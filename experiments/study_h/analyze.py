"""Audit all searches before describing fixed Study H endpoints and costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_h.audit import audit_run
from experiments.study_h.run import ARMS, BUDGET
from experiments.study_h.search import encoded_length


CHECKPOINTS = (4096, 16384, 65536)


def summarize(rows, memories):
    seeds = sorted(memories)
    records, costs = [], []
    for seed in seeds:
        experience = [r for r in rows if r["seed"] == seed and r["family"] == "experience"]
        programs = [tuple(r["result"]["candidate_program"]) for r in experience
                    if r["result"]["status"] == "accepted"]
        for arm in ARMS:
            evaluated = [r for r in rows if r["seed"] == seed and r["family"] != "experience" and r["arm"] == arm]
            for family in ("composition", "shifted"):
                group = [r for r in evaluated if r["family"] == family]
                for budget in CHECKPOINTS:
                    available = [r for r in group if r["result"]["status"] == "accepted"
                                 and r["result"]["primitive_ops"] <= budget]
                    records.append(dict(seed=seed, arm=arm, family=family, budget=budget, tasks=len(group),
                        accepted=len(available), exact_count=sum(r["evaluation"]["exact"] for r in available),
                        exact_rate=sum(r["evaluation"]["exact"] for r in available) / len(group),
                        test_accuracy=sum(r["evaluation"]["test_correct"] for r in available) / (8 * len(group))))
            library = [tuple(f) for f in memories[seed]["libraries"][arm]]
            previous = experience if arm != "none" else []
            induction = memories[seed]["seconds"]
            library_seconds = (0 if arm == "none" else induction["whole"] if arm == "whole"
                               else induction["compressed"] + (induction["renamed"] if arm == "renamed" else 0))
            costs.append(dict(seed=seed, arm=arm, library_entries=len(library),
                definition_tokens=sum(len(f) + 1 for f in library),
                encoded_experience_tokens=sum(encoded_length(p, library) for p in programs),
                experience_tasks=len(previous), experience_accepted=sum(r["result"]["status"] == "accepted" for r in previous),
                experience_false_acceptances=sum(r["result"]["status"] == "accepted" and not r["evaluation"]["exact"] for r in previous),
                experience_primitive_ops=sum(r["result"]["primitive_ops"] for r in previous),
                evaluation_search_primitive_ops=sum(r["result"]["primitive_ops"] for r in evaluated),
                standalone_search_primitive_ops=sum(r["result"]["primitive_ops"] for r in previous + evaluated),
                standalone_candidates=sum(r["result"]["candidates"] for r in previous + evaluated),
                standalone_comparisons=sum(r["result"]["comparisons"] for r in previous + evaluated),
                experience_search_seconds=sum(r["search_seconds"] for r in previous),
                evaluation_search_seconds=sum(r["search_seconds"] for r in evaluated),
                induction_seconds=library_seconds,
                standalone_seconds=sum(r["search_seconds"] for r in previous + evaluated) + library_seconds))
    summaries = []
    for family in ("composition", "shifted"):
        for arm in ARMS:
            for budget in CHECKPOINTS:
                group = [r for r in records if (r["family"], r["arm"], r["budget"]) == (family, arm, budget)]
                summaries.append(dict(family=family, arm=arm, budget=budget, n=len(seeds),
                    tasks=sum(r["tasks"] for r in group), accepted=sum(r["accepted"] for r in group),
                    exact_count=sum(r["exact_count"] for r in group),
                    exact_rate=mean(r["exact_rate"] for r in group),
                    test_accuracy=mean(r["test_accuracy"] for r in group)))
    end = {(r["seed"], r["arm"]): r["exact_rate"] for r in records
           if r["family"] == "composition" and r["budget"] == BUDGET}
    differences = [end[(seed, "compressed")] - end[(seed, "whole")] for seed in seeds]
    primary = dict(contrast="compressed minus whole", family="composition", budget=BUDGET,
                   n=len(seeds), mean=mean(differences), differences=differences, ci95=None, bootstrap95=None)
    if len(seeds) > 1:
        standard_error = stdev(differences) / len(seeds)**0.5
        half = float(t.ppf(.975, len(seeds) - 1)) * standard_error
        bootstrap = np.random.default_rng(829).choice(differences, (10000, len(seeds))).mean(axis=1)
        primary.update(ci95=[primary["mean"] - half, primary["mean"] + half],
                       bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                       degenerate=standard_error == 0,
                       positive_primary=primary["mean"] - half > 0)
        primary["interval_disagreement"] = (
            primary["positive_primary"] != (primary["bootstrap95"][0] > 0)
            or (primary["ci95"][1] < 0) != (primary["bootstrap95"][1] < 0))
    cost_totals = []
    for arm in ARMS:
        group = [c for c in costs if c["arm"] == arm]
        cost_totals.append(dict(arm=arm, **{key: sum(c[key] for c in group)
                                          for key in group[0] if key not in ("seed", "arm")}))
    actual = dict(searches=len(rows), primitive_ops=sum(r["result"]["primitive_ops"] for r in rows),
                  search_seconds=sum(r["search_seconds"] for r in rows),
                  final_evaluator_primitive_ops=sum(r["evaluation"]["primitive_ops"] for r in rows),
                  final_evaluator_seconds=sum(r["evaluation_seconds"] for r in rows),
                  induction_seconds=sum(sum(m["seconds"].values()) for m in memories.values()))
    return dict(status="audited finite symbolic experiment; no neural model", seeds=seeds,
                statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__),
                primary=primary, summaries=summaries, actual_cost=actual, standalone_cost_totals=cost_totals,
                seed_records=records, seed_costs=costs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        rows, memories, audit = audit_run(args.input)
        summary = summarize(rows, memories)
        audit["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(dict(primary=summary["primary"],
                              endpoints=[s for s in summary["summaries"] if s["budget"] == BUDGET]), indent=2))
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps(dict(
            error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
