"""Post-hoc selection from fixed proposal pools using disclosed examples only."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

from experiments.study_b.tasks import COMMANDS, execute
from experiments.study_f.rules import parse_rule

CHECKPOINTS = (0, 1, 2, 4, 8, 16)


def matches(rule, queries):
    if rule is None:
        return [False] * len(queries)
    offsets = [rule[c] for c in COMMANDS]
    return [execute(q["state"], q["commands"], offsets) == q["target"] for q in queries]


def select_history(rules, examples):
    """Keep the earliest best rule at each prefix; invalid ranks below valid."""
    ranks = [sum(matches(rule, examples)) if rule is not None else -1 for rule in rules]
    selected, best = [], 0
    for stage in range(len(rules)):
        if ranks[stage] > ranks[best]:
            best = stage
        selected.append(best)
    return ranks, selected


def describe_trajectory(rules, environment):
    ranks, selected = select_history(rules, environment["train"])
    records = []
    for stage in CHECKPOINTS:
        chosen = selected[stage]
        record = dict(seed=environment["seed"], stage=stage, selected_stage=chosen,
                      current_valid=rules[stage] is not None, selected_valid=rules[chosen] is not None,
                      current_train_passes=max(0, ranks[stage]), selected_train_passes=max(0, ranks[chosen]),
                      current_exact=ranks[stage] == 16, selected_exact=ranks[chosen] == 16,
                      candidate_count=stage + 1,
                      selection_example_executions=16 * sum(rule is not None for rule in rules[:stage + 1]))
        for split in ("direct", "compose"):
            current = matches(rules[stage], environment[split])
            chosen_answers = matches(rules[chosen], environment[split])
            record[f"current_{split}"] = mean(current)
            record[f"selected_{split}"] = mean(chosen_answers)
            record[f"gained_{split}"] = sum(not before and after for before, after in zip(current, chosen_answers))
            record[f"lost_{split}"] = sum(before and not after for before, after in zip(current, chosen_answers))
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    input_hashes, records, summaries = {}, [], []
    for size in ("0.5B", "7B"):
        folder = args.input / f"formal-{size}-v1"
        for name in ("environments.json", "proposals.jsonl"):
            input_hashes[f"formal-{size}-v1/{name}"] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        environments = json.loads((folder / "environments.json").read_text())
        rows = [json.loads(line) for line in (folder / "proposals.jsonl").read_text().splitlines()]
        rules = {(row["seed"], row["arm"], row["stage"]): parse_rule(row["text"]) for row in rows}
        for arm in ("last_only", "full_history"):
            group = []
            for environment in environments:
                seed = environment["seed"]
                trajectory = [rules[(seed, "initial", 0)]] + [rules[(seed, arm, stage)] for stage in range(1, 17)]
                group.extend(dict(size=size, arm=arm, **row) for row in describe_trajectory(trajectory, environment))
            records.extend(group)
            for stage in CHECKPOINTS:
                checkpoint = [row for row in group if row["stage"] == stage]
                summary = dict(size=size, arm=arm, stage=stage, n=len(checkpoint),
                    earlier_selected=sum(row["selected_stage"] < stage for row in checkpoint),
                    exact_discarded_by_current=sum(row["selected_exact"] and not row["current_exact"] for row in checkpoint))
                for key in ("current_valid", "selected_valid", "current_exact", "selected_exact",
                            "selection_example_executions", "gained_direct", "lost_direct", "gained_compose", "lost_compose"):
                    summary[key] = sum(row[key] for row in checkpoint)
                for split in ("direct", "compose"):
                    summary[f"current_{split}"] = mean(row[f"current_{split}"] for row in checkpoint)
                    summary[f"selected_{split}"] = mean(row[f"selected_{split}"] for row in checkpoint)
                    summary[f"difference_{split}"] = summary[f"selected_{split}"] - summary[f"current_{split}"]
                summaries.append(summary)
    source_paths = (Path("experiments/study_g/selection.py"), Path("experiments/study_b/tasks.py"), Path("experiments/study_f/rules.py"),
                    Path("notes/study-g-selection-plan.md"))
    output = dict(status="exploratory post-hoc; fixed pools; no new model calls or formal-score changes",
                  input_sha256=input_hashes,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  elapsed_seconds=time.perf_counter() - started,
                  summaries=summaries, records=records)
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps([row for row in summaries if row["stage"] == 16], indent=2))


if __name__ == "__main__":
    main()
