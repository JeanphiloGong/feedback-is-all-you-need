"""Audit feedback-history interventions and their held-out rule execution."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from transformers import AutoConfig, AutoTokenizer

from experiments.study_b.run import REVISIONS
from experiments.study_b.tasks import COMMANDS
from experiments.study_f.rules import check_rule, parse_rule
from experiments.study_f.diagnose import transition as component_transition
from experiments.study_g.history import history_messages
from experiments.study_g.run import DATA_SHA256, SOURCES, STAGES


def interval(values, bootstrap=False):
    values = np.asarray(values, dtype=float)
    result = dict(n=len(values), mean=float(values.mean()))
    if len(values) == 1:
        return result
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    result.update(sd=sd, low=result["mean"] - half, high=result["mean"] + half)
    if bootstrap:
        rng = np.random.default_rng(823)
        samples = values[rng.integers(0, len(values), (10_000, len(values)))].mean(axis=1)
        result["bootstrap95"] = np.quantile(samples, [0.025, 0.975]).tolist()
    return result


def audit(root, tokenizer):
    metadata = json.loads((root / "metadata.json").read_text())
    seeds = [19] if metadata["development"] else list(range(9000, 9048))
    assert metadata["study"] == "G" and metadata["size"] in ("0.5B", "7B")
    assert metadata["model"] == f"Qwen/Qwen2.5-{metadata['size']}-Instruct"
    assert metadata["revision"] == REVISIONS[metadata["size"]]
    assert metadata["seeds"] == seeds and metadata["completed"]
    assert metadata["finished_unix"] > metadata["started_unix"]
    assert metadata["elapsed_seconds"] == metadata["finished_unix"] - metadata["started_unix"]
    assert metadata["stages"] == [list(s) for s in STAGES] and metadata["batch_size"] == 16
    assert metadata["dtype"] == "bfloat16" and metadata["attention"] == "sdpa"
    assert metadata["packages"] == {p: importlib.metadata.version(p)
                                   for p in ("torch", "transformers", "numpy", "scipy", "matplotlib")}
    assert metadata["generation_overrides"] == dict(do_sample=False, max_new_tokens=512)
    assert metadata["max_position_embeddings"] == AutoConfig.from_pretrained(
        metadata["model"], revision=metadata["revision"], local_files_only=True).max_position_embeddings
    assert set(metadata["source_sha256"]) == set(SOURCES)
    for source, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    frozen = Path("experiments/study_g/data/environments-v1.json").read_bytes()
    assert metadata["input_environments_sha256"] == hashlib.sha256(frozen).hexdigest() == DATA_SHA256
    executed = (root / "environments.json").read_bytes()
    assert hashlib.sha256(executed).hexdigest() == metadata["executed_environments_sha256"]
    environments = json.loads(executed)
    assert environments == [e for e in json.loads(frozen)["environments"] if e["seed"] in seeds]
    if not metadata["development"]:
        gate = metadata["development_evidence"]
        assert gate["passed"] and gate["metadata"]["development"] and gate["metadata"]["completed"]
        assert gate["metadata"]["seeds"] == [19]
        for field in ("model", "revision", "source_sha256", "packages", "dtype", "attention",
                      "batch_size", "generation_config", "generation_overrides"):
            assert gate["metadata"][field] == metadata[field]
        assert gate["metadata"]["finished_unix"] < metadata["started_unix"]
    envs = {e["seed"]: e for e in environments}
    symbolic = {}
    for environment in environments:
        train_keys = {(q["state"], tuple(q["commands"])) for q in environment["train"]}
        test_keys = {(q["state"], tuple(q["commands"])) for q in environment["direct"] + environment["compose"]}
        assert len(train_keys) == 16 and len(test_keys) == 48 and not train_keys & test_keys
        true_rule = dict(zip(COMMANDS, environment["offsets"]))
        inferred = {}
        for query in environment["train"]:
            command, = query["commands"]
            value = (query["target"] - query["state"]) % 8
            assert command not in inferred or inferred[command] == value
            inferred[command] = value
        assert inferred == true_rule
        symbolic[environment["seed"]] = inferred
        for query in environment["train"] + environment["direct"] + environment["compose"]:
            assert query["target"] == (query["state"] + sum(true_rule[c] for c in query["commands"])) % 8
    rows = [json.loads(line) for line in (root / "proposals.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (root / "checks.jsonl").read_text().splitlines()]
    input_batches = [json.loads(line) for line in (root / "batches.jsonl").read_text().splitlines()]
    assert [(b["arm"], b["stage"], b["batch_index"]) for b in input_batches] == [
        (arm, stage, start // 16) for arm, stage in STAGES for start in range(0, len(seeds), 16)]
    input_batches = {(b["arm"], b["stage"], b["batch_index"]): b for b in input_batches}
    keys = [(seed, arm, stage) for arm, stage in STAGES for seed in seeds]
    assert [(r["seed"], r["arm"], r["stage"]) for r in rows] == keys
    assert [(r["seed"], r["arm"], r["stage"]) for r in checks] == keys
    assert {p.name for p in (root / "rules").iterdir()} == {f"{s}-{a}-{t}.json" for s, a, t in keys}
    indexed, batches, scored = {}, {}, []
    eos = metadata["generation_config"]["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    for row, check in zip(rows, checks):
        seed, arm, stage = row["seed"], row["arm"], row["stage"]
        environment = envs[seed]
        previous_keys = [] if stage == 0 else [(seed, "initial", 0)] + [
            (seed, arm, i) for i in range(1, stage)]
        history = [dict(text=indexed[k]["text"], check=check_rule(parse_rule(indexed[k]["text"]), environment["train"]))
                   for k in previous_keys]
        used_keys = previous_keys if arm == "full_history" else previous_keys[-1:]
        assert row["history_keys"] == [list(k) for k in used_keys]
        prompt = tokenizer.apply_chat_template(history_messages(environment["train"], history, arm),
                                                tokenize=False, add_generation_prompt=True)
        assert row["prompt"] == prompt and row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert row["input_tokens"] == len(tokenizer.encode(prompt, add_special_tokens=False))
        ids = row["output_token_ids"]
        assert 1 <= len(ids) <= 512 and row["output_tokens"] == len(ids)
        assert not any(token in eos for token in ids[:-1])
        assert row["eos_terminated"] == (ids[-1] in eos)
        assert row["eos_terminated"] or len(ids) == 512
        assert row["text"] == tokenizer.decode(ids, skip_special_tokens=True)
        rule = parse_rule(row["text"])
        memory = root / "rules" / f"{seed}-{arm}-{stage}.json"
        assert json.loads(memory.read_text()) == check["rule"] == rule
        assert check["memory_bytes"] == memory.stat().st_size
        assert check["check"] == check_rule(rule, environment["train"])
        assert np.isfinite(check["check_seconds"]) and check["check_seconds"] >= 0
        assert np.isfinite(row["batch_wall_seconds"]) and row["batch_wall_seconds"] >= 0
        assert row["batch_index"] == seeds.index(seed) // 16
        batches.setdefault((arm, stage, row["batch_index"]), []).append(row)
        if arm == "full_history" and stage == 1:
            paired = indexed[(seed, "last_only", 1)]
            assert row["batch_input_sha256"] == paired["batch_input_sha256"]
            assert row["output_token_ids"] == paired["output_token_ids"]
        indexed[(seed, arm, stage)] = row
        for split in ("direct", "compose"):
            for index, query in enumerate(environment[split]):
                prediction = None if rule is None else (query["state"] + sum(rule[c] for c in query["commands"])) % 8
                scored.append(dict(seed=seed, arm=arm, stage=stage, split=split, index=index,
                    query=query, prediction=prediction, target=query["target"], correct=prediction == query["target"]))
    for (arm, stage, batch_index), group in batches.items():
        assert [r["seed"] for r in group] == seeds[batch_index * 16:batch_index * 16 + 16]
        assert all(r["batch_size"] == len(group) for r in group)
        for field in ("batch_wall_seconds", "padded_input_tokens", "padded_generation_tokens"):
            assert len({r[field] for r in group}) == 1
        assert group[0]["padded_input_tokens"] == len(group) * max(r["input_tokens"] for r in group)
        assert group[0]["padded_generation_tokens"] == len(group) * max(r["output_tokens"] for r in group)
        encoded = tokenizer([r["prompt"] for r in group], padding=True, add_special_tokens=False)
        saved = input_batches[(arm, stage, batch_index)]
        assert saved["seeds"] == [r["seed"] for r in group]
        assert saved["input_ids"] == encoded["input_ids"]
        assert saved["attention_mask"] == encoded["attention_mask"]
        digest = hashlib.sha256(json.dumps(dict(input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]), sort_keys=True).encode()).hexdigest()
        assert saved["input_sha256"] == digest
        assert all(r["batch_input_sha256"] == digest for r in group)
        assert len(encoded["input_ids"][0]) + 512 <= metadata["max_position_embeddings"]
    assert metadata["elapsed_seconds"] >= sum(g[0]["batch_wall_seconds"] for g in batches.values())
    return metadata, rows, checks, scored, batches, symbolic, envs


def summarize(metadata, rows, checks, scored, batches, envs):
    seeds = metadata["seeds"]
    values, summaries = {}, []
    score_index = {}
    for row in scored:
        score_index.setdefault((row["seed"], row["arm"], row["stage"], row["split"]), []).append(row["correct"])
    for arm, stage in STAGES:
        for split in ("direct", "compose"):
            means = np.array([np.mean(score_index[(seed, arm, stage, split)]) for seed in seeds])
            values[(arm, stage, split)] = means
            summaries.append(dict(arm=arm, stage=stage, split=split, **interval(means)))
    contrasts, transitions = [], []
    for split in ("direct", "compose"):
        for arm, reference in (("full_history", "last_only"), ("last_only", "initial"),
                               ("full_history", "initial")):
            before_stage = 0 if reference == "initial" else 16
            primary = metadata["size"] == "0.5B" and split == "direct" and reference == "last_only"
            contrast = arm + "_minus_" + reference
            contrasts.append(dict(contrast=contrast, split=split, primary=primary,
                **interval(values[(arm, 16, split)] - values[(reference, before_stage, split)], bootstrap=primary)))
            pairs, gains, losses = [], [], []
            for seed in seeds:
                before = score_index[(seed, reference, before_stage, split)]
                after = score_index[(seed, arm, 16, split)]
                local_pairs = list(zip(before, after))
                pairs.extend(local_pairs)
                gains.append(np.mean([not a and b for a, b in local_pairs]))
                losses.append(np.mean([a and not b for a, b in local_pairs]))
            transitions.append(dict(contrast=contrast, split=split, n=len(pairs),
                wrong_to_correct=sum(not a and b for a, b in pairs), correct_to_wrong=sum(a and not b for a, b in pairs),
                wrong_to_correct_rate=interval(gains), correct_to_wrong_rate=interval(losses)))
    diagnostics, costs = [], []
    for arm, stage in STAGES:
        selected = [r for r in rows if (r["arm"], r["stage"]) == (arm, stage)]
        checked = [r for r in checks if (r["arm"], r["stage"]) == (arm, stage)]
        grouped = [g for (a, t, _), g in batches.items() if (a, t) == (arm, stage)]
        passed = sum(r["check"] == "External check: all 16 examples pass." for r in checked)
        nonexact_perfect = sum(values[(arm, stage, "compose")][i] == 1 and
                              check["check"] != "External check: all 16 examples pass."
                              for i, check in enumerate(checked))
        diagnostics.append(dict(arm=arm, stage=stage, n=len(selected),
            valid=sum(r["rule"] is not None for r in checked), exact_rule=passed, train_check_pass=passed,
            perfect_composition_nonexact=int(nonexact_perfect),
            eos_terminated=sum(r["eos_terminated"] for r in selected),
            memory_bytes=sum(r["memory_bytes"] for r in checked)))
        for endpoint in ("valid", "exact_rule", "train_check_pass"):
            bounds = stats.binomtest(diagnostics[-1][endpoint], len(selected)).proportion_ci(method="wilson")
            diagnostics[-1][endpoint + "_rate"] = dict(mean=diagnostics[-1][endpoint] / len(selected),
                                                       low=bounds.low, high=bounds.high, method="Wilson 95%")
        costs.append(dict(arm=arm, stage=stage, continuations=len(selected), batches=len(grouped),
            input_tokens=sum(r["input_tokens"] for r in selected),
            output_tokens=sum(r["output_tokens"] for r in selected),
            padded_input_tokens=sum(g[0]["padded_input_tokens"] for g in grouped),
            padded_generation_tokens=sum(g[0]["padded_generation_tokens"] for g in grouped),
            generation_seconds=sum(g[0]["batch_wall_seconds"] for g in grouped),
            check_seconds=sum(r["check_seconds"] for r in checked)))
    row_index = {(r["seed"], r["arm"], r["stage"]): r for r in rows}
    check_index = {(r["seed"], r["arm"], r["stage"]): r for r in checks}
    repetition, components = [], []
    for seed in seeds:
        for arm in ("last_only", "full_history"):
            previous_keys = [(seed, "initial", 0)]
            for stage in range(1, 17):
                key = (seed, arm, stage)
                current = row_index[key]
                rule = check_index[key]["rule"]
                previous = row_index[previous_keys[-1]]
                prior_rules = [check_index[k]["rule"] for k in previous_keys]
                stratum = "invalid" if rule is None else ("exact" if check_index[key]["check"] ==
                          "External check: all 16 examples pass." else "valid_wrong")
                repetition.append(dict(seed=seed, arm=arm, stage=stage, stratum=stratum,
                    immediate_text=current["text"] == previous["text"],
                    seen_text=any(current["text"] == row_index[k]["text"] for k in previous_keys),
                    immediate_mapping=None if rule is None else rule == prior_rules[-1],
                    seen_mapping=None if rule is None else rule in prior_rules))
                components.append(dict(seed=seed, arm=arm, stage=stage,
                    **component_transition(prior_rules[-1], rule, envs[seed]["train"])))
                previous_keys.append(key)
    repetition_summary = []
    for arm in ("last_only", "full_history"):
        for stratum in ("invalid", "valid_wrong", "exact"):
            group = [r for r in repetition if r["arm"] == arm and r["stratum"] == stratum]
            repetition_summary.append(dict(arm=arm, stratum=stratum, transitions=len(group),
                mapping_denominator=0 if stratum == "invalid" else len(group),
                **{metric: sum(r[metric] is True for r in group) for metric in
                   ("immediate_text", "seen_text", "immediate_mapping", "seen_mapping")}))
    cost_fields = ("continuations", "batches", "input_tokens", "output_tokens", "padded_input_tokens",
                   "padded_generation_tokens", "generation_seconds", "check_seconds")
    actual_cost = {field: sum(c[field] for c in costs) for field in cost_fields}
    standalone_cost = {arm: {field: sum(c[field] for c in costs if c["arm"] in ("initial", arm))
                            for field in cost_fields} for arm in ("last_only", "full_history")}
    return dict(size=metadata["size"], summaries=summaries, contrasts=contrasts, transitions=transitions,
                diagnostics=diagnostics, costs=costs, elapsed_seconds=metadata["elapsed_seconds"],
                actual_cost=actual_cost, hypothetical_standalone_cost=standalone_cost,
                peak_allocated_bytes=metadata["peak_allocated_bytes"],
                repetition_summary=repetition_summary, repetition_records=repetition, component_records=components,
                interpretation="Environment-paired intervals; only 0.5B final full-history-minus-last-only direct accuracy is primary. Shared initial generation is counted once. Repetition and component records are nested within environments; zero-variance empirical intervals do not establish zero population effects.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    meta = json.loads((root / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"], local_files_only=True)
    tokenizer.padding_side = "left"
    metadata, rows, checks, scored, batches, symbolic, envs = audit(root, tokenizer)
    with (root / "evaluations.jsonl").open("w") as stream:
        for row in scored:
            stream.write(json.dumps(row) + "\n")
    (root / "symbolic.json").write_text(json.dumps(symbolic, indent=2) + "\n")
    summary = summarize(metadata, rows, checks, scored, batches, envs)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not metadata["development"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
        for axis, split in zip(axes, ("direct", "compose")):
            for arm in ("last_only", "full_history"):
                points = [next(r for r in summary["summaries"] if (r["arm"], r["stage"], r["split"]) ==
                              ("initial" if stage == 0 else arm, stage, split)) for stage in (0, 1, 2, 4, 8, 16)]
                axis.errorbar((0, 1, 2, 4, 8, 16), [p["mean"] for p in points],
                              yerr=[p["high"] - p["mean"] for p in points], marker="o", label=arm)
            axis.set(title=f"{metadata['size']} {split}", xlabel="Revision calls", xticks=(0, 1, 2, 4, 8, 16), ylim=(-0.04, 1.04))
        axes[0].set_ylabel("Held-out executor accuracy")
        axes[1].legend()
        fig.savefig(root / "feedback-history.pdf")
        fig.savefig(root / "feedback-history.png", dpi=180)
        plt.close(fig)
    evidence = dict(passed=True, metadata=metadata,
        files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p.name not in ("audit.json", "README.md")})
    (root / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Verified {len(rows)} proposals, {len(scored)} evaluations, and {len(checks)} saved memories.")
    for contrast in summary["contrasts"]:
        print(json.dumps(contrast))


if __name__ == "__main__":
    main()
