"""Audit the feedback journal before computing environment-paired changes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from experiments.study_b.run import REVISIONS, text_prompt
from experiments.study_b.tasks import execute

CHECKPOINTS = (0, 1, 2, 4, 8, 16)
EVAL_GRID = {"none": (0,), "context": CHECKPOINTS, "online_lora": CHECKPOINTS,
             "error_only_lora": CHECKPOINTS, "online_lora_with_context": (16,),
             "error_only_lora_with_context": (16,)}
SOURCE_PATHS = {"experiments/study_e/run.py", "experiments/study_b/run.py",
                "experiments/study_b/tasks.py", "experiments/study_e/freeze_data.py"}
DATA_SHA256 = "67fd60a700203034dccb8e79fd95c119bb1315f08d69f040ae2b0ae24784b0c9"


def interval(values, bootstrap=False):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    result = dict(n=len(values), mean=mean, sd=sd, low=mean - half, high=mean + half)
    if bootstrap:
        rng = np.random.default_rng(821)
        samples = values[rng.integers(0, len(values), (10_000, len(values)))].mean(axis=1)
        result["bootstrap95"] = np.quantile(samples, [0.025, 0.975]).tolist()
    return result


def verify_score(row, query, history, tokenizer, digit_ids):
    assert row["query"] == query and row["target"] == query["target"]
    assert row["prediction"] in range(8)
    assert row["correct"] == (row["prediction"] == row["target"])
    assert row["unconstrained_text"] == tokenizer.decode([row["unconstrained_token"]])
    assert row["unconstrained_first_token_correct"] == (row["unconstrained_token"] == digit_ids[row["target"]])
    prompt = text_prompt(tokenizer, query, history)
    assert hashlib.sha256(prompt.encode()).hexdigest() == row["prompt_sha256"]
    assert len(tokenizer.encode(prompt, add_special_tokens=False)) == row["input_tokens"]
    assert prompt == text_prompt(tokenizer, dict(query, target=(query["target"] + 1) % 8), history)
    assert np.isfinite(row["batch_wall_seconds"]) and row["batch_wall_seconds"] >= 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    root = args.input
    metadata = json.loads((root / "metadata.json").read_text())
    seeds = [13] if args.development else list(range(7000, 7024))
    assert metadata["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert metadata["revision"] == REVISIONS["0.5B"]
    assert metadata["seeds"] == seeds and metadata["development"] == args.development
    assert metadata["completed"] and metadata["finished_unix"] > metadata["started_unix"]
    assert metadata["checkpoints"] == list(CHECKPOINTS) and metadata["batch_size"] == 16
    assert metadata["dtype"] == "bfloat16" and metadata["attention"] == "sdpa"
    assert metadata["generation_interface"] == "constrained first-token digit argmax"
    assert metadata["base_initial_sha256"] == metadata["base_final_sha256"]
    assert set(metadata["source_sha256"]) == SOURCE_PATHS
    for source, digest in metadata["source_sha256"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    frozen = Path("experiments/study_e/data/environments-v1.json").read_bytes()
    assert metadata["input_environments_sha256"] == hashlib.sha256(frozen).hexdigest() == DATA_SHA256
    executed = (root / "environments.json").read_bytes()
    assert metadata["executed_environments_sha256"] == hashlib.sha256(executed).hexdigest()
    environments = json.loads(executed)
    if not args.development:
        assert environments == json.loads(frozen)["environments"]
        gate = metadata["development_evidence"]
        assert gate["passed"] and gate["metadata"]["development"] and gate["metadata"]["seeds"] == [13]
        for field in ("model", "revision", "source_sha256", "packages", "base_initial_sha256",
                      "dtype", "attention", "generation_interface", "generation_config"):
            assert gate["metadata"][field] == metadata[field]
        assert gate["metadata"]["finished_unix"] < metadata["started_unix"]
    envs = {environment["seed"]: environment for environment in environments}
    assert set(envs) == set(seeds) and len(environments) == len(seeds)
    for environment in environments:
        adaptation = {(q["state"], tuple(q["commands"])) for q in environment["train"]}
        held_out = {(q["state"], tuple(q["commands"])) for q in environment["direct"] + environment["compose"]}
        assert len(adaptation) == 16 and len(held_out) == 48 and not adaptation & held_out
        for query in environment["train"] + environment["direct"] + environment["compose"]:
            assert query["target"] == execute(query["state"], query["commands"], environment["offsets"])
    tokenizer = AutoTokenizer.from_pretrained(metadata["model"], revision=metadata["revision"], local_files_only=True)
    digit_ids = [tokenizer.encode(str(digit), add_special_tokens=False)[0] for digit in range(8)]
    assert digit_ids == metadata["digit_token_ids"]
    with (root / "predictions.jsonl").open() as stream:
        predictions = [json.loads(line) for line in stream]
    with (root / "training.jsonl").open() as stream:
        journal = [json.loads(line) for line in stream]
    expected = {(seed, condition, count, split, index) for seed in seeds
                for condition, counts in EVAL_GRID.items() for count in counts
                for split, size in (("direct", 16), ("compose", 32)) for index in range(size)}
    keys = [(r["seed"], r["condition"], r["feedback_count"], r["split"], r["index"]) for r in predictions]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    indexed = dict(zip(keys, predictions))
    batches = {}
    for row in predictions:
        environment = envs[row["seed"]]
        history = environment["train"][:row["feedback_count"]] if row["condition"] == "context" else []
        if row["condition"].endswith("_with_context"):
            history = environment["train"]
        verify_score(row, environment[row["split"]][row["index"]], history, tokenizer, digit_ids)
        if row["feedback_count"] == 0:
            baseline = indexed[(row["seed"], "none", 0, row["split"], row["index"])]
            assert (row["prediction"], row["unconstrained_token"]) == (baseline["prediction"], baseline["unconstrained_token"])
        assert row["batch_index"] == row["index"] // 16
        key = (row["seed"], row["condition"], row["feedback_count"], row["split"], row["batch_index"])
        batches.setdefault(key, []).append(row)
    for group in batches.values():
        assert len(group) == 16 and all(r["batch_size"] == 16 for r in group)
        for field in ("batch_wall_seconds", "padded_batch_tokens"):
            assert len({r[field] for r in group}) == 1
        assert group[0]["padded_batch_tokens"] == 16 * max(r["input_tokens"] for r in group)
    expected_journal = [(seed, condition, step, event) for seed in seeds
                        for condition in ("context", "online_lora", "error_only_lora")
                        for step in range(16) for event in ("prediction", "feedback")]
    assert [(r["seed"], r["condition"], r["step"], r["event"]) for r in journal] == expected_journal
    for prediction, feedback in zip(journal[::2], journal[1::2]):
        environment = envs[prediction["seed"]]
        query = environment["train"][prediction["step"]]
        condition = prediction["condition"]
        history = environment["train"][:prediction["step"]] if condition == "context" else []
        verify_score(prediction, query, history, tokenizer, digit_ids)
        assert prediction["batch_size"] == 1 and prediction["padded_batch_tokens"] == prediction["input_tokens"]
        expected_update = condition == "online_lora" or (condition == "error_only_lora" and not prediction["correct"])
        assert feedback["target"] == query["target"] and feedback["update_applied"] == expected_update
        update = feedback["update"]
        if expected_update:
            prompt = text_prompt(tokenizer, query, [])
            assert update["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
            assert update["input_tokens"] == update["padded_input_tokens"] == len(tokenizer.encode(prompt, add_special_tokens=False))
            assert np.isfinite(update["loss"]) and update["loss"] >= 0
            assert np.isfinite(update["wall_seconds"]) and update["wall_seconds"] >= 0
        else:
            assert update is None
    accounting = metadata["accounting"]
    assert [(r["seed"], r["condition"]) for r in accounting] == [
        (seed, condition) for seed in seeds for condition in ("context", "online_lora", "error_only_lora")]
    for record in accounting:
        expected_updates = sum(r["update_applied"] for r in journal[1::2]
                               if (r["seed"], r["condition"]) == (record["seed"], record["condition"]))
        assert record["applied_updates"] == expected_updates
        if args.development:
            assert record["base_after_sha256"] == metadata["base_initial_sha256"]
        if record["condition"] != "context":
            assert record["trainable_parameters"] == 540672
            assert record["initialization_seed"] == record["seed"] + 4_000_000
            assert "adapter_model.safetensors" in record["adapter_sha256"]
            for name, digest in record["adapter_sha256"].items():
                path = root / "adapters" / str(record["seed"]) / record["condition"] / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for seed in seeds:
        assert len({r["initial_adapter_sha256"] for r in accounting if r["seed"] == seed and r["condition"] != "context"}) == 1
    print(f"Verified {len(predictions)} evaluation rows, {len(journal)//2} prediction/feedback pairs, and all adapters.")
    if args.development:
        evidence = dict(passed=True, metadata=metadata,
                        analysis_sha256=hashlib.sha256(Path("experiments/study_e/analyze.py").read_bytes()).hexdigest(),
                        files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in root.rglob("*") if p.is_file() and p.name != "audit.json"})
        (root / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
        return
    values, summaries = {}, []
    for condition, counts in EVAL_GRID.items():
        for count in counts:
            for split in ("direct", "compose"):
                per_environment = np.array([np.mean([indexed[(seed, condition, count, split, i)]["correct"]
                                                     for i in range(16 if split == "direct" else 32)]) for seed in seeds])
                values[(condition, count, split)] = per_environment
                summaries.append(dict(condition=condition, feedback_count=count, split=split, **interval(per_environment)))
    contrasts = []
    for condition in ("context", "online_lora", "error_only_lora"):
        for split in ("direct", "compose"):
            delta = values[(condition, 16, split)] - values[("none", 0, split)]
            primary = condition == "online_lora" and split == "compose"
            contrasts.append(dict(condition=condition, split=split, contrast="16_minus_none", primary=primary,
                                  **interval(delta, bootstrap=primary)))
    for split in ("direct", "compose"):
        contrasts.append(dict(condition="online_lora_minus_context", split=split, contrast="at_16", primary=False,
                              **interval(values[("online_lora", 16, split)] - values[("context", 16, split)])))
        for condition in ("online_lora", "error_only_lora"):
            contrasts.append(dict(condition=condition + "_with_context_minus_adapter", split=split,
                                  contrast="at_16", primary=False,
                                  **interval(values[(condition + "_with_context", 16, split)] - values[(condition, 16, split)])))
    transitions = []
    for condition in ("context", "online_lora", "error_only_lora"):
        for split, size in (("direct", 16), ("compose", 32)):
            pairs = [(indexed[(s, "none", 0, split, i)]["correct"],
                      indexed[(s, condition, 16, split, i)]["correct"])
                     for s in seeds for i in range(size)]
            transitions.append(dict(condition=condition, split=split, n=len(pairs),
                                    wrong_to_correct=sum(not a and b for a, b in pairs),
                                    correct_to_wrong=sum(a and not b for a, b in pairs)))
    costs = {}
    for condition in ("none", "context", "online_lora", "error_only_lora",
                      "online_lora_with_context", "error_only_lora_with_context"):
        eval_rows = [r for r in predictions if r["condition"] == condition]
        eval_batches = [g for g in batches.values() if g[0]["condition"] == condition]
        pre = [r for r in journal[::2] if r["condition"] == condition]
        updates = [r["update"] for r in journal[1::2] if r["condition"] == condition and r["update_applied"]]
        costs[condition] = dict(evaluation_calls=len(eval_batches), adaptation_prediction_calls=len(pre),
            evaluation_input_tokens=sum(r["input_tokens"] for r in eval_rows),
            evaluation_padded_input_tokens=sum(g[0]["padded_batch_tokens"] for g in eval_batches),
            adaptation_input_tokens=sum(r["input_tokens"] for r in pre),
            adaptation_padded_input_tokens=sum(r["padded_batch_tokens"] for r in pre),
            applied_updates=len(updates), optimization_input_tokens=sum(u["input_tokens"] for u in updates),
            optimization_padded_input_tokens=sum(u["padded_input_tokens"] for u in updates),
            optimization_wall_seconds=sum(u["wall_seconds"] for u in updates),
            evaluation_wall_seconds=sum(g[0]["batch_wall_seconds"] for g in eval_batches),
            adaptation_prediction_wall_seconds=sum(r["batch_wall_seconds"] for r in pre))
    summary = dict(summary=summaries, contrasts=contrasts, transitions=transitions, costs=costs,
                   elapsed_seconds=metadata["elapsed_seconds"], peak_allocated_bytes=metadata["peak_allocated_bytes"],
                   interpretation="24 environment units; online LoRA composition change primary; other intervals descriptive.")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, split in zip(axes, ("direct", "compose")):
        for condition in ("context", "online_lora", "error_only_lora"):
            points = [values[(condition, count, split)] for count in CHECKPOINTS]
            axis.errorbar(CHECKPOINTS, [p.mean() for p in points],
                          yerr=[interval(p)["high"] - p.mean() for p in points], marker="o", label=condition)
        axis.set(title=split, xlabel="Verified feedback events", xticks=CHECKPOINTS, ylim=(-0.03, 1.03))
    axes[0].set_ylabel("Held-out accuracy")
    axes[1].legend()
    fig.savefig(root / "online-feedback.pdf")
    fig.savefig(root / "online-feedback.png", dpi=180)
    plt.close(fig)
    for contrast in contrasts:
        print(json.dumps(contrast))


if __name__ == "__main__":
    main()
