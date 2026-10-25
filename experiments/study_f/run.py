"""Generate and retain all rule proposals before independent held-out scoring."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.study_b.run import REVISIONS
from experiments.study_f.rules import check_rule, parse_rule, proposal_messages

DATA_SHA256 = "3713e3d93f4221224e247b9c2b6454f1e4fec4df03fa8cfd93b1babf7c150215"
STAGES = (("initial", 0), ("self_review", 1), ("checked_review", 1),
          ("self_review", 2), ("checked_review", 2))
SOURCES = ("experiments/study_f/run.py", "experiments/study_f/rules.py",
           "experiments/study_f/analyze.py", "experiments/study_f/freeze_data.py",
           "experiments/study_b/run.py", "experiments/study_b/tasks.py",
           "notes/protocol-v6-rule-memory.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("0.5B", "7B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    raw = Path("experiments/study_f/data/environments-v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    seeds = [17] if args.development else list(range(8000, 8024))
    environments = [e for e in json.loads(raw)["environments"] if e["seed"] in seeds]
    assert [e["seed"] for e in environments] == seeds
    sources = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCES}
    name = f"Qwen/Qwen2.5-{args.size}-Instruct"
    packages = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scipy", "matplotlib")}
    gate = None
    if not args.development:
        if args.development_audit is None:
            raise ValueError("formal inference requires a completed development audit")
        gate = json.loads(args.development_audit.read_text())
        development = gate["metadata"]
        assert gate["passed"] and development["completed"] and development["development"]
        assert development["seeds"] == [17] and development["source_sha256"] == sources
        assert development["model"] == name and development["revision"] == REVISIONS[args.size]
        assert development["packages"] == packages
        for path, digest in gate["files_sha256"].items():
            assert hashlib.sha256((args.development_audit.parent / path).read_bytes()).hexdigest() == digest
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        (args.output / "metadata.json").write_text(json.dumps(dict(
            study="F", size=args.size, model=name, revision=REVISIONS[args.size],
            development=args.development, seeds=seeds, source_sha256=sources,
            input_environments_sha256=DATA_SHA256, packages=packages,
            git_revision=os.environ.get("FEEDBACK_GIT_REVISION"),
            preflight_unix=time.time(), completed=False), indent=2) + "\n")
        (args.output / "rules").mkdir()
        torch.set_num_threads(4)
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(name, revision=REVISIONS[args.size], local_files_only=True)
        tokenizer.padding_side = "left"
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=REVISIONS[args.size], dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        actual = (json.dumps(environments, sort_keys=True) + "\n").encode()
        (args.output / "environments.json").write_bytes(actual)
        metadata = dict(study="F", size=args.size, model=name, revision=REVISIONS[args.size],
                        development=args.development, seeds=seeds, source_sha256=sources,
                        input_environments_sha256=DATA_SHA256,
                        executed_environments_sha256=hashlib.sha256(actual).hexdigest(),
                        git_revision=os.environ.get("FEEDBACK_GIT_REVISION"), packages=packages,
                        device=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
                        batch_size=16, stages=STAGES, dtype="bfloat16", attention="sdpa",
                        generation_config=model.generation_config.to_dict(),
                        generation_overrides=dict(do_sample=False, max_new_tokens=512),
                        load_seconds=time.perf_counter() - started, started_unix=time.time(),
                        development_evidence=gate, completed=False)
        if gate:
            for field in ("dtype", "attention", "batch_size", "generation_config", "generation_overrides"):
                assert metadata[field] == gate["metadata"][field]
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        torch.cuda.reset_peak_memory_stats()
        records = {}
        with (args.output / "proposals.jsonl").open("x") as output:
            for arm, stage in STAGES:
                for start in range(0, len(environments), 16):
                    chunk = environments[start:start + 16]
                    prompts, previous_texts, feedbacks = [], [], []
                    for environment in chunk:
                        previous = None if stage == 0 else records[(environment["seed"],
                            "initial" if stage == 1 else arm, stage - 1)]
                        previous_text = None if previous is None else previous["text"]
                        feedback = previous["check"] if arm == "checked_review" else None
                        prompts.append(tokenizer.apply_chat_template(
                            proposal_messages(environment["train"], previous_text, feedback),
                            tokenize=False, add_generation_prompt=True))
                        previous_texts.append(previous_text)
                        feedbacks.append(feedback)
                    inputs = tokenizer(prompts, padding=True, add_special_tokens=False,
                                       return_tensors="pt").to(model.device)
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - began
                    tokens = generated[:, inputs["input_ids"].shape[1]:].tolist()
                    lengths = inputs["attention_mask"].sum(-1).tolist()
                    raw_rows = []
                    for environment, prompt, previous_text, feedback, sequence, length in zip(
                            chunk, prompts, previous_texts, feedbacks, tokens, lengths):
                        stop = next((i + 1 for i, token in enumerate(sequence) if token in eos), len(sequence))
                        sequence = sequence[:stop]
                        text = tokenizer.decode(sequence, skip_special_tokens=True)
                        row = dict(seed=environment["seed"], arm=arm, stage=stage,
                                   prompt=prompt, previous_text=previous_text, feedback=feedback,
                                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                   output_token_ids=sequence, text=text, input_tokens=length,
                                   output_tokens=len(sequence), eos_terminated=sequence[-1] in eos,
                                   batch_index=start // 16, batch_size=len(chunk),
                                   padded_input_tokens=int(inputs["input_ids"].numel()),
                                   padded_generation_tokens=sum(map(len, tokens)), batch_wall_seconds=elapsed)
                        raw_rows.append(row)
                    # Preserve the entire completed batch before any row can fail validation.
                    for row in raw_rows:
                        output.write(json.dumps(row) + "\n")
                    output.flush()
                    for environment, row in zip(chunk, raw_rows):
                        began = time.perf_counter()
                        rule = parse_rule(row["text"])
                        check = check_rule(rule, environment["train"])
                        check_seconds = time.perf_counter() - began
                        memory = args.output / "rules" / f"{environment['seed']}-{arm}-{stage}.json"
                        memory.write_text(json.dumps(rule, separators=(",", ":")) + "\n")
                        row.update(rule=rule, check=check)
                        records[(environment["seed"], arm, stage)] = row
                        with (args.output / "checks.jsonl").open("a") as checks:
                            checks.write(json.dumps(dict(seed=environment["seed"], arm=arm, stage=stage,
                                rule=rule, check=check, check_seconds=check_seconds,
                                memory_bytes=memory.stat().st_size)) + "\n")
                    print(args.size, arm, stage, start, "complete", flush=True)
        metadata.update(completed=True, finished_unix=time.time(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
        metadata["elapsed_seconds"] = metadata["finished_unix"] - metadata["started_unix"]
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(
            error_type=type(error).__name__, message=str(error), unix=time.time()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
