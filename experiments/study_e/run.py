"""Run a fixed online-feedback stream with context or persistent LoRA state."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.study_b.run import REVISIONS, inputs, score, text_prompt
from experiments.study_b.tasks import make_environment

CHECKPOINTS = (0, 1, 2, 4, 8, 16)
DATA_SHA256 = "67fd60a700203034dccb8e79fd95c119bb1315f08d69f040ae2b0ae24784b0c9"
SOURCES = ("experiments/study_e/run.py", "experiments/study_b/run.py",
           "experiments/study_b/tasks.py", "experiments/study_e/freeze_data.py")


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def attach_adapter(base, seed):
    from peft import LoraConfig, TaskType, get_peft_model
    torch.manual_seed(seed + 4_000_000)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
        revision=REVISIONS["0.5B"],
    ))
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert all("lora_" in name for name, _ in trainable)
    assert all(torch.count_nonzero(p) == 0 for name, p in trainable if "lora_B" in name)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-4, weight_decay=0)
    return model, optimizer, trainable


def update_once(model, tokenizer, digit_ids, query, optimizer):
    prompt = text_prompt(tokenizer, query, [])
    batch = inputs(tokenizer, [prompt], model.device)
    target = torch.tensor([digit_ids[query["target"]]], device=model.device)
    model.train()
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits.float(), target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return dict(loss=float(loss.detach()),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_tokens=int(batch["attention_mask"].sum()),
                padded_input_tokens=int(batch["input_ids"].numel()),
                wall_seconds=time.perf_counter() - started)


def evaluate(model, tokenizer, digit_ids, environment, condition, count, history, output):
    records = []
    for split in ("direct", "compose"):
        queries = environment[split]
        for start in range(0, len(queries), 16):
            rows = score(model, tokenizer, digit_ids, queries[start:start + 16], history, 16)
            for index, row in enumerate(rows, start):
                row["batch_index"] = start // 16
                record = dict(seed=environment["seed"], condition=condition,
                              feedback_count=count, split=split, index=index, **row)
                output.write(json.dumps(record) + "\n")
                records.append(record)
            output.flush()
    return records


def run_feedback(base, tokenizer, digit_ids, environment, condition, output, training,
                 adapter_root, baseline):
    model = base
    initial_hash = None
    if condition != "context":
        model, optimizer, trainable = attach_adapter(base, environment["seed"])
        initial_hash = tensor_hash(trainable)
    initial = evaluate(model, tokenizer, digit_ids, environment, condition, 0, [], output)
    for row, reference in zip(initial, baseline):
        assert row["prediction"] == reference["prediction"]
        assert row["unconstrained_token"] == reference["unconstrained_token"]
    history = []
    applied_updates = 0
    for step, query in enumerate(environment["train"]):
        prediction = score(model, tokenizer, digit_ids, [query], history, 1)[0]
        identity = dict(seed=environment["seed"], condition=condition, step=step)
        # Persist the complete pre-update prediction even if the next update fails.
        training.write(json.dumps(dict(event="prediction", **identity, **prediction)) + "\n")
        training.flush()
        apply_update = condition == "online_lora" or (
            condition == "error_only_lora" and not prediction["correct"])
        update = update_once(model, tokenizer, digit_ids, query, optimizer) if apply_update else None
        applied_updates += int(apply_update)
        training.write(json.dumps(dict(event="feedback", **identity, target=query["target"],
                                       update_applied=apply_update, update=update)) + "\n")
        training.flush()
        if condition == "context":
            history.append(dict(query))
        if step + 1 in CHECKPOINTS[1:]:
            evaluate(model, tokenizer, digit_ids, environment, condition, step + 1, history, output)
    accounting = dict(seed=environment["seed"], condition=condition, applied_updates=applied_updates)
    if condition != "context":
        evaluate(model, tokenizer, digit_ids, environment, condition + "_with_context", 16,
                 environment["train"], output)
        save = adapter_root / str(environment["seed"]) / condition
        save.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(save, save_embedding_layers=False)
        accounting.update(initial_adapter_sha256=initial_hash,
                          initialization_seed=environment["seed"] + 4_000_000,
                          trainable_parameters=sum(p.numel() for _, p in trainable),
                          adapter_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in save.iterdir() if p.is_file()})
        base = model.unload()
        del model, optimizer, trainable
        torch.cuda.empty_cache()
    return base, accounting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    raw = Path("experiments/study_e/data/environments-v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    environments = [make_environment(13)] if args.development else json.loads(raw)["environments"]
    assert [e["seed"] for e in environments] == ([13] if args.development else list(range(7000, 7024)))
    sources = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCES}
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    packages = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "peft")}
    development_evidence = None
    if not args.development:
        if args.development_audit is None:
            raise ValueError("formal inference requires a completed development audit")
        audit = json.loads(args.development_audit.read_text())
        development = audit["metadata"]
        assert audit["passed"] and development["completed"] and development["development"]
        assert development["seeds"] == [13] and development["source_sha256"] == sources
        assert development["model"] == name and development["revision"] == REVISIONS["0.5B"]
        assert development["packages"] == packages and development["batch_size"] == 16
        assert audit["analysis_sha256"] == hashlib.sha256(Path("experiments/study_e/analyze.py").read_bytes()).hexdigest()
        for path, digest in audit["files_sha256"].items():
            assert hashlib.sha256((args.development_audit.parent / path).read_bytes()).hexdigest() == digest
        development_evidence = audit
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=REVISIONS["0.5B"], local_files_only=True)
    tokenizer.padding_side = "left"
    digit_ids = []
    for digit in range(8):
        ids = tokenizer.encode(str(digit), add_special_tokens=False)
        assert len(ids) == 1
        digit_ids.append(ids[0])
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=REVISIONS["0.5B"], dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    actual = (json.dumps(environments, sort_keys=True) + "\n").encode()
    (args.output / "environments.json").write_bytes(actual)
    metadata = dict(study="E", model=name, revision=REVISIONS["0.5B"], development=args.development,
                    seeds=[e["seed"] for e in environments], source_sha256=sources,
                    input_environments_sha256=hashlib.sha256(raw).hexdigest(),
                    executed_environments_sha256=hashlib.sha256(actual).hexdigest(),
                    git_revision=os.environ.get("FEEDBACK_GIT_REVISION"), packages=packages,
                    device=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
                    batch_size=16, checkpoints=list(CHECKPOINTS), digit_token_ids=digit_ids,
                    dtype="bfloat16", attention="sdpa", generation_interface="constrained first-token digit argmax",
                    generation_config=model.generation_config.to_dict(),
                    load_seconds=time.perf_counter() - started, started_unix=time.time(),
                    base_initial_sha256=tensor_hash(model.named_parameters()),
                    development_evidence=development_evidence, completed=False, accounting=[])
    if development_evidence:
        for field in ("dtype", "attention", "generation_interface", "generation_config", "base_initial_sha256"):
            assert metadata[field] == development_evidence["metadata"][field]
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "predictions.jsonl").open("x") as output, (args.output / "training.jsonl").open("x") as training:
        for environment in environments:
            baseline = evaluate(model, tokenizer, digit_ids, environment, "none", 0, [], output)
            for condition in ("context", "online_lora", "error_only_lora"):
                model, record = run_feedback(model, tokenizer, digit_ids, environment, condition,
                                             output, training, args.output / "adapters", baseline)
                metadata["accounting"].append(record)
                if args.development:
                    record["base_after_sha256"] = tensor_hash(model.named_parameters())
                    assert record["base_after_sha256"] == metadata["base_initial_sha256"]
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(environment["seed"], "completed", flush=True)
    metadata["base_final_sha256"] = tensor_hash(model.named_parameters())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert metadata["base_final_sha256"] == metadata["base_initial_sha256"]
    metadata.update(finished_unix=time.time(), peak_allocated_bytes=torch.cuda.max_memory_allocated(), completed=True)
    metadata["elapsed_seconds"] = metadata["finished_unix"] - metadata["started_unix"]
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
