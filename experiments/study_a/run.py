#!/usr/bin/env python3
"""Run Study A's frozen capacity-by-feedback experiment.

The runner is deliberately self-contained. It pre-generates all task and
evaluation data per seed, clones one warmed-up model state into every arm, and
writes one JSON object per checkpoint. The resulting file is suitable for
analysis without rerunning the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


INPUT_DIM = 8
BATCH_SIZE = 32
WARMUP_BATCHES = 256
ADAPTATION_BATCHES = 256
EVAL_EXAMPLES = 2048
CHECKPOINTS = tuple(range(0, ADAPTATION_BATCHES + 1, 16))
FAMILIES = ("linear", "xor")
ARMS = ("frozen", "clean", "noise25", "noise50", "inverted", "delay32", "replay25")
WIDTHS = (0, 16, 64, 256)


def stream_seed(seed: int, family: str, role: str, *details: object) -> int:
    """Derive separate reproducible PRNG streams for each experimental role."""
    encoded = json.dumps([seed, family, role, *details]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


@dataclass(frozen=True)
class Rule:
    """A deterministic binary task rule."""

    family: str
    weight: Tensor | None = None
    pair: tuple[int, int] | None = None
    invert: bool = False

    def labels(self, x: Tensor) -> Tensor:
        if self.family == "linear":
            assert self.weight is not None
            y = (x @ self.weight >= 0).float()
        elif self.family == "xor":
            assert self.pair is not None
            y = ((x[:, self.pair[0]] > 0) != (x[:, self.pair[1]] > 0)).float()
        else:
            raise ValueError(f"unknown family: {self.family}")
        return 1.0 - y if self.invert else y


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def make_rule(family: str, seed: int) -> Rule:
    generator = make_generator(seed)
    if family == "linear":
        weight = torch.randn(INPUT_DIM, generator=generator)
        weight = weight / weight.norm()
        return Rule(family=family, weight=weight)
    if family == "xor":
        pair = tuple(torch.randperm(INPUT_DIM, generator=generator)[:2].tolist())
        invert = bool(torch.randint(0, 2, (), generator=generator).item())
        return Rule(family=family, pair=pair, invert=invert)
    raise ValueError(f"unknown family: {family}")


def make_data(rule: Rule, batches: int, seed: int, distribution: str = "gaussian") -> tuple[Tensor, Tensor]:
    generator = make_generator(seed)
    if distribution == "gaussian":
        x = torch.randn(batches, BATCH_SIZE, INPUT_DIM, generator=generator)
    elif distribution == "uniform":
        bound = 3.0**0.5
        x = torch.rand(batches, BATCH_SIZE, INPUT_DIM, generator=generator) * (2 * bound) - bound
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    y = torch.stack([rule.labels(batch) for batch in x])
    return x, y


class Agent(nn.Module):
    """A logistic or one-hidden-layer binary classifier."""

    def __init__(self, width: int):
        super().__init__()
        if width == 0:
            self.network = nn.Linear(INPUT_DIM, 1)
        else:
            self.network = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


def parameter_count(width: int) -> int:
    return INPUT_DIM + 1 if width == 0 else (INPUT_DIM + 2) * width + 1


def evaluate(agent: Agent, x: Tensor, y: Tensor) -> float:
    agent.eval()
    with torch.no_grad():
        prediction = (agent(x) >= 0).float()
    return float((prediction == y).float().mean().item())


def train_batch(agent: Agent, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor) -> float:
    agent.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.binary_cross_entropy_with_logits(agent(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def noisy_labels(y: Tensor, arm: str, seed: int) -> Tensor:
    if arm == "clean" or arm == "delay32" or arm == "replay25":
        return y.clone()
    if arm == "inverted":
        return 1.0 - y
    if arm not in {"noise25", "noise50"}:
        return y.clone()
    probability = 0.25 if arm == "noise25" else 0.50
    generator = make_generator(seed)
    flip = torch.rand(y.shape, generator=generator) < probability
    return torch.where(flip, 1.0 - y, y)


def clone_agent(state: dict[str, Tensor], width: int, device: torch.device) -> Agent:
    agent = Agent(width).to(device)
    agent.load_state_dict({name: value.to(device) for name, value in state.items()})
    return agent


def update_record(
    agent: Agent,
    family: str,
    arm: str,
    width: int,
    seed: int,
    checkpoint: int,
    eval_sets: dict[str, tuple[Tensor, Tensor]],
    endpoint: str = "online",
    last_loss: float | None = None,
    updates: int = 0,
    fresh_examples: int = 0,
    replay_examples: int = 0,
    update_seconds: float = 0.0,
    evaluation_seconds: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "family": family,
        "arm": arm,
        "width": width,
        "parameters": parameter_count(width),
        "seed": seed,
        "checkpoint": checkpoint,
        "endpoint": endpoint,
        "updates": updates,
        "fresh_examples": fresh_examples,
        "replay_examples": replay_examples,
        "last_loss": last_loss,
        "update_seconds": update_seconds,
        "prior_evaluation_seconds": evaluation_seconds,
        "adaptation_flops_proxy": 6 * (INPUT_DIM if width == 0 else (INPUT_DIM + 1) * width) * (fresh_examples + replay_examples),
    }
    for name, (x, y) in eval_sets.items():
        row[f"accuracy_{name}"] = evaluate(agent, x, y)
    return row


def run_arm(
    family: str,
    arm: str,
    width: int,
    seed: int,
    warm_state: dict[str, Tensor],
    warm_x: Tensor,
    warm_y: Tensor,
    adapt_x: Tensor,
    adapt_y: Tensor,
    eval_sets: dict[str, tuple[Tensor, Tensor]],
    device: torch.device,
) -> list[dict[str, object]]:
    agent = clone_agent(warm_state, width, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    records: list[dict[str, object]] = []
    pending: list[tuple[Tensor, Tensor]] = []
    replay_generator = make_generator(stream_seed(seed, family, "replay"))
    replay_indices = torch.randint(
        0, WARMUP_BATCHES * BATCH_SIZE, (ADAPTATION_BATCHES, 8), generator=replay_generator
    )
    updates = 0
    fresh_examples = 0
    replay_examples = 0
    last_loss: float | None = None
    update_seconds = 0.0
    evaluation_seconds = 0.0

    def record(checkpoint: int) -> None:
        nonlocal evaluation_seconds
        evaluation_start = time.perf_counter()
        records.append(
            update_record(
                agent,
                family,
                arm,
                width,
                seed,
                checkpoint,
                eval_sets,
                last_loss=last_loss,
                updates=updates,
                fresh_examples=fresh_examples,
                replay_examples=replay_examples,
                update_seconds=update_seconds,
                evaluation_seconds=evaluation_seconds,
            )
        )
        evaluation_seconds += time.perf_counter() - evaluation_start

    record(0)
    for step in range(ADAPTATION_BATCHES):
        update_start = time.perf_counter()
        x = adapt_x[step]
        y = noisy_labels(adapt_y[step], arm, stream_seed(seed, family, "corruption", step))
        if arm == "frozen":
            pass
        elif arm == "replay25":
            flat_x = warm_x.reshape(-1, INPUT_DIM)
            flat_y = warm_y.reshape(-1)
            index = replay_indices[step]
            x = torch.cat([x[:24], flat_x[index]], dim=0)
            y = torch.cat([y[:24], flat_y[index]], dim=0)
            last_loss = train_batch(agent, optimizer, x.to(device), y.to(device))
            updates += 1
            fresh_examples += 24
            replay_examples += 8
        elif arm == "delay32":
            pending.append((x, y))
            if len(pending) > 32:
                delayed_x, delayed_y = pending.pop(0)
                last_loss = train_batch(agent, optimizer, delayed_x.to(device), delayed_y.to(device))
                updates += 1
                fresh_examples += BATCH_SIZE
        else:
            last_loss = train_batch(agent, optimizer, x.to(device), y.to(device))
            updates += 1
            fresh_examples += BATCH_SIZE
        update_seconds += time.perf_counter() - update_start
        checkpoint = step + 1
        if checkpoint in CHECKPOINTS:
            record(checkpoint)

    if arm == "delay32":
        update_start = time.perf_counter()
        while pending:
            delayed_x, delayed_y = pending.pop(0)
            last_loss = train_batch(agent, optimizer, delayed_x.to(device), delayed_y.to(device))
            updates += 1
            fresh_examples += BATCH_SIZE
        update_seconds += time.perf_counter() - update_start
        records.append(
            update_record(
                agent,
                family,
                arm,
                width,
                seed,
                ADAPTATION_BATCHES,
                eval_sets,
                endpoint="flushed",
                last_loss=last_loss,
                updates=updates,
                fresh_examples=fresh_examples,
                replay_examples=replay_examples,
                update_seconds=update_seconds,
                evaluation_seconds=evaluation_seconds,
            )
        )
    return records


def git_revision() -> str | None:
    configured = os.environ.get("FEEDBACK_GIT_REVISION")
    if configured:
        return configured
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    families = tuple(args.families)
    widths = tuple(args.widths)
    arms = tuple(args.arms)
    started = time.time()
    metadata = {
        "study": "A",
        "random_stream_scheme": "sha256(seed, family, role, details), first 64 bits little-endian",
        "protocol": "notes/protocol-v1.md",
        "git_revision": git_revision(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "families": families,
        "widths": widths,
        "arms": arms,
        "seeds": seeds,
        "warmup_batches": WARMUP_BATCHES,
        "adaptation_batches": ADAPTATION_BATCHES,
        "batch_size": BATCH_SIZE,
        "eval_examples": EVAL_EXAMPLES,
        "started_unix": started,
        "threads": torch.get_num_threads(),
        "feedback_semantics": "labels after prediction; predictions do not affect input sampling",
        "walltime_scope": "per-arm update time includes host preparation and transfers; loss.item synchronizes GPU; evaluation reported separately",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=list) + "\n")
    rows_path = output / "results.jsonl"
    with rows_path.open("w") as rows:
        total = len(families) * len(widths) * len(arms) * len(seeds)
        completed = 0
        for family in families:
            for seed in seeds:
                initial_rule = make_rule(family, stream_seed(seed, family, "initial_rule"))
                post_rule = make_rule(family, stream_seed(seed, family, "post_rule"))
                warm_x, warm_y = make_data(initial_rule, WARMUP_BATCHES, stream_seed(seed, family, "warmup_data"))
                adapt_x, adapt_y = make_data(post_rule, ADAPTATION_BATCHES, stream_seed(seed, family, "adaptation_data"))
                eval_sets = {
                    "post_gaussian": make_data(post_rule, EVAL_EXAMPLES // BATCH_SIZE, stream_seed(seed, family, "post_gaussian")),
                    "post_uniform": make_data(post_rule, EVAL_EXAMPLES // BATCH_SIZE, stream_seed(seed, family, "post_uniform"), "uniform"),
                    "old_gaussian": make_data(initial_rule, EVAL_EXAMPLES // BATCH_SIZE, stream_seed(seed, family, "old_gaussian")),
                    "old_uniform": make_data(initial_rule, EVAL_EXAMPLES // BATCH_SIZE, stream_seed(seed, family, "old_uniform"), "uniform"),
                }
                warm_x_device = warm_x.to(device)
                warm_y_device = warm_y.to(device)
                for width in widths:
                    torch.manual_seed(stream_seed(seed, family, "initialization", width))
                    warm_agent = Agent(width).to(device)
                    optimizer = torch.optim.Adam(warm_agent.parameters(), lr=0.01)
                    warm_start = time.perf_counter()
                    for batch_x, batch_y in zip(warm_x_device, warm_y_device):
                        train_batch(warm_agent, optimizer, batch_x, batch_y)
                    warm_seconds = time.perf_counter() - warm_start
                    warm_state = {name: value.detach().cpu().clone() for name, value in warm_agent.state_dict().items()}
                    for arm in arms:
                        for row in run_arm(
                            family,
                            arm,
                            width,
                            seed,
                            warm_state,
                            warm_x,
                            warm_y,
                            adapt_x,
                            adapt_y,
                            {name: (x.to(device), y.to(device)) for name, (x, y) in eval_sets.items()},
                            device,
                        ):
                            row["warmup_seconds"] = warm_seconds
                            row["warmup_flops_proxy"] = 6 * (INPUT_DIM if width == 0 else (INPUT_DIM + 1) * width) * WARMUP_BATCHES * BATCH_SIZE
                            row["replay_storage_bytes"] = warm_x.numel() * warm_x.element_size() + warm_y.numel() * warm_y.element_size() if arm == "replay25" else 0
                            rows.write(json.dumps(row) + "\n")
                        rows.flush()
                        completed += 1
                        if args.progress:
                            print(f"completed {completed}/{total}: family={family} seed={seed} width={width} arm={arm}", flush=True)
    metadata["finished_unix"] = time.time()
    metadata["elapsed_seconds"] = metadata["finished_unix"] - started
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=list) + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--widths", nargs="+", type=int, choices=WIDTHS, default=list(WIDTHS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
