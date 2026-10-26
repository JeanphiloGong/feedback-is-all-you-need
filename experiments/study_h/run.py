"""Collect the fixed Study H grid; preserve each completed search and failure."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from experiments.study_h.search import compress, execute, search
from experiments.study_h.tasks import DEVELOPMENT_SEED, FORMAL_SEEDS, rng


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "experiments/study_h/data/environments-v1.json"
ARMS = ("none", "whole", "compressed", "renamed")
BUDGET = 65536


def source_hashes():
    paths = sorted((ROOT / "experiments/study_h").glob("*.py"))
    paths.append(ROOT / "notes/protocol-v8-library-transfer.md")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def evaluate(primitives, task, result):
    if result["status"] != "accepted":
        return dict(train_correct=0, test_correct=0, exact=False, predictions=None, primitive_ops=0)
    program = result["candidate_program"]
    predictions = [execute(primitives, program, state) for state in range(16)]
    train = sum(predictions[state] == target for state, target in task["train"])
    test = sum(predictions[state] == target for state, target in task["test"])
    return dict(train_correct=train, test_correct=test, exact=train + test == 16,
                predictions=predictions, primitive_ops=16 * len(program))


def make_libraries(programs, seed):
    started = time.perf_counter()
    whole = sorted({program for program in programs if len(program) > 1})
    whole_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compressed, compression = compress(programs)
    compressed_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mappings = [mapping for mapping in permutations(range(4)) if mapping != (0, 1, 2, 3)]
    mapping = rng(seed, "library_renaming").choice(mappings)
    renamed = [tuple(mapping[op] for op in fragment) for fragment in compressed]
    renamed_seconds = time.perf_counter() - started
    return dict(none=[], whole=whole, compressed=compressed, renamed=renamed), dict(
        accepted_program_count=len(programs), compression=compression, renaming=mapping,
        seconds=dict(whole=whole_seconds, compressed=compressed_seconds, renamed=renamed_seconds))


def collect(environments, output):
    with (output / "searches.jsonl").open("x") as searches, (output / "evaluations.jsonl").open("x") as evaluations:
        for environment in environments:
            seed, primitives = environment["seed"], environment["primitives"]
            programs, libraries = [], {"none": []}
            for phase in ("experience", "evaluation"):
                if phase == "evaluation":
                    libraries, induction = make_libraries(programs, seed)
                    with (output / f"memory-{seed}.json").open("x") as stream:
                        stream.write(json.dumps(dict(seed=seed, libraries=libraries, **induction), indent=2) + "\n")
                families = ("experience",) if phase == "experience" else ("composition", "shifted")
                for arm in (("none",) if phase == "experience" else ARMS):
                    for family in families:
                        for task in environment["tasks"][family]:
                            key = dict(seed=seed, family=family, task_id=task["id"], arm=arm)
                            started = time.perf_counter()
                            result = search(primitives, task["train"], libraries[arm], BUDGET)
                            elapsed = time.perf_counter() - started
                            searches.write(json.dumps(dict(**key, result=result, search_seconds=elapsed)) + "\n")
                            searches.flush()
                            # Only visible-example acceptance enters persistent memory.
                            if phase == "experience" and result["status"] == "accepted":
                                programs.append(result["candidate_program"])
                            started = time.perf_counter()
                            evaluation = evaluate(primitives, task, result)
                            evaluations.write(json.dumps(dict(**key, evaluation=evaluation,
                                evaluation_seconds=time.perf_counter() - started)) + "\n")
                            evaluations.flush()
            print(f"Completed environment {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        hashes = source_hashes()
        data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        if not args.development:
            if args.development_audit is None:
                raise ValueError("Formal collection requires a passing development audit.")
            audit = json.loads(args.development_audit.read_text())
            if (audit["status"] != "passed" or audit["seeds"] != [DEVELOPMENT_SEED]
                    or audit["source_sha256"] != hashes or audit["data_sha256"] != data_hash):
                raise ValueError("Development audit must pass with the current source and frozen data.")
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        environments = [row for row in json.loads(DATA_PATH.read_text()) if row["seed"] in seeds]
        if [row["seed"] for row in environments] != seeds:
            raise ValueError("Frozen data does not contain the required ordered seed grid.")
        metadata = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        source_sha256=hashes, data_sha256=data_hash, seeds=seeds, budget=BUDGET,
                        arms=ARMS, python=sys.version, platform=platform.platform(),
                        processor=platform.processor(), argv=sys.argv,
                        cpu_model=next(line.split(":", 1)[1].strip()
                                       for line in Path("/proc/cpuinfo").read_text().splitlines()
                                       if line.startswith("model name")),
                        development_audit_sha256=(hashlib.sha256(args.development_audit.read_bytes()).hexdigest()
                                                  if args.development_audit else None))
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "environments.json").write_text(json.dumps(environments, indent=2) + "\n")
        collect(environments, args.output)
        metadata.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps(dict(
            error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
