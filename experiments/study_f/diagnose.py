"""Post-hoc transition descriptions; never alter Study F scores or memories."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.study_b.tasks import COMMANDS, symbolic_offsets


def transition(before, after, examples):
    truth = dict(zip(COMMANDS, symbolic_offsets(examples)))
    result = dict(before_valid=before is not None, after_valid=after is not None,
                  before_exact=before == truth, after_exact=after == truth,
                  first_failed_command=None, failed_offset_changed=None,
                  failed_offset_corrected=None, gained_components=None, lost_components=None)
    if before is not None and before != truth:
        command = next(q["commands"][0] for q in examples
                       if before[q["commands"][0]] != truth[q["commands"][0]])
        result["first_failed_command"] = command
        if after is not None:
            result["failed_offset_changed"] = after[command] != before[command]
            result["failed_offset_corrected"] = after[command] == truth[command]
    if before is not None and after is not None:
        result["gained_components"] = sum(before[c] != truth[c] and after[c] == truth[c] for c in COMMANDS)
        result["lost_components"] = sum(before[c] == truth[c] and after[c] != truth[c] for c in COMMANDS)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_files = {}
    detail, summary = [], []
    for size in ("0.5B", "7B"):
        root = args.input / f"formal-{size}-v1"
        for name in ("environments.json", "proposals.jsonl", "checks.jsonl"):
            source_files[f"formal-{size}-v1/{name}"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        environments = json.loads((root / "environments.json").read_text())
        proposals = {(r["seed"], r["arm"], r["stage"]): r for r in map(json.loads, (root / "proposals.jsonl").read_text().splitlines())}
        checks = {(r["seed"], r["arm"], r["stage"]): r for r in map(json.loads, (root / "checks.jsonl").read_text().splitlines())}
        for arm in ("self_review", "checked_review"):
            records = []
            cycles = 0
            for environment in environments:
                seed = environment["seed"]
                keys = [(seed, "initial", 0), (seed, arm, 1), (seed, arm, 2)]
                texts = [proposals[k]["text"] for k in keys]
                cycles += int(texts[0] == texts[2] and texts[0] != texts[1])
                for stage in (1, 2):
                    before, after = [checks[k]["rule"] for k in keys[stage - 1:stage + 1]]
                    record = dict(size=size, seed=seed, arm=arm, stage=stage,
                                  repeated_text=texts[stage - 1] == texts[stage],
                                  **transition(before, after, environment["train"]))
                    records.append(record)
            valid = [r for r in records if r["before_valid"] and r["after_valid"]]
            failed = [r for r in records if r["first_failed_command"] is not None]
            result = dict(size=size, arm=arm, environments=len(environments), transitions=len(records),
                repeated_text=sum(r["repeated_text"] for r in records),
                repeated_final_text_environments=sum(r["repeated_text"] for r in records if r["stage"] == 2),
                return_to_initial_text_after_change=cycles,
                repeated_invalid=sum(r["repeated_text"] and not r["before_valid"] for r in records),
                repeated_valid_wrong=sum(r["repeated_text"] and r["before_valid"] and not r["before_exact"] for r in records),
                repeated_exact=sum(r["repeated_text"] and r["before_exact"] for r in records),
                valid_to_invalid=sum(r["before_valid"] and not r["after_valid"] for r in records),
                invalid_to_valid=sum(not r["before_valid"] and r["after_valid"] for r in records),
                valid_pairs=len(valid), gained_components=sum(r["gained_components"] for r in valid),
                lost_components=sum(r["lost_components"] for r in valid),
                transitions_with_gain=sum(r["gained_components"] > 0 for r in valid),
                transitions_with_loss=sum(r["lost_components"] > 0 for r in valid),
                previously_failed_valid_rules=len(failed), failed_rule_to_invalid=sum(not r["after_valid"] for r in failed),
                failed_offset_changed=sum(r["failed_offset_changed"] is True for r in failed),
                failed_offset_corrected=sum(r["failed_offset_corrected"] is True for r in failed),
                previously_exact=sum(r["before_exact"] for r in records),
                exact_retained=sum(r["before_exact"] and r["after_exact"] for r in records))
            summary.append(result)
            detail.extend(records)
    output = dict(status="exploratory post-hoc; no new model calls or score changes",
                  source_sha256=source_files, analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=summary, transitions=detail)
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
