"""Exact post-hoc uniform-proposal expectations for the supplied finite grammar."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time


def match_counts():
    return dict(sorted(Counter(sum(value == position for position, value in enumerate(rule))
                               for rule in permutations(range(8), 4)).items()))


def expected_selection(initial_matches, draws, counts):
    total = sum(counts.values())
    correct = Fraction(initial_matches)
    for threshold in range(initial_matches + 1, 5):
        below = sum(count for matches, count in counts.items() if matches < threshold)
        correct += 1 - Fraction(below, total) ** draws
    exact = Fraction(1) if initial_matches == 4 else 1 - Fraction(total - counts[4], total) ** draws
    return correct / 4, exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    selection = json.loads(args.input.read_text())
    counts = match_counts()
    pure_direct, pure_exact = expected_selection(0, 17, counts)
    rows, summaries = [], []
    for size in ("0.5B", "7B"):
        initial = [r for r in selection["records"] if (r["size"], r["arm"], r["stage"]) == (size, "last_only", 0)]
        expected_direct, expected_exact = [], []
        example_executions = 0
        for row in initial:
            k = row["current_train_passes"] // 4
            direct, exact = expected_selection(k, 16, counts)
            expected_direct.append(direct)
            expected_exact.append(exact)
            executions = 16 * 16 + 16 * row["current_valid"]
            example_executions += executions
            rows.append(dict(size=size, seed=row["seed"], initial_correct_offsets=k,
                             initial_valid=row["current_valid"],
                             expected_direct=float(direct), expected_direct_fraction=str(direct),
                             exact_probability=float(exact), exact_probability_fraction=str(exact),
                             example_executions=executions))
        n = len(initial)
        mean_direct, exact_count = sum(expected_direct) / n, sum(expected_exact)
        for arm in ("last_only", "full_history"):
            final = [r for r in selection["records"] if (r["size"], r["arm"], r["stage"]) == (size, arm, 16)]
            observed_direct = sum(Fraction(r["selected_train_passes"], 16) for r in final) / n
            observed_exact = sum(r["selected_exact"] for r in final)
            summaries.append(dict(size=size, arm=arm, n=n, uniform_revision_draws=16 * n,
                initial_neural_proposals=n, reference_example_executions=example_executions,
                observed_selection_example_executions=sum(r["selection_example_executions"] for r in final),
                expected_direct=float(mean_direct), expected_direct_fraction=str(mean_direct),
                expected_exact_count=float(exact_count), expected_exact_count_fraction=str(exact_count),
                observed_selected_direct=float(observed_direct), observed_exact_count=observed_exact,
                observed_minus_expected_direct=float(observed_direct - mean_direct),
                observed_minus_expected_exact_count=float(observed_exact - exact_count)))
    source_paths = (Path("experiments/study_g/random_reference.py"), Path("notes/study-g-random-reference-plan.md"))
    output = dict(status="exploratory post-hoc exact expectations; no sampled agents or new neural inference",
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        legal_rule_count=sum(counts.values()), correct_offset_counts=counts,
        pure_uniform=dict(draws_per_environment=17, example_executions_per_environment=17 * 16,
                          expected_direct=float(pure_direct), expected_direct_fraction=str(pure_direct),
                          exact_probability=float(pure_exact), exact_probability_fraction=str(pure_exact)),
        elapsed_seconds=time.perf_counter() - started, summaries=summaries, records=rows)
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(pure_uniform=output["pure_uniform"], summaries=summaries), indent=2))


if __name__ == "__main__":
    main()
