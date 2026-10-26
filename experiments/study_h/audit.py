"""Independent full-output and exhaustive-segmentation audit of Study H runs."""

import hashlib
from itertools import count, permutations, product
import json

from experiments.study_h.run import ARMS, BUDGET, DATA_PATH, source_hashes
from experiments.study_h.tasks import DEVELOPMENT_SEED, FORMAL_SEEDS, make_environment, rng


def reference_search(primitives, examples, library, budget):
    """Infer early-exit costs from complete candidate outputs, not loop counters."""
    vocabulary = [(i,) for i in range(len(primitives))] + [tuple(f) for f in library]
    used = comparisons = rejected = candidates = 0
    for length in count():
        for tokens in product(range(len(vocabulary)), repeat=length):
            program = tuple(op for token in tokens for op in vocabulary[token])
            candidates += 1
            predictions = [state for state, _ in examples]
            for op in program:
                predictions = [primitives[op][state] for state in predictions]
            first_error = next((i for i, (value, (_, target)) in enumerate(zip(predictions, examples))
                                if value != target), len(examples))
            checked = min(first_error + 1, len(examples))
            required = len(program) * checked
            if used + required > budget:
                complete, partial = divmod(budget - used, len(program))
                state = examples[complete][0]
                for op in program[:partial]:
                    state = primitives[op][state]
                return dict(status="exhausted", candidate_tokens=list(tokens), candidate_program=list(program),
                            candidates=candidates, rejected=rejected, comparisons=comparisons + complete,
                            primitive_ops=budget, example_index=complete,
                            operation_index=partial, terminal_state=state)
            used += required
            comparisons += checked
            if first_error == len(examples):
                return dict(status="accepted", candidate_tokens=list(tokens), candidate_program=list(program),
                            candidates=candidates, rejected=rejected, comparisons=comparisons,
                            primitive_ops=used, example_index=len(examples) - 1,
                            operation_index=len(program), terminal_state=predictions[-1])
            rejected += 1


def brute_length(program, library):
    if not program:
        return 0
    lengths = []
    for cuts in product((False, True), repeat=len(program) - 1):
        boundaries = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [len(program)]
        pieces = [program[a:b] for a, b in zip(boundaries, boundaries[1:])]
        if all(len(piece) == 1 or piece in library for piece in pieces):
            lengths.append(len(pieces))
    return min(lengths)


def reference_compression(programs):
    candidates = {program[a:b] for program in programs for a in range(len(program))
                  for b in (a + 2, a + 3) if b <= len(program)}
    library, steps, evaluations = [], [], 0
    current = sum(map(len, programs))
    for _ in range(4):
        scored = []
        for candidate in candidates.difference(library):
            proposed = library + [candidate]
            cost = sum(len(f) + 1 for f in proposed) + sum(brute_length(p, proposed) for p in programs)
            scored.append((cost, candidate))
        evaluations += len(scored)
        if not scored:
            break
        cost, fragment = min(scored)
        if cost >= current:
            break
        library.append(fragment)
        steps.append(dict(fragment=list(fragment), before=current, after=cost))
        current = cost
    return library, dict(steps=steps, candidate_evaluations=evaluations,
                         original_tokens=sum(map(len, programs)),
                         definition_tokens=sum(len(f) + 1 for f in library),
                         encoded_corpus_tokens=sum(brute_length(p, library) for p in programs),
                         objective=current)


def audit_run(folder):
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["seeds"] in ([DEVELOPMENT_SEED], list(FORMAL_SEEDS)), "Incomplete fixed seed grid"
    assert metadata["source_sha256"] == source_hashes()
    assert metadata["data_sha256"] == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert metadata["budget"] == BUDGET and metadata["arms"] == list(ARMS)
    environments = json.loads((folder / "environments.json").read_text())
    source_data = json.loads(DATA_PATH.read_text())
    assert environments == [e for e in source_data if e["seed"] in metadata["seeds"]]
    assert [e["seed"] for e in environments] == metadata["seeds"]
    rows = [json.loads(line) for line in (folder / "searches.jsonl").read_text().splitlines()]
    indexed = {(r["seed"], r["family"], r["task_id"], r["arm"]): r for r in rows}
    assert len(indexed) == len(rows)
    evaluations = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
    scored = {(r["seed"], r["family"], r["task_id"], r["arm"]): r for r in evaluations}
    assert len(scored) == len(evaluations) and set(scored) == set(indexed)
    for key, row in indexed.items():
        row["evaluation"] = scored[key]["evaluation"]
        row["evaluation_seconds"] = scored[key]["evaluation_seconds"]
    expected_keys, memories = set(), {}
    for environment in environments:
        seed, primitives = environment["seed"], environment["primitives"]
        assert environment == json.loads(json.dumps(make_environment(seed))), "Frozen task generation mismatch"
        assert len(primitives) == 4 and len(set(map(tuple, primitives))) == 4
        assert all(sorted(p) == list(range(16)) and p != list(range(16)) for p in primitives)
        seen = set()
        tasks = environment["tasks"]
        for family, group in tasks.items():
            if family != "experience":
                assert len(group) == 16
                assert [len(t["generating_program"]) for t in group] == [6] * 8 + [9] * 8
            for task in group:
                assert len(task["train"]) == len(task["test"]) == 8
                table = list(range(16))
                for op in task["generating_program"]:
                    table = [primitives[op][state] for state in table]
                assert sorted(task["train"] + task["test"]) == [[i, value] for i, value in enumerate(table)]
                assert tuple(table) not in seen
                seen.add(tuple(table))
        memory = json.loads((folder / f"memory-{seed}.json").read_text())
        memories[seed] = memory
        programs = []
        for task in tasks["experience"]:
            result = indexed[(seed, "experience", task["id"], "none")]["result"]
            if result["status"] == "accepted":
                programs.append(tuple(result["candidate_program"]))
        compressed, detail = reference_compression(programs)
        assert memory["compression"] == detail
        assert memory["accepted_program_count"] == len(programs)
        mapping = rng(seed, "library_renaming").choice(list(permutations(range(4)))[1:])
        expected = dict(none=[], whole=sorted({p for p in programs if len(p) > 1}), compressed=compressed,
                        renamed=[tuple(mapping[op] for op in f) for f in compressed])
        assert memory["renaming"] == list(mapping)
        assert memory["libraries"] == json.loads(json.dumps(expected))
        assert all(value >= 0 for value in memory["seconds"].values())
        for family, group in tasks.items():
            for arm in (("none",) if family == "experience" else ARMS):
                for task in group:
                    key = (seed, family, task["id"], arm)
                    expected_keys.add(key)
                    row = indexed[key]
                    result = reference_search(primitives, task["train"], expected[arm], BUDGET)
                    assert row["result"] == result, key
                    if result["status"] == "accepted":
                        predictions = list(range(16))
                        for op in result["candidate_program"]:
                            predictions = [primitives[op][state] for state in predictions]
                        train = sum(predictions[state] == label for state, label in task["train"])
                        test = sum(predictions[state] == label for state, label in task["test"])
                        evaluated = dict(train_correct=train, test_correct=test, exact=train + test == 16,
                                         predictions=predictions, primitive_ops=16 * len(result["candidate_program"]))
                    else:
                        evaluated = dict(train_correct=0, test_correct=0, exact=False,
                                         predictions=None, primitive_ops=0)
                    assert row["evaluation"] == evaluated, key
                    assert row["search_seconds"] >= 0 and row["evaluation_seconds"] >= 0
        print(f"Audited environment {seed}", flush=True)
    assert expected_keys == set(indexed)
    paths = [folder / name for name in ("metadata.json", "environments.json", "searches.jsonl", "evaluations.jsonl")]
    paths.extend(folder / f"memory-{seed}.json" for seed in metadata["seeds"])
    audit = dict(status="passed", seeds=metadata["seeds"], source_sha256=metadata["source_sha256"],
                 data_sha256=metadata["data_sha256"], searches=len(rows),
                 input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return rows, memories, audit
