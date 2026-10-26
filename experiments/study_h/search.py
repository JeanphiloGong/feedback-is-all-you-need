"""Fixed enumeration and flat fragment induction for the Study H protocol."""

from itertools import count, product


def execute(primitives, program, state):
    for operation in program:
        state = primitives[operation][state]
    return state


def search(primitives, examples, library, budget=65536):
    """Accept the first visible-example fit, charging expanded primitive calls."""
    vocabulary = [(i,) for i in range(len(primitives))] + list(library)
    candidates = rejected = comparisons = operations = 0
    for length in count():
        for tokens in product(range(len(vocabulary)), repeat=length):
            program = tuple(op for token in tokens for op in vocabulary[token])
            candidates += 1
            for example_index, (start, target) in enumerate(examples):
                state = start
                for operation_index, op in enumerate(program):
                    if operations == budget:
                        return dict(status="exhausted", candidate_tokens=tokens,
                                    candidate_program=program, candidates=candidates,
                                    rejected=rejected, comparisons=comparisons,
                                    primitive_ops=operations, example_index=example_index,
                                    operation_index=operation_index, terminal_state=state)
                    state = primitives[op][state]
                    operations += 1
                comparisons += 1
                if state != target:
                    rejected += 1
                    break
            else:
                return dict(status="accepted", candidate_tokens=tokens,
                            candidate_program=program, candidates=candidates,
                            rejected=rejected, comparisons=comparisons,
                            primitive_ops=operations, example_index=example_index,
                            operation_index=len(program), terminal_state=state)


def encoded_length(program, library):
    """Minimum number of primitive/library tokens encoding a flat program."""
    best = [0] + [len(program) + 1] * len(program)
    for end in range(1, len(program) + 1):
        best[end] = best[end - 1] + 1
        for fragment in library:
            start = end - len(fragment)
            if start >= 0 and program[start:end] == fragment:
                best[end] = min(best[end], best[start] + 1)
    return best[-1]


def description_length(programs, library):
    return (sum(len(fragment) + 1 for fragment in library)
            + sum(encoded_length(program, library) for program in programs))


def compress(programs):
    """Greedily reduce the declared token objective with at most four macros."""
    candidates = sorted({program[start:start + length] for program in programs
                         for length in (2, 3)
                         for start in range(len(program) - length + 1)})
    library, steps, evaluations = [], [], 0
    current = description_length(programs, library)
    for _ in range(4):
        best_score, chosen = current, None
        for fragment in candidates:
            if fragment in library:
                continue
            score = description_length(programs, library + [fragment])
            evaluations += 1
            if score < best_score:
                best_score, chosen = score, fragment
        if chosen is None:
            break
        library.append(chosen)
        steps.append(dict(fragment=chosen, before=current, after=best_score))
        current = best_score
    return library, dict(steps=steps, candidate_evaluations=evaluations,
                         original_tokens=sum(map(len, programs)),
                         definition_tokens=sum(len(f) + 1 for f in library),
                         encoded_corpus_tokens=sum(encoded_length(p, library) for p in programs),
                         objective=current)
