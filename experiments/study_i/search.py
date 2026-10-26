"""Finite prior-ranked search with explicitly charged primitive execution."""

import heapq
import math


def token_counts(programs):
    counts = [[[0] * 4 for _ in range(5)] for _ in range(5)]
    for program in programs:
        previous, last = 4, 4
        for operation in program:
            counts[previous][last][operation] += 1
            previous, last = last, operation
    return counts


def count_probabilities(counts, prefix):
    previous, last = ((4, 4) + tuple(prefix))[-2:]
    row = counts[previous][last]
    return [(value + 1) / (sum(row) + 4) for value in row]


def rank_programs(probabilities, budget=65536, max_length=9):
    """Rank by rounded log costs; retain enough candidates for any task."""
    heap = [(0, ())]
    programs = []
    minimum_ops = predictions = prefix_positions = children = 0
    maximum_heap = 1
    while heap:
        score, program = heapq.heappop(heap)
        programs.append(program)
        minimum_ops += len(program)
        if minimum_ops > budget:
            break
        if len(program) == max_length:
            continue
        row = probabilities(program)
        predictions += 1
        prefix_positions += len(program) + 1
        for operation, probability in enumerate(row):
            edge = round(-math.log(0.9 * probability + 0.025) * 1000000)
            heapq.heappush(heap, (score + edge, program + (operation,)))
            children += 1
        maximum_heap = max(maximum_heap, len(heap))
    return programs, dict(candidates=len(programs), minimum_primitive_ops=minimum_ops,
                         predictions=predictions, prefix_positions=prefix_positions,
                         children=children, maximum_heap_entries=maximum_heap)


def search(primitives, examples, programs, budget=65536):
    """Accept the first visible fit; an incomplete check is never accepted."""
    rejected = comparisons = operations = 0
    for candidate, program in enumerate(programs, 1):
        for example_index, (start, target) in enumerate(examples):
            state = start
            for operation_index, operation in enumerate(program):
                if operations == budget:
                    return dict(status="exhausted", candidate_program=program,
                                candidates=candidate, rejected=rejected,
                                comparisons=comparisons, primitive_ops=operations,
                                example_index=example_index,
                                operation_index=operation_index, terminal_state=state)
                state = primitives[operation][state]
                operations += 1
            comparisons += 1
            if state != target:
                rejected += 1
                break
        else:
            return dict(status="accepted", candidate_program=program,
                        candidates=candidate, rejected=rejected,
                        comparisons=comparisons, primitive_ops=operations,
                        example_index=example_index, operation_index=len(program),
                        terminal_state=state)
    return dict(status="space_exhausted", candidate_program=None,
                candidates=len(programs), rejected=rejected,
                comparisons=comparisons, primitive_ops=operations)
