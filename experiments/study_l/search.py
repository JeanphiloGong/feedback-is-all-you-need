"""Re-rank unchecked programs and continue interrupted visible checks exactly."""

import heapq
import math


def rank_unchecked(probabilities, excluded, budget, max_length=9):
    frontier, ordered = [(0, ())], []
    minimum_ops = predictions = positions = children = 0
    maximum_heap = 1
    while frontier:
        cost, program = heapq.heappop(frontier)
        if program not in excluded:
            ordered.append(program)
            minimum_ops += len(program)
            if minimum_ops > budget:
                break
        if len(program) == max_length:
            continue
        context = program[-2:]
        index = (0 if not context else 1 + context[0] if len(context) == 1
                 else 5 + context[0] * 4 + context[1])
        for operation, probability in enumerate(probabilities[index]):
            edge = round(-math.log(.9 * probability + .025) * 1000000)
            heapq.heappush(frontier, (cost + edge, program + (operation,)))
            children += 1
        predictions += 1
        positions += len(program) + 1
        maximum_heap = max(maximum_heap, len(frontier))
    return ordered, dict(candidates=len(ordered), minimum_primitive_ops=minimum_ops,
        predictions=predictions, prefix_positions=positions, children=children,
        maximum_heap_entries=maximum_heap)


def advance_search(primitives, examples, programs, state, limit):
    """Mutate one task's state and yield every completed check or boundary."""
    candidates = iter(programs)
    state['status'] = 'running'
    while True:
        if state['pending'] is None:
            if state['primitive_ops'] == limit:
                state['status'] = 'boundary'
                yield dict(status='boundary', pending=None, primitive_ops=limit,
                           comparisons=state['comparisons'])
                return
            program = next(candidates, None)
            if program is None:
                state['status'] = 'space_exhausted'
                yield dict(status='space_exhausted', primitive_ops=state['primitive_ops'],
                           comparisons=state['comparisons'])
                return
            state['candidates'] += 1
            state['pending'] = dict(program=program, example_index=0,
                                    operation_index=0, value=examples[0][0])
        pending = state['pending']
        program, example = pending['program'], pending['example_index']
        while pending['operation_index'] < len(program):
            if state['primitive_ops'] == limit:
                state['status'] = 'boundary'
                yield dict(status='boundary', pending=dict(pending), primitive_ops=limit,
                           comparisons=state['comparisons'])
                return
            operation = program[pending['operation_index']]
            pending['value'] = primitives[operation][pending['value']]
            pending['operation_index'] += 1
            state['primitive_ops'] += 1
        state['comparisons'] += 1
        wrong = pending['value'] != examples[example][1]
        if wrong or example == len(examples) - 1:
            event = dict(status='rejected' if wrong else 'accepted', program=program,
                example_index=example, output=pending['value'], target=examples[example][1],
                primitive_ops=state['primitive_ops'], comparisons=state['comparisons'])
            if wrong:
                state['rejected'].append(program)
                state['pending'] = None
            else:
                state['status'] = 'accepted'
            yield event
            if not wrong:
                return
        else:
            pending.update(example_index=example + 1, operation_index=0,
                           value=examples[example + 1][0])
