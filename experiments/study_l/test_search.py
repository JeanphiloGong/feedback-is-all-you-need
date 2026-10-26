"""Check complete-string exclusion and execution boundaries independently."""

from itertools import product
import math
import unittest

from experiments.study_i.search import rank_programs, search
from experiments.study_l.feedback import CONTEXTS
from experiments.study_l.search import advance_search, rank_unchecked


def new_state():
    return dict(status='running', primitive_ops=0, comparisons=0,
                candidates=0, rejected=[], pending=None)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.primitives = [[(state + offset) % 16 for state in range(16)] for offset in (1, 2, 3, 4)]

    def test_excluded_strings_keep_descendants_and_exact_rank(self):
        table = [[.1, .2, .3, .4] if i % 2 else [.4, .3, .2, .1] for i in range(21)]
        excluded = {(), (0,), (3, 2), (1, 2, 3)}
        actual, details = rank_unchecked(table, excluded, 10000, max_length=3)
        candidates = [p for n in range(4) for p in product(range(4), repeat=n) if p not in excluded]

        def cost(program):
            return sum(round(-math.log(.9 * table[CONTEXTS.index(program[max(0, i - 2):i])][a]
                                        + .025) * 1000000) for i, a in enumerate(program))

        self.assertEqual(actual, sorted(candidates, key=lambda p: (cost(p), p)))
        self.assertIn((0, 0), actual)
        self.assertEqual(details['minimum_primitive_ops'], sum(map(len, candidates)))

    def test_partial_candidate_continues_without_repeated_execution(self):
        state = new_state()
        examples = [(0, 2), (1, 3)]
        events = list(advance_search(self.primitives, examples, [(0, 0)], state, 1))
        self.assertEqual(events[-1]['pending']['operation_index'], 1)
        self.assertEqual(state['rejected'], [])
        events = list(advance_search(self.primitives, examples, [], state, 3))
        self.assertEqual(events[-1]['pending']['example_index'], 1)
        self.assertEqual(events[-1]['pending']['operation_index'], 1)
        self.assertEqual(state['comparisons'], 1)
        events = list(advance_search(self.primitives, examples, [], state, 4))
        self.assertEqual(events[-1]['status'], 'accepted')
        self.assertEqual(state['primitive_ops'], 4)
        self.assertEqual(state['comparisons'], 2)
        self.assertEqual(state['candidates'], 1)

    def test_mismatch_at_boundary_enters_loss_but_partial_does_not(self):
        state = new_state()
        list(advance_search(self.primitives, [(0, 0)], [(0, 1)], state, 1))
        self.assertEqual(state['rejected'], [])
        events = list(advance_search(self.primitives, [(0, 0)], [], state, 2))
        self.assertEqual([e['status'] for e in events], ['rejected', 'boundary'])
        self.assertEqual(events[0]['output'], 3)
        self.assertEqual(state['rejected'], [(0, 1)])
        self.assertIsNone(state['pending'])

    def test_empty_program_has_free_comparisons(self):
        state = new_state()
        events = list(advance_search(self.primitives, [(0, 1)], [(), (0,)], state, 1))
        self.assertEqual([e['status'] for e in events], ['rejected', 'accepted'])
        self.assertEqual(state['primitive_ops'], 1)
        self.assertEqual(state['comparisons'], 2)
        state = new_state()
        events = list(advance_search(self.primitives, [(0, 0), (1, 1)], [()], state, 1))
        self.assertEqual(events[-1]['status'], 'accepted')
        self.assertEqual(state['primitive_ops'], 0)

    def test_exhaustion_at_rejection_boundary_does_not_select_a_stale_candidate(self):
        examples, programs = [(0, 0), (1, 2)], [(), (0,), (1,)]
        reference = search(self.primitives, examples, programs, budget=1)
        state = new_state()
        events = list(advance_search(self.primitives, examples, programs, state, 1))
        self.assertEqual(reference['status'], 'exhausted')
        self.assertEqual(state['status'], 'boundary')
        self.assertIsNone(state['pending'])
        self.assertEqual(state['primitive_ops'], reference['primitive_ops'])
        self.assertEqual(state['comparisons'], reference['comparisons'])
        self.assertEqual(len(state['rejected']), reference['rejected'])
        self.assertEqual(state['candidates'] + 1, reference['candidates'])
        self.assertEqual(events[-1]['status'], 'boundary')

    def test_no_update_matches_static_search_across_many_interruptions(self):
        table = [[.25] * 4 for _ in range(21)]
        examples = [(0, 13), (7, 4), (15, 12)]
        order, _ = rank_programs(lambda _: [.25] * 4, budget=1024)
        expected = search(self.primitives, examples, order, budget=1024)
        state, completed = new_state(), []
        for limit in range(7, 1025, 7):
            excluded = set(state['rejected'])
            if state['pending'] is not None:
                excluded.add(state['pending']['program'])
            ranked, _ = rank_unchecked(table, excluded, limit - state['primitive_ops'])
            events = list(advance_search(self.primitives, examples, ranked, state, limit))
            completed.extend(tuple(e['program']) for e in events if e['status'] != 'boundary')
            if state['status'] == 'accepted':
                break
        self.assertEqual(state['status'], 'accepted')
        self.assertEqual(state['primitive_ops'], expected['primitive_ops'])
        self.assertEqual(state['pending']['program'], expected['candidate_program'])
        self.assertEqual(state['comparisons'], expected['comparisons'])
        self.assertEqual(len(completed), len(set(completed)))


if __name__ == '__main__':
    unittest.main()
