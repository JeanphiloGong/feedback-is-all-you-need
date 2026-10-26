import itertools
import math
import random
import unittest

from experiments.study_i.search import count_probabilities, rank_programs, search, token_counts


class PriorSearchTests(unittest.TestCase):
    def test_heap_matches_complete_independent_sort(self):
        def probabilities(prefix):
            values = [1 + ((sum(prefix) + operation * 3) % 7) for operation in range(4)]
            return [value / sum(values) for value in values]

        def score(program):
            return sum(round(-math.log(0.9 * probabilities(program[:i])[op] + 0.025)
                             * 1000000) for i, op in enumerate(program))

        all_programs = [p for length in range(5)
                        for p in itertools.product(range(4), repeat=length)]
        expected = sorted(all_programs, key=lambda p: (score(p), p))
        actual, meta = rank_programs(probabilities, budget=100000, max_length=4)
        self.assertEqual(actual, expected)
        self.assertEqual(meta['minimum_primitive_ops'], sum(map(len, all_programs)))
        self.assertEqual(meta['predictions'], 1 + 4 + 16 + 64)

    def test_uniform_and_lower_bound_cover_every_allowed_search(self):
        order, meta = rank_programs(lambda _: [0.25] * 4, budget=20, max_length=9)
        expected = [p for n in range(3) for p in itertools.product(range(4), repeat=n)]
        self.assertEqual(order, expected[:len(order)])
        self.assertGreater(meta['minimum_primitive_ops'], 20)
        self.assertLessEqual(sum(map(len, order[:-1])), 20)
        result = search([list(range(2))] * 4, [(0, 1)], order, budget=20)
        self.assertEqual(result['status'], 'exhausted')
        self.assertEqual(result['primitive_ops'], 20)

    def test_counts_keep_prefix_context_and_laplace_smoothing(self):
        counts = token_counts([(0, 1, 2), (0, 1, 3), (1,)])
        self.assertEqual(sum(v for plane in counts for row in plane for v in row), 7)
        self.assertEqual(count_probabilities(counts, ()), [3/7, 2/7, 1/7, 1/7])
        self.assertEqual(count_probabilities(counts, (3, 0, 1)), [1/6, 1/6, 2/6, 2/6])
        self.assertEqual(count_probabilities(counts, (3, 3)), [0.25] * 4)

    def test_exact_cap_and_partial_example(self):
        primitives = [(1, 0), (0, 1), (0, 1), (0, 1)]
        order = [(), (0,)]
        complete = search(primitives, [(0, 1), (1, 0)], order, budget=2)
        self.assertEqual(complete['status'], 'accepted')
        self.assertEqual(complete['primitive_ops'], 2)
        interrupted = search(primitives, [(0, 1), (1, 0)], order, budget=1)
        self.assertEqual(interrupted['status'], 'exhausted')
        self.assertEqual(interrupted['example_index'], 1)
        self.assertEqual(interrupted['operation_index'], 0)
        self.assertEqual(interrupted['comparisons'], 2)
        identity = search(primitives, [(0, 0)], order, budget=0)
        self.assertEqual(identity['status'], 'accepted')
        self.assertEqual(identity['primitive_ops'], 0)

    def test_costs_against_full_output_oracle(self):
        rng = random.Random(102)
        order, _ = rank_programs(lambda _: [.1, .2, .3, .4], budget=100, max_length=4)
        for _ in range(40):
            primitives = [rng.sample(range(5), 5) for _ in range(4)]
            examples = [(x, rng.randrange(5)) for x in range(3)]
            costs, outcomes = [], []
            for program in order:
                outputs = []
                for start, target in examples:
                    value = start
                    for operation in program:
                        value = primitives[operation][value]
                    outputs.append(value == target)
                checked = next((i + 1 for i, ok in enumerate(outputs) if not ok), len(examples))
                costs.append(len(program) * checked)
                outcomes.append(all(outputs))
            for budget in (0, 1, 7, 30, 100):
                total = 0
                for index, (cost, success) in enumerate(zip(costs, outcomes), 1):
                    if total + cost > budget:
                        expected = ('exhausted', budget, index)
                        break
                    total += cost
                    if success:
                        expected = ('accepted', total, index)
                        break
                else:
                    expected = ('space_exhausted', total, len(order))
                actual = search(primitives, examples, order, budget)
                self.assertEqual((actual['status'], actual['primitive_ops'], actual['candidates']), expected)


if __name__ == '__main__':
    unittest.main()
