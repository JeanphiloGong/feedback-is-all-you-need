"""Synthetic checks only: no official Study K environments are generated."""

from itertools import product
import math
import unittest

import numpy as np
import torch

from experiments.study_i.model import ProgramPrior
from experiments.study_i.search import rank_programs
from experiments.study_j.audit import gru_probabilities
from experiments.study_j.local import local_replay_steps
from experiments.study_k.stream import local_probability_table, task_roles


class StreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(67)

    def test_roles_partition_families_and_alternate_lengths(self):
        environment = {'tasks': {family: [dict(id=f'{family}-{i}', length=6 if i < 8 else 9)
                                          for i in range(16)]
                                 for family in ('composition', 'shifted')}}
        streams, probes = task_roles(environment)
        self.assertEqual([(family, task['id']) for family, task in probes],
                         [(family, f'{family}-{i}')
                          for family in ('composition', 'shifted')
                          for i in (1, 9, 3, 11, 5, 13, 7, 15)])
        expected_first = [('composition', i) for i in (0, 8, 2, 10, 4, 12, 6, 14)]
        expected_first += [('shifted', i) for i in (0, 8, 2, 10, 4, 12, 6, 14)]
        expected_reverse = expected_first[8:] + expected_first[:8]
        for order, expected in zip(('composition_first', 'shifted_first'),
                                   (expected_first, expected_reverse)):
            self.assertEqual([(family, task['id']) for family, task in streams[order]],
                             [(family, f'{family}-{i}') for family, i in expected])
            self.assertEqual([task['length'] for _, task in streams[order]], [6, 9] * 8)
            stream_ids = {task['id'] for _, task in streams[order]}
            probe_ids = {task['id'] for _, task in probes}
            self.assertFalse(stream_ids & probe_ids)
            self.assertEqual(len(stream_ids | probe_ids), 32)
        for family in ('composition', 'shifted'):
            selected = [task for name, task in probes if name == family]
            self.assertEqual([task['length'] for task in selected], [6, 9] * 4)
            self.assertIs(selected[0], environment['tasks'][family][1])

    def test_table_has_exact_physical_work_and_direct_predictions(self):
        model = ProgramPrior(16)
        calls = []
        hook = model.register_forward_pre_hook(lambda module, args: calls.append(tuple(args[0].shape)))
        table = local_probability_table(model)
        hook.remove()
        self.assertEqual(len(table), 21)
        self.assertEqual(calls, [(1, 1)] + [(1, 2)] * 4 + [(1, 3)] * 16)
        self.assertEqual(sum(rows * width for rows, width in calls), 57)
        self.assertEqual(sum(len(row) for row in table.values()), 84)
        for length in range(6):
            for prefix in product(range(4), repeat=length):
                self.assertEqual(table[prefix[-2:]], model.probabilities(prefix[-2:]))

    def test_table_values_and_order_match_independent_gru_equations(self):
        model = ProgramPrior(16)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
        list(local_replay_steps(model, optimizer, [(0, 1, 3), (2,), (1, 0, 1, 2, 0)], 3))
        table = local_probability_table(model)
        weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
        cache = {}
        for context, row in table.items():
            np.testing.assert_allclose(row, gru_probabilities(weights, context, cache), rtol=1e-12, atol=1e-12)
        scored = []
        for length in range(5):
            for program in product(range(4), repeat=length):
                cost = 0
                for i, token in enumerate(program):
                    probability = gru_probabilities(weights, program[max(0, i - 2):i], cache)[token]
                    cost += round(-math.log(.9 * float(probability) + .025) * 1000000)
                scored.append((cost, program))
        complete = [program for _, program in sorted(scored)]
        for budget in (0, 1, 3, 8, 19, 65, 10000):
            expected, used = [], 0
            for program in complete:
                expected.append(program)
                used += len(program)
                if used > budget:
                    break
            actual, detail = rank_programs(lambda prefix: table[prefix[-2:]], budget, max_length=4)
            self.assertEqual(actual, expected)
            self.assertEqual(detail['minimum_primitive_ops'], used)

    def test_archived_table_does_not_change_after_learning(self):
        model = ProgramPrior(16)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
        before = local_probability_table(model)
        snapshot = {context: row[:] for context, row in before.items()}
        list(local_replay_steps(model, optimizer, [(0, 0, 1), (0, 2)], 2))
        after = local_probability_table(model)
        self.assertEqual(before, snapshot)
        self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
