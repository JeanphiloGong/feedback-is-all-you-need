import copy
from itertools import product
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.study_j.audit import gru_probabilities, reference_search
from experiments.study_i.model import ProgramPrior
from experiments.study_i.search import rank_programs, search
from experiments.study_j.local import local_replay_steps


class LocalContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(41)
        self.model = ProgramPrior(16)

    def test_local_prediction_ignores_earlier_history_and_matches_short_prefixes(self):
        for prefix in [p for n in range(3) for p in product(range(4), repeat=n)]:
            self.assertEqual(self.model.probabilities(prefix), self.model.probabilities(prefix[-2:]))
        a, b = (0, 1, 2, 3), (3, 0, 2, 3)
        self.assertEqual(self.model.probabilities(a[-2:]), self.model.probabilities(b[-2:]))
        self.assertNotEqual(self.model.probabilities(a), self.model.probabilities(b))
        self.assertEqual(sum(t.numel() for t in self.model.parameters()), 1172)

    def test_batched_loss_gradient_and_cost_match_individual_causal_contexts(self):
        programs = [(0, 1, 2, 3, 0), (2,)]
        reference = copy.deepcopy(self.model)
        rows = []
        for program in programs:
            for index, target in enumerate(program):
                context = [4, *program[max(0, index - 2):index]]
                logits = reference(torch.tensor([context]))[0, -1]
                rows.append(F.cross_entropy(logits[None], torch.tensor([target])))
        loss = torch.stack(rows).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        expected_optimizer = torch.optim.Adam(reference.parameters(), lr=.01, foreach=False, fused=False)
        expected_optimizer.step()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01, foreach=False, fused=False)
        actual = next(local_replay_steps(self.model, optimizer, programs, steps=1))
        self.assertAlmostEqual(actual['loss_before_step'], loss.item(), places=13)
        self.assertAlmostEqual(actual['gradient_norm_before_clip'], norm.item(), places=13)
        self.assertEqual((actual['target_tokens'], actual['input_rows'], actual['input_positions']), (6, 6, 18))
        for expected, observed in zip(reference.parameters(), self.model.parameters()):
            torch.testing.assert_close(expected, observed, rtol=1e-12, atol=1e-12)

    def test_updates_persist_without_changing_frozen_weights_and_empty_corpus_skips(self):
        frozen = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01, foreach=False, fused=False)
        first = list(local_replay_steps(self.model, optimizer, [(1, 1, 1)], steps=8))
        second = list(local_replay_steps(self.model, optimizer, [(1, 1, 1), (1,)], steps=8))
        self.assertLess(second[-1]['loss_before_step'], first[0]['loss_before_step'])
        self.assertEqual({int(s['step']) for s in optimizer.state.values()}, {16})
        self.assertNotEqual(frozen.probabilities((1, 1)), self.model.probabilities((1, 1)))
        before = copy.deepcopy(self.model.state_dict())
        self.assertEqual(list(local_replay_steps(self.model, optimizer, [])), [])
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_local_heap_matches_exhaustive_numpy_order_and_execution_boundaries(self):
        weights = {name: value.detach().numpy() for name, value in self.model.state_dict().items()}
        cache = {}
        all_programs = [p for n in range(5) for p in product(range(4), repeat=n)]
        scored = []
        for program in all_programs:
            score = 0
            for index, token in enumerate(program):
                prefix = program[max(0, index - 2):index]
                probabilities = gru_probabilities(weights, prefix, cache)
                np.testing.assert_allclose(probabilities, self.model.probabilities(prefix), rtol=1e-12, atol=1e-12)
                score += round(-math.log(.9 * float(probabilities[token]) + .025) * 1_000_000)
            scored.append((score, program))
        order, detail = rank_programs(lambda p: self.model.probabilities(p[-2:]), 100000, 4)
        self.assertEqual(order, [program for _, program in sorted(scored)])
        self.assertEqual(detail['predictions'], 85)
        primitives = [(1, 2, 3, 0), (2, 0, 3, 1), (3, 2, 1, 0), (0, 2, 1, 3)]
        examples = [(0, 2), (1, 0), (3, 1)]
        for budget in range(101):
            actual = search(primitives, examples, order, budget)
            actual['candidate_program'] = list(actual['candidate_program']) if actual['candidate_program'] is not None else None
            self.assertEqual(actual, reference_search(primitives, examples, order, budget, {}))


if __name__ == '__main__':
    unittest.main()
