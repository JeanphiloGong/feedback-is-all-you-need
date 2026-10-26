"""Check probability semantics, derivatives, and actual corrective updates."""

import copy
from itertools import product
import unittest

import numpy as np
import torch

from experiments.study_i.model import ProgramPrior
from experiments.study_j.audit import gru_probabilities
from experiments.study_l.feedback import (CONTEXTS, feedback_steps, local_probabilities,
                                         program_indices, rejected_mass)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(304)
        self.model = ProgramPrior(16)

    def test_padded_local_batch_matches_independent_gru_equations(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 1172)
        actual = local_probabilities(self.model).detach().numpy()
        weights = {name: tensor.detach().numpy() for name, tensor in self.model.state_dict().items()}
        expected = [gru_probabilities(weights, context, {}) for context in CONTEXTS]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_complete_length_slices_have_fixed_mass_and_zero_gradient(self):
        probabilities = .9 * local_probabilities(self.model) + .025
        programs = [p for n in range(4) for p in product(range(4), repeat=n)]
        mass = rejected_mass(probabilities, program_indices(programs))
        self.assertAlmostEqual(mass.item(), .4, places=14)
        mass.backward()
        self.assertLess(max(p.grad.abs().max().item() for p in self.model.parameters()), 1e-14)

    def test_failure_loss_gradient_matches_finite_differences(self):
        logits = torch.randn(21, 4, dtype=torch.float64, requires_grad=True)
        programs = [(), (0,), (1, 2), (3, 1, 0), (0, 1, 2, 3, 0)]
        indices = program_indices(programs)
        loss = -torch.log1p(-rejected_mass(.9 * logits.softmax(-1) + .025, indices))
        gradient, = torch.autograd.grad(loss, logits)

        def independent_loss(values):
            rows = np.exp(values - values.max(axis=1, keepdims=True))
            rows = .9 * rows / rows.sum(axis=1, keepdims=True) + .025
            total = 0.
            for program in programs:
                probability = .1
                for i, op in enumerate(program):
                    context = tuple(program[max(0, i - 2):i])
                    probability *= rows[CONTEXTS.index(context), op]
                total += probability
            return -np.log1p(-total)

        values = logits.detach().numpy().copy()
        self.assertAlmostEqual(loss.item(), independent_loss(values), places=14)
        for row, column in ((0, 0), (0, 3), (1, 1), (6, 2), (11, 0), (20, 3)):
            before = values[row, column]
            values[row, column] = before + 1e-6
            upper = independent_loss(values)
            values[row, column] = before - 1e-6
            lower = independent_loss(values)
            values[row, column] = before
            self.assertAlmostEqual(gradient[row, column].item(), (upper - lower) / 2e-6, places=8)

    def test_rejected_programs_change_weights_and_lose_probability_mass(self):
        programs = [(0,), (0, 0), (0, 1, 0), (1, 0, 1)]
        indices = program_indices(programs)
        initial = copy.deepcopy(self.model.state_dict())
        before = rejected_mass(.9 * local_probabilities(self.model) + .025, indices).item()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01, foreach=False, fused=False)
        records = list(feedback_steps(self.model, optimizer, 'rejection', programs))
        after = rejected_mass(.9 * local_probabilities(self.model) + .025, indices).item()
        self.assertLess(after, before)
        self.assertEqual(len(records), 16)
        self.assertTrue(any(not torch.equal(initial[name], tensor)
                            for name, tensor in self.model.state_dict().items()))
        self.assertTrue(all(r['input_positions'] == 63 and r['input_rows'] == 21 for r in records))

    def test_uniformizing_ignores_rejection_identities_and_empty_rejections_skip(self):
        other = copy.deepcopy(self.model)
        first = torch.optim.Adam(self.model.parameters(), lr=.01, foreach=False, fused=False)
        second = torch.optim.Adam(other.parameters(), lr=.01, foreach=False, fused=False)
        a = list(feedback_steps(self.model, first, 'uniformizing', [(0,)]))
        b = list(feedback_steps(other, second, 'uniformizing', [(3, 2, 1), ()]))
        self.assertEqual(a, b)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(self.model.parameters(), other.parameters())))
        original = copy.deepcopy(self.model.state_dict())
        self.assertEqual(list(feedback_steps(self.model, first, 'rejection', [])), [])
        self.assertTrue(all(torch.equal(original[n], p) for n, p in self.model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
