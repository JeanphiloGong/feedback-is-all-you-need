"""Synthetic tests only: no official study seed is evaluated here."""

import copy
from math import comb
import random
import unittest

import torch

from experiments.study_m.model import TransitionModel
from experiments.study_m.policy import choose_action
from experiments.study_n.learning import update_steps
from experiments.study_n.tasks import conjugate, make_world, signed_candidates, signed_prediction


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_balanced_disjoint_queries_and_conjugate_physical_evidence(self):
        for seed in range(-111, -101):
            world = make_world(seed)
            labels = world['labels']
            self.assertEqual(sorted(labels), list(range(32)))
            maps = world['primitives']
            changed = conjugate(maps, labels)
            train, test = set(map(tuple, world['queries'])), set(map(tuple, world['test']))
            self.assertEqual(len(train), 64)
            self.assertEqual(len(test), 64)
            self.assertFalse(train & test)
            self.assertEqual(train | test, {(s, a) for a in range(4) for s in range(32)})
            for n in (4, 16, 32, 64):
                self.assertEqual([sum(a == k for _, a in world['queries'][:n]) for k in range(4)],
                                 [n // 4] * 4)
            for s, a in train | test:
                self.assertEqual(changed[a][labels[s]], labels[maps[a][s]])

    def test_planner_is_equivariant_on_wrong_and_missing_graphs(self):
        source = random.Random(112)
        for _ in range(6):
            labels = source.sample(range(32), 32)
            graph = [[source.choice([None, *range(32)]) for _ in range(32)] for _ in range(4)]
            changed = conjugate(graph, labels)
            for state in range(32):
                for goal in range(32):
                    if state == goal:
                        continue
                    fallback = source.randrange(4)
                    a = choose_action(graph, state, goal, fallback)
                    b = choose_action(changed, labels[state], labels[goal], fallback)
                    self.assertEqual(a, b)

    def test_symbolic_matcher_never_uses_test_targets_or_guesses_ambiguous_bits(self):
        world = make_world(-113)
        candidates, comparisons = signed_candidates([])
        self.assertEqual(comparisons, 0)
        self.assertIsNone(signed_prediction(candidates, 0, 0))
        triples = [(s, a, world['primitives'][a][s]) for s, a in world['queries']]
        candidates, comparisons = signed_candidates(triples)
        self.assertGreater(comparisons, 0)
        for s, a in world['test']:
            predicted = signed_prediction(candidates, s, a)
            if predicted is not None:
                self.assertEqual(predicted, world['primitives'][a][s])
        complete = [(s, a, s ^ 17) for a in range(4) for s in range(32)]
        identified, _ = signed_candidates(complete)
        for state in range(32):
            self.assertEqual(signed_prediction(identified, state, 2), state ^ 17)
        ambiguous = copy.deepcopy(identified)
        ambiguous[2][4].append((4, 0))
        self.assertIsNone(signed_prediction(ambiguous, 0, 2))
        for bit in range(5):
            for sign in (0, 1):
                counts = [sum(((s >> bit & 1) ^ sign) == ((s >> other & 1) ^ flip)
                              for s in range(32)) for other in range(5) for flip in (0, 1)
                          if (other, flip) != (bit, sign)]
                self.assertEqual(sorted(counts), [0] + [16] * 8)
        self.assertAlmostEqual(160 / comb(32, 16), 2.6618735640335896e-7)

    def test_interrupted_updates_retain_weights_moments_and_ordered_sampling(self):
        for hidden in (32, 128):
            torch.manual_seed(29)
            model = TransitionModel(hidden)
            original = copy.deepcopy(model.state_dict())
            other = copy.deepcopy(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
            other_optimizer = torch.optim.Adam(other.parameters(), lr=.01, foreach=False, fused=False)
            observations = [(0, 1, 13), (7, 2, 9), (9, 3, 4)]
            first = list(update_steps(model, optimizer, observations, -114, 1, 3))
            second = list(update_steps(model, optimizer, observations, -114, 4, 4))
            continuous = list(update_steps(other, other_optimizer, observations, -114, 1, 7))
            self.assertEqual(first + second, continuous)
            self.assertTrue(any(not torch.equal(original[k], v) for k, v in model.state_dict().items()))
            for p, q in zip(model.parameters(), other.parameters()):
                self.assertTrue(torch.equal(p, q))
                for k, v in optimizer.state[p].items():
                    self.assertTrue(torch.equal(v, other_optimizer.state[q][k]))
            changed = copy.deepcopy(other)
            changed_optimizer = torch.optim.Adam(changed.parameters(), lr=.01, foreach=False, fused=False)
            renamed = [(31-s, a, 31-y) for s, a, y in observations]
            records = list(update_steps(changed, changed_optimizer, renamed, -114, 1, 7))
            self.assertEqual([r['indices_sha256'] for r in records],
                             [r['indices_sha256'] for r in continuous])

    def test_actual_minibatch_rows_match_under_relabeling(self):
        observed = [(19, 2, 5), (0, 3, 17), (12, 0, 31), (7, 1, 2)]
        renamed = [(31-s, a, 31-y) for s, a, y in observed]
        batches = []
        for data in (observed, renamed):
            model = TransitionModel(32)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
            captured = []
            handle = model.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
            list(update_steps(model, optimizer, data, -115, 1, 5))
            handle.remove()
            batches.append(captured)
        for canonical, relabeled in zip(*batches):
            self.assertTrue(torch.equal(canonical[:, :5], 1 - relabeled[:, :5]))
            self.assertTrue(torch.equal(canonical[:, 5:], relabeled[:, 5:]))


if __name__ == '__main__':
    unittest.main()
