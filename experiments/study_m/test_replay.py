"""Crosscheck independent numerical and environment reconstructions."""

import copy
import unittest

import torch

from experiments.study_h.tasks import rng
from experiments.study_m.model import TransitionModel, predicted_graph, replay_steps
from experiments.study_m.policy import choose_action
from experiments.study_m.replay import (independent_graph, independent_initial,
    independent_replay, reference_policy, reference_world)
from experiments.study_m.tasks import make_world


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_independent_worlds_match_both_families_on_synthetic_seeds(self):
        for seed in (-41, -73, -99):
            for family in ('structured', 'random'):
                self.assertEqual(make_world(seed, family), reference_world(seed, family))

    def test_independent_updates_match_model_adam_and_all_graph_logits(self):
        for hidden in (32, 128):
            parameters, optimizer = independent_initial(-847, 'synthetic', hidden)
            torch.manual_seed(rng(-847, f'study_m:initial:synthetic:{hidden}').randrange(2 ** 31))
            model = TransitionModel(hidden)
            other = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
            corpus = {(s, a): (s + a + 1) % 32 for s in range(12) for a in range(4)}
            actual = list(replay_steps(model, other, corpus, -847, 'synthetic', 1, 7))
            expected = list(independent_replay(parameters, optimizer, corpus, -847, 'synthetic', 1, 7))
            self.assertEqual(actual, expected)
            for p, q in zip(model.parameters(), parameters):
                self.assertTrue(torch.equal(p, q))
                for name, value in other.state[p].items():
                    self.assertTrue(torch.equal(value, optimizer.state[q][name]))
            self.assertEqual(predicted_graph(model), independent_graph(parameters))

    def test_independent_policy_counts_and_paths_match_with_unknown_edges(self):
        source = rng(-29, 'fixture')
        for _ in range(20):
            graph = [[source.choice([None, *range(32)]) for _ in range(32)] for _ in range(4)]
            original = copy.deepcopy(graph)
            for _ in range(20):
                start, goal = source.sample(range(32), 2)
                fallback = source.randrange(4)
                self.assertEqual(choose_action(graph, start, goal, fallback),
                                 reference_policy(graph, start, goal, fallback))
            self.assertEqual(graph, original)


if __name__ == '__main__':
    unittest.main()
