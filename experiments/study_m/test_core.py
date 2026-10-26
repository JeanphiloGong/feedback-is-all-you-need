"""Synthetic checks of predictive learning, expressivity, and planning."""

import copy
import random
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from experiments.study_m.model import TransitionModel, features, predicted_graph, replay_steps, state_bits
from experiments.study_m.policy import choose_action
from experiments.study_m.tasks import make_world, structured_map, strongly_connected


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(733)

    def test_bit_features_and_threshold_decode(self):
        keys = [(0, 0), (19, 2), (31, 3)]
        expected = [[int(x) for x in format(s, '05b')[::-1]] + [int(a == i) for i in range(4)]
                    for s, a in keys]
        self.assertEqual(features(keys).tolist(), expected)
        model = TransitionModel(32)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.output_layer.bias.copy_(torch.tensor([0., -1., 2., -3., 0.]))
        graph, detail = predicted_graph(model)
        self.assertEqual(graph, [[21] * 32 for _ in range(4)])
        self.assertEqual(detail['input_rows'], 128)
        self.assertEqual(detail['dense_forward_macs'], 57344)

    def test_parameter_counts_and_training_matrix_budget(self):
        for hidden, parameters in ((32, 485), (128, 1925)):
            model = TransitionModel(hidden)
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
            self.assertFalse(features([(1, 2)]).requires_grad)
        self.assertEqual(8 * 32 * 33 * 32, 2 * 32 * 33 * 128)
        self.assertEqual(8 * 32 * 33 * 32, 270336)

    def test_numpy_forward_loss_and_analytical_gradients(self):
        model = TransitionModel(32)
        x = features([(0, 0), (19, 2), (31, 3), (7, 1)])
        y = state_bits(torch.tensor([13, 4, 31, 0]))
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        w, b, v, c = [p.detach().numpy() for p in model.parameters()]
        z = x.numpy() @ w.T + b
        h = np.maximum(z, 0)
        logits = h @ v.T + c
        expected_loss = np.mean(np.logaddexp(0, logits) - y.numpy() * logits)
        self.assertAlmostEqual(loss.item(), expected_loss, places=14)
        np.testing.assert_allclose(model(x).detach().numpy(), logits, rtol=1e-12, atol=1e-12)
        d = (1 / (1 + np.exp(-logits)) - y.numpy()) / y.numel()
        dz = (d @ v) * (z > 0)
        gradients = [dz.T @ x.numpy(), dz.sum(axis=0), d.T @ h, d.sum(axis=0)]
        for p, expected in zip(model.parameters(), gradients):
            np.testing.assert_allclose(p.grad.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_small_constructive_witness_covers_all_structured_transitions(self):
        permutations = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 0], [2, 0, 4, 1, 3]]
        masks = [0, 31, 21, 10]
        model = TransitionModel(32)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.output_layer.bias.fill_(-3)
            for action in range(4):
                model.input_layer.weight[20 + action, 5 + action] = 1
                for bit in range(5):
                    unit = action * 5 + bit
                    model.input_layer.weight[unit, bit] = 1
                    model.input_layer.weight[unit, 5 + action] = 1
                    model.input_layer.bias[unit] = -1
                for bit, source in enumerate(permutations[action]):
                    flipped = (masks[action] >> bit) & 1
                    model.output_layer.weight[bit, action * 5 + source] = 6 * (1 - 2 * flipped)
                    model.output_layer.weight[bit, 20 + action] = 6 * flipped
        expected = []
        for permutation, mask in zip(permutations, masks):
            expected.append([int(''.join(format(s, '05b')[::-1][i] for i in permutation)[::-1], 2) ^ mask
                             for s in range(32)])
        self.assertEqual(predicted_graph(model)[0], expected)
        self.assertEqual([structured_map(p, m) for p, m in zip(permutations, masks)], expected)

    def test_large_constructive_witness_covers_an_arbitrary_transition_table(self):
        source = random.Random(389)
        expected = [[source.randrange(32) for _ in range(32)] for _ in range(4)]
        model = TransitionModel(128)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            for action in range(4):
                for state in range(32):
                    unit = action * 32 + state
                    bits = [int(b) for b in format(state, '05b')[::-1]]
                    model.input_layer.weight[unit, :5] = torch.tensor([2*b - 1 for b in bits])
                    model.input_layer.weight[unit, 5 + action] = 1
                    model.input_layer.bias[unit] = bits.count(0) - 5
                    model.output_layer.weight[:, unit] = torch.tensor([
                        3 if expected[action][state] & (1 << bit) else -3 for bit in range(5)])
        self.assertEqual(predicted_graph(model)[0], expected)

    def test_replay_continues_weights_and_optimizer_across_calls(self):
        model = TransitionModel(32)
        other = copy.deepcopy(model)
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
        other_optimizer = torch.optim.Adam(other.parameters(), lr=.01, foreach=False, fused=False)
        corpus = {(state, action): state ^ (action + 1) for state in range(4) for action in range(4)}
        first = list(replay_steps(model, optimizer, corpus, -733, 'fixture', 1, 8))
        second = list(replay_steps(model, optimizer, corpus, -733, 'fixture', 9, 8))
        reverse = dict(reversed(list(corpus.items())))
        continuous = list(replay_steps(other, other_optimizer, reverse, -733, 'fixture', 1, 16))
        self.assertEqual(first + second, continuous)
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()))
        for p, q in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(p, q))
            for name in optimizer.state[p]:
                self.assertTrue(torch.equal(optimizer.state[p][name], other_optimizer.state[q][name]))
            self.assertEqual(optimizer.state[p]['step'].item(), 16)


class PolicyTests(unittest.TestCase):
    def test_unknown_edges_are_absent_and_known_paths_use_numeric_ties(self):
        graph = [[None] * 32 for _ in range(4)]
        self.assertTrue(choose_action(graph, 3, 0, 2)['fallback'])
        graph[0][3], graph[2][1] = 1, 0
        graph[1][3], graph[0][2] = 2, 0
        result = choose_action(graph, 3, 0, 2)
        self.assertEqual(result['planned_path'], [0, 2])
        self.assertEqual(result['action'], 0)
        self.assertFalse(result['fallback'])
        graph[3][3] = 0
        self.assertEqual(choose_action(graph, 3, 0, 2)['planned_path'], [3])

    def test_exact_eight_edge_plan_is_allowed_but_nine_is_not(self):
        graph = [[(state + 1) % 32 for state in range(32)]] + [[None] * 32 for _ in range(3)]
        self.assertEqual(choose_action(graph, 0, 8, 3)['planned_path'], [0] * 8)
        self.assertEqual(choose_action(graph, 0, 9, 3)['action'], 3)
        self.assertTrue(choose_action(graph, 0, 9, 3)['fallback'])
        self.assertEqual(choose_action(None, 0, 1, 2)['edge_checks'], 0)

    def test_bfs_matches_independent_bounded_distance_relaxation(self):
        source = random.Random(537)
        for _ in range(20):
            graph = [[source.randrange(32) if source.random() > .3 else None
                      for _ in range(32)] for _ in range(4)]
            for _ in range(5):
                start, goal = source.sample(range(32), 2)
                distances = [99] * 32
                distances[goal] = 0
                for _ in range(8):
                    distances = [min([distances[s]] + [1 + distances[g[s]]
                                 for g in graph if g[s] is not None]) for s in range(32)]
                result = choose_action(graph, start, goal, 3)
                if distances[start] > 8:
                    self.assertTrue(result['fallback'])
                else:
                    expected = min(a for a in range(4) if graph[a][start] is not None
                                   and distances[graph[a][start]] == distances[start] - 1)
                    self.assertEqual(result['action'], expected)
                    self.assertEqual(len(result['planned_path']), distances[start])


class WorldTests(unittest.TestCase):
    def test_synthetic_worlds_have_disjoint_goal_roles_and_complete_queues(self):
        for family in ('structured', 'random'):
            world = make_world(-511, family)
            self.assertEqual(world, make_world(-511, family))
            self.assertEqual(len(world['episodes']), 513)
            self.assertEqual(len(world['probes']), 248)
            self.assertEqual(len(world['warmup']['actions']), 16)
            self.assertEqual(set(world['held_out_goals']) & set(world['training_goals']), set())
            self.assertEqual(len(set(world['held_out_goals'])), 8)
            self.assertEqual(len(set(world['training_goals'])), 24)
            for episode in world['episodes']:
                self.assertIn(episode['goal'], world['training_goals'])
                self.assertNotEqual(episode['start'], episode['goal'])
            for probe in world['probes']:
                self.assertIn(probe['goal'], world['held_out_goals'])
                self.assertNotEqual(probe['start'], probe['goal'])
            for action in world['primitives']:
                self.assertEqual(sorted(action), list(range(32)))
                self.assertNotEqual(action, list(range(32)))
            self.assertEqual(len({tuple(a) for a in world['primitives']}), 4)
            reachable = np.eye(32, dtype=bool)
            for action in world['primitives']:
                reachable[np.arange(32), action] = True
            for pivot in range(32):
                reachable |= reachable[:, pivot, None] & reachable[None, pivot, :]
            self.assertTrue(reachable.all())
            self.assertTrue(strongly_connected(world['primitives']))
        self.assertFalse(strongly_connected([list(range(32)) for _ in range(4)]))


if __name__ == '__main__':
    unittest.main()
