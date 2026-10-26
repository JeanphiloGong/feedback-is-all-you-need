import copy
import unittest

import torch

from experiments.study_i.model import ProgramPrior, replay_steps


class ProgramPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        torch.manual_seed(7)

    def test_full_parameter_counts_and_probability_support(self):
        for hidden, expected in ((16, 1172), (64, 13892)):
            model = ProgramPrior(hidden)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            values = model.probabilities((0, 1))
            self.assertAlmostEqual(sum(values), 1)
            self.assertTrue(all(0 < value < 1 for value in values))

    def test_future_tokens_and_padding_cannot_change_earlier_logits(self):
        model = ProgramPrior(16)
        a = model(torch.tensor([[4, 1, 0, 2, 3]]))
        b = model(torch.tensor([[4, 1, 0, 3, 2]]))
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
        c = model(torch.tensor([[4, 1, 0]]))
        torch.testing.assert_close(a[:, :3], c, rtol=0, atol=0)

    def test_replay_updates_persist_and_frozen_copy_does_not_change(self):
        model = ProgramPrior(16)
        frozen = copy.deepcopy(model)
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        first = list(replay_steps(model, optimizer, [(0, 0, 0), (0,)]))
        second = list(replay_steps(model, optimizer, [(0, 0, 0), (0,), (0, 0)]))
        self.assertEqual(len(first + second), 32)
        self.assertLess(second[-1]['loss_before_step'], first[0]['loss_before_step'])
        self.assertEqual({int(state['step']) for state in optimizer.state.values()}, {32})
        self.assertEqual({row['target_tokens'] for row in first}, {4})
        self.assertEqual({row['target_tokens'] for row in second}, {6})
        for name, tensor in frozen.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(t, original[n]) for n, t in model.state_dict().items()))

    def test_padding_loss_matches_individually_scored_targets(self):
        model = ProgramPrior(16)
        corpus = [(0, 1, 2), (3,)]
        losses = []
        with torch.no_grad():
            for program in corpus:
                logits = model(torch.tensor([[4, *program[:-1]]]))[0]
                losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor(program), reduction='sum'))
        expected = sum(losses).item() / 4
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        record = next(replay_steps(model, optimizer, corpus, steps=1))
        self.assertAlmostEqual(record['loss_before_step'], expected, places=13)
        self.assertEqual(list(replay_steps(model, optimizer, [])), [])


if __name__ == '__main__':
    unittest.main()
