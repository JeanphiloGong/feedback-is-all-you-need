"""Check selection causality, interface constraints, and interpretation limits."""

import copy
import json
import unittest

from experiments.study_g.selection import describe_trajectory, matches, select_history
from experiments.study_f.rules import parse_rule


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.truth = dict(dax=0, wug=1, zup=2, fep=3)
        self.examples = [dict(state=state, commands=[c], target=(state + value) % 8)
                         for c, value in self.truth.items() for state in range(4)]
        self.wrong = dict(self.truth, dax=4)

    def test_earliest_tie_and_recovery_after_regression(self):
        ranks, selected = select_history([self.wrong, self.truth, self.truth, self.wrong], self.examples)
        self.assertEqual(ranks, [12, 16, 16, 12])
        self.assertEqual(selected, [0, 1, 1, 1])

    def test_invalid_candidates_never_beat_valid_zero(self):
        wrong = {c: (value + 4) % 8 for c, value in self.truth.items()}
        ranks, selected = select_history([None, None, wrong, None], self.examples)
        self.assertEqual(ranks, [-1, -1, 0, -1])
        self.assertEqual(selected, [0, 0, 2, 2])

    def test_negative_representatives_are_not_repaired(self):
        negative = dict(self.truth, fep=-5)
        self.assertIsNone(parse_rule(json.dumps(negative)))
        _, selected = select_history([self.wrong, parse_rule(json.dumps(negative))], self.examples)
        self.assertEqual(selected, [0, 0])

    def test_selection_uses_only_the_available_prefix(self):
        rules = [None, self.wrong, self.truth, self.wrong]
        ranks, selected = select_history(rules, self.examples)
        for end in range(1, len(rules) + 1):
            with self.subTest(end=end):
                prefix_ranks, prefix_selected = select_history(rules[:end], self.examples)
                self.assertEqual((prefix_ranks, prefix_selected), (ranks[:end], selected[:end]))

    def test_higher_direct_accuracy_can_reduce_composition(self):
        alias = {c: (value + 4) % 8 for c, value in self.truth.items()}
        _, selected = select_history([alias, self.wrong], self.examples)
        queries = [dict(state=0, commands=[a, b], target=(self.truth[a] + self.truth[b]) % 8)
                   for a in self.truth for b in self.truth]
        self.assertEqual(selected, [0, 1])
        self.assertEqual(sum(matches(alias, queries)), len(queries))
        self.assertLess(sum(matches(self.wrong, queries)), len(queries))

    def test_hidden_mutation_does_not_select_a_candidate(self):
        environment = dict(seed=1, train=self.examples, direct=self.examples, compose=self.examples)
        rules = [self.wrong, self.truth] + [self.wrong] * 15
        before = describe_trajectory(rules, environment)
        changed = copy.deepcopy(environment)
        changed["offsets"] = [7, 6, 5, 4]
        for split in ("direct", "compose"):
            changed[split] = [dict(q, target=(q["target"] + 1) % 8) for q in changed[split]]
        after = describe_trajectory(rules, changed)
        self.assertEqual([r["selected_stage"] for r in before], [r["selected_stage"] for r in after])
        self.assertEqual(before[-1]["selection_example_executions"], 17 * 16)
        self.assertEqual(before[-1]["selected_direct"], 1.0)
        self.assertEqual(after[-1]["selected_direct"], 0.0)


if __name__ == "__main__":
    unittest.main()
