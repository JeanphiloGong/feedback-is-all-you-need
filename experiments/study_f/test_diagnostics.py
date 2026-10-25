"""Distinguish targeted repairs, collateral changes, and invalid transitions."""

import unittest

from experiments.study_f.diagnose import transition


class TransitionTest(unittest.TestCase):
    def setUp(self):
        self.truth = dict(dax=0, wug=1, zup=2, fep=3)
        self.examples = [dict(state=0, commands=[c], target=v) for c, v in self.truth.items()]
        self.wrong = dict(self.truth, dax=4)

    def test_targeted_repair_with_collateral_damage(self):
        result = transition(self.wrong, dict(self.truth, wug=4), self.examples)
        self.assertEqual(result["first_failed_command"], "dax")
        self.assertTrue(result["failed_offset_corrected"])
        self.assertEqual((result["gained_components"], result["lost_components"]), (1, 1))
        self.assertFalse(result["after_exact"])

    def test_unchanged_error_is_not_a_repair(self):
        result = transition(self.wrong, self.wrong, self.examples)
        self.assertFalse(result["failed_offset_changed"])
        self.assertFalse(result["failed_offset_corrected"])
        self.assertEqual((result["gained_components"], result["lost_components"]), (0, 0))

    def test_invalid_pair_has_no_component_attribution(self):
        for before, after in ((self.wrong, None), (None, self.truth), (None, None)):
            result = transition(before, after, self.examples)
            self.assertIsNone(result["failed_offset_corrected"])
            self.assertIsNone(result["gained_components"])
            self.assertIsNone(result["lost_components"])

    def test_exact_retention_and_regression_are_separate(self):
        kept = transition(self.truth, self.truth, self.examples)
        lost = transition(self.truth, self.wrong, self.examples)
        self.assertTrue(kept["before_exact"] and kept["after_exact"])
        self.assertIsNone(kept["first_failed_command"])
        self.assertTrue(lost["before_exact"])
        self.assertFalse(lost["after_exact"])
        self.assertEqual(lost["lost_components"], 1)


if __name__ == "__main__":
    unittest.main()
