"""Independent combinatorial checks for uniform candidate selection."""

from fractions import Fraction
from math import comb, perm
import unittest

from experiments.study_g.random_reference import expected_selection, match_counts


class RandomReferenceTest(unittest.TestCase):
    def test_enumeration_matches_inclusion_exclusion(self):
        counts = match_counts()
        expected = {k: comb(4, k) * sum((-1) ** j * comb(4 - k, j) * perm(8 - k - j, 4 - k - j)
                                       for j in range(5 - k)) for k in range(5)}
        self.assertEqual(counts, expected)
        self.assertEqual(sum(counts.values()), 1680)

    def test_one_uniform_draw_has_chance_marginals(self):
        direct, exact = expected_selection(0, 1, match_counts())
        self.assertEqual(direct, Fraction(1, 8))
        self.assertEqual(exact, Fraction(1, 1680))

    def test_two_draws_match_weighted_pair_enumeration(self):
        counts = match_counts()
        for initial in range(5):
            with self.subTest(initial=initial):
                direct = Fraction(sum(max(initial, a, b) * ca * cb for a, ca in counts.items()
                                      for b, cb in counts.items()), 4 * 1680 ** 2)
                exact = Fraction(sum((max(initial, a, b) == 4) * ca * cb for a, ca in counts.items()
                                     for b, cb in counts.items()), 1680 ** 2)
                self.assertEqual(expected_selection(initial, 2, counts), (direct, exact))

    def test_existing_exact_rule_is_preserved(self):
        for draws in (0, 1, 16, 17):
            with self.subTest(draws=draws):
                self.assertEqual(expected_selection(4, draws, match_counts()), (1, 1))

    def test_no_draws_preserve_initial_score(self):
        for initial in range(5):
            with self.subTest(initial=initial):
                self.assertEqual(expected_selection(initial, 0, match_counts()),
                                 (Fraction(initial, 4), Fraction(initial == 4)))


if __name__ == "__main__":
    unittest.main()
