"""Behavioral checks for first-fit selection, accounting, and compression."""

import copy
from itertools import product
import unittest

from experiments.study_h.search import compress, description_length, encoded_length, execute, search


class SearchTests(unittest.TestCase):
    def test_acceptance_exactly_at_budget_and_interruption(self):
        primitives = ((1, 2, 0), (2, 0, 1))
        examples = ((0, 1), (1, 2))
        accepted = search(primitives, examples, [], budget=2)
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["candidate_program"], (0,))
        self.assertEqual((accepted["primitive_ops"], accepted["comparisons"]), (2, 3))
        interrupted = search(primitives, examples, [], budget=1)
        self.assertEqual(interrupted["status"], "exhausted")
        self.assertEqual((interrupted["example_index"], interrupted["operation_index"]), (1, 0))
        self.assertEqual(interrupted["comparisons"], 2)

    def test_identity_needs_no_operations_but_checks_examples(self):
        result = search(((1, 0),), ((0, 0), (1, 1)), [], budget=0)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["candidate_program"], ())
        self.assertEqual((result["primitive_ops"], result["comparisons"]), (0, 2))

    def test_macro_is_expanded_and_charged(self):
        primitives = ((1, 2, 3, 0),)
        result = search(primitives, ((0, 3), (1, 0)), [(0, 0, 0)], budget=7)
        self.assertEqual(result["candidate_tokens"], (1,))
        self.assertEqual(result["candidate_program"], (0, 0, 0))
        self.assertEqual(result["primitive_ops"], 7)
        self.assertEqual(search(primitives, ((0, 3), (1, 0)), [(0, 0, 0)], budget=6)["status"], "exhausted")
        for state in range(4):
            self.assertEqual(execute(primitives, result["candidate_program"], state), (state + 3) % 4)

    def test_visible_fit_can_be_wrong_on_hidden_inputs(self):
        primitives = ((1, 0, 2, 3), (1, 0, 3, 2))
        task = dict(examples=((0, 1), (1, 0)), hidden=((2, 3), (3, 2)))
        changed = copy.deepcopy(task)
        changed["hidden"] = ((2, 0), (3, 0))
        first = search(primitives, task["examples"], [])
        second = search(primitives, changed["examples"], [])
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_program"], (0,))
        self.assertNotEqual(execute(primitives, first["candidate_program"], 2), task["hidden"][0][1])
        self.assertEqual(compress([first["candidate_program"]]), compress([second["candidate_program"]]))

    def test_encoding_matches_exhaustive_segmentation(self):
        program = (0, 1, 0, 1, 0)
        library = [(0, 1, 0), (0, 1)]
        vocabulary = [(0,), (1,)] + library
        shortest = next(length for length in range(1, len(program) + 1)
                        if any(tuple(op for token in tokens for op in vocabulary[token]) == program
                               for tokens in product(range(len(vocabulary)), repeat=length)))
        self.assertEqual(encoded_length(program, library), shortest)
        self.assertEqual(shortest, 2)

    def test_compression_charges_definitions_and_retains_multiplicity(self):
        motif = (0, 1, 2)
        self.assertEqual(compress([motif])[0], [])
        library, record = compress([motif] * 4)
        self.assertEqual(library, [motif])
        self.assertEqual(record["objective"], 8)
        self.assertEqual(record["original_tokens"], 12)
        self.assertEqual(description_length([motif] * 4, library), 8)

    def test_equal_compression_gain_uses_lexicographic_tie(self):
        library, record = compress([(1, 1)] * 4 + [(0, 0)] * 4)
        self.assertEqual(library, [(0, 0), (1, 1)])
        self.assertTrue(all(step["after"] < step["before"] for step in record["steps"]))


if __name__ == "__main__":
    unittest.main()
