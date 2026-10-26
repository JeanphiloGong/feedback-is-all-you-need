"""Check controlled labels, exact split novelty, and independent RNG roles."""

import unittest

from experiments.study_h.tasks import make_environment, rng


class TaskTests(unittest.TestCase):
    def test_reproducible_generation_and_stream_separation(self):
        self.assertEqual(make_environment(23), make_environment(23))
        draws = [rng(seed, role).getrandbits(128)
                 for seed in (7, 23) for role in ("primitive_tables", "latent_motifs", "experience_order")]
        self.assertEqual(len(draws), len(set(draws)))

    def test_complete_labels_and_no_semantic_overlap(self):
        environment = make_environment(23)
        primitives = environment["primitives"]
        functions = []
        for family, tasks in environment["tasks"].items():
            if family != "experience":
                self.assertEqual(len(tasks), 16)
                self.assertEqual([len(task["generating_program"]) for task in tasks], [6] * 8 + [9] * 8)
            for task in tasks:
                self.assertEqual(len(task["train"]), 8)
                self.assertEqual(len(task["test"]), 8)
                self.assertEqual(sorted(x for x, _ in task["train"] + task["test"]), list(range(16)))
                # Independently compose full tables, rather than call the executor.
                table = list(range(16))
                for op in task["generating_program"]:
                    table = [primitives[op][state] for state in table]
                self.assertEqual(sorted(task["train"] + task["test"]), list(enumerate(table)))
                functions.append(tuple(table))
        self.assertEqual(len(functions), len(set(functions)))
        for primitive in primitives:
            self.assertEqual(sorted(primitive), list(range(16)))
            self.assertNotEqual(primitive, list(range(16)))
        self.assertEqual(len(set(map(tuple, primitives))), 4)

    def test_experience_length_fits_fixed_budget(self):
        self.assertLessEqual(8 * sum(4**length * length for length in range(6)), 65536)
        for task in make_environment(23)["tasks"]["experience"]:
            self.assertEqual(len(task["generating_program"]), 5)


if __name__ == "__main__":
    unittest.main()
