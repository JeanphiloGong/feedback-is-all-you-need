"""Check collector data boundaries and the independent budget replay."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.study_h import run
from experiments.study_h.audit import reference_compression, reference_search
from experiments.study_h.run import collect, make_libraries
from experiments.study_h.search import compress, search
from experiments.study_h.tasks import make_environment


class CollectionTests(unittest.TestCase):
    def test_completed_search_survives_evaluator_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("experiments.study_h.run.evaluate", side_effect=RuntimeError("evaluator failed")):
                with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
                    collect([make_environment(23)], output)
            records = (output / "searches.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["result"]["status"], "accepted")
            self.assertEqual((output / "evaluations.jsonl").read_text(), "")

    def test_initialization_failure_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-run"
            with patch.object(run, "DATA_PATH", Path(directory) / "missing.json"), patch(
                    "sys.argv", ["run", "--development", "--output", str(output)]):
                with self.assertRaises(FileNotFoundError):
                    run.main()
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["error_type"], "FileNotFoundError")

    def test_hidden_labels_cannot_change_collected_search_or_memory(self):
        environment = make_environment(23)
        environment["tasks"] = {name: tasks[:1] for name, tasks in environment["tasks"].items()}
        changed = copy.deepcopy(environment)
        changed["latent_motifs"] = []
        for tasks in changed["tasks"].values():
            for task in tasks:
                task["test"] = [(state, (label + 1) % 16) for state, label in task["test"]]
                task["generating_program"] = []
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / label for label in ("original", "changed")]
            for item, path in zip((environment, changed), paths):
                path.mkdir()
                collect([item], path)
            rows = [[json.loads(line) for line in (path / "searches.jsonl").read_text().splitlines()] for path in paths]
            self.assertEqual([r["result"] for r in rows[0]], [r["result"] for r in rows[1]])
            memories = [json.loads((path / "memory-23.json").read_text()) for path in paths]
            for memory in memories:
                del memory["seconds"]
            self.assertEqual(*memories)

    def test_reference_counts_cover_full_and_partial_checks(self):
        primitives = ((1, 2, 0), (0, 2, 1))
        examples = ((0, 2), (1, 1), (2, 0))
        for library in ([], [(0, 1)], [(0, 0, 1)]):
            for budget in range(21):
                actual = search(primitives, examples, library, budget)
                self.assertEqual(reference_search(primitives, examples, library, budget), json.loads(json.dumps(actual)))

    def test_reference_compression_and_renaming(self):
        programs = [(0, 1, 2)] * 4 + [(3, 1, 2)] * 3
        library, detail = compress(programs)
        expected, reference = reference_compression(programs)
        self.assertEqual(library, expected)
        self.assertEqual(json.loads(json.dumps(detail)), reference)
        libraries, _ = make_libraries(programs, 23)
        self.assertEqual(list(map(len, libraries["compressed"])), list(map(len, libraries["renamed"])))
        self.assertEqual(libraries["whole"], [(0, 1, 2), (3, 1, 2)])


if __name__ == "__main__":
    unittest.main()
