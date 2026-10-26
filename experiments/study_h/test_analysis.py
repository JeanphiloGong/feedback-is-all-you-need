"""Prevent incomplete-grid acceptance and missed negative interval disagreement."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.study_h import audit
from experiments.study_h.analyze import summarize
from experiments.study_h.run import ARMS, BUDGET, source_hashes
from experiments.study_h.tasks import make_environment


class AnalysisTests(unittest.TestCase):
    def test_incomplete_seed_grids_fail_before_replay(self):
        for seeds in ([], [10000], [10000, 10001]):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "metadata.json").write_text(json.dumps(dict(status="completed", seeds=seeds)))
                with self.assertRaisesRegex(AssertionError, "Incomplete fixed seed grid"):
                    audit.audit_run(folder)

    def test_changed_experience_or_composition_family_fails(self):
        original = make_environment(23)
        changed = copy.deepcopy(original)
        changed["tasks"]["experience"] = changed["tasks"]["experience"][:1]
        swapped = copy.deepcopy(original)
        swapped["tasks"]["composition"], swapped["tasks"]["shifted"] = (
            swapped["tasks"]["shifted"], swapped["tasks"]["composition"])
        for environment in (changed, swapped):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                data = folder / "environments.json"
                data.write_text(json.dumps([environment]))
                (folder / "metadata.json").write_text(json.dumps(dict(status="completed", seeds=[23],
                    source_sha256=source_hashes(), data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    budget=BUDGET, arms=ARMS)))
                (folder / "searches.jsonl").write_text("")
                (folder / "evaluations.jsonl").write_text("")
                with patch.object(audit, "DATA_PATH", data), self.assertRaisesRegex(
                        AssertionError, "Frozen task generation mismatch"):
                    audit.audit_run(folder)

    def test_negative_bootstrap_disagreement_is_reported(self):
        rows, memories = [], {}
        for seed in range(32):
            memories[seed] = dict(libraries={arm: [] for arm in ARMS},
                                 seconds=dict(whole=0, compressed=0, renamed=0))
            for arm in ARMS:
                for family in ("composition", "shifted"):
                    for task in range(16):
                        exact = family == "composition" and (
                            (seed < 6 and arm == "whole") or
                            (seed >= 6 and arm == "compressed" and task == 0))
                        rows.append(dict(seed=seed, arm=arm, family=family,
                            result=dict(status="accepted", primitive_ops=1, candidates=1, comparisons=8),
                            evaluation=dict(exact=exact, test_correct=8 * exact, primitive_ops=16),
                            search_seconds=0, evaluation_seconds=0))
        primary = summarize(rows, memories)["primary"]
        self.assertEqual(primary["differences"], [-1] * 6 + [1 / 16] * 26)
        self.assertLess(primary["ci95"][0], 0)
        self.assertGreater(primary["ci95"][1], 0)
        self.assertLess(primary["bootstrap95"][1], 0)
        self.assertTrue(primary["interval_disagreement"])


if __name__ == "__main__":
    unittest.main()
