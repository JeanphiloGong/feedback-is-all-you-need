"""Checks for transcript ordering, branch equality, and hidden-data separation."""

import copy
import json
from pathlib import Path
import unittest

from experiments.study_b.tasks import make_environment
from experiments.study_f.rules import REVIEW, proposal_messages
from experiments.study_g.history import history_messages


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_environment(19)
        self.history = [dict(text="old incorrect proposal", check="first failure"),
                        dict(text="new proposal", check="all examples pass")]

    def test_initial_and_first_revision_prompts_match(self):
        for history in ([], self.history[:1]):
            self.assertEqual(history_messages(self.environment["train"], history, "last_only"),
                             history_messages(self.environment["train"], history, "full_history"))
        self.assertEqual(history_messages(self.environment["train"], [], "last_only"),
                         proposal_messages(self.environment["train"]))

    def test_last_only_matches_the_frozen_f_checked_prompt(self):
        previous = self.history[-1]
        expected = proposal_messages(self.environment["train"], previous["text"], previous["check"])
        self.assertEqual(history_messages(self.environment["train"], self.history, "last_only"), expected)

    def test_full_history_retains_ordered_proposals_then_checks(self):
        messages = history_messages(self.environment["train"], self.history, "full_history")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual([m["content"] for m in messages[2:]], [
            "old incorrect proposal", "first failure\n" + REVIEW,
            "new proposal", "all examples pass\n" + REVIEW])
        self.assertEqual(self.history[0]["text"], "old incorrect proposal")

    def test_hidden_queries_and_offsets_do_not_enter_prompts(self):
        altered = copy.deepcopy(self.environment)
        altered["offsets"] = [999] * 4
        altered["direct"] = altered["compose"] = [{"state": -100, "commands": ["hidden"], "target": 999}]
        for arm in ("last_only", "full_history"):
            self.assertEqual(history_messages(self.environment["train"], self.history, arm),
                             history_messages(altered["train"], self.history, arm))

    def test_frozen_languages_have_identifiable_disjoint_splits(self):
        data = json.loads(Path("experiments/study_g/data/environments-v1.json").read_text())
        self.assertEqual([e["seed"] for e in data["environments"]], [19, *range(9000, 9048)])
        for environment in data["environments"]:
            offsets = dict(zip(("dax", "wug", "zup", "fep"), environment["offsets"]))
            inferred = {q["commands"][0]: (q["target"] - q["state"]) % 8 for q in environment["train"]}
            self.assertEqual(inferred, offsets)
            train = {(q["state"], tuple(q["commands"])) for q in environment["train"]}
            test = {(q["state"], tuple(q["commands"])) for q in environment["direct"] + environment["compose"]}
            self.assertEqual((len(train), len(test)), (16, 48))
            self.assertFalse(train & test)
            for q in environment["train"] + environment["direct"] + environment["compose"]:
                self.assertEqual(q["target"], (q["state"] + sum(offsets[c] for c in q["commands"])) % 8)


if __name__ == "__main__":
    unittest.main()
