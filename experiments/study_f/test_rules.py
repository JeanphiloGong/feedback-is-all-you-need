"""Behavioral checks for rule boundaries, leakage, and score interpretation."""

import copy
import json
from pathlib import Path
import unittest

from experiments.study_b.tasks import COMMANDS, execute, make_environment, symbolic_offsets
from experiments.study_f.rules import check_rule, parse_rule, proposal_messages


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_environment(17)
        self.rule = dict(zip(COMMANDS, self.environment["offsets"]))
        self.text = json.dumps(self.rule)

    def test_accepts_exact_object_or_single_outer_fence(self):
        for response in (self.text, f"```json\n{self.text}\n```", f"```\n{self.text}\n```", f" \n{self.text}\n "):
            self.assertEqual(parse_rule(response), self.rule)

    def test_rejects_malformed_or_nonobject_output(self):
        for response in ("", "{}", self.text[:-1], "Here is the rule: " + self.text,
                         self.text + "\nDone", json.dumps(list(self.rule.items())),
                         "null", "true", "7", self.text + self.text):
            with self.subTest(response=response):
                self.assertIsNone(parse_rule(response))

    def test_rejects_duplicate_keys_even_if_values_agree(self):
        response = '{"dax":0,"wug":1,"zup":2,"fep":3,"dax":0}'
        self.assertIsNone(parse_rule(response))

    def test_deep_malformed_response_counts_as_invalid(self):
        # The pinned Qwen tokenizer encodes these 1,024 brackets in 512 tokens.
        self.assertIsNone(parse_rule("[" * 1024))

    def test_rejects_wrong_types_ranges_keys_and_repeated_values(self):
        for value in (True, 1.0, "1", -1, 8, None, {}, [], float("nan")):
            self.assertIsNone(parse_rule(json.dumps(dict(self.rule, dax=value))))
        self.assertIsNone(parse_rule('{"dax":0,"wug":1,"zup":2,"other":3}'))
        self.assertIsNone(parse_rule('{"dax":0,"wug":1,"zup":2,"fep":2}'))

    def test_checker_uses_first_failure_and_disclosed_target(self):
        wrong = {c: (v + 1) % 8 for c, v in self.rule.items()}
        first = self.environment["train"][0]
        expected = (first["target"] + 1) % 8
        message = check_rule(wrong, self.environment["train"])
        self.assertIn(f"Start: {first['state']}; commands: {first['commands'][0]}; result: {first['target']}", message)
        self.assertIn(f"Your proposal predicts {expected}.", message)
        self.assertEqual(check_rule(self.rule, self.environment["train"]), "External check: all 16 examples pass.")
        self.assertEqual(check_rule(None, self.environment["train"]),
                         "External check: the proposal violates the declared output format.")

    def test_prompts_ignore_heldout_inputs_labels_and_secret_offsets(self):
        mutated = copy.deepcopy(self.environment)
        mutated["offsets"] = [100, 101, 102, 103]
        for split in ("direct", "compose"):
            for query in mutated[split]:
                query.update(state=999, commands=["secret"], target=888)
        for previous, checked in ((None, False), (self.text, False), (self.text, True)):
            before = check_rule(self.rule, self.environment["train"]) if checked else None
            after = check_rule(self.rule, mutated["train"]) if checked else None
            self.assertEqual(proposal_messages(self.environment["train"], previous, before),
                             proposal_messages(mutated["train"], previous, after))

    def test_all_frozen_splits_labels_and_symbolic_solutions(self):
        data = json.loads(Path("experiments/study_f/data/environments-v1.json").read_text())
        self.assertEqual([e["seed"] for e in data["environments"]], [17, *range(8000, 8024)])
        for environment in data["environments"]:
            self.assertEqual(symbolic_offsets(environment["train"]), environment["offsets"])
            train = {(q["state"], tuple(q["commands"])) for q in environment["train"]}
            test = {(q["state"], tuple(q["commands"])) for q in environment["direct"] + environment["compose"]}
            self.assertEqual((len(train), len(test)), (16, 48))
            self.assertFalse(train & test)
            for query in environment["train"] + environment["direct"] + environment["compose"]:
                self.assertEqual(query["target"], execute(query["state"], query["commands"], environment["offsets"]))

    def test_composition_can_pass_with_an_entirely_wrong_rule(self):
        shifted = [(value + 4) % 8 for value in self.environment["offsets"]]
        self.assertTrue(all(a != b for a, b in zip(shifted, self.environment["offsets"])))
        for split, expected_correct in (("direct", False), ("compose", True)):
            for query in self.environment[split]:
                self.assertEqual(execute(query["state"], query["commands"], shifted) == query["target"], expected_correct)


if __name__ == "__main__":
    unittest.main()
