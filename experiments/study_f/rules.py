"""Structured rule proposals; model text is parsed as data, never executed."""

import json
import re

from experiments.study_b.tasks import COMMANDS, execute, query_text

SYSTEM = "Infer the command offsets from the examples. Return only the requested JSON object."
TASK = (
    "States are integers 0 through 7. Each command (dax, wug, zup, fep) adds "
    "its own fixed but unknown integer offset modulo 8. The four offsets are "
    "distinct and may include 0. Infer the four offsets from all examples. "
    "Return a JSON object with exactly the keys dax, wug, zup, fep and four "
    "distinct integer values from 0 to 7. Do not return explanations or other text."
)
REVIEW = "Check your previous proposal against all the examples and revise it if needed. Return only the requested JSON object."


def parse_rule(text):
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        pairs = json.loads(body, object_pairs_hook=list)
    except (json.JSONDecodeError, RecursionError):
        return None
    # The raw text must encode an object, not a list of key/value pairs.
    if not body.lstrip().startswith("{") or not isinstance(pairs, list):
        return None
    if len(pairs) != 4 or {key for key, _ in pairs} != set(COMMANDS):
        return None
    rule = dict(pairs)
    if any(type(value) is not int or value not in range(8) for value in rule.values()):
        return None
    if len(set(rule.values())) != 4:
        return None
    return {command: rule[command] for command in COMMANDS}


def check_rule(rule, examples):
    if rule is None:
        return "External check: the proposal violates the declared output format."
    offsets = [rule[command] for command in COMMANDS]
    for query in examples:
        prediction = execute(query["state"], query["commands"], offsets)
        if prediction != query["target"]:
            return (f"External check: first failing example: {query_text(query)} "
                    f"{query['target']}. Your proposal predicts {prediction}.")
    return "External check: all 16 examples pass."


def proposal_messages(examples, previous=None, check=None):
    task = TASK + "\nExamples:\n" + "\n".join(
        f"{query_text(query)} {query['target']}" for query in examples)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    if previous is not None:
        review = REVIEW if check is None else check + "\n" + REVIEW
        messages.extend([{"role": "assistant", "content": previous},
                         {"role": "user", "content": review}])
    return messages
