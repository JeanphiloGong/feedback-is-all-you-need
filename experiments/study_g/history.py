"""Retain prior proposal/check turns without changing the review instruction."""

from experiments.study_f.rules import REVIEW, proposal_messages


def history_messages(examples, history, arm):
    messages = proposal_messages(examples)
    selected = history if arm == "full_history" else history[-1:]
    for previous in selected:
        messages.extend([{"role": "assistant", "content": previous["text"]},
                         {"role": "user", "content": previous["check"] + "\n" + REVIEW}])
    return messages
