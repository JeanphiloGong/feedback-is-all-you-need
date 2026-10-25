"""Test feedback separation and failure-record persistence without a model download."""

import io
import json
import pytest

from experiments.study_e import run
from experiments.study_b.tasks import make_environment, messages


def test_study_e_stream_is_disjoint_from_held_out_queries():
    environment = make_environment(7000)
    train = {(query["state"], tuple(query["commands"])) for query in environment["train"]}
    held_out = {(query["state"], tuple(query["commands"]))
                for query in environment["direct"] + environment["compose"]}
    assert not train & held_out
    assert len(train) == 16 and len(held_out) == 48


def test_feedback_target_is_not_rendered_by_the_query_message():
    query = dict(state=3, commands=["dax"], target=6)
    altered = dict(query, target=1)
    assert messages(query, []) == messages(altered, [])


def test_failed_update_keeps_its_pre_update_prediction(monkeypatch, tmp_path):
    environment = make_environment(13)
    monkeypatch.setattr(run, "attach_adapter", lambda *args: (object(), object(), []))
    monkeypatch.setattr(run, "tensor_hash", lambda *args: "initial")
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: [])
    monkeypatch.setattr(run, "score", lambda *args: [dict(
        query=environment["train"][0], target=environment["train"][0]["target"],
        prediction=0, correct=False, prompt_sha256="saved-prompt", input_tokens=10,
        batch_wall_seconds=0.1,
    )])
    def failed_update(*args):
        raise RuntimeError("simulated optimizer failure")
    monkeypatch.setattr(run, "update_once", failed_update)
    training = io.StringIO()
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        run.run_feedback(object(), object(), [], environment, "online_lora",
                         io.StringIO(), training, tmp_path, [])
    rows = [json.loads(line) for line in training.getvalue().splitlines()]
    assert len(rows) == 1 and rows[0]["event"] == "prediction"
    assert rows[0]["prompt_sha256"] == "saved-prompt"


@pytest.mark.parametrize("condition,correct,expected_updates", [
    ("online_lora", True, 16), ("online_lora", False, 16),
    ("error_only_lora", True, 0), ("error_only_lora", False, 16),
    ("context", False, 0),
])
def test_feedback_changes_only_the_assigned_state(monkeypatch, tmp_path, condition, correct, expected_updates):
    class Model:
        def save_pretrained(self, path, **kwargs):
            (path / "weights").write_bytes(b"adapter")
        def unload(self):
            return self
    model = Model()
    environment = make_environment(13)
    seen_history = []
    updates = []
    def predict(model, tokenizer, digits, queries, history, batch):
        seen_history.append(len(history))
        return [dict(query=queries[0], prediction=0, correct=correct)]
    monkeypatch.setattr(run, "score", predict)
    monkeypatch.setattr(run, "attach_adapter", lambda *args: (model, object(), []))
    monkeypatch.setattr(run, "tensor_hash", lambda *args: "initial")
    monkeypatch.setattr(run, "evaluate", lambda *args: [])
    monkeypatch.setattr(run, "update_once", lambda *args: updates.append(args[3]))
    training = io.StringIO()
    _, record = run.run_feedback(model, object(), [], environment, condition,
                                 io.StringIO(), training, tmp_path, [])
    events = [json.loads(line) for line in training.getvalue().splitlines()]
    assert [row["event"] for row in events] == ["prediction", "feedback"] * 16
    assert len(updates) == record["applied_updates"] == expected_updates
    assert seen_history == (list(range(16)) if condition == "context" else [0] * 16)
