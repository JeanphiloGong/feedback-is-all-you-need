from __future__ import annotations

import torch

from experiments.study_a import run
from experiments.study_a.run import Agent, Rule, make_data, make_rule, parameter_count


def test_parameter_counts_are_frozen_by_protocol() -> None:
    assert [parameter_count(width) for width in (0, 16, 64, 256)] == [9, 161, 641, 2561]


def test_rule_generation_and_data_are_deterministic() -> None:
    rule_a = make_rule("xor", 1000)
    rule_b = make_rule("xor", 1000)
    x_a, y_a = make_data(rule_a, 2, 2000)
    x_b, y_b = make_data(rule_b, 2, 2000)
    assert rule_a.pair == rule_b.pair
    assert rule_a.invert == rule_b.invert
    assert torch.equal(x_a, x_b)
    assert torch.equal(y_a, y_b)


def test_xor_rule_is_not_linearly_constant() -> None:
    rule = Rule(family="xor", pair=(0, 1))
    x = torch.tensor([[1.0, 1.0, 0, 0], [1.0, -1.0, 0, 0], [-1.0, 1.0, 0, 0], [-1.0, -1.0, 0, 0]])
    assert rule.labels(x).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_agent_outputs_one_logit_per_example() -> None:
    x = torch.randn(5, 8)
    for width in (0, 16, 64, 256):
        assert Agent(width)(x).shape == (5,)


def test_frozen_weights_and_delayed_flush_preserve_expected_contract(monkeypatch) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(run, "ADAPTATION_BATCHES", 40)
    monkeypatch.setattr(run, "CHECKPOINTS", (0, 16, 32, 40))
    rule = make_rule("linear", 7)
    x, y = make_data(rule, 40, 9)
    model = Agent(0)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    kwargs = dict(family="linear", width=0, seed=7, warm_state=state,
                  warm_x=x, warm_y=y, adapt_x=x, adapt_y=y,
                  eval_sets={"heldout": make_data(rule, 2, 11)}, device=torch.device("cpu"))
    frozen = run.run_arm(arm="frozen", **kwargs)
    assert all(r["accuracy_heldout"] == frozen[0]["accuracy_heldout"] for r in frozen)
    assert frozen[-1]["updates"] == 0
    clean = run.run_arm(arm="clean", **kwargs)
    delayed = run.run_arm(arm="delay32", **kwargs)
    assert delayed[-2]["updates"] == 8
    assert delayed[-1]["updates"] == 40
    assert delayed[-1]["accuracy_heldout"] == clean[-1]["accuracy_heldout"]
    assert delayed[-1]["last_loss"] == clean[-1]["last_loss"]


def test_noise50_does_not_leak_labels_and_inversion_is_exact() -> None:
    labels = torch.zeros(10000)
    noisy = run.noisy_labels(labels, "noise50", 3)
    assert 0.48 < noisy.mean().item() < 0.52
    assert torch.equal(run.noisy_labels(labels, "inverted", 3), torch.ones_like(labels))
