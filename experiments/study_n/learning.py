"""Retain neural weights and Adam state on a controlled observation stream."""

import hashlib
import json

import torch
from torch.nn import functional as F

from experiments.study_h.tasks import rng
from experiments.study_m.model import features, state_bits


def update_steps(model, optimizer, observations, seed, first_step, steps):
    inputs = features([(s, a) for s, a, _ in observations])
    targets = state_bits(torch.tensor([following for _, _, following in observations]))
    hidden = model.input_layer.out_features
    model.train()
    for step in range(first_step, first_step + steps):
        source = rng(seed, f'study_n:sampling:{hidden}:{step}')
        selected = [source.randrange(len(observations)) for _ in range(32)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(inputs[selected]), targets[selected])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(gradient_step=step, loss=loss.item(), gradient_norm=norm.item(),
                   indices_sha256=hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode()).hexdigest())
