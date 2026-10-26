"""Learn observed next-state bits, retaining weights across goal episodes."""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from experiments.study_h.tasks import rng


def state_bits(states):
    return ((states[:, None] >> torch.arange(5)) & 1).to(torch.float64)


def features(keys):
    values = torch.tensor(keys, dtype=torch.long)
    return torch.cat((state_bits(values[:, 0]),
                      F.one_hot(values[:, 1], num_classes=4).to(torch.float64)), dim=1)


class TransitionModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.input_layer = nn.Linear(9, hidden, dtype=torch.float64)
        self.output_layer = nn.Linear(hidden, 5, dtype=torch.float64)

    def forward(self, inputs):
        return self.output_layer(F.relu(self.input_layer(inputs)))


@torch.no_grad()
def predicted_graph(model):
    keys = [(state, action) for action in range(4) for state in range(32)]
    logits = model(features(keys))
    following = ((logits >= 0).long() * (2 ** torch.arange(5))).sum(dim=1).tolist()
    graph = [following[action * 32:(action + 1) * 32] for action in range(4)]
    return graph, dict(logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        input_rows=128, dense_forward_macs=128 * 14 * model.input_layer.out_features)


def replay_steps(model, optimizer, corpus, seed, family, first_step, steps):
    """The controlled caller supplies a nonempty stationary observed corpus."""
    keys = sorted(corpus)
    inputs = features(keys)
    targets = state_bits(torch.tensor([corpus[key] for key in keys]))
    hidden = model.input_layer.out_features
    model.train()
    for step in range(first_step, first_step + steps):
        source = rng(seed, f'study_m:sampling:{family}:{hidden}:{step}')
        indices = [source.randrange(len(keys)) for _ in range(32)]
        sampled = json.dumps([keys[index] for index in indices], separators=(',', ':')).encode()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(inputs[indices]), targets[indices])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(gradient_step=step, loss_before_step=loss.item(),
            gradient_norm_before_clip=norm.item(), sampled_keys_sha256=hashlib.sha256(sampled).hexdigest(),
            input_rows=32, target_bits=160, dense_training_macs=32 * 33 * hidden)
