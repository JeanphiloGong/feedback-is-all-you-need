"""Separate arithmetic for world generation and ordered gradient replay."""

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.study_m.replay import reference_rng


def reference_world(seed):
    source = reference_rng(seed, 'study_n:maps')
    rejected = dict(actions=0, worlds=0)
    while True:
        maps, permutations, masks = [], [], []
        while len(maps) < 4:
            permutation, mask = source.sample(range(5), 5), source.randrange(32)
            values = [sum(((s >> permutation[j]) & 1) << j for j in range(5)) ^ mask for s in range(32)]
            if values in maps or values == list(range(32)):
                rejected['actions'] += 1
                continue
            maps.append(values)
            permutations.append(permutation)
            masks.append(mask)
        reach = np.eye(32, dtype=bool)
        for action in maps:
            reach[np.arange(32), action] = True
        for pivot in range(32):
            reach |= reach[:, pivot, None] & reach[None, pivot, :]
        if reach.all():
            break
        rejected['worlds'] += 1
    labels = reference_rng(seed, 'study_n:labels').sample(range(32), 32)
    source_lists = [reference_rng(seed, f'study_n:sources:{a}').sample(range(32), 32) for a in range(4)]
    source = reference_rng(seed, 'study_n:query_order')
    queries = []
    for index in range(16):
        for action in source.sample(range(4), 4):
            queries.append([source_lists[action][index], action])
    test = [[s, a] for a in range(4) for s in range(32) if [s, a] not in queries]
    return dict(seed=seed, primitives=maps, bit_permutations=permutations,
                masks=masks, rejections=rejected, labels=labels, queries=queries, test=test)


def reference_initial(seed, hidden):
    torch.manual_seed(reference_rng(seed, f'study_n:initial:{hidden}').randrange(2 ** 31))
    layers = [nn.Linear(9, hidden, dtype=torch.float64), nn.Linear(hidden, 5, dtype=torch.float64)]
    parameters = [p for layer in layers for p in layer.parameters()]
    return parameters, torch.optim.Adam(parameters, lr=.01, foreach=False, fused=False)


def reference_inputs(pairs):
    return torch.tensor([[int(c) for c in f'{state:05b}'[::-1]] + [int(a == action) for a in range(4)]
                         for state, action in pairs], dtype=torch.float64)


def reference_logits(parameters, inputs):
    w, b, v, c = parameters
    return F.linear(torch.relu(F.linear(inputs, w, b)), v, c)


def reference_updates(parameters, optimizer, observations, seed, first, count):
    inputs = reference_inputs([(s, a) for s, a, _ in observations])
    targets = torch.tensor([[int(c) for c in f'{y:05b}'[::-1]] for _, _, y in observations], dtype=torch.float64)
    for step in range(first, first + count):
        source = reference_rng(seed, f'study_n:sampling:{len(parameters[1])}:{step}')
        indices = [source.randrange(len(observations)) for _ in range(32)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(reference_logits(parameters, inputs[indices]), targets[indices])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(gradient_step=step, loss=loss.item(), gradient_norm=norm.item(),
            indices_sha256=hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest())
