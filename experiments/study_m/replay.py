"""Independent reference arithmetic; never calls the collector's update loop."""

import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def reference_rng(seed, role):
    digest = hashlib.sha256(json.dumps([seed, role], separators=(',', ':')).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'little'))


def independent_initial(seed, family, hidden):
    torch.manual_seed(reference_rng(seed, f'study_m:initial:{family}:{hidden}').randrange(2 ** 31))
    layers = (nn.Linear(9, hidden, dtype=torch.float64), nn.Linear(hidden, 5, dtype=torch.float64))
    parameters = [p for layer in layers for p in layer.parameters()]
    return parameters, torch.optim.Adam(parameters, lr=.01, foreach=False, fused=False)


def independent_replay(parameters, optimizer, corpus, seed, family, first_step, steps):
    keys = sorted(corpus)
    inputs = torch.tensor([[int(bit) for bit in format(s, '05b')[::-1]] +
                           [int(a == k) for k in range(4)] for s, a in keys], dtype=torch.float64)
    targets = torch.tensor([[int(bit) for bit in format(corpus[key], '05b')[::-1]]
                            for key in keys], dtype=torch.float64)
    w, b, v, c = parameters
    for step in range(first_step, first_step + steps):
        generator = reference_rng(seed, f'study_m:sampling:{family}:{len(b)}:{step}')
        selected = [generator.randrange(len(keys)) for _ in range(32)]
        sampled = json.dumps([keys[i] for i in selected], separators=(',', ':')).encode()
        optimizer.zero_grad(set_to_none=True)
        logits = F.linear(torch.relu(F.linear(inputs[selected], w, b)), v, c)
        loss = F.binary_cross_entropy_with_logits(logits, targets[selected])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(gradient_step=step, loss_before_step=loss.item(),
            gradient_norm_before_clip=norm.item(), sampled_keys_sha256=hashlib.sha256(sampled).hexdigest(),
            input_rows=32, target_bits=160, dense_training_macs=32 * 33 * len(b))


@torch.no_grad()
def independent_graph(parameters):
    inputs = torch.tensor([[int(bit) for bit in format(s, '05b')[::-1]] +
                           [int(a == k) for k in range(4)]
                           for a in range(4) for s in range(32)], dtype=torch.float64)
    w, b, v, c = parameters
    logits = F.linear(torch.relu(F.linear(inputs, w, b)), v, c)
    matrices = [p.detach().numpy() for p in parameters]
    expected = np.maximum(inputs.numpy() @ matrices[0].T + matrices[1], 0) @ matrices[2].T + matrices[3]
    np.testing.assert_allclose(logits.numpy(), expected, rtol=1e-11, atol=1e-12)
    values = [sum(2 ** j for j, value in enumerate(row) if value >= 0) for row in logits.tolist()]
    return [values[i:i + 32] for i in range(0, 128, 32)], dict(
        logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        input_rows=128, dense_forward_macs=128 * 14 * len(b))


def reference_policy(graph, state, goal, fallback):
    queue, parents, depth = [state], {state: None}, {state: 0}
    expanded = edges = 0
    if graph is not None:
        for current in queue:
            if depth[current] >= 8:
                continue
            expanded += 1
            for action in range(4):
                edges += 1
                following = graph[action][current]
                if following is None or following in parents:
                    continue
                parents[following] = current, action
                depth[following] = depth[current] + 1
                if following == goal:
                    path, cursor = [], following
                    while parents[cursor] is not None:
                        cursor, token = parents[cursor]
                        path.insert(0, token)
                    return dict(action=path[0], planned_path=path, expanded_states=expanded,
                                edge_checks=edges, fallback=False)
                queue.append(following)
    return dict(action=fallback, planned_path=[], expanded_states=expanded,
                edge_checks=edges, fallback=True)


def reference_world(seed, family):
    generator = reference_rng(seed, f'study_m:{family}:maps')
    rejections = dict(worlds=0, actions=0)
    while True:
        maps, permutations, masks = [], [], []
        while len(maps) < 4:
            if family == 'structured':
                permutation = generator.sample(range(5), 5)
                mask = generator.randrange(32)
                candidate = [sum(((s >> permutation[j]) & 1) << j for j in range(5)) ^ mask
                             for s in range(32)]
            else:
                candidate = generator.sample(range(32), 32)
            if candidate in maps or candidate == list(range(32)):
                rejections['actions'] += 1
                continue
            maps.append(candidate)
            if family == 'structured':
                permutations.append(permutation)
                masks.append(mask)
        reachability = np.eye(32, dtype=bool)
        for action in maps:
            reachability[np.arange(32), action] = True
        for pivot in range(32):
            reachability |= reachability[:, pivot, None] & reachability[None, pivot, :]
        if reachability.all():
            break
        rejections['worlds'] += 1
    held_out = sorted(reference_rng(seed, f'study_m:{family}:goals').sample(range(32), 8))
    training = [s for s in range(32) if s not in held_out]
    generator = reference_rng(seed, f'study_m:{family}:episodes')
    episodes = []
    for _ in range(513):
        goal = generator.choice(training)
        episodes.append(dict(start=generator.choice([s for s in range(32) if s != goal]), goal=goal))
    generator = reference_rng(seed, f'study_m:{family}:warmup')
    result = dict(seed=seed, family=family, primitives=maps, rejections=rejections,
        held_out_goals=held_out, training_goals=training, episodes=episodes,
        warmup=dict(start=generator.randrange(32), actions=[generator.randrange(4) for _ in range(16)]),
        probes=[dict(start=s, goal=g) for g in held_out for s in range(32) if s != g])
    if family == 'structured':
        result.update(bit_permutations=permutations, masks=masks)
    return result
