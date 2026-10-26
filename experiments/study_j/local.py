"""Fit next-token targets using only two previous primitives and a fresh state."""

import torch
from torch.nn import functional as F


def local_replay_steps(model, optimizer, programs, steps=16):
    if not programs:
        return
    contexts, targets = [], []
    for program in programs:
        for index, token in enumerate(program):
            contexts.append([4, *program[max(0, index - 2):index]])
            targets.append(token)
    width = max(map(len, contexts))
    inputs = torch.tensor([row + [0] * (width - len(row)) for row in contexts])
    positions = torch.tensor([len(row) - 1 for row in contexts])
    labels = torch.tensor(targets)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)[torch.arange(len(contexts)), positions]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
                   target_tokens=len(targets), input_rows=len(contexts),
                   input_positions=inputs.numel())
