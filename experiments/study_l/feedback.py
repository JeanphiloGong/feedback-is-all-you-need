"""Task-specific weight losses using the same smoothed local program prior."""

from itertools import product

import torch


CONTEXTS = [context for length in range(3) for context in product(range(4), repeat=length)]
INPUTS = torch.tensor([[4, *context, *([0] * (2 - len(context)))] for context in CONTEXTS])
POSITIONS = torch.tensor([len(context) for context in CONTEXTS])


def local_probabilities(model):
    logits = model(INPUTS)[torch.arange(21), POSITIONS]
    return logits.softmax(dim=-1)


def program_indices(programs):
    """Index 84 is a zero log-factor used only to pad shorter programs."""
    width = max(map(len, programs))
    indices = torch.full((len(programs), width), 84, dtype=torch.long)
    for row, program in enumerate(programs):
        for position, token in enumerate(program):
            context = (() if position == 0 else tuple(program[max(0, position - 2):position]))
            context_index = (0 if not context else 1 + context[0] if len(context) == 1
                             else 5 + context[0] * 4 + context[1])
            indices[row, position] = 4 * context_index + token
    return indices


def rejected_mass(smoothed, indices):
    log_factors = torch.cat((smoothed.log().flatten(), smoothed.new_zeros(1)))
    return (log_factors[indices].sum(dim=1).exp() / 10).sum()


def feedback_steps(model, optimizer, arm, rejected, steps=16):
    if arm == 'rejection' and not rejected:
        return
    indices = program_indices(rejected) if arm == 'rejection' else None
    tokens = sum(map(len, rejected)) if arm == 'rejection' else 0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        smoothed = .9 * local_probabilities(model) + .025
        mass = rejected_mass(smoothed, indices) if arm == 'rejection' else None
        loss = -torch.log1p(-mass) if arm == 'rejection' else -smoothed.log().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
                   rejection_mass_before_step=mass.item() if mass is not None else None,
                   rejected_programs=len(rejected) if arm == 'rejection' else 0,
                   rejected_tokens=tokens, input_rows=21, input_positions=63)
