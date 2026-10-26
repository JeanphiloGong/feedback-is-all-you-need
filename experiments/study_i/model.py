"""A program-prefix GRU trained only on accepted experience programs."""

import torch
from torch import nn
from torch.nn import functional as F


class ProgramPrior(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.recurrent = nn.GRU(5, hidden, batch_first=True, dtype=torch.float64)
        self.head = nn.Linear(hidden, 4, dtype=torch.float64)

    def forward(self, tokens):
        features = F.one_hot(tokens, num_classes=5).to(torch.float64)
        states, _ = self.recurrent(features)
        return self.head(states)

    @torch.no_grad()
    def probabilities(self, prefix):
        tokens = torch.tensor([[4, *prefix]], dtype=torch.long)
        return self(tokens)[0, -1].softmax(dim=-1).tolist()


def replay_steps(model, optimizer, programs, steps=16):
    """Yield each completed update's loss and unclipped norm for journaling."""
    if not programs:
        return
    width = max(map(len, programs))
    inputs = torch.zeros((len(programs), width), dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for row, program in enumerate(programs):
        inputs[row, :len(program)] = torch.tensor([4, *program[:-1]])
        targets[row, :len(program)] = torch.tensor(program)
    target_tokens = sum(map(len, programs))
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).reshape(-1, 4), targets.reshape(-1))
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
                   target_tokens=target_tokens)
