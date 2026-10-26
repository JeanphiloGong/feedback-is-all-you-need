"""Check a conditional architecture formula, not any published DreamCoder run."""

import json

import torch
from torch import nn


records = []
for vocabulary in (16, 100):
    for grammar_size in (8, 32, 128):
        components = nn.ModuleDict({
            "embedding": nn.Embedding(vocabulary, 64),
            "recurrent": nn.GRU(64, 64, num_layers=1, bidirectional=True),
            "mlp": nn.Linear(64, 64),
            "grammar": nn.Linear(64, grammar_size + 1),
            "auxiliary": nn.Linear(64, grammar_size),
        })
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in components.items()}
        total = sum(counts.values())
        formula = 64 * vocabulary + 54145 + 130 * grammar_size
        assert total == formula
        records.append(dict(vocabulary=vocabulary, grammar_size=grammar_size,
                            counts=counts, total=total, formula=formula))
print(json.dumps(dict(torch=torch.__version__,
    purpose="Synthetic architecture-count checks only; no DreamCoder code executed or published model measured",
    records=records), indent=2))
