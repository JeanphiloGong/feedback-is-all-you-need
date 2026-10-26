# Study I: learned program priors

The frozen design is [protocol v9](../../notes/protocol-v9-neural-prior.md).
This experiment learns a distribution over primitive sequences from solutions
accepted by an external checker. The network does not read task examples;
symbolic search and verification remain supplied parts of the complete system.

The core contains a 1,172/13,892-parameter GRU, sequential replay updates,
a count-based prior, finite ranked enumeration, and a budgeted first-fit checker.
No new task data or official outcomes are included in this implementation slice.

```bash
# Pure search checks (standard library only).
python3 -s -m unittest experiments.study_i.test_search -v

# All core checks, in the recorded PyTorch environment.
python -m unittest discover -s experiments/study_i -p 'test_*.py' -v
```

The hand-built fixtures verify full finite ordering, count smoothing, budget
boundaries, independent execution costs, actual tensor counts, causal prefixes,
padding exclusion, persistent optimizer state, and unchanged frozen weights.

Data flow is accepted primitive sequences → replay tensors → GRU updates →
prior-ranked candidate programs → visible-example execution → fixed result →
separate held-out evaluation. Hidden labels and latent generating programs are
not accepted by the core training or search interfaces. Controlled internal
programs use A–D encoded as 0–3, and token 4 is START. The count table and
probability callback serve actual alternative priors; there is no compatibility
layer or generic agent framework.
