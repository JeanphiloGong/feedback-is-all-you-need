# Study I: learned program priors

The frozen design is [protocol v9](../../notes/protocol-v9-neural-prior.md).
This experiment learns a distribution over primitive sequences from solutions
accepted by an external checker. The network does not read task examples;
symbolic search and verification remain supplied parts of the complete system.

The core contains a 1,172/13,892-parameter GRU, sequential replay updates,
a count-based prior, finite ranked enumeration, and a budgeted first-fit checker.
No official task outcomes have been collected at this implementation checkpoint.

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

## Collection and evidence

The collector records searches before held-out evaluation, journals every
completed gradient step and resource phase, and explicitly records attempts
skipped for an empty corpus. Initial weights and optimizer-bearing checkpoints
are retained. `active-phase.json` identifies an interrupted operation; failure
records reference that file as raw evidence even if its write was interrupted.
No failure-handling code assumes its contents are valid JSON.

Accepted-solution controls use zero-based experience indices in their shuffle
RNG roles. The frozen input grid must contain at least 16 experience attempts
per environment to support the declared intermediate checkpoints; the collector
refuses an incompatible grid rather than substituting another checkpoint.

After the source and data freeze, use the recorded PyTorch environment:

```bash
python -m experiments.study_i.run --development \
  --source-revision SOURCE_COMMIT --output runs/study-i-development
python -m experiments.study_i.run \
  --source-revision SOURCE_COMMIT --output runs/study-i-formal \
  --development-audit runs/study-i-development-audit/audit.json
```

`SOURCE_COMMIT` must identify the source archive actually transferred to the
execution host. File hashes are also recorded, and formal collection requires
a passing development audit with the same source and data. Each attempt uses
a new output directory. No incomplete grid supports a primary conclusion.
The audit/analysis implementation is a separate checkpoint; the commands above
do not bypass that gate.
