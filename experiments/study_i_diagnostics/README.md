# Study I: post-hoc likelihood diagnostics

The [frozen analysis plan](../../notes/study-i-likelihood-diagnostic-plan.md)
describes all formal checkpoints after manuscript v11. This directory is
separate from the original experiment so its frozen source hashes remain
unchanged. There is no training, search rerun, or change to the primary result.
The [verified results](../../results/study-i-likelihood-v1/README.md) retain all
formal checkpoints, input hashes, and independent summary reconstructions.

Input is the controlled archive of 967 files, checked once against the passing
original audit before checkpoint loading. A mismatch aborts the analysis and
is recorded. Output is a new directory of program-level token scores, complete
summaries, source/input hashes, and measured analysis time. A failed attempt
retains its flushed records and error; an existing directory is never reused.
The fixed archive has 24 accepted length-five programs in every environment
and eight transfer programs of each length six/nine per family. Every reported
length/position stratum therefore contains all 32 environments; the verifier
checks that known shape rather than supporting unrelated datasets.

In the original PyTorch environment, run:

```bash
python -m unittest experiments.study_i_diagnostics.test_likelihood -v
python -m experiments.study_i_diagnostics.likelihood \
  --input results/study-i-v1/formal-v1 \
  --audit results/study-i-v1/formal-audit-v1/audit.json \
  --source-revision DIAGNOSTIC_SOURCE_COMMIT \
  --output runs/study-i-likelihood-v1
```

Every neural target probability and its log are compared with independent
NumPy GRU equations. Hand-built tests check conflicting-prefix entropy,
smoothing, causal targets, attempted-task membership, and environment weights.
The original audit already exactly replayed all optimizer and model states.

Independently reconstruct the full expected program grid and every summary
with the standard-library verifier:

```bash
python3 -s -m experiments.study_i_diagnostics.verify \
  --input results/study-i-v1/formal-v1 \
  --diagnostic runs/study-i-likelihood-v1 \
  --output runs/study-i-likelihood-verification-v1.json
```

Losses use natural logarithms per target token, excluding EOS and the common
length prior. Aggregates weight environments equally. The empirical prefix
bound is an unrestricted in-sample optimum. Generating-program likelihood is
not total semantic solution mass or proof of the cause of transfer failure.
