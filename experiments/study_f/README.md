# Study F: externally checked rule revision

The prospective contract is in
[`../../notes/protocol-v6-rule-memory.md`](../../notes/protocol-v6-rule-memory.md).
This study compares two fixed revisions with or without an external check of
already-disclosed examples. A shared initial proposal is generated once. The
saved four-offset JSON rule is used by a supplied deterministic executor; no
model-generated code is executed, and the model never sees held-out queries.

The model-output boundary accepts only the declared JSON object or one outer
newline-delimited code fence. Malformed output is a realistic model failure:
it produces a null memory and scores zero, with the raw response retained.
The frozen task file is a controlled input checked by its SHA-256; the parser
is applied to model text once per proposal, and the analyzer repeats that
parse from decoded raw token IDs for audit.

Run from the repository root with the pinned models cached locally and a CUDA
GPU. Dependencies match the Study E environment: PyTorch 2.8.0+cu126,
Transformers 4.57.1, NumPy 2.3.4, SciPy 1.16.3, Matplotlib 3.10.7.

```bash
python -m unittest experiments.study_f.test_rules -v
python -m experiments.study_f.run --size 0.5B --development --output runs/f-dev-0.5B
python -m experiments.study_f.analyze --input runs/f-dev-0.5B
python -m experiments.study_f.run --size 0.5B --output runs/f-formal-0.5B \
  --development-audit runs/f-dev-0.5B/audit.json
python -m experiments.study_f.analyze --input runs/f-formal-0.5B
```

Repeat with `--size 7B` and separate output directories. Set
`FEEDBACK_GIT_REVISION` to the frozen source commit when launching. Each formal
run requires a prior matching development audit; output directories cannot be
reused. All development/formal environments are frozen together at SHA-256
`3713e3d93f4221224e247b9c2b6454f1e4fec4df03fa8cfd93b1babf7c150215`.

The runner records raw proposals, prompts, checks, every intermediate memory,
tokens through EOS, batch padding, and measured times. The analyzer checks the
complete grid and recomputes all held-out scores. Token counts are work proxies,
not FLOPs. Its per-stage costs count the initial call once in the actual run;
a hypothetical standalone arm would add that same initial cost to its two
revision stages. Checker time includes diagnostic checks of every proposal;
only the initial check and checked arm's first revision check enter prompts.
Evaluation and plotting time are outside runner elapsed time.

The primary contrast is final checked-minus-self composition accuracy in
0.5B, paired across 24 environments. Direct accuracy and exact-rule recovery
must also be reported: shifting every offset by four gives a fully wrong rule
with perfect two-command composition accuracy. Intervals on the remaining
outcomes are descriptive. No neural retention, cross-language learning, or
general-intelligence claim follows from this file-based memory experiment.
