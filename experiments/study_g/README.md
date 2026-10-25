# Study G: complete feedback history

The contract is [`../../notes/protocol-v7-feedback-history.md`](../../notes/protocol-v7-feedback-history.md).
Forty-eight fresh environments compare 16 checked revisions with only the last
proposal against 16 revisions retaining the complete proposal/check transcript.
Initial generation is shared. The first revision uses identical ordered batches,
inputs, and greedy outputs; any mismatch is preserved and fails the suite.

No weights change. All examples are available initially, checks compute known
information, the declared F parser is retained, and a supplied executor answers
held-out queries from the last saved rule. Additional context changes cost;
equal call caps do not establish equal compute. Every intermediate proposal is
scored and retained; no better earlier candidate replaces the final one.

From the repository root, using the same packages and pinned models as F:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest experiments.study_g.test_history experiments.study_g.test_pairing -v
python -m experiments.study_g.run --size 0.5B --development --output runs/g-dev-0.5B
python -m experiments.study_g.analyze --input runs/g-dev-0.5B
python -m experiments.study_g.run --size 0.5B --output runs/g-formal-0.5B \
  --development-audit runs/g-dev-0.5B/audit.json
python -m experiments.study_g.analyze --input runs/g-formal-0.5B
```

Set `FEEDBACK_GIT_REVISION` to the frozen source commit before launching.
Repeat with 7B and separate output directories. Frozen data SHA-256:
`a520ddb1a69ff40502abe49659de80d310dadd91cd3f7a50c160d83617e38b65`.
Development uses only seed 19; formal seeds are 9000–9047.

The primary contrast is final 0.5B direct accuracy, full history minus last only,
paired over environments. Exact recovery, composition, repetition, and 7B are
descriptive. Repetition strata separate valid wrong, exact, and invalid rules;
repeated success is not failure. Component diagnostics use only valid pairs.
All initialization and collection failures are recorded after run creation;
preflight rejection of missing development evidence occurs before creating a run.

## Exploratory selection after manuscript v8

The separate post-hoc plan is
[`../../notes/study-g-selection-plan.md`](../../notes/study-g-selection-plan.md).
Select the earliest best training-fit rule from each recorded prefix without
changing formal scores, generation histories, or model weights:

```bash
python3 -s -m unittest experiments.study_g.test_selection -v
python3 -s -m experiments.study_g.selection --input results/study-g-v1 \
  --output /tmp/study-g-selection.json
```

The selector evaluates all disclosed examples, adding supplied computation.
Its direct-score monotonicity follows from this task family's structure;
composition can decline. This is not a run with gated model feedback or a
measurement of learning across tasks. The output must be a new file.

The later uniform-proposal reference is specified in
[`../../notes/study-g-random-reference-plan.md`](../../notes/study-g-random-reference-plan.md).
It calculates exact expectations for 17 uniform legal rules or 16 uniform
revisions after the shared neural initial proposal. It adds no model calls:

```bash
python3 -s -m unittest experiments.study_g.test_random_reference -v
python3 -s -m experiments.study_g.random_reference \
  --input results/study-g-v1/selection-diagnostics.json \
  --output /tmp/study-g-random-reference.json
```

The sampler has a supplied finite grammar and always produces valid rules.
Candidate opportunities do not match measured compute or checker executions;
conditional expectations are not observed runs or confidence intervals.
