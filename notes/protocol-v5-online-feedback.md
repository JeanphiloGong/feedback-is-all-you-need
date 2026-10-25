# Study E protocol: online correction, context memory, and parameter updates

Status: prospective protocol. Freeze this document, the runner, and the
environment file before viewing formal outcomes. Study E is a mechanism test,
not a general-intelligence benchmark.

## Objective and assumptions

The question is whether repeated, externally verified outcomes produce durable
held-out improvement through one explicit online update rule. The base model,
task family, and feedback stream are fixed. The experiment does not assume that
one LoRA rule represents all feedback systems, and it does not treat a result
as evidence about human intelligence.

The environment is trusted because it is generated and frozen in the
repository. The model receives a target only after it has produced its current
prediction. The runner must keep the target out of the prediction prompt and
must record the prediction before applying any update.

## Environment and independent units

Use 24 fresh hidden command languages, seeds 7000--7023, generated with the
existing Study B generator and NumPy 2.2.4. Four commands add distinct hidden
offsets modulo eight. Each environment has 16 ordered adaptation queries,
16 held-out direct queries, and 32 held-out length-two composition queries.
The adaptation queries are the generator's `train` split; their targets are
released by the environment only after each prediction. Direct and composition
queries are never used for updates. Compositions commute, so they test transfer
within a disclosed arithmetic family rather than general sequence reasoning.

Freeze the exact JSON and its SHA-256 before formal inference. Record the model
revision, source hashes, package versions, generation settings, and complete
query/prompt identities. Environment is the statistical unit; query-level rows
are aggregated to one mean per environment before intervals.

The frozen file is `experiments/study_e/data/environments-v1.json`, SHA-256
`67fd60a700203034dccb8e79fd95c119bb1315f08d69f040ae2b0ae24784b0c9`.

## Model and constrained scoring

Use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, loaded in bfloat16 with the same
chat template and first-token scoring used by Study B. Greedy inference selects
the largest logit among the eight one-token digits. Record the unrestricted
argmax token, its decoded text, and timing as diagnostics; full logit vectors
are not retained. No sampling, chain of thought, judge, retrieval, or
additional attempts are used. This interface measures a constrained prediction
capacity and avoids Study C's output-format confound. Studies C and D showed
material response-interface and token-budget constraints; a null
result here remains conditional on this constrained interface.

## Feedback stream and conditions

For every adaptation query, the model first predicts with its current state.
The environment then reveals the correct target. The conditions are:

- `none`: frozen base model, no adaptation, evaluated at feedback count 0.
- `context`: frozen base model; after each response, append the correct query
  and target to an external history. Evaluate after 0, 1, 2, 4, 8, and 16
  feedback pairs.
- `online_lora`: add a fresh rank-8 LoRA adapter to `q_proj` and `v_proj` and
  perform one gradient update after every feedback target, using only the
  current query and revealed target. At each checkpoint evaluate with empty
  context, so any retained gain must reside in the adapter.
- `error_only_lora`: use the same adapter and optimizer, but apply the update
  only when the pre-update prediction is wrong. Correct responses still count
  as feedback events and advance the checkpoint.

For both adapter conditions, evaluate an additional final `with_context` row
at 16 feedback pairs using the complete correct history. This separates a
parameter update that persists without memory from a gain that still depends on
the transcript. The context condition is deliberately an external-memory
baseline; it is not a parameter-learning claim.

LoRA uses alpha 16, dropout 0, no bias, AdamW learning rate `2e-4`, weight
decay 0, default betas and epsilon, no scheduler or gradient clipping, and
one update per applied feedback. The loss is full-vocabulary cross-entropy on
the next digit token only. Adapters and optimizers reset to identical base
weights for every environment and condition. Both adapter conditions use the
same initialization seed `environment_seed + 4000000`; their initial parameter
hashes must match. No best checkpoint is selected. There are at most 16 updates,
fewer than Study B's 128-update schedule; this dose tests one-pass adaptation,
not the best achievable optimization schedule.

The runner asserts equal zero-feedback predictions for every condition. A full
base-parameter hash is recorded before and after the suite; development also
checks it after each condition. Only adapter parameters can require gradients.
Final adapter bytes and their hashes are retained and audited. Persistence
means retaining performance on this environment after removing demonstrations;
the study does not test retention after learning another environment.

Input selection is fixed independently of predictions. For `online_lora`,
offline replay of the same ordered labels and updates is equivalent; the
experiment makes no online-versus-offline superiority claim. `error_only_lora`
uses its prediction to gate updates and can therefore use fewer updates, so its
comparison changes the update rule and realized optimization cost together.

## Endpoints and predictions

Primary endpoint: environment-paired composition accuracy change from no
feedback to 16 verified feedback events for `online_lora`, with a 95% paired
t interval and a 10,000-resample environment bootstrap (RNG seed 821).

Secondary endpoints are the corresponding `context` and `error_only_lora`
changes, direct accuracy, the full accuracy curves, online-versus-context
accuracy at 16 events, and the gap between adapter-only and adapter-plus-context
evaluation. Report transitions and every failed or inconclusive run. A
positive adapter change demonstrates only this specified update rule on this
task family; a null result does not refute other feedback mechanisms.

## Accounting and audit requirements

Store every pre-update adaptation prediction, revealed target, whether an
update was applied, update loss, checkpoint, raw constrained and unrestricted
predictions, prompt hashes, input tokens, padded batch tokens, batch timings,
and final adapter files with hashes. Distinguish model forward time from
optimization time. Do not compare this loop's cost with static inference unless
all calls and optimization tokens are reported.

The analyzer must verify the complete grid, disjoint train/test queries,
target/prompt identity, source and model hashes, update gating, adapter reset
metadata, and environment-level aggregation. A target appearing in a
pre-update prompt is a fatal audit failure. Formal collection must stop on any
failed prerequisite rather than silently proceeding.

Per environment, the complete held-out grid contains 1,008 query records in
63 batches. Formal totals are 24,192 evaluation rows, 1,512 evaluation batches,
1,152 pre-update predictions, and 1,152 corresponding feedback events.
The journal writes and flushes each prediction before applying an update,
then writes its feedback/update result. Partial failed runs remain on disk.
These repeated checkpoints are not independent statistical replicates.

## Reproduction and stopping

Run unit tests and development seed 13 before formal inference. Formal
seeds are fixed once and are not extended after outcomes are viewed. The
reproduction command uses the pinned offline model and the archived source
revision. A later change to the optimizer, prompt, feedback stream, or model
requires a new prospective protocol and fresh environments.

From the repository root, the intended commands are:

```bash
python -m pytest -q experiments/study_e/test_online.py experiments/study_b/test_tasks.py experiments/study_b/test_adapter.py
# Check out the frozen revision and use the supplied frozen JSON.
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) python -m experiments.study_e.run \
  --development --output runs/study-e-development
HF_HUB_OFFLINE=1 python -m experiments.study_e.analyze --development \
  --input runs/study-e-development
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) python -m experiments.study_e.run \
  --development-audit runs/study-e-development/audit.json \
  --output runs/study-e-formal
HF_HUB_OFFLINE=1 python -m experiments.study_e.analyze --input runs/study-e-formal
```

The formal command verifies a successful development audit, all artifact hashes,
matching source/package/model/configuration identities, and the frozen data
before collecting formal predictions. Development statistics are exploratory;
one environment supplies no interval estimate. No cross-model external review
CLI is invoked in this authorized autonomous run.

## Success criteria

The study is complete only if the frozen runner produces a complete raw grid,
the analyzer independently reproduces every reported mean and interval, and
the manuscript clearly separates external context memory, parameter updates,
and the absence of persistent cross-task learning. No success criterion grants
the hypothesis that feedback alone explains intelligence.
