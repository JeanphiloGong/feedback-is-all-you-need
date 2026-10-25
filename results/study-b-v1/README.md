# Study B: pretrained-model feedback and persistence

All four context suites and the 24-environment LoRA suite completed using
runner revision `8010c3d`. Metadata records exact model revisions, executed
environment hashes, source hashes, package versions, times, and peak memory.
Each size directory contains every prediction and the self-history or LoRA
training record, compressed without dropping failed answers.

Context suites contain 8,064 held-out predictions each. The LoRA suite contains
2,304. Models share 24 independent hidden-command languages. Primary statistics
use environment means; individual queries are not independent replicates.
The main response metric is constrained first-token accuracy. It is not
free-generation answer accuracy or a measure of general intelligence.

```bash
python -m pytest -q experiments/study_b/test_tasks.py experiments/study_b/test_adapter.py
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) \
  python -m experiments.study_b.run --model 0.5B --output runs/reproduction-0.5B
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) \
  python -m experiments.study_b.run --model 0.5B --lora --output runs/reproduction-lora
python -m experiments.study_b.analyze --input results/study-b-v1
```

Repeat the context command for 1.5B, 3B, and 7B. Download the pinned model
revisions before offline execution. `environment.txt` records the actual
environment. Optimizer defaults and adapter specifications are frozen in
`notes/protocol-v2-language-models.md`.

Per-row batch times and padded token counts repeat for all rows in the batch;
the analysis divides them by batch size before summing. Input-token counts
are per query. Adapter hashes and all 128 losses per environment appear in
`lora/training.jsonl.gz`; weights remain in the authorized lens workspace at
`/src/users/lens/feedback-is-all-you-need/runs/study-b-v1-lora/adapters/`.

The remote disk filled after all original runs completed, during the first
export attempt. The successful export used `/tmp/feedback-study-b-v1-export`.
No model run was restarted or selected based on its outcome.
