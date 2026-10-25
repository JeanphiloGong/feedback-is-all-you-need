# Study E: online outcomes, context memory, and LoRA updates

The frozen `84f2e52` runner completed all 24 environments (7000–7023) in
72.80 seconds after model loading. It retains 24,192 evaluation records,
1,152 ordered prediction/feedback pairs, and all 48 final adapter states.
The model is the pinned Qwen2.5-0.5B-Instruct checkpoint, scored through the
same constrained first-token interface as Study B.

| Condition | Direct accuracy | Composition accuracy |
| --- | ---: | ---: |
| No feedback | 14.06% | 14.32% |
| Context after 16 outcomes | 12.50% | 14.45% |
| One LoRA update per outcome | 13.28% | 13.67% |
| LoRA update only after an error | 10.68% | 11.98% |
| Per-outcome LoRA plus context | 5.47% | 13.54% |
| Error-only LoRA plus context | 5.99% | 12.89% |

The primary composition change for per-outcome LoRA is **−0.65 percentage
points**, with 95% environment-paired t interval **[−4.04, 2.74]** and
10,000-resample bootstrap interval **[−3.78, 2.47]**. This schedule does not
demonstrate a retained compositional gain. It changes 99 previously wrong
composition answers to correct and 104 previously correct answers to wrong.
Error-only updating changes 86 to correct and 104 to wrong, for a descriptive
net change of −2.34 points (95% interval [−5.53, 0.85]).

Adding the correct transcript after per-outcome LoRA reduces direct accuracy
from 13.28% to 5.47%: a descriptive paired difference of −7.81 points
([−12.16, −3.46]). The corresponding error-only difference is −4.69 points
([−8.67, −0.70]). Parameter updates and context do not combine reliably under
this interface and schedule. These are secondary findings, not separately
confirmed hypotheses.

Both adapter conditions start from identical adapter tensors for each
environment. All zero-feedback predictions match the base, and the full base
parameter hash is unchanged at completion. Per-outcome LoRA applies 384 total
updates; the error-only condition applies 337. They process 46,464 and 40,777
optimization input tokens in 7.82 and 6.82 measured seconds, respectively.
All 1,512 evaluation batches and 1,152 adaptation-prediction calls are counted
separately in `summary.json`; whole-suite peak allocation is 1,305,673,728 bytes.
Costs are reported, not matched across conditions.

The full audit checks the pinned model and sources, frozen data and complete
query grid, target-independent prompts, unrestricted token decoding, ordered
prediction/feedback events, update gating, adapter files, reset evidence,
batch padding, and costs. Every transferred file matches `sha256.json`.
Independent local aggregation reproduces all 42 means and the primary interval.
The original developer-generated PEFT model-card templates are preserved with
the adapter bytes for exact hash verification; this README describes the study.

This is passive one-pass learning from only 16 outcomes, with no rehearsal,
active exploration, cross-environment retention, or full answer generation.
An equivalent ordered offline replay reproduces the per-outcome updates.
The null result does not establish that other feedback mechanisms fail, or
that the model cannot improve under a different interface or optimization dose.

With the pinned offline model cache, from the repository root:

```bash
HF_HUB_OFFLINE=1 python -m experiments.study_e.analyze --input results/study-e-v1
```

The analysis rewrites summaries and figures; regenerate in a copy to preserve
the archived hashes. Exact development/formal commands are in
[`../../notes/protocol-v5-online-feedback.md`](../../notes/protocol-v5-online-feedback.md).
Development evidence is retained separately under
[`../development/study-e-v1`](../development/study-e-v1).
