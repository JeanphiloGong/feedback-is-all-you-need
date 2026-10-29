# Study M: retained predictive learning from action outcomes

Full checkpoint replay and numerical reconstruction now run on `lens`;
weights are excluded from Git. See [server storage and execution instructions](../STORAGE.md).
Earlier local verification certificates remain historical evidence.

The [prospective protocol](../../notes/protocol-v13-retained-world-model.md)
and [implementation](../../experiments/study_m/README.md) freeze two predictor
widths, eight controls, all interaction budgets, and the sole paired AUC primary
before outcomes. Weights, Adam moments, and observed replay persist across goal
episodes. The learned model predicts next-state bits; a supplied planner uses
those predictions to pursue goals.

Source/data revision `13f1c71c991ed991507f8ccc66609d4e5b88ec04` was copied to
`lens` (`aorus`). All 16 execution files match Git bytes and host bytes.
The frozen checkout passed all 37 synthetic checks under PyTorch 2.8.0+cu126
in 12.882 seconds. Collection uses CPU float64 and one thread; the GPU is not
used by this small-network experiment.

## Development evidence

Development seed 43 completed once in 7.771 seconds; independent audit and
analysis passed in 10.058 seconds. The audit reconstructed 8,192 main actions,
20,800 gradient steps, 88 checkpoints, and 13,888 probe tasks. Structured-world
small/large immediate AUCs are 0.930192/0.915575; both finish at 248/248 goal
successes. This single development observation is descriptive, not the primary
inference or a reason to change settings.

All 111 transferred files (41,927,964 bytes), including 104 raw audit entries,
match the execution host. An independent local reconstruction imports no
experiment implementation and reproduces all 16 curves, 16 aggregates, 28
contrasts, and 16 standalone cost records. Its scope and result are recorded
in `check_summary.py` and `development-local-verification-v2.json`.
The earlier verification report is preserved; review subsequently added
complete-cell and primary-decision checks without changing experiment results.

## Formal evidence

The unchanged formal grid started only after the matching development
certificate passed. All 32 seeds (15000--15031), each with structured and
random worlds, completed in 236.915 seconds on the lens AMD Ryzen 9 9950X.
Independent audit and analysis passed in 322.529 seconds, reconstructing
262,144 main actions, 665,600 gradient steps, 2,816 model/Adam/replay
checkpoints, and 444,416 probe tasks. All 69,632 main adaptive batches changed
at least one parameter; all frozen model tensors stayed fixed.

The small network has 485 parameters, the large network 1,925. Both predict
the five bits of the actual next state given the current state and action.
Immediate feedback adds each observed transition to replay, computes binary
cross-entropy, backpropagates, and applies eight small-model or two large-model
Adam steps. Weights, optimizer moments, and replay persist across goal resets.
A supplied breadth-first planner uses the predicted graph to choose actions.
This is online learning of a transition model, not a pretrained language model.

Every checkpoint is evaluated on the same 248 start/held-out-goal pairs per
world, after all learning has finished, with no parameter updates during probes.
The 32 environments are the paired inferential units; the 7,936 tasks per
family/checkpoint are not independent replications. AUC is the normalized
trapezoidal area through checkpoints 0, 64, 128, 256, and 512.

| Arm | Structured AUC | Structured final / 7,936 | Random AUC | Random final / 7,936 |
| --- | ---: | ---: | ---: | ---: |
| Small immediate | 0.914299 | 7,936 (100.00%) | 0.675238 | 6,763 (85.22%) |
| Small delayed | 0.896311 | 7,936 (100.00%) | 0.684082 | 6,863 (86.48%) |
| Small frozen | 0.232233 | 1,843 (23.22%) | 0.259703 | 2,061 (25.97%) |
| Large immediate | 0.916772 | 7,936 (100.00%) | 0.715261 | 7,705 (97.09%) |
| Large delayed | 0.894783 | 7,936 (100.00%) | 0.716758 | 7,689 (96.89%) |
| Large frozen | 0.227319 | 1,804 (22.73%) | 0.248110 | 1,969 (24.81%) |
| Observed table | 0.881718 | 7,648 (96.37%) | 0.892413 | 7,696 (96.98%) |
| Uniform actions | 0.215726 | 1,712 (21.57%) | 0.224294 | 1,780 (22.43%) |

The sole registered primary, structured-world AUC for small immediate minus
large immediate, is **-0.2473 percentage points**, paired t95
**[-1.1277, 0.6332]** and percentile bootstrap95 **[-1.0813, 0.5694]**
(10,000 resamples, seed 983). Sixteen seed differences favor each width.
The criterion for small-model superiority is not met; these intervals also
do not establish equivalence. Other curves and comparisons are descriptive.

In structured worlds the small immediate learner rises from 23.22% to 100%
success across new goals in the same dynamics. Frozen weights remain at
23.22%, while the large immediate learner rises from 22.73% to 100%.
Random-world final success is lower for the small learner than the large
learner. Thus useful retained weight learning is demonstrated in this finite
domain, while a general advantage for the smaller model is not. Faster updates
also do not consistently dominate delayed updates across both world families.

The supplied five-bit state encoding, four-action interface, stationary
transition memory, complete graph enumeration, planner, dense feedback, and
goal resets are parts of the system. Held-out goals may occur as physical
states during learning; no new dynamics are introduced at evaluation. Frozen
probes still receive their current true state. This experiment does not
establish feedback sufficiency, unrestricted intelligence, or a universal
minimum model size.

## Resources and archive

The immediate widths match a dense training-matrix proxy of 270,336 MACs per
main action. Adam, nonlinearities, replay, graph inference, and planning are
outside that proxy. Across both families, standalone small/large immediate
costs are respectively 270,336/67,584 gradient steps, the same 9,135,194,112
training MACs, 1,882,718,208/7,530,872,832 graph-forward MACs, and
85.185/29.218 instrumented phase seconds. Each standalone arm charges necessary
warmup and initialization once; summing these arms would count shared phases
again. Matching the proxy does not match total compute or elapsed time.

Actual collection includes 2,463,567 environment actions (warmup, main, and
probe actions), 964,917,075 journal bytes, and 74,892,864 checkpoint bytes.
These archive sizes are not inference RAM or model capacity. The existing
server GPU is not used in this deterministic CPU experiment.

All 3,211 transferred files (1,336,399,264 bytes) match the host manifest;
all 3,205 raw-file hashes in the audit match locally. The independent local
checker imports no experiment code and reproduces all 512 seed curves,
16 aggregates, 28 descriptive contrast means, 512 standalone cost records,
and the primary intervals and decision. Full tensor replay remains the host
audit. `formal-local-verification-v2.json` is the earlier host-side execution
of that checker; `formal-local-reconstruction.json` records the local rerun.

The 291,204,229-byte phase journal is stored as `formal-phases.jsonl.gz` outside
the raw directory to keep each Git object below GitHub's file limit. Its
round-trip bytes and hashes are recorded in `formal-compression.json`.
All other raw files retain their original names and bytes. Restore the journal
before either independent summary reconstruction or full replay:

```bash
gzip -dc results/study-m-v1/formal-phases.jsonl.gz \
  > results/study-m-v1/formal-v1/phases.jsonl
python3 -s results/study-m-v1/check_summary.py \
  --raw results/study-m-v1/formal-v1 \
  --summary results/study-m-v1/formal-audit-v1/summary.json \
  --output /tmp/study-m-summary-recheck.json
python3 -s results/study-m-v1/plot_learning.py
```

For full replay, use the registered PyTorch runtime and a new output directory
with the command in the [experiment instructions](../../experiments/study_m/README.md).
The restored journal is ignored by Git, remains in the host manifest, and
must match its original SHA-256. No failed official run or intermediate
checkpoint was discarded.
