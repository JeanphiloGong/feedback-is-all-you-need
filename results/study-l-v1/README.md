# Study L: task-specific weight updates from rejected programs

The [prospective protocol](../../notes/protocol-v12-failure-conditioned-weights.md)
compares direct failure-loss updates with uniformizing updates, frozen learned
weights, and uniform search. The model remains a supplied 1,172-parameter
local program prior. Each adaptive task starts from the same learned model
with a fresh optimizer, receives only visible rejections, and updates between
execution blocks. This tests within-task adaptation, not retained learning
across different goals.

Source/data revision: `69cc5bc503d030a7e5ed9912c71bd366c0f0ef28`, following
reviewed implementation freeze `fd08446`. All 20 source/data files match Git
and the `lens` execution copy. The frozen checkout passed all 27 synthetic
checks in 5.725 seconds under PyTorch 2.8.0+cu126, CPU float64, one thread.
The complete fixed grid uses development seed 41 and formal seeds 14000--14031.
No source, seed, budget, objective, or update setting changed after development.

## Development

The development run completed once in 35.070 seconds on the lens AMD Ryzen
9 9950X. Independent audit/analysis passed in 23.279 seconds, reproducing
152 searches, 12,224 gradient steps, and 882,800 candidate/check-boundary
events. The run includes initial-experience learning and both adaptive arms;
those step totals are not exclusively rejection-loss updates.

| Arm | Composition exact / 16 | Shifted exact / 16 | Balanced exact / 32 |
| --- | ---: | ---: | ---: |
| Uniform | 8 | 8 | 16 |
| Frozen learned weights | 10 | 0 | 10 |
| Uniformizing updates | 8 | 8 | 16 |
| Rejection-loss updates | 9 | 6 | 15 |

The primary contrast is therefore -3.125 percentage points in this single
development environment. It is descriptive, with no confirmatory interval.
Improvement over frozen weights alone does not establish a benefit from
informative failure feedback over uniformization or uniform search. These
observations did not change the formal experiment.

All 2,577 transferred raw/audit files, totaling 187,163,676 bytes, match their
execution-host originals. The auditor's 2,575 raw-file hashes also match the
local archive. Every intermediate state, rank order, rejected candidate,
partial execution, and failed attempt remains available. Console logs and the
source-transfer record are saved separately from the host file manifest.

The unchanged 32-environment formal grid completed with this exact passing
certificate. Independent audit/analysis passed in 688.739 seconds after
985.204 seconds of collection. The complete host archive contains 73,076
raw/audit files and 5,442,329,227 bytes; the audit covers 73,074 raw files.
Transfer verification and independent summary reconstruction are recorded
alongside the original host evidence. No official failed attempt occurred.

## Formal results

All 32 registered environments (14000--14031) completed with unchanged frozen
settings. The inferential unit is an environment, averaging its 16 composition
and 16 shifted tasks equally. The 4,096 transfer searches share 1,024 tasks;
neither four arms nor 32 tasks per environment are independent replications.

| Arm | Composition / 512 | Shifted / 512 | Balanced / 1,024 |
| --- | ---: | ---: | ---: |
| Uniform | 256 (50.00%) | 256 (50.00%) | 512 (50.00%) |
| Frozen learned weights | 378 (73.83%) | 58 (11.33%) | 436 (42.58%) |
| Uniformizing updates | 262 (51.17%) | 256 (50.00%) | 518 (50.59%) |
| Rejection-loss updates | 258 (50.39%) | 235 (45.90%) | 493 (48.14%) |

The sole primary, final balanced rejection minus uniformizing, is **-2.4414
percentage points**, paired t95 **[-4.4998, -0.3830]** and paired percentile
bootstrap95 **[-4.4946, -0.6836]** (10,000 resamples, seed 977). Both intervals
are below zero. Rejection wins in three environments, ties in 17, and loses
in 12. The registered positive criterion is not met; there is evidence of a
deficit for this intervention relative to the uniformizing control.

Rejection exceeds frozen weights by 5.5664 points but trails uniform search
by 1.8555 points, descriptively. The shifted recovery over frozen is 34.5703
points, accompanied by a 23.4375-point composition loss. Uniformizing recovers
at least as much balanced success while its loss ignores rejected identities.
This limits attribution of recovery over freezing to informative failure
content. It neither identifies the cause of the rejection deficit nor rules
out other online learning mechanisms. There are no false acceptances in the
recorded grid; this does not guarantee the visible checker on arbitrary tasks.

The parameter updates are real. Rejection executes 178,848 Adam steps in
11,178 batches; uniformizing executes 151,280 steps in 9,455 batches. Each
batch changes all 1,172 parameter values, with positive recorded L2 change.
There are no skipped adaptive batches. Both arms start each new goal from
the common learned weights and a fresh optimizer. These data therefore do
not demonstrate retained learning across transfer tasks.

| Primitive budget | Uniform | Frozen | Uniformizing | Rejection |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 0.00% | 14.16% | 14.16% | 14.16% |
| 16,384 | 17.77% | 25.29% | 22.95% | 17.58% |
| 65,536 | 50.00% | 42.58% | 50.59% | 48.14% |

These balanced lower-budget endpoints follow the observed full-budget path,
without retraining, selecting checkpoints, or replacing the primary endpoint.
The first adaptive update occurs only after 4,096 operations if unresolved,
so the three learned arms agree there.

[Success budgets and paired final differences](success-and-contrasts.pdf)

The figure shows means at the three registered budgets and every final paired
environment difference. Its t95 interval concerns only the registered primary;
curves and family comparisons are descriptive.

## Costs and interpretation

| Standalone arm | Primitive executions | Gradient steps | Physical training positions | Instrumented phase seconds | Included journal seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 43,948,916 | 0 | 0 | 21.907 | 13.913 |
| Frozen | 48,940,473 | 12,288 | 2,304,000 | 27.587 | 12.420 |
| Uniformizing | 45,052,778 | 163,568 | 11,834,640 | 215.281 | 15.377 |
| Rejection | 51,995,043 | 191,136 | 13,571,424 | 713.812 | 17.573 |

Learned arms each include necessary initial experience/learning once per
environment. The 12,288 initial steps are charged to each standalone learned
arm, but occur only once in actual collection. Padding counts as physical
training work. Rejection also processes 772,391,024 rejected-program exposures
and 5,073,457,216 corresponding token exposures across updates. These are
repeated processing counts, not distinct programs or extra neural positions.
Equal primitive caps and update opportunities do not match lifetime work.

Phase sums come from shared instrumented collection, not independent timing
trials or pure-compute measurements. They include separately measured journal
serialization/write/flush time; explicit checkpoint, result, table, and order
archives plus hidden scoring remain in actual collection. Phase-ledger, status,
and file-management overhead stays in total collection time. Standalone totals
must not be summed as actual elapsed time because initial phases are reused.

Actual collection takes 985.204 seconds on the lens AMD Ryzen 9 9950X, using
one CPU thread and float64; the server's GPU is not used by Study L. It includes
4,864 searches (768 initial and 4,096 transfer), 183,561,514 primitive executions,
257,088 hidden-evaluator executions, 342,416 gradient steps, and 25,936,224
candidate/boundary events. All 25,529 checkpoints occupy 863,577,169 bytes;
this is an accumulated research archive, not live inference RAM or a neural
capacity measurement. Supplied primitive semantics, local context, search,
verification, and external rejection/ranking storage remain part of the system.

The loss penalizes probability mass assigned to already known failed strings;
finite Adam updates need not reduce that mass monotonically.
It does not identify correct untested strings or learn an explanation of a
mismatch. Shared token factors can also change probabilities of untested
solutions. The present evidence separates real parameter changes from useful
adaptation; it does not establish the optimal feedback objective, a small-model
capacity limit, or a general refutation of feedback-based intelligence.


## Reproduce the summary and figure

From the repository root, after installing the recorded NumPy/SciPy and
Matplotlib dependencies:

```bash
python3 -s results/study-l-v1/check_summary.py \
  results/study-l-v1/formal-v1 \
  results/study-l-v1/formal-audit-v1/summary.json \
  /tmp/study-l-summary-recheck.json
python3 -s results/study-l-v1/plot_success.py
```

The first command checks the archived outcomes/statistics/cost aggregation
without importing experiment code; it does not replace the full tensor and
search replay audit. The [experiment instructions](../../experiments/study_l/README.md)
record the pinned runtime and complete audit commands. Historical source,
development, negative results, and all intermediate records remain intact.
