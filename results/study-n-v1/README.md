# Study N: encoding and unqueried transition learning

The [prospective protocol](../../notes/protocol-v14-encoding-and-unseen-transitions.md)
and [implementation](../../experiments/study_n/README.md) compare retained
gradient learning under canonical and randomly relabeled coordinates, using
identical physical feedback, initial tensors, and replay indices. The fixed
64 test transitions never enter training. The primary compares the small
model's learning gains across encodings; it is not a small-versus-large test.

Source revision `df2ecb3310e575a278c93bdec753434e85b6e161` freezes all 17
execution/protocol files. Data commit `5687bf0` freezes the complete 33-world
grid before outcomes. The server verified every source/data digest and passed
all 25 synthetic checks in 28.431 seconds. Execution uses lens/aorus, AMD
Ryzen 9 9950X, Python 3.11.15, Torch 2.8.0+cu126, NumPy 2.3.4, SciPy 1.16.3,
CPU float64, one thread, and deterministic algorithms. The small networks use
the server CPU; no GPU is needed for these runs.

## Development evidence

Seed 47 completed once in 2.180 seconds, followed by 2.711 seconds of
independent reconstruction. The audit passed all 64 physical queries, 1,280
gradient steps, 16 model/Adam/replay checkpoints, 20 direct-test records, and
27,776 fixed-graph goal probes. Every one of the 36 transferred evidence files
matches the server digest (28,705,937 bytes).

The small canonical predictor's unqueried exact accuracy rises from 3.125%
to 79.6875%; relabeled accuracy changes from 4.6875% to 0%. This single world
is descriptive, with no confidence interval or confirmatory claim. No
hyperparameter, endpoint, data, or source change follows these observations.
Formal collection starts only after the matching development certificate and
all its raw file hashes are checked by the frozen collector.

## Formal evidence

All 32 registered seeds (16000--16031) completed once with unchanged settings
in 59.883 seconds; full independent replay passed in 73.516 seconds. The
archive retains 2,048 physical queries, 40,960 gradient steps, 512 complete
checkpoints, 640 direct-test records, and 888,832 goal probes. All 8,192
adaptive batches changed at least one parameter. All 810 transferred files
(932,832,984 bytes) match host SHA-256 digests; the largest is 28,799,364 bytes.

The small model's unqueried exact accuracy changes from 64/2,048 (3.125%) to
1,101/2,048 (53.760%) canonically, a gain of **50.635 percentage points**.
After relabeling it changes from 70/2,048 (3.418%) to 47/2,048 (2.295%), a
gain of **-1.123 points**. The registered difference in gains is therefore
**+51.758 points**, paired t95 **[48.403, 55.113]**, bootstrap95
**[48.584, 54.932]** (10,000 resamples, seed 991). All 32 paired differences
are positive; the registered positive-primary criterion is met. World seeds,
not the 2,048 individual test pairs, are the inferential units.

| Encoding | Arm | Initial unqueried (%) | Final unqueried (%) | Final observed fit (%) | Final goal success (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Canonical | Small online | 3.125 | 53.760 | 97.803 | 84.217 |
| Canonical | Large online | 3.174 | 47.656 | 90.381 | 75.680 |
| Canonical | Small frozen | 3.125 | 3.125 | Unmeasured | 20.561 |
| Canonical | Large frozen | 3.174 | 3.174 | Unmeasured | 20.309 |
| Canonical | Observed table | 0 | 0 | 100 | 87.806 |
| Relabeled | Small online | 3.418 | 2.295 | 48.535 | 29.517 |
| Relabeled | Large online | 2.881 | 2.295 | 39.258 | 25.016 |
| Relabeled | Small frozen | 3.418 | 3.418 | Unmeasured | 20.675 |
| Relabeled | Large frozen | 2.881 | 2.881 | Unmeasured | 20.186 |
| Relabeled | Observed table | 0 | 0 | 100 | 87.806 |

All comparisons in the table besides the registered difference in small-model
learning gains are descriptive. Frozen direct and goal scores are evaluated
once and reused across checkpoints. Table prediction abstains on unqueried
pairs; its observed fit is exact by construction. Goal denominators are
31,744 distinct start/goal pairs per checkpoint, dependent within each world.

The stronger-prior canonical signed-coordinate matcher reaches 58.64%,
99.41%, and 100.00% unqueried accuracy after 16, 32, and 64 observations,
with coverage equal to accuracy. At the final checkpoint it retains one rule
per output and averages 644.94 rule comparisons. The uniform consistent-
permutation completion expectation is 6.25% there. These are distinct priors,
not matched-compute neural controls. The supplied family makes the canonical
toy easy to identify, which limits any interpretation as exceptional intelligence.

The table's physical goal traces are identical across encodings and reach
87.81% success without predicting any unqueried transition. Supplied BFS can
use acquired edges and current-state feedback to achieve goals. All true-map
shortest distances are at most six, within the eight-action cap. Goal success,
predictive generalization, and real parameter updates are separate outcomes.

The positive primary establishes an encoding-dependent learning effect for
this fixed architecture and update budget under identical physical evidence.
Relabeling preserves information while changing neural geometry. Relabeled
observed fit is also poor, so these results do not separate optimization,
effective capacity, and prior mismatch, or rule out a small agent that learns
representations. Larger-model comparisons also change gradient-step count.
No feedback-sufficiency or human-intelligence conclusion follows.

## Resources

Each standalone learner receives 64 queries and 17,301,504 training matrix
MACs, with 512 small-model or 128 large-model steps. Actual shared work totals
2,214,592,512 training MACs, 9,175,040 pre-update forward MACs, 91,750,400
evaluation-graph MACs, and 5,911,932 environment actions including probes.
Probe phases account for 45.746 seconds of the collection time.

Per-world learner phase subtotals excluding known journal I/O average
158.71/47.84 ms for small/large canonical learners and 146.90/47.25 ms after
relabeling. Shared width initialization, physical queries, and encoded replay
insertions are charged once per standalone learner. Some instrumentation
remains, so these are not pure deployment benchmarks or equal wall-time
comparisons. Model tensors use 3,880/15,400 bytes and Adam tensors
7,776/30,816 bytes, with about 600 serialized replay bytes per branch; these
are not resident memory measurements. Archives and probes remain in actual
costs rather than learner subtotals.

## Reproduction and scope

Run the commands in the implementation README under the registered runtime,
using new output directories. Analysis performs full gradient and checkpoint
replay before writing a summary. The certificate's remote raw-directory path
records where development evidence was verified at collection time; portable
formal replay checks its identity, digest, complete manifest, and counts.

An independent local verifier imports no experiment implementation and loads
no model. It reconstructs raw predictions, every goal trace and fallback,
symbolic references, complete summary cells, paired inference, and all costs.
Development and formal checks passed; formal reconstruction took 334.073
seconds and checked all 804 raw file hashes before and after. Nineteen
adversarial controls rejected altered summaries, incomplete/duplicate cells,
modified raw evidence, optimized Python, and output overwrites. Certificates
state the limits: gradient/tensor replay and timing measurements belong to
the pinned server audit, not this local numerical reconstruction.

```bash
mkdir -p runs
python3 results/study-n-v1/check_summary.py \
  --raw results/study-n-v1/formal-v1 \
  --summary results/study-n-v1/formal-audit-v1/summary.json \
  --output runs/study-n-local-verification.json
```

The [learning-curve figure](encoding-and-contrasts.pdf) uses every fixed
checkpoint and all 32 primary differences. Regenerate it with
`python3 -s results/study-n-v1/plot_learning.py` in the local system plotting
environment; disabling user-site packages avoids the local NumPy/Matplotlib
binary mismatch. This plotting environment is separate from training.

This experiment supplies query/reset access, state encoding, stationary
deterministic dynamics, replay, graph construction, and BFS. It tests unseen
transition pairs in the same physical world, not unseen states or domains.
Actual weight changes are required evidence of learning execution, but do not
by themselves establish useful generalization. All official runs, including
any failure, remain part of the research record.
