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
