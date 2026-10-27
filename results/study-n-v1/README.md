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

## Reproduction and scope

Run the commands in the implementation README under the registered runtime,
using new output directories. Analysis performs full gradient and checkpoint
replay before writing a summary. The certificate's remote raw-directory path
records where development evidence was verified at collection time; portable
formal replay checks its identity, digest, complete manifest, and counts.

This experiment supplies query/reset access, state encoding, stationary
deterministic dynamics, replay, graph construction, and BFS. It tests unseen
transition pairs in the same physical world, not unseen states or domains.
Actual weight changes are required evidence of learning execution, but do not
by themselves establish useful generalization. All official runs, including
any failure, remain part of the research record.
