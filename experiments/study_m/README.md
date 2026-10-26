# Study M: retained predictive weight learning

The [prospective protocol](../../notes/protocol-v13-retained-world-model.md)
tests whether a small neural transition predictor learns reusable behavior
from action outcomes. Weights, Adam moments, and observed transitions persist
across goal episodes. This is a fully observed finite-world experiment with a
supplied state encoding and planner, not a pretrained language model.

The core implements the two world families, 485- and 1,925-parameter predictors,
sampled gradient replay, and bounded shortest-path planning. Synthetic checks
verify both constructive capacity witnesses, independently calculated gradients,
parameter changes, continued Adam state, and planning boundaries:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_m.test_core -v
```

The collector retains action outcomes, every gradient-step record, every graph
used for planning, and the five registered model/Adam/replay checkpoints. Goal
episode resets retain the neural state. Probes start only after all learning
streams finish and use fixed graphs without updates. An uncapped evaluator
also records exact true-graph distances, including distances beyond eight.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_m.test_collection -v
```

Startup requires the full frozen world grid and registered runtime. Formal
collection additionally requires a passing development audit with matching
source/data/runtime and complete file, action, gradient, checkpoint, and probe
coverage. The certificate is a controlled attestation, not authentication.
Output directories must be new, and failures preserve partial evidence.

Training, graph, action, warmup, and evaluation journals record serialization,
write/flush time and bytes separately. Remaining phase/status/file-management
overhead stays in collection elapsed time. Replay/graph JSON bytes describe
serialized representations, not Python resident memory or inference RAM.

Independent audit regenerates worlds, reconstructs each action and gradient,
checks every model/Adam/replay checkpoint, and rebuilds fixed-graph probes.
PyTorch tensors and journal values/digests match exactly; an additional NumPy
forward check uses rtol 1e-11 and atol 1e-12. Tensor representations, scalar
types, canonical JSON bytes, complete timing fields, and all physical journal
rows are checked. Optimized Python is rejected because it disables assertions.
Raw file hashes are checked before and after reconstruction; this is not file
locking or protection against changes after verification.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_m.test_replay experiments.study_m.test_audit -v
```

No official world data or experiment outcome has been generated. Official
collection must use the registered source/data freeze and PyTorch 2.8.0+cu126
runtime; local synthetic checks under another runtime do not satisfy that
requirement.
