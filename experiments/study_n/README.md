# Study N: encoding and unqueried-transition learning

The [prospective protocol](../../notes/protocol-v14-encoding-and-unseen-transitions.md)
compares learning gains from identical physical transitions under canonical
and randomly relabeled state encodings. Half of the transition pairs never
enter learning. This is a controlled supervised online mechanism study;
Study M supplies the interactive goal-learning experiment.

The core implements paired worlds/queries, isomorphic graph relabeling,
ordered replay updates with persistent Adam, and the family-informed symbolic
reference. It reuses Study M's fixed predictor and action-tied BFS planner.
No official data or outcomes have been generated at this checkpoint.

The collector now records all four predictions before each shared physical
query, then updates encoded replay and persistent model/Adam states. All
learning finishes before any direct or goal evaluation. Frozen controls are
evaluated once, and graph/transition/probe records retain physical pairing.
Training, prediction, and query phases separate journal time from measured
work; pure research archives remain outside standalone learning costs.

Startup checks the frozen source/data manifest and the exact development
certificate file grid before formal collection. The independent auditor and
frozen world file are still required before an official run can start.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_n.test_core -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_n.test_collection -v
```
