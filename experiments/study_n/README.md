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

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_n.test_core -v
```
