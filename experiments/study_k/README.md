# Study K: continued local-prior learning

The [prospective protocol](../../notes/protocol-v11-continual-local-feedback.md)
tests continued accepted-solution replay through both orders of a distribution
change. Frozen and extra-replay controls separate the initial learner from
further optimization, while untouched probes measure later transfer.

The initial implementation assigns disjoint, length-balanced stream/probe
roles and computes the existing local GRU's 21 possible prediction contexts.
The derived table has 84 probabilities and requires 21 model calls consuming
57 recurrent positions. It replaces repeated identical calls during ranking;
the model and search algorithm are reused directly, without a new class,
adapter, or compatibility layer.

In the pinned PyTorch 2.8.0+cu126 environment:

```bash
python -m unittest experiments.study_k.test_stream -v
```

Synthetic checks cover task-role disjointness and ordering, physical forward
calls, exact agreement with direct predictions, independent NumPy GRU
probabilities and finite ordering, and preservation of an earlier table after
updates. They do not generate official environments.

The collector preserves every search, outcome, update, probability table,
candidate order, and checkpoint. Both continual branches clone the common
model and complete Adam state independently for each stream order. Online
adds only visibly accepted nonempty programs; replay always uses the initial
corpus. Frozen/uniform stream outcomes are shared explicitly between orders.
All stream updates finish before any untouched probe is searched.

```bash
python -m unittest experiments.study_k.test_collection -v
```

Synthetic collection checks cover cloned model/optimizer ownership, retained
Adam steps, real forward-call costs, reuse of recorded rankings, false accepted
solutions, hidden/probe isolation, empty acceptance, replay after exhaustion,
complete-grid/runtime gates, and preservation of an original failure. These
are artificial fixtures, not Study K development or formal observations.

The auditor independently regenerates tasks and initial tensors, reconstructs
each corpus from visible search outcomes, replays every model/Adam update,
and checks the archived local tables with NumPy GRU equations. Independent
full-function search checks preserve primitive costs and exact outcomes.
Every phase, physical input count, ordering artifact, and file must match.

```bash
python -m unittest discover -s experiments/study_k -p 'test_*.py' -v
python -m experiments.study_k.run --development \
  --source-revision SOURCE_COMMIT --output runs/study-k-development
python -m experiments.study_k.analyze --input runs/study-k-development \
  --output runs/study-k-development-audit
python -m experiments.study_k.run --source-revision SOURCE_COMMIT \
  --output runs/study-k-formal \
  --development-audit runs/study-k-development-audit/audit.json
python -m experiments.study_k.analyze --input runs/study-k-formal \
  --output runs/study-k-formal-audit
```

The primary averages final online-minus-frozen held-out success over both
orders/families within each environment. Reused controls and checkpoint-zero
scores are explicitly referenced, not independent observations. Both-order
tables count evaluation conditions and separately identify unique tasks.
Lower-budget stream scores describe first acceptances along the original
65,536-budget learning path, not agents retrained under those lower budgets.

Standalone cost serves final probes only: replay's stream searches and
intermediate orders are diagnostic, while online needs its stream outcomes.
All actual research work remains in the complete collection totals. Replay
phase durations include per-step journal writes and flushes; explicit archive,
clone, and load phases are excluded from standalone sums. These are measured
instrumented phases, not pure training-compute time or independent timing trials,
and they do not establish equal lifetime compute.
Synthetic audit checks reject altered states, corpora, probabilities, scores,
cost phases, and incomplete grids, and check environment-level intervals and
the fixed bootstrap. Formal archives retain the exact matching development
audit certificate. Auditing refuses optimized Python, where assertions would
be disabled, and requires collection/replay durations to cover their recorded
work.

The task data is now frozen after reviewed implementation commit
`53e250d52e31aa9b1e5db4d9512510febc70f783`: 33 environments, 1,848 tasks,
and 24 initial experience attempts in every environment. No official search
or learning outcome preceded this data freeze. The 2,378,541-byte JSON has
SHA-256 `68f068206234e43eb66b444e02a93640cb4611f33298a9a1eaaedae40024675d`.
Task labels, split uniqueness, and the declared length counts passed checks.

Reproduce the unchanged generator into a new file:

```bash
python3 -s - <<'PY'
import json
from pathlib import Path
from experiments.study_h.tasks import make_environment

data = [make_environment(seed) for seed in [37, *range(13000, 13032)]]
with Path('/tmp/study-k-environments-regenerated.json').open('x') as stream:
    stream.write(json.dumps(data, indent=2) + '\n')
PY
```
