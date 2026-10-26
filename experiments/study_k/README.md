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
The complete independent auditor and inference remain to be implemented;
official data generation and collection have not begun.
