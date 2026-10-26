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
updates. They do not generate official environments or test the collection,
continual-learning control, or complete audit, which remain to be implemented.
