# Study L: failure-conditioned weight updates

The [prospective protocol](../../notes/protocol-v12-failure-conditioned-weights.md)
tests direct updates from rejected programs during a new task. A separate
copy of the learned 1,172-parameter prior adapts to each task's failures.
The primary compares rejection-loss updates with uniformizing updates;
frozen and uniform search provide additional references.

The loss reduces total normalized probability mass assigned to the current
task's rejected complete strings. All search arms avoid repeated strings.
Updates therefore have to improve the ranking of unchecked programs to help.
An interrupted candidate continues after an update without repeating work.
Task weights reset between targets; this is not persistent cross-task learning.

In the pinned CPU float64 PyTorch 2.8.0+cu126 environment:

```bash
python -m unittest experiments.study_l.test_feedback experiments.study_l.test_search -v
```

Eleven synthetic tests check independent NumPy probabilities, finite-difference
gradients, actual weight changes, uniformizing independence, excluded-prefix
descendants, exact execution boundaries, and equivalence to static search.
After rejection exactly exhausts a budget, the new checker stops before
choosing an unexecuted candidate; this preserves the next updated ranking.
That bookkeeping difference does not change acceptance or execution counts.

No official Study L environment or outcome has been generated. Implementation
and audit must be frozen before the development and formal runs.

The collector resets both model copies and fresh feedback optimizers for every
transfer target and retains each candidate, partial check, update, checkpoint,
table, and rank order. Hidden outputs are evaluated only after the complete
visible result is saved. Initial-experience optimizer states are retained as
evidence but are not carried into the different transfer losses.

```bash
python -m unittest experiments.study_l.test_collection -v
```

Candidate/update/evaluation phases separately measure journal serialization,
writes, and flushes, with byte counts. Result records and table/order/checkpoint
archives have explicit I/O phases. Phase-ledger writes, active-status writes,
file opening/closing, directories, and run metadata remain collection overhead;
phase sums are instrumented measurements, not pure-compute benchmarks.
Update journals are stored per environment so the complete fixed grid does
not put its potentially hundreds of thousands of steps into one oversized file.
