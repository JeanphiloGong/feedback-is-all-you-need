# Study L: failure-conditioned weight updates

The [prospective protocol](../../notes/protocol-v12-failure-conditioned-weights.md)
tests direct updates from rejected programs during a new task. A separate
copy of the learned 1,172-parameter prior adapts to each task's failures.
The primary compares rejection-loss updates with uniformizing updates;
frozen and uniform search provide additional references.

The loss penalizes total normalized probability mass assigned to the current
task's rejected complete strings; finite Adam updates need not reduce it
monotonically. All search arms avoid repeated strings.
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

The complete task grid is frozen after reviewed implementation `fd08446`:
development seed 41 and formal seeds 14000--14031, totaling 1,848 tasks with
24 initial experience tasks per environment. No search/learning outcome
preceded this freeze. The 2,378,541-byte data file has SHA-256
`bb900adf1c562b5229486556c9df2fc0af6a28be2d3e4a3ccac63456cdb4ce81`.

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

The development certificate is a controlled audit-tool attestation. Its gate
checks source/data/runtime, full required search/file/checkpoint coverage, and
step totals reconciled with final checkpoints. It is not a signature proving
who authored a certificate; execution and transfer retain the real development
archive. Journal byte checks read raw bytes so newline conversions cannot hide
changed physical storage behind text-mode normalization.

The independent auditor reconstructs initial tensors, every Adam update,
rejection losses, NumPy GRU probabilities, complete ranked orders, and partial
checks using full functions and arithmetic. It verifies exact journal order,
all files, raw byte counts, source/data/runtime gates, and timing consistency.
The formal development gate requires every intermediate checkpoint, round,
and rebuilt table implied by its final step and operation counters.

```bash
python -m unittest discover -s experiments/study_l -p 'test_*.py' -v
python -m experiments.study_l.run --development \
  --source-revision SOURCE_COMMIT --output runs/study-l-development
python -m experiments.study_l.analyze --input runs/study-l-development \
  --output runs/study-l-development-audit
python -m experiments.study_l.run --source-revision SOURCE_COMMIT \
  --output runs/study-l-formal \
  --development-audit runs/study-l-development-audit/audit.json
python -m experiments.study_l.analyze --input runs/study-l-formal \
  --output runs/study-l-formal-audit
```

Analysis follows a passing full audit. The single primary uses 32 environment
means, with equal composition/shifted weight and fixed bootstrap seed 977.
Other comparators, families, and lower operation budgets are descriptive.
Task-specific updates do not establish retained capability on a later goal;
new ranking, rejection records, supplied search, and additional computation
remain part of the measured system.
