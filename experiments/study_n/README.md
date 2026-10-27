# Study N: encoding and unqueried transition learning

The [prospective protocol](../../notes/protocol-v14-encoding-and-unseen-transitions.md)
asks whether retained online gradients predict never-observed transitions and
how much the answer depends on supplied state coordinates. A physical world
and its random relabeling share the same 64 ordered queries, initial tensors,
and sampled observation indices. Half of all state/action pairs never enter
learning. The 485- and 1,925-parameter predictors retain weights, Adam moments,
and replay across the entire stream. This is controlled online identification
with supplied query/reset access; Study M is the action-selection experiment.

CPU float64, one thread, deterministic PyTorch **2.8.0+cu126** are required.
The small/large branches take 512/128 gradient steps, respectively, for equal
training matrix MACs. This does not equalize full computation or time. Every
pre-update prediction, actual observation, gradient, model/optimizer/replay
checkpoint, direct test, and fixed-graph goal probe is retained. Evaluation
begins after every learning stream finishes; probes cannot update weights.

Independent replay regenerates worlds, updates, optimizer moments, complete
checkpoints, direct predictions, planning traces, symbolic reference results,
and true distances. It uses separate arithmetic and an additional NumPy
forward check. Exact file/row grids, scalar and tensor representations, source
and data provenance, and file hashes before and after replay are checked.
Python optimization is refused because assertions are required. These checks
are evidence reconstruction, not file locking or authenticated attestations.

The primary is the small model's canonical-minus-relabeled difference in
learning gains between observations 0 and 64, paired across 32 world seeds.
Both absolute gains and final accuracies accompany t95 and fixed-seed paired
bootstrap intervals. Other curves, widths, planner outcomes, and baselines
are descriptive. A family-informed symbolic matcher exposes the ease of the
canonical task; a consistent-permutation completion gives an exact expected
accuracy, not a sampled algorithm. No general-intelligence claim is tested.

Synthetic checks use negative seeds, without generating official worlds:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/study_n -p 'test_*.py' -v
```

After reviewed source/runtime and complete data freezes, use new output
directories. Formal collection requires a matching development certificate
and verifies every development raw file before it starts:

```bash
python -m experiments.study_n.run --development --output runs/study-n-development
python -m experiments.study_n.analyze --input runs/study-n-development --output runs/study-n-development-audit
python -m experiments.study_n.run --development-audit runs/study-n-development-audit/audit.json --output runs/study-n-formal
python -m experiments.study_n.analyze --input runs/study-n-formal --output runs/study-n-formal-audit
```

Analysis always performs complete replay before writing a summary. Formal
archives embed the development certificate with its digest; later portable
audits check its frozen identity and full evidence manifest. Collection checks
the actual development bytes at their recorded path before starting; portable
formal replay does not require that original remote path to still exist.
No result is emitted for an incomplete grid. Failures keep their partial
outputs and require the protocol's explicit amendment policy before any rerun.

Costs distinguish actual shared collection from standalone online learning.
Each learner is charged its width's initialization, encoded replay insertion,
and all 64 physical queries once, plus its own prediction/update work. Known
journal I/O is subtracted from this phase subtotal; sampling digests and some
instrumentation remain, so the result is not a deployment-speed benchmark.
Checkpoint archives, evaluation graphs, and probes remain in actual totals.
Tensor and JSON byte counts are not resident memory measurements.
