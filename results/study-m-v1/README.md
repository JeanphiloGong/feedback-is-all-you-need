# Study M: retained predictive learning from action outcomes

The [prospective protocol](../../notes/protocol-v13-retained-world-model.md)
and [implementation](../../experiments/study_m/README.md) freeze two predictor
widths, eight controls, all interaction budgets, and the sole paired AUC primary
before outcomes. Weights, Adam moments, and observed replay persist across goal
episodes. The learned model predicts next-state bits; a supplied planner uses
those predictions to pursue goals.

Source/data revision `13f1c71c991ed991507f8ccc66609d4e5b88ec04` was copied to
`lens` (`aorus`). All 16 execution files match Git bytes and host bytes.
The frozen checkout passed all 37 synthetic checks under PyTorch 2.8.0+cu126
in 12.882 seconds. Collection uses CPU float64 and one thread; the GPU is not
used by this small-network experiment.

## Development evidence

Development seed 43 completed once in 7.771 seconds; independent audit and
analysis passed in 10.058 seconds. The audit reconstructed 8,192 main actions,
20,800 gradient steps, 88 checkpoints, and 13,888 probe tasks. Structured-world
small/large immediate AUCs are 0.930192/0.915575; both finish at 248/248 goal
successes. This single development observation is descriptive, not the primary
inference or a reason to change settings.

All 111 transferred files (41,927,964 bytes), including 104 raw audit entries,
match the execution host. An independent local reconstruction imports no
experiment implementation and reproduces all 16 curves, 16 aggregates, 28
contrasts, and 16 standalone cost records. Its scope and result are recorded
in `check_summary.py` and `development-local-verification-v2.json`.
The earlier verification report is preserved; review subsequently added
complete-cell and primary-decision checks without changing experiment results.

The unchanged formal grid was started only after the matching development
certificate passed. Its outcome and interpretation will be archived separately.
