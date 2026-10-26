# Study H: learned fragments across tasks

The frozen contract is [`../../notes/protocol-v8-library-transfer.md`](../../notes/protocol-v8-library-transfer.md).
This finite symbolic study uses a supplied program enumerator, four primitive
permutations, and flat library compression. No neural model is involved.
The primary contrast compares compressed with whole-program memory on fresh
motif compositions, over 32 environments. Random primitive sequences measure
a distribution change. The renamed control preserves library entry lengths.

From the repository root, with Python 3.10, NumPy, and SciPy:

```bash
python3 -s -m unittest discover -s experiments/study_h -p 'test_*.py' -v
python3 -s -m experiments.study_h.run --development --output runs/h-dev-v1
python3 -s -m experiments.study_h.analyze --input runs/h-dev-v1 \
  --output runs/h-dev-audit-v1
python3 -s -m experiments.study_h.run --output runs/h-formal-v1 \
  --development-audit runs/h-dev-audit-v1/audit.json
python3 -s -m experiments.study_h.analyze --input runs/h-formal-v1 \
  --output runs/h-formal-audit-v1
```

Each output directory must be new. The formal runner requires a passing
seed-23 audit with identical source and data hashes. The analyst independently
replays all search counts and acceptance decisions from complete candidate
outputs and checks compression with exhaustive segmentation. Runtime failures
are preserved; completed searches are flushed before the final evaluator runs,
and evaluations are written to a separate journal. The audit requires matching
search/evaluation keys and the complete frozen seed/task grid. Formal seeds are
10000–10031, with 16 composition and 16 shifted tasks per environment per arm.

The shared experience corpus is solved with primitive-only search, then each
library is frozen. Eight visible examples guide search; eight hidden inputs
contribute to the complete-function success criterion. A false fit remains in
memory and is reported. No hidden label determines acceptance or memory entry.

Every library call pays for all expanded primitive operations. The 65,536-unit
cap excludes enumeration, comparisons, expansion overhead, induction, and
storage, which are reported separately. Standalone cost estimates charge
experience to the memory arms, and compression additionally to the renamed
control; the no-memory arm needs no prior experience. These are not equal
lifetime FLOPs. The encoded-corpus token metric is a compression diagnostic,
not additional runtime memory: only the library persists during evaluation.
Phase timings exclude imports, data generation, final auditing, and report
serialization; final evaluator costs are separately recorded. The run metadata
also records elapsed wall time including serialization and journaling.

To reproduce the fixed data without replacing the archive:

The archived data contains 33 environments and 1,848 tasks, including seed 23.
SHA-256: `cea3c10966a5b8e48225afaa2bd42722c6bb0cdfa82be6da5d9802d17118d432`.

```bash
python3 -s -m experiments.study_h.tasks --output /tmp/study-h-environments.json
cmp experiments/study_h/data/environments-v1.json /tmp/study-h-environments.json
```
