# Study A: registered run

**Superseded for inference:** an implementation audit found shared corruption
streams across seeds and some initialization/task stream collisions. Retained
for the research record. Use study-a-v2 for primary statistics; reproduce this
historical version from its recorded Git revision, not the amended runner.

Source revision: `b2eeda2`. Full source SHA-256 and hardware are in metadata.
The compressed JSONL contains all 19,200 checkpoint rows from 20 seeds per
family. It is raw output, including negative results. The two figures and
summary are derived by `experiments/study_a/analyze.py`.

Use the packages in `environment.txt`. From the repository root:

```bash
python -m pytest -q experiments/study_a/test_run.py
FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) python -m experiments.study_a.run \
  --output runs/study-a-reproduction --progress
python -m experiments.study_a.analyze --input results/study-a-v1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build \
  paper/versions/v1-controlled-study.tex
```

The analysis checks all keys, frozen-score invariance, equal initial scores,
and delayed-flush equivalence before computing intervals. Statistics use
independent task seeds as units. Co-primary paired intervals use 97.5%
confidence; secondary intervals are descriptive 95% intervals.

Training inputs and labels are generated deterministically by the archived
runner. Reproduction timings and GPU rounding may vary. Outputs from the
exploratory 1000/1001 smoke run are excluded from these registered results.
