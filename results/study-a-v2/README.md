# Study A: registered run

Source revision: `0b4f0fb`. Full source SHA-256 and hardware are in metadata.
The compressed JSONL contains all 19,200 checkpoint rows from 20 seeds per
family. It is raw output, including negative results. The two figures and
summary are derived by `experiments/study_a/analyze.py`.

Use the packages in `environment.txt`. From the repository root:

```bash
python -m pytest -q experiments/study_a/test_run.py
FEEDBACK_GIT_REVISION=$(git rev-parse HEAD) python -m experiments.study_a.run \
  --output runs/study-a-reproduction --progress
python -m experiments.study_a.analyze --input results/study-a-v2
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build \
  paper/versions/v2-replication.tex
```

The analysis checks all keys, frozen-score invariance, equal initial scores,
and delayed-flush equivalence before computing intervals. Statistics use
independent task seeds as units. Co-primary paired intervals use 97.5%
confidence; secondary intervals are descriptive 95% intervals.

Training inputs and labels are generated deterministically by the archived
runner. Reproduction timings and GPU rounding may vary. Outputs from the
exploratory 1000/1001 smoke run are excluded from these registered results.

The first full run is preserved under `../study-a-v1/`. This replication
uses disjoint SHA-256-derived streams and the untouched seed range 4000–4019.
