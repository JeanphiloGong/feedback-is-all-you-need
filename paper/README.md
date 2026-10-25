# Manuscript versions

The title is **Feedback Is All You Need**. The original NeurIPS paper is in
[`../references/attention-is-all-you-need.pdf`](../references/attention-is-all-you-need.pdf).
Its seven-section organization guides these manuscripts; no empirical claim
is borrowed from it.

| Version | Evidence and status |
| --- | --- |
| [v0](versions/v0-hypothesis.md) | Conceptual hypothesis, definitions, and prospective experiment plan. |
| [v1](versions/v1-controlled-study.pdf) | First controlled study; archived with a warning because a random-stream audit found dependencies. |
| [v2](versions/v2-replication.pdf) | Corrected study on untouched seeds; complete raw results and analysis. |
| [v3](versions/v3-feedback-and-capacity.pdf) | Adds four pretrained-model sizes, feedback controls, and persistent LoRA adaptation; evidence-reviewed and rendered. |

All manuscripts are working drafts. The current results support conditional
adaptation, not the sufficiency of feedback for general intelligence.
The negative language-model result applies to a first-token response interface;
full-generation and stronger feedback mechanisms remain research questions.

From the repository root, compile a version twice to resolve citations:

```bash
mkdir -p paper/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build \
  paper/versions/v3-feedback-and-capacity.tex
```

Compiled files in `paper/versions/` are inspected snapshots. `paper/build/`
contains disposable intermediates. Figures are generated from archived data,
and historical drafts remain available even when later evidence supersedes them.
