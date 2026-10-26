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
| [v4](versions/v4-generation-and-feedback.pdf) | Adds full-generation evidence on fresh environments, format and token-budget failures, a memory bound, and full-text comparisons with prior work. |
| [v5](versions/v5-generation-budget.pdf) | Adds a fixed 128/512-token comparison on 24 fresh environments: substantial recovery from invalid shorter prefixes, with no additional feedback during generation. |
| [v6](versions/v6-online-feedback.pdf) | Adds repeated parameter updates after verified outcomes on 24 fresh environments: no demonstrated composition gain, with context dependence and all adapter states retained. |
| [v7](versions/v7-rule-memory.pdf) | Adds externally checked rule revision, all intermediate rule memories, explicit executor attribution, and comparisons with Voyager, Self-Debugging, and ExpeL. |
| [v8](versions/v8-feedback-history.pdf) | Adds 16 checked revisions with last-only or complete history on 48 fresh environments: no demonstrated direct-accuracy benefit, with exact-rule regressions and unequal costs reported. |
| [v9](versions/v9-selection-and-search.pdf) | Adds explicitly post-hoc candidate selection and exact uniform-proposal references, distinguishing discarded progress from absent exact candidates and attributing the supplied grammar and selector. |
| [v10](versions/v10-compression-and-transfer.pdf) | Adds DreamCoder as an abstraction-learning precedent and a 32-environment symbolic transfer study: learned fragments help new motif compositions but reduce success on shifted programs, with prior experience and library costs accounted for. |

All manuscripts are working drafts. The current results support conditional
adaptation, not the sufficiency of feedback for general intelligence.
Full generation reveals stronger oracle execution in 7B. Extending the fixed
generation cap raises clean composition accuracy from 7.94% to 70.57%; all 481
recovered responses had invalid shorter prefixes. The 0.5B failure
under Study C's prompt remains unresolved; Study F's different rule interface
yields mostly valid proposals but no exact recovery. A one-pass LoRA feedback loop shows no demonstrated
composition gain; more effective mechanisms and sustained cross-task learning
remain open. The Study F checked-minus-self composition contrast is +4.04
points with 95% CI [-2.78, 10.85], which does not demonstrate a positive effect.
Study G retains complete correction transcripts for 16 revisions. The primary
0.5B direct-accuracy contrast is 0.00 points, CI [-3.96, 3.96]. Higher final
exact-rule counts in 7B coexist with loss of three of four initially exact
rules. Full history uses over twice the input tokens; neither extra iterations
nor transcript retention ensures cumulative improvement under this mechanism.
Post-hoc selection recovers exact rules from 17/48 full-history 7B pools
instead of the final 9/48. The 0.5B pools contain none. Their selected direct
score remains below the exact expectation for uniform legal revisions with
the same initial proposal; the reference supplies the action grammar and
does not match checking cost. These analyses preserve the original endpoints.

Study H tests cross-task fragment reuse with a supplied symbolic searcher and
no neural model. Composition success is 85.16% with compressed memory, 2.34%
with whole-program memory, and 50.39% without memory. The primary difference
is +82.81 points, t95 [77.12, 88.51]. On shifted programs, compressed memory
scores 9.96% versus 50.00% without memory. The result supports conditional
representation reuse, with no claim of general or neural intelligence.

From the repository root, compile a version twice to resolve citations:

```bash
mkdir -p paper/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build \
  paper/versions/v10-compression-and-transfer.tex
```

Compiled files in `paper/versions/` are inspected snapshots. `paper/build/`
contains disposable intermediates. Figures are generated from archived data,
and historical drafts remain available even when later evidence supersedes them.
