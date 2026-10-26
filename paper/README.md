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
| [v11](versions/v11-neural-search-priors.pdf) | Adds 32 new environments with neural program priors learned from accepted solutions: the small trained prior reduces composition success relative to uniform search; counts help compositions but lose that advantage on the balanced mixture. Includes replay audits, intermediate checkpoints, costs, and a qualified DreamCoder capacity audit. |
| [v12](versions/v12-experience-fit-and-transfer.pdf) | Adds explicitly post-hoc diagnostics of the frozen priors: near-minimal experience loss coexists with worse transfer-program likelihood and search success. Reports empirical prefix bounds, every prior, and the limits of causal and semantic interpretation. |
| [v13](versions/v13-local-context-and-transfer.pdf) | Adds a prospective comparison on 32 fresh environments: the same 1,172-parameter GRU with supplied local context solves 79.30% of compositions versus 24.22% with full prefixes. Shifted success is only 11.13%, and the balanced total remains below uniform search. Includes both frozen controls, every checkpoint, and distinct physical training/ranking costs. |
| [v14](versions/v14-continued-feedback-and-transfer.pdf) | Adds continued weight updates across two task-family blocks on 32 fresh environments: balanced probe success rises from 43.75% frozen to 51.56%, with a primary gain of +7.81 points (95% CI [4.74, 10.88]). Composition success reaches 92.38%, while shifted success remains 10.74%. Includes initial-corpus replay, both stream orders, every probe checkpoint, and lifetime-resource accounting. |

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

Study I trains unconditional GRU program priors from accepted experience
solutions supplied by symbolic search. The 1,172-parameter trained prior
solves 22.85% of new compositions versus 50.00% for uniform search: a primary
difference of -27.15 points, t95 [-32.74, -21.55]. The same small network frozen
at initialization scores 50.00%; the 13,892-parameter trained prior scores
17.58%. These results concern the fixed architecture and update schedule,
not a general capacity limit. A count prior solves 66.02% of compositions but
33.59% of shifted tasks, totaling 510/1,024 versus uniform's 512/1,024 on the
balanced mixture. Training, ranking, and supplied search remain explicit parts
of the system; the cause of neural transfer regression is not isolated.

The post-hoc diagnostics in v12 show that the small prior's full-experience
loss falls to 0.662957 nats/token from its initial 1.393215, with an unrestricted
empirical prefix bound of 0.635611. Its composition generating-program loss
is 2.596447 versus uniform's 1.386294. Thus improved in-sample fitting coexists
with worse transfer likelihood and original search success. Every checkpoint
is retained; a generating sequence is only one possible solution, and these
descriptions do not identify the cause of failure.

Study J in v13 restricts the same GRU to the last two primitive tokens, with
identical initial tensors, accepted experience, targets, and update counts.
The primary local-minus-full composition difference is +55.08 points,
t95 [45.60, 64.56], bootstrap95 [45.90, 63.87]. Local-trained composition
success is 79.30% versus uniform's 50.00%, descriptively, but its shifted
success is 11.13% and its balanced total is 463/1,024 versus uniform's 512.
The restriction changes context, recurrent state, gradients, and physical
work together; it does not isolate the cause of Study I's failure or establish
general intelligence, overall superiority, or equal computation.

Study K in v14 continues actual neural weight updates between stream tasks,
then evaluates all fixed checkpoints on untouched probes. Its balanced
online-minus-frozen difference is +7.81 points, t95 [4.74, 10.88],
bootstrap95 [4.98, 10.84]. Initial-corpus replay scores 43.16% and uniform
search 50.20%, compared with online's 51.56%. The gain is concentrated in
compositions (92.38%); shifted success remains 10.74%. New targets come only
from accepted solutions: failed searches cause extra replay without a new
failure-specific target. Thus this is a short continued-learning result,
not demonstrated shifted-task adaptation or feedback sufficiency. Both orders
are dependent conditions, and update counts do not match lifetime resources.

From the repository root, compile a version twice to resolve citations:

```bash
mkdir -p paper/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build \
  paper/versions/v14-continued-feedback-and-transfer.tex
```

Compiled files in `paper/versions/` are inspected snapshots. `paper/build/`
contains disposable intermediates. Figures are generated from archived data,
and historical drafts remain available even when later evidence supersedes them.
