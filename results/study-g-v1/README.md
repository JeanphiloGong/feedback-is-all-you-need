# Study G: retaining the complete feedback history

Frozen source: `c4b4054`. The prospective contract is
[`../../notes/protocol-v7-feedback-history.md`](../../notes/protocol-v7-feedback-history.md).
Each pinned Qwen2.5 model receives all 16 examples for 48 fresh languages
(seeds 9000–9047), generates a shared initial rule, and performs exactly 16
checked revisions per branch. `last_only` retains the immediately previous
proposal and check; `full_history` retains every proposal and check. The
known additive family, strict parser, review instruction, 512-token cap, and
supplied executor are identical. No weights change, no new task information
arrives, and no gate preserves a better earlier rule. The final proposal is
scored even when an earlier one was correct.

Each model produces 1,584 proposals and saved rule files, with 76,032 held-out
evaluations. Both first-revision branches have identical ordered seed batches,
actual padded input IDs, attention masks, and output token IDs. All proposals
terminate at EOS before the cap. All intermediate outputs and failures remain
in the archive; no held-out query or target enters a prompt.

| Model | Proposal | Direct % | Composition % | Valid / 48 | Exact / 48 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | Initial | 13.54 | 13.02 | 44 | 0 |
| 0.5B | Last only, revision 16 | 10.42 | 7.94 | 32 | 0 |
| 0.5B | Full history, revision 16 | 10.42 | 7.23 | 30 | 0 |
| 7B | Initial | 53.13 | 39.52 | 45 | 4 |
| 7B | Last only, revision 16 | 47.40 | 34.51 | 44 | 5 |
| 7B | Full history, revision 16 | 47.92 | 36.65 | 41 | 9 |

The primary 0.5B final direct-accuracy difference, full history minus last
only, is **0.00 percentage points**, paired t 95% CI **[-3.96, 3.96]**,
percentile bootstrap **[-3.65, 3.65]**. The zero mean combines 20 wrong-to-correct
and 20 correct-to-wrong answer changes between branches; individual outcomes
are not identical. Both branches finish 3.13 points below the initial direct
score (last-only CI [-8.01, 1.76]; full-history CI [-8.23, 1.98]). There is no
exact 0.5B proposal at any stage; each final 0/48 rate has Wilson 95% CI
[0.00%, 7.41%]. These results do not establish a positive history effect or a
universal inability of small models to learn.

Secondary 0.5B composition changes from the initial proposal are -5.08 points
for last only (CI [-9.22, -0.94]) and -5.79 for full history
(CI [-10.04, -1.55]). These descriptive intervals exclude zero but were not
the primary endpoint and have no multiplicity adjustment. The between-branch
composition contrast is -0.72 points, CI [-2.84, 1.41]. Direct accuracy was
chosen as primary before collection because adding four modulo eight to every
offset can preserve all two-command outputs despite every offset being wrong.

The secondary 7B full-minus-last direct contrast is +0.52 points,
CI [-11.69, 12.73]; composition is +2.15, CI [-9.55, 13.85]. Direct changes
from the initial proposal are -5.73 points for last only (CI [-13.54, 2.09])
and -5.21 for full history (CI [-16.31, 5.89]). Exact final counts are 5/48
(10.42%, Wilson CI [4.53%, 22.17%]) and 9/48
(18.75%, CI [10.19%, 31.94%]). These marginal rates do not establish a
between-branch effect. Relative to the four initially exact rules, last only
gains two and loses one; full history gains eight and loses three. Higher
final exact recovery must not be mistaken for reliable preservation of prior
success. Direct answers gained/lost from the initial proposal are 48/92 for
last only and 88/128 for full history.

## Repetition and component changes

Each branch has 768 revision transitions nested within 48 environments.
The table stratifies by the **current** proposal. Repeated exact rules can be
desirable; reducing repetition alone does not demonstrate learning. Text and
valid-mapping repetition counts happen to match in these runs. Invalid
proposals have no valid-mapping denominator, and their per-record mapping
metrics are null.

| Model / branch | Current state | Transitions | Same text as previous | Text seen at any earlier stage |
| --- | --- | ---: | ---: | ---: |
| 0.5B last | Invalid | 246 | 231 | 231 |
| 0.5B last | Valid, wrong | 522 | 441 | 483 |
| 0.5B last | Exact | 0 | 0 | 0 |
| 0.5B full | Invalid | 283 | 254 | 262 |
| 0.5B full | Valid, wrong | 485 | 363 | 436 |
| 0.5B full | Exact | 0 | 0 | 0 |
| 7B last | Invalid | 61 | 46 | 58 |
| 7B last | Valid, wrong | 603 | 74 | 525 |
| 7B last | Exact | 104 | 62 | 98 |
| 7B full | Invalid | 84 | 26 | 66 |
| 7B full | Valid, wrong | 525 | 244 | 425 |
| 7B full | Exact | 159 | 123 | 146 |

| Model / branch | Valid-to-valid pairs | Correct offsets gained / lost | First failed command repaired / valid incorrect-to-valid pairs |
| --- | ---: | ---: | ---: |
| 0.5B last | 520 | 11 / 9 | 4 / 520 |
| 0.5B full | 474 | 37 / 32 | 18 / 474 |
| 7B last | 693 | 212 / 216 | 119 / 590 |
| 7B full | 645 | 66 / 56 | 48 / 509 |

Component totals exclude invalid predecessors or successors, so they cannot
explain the complete accuracy change by themselves. The repair denominator
also excludes previously exact rules. Across all transitions from exact 7B
rules, including invalid successors, last only preserves exactness in 62/103
and full history in 123/154. Full history has 18 invalid successors among
those 154 transitions. These transition fractions differ from retention of the four
initially exact environments at the final stage. None of these nested
transition totals is treated as a count of independent experiments.

## Resource costs

| Measured actual suite | 0.5B | 7B |
| --- | ---: | ---: |
| Continuations / generation batches | 1,584 / 99 | 1,584 / 99 |
| Input tokens | 1,244,712 | 1,168,726 |
| Output tokens through EOS | 61,776 | 44,253 |
| Padded input slots | 1,284,288 | 1,211,328 |
| Padded generation slots | 61,776 | 45,936 |
| Generation seconds | 41.294 | 227.323 |
| Diagnostic checker seconds | 0.01260 | 0.01432 |
| Runner seconds after loading | 42.085 | 228.098 |
| Peak allocated GPU memory, GiB | 2.380 | 19.485 |

Actual costs count the shared initial generation once. A hypothetical
standalone branch would include its own initial generation and requires 816
continuations in 51 batches. Summing those standalone costs would double-count
the initial generation; they are accounting projections from the actual run,
not separately timed reruns.

| Standalone accounting, initial included | 0.5B last | 0.5B full | 7B last | 7B full |
| --- | ---: | ---: | ---: | ---: |
| Input tokens | 367,542 | 894,642 | 360,150 | 826,048 |
| Generation seconds | 15.836 | 26.315 | 78.413 | 152.625 |

Full history uses 2.43 times as many input tokens in 0.5B and 2.29 times in
7B; accounted generation time is 1.66 and 1.95 times as large. Equal call caps
do not match compute. Costs exclude model loading and offline analysis, and
do not account for pretraining. Model-specific repetition penalties remain
1.1 and 1.05. File persistence and modular execution are supplied components,
not neural retention or cross-language skill accumulation.

## Audit and reproduction

Each model's full remote audit passes complete grids, pinned sources/models,
packages, development evidence, prompt/history reconstruction, raw decoding,
actual input tensors, first-revision pairing, rules, targets, and cost totals.
All 3,188 transferred manifest entries match. Independent local parsing and
scalar arithmetic reproduce 132 means, 12 contrasts, the primary bootstrap,
66 diagnostic rows and Wilson intervals, answer and exact-rule transitions,
repetition/component records, first-revision tensors, and cost accounting.
Both generated figures were inspected.

The local independent check initially used SciPy 1.8.0's Student-t quantile,
whose last digits differ from the recorded remote SciPy 1.16.3 value. This
caused a roughly 7.6e-11 bound discrepancy under a tight comparison. Solving
the Student-t CDF with the incomplete beta function and root finding reproduced
the recorded bounds at the original tolerance. Formal results were unchanged.

The source/data SHA-256 records and exact package versions are in each
`metadata.json`; every audited file hash is in `audit.json`. Complete seed-19
development evidence is in
[`../development/study-g-v1`](../development/study-g-v1).
Reproduce using
[`../../experiments/study_g/README.md`](../../experiments/study_g/README.md)
at frozen source `c4b4054`, with the pinned cached models and
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Exploratory selection after manuscript v8

This analysis was specified after viewing the complete results and committing
v8; its plan and implementation are frozen at `893d1c6` in
[`../../notes/study-g-selection-plan.md`](../../notes/study-g-selection-plan.md).
`selection-diagnostics.json` retains 1,152 checkpoint records, input/source
hashes, and all descriptive summaries. The selector uses only disclosed
training examples, ranks invalid rules below every valid rule, and keeps the
earliest maximum within each available prefix. It does not change the original
scores or infer what a model would generate after a changed acceptance decision.

| Model / pool | Final direct % | Selected direct % | Final composition % | Selected composition % | Exact final / selected, out of 48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B last only | 10.42 | 17.71 | 7.94 | 14.84 | 0 / 0 |
| 0.5B full history | 10.42 | 19.27 | 7.23 | 15.30 | 0 / 0 |
| 7B last only | 47.40 | 63.54 | 34.51 | 49.93 | 5 / 10 |
| 7B full history | 47.92 | 70.31 | 36.65 | 58.14 | 9 / 17 |

The 7B pools contain exact rules in 10 last-only and 17 full-history
environments, of which five and eight respectively are lost by the final
proposal. Selection retrieves those candidates. No 0.5B pool contains an
exact valid rule, so preserving an earlier candidate cannot supply one.
All 48 trajectories in each group select an earlier stage at the final
checkpoint; this includes earliest ties and does not mean that all final
proposals are strictly worse.

Selection improves final direct accuracy in 10/48 and 13/48 small-model
environments, and 20/48 and 22/48 large-model environments, for last only and
full history respectively. It never lowers the per-environment direct score:
that property follows algebraically from the known additive family and
balanced train/direct queries. It is not evidence that the model learned.
Individual direct answers can still regress when a different subset of
offsets becomes correct: gained/lost counts are 60/4, 76/8, 132/8, and 172/0
in table order. Composition scores fall in 5, 4, 2, and 3 environments despite
the group mean increases. These are descriptive results from the same fixed
pools, with no new significance claim or formal primary-endpoint replacement.

Selection adds full-example checks beyond the original first-failure checker:
9,056, 8,464, 12,032, and 11,664 training-example executions per standalone
pool group in table order. These counts include repeated candidates and the
initial proposal within each group; invalid candidates require parsing but
no example execution. The local offline analysis takes 0.668 seconds before
output serialization, excluding imports and all original model generation.
Its generation cost is already incurred in the formal archive; selection
does not establish a free intervention, a new neural capability, or a
prospective effect of feedback gating.

Reproduce without model inference using the selection commands in
[`../../experiments/study_g/README.md`](../../experiments/study_g/README.md).
Independent calculation from the audited rule memories reproduces all 1,152
selected-prefix records and 24 summaries, including gains/losses and extra
example checks; all input/source hashes match. Fresh reporting review found
no material issue. Cross-model CLI review was skipped in the autonomous run.

## Exploratory uniform-proposal reference

This later post-hoc calculation follows the plan and source at `7da2453`:
[`../../notes/study-g-random-reference-plan.md`](../../notes/study-g-random-reference-plan.md).
It evaluates expectations over a supplied sampler of the 1,680 distinct-offset
rules, with independent draws and replacement. The counts of rules matching
0, 1, 2, 3, or 4 true offsets are **1,001, 536, 126, 16, and 1**. The same
distribution holds for every legal true rule by relabeling the offset values.

Seventeen uniform legal candidates, selected using all disclosed examples,
have expected direct accuracy **48.71%** and exact-rule probability **1.01%**.
These are exact policy expectations, not observed agents or confidence bounds.
The sampler has no learned proposal generator, but its legal rule grammar,
example scorer, memory, and executor are all supplied algorithms.

The second reference retains each observed neural initial proposal and adds
16 uniform legal candidates. Thus it retains that model's initial generation
and prior learning. The following comparison conditions on those same 48
initial proposals and uses the training-only selector on both candidate pools.

| Model / observed pool | Observed selected direct % | Uniform-revision expected direct % | Observed minus expected, pp | Observed exact / expected exact count, out of 48 |
| --- | ---: | ---: | ---: | ---: |
| 0.5B last only | 17.71 | 48.47 | -30.76 | 0 / 0.46 |
| 0.5B full history | 19.27 | 48.47 | -29.20 | 0 / 0.46 |
| 7B last only | 63.54 | 63.84 | -0.30 | 10 / 4.42 |
| 7B full history | 70.31 | 63.84 | +6.47 | 17 / 4.42 |

The small-model pools have less direct-score utility than this uniform-policy
expectation under the supplied grammar and selector. In 7B, the full-history
pool has a higher observed direct mean and more exact rules than the reference
expectation. These are descriptive observed-versus-expected comparisons,
without a significance or universal capability claim. In particular, observing
zero exact small-model rules is not evidence of falling below a reference
whose expected exact count is only 0.46. Candidate validity, dependence, and
learned priors differ across policies; this is not an isolated causal effect
of feedback, neural parameters, or semantic candidate quality.

Each conditional reference has 48 initial neural proposals and 768 uniform
revision candidates. Full-example execution counts are 12,992 for the 0.5B
initial states and 13,008 for the 7B initial states, versus 9,056/8,464 and
12,032/11,664 for the respective observed selected pools. Uniform candidates
are always valid; invalid neural candidates are excluded without executing
examples. The pure seventeen-draw reference instead uses 272 example
executions per environment, or 13,056 for 48 environments, with no neural
initial generation. Candidate counts do not match these checker costs or
overall computation, and the pure reference has a different initial source.

`random-reference.json` preserves the exact rational expectations, 96
conditional records, four comparisons, and analysis/input hashes. The offline
expectation calculation takes 0.037 seconds before output serialization,
excluding imports. This is not a measured runtime of deployed random agents.
No neural inference or original result modification is involved. Reproduction
commands are in [`../../experiments/study_g/README.md`](../../experiments/study_g/README.md).
An independent rational dynamic program, using the original audited rule
memories, reproduces the pure reference, all 96 conditional records, four
comparisons, and checker totals. All input/source hashes match; independent
reporting review found no numerical or attribution issue in this section.
