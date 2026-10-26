# Study I: post-hoc likelihood diagnostics

Status: specified after the formal search outcomes and manuscript v11, before
computing these diagnostics. This is an exploratory analysis of the existing
archive, not a new prospective test of feedback efficacy or an intervention.
The unfavorable primary result and all original endpoints remain unchanged.

## Question and competing interpretations

Did the trained priors fail to fit their accepted experience, or does a lower
experience prediction loss coexist with poor transfer search? A second
description asks whether probability assigned to transfer generating programs
differs by their length and token position. These observations can narrow the
next experiment; they cannot isolate the cause of failure. In particular,
positional dependence can reflect content and prefix differences, and a
generating sequence is only one of possibly many programs for the same task.

No optimization, search, model selection, or new task generation will occur.
The original neural policy still has no task examples as input. Hidden
generating programs are used only by this separate offline diagnostic, after
the original searches and training are complete.

## Fixed inputs and complete grid

Use all formal seeds 11000–11031 from source/data freeze 6365bec. Verify all
967 raw files against the archived passing audit's hashes before analysis;
record that audit's own digest, the diagnostic source hashes, package versions,
and elapsed analysis time. Preserve every input and write to a new output
directory. Development seed 29 is not pooled into these descriptions.

Include all six final arms and both intermediate probes: uniform, counts,
small frozen, small trained, small shuffled, large trained, small after 8,
and small after 16. Use the exact saved checkpoints and count tables. Neither
the best checkpoint nor a subset of environments may replace this grid.

Score each arm on these common sets:

- All accepted nonempty experience programs, in stored corpus order. These
  are solutions from search, not the generator's experience programs.
- All 16 composition and all 16 shifted generating programs. Retain their
  task identifiers, lengths, and every token position.

Also score each trained neural arm on its own actual training corpus: the
accepted prefix after 8/16 attempts for the two probes, all accepted programs
for the final small/large models, and the stored shuffled programs for the
shuffled model. Select a prefix by accepted task identifiers and the original
attempt order, not by assuming that every attempt succeeded. This extra view
is distinct from the common full-experience comparison.

## Measures and aggregation

For program p, compute teacher-forced next-token probabilities at prefixes
p[:i], including START exactly as in the original model. Retain per-token
log probabilities for both the raw prior and the search prior 0.9 pi + 0.025.
There is no EOS target. Report negative log probability in natural units
(nats), averaged over real target tokens. A uniform token prior has loss
log(4). The common uniform length prior adds log(10) to whole-program loss;
it is excluded from token loss for every arm and adds no ranking difference.

For each actual training corpus, also compute the unrestricted empirical
prefix predictor's minimum token loss:

    H_emp = -sum_{prefix, token} n(prefix, token)
                    * log(n(prefix, token) / n(prefix)) / N_tokens.

This bound allows a different distribution at every complete prefix. It
accounts for conflicting continuations in an unconditional corpus, but makes
no neural capacity or generalization claim. Report raw loss minus H_emp;
zero would attain the unrestricted in-sample bound, not prove useful learning.

Retain program-level records and per-environment summaries. For each common
set, report token means overall, by generating length, and by one-based token
position within length. Aggregate by equal weight over the 32 environment
means; do not treat nested tokens or tasks as independent replicates. For
training-corpus views, report loss, bound, and excess for every environment
and each trained arm. Keep all positions and lengths, including unfavorable
ones; do not choose a favorable range after inspection.

Report two explicit descriptive differences with every paired value:
small trained minus small frozen on the common full accepted corpus, and
small trained minus uniform on each transfer family's raw and smoothed token
loss. Give means and counts of negative, zero, and positive environments.
No confirmatory p-values, post-hoc success threshold, or new primary endpoint
is introduced. Checkpoint trajectories use the same common full corpus and
transfer programs; their separate own-corpus losses use different datasets.

## Verification and interpretation boundary

Test the prefix lower bound against hand-computed conflicting continuations,
verify uniform loss and smoothing, and compare GRU scores with the independent
NumPy gate equations already used in the original audit. Every neural token
probability in the official diagnostic must agree with that independent path
within absolute and relative tolerance 1e-12. Reconstruct all summaries from
the retained records independently after collection, and verify completeness
against the fixed input grid before reporting. Freeze the reviewed analysis
plan and implementation before computing the new official diagnostics.

Training fit and transfer sequence likelihood are different outcomes from
budgeted task success. Low probability for the generator's program does not
measure total probability of its semantic equivalence class, exclude a
shorter solution, or prove that length extrapolation caused the search result.
The same generator creates both experience and transfer, so their structural
relationship is supplied. Any causal explanation or changed training method
requires a separate frozen intervention on fresh environments. This analysis
adds only descriptions of existing checkpoints and cannot establish that
additional feedback, parameters, or optimization steps would fix the failure.
