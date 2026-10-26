# Study N: encoding and learning of unobserved transitions

Status: prospective after Study M/v16, before implementation, official data
generation, or outcomes. This is a controlled mechanism experiment. It does
not extend the action-world result to open-ended intelligence or introduce a
new learning algorithm.

## Question and estimand

Study M reached perfect new-goal success in a stationary structured world,
using a supplied bit encoding, transition replay, and breadth-first planner.
It did not reserve transition pairs from learning. Study N asks whether
online gradients predict genuinely unqueried transitions, and how much that
learning depends on the supplied encoding under identical physical evidence.

Use fresh development seed **47** and formal seeds **16000--16031**. Each
seed supplies one structured world and one independent random state-label
bijection. Both encodings use exactly the same ordered physical queries.
The sole primary is the small model's difference in learning gains:

    delta = (canonical_accuracy_64 - canonical_accuracy_0)
            - (relabeled_accuracy_64 - relabeled_accuracy_0).

Each accuracy uses the same fixed 64 unqueried state/action pairs; "64" is
the number of observed transitions. The unit of inference is a world seed,
not a transition, encoding, or checkpoint. Report the paired mean, two-sided
Student t95 interval, and 10,000 paired bootstrap resamples with seed **991**.
A positive primary requires the t lower bound strictly above zero. Report
degenerate intervals, sign counts, and disagreement with the bootstrap.
No other endpoint replaces this one after seeing outcomes.

Positive delta establishes an encoding-dependent learning effect under this
algorithm and budget, not by itself a positive gain in either encoding. Report
both gains and both final scores beside it. It does not establish a meaningful
degradation threshold or that high performance cannot survive relabeling. The
original final-score contrast,
large-model comparisons, bit accuracy, training fit, goal success, all curves,
and costs are descriptive. No equivalence test or universal capacity claim
is implied by an interval spanning zero.

## Worlds, encodings, and equal observations

Physical states are the 32 five-bit integers, and there are four actions.
Each action permutes the five bits and XORs a five-bit mask. Independently
sample permutations and masks, rejecting identity/duplicate actions and then
whole worlds whose directed union is not strongly connected. Record rejection
counts. The generator follows Study M's distribution, with independent
`study_n:` random-stream roles. No seed replacement depends on learning.

Canonical encoding is the identity. Relabeled encoding is a uniformly sampled
permutation b of the 32 state IDs, independent of maps and query selection.
An action becomes b(T_a(b^{-1}(x))). Inputs and outputs still have five bits.
The encoding changes neural geometry and the effective prior induced by the
architecture; it does not remove physical information, but learners receive
no decoder or cross-encoding correspondences. This is not a test of a
representation-invariant learner or of equal optimization difficulty.

For each action independently shuffle all 32 physical source states. For each
of 16 rounds, take the next source for every action and order those four
queries using a separate seeded shuffle. This yields 64 unique ordered
queries, 16 per action. The remaining 64 pairs form a fixed test set in
action-major, physical-state order. They never enter training or replay.
Source states and successor states can overlap between sets; what is held
out is a transition query, not an unseen state vocabulary or world.

The collector executes the queries one by one and reveals the true successor
after recording each pre-update prediction. Each encoding sees the same
physical outcomes in the same order, transformed consistently. Neither model
selects queries or follows a trajectory; arbitrary state reset/query access is
supplied. This is supervised online system identification, not active
exploration. Study M remains the interactive test. Record 64 physical queries
per world shared across branches, and charge 64 to each standalone learner.

Only the simulator/evaluator has full maps, labels b, and test targets. The
learner receives encoded triples in arrival order. Sampling uses observation
indices, never a sort by encoded state. Full world files remain outside the
trainer's input. Checkpoint evaluation begins only after all learning finishes.

## Predictor, updates, and branches

Reuse Study M's network at repository revision **13f1c71**: nine inputs,
ReLU hidden units, and five logits, with five current-state
bits plus four one-hot action bits. Hidden widths 32 and 128 have 485 and
1,925 total parameters. Initialize separately by width and seed; clone exact
initial tensors across encodings. There is no pretraining or warmup.

CPU float64, one thread, deterministic algorithms, PyTorch **2.8.0+cu126**.
Use Adam with lr .01, betas (.9,.999), eps 1e-8, weight_decay=0, and
amsgrad/maximize/foreach/fused false; clip total gradient norm at 1. For each
observed transition take eight small-model or two large-model steps. Each
step samples 32 entries with replacement from the current observed prefix,
using a stream keyed by world seed, width, and cumulative gradient index.
Use identical index draws across encodings. Loss is mean binary cross-entropy
with logits across the 32 rows and five successor bits.

Maintain weights, full Adam moments, and the ordered replay prefix throughout
all 64 queries. Archive checkpoints at **0, 16, 32, 64** observations; no
checkpoint selection. Frozen controls use each width's original tensors
throughout and have no unused training. A non-neural observed-table branch
retains exactly the acquired edges. Ten encoding/branch cells result from
two encodings times small online, large online, small frozen, large frozen,
and observed table. Uniform fallback behavior is visible in every branch.

Record each pre-update prediction before target exposure, each gradient's
loss, norm and sampled-index digest, each batch's actual parameter-change
norm/count, and each saved model/optimizer/corpus. No record of an update
counts as proof of improvement. Frozen tensors must remain byte-identical.

## Evaluations and planner controls

After all training, compute the full 128-edge predicted graph and logit digest
at each neural checkpoint. Exact test accuracy compares decoded successor IDs
on the 64 never queried pairs; bit accuracy uses encoded bits and is explicitly
encoding-dependent. Report observed-prefix accuracy separately, with no
training score at the empty prefix. All test pairs are fixed across checkpoints.
Frozen direct predictions are evaluated once then reused explicitly.

Report two non-neural reference calculations. First, an action with m distinct
observed sources has 32-m unassigned successors because it is a permutation.
A uniformly sampled consistent permutation completion has expected held-out
exact accuracy 1/(32-m): 1/32, 1/28, 1/24, and 1/16 at the four checkpoints.
These are exact expectations, not measured samples or a neural baseline.

Second, in canonical coordinates a family-informed matcher considers ten
signed input-coordinate rules for each of the five output bits of every
action. Retain rules consistent with observed triples; predict an unqueried
output only if every remaining rule agrees on all five bits, otherwise abstain
and count it incorrect. Report coverage and exact accuracy on the same fixed
test pairs. This comparator is supplied the signed-coordinate family; it is
not given the relabeling inverse, is reported only in canonical coordinates,
and has a stronger explicit prior than the generic neural network. Report its
rule-comparison work and surviving candidate counts; no neural parameters.

For a fixed true rule, the opposite sign of the same coordinate never agrees,
and each of the other eight signed coordinates agrees on exactly 16 of the
32 inputs. With m distinct uniformly sampled sources, the union bound for
any ambiguity among the 20 output rules is

    min(1, 160 * choose(16, m) / choose(32, m)).

At m=16 this is about 2.662e-7. Query selection is independent of maps, so
the bound holds conditional on any generated world. It is conservative because
the union bound counts overlapping ambiguity events separately. The reference
need not enforce a joint input-bit permutation: a consistent coordinate
substitution can be extended by swapping equivalent observed columns and
adjusting their signs in the original true permutation.
This exposes the task's low identification difficulty with the supplied
canonical family, and prevents interpreting high scores as surprising evidence
of intelligence. The reference is not a general or matched-prior learner.

The supplied BFS planner is unchanged from Study M: numeric action order,
one visit per graph state, an eight-edge search bound, and a seeded uniform
fallback when no path exists. Test all 992 different start/goal pairs with an
eight-action execution cap. Replan from the actual state after each action;
weights and cached graph stay fixed, and the branch cannot read replay or
learn probe transitions. Goal success therefore includes supplied planning
and current-state feedback, not just neural prediction.

Fallback tapes are keyed by seed, physical start, physical goal, and step,
then supplied as actions to both encodings; no physical ID is an input to
the learner or planner. Exact-table graphs contain only acquired edges, with
unknown successors absent. Evaluate every table prefix. Both frozen controls
run once per encoding and are reused across checkpoints. Record every probe's
action/state trace, success, planned paths and planning-operation counts.

Independently establish planner equivariance for arbitrary, possibly wrong,
graphs under consistent relabeling, including missing edges, cycles and
fallbacks. This is a synthetic invariant, not only an observation on the true
or exact-table graph. True shortest distances and exact-table probe outcomes
must agree across encodings on the official grid. A learned graph need not
equal the conjugate of the other learner's graph; such divergence is part of
the encoding intervention. True-map evaluation is an information bound only.

## Resources and verification

Training's dense matrix proxy is 33*h MACs per row, 32 rows per step, and
270,336 MACs per observed transition at either width. Small/large models take
512/128 steps, respectively, totaling 17,301,504 training MACs each. This is
not total FLOPs or matched wall time: Adam, nonlinearities, sampling, journals,
graph construction, and planning differ. Count all model forwards, gradient
rows/steps, query actions, graph inference, probe actions, planner edges,
model/Adam/replay storage and measured phases. Include pre-update prediction
forwards. Report shared actual collection and standalone costs separately;
evaluation and archive work belong to actual collection, not deployed cost.

Before official generation, verify synthetic encoding bijections/conjugacy,
query/holdout separation and balance, identical physical evidence, initial
tensors and sampled indices across encodings, weight changes and Adam
persistence, exact budget boundaries, read-only test evaluation, arbitrary
graph planner equivariance, and inference statistics on hand-built counts.

Independently reconstruct world generation, action outcomes, all gradients,
Adam states, saved tensors and corpora, graph probabilities, direct predictions,
probe traces and operation counts. Never call the collector's training or
evaluation loop from the auditor. Tensor/checkpoint values and journal digests
match exactly; an independent NumPy forward uses rtol 1e-11/atol 1e-12. Check
complete row/file grids and hashes before and after replay. A matching
development audit is required before formal collection. Formal output must
be a new directory; failures remain archived and do not trigger silent reruns.

Review and commit this protocol, implement with synthetic fixtures, review
and freeze source/runtime, then freeze the complete 33-world data file.
Run development once, audit it, and run the unchanged formal grid only after
the certificate passes. No tuning, early stopping, or selected reruns follow
development. Any defect requires a recorded amendment and new freeze before
replacement evidence; retain the affected run and qualify its status.
No incomplete seed/encoding pair is excluded or replaced. Any incomplete grid
has no confirmatory summary. Infrastructure failures before collection may
be retried with unchanged settings and recorded provenance; once any record
is produced, preserve that attempt and document the reason and scope of any
replacement run before executing it. A fixed-seed rerun after an implementation
change cannot silently regain confirmatory status.

This experiment isolates a representation intervention at matched physical
observations. It cannot establish general intelligence, innate human
representations, performance on natural domains, optimal learning rates,
or that larger models are inherently necessary. It neither explains all of
Study M's success nor tests every way a small agent could learn representations.
