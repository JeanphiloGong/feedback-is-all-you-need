# Study O: more optimization from unchanged feedback

Status: prospective after Study N/v17, before implementation, official data
generation, or outcomes. The previous goal turn made substantive progress:
the complete Study N experiment, independent evidence reconstruction, and
manuscript v17 are committed. Its relabeled learners fit observed transitions
poorly, so their low unqueried accuracy does not isolate a capacity limit.

## Question and primary comparison

Does increasing the number of gradient steps per observed transition improve
unqueried prediction while keeping physical feedback, architecture, and
initialization fixed? This tests optimization work per observation, not the
number of informative feedback events, a faster physical environment, or
learned active exploration. More updates reuse existing evidence.

Use development seed **53** and formal seeds **17000--17031**. The sole
primary is the small relabeled predictor's final exact-accuracy difference:

    delta = unqueried_accuracy_16x_at64 - unqueried_accuracy_1x_at64.

The initial tensors and initial predictions are identical within this paired
contrast, so it is also a difference in learning gains. Accuracy uses the
same 64 permanently unqueried transition pairs. Pair by physical world seed,
not by transition, encoding, model size, budget, or checkpoint. Report the
mean difference, two-sided Student t95 interval, and 10,000 paired bootstrap
resamples with seed **997**. A positive primary requires the t lower bound
strictly above zero. Report sign counts, interval degeneracy/disagreement,
both absolute final scores, and both gains from initialization.

A positive primary establishes an effect of the registered update multiplier;
it does not establish high capability or recovery to canonical performance.
An interval spanning zero does not establish equivalence. No meaningful-
degradation margin, optimal budget, or universal capacity boundary is tested.
All other endpoints and comparisons are descriptive: 4x, canonical encoding,
large networks, observed fitting, losses, pre-update prediction, learning
curves, and resource costs. No endpoint replaces the primary after outcomes.

## Fixed physical evidence and representation

Reuse Study N's world generator at source revision **df2ecb3**, including its
`study_n:` random-stream roles, on the fresh seeds above. Each world has 32
five-bit states and four actions. Each action is a bit permutation XOR mask;
identity/duplicate actions and worlds without directed strong connectivity
are rejected with counts recorded. The complete 33-world file is generated
once only after source/runtime freeze, before development.

Each world is represented canonically or by its independent uniform random
bijection of the 32 state IDs. Actions are conjugated by that bijection.
Both encodings retain the same physical information; input geometry and the
architecture's induced prior change. Learners receive neither the inverse
encoding nor cross-encoding correspondences.

The frozen query schedule contains 16 distinct sources per action, interleaved
in 16 independently ordered four-action rounds: 64 unique physical queries.
The complement of 64 pairs, in action-major physical-state order, is never
used by any learner. States, successors, and the same dynamics can occur in
both sets. This is not a disjoint-state or unseen-world transfer experiment.
All branches receive identical ordered physical evidence. Arbitrary state
query/reset access is supplied; no model chooses the observation sequence.

Record every branch's prediction before executing the common physical query
and revealing its successor. Only the observed encoded triple enters that
branch's arrival-ordered replay. Full world maps and test labels are owned by
the simulator/evaluator, not supplied as training inputs. No tuning or data
selection follows development or formal outcomes.

## Models, optimizer, and update intervention

Reuse the unchanged Study M predictor and Study N gradient routine: nine
inputs (five state bits and four action indicators), a ReLU hidden layer,
and five successor-bit logits. Widths 32 and 128 contain 485 and 1,925
parameters. There is no pretraining or warmup. Initialize with Study N's
width/seed random role and clone exact tensors across encodings and budgets.

Registered online multipliers are **1, 4, 16**. Per newly observed transition,
small models take **8, 32, 128** Adam steps and large models take **2, 8, 32**.
At 64 observations, total updates are 512/2,048/8,192 for small models and
128/512/2,048 for large models. Each step samples 32 entries with replacement
from the current arrival-ordered prefix. Use the existing random role keyed
by world seed, width, and cumulative gradient step, without a multiplier key.
Within each budget/width, physical minibatch indices are identical across
encodings. Across budgets, cumulative steps can occur at different observed
prefix sizes: common random streams do not imply identical sampled evidence
at every step. The first shared steps on the first observation coincide.

Weights, full Adam moments, replay, and cumulative gradient count persist
through every observation. Mean binary cross-entropy over rows and five bits,
learning rate .01, betas (.9,.999), epsilon 1e-8, zero weight decay, gradient-
norm clip 1, and amsgrad/maximize/foreach/fused false remain unchanged.
Use deterministic CPU float64, one thread, PyTorch **2.8.0+cu126**. Multipliers
are fixed before synthetic implementation checks and are not adjusted using
official or synthetic accuracy. Runtime-only feasibility checks may be made
on negative-seed fixtures without inspecting their outcome metrics.

Archive each gradient's loss, norm, and sampled-index digest, and every
observation batch's actual parameter-change count/norm. Archive full
model/optimizer/replay states at **0, 16, 32, 64** observations, with no
checkpoint selection. Each encoding/width also has an untrained frozen
control, evaluated once at initialization and explicitly reused along curves.
There are 12 adaptive streams and four frozen controls per world.

## Evaluation and the capacity distinction

Evaluation starts after every learning stream in the run finishes. Each
saved predictor produces all 128 state/action logit vectors once. Archive
the full float64 logits, their byte digest, threshold-decoded graph, and
derived direct predictions. Record unqueried exact accuracy, encoded-bit
accuracy, and mean bitwise BCE on all 64 held-out pairs. Record observed-
prefix exact/bit accuracy and BCE separately; observed metrics at the empty
prefix are undefined, not zero. Frozen scores are evaluated once and reused;
no final-prefix training-fit score is inferred for a frozen model.

Report all four checkpoints and every width/encoding/multiplier. Unqueried
BCE can expose increasing confidence in incorrect predictions even when
exact accuracy changes little. Pre-update predictions on the acquisition
stream are descriptive and are never substituted for held-out performance.
Evaluation cannot update model tensors, Adam state, or replay. There is no
goal-planning endpoint in this diagnostic: Studies M and N retain the evidence
about supplied BFS and live state feedback. This experiment isolates the
predictor's fitting/generalization response to more optimization.

Reuse the two reference calculations from Study N at every checkpoint: the
canonical family-informed signed-coordinate matcher, and the uniform
consistent-permutation completion expectation 1/(32-m), where m is the
observed-source count per action. The latter is an expectation under that
specific completion procedure, not a universal chance level or information
bound for the structured hidden-family distribution. The symbolic matcher
has a stronger explicit prior and receives no inverse relabeling. Its rule
comparisons, candidates, coverage, and predictions are retained. An observed-
edge table would abstain on every unqueried pair; no extra execution trial
is needed to establish that fact.

Expressive capacity and success of optimization must remain separate. The
unchanged 128-hidden-unit architecture has a constructive exact solution for
any full 128-row transition table, including either encoding. For a source
state b and action a, create a hidden unit

    h_(b,a)(x,u) = ReLU(sum_j (2*b_j-1)*x_j + u_a + count_zeros(b) - 5).

It equals one exactly on that valid state/action input and zero on every
other valid input. Set each output weight to +3 or -3 according to the
corresponding true successor bit and set output biases to zero. Every output
logit has the correct sign and magnitude three. This is an existence proof
using the full true table, not an agent trained from 64 observations or a
data-efficiency result. Synthetic arbitrary-table tests in Study M already
verify the construction; Study O rechecks it against the reused model.
The small canonical family also has Study M's 24-unit construction. No
corresponding universal small-model witness for relabeled tables is claimed.

If extra steps improve observed fitting without a demonstrated unqueried
gain, the conclusion is limited to the tested objective and budget. It is
not proof that optimization can never recover generalization. If the large
model still does not fit observations, lack of an expressive solution is
not a sufficient explanation, given its witness; the schedule/optimization
path remains unresolved. There is no claim that 16x is convergence.

## Resource accounting, audit, and failure policy

Per learner the dense training-matrix MAC totals are **17,301,504**,
**69,206,016**, and **276,824,064** at 1x, 4x, and 16x, respectively, at either
width. The primary deliberately increases optimization resources sixteenfold
while holding unique feedback fixed; it is not a matched-compute comparison.
Width comparisons share that matrix proxy but differ in step count and full
work. Record all gradients, sampled training rows, target bits, pre-update
and evaluation forwards, storage bytes, phases, and total elapsed time.

Actual shared collection uses 64 physical queries per world. Standalone
learning charges the width's initialization, encoded replay insertion, and
all 64 queries once per learner, plus that learner's own prediction/update
work. Separate known journal I/O from learner phase subtotals. Sampling
digests and some instrumentation remain; these are not pure deployment-time
measurements. Archive/evaluation/reference work stays in actual totals.
Tensor/serialized-replay byte counts are not resident RAM.

Before official data generation, verify the exact update budgets, shared
initial states, physical query/minibatch pairing, retained optimizer moments,
held-out separation, frozen controls, full-logit metrics, capacity witnesses,
and primary inference on hand-built values. Synthetic seeds must be negative;
no official outcomes are used to repair implementation.

Independent audit regenerates worlds, all query outcomes and gradients,
full checkpoint states, logits (with a separate NumPy forward), direct
metrics, references, and resource counts. Never call the collector's training
or evaluation loop from the auditor. Require the complete row/file grid,
source/data freeze, pinned runtime, and raw hashes before/after reconstruction.
Reject optimized Python. A complete matching development certificate and
unchanged development raw bytes are required before formal collection.

Freeze reviewed source/runtime, then the complete official data grid, run
development once, audit it, and only then run the unchanged formal grid.
Use new output directories. Failures preserve partial evidence and have no
confirmatory summary; never exclude or silently replace a seed or branch.
Infrastructure failures before any collection record may be retried unchanged
with provenance recorded. Any implementation defect after outcomes requires
an explicit amendment and new freeze; fixed-seed reruns cannot silently regain
confirmatory status. The research hypothesis remains open regardless of this
single finite experiment's result.
