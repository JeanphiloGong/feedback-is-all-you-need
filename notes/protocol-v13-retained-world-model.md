# Study M: retained predictive models from action feedback

Status: prospective after Study L and manuscript v15, before implementation,
official task generation, or outcomes. The previous studies learned program
priors from solutions or rejections. This study instead learns the next state
produced by an action, updates during interaction, and retains that model for
new goals. It changes the domain, feedback content, and objective together;
cross-study differences cannot identify the cause of Study L's deficit.

## Question and primary comparison

Can a small neural world model acquire useful behavior on previously unused
goals through persistent action-outcome learning? How does that learning curve
compare with a larger model under the same dense training-multiplication
allocation, and with delayed updates, freezing, and exact observed memory?

The sole primary is **small immediate minus large immediate normalized area
under the held-out-goal success curve in structured worlds**, paired across
32 environment seeds. Fixed checkpoints are 0, 64, 128, 256, and 512 main
interactions. For success rates a(t), compute

    AUC = sum((t_next - t) * (a(t) + a(t_next)) / 2) / 512.

This is a prespecified trapezoidal summary of five frozen-probe evaluations,
not measured success at every intervening interaction. Report the paired mean,
two-sided t95, and 10,000-resample percentile bootstrap with seed **983**.
A positive primary requires the t lower bound above zero. Report interval
disagreement and degeneracy. Failure to find superiority does not establish
equivalence. All final scores, random-world comparisons, cadence contrasts,
freeze comparisons, curve changes from the initial score, and costs are
descriptive; none replaces the primary after outcomes.

The comparison couples capacity and gradient-step allocation under a declared
training arithmetic proxy. It does not isolate parameter count at fixed
updates, equal complete FLOPs, equal elapsed time, or a universal efficiency
law. The initial learned states also differ across widths. The factorial
cadence/freeze controls distinguish some narrower explanations.

## Worlds, goals, and hidden information

Use development seed **43** and formal seeds **15000--15031**. Each seed has
two separately generated worlds, named `structured` and `random`, on the
32 five-bit states with four actions. World generation uses SHA-256-derived
role-specific standard-library random streams, following the existing `rng`
convention. Freeze the complete world/goal data and digest before outcomes.

In a structured world, each action permutes the five input bits and XORs a
five-bit mask: output bit j is input bit permutation[j] XOR mask[j]. Sample
the permutation uniformly and the mask uniformly from 0--31. In a random
world, each action is a uniform permutation of the 32 states. Within either
world reject identity/duplicate actions, then reject the entire four-action
world unless its directed union is strongly connected. Record all rejection
counts. These are designed solvable finite worlds, not natural environments.
No official seed or world is replaced in response to learner performance.

Choose eight held-out goal states uniformly without replacement; the other
24 states are training goals. The 248 probe tasks contain every held-out goal
paired with every different start state. States used as held-out goals may
occur in ordinary transition observations; what is withheld is their use as
training goal cues. The neural transition predictor receives no goal input.
Transfer therefore concerns composing learned transitions toward new goals
with a supplied planner, not zero-shot prediction of unseen state values.

Store a common warmup start and 16 uniform warmup actions. Execute that walk
once per world and share its observed triples across learned/table arms.
Generate a 513-entry training episode queue: each goal is sampled uniformly
from the 24 allowed goals, and its start uniformly from the 31 other states.
The queue is long enough even if every main action finishes an episode.
All arms start at queue entry zero, independent of the warmup's final state.
An episode ends upon reaching its goal or after 16 executed actions; the next
queued start/goal is installed without charging an action. Count such resets
separately. Faster arms can encounter more goals under the fixed 512-action
budget; paths, goal exposure, and acquired observations are endogenous.

The environment supplies only an executed action's true next state to each
agent. Exact action maps, structured permutations/masks, unused goals, and
probe results are unavailable to its predictor, replay sampling, update rule,
planner, and stopping decisions. The simulator and later evaluator necessarily
retain these data. All states/actions and the five-bit encoding are supplied.

## Predictor, representation, and arithmetic

Input is five state bits, least significant first, followed by four one-hot
action bits. Use a fully connected layer from 9 inputs to h hidden units,
ReLU, and a linear head with five output logits. A next-state bit is predicted
as one exactly when its logit is nonnegative. Predicted tables need not be
permutations. Widths **h=32** and **h=128** have **485** and **1,925** parameters,
including biases. No pretrained weights or learned task encoder are supplied.

The smaller architecture can represent every structured action family:
20 ReLU units implement input-bit AND action-indicator gates; four additional
units copy action indicators to implement mask offsets. A linear head then
produces each required bit with a finite signed margin. The larger width can
represent every random table using one ReLU indicator per valid state-action
pair (128 units). Verify both constructions on synthetic fixtures before any
official data. These are expressivity witnesses, not evidence that gradient
training finds those weights, and not a universal capacity threshold.

Use CPU float64, one thread, deterministic PyTorch **2.8.0+cu126**, default
Linear initialization with separately seeded widths, and Adam lr .01,
betas (.9,.999), eps 1e-8, no weight decay, amsgrad/maximize/foreach/fused false.
Clip gradient norm at 1. Loss is binary cross-entropy with logits, averaged
over all 32 rows and five output bits. Never clip or threshold logits before
the loss. Initializers and all four parameter tensors are archived.

The replay corpus contains one true observed triple per distinct state-action
pair, sorted by (state, action), at most 128 entries. Revisiting a pair does
not add duplicates. This deduplication assumes a stationary deterministic
world and supplies an explicit external memory, in addition to the network.
Each gradient step samples 32 entries uniformly with replacement. Its random
stream is independently derived from seed, world family, hidden width, and
cumulative gradient-step index; cadence branches within a width use the same
draw stream at a given index, although their available corpora can differ.

Collect all warmup observations before training either initial model. Take
128 steps for the small model and 32 for the large model on the complete
warmup corpus. Clone the resulting weights and full Adam state into each
cadence branch; the frozen branch retains those weights. Continue optimizer
state and cumulative gradient indices throughout main interaction. Never
reset weights, moments, or replay at an episode boundary.

Each main observation allocates eight small-model or two large-model steps.
For this architecture, dense training matrix products require 33*h MACs per
row: two products through the first layer and three through the second;
the input does not require a gradient. With 32 rows, both allocations have
270,336 MACs per observed transition. Count one multiply-accumulate as one MAC.
Bias, activation, loss, sampling, clipping, Adam, and I/O work is outside that
proxy and is separately timed or counted. Do not call it total FLOPs.

A complete predicted table evaluates all 128 state-action pairs, costing
128*14*h forward MACs: 57,344 small or 229,376 large. Cache this table between
weight updates. Table inference and planner work are not matched between
widths or cadences and remain explicit lifetime costs.

## Eight arms and interaction policy

1. `small_immediate`: update eight steps after every main observation.
2. `small_delayed`: collect 16 observations, then update 128 steps.
3. `small_frozen`: keep the common small learned initialization.
4. `large_immediate`: update two steps after every main observation.
5. `large_delayed`: collect 16 observations, then update 32 steps.
6. `large_frozen`: keep the common large learned initialization.
7. `observed_table`: retain exact observed transitions and update after each
   observation, with no neural model or optimization.
8. `uniform`: choose an action uniformly, with no learning or warmup requirement.

All arms execute exactly 512 main actions, including actions on unsuccessful
episodes. Delayed batches occur at multiples of 16, including 512, irrespective
of episode boundaries. Their replay receives observations immediately, but
their policy cannot read it; only the scheduled weight updates expose its
content to the predictor. At these fixed probe checkpoints, all allocated
delayed updates have completed. Thus within a width, immediate/delayed arms
have identical scheduled cumulative gradient counts, but different experience
and update timing. Frozen/table/uniform arms need no unused optimizer work.

Neural arms use breadth-first search on their cached predicted graph from the
actual current state to the current goal, considering actions in numeric
order and each graph state once. A shortest path of at most eight edges gives
the first action; an absent path gives a uniform fallback. Replan after each
observed transition. The table arm uses only known edges; an unknown edge is
absent, not an identity or oracle prediction. Uniform uses the same fallback
action tape. Training fallback draws are keyed by seed, family, and main
action index; probe fallback draws by seed, family, start, goal, and step.
They never depend on arm identity or hidden maps. A prediction error is logged
before any update. No planning simulation supplies a new learning target.

The controller always observes its true current state for replanning, also
in frozen-probe evaluation. Therefore even a frozen model has a feedback
control loop; the neural contrast concerns weight adaptation, not presence
versus complete absence of feedback. The BFS algorithm, state/action encoding,
known state set, fallback policy, and episode resets are supplied capability.

## Frozen probes, integrity, and costs

Archive model/optimizer/corpus checkpoints after warmup (t=0) and after main
actions 64, 128, 256, and 512. At every completed update batch journal step
losses, unclipped norms, sampled-key digests, cumulative step index, and the
before/after weight-change norm/count. Retain all action outcomes, episode
boundaries, corpus additions, predicted discrete tables, and their underlying
logit-byte digests. Keep every failure and partial run.

Complete all main interactions before evaluating any probe. Each probe starts
fresh with a fixed saved graph, no replay access, no parameter updates, no
memorized probe transitions, and an eight-action cap. Reaching the goal is
success, including on the eighth action. Record the complete trace, success,
actual actions, fallback choices, and original predicted-next-state errors.
Frozen/uniform probes are executed once and reused across the curve; they
are not independent timing or statistical replications. The 32 paired seed
means, not goals, checkpoints, two world families, or eight arms, are the
replication units for the stated primary.

Also report the exact shortest-path reachability of each probe under the true
graph, computed by the separate evaluator only after collection. This is a
supplied-information bound, not a trained agent or a matched-information arm.
Report all fixed curves, goal success during interaction, pre-update next-state
bit/exact accuracy, distinct transition coverage, episode/reset counts, and
the exact-table reference. Scores alone cannot attribute capability to weights.

Record actual collection work and standalone arm costs: necessary shared
warmup acquisition, initial training, own replay/updates, graph inference,
planning, fallback decisions, and actual environment actions. Charge shared
initial learning once per standalone learned arm/world, not once per episode.
Evaluation and explicit checkpoint/table/research archives stay in actual
collection costs; journal serialization/write/flush inside phases is timed
separately. Remaining phase/status/file-management overhead stays in total
elapsed time. Shared phase sums are not independent wall-time benchmarks.
Include model/Adam tensor bytes, replay entries/bytes, graph storage, planned
edges, gradient/forward MACs, physical row counts, and measured CPU durations.
Archive size is not inference RAM. Report feedback content as true next-state
observations, not one-bit failure flags comparable to Study L.

## Freeze, independent checks, and interpretation

Review and commit this protocol before implementation and official generation.
Before the source freeze, use only hand-built/synthetic fixtures. Test bit
encoding, exact parameter counts, both constructive capacity witnesses,
analytical gradients, actual weight changes, sampled replay, persistence,
delayed boundaries, identical-width initial states, known-only table planning,
shortest-path ties, probe isolation, exact-cap success, and complete budgets.

Independently review the implementation and audit, then freeze source, runtime,
and the complete generated grid. Run development once, audit it, and execute
the unchanged full formal grid only with matching evidence. No tuning, seed
replacement, outcome-driven stopping, or best-checkpoint selection follows
development. An implementation defect requires a recorded amendment/new freeze;
retain affected evidence rather than silently restoring confirmatory status.

Independent audit regenerates worlds/roles, reconstructs every physical
transition and policy choice, replays every gradient/Adam state without
calling the production update loop, checks neural forward values with NumPy,
and verifies each checkpoint, journal, corpus, cost ledger, and file manifest.
Pin numerical tolerances before official outcomes. Reconstruct the full AUC
and paired inference separately from raw probe traces after transfer.

This tests retained predictive learning and goal transfer in small, fully
observed, stationary finite worlds. A positive primary would support one
resource allocation and representation, not unrestricted intelligence,
autonomous representation discovery, or feedback sufficiency. Model-based
learning and planning have substantial prior work; no novelty for their basic
combination is claimed. Exact memory, dense feedback, and a supplied planner
may account for much of system performance. Failures may concern optimization,
exploration, representation, or model error; none is inferred from scores alone.
