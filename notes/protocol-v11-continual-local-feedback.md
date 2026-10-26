# Study K: continuing local-prior learning across a distribution change

Status: prospective design after Study J and manuscript v13, before Study K
source implementation, task generation, or outcomes. Study J showed favorable
composition transfer with a supplied local context, but poor shifted and
balanced performance. It ended all learning before transfer. This study tests
whether learning from further visibly accepted solutions improves performance
on untouched tasks, with a control for additional optimization alone.

## Question and primary endpoint

Does a 1,172-parameter local program prior improve its final balanced held-out
exact success by continuing to learn from its own accepted stream solutions,
relative to freezing the same initial learned model? Use both stream orders:
composition then shifted, and shifted then composition. Average the two orders
and both held-out families equally within each of 32 environments. The single
primary contrast is `online` minus `frozen` at the final checkpoint, under the
same 65,536-primitive-execution cap per held-out task.

Use a two-sided 95% t interval over 32 paired environment means and a
10,000-resample paired percentile bootstrap with seed **967**. A prespecified
positive result requires the t lower bound above zero; disclose disagreement
with bootstrap or degenerate intervals. A complete audited grid, not a
favorable result, defines successful execution. No primary inference is
permitted on an incomplete grid.

Uniform search and extra replay without new solutions are essential
comparators. Beating frozen alone does not establish superiority to uniform
or that the gains require new outcomes. All other contrasts, families, stream
orders, checkpoint curves, and prequential stream scores are descriptive.
There is no best-checkpoint selection, stopping-rule selection, or tuning.

## Fresh domain and disjoint task roles

Reuse the unchanged Study H generator with development seed **37** and formal
seeds **13000–13031**. Freeze all 33 environments, their digest, and task-role
mapping before any official search or learning. Four primitive permutations
act on 16 states; experience programs contain the hidden length-three motifs.
Keep the generator's full-function uniqueness/rejections and actual experience
count, without resampling seeds or requiring exactly 24 experience programs.

For each generated composition/shifted family, allocate tasks by their original
zero-based index:

- Online stream: indices **0, 8, 2, 10, 4, 12, 6, 14**, in that order.
- Held-out probes: indices **1, 9, 3, 11, 5, 13, 7, 15**, in that order.

Both subsets have four length-six and four length-nine tasks; the stream
alternates lengths to avoid a length change coinciding with the family change.
There are eight stream tasks per family and eight held-out tasks per family.
Each environment runs both orders, with independent branch states cloned from
the same model/optimizer after initial experience. Within each family, the
order above stays fixed. The supplied family switch defines the experiment;
the model/update rule receives no family labels, length labels, latent motifs,
generating programs, hidden labels, or probe scores. Stream task signatures
are distinct from experience and held-out signatures under the fixed generator.

Each task reveals its usual eight input/output examples to the searcher and
reserves eight for evaluation. Neural prediction sees only the last two
program tokens. Primitive tables and visible examples remain supplied to the
searcher; this is not a task-conditioned neural policy.

## Initial learning and branch definitions

Use the unchanged Study I `ProgramPrior(16)` and Study J local training and
prediction rules, including one zero-state row per target, START followed by
up to two preceding tokens, right padding, and last-real-output selection.
The model has 1,172 parameters, with no pretraining or additional embeddings.
Initialize directly in CPU float64 using
`rng(seed, 'study_k:initial').getrandbits(63)` and `torch.manual_seed`, then the
unchanged default GRU/head initializers. Pin PyTorch **2.8.0+cu126**, one thread,
and deterministic algorithms.

Acquire initial experience once with the unchanged primitive-only breadth-first
search at 65,536 primitive applications per task. Append every accepted
nonempty program, even if hidden evaluation marks it wrong. Preserve repeated
solutions rather than deduplicating. After each initial attempt, take 16
full-corpus Adam steps with lr .01, betas (.9, .999), eps 1e-8, weight decay
zero, amsgrad/maximize/foreach/fused false, and gradient-norm clipping at one.
The loss is raw next-token cross-entropy averaged over real targets, excluding
padding and EOS. Retain optimizer state. Empty corpora create explicit skips.

At the end of initial experience, preserve the model, complete optimizer,
accepted corpus, actual step count, and physical training costs. This common
learned state is checkpoint **0**, distinct from random initialization.

Four arms:

1. **Uniform:** unchanged uniform primitive prior, with no experience or
   training required for a standalone system.
2. **Frozen:** the common local prior after initial experience; no subsequent
   updates or memory additions.
3. **Replay:** clone the common model and optimizer separately for each stream
   order. After every stream attempt, take 16 Adam steps on the original
   initial corpus only. Neither accepted stream programs nor failures enter
   its learning data. This controls for further optimization and attempt count.
4. **Online:** clone the same common model and optimizer separately for each
   order. Search each stream task using the current prior. Append its first
   accepted nonempty program to the replay corpus, regardless of hidden
   correctness, then take 16 Adam steps on the entire accumulated corpus.
   After an exhausted or empty accepted search, add no target but still
   replay the available corpus. Empty corpora create explicit skips.

The stream search result is journaled before any update or hidden evaluation.
Hidden scoring cannot filter accepted programs. Save model and Adam states
for replay and online after stream attempts **8** and **16**, plus the common
checkpoint 0 and the random initialization. Optimizers continue from the
initial learned state, not newly reset Adam states. No branch shares mutable
weights, optimizer tensors, or corpus lists with another.

Frozen/uniform stream searches do not change any state. Their deterministic
outcomes may be reused between the two orders with reuse made explicit, but
not represented as separately observed or separately paid work. Both learned
branches actually search each task in each order. Report prequential stream
success before each task's update, never in-sample post-update success.

This is positive-solution learning: failures cause further replay but do not
provide a failure-specific gradient. A failed search obtains no corrective
oracle program or rescue search. Additional accepted experience is endogenous
to each branch's current prior. The interpretation must retain these limits.

## Search, local probability tables, and evaluation

Reuse Study I's finite ranked primitive search unchanged: lengths 0–9,
uniform length prior, smoothing `0.9 p + 0.025`, integer edge costs
`round(-log(probability) * 1,000,000)`, and lexicographic ties. Check visible
examples until the first mismatch and accept the first complete fit. Charge
every primitive application, accepting a complete fit exactly at 65,536 but
not an interrupted fit. No semantic deduplication or pruning occurs.

The supplied local restriction permits exactly **21** distinct prediction
contexts: the empty prefix, four one-token prefixes, and 16 two-token
prefixes. Construct the table once immediately before a state first supplies
a search, including the final state used for probing. Compute the 21 original
local predictions individually, with `no_grad`, and store their four probabilities.
Do not construct tables for intermediate Adam states within an update batch.
The ranker's callback directly looks up the last up-to-two tokens. This
explicit finite table replaces repeated identical neural calls, not the
probabilities or candidate order. Test exact agreement with direct prediction
and charge table construction, 21 neural calls, and 57 recurrent positions
(1 + 4*2 + 16*3) separately from logical ranker queries. No new predictor or
compatibility class is needed. A table is 84 float64 values (672 numeric
bytes), excluding keys/serialization; it is part of system storage.

Build only the candidate prefix whose cumulative primitive lengths first
exceed 65,536, as in I/J. Save full candidate orders and their probability
tables for every state used. Frozen, replay, and online share the common
checkpoint-0 neural table/order; uniform uses its separate uniform order.
Beyond that point, build separately for each branch/order and
completed update batch, even when replay states happen to be equal across
orders; do not add cross-branch equality detection or caching. If a batch
skips every update because its corpus is empty, reuse that branch's last
table/order. Frozen/uniform always reuse their existing order. Delayed probes
reuse checkpoint 8's already archived table/order and build checkpoint 16's
table/order once if its state has not yet supplied a search. No probe rebuilds
an existing table/order. Log actual builds, reused references, predictions,
heap sizes, logical prefix positions, file bytes, and phase times; do not
charge a neural call for a table lookup.

All stream learning in both orders finishes before any held-out probe search.
Then evaluate uniform, frozen, and both learned branches at checkpoints 0,
8, and 16 on the same 16 untouched held-out tasks. Identical checkpoint-0
neural states share a single actually evaluated frozen result, with explicit
references in summaries, and uniform is also evaluated once. No held-out
result changes a model, corpus, search rule, or checkpoint choice. Repeated
probes are not independent observations; the environment is the inferential
unit. Preserve every score and report both families at every checkpoint.

Exact success requires equality on all 16 states, independently checked from
the accepted program after the original result is saved. Exhaustion is zero.
Record held-out accuracy, false acceptance, terminal partial candidates,
comparisons, lengths, and success at 4,096/16,384/65,536 executed operations.
Stream outcomes and held-out outcomes must be unambiguously distinguishable.

## Costs and interpretation

Keep both actual collection totals and reconstructed standalone phase sums.
The standalone contract is **final held-out predictions only**, not serving
the stream as an additional output. Uniform needs only its final order and
probe searches. Frozen additionally pays for initial experience and initial
learning. Replay additionally pays for its 256 scheduled extra Adam steps
(or recorded skips) on the initial corpus, but its stream searches and
intermediate tables/orders are diagnostic costs: their outcomes never affect
its final state. Online additionally requires its stream searches, their
tables/orders, and updates to obtain its final state. Each arm pays for its
final table/order as applicable and final probe searches. Charge a reused
required artifact once. Report each stream order separately before any mean;
duplicated frozen/uniform descriptions do not multiply actual work.

All actual stream searches, intermediate probes, checkpoint I/O, and other
research work remain in collection totals even when not required for this
final-only standalone contract. State exactly which work is amortized or
shared; shared-phase sums are not independent wall-time measurements.

Record real target tokens, padded rows/positions, gradient steps, training
seconds, primitive executions, hidden evaluator executions, 21-context table
construction, ranker queries, ordering/storage/heap work, and total time.
Online's growing corpus and replay's fixed corpus imply different work even
when update counts agree. Sequential stream scores consume computation and
provide data; final-task execution caps do not equate lifetime resources.

Report final online-minus-frozen, online-minus-replay, and online-minus-uniform
balanced scores, each family, each order, intermediate checkpoints, and changes
from common checkpoint 0. The first is the sole confirmatory contrast. For a
family encountered first, later performance on its untouched held-out tasks
can describe retention or interference across the switch. It does not measure
lifelong retention, general intelligence, or knowledge across environments.

The model, local restriction, grammar, ranker, verifier, replay rule, and
favorable motifs are supplied. This is a short, two-phase learning test in a
finite domain. No result can establish feedback sufficiency or claim that
model parameters represent total system capacity. A failure would concern
this update/search rule and horizon, not all small agents.

## Freeze, verification, and failure handling

1. Independently review and commit this protocol before official task generation.
2. Test on hand-built fixtures: disjoint/mixed-length roles, cloning of all
   model/Adam state, local table equivalence, exact orders/budget boundaries,
   accepted-solution membership including hidden-wrong/empty cases, persistent
   updates after failures, replay control exclusion of stream solutions, and
   isolation from family/hidden/probe information. Verify cost accounting
   against actual model calls and all completed phases.
3. Freeze reviewed source, then generate and hash the complete task grid.
   Run development once. The auditor independently regenerates tasks and
   initial tensors, reconstructs every corpus from the appropriate visible
   search outcomes, replays every Adam update, reconstructs local probabilities
   with NumPy GRU equations, and verifies orders/searches/full functions/costs.
4. Require a passing development audit with the exact same source/data/runtime
   before the unchanged complete formal grid. Audit before inference, verify
   the full transferred file manifest, and independently reconstruct paired
   environment means and both primary intervals from raw records.

Journal each completed phase, search, and update with immediate flush; retain
active-phase metadata and the original exception on failure. Preserve all
partial/failed attempts. An infrastructure retry uses a fresh directory and
unchanged complete grid, never a favorable subset. An implementation defect
requires an explicit amendment and new source freeze; affected evidence stays
archived and cannot silently enter confirmatory results. No outcome-driven
seed replacement, setting change, early stopping, or selective reporting.
