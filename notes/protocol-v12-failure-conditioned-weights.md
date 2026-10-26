# Study L: weight updates from rejected programs during a task

Status: prospective after Study K and manuscript v14, before implementation,
official task generation, or outcomes. Study K updates weights between tasks
using accepted programs; failure supplies no new target. This study tests
whether verified failures can usefully change weights during a new task.

## Question and inferential unit

Does removing probability mass from visibly rejected programs outperform a
feedback-independent uniformizing update when both start from the same small
learned prior and use the same update opportunities? The sole primary is
final exact success of `rejection` minus `uniformizing`, equally averaged
across composition and shifted tasks within each of 32 environments, at
65,536 executed primitive operations per task.

Report a two-sided paired t95 interval and a 10,000-resample paired percentile
bootstrap with seed **977**. A positive primary requires the t lower bound
above zero; disclose bootstrap disagreement or degenerate intervals. Frozen
and uniform comparators, each family, all intermediate budgets, and update
trajectories are descriptive. A positive contrast against uniformizing alone
does not establish improvement over the learned starting point or uniform
search. A complete audited grid defines execution success regardless of result.

## Domain and task boundaries

Reuse the unchanged Study H generator with development seed **41** and formal
seeds **14000--14031**. Freeze all environments and their digest before any
official outcome. Keep its actual experience count, full-function uniqueness,
four primitive permutations on 16 states, and eight visible/eight hidden
examples per task. Use all 16 composition and all 16 shifted transfer tasks;
there is no new stream/probe subdivision. Retain the original task order.

Primitive tables, the legal program grammar, the verifier, and local-context
restriction are supplied. Prediction is not task-conditioned by an input
encoder. Instead, a separate copy of the same learned prior adapts to the
current task's rejection history through weights. Reset weights and the new
feedback optimizer for every transfer task. A program rejected for one goal
may solve another, so rejection records never become global negative targets.
This tests within-task adaptation and generalization to that task's eight
hidden outputs, not retained learning across new tasks or lifelong capability.
Hidden outputs, generating programs, task-family labels, and latent motifs
never enter the policy, rejection loss, optimizer, or stopping decision.

## Common learned initialization

Use unchanged `ProgramPrior(16)` with 1,172 parameters, CPU float64, one thread,
deterministic algorithms, and PyTorch **2.8.0+cu126**. Initialize using
`rng(seed, 'study_l:initial').getrandbits(63)` and `torch.manual_seed`.
Acquire initial experience by the unchanged H primitive-only search with
65,536 operations per task. Append every accepted nonempty program, regardless
of hidden correctness, and take 16 unchanged J local next-token replay steps
after each initial attempt. Preserve duplicates and full Adam state. Empty
corpora yield explicit skips. The initial optimizer is Adam lr .01, betas
(.9,.999), eps 1e-8, no decay, amsgrad/maximize/foreach/fused false, with norm
clipping at 1. Checkpoint the random model, every completed initial update
batch, and the common learned model/optimizer/corpus.

## Four transfer arms

1. **Uniform:** fixed uniform primitive probabilities, no model or experience.
2. **Frozen:** common learned local model, with no transfer updates.
3. **Uniformizing:** task-specific copy of the common learned model, updated
   toward uniform next-token probabilities at the same operation checkpoints
   as rejection. Its loss contains no rejected-program identities.
4. **Rejection:** task-specific copy, trained to reduce total probability
   mass assigned to the cumulative set of visibly rejected programs.

Both adaptive arms get a fresh Adam optimizer with the same settings as above;
experience Adam moments are not carried into these different objectives.
Updates occur at exactly 4,096, 8,192, ..., 61,440 executed primitive operations
when the task is still unresolved. Each opportunity has 16 gradient steps.
There is no update after acceptance or at the terminal 65,536 cap. Early
acceptance therefore changes realized update counts; matched opportunities
do not imply matched compute or matched steps between arms. An empty rejected
set causes an explicit rejection-update skip; uniformizing still follows its
scheduled objective. Preserve all completed step losses, norms, times, model
and optimizer checkpoints, and the parameter-change norm per update batch.

## Losses and local probabilities

Contexts are (), (0),...,(3), (0,0),...,(3,3), in length/lexicographic order.
For each context the GRU receives START=4 followed by that context, starts at
zero state, and uses the last real output. Compute the 21 rows as one padded
21-by-3 batch and select their last real positions. Thus each table requires
one neural call and 63 physical recurrent input positions, including padding.
Archive all 84 probabilities. Use this same batch convention for search and
both losses; it is a declared implementation change from K's 21 separate calls.

Let p_theta(a|c) be the raw softmax and s_theta(a|c)=.9 p_theta(a|c)+.025,
the same smoothing as the ranked search. The finite normalized program prior is

    q_theta(program) = (1/10) product_t s_theta(a_t | last_two_tokens),

with lengths zero through nine equally weighted. The empty program has mass
0.1 and no trainable factors. For the cumulative unique rejected set R,

    L_rejection = -log(1 - sum_{program in R} q_theta(program)).

This is the negative log probability of avoiding the observed rejected set,
not a probability of solving the task. It penalizes only completed visible
mismatches. It neither receives a correct program nor identifies all other
programs as correct. Use all accumulated rejections for all 16 steps in a
batch, without sampling or deduplication by semantic equivalence. The search
never proposes the same rejected string twice; the loss can only help by
changing the order of still-unchecked strings. Store the rejection-mass and
unclipped gradient norm before each step.

Uniformizing minimizes the cross-entropy of a uniform target against the
smoothed token probabilities, equally averaged over all 21 contexts and four
tokens: `-mean(log(s_theta))`. This is a control for flattening a specialized
prior, not a matched-information or matched-loss-scale intervention. Adam and
clipping are identical, but gradient norms and physical work can differ.

## Search and partial checks

Keep the I grammar (lengths 0--9), rounded edge costs
`round(-log(s_theta)*1_000_000)`, and lexicographic ties. The length prior is
constant and does not affect ranking. Expand prefixes even when they are in
the rejected set: excluding a complete string must not prune its descendants.
Exclude only already rejected complete strings from subsequent candidates.

Check the eight visible examples in their stored order, stopping at the first
mismatch and accepting only a complete fit. Charge every primitive execution.
At an update boundary, retain an interrupted candidate's program, example
index, operation index, and intermediate state. Continue its check first after
updating, without repeating any execution; its identity was chosen before the
update. Only after it completes does the next candidate use the new ranking.
If the last required primitive lands exactly on a boundary, perform the free
equality check before deciding whether an update is needed. A complete fit at
the terminal cap is accepted. An unfinished fit at that cap is exhaustion.
A partial candidate is not a rejection and cannot enter the loss.

At each adaptive round, rank only enough unchecked candidates that their
summed lengths first exceed the remaining operation allowance for that round.
Exclude the pending candidate from that list but keep expanding its prefix.
The pending candidate is continued separately. This is a safe lower bound on
check cost; empty strings and free comparisons are handled explicitly. Rebuild
one table/order only when the model changed, but each round can construct its
needed remaining order with the updated exclusion set. Record actual ranking,
heap, query, and storage costs. Uniform/frozen can use one full-cap static
order per environment, shared across tasks, since neither changes its weights.
Their results must exactly match unchanged I search on the same tasks/order.

Archive every candidate outcome including its first visible mismatch, every
partial boundary state, all selected orders and tables, all updates, and all
checkpoints. Save a complete accepted/exhausted result before hidden scoring.
Exact success requires equality on all 16 states from independent evaluation;
record hidden accuracy and false acceptance separately. Report success at
4,096, 16,384, and 65,536 from the actually observed learning path, without
claiming lower-budget retraining or selected early stopping.

## Resource accounting and interpretation

Report actual collection work and standalone full-task service costs per arm.
Uniform needs its shared order and task searches. Frozen additionally needs
initial experience and its replay. Each adaptive arm also needs its own task
clones, optimizer setup, tables/orders, feedback updates, and searches. Shared
initial resources are charged once per standalone arm/environment, not once
per task. No arm pays for the other arm's tasks or hidden evaluation. Actual
collection retains all research overhead and hidden scoring. Report explicit
artifact/checkpoint I/O separately, including any journal flushes inside timed
phases. Phase sums are shared/instrumented measurements, not independent
wall-time benchmarks or equal lifetime FLOPs.

Track initial target tokens/padded positions, feedback steps, rejected strings
and tokens processed, physical model rows/positions, table/ranker work,
primitive executions, comparisons, optimizer and checkpoint bytes, and phase
times. New rank orders and stored rejections are external memory; 1,172 neural
parameters do not describe total system capacity. All arms receive exact,
cheap synthetic feedback and supplied primitives/search. A positive result
would establish only this task-specific failure-loss intervention. A null or
negative result would not refute other online update mechanisms.

## Freeze and audit

1. Independently review and commit this protocol before task generation.
2. Implement/test with hand-built fixtures, including finite normalized mass,
   independent analytical/finite-difference gradient checks, actual weight
   changes, task reset, excluded-prefix descendants, boundary continuations,
   acceptance exactly at cap, partial-candidate exclusion, no duplicate
   proposals, control independence, and rejection of corrupted evidence.
3. Independently review and freeze the complete implementation before official
   task generation. Generate/hash the complete grid once. Run development
   once; require a passing audit with matching source/data/runtime before the
   unchanged formal grid. No tuning from development outcomes.
4. Audit task/init regeneration, every learned tensor/Adam state by independent
   update replay, probabilities by NumPy GRU equations, search orders and
   execution/mismatch histories, all result/cost counts, and complete files.
   Refuse optimized Python if assertions implement checks. Reconstruct the
   full primary inference independently from raw outcomes after transfer.

Journal completed operations with flush and retain active-phase metadata plus
the original exception on failure. Preserve partial/failed attempts. Retries
use fresh paths and unchanged complete grids. An implementation defect needs
an explicit amendment and new freeze; affected evidence is retained and
cannot silently become confirmatory data. No outcome-driven task replacement,
stopping, hyperparameter changes, or best-checkpoint selection is allowed.
