# Study J: restricting a neural program prior to local context

Status: prospective design, after Study I and its explicitly post-hoc
likelihood diagnostics, before new task generation or outcome collection.
The diagnostics show good fitting of accepted experience alongside poor
transfer. They do not identify a cause. This study intervenes on how a fixed
small network receives program history, using fresh environments.

## Question and primary comparison

Does resetting a neural predictor to the last two primitive tokens improve
composition search relative to retaining the complete prefix, with the same
neural tensors, initial values, experience, targets, and update schedule?
The single primary endpoint is final `local_trained` minus `full_trained`
exact composition success, paired over 32 environment means. A positive
prediction requires a two-sided 95% t interval whose lower bound exceeds zero.
A 10,000-resample paired percentile bootstrap, seed 953, is a sensitivity
check; report disagreement and degenerate intervals. Completion and valid
inference, not a favorable outcome, are the research success criterion.

Uniform, counts, and frozen controls are essential descriptive comparators.
An advantage over full-prefix learning alone does not establish an advantage
over uniform search or useful learning relative to a frozen local predictor.
Report both of these additional differences, shifted and balanced scores,
and all intermediate checkpoints without changing the primary comparator.
No general claim that shorter context, fewer effective states, or small neural
models are superior follows from this finite intervention.

## Unchanged domain with new environments

Use the unchanged Study H generator from c582467, as used in Study I, with
development seed **31** and formal seeds **12000–12031**. Freeze all 33
environments and their digest before official search. Four primitive
permutations act on 16 states. Experience comprises up to 24 length-five
programs containing hidden length-three motifs and primitive wrappers;
transfer has 16 motif compositions and 16 shifted uniform programs, eight
at each length six and nine. Full-function split uniqueness and the fixed
generator's rejections remain unchanged. Do not require 24 tasks by resampling
an environment: use its generated attempt count and record any omissions.

Each task reveals eight ordered input/output examples and holds out eight.
The learner receives only accepted experience programs. The searcher receives
the public primitive tables and visible examples. Latent motifs, generating
programs, hidden labels, and transfer scores never guide learning or acceptance.
All accepted nonempty experience programs enter replay even if hidden
evaluation later marks them wrong. Retain every failed or false accepted task.
Use primitive-only breadth-first search, first complete visible fit, and
65,536 primitive applications per experience task, exactly as in H/I.

## Matched model tensors and context intervention

Both neural learners use the unmodified Study I `ProgramPrior(16)`: five-way
one-hot inputs, one 16-unit GRU layer, and a four-output linear head, with
1,172 trainable parameters including every bias. Initialize once per
environment, directly in CPU float64 with PyTorch 2.8.0+cu126. Derive the seed
as Study H `rng(seed, 'study_j:initial').getrandbits(63)`, call
`torch.manual_seed`, construct the GRU followed by the linear head using the
same default initializers as protocol v9, and clone its tensors for both
learners and both frozen controls. No pretraining or extra embeddings occur.

- **Full prefix:** to predict after prefix p, start from zero recurrent state,
  process `[START, *p]`, and use the last output. For training, one sequence
  `[START, *program[:-1]]` supplies the logits for all next-token targets.
- **Local context:** to predict after p, start from zero recurrent state,
  process `[START, *p[-2:]]`, and use the last output. For training, create one
  separate row for each target position i, containing
  `[START, *program[max(0, i-2):i]]`; take its last real output. Pad rows on the
  right with token zero to their batch maximum length, and gather only the
  last real output for cross-entropy. Padding never supplies a target.

The prediction paths agree for prefixes of length at most two when weights
are equal. For longer prefixes, the local path cannot distinguish histories
sharing the last two tokens. START is still processed once per prediction;
it is not repeated as left padding. This differs from learning an unrestricted
recurrent history despite matching parameter counts. It changes context,
state reset, gradient paths, and actual computation together; it does not
isolate absolute position from content or prove the cause of Study I's failure.

After each attempted experience task, both persistent optimizers take 16
full-batch updates on the same accumulated accepted corpus. Minimize raw
next-token cross-entropy averaged over all real target tokens. Use Adam
lr .01, betas (.9, .999), eps 1e-8, weight decay zero, amsgrad false, maximize
false, foreach false, fused false, gradient norm clipping at 1, CPU float64,
one PyTorch thread, and deterministic algorithms. Retain optimizer state.
Empty corpora yield explicit skip records. No replay reweighting, target
smoothing in the training loss, early stopping, checkpoint selection, or
hyperparameter tuning occurs. Matched update and target-token counts do not
mean matched FLOPs: local rows recompute their short contexts for each target.

## Arms, checkpoints, and search

Six final arms:

1. `uniform`: next-token probability 1/4, no experience acquisition.
2. `counts`: Study I add-one counts with two previous tokens, using the same
   accepted programs; 100 stored integer cells and 21 reachable contexts.
3. `full_frozen`: the shared initial network, full-prefix prediction.
4. `local_frozen`: the same initial tensors, local-context prediction.
5. `full_trained`: the full-prefix learner after all attempted experience.
6. `local_trained`: the local-context learner after all attempted experience.

Save both trained models with their optimizers after attempts 8, 16, and the
last attempt, plus the shared initial state. Evaluate both intermediate
checkpoints as descriptive probes, even if they are unfavorable. If the fixed
generator ever supplies fewer than 16 attempts, stop before collection and
record the design incompatibility; do not substitute seeds. Complete all
experience and updates before any transfer evaluation. Transfer never updates
weights or selects a prior.

Reuse Study I's exact ranked finite enumeration and checker: lengths 0–9,
uniform length prior, conditional smoothing `0.9 p + 0.025`, summed edge
cost `round(-log(probability) * 1,000,000)`, lexicographic ties, and positive
cost heap expansion. The local neural callback truncates the prefix according
to the rule above; no semantic caching, duplicate removal, beam pruning, or
task-dependent reranking is added. The exact count implementation is unchanged.

Retain the ranked candidate prefix until cumulative primitive lengths exceed
65,536; share it across the 32 transfer tasks per arm/environment. Check fixed
visible examples until the first mismatch and accept the first complete fit.
Every actual primitive application is charged, with a per-task cap of 65,536;
a complete fit exactly at the cap is accepted, an interrupted fit is not.
Exact success requires equality on all 16 states, scored only after the search
result is journaled. Exhaustion scores zero. Also retain held-out accuracy,
false acceptance, counts, comparisons, and the terminal partial candidate.

## Costs and evidence

Journal every completed search before hidden evaluation, every gradient step
before the next step, and every completed resource phase with immediate flush.
Keep the active-phase record and original exception if a runtime failure occurs;
do not parse a possibly truncated active record while handling that error.
Save losses, pre-clipping norms, actual input rows/positions including padding,
target tokens, optimizer/model tensors, accepted corpus, counts, full orders,
file sizes, and elapsed phase/collection times. Regenerate initial states and
replay all optimizer updates in the audit.

The ranker's original `prefix_positions` is the logical full-prefix count.
For neural arms, additionally record actual recurrent input positions: sum
`1 + len(prefix)` for full calls or `1 + min(2, len(prefix))` for local calls.
No probability table cache avoids repeated neural calls in this comparison.
Training similarly reports padded physical input positions and real targets
separately. Retain stored tensor bytes, count cells, and maximum heap entries;
none alone measures total system capacity or peak RAM.

Experience acquisition is shared in actual collection but charged to every
standalone trained/count arm. Frozen and uniform arms need no experience.
Charge each learner's own training and ranking; ranking is amortized across
32 transfer tasks. Probes add research cost and use only their available
experience prefix. Separate actual collection totals, held-out evaluation,
checkpoint overhead, and hypothetical standalone phase sums. Shared-phase
sums are not independent wall-time trials or matched lifetime compute.

Report all seeds, arms, checkpoints, both families, the balanced mixture,
and 4,096/16,384/65,536-operation descriptions from the first accepted fit's
actual cost. The complete formal grid must pass audit before primary inference.
Do not pool H/I seeds into this study or compare their outcomes as paired data.
There are no additional planned likelihood or motif-recovery endpoints here.

## Verification, failures, and freeze sequence

1. Independently review and commit this protocol before task generation.
2. Verify local history invariance, agreement on short prefixes, causal
   target construction, exclusion of padding, matching tensors, persistent
   optimizers, and absence of hidden-label inputs on hand-built fixtures.
   Reuse the verified original search functions; test their local callback
   against independently enumerated finite orders and boundary cases.
3. Freeze source and the complete task grid, then run development once.
   Independent audit regenerates tasks, initial tensors, corpus membership,
   every update and saved optimizer, local/full NumPy probabilities, ranked
   orders, full-function execution costs, hidden scores, and complete cost maps.
   The local NumPy path receives only the last two primitive tokens.
4. Formal collection requires a passing development audit for exactly the
   same source/data and settings. No unfavorable development result can
   trigger tuning, selection, or omission. Run the complete formal grid once,
   audit it, preserve raw records, and independently reconstruct the primary
   paired differences and intervals before reporting.

Runtime/initialization failures produce an incomplete attempt, not zero-success
agents; incomplete grids support no primary conclusion. Retain all records and
record the costs of completed work. An infrastructure retry uses a new attempt
directory and the entire unchanged grid; do not retry selected environments.
An implementation defect requires a documented amendment and new source freeze,
with affected evidence retained but excluded from confirmatory interpretation.

This study tests an explicit, supplied context restriction within one finite
experience-to-transfer cycle. It does not learn the restriction, replace the
searcher, create task-conditioned reasoning, accumulate skills across changing
environments, or test whether feedback suffices for general intelligence.
