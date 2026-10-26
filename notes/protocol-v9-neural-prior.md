# Study I: learning a small neural search prior from checked experience

Status: prospective design, before development or formal task generation and
outcomes. This study follows Study H and the DreamCoder source audit. It tests
whether verified solutions can teach a small, initially random network a useful
distribution over programs. It does not replicate DreamCoder, condition the
network on task examples, or test general intelligence. A supplied symbolic
searcher obtains the experience and verifies every proposed solution.

## Question and success criterion

Does a learned 1,172-parameter program prior improve exact recovery on new
compositions over uniform program search at the same primitive-execution cap?
Primary: final small-network minus uniform exact success on composition tasks,
paired over 32 independently generated environments. The prediction is positive,
with a two-sided 95% t interval whose lower bound exceeds zero. A failure to
meet that criterion remains an inconclusive or contrary result for this setup.
Successful research means completing and auditing the declared comparison,
regardless of its sign; it does not mean obtaining a positive effect.

A count-based prior tests whether any benefit needs a neural implementation.
A frozen initial network separates learning from an arbitrary neural ordering.
Training on shuffled solution tokens tests dependence on observed sequential
structure. A larger network and intermediate checkpoints describe capacity and
learning progression. Shifted tasks test the cost of a specialized prior.
These comparisons are descriptive, not additional primary hypotheses.

## Domain and data boundary

Reuse the unmodified `experiments.study_h.tasks.make_environment` generator
from commit c582467 with **new** development seed 29 and formal seeds
11000–11031. Its four public primitive permutations act on 16 states. Each
environment has up to 24 experience programs, generated as a primitive prefix,
a hidden length-three motif, and a primitive suffix. There are 16 novel
compositions of two/three motifs and 16 shifted uniform primitive programs of
length six/nine. Full-function duplicates are excluded across splits. Eight
examples per task are visible; eight are held out. The supplied task structure
and semantic split restrictions remain deliberate inductive biases.

Archive all 33 environments and their hash before official search. Regenerate
the complete grid during audit; do not select seeds after viewing outcomes.
Study H's scores do not enter this study's estimates or paired comparisons.
Latent motifs, generating programs, held-out labels, and evaluation outcomes
never enter the prior, its optimizer, or search acceptance. The learner receives
only accepted experience programs; the searcher receives primitive tables and
visible examples. Final evaluation is a separate call after acceptance is fixed.

## Experience and sequential updates

Solve each experience task once with Study H's unchanged primitive-only
breadth-first search and 65,536 executed primitive operations. Accept the first
program fitting all eight visible examples, even if later hidden evaluation
finds it incorrect. Keep failures and false acceptances. All accepted nonempty
programs enter an ordered replay corpus; there are no latent-program targets.
The search routine, not the network, solves these experience tasks.

After each experience task, train on all accepted nonempty programs so far for
16 full-batch Adam steps. If the corpus is empty, skip those steps and record
that fact. Retain weights and Adam state between tasks. There is no reweighting,
minibatch sampling, early stopping, validation selection, or best-checkpoint
selection. Final, eight-task, and sixteen-task models are saved; an intermediate
checkpoint means that many attempted tasks, not that many successful solutions.
Complete all experience and updates before evaluating any transfer task.

The model predicts the next primitive from a prefix. Its input is a fixed
five-way one-hot vector (A–D plus START), with zero initial recurrent state,
one GRU layer, and a linear four-output head. To predict a program, feed START
followed by its previous tokens. Train next-token cross-entropy averaged over
all nonpadding primitive targets in the corpus. There is no EOS target or
learned length distribution. Padding contributes no loss.

Two hidden widths are fixed: H=16 and H=64. Count every trainable tensor and
bias: `3 H^2 + 25 H + 4`, giving **1,172** and **13,892** parameters. There is no
pretraining, embedding model, learned feature extractor, library of macros, or
neural task representation. Use CPU float64, one PyTorch thread, deterministic
algorithms, Adam lr 0.01, betas (0.9, 0.999), eps 1e-8, no weight decay, and
gradient norm clipping at 1.0. Use Adam with `amsgrad=False`, `maximize=False`,
`foreach=False`, and `fused=False`. Archive versions and actual tensor counts.

Derive seeds through Study H's SHA-256-based `rng(seed, role)`: draw 63 bits
from roles `study_i:small_initial` and `study_i:large_initial` for PyTorch
initialization. Small trained, frozen, and shuffled arms start with identical
weights. For the shuffled arm, independently permute each accepted program's
tokens once using role `study_i:shuffle:<experience index>`, preserving its
length and primitive counts. Store those controls and never reshuffle on replay.
They need not solve the experience tasks; disrupting that structure is the
intervention. This is not a matched-label-noise estimate.

For each independent model initialization, call `torch.manual_seed` with its
derived seed, then construct `nn.GRU(5, H, num_layers=1, bias=True,
batch_first=True, dropout=0, bidirectional=False, dtype=torch.float64)` followed
by `nn.Linear(H, 4, bias=True, dtype=torch.float64)`, both on CPU. Use the
recorded PyTorch version's default `reset_parameters` distributions: GRU tensors
uniform in [-1/sqrt(H), 1/sqrt(H)], and the linear head's default Kaiming-uniform
weights with `a=sqrt(5)` and its default uniform bias. Do not construct float32
parameters and convert them. Clone the resulting small initial state for its
three arms without drawing replacement tensors. Independently regenerate and
compare the saved initial tensors during audit.

## Six final arms and intermediate probes

- `uniform`: probability 1/4 for each primitive after every prefix.
- `counts`: next-token counts from the same accepted corpus, conditioned on
  the last two tokens, left-padded with START. Add one to each of four counts
  before normalization. Store a 5 by 5 by 4 integer table (21 reachable
  contexts); unseen contexts are uniform. No neural training is involved.
- `small_frozen`: the initial H=16 network, with no updates.
- `small_trained`: H=16 after all experience attempts and replay updates.
- `small_shuffled`: H=16 trained under the same schedule on shuffled programs.
- `large_trained`: H=64 after all experience attempts and replay updates.

Also evaluate `small_trained` after attempts eight and sixteen; the initial
probe is `small_frozen`. These probes cannot affect subsequent training.
All arms use the same visible examples, primitive vocabulary, acceptance rule,
maximum program length, and operation cap. No arm receives whole solutions or
compressed macros as callable primitives. The count table is a serious simpler
alternative, not a claim that neural guidance is necessary.

## Proposal order and execution budget

Supply a maximum primitive-program length of nine and a uniform prior over
lengths zero through nine. The network/count prior supplies token probabilities
only. Smooth each conditional probability as `0.9 p + 0.025`, including frozen
and count arms; uniform remains unchanged. All legal next tokens keep support.
The intended program mass is its token-probability product divided by ten.
The finite length bound and its distribution are supplied to every arm, not
inferred from evaluation targets or the hidden generating length.

For reproducible ordering, each edge costs
`round(-log(smoothed_probability) * 1,000,000)` integer units. Rank programs
by increasing summed edge cost, breaking ties lexicographically by primitive
sequence. The per-program rounding error is at most 4.5e-6 nats. Start with
the empty program in a minimum heap; after emitting a program of length less
than nine, append each of four primitives to construct its children. Positive
edge costs make this enumerate the complete finite space in ranked order.
There is no beam pruning, semantic cache, program deduplication, or target-aware
reranking. Uniform gives breadth-first lexicographic order.

The order is independent of the transfer task and can be constructed once per
environment and arm, before evaluation. Store the emitted programs. Stop order
construction once their cumulative primitive lengths **exceed** 65,536: checking
each nonempty candidate requires at least that many primitive operations on its
first example, so this prefix covers every permitted search. This is a lower
bound on checking cost, not the actual per-task charge. Count ranking work and
heap storage separately; it is not free model inference.

Check candidates in that stored order, visible examples in their fixed order,
and reject at the first mismatch. Charge every executed primitive application
to one example, with a cap of 65,536. An interrupted candidate cannot be
accepted; a full match completing exactly at the cap can be. The identity
program has zero primitive cost but its comparisons count. Stop at the first
complete visible fit; do not choose by hidden evaluation. Exhaustion emits no
answer and scores zero. Exact success means equality on all 16 states; also
report held-out accuracy and incorrect accepted programs.

## Resource and result accounting

Record experience and evaluation primitive operations, candidates, comparisons,
ranking nodes, conditional predictions, maximum heap entries, training steps,
replay target tokens, loss/gradient norms, parameter/storage bytes, checkpoint
sizes, corpus size, count-table cells, and measured phase times. Save initial
and learned model tensors plus optimizer states at the declared checkpoints.
No training or search FLOP equivalence is claimed from the operation cap.

Experience is shared during actual collection but charged to every standalone
trained or count arm. Uniform and frozen arms need no experience collection.
Charge model training to each relevant arm; count construction is separately
charged. Ranking is shared across an arm's 32 transfer tasks, and amortization
must be stated. Intermediate probes add actual research cost and must not be
hidden in the final deployed-arm estimates. Report actual total work separately
from hypothetical standalone totals, and do not add shared phases repeatedly
to a claim of measured wall time.

Primary paired t95 and 10,000 paired percentile-bootstrap draws use environment
means of 16 composition exact scores, bootstrap seed 947. Report disagreement
in positive or negative conclusions and degenerate intervals. Report every
environment, both families, the balanced combined score, all six final arms,
and both intermediate probes. Curves at 4,096, 16,384, and 65,536 operations use
the first accepted program's actual cost; they do not select better candidates.
There is one formal grid and no favorable seed or checkpoint selection.

Algorithmic budget or finite-space exhaustion is an observed zero-success
outcome. Initialization/runtime failure instead makes the attempt incomplete:
no primary interval or confirmatory conclusion is issued until the complete
32-pair grid passes audit. Do not drop failed pairs or score infrastructure
failures as unsuccessful agents. An infrastructure failure permits a new,
versioned attempt of the complete grid with identical frozen source, inputs,
seeds, settings, and regenerated initial states. Retain all prior attempt files
and their resource records, distinguishing completed work from any unmeasured
interrupted operation. Do not retry only unfavorable completed environments.
An implementation defect requires an explicit amendment identifying affected
records; those records remain archived but cannot support the confirmatory
analysis. Corrected collection uses a new source freeze and attempt directory.

## Verification and freeze sequence

1. Freeze this protocol after independent review, before generating data or
   seeing outcomes; preserve any subsequent amendment explicitly.
2. Verify counts, probability normalization, causal prefix inputs, padding,
   persistent updates, matched initialization, and no hidden-label path. Check
   heap order against exhaustive finite enumeration and compare search costs
   with an independent executor, including interruption/exact-cap cases.
3. Freeze source and data. Run the development seed once, then independently
   audit corpus construction, learned/shuffled data, training replay, model
   counts, candidate orders, first-fit selection, full labels, and budgets.
   Record all development outcomes without tuning to them.
4. Run the formal grid once only after the development audit. Flush completed
   searches before final evaluation and record initialization/runtime failures.
   Independently reproduce summaries from raw records and verify every archived
   file against its manifest before reporting or writing a new manuscript.

Reference code remains source material; it is not executed or imported. Studies
A–H and manuscripts through v10 remain historical evidence. This study tests
one pass of accepted-solution accumulation and transfer in a supplied finite
domain, not an autonomous curriculum, lifelong learning, or sufficiency of a
small model for unrestricted intelligence.
