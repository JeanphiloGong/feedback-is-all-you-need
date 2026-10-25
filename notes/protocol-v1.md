# Study A protocol: capacity and online feedback

Status: frozen and amended; see dated amendment below for the current seed set.
This is a repository-timestamped protocol, not an externally registered study.
Changes after the freeze must be dated and justified in the research log.

## Question and scope

When a prediction environment changes, how do feedback quality, delay, and
model capacity affect adaptation? This controlled study tests necessary parts
of the broad hypothesis. It does not test general intelligence or an LLM.

The agent observes an eight-dimensional vector, predicts a binary action, and
receives the correct binary label as environmental feedback after prediction.
In this binary setting, correctness feedback together with the action conveys
the same label. Learning is supervised online adaptation, not a new RL method.

## Data-generating process

- Independent experimental units: 20 seeds, 4000 through 4019. Development and
  correctness checks use seeds below 100. Seed outcomes must not tune settings.
  Amendment before formal collection: an implementation smoke run accidentally
  used seeds 1000 and 1001 and its outcomes were viewed; these are now excluded
  from formal analysis. See the research log. Model settings were not tuned.
- Features: independent standard-normal coordinates, dimension 8.
- Linear family: `y = 1[x dot w >= 0]`, with a normalized random Gaussian `w`.
- XOR family: `y = 1[(x[i] > 0) != (x[j] > 0)]`, with a uniformly sampled
  distinct pair of coordinates. Independently for each rule draw
  `c ~ Bernoulli(0.5)` and complement all labels exactly when `c=1`.
- Each seed has an initial rule and an independently sampled post-change rule.
  Identical XOR rules may occur naturally and are retained, not resampled.
- Warmup: 256 mini-batches of 32 labeled examples from the initial rule.
- Adaptation: 256 new mini-batches of 32 examples from the new rule.
- Evaluation: 2048 fresh examples for each rule, generated with separate RNG
  streams. Fixed evaluation examples are never used in updates or selection.
- A second 2048-example set per rule, drawn from independent uniform
  `[-sqrt(3), sqrt(3)]` coordinates, measures distribution transfer. This
  preserves each rule but changes the input distribution, not task family.

## Models and optimizer

- Logistic regression (width 0) and one-hidden-layer ReLU networks with widths
  16, 64, and 256; one scalar output with binary cross-entropy.
- Full counts: 9, 161, 641, and 2561 trainable parameters respectively.
- Adam, learning rate 0.01, betas (0.9, 0.999), epsilon 1e-8, no weight decay.
- Initial weights use PyTorch Linear defaults; independent initialization per
  seed. Each capacity/seed uses the identical warmed-up state in all arms.
- Reset optimizer moments after warmup in every adapting arm. No arm receives
  tuning on evaluation seeds. All arms receive the same ordered input stream.

## Arms

1. `frozen`: warmup weights remain fixed; no post-change update.
2. `clean`: one update per arriving batch, with correct labels.
3. `noise25`: independent label flips with probability 0.25.
4. `noise50`: independent label flips with probability 0.50.
5. `inverted`: every feedback label is flipped, unknown to the learner.
6. `delay32`: correct labels arrive 32 batches after prediction. Evaluate both
   the online endpoint and a separately labeled endpoint after flushing pending
   batches, so feedback latency is not confused with final information count.
   Checkpoint numbers count arriving batches, not completed updates: after
   arrival t, this arm has max(0, t-32) updates. At t=256 it has 224 updates;
   flushing processes the remaining 32 in order, reaching 256 total updates.
   Equality with clean after flushing is an implementation check because the
   update rule has no dependence on arrival time and uses the same batch order.
7. `replay25`: each update uses 24 new examples plus 8 randomly selected
   examples retained from warmup; clean post-change labels. This reduces fresh
   examples per update and therefore is not a pure memory ablation. Report
   fresh and replay example counts separately.
   Use the first 24 examples of each common 32-example batch and discard the
   other 8 for this arm. Draw 8 warmup examples with replacement independently
   per update from all 8192 stored examples. Retain their old-rule labels.

No checkpoints are selected using evaluation scores. Checkpoints are fixed at
0, 16, 32, ..., 256 adaptation steps. Save per-seed accuracy for both rules and
both distributions. Old-rule performance is a forgetting diagnostic; with
unannounced conflicting rules, perfect retention and perfect adaptation need
not be simultaneously feasible.

## Endpoints and comparisons

Primary endpoint: post-change held-out Gaussian accuracy after 256 batches.
Primary contrast in each task family: width-16 clean minus width-256 frozen.
This intentionally unequal-information comparison is a positive control for
adaptation, not proof that feedback is more compute-efficient than scaling.

Required secondary comparisons:

- Same width: clean vs frozen, noise25, noise50, inverted, and delay32.
- Same feedback stream: all widths under clean feedback.
- Capacity boundary: linear vs nonlinear models on XOR.
- Old-rule retention and uniform-distribution transfer.
- Learning curves and accuracy at a matched adaptation arithmetic budget,
  using the last observed checkpoint not exceeding the small model's full
  budget. If this gives too coarse a match, mark it inconclusive and record a
  prospective finer-budget follow-up rather than interpolate an unmeasured run.

For a dense network with W matrix weights, estimate update arithmetic as
`6 * W * number_of_training_examples` FLOPs. Report biases/nonlinearities,
optimizer operations, and hardware overhead as excluded; this is a proxy,
not measured total compute. Also report real wall time, parameter count,
warmup cost, delivered examples, update count, and replay storage. Evaluation
cost is recorded separately and not treated as useful adaptation compute.
No claim of equal lifetime compute is permitted from this proxy.

## Statistics, reporting, and stopping

- Report each seed, mean, standard deviation, and a 95% t interval for means
  and paired differences. The seed is the replication unit, not a checkpoint
  or individual example. Interval assumptions must be stated.
- Treat the two primary family contrasts as co-primary; report 97.5% intervals
  for a Bonferroni familywise 95% confidence interpretation.
- Secondary intervals are descriptive; no search for a favorable p-value.
- Run the full fixed grid once; no optional stopping based on effect size.
- Save full configurations, package versions, source SHA-256, Git revision,
  hardware, wall times, all seed outcomes, and failed job records.
- Rerun only for identified defects or infrastructure failures; preserve failed
  evidence and explain which results are superseded.

## Falsification and subsequent studies

The broad sufficiency claim is weakened by uninformative feedback, capacity
limits, non-transfer, or catastrophic forgetting. A positive result on these
toy tasks only supports a conditional adaptation claim. Stronger tests remain
required: pretrained language models, new task families, persistent knowledge,
equal-information offline controls, equal-inference-budget controls, and a
capacity sweep with informative feedback supplied to large models as well.
Study B will freeze its own protocol before scored language-model runs.

## Amendment: 2026-10-25 UTC, after implementation audit

The first full run (2000–2019, archived as study-a-v1) used additive seed
offsets. This caused overlapping corruption masks across environments and
some initialization/task stream collisions. Treat its intervals as superseded;
keep all raw outcomes and the original implementation in Git history. The
replication uses untouched seeds 4000–4019 and derives each PRNG seed from
SHA-256 of `(seed, family, role, details)`, taking the first eight bytes as an
unsigned little-endian integer. Test for stream collisions over the complete
grid. Feedback arms deliberately share within-environment data and noise
draws for paired comparisons. No model, optimizer, endpoint, sample size,
or feedback schedule changes. Replication results will replace primary tables.
