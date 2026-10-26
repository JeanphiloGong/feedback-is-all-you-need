# Study K: continued local-prior learning

The [prospective protocol](../../notes/protocol-v11-continual-local-feedback.md)
tests continued learning from visibly accepted solutions with the same
1,172-parameter local GRU. Each environment has 24 initial experience tasks,
eight stream tasks per family, and eight untouched probes per family. Both
family orders start from identical model/Adam states. Online adds accepted
stream solutions; replay only revisits the initial corpus; frozen stops
updating; uniform requires no experience. No probe outcome guides learning.

The complete source/data freeze is
`a22026af5be6b77fdcb39ad33e717f296f422987`. The local context, primitive
grammar/search, verifier, and replay rule are supplied mechanisms. Update
counts and final search caps do not equate lifetime computation.

## Development record

Seed 37 completed once with 280 searches and 1,408 gradient steps. Collection
took 3.377 seconds; complete independent audit/analysis took 4.974 seconds.
All tensor/Adam states, corpora, probability tables, candidate orders,
full-function outcomes, journals, timing consistency, and physical input
records passed audit.

| Arm, final checkpoint | Composition exact / 16 conditions | Shifted exact / 16 conditions | Balanced exact / 32 conditions |
| --- | ---: | ---: | ---: |
| uniform | 8 | 8 | 16 |
| frozen | 10 | 0 | 10 |
| replay | 10 | 0 | 10 |
| online | 14 | 0 | 14 |

Each family has eight unique probe tasks evaluated under both stream orders.
Frozen/uniform actually run once; their outcomes are explicitly reused. The
two orders are dependent conditions, not independent samples. The descriptive
online-minus-frozen balanced difference is +12.50 points, with no confidence
interval or confirmatory inference from this single environment. Online
remains below uniform's balanced score and has no shifted success here.
All orders, probes, and checkpoints remain in the archive. No source, task,
seed, setting, budget, or checkpoint selection changed after these outcomes.
The formal grid started only after the matching development audit.

All 152 development raw/audit files, 6,507,233 bytes, match the execution-host
originals in [development-manifest.json](development-manifest.json); all 150
raw hashes match the original independent audit. All 18 source/data files
match both the host and Git freeze in [source-transfer.json](source-transfer.json).
The [implementation instructions](../../experiments/study_k/README.md) describe
reproduction and the 19 synthetic checks completed before official generation.

Before collection, source extraction used an incorrect directory layout and
one module launch used its parent directory. These startup errors occurred
before experiment execution; the corrected frozen checkout passed all 19
tests and the successful official development run was not repeated. The first
launch console was overwritten by the successful command's log; this note
preserves that event without presenting a reconstructed log as an original.

## Formal results

All 32 formal environments (13000–13031) completed with the same source,
data, settings, and archived matching development certificate. Each held-out
family has eight tasks per environment, evaluated under two dependent stream
orders. Thus a family has 256 unique tasks and 512 task/order conditions;
the balanced total has 512 unique tasks and 1,024 conditions. Frozen/uniform
results are measured once and reused. The environment is the inferential unit.

| Arm, final checkpoint | Composition exact / 512 conditions | % | Shifted exact / 512 conditions | % | Balanced exact / 1,024 conditions | % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 256 | 50.00 | 258 | 50.39 | 514 | 50.20 |
| frozen | 390 | 76.17 | 58 | 11.33 | 448 | 43.75 |
| replay | 388 | 75.78 | 54 | 10.55 | 442 | 43.16 |
| online | 473 | 92.38 | 55 | 10.74 | 528 | 51.56 |

The prespecified primary, **final online-minus-frozen balanced exact success**,
is **+7.81 percentage points**, paired over 32 environment means after averaging
both orders and families: t95 **[4.74, 10.88]**, bootstrap95 **[4.98, 10.84]**
(10,000 resamples, seed 967). Twenty-two environments improve, eight tie, and
two decline. The intervals agree and satisfy the protocol's positive criterion.

The extra-replay control scores 43.16% on the balanced mixture; the descriptive
online-minus-replay difference is +8.40 points (25 positive, five zero, two
negative environments). Thus simply running the additional fixed number of
updates on the original corpus does not account for the observed online gain.
However, online's growing corpus changes both training data and physical work;
this is not an equal-compute estimate of the effect of new feedback alone.

Against uniform search, online is only **+1.37 points** on the balanced mixture,
with 15 positive environments, seven ties, and ten negatives. This is a
secondary description, without a new confirmatory claim of superiority. The
primary contrasts against an already specialized learned prior, not uniform.

The gain is concentrated in compositions: online exceeds frozen by 16.21
points and replay by 16.60. Shifted success is 0.59 points below frozen and
0.20 above replay. Relative to uniform, the composition difference is +42.38
points, but the shifted difference is **−39.65 points**, negative in every
environment. Continuing accepted-solution learning does not demonstrate
adaptation to the shifted family under this mechanism.

## Both orders and every checkpoint

| Stream order | Arm | Attempts | Composition % | Shifted % | Balanced % |
| --- | --- | ---: | ---: | ---: | ---: |
| either | common learned state | 0 | 76.17 | 11.33 | 43.75 |
| either | replay | 8 | 75.39 | 10.16 | 42.77 |
| either | replay | 16 | 75.78 | 10.55 | 43.16 |
| composition then shifted | online | 8 | 92.19 | 10.55 | 51.37 |
| composition then shifted | online | 16 | 92.19 | 10.55 | 51.37 |
| shifted then composition | online | 8 | 76.17 | 10.16 | 43.16 |
| shifted then composition | online | 16 | 92.58 | 10.94 | 51.76 |

The [learning-progress figure](learning-progress.pdf) shows all checkpoints,
both orders, all 32 online environment trajectories, and all four mean
comparators. Checkpoint zero is after initial experience, not random weights.
Frozen/uniform stay unchanged; replay follows the same fixed-corpus updates
in each order. No point was selected as a stopping rule. Reproduce it with
`python3 -s results/study-k-v1/plot_progress.py`.

Composition-first online means are unchanged between attempts 8 and 16, but
equal aggregate scores alone do not show preservation of each solved task.
The stream search outcomes also remain separate from these untouched probes:
online accepts 220/256 composition and 23/256 shifted tasks in composition-first
order, and 219/256 composition and 25/256 shifted tasks in the reverse order.
Replay accepts 191/256 and 26/256, or 192/256 and 25/256, respectively.
Each stream outcome precedes its task's update. These are finite dependent
trajectories, not independent repetitions or in-sample post-update accuracy.

Every per-environment checkpoint, family, order, and 4,096/16,384/65,536-budget
score remains in [formal-audit-v1/summary.json](formal-audit-v1/summary.json).
Lower-budget stream descriptions rescore the original full-budget learning
path; they do not describe agents retrained with less acquisition budget.

## Computation and audit

Actual collection performs 8,960 searches: 768 initial experience, 3,072
stream, and 5,120 probe searches. All 4,534 acceptances are exact; 4,426 searches
exhaust their caps. No hidden-wrong accepted solution or skipped replay batch
occurs. The shared initial experience costs 3,167,190 primitive executions.
All actual search costs total 353,869,176, plus 478,640 hidden-evaluator
executions. The 45,056 gradient steps process 5,169,744 real target tokens
and rows, totaling 15,509,232 physical padded training input positions.

The fixed local prior has 21 possible contexts. The run builds 2,080 neural
tables with 43,680 model calls and 118,560 recurrent input positions; 2,112
ranked orders use 11,966,349 logical prior queries and 94,509,370 logical
prefix positions. Table lookup is not a neural forward call. Each table stores
84 float64 probabilities (672 numeric bytes), excluding keys and serialization.
The 320 checkpoint files occupy 11,215,168 bytes including optimizer/corpus
records; this is not an inference-RAM measurement.

| Final-only standalone system | Order | Search executions | Gradient steps | Physical training positions | Table model calls | Charged phase seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform | either | 21,793,204 | 0 | 0 | 0 | 1.627 |
| frozen | either | 25,747,732 | 12,288 | 2,304,000 | 672 | 10.073 |
| replay | composition first | 25,815,114 | 20,480 | 5,253,120 | 672 | 16.623 |
| replay | shifted first | 25,815,114 | 20,480 | 5,253,120 | 672 | 16.469 |
| online | composition first | 44,012,797 | 20,480 | 6,233,760 | 11,424 | 25.255 |
| online | shifted first | 44,046,338 | 20,480 | 5,681,232 | 11,424 | 25.407 |

These totals sum relevant measured phases across 32 environments for final
held-out predictions only. Replay's diagnostic stream searches/intermediate
orders are excluded because they cannot change its final state; online needs
its stream outcomes and corresponding orders. Uniform/frozen rows are shared
between orders, not separately observed trials. Required reused artifacts
are charged once. The complete actual totals retain all diagnostic work.

Standalone sums exclude explicit clone/load/checkpoint/table/order archive
phases. Replay timings include per-step journal writes/flushes and are
instrumented phase times, not pure training-compute measurements. Actual
training takes 33.356 seconds, table construction 2.197, ranking 31.792, and
search 14.859. Total collection takes 92.183 seconds and independent
audit/analysis takes 142.827 seconds on one AMD Ryzen 9 9950X CPU thread,
PyTorch 2.8.0+cu126, float64, with deterministic algorithms. These shared-phase
sums do not establish equal lifetime compute or independently timed speedups.

The full auditor independently regenerates tasks and initialization, rebuilds
accepted corpora, exactly replays every model/Adam tensor, reconstructs local
probabilities using NumPy GRU equations, and checks full-function search,
complete phase/file maps, and duration consistency. It passed all 8,960
searches and 45,056 steps. A separate local reconstruction from raw records
reproduces all 4,608 probe cells, 1,536 prequential cells, 324 aggregates, nine
contrasts, and both primary intervals. All 4,586 formal raw/audit files,
205,057,755 bytes, match the execution host via
[formal-manifest.json](formal-manifest.json); all 4,584 raw audit hashes match.
The retained development certificate matches the frozen source, data, and
runtime. No official source or setting changed after development.

## Interpretation

The small local network gains useful held-out composition capability when it
continues fitting its own accepted solutions, compared with stopping updates
or repeatedly fitting only initial experience. The untouched probes exclude
direct reuse of the same task functions. The experiment therefore extends
the preceding one-cycle transfer result to one short continued-learning loop.

Its failures supply no corrective target, only another replay of available
solutions. Accepted experience is endogenous to the current prior, and most
new accepted programs come from the motif-composition family. The resulting
specialization does not resolve the shifted-family deficit. This supports a
conditional mechanism with a supplied local representation, search, checker,
and task structure; it does not establish lifelong improvement, autonomous
representation discovery, feedback sufficiency, or general intelligence.
