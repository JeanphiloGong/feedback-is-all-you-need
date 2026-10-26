# Study J: full versus local context

The [prospective protocol](../../notes/protocol-v10-local-context-prior.md)
compares two ways of using the same 1,172-parameter GRU: the complete program
prefix or only its last two primitive tokens with a reset recurrent state.
Both start from identical tensors and receive the same accepted experience,
targets, and update counts. Source and data are frozen at
`ce232771cd35a61e6b6d7678d4a31c79b9f4a099`. The context restriction and symbolic
search are supplied mechanisms; matching parameters/updates does not match
recurrent input work, FLOPs, or total system capacity.

## Development record

Seed 31 completed once: 344 searches and 768 gradient steps. Collection took
5.265 seconds on the lens CPU; complete independent audit/analysis took
2.784 seconds. All initial and learned tensors, optimizer states, ranked
orders, execution/hidden scores, and physical input records passed audit.

| Prior | Composition exact / 16 | Shifted exact / 16 |
| --- | ---: | ---: |
| uniform | 8 | 8 |
| counts | 11 | 5 |
| full_frozen | 8 | 8 |
| local_frozen | 8 | 8 |
| full_trained | 1 | 2 |
| local_trained | 15 | 1 |
| full_after_8 | 2 | 3 |
| full_after_16 | 1 | 1 |
| local_after_8 | 11 | 2 |
| local_after_16 | 12 | 2 |

These are descriptive single-environment observations, not confirmatory
inference. Local training's favorable composition result coexists with lower
shifted success than uniform and an unchanged balanced score of 16/32. No
source, model, seed, setting, budget, or checkpoint choice changed afterward.
The complete formal grid started only after the matching development audit.

All 39 development/audit files, 1,553,116 bytes, match their remote originals;
all 37 raw hashes also match the independent audit.
[development-manifest.json](development-manifest.json) preserves those digests.
[source-transfer.json](source-transfer.json) records all 15 execution source/data
files verified against both Git and the lens host before collection.
The [implementation instructions](../../experiments/study_j/README.md) describe
reproduction. No official development result was used for tuning.

## Formal results

All 32 frozen formal environments (12000–12031) completed with the unchanged
source/data/settings and matching development audit. Each has 24 experience
tasks, 16 new motif compositions, and 16 shifted random programs. Training
ends before transfer; transfer outcomes never update either model. The supplied
searcher checks eight visible examples and the evaluator checks eight hidden
examples and the full 16-state function. Results below use a cap of 65,536
primitive executions per task, excluding training, ranking, and overhead.

| Prior | Composition exact / 512 | % | Shifted exact / 512 | % | Balanced exact / 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 256 | 50.00 | 256 | 50.00 | 512 |
| counts | 338 | 66.02 | 170 | 33.20 | 508 |
| full_frozen | 250 | 48.83 | 256 | 50.00 | 506 |
| local_frozen | 250 | 48.83 | 256 | 50.00 | 506 |
| full_trained | 124 | 24.22 | 65 | 12.70 | 189 |
| local_trained | 406 | 79.30 | 57 | 11.13 | 463 |
| full_after_8 | 142 | 27.73 | 61 | 11.91 | 203 |
| full_after_16 | 141 | 27.54 | 55 | 10.74 | 196 |
| local_after_8 | 313 | 61.13 | 67 | 13.09 | 380 |
| local_after_16 | 384 | 75.00 | 56 | 10.94 | 440 |

The prespecified primary is final **local-trained minus full-trained composition
success**, paired over 32 environment means: **+55.08 percentage points**,
t95 **[45.60, 64.56]**, bootstrap95 **[45.90, 63.87]** (10,000 resamples,
seed 953). Thirty environments improve and two tie. Both intervals are
positive, satisfying the protocol's primary criterion. The full-trained
comparator itself performs below uniform, so this contrast alone does not
establish improvement over uninformed search.

The separate descriptive comparisons show local-trained composition gains of
29.30 points over uniform and 30.47 points over local-frozen; each comparison
has 31 positive environments and one tie. On shifted programs local-trained
loses 38.87 points to both controls, in all 32 environments. Its balanced
score is 45.21%, **4.79 points below uniform** (20 environments worse, six
equal, six better). Counts also fall short of uniform on the balanced total.
These secondary comparisons and intermediate checkpoints are descriptive;
none replaces the primary endpoint.

The [learning-progress figure](learning-progress.pdf) retains all 32 individual
trajectories and the means at 0/8/16/24 experience attempts. Local composition
means rise from 48.83% through 61.13% and 75.00% to 79.30%; shifted performance
falls from 50.00% to 11.13%. This is a finite experience-to-transfer comparison,
not evidence of unbounded or monotonic improvement on every environment.
Reproduce the figure with `python3 -s results/study-j-v1/plot_progress.py`.
All three execution budgets and all seed-level scores remain in
[formal-audit-v1/summary.json](formal-audit-v1/summary.json).

## Costs and verification

Formal collection performed 11,008 searches: 768 shared experience attempts
and 10,240 transfer searches across six final arms and four probes. All
experience solutions were exact; all 4,671 accepted searches across both
phases were exact and 6,337 exhausted their budgets. No false accepted transfer
solution or skipped replay attempt occurred. The actual combined run used
499,770,365 primitive search executions, with 463,248 further primitive
executions for hidden evaluation. Shared experience used 3,152,551 of the
search executions.

Both trained models perform 16 persistent-Adam steps after each experience
attempt: together 24,576 gradient steps and 1,536,000 replay target tokens.
Local training constructs a fresh padded row for every target, while full
training processes each whole sequence. Together they use 921,600 rows and
3,072,000 physical padded input positions. Ranking makes 2,390,113 prior
predictions, representing 18,153,098 logical prefix positions but 9,416,727
actual recurrent input positions. Equal parameter and update counts do not
imply equal recurrent work or FLOPs.

| Final arm | Search executions including required experience | Charged phase seconds | Physical training positions | Recurrent ranking positions |
| --- | ---: | ---: | ---: | ---: |
| uniform | 43,571,131 | 2.577 | 0 | 0 |
| counts | 46,670,297 | 2.915 | 0 | 0 |
| full_frozen | 44,515,010 | 26.564 | 0 | 2,429,656 |
| local_frozen | 44,464,513 | 17.431 | 0 | 997,668 |
| full_trained | 63,289,610 | 25.095 | 768,000 | 1,414,642 |
| local_trained | 48,238,112 | 18.147 | 2,304,000 | 544,350 |

These standalone totals sum relevant shared measured phases across 32
environments; they are not independently timed runs or equal-FLOP comparisons.
Ranking is amortized over 32 transfer tasks per environment. Actual combined
training, ranking, and search times were 15.519, 110.839, and 20.520 seconds.
Total collection took 149.137 seconds and independent audit/analysis took
74.098 seconds, on one AMD Ryzen 9 9950X CPU thread, PyTorch 2.8.0+cu126,
float64, with deterministic algorithms. The 224 archived checkpoint files
occupy 7,333,536 bytes including optimizer states; that is not inference RAM.

The full auditor regenerates tasks and initialization, replays every learned
tensor and Adam state, reconstructs full/local GRU probabilities using NumPy,
and verifies candidate ordering, full-function scores, and the complete
physical-input/cost ledger. It passed all 11,008 searches and 24,576 steps.
A separate local reconstruction from raw search/evaluation records reproduced
every seed/budget score, descriptive contrast, and both primary intervals.
All 969 formal raw/audit files (48,537,188 bytes) match the execution-host
originals via [formal-manifest.json](formal-manifest.json); all 967 raw hashes
also match the original audit. The implementation passed 20 synthetic tests
before data generation and collection.

## Interpretation

This intervention demonstrates that the same small network, accepted
experience, and update schedule can transfer much better to new motif
compositions when each prediction sees a supplied local context. It does not
isolate a single causal explanation: truncating history, resetting recurrent
state, shortening gradient paths, and changing physical computation occur
together. The last-two-token restriction is an experimenter-provided inductive
bias, not a discovered representation. The network remains an unconditional
program prior inside a supplied symbolic search and verification system.

The shifted and balanced deficits prevent an overall superiority claim. The
result does not establish feedback sufficiency, unrestricted intelligence,
lifelong improvement, or a universal advantage of fewer parameters. It does
show why Study I's failure cannot be read as a general incapacity of a
1,172-parameter network: performance depends on how experience is represented
and used. Study I and J use different formal seeds; cross-study numerical
differences are not paired treatment effects.
