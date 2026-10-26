# Study I: fitting experience and scoring transfer programs

This is an **explicitly post-hoc descriptive analysis**, specified after the
formal search results and manuscript v11. The [analysis plan](../../notes/study-i-likelihood-diagnostic-plan.md)
was frozen at `70d6558`; the reviewed implementation is frozen at
`85f1113ed3a978c7d26b5752c3d19059a5bc4a8b`. No new training, search, task
generation, or checkpoint selection occurred. The original negative primary
result and every archived experiment endpoint remain unchanged.

## Observations

The small trained prior lowers accepted-experience token loss in **all 32
environments** relative to its initial state. Its mean loss is **0.662957**
nats per token, versus **1.393215** when frozen. The unrestricted empirical
prefix predictor's bound is **0.635611**, leaving a mean excess of **0.027347**.
This establishes improved in-sample fitting under the stated objective.

On the composition generating programs, its raw loss is **2.596447**, compared
with uniform's **1.386294**; every environment has higher loss than uniform.
After applying the original search smoothing, the difference is **+0.437787**
nats per token, positive in 31 environments and negative in one. On shifted
programs, raw and smoothed losses are higher than uniform in all 32 environments.
Improved experience fitting therefore coexists with poor transfer likelihood
and the previously observed lower budgeted search success.

All losses below are nats per target token; lower is better. Raw probabilities
are used by the training objective. Smoothed probabilities are the original
search prior, `0.9 pi + 0.025`. There is no EOS loss, and the common uniform
length prior is excluded. Means weight environments equally; tokens and tasks
are nested observations, not independent replicates.

| Prior | Full experience, raw | Composition, raw | Composition, smoothed | Shifted, raw | Shifted, smoothed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 1.3863 | 1.3863 | 1.3863 | 1.3863 | 1.3863 |
| Counts | 1.0298 | 1.0913 | 1.1040 | 1.5098 | 1.4830 |
| Small frozen | 1.3932 | 1.3897 | 1.3880 | 1.4000 | 1.3974 |
| Small trained | 0.6630 | 2.5964 | 1.8241 | 3.3573 | 2.1445 |
| Small shuffled | 2.2551 | 2.3209 | 1.7631 | 2.9893 | 2.0753 |
| Large trained | 0.6473 | 2.9522 | 1.9494 | 3.6507 | 2.1815 |
| Small after 8 | 1.4153 | 2.1319 | 1.7213 | 2.7116 | 2.0346 |
| Small after 16 | 0.9803 | 2.4225 | 1.8117 | 3.0870 | 2.1175 |

The count prior fits experience less closely than the final small and large
priors trained on real accepted experience, but assigns higher average
likelihood to composition generating programs.
It assigns lower likelihood than uniform to shifted programs. This matches
the direction of its original task-family search results, but likelihood of
one generating program is not total probability of all programs that solve
the task. No causal mediation or semantic-mass estimate follows.

## Each learner's actual training corpus

The bound permits a separate next-token distribution at every complete prefix.
Conflicting continuations make its minimum loss nonzero. It is an unrestricted
in-sample bound, not a neural capacity bound or a generalization guarantee.
For the shuffled arm, this table uses its stored shuffled sequences; the table
above always uses the common real accepted experience. Intermediate own-corpus
rows use only solutions accepted during the first 8/16 attempted tasks.

| Learner | Actual-corpus raw loss | Empirical prefix bound | Raw excess |
| --- | ---: | ---: | ---: |
| Small trained | 0.662957 | 0.635611 | 0.027347 |
| Small shuffled | 0.703128 | 0.630918 | 0.072211 |
| Large trained | 0.647273 | 0.635611 | 0.011663 |
| Small after 8 | 0.593528 | 0.415888 | 0.177640 |
| Small after 16 | 0.608561 | 0.554518 | 0.054043 |

The larger network has a smaller mean in-sample excess than the small network,
while its transfer likelihood and original search success are worse under the
fixed schedule. More complete fitting alone is therefore not a sufficient
description of improved transfer in this archive. Different optimization or
regularization could still change transfer; this analysis does not test them.
Own-corpus losses at different checkpoints concern different datasets and
cannot be treated as a fixed-dataset learning curve.

## All paired descriptive differences

Each entry is small trained minus the named comparator. Full paired values
and all seed-level cells are retained in the JSON summary. These are descriptive
sign counts and means; no new hypothesis test or primary endpoint is introduced.

| Dataset | Comparator | Loss | Mean difference | Negative / zero / positive |
| --- | --- | --- | ---: | ---: |
| experience_all | small_frozen | raw_loss | -0.730258 | 32 / 0 / 0 |
| experience_all | small_frozen | smoothed_loss | -0.685703 | 32 / 0 / 0 |
| composition | uniform | raw_loss | +1.210152 | 0 / 0 / 32 |
| composition | uniform | smoothed_loss | +0.437787 | 1 / 0 / 31 |
| shifted | uniform | raw_loss | +1.970991 | 0 / 0 / 32 |
| shifted | uniform | smoothed_loss | +0.758164 | 0 / 0 / 32 |

## Length and position descriptions

The following table retains every transfer length and position for the small
trained prior. The uniform value is 1.386294 at every position. Complete
length/position descriptions for **all eight priors** are in the archived
summary. Positions are one-based and are scored with the actual preceding
primitive tokens, not generated model continuations.

| Program length | Position | Composition, raw | Composition, smoothed | Shifted, raw | Shifted, smoothed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 1.4680 | 1.4527 | 1.4206 | 1.4114 |
| 6 | 2 | 2.8638 | 1.9228 | 3.1616 | 1.9281 |
| 6 | 3 | 3.6106 | 2.3247 | 4.6223 | 2.5204 |
| 6 | 4 | 3.7014 | 2.2911 | 4.2910 | 2.4251 |
| 6 | 5 | 2.3745 | 1.7199 | 3.2313 | 2.1681 |
| 6 | 6 | 2.1032 | 1.5195 | 3.4802 | 2.2657 |
| 9 | 1 | 1.4507 | 1.4378 | 1.4228 | 1.4137 |
| 9 | 2 | 3.1184 | 2.0694 | 3.2994 | 2.0645 |
| 9 | 3 | 3.5545 | 2.2217 | 4.7575 | 2.6068 |
| 9 | 4 | 3.6714 | 2.2960 | 4.1045 | 2.3321 |
| 9 | 5 | 2.4163 | 1.7598 | 3.4415 | 2.2985 |
| 9 | 6 | 1.8493 | 1.3942 | 3.1303 | 2.1443 |
| 9 | 7 | 2.7147 | 1.9245 | 3.2995 | 2.1967 |
| 9 | 8 | 2.4932 | 1.7410 | 3.5569 | 2.2890 |
| 9 | 9 | 1.5567 | 1.2859 | 3.1397 | 2.1022 |

Higher mean loss than uniform is already present at positions within the
length-five training range. Content, prefixes, and the position of motifs also
differ between experience and transfer, so these cells do not isolate any one
cause. In particular, they do not prove positional overfitting or identify
length extrapolation as the sole mechanism.

## Evidence and reproducibility

The single diagnostic run took **4.349 seconds** on the lens CPU, using the
original PyTorch 2.8.0+cu126 environment, float64, and one Torch thread. This is
analysis time, not training or a deployment benchmark. It preserved **17,408
program records**, with **84,480 neural target probabilities and logs** checked
against independent NumPy GRU equations at absolute/relative tolerance 1e-12.
The duplicated own-corpus views are included in these record/token counts.

All 967 original raw files matched the passing experiment audit before scoring.
An independent standard-library reconstruction verified the complete program
grid, 12,128 environment cells, 379 aggregates, 160 training corpora, and six
paired descriptive contrasts. Repeating that reconstruction locally produced
the same verification results. All four transferred raw/verification files,
10,794,297 bytes, match the remote originals in [manifest.json](manifest.json).
All 18 diagnostic and dependency source hashes match the implementation freeze.

- [Program records](analysis-v1/programs.jsonl): actual sequences and both per-token log probabilities.
- [Full summary](analysis-v1/summary.json): every arm, dataset, environment, length, position, bound, and difference.
- [Provenance](analysis-v1/metadata.json): source/input/output digests, versions, and elapsed time.
- [Remote verification](verification-v1.json) and [local verification](local-verification-v1.json).
- [Code and rerun instructions](../../experiments/study_i_diagnostics/README.md).

The generating programs are used only in this offline diagnostic after the
original search and learning have finished. Their likelihood does not include
all semantically equivalent or shorter solutions. These results do not isolate
a causal explanation, establish an optimal capacity, show unrestricted
intelligence, or establish that a different feedback mechanism would fail.
Any intervention on the learning rule requires a separate frozen comparison
on new environments.
