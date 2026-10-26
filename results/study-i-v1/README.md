# Study I: learned program priors

The [prospective protocol](../../notes/protocol-v9-neural-prior.md) compares
program-search priors learned from verified experience. Source and data are
frozen at `6365bec2b42cbc4c54b1d31f9200512a933357c8`. The standalone neural
components have 1,172 and 13,892 parameters; supplied symbolic search, checking,
task structure, and prior experience remain part of the complete system.

## Formal result

All 32 fixed environments completed once: 768 experience searches, 8,192
transfer searches, and 36,864 gradient updates. The complete audit passed.
The primary small-trained-minus-uniform composition difference is **−27.1484
percentage points**, paired t95 **[−32.7449, −21.5519]**, bootstrap95
**[−32.2266, −21.6797]**. Differences are negative in 29 environments, positive
in two, and zero in one. The two intervals agree. This is contrary evidence
for the declared positive prediction in this configuration.

| Prior | Neural parameters | Composition exact / 512 | Composition % | Shifted exact / 512 | Shifted % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0 | 256 | 50.00 | 256 | 50.00 |
| Counts | 0 | 338 | 66.02 | 172 | 33.59 |
| Small frozen | 1,172 | 256 | 50.00 | 255 | 49.80 |
| Small trained | 1,172 | 117 | 22.85 | 52 | 10.16 |
| Small shuffled | 1,172 | 137 | 26.76 | 70 | 13.67 |
| Large trained | 13,892 | 90 | 17.58 | 51 | 9.96 |
| Small after 8 attempts | 1,172 | 146 | 28.52 | 61 | 11.91 |
| Small after 16 attempts | 1,172 | 119 | 23.24 | 55 | 10.74 |

All comparisons other than the declared primary are descriptive. Counts use
100 integer cells with 21 reachable two-token contexts, not zero system memory.
On the balanced mixture, counts solve 510/1,024 tasks versus uniform's
512/1,024; the small trained network solves 169/1,024. The larger network also
has a lower observed score under the same fixed update schedule; this is not
a general conclusion about scaling or an established capacity optimum.

All 768 experience solutions are exact and supply 3,840 primitive tokens.
No accepted transfer solution fails its hidden evaluation. These failures are
therefore failures to find a complete visible fit within the budget, not false
acceptance or an output-format problem. The small-network composition mean
falls from 50.00% initially to 28.52%, 23.24%, and 22.85% at the prespecified
checkpoints. Individual trajectories vary; this does not mean every update
or every environment deteriorates.

[learning-progress.pdf](learning-progress.pdf) shows those four checkpoints,
with all 32 individual trajectories and the uniform reference.
Regenerate it with `python3 -s results/study-i-v1/plot_progress.py`.

## Interpretation and cost boundary

This network learns an unconditional sequence prior from length-five accepted
solutions; it does not read task examples or learn from each failed transfer
proposal. Transfer includes new lengths six and nine. Next-token training and
successful synthesis are different objectives. The present comparison cannot
isolate whether the failure arises from architecture, optimization, positional
overfitting, or this objective/distribution mismatch. No explanation was selected
by a new intervention after seeing the result.

The count baseline's composition gain supports useful reuse of observed local
structure by a supplied statistical rule. Its shifted-task reduction and nearly
unchanged balanced score prevent an overall-superiority claim. This study does
not test a task-conditioned policy, online transfer-task adaptation, learned
symbolic compression, or unrestricted intelligence. Study H uses other seeds
and a different search representation, so its scores are not a paired comparator.

Experience collection costs 3,188,241 primitive operations. Each standalone
trained/count arm pays for it; uniform and frozen priors do not. Intermediate
probes pay only their available experience prefix and updates. Ranking is
amortized over 32 transfer tasks per environment. All arms have the same 65,536
executed-primitive cap per task, but their neural and other compute differs.

| Final prior | Standalone search primitive operations | Charged phase seconds |
| --- | ---: | ---: |
| Uniform | 44,144,794 | 2.739 |
| Counts | 46,635,157 | 2.847 |
| Small frozen | 43,858,224 | 27.705 |
| Small trained | 63,641,689 | 25.907 |
| Small shuffled | 62,135,844 | 25.969 |
| Large trained | 65,071,471 | 29.562 |

The time column adds measured initialization, optimizer setup, training,
count construction, ranking, and search phases as applicable. Shared phases
are reused for these hypothetical standalone estimates; these are not
independent wall-time benchmarks or equal-FLOP comparisons. Checkpoint I/O,
cloning, final hidden evaluation, and other research overhead are separately
recorded in actual totals.

Actual collection, including both intermediate probes, uses 434,569,422 search
primitive operations, 2,304,000 replay target tokens, and 9,606,438 neural
prefix positions. Search, training, and ranking take 17.751, 26.203, and 103.961
seconds respectively. Complete collection takes 150.076 seconds on the lens
AMD Ryzen 9 9950X CPU; audit/analysis takes 84.424 seconds. The final evaluator
adds 302,256 primitive operations outside search. There are no skipped replay
attempts. The 352 archived checkpoint files occupy 43,789,664 bytes, including
optimizer states; these archive bytes are not an inference-RAM measurement.

All 969 formal raw/audit files (80,380,327 bytes) match the remote originals
byte-for-byte. [formal-manifest.json](formal-manifest.json) records them, while
the independent audit hashes every one of the 967 raw files. Local independent
reconstruction also reproduces the paired differences and both intervals.
All raw attempts, every checkpoint, both task families, and all budget curves
remain in the archive; no design or source changed after official outcomes.

## Development record

Development seed 29 completed once: 280 searches and 1,152 gradient updates.
Collection took 5.428 seconds on a lens AMD Ryzen 9 9950X CPU; independent
audit/analysis took 3.398 seconds. All initial/learned tensors, optimizer states,
rankings, execution costs, scores, and phase records passed audit.

| Prior | Composition exact / 16 | Shifted exact / 16 |
| --- | ---: | ---: |
| Uniform | 8 | 8 |
| Counts | 8 | 4 |
| Small frozen | 8 | 8 |
| Small trained | 1 | 0 |
| Small shuffled | 4 | 4 |
| Large trained | 2 | 2 |
| Small after 8 attempts | 2 | 1 |
| Small after 16 attempts | 1 | 0 |

These are single-environment development observations, without a primary
interval or confirmatory claim. The unfavorable result did not trigger a
configuration, budget, seed, task, or source change. The fixed formal grid was
started only after this development audit passed.

All 39 development/audit files are byte-identical to their remote originals,
verified by [development-manifest.json](development-manifest.json). The audit
also hashes all 37 raw files. [source-transfer.json](source-transfer.json)
records the 15 source/protocol/data hashes checked against both the Git commit
and the execution host before the first official run. Checkpoints are controlled
PyTorch tensor/optimizer archives and were audited with `weights_only=True`.
