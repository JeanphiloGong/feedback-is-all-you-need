# Study I: learned program priors

The [prospective protocol](../../notes/protocol-v9-neural-prior.md) compares
program-search priors learned from verified experience. Source and data are
frozen at `6365bec2b42cbc4c54b1d31f9200512a933357c8`. The standalone neural
components have 1,172 and 13,892 parameters; supplied symbolic search, checking,
task structure, and prior experience remain part of the complete system.

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
started only after the development audit passed; formal conclusions require
all 32 environments and their complete audit.

All 39 development/audit files are byte-identical to their remote originals,
verified by [development-manifest.json](development-manifest.json). The audit
also hashes all 37 raw files. [source-transfer.json](source-transfer.json)
records the 15 source/protocol/data hashes checked against both the Git commit
and the execution host before the first official run. Checkpoints are controlled
PyTorch tensor/optimizer archives and were audited with `weights_only=True`.
