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
