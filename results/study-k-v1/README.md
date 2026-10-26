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
