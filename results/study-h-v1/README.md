# Study H: cross-task library transfer

Protocol: [`../../notes/protocol-v8-library-transfer.md`](../../notes/protocol-v8-library-transfer.md).
Implementation: [`../../experiments/study_h/README.md`](../../experiments/study_h/README.md).
Source/data freeze: `c582467`. This is a finite symbolic search experiment,
with no neural model. All four arms use supplied primitive tables, a program
enumerator, visible-example checks, and an interpreter.

## Development evidence

`development-v1/` retains seed 23's metadata, complete tasks, 152 raw search
records, 152 separate evaluations, and induced memories.
`development-audit-v1/` contains the independent replay audit and descriptive
summary. `development-manifest.json` hashes all seven archived files; each
matches its original byte for byte.

| Memory | Composition exact / 16 | Shifted exact / 16 |
| --- | ---: | ---: |
| None | 8 | 8 |
| Whole programs | 0 | 0 |
| Compressed fragments | 14 | 0 |
| Renamed fragments | 0 | 2 |

These are development observations from one environment, not population
estimates or formal inference. The 24 experience tasks and 128 evaluation
searches pass replay of every decision, budget count, memory, and score.
Collection took 4.348 seconds including journaling; audit/analysis took 14.181
seconds. The protocol, data, implementation, and budget were not changed after
these outcomes. Formal collection uses the prespecified 32 fresh environments.
