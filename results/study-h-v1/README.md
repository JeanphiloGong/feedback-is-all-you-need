# Study H: cross-task library transfer

Protocol: [`../../notes/protocol-v8-library-transfer.md`](../../notes/protocol-v8-library-transfer.md).
Implementation: [`../../experiments/study_h/README.md`](../../experiments/study_h/README.md).
Source/data freeze: `c582467`. This is a finite symbolic search experiment,
with no neural model. All four arms use supplied primitive tables, a program
enumerator, visible-example checks, and an interpreter.

## Formal results

All 32 declared environments completed once, with 768 experience searches and
4,096 evaluation searches. Independent replay passed all 4,864 search records
and separate evaluations. Every experience solution is exact; none of the
accepted evaluation programs is a false fit on the hidden inputs. This is an
observed result, not a guarantee from eight visible examples.

| Memory | Composition exact / 512 | Composition % | Shifted exact / 512 | Shifted % |
| --- | ---: | ---: | ---: | ---: |
| None | 258 | 50.39 | 256 | 50.00 |
| Whole programs | 12 | 2.34 | 6 | 1.17 |
| Compressed fragments | 436 | 85.16 | 51 | 9.96 |
| Renamed fragments | 75 | 14.65 | 62 | 12.11 |

The primary compressed-minus-whole composition difference is **+82.8125
percentage points**, t95 **[77.1176, 88.5074]**, bootstrap95 **[77.3438, 88.0859]**.
The independent unit is the environment, n=32; both intervals support the
declared positive criterion. Every environment's primary contrast is positive.
All other comparisons are descriptive. Each exact rate also equals held-out
input accuracy in this sample because all accepted programs are exact and
unaccepted tasks score zero.

The weaker whole-memory result matters: the fixed breadth-first searcher now
has 28 tokens instead of four. Compressed and renamed vocabularies have eight
tokens, including primitives. Learned content improves composition performance
over the control with renamed fragments and exceeds no memory by 34.77 points.
But it performs worse than no memory on shifted tasks. Across the fixed
balanced mixture, compressed memory solves 487/1,024 versus 514/1,024 without
memory. This does not establish overall superiority, neural learning, or an
enlarged primitive policy class. It is a conditional effect of the learned
representation on a supplied search order and deliberately shared task structure.

## Resource accounting

The cap charges expanded primitive applications, not comparisons, enumeration,
induction, or storage. Shared experience costs 3,110,069 operations and is
charged to each hypothetical standalone memory system. The no-memory system
needs no such experience. Totals below cover both evaluation families over all
32 environments; they have different success counts.

| Memory | Definition tokens | Standalone search operations | Sum of charged phase times (s) |
| --- | ---: | ---: | ---: |
| None | 0 | 43,488,101 | 30.721 |
| Whole programs | 4,608 | 69,104,664 | 34.947 |
| Compressed fragments | 473 | 42,256,312 | 27.109 |
| Renamed fragments | 473 | 63,662,017 | 39.949 |

The times add measured search and induction phases from the shared experiment;
they are not fresh standalone wall-time benchmarks. Each environment has 24
whole-program entries or four compressed/renamed entries. Definition tokens
exclude shared primitive tables and implementation. Compression changes the
3,840-token source corpus to 2,434 encoded tokens plus 473 definition tokens,
using 5,048 candidate-library evaluations and 0.833 seconds. Encoded corpus
size is a diagnostic; only the library persists during evaluation. The renamed
system pays both compression and its renaming cost.

Actual collection uses 212,290,956 primitive operations, including shared
experience only once. Final evaluators separately execute 180,864 operations.
The run takes 127.832 seconds including journaling, on an Intel Core i5-1135G7
CPU with Python 3.10.12. Audit/analysis takes 429.540 seconds and is not charged
as agent inference. There is no neural training or GPU use. Additional raw
counts, per-phase times, and every environment's costs are in the summary.

## Fixed budget curves

Exact success (%); an answer is available only once its first visible fit has
completed within the specified budget. No best checkpoint is selected.

| Family | Operations | None | Whole | Compressed | Renamed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Composition | 4,096 | 0.59 | 1.95 | 63.87 | 3.32 |
| Composition | 16,384 | 21.48 | 2.15 | 69.73 | 10.16 |
| Composition | 65,536 | 50.39 | 2.34 | 85.16 | 14.65 |
| Shifted | 4,096 | 0.00 | 0.59 | 1.17 | 1.95 |
| Shifted | 16,384 | 17.77 | 1.17 | 6.45 | 5.86 |
| Shifted | 65,536 | 50.00 | 1.17 | 9.96 | 12.11 |

## Formal archive

`formal-v1/` preserves complete task definitions, source/data hashes, metadata,
raw search and evaluation journals, and all 32 induced memories.
`formal-audit-v1/` contains the passing independent replay audit and all results,
including 768 environment/arm/family/budget records. `formal-manifest.json`
hashes all 38 files, each byte-identical to its original. No outcome or failure
was removed, and no formal rerun or protocol amendment occurred.

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
