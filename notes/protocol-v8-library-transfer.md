# Study H: reusable fragments and cross-task search

Status: prospective design, frozen before development outcomes or formal
collection. This is a finite symbolic mechanism study motivated by
DreamCoder and the missing cross-task component in Studies F–G. It does not
replicate DreamCoder or introduce a new compression algorithm. Its controller
is a supplied enumerative program synthesizer, with **no neural model or weight
updates**. A positive result would concern system-level search with learned
external representations, not a parameter-count comparison or general intelligence.

## Question and falsifiable prediction

Can fragments compressed from previously accepted programs increase exact
recovery of new task functions, at a fixed budget of executed primitive
operations, relative to storing the accepted programs whole? The task generator
explicitly contains shared structure. The test is conditional on that structure,
not evidence that arbitrary unrelated experience should transfer.

Primary: compressed minus whole-program memory on new motif-composition tasks,
paired over 32 environments. The prediction is positive, with a two-sided 95%
t interval whose lower bound exceeds zero. Failure to meet that criterion is
inconclusive or contrary evidence for this configuration, not a reason to
change seeds, budgets, or the task family. Success means completing the audited
comparison regardless of sign. A second family of random primitive sequences
measures performance under a change in the program distribution, descriptively.

## Supplied domain and fixed split

Use development seed 23 and formal seeds 10000–10031. Derive independent
Python Random streams from SHA-256 of the compact UTF-8 JSON array [seed, role]
(separators comma and colon, without spaces), taking the
first eight digest bytes as a little-endian integer. Archive generated data
before formal search. No formal outcome can tune the protocol.

Each environment has 16 integer states and four named primitive operations,
A–D. Each primitive is an independently drawn permutation of those states;
reject the identity or a duplicate primitive. These complete transition tables
are known to every search arm. Programs are finite sequences applied left to
right. The identity program is permitted. This finite deterministic domain
provides an exact executor; it does not test natural-language interpretation.

Sample four distinct latent motifs uniformly from the 64 primitive sequences
of length three. For each motif, sample six different ordered pairs of
primitive prefix and suffix from the 16 possible pairs. These 24 length-five
programs define the experience tasks. If two generated experience programs
have the same full 16-state function, retain the first and omit the duplicate;
do not replace it. Shuffle the retained task order with its own RNG stream.
Neither latent motifs nor generating programs are provided to the learner.

Construct 16 held-out composition tasks: eight concatenations of two latent
motifs and eight of three motifs, drawn with replacement. Construct 16 shifted
tasks: eight independent uniform primitive sequences of length six and eight
of length nine. In each family, reject a task if its full function duplicates
an experience task or an already accepted evaluation task in either family.
This is a benchmark split constraint, checked by the data generator, not a
signal available to the learner. Archive rejection counts. Random tasks can
incidentally contain familiar fragments; do not reject them for that reason.

For every task independently, shuffle the 16 inputs. Disclose eight input/output
examples in that order; reserve the other eight for evaluation. Search receives
only primitive tables and the eight examples. Final exact success means that
the accepted program matches the complete 16-state function, including the
held-out inputs. Report held-out accuracy separately. Hidden labels never affect
acceptance, the learned library, search ordering, or future updates. Evaluator
access is a separate call after the search result has been fixed.

## Experience collection and memory construction

Solve each experience task once using primitive-only breadth-first enumeration:
identity, then sequences in increasing token length and lexicographic token
order A–D. Check its eight examples in fixed order and stop on the first
mismatch. Accept and stop at the first program passing all eight examples.
The per-task budget is 65,536 primitive state transitions, as defined below.
The generating program has length five, so exhaustive search through length
five fits this budget even if every candidate needs all eight examples.
Nevertheless preserve and report every unexpected failure or false acceptance.
Only accepted programs enter memory, including any that fail hidden evaluation.

Freeze four evaluation arms after the same experience collection:

- `none`: only the four primitives; no persisted experience programs.
- `whole`: append every distinct accepted nonempty program of length greater
  than one, ordered lexicographically, as a callable library entry. No retrieval
  model or favorable subset selection is used.
- `compressed`: extract all distinct length-two and length-three contiguous
  fragments from accepted programs. Starting with no library, greedily add at
  most four fragments. At each step, choose the candidate that most decreases
  `sum(len(fragment) + 1 for library definitions) + sum(encoded_length(program))`.
  Encoding uses minimum-token segmentation into primitives and current library
  calls, calculated by dynamic programming. Each token costs one; the extra
  definition token charges the new name. Break equal improvements by the
  primitive-string lexicographic order. Add only a strictly improving fragment.
  Candidate fragments and definitions are flat primitive sequences; this study
  does not induce recursive or hierarchical abstractions.
- `renamed`: apply one independently sampled nonidentity permutation of A–D
  to every compressed fragment, preserving entry order, count, and lengths.
  This control alters learned content at equal library-token storage. Some
  fragments may remain unchanged or accidentally help; do not resample on
  their performance. Its induction cost includes learning the original library.

For every arm, primitives come first, followed by library entries in the order
specified above (compressed entries use their induction order). Enumeration is
breadth first in these tokens, with the same fixed ordering rule for every task.
Duplicate expanded programs are retained and charged; there is no semantic
cache, hidden deduplication, learned ranking, or task-dependent retrieval.
Whole and compressed memory have different branching factors and storage.
Those differences are part of the representation intervention, not matched
memory capacity. The renamed control provides a narrower content comparison.
Libraries remain frozen throughout evaluation; evaluation tasks cannot teach
later tasks. This is one experience-to-transfer cycle, not lifelong learning.

## Feedback and resource accounting

At training and evaluation, examples supply correctness feedback on proposed
programs. A check can reject a proposal or accept the first complete fit.
There are no gradients and no error-directed revisions within a search: failed
checks advance a fixed enumeration. Experience changes the subsequent program
vocabulary through a supplied compression routine. This separates the effect of
external representation learning from an autonomous discovery of that routine.

Every primitive application to one example input costs one budget unit. Expand
library calls and charge every underlying primitive operation; a macro is not
a free execution step. Stop immediately when the budget would be exceeded.
A candidate interrupted during checking cannot be accepted. Completing the
last required operation exactly at the limit can still yield acceptance.
The identity program uses zero primitive operations, but its example checks
are counted. All four arms have the same 65,536-unit limit per evaluation task.

Also record candidate count, completed/rejected/accepted checks, comparison
count, expanded program lengths, library definition tokens, compressed corpus
tokens, induction candidate evaluations, and measured wall time for experience,
compression, renaming, search, and final evaluation. Enumeration, expansion,
comparisons, compression, memory storage, and interpreter overhead are excluded
from the primitive-operation cap; report them rather than calling this equal
FLOPs or equal wall time. No precomputed macro transition tables are permitted.

Experience collection is shared in the actual experiment and charged to each
hypothetical standalone memory arm; `none` needs no experience collection.
Report both actual totals and these separate system costs. Compression is paid
before evaluation, not hidden as free prior knowledge. Equal online budgets do
not establish equal lifetime resource costs. Secondary curves at 4,096, 16,384,
and 65,536 operations use the cost of the first accepted proposal; an arm that
has not accepted by a threshold emits no answer and scores zero. These curves
do not choose a best checkpoint or a better earlier candidate.

## Statistics, auditing, and stop rule

Each environment contributes its mean exact-success rate across the 16
composition tasks. The environment is the independent unit; tasks, candidates,
and budgets are not independent replicates. Compute the primary paired t
interval and a 10,000-resample paired percentile bootstrap with seed 829.
Report disagreement. All other comparisons, shifted-task scores, held-out
accuracies, storage, cost, and curves are descriptive. Disclose degenerate
intervals if all observed differences coincide. Report every environment.

Before formal collection, require a development run and independent audit of:
all permutation tables and labels; absence of semantic split overlap; first-fit
selection; exact budget boundaries and interruption; content of each memory;
compression objective and tie-breaking; library-call expansion costs; final
held-out scores; and costs/statistics. Tests must show that modifying hidden
labels cannot change search or induction, and that macro execution agrees with
its primitive expansion. Independent calculation must reproduce the reported
costs and scores without trusting saved summary fields.

Archive source/data hashes, Git revision, Python/package versions, hardware,
complete task definitions, all accepted programs, compression choices, library
contents, per-search counts and terminal state, timings, and initialization or
runtime failures. Flush each completed search result to disk. Enumeration is
deterministic and reproducible from the frozen source, examples, library, and
terminal candidate index; do not describe counts as an archive of every rejected
proposal. Run the fixed formal grid once after review and development audit.
Correct defects with an explicit amendment and retain failed runs. Do not alter
historical Studies A–G or manuscript v9.
