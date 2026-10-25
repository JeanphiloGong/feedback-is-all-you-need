# Study G: feedback history and repeated rule proposals

Status: prospective design, before development or formal model calls. Study F's post-hoc
transition counts motivate this follow-up: 0.5B repeats its final checked
response in 16/24 environments and repairs few checked command errors. This
study tests whether retaining the full available proposal/check history changes
that behavior over a longer fixed horizon. It does not assume history helps.
The broad method has precedents in reflection and experience-memory systems;
this is a controlled diagnostic extension, not an invention of agent memory.

## Tasks, models, and fixed budget

Freeze 48 new modulo-eight languages, seeds 9000–9047, plus development seed 19.
Use the same generator and splits as F: four distinct offsets, 16 disclosed
examples, 16 direct held-out queries, and 32 two-command held-out queries.
Both branches receive all examples initially. True offsets are identifiable
from those examples; checks add computed verification, not new task information.
Neither held-out inputs nor targets enter a generation or revision prompt.

Use the pinned Qwen2.5-0.5B-Instruct and 7B-Instruct models from F. Preserve
bfloat16/SDPA, left-padded batches of 16 independent languages, greedy decoding,
a 512-token cap per call, and inherited model defaults including repetition
penalties 1.1 and 1.05. No parameter updates or teacher model. Execute exactly
16 revisions per branch, even after passing checks or repeating a response.
Sample count, checkpoints, and cap remain fixed after outcomes. There is no
claim that 48 environments proves absence of smaller effects; report interval
widths and all failures. Model-size comparisons remain descriptive.

## Intervention

Generate one initial proposal and fork it into two branches. Retain F's exact
four-key JSON parser, range/distinctness constraints, training-example checker,
and supplied executor. Invalid rules still score zero. Do not normalize negative
values, repair JSON, choose a better earlier candidate, or stop when correct.
The last proposal is evaluated at each checkpoint.

Both branches use identical task/example instructions and the same review
instruction from F. The current review contains the previous proposal and
its format error, first failed example, or all-pass check. Branches differ only
in earlier interaction history:

- `last_only`: construct the same fresh task/examples + previous assistant
  proposal + current check/review messages used by F's checked branch.
- `full_history`: start with the same task/examples and retain every assistant
  proposal and its following check/review message. A check appears only after
  the corresponding proposal. Earlier successes and failures are both retained.
  The current review instruction is identical to `last_only`.

At revision 1, both branches have identical prompts, ordered seed batches,
token IDs, attention masks, and padding. Audit those inputs and generated token
equality before attributing later differences to history. Preserve any mismatch
and stop the suite as failed; do not rerun until equality or interpret that
suite as a successful history comparison. A changed implementation requires
a separately recorded development run and source freeze. Run and count each branch's
revision calls; the single shared initial generation is counted once in actual
cost and included in each hypothetical standalone-arm cost. Preserve all
responses and all memories, including intermediate failures and regressions.
No acceptance gate preserves a prior better rule. This isolates availability
of earlier transcripts as a complete intervention, including added context
and computation; it does not isolate memory semantics from token cost.

## Endpoints

Primary: 0.5B final direct held-out accuracy for `full_history` minus `last_only`,
paired across the 48 environments. Use a 95% t interval as the primary interval
and a 10,000-resample paired percentile bootstrap with seed 823 as sensitivity.
A positive primary claim requires the t lower bound above zero; report any
bootstrap disagreement. Also report each branch's final-minus-initial effect.

Direct accuracy is primary because F exposed that a fully incorrect offset
vector can produce perfect two-command composition accuracy. Secondary outcomes
are exact-rule recovery and format validity (Wilson 95% intervals), composition
accuracy, transitions, repeated proposals, and descriptive 7B outcomes. Report
checkpoints 0, 1, 2, 4, 8, and 16, while retaining and scoring every proposal
for auditing and transition counts. No checkpoint selection is permitted.

Report exact response repetitions and repeated valid mappings separately;
whitespace differences can change text without changing a rule. Count both
immediate repetitions and previously seen proposals within each branch.
Stratify both by current invalid, valid-but-incorrect, and exact proposals.
Report total transition and valid-output denominators separately: repetition
of a correct rule can be desirable, and reduced repetition alone is not learning.
Transition counts remain nested in environments. Component repair/loss metrics
use F's declared valid-pair semantics; invalid successors are separate.
Descriptive t intervals on query scores use environment means. If a contrast
has zero empirical variance, disclose the degenerate t/bootstrap interval;
it does not establish a zero population effect. Marginal Wilson bounds on
exact recovery and validity remain informative at all-zero/all-one samples.

## Accounting and validation

Record all raw token IDs, decoded outputs, prompts, checked messages, rule
files, histories at each call, model/source/data hashes, package versions,
batches, input/output tokens, padded token slots, elapsed generation/checker
time, and peak allocated memory. Full history can consume much more input
compute. A fixed number of calls is not a compute-matched or lifetime-resource
comparison. Raw completed batches must be flushed before parsing; preserve
initialization errors as well as failures during collection.

Require an audited development run on seed 19 with matching source, model,
configuration, and packages before either formal suite. Test that the first
revision prompts match across branches, that earlier incorrect proposals and
checks are retained only in the full-history branch, that check messages never
precede their own proposal, and that hidden query/target mutation cannot affect
any prompt. Verify every symbolic label, complete grids, raw decoding, strict
parses, checkpoint scores, response histories, costs, and paired statistics.
The full history of at most 16 prior 512-token proposals must fit the model's
context limit without truncation; verify tokenized lengths before generation.

Success means a complete audited comparison, not a positive result. No outcome
would establish cross-language skill accumulation, neural retention, feedback
sufficiency for general intelligence, or the irrelevance of model capacity.
