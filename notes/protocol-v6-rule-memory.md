# Study F: rule memory and externally checked revision

Status: prospective design, before development model calls. The goal is to test
whether a model can extract and revise a compact rule stored externally,
using explicit checks. This is structured rule induction with a supplied
executor, not general program synthesis, autonomous exploration, or a claim of
newly invented agent memory. Relevant precedents are Voyager, Self-Debugging,
and ExpeL; see `full-text-review.md` for the source comparison.

## Fixed environment and models

Use the existing modulo-eight command generator on 24 fresh seeds 8000--8023.
Each language has four distinct offsets, 16 disclosed single-command examples,
16 held-out direct queries, and 32 held-out compositions. Freeze JSON and
SHA-256 before inference. Development uses seed 17 only, included in the same
frozen data file. The same 16 examples
are available from the start in every condition. All four offsets are
identifiable from those examples. Held-out inputs and targets never enter a
rule-generation or revision prompt.

Use the pinned Qwen2.5-0.5B-Instruct and 7B-Instruct revisions from Study B,
with bfloat16/SDPA and greedy generation capped at 512 new tokens. Batch 16
independent environments where possible; left-pad. Inherit model defaults and
save them, including model-specific repetition penalties. No parameter updates
or teacher model. The two models have different pretraining histories; their
comparison is descriptive rather than a causal parameter scaling estimate.

## Rule representation and update loop

The model receives the known additive family and all examples. It proposes a
JSON object mapping `dax`, `wug`, `zup`, and `fep` to integer offsets 0--7.
The exact four keys, integer type (excluding bool), range, and distinctness
are required. Parsing accepts a JSON object alone, or one outer fenced code
block with optional lowercase `json` language tag and newline-delimited body;
no surrounding commentary. Duplicate
keys invalidate the response. These are interface checks, not inferred truths.
No model code is executed. A supplied deterministic modular-addition executor
uses the proposed offsets to answer all held-out queries.

Generate one initial proposal per environment, then fork its stored text into
two revision paths. Each path uses two further model calls with the same cap:

- `self_review`: repeat the original task/examples and the previous proposal,
  and ask the model to check and revise its rule. No evaluator result enters
  this path.
- `checked_review`: use the same content, plus an exact check of the previous
  proposal against the disclosed examples. If parsing fails, state only that
  the rule violates the declared output format. Otherwise report either that
  all examples pass or the first failing example in the fixed training order,
  with the proposed rule's predicted output and the already-disclosed correct
  output. The verifier never supplies the true offset vector.

Use a fresh prompt containing the task/examples, immediately previous proposal,
and the current review instruction on each call. Do not accumulate older
proposal histories. Save all intermediate responses. Always make both revisions,
including after a passing proposal; evaluate the final proposal without
selecting a better earlier candidate. Improvement is not assumed, and regression
must be reported. The initial shared call counts once in actual run cost and
once per arm in an explicitly hypothetical standalone-arm cost, never twice
in measured total runtime.

The checker derives its message entirely from examples already in the prompt.
It supplies computation and a reliability signal, not new task information.
The proposed four-offset memory persists as a file; all test execution removes
the examples and uses only that memory. The executor and rule-family knowledge
are part of system capability and must not be attributed to the neural model.
There is no learning across distinct languages or unbounded memory library.

## Endpoints and controls

Primary: 0.5B environment-paired final composition accuracy for `checked_review`
minus `self_review`, with a 95% t interval and a 10,000-resample bootstrap
(RNG seed 822). The t interval is primary; a lower bound above zero supports
a positive composition-score effect. The bootstrap is a percentile 95%
sensitivity interval, resampling the 24 paired environment differences. Report
any disagreement. Also report each arm's final-minus-initial effect: a positive
primary result alone might arise from self-review degradation.

Secondary, with descriptive intervals: initial and each revision's direct and
composition accuracy; exact rule recovery; valid-format rates; training-check
success; wrong-to-correct and correct-to-wrong transitions; all 7B outcomes.
Use Wilson 95% intervals for binary environment-level format, exact-rule, and
training-pass rates, and t intervals over environment means for query accuracy
and per-environment answer-transition proportions. Report transition counts too.
Include the exact symbolic estimator from examples as an identifiability
control, clearly labeled as a human-supplied algorithm. A correct rule implies
perfect direct and composition execution; these endpoints are not independent
skills. Conversely, shifting every offset by four modulo eight gives a fully
wrong rule with perfect two-command composition accuracy and zero direct
accuracy. Report perfect-composition/nonexact rules explicitly. Answer
wrong-to-correct transitions concern individual held-out queries, not exact
rule recovery. Invalid proposals score zero on every query and remain in denominators.

Equal revision-call caps do not imply equal compute: prompt lengths and early
EOS can differ. Record input tokens, generated tokens through EOS, padded
work, batch times, checker time, peak GPU memory, and memory byte size. Comparisons
with a frozen proposal must include the two extra revision calls. There is no
claim of equal lifetime cost or pure isolation of externally supplied feedback
from its extra instruction tokens.

## Audit, stop rule, and reproducibility

Formal inference requires a complete audited seed-17 development run with
matching source/model/configuration. Store raw IDs and decoded text, all prompts
or their hashes, exact rule parses, checker results, symbolic outputs, and
all held-out query scores. Save failures before raising. Formal seed count,
number of revisions, and cap remain fixed after outcomes. New prompts or
budgets after formal collection require a separate experiment.

Verify task split separation and all symbolic labels; mutate held-out targets
and confirm proposal prompts do not change; test malformed JSON, duplicate keys,
bools, wrong key sets/ranges, and valid wrong rules. The analyzer independently
decodes token IDs, reparses rules, reconstructs prompts/checker messages, checks
complete grids, recomputes targets and costs, and estimates intervals over
24 environment means. No held-out outcome may enter an acceptance decision or
repair. Source revisions, packages, model revisions, and data hashes must match.

Success means a complete audited comparison and an honest report, whether
positive, negative, or inconclusive. A positive result would establish only a
conditional benefit from verified computation and this rule representation;
it would not show that feedback alone produces general intelligence.
