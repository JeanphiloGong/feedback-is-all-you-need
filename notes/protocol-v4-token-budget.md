# Study D: generation-budget sensitivity with fixed instructions

Status: prospective follow-up specified after Study C, before viewing any
Study D outcomes. Study C's clean-example 7B composition responses truncated
88.28% of the time. This motivates a budget test, not a conclusion that the
model cannot infer offsets from examples.

Frozen data: `experiments/study_d/data/environments-v1.json`, NumPy 2.2.4,
SHA-256 `356e4439fcb43efea4556a6cea0a5494a814f1bd41307395e602b3885edecdd4`.
All targets, train/test separation, and symbolic recovery were checked before
inference. Formal inference reads the frozen file rather than regenerating it.

## Fixed design

- Use the pinned Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28` with unchanged weights, tokenizer,
  chat template, bfloat16/SDPA, batch size 16, and inherited repetition penalty
  1.05. All model calls are local on lens.
- Freeze 24 new environments, seeds 6000–6023, with the existing generator.
  Each has 16 clean single-command examples, 16 held-out direct queries, and
  32 held-out composition queries. No previous formal environment is reused.
- Conditions: no examples, clean16, and oracle offsets. No self-history, LoRA,
  reflection, or active query selection in this budget-only comparison.
- Use exactly Study C's reasoning prompt and parser, including special-token
  removal before the final-line grammar. Keep all invalid responses wrong.
- Generate once per query with `do_sample=False`, `max_new_tokens=512`.
  Evaluate the completed continuation and its prefix through at most 128
  tokens, trimming each at its first EOS. This yields paired 512/128-budget
  scores without treating duplicate calls as replications. A model may revise
  or invalidate a previously parseable answer, so improvement is not assumed.
- The prefix represents the shorter-budget output because this deterministic
  configuration has no forced EOS, minimum length, maximum time, or beam
  search depending on the length limit. Verify exact prefix equality against
  a separate 128-token call on every development batch, using only seed 11.
  A mismatch blocks formal inference until understood and documented.
  Preserve both token sequences and query/prompt identity for every check,
  including a mismatch before raising an error. The launch workflow must
  inspect a completed seed-11 artifact with all 144 checks passing, matching
  runner/source hashes, model revision, and generation settings. Record that
  gate in the research log before launching formal inference.
- The 0.5B model is excluded: its Study C outputs ended at EOS after two
  tokens, so raising the cap cannot change those completed greedy responses
  under the same instructions. This does not address alternative prompts.

## Endpoints and resource accounting

Primary: environment-paired difference in clean16 composition accuracy,
`accuracy_512 - accuracy_128`. Report a 95% t interval and a 10,000-resample
environment bootstrap interval with RNG seed 820. Do not stop early or alter
the cap based on outcomes.

Secondary: every condition/split/budget mean; clean-minus-none effects at each
budget and their difference; valid-format and truncation rates; transitions
from wrong to correct and from correct to wrong; oracle performance. Secondary
intervals are descriptive. A higher clean score diagnoses a budget constraint
on this interface, not the sufficiency of feedback or general intelligence.

Store all 512-budget token IDs and text, both prefix evaluations, prompt hashes,
model/source/data hashes, generation settings, package versions, actual input
and generated tokens, padded token slots, timings, and peak GPU allocation.
There are 1,152 environment/query pairs, each evaluated under three conditions:
3,456 generation records and 216 actual generation calls. The
128-token evaluations are derived prefixes, so report no invented 128-budget
wall-time measurements or duplicated generation cost. Total 512-budget output
is bounded by 1,769,472 unpadded tokens. Export completed runs from `/tmp` and
retain failed or inconclusive outcomes.

Any remaining truncation is a limit of this registered budget; it is not a
license to continue extending the same experiment until a desirable score
appears. A later mechanism change requires a separate prospective design.
