# Research log

## 2026-10-25: initialization and version 0

- Preserved the initial conversation-derived hypothesis in commit `03df400`.
- Created the GitHub repository and verified the initial commit on its remote.
  The create response did not yield the requested private visibility; a
  subsequent explicit API update set `private=true`. Verify visibility again
  before further pushes.
- Connected to the authorized `lens` host: RTX 4090 D, 24 GB, driver 580.159.03.
  Created a project-specific Python environment; installation is in progress.
  Early SSH connection failures were transient; an authenticated multiplexed
  connection subsequently succeeded. No experiment was restarted.
- Downloaded the original Attention paper and verified its PDF metadata and
  page rendering. Follow its seven-section structure and explicit ablations.
- Found the Chen et al. (2023) title precedent. Verified DOI metadata, but
  publisher PDF attempts returned non-PDF responses. Do not cite its detailed
  arguments until full text is read.
- Saved arXiv metadata and abstracts for eleven foundational or directly
  related papers. These do not constitute a full systematic literature review.
- Drafted version 0 and Study A's prospective protocol. No experimental
  outcomes have been inspected. English commit messages are required.

## 2026-10-25: development run and protocol amendment

- The first smoke command accidentally used registered seeds 1000 and 1001;
  their outcomes were inspected. They are exploratory only. Before any full
  run, move the formal seed set to 2000–2019. No learning-rate or architecture
  tuning followed these outcomes. Future smoke runs explicitly use seeds <100.
- Preserve `runs/study-a-smoke` and its metadata. It lacks a Git revision
  because the remote copy was not a checkout; the corrected runner records
  both a supplied source revision and its own SHA-256.
- Four initial correctness checks passed on lens. A smoke-output check failed
  because the expected count was mistyped as 36 rather than 35 (17 checkpoints
  for each of two arms plus one flushed endpoint); the correct count passed.
- Output directories are now exclusive-create to prevent overwriting runs.
  The missing old-rule uniform-distribution evaluation was added before the
  formal run to match the protocol. Formal data collection remains pending.
- The reference PDF is tracked explicitly. Its ignore rule was repository-local,
  not global; the earlier commentary misidentified it. JSON source files were
  already tracked. Repository visibility was verified as private after the
  explicit update; future pushes must recheck it.

## 2026-10-25: adversarial protocol review

A fresh-context reviewer found four actionable issues before formal collection:
the Fano statement omitted initial-memory/side-information independence; delay
checkpoints were ambiguous; replay sampling was underspecified; XOR label
complementation was optional. All four were corrected explicitly. The runner
now includes a behavioral test for clean/delayed-flush equivalence, and six
tests passed on lens. Cross-model review was skipped because this is the
user-requested autonomous run; no external reviewer CLI was invoked.

The review's replay counting concern is addressed by reporting arriving examples
as `checkpoint * 32` and storing fresh examples used and replay examples used
separately. All outcomes from development seeds remain exploratory.

## 2026-10-25: Study A results and language protocol review

- The registered Study A run completed all 1120 arms in 105.97 seconds on lens.
  The 19,200 rows passed completeness, identical initial scores, frozen scores,
  and delayed-flush equivalence checks. The runner SHA-256 matches metadata.
- Width 16 clean vs width 256 frozen improved by 46.76 percentage points for
  linear and 38.06 for XOR. Against width 256 clean, width 16 differed by
  +0.11 points for linear and -7.67 for XOR. These are conditional adaptation
  results; the larger learner's XOR advantage contradicts capacity irrelevance.
- Local plotting failed due to the system Matplotlib/NumPy ABI combination.
  Analysis succeeded inside lens's pinned environment; no result was discarded.
  Both generated figures were visually inspected.
- A fresh reviewer of Study B required exact LoRA prompt/loss rules, constrained
  self-generated labels, comparisons against no feedback, named random streams,
  a first-token metric label, and the commutativity limitation. All six were
  incorporated before scored inference. Cross-model review remains skipped in
  this autonomous run.
- Five Study B task tests pass. The generated 24-environment dataset is frozen
  with SHA-256 ee556574a989ed8db2caac66030daaf5b6dfc0acbeeb1eca4276d7607983bceb.
- A development-only Qwen 0.5B run on seed 7 completed. Its near-constant output
  motivates inference diagnostics before any interpretation or formal run.

## 2026-10-25 UTC: post-run code audit and replication

- An independent code reviewer found overlapping random streams in Study A.
  Adjacent environment seeds reused 255/256 corruption masks; some initial
  model seeds also collided with other environments' rule seeds. Preserve
  study-a-v1 and its historical analysis; mark its confidence claims superseded.
- Derive streams by SHA-256 over environment, family, role, and step/width.
  Seven tests pass, including collision checks over all formal streams.
  Replicate unchanged settings on untouched seeds 4000–4019 as study-a-v2.
- The language-model code review found no target leakage or scoring-index bug,
  but required peak-memory accounting, actual package versions, and a hash of
  the actually executed development environment. These were added.
- The 0.5B checkpoint's SHA-256 exactly matches the official blob hash.
  CPU float32, GPU float32 and bfloat16, and eager/SDPA inference agree on the
  surprising calibration response. This is evidence against file corruption
  or a device-only numerical fault, not a guarantee of general runtime validity.
- The seed-7 LoRA development run completed. Its saved adapter encountered a
  nonfatal remote-config lookup failure; vocabulary was unchanged. Formal
  saving will explicitly omit frozen embeddings and run offline.

## 2026-10-25 UTC: independent replication and pretrained-model runs

- Study A v2 completed 1,120 arms / 19,200 checkpoints in 109.54 seconds,
  with source hash verified. Primary small-clean vs large-frozen differences
  are +52.91 points for linear and +37.32 for XOR. Under the same clean
  feedback, small-minus-large is 0.00 and -10.93 points respectively.
- Manuscript v2 reports the corrected run; v1 retains an audit notice. Both
  compile without final-pass LaTeX warnings, and result pages were inspected.
- All four language-model context runs and 24 LoRA environments completed.
  The remote disk then filled during export. The partial export is excluded;
  original run metadata and data remained complete. Export and analysis moved
  to /tmp, which is a separate filesystem with sufficient free space.
- Actual context runtimes were 21.81, 56.22, 109.02, and 234.69 seconds for
  0.5B, 1.5B, 3B, and 7B; the LoRA suite took 94.65 seconds. Source hashes
  match the archived runner. The run scripts and packages remained unchanged.

## 2026-10-25 UTC: manuscript v3 evidence review

- The Study B primary 0.5B composition contrast is 0.00 points, t CI
  [-1.03, 1.03], bootstrap [-0.78, 1.04]. The registered result is negative.
- A fresh manuscript reviewer required two material reporting corrections:
  logistic regression's chance XOR result must not be confused with the
  optimal zero-one accuracy of a biased linear classifier; LoRA's direct
  score drops from 26.30% without examples to 9.90% with correct context.
  Both corrections and the paired direct-gain interval are now in v3.
- The same review corrected Study C interpretation: a positive interaction
  cannot establish helpful feedback without examining generation's own
  clean-minus-none difference. This was amended before collection.
- Study C uses untouched seeds 5000–5023 and tests 128-token generation against
  the original interface. No generation outcome has yet been inspected.

## 2026-10-25 UTC: generation-interface development and source freeze

- The local HTTPS proxy failed TLS, while direct verified HTTPS succeeded.
  Per-command proxy removal restored GitHub access; private visibility was
  verified and commits through v3 were pushed. Global proxy settings were not
  changed. SSH handshake failures subsequently cleared; no C job had started
  during the failed connections. The GPU was idle before development resumed.
- Froze 24 Study C environments with NumPy 2.2.4, SHA-256
  `1b30fb9836729e2775b6db993785cc6de7a90ce9a0cb3c6133d762a2606c6bb7`.
  All targets, train/test separation, and symbolic recoverability were checked.
- Seed-9 development on 0.5B returned literal `N` plus EOS under the reasoning
  prompt. No prompt was tuned to improve this outcome. Separate ordinary
  calibration questions produced coherent answers, including correct 2+2,
  whereas the one-digit instruction reproduced the wrong answer 1. This
  demonstrates prompt sensitivity, not a general arithmetic incapacity.
- The adversarial generation-code review required retaining raw token IDs and
  specifying special-token handling. The runner now saves tokens through EOS,
  all generation defaults and overrides, and padded generation work. Parsing
  explicitly strips tokenizer special tokens before applying the final-line
  grammar. The inherited repetition penalty is 1.1 and is disclosed.
- Parser, prompt, EOS/padding accounting, and task checks passed (eight tests).
  Cross-model review was skipped in this user-requested autonomous context.
  Formal Study C seeds remain unscored at this source freeze.

- During formal collection, analysis review found a documentation error: the
  inherited repetition penalty is 1.1 for 0.5B and 1.05 for 7B, as shown by
  development metadata. The protocol text was corrected with an explicit
  after-launch note. The rule to inherit checkpoint defaults and the frozen
  runner were unchanged. Analysis now checks model/revision identities,
  decodes raw tokens against saved text, and requires EOS or the token cap.

## 2026-10-25 UTC: Study C completion and audit

- Both runs completed without model restarts: 28.92 seconds for 0.5B and
  1,831.46 seconds for 7B. Each produced 9,216 held-out predictions and 768
  self-history records. All model/revision, source, environment, grid, raw
  decoding, EOS, and answer-score checks passed. Local independent aggregation
  reproduced all means and the primary t interval.
- The primary interaction is -0.52 points, t CI [-1.16, 0.11], bootstrap
  [-1.17, 0.00]. Every small-model generation was literal N plus EOS. Its
  failure is an interface result, not a general reasoning-capacity estimate.
- 7B oracle generation reaches 100% direct and 97.27% composition accuracy.
  With correct examples, composition accuracy is 6.90%, and 88.28% of those
  outputs hit the token cap. Report this material budget limit and register
  any longer-budget follow-up separately.
- SSH handshake failures delayed copying completed results. The first SCP
  directory-copy syntax was rejected; a tar transfer succeeded, with every
  remote archive hash matching locally. Raw remote outputs remain intact.
- A reference PDF was initially classified as text by Git; its unchanged
  streams triggered whitespace diagnostics. PDF binary attributes resolved
  the false positives without altering any source PDF. Text-only checks and
  subsequent reference-diff checks passed.

## 2026-10-25 UTC: manuscript v4 review

- Version 4 integrates all three studies, the memory counting bound, and a
  targeted full-text review. A fresh reviewer independently checked the new
  numerical results and theoretical bound and found no substantive error.
  Its suggested clarification, "clean-example" truncation in the abstract,
  was incorporated. Cross-model review was skipped in this autonomous run.
- The 11-page PDF compiled twice without final-pass warnings. All pages were
  rendered; layout, new results, equations, figures, and references were
  inspected. Earlier versions and raw failed responses remain intact.

## 2026-10-25 UTC: fixed-budget follow-up preparation

- Study D freezes 24 new environments (seeds 6000–6023) and the same 7B
  reasoning interface. It compares 128/512-token answers from shared greedy
  continuations, with clean examples, no examples, and oracle rules. It tests
  the material truncation limit without changing prompts or update mechanisms.
- All frozen targets, symbolic recovery, and train/test separation passed.
  Two prefix tests passed, including a case in which further generation makes
  a correct prefix invalid; improvement is not assumed.
- The first seed-11 development run completed all 144 direct-versus-prefix
  equivalence checks. A subsequent review required persisting every compared
  token pair before an assertion and explicitly enforcing the development
  prerequisite in the launch workflow. The updated development run must pass
  with matching source/configuration before formal collection. Both development
  versions will be retained; this change affects auditing, not generation.

- The updated seed-11 run completed in 79.95 seconds. Before formal launch,
  the workflow verified all 144 saved prefix matches, completion metadata,
  source hashes, revision `8507524`, pinned 7B identity, and exact generation
  settings. The prerequisite passed. No Study D formal outputs were viewed
  before this gate.

## 2026-10-25 UTC: Study D completion and changed interpretation

- The fixed 512-token run completed all 3,456 continuations in 216 batches,
  taking 1,436.61 seconds. All source/model/data/grid/prompt/token/score/batch
  checks pass; every transferred hash matches the remote manifest. Independent
  local aggregation matches all means and the primary interval.
- Clean-example composition accuracy rises from 7.94% at the 128-token prefix
  to 70.57% at 512 tokens: +62.63 points, t CI [56.10,69.16], bootstrap
  [56.38,68.75]. The no-example score also rises from 0% to 14.06%.
- Clean composition truncation falls from 87.76% to 3.26%; 481 queries become
  correct and none become incorrect. This materially revises the interpretation
  of short-budget failures. It is an inference/interface result, not evidence
  of ongoing parameter learning or the sufficiency of feedback for intelligence.
- The analysis review required explicit presence of all four source hashes,
  prefix-relevant generation settings, and a check of padded input costs.
  These checks were added before interpreting formal results; no rerun was
  required. Cross-model review was skipped in this autonomous run.

- All 481 recovered clean composition responses had invalid shorter prefixes.
  Report this decomposition: the result establishes a completion-budget effect
  on the measured use of examples, not 481 corrections of valid wrong answers.

## 2026-10-25 UTC: manuscript v5 review

- A fresh reviewer independently reproduced the Study D means, paired t and
  bootstrap intervals, invalid-prefix transitions, development equality checks,
  and costs; no substantive reporting error was found. The reviewer could
  reparse saved text but lacked the local tokenizer cache for decoding token
  IDs again; the full remote decoding audit had passed.
- The new attribution was checked against the archived Era of Experience
  text. Cross-model review was skipped in this user-requested autonomous run.
- Version 5 compiles twice without final-pass warnings. All 13 pages were
  rendered and inspected, including the new result table and figure. Earlier
  versions remain intact. Study D and the manuscript are separate commits
  because the data audit and paper snapshot are independently reviewable.

## 2026-10-25 UTC: online-feedback mechanism preparation

- Study E compares context retention, one LoRA update per verified outcome,
  and updating only after a wrong prediction. It uses 24 fresh environments,
  seeds 7000–7023, with the constrained 0.5B interface. Only 16 feedback events
  are used; no claim about optimal schedules or lifelong learning is intended.
- Initial review found mismatched adapter initialization between conditions,
  missing pre-update failure records, and analyzer/source-path inconsistencies.
  All were corrected before any Study E model execution. An injected optimizer
  failure first reproduced the loss of the pre-update prediction; its regression
  test now passes with the prediction retained. No formal results were viewed.
- All 24 frozen task splits, targets, and symbolic solutions passed. Local
  E/task tests passed (13); the local PEFT reset test could not collect because
  PEFT is absent. The authorized lens environment ran all 14 tests successfully,
  including two actual adapter train/unload cycles with exact base equality.
  PEFT emitted its existing peft_config warning on reattachment; tensor equality
  passed. Development will also hash the full pretrained base after each arm.
- The standalone generator's first invocation used a file path and failed to
  resolve its package; invoking it as a module succeeded. No model data was
  collected during that failure. Frozen JSON creation now refuses overwrite.
- The second fresh review found no remaining substantive issue. An isolated
  synthetic full-grid audit produced all 42 summaries and 12 contrasts with
  environment-level n=24. Its reproduction-path correction was applied; no
  model outputs were generated for that check. Cross-model review was skipped
  in this autonomous run. Newly introduced unused imports were removed.

- Before formal Study E launch, seed 13 completed in 5.92
  seconds and passed the full audit: 1,008 evaluation rows, 48 ordered
  prediction/feedback pairs, initial adapter equality, all saved adapters, and
  exact base recovery after each condition. The transferred development files
  and analysis/source hashes match revision 84f2e52. Formal source/model/config
  gate is enforced by the runner; no formal Study E predictions exist yet.

## 2026-10-25 UTC: Study E completion

- All 24 environments finished in 72.80 seconds after loading. The full audit
  passed 24,192 evaluations, 1,152 prediction/feedback pairs, all 48 adapters,
  model and source identities, reset evidence, prompt separation, and costs.
  All 152 transferred artifact hashes match; local independent aggregation
  reproduces all 42 means and the primary t interval.
- The primary one-update-per-outcome composition change is -0.65 points,
  t CI [-4.04,2.74], bootstrap [-3.78,2.47]. It is a failure to demonstrate
  improvement under this 16-event, constrained-output schedule. Error-only
  updating also has no demonstrated benefit: -2.34 points, CI [-5.53,0.85].
- Per-outcome LoRA changes 99 composition responses from wrong to correct and
  104 from correct to wrong. Adding correct context reduces direct accuracy
  from 13.28% to 5.47%; this secondary context dependence is retained.
- Both arms have identical initial adapters and unchanged base parameters.
  The per-outcome and error-only arms apply 384 and 337 updates, respectively;
  their different costs are measured and are not described as matched. All
  final adapter files are archived along with raw records and manifests.

## 2026-10-25 UTC: manuscript v6 review

- A fresh manuscript reviewer independently reproduced all new means and
  intervals, contrasts, transitions, bootstrap, and update costs, and checked
  all 48 adapter manifests. No substantive issue remained. Cross-model review
  was skipped in the authorized autonomous context.
- Version 6 retains the five studies and all earlier versions. It explicitly
  distinguishes changed predictions from useful retained improvement and
  reports the small-model null result without a general incapacity claim.
- The 15-page PDF compiles twice without final-pass warnings; all pages were
  rendered and inspected, including the new table, curves, and conclusion.
  A line overflow in the result paragraph was fixed before the final render.

## 2026-10-25 UTC: rule-memory experiment preparation

- Added selected-section reviews and exact source records for Voyager,
  Self-Debugging, and ExpeL. Independent review corrected simulated versus
  observed traces, SQL execution observations, and the target-example/GPT-4
  conditions on ExpeL transfer. The three unmodified PDFs are archived.
- Study F freezes 24 fresh languages, seeds 8000–8023, plus development seed 17.
  It compares two fixed self-review or externally checked revisions of a
  shared initial rule, using a supplied executor and no hidden test inputs
  or targets in prompts. No model calls have occurred at this checkpoint.
- Protocol review exposed a composition alias: adding four to every offset
  makes all direct answers wrong but preserves every two-command result.
  Exact-rule recovery and nonexact perfect-composition diagnostics prevent
  interpreting that score as faithful memory. The primary t-interval rule
  and paired percentile bootstrap are specified before inference.
- Implementation review reproduced a 512-token malformed bracket response
  that raised RecursionError. The boundary now scores it as invalid. Raw
  outputs from the whole completed batch are persisted before checking any
  row; an injected first-row failure verifies that all 16 proposals survive.
- Local rule tests pass (9). A local analyzer smoke import was unavailable
  because python3 -s cannot import Transformers; it is not a passing analysis
  check. The existing remote environment passes 16 tests and 10 subtests.
  Analysis now verifies the runtime versions of all five scientific/model
  packages; source/configuration and development evidence are also checked.
- No production abstractions, adapters, or compatibility layers were added.
  The test-only batch/model fixtures support the injected-failure check.
  Cross-model CLI review is skipped in this autonomous research run.
- The final review also reproduced a setup-failure gap. Provenance is now
  saved before tokenizer/model loading; an injected cache failure retains
  incomplete metadata and the exception record. The narrow correction passed
  independent review. A synthetic full-grid audit checked forks, final scoring,
  package rejection, Wilson/transition intervals, and JSON serialization.

- Study F development completed five proposals and 240 evaluations per model.
  The full audits pass and transferred hashes match frozen source `08725bb`.
  0.5B took 1.11 seconds after loading. The first 7B launch failed before
  inference because an optional adapter lookup attempted a metadata write on
  the full root filesystem. Its metadata, traceback, and failure record are
  retained. An offline retry using the same cached pinned weights completed
  in 2.29 seconds; both formal runs will use the same offline environment flags.
  No prompt, cap, source, or model parameter was changed after development.

## 2026-10-25 UTC: Study F completion

- Both frozen suites completed: 120 proposals, 120 retained rule files, and
  5,760 held-out evaluations per model. Full remote audits pass; all 258
  transferred manifest entries match. An independent local parser and scalar
  executor reproduce 20 means, 12 contrasts, bootstrap, Wilson intervals,
  answer transitions, and generation costs. The two figures were inspected.
- The primary 0.5B checked-minus-self composition effect is +4.04 points,
  t CI [-2.78,10.85], bootstrap [-1.56,11.07]. No small-model proposal recovers
  all four rules. This is inconclusive for a composition benefit and negative
  for observed exact recovery under the tested interface, not a universal
  impossibility result. The 7B final exact counts are 6/24 checked and 3/24 self.
- All responses finish before the token cap. Invalid 0.5B proposals repeat
  offsets; nine of the 12 invalid 7B proposals instead use negative modular
  representatives disallowed by the declared interface. Raw outputs are
  retained and scores are not silently normalized after collection.
- The measured post-loading suites take 2.76 and 12.50 seconds; shared initial
  generation is counted once. Storage and modular execution remain supplied
  system components. HTTP pushes failed twice with TLS termination errors;
  the authenticated SSH transport successfully synchronized through e195e6d.

## 2026-10-25 UTC: manuscript v7 review

- A fresh reviewer independently recomputed all new raw-rule scores, 20 means,
  12 contrasts, primary bootstrap, Wilson/transition intervals, prompts, hashes,
  and costs. No substantive reporting issue was found. The review reparsed
  decoded text; remote analysis performed token-ID decoding. Cross-model CLI
  review was skipped in this autonomous run.
- The review identified a useful concrete interface example: seed 8012's final
  checked 7B rule is semantically exact with -1 interpreted as 7, while its
  registered canonical-format score is zero. The manuscript discloses this
  without changing formal scores or overstating neural incapacity.
- Version 7 incorporates Study F and the three new literature precedents.
  All earlier snapshots remain intact. The 18-page PDF compiles twice without
  final-pass warnings; all pages were rendered and inspected. A line overflow
  and an isolated concluding line were removed before the final render.

## 2026-10-25 UTC: exploratory Study F mechanism diagnostics

- After v7, a separate post-hoc plan and analyzer describe all 192 revision
  transitions. Four behavioral tests pass; fresh independent review checked
  every transition. No new inference or formal-score changes were made.
- In 0.5B, self-review repeats text in 47/48 transitions; checked review repeats
  it in 26/48, with the final response unchanged in 16/24 environments. Among
  32 valid incorrect-to-valid checked transitions, three repair the first
  failing command. Five offsets become correct and six become wrong.
- These counts motivate rejection memory and preservation of verified progress
  as future interventions. They are not evidence of permanent cycles or a
  causal claim for an untested memory mechanism. The 48 transitions are nested
  within 24 environments and are never treated as independent replicates.

## 2026-10-25 UTC: feedback-history experiment preparation

- Study G compares last-only versus full proposal/check history for 16 fixed
  revisions on 48 fresh environments per model, seeds 9000–9047. Development
  uses seed 19. Its primary endpoint is 0.5B final direct accuracy; composition
  is secondary because of the demonstrated two-command alias. No G model
  inference has occurred at this checkpoint.
- Protocol review required matching ordered first-revision batches, padded
  token IDs, masks, and generated outputs before attributing later changes
  to history. Any mismatch preserves evidence and fails the suite. Repetition
  counts distinguish invalid, valid-wrong, and exact states.
- Implementation review found that input hashes alone would lose the actual
  mismatching tensors, and that stage costs omitted explicit standalone-arm
  totals. Both were corrected before inference. A synthetic audit rejected
  mutations of IDs, masks, seeds, digest, and batch order; scalar verification
  confirmed 1,584 actual continuations versus 816 per hypothetical standalone
  arm, including its initial generation.
- Local history/parser/diagnostic tests pass (18). The first remote combined
  run lacked an inherited F data fixture, yielding one missing-file failure;
  copying that unchanged fixture resolved the packaging omission. The complete
  suite passes 24 tests and 12 subtests, including injected first-revision
  input/output divergence with raw evidence retained. No implementation layer
  or compatibility wrapper was added. Cross-model CLI review was skipped in
  this autonomous run.

- Before formal G inference, both seed-19 runs completed 33 proposals and
  1,584 evaluations, with exact first-revision batch inputs/masks/output
  equality. Full audits pass; all 82 transferred manifest entries and source
  hashes match `c4b4054`. Post-loading elapsed times are 6.50 and 19.33 seconds.
  Actual/standalone continuation counts are 33/17 per development language.

## 2026-10-25 UTC: Study G completion

- Both frozen formal suites completed 1,584 proposals, 1,584 saved rules,
  and 76,032 held-out evaluations each. Full remote audits pass; all 3,188
  transferred manifest entries match. Actual first-revision seed batches,
  input IDs/masks, and generated IDs are identical across branches. Every
  proposal finishes at EOS before the cap. No model rerun was needed.
- The primary 0.5B full-history-minus-last-only direct effect is 0.00 points,
  t CI [-3.96, 3.96], bootstrap [-3.65, 3.65]. Both branches finish 3.13 points
  below initial direct accuracy, with intervals containing zero. No exact
  small-model rule appears. Secondary composition declines have descriptive
  intervals below zero; they do not replace the registered primary endpoint.
- The secondary 7B direct contrast is +0.52 points, CI [-11.69, 12.73]. Final
  exact counts are 5/48 last-only and 9/48 full-history. The latter gains eight
  exact rules but loses three of the four initially exact rules; final count
  and retention of initial successes must remain distinct. All repetition
  and component transitions remain nested within 48 environments per branch.
- Independent raw parsing/arithmetic reproduces 132 means, 12 contrasts,
  primary bootstrap, 66 diagnostic rows, Wilson intervals, answer/exact-rule
  transitions, repetition/component records, first-revision tensors, and
  actual/standalone costs. An initially over-tight comparison using local
  SciPy 1.8.0 failed on a roughly 7.6e-11 interval-bound difference from remote
  SciPy 1.16.3. Independent CDF root solving reproduced the recorded quantile
  at the original tolerance; no formal result or source was changed.
- Post-loading elapsed times are 42.08 and 228.10 seconds. Each standalone
  branch accounts for 816 continuations including its initial proposal;
  actual suites count the shared initial generation once. Full history uses
  2.43/2.29 times the standalone input tokens in 0.5B/7B. This is not a
  matched-compute experiment. Both result figures were inspected.
- Fresh results review found a mislabeled retention denominator in the new
  README: 123/154 counts all transitions from exact 7B full-history rules,
  including 18 invalid successors, rather than only valid-to-valid pairs.
  The label and invalid count are now explicit; no numerical result changed.
  No material findings remain. Cross-model CLI review was skipped in this
  authorized autonomous run.

## 2026-10-25 UTC: manuscript v8 review

- Version 8 adds the complete-history comparison and separates Study F's
  post-hoc motivation from Study G's prospective endpoint. It reports the
  primary interval, regressions, exact-rule recovery versus retention,
  correctness-stratified repetition, and the extra cost of full history.
- Fresh manuscript review checked the added methods, raw-rule arithmetic,
  uncertainty, costs, figures, and changed interpretation against the frozen
  protocol. No actionable scientific-reporting issue remained. Cross-model
  CLI review was skipped in this authorized autonomous run.
- The 22-page manuscript compiles twice at its final revision without warnings.
  All pages were rendered and inspected; a stranded result line was removed
  and the changed final pages were rendered and inspected again. All earlier
  manuscript snapshots remain intact. No implementation abstraction was added;
  compilation and rendering intermediates stay outside the tracked archive.

## 2026-10-25 UTC: exploratory selection analysis preparation

- After completing v8, specified a separate post-hoc selector over Study G's
  fixed candidate pools. It ranks only disclosed-example accuracy, keeps the
  earliest maximum at each prefix, and retains the strict invalid-rule policy.
  This measures available candidates discarded by final-proposal scoring,
  not the unobserved trajectory of a model with an acceptance gate.
- The plan explicitly identifies why direct-score monotonicity follows from
  the supplied balanced additive family, while composition can decline.
  Extra full-example checks and offline analysis time are recorded separately.
- Six focused tests pass, covering ties, regression recovery, invalid rules,
  prefix causality, hidden-data independence, and the composition alias.
  Fresh independent review found no substantive issue in the plan or code.
  Cross-model CLI review was skipped in this autonomous context. Only direct
  research analysis functions were added; no compatibility layer or temporary
  implementation was introduced.

## 2026-10-25 UTC: exploratory selection outcomes

- The supplied training-example selector retrieves exact rules from 10/48
  last-only and 17/48 full-history 7B pools, versus final counts of 5/48 and
  9/48. The full-history direct score becomes 70.31% instead of 47.92%.
  No 0.5B pool contains an exact valid rule; its selected direct scores are
  17.71% last-only and 19.27% full-history. These fixed-pool descriptions
  cannot predict the candidates generated under a different feedback policy.
- All groups choose an earlier stage in 48/48 environments, including earliest
  ties. Per-environment direct scores cannot fall under this supplied task
  structure, but individual answers and composition scores can regress.
  Composition declines in 5, 4, 2, and 3 environments in model/branch order.
- The analysis adds 9,056/8,464/12,032/11,664 full-example executions across
  the four standalone pool groups. It takes 0.668 seconds locally before
  serialization, excluding imports and all prior generation. Input and
  source hashes are archived with 1,152 records and 24 summaries.
- Independent calculation from audited rule memories reproduces every record
  and summary; fresh results review found no material issue. Cross-model CLI
  review was skipped. No formal scores, source freezes, or earlier manuscripts
  changed, and no additional model inference was performed.

## 2026-10-25 UTC: uniform-proposal reference preparation

- After observing selection outcomes, specified two exact post-hoc references:
  17 uniform legal proposals, and the shared neural initial proposal followed
  by 16 uniform proposals. Both use the supplied training-example selector;
  neither changes any frozen model trajectory. The latter retains the neural
  initial call and must not be called a fully model-free system.
- Enumerating the 1,680 legal rules yields the correct-offset distribution;
  a tail-sum formula gives conditional selected-direct expectations and
  exact-recovery probabilities with rational arithmetic. The sampler builds
  the grammar into its action space and always returns valid rules. Equal
  candidate counts do not match inference or checker costs.
- Five tests pass, including independent inclusion-exclusion, single-draw
  marginals, and a weighted two-draw calculation. Fresh review found no issue
  in the formula, initial-rule conditioning, implementation, or attribution.
  Cross-model CLI review was skipped. The added research functions introduce
  no compatibility layer or temporary implementation.

## 2026-10-26 UTC: uniform-proposal reference verification

- The exact match-count distribution is 1,001/536/126/16/1 across zero through
  four correct offsets. Seventeen uniform legal proposals with supplied
  selection have expected direct accuracy 48.71% and exact recovery 1.01%.
  These are policy expectations, not observed random-agent runs.
- Keeping the shared neural initial rule and adding 16 uniform revisions
  gives mean expected direct accuracy 48.47% for the 0.5B initial states and
  63.84% for the 7B initial states. Observed selected full-history pools score
  19.27% and 70.31%. Expected exact counts are 0.46 and 4.42, versus observed
  selected counts of zero and 17. All comparisons remain post-hoc/descriptive.
- The random reference supplies a valid rule grammar and uses more example
  executions than the observed pools: 12,992/13,008 per conditional group.
  The pure reference has no neural initial call and uses 13,056 checks per
  48 environments. Its 0.037-second offline expectation calculation is not
  a runtime measurement for a deployed random agent.
- Independent rational dynamic programming from the original audited rule
  memories reproduces the pure reference, all 96 conditional records, four
  comparisons, and cost totals; input/source hashes match. Fresh reporting
  review found no numerical or attribution issue in the result section.
  Cross-model CLI review was skipped. No new model calls or changes to
  earlier formal scores, sources, or manuscripts occurred in this analysis.

## 2026-10-26 UTC: manuscript v9 review

- Version 9 adds the two sequentially declared post-hoc analyses without
  changing Study G's original endpoints. It distinguishes selected candidates,
  exact-rule availability, externally supplied verification, and uniform-policy
  expectations from neural learning or a measured interactive intervention.
  The abstract was condensed to retain the main evidence and its limits.
- Fresh review found two wording ambiguities: the selection boundary could
  be read as excluding training targets, and the abstract's 19.27% score did
  not name the full-history branch. Both were corrected, and narrow follow-up
  review confirmed resolution. No numerical or inferential finding remained.
  Cross-model CLI review was skipped in the authorized autonomous context.
- The final 23-page PDF compiles twice without warnings. All pages were
  rendered and inspected; paragraph pagination was adjusted and every changed
  page inspected again. Earlier versions remain unchanged. No architectural
  abstraction or compatibility layer was added; build/render files are
  disposable and excluded from the manuscript archive.

## 2026-10-26 UTC: DreamCoder source review

- Archived the unmodified 22-page arXiv 2006.08381v1 PDF with its exact title,
  source, retrieval time, byte count, and SHA-256. Read the main mechanisms,
  selected task setups, quantitative comparisons, and discussion; visually
  checked seven relevant pages. The separate supplement was not reviewed.
- The method learns a reusable program library and a neural search model,
  using repeated search, replay, and synthetic tasks. The record distinguishes
  these mechanisms from transcript memory and preserves the reported CPU
  costs, supplied primitives, domain-specific setups, and limited transfer scope.
- Fresh independent review found that the draft overstated the task-generation
  limitation. Corrected it to purposeful task selection/generation: random
  dream-task generation is already implemented. Cross-model CLI review was
  skipped in the authorized autonomous context. Manuscript v9 remains frozen;
  this precedent will enter a subsequent version with the relevant experiment.
- Source size/hash and manifest consistency were checked. This documentation
  change adds no implementation abstraction, compatibility layer, or temporary
  implementation. Temporary PDF extraction/render files remain outside Git.

## 2026-10-26 UTC: cross-task library protocol

- Specified Study H before any development or formal task outcomes: a supplied
  symbolic searcher learns flat fragments from accepted experience programs,
  then freezes them for new tasks. Four arms separate no memory, whole-program
  memory, compressed fragments, and renamed fragments. No neural model is used.
- New tasks have explicitly shared motif structure or a shifted random-program
  distribution. Hidden labels and latent generating programs cannot enter
  search or memory induction. All calls pay for expanded primitive execution;
  prior experience, compression, memory, and interpreter costs remain separate.
- The primary paired comparison uses 32 frozen environment seeds and a fixed
  65,536-operation cap. Independent protocol review found no material issue.
  Cross-model CLI review was skipped. Seven hand-built unit checks of the core
  search and compression routines passed while the protocol was being reviewed;
  no development environments or formal outcomes had been generated.
- The protocol adds no software layer. The initial implementation uses direct
  research functions and flat tuples; no compatibility interface is planned.

## 2026-10-26 UTC: library-search core verification

- Implemented the finite task generator, first-fit enumeration with expanded
  primitive-operation accounting, minimum-token segmentation, and greedy flat
  fragment induction. Only visible examples enter search; only accepted
  programs enter compression. Full-function split novelty is enforced by the
  benchmark generator, independently of learner success.
- Ten focused tests pass. Fresh core review additionally compared 2,880 search
  outcomes/counts, 20,145 segmentations, and 500 compression outcomes with
  independent calculations, checked the four-entry limit, and inspected seed-23
  generation. It found no material issue. No formal searches were performed.
  Cross-model CLI review was skipped in this autonomous context.
- The core adds direct research functions, not a software abstraction layer or
  compatibility interface. No temporary implementation was introduced. Runtime
  collection and result auditing are a subsequent implementation slice.

## 2026-10-26 UTC: library collection and audit implementation

- Added collection journals, fixed memory controls, independent search/compression
  replay, paired analysis, cost summaries, and reproduction instructions.
  Search takes primitive tables plus visible examples; accepted programs feed
  memory; the separate evaluator reads held-out labels only after search ends.
- Fresh review found five actionable defects before the official development
  run: a completed search could be lost if evaluation failed; initialization
  errors escaped the failure archive; negative bootstrap/t-interval disagreement
  was missed; truncated seed grids could pass; and truncated or reassigned task
  families could pass. All were corrected with direct changes and regression
  tests. Raw searches now flush before evaluation into a separate journal;
  audit requires complete grids and exact deterministic task generation.
- All 18 tests pass. Narrow follow-up reviews confirmed resolution, checked
  phase ordering and false-fit retention, and exercised journal reordering,
  missing/duplicate rows, and mismatched keys. The auditor's temporary seed-23
  fixture covered 152 searches and was deleted by its TemporaryDirectory after
  the check; it is transient test evidence, not an archived development run.
  The official development suite will be retained after source/data freezing.
  No formal search has run. Cross-model CLI review was skipped.
- The two journals represent actual search and evaluation lifecycle boundaries.
  No DTO, compatibility layer, or temporary migration path was added. The
  original duplicated collection loops were consolidated within the collector;
  obsolete combined-record construction was removed before the first commit.

## 2026-10-26 UTC: Study H data freeze

- Archived development seed 23 and formal seeds 10000–10031: 33 environments,
  1,848 tasks, 2,378,541 bytes, SHA-256
  `cea3c10966a5b8e48225afaa2bd42722c6bb0cdfa82be6da5d9802d17118d432`.
  All full-function labels and split uniqueness checks pass; no experience
  task was omitted for duplicate semantics. Regeneration reproduces the data.
- Generator source is unchanged from core commit 8410cf1; collection and audit
  are frozen at 9e34d23. No official development or formal search outcome was
  used to select this data. The large generated JSON is a separate data commit
  so that it does not obscure the reviewed implementation.

## 2026-10-26 UTC: archived Study H development

- Ran the official seed-23 suite from source/data freeze c582467: 152 searches
  and evaluations, 4.348 seconds including journaling. Independent replay and
  exhaustive compression audit passed, taking 14.181 seconds with analysis.
  All seven raw/audit files were archived byte-identically with a manifest.
- At the fixed budget, composition exact counts are 8/0/14/0 of 16 for
  none/whole/compressed/renamed memory. Shifted counts are 8/0/0/2. These are
  single-environment development observations, not formal effect estimates.
  No configuration, seed, task, or implementation changed after inspection.
- Started the full 32-environment formal suite using the matching development
  audit. The audit did not trigger a design amendment or a favorable rerun.

## 2026-10-26 UTC: formal cross-task compression outcomes

- Completed all 32 environments once from source/data freeze c582467: 768
  experience searches and 4,096 evaluation searches. Independent full-output
  replay and exhaustive compression checks passed every search, memory, cost,
  and score. Collection took 127.832 seconds; audit/analysis 429.540 seconds.
  All 38 formal raw/audit files were archived byte-identically with a manifest.
- Composition exact success is 258/512 none, 12/512 whole, 436/512 compressed,
  and 75/512 renamed. The primary paired compressed-minus-whole effect is
  +82.8125 points, t95 [77.1176, 88.5074], bootstrap95 [77.3438, 88.0859].
  Every environment contrast is positive. No accepted evaluation program is
  wrong on hidden inputs. All 768 experience solutions match their full functions.
- The unfavorable distribution change is retained: shifted success is 256/512,
  6/512, 51/512, and 62/512. Across the balanced evaluation mixture, compressed
  memory solves 487/1,024 compared with 514/1,024 without memory. This result
  concerns supplied symbolic search, learned flat representations, and deliberate
  shared structure, not neural learning or general intelligence.
- Shared experience costs 3,110,069 primitive operations. Each standalone memory
  arm is charged this cost; the renamed arm also pays compression. Actual
  collection uses 212,290,956 operations with experience counted once. All
  library contents, storage, candidate counts, comparison costs, timings, and
  budget curves are reported; the cap is not total FLOPs or lifetime compute.
- Fresh results review independently checked outcomes, intervals, compression,
  costs, and archive completeness and found no material issue. Cross-model CLI
  review was skipped. No formal rerun, protocol amendment, software abstraction,
  or compatibility layer was introduced in this results checkpoint.

## 2026-10-26 UTC: manuscript v10 review

- Version 10 adds the verified DreamCoder precedent and Study H's fixed
  cross-task comparison. The abstract, methods, results, and interpretation
  distinguish symbolic representation learning from neural updates, disclose
  supplied task structure/search, and retain unfavorable shifted-task outcomes
  and the lower combined success versus no memory. Earlier manuscripts remain
  frozen. The manuscript now covers eight studies in the original seven-section
  organization.
- Fresh scientific review found that the rewritten abstract omitted Study G's
  comparator. Restored the latest-proposal-and-check arm for both the zero-point
  accuracy difference and doubled input cost; narrow follow-up confirmed the
  finding resolved. No other material scientific issue remained. Cross-model
  CLI review was skipped in the authorized autonomous context.
- The final source compiles twice without warnings into 26 pages. All 26 pages
  were rendered and inspected for text, equations, tables, plots, references,
  margins, and pagination. The archived PDF is byte-identical to the inspected
  build, SHA-256 0665470322a8104b24120a88f9014a9b376597e7e94f9217caa9b1062ca2b4a3.
  Build/render intermediates and Study H bytecode caches were removed. No new
  software abstraction, compatibility layer, or temporary implementation was
  added in the manuscript change.

## 2026-10-26 UTC: DreamCoder capacity source audit

- The printed supplement URL returned 404 and the ACM endpoint returned 403.
  Internet Archive lookup failed through TLS/timeout errors; this is not
  evidence that no archived supplement exists. The arXiv v1 source package
  contains an appendix cross-reference file but no supplement body. Its hash
  and member list are recorded; the package itself is not added to the repository.
- Preserved five unmodified source files from the authors' historical commit
  e6fd72804194ed6cc0c88eee652413b0b44fc63e (2020-04-25). Their byte counts,
  SHA-256 values, and Git blob hashes match the manifest. The snapshot's link
  to the publication's actual configurations/checkpoints remains unestablished.
  A 64-dimensional feature is not a complete neural parameter count.
- Derived a conditional non-contextual count, `64 V + 54,145 + 130 G`, including
  embedding, bidirectional GRU, intermediate layer, grammar head, and auxiliary
  head. Six synthetic PyTorch shape checks passed on the authorized lens host;
  the standalone script and exact output hashes are archived. No external
  DreamCoder code, checkpoint, or training run was executed. These counts are
  not reported as model sizes measured in the publication.
- Fresh source review found an auxiliary-loss flag misinterpretation. Traced
  both loss paths and the optimizer call: `auxLoss=False` detaches features for
  that loss but still trains the auxiliary head. Corrected the note; the
  parameter formula is unaffected. Narrow follow-up review confirmed the
  correction resolved the finding. Cross-model CLI review was skipped in the
  authorized autonomous context.
- The reference snapshot and its audit form one archival slice; most added
  lines are exact external source, not new project implementation. The full
  staged whitespace check flags 165 original trailing-whitespace lines and
  one original blank line at EOF; all are preserved for source fidelity.
  The separate whitespace check on authored artifacts passes.
  No software abstraction, compatibility layer, or temporary implementation
  was introduced. Manuscript v10 and previous experimental records are unchanged.

## 2026-10-26 UTC: neural search-prior protocol

- Declared Study I before generating its development or formal data: a small
  untrained GRU learns from programs accepted by supplied symbolic search.
  Uniform search is the single primary comparator; count-based, frozen,
  shuffled-training, larger-network, intermediate, and shifted-task outcomes
  remain descriptive. The network learns a program prior, not a task encoder.
- Fixed 1,172/13,892 parameters, sequential replay, a common finite program
  space, first-fit acceptance, execution caps, independent environment units,
  and separate training/ranking/storage costs. Fresh seeds reuse Study H's
  unchanged generator without importing its outcomes into this comparison.
- Independent protocol review identified two execution gaps: failed-grid
  disposition and fully specified neural initialization. Added the distinction
  between algorithmic exhaustion and incomplete infrastructure attempts, the
  no-incomplete-primary rule, retained versioned retries/amendments, and exact
  CPU float64 constructor order and initialization. Narrow follow-up confirmed
  both findings resolved. Cross-model CLI review was skipped in the authorized
  autonomous context.
- Five hand-built core-search checks pass while review proceeds, including
  complete finite sorting and 200 independent cost comparisons. These are unit
  fixtures, not development/formal task outcomes. No new benchmark tasks have
  been generated or searched, and no data-dependent design tuning has occurred.
  All nine core checks, including four neural checks, also pass on lens with
  PyTorch 2.8.0+cu126. The initial implementation remains a separate uncommitted
  slice pending its own review.

## 2026-10-26 UTC: neural-prior core implementation

- Implemented the small/large program-prefix GRU, sequential replay with an
  externally retained optimizer, count prior, rounded-cost enumeration, and
  visible-example checking. The model takes primitive prefixes; no hidden
  labels, generating programs, or task features enter these interfaces.
- Nine hand-built checks pass on lens with PyTorch 2.8.0+cu126, Python 3.11.15,
  CPU float64 and one thread. They check real tensor counts, causal inputs,
  padding loss, persistent updates, frozen weights, independent complete finite
  ordering, and 200 full-output cost comparisons including exact-cap and
  interrupted checking. No benchmark environment was generated or searched.
- Fresh independent review found no material issue in this core subset.
  Cross-model CLI review was skipped. Collection, checkpoint persistence,
  full audit, and inference remain separate work, not verified by this slice.
  The GRU class directly implements the experimental model; no compatibility
  layer, temporary implementation, or generic agent abstraction was added.

## 2026-10-26 UTC: neural-prior collection and failure records

- Added collection with three persistent optimizers, matched initial small
  weights, intermediate/final checkpoints, six final priors, and two frozen
  probes. Separate search, evaluation, update, and completed-phase journals
  preserve the observable data flow. Runtime failures include the active-phase
  file reference; official collection still has not started.
- Independent collection review found two missing records: completed phase
  costs could be lost before the final environment summary, and empty corpora
  had no explicit per-attempt skip. Both were fixed with flushed phase records.
  Follow-up exposed parsing of a potentially truncated phase file inside the
  error handler; removed that parsing and retained the raw file by reference.
  The final narrow review confirmed resolution. Cross-model CLI review was
  skipped in the authorized autonomous context.
- All five collection tests pass on lens. Hand-built fixtures show that hidden
  label changes cannot affect training or search, incorrect hidden fits remain
  in the learning corpus, completed work survives evaluator failure, all empty
  attempts are recorded, and a damaged phase file cannot mask the original
  exception. Initialization failure is also archived. Temporary fixture trees
  are removed by the tests; no benchmark environment was generated or searched.
- This checkpoint adds direct experiment collection and evidence files, with
  no compatibility layer or generic agent abstraction. Full audit and analysis
  remain a separate uncommitted review slice; their presence is not an official
  development result or a completed scientific comparison.

## 2026-10-26 UTC: independent neural-prior audit and inference

- Added complete-grid auditing, independent full-function search accounting,
  initial-state regeneration, exact replay of all model and optimizer tensors,
  and NumPy GRU gate equations for candidate ordering. Statistical summaries
  retain the single paired environment-level comparison, shifted/balanced
  outcomes, intermediate probes, and separate actual/standalone costs.
- Independent audit review demonstrated that extra unrecorded keys in cost
  maps could inflate actual totals while passing per-entry checks. Replaced
  those checks with complete map equality against declared phase evidence;
  actual aggregate timings and prediction counts now come from the verified
  phase ledger. A new hand-built complete-audit fixture passes normally and
  rejects each of four separate injected cost entries. Narrow follow-up review
  confirmed resolution. Cross-model CLI review was skipped.
- All 22 tests pass on lens. In addition to collection/core checks, verification
  reproduces all tensors and optimizer states in a controlled replay, compares
  200 NumPy/Torch prefix probabilities, checks 101 full-output terminal-state
  cases, rejects incomplete grids and changed training records, and detects
  negative t/bootstrap disagreement. No development or formal benchmark data
  or outcomes were used. All transient test directories were removed.
- This adds direct audit/analysis functions, with no compatibility layer or
  new production abstraction. Redundant per-entry cost checks were replaced
  where full-map comparison owns the evidence boundary; no temporary bypass
  or permissive audit path remains in the experiment implementation.

## 2026-10-26 UTC: neural-prior data freeze

- Archived development seed 29 and formal seeds 11000–11031: 33 environments,
  1,848 tasks, and 24 experience attempts in every environment. The frozen JSON
  has 2,378,541 bytes and SHA-256
  `8d44fdd010df018097ffc5a8e5ee08e7f0270b1f3351846e3671293cc657aa78`.
  Full-function labels and within-environment split uniqueness were checked.
- The first pre-write validation command compared tuple-valued examples with
  list-valued expected pairs and failed. Corrected that check's representation
  and regenerated the identical seed grid; no generator, task, label, seed,
  model, or protocol changed. No data file or official run had been written by
  the failed command. This was a validation-script defect, not a task outcome.
- Implementation is frozen at 992085f, with Study H's unchanged generator and
  experience search. This large generated JSON is a separate archival commit
  so it does not obscure the reviewed source. No development/formal search or
  neural training outcome was used to select the data or implementation.
