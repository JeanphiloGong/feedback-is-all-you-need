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
