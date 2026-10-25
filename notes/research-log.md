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
