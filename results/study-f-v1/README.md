# Study F: checked revision of external rule memory

Source and prospective protocol: `08725bb`,
[`../../notes/protocol-v6-rule-memory.md`](../../notes/protocol-v6-rule-memory.md).
Each pinned model generates one initial four-offset rule per language and two
revisions per branch, with or without checks against already-disclosed examples.
There are 24 fresh environments (8000–8023), 120 actual proposals and 5,760
held-out executor evaluations per model. All 240 intermediate rule files are
retained across both models. No held-out input or answer enters a model prompt.

| Model | Proposal | Direct % | Composition % | Valid / 24 | Exact / 24 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | Initial | 11.46 | 9.38 | 18 | 0 |
| 0.5B | Self-review, final | 11.46 | 9.38 | 18 | 0 |
| 0.5B | Checked review, final | 10.42 | 13.41 | 17 | 0 |
| 7B | Initial | 51.04 | 43.23 | 21 | 4 |
| 7B | Self-review, final | 52.08 | 42.32 | 22 | 3 |
| 7B | Checked review, final | 54.17 | 45.44 | 21 | 6 |

The primary 0.5B final checked-minus-self composition contrast is **+4.04
percentage points**, t 95% CI **[-2.78, 10.85]**, paired percentile bootstrap
**[-1.56, 11.07]**. Neither interval demonstrates a positive effect under this
protocol. Checked-minus-initial has the same mean and t interval; self-review
has zero net composition change, with two answers gained and two lost.
Compared with self-review, checked review changes 60 composition answers from
wrong to correct and 29 from correct to wrong. Direct accuracy decreases by
1.04 points, CI [-10.11, 8.02]. No small-model proposal recovers all four rules;
a final 0/24 exact-recovery rate has Wilson 95% CI [0.00%, 13.80%].

The secondary 7B checked-minus-self composition contrast is +3.13 points,
CI [-9.32, 15.57]; checked-minus-initial is +2.21 points, CI [-9.93, 14.36].
Checked review recovers two previously incorrect full rules and retains all
four initially correct rules. Self-review loses one initially correct rule
and gains none. Final exact recovery is 6/24 (25.00%, Wilson CI [12.00%, 44.90%])
versus 3/24 (12.50%, CI [4.34%, 31.00%]); this is a descriptive comparison.
Relative to the initial proposal, checked review gains 81 composition answers
and loses 64. Query transitions and exact-rule transitions are different units.

All proposals terminate at EOS before the 512-token cap. Across all five
stages, 0.5B has 32 invalid proposals, all with repeated offsets. The 12 invalid
7B proposals include nine with negative offsets outside the required 0–7
range, two malformed JSON objects, and one repeated offset. Negative modular
representatives are rejected by the preregistered interface; their failure
must not be equated with inability to express the corresponding modular rule.
For example, seed 8012's final checked 7B rule is semantically exact after
interpreting -1 as 7, while its canonical-interface score remains zero.
No post-hoc normalization or candidate selection enters these scores.

A supplied symbolic estimator recovers every language from its training
examples and executes all held-out targets correctly. No observed nonexact
rule achieves perfect composition accuracy, although the +4-offset
counterexample remains valid. Successful rule files occupy 34 bytes, including
JSON syntax and newline. Persistence and execution are supplied by file storage
and the deterministic executor; this is not evidence of neural retention,
lifelong learning, or a novel external-memory mechanism.

| Whole-suite measured cost | 0.5B | 7B |
| --- | ---: | ---: |
| Input tokens | 51,427 | 50,283 |
| Output tokens through EOS | 4,680 | 3,305 |
| Padded input token slots | 51,648 | 50,688 |
| Padded generation slots | 4,680 | 3,480 |
| Actual generation batches | 10 | 10 |
| Generation seconds | 2.724 | 12.457 |
| Diagnostic checker seconds | 0.00114 | 0.00123 |
| Runner seconds after loading | 2.763 | 12.497 |
| Peak allocated GPU memory, GiB | 1.269 | 15.546 |

The shared initial call is counted once. Each standalone branch would incur
that initial cost plus its two revisions. All proposal checks are timed for
audit; only initial and checked-first-revision checks enter feedback prompts.
Costs are measured, not matched, and exclude offline evaluation/plotting.
The model-specific repetition penalties remain 1.1 and 1.05.

Each model's `audit.json` verifies complete grids, pinned sources and models,
packages, development gating, raw-token decoding, prompt reconstruction,
parsing, targets, all memory files, and batch costs. All 258 transferred
manifest entries match. Independent local parsing and arithmetic reproduce
all 20 accuracy means, 12 contrasts, the primary bootstrap, diagnostic Wilson
intervals, and answer-transition counts. Both generated figures were inspected.
The complete development evidence, including the preserved pre-inference
7B cache-write failure, is in
[`../development/study-f-v1`](../development/study-f-v1).

Reproduce with the commands in
[`../../experiments/study_f/README.md`](../../experiments/study_f/README.md),
using `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` with the cached model revisions.
