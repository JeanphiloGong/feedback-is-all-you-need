# Study C: response-interface sensitivity

Both formal runs completed using frozen runner revision `f0a8841` on the same
RTX 4090 D. Each model has 9,216 held-out predictions and 768 sequential
self-history records across 24 environments (seeds 5000–5023), two interfaces,
and four conditions. The primary replication unit is the environment.

Model identities and revisions, source/data hashes, full outcome grids,
recomputed scores, and raw-token decoding were verified before interpretation.
All transferred files match `sha256.json`, generated on the remote host.
No answer, invalid format, or truncated response was dropped. Source protocols
and all development outputs remain separate from formal data.

| Model and interface | Composition: none | clean16 | self | oracle |
| --- | ---: | ---: | ---: | ---: |
| 0.5B, first token | 10.68% | 11.20% | 10.68% | 10.68% |
| 0.5B, generation | 0.00% | 0.00% | 0.00% | 0.00% |
| 7B, first token | 13.15% | 16.15% | 12.24% | 24.74% |
| 7B, generation | 0.00% | 6.90% | 0.26% | 97.27% |

The primary 0.5B composition interaction is **−0.52 percentage points**:
95% t interval **[−1.16, 0.11]**, bootstrap **[−1.17, 0.00]**. Generation's
clean-minus-none effect is exactly zero in this sample: every 0.5B generation
returned literal `N` plus EOS and failed the answer grammar. This is an
instruction/interface failure and does not measure its unrestricted reasoning.

For 7B generation, clean-minus-none composition accuracy is +6.90 points
(descriptive 95% interval [4.04, 9.76]). Absolute clean composition accuracy
falls relative to first-token scoring, while **88.28%** of clean composition
outputs hit the 128-token limit. The no-example composition condition truncates
on every query. In contrast, oracle-offset generation reaches 100% direct and
97.27% composition accuracy, with only 0.26% composition truncation. The result
does not isolate reasoning compute: prompts, constraints, parsing, and runtime
all change between interfaces. A longer-budget test requires a new protocol.

The self-history feedback is also affected: all 384 small-model generated
examples and 354/384 large-model generated examples are invalid. These are
stored as the explicit string `invalid` in subsequent context. The self
condition must not be interpreted as 16 valid independent critiques.

Whole-suite measured times are 28.92 seconds (0.5B) and 1,831.46 seconds (7B),
excluding model loading. Each interface makes 672 calls per model including
self-history acquisition. 7B generation uses 554,021 unpadded output tokens
and 599,945 output token slots including batch padding. The analysis sums
repeated batch fields once. `summary.json` includes all means, intervals,
format/truncation rates, resource totals, and provenance checks.

The initial prose mistakenly assigned repetition penalty 1.1 to both models.
The frozen implementation inherits model defaults: 1.1 for 0.5B, 1.05 for 7B.
This was corrected transparently during collection; no run setting changed.
Intermittent SSH failures delayed export after collection. An initial SCP
directory-copy command rejected a `.` filename; a tar transfer then succeeded
and all remote hashes matched. Model inference was never repeated.

To reproduce, use the package versions in `analysis-environment.json`, download
the pinned models first, and run on a CUDA GPU with sufficient memory:

```bash
python -m pytest -q experiments/study_c/test_generation.py experiments/study_b/test_tasks.py
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=f0a8841 python -m experiments.study_c.run \
  --model 0.5B --output runs/study-c-reproduction-0.5B
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=f0a8841 python -m experiments.study_c.run \
  --model 7B --output runs/study-c-reproduction-7B
HF_HUB_OFFLINE=1 python -m experiments.study_c.analyze --input results/study-c-v1
```

The archived analysis command reads gzip JSONL files, loads pinned tokenizers
locally, and requires the frozen runner/task hashes. Analyze an independent
reproduction only after preserving its raw outputs and metadata in the same
directory layout; do not replace this raw archive with a reproduction. Analysis
regenerates derived summaries and figures. Figures were visually checked.
