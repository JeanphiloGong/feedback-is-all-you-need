# Study D: fixed generation-budget comparison

The registered 7B run completed all 24 new environments, seeds 6000–6023,
using frozen runner revision `8507524`. It retains 3,456 continuations in 216
actual generation batches. Both 128-token and 512-token scores are computed
from each continuation; they are paired observations, not independent runs.
The model, instructions, parser, and repetition penalty remain fixed.

| Condition | Direct, 128 | Direct, 512 | Composition, 128 | Composition, 512 |
| --- | ---: | ---: | ---: | ---: |
| No examples | 3.12% | 11.98% | 0.00% | 14.06% |
| Clean 16 | 20.05% | 59.90% | 7.94% | 70.57% |
| Oracle offsets | 100.00% | 100.00% | 96.61% | 96.88% |

The primary clean composition difference is **+62.63 percentage points**:
95% paired t interval **[56.10, 69.16]**; 10,000-resample environment bootstrap
**[56.38, 68.75]**. Clean-minus-none composition performance is +56.51 points
at 512 tokens (descriptive 95% interval [48.99, 64.03]), compared with +7.94
at 128. The no-example control also improves as responses become complete.

Clean composition truncation falls from 87.76% to 3.26%, while invalid output
falls from 87.76% to 2.99%. Of 768 clean composition queries, 481 change from
wrong to correct and none change from correct to wrong. Thus allowing a longer
continuation materially changes the measured feedback effect under this
interface. It does not isolate reasoning from formatting completion, introduce
new feedback during generation, establish lasting learning, or show that a
smaller model outperforms a larger one. The 0.5B model was prospectively excluded
because its previous outputs had already terminated after two tokens.

All 481 recovered clean composition queries had invalid 128-token prefixes.
This is not evidence that 481 valid wrong answers were revised successfully.

The actual run took 1,436.61 seconds, including 1,435.81 measured generation
seconds, and processed 853,632 unpadded input tokens and 654,495 generated
tokens. Batch padding raises output token slots to 1,044,320. Prefix scores
have no separately measured wall times; do not invent a 128-token speed ratio.

Before formal collection, all 144 development responses matched separate
128-token calls exactly; raw paired evidence is in
[`../development/study-d-v1`](../development/study-d-v1). Formal model settings,
all four required source hashes, data hashes, complete grids, prompt hashes,
input lengths, raw decoding, EOS/cap status, scores, and batch costs passed
analysis checks. Independent local aggregation reproduces every mean and the
primary interval. Transferred files match the remote `sha256.json` manifest.

Run with the pinned offline model and package versions in
`analysis-environment.json`:

```bash
python -m pytest -q experiments/study_d/test_prefix.py
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=8507524 python -m experiments.study_d.run \
  --development --output runs/study-d-development
# Inspect all 144 prefix matches and matching source/configuration before:
HF_HUB_OFFLINE=1 FEEDBACK_GIT_REVISION=8507524 python -m experiments.study_d.run \
  --output runs/study-d-reproduction
HF_HUB_OFFLINE=1 python -m experiments.study_d.analyze --input results/study-d-v1
```

The analysis consumes gzip JSONL, regenerates summaries and figures, and
requires the frozen source hashes. Preserve independent reproductions in their
own directory. No formal inference was restarted or extended after outcomes.
