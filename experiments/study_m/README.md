# Study M: retained predictive weight learning

The [prospective protocol](../../notes/protocol-v13-retained-world-model.md)
tests whether a small neural transition predictor learns reusable behavior
from action outcomes. Weights, Adam moments, and observed transitions persist
across goal episodes. This is a fully observed finite-world experiment with a
supplied state encoding and planner, not a pretrained language model.

The core implements the two world families, 485- and 1,925-parameter predictors,
sampled gradient replay, and bounded shortest-path planning. Synthetic checks
verify both constructive capacity witnesses, independently calculated gradients,
parameter changes, continued Adam state, and planning boundaries:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.study_m.test_core -v
```

The collector and independent audit are still being implemented. No official
world data or experiment outcome has been generated. Official collection must
use the registered source/data freeze and PyTorch 2.8.0+cu126 runtime; local
synthetic checks under another runtime do not satisfy that requirement.
