# Historical DreamCoder source reference

The five files under `dreamcoder/` are unmodified copies from the authors'
public repository at commit `e6fd72804194ed6cc0c88eee652413b0b44fc63e`
(2020-04-25). Their original authorship is retained. They are research source
material, not this project's implementation or a runnable vendored dependency.
`manifest.json` records provenance and hashes. We have not established that
this snapshot and particular options reproduce the 2020 paper's experiments.

The focused reading and conditional parameter formula are in
[`../../notes/dreamcoder-capacity-audit.md`](../../notes/dreamcoder-capacity-audit.md).

`verify_parameter_count.py` is this project's standalone arithmetic check.
It constructs standard PyTorch module shapes without importing the external
source. `parameter-count-check.json` preserves its output with PyTorch
2.8.0+cu126. These are synthetic dimensions, not measured published models:

```bash
python verify_parameter_count.py
```

No experiment weights, performance reproduction, or original supplement is
included in this source reference.
