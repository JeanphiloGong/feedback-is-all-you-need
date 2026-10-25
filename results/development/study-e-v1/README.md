# Study E development evidence

Seed 13 was run before formal collection using frozen revision `84f2e52`.
The complete run took 5.92 seconds after model loading and retains 1,008
held-out predictions, 48 pre-update predictions, 48 corresponding feedback
records, and both final adapters. Outcomes are exploratory, not replicates
of the formal experiment; no prompt or update schedule was tuned to them.

The full development audit passed. It checks query and prompt identity,
complete ordered grids, update gating, independent parsing of token identities,
batch/input costs, source hashes, model identity, and both adapter manifests.
Both arms start from identical adapter tensors; the full pretrained base hash
matches after every condition. Zero-feedback predictions agree with the base.

`audit.json` retains the exact metadata, analyzer hash, and artifact hashes
verified before formal launch. The formal runner requires this matching audit.
All transferred artifact hashes were checked locally. The PEFT reattachment
warning was retained in the research log; base equality checks passed.

Run the auditor from the repository root with the pinned offline model cache:

```bash
HF_HUB_OFFLINE=1 python -m experiments.study_e.analyze --development \
  --input results/development/study-e-v1
```

The original audit covers files present at the gate; this explanatory README
was added afterward. Regenerating the audit may therefore include this file
in its manifest. Preserve the original evidence when reproducing the run.
