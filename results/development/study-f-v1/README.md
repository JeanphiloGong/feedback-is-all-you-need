# Study F development evidence

Frozen source: `08725bb`. Development seed: 17 only. The two successful runs
retain five proposals, five rule files, and 240 held-out evaluations each.
Every source and transferred artifact hash matches the completed audit.

- `dev-0.5B-v1`: complete, 1.11 seconds after model loading.
- `dev-7B-v1`: failed during model loading, before any proposal generation.
  Transformers' optional PEFT adapter lookup attempted a cache metadata write
  on the full root filesystem. The original traceback, preflight metadata,
  and failure record are preserved.
- `dev-7B-v2`: complete, 2.29 seconds after model loading. Retried with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` using the existing pinned weights.
  Prompts, source, model revision, and generation settings were unchanged.

The offline environment flags will also be used for formal runs. This changes
cache/network behavior, not the experimental treatment. The logs' warnings
about temperature, top-p, and top-k are from checkpoint sampling defaults;
`do_sample=False` remains the saved generation override.

Development outcomes are exploratory and are excluded from the 24 formal
seeds. Formal inference requires each model's matching `audit.json`. Audits
were produced before this README and retain their original file manifests.
