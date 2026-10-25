# Study G development checkpoint

Frozen source: `c4b4054`. Seed 19 only. Both models completed 33 proposals,
33 saved rule files, and 1,584 held-out evaluations. Each full audit passed,
including identical first-revision ordered input IDs, attention masks, and
output IDs. Saved batch tensors are retained alongside their hashes.

- 0.5B: 6.50 seconds after model loading.
- 7B: 19.33 seconds after model loading.

Each actual suite contains 33 continuations; each hypothetical standalone arm
contains 17, counting the shared initial proposal. All 82 transferred manifest
entries and all source hashes match. Formal inference requires these original
model-specific audits, with matching packages/configuration/source. Both runs
use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and the cached pinned weights.

Development outcomes are exploratory and excluded from the 48 formal seeds.
This README was added after the audit and is outside its original manifest.
