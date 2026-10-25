# Study C development records

These are exploratory calibration runs, excluded from formal inference.
Both model runs use seed 9. Each has 384 test predictions, 32 sequential
self-history predictions, completion metadata, and a matching runner snapshot.
The snapshots preserve the code as it existed when each run began; the later
formal runner adds raw-token/configuration and padded-work accounting.

The 0.5B reasoning prompt returned literal `N` plus EOS, producing invalid
answers under the fixed grammar. The 7B run could produce valid answers and
achieved 100% direct / 93.75% composition accuracy with oracle offsets on this
single development environment. These are diagnostics, not inferential results
or criteria used to select a favorable formal seed set.

`calibrate.py` and `calibration-current.txt` preserve five ordinary language
prompts evaluated with the pinned 0.5B model in float32/eager mode. The ordinary
2+2 question returned 4, while the one-digit instruction returned 1. Other
responses were coherent. This establishes prompt sensitivity in these examples;
it does not establish a general capability or remove every possible runtime
limitation. Prompts and token IDs are retained in the log.

JSONL compression is lossless, with a fixed gzip timestamp. No failed or invalid
response was removed. Formal evaluation uses seeds 5000–5023 and the frozen
source revision `f0a8841`.
