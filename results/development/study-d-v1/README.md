# Study D prefix-equivalence development

Both runs use seed 11, excluded from formal statistics. Each completed 144
test generations and 144 separate 128-token generations. For every query,
the 512-token continuation's first 128 tokens, trimmed at EOS, exactly matched
the shorter generation. Tests compare within the same batch shape and settings.

Version 1 performed the comparison but only retained the longer outputs and
the success count. Before formal collection, review required raw evidence for
every comparison, including a failure. Version 2 adds `prefix-checks.jsonl.gz`
with both token sequences, query identity, prompt hash, and equality result.
It was verified against frozen source revision `8507524`, model revision, and
generation settings before the formal run began. Both runner snapshots match
their metadata hashes; both original runs are retained.

Development times include extra short generations and must not be used as
formal inference-cost estimates. JSONL files were compressed losslessly.
