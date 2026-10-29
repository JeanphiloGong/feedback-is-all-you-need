# Research evidence storage

Model weights and optimizer checkpoints live on **`lens`**, outside Git,
under `/src/users/lens/feedback-research-archive/`. This is persistent disk;
the original `/tmp` execution directories are not the archival location.
Git retains source, protocols, papers, lightweight journals, summaries,
checksums, and verification certificates. Do not download or commit weights
unless the researcher explicitly requests them.

On 28 October 2026, all 36,492 local checkpoint files (13,774,200,908 bytes)
were matched to server files by size and SHA-256 before local removal.
The [verification record](checkpoint-storage-verification.json) records the
totals and groups. A complete mapping from repository paths to server paths
is stored on the server as `checkpoints-2026-10-28-v2.json` in the archive
root, with SHA-256
`f4a8f2a3b6d70d4624369f3a533d5a975a8883bbe5059a8b2cb038334be782be`.
The 11 original snapshots contained 95,133 files, verified at migration in
`migration-2026-10-28.json` at the same root. Study R later received the
unchanged numerical checker and its verification report. The first
checkpoint check used an incorrect Study N directory; the corrected v2 map
matches every file. Both verification records remain on the server.

Checkpoint/raw locations below are relative to
`/src/users/lens/feedback-research-archive/`. Full reconstruction also needs
the sibling audit, source, and transfer files in each `feedback-research-*`
directory; the copied snapshots retain those files.

| Study | Checkpoint and raw evidence location |
| --- | --- |
| E | `feedback-research-e/dev-e-v1/`, `feedback-research-e/formal-e-v1/` |
| I | `feedback-research-i/development-v1/`, `feedback-research-i/formal-v1/` |
| J | `feedback-research-j/development-v1/`, `feedback-research-j/formal-v1/` |
| K | `feedback-research-k/development-v1/`, `feedback-research-k/formal-v1/` |
| L | `feedback-research-l/development-v1/`, `feedback-research-l/formal-v1/` |
| M | `feedback-research-m/development-v1/`, `feedback-research-m/formal-v1/` |
| N | `feedback-research-n/frozen-v1/evidence/` |
| O | `feedback-research-o/frozen-v1/results/study-o-v1/` |
| P | `feedback-research-p/frozen-v1/results/study-p-v1/` |
| Q | `feedback-research-q/frozen-v1/results/study-q-v1/` |
| R | `feedback-research-r/frozen-v1/results/study-r-v1/` |

Full gradient/state replay and numerical verifiers that bind checkpoint
membership, hashes, or storage require the complete server evidence. They
are expected to reject a lightweight local checkout missing those files.
Do not weaken their checks to make such a checkout pass. Recorded earlier
local verification certificates describe checks performed before migration.
Summary-only plotting still works locally.

Use the pinned Python environment
`/src/users/lens/feedback-is-all-you-need/.venv/bin/python` and the frozen
source for each study. For example, Study R's unchanged numerical verifier
has been copied to its persistent frozen source root and passes there:

```bash
ssh lens
cd /src/users/lens/feedback-research-archive/feedback-research-r/frozen-v1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python \
  results/study-r-v1/check_summary.py --mode formal \
  --output /tmp/study-r-formal-verification-new.json
```

The output must be a new path. For older verifiers whose relative paths
assume a repository layout, use a disposable **server** checkout and copy
the required evidence from the paths above into its expected `results/`
paths. Preserve the registered source/runtime and original raw directories.
Transfer only the small verification report back to the local repository.

Removing tracked checkpoints and adding `.gitignore` does not remove blobs
from previous commits. This migration does not rewrite published history or
force-push. The remote cleanup is published independently from unpublished
research commits so their historical checkpoint blobs are not uploaded.
