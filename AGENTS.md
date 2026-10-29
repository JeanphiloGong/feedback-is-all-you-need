# AGENTS.md (Project Rules: Feedback Is All You Need)

## Overview

This repository develops the paper **“Feedback Is All You Need”** and its
reproducible experiments. The owner is the human researcher who initiated this
repository. The default focus is research notes, manuscript drafts, and small
verifiable experiments.

## Core Principles

- Treat the feedback hypothesis as unproven until experiments support it.
- Distinguish model capability from system capability supplied by memory,
  tools, data, and the environment.
- State assumptions and define measurable success criteria before implementing
  an experiment.
- Prefer the smallest experiment that can distinguish competing explanations.
- Keep claims, evidence, and speculation visibly separate.

## Scope Boundaries

- Include manuscript sources, research notes, experiment code, configurations,
  result summaries, and reproducibility instructions.
- Do not commit secrets, credentials, private user data, or unreviewed generated
  artifacts.
- Keep model weights and optimizer checkpoints on `lens` in persistent storage
  outside Git. Track code, manuscripts, result records, hashes, and server
  locations instead. Run full checkpoint verification on the server; do not
  download checkpoints locally unless the user explicitly requests it.
- Keep external publication, public announcements, and irreversible releases
  subject to explicit human approval.

## Execution Rules

- Use short, atomic commits with behavior-first messages.
- Record the baseline, feedback signal, update rule, compute budget, and
  evaluation protocol for every experiment.
- Report failed runs and inconclusive results rather than deleting them from the
  research record.
- Run the narrowest relevant verification before committing; record checks that
  were not run and why.

## Quality Bar

The paper is complete only when its central claim has a precise definition,
falsifiable predictions, comparison baselines, ablations, reproducible evidence,
and a limitations section covering capacity, feedback quality, stability, and
reward-hacking risks.
