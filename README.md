# Feedback Is All You Need

Research repository for the paper **“Feedback Is All You Need”**.

## Central hypothesis

Once a model has a minimum level of representational and world-modeling
capacity, the main source of further intelligence is not parameter count alone,
but the quality and efficiency of a closed feedback loop:

```text
observe → predict → act → measure the outcome → identify the error
→ update memory or policy → act again
```

The paper investigates whether a small agent can acquire substantial, general,
and transferable capability through persistent interaction, error correction,
memory, and skill consolidation.

This is a research hypothesis, not an established result. The project must
separate the contribution of the base model, feedback, memory, tools, and
additional compute.

## Why this matters

The dominant scaling approach treats intelligence largely as a function of
model size and training data. This project asks whether the more fundamental
variable is **feedback efficiency**: how much durable, transferable capability
an agent gains from each reliable interaction with reality.

The working view is that books, language, science, and institutions are
externalized records of accumulated feedback. Human intelligence may therefore
be better understood as a brain participating in a larger individual–tool–social
feedback system than as a static parameter store.

## Research questions

1. Can a smaller model with a persistent feedback loop outperform a larger
   static model on tasks that require adaptation?
2. Does capability growth correlate more strongly with feedback quality and
   feedback efficiency than with parameter count after a minimum threshold?
3. Which feedback signals matter most: outcome, process, prediction error,
   social correction, or long-horizon utility?
4. How should memory, policy updates, and skill consolidation be separated to
   make continual improvement stable?
5. When does a fast feedback loop amplify error, reward hacking, or
   catastrophic forgetting?

## Repository map

- [`paper/outline.md`](paper/outline.md): working paper structure and evidence plan.
- [`notes/seed-idea.md`](notes/seed-idea.md): the initial conceptual argument.
- [`references/README.md`](references/README.md): source records and title precedent.
- `experiments/`: reproducible environments, agents, baselines, and results as
  they are added.
- `paper/`: manuscript sources and, later, reviewed figures and references.

## Current status

The project is at the hypothesis and research-design stage. No empirical claim
has been accepted yet. Every strong statement in the manuscript should be
paired with a falsifiable prediction or an explicitly marked limitation.
