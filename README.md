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

Scaling research studies the joint contribution of model size, training data,
and computation. This project investigates **feedback efficiency**: how much
durable, transferable capability an agent gains from reliable interaction,
and how that gain depends on capacity, memory, and computational resources.

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

The hypothesis has now been tested in controlled classification and pretrained
language-model studies. Results support conditional adaptation, but do not
establish feedback sufficiency for general intelligence.

- [Manuscript versions](paper/README.md): v0 hypothesis through v8, including
  audited results, pretrained models, response budgets, online updates, and
  checked revision of external rule memory and complete correction history.
- [Study A replication](results/study-a-v2/README.md): 20 seeds, four capacities,
  seven feedback conditions; all 19,200 records retained.
- [Study B results](results/study-b-v1/README.md): four Qwen2.5 sizes, 24 new
  command languages, raw predictions, and persistent LoRA updates.
- [Research log](notes/research-log.md): protocol amendments, failed checks,
  independent reviews, and infrastructure failures.
- [Study C results](results/study-c-v1/README.md): first-token versus generation
  on 24 new environments; all raw tokens, format failures, and truncations.
- [Study D results](results/study-d-v1/README.md): fixed 128/512-token comparison
  on 24 further environments; clean composition rises from 7.94% to 70.57%.
- [Study E results](results/study-e-v1/README.md): 24 fresh online-feedback
  environments; no demonstrated composition gain after 16 LoRA updates.
- [Study F results](results/study-f-v1/README.md): 24 fresh rule-memory
  environments per model; external checks do not establish a small-model
  composition gain over two self-revisions.
- [Study G results](results/study-g-v1/README.md): 48 fresh environments per
  model and 16 checked revisions; complete history does not demonstrate a
  direct-accuracy benefit, with regressions and additional compute retained.
- [Full-text literature review](notes/full-text-review.md): methods and limits
  of nine close precedents, including Reward is enough, Voyager, ExpeL, and the
  Era of Experience.

The smallest language model did not improve composition accuracy under the
first tested feedback mechanism. This negative result is conditional on the
prompt, update rule, and inference interface. In 7B, longer continuations
recover substantial example-based performance; all 481 recovered composition
answers had invalid shorter prefixes. This is a response-completion effect.
Explicit per-outcome updates and error-only updates in 0.5B also fail to
demonstrate a retained composition gain under the short tested schedule.
Checked revision of stored rules yields a 0.5B composition difference of
+4.04 points (95% CI [-2.78, 10.85]) over self-review and no exact rule recovery.
The supplied executor and strict output schema remain part of that result.
Retaining complete history through 16 checked revisions yields a 0.5B direct
difference of 0.00 points (95% CI [-3.96, 3.96]) over last-only history. In 7B,
the full-history branch finishes with nine exact rules but loses three of the
four initially exact rules. More revisions do not ensure cumulative progress
under this mechanism, and the full-history branch consumes over twice the
input tokens.
More effective feedback systems and sustained cross-task learning remain open.
All commits use English. External paper submission has not been authorized.
