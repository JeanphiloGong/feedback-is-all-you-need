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

Model and optimizer checkpoints are stored on the model server, outside Git.
See [evidence storage and verification](results/STORAGE.md).

- [`paper/outline.md`](paper/outline.md): working paper structure and evidence plan.
- [`notes/seed-idea.md`](notes/seed-idea.md): the initial conceptual argument.
- [`references/README.md`](references/README.md): source records and title precedent.
- `experiments/`: reproducible environments, agents, baselines, and results as
  they are added.
- `paper/`: manuscript sources and, later, reviewed figures and references.

## Current status

The hypothesis has now been tested in controlled classification, pretrained
language-model, symbolic program-search, and action-world studies. Results support conditional adaptation, but do not
establish feedback sufficiency for general intelligence.

- [Manuscript versions](paper/README.md): v0 hypothesis through v17, including
  audited results, pretrained models, response budgets, online updates, and
  checked rule revision, complete correction history, and explicit separation
  of candidate generation from external selection, cross-task fragment reuse,
  neural program priors learned from accepted solutions, and post-hoc
  diagnostics separating experience fitting from transfer likelihood, and
  a prospective local-context intervention with distribution-specific gains,
  and continued weight updates with untouched transfer probes, followed by
  direct failure-conditioned updates during task solving and predictive
  weights retained across goal episodes, and paired encoding interventions
  on unqueried transitions.
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
  Subsequent post-hoc selection and exact uniform-proposal references quantify
  recoverable progress and the contribution of supplied algorithms.
- [Study H results](results/study-h-v1/README.md): 32 new symbolic environments;
  compressed fragments solve 85.16% of motif-composition tasks versus 2.34%
  with whole-program memory and 50.39% without memory. On shifted programs,
  compressed memory falls to 9.96%, versus 50.00% without memory. This is a
  supplied symbolic search mechanism with learned representations, no neural model.
- [Study I results](results/study-i-v1/README.md): 32 further environments;
  the 1,172-parameter trained program prior solves 22.85% of compositions versus
  50.00% for uniform search, a primary difference of -27.15 points
  (95% CI [-32.74, -21.55]). A count prior reaches 66.02% on compositions but
  33.59% on shifted tasks; its balanced total is 510/1,024 versus uniform's
  512/1,024. All checkpoints, exact replays, and separate training/ranking costs
  are retained. This is unconditional program learning with supplied symbolic
  search, not a task-conditioned or lifelong agent.
- [Study I likelihood diagnostics](results/study-i-likelihood-v1/README.md):
  explicitly post-hoc scoring of every fixed checkpoint finds improved
  experience fitting alongside worse transfer-program likelihood. The small
  prior's experience loss is 0.663 nats/token versus an empirical bound of
  0.636; composition-generating-program loss is 2.596 versus uniform's 1.386.
  This does not identify the cause of the original search regression.
- [Study J results](results/study-j-v1/README.md): the same 1,172-parameter GRU
  trained with a supplied last-two-token context solves 79.30% of compositions,
  versus 24.22% with full-prefix training, a primary difference of +55.08 points
  (95% CI [45.60, 64.56]). Its shifted score is 11.13%; the balanced total,
  463/1,024, remains below uniform search's 512/1,024. All experience checkpoints,
  physical input costs, and independent replay audits are retained.
- [Study K results](results/study-k-v1/README.md): continued accepted-solution
  replay raises balanced held-out success from 43.75% frozen to 51.56%, a
  primary gain of +7.81 points (95% CI [4.74, 10.88]) over 32 environments.
  Composition success reaches 92.38%; shifted success remains 10.74%. The
  uniform comparator scores 50.20% overall, and additional optimization of
  only initial experience scores 43.16%. Both stream orders and all probes
  are retained with explicit reuse and lifetime-resource accounting.
- [Study L results](results/study-l-v1/README.md): weights change after rejected
  programs during task solving in 32 fresh environments. Rejection scores
  48.14% balanced exact success versus 50.59% for uniformizing updates, a
  primary difference of -2.44 points (95% CI [-4.50, -0.38]). All 11,178
  rejection batches change all 1,172 parameters, yet this objective does not
  improve the primary endpoint. Weights reset for each task; the result does
  not measure cross-task retention. Full journals, checkpoints, and unequal
  lifetime costs are archived.
- [Study M results](results/study-m-v1/README.md): a 485-parameter predictor
  updates after action outcomes and retains weights across goal episodes.
  Structured-world held-out-goal success rises from 23.22% to 100%; the
  1,925-parameter model also reaches 100%. The small-minus-large AUC primary
  is -0.247 points (95% CI [-1.128, 0.633]), without demonstrated superiority
  or equivalence. Random-world final scores are 85.22% and 97.09%.
  All gradients and checkpoints were replayed on lens; supplied encoding,
  replay memory, planning, and same-world dynamics limit the interpretation.
- [Study N results](results/study-n-v1/README.md): identical physical feedback
  and replay indices yield a +51.76-point encoding difference in small-model
  learning gains (95% CI [48.40, 55.11]) across 32 fresh worlds. Unqueried exact
  accuracy reaches 53.76% canonically and 2.29% after relabeling. A supplied-
  family symbolic matcher reaches 100%; an observed-edge table reaches 87.81%
  goal success while predicting no unqueried pair. All 40,960 gradients,
  512 checkpoints, and 888,832 probes are audited; the result concerns the
  fixed representation/learning mechanism, not general intelligence.
- [Full-text literature review](notes/full-text-review.md): methods and limits
  of eleven close precedents, including Reward is enough, DreamCoder, ExpeL,
  the Era of Experience, and unlikelihood training.

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
Post-hoc selection retrieves exact rules from 17/48 full-history 7B pools
instead of the final 9/48, but from none of the 0.5B pools. The latter's
selected direct score is 19.27%, compared with an exact 48.47% expectation
for 16 uniform legal revisions after the same initial proposal. This reference
supplies the rule grammar and selector and has different checker costs; it
is not an observed model run or a general intelligence comparison.
More effective feedback systems and sustained cross-task learning remain open.
All commits use English. External paper submission has not been authorized.
