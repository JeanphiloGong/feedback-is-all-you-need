# Targeted full-text review: feedback, capacity, and prior learning

Reviewed on 2026-10-25 UTC. This is a targeted comparison, not an exhaustive
novelty search. Copies and exact hashes are in
[`../references/full-text-manifest.json`](../references/full-text-manifest.json).
Page numbers below are printed article pages. Main methods, controls, and
limitations were read; appendices were consulted selectively. Reward is enough
was read visually because the author's PDF copy has no text layer.

## Reward is enough — Silver, Singh, Precup, and Sutton (2021)

DOI: [10.1016/j.artint.2021.103535](https://doi.org/10.1016/j.artint.2021.103535).
All main-text pages 1–11 were read.

The central hypothesis is that many abilities associated with intelligence can
arise in service of maximizing reward in a rich environment. Section 4 adds a
conjecture about learning such behavior through ongoing interaction. This is a
direct intellectual precedent for our broad motivating idea.

The paper does not claim that a reward scalar supplies an entire implementation
or makes capacity irrelevant. Section 2.2 (p. 3) explicitly includes finite
memory, finite neurons, and real-time computational constraints. Section 3.1
(p. 5) distinguishes innate from learned knowledge. Section 4 (p. 9) offers no
sample-efficiency guarantee and states that learning depends on environment,
algorithm, and inductive bias; artificial environments can make learning fail.
Section 6 (p. 11) argues that online interaction can reveal gaps not covered by
offline data, while allowing additional experiential signals and learned world
models to assist reward maximization.

Implication: a feedback-centered account of broad intelligence is not new. Our
synthetic negative results do not falsify their practical conjecture about
sufficiently powerful agents in rich environments. Any contribution here must
identify a mechanism, measurable resource tradeoff, or reproducible boundary.

## RL² — Duan et al. (2016), arXiv v2

[arXiv:1611.02779v2](https://arxiv.org/abs/1611.02779v2).
Sections 1–2, discussion, and Appendix A.1–A.2 were inspected.

RL² learns a fast adaptation algorithm in an RNN's weights through a slower
outer reinforcement-learning process. Runtime inputs include observations,
previous actions, rewards, and episode-end flags. Hidden state persists across
episodes of the same task and resets between tasks (pp. 2–3). Thus rapid
adaptation without runtime weight updates already has a concrete precedent.

Appendix A lists 256 GRU units and substantial outer training schedules for
bandits and tabular MDPs. The discussion identifies outer-loop optimization and
long-horizon architecture as bottlenecks. A compact runtime learner is therefore
compatible with expensive prior training; these resources must not be conflated.

Poppler reported reconstruction of one missing PDF cross-reference entry.
Text extraction produced 14 pages, and Appendix A.1 was also checked visually.

## Reflexion — Shinn et al. (2023), arXiv v4

[arXiv:2303.11366v4](https://arxiv.org/abs/2303.11366v4).
Section 3, the reasoning ablation, programming Section 4.3, and Section 5 were
inspected.

The mechanism combines an actor, an evaluator, verbal self-reflection, and a
bounded memory of recent reflections (pp. 4–5). Feedback sources differ by task:
exact-match grading, environment heuristics, or model-generated tests. The
programming setup uses generated tests rather than hidden evaluation answers
(p. 7); it must not be described as having the same oracle feedback as its
reasoning setup.

The paper reports both benefits and failures. Its programming analysis links
false-positive generated tests to premature acceptance (p. 8), and an ablation
without test guidance performs below its baseline on the selected Rust subset.
Memory is bounded to a small sliding window, and local optima remain possible
(p. 9). These are direct precedents for feedback-quality and memory ablations.

Our self-generated-example condition produces guesses without critiques or
iterative repair of a solution. It is not a reproduction of Reflexion.

## Self-Refine — Madaan et al. (2023), arXiv v2

[arXiv:2303.17651v2](https://arxiv.org/abs/2303.17651v2).
Sections 2–4 and 6 were inspected.

The same model generates an answer, supplies a task-specific critique, and
refines the answer. Few-shot prompts demonstrate each operation; history is
appended to the prompt. The studied schedule allows up to four iterations,
with task-dependent stopping (pp. 3–4). There are no extra weight updates.
Section 4 compares specific, generic, and absent critiques and reports
diminishing returns and some non-monotonic changes (pp. 6–7).

Section 6 (p. 9) explicitly requires adequate few-shot or instruction-following
ability in the base model. Its experiments use strong proprietary models and
English datasets. It provides no evidence that an arbitrarily small controller
can bootstrap unrestricted capability from feedback alone. Our sequential
guess control does not reproduce this method either.

## Large Language Models Cannot Self-Correct Reasoning Yet — Huang et al.

ICLR 2024; [arXiv:2310.01798v2](https://arxiv.org/abs/2310.01798v2).
Definitions, oracle-feedback controls, and Sections 4–7 were inspected.

The target is **intrinsic** self-correction without external correctness
feedback. The paper separates it from correction assisted by oracle labels
(pp. 2–4). It also identifies two important comparisons: equivalent inference
cost, and placing informative correction instructions in the initial prompt
(Sections 4–5). Reported failures apply to tested models, reasoning tasks, and
protocols; they are not an impossibility theorem for all feedback systems.

Our information bound concerns genuinely new task-dependent observations.
Internal reasoning can still improve use of existing information, and external
feedback can still fail if the learner cannot use it. Neither statement implies
that all self-refinement is useless or that any external feedback is sufficient.

## Welcome to the Era of Experience — Silver and Sutton (2025)

[Author-linked preprint](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf).
All main-text pages 1–8 were read. The PDF, dated April 2025 in its metadata,
identifies itself as a forthcoming chapter in *Designing an Intelligence*,
MIT Press. It is a perspective and research agenda, not a new controlled test
establishing general intelligence. The author publication list links this copy.

This is especially close to the original project motivation. It argues for
agents that learn continuously from their own interactions, retain information
across long experience streams, obtain rewards from observable consequences,
and revise world models from prediction errors (pp. 1–5). Its definition of
adaptation includes weight updates and adaptation in context (p. 2, footnote 1).
It explicitly says experience and human interaction are not opposites (p. 4,
footnote 2), so books and human knowledge need not be excluded from a feedback
system.

The perspective also recognizes simultaneous scaling with network size,
interactive experience, and thinking time (p. 6). It does not establish that
small controllers make capacity irrelevant. Its discussion of adaptable reward
functions explicitly provides no perfect-alignment guarantee and notes the
physical delays of real-world experimentation (p. 8).

Our paper should cite this close precedent and avoid presenting the whole
research agenda as novel. Studies B–E remain short, synthetic inference/adaptation
experiments; they do not instantiate the lifelong, consequence-driven learning
system described here.

## Consequences for this manuscript

1. Retain the title as a hypothesis; acknowledge both the 2023 title precedent
   and the conceptual precedents from Reward is enough and the Era of Experience.
2. Report feedback source, update mechanism, memory, prior learning, and compute
   separately. Parameter count is one resource, not a complete system measure.
3. Describe Studies B and C as diagnostic example-conditioning experiments,
   plus Study B's limited LoRA schedule. Do not claim a direct replication or
   refutation of Reflexion, Self-Refine, RL², or the reward-is-enough conjecture.
4. A further agent experiment must include actual repeated error-directed
   updates, a control with the same inference budget, an independent held-out
   evaluator, and tests of retention and transfer before making broader claims.

## Voyager — Wang et al. (2023), arXiv v2

[arXiv:2305.16291v2](https://arxiv.org/abs/2305.16291v2). Selected methods
(Section 2), setup/baselines, skill-reuse results, ablations (3.4), and
limitations (4) were read. Title and result pages were inspected visually;
appendices were not exhaustively reviewed.

Voyager is a close precedent for the project's agent framing. It uses GPT-4 as
a black-box generator, an automatic curriculum, iterative code generation,
Minecraft environment feedback, execution errors, self-verification, and an
embedding-indexed executable skill library. It explicitly avoids parameter
fine-tuning; the durable state is a retrievable program library. Its ablations
report large drops without the library or self-verification, and its new-world
experiments test reuse of stored skills. These results motivate separating
feedback, verification, memory representation, and retrieval in our design.

The system is not a small model that bootstraps itself from scratch: GPT-4 and
GPT-3.5 perform generation, curriculum, and verification, with a separate
text-embedding-ada-002 embedding model, while
Minecraft supplies a rich simulator and a hand-defined control API. The paper's
success curves therefore do not establish that feedback makes parameter count
irrelevant. Our Study E is a narrow synthetic update comparison and does not
replicate Voyager's executable memory or open-ended exploration.

## Teaching Large Language Models to Self-Debug — Chen et al. (2023), arXiv v2

[arXiv:2304.05128v2](https://arxiv.org/abs/2304.05128v2). Selected framework
and feedback variants (3), applications (4), setup and sample-efficiency/code-
execution ablations (5), and conclusion (7) were read. Title and prompt pages
were inspected visually; appendices were not exhaustively reviewed.

Self-Debugging combines execution/test results with model-generated code
explanations or simulated traces, then asks the same model to revise its
program. The traces are model simulations, not observed intermediate states.
It reports improvements on Spider, TransCoder, and MBPP and improved sample
efficiency. The feedback can be externally evaluated by tests or generated by
the model; these cases must not be treated as equivalent. The method repeatedly
reuses failed predictions and spends additional inference compute, without
updating model weights.

This is a direct precedent for error-directed revision. Its demonstrated results
use already capable models. Correctness may be test-checked or model-judged;
even Spider supplies executed SQL results despite lacking a unit-test
correctness oracle. Model-generated explanations do not remove that external
execution information. It
does not show that an arbitrarily small model acquires general intelligence from
feedback alone, and its iterative inference cost must be matched when comparing
against a static baseline.

## ExpeL — Zhao et al. (2023/2024), arXiv v3

[arXiv:2308.10144v3](https://arxiv.org/abs/2308.10144v3). Selected mechanism
and transfer sections (4), setup/results/ablations (5), and limitations (6)
were read. Title and insight/transfer pages were inspected visually; appendices
were not exhaustively reviewed.

ExpeL gathers successful and failed trajectories, extracts editable natural-
language insights using an LLM, and appends the full insight list alongside
retrieved similar successful trajectories at evaluation. Insight retrieval is
a proposed future extension when the list grows beyond the context budget. It reports gains over ReAct and a transfer
experiment from HotpotQA to FEVER. That transfer adapts insights with GPT-4 and
target-task few-shot examples, using fixed examples without experience-pool
retrieval. The paper emphasizes learning without
parameter updates, interpretability, and explicit edit/upvote/downvote operations
on the insight list. It also notes that the base LLM remains important and that
insight extraction can hallucinate or be misled by poor trajectories.

ExpeL is the closest precedent to the proposed external knowledge compression
idea. Its experience pool, abstraction model, retrieval, task distribution, and
strong GPT-3.5/GPT-4 components are part of the system. Our paper must therefore
state that rule-memory accumulation and cross-task transfer already have an
empirical precedent. Study E tests neither natural-language abstraction nor
retrieval; a future experiment must compare raw transcripts with compressed,
verifiable rules and an independently held-out evaluator.

## Updated consequences for this manuscript

5. Voyager, Self-Debugging, and ExpeL make the intended novelty boundary more
   precise: the open question is a controlled resource comparison among feedback
   quality, memory representation, update rule, evaluator reliability, and
   persistent transfer, not the existence of feedback loops or agent memory.
6. A follow-up should compare raw transcripts, compressed rules, and error
   feedback under explicit inference budgets. Keep acceptance examples separate
   from a final held-out evaluator: final test answers must never guide repairs
   or entry into memory. Include rule extraction and verification in costs.
