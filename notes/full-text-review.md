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
research agenda as novel. Studies B–D remain short, synthetic inference
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
