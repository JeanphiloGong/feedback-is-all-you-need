# Feedback Is All You Need

Version 0: conceptual manuscript, 2026-10-25. No experimental results yet.

## Abstract

Can efficient feedback compensate for a small cognitive core? We investigate
the hypothesis that a model with sufficient task representation can acquire
substantial adaptive capability through repeated interaction, correction, and
retention. We separate model parameters from external memory, useful feedback
from self-generated critique, and persistent learning from repeated inference.
We propose controlled capacity-by-feedback experiments followed by pretrained
language-model studies. The universal claim that feedback alone is sufficient
for intelligence remains unproven; this manuscript develops tests that can
support or reject restricted forms of it. No empirical results are reported in
this version.

## 1. Introduction

An agent's deployed behavior depends on more than its number of parameters.
It can act, observe consequences, revise its state, and preserve knowledge for
later decisions. The motivating hypothesis is that improving this loop can
yield large gains without proportional increases in the core model's size.
External records, including books, suggest a useful analogy with cumulative
human knowledge, but the analogy is not empirical evidence about AI or brains.

The question is not whether feedback matters: reinforcement learning, control,
meta-learning, and gradient-based training already make it central. The
question is when feedback quality and update efficiency substitute for capacity,
how far the substitution extends, and what prevents further improvement.

## 2. Background

The title follows Vaswani et al. (2017) as requested by the researcher. Chen
et al. (2023) already use a closely related title and feedback-centered framing.
Reflexion, Self-Refine, RL2, Algorithm Distillation, and test-time scaling also
make important parts of the proposed idea precedented. Novelty must reside in
a precise mechanism or a new, well-controlled finding, not the slogan.

Scaling-law studies measure empirical relationships between loss, parameters,
data, and compute. They do not exclude feedback, and they do not claim to
explain all intelligence. Our comparison must therefore evaluate competing
resource allocations rather than assume the existing literature is false.

## 3. Agent and feedback model

Let theta be the controller's learned parameters, m its persistent memory,
and h its current interaction history. An action is sampled from
`pi_theta(a | observation, m, h)`. The environment returns an observation and
feedback f. An update rule U maps the existing state and this new evidence to
the next state. U can change the context, memory, parameters, or a learned skill.
Experiments must identify exactly which of these components change.

A small parameter count does not imply a small total system. We separately
account for memory, evaluator knowledge, inference, interaction, and update
cost. We do not assume a universal representation threshold exists.

## 4. What feedback can and cannot provide

### 4.1 Identification requires information

Consider a hidden task Theta drawn uniformly from K possibilities, independent
of the agent's initial parameters and random seed. If n responses contain at
most b bits each, including all task-dependent response channels, their total
mutual information about Theta is at most nb bits. Adaptive queries do not
create additional information by themselves. Fano's inequality then gives
`P(task-identification error) >= 1 - (nb + 1) / log2(K)` for K > 1, clipped at
zero. This bound concerns exact task identification, not every possible
prediction metric. Informative prior knowledge changes the bound.

For a binary symmetric channel with independent flip probability p, each
response conveys at most `1 - H_2(p)` bits. At p = 0.5 the feedback carries no
task information. At p = 1 the channel is invertible if its corruption rule is
known. Therefore learner failure under inverted feedback is not information
destruction; it is failure to model the feedback channel.

### 4.2 Feedback is constrained by representation

If the permitted complete agent cannot represent a successful policy, updating
its parameters more often cannot make that policy attainable. External memory
or tool computation can enlarge the permitted policy class. The constraint
must apply to the whole evaluated agent, not just its neural core.

### 4.3 A loop is not automatically a new learning algorithm

For a deterministic update rule, replaying exactly the same ordered transcript
from the same initial state produces the same final state, by induction over
updates. This statement assumes the recorded observations remain the inputs
and excludes new counterfactual interaction. An online loop can help by
selecting informative actions or reacting to changing environments; it does
not obtain an unexplained benefit merely from being called a loop.

## 5. Experimental protocol

Study A factors model size, corrective feedback, label noise, delivery delay,
and replay on changing linear and XOR tasks. Independent evaluation examples
measure adaptation, forgetting, and distribution transfer. Study B must test
pretrained language models, durable learning, equal-information controls, and
inference budgets. Protocols are frozen before scored runs.

## 6. Results

Not available in this version. No table, effect size, or conclusion is inferred
from the hypothesis. The proposed protocol is in `notes/protocol-v1.md`.

## 7. Conclusion

The project asks how much intelligence a compact controller can acquire through
feedback. A defensible answer requires both successful adaptation and explicit
failure boundaries. Neither a small-model win nor a toy learning curve alone
can establish the nature of intelligence.

## References

Verified source identifiers and the original Attention paper are retained in
`references/`; the evolving related-work map is in `notes/literature.md`.
