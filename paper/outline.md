# Paper outline

## Title

**Feedback Is All You Need**

Possible explanatory subtitle: *A Closed-Loop Theory of Intelligence*.

## Abstract

State the scaling assumption, introduce the feedback hypothesis, define the
minimum-capacity boundary, describe the small-agent comparison, and summarize
the evidence needed to distinguish feedback-driven improvement from extra
memory, tools, data, or test-time compute.

## 1. Introduction

- The success of parameter and data scaling.
- The difference between stored competence and adaptive intelligence.
- The feedback hypothesis and its precise scope.
- Contributions and falsifiable predictions.

## 2. What “feedback” means

- Outcome feedback from the environment.
- Process feedback from tests, verifiers, and critics.
- Prediction error and model mismatch.
- Social and instructional correction.
- Delayed feedback, credit assignment, and long-horizon objectives.

## 3. Related perspectives

- Transformer attention and static sequence modeling.
- Scaling laws and compute-optimal training.
- Reinforcement learning and policy improvement.
- Continual learning and memory consolidation.
- Predictive processing, active inference, and control theory.
- Tool-using and environment-interacting agents.

## 4. A feedback-centric model of intelligence

Define a minimal agent with a base model, state, memory, action policy,
environment, evaluator, and update rule. Separate:

- representational capacity;
- feedback quality;
- feedback latency and density;
- adaptation rate;
- memory horizon;
- transfer across tasks.

## 5. Experimental design

- Small and large model baselines.
- Static, offline-feedback, and online-feedback conditions.
- Matched data, compute, interaction, and tool budgets.
- Ablations for memory, feedback type, update frequency, and credit assignment.
- Evaluation on adaptation, transfer, recovery, and long-horizon tasks.

## 6. Results

Report both positive and negative results. Measure capability gain per unit of
feedback and compare it with capability gain per additional parameter or
training token.

## 7. Failure modes and limits

- Insufficient representational capacity.
- Noisy or misaligned feedback.
- Reward hacking and feedback-loop instability.
- Catastrophic forgetting and over-specialization.
- External-memory and tool-use confounds.
- The difference between system intelligence and model intelligence.

## 8. Discussion

Explain what the results imply for scaling, agent design, human learning, and
externalized cultural memory without claiming more than the experiments show.

## 9. Conclusion

State whether the evidence supports feedback as the primary driver of adaptive
intelligence after the minimum-capacity threshold.
