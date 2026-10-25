# Literature map, first pass

Status: source metadata and abstracts verified on 2026-10-25; the original
Attention paper's relevant sections were read in full. This is a targeted
review, not a systematic or exhaustive novelty search.

Update: a [targeted full-text review](full-text-review.md) now covers the methods
and limitations of five directly relevant works, including Reward is enough.
The table below remains the original discovery map; see the review for exact
versions and reading scope.

| Work | Relevance and constraint on our claim |
| --- | --- |
| Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Demonstrates an architecture with task-specific experiments and ablations; does not establish that attention alone is sufficient for all intelligence. |
| Kaplan et al., 2020, [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Studies loss as a function of parameters, data, and compute; it is not a claim that feedback is unnecessary. |
| Hoffmann et al., 2022, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Optimizes joint model/data allocation. A critique of parameter count alone must not misrepresent this result. |
| Duan et al., 2016, [RL2](https://arxiv.org/abs/1611.02779) | A recurrent controller can implement a fast learning algorithm in state. Small-core adaptation is not a new concept. |
| Laskin et al., 2022, [Algorithm Distillation](https://arxiv.org/abs/2210.14215) | Learns reinforcement learning in context without runtime weight changes. Context adaptation must be distinguished from parameter learning. |
| Christiano et al., 2017, [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | Feedback-driven learning is already central to established AI research. |
| Shinn et al., 2023, [Reflexion](https://arxiv.org/abs/2303.11366) | Stores verbal reflections from feedback in episodic memory. Merely adding reflection and memory is not a novel contribution. |
| Madaan et al., 2023, [Self-Refine](https://arxiv.org/abs/2303.17651) | Iterative self-feedback can improve outputs; an extra-generation control is required. |
| Huang et al., 2023, [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | Intrinsic self-correction can fail. Distinguish independent environmental evidence from self-generated critique. |
| Snell et al., 2024, [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) | Already reports smaller models outperforming much larger models under some compute-matched conditions. A small-versus-large win alone is insufficient novelty. |
| Dohare et al., 2021, [Continual Backprop](https://arxiv.org/abs/2108.06325) | Continued optimization does not guarantee retained plasticity. Learning stability needs measurement. |
| Chen et al., 2023, [Feedback is all you need: from ChatGPT to autonomous driving](https://doi.org/10.1007/s11432-023-3740-x) | Direct title/framing precedent; full text still to be obtained. |

## Proposed contribution to investigate

A controlled account of when feedback substitutes for extra model capacity,
and when that substitution fails: information quality, delay, representation,
retention, and cost. Whether this is a publishable contribution depends on the
experiments and further literature review. We make no priority claim.

## Corrections to the seed framing

- Training already uses feedback through gradients, reward, and preference
  supervision. Our intervention is deployment-time adaptation and its budget.
- Parameter scaling and feedback are interacting variables, not disjoint
  scientific paradigms.
- Better performance on a changing task does not establish general intelligence
  or explain human intelligence.
- A universal threshold after which capacity ceases to matter is an additional
  hypothesis, not a premise that can be assumed into the experiment.
