# Reference material

`attention-is-all-you-need.pdf` is the original NeurIPS 2017 proceedings paper
by Vaswani et al., downloaded from the publisher's proceedings site at the
researcher's request. `attention-manifest.json` records its URL, retrieval
time, size, and SHA-256. The PDF retains the original authorship; it is source
material, not part of this project's original manuscript.

The proceedings version is 11 pages and reports 41.0 English-French BLEU;
the later arXiv abstract reports 41.8. Do not silently mix these versions.

## Structural reference

| Original paper | Our manuscript |
| --- | --- |
| Abstract | Research question, intervention, measured findings, limits |
| 1. Introduction | Feedback hypothesis and competing explanations |
| 2. Background | Control, reinforcement learning, scaling, continual adaptation |
| 3. Model Architecture | Agent state, feedback channel, memory, update rule |
| 4. Why Self-Attention | What feedback can and cannot identify |
| 5. Training | Environments, budgets, optimizer, seeds, evaluation |
| 6. Results / Model Variations | Capacity comparisons, controls, ablations |
| 7. Conclusion | Supported conclusions and unresolved claims |
| References | Verified primary sources |

The reference supplies an organizational model. It does not supply our
experimental findings, prose, or a theory of intelligence.

## Source records

`source-records.json` preserves metadata and abstracts retrieved from arXiv.
These are discovery records, not evidence that every full paper was reviewed.
`title-precedent.json` is Crossref's record for Chen et al. (2023),
*Feedback is all you need: from ChatGPT to autonomous driving*,
DOI [10.1007/s11432-023-3740-x](https://doi.org/10.1007/s11432-023-3740-x).
The closely related title and framing must be acknowledged. The publisher
endpoint returned a browser challenge instead of a PDF, so only metadata has
been verified for that work at this checkpoint.

`full-text-manifest.json` records exact copies and versions of Reward is enough,
RL², Reflexion, Self-Refine, the intrinsic self-correction study, and Welcome to
the Era of Experience. The
unmodified PDFs retain their authorship and original license notices.
[`../notes/full-text-review.md`](../notes/full-text-review.md) records the
sections read and how their methods constrain our claims. Reward is enough
was read from an image-only author copy; its Crossref metadata is preserved in
`reward-is-enough.json`.
