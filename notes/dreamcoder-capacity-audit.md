# DreamCoder capacity follow-up: source evidence and limits

Reviewed on 2026-10-26 UTC, after manuscript v10. This is a targeted source
inspection, not a reproduction or a new performance experiment. The main-paper
record remains [arXiv 2006.08381v1](https://arxiv.org/abs/2006.08381v1).
No exact parameter count for its reported experiments has been established.

## Retrieval and version boundary

The supplement URL printed on p. 22 returns HTTP 404 in this session. The ACM
publication endpoint returns HTTP 403. Internet Archive lookup failed through
a TLS error and a direct-connection timeout; this does not establish that no
archived copy exists. The official arXiv v1 source package includes
`appendix.aux`, but no supplement PDF or appendix source body. Cross-reference
labels and figure captions do not substitute for the missing text.

The author's public repository provides a historical code snapshot:
[`ellisk42/ec@e6fd728`](https://github.com/ellisk42/ec/tree/e6fd72804194ed6cc0c88eee652413b0b44fc63e),
committed 2020-04-25. It is the latest commit returned from the repository's
master history before 2020-06-16 UTC. Its correspondence to the paper's actual
experimental checkpoints, invocation flags, and learned libraries is unknown.
A newer master snapshot is dated March 2022; it must not silently stand in for
2020 experimental evidence.

Five unmodified source files and their Git blob hashes, SHA-256 values, URLs,
retrieval times, and reading scope are recorded in
[`../references/dreamcoder-code-2020/manifest.json`](../references/dreamcoder-code-2020/manifest.json).
Only the relevant constructors and callers were inspected; this is not a
correctness audit of the entire external codebase.

## What the inspected code establishes

- The list and text domain `LearnedFeatureExtractor` constructors pass `H=64`
  and `bidirectional=True` to a recurrent feature extractor. It uses a
  64-dimensional symbol embedding and a one-layer bidirectional GRU. The two
  final directional states are added, giving a 64-dimensional task feature.
  Sixty-four hidden dimensions are not a count of all neural parameters.
- `RecognitionModel` defaults to one hidden layer of width 64 between features
  and a grammar-prediction head. Its feature extractor is a registered neural
  submodule. A separate auxiliary-prediction linear head is registered and
  trained even when `auxLoss=False`; that flag controls whether auxiliary-loss
  gradients also update the feature extractor. The inspected training loop
  always adds the auxiliary loss to a valid program loss before backpropagation.
- A non-contextual grammar head has one output per grammar production plus
  a variable output. Contextual variants additionally depend on parent/argument
  contexts and may use a dense, factored, or masked construction. Thus head
  shape depends on the current grammar and configuration. A fixed hidden width
  does not imply a fixed total parameter count as the library changes.
- The list and text extractors build their symbol vocabularies using the
  supplied training and testing task examples. The list extractor also derives
  a maximum sequence length from those examples. These preprocessing details
  are further reasons that a constructor width alone cannot identify an exact
  experiment parameter count; no performance-leakage conclusion is inferred.
- The main training caller creates new recognition-model objects during sleep.
  Weight reuse is configurable and defaults to false in the inspected outer
  interface. When supplied, `previousRecognitionModel` copies feature-extractor
  and intermediate-MLP states; it does not copy the new grammar or auxiliary
  head in that constructor. These are code paths and defaults, not verified
  claims about which flags were used in the reported paper runs.
- The repository offers a `countParameters` command that reads a saved model
  and counts registered tensors. We have not recovered or loaded the paper's
  corresponding trained checkpoints, so that route supplies no measured count
  for the publication here.

## Conditional count, with an independent shape check

For the inspected list/text feature architecture and the **non-contextual**
recognition path with default hidden width 64, let `V` be the number of embedding
rows after adding special symbols and `G` the number of grammar productions.
The source's primitive list and grammar length both have `G` entries. Counting
all registered parameters, including biases and the auxiliary head, gives:

| Component | Registered parameter count |
| --- | ---: |
| Symbol embedding | `64 V` |
| One-layer bidirectional GRU, input/hidden width 64 | `49,920` |
| Intermediate linear layer, 64 to 64 | `4,160` |
| Non-contextual grammar head, 64 to G+1 | `65 (G+1)` |
| Auxiliary head, 64 to G | `65 G` |
| Total | `64 V + 54,145 + 130 G` |

The GRU count follows from two directions, each with three input-gate matrices,
three recurrent-gate matrices, and both input and recurrent biases:
`2 × (3×64×64 + 3×64×64 + 3×64 + 3×64) = 49,920`.
The formula counts tensors, not the number of parameters actually receiving
nonzero gradients in a particular loss computation.

A small standalone script instantiates these module shapes in PyTorch
2.8.0+cu126 and verifies the formula for six synthetic `(V, G)` pairs. It does
not import or execute the archived DreamCoder code and performs no training.
For example, `(V=100, G=32)` gives 64,705 parameters **for that illustrative
configuration**, not an observed parameter count from the paper. The script
and exact output are in the source-reference directory. Other feature
architectures, hidden-layer settings, and contextual heads need different
counts. This check validates the arithmetic under the stated construction;
it does not recreate the historical framework or experimental checkpoint.

## Implication for this project

DreamCoder remains relevant evidence for combining learned search guidance
with external reusable representations. The inspected implementation contains
compact neural components, but its achieved performance cannot be assigned to
an arbitrary fixed small network without identifying the actual grammar,
configuration, prior training, synthetic tasks, search, and checkpoint. Our
Study H supplies symbolic search without a neural component; bridging that gap
requires a prospective test of learned search guidance with explicit parameter
and total-system accounting. Manuscript v10 remains unchanged by this follow-up.
