# Study B protocol: feedback, size, and compositional transfer

Status: prospective draft, before scored model runs. Freeze by Git commit
after review. This study complements Study A with pretrained language models.
It remains a controlled synthetic study, not a general-intelligence benchmark.

## Environment and independent units

Each environment contains a new command language operating on integer states
0 through 7. Commands `dax`, `wug`, `zup`, and `fep` each add a hidden offset
modulo 8. Four distinct offsets are sampled without replacement from 0–7 for
each environment. Execute sequences left to right. The agent knows the family
(unknown additive offsets modulo 8), but does not see the offsets except in
the explicit-rule oracle control. Zero offsets are permitted.

Use 24 independent environments, seeds 3000–3023. Seeds below 100 are reserved
for development. All capacities receive exactly the same environments and
feedback streams. Use NumPy's default_rng, whose version is recorded.
Use separate RNG streams seeded by `[environment_seed, stream_id]`, with IDs
0 for offsets, 1 for train/test states and training order, 2 for composition
selection, and 3 for corruption. Enumerate commands in the order above and
composition candidates in state-major then command1-major then command2 order.
Freeze the generated environment JSON and its hash before inference.
For each command, randomly permute states 0–7. The first four states form
training queries and the remaining four form single-command test queries.
Shuffle the 16 training queries once. Sample 32 distinct length-two
(initial-state, command1, command2) queries without replacement from the 128
possibilities. Sequences are absent from the training feedback. They are a
compositional transfer endpoint, not a new task family. The two tests contain
16 and 32 queries respectively and are never used for selection or updates.
Modular additions commute. This endpoint cannot test sensitivity to command
order or general sequential execution; conclusions must say commutative
arithmetic composition.

## Models and scoring

Qwen2.5-Instruct at 0.5B, 1.5B, 3B, and 7B, with exact revisions in
`references/models/`. Load bfloat16 weights with Transformers 4.57.1. Run
deterministic inference without sampling, chain-of-thought, a judge model,
retrieval tools, or repeated attempts. Prompts explain the task family and ask
for exactly one state digit. Score the most likely token among `0` through `7`
at the first assistant response position. Verify each is one tokenizer token.
Also record the unrestricted most likely token and unconstrained first-token
accuracy. This diagnoses first-token behavior, not exact full-response format.

All conditions use the same system instruction, query formatting, and tokenizer
chat template. Feedback consists of the observed correct output for a query.
Input queries are fixed independently of actions. Therefore the studied loop
is passive online supervised adaptation, not active exploration or RL.

## Conditions

- `none`: no environment examples, frozen weights.
- `clean`: append correct query/output pairs to context. Evaluate after 4, 8,
  and 16 feedback pairs; endpoints are fixed rather than selected.
- `corrupted`: after 16 pairs, replace each output independently with probability
  0.5 by a uniformly sampled wrong digit. Same input and example counts as clean.
  This channel retains some information, unlike binary noise50 in Study A.
- `self`: predict the 16 training outputs sequentially and retain those guesses
  as the example history, with no environmental answers. Evaluate after 16.
  This supplies additional inference and a same-length history; wrong guesses
  may reinforce themselves. It is a pseudo-label control, not a full verbal
  Self-Refine implementation. Report its extra calls and tokens.
  Query i uses only guesses 1 through i-1, selecting the constrained digit
  argmax and appending it with exactly the clean demonstration format. Save
  every guess and prompt hash. Self histories are specific to each model;
  externally labeled histories are common to all capacities.
- `oracle`: disclose the correct four offsets in the instruction, with zero
  examples. This tests execution capacity given a known rule and is not a
  resource-matched competitor.
- `offline`: the prompt after 16 feedback pairs is exactly the clean transcript
  supplied at once. Verify byte identity. No repeat of the identical model call
  is counted as an independent observation. This equality exposes that this
  passive context loop has no final-state advantage over equivalent examples.
- `symbolic`: from one correct example for each command compute its offset and
  execute compositions exactly. This hand-coded comparator reports the effect
  of supplied algorithmic structure. It is not a zero-cost learned model.

## Persistent-update extension

Only on the 0.5B model, reset to identical base weights for each environment,
add rank-8 LoRA adapters to all q_proj and v_proj matrices (alpha=16, dropout=0,
no trained bias), and train on the same 16 clean query/output examples.
Use AdamW with learning rate 2e-4, weight decay 0, betas (0.9,0.999), eps=1e-8;
32 passes through a fixed shuffled stream in batches of 4 (128 updates). Loss
is only on the next correct digit, not on prompt tokens. Report actual trainable
parameters, update time, processed tokens, and peak allocated GPU memory.
Seeds 3000–3023 seed adapter initialization and training order independently.
Training prompts contain only the task instruction and the current query,
with no demonstrations. Use the same chat template and assistant-generation
prefix as evaluation, with full-vocabulary cross-entropy on its next-token
logits. Left-pad batches with an attention mask; no prompt-token loss.
Use PyTorch 2.8.0, PEFT 0.17.1, default LoRA initialization (Kaiming A and zero
B), fresh adapters/optimizer per environment, and constant learning rate with
no scheduler. Adapter initialization uses `seed + 4000000`; training uses
the frozen feedback order repeated identically for each of the 32 passes.

Evaluate the final adapter on both test sets with an empty example context
(`lora`) and with the 16 feedback pairs (`lora_context`). No intermediate best
checkpoint selection. Save adapter state hashes, configuration, and predictions.
This tests persistence under removal of context, not lifelong retention across
unrelated environments. The 32 passes are extra optimization and must not be
portrayed as matched cost with frozen inference. If the learning schedule fails,
report it and design any tuned follow-up on fresh seeds.

## Endpoints and statistics

Primary: mean length-two accuracy of clean-16 minus self-16 within the 0.5B
model, paired by environment. Report its 95% paired t interval with all 24
environments, plus an environment bootstrap interval as sensitivity analysis
(10,000 resamples, fixed RNG 817). This tests grounded evidence versus guesses;
it does not isolate environment information from every other difference.
Also report paired clean-minus-none and self-minus-none. A positive primary
contrast alone may reflect harmful guesses and cannot establish an improvement
over receiving no feedback.

Secondary: all capacity-by-condition means, paired small-clean vs large-none
and small-clean vs large-clean, direct-query versus compositional performance,
corruption, oracle execution, and lora without context. Secondary intervals
are descriptive, without multiple-testing significance claims. Report raw
per-query predictions, environment aggregates, and mean/SD across environments.
All benchmark examples are synthetic but pretrained knowledge of arithmetic
and instruction following remains a substantial prior.

## Accounting and stopping

Record input tokens, scored tokens, forward calls, optimization tokens, memory,
and wall time separately. Use no parameter-only FLOP claim: sequence length,
attention, pretraining, feedback construction, and quantization matter. The
four model sizes have different training histories; this is a released-model
comparison, not a causal parameter-count experiment. No total-compute equality
is assumed. API models or undisclosed larger teachers are not used.

Run the fixed grid once. Preserve failures and any protocol amendment. No
successful table proves unrestricted feedback sufficiency, and failure of a
specific prompt or adapter schedule does not refute every possible mechanism.
Study completion requires raw predictions, exact revisions, runnable commands,
checks of split separation and prompt identity, and manuscript limitations.
