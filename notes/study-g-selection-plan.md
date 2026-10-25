# Exploratory selection from Study G's recorded proposals

Declared after the complete Study G results and manuscript v8 were reviewed
and committed. This is a post-hoc description of the frozen proposal pools,
not a new confirmatory study or a replacement of the registered final-proposal
scores. No new model calls, changed prompts, normalized invalid rules, or
parameter updates are involved. The original collection contract remains
`protocol-v7-feedback-history.md` at source `c4b4054`.

Question: how much observed progress is discarded by using only the last
proposal, and how much could a supplied selector recover from that same pool?
The existing outputs already disclose failures and regressions. This analysis
cannot establish what a model would generate if an acceptance decision changed
its later prompt or feedback; that requires a new prospective intervention.

## Fixed selection rule

For each of 48 environments, model, and branch, take the shared initial rule
followed by that branch's 16 recorded proposals. Reparse with the unchanged
strict F/G parser. At checkpoints 0, 1, 2, 4, 8, and 16, consider only the
prefix through that checkpoint; never use a later proposal.

Rank valid candidates by the number of the 16 disclosed training examples
they execute correctly. Rank invalid proposals below every valid proposal,
including a valid rule that passes no examples. Break ties by the earliest
stage. If the prefix contains only invalid outputs, retain its initial invalid
proposal and score it zero. The selector receives rules and training examples
only: neither hidden queries/answers nor the true offset vector may select a
candidate. Exactness and held-out scores are evaluated after selection.

Preserve the selected stage, validity, training passes, exactness, direct and
composition scores, and selected-minus-current answer transitions at every
checkpoint. Report means and counts over 48 environments per model/branch,
including the number of ever-exact pools, exact rules discarded by the final
proposal, earlier-stage selections, and direct/composition gains and losses.
All outcomes are descriptive; no new primary endpoint, significance claim,
or checkpoint selection is introduced. Keep full-history and last-only pools
separate and include the shared initial candidate in each.

## Supplied capability and verification

This selector adds an algorithm that scores **all** examples for every valid
candidate. G's generation feedback reported only the first failing example.
Count these extra example executions per standalone branch, including repeated
candidates; invalid proposals require parsing but no example execution. Report
the offline analysis runtime separately from the already incurred generation
cost. Do not describe this as a free or matched-compute intervention.

The known family makes training accuracy equal to direct accuracy for each
valid rule: every command has four train inputs and four disjoint direct
inputs, and a wrong additive offset fails every input for that command.
Invalid rules score zero on both. Selecting maximal training accuracy thus
guarantees nondecreasing direct accuracy over these fixed prefixes. This is
a property of the supplied representation and selector, not evidence of neural
learning, generalization to a different rule family, or a novel algorithm.
Composition can fall even while training accuracy rises because of the
previously disclosed +4-offset alias. Exact recovery under selection is simply
whether an exact valid rule ever appears in the prefix.

Trust the previously audited local journal schema, record hashes of all input
files and directly used analysis sources, and preserve the original artifacts.
Verify earliest ties, invalid-versus-valid ranking, recovery after regression,
prefix causality, and the composition counterexample with focused tests.
Independently check the reported counts and selected stages before archiving.
