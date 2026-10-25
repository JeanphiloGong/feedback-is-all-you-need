# Exploratory uniform-proposal reference after Study G selection

Declared after the selection diagnostics at `d961cf6`. The new question is
whether the observed candidate pools offer more useful rules than uniform
legal proposals given the same supplied training-example selector. This is a
post-hoc reference calculation, not a new primary endpoint or neural run.

## Policies and exact calculation

There are 8 * 7 * 6 * 5 = 1,680 ordered legal four-offset rules. A uniform
proposal draws independently with replacement from this set. The action
space and validity constraints are supplied; no model learns them. Uniform
proposals do not use feedback. The external selector scores all disclosed
examples and retains the earliest maximum exactly as in the selection plan.

Report two policies, without choosing between them after seeing results:

1. Seventeen uniform proposals, with no neural initial proposal.
2. The observed shared neural initial proposal followed by sixteen uniform
   proposals. This is a model-free revision policy, not a fully model-free
   agent: the initial generation and its cost remain supplied by that model.

Enumerate all legal rules against canonical truth (0, 1, 2, 3) and count their
number M of correctly matched positions. Every legal true vector is an
injective relabeling of that canonical vector, so the distribution of M is
invariant. Let F(j) = P(M <= j). For n independent draws and k initially
correct offsets, expected selected direct accuracy is exactly

    [k + sum_{j=k+1}^4 (1 - F(j-1)^n)] / 4.

The no-neural-initial policy uses k=0 and n=17. Invalid initial proposals also
have direct score zero; after at least one draw a valid proposal is available.
The probability of an exact selected rule is 1 when the initial rule is exact,
and otherwise 1 - (1 - 1/1680)^n. Compute with rational arithmetic and report
both exact fractions and decimal values, without Monte Carlo sampling.

For each model, condition the sixteen-draw reference on all 48 recorded initial
rules. Compare its mean expected direct score and expected exact count with
both observed selected pools. Use only the already audited training-pass
counts to identify k; hidden queries, true offsets, and final model candidates
do not determine the random proposal law. Report per-environment conditional
expectations and aggregate differences. These are descriptive comparisons of
observed outcomes with policy expectations, not new hypothesis tests or
confidence intervals. No composition reference is claimed: the number of
correct offsets alone does not determine composition under the known alias.

## Costs, attribution, and checks

Report additional proposal counts and full-example executions. Each uniform
candidate is valid and incurs 16 example executions, including repeats.
Count checking the shared neural initial proposal only when it is valid.
The pure seventeen-draw reference has no neural generation; the sixteen-draw
reference retains the original initial-generation cost. These policies match
candidate opportunities, not measured compute, checker counts, learned prior
knowledge, or formatting competence. They build the finite rule grammar into
the sampler. Exact expectation calculation is an offline analysis, not a
timed deployment of random agents. Record its runtime separately.

Check the enumerated match counts against inclusion-exclusion, one-draw
expectation against the 1/8 marginal matching probability, two-draw expectation
against a direct weighted pair sum, and preservation of an already exact
initial rule. Independent review must verify the formula, attribution, raw
initial training counts, and all reported comparisons before archiving.
Record analysis/input hashes and leave all previous sources/results unchanged.
