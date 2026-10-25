# Exploratory transition diagnostics after Study F

Declared after viewing the complete formal Study F results and finishing
manuscript v7. This is a post-hoc description of the existing frozen journals,
not a new confirmatory experiment or a change to the registered scores. No
model calls, prompt changes, or rescoring of invalid proposals are involved.
The baseline, feedback, update rule, inference budget, and sampling protocol
remain those in `protocol-v6-rule-memory.md` at source `08725bb`.

The next mechanism question is whether revisions repeat a failed proposal,
repair the specific checked command, or damage already-correct components.
For each model and branch, examine the 24 initial-to-first and 24 first-to-final
transitions. Treat them as repeated observations within 24 environments, not
48 independent experimental units. Report counts and denominators only.

1. Compare exact response text for immediate repetition and a return to the
   initial text after a different first revision. Report invalid repetitions
   separately from valid incorrect repetitions and correct repetitions.
2. Where the previous rule is valid but fails the disclosed examples, identify
   the first failing command in their fixed order. If the next rule is valid,
   record whether that command's offset changed and whether it became correct.
   Invalid next rules are reported separately, without claiming a component
   repair or a semantic loss based on the null parsed memory.
3. For valid-to-valid transitions, count offsets that were correct before and
   become wrong afterward, and incorrect offsets that become correct. Report
   both component totals and the number of transitions that contain them.
4. Report all-example-pass retention, valid-to-invalid and invalid-to-valid
   transitions, and the number of environments with a repeated final text.

Use only the training examples to infer true offsets for diagnosis; hidden
query answers are unnecessary. No causal attribution follows from these
counts. Repeated text is evidence of an observed repetition, not proof of a
permanent cycle: a future changed prompt, state, or numerical batching could
change the output. Candidate memory, monotone acceptance, longer horizons,
and arithmetic tools remain interventions for separately frozen experiments.
