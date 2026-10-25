# Experiments

Experiments are organized by study and must remain tied to a frozen protocol.
Each study records the code revision, environment, model configuration,
feedback signal, update rule, compute proxy, seeds, raw outputs, and analysis.

## Study A

Study A is a small online-learning experiment for the conditional claim that
feedback can produce adaptation after a task changes. It uses linear and XOR
classification rules so that the experiment can expose both feedback effects
and a representation-capacity boundary. It does not measure general
intelligence, language reasoning, or human learning.

- Protocol: [`../notes/protocol-v1.md`](../notes/protocol-v1.md)
- Runner: [`study_a/run.py`](study_a/run.py)
- Tests: [`study_a/test_run.py`](study_a/test_run.py)
- Output: a directory outside Git containing metadata, JSONL rows, and logs.

Run the smoke test before a full study. The fixed grid is intentionally
specified in the protocol rather than inferred from the runner's defaults.
