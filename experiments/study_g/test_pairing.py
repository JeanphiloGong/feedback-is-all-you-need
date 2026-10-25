"""Do not accept a history comparison whose identical first revision diverges."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.study_g import run


class Batch(dict):
    def to(self, device):
        return self


class PairingTest(unittest.TestCase):
    def test_divergence_keeps_raw_output_and_fails_the_suite(self):
        for mismatch in ("inputs", "outputs"):
            with self.subTest(mismatch=mismatch), TemporaryDirectory() as temporary:
                class Tokenizer:
                    calls = 0

                    def apply_chat_template(self, messages, **kwargs):
                        return json.dumps(messages)

                    def __call__(self, prompts, **kwargs):
                        self.calls += 1
                        token = 7 if mismatch == "inputs" and self.calls == 3 else 1
                        return Batch(input_ids=torch.full((1, 4), token, dtype=torch.long),
                                     attention_mask=torch.ones((1, 4), dtype=torch.long))

                    def decode(self, ids, **kwargs):
                        return '{"dax":0,"wug":1,"zup":2,"fep":3}'

                class Model:
                    calls = 0
                    device = "cpu"
                    config = type("Config", (), {"max_position_embeddings": 32768})()
                    generation_config = type("GenerationConfig", (), {
                        "to_dict": lambda self: {"eos_token_id": [0]}, "eos_token_id": [0]})()

                    def eval(self):
                        return self

                    def parameters(self):
                        return [torch.zeros(1)]

                    def generate(self, input_ids, **kwargs):
                        self.calls += 1
                        token = 3 if mismatch == "outputs" and self.calls == 3 else 2
                        return torch.cat((input_ids, torch.tensor([[token, 0]])), dim=1)

                output = Path(temporary) / "run"
                argv = ["run", "--size", "0.5B", "--development", "--output", str(output)]
                with patch("sys.argv", argv), \
                     patch.object(run.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                     patch.object(run.AutoModelForCausalLM, "from_pretrained", return_value=Model()), \
                     patch.object(torch.cuda, "get_device_name", return_value="synthetic CPU fixture"), \
                     patch.object(torch.cuda, "reset_peak_memory_stats"), \
                     patch.object(torch.cuda, "synchronize"):
                    with self.assertRaisesRegex(AssertionError, f"First-revision .*{mismatch} differ"):
                        run.main()
                rows = [json.loads(x) for x in (output / "proposals.jsonl").read_text().splitlines()]
                batches = [json.loads(x) for x in (output / "batches.jsonl").read_text().splitlines()]
                self.assertEqual(len(batches), 3)
                self.assertEqual(batches[1]["attention_mask"], batches[2]["attention_mask"])
                self.assertEqual(batches[1]["input_ids"] == batches[2]["input_ids"], mismatch == "outputs")
                self.assertEqual([(r["arm"], r["stage"]) for r in rows],
                                 [("initial", 0), ("last_only", 1), ("full_history", 1)])
                self.assertFalse(json.loads((output / "metadata.json").read_text())["completed"])
                self.assertEqual(json.loads((output / "failure.json").read_text())["error_type"], "AssertionError")


if __name__ == "__main__":
    unittest.main()
