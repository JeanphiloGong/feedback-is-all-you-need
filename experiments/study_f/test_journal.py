"""Preserve all already-generated proposals when checking the first one fails."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.study_b.run import REVISIONS
from experiments.study_f import run


class Batch(dict):
    def to(self, device):
        return self


class JournalTest(unittest.TestCase):
    def test_model_setup_failure_keeps_provenance_and_error(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            argv = ["run", "--size", "7B", "--development", "--output", str(output)]
            with patch("sys.argv", argv), patch.object(run.AutoTokenizer, "from_pretrained",
                    side_effect=OSError("injected cache failure")):
                with self.assertRaisesRegex(OSError, "injected cache failure"):
                    run.main()
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["completed"])
            self.assertEqual(metadata["revision"], REVISIONS["7B"])
            self.assertEqual(json.loads((output / "failure.json").read_text())["message"], "injected cache failure")

    def test_first_check_failure_keeps_the_whole_completed_batch(self):
        config = dict(eos_token_id=[0])

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)

            def __call__(self, prompts, **kwargs):
                return Batch(input_ids=torch.ones((len(prompts), 4), dtype=torch.long),
                             attention_mask=torch.ones((len(prompts), 4), dtype=torch.long))

            def decode(self, ids, **kwargs):
                return '{"dax":0,"wug":1,"zup":2,"fep":3}'

        class Model:
            device = "cpu"
            generation_config = type("Config", (), {"to_dict": lambda self: config, "eos_token_id": [0]})()

            def eval(self):
                return self

            def parameters(self):
                return [torch.zeros(1)]

            def generate(self, input_ids, **kwargs):
                return torch.cat((input_ids, torch.tensor([[2, 0]] * len(input_ids))), dim=1)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = dict(passed=True, files_sha256={}, metadata=dict(
                completed=True, development=True, seeds=[17],
                source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in run.SOURCES},
                model="Qwen/Qwen2.5-0.5B-Instruct", revision=REVISIONS["0.5B"],
                packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scipy", "matplotlib")},
                dtype="bfloat16", attention="sdpa", batch_size=16, generation_config=config,
                generation_overrides=dict(do_sample=False, max_new_tokens=512)))
            (root / "gate.json").write_text(json.dumps(gate))
            output = root / "output"
            argv = ["run", "--size", "0.5B", "--output", str(output),
                    "--development-audit", str(root / "gate.json")]
            with patch("sys.argv", argv), \
                 patch.object(run.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                 patch.object(run.AutoModelForCausalLM, "from_pretrained", return_value=Model()), \
                 patch.object(torch.cuda, "get_device_name", return_value="synthetic CPU fixture"), \
                 patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 patch.object(torch.cuda, "synchronize"), \
                 patch.object(run, "parse_rule", side_effect=RuntimeError("injected checker failure")):
                with self.assertRaisesRegex(RuntimeError, "injected checker failure"):
                    run.main()
            rows = [json.loads(line) for line in (output / "proposals.jsonl").read_text().splitlines()]
            self.assertEqual([r["seed"] for r in rows], list(range(8000, 8016)))
            self.assertFalse(json.loads((output / "metadata.json").read_text())["completed"])
            self.assertEqual(json.loads((output / "failure.json").read_text())["message"], "injected checker failure")


if __name__ == "__main__":
    unittest.main()
