"""Reject altered feedback, weights, Adam state, probes, and resource records."""

import json
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_m import audit, run
from experiments.study_m.tasks import make_world


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.world = make_world(-937, 'structured')
        cls.folder = cls.root / 'seed--937' / 'structured'
        with patch.object(run, 'INTERACTIONS', 16), patch.object(run, 'CHECKPOINTS', (0, 16)):
            run.collect([cls.world], cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def check(self):
        phases = audit.read_records(self.root / 'phases.jsonl')
        with patch.object(audit, 'INTERACTIONS', 16), patch.object(audit, 'CHECKPOINTS', (0, 16)):
            return audit.audit_world(self.world, self.folder, phases)

    def test_complete_replay_matches_every_journal_and_checkpoint(self):
        result = self.check()
        self.assertEqual(result['gradient_steps'], 480)
        self.assertEqual(result['main_actions'], 128)
        self.assertEqual(result['checkpoints'], 20)
        self.assertEqual(result['probe_tasks'], 13 * 248)

    def test_modified_action_update_graph_and_probe_are_rejected(self):
        corruptions = [('actions.jsonl', 'next_state', 999),
                       ('updates.jsonl', 'loss_before_step', -1),
                       ('graphs.jsonl', 'logits_sha256', 'altered'),
                       ('probes.jsonl', 'success', None),
                       ('oracle.jsonl', 'distance', -1)]
        for filename, field, value in corruptions:
            path = self.folder / filename
            original = path.read_bytes()
            try:
                rows = audit.read_records(path)
                rows[0][field] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.subTest(filename=filename), self.assertRaises((AssertionError, ValueError)):
                    self.check()
            finally:
                path.write_bytes(original)

    def test_model_and_adam_changes_are_rejected(self):
        path = self.folder / 'small_immediate/16.pt'
        original = path.read_bytes()
        for target in ('model', 'optimizer'):
            try:
                data = torch.load(path, weights_only=True)
                if target == 'model':
                    next(iter(data['model'].values())).flatten()[0] += .01
                else:
                    data['optimizer']['state'][0]['exp_avg'].flatten()[0] += .01
                torch.save(data, path)
                with self.subTest(target=target), self.assertRaises((AssertionError, ValueError)):
                    self.check()
            finally:
                path.write_bytes(original)

    def test_checkpoint_dtype_change_is_rejected_even_when_file_size_matches(self):
        path = self.folder / 'small_immediate/16.pt'
        ledger = self.root / 'phases.jsonl'
        original, original_ledger = path.read_bytes(), ledger.read_bytes()
        try:
            data = torch.load(path, weights_only=True)
            data['model'] = {k: v.to(torch.complex128) for k, v in data['model'].items()}
            torch.save(data, path)
            rows = audit.read_records(ledger)
            for row in rows:
                if row['stage'] == 'checkpoint' and row['arm'] == 'small_immediate' and row['at'] == 16:
                    row['bytes'] = path.stat().st_size
            ledger.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(AssertionError):
                self.check()
        finally:
            path.write_bytes(original)
            ledger.write_bytes(original_ledger)

    def test_phase_costs_duplicate_rows_and_newline_changes_are_rejected(self):
        path = self.root / 'phases.jsonl'
        original = path.read_bytes()
        try:
            for field, value in (('environment_actions', 999), ('seconds', -1), ('journal_bytes', 0)):
                rows = audit.read_records(path)
                rows[0][field] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                    self.check()
                path.write_bytes(original)
            path.write_bytes(original + original.splitlines(keepends=True)[0])
            with self.assertRaises((AssertionError, ValueError)):
                self.check()
        finally:
            path.write_bytes(original)


    def test_null_followed_by_extra_rows_is_rejected(self):
        path = self.folder / 'updates.jsonl'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'null\n{"unaccounted_extra_record": true}\n')
            with self.assertRaises(AssertionError):
                self.check()
        finally:
            path.write_bytes(original)

    def test_missing_journal_timing_is_rejected(self):
        path = self.root / 'phases.jsonl'
        original = path.read_bytes()
        try:
            rows = audit.read_records(path)
            del rows[0]['journal_seconds']
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(AssertionError):
                self.check()
        finally:
            path.write_bytes(original)

    def test_changed_journal_newlines_are_rejected(self):
        path = self.folder / 'updates.jsonl'
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b'\n', b'\r\n'))
            with self.assertRaises(AssertionError):
                self.check()
        finally:
            path.write_bytes(original)


class BoundaryTests(unittest.TestCase):
    def test_complete_development_formal_gate_and_mutation_during_audit(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for module in (run, audit):
                for name, value in (('INTERACTIONS', 16), ('CHECKPOINTS', (0, 16)),
                                    ('DEVELOPMENT_SEED', -947), ('FORMAL_SEEDS', (-948,))):
                    stack.enter_context(patch.object(module, name, value))
            # Synthetic boundary fixture only; registered-runtime checks run on lens.
            stack.enter_context(patch.object(torch, '__version__', '2.8.0+cu126'))
            data = root / 'worlds.json'
            data.write_text(json.dumps([make_world(seed, family) for seed in (-947, -948)
                                       for family in ('structured', 'random')], indent=2) + '\n')
            stack.enter_context(patch.object(run, 'DATA_PATH', data))
            stack.enter_context(patch.object(audit, 'DATA_PATH', data))
            development, formal = root / 'development', root / 'formal'
            with patch.object(sys, 'argv', ['run', '--output', str(development),
                                          '--source-revision', 'synthetic', '--development']):
                run.main()
            certificate = audit.audit_run(development)[-1]
            self.assertEqual(certificate['main_actions'], 256)
            self.assertEqual(certificate['gradient_steps'], 960)
            path = root / 'audit.json'
            path.write_text(json.dumps(certificate, indent=2) + '\n')
            with patch.object(sys, 'argv', ['run', '--output', str(formal),
                    '--source-revision', 'synthetic', '--development-audit', str(path)]):
                run.main()
            self.assertEqual(audit.audit_run(formal)[-1]['status'], 'passed')
            reconstruct = audit.audit_world
            def mutate(world, folder, phases):
                result = reconstruct(world, folder, phases)
                (formal / 'active-phase.json').write_text(json.dumps(dict(stage='altered')) + '\n')
                return result
            with patch.object(audit, 'audit_world', side_effect=mutate), self.assertRaises(AssertionError):
                audit.audit_run(formal)

    def test_scalar_representation_must_match(self):
        for actual, expected in ((1, True), (8., 8), (False, 0)):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                audit.same(actual, expected)

    def test_tensor_representation_must_match(self):
        with self.assertRaises(AssertionError):
            audit.same(torch.ones(2, dtype=torch.complex128), torch.ones(2, dtype=torch.float64))

    def test_duplicate_document_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.json'
            path.write_text('{"status": "failed", "status": "completed"}\n')
            with self.assertRaises(AssertionError):
                audit.read_document(path, 2)

    def test_optimized_python_is_rejected_before_reading_inputs(self):
        result = subprocess.run([sys.executable, '-B', '-O', '-c',
            "from pathlib import Path; from experiments.study_m.audit import audit_run; audit_run(Path('/no-study-m-input'))"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires Python assertions', result.stderr)


if __name__ == '__main__':
    unittest.main()
