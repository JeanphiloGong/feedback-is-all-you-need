"""Corrupt synthetic evidence and verify independent reconstruction rejects it."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_l import analyze, audit, run
from experiments.study_l.test_collection import synthetic_environment


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.root = Path(cls.scratch.name)
        cls.environments = [synthetic_environment(seed) for seed in (41, *range(14000, 14032))]
        cls.data = cls.root / 'synthetic-grid.json'
        cls.data.write_text(json.dumps(cls.environments))
        cls.original = cls.root / 'original'
        argv = ['study_l.run', '--development', '--source-revision', 'synthetic', '--output', str(cls.original)]
        with patch.multiple(run, DATA_PATH=cls.data, BUDGET=512, INTERVAL=128, STEPS=2), \
             patch.object(run, 'source_hashes', return_value={}), \
             patch.object(run.torch, '__version__', '2.8.0+cu126'), patch.object(run.sys, 'argv', argv):
            run.main()

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def setUp(self):
        self.output = Path(tempfile.mkdtemp(dir=self.root)) / 'archive'
        shutil.copytree(self.original, self.output)
        self.addCleanup(shutil.rmtree, self.output.parent)

    def audited(self):
        lookup = {e['seed']: e for e in self.environments}
        with patch.multiple(audit, DATA_PATH=self.data, BUDGET=512, INTERVAL=128, STEPS=2), \
             patch.object(audit, 'source_hashes', return_value={}), \
             patch.object(audit, 'make_environment', side_effect=lookup.__getitem__), \
             patch.object(audit.torch, '__version__', '2.8.0+cu126'):
            return audit.audit_run(self.output)

    def test_complete_weight_replay_search_and_resource_reconstruction(self):
        rows, phases, updates, metadata, certificate = self.audited()
        self.assertEqual(len(rows), 130)
        self.assertGreater(len(updates), 4)
        self.assertGreater(certificate['candidate_events'], len(rows))
        self.assertEqual(certificate['gradient_steps'], len(updates))
        self.assertEqual(len(certificate['files']), len(list(p for p in self.output.rglob('*') if p.is_file())))
        with patch.multiple(analyze, BUDGET=512, BUDGETS=(128, 256, 512)):
            summary = analyze.summarize(rows, phases, updates, metadata, certificate)
        self.assertFalse(summary['primary']['confirmatory'])
        self.assertIsNone(summary['primary']['ci95'])
        self.assertEqual(summary['actual_cost']['gradient_steps'], len(updates))
        for cost in summary['standalone_cost_totals']:
            arm = cost['arm']
            searches = [r for r in rows if r['arm'] == arm or (arm != 'uniform' and r['arm'] == 'initial')]
            self.assertEqual(cost['primitive_ops'], sum(r['result']['primitive_ops'] for r in searches))
            self.assertLessEqual(cost['included_journal_seconds'], cost['instrumented_phase_seconds'])
            if arm == 'uniform':
                self.assertEqual(cost['gradient_steps'], 0)
                self.assertEqual(cost['table_calls'], 0)
            elif arm == 'frozen':
                self.assertEqual(cost['gradient_steps'], 4)
                self.assertEqual(cost['table_calls'], 1)

    def test_changed_weights_optimizer_and_partial_state_are_rejected(self):
        path = next((self.output / 'seed-41/rejection').glob('*/after-*.pt'))
        original = path.read_bytes()
        for mutation in ('weight', 'optimizer', 'rejected_count'):
            with self.subTest(mutation=mutation):
                saved = torch.load(path, weights_only=True)
                if mutation == 'weight':
                    next(iter(saved['model'].values())).flatten()[0] += 1
                elif mutation == 'optimizer':
                    next(iter(saved['optimizer']['state'].values()))['step'] += 1
                else:
                    saved['rejected'] += 1
                torch.save(saved, path)
                with self.assertRaises(AssertionError):
                    self.audited()
                path.write_bytes(original)

    def test_certificate_cannot_omit_rounds_tables_or_intermediate_checkpoints(self):
        certificate = self.audited()[-1]
        with patch.multiple(run, BUDGET=512, INTERVAL=128, STEPS=2):
            run.check_development_evidence(certificate, self.environments[0])
            for fragment in ('/round-', '/after-', '/table-'):
                altered = json.loads(json.dumps(certificate))
                removed = {name for name in altered['files'] if fragment in name}
                self.assertTrue(removed)
                altered['files'] = {name: value for name, value in altered['files'].items() if name not in removed}
                altered['checkpoints'] = [row for row in altered['checkpoints'] if row['file'] not in removed]
                with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                    run.check_development_evidence(altered, self.environments[0])

    def test_changed_failure_event_and_interrupted_state_are_rejected(self):
        paths = list((self.output / 'seed-41/rejection').glob('*/candidates.jsonl'))
        for event_type in ('rejected', 'partial'):
            for path in paths:
                original = path.read_text()
                events = [json.loads(line) for line in original.splitlines()]
                event = next((e for e in events if (e['status'] == 'rejected' if event_type == 'rejected'
                             else e['status'] == 'boundary' and e['pending'] is not None)), None)
                if event is not None:
                    break
            self.assertIsNotNone(event)
            if event_type == 'rejected':
                event['output'] += 1
            else:
                event['pending']['value'] += 1
            path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            with self.assertRaises(AssertionError):
                self.audited()
            path.write_text(original)

    def test_changed_update_table_and_order_are_rejected(self):
        for name in ('update', 'table', 'order'):
            if name == 'update':
                path = self.output / 'seed-41/updates.jsonl'
                original = path.read_text()
                rows = [json.loads(line) for line in original.splitlines()]
                next(r for r in rows if r['arm'] == 'rejection')['loss_before_step'] += .01
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            elif name == 'table':
                path = next((self.output / 'seed-41/rejection').glob('*/table-*.json'))
                original = path.read_text()
                table = json.loads(original)
                table[0][0] += .01
                path.write_text(json.dumps(table))
            else:
                path = next((self.output / 'seed-41/rejection').glob('*/round-*.json'))
                original = path.read_text()
                order = json.loads(original)
                order['programs'][0], order['programs'][1] = order['programs'][1], order['programs'][0]
                path.write_text(json.dumps(order))
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.audited()
            path.write_text(original)

    def test_changed_journal_costs_impossible_times_and_extra_phases_are_rejected(self):
        path = self.output / 'phases.jsonl'
        original = path.read_text()
        for mutation in ('bytes', 'journal_seconds', 'phase_seconds', 'input_positions', 'extra'):
            rows = [json.loads(line) for line in original.splitlines()]
            phase = next(r for r in rows if r['stage'] == 'feedback')
            if mutation == 'bytes':
                phase['journal_bytes'] += 1
            elif mutation == 'journal_seconds':
                phase['journal_seconds'] = phase['seconds'] + 1
            elif mutation == 'phase_seconds':
                phase['seconds'] = 0.
            elif mutation == 'input_positions':
                next(r for r in rows if r['stage'] == 'table')['input_positions'] += 1
            else:
                rows.append(rows[-1])
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.audited()
            path.write_text(original)

    def test_changed_outcomes_incomplete_grid_and_short_total_are_rejected(self):
        path = self.output / 'evaluations.jsonl'
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[-1]['evaluation']['exact'] = not rows[-1]['evaluation']['exact']
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaises(AssertionError):
            self.audited()
        path.write_text(original)
        path = self.output / 'metadata.json'
        original = path.read_text()
        for mutation in ('seed', 'time'):
            metadata = json.loads(original)
            if mutation == 'seed':
                metadata['seeds'] = [14000, 14001]
            else:
                metadata['elapsed_seconds'] = 0.
            path.write_text(json.dumps(metadata))
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.audited()
            path.write_text(original)

    def test_optimized_python_cannot_disable_checks(self):
        result = subprocess.run([sys.executable, '-O', '-c',
            "from pathlib import Path; from experiments.study_l.audit import audit_run; audit_run(Path('unused'))"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Audit requires assertions', result.stderr)

    def test_newline_conversion_cannot_hide_changed_journal_bytes(self):
        paths = [self.output / 'searches.jsonl', self.output / 'evaluations.jsonl',
                 self.output / 'seed-41/updates.jsonl',
                 next((self.output / 'seed-41/rejection').glob('*/candidates.jsonl'))]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original.replace(b'\n', b'\r\n'))
            with self.subTest(file=path.name), self.assertRaises(AssertionError):
                self.audited()
            path.write_bytes(original)

    def test_matching_certificate_allows_complete_synthetic_formal_grid(self):
        # Two environments and a one-operation budget keep this gate test small.
        # Both generator and fixed grid are explicitly patched to these fixtures.
        data = self.output.parent / 'small-grid.json'
        fixtures = [synthetic_environment(seed) for seed in (41, 14000, 14001)]
        data.write_text(json.dumps(fixtures))
        lookup = {e['seed']: e for e in fixtures}
        development, formal = self.output.parent / 'development', self.output.parent / 'formal'
        certificate_path = self.output.parent / 'certificate.json'
        with ExitStack() as stack:
            for module in (run, audit):
                stack.enter_context(patch.multiple(module, DATA_PATH=data, BUDGET=1, INTERVAL=1,
                                                   STEPS=1, FORMAL_SEEDS=(14000, 14001)))
                stack.enter_context(patch.object(module, 'source_hashes', return_value={}))
            stack.enter_context(patch.object(audit, 'make_environment', side_effect=lookup.__getitem__))
            stack.enter_context(patch.object(run.torch, '__version__', '2.8.0+cu126'))
            for output, extra in ((development, ['--development']),
                                  (formal, ['--development-audit', str(certificate_path)])):
                argv = ['study_l.run', '--source-revision', 'synthetic', '--output', str(output), *extra]
                with patch.object(run.sys, 'argv', argv):
                    run.main()
                checked = audit.audit_run(output)[-1]
                self.assertEqual(checked['status'], 'passed')
                if output == development:
                    certificate_path.write_text(json.dumps(checked))
            self.assertEqual((formal / 'development-audit.json').read_bytes(), certificate_path.read_bytes())
            forged = json.loads(certificate_path.read_bytes())
            forged.update(files={}, searches=0, gradient_steps=0, candidate_events=0, checkpoints=[])
            certificate_path.write_text(json.dumps(forged))
            (formal / 'development-audit.json').write_bytes(certificate_path.read_bytes())
            metadata_path = formal / 'metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['development_audit_sha256'] = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'complete search, file, and checkpoint evidence'):
                audit.audit_run(formal)
            argv = ['study_l.run', '--source-revision', 'synthetic', '--output', str(self.output.parent / 'forged'),
                    '--development-audit', str(certificate_path)]
            with patch.object(run.sys, 'argv', argv), \
                 self.assertRaisesRegex(ValueError, 'complete search, file, and checkpoint evidence'):
                run.main()


if __name__ == '__main__':
    unittest.main()
