"""Audit tampering and inference checks on explicitly hand-built environments."""

import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import shutil
from statistics import mean, stdev
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.study_k import analyze, audit, run


def synthetic_environment(seed):
    source = random.Random(441)
    primitives = []
    for _ in range(2):
        permutation = source.sample(range(16), 16)
        primitives.extend([permutation, [permutation.index(state) for state in range(16)]])
    seen, tasks = set(), {'experience': [], 'composition': [], 'shifted': []}

    def add(family, program):
        function = list(range(16))
        for operation in program:
            function = [primitives[operation][state] for state in function]
        if tuple(function) in seen:
            return False
        seen.add(tuple(function))
        examples = [[state, target] for state, target in enumerate(function)]
        tasks[family].append(dict(id=f'{family}-{len(tasks[family]):02d}', generating_program=list(program),
                                  train=examples[:8], test=examples[8:]))
        return True

    add('experience', (0,))
    add('experience', (2, 0, 2))
    for family in ('composition', 'shifted'):
        for length in (6, 9):
            count = 0
            for short_length in ((2, 4) if length == 6 else (1, 3)):
                for short_program in product(range(4), repeat=short_length):
                    program = short_program + (0, 1) * ((length - short_length) // 2)
                    count += add(family, program)
                    if count == 8:
                        break
                if count == 8:
                    break
    return dict(seed=seed, primitives=primitives, tasks=tasks, latent_motifs=[])


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.root = Path(cls.scratch.name)
        cls.environments = [synthetic_environment(seed) for seed in [37, *range(13000, 13032)]]
        cls.data = cls.root / 'synthetic-grid.json'
        cls.data.write_text(json.dumps(cls.environments))
        cls.original = cls.root / 'original'
        argv = ['study_k.run', '--development', '--output', str(cls.original), '--source-revision', 'synthetic']
        with patch.object(run, 'DATA_PATH', cls.data), patch.object(run, 'source_hashes', return_value={}), \
             patch.object(run, 'BUDGET', 512), patch.object(run.sys, 'argv', argv):
            run.main()

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def setUp(self):
        self.copy = Path(tempfile.mkdtemp(dir=self.root)) / 'archive'
        shutil.copytree(self.original, self.copy)
        self.addCleanup(shutil.rmtree, self.copy.parent)

    def audited(self):
        lookup = {environment['seed']: environment for environment in self.environments}
        with patch.object(audit, 'DATA_PATH', self.data), patch.object(audit, 'source_hashes', return_value={}), \
             patch.object(audit, 'BUDGET', 512), patch.object(audit, 'make_environment', side_effect=lookup.__getitem__):
            return audit.audit_run(self.copy)

    def test_complete_replay_and_final_only_resource_accounting(self):
        searches, phases, states, metadata, verified = self.audited()
        self.assertEqual(verified['searches'], 258)
        self.assertEqual(verified['gradient_steps'], 1056)
        self.assertEqual(len(states), 10)
        base = torch.load(self.copy / 'seed-37/base.pt', weights_only=True)
        self.assertGreater(max(map(len, base['programs'])), 2)
        self.assertGreater(len({len(program) for program in base['programs']}), 1)
        final_online = [state for state in states if state['arm'] == 'online' and state['attempt'] == 16]
        self.assertTrue(all(state['programs'] > len(base['programs']) for state in final_online))
        self.assertTrue(all(state['programs'] == len(base['programs']) for state in states if state['arm'] == 'replay'))
        summary = analyze.summarize(searches, phases, states, metadata)
        self.assertFalse(summary['primary']['confirmatory'])
        self.assertIsNone(summary['primary']['ci95'])
        self.assertEqual(summary['actual_cost']['gradient_steps'], 1056)
        self.assertEqual(summary['actual_cost']['table_model_calls'], 65 * 21)
        self.assertEqual(summary['actual_cost']['table_recurrent_input_positions'], 65 * 57)
        for row in summary['standalone_seed_costs']:
            arm = row['arm']
            self.assertEqual(row['stream_searches'], 16 if arm == 'online' else 0)
            self.assertEqual(row['final_probe_searches'], 16)
            self.assertEqual(row['experience_searches'], 0 if arm == 'uniform' else 2)
            self.assertEqual(row['gradient_steps'], {'uniform': 0, 'frozen': 32, 'replay': 288, 'online': 288}[arm])
            self.assertEqual(len(row['required_priors']), 17 if arm == 'online' else 1)
            self.assertEqual(row['table_model_calls'], {'uniform': 0, 'frozen': 21, 'replay': 21, 'online': 357}[arm])
        final = [row for row in summary['summaries'] if row['order'] == 'both' and row['checkpoint'] == 16
                 and row['family'] == 'balanced' and row['budget'] == 65536]
        self.assertTrue(all(row['condition_tasks'] == 32 and row['unique_tasks'] == 16 for row in final))

    def test_changed_adam_state_and_corpus_are_rejected(self):
        path = self.copy / 'seed-37/online-composition_first-8.pt'
        original = path.read_bytes()
        for mutation in ('optimizer', 'corpus'):
            with self.subTest(mutation=mutation):
                saved = torch.load(path, weights_only=True)
                if mutation == 'optimizer':
                    next(iter(saved['optimizer']['state'].values()))['step'] += 1
                else:
                    saved['accepted_task_ids'] = saved['accepted_task_ids'][:-1]
                torch.save(saved, path)
                with self.assertRaises(AssertionError):
                    self.audited()
                path.write_bytes(original)

    def test_changed_update_and_probability_table_are_rejected(self):
        path = self.copy / 'updates.jsonl'
        original = path.read_text()
        records = [json.loads(line) for line in original.splitlines()]
        records[0]['loss_before_step'] += .01
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        with self.assertRaises(AssertionError):
            self.audited()
        path.write_text(original)
        table = self.copy / 'seed-37/table-base.json'
        recorded = json.loads(table.read_text())
        recorded[0]['probabilities'][0] += .01
        table.write_text(json.dumps(recorded))
        with self.assertRaises(AssertionError):
            self.audited()

    def test_extra_or_changed_cost_phases_are_rejected(self):
        path = self.copy / 'phases.jsonl'
        original = path.read_text()
        for mutation in ('extra_table', 'physical_inputs', 'ranker_queries', 'nonfinite_time', 'short_replay'):
            with self.subTest(mutation=mutation):
                records = [json.loads(line) for line in original.splitlines()]
                if mutation == 'extra_table':
                    row = dict(next(r for r in records if r['stage'] == 'probability_table'), prior_id='unrecorded')
                    records.append(row)
                elif mutation == 'physical_inputs':
                    next(r for r in records if r['stage'] == 'replay')['input_positions'] += 1
                elif mutation == 'ranker_queries':
                    next(r for r in records if r['stage'] == 'ranking')['predictions'] += 1
                elif mutation == 'nonfinite_time':
                    records[0]['seconds'] = float('inf')
                else:
                    next(r for r in records if r['stage'] == 'replay')['seconds'] = 0.
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                with self.assertRaises(AssertionError):
                    self.audited()
                path.write_text(original)

    def test_held_out_score_and_incomplete_grid_are_rejected(self):
        path = self.copy / 'evaluations.jsonl'
        original = path.read_text()
        records = [json.loads(line) for line in original.splitlines()]
        record = next(row for row in records if row['phase'] == 'probe')
        record['evaluation']['exact'] = not record['evaluation']['exact']
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        with self.assertRaises(AssertionError):
            self.audited()
        path.write_text(original)
        metadata = self.copy / 'metadata.json'
        value = json.loads(metadata.read_text())
        value['seeds'] = [13000, 13001]
        metadata.write_text(json.dumps(value))
        with self.assertRaisesRegex(AssertionError, 'Incomplete seed grid'):
            self.audited()

    def test_impossible_collection_time_is_rejected(self):
        path = self.copy / 'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['elapsed_seconds'] = 0.
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(AssertionError, 'Collection shorter'):
            self.audited()

    def test_optimized_python_cannot_disable_audit(self):
        result = subprocess.run([sys.executable, '-O', '-c',
            "from pathlib import Path; from experiments.study_k.audit import audit_run; audit_run(Path('unused'))"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Audit requires assertions', result.stderr)

    def test_formal_certificate_is_required_and_must_match(self):
        # A complete zero-budget synthetic grid keeps this handoff test small.
        # Its task generator is patched to the explicit handcrafted fixtures.
        development = self.copy.parent / 'development-zero'
        formal = self.copy.parent / 'formal-zero'
        certificate_source = self.copy.parent / 'development-certificate.json'
        lookup = {environment['seed']: environment for environment in self.environments}
        with patch.object(run, 'DATA_PATH', self.data), patch.object(run, 'source_hashes', return_value={}), \
             patch.object(run, 'BUDGET', 0), patch.object(audit, 'DATA_PATH', self.data), \
             patch.object(audit, 'source_hashes', return_value={}), patch.object(audit, 'BUDGET', 0), \
             patch.object(audit, 'make_environment', side_effect=lookup.__getitem__):
            argv = ['study_k.run', '--development', '--output', str(development), '--source-revision', 'synthetic']
            with patch.object(run.sys, 'argv', argv):
                run.main()
            certificate = audit.audit_run(development)[-1]
            certificate_source.write_text(json.dumps(certificate))
            argv = ['study_k.run', '--output', str(formal), '--source-revision', 'synthetic',
                    '--development-audit', str(certificate_source)]
            with patch.object(run.sys, 'argv', argv):
                run.main()
            self.assertEqual(audit.audit_run(formal)[-1]['seeds'], list(range(13000, 13032)))
            metadata_path = formal / 'metadata.json'
            original_metadata = json.loads(metadata_path.read_text())
            certificate_path = formal / 'development-audit.json'
            original_bytes = certificate_path.read_bytes()
            certificate_path.unlink()
            with self.assertRaises(FileNotFoundError):
                audit.audit_run(formal)
            certificate_path.write_bytes(original_bytes)
            metadata = dict(original_metadata, development_audit_sha256=None)
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(AssertionError):
                audit.audit_run(formal)
            for field, value in (('status', 'failed'), ('seeds', [999]), ('source_sha256', {'bad': 'hash'}),
                                 ('data_sha256', 'wrong'), ('runtime', {})):
                with self.subTest(field=field):
                    changed = dict(certificate, **{field: value})
                    certificate_path.write_text(json.dumps(changed))
                    metadata = dict(original_metadata,
                        development_audit_sha256=hashlib.sha256(certificate_path.read_bytes()).hexdigest())
                    metadata_path.write_text(json.dumps(metadata))
                    with self.assertRaises(AssertionError):
                        audit.audit_run(formal)

    def test_primary_uses_environment_means_and_frozen_bootstrap(self):
        searches, phases, states, metadata, _ = self.audited()
        all_searches, all_phases, all_states = [], [], []
        for offset, seed in enumerate(range(13000, 13032)):
            seen = {}
            for original in searches:
                row = copy.deepcopy(original)
                row['seed'] = seed
                if row['phase'] == 'probe':
                    key = (row['arm'], row['order'], row['checkpoint'], row['family'])
                    position = seen.get(key, 0)
                    seen[key] = position + 1
                    count = offset % 5 if row['arm'] == 'online' and row['checkpoint'] == 16 else 0
                    row['result'].update(status='accepted', primitive_ops=4096)
                    row['evaluation'].update(exact=position < count, test_correct=8 if position < count else 0)
                all_searches.append(row)
            all_phases.extend(dict(row, seed=seed) for row in phases)
            all_states.extend(dict(row, seed=seed) for row in states)
        metadata = dict(metadata, seeds=list(range(13000, 13032)))
        summary = analyze.summarize(all_searches, all_phases, all_states, metadata)
        primary = summary['primary']
        differences = [(i % 5) / 8 for i in range(32)]
        self.assertEqual(primary['differences'], differences)
        self.assertEqual(primary['n'], 32)
        expected_half = 2.0395134463964077 * stdev(differences) / 32**.5
        np.testing.assert_allclose(primary['ci95'], [mean(differences) - expected_half, mean(differences) + expected_half])
        bootstrap = np.random.default_rng(967).choice(differences, (10000, 32)).mean(axis=1)
        self.assertEqual(primary['bootstrap95'], np.quantile(bootstrap, [.025, .975]).tolist())
        self.assertTrue(primary['positive_primary'])
        self.assertFalse(primary['degenerate'])
        with patch.object(analyze.t, 'ppf', return_value=100):
            disagreement = analyze.summarize(all_searches, all_phases, all_states, metadata)['primary']
        self.assertTrue(disagreement['interval_disagreement'])
        self.assertFalse(disagreement['positive_primary'])
        # Opposite order effects cancel inside every environment. They must
        # not be treated as 64 independent observations with nonzero variance.
        positions = {}
        for row in all_searches:
            if row['phase'] == 'probe' and row['arm'] == 'online' and row['checkpoint'] == 16:
                key = (row['seed'], row['order'], row['family'])
                position = positions.get(key, 0)
                positions[key] = position + 1
                count = (row['seed'] - 13000) % 9
                if row['order'] == 'shifted_first':
                    count = 8 - count
                row['evaluation']['exact'] = position < count
        degenerate = analyze.summarize(all_searches, all_phases, all_states, metadata)['primary']
        self.assertEqual(degenerate['differences'], [.5] * 32)
        self.assertEqual(degenerate['ci95'], [.5, .5])
        self.assertEqual(degenerate['bootstrap95'], [.5, .5])
        self.assertTrue(degenerate['degenerate'])


if __name__ == '__main__':
    unittest.main()
