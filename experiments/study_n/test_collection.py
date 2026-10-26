"""Synthetic full transcript, persistent weights, and isolated evaluations."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import torch

from experiments.study_n import run
from experiments.study_n.tasks import CHECKPOINTS, ENCODINGS, make_world


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.world = make_world(-171)
        run.collect([cls.world], cls.root)
        cls.folder = cls.root / 'seed--171'
        cls.records = {name: [json.loads(line) for line in (cls.folder / f'{name}.jsonl').read_text().splitlines()]
                       for name in run.JOURNALS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def checkpoint(self, encoding, size, at):
        return torch.load(self.folder / encoding / size / f'{at}.pt', weights_only=True)

    def test_prediction_before_query_and_exact_update_and_checkpoint_budgets(self):
        self.assertEqual(len(self.records['queries']), 64)
        self.assertEqual(len(self.records['predictions']), 256)
        self.assertEqual(len(self.records['updates']), 1280)
        self.assertEqual(len(list(self.folder.rglob('*.pt'))), 16)
        phases = self.records['phases']
        first_evaluation = next(i for i, p in enumerate(phases) if p['stage'] == 'evaluation_graph')
        self.assertTrue(all(p['stage'] not in ('query', 'training', 'prediction') for p in phases[first_evaluation:]))
        for at in range(1, 65):
            flow = [p['stage'] for p in phases if p.get('at') == at and
                    p['stage'] in ('prediction', 'query', 'training')]
            self.assertEqual(flow, ['prediction'] * 4 + ['query'] + ['training'] * 4)
        for encoding in ENCODINGS:
            for size, _, count in run.WIDTHS:
                for at in CHECKPOINTS:
                    saved = self.checkpoint(encoding, size, at)
                    self.assertEqual(saved['steps'], at * count)
                    self.assertEqual(len(saved['observations']), at)
                    for state in saved['optimizer']['state'].values():
                        self.assertEqual(state['step'].item(), at * count)
                initial, final = self.checkpoint(encoding, size, 0), self.checkpoint(encoding, size, 64)
                self.assertTrue(any(not torch.equal(initial['model'][k], v) for k, v in final['model'].items()))
                other = self.checkpoint('canonical', size, 0)
                for name, tensor in initial['model'].items():
                    self.assertTrue(torch.equal(tensor, other['model'][name]))

    def test_all_test_pairs_stay_outside_updates_and_sample_indices_pair(self):
        test = set(map(tuple, self.world['test']))
        for encoding in ENCODINGS:
            labels = list(range(32)) if encoding == 'canonical' else self.world['labels']
            inverse = {v: i for i, v in enumerate(labels)}
            for size, _, _ in run.WIDTHS:
                saved = self.checkpoint(encoding, size, 64)
                physical = {(inverse[s], a) for s, a, _ in saved['observations']}
                self.assertFalse(test & physical)
                self.assertEqual(physical, set(map(tuple, self.world['queries'])))
                signatures = [[r['indices_sha256'] for r in self.records['updates']
                               if r['encoding'] == e and r['arm'] == size + '_online'] for e in ENCODINGS]
                self.assertEqual(*signatures)

    def test_frozen_models_static_and_table_and_true_graphs_are_equivariant(self):
        self.assertEqual(len(self.records['direct']), 20)
        self.assertEqual(len(self.records['probes']), 27776)
        for encoding in ENCODINGS:
            for size, _, _ in run.WIDTHS:
                rows = [r for r in self.records['direct'] if r['encoding'] == encoding and
                        r['arm'] == size + '_frozen']
                self.assertEqual([r['at'] for r in rows], [0])
                initial = next(r for r in self.records['direct'] if r['encoding'] == encoding and
                               r['arm'] == size + '_online' and r['at'] == 0)
                self.assertEqual(rows[0]['predictions'], initial['predictions'])
        for at in CHECKPOINTS:
            tables = [[(r['start'], r['goal'], r['success'], [x['action'] for x in r['trace']])
                       for r in self.records['probes'] if r['encoding'] == e and
                       r['arm'] == 'observed_table' and r['at'] == at] for e in ENCODINGS]
            self.assertEqual(*tables)
        distances = json.loads((self.folder / 'distances.json').read_text())
        self.assertEqual([r['distances'] for r in distances[:32]], [r['distances'] for r in distances[32:]])

    def test_evaluation_cannot_call_updates_or_mutate_model_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'world'
            run.collect_learning(self.world, folder)
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*.pt')}
            with patch.object(run, 'update_steps', side_effect=AssertionError('evaluation trained')):
                run.evaluate(self.world, folder)
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*.pt')})

    def test_empty_development_manifest_cannot_start_formal_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, certificate = [root / name for name in ('data.json', 'manifest.json', 'audit.json')]
            # Seed labels only; no official environment or outcome is generated.
            run.document(data, [dict(seed=s) for s in (run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS)])
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            run.document(manifest, dict(source_sha256={}, data_sha256=digest, source_revision='synthetic'))
            evidence = dict(status='passed', seeds=[run.DEVELOPMENT_SEED],
                source_sha256={}, data_sha256=digest, runtime='2.8.0+cu126',
                counts=dict(queries=64, gradients=1280, checkpoints=16, direct=20, probes=27776),
                files={}, raw_directory=str(root / 'absent'))
            with patch.object(run, 'DATA', data), patch.object(run, 'MANIFEST', manifest), \
                    patch.object(run, 'source_hashes', return_value={}), \
                    patch.object(torch, '__version__', '2.8.0+cu126'), \
                    patch.object(run, 'collect') as collect, \
                    patch.object(sys, 'argv', ['run', '--output', str(root / 'formal'),
                                             '--development-audit', str(certificate)]):
                for files in ({}, {'metadata.json': 'missing'}):
                    evidence['files'] = files
                    run.document(certificate, evidence)
                    with self.assertRaisesRegex(ValueError, 'certificate'):
                        run.main()
                collect.assert_not_called()
                self.assertFalse((root / 'formal').exists())

    def test_journal_time_is_separable_from_training_time(self):
        clock = [0.]
        original_emit = run.emit
        def delayed_emit(stream, value):
            if str(stream.name).endswith('/updates.jsonl'):
                clock[0] += 1.
                original_emit(stream, value)
                return 1.
            return original_emit(stream, value)
        with tempfile.TemporaryDirectory() as temporary:
            world = dict(self.world, queries=self.world['queries'][:1])
            folder = Path(temporary) / 'world'
            with patch.object(run.time, 'perf_counter', side_effect=lambda: clock[0]), \
                    patch.object(run, 'emit', side_effect=delayed_emit):
                run.collect_learning(world, folder)
            phases = [json.loads(line) for line in (folder / 'phases.jsonl').read_text().splitlines()]
            training = [p for p in phases if p['stage'] == 'training']
            self.assertEqual(len(training), 4)
            for phase in training:
                self.assertEqual(phase['seconds'], phase['journal_seconds'])
                self.assertEqual(phase['journal_seconds'], 8 if phase['arm'] == 'small_online' else 2)


if __name__ == '__main__':
    unittest.main()
