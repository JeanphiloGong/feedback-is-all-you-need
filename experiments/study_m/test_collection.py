"""Synthetic checks of retained weights, update cadence, and probe isolation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_m import run


def fixture(family='structured'):
    return dict(seed=-811, family=family,
        primitives=[[(s + a + 1) % 32 for s in range(32)] for a in range(4)],
        warmup=dict(start=0, actions=[0] * 16),
        episodes=[dict(start=0, goal=1)] * 513,
        probes=[dict(start=0, goal=8), dict(start=0, goal=31)])


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CollectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch.object(run, 'INTERACTIONS', 32), patch.object(run, 'CHECKPOINTS', (0, 16, 32)):
            run.collect([fixture()], self.root)
        self.folder = self.root / 'seed--811' / 'structured'
        self.phases = records(self.root / 'phases.jsonl')
        self.actions = records(self.folder / 'actions.jsonl')

    def checkpoint(self, arm, at):
        return torch.load(self.folder / arm / f'{at}.pt', weights_only=True)

    def test_complete_actions_cadence_and_optimizer_persistence(self):
        self.assertEqual(len(self.actions), 8 * 32)
        for arm in run.ARMS:
            actions = [r for r in self.actions if r['arm'] == arm]
            self.assertEqual([r['at'] for r in actions], list(range(1, 33)))
            batches = [r for r in self.phases if r['stage'] == 'training' and r['arm'] == arm]
            if arm.endswith('immediate'):
                self.assertEqual([r['at'] for r in batches], list(range(1, 33)))
            elif arm.endswith('delayed'):
                self.assertEqual([r['at'] for r in batches], [16, 32])
            else:
                self.assertEqual(batches, [])
            checkpoint = self.checkpoint(arm, 32)
            if arm.startswith(('small', 'large')):
                expected = (128 if arm.startswith('small') else 32)
                if not arm.endswith('frozen'):
                    expected += 32 * (8 if arm.startswith('small') else 2)
                self.assertEqual(checkpoint['gradient_steps'], expected)
                for state in checkpoint['optimizer']['state'].values():
                    self.assertEqual(state['step'].item(), expected)
            self.assertEqual(checkpoint['episode_index'], sum(r['episode_ended'] for r in actions))
            self.assertEqual(len(checkpoint['corpus']), len({tuple(r[:2]) for r in checkpoint['corpus']}))
        small = sum(r['dense_training_macs'] for r in self.phases
                    if r['stage'] == 'training' and r['arm'] == 'small_immediate')
        large = sum(r['dense_training_macs'] for r in self.phases
                    if r['stage'] == 'training' and r['arm'] == 'large_immediate')
        self.assertEqual(small, large)

    def test_initial_branches_match_frozen_weights_and_real_updates(self):
        for size in ('small', 'large'):
            initial = self.checkpoint(f'{size}_frozen', 0)
            for cadence in ('immediate', 'delayed', 'frozen'):
                arm = f'{size}_{cadence}'
                current = self.checkpoint(arm, 0)
                for name, tensor in initial['model'].items():
                    self.assertTrue(torch.equal(tensor, current['model'][name]))
                final = self.checkpoint(arm, 32)
                changed = any(not torch.equal(tensor, final['model'][name])
                              for name, tensor in initial['model'].items())
                self.assertEqual(changed, cadence != 'frozen')
        batches = [r for r in self.phases if r['stage'] == 'training']
        self.assertTrue(all(r['changed_parameter_values'] > 0 and r['parameter_change_l2'] > 0
                            for r in batches))

    def test_decisions_use_only_the_previous_scheduled_graph(self):
        for row in self.actions:
            arm, at = row['arm'], row['at']
            expected = ((at - 1) // 16 * 16 if arm.endswith('delayed') else
                        0 if arm.endswith('frozen') or arm == 'uniform' else at - 1)
            self.assertEqual(row['graph_at'], expected)
            if row['predicted_next_state'] is not None:
                expected_error = (row['predicted_next_state'] ^ row['next_state']).bit_count()
                self.assertEqual(row['bit_errors'], expected_error)
        for phase in self.phases:
            self.assertGreaterEqual(phase['seconds'], 0)
            self.assertLessEqual(phase.get('journal_seconds', 0), phase['seconds'])

    def test_static_probes_are_executed_once_and_checkpoint_graphs_stay_fixed(self):
        probes = records(self.folder / 'probes.jsonl')
        for arm in run.ARMS:
            values = [r for r in probes if r['arm'] == arm]
            expected = [0] if arm.endswith('frozen') or arm == 'uniform' else [0, 16, 32]
            self.assertEqual([r['at'] for r in values], [at for at in expected for _ in range(2)])
            for value in values:
                graph = self.checkpoint(arm, value['at'])['graph']
                for step in value['trace']:
                    predicted = graph[step['action']][step['state']] if graph is not None else None
                    self.assertEqual(predicted, step['predicted_next_state'])


class ProbeTests(unittest.TestCase):
    def test_oracle_retains_exact_distance_beyond_the_probe_budget(self):
        graph = [[(s + 1) % 32 for s in range(32)] for _ in range(4)]
        for action, pair in enumerate(((30, 31), (28, 29), (26, 27)), 1):
            left, right = pair
            graph[action][left], graph[action][right] = graph[action][right], graph[action][left]
        value = run.oracle_path(graph, 0, 9)
        self.assertEqual(value['distance'], 9)
        self.assertEqual(value['shortest_path'], [0] * 9)
        self.assertFalse(value['reachable_within_budget'])

    def test_success_on_eighth_action_and_failure_after_eight(self):
        graph = [[(s + 1) % 32 for s in range(32)]] * 4
        for goal, expected in ((8, True), (9, False)):
            value = run.probe(graph, graph, 0, goal, -313, 'fixture')
            self.assertEqual(value['success'], expected)
            self.assertEqual(len(value['trace']), 8)

    def test_no_probe_access_until_all_world_streams_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            class Probes(list):
                def __iter__(self):
                    for family in ('structured', 'random'):
                        rows = records(output / 'seed--811' / family / 'actions.jsonl')
                        self_test.assertEqual(len(rows), 8 * 16)
                    return super().__iter__()
            self_test = self
            worlds = [fixture(family) for family in ('structured', 'random')]
            for world in worlds:
                world['probes'] = Probes(world['probes'])
            with patch.object(run, 'INTERACTIONS', 16), patch.object(run, 'CHECKPOINTS', (0, 16)):
                run.collect(worlds, output)


class GateTests(unittest.TestCase):
    def test_development_attestation_requires_complete_files_and_counts(self):
        files = run.required_files([43])
        certificate = dict(files={path: '0' * 64 for path in files},
            checkpoints=88, gradient_steps=20800, main_actions=8192, probe_tasks=13888)
        run.check_development_evidence(certificate)
        for field in ('checkpoints', 'gradient_steps', 'main_actions', 'probe_tasks'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.check_development_evidence(dict(certificate, **{field: certificate[field] - 1}))
        for missing in ('seed-43/structured/small_immediate/64.pt',
                        'seed-43/random/probes.jsonl', 'seed-43/random/updates.jsonl'):
            changed = dict(certificate, files={k: v for k, v in certificate['files'].items() if k != missing})
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                run.check_development_evidence(changed)


if __name__ == '__main__':
    unittest.main()
