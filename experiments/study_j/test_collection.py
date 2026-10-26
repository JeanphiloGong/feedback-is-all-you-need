"""Hand-built boundary fixtures, not development or formal environments."""

import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_j import run


def fixture():
    primitives = [[(i + 1) % 16 for i in range(16)],
                  [(i + 2) % 16 for i in range(16)],
                  [15 - i for i in range(16)], [i ^ 1 for i in range(16)]]
    task = dict(train=[[i, (i + 1) % 16] for i in range(8)],
                test=[[i, (i + 1) % 16] for i in range(8, 16)], generating_program=[0])
    return dict(seed=7, primitives=primitives, latent_motifs=[], tasks={
        family: [dict(id=f'{family}-{i}', **copy.deepcopy(task)) for i in range(n)]
        for family, n in (('experience', 16), ('composition', 1), ('shifted', 1))})


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_search_and_updates_survive_evaluator_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(run, 'evaluate', side_effect=RuntimeError('evaluator failed')):
                with self.assertRaisesRegex(RuntimeError, 'evaluator failed'):
                    run.collect([fixture()], output)
            searches = records(output / 'searches.jsonl')
            self.assertEqual(len(searches), 1)
            self.assertEqual(searches[0]['result']['status'], 'accepted')
            self.assertEqual(len(records(output / 'updates.jsonl')), 32)
            self.assertEqual((output / 'evaluations.jsonl').read_text(), '')
            phases = records(output / 'phases.jsonl')
            self.assertEqual(sum(r['stage'] == 'initialization' for r in phases), 1)
            self.assertEqual(sum(r['stage'] == 'clone' for r in phases), 1)
            self.assertEqual(sum(r['stage'] == 'optimizer_setup' for r in phases), 2)
            self.assertEqual(sum(r['stage'] == 'replay' for r in phases), 2)
            active = json.loads((output / 'active-phase.json').read_text())
            self.assertEqual(active['stage'], 'evaluation')
            self.assertEqual(active['task_id'], 'experience-0')

    def test_empty_corpus_has_one_explicit_skip_per_attempt_and_arm(self):
        environment = fixture()
        for tasks in environment['tasks'].values():
            for task in tasks:
                task['train'] = [[state, state] for state in range(8)]
                task['test'] = [[state, state] for state in range(8, 16)]
                task['generating_program'] = []
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'BUDGET', 1):
            output = Path(directory)
            run.collect([environment], output)
            self.assertEqual((output / 'updates.jsonl').read_text(), '')
            skipped = [r for r in records(output / 'phases.jsonl') if r['stage'] == 'replay']
            self.assertEqual(len(skipped), 32)
            self.assertEqual({(r['arm'], r['attempt']) for r in skipped},
                             {(arm, n) for arm in run.LEARNED for n in range(1, 17)})
            self.assertTrue(all(r['skip_reason'] == 'empty_corpus' and r['steps'] == r['replay_tokens'] == r['input_rows'] == r['input_positions'] == 0
                                for r in skipped))

    def test_initialization_failure_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            with patch.object(run, 'DATA_PATH', Path(directory) / 'absent.json'), patch(
                    'sys.argv', ['run', '--development', '--output', str(output),
                                 '--source-revision', 'unit-fixture']):
                with self.assertRaises(FileNotFoundError):
                    run.main()
            self.assertEqual(json.loads((output / 'failure.json').read_text())['error_type'], 'FileNotFoundError')

    def test_truncated_phase_file_does_not_mask_original_failure(self):
        def interrupted(environments, output):
            (output / 'active-phase.json').write_text('{"stage":')
            raise OSError('phase write interrupted')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            data = Path(directory) / 'fixture.json'
            data.write_text(json.dumps([dict(fixture(), seed=seed)
                                        for seed in (run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS)]))
            with patch.object(run, 'DATA_PATH', data), patch.object(run, 'collect', side_effect=interrupted), patch(
                    'sys.argv', ['run', '--development', '--output', str(output), '--source-revision', 'unit-fixture']):
                with self.assertRaisesRegex(OSError, 'phase write interrupted'):
                    run.main()
            failure = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failure['error_type'], 'OSError')
            self.assertEqual(failure['active_phase_file'], 'active-phase.json')
            self.assertEqual((output / failure['active_phase_file']).read_text(), '{"stage":')

    def test_development_rejects_incomplete_or_incompatible_full_grid(self):
        for scenario in ('missing_formal', 'short_formal'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                data, output = Path(directory) / 'data.json', Path(directory) / 'run'
                grid = [dict(fixture(), seed=seed) for seed in (run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS)]
                if scenario == 'missing_formal':
                    grid.pop()
                else:
                    grid[-1]['tasks']['experience'].pop()
                data.write_text(json.dumps(grid))
                with patch.object(run, 'DATA_PATH', data), patch.object(run, 'collect') as collect, patch(
                        'sys.argv', ['run', '--development', '--output', str(output), '--source-revision', 'fixture']):
                    with self.assertRaises(ValueError):
                        run.main()
                    collect.assert_not_called()
                self.assertEqual(json.loads((output / 'failure.json').read_text())['error_type'], 'ValueError')

    def test_formal_rejects_changed_torch_or_development_runtime(self):
        import hashlib

        for scenario in ('wrong_torch', 'different_development'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                data, audit, output = (Path(directory) / name for name in ('data.json', 'audit.json', 'run'))
                data.write_text(json.dumps([dict(fixture(), seed=seed)
                                            for seed in (run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS)]))
                runtime = dict(torch='2.8.0+cu126', threads=1, device='cpu', dtype='float64', deterministic_algorithms=True)
                if scenario == 'different_development':
                    runtime['threads'] = 2
                audit.write_text(json.dumps(dict(status='passed', seeds=[run.DEVELOPMENT_SEED],
                    source_sha256=run.source_hashes(), data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    runtime=runtime)))
                version = '9.9.9-test' if scenario == 'wrong_torch' else torch.__version__
                with patch.object(run, 'DATA_PATH', data), patch.object(run, 'collect') as collect, patch.object(
                        torch, '__version__', version), patch('sys.argv', ['run', '--output', str(output),
                            '--source-revision', 'fixture', '--development-audit', str(audit)]):
                    with self.assertRaises(ValueError):
                        run.main()
                    collect.assert_not_called()

    def test_hidden_labels_do_not_change_training_or_search(self):
        original = fixture()
        changed = copy.deepcopy(original)
        changed['latent_motifs'] = [[3, 2, 1]]
        for tasks in changed['tasks'].values():
            for task in tasks:
                task['generating_program'] = [3]
                task['test'] = [[state, (label + 1) % 16] for state, label in task['test']]
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'BUDGET', 64):
            a, b = Path(directory) / 'original', Path(directory) / 'changed'
            for environment, output in ((original, a), (changed, b)):
                output.mkdir()
                run.collect([environment], output)
            self.assertEqual([r['result'] for r in records(a / 'searches.jsonl')],
                             [r['result'] for r in records(b / 'searches.jsonl')])
            left, right = records(a / 'updates.jsonl'), records(b / 'updates.jsonl')
            for row in left + right:
                del row['step_seconds']
            self.assertEqual(left, right)
            self.assertEqual((a / 'seed-7/corpus.json').read_bytes(), (b / 'seed-7/corpus.json').read_bytes())
            for path in (a / 'seed-7').glob('*.pt'):
                first = torch.load(path, weights_only=True)
                second = torch.load(b / 'seed-7' / path.name, weights_only=True)
                for name, tensor in first['model'].items():
                    torch.testing.assert_close(tensor, second['model'][name], rtol=0, atol=0)
            self.assertTrue(records(a / 'evaluations.jsonl')[0]['evaluation']['exact'])
            self.assertFalse(records(b / 'evaluations.jsonl')[0]['evaluation']['exact'])
            corpus = json.loads((b / 'seed-7/corpus.json').read_text())
            self.assertEqual(corpus['programs'], [[0]] * 16)
            costs = json.loads((a / 'seed-7/costs.json').read_text())
            self.assertEqual(costs['training_steps'], {arm: 256 for arm in run.LEARNED})
            self.assertEqual(costs['parameters']['full_trained'], 1172)
            self.assertEqual(costs['parameters']['local_trained'], 1172)

    def test_recorded_physical_compute_matches_observed_forward_calls(self):
        original_forward = run.ProgramPrior.forward
        calls, positions, input_rows = Counter(), Counter(), Counter()
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'BUDGET', 128):
            output = Path(directory)

            def observed(model, tokens):
                phase = json.loads((output / 'active-phase.json').read_text())
                key = phase['stage'], phase['arm']
                calls[key] += 1
                positions[key] += tokens.numel()
                input_rows[key] += len(tokens)
                return original_forward(model, tokens)

            with patch.object(run.ProgramPrior, 'forward', observed):
                run.collect([fixture()], output)
            costs = json.loads((output / 'seed-7/costs.json').read_text())
            for arm in run.LEARNED:
                self.assertEqual(costs['training_steps'][arm], calls['replay', arm])
                self.assertEqual(costs['input_rows'][arm], input_rows['replay', arm])
                self.assertEqual(costs['input_positions'][arm], positions['replay', arm])
            for arm in run.ARMS + run.PROBES:
                ranking = costs['ranking'][arm]
                self.assertEqual(ranking['recurrent_input_positions'], positions['ranking', arm])
                if arm not in ('uniform', 'counts'):
                    self.assertEqual(ranking['predictions'], calls['ranking', arm])
            initial = torch.load(output / 'seed-7/initial.pt', weights_only=True)
            self.assertEqual(initial['attempts'], 0)
            self.assertIsNone(initial['optimizer'])
            self.assertEqual(set(costs['checkpoint_bytes']), {'initial.pt'} | {
                f'{arm}-{attempt}.pt' for arm in run.LEARNED for attempt in (8, 16)})


if __name__ == '__main__':
    unittest.main()
