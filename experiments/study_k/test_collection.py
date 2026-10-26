"""Hand-built collection fixtures; official K tasks are never generated."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_h.search import execute
from experiments.study_i.model import ProgramPrior
from experiments.study_j.audit import compare_tensors
from experiments.study_k import run


def fixture(seed=67, empty_experience=False):
    primitives = [[(state + step) % 16 for state in range(16)] for step in (1, 3, 5, 7)]
    tasks = {}
    for family, count in (('experience', 2), ('composition', 16), ('shifted', 16)):
        tasks[family] = []
        for index in range(count):
            program = () if empty_experience and family == 'experience' else (0,)
            examples = [(state, execute(primitives, program, state)) for state in range(16)]
            tasks[family].append(dict(id=f'{family}-{index:02d}', generating_program=program,
                                      train=examples[:8], test=examples[8:]))
    return dict(seed=seed, primitives=primitives, tasks=tasks, latent_motifs=[])


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CollectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.addCleanup(self.scratch.cleanup)

    def collect(self, name, environment):
        output = self.root / name
        output.mkdir()
        with patch.object(run, 'BUDGET', 64):
            run.collect([environment], output)
        return output

    def test_continual_membership_cloning_and_physical_costs(self):
        calls, retained_models = [], []
        forward = ProgramPrior.forward
        original_replay = run.record_replay

        def observed_forward(model, tokens):
            calls.append((torch.is_grad_enabled(), tokens.shape[0], tokens.numel()))
            return forward(model, tokens)

        def observed_replay(model, optimizer, programs, key, *rest):
            if not any(model is item[0] for item in retained_models):
                retained_models.append((model, optimizer))
            return original_replay(model, optimizer, programs, key, *rest)

        with patch.object(ProgramPrior, 'forward', observed_forward), patch.object(run, 'record_replay', observed_replay):
            output = self.collect('complete', fixture())
        folder = output / 'seed-67'
        checkpoints = json.loads((folder / 'checkpoints.json').read_text())
        initial = torch.load(folder / 'base.pt', weights_only=True)
        self.assertEqual(initial['counters']['steps'], 32)
        self.assertEqual(len(retained_models), 5)
        pointer_sets = [{p.data_ptr() for p in model.parameters()} for model, _ in retained_models]
        for i, pointers in enumerate(pointer_sets):
            self.assertTrue(all(not pointers & other for other in pointer_sets[i + 1:]))
        compare_tensors(retained_models[0][0].state_dict(), initial['model'])
        compare_tensors(retained_models[0][1].state_dict(), initial['optimizer'])
        for model, optimizer in retained_models:
            self.assertEqual({id(p) for p in model.parameters()},
                             {id(p) for group in optimizer.param_groups for p in group['params']})
        for checkpoint in checkpoints:
            saved = torch.load(folder / (checkpoint['name'] + '.pt'), weights_only=True)
            attempt = checkpoint['checkpoint']
            expected_ids = ['experience-00', 'experience-01']
            if checkpoint['arm'] == 'online':
                roles = json.loads((folder / 'roles.json').read_text())
                expected_ids += roles['streams'][checkpoint['order']][:attempt]
            self.assertEqual(saved['accepted_task_ids'], expected_ids)
            self.assertEqual(saved['counters']['steps'], 32 + 16 * attempt)
            for state in saved['optimizer']['state'].values():
                self.assertEqual(state['step'].item(), 32 + 16 * attempt)
            self.assertTrue((folder / f"order-{saved['prior_id']}.txt").is_file())
        for attempt in (8, 16):
            left = torch.load(folder / f'replay-composition_first-{attempt}.pt', weights_only=True)
            right = torch.load(folder / f'replay-shifted_first-{attempt}.pt', weights_only=True)
            compare_tensors(left['model'], right['model'])
            compare_tensors(left['optimizer'], right['optimizer'])
        phases = rows(output / 'phases.jsonl')
        updates = rows(output / 'updates.jsonl')
        self.assertEqual(sum(r['input_rows'] for r in updates), sum(r for grad, r, _ in calls if grad))
        self.assertEqual(sum(r['input_positions'] for r in updates), sum(n for grad, _, n in calls if grad))
        tables = [r for r in phases if r['stage'] == 'probability_table']
        self.assertEqual(len(tables), 65)
        self.assertEqual(sum(r['model_calls'] for r in tables), sum(not grad for grad, _, _ in calls))
        self.assertEqual(sum(r['recurrent_input_positions'] for r in tables),
                         sum(n for grad, _, n in calls if not grad))
        ranking = [r for r in phases if r['stage'] == 'ranking']
        self.assertEqual(len(ranking), 66)
        self.assertEqual(len({r['prior_id'] for r in ranking}), 66)
        searches = rows(output / 'searches.jsonl')
        self.assertEqual(len(searches), 258)
        self.assertEqual(sum(r['phase'] == 'probe' for r in searches), 160)
        first_probe = next(i for i, r in enumerate(searches) if r['phase'] == 'probe')
        self.assertTrue(all(r['phase'] != 'stream' for r in searches[first_probe:]))
        self.assertEqual(sum(r['phase'] == 'stream' and r['order'] == 'shared' for r in searches), 32)
        for arm in run.LEARNED:
            for order in run.ORDER_FAMILIES:
                first = next(r for r in searches if r['arm'] == arm and r['order'] == order and r['phase'] == 'stream')
                self.assertEqual(first['prior_id'], 'base')

    def test_hidden_labels_and_probe_inputs_cannot_change_learning(self):
        original = fixture()
        modified = copy.deepcopy(original)
        for family, tasks in modified['tasks'].items():
            for index, task in enumerate(tasks):
                task['test'] = [(state, (target + 1) % 16) for state, target in task['test']]
                task['generating_program'] = (3, 2, 1, 0)
                if family != 'experience' and index in run.PROBE_INDICES:
                    task['train'] = [(state, (target + 2) % 16) for state, target in task['train']]
        left, right = self.collect('original', original), self.collect('hidden_changed', modified)
        for path in (left / 'seed-67').glob('*.pt'):
            a = torch.load(path, weights_only=True)
            b = torch.load(right / 'seed-67' / path.name, weights_only=True)
            compare_tensors(a['model'], b['model'])
            compare_tensors(a['optimizer'], b['optimizer'])
            self.assertEqual(a['programs'], b['programs'])
            self.assertEqual(a['accepted_task_ids'], b['accepted_task_ids'])
        for path in (left / 'seed-67').glob('order-*.txt'):
            self.assertEqual(path.read_bytes(), (right / 'seed-67' / path.name).read_bytes())
        self.assertTrue(all(not r['evaluation']['exact'] for r in rows(right / 'evaluations.jsonl')))
        online = torch.load(right / 'seed-67/online-composition_first-16.pt', weights_only=True)
        self.assertEqual(len(online['programs']), 18)

    def test_empty_initial_corpus_and_empty_stream_acceptances(self):
        environment = fixture(empty_experience=True)
        for family in ('composition', 'shifted'):
            for task in environment['tasks'][family]:
                task['train'] = [(state, state) for state in range(8)]
                task['test'] = [(state, state) for state in range(8, 16)]
        output = self.collect('empty', environment)
        self.assertEqual(rows(output / 'updates.jsonl'), [])
        phases = rows(output / 'phases.jsonl')
        self.assertEqual(sum(r['stage'] == 'probability_table' for r in phases), 1)
        self.assertEqual(sum(r['stage'] == 'ranking' for r in phases), 2)
        replays = [r for r in phases if r['stage'] == 'replay']
        self.assertEqual(len(replays), 66)
        self.assertTrue(all(r['skip_reason'] == 'empty_corpus' and r['steps'] == 0 for r in replays))
        for checkpoint in json.loads((output / 'seed-67/checkpoints.json').read_text()):
            self.assertEqual(checkpoint['prior_id'], 'base')

    def test_failed_searches_still_replay_without_new_targets(self):
        environment = fixture()
        for family in ('composition', 'shifted'):
            for task in environment['tasks'][family]:
                task['train'] = [(state, 0) for state in range(8)]
        output = self.collect('failed_stream', environment)
        for row in rows(output / 'searches.jsonl'):
            if row['phase'] == 'stream':
                self.assertEqual(row['result']['status'], 'exhausted')
        for arm in run.LEARNED:
            for order in run.ORDER_FAMILIES:
                saved = torch.load(output / f'seed-67/{arm}-{order}-16.pt', weights_only=True)
                self.assertEqual(saved['accepted_task_ids'], ['experience-00', 'experience-01'])
                self.assertEqual(saved['counters']['steps'], 288)
        online = torch.load(output / 'seed-67/online-composition_first-16.pt', weights_only=True)
        replay = torch.load(output / 'seed-67/replay-composition_first-16.pt', weights_only=True)
        compare_tensors(online['model'], replay['model'])
        compare_tensors(online['optimizer'], replay['optimizer'])

    def invoke_main(self, name, environments, *, development=True, audit=None, collect=None, version='2.8.0+cu126'):
        data = self.root / (name + '-data.json')
        data.write_text(json.dumps(environments))
        output = self.root / name
        argv = ['study_k.run', '--output', str(output), '--source-revision', 'synthetic']
        if development:
            argv.append('--development')
        if audit is not None:
            path = self.root / (name + '-audit.json')
            path.write_text(json.dumps(audit))
            argv += ['--development-audit', str(path)]
        with patch.object(run, 'DATA_PATH', data), patch.object(run, 'source_hashes', return_value={}), \
             patch.object(run.sys, 'argv', argv), patch.object(run.torch, '__version__', version), \
             patch.object(run, 'collect', side_effect=collect) as collector:
            run.main()
        return output, collector

    def test_full_grid_and_runtime_are_required_before_collection(self):
        complete = [fixture(seed) for seed in [run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS]]
        incomplete_family = copy.deepcopy(complete)
        incomplete_family[-1]['tasks']['shifted'].pop()
        cases = [('missing_seed', complete[:-1], '2.8.0+cu126'),
                 ('missing_probe', incomplete_family, '2.8.0+cu126'),
                 ('wrong_runtime', complete, '2.5.1')]
        for name, data, version in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.invoke_main(name, data, version=version,
                                 collect=lambda *args: self.fail('Collection started before gate checks.'))
            self.assertTrue((self.root / name / 'failure.json').is_file())
        output, collector = self.invoke_main('valid_development', complete)
        self.assertEqual([e['seed'] for e in collector.call_args.args[0]], [37])
        self.assertEqual(json.loads((output / 'metadata.json').read_text())['status'], 'completed')

    def test_formal_gate_matches_runtime_and_preserves_original_failure(self):
        complete = [fixture(seed) for seed in [run.DEVELOPMENT_SEED, *run.FORMAL_SEEDS]]
        audit = dict(status='passed', seeds=[37], source_sha256={},
            data_sha256=hashlib.sha256(json.dumps(complete).encode()).hexdigest(),
            runtime=dict(torch='2.8.0+cu126', threads=1, device='cpu', dtype='float64', deterministic_algorithms=True))
        changed = copy.deepcopy(audit)
        changed['runtime']['threads'] = 2
        with self.assertRaises(ValueError):
            self.invoke_main('bad_handoff', complete, development=False, audit=changed,
                             collect=lambda *args: self.fail('Invalid formal handoff reached collection.'))
        output, collector = self.invoke_main('valid_formal', complete, development=False, audit=audit)
        self.assertEqual([e['seed'] for e in collector.call_args.args[0]], list(run.FORMAL_SEEDS))

        def interrupted(environments, output):
            (output / 'searches.jsonl').write_text('{"completed": true}\n')
            (output / 'active-phase.json').write_text('{partial')
            raise RuntimeError('original synthetic interruption')

        with self.assertRaisesRegex(RuntimeError, 'original synthetic interruption'):
            self.invoke_main('interrupted', complete, collect=interrupted)
        saved = self.root / 'interrupted'
        self.assertEqual(rows(saved / 'searches.jsonl'), [{'completed': True}])
        self.assertEqual((saved / 'active-phase.json').read_text(), '{partial')
        failure = json.loads((saved / 'failure.json').read_text())
        self.assertEqual(failure['message'], 'original synthetic interruption')
        self.assertEqual(failure['active_phase_file'], 'active-phase.json')


if __name__ == '__main__':
    unittest.main()
