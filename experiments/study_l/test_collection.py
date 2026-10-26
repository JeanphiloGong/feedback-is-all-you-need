"""Synthetic checks for task resets, visible-only learning, and run journals."""

import copy
import hashlib
import io
from itertools import chain, product
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_i.search import search
from experiments.study_j.audit import compare_tensors
from experiments.study_l import run
from experiments.study_l.feedback import local_probabilities


def synthetic_environment(seed):
    source = random.Random(441)
    primitives = []
    for _ in range(2):
        permutation = source.sample(range(16), 16)
        primitives.extend([permutation, [permutation.index(state) for state in range(16)]])
    tasks = dict(experience=[], composition=[], shifted=[])
    seen = set()
    for family, lengths in (('experience', (1, 3)), ('composition', (6, 9)), ('shifted', (6, 9))):
        for length in lengths:
            count = 0
            short_lengths = ((length,) if family == 'experience' else
                             (2, 4) if length == 6 else (1, 3))
            for short in chain.from_iterable(product(range(4), repeat=n) for n in short_lengths):
                program = short + (0, 1) * ((length - len(short)) // 2)
                function = list(range(16))
                for op in program:
                    function = [primitives[op][state] for state in function]
                if tuple(function) in seen:
                    continue
                seen.add(tuple(function))
                examples = [[state, value] for state, value in enumerate(function)]
                tasks[family].append(dict(id=f'{family}-{len(tasks[family]):02d}',
                    generating_program=list(program), train=examples[:8], test=examples[8:]))
                count += 1
                if count == (1 if family == 'experience' else 8):
                    break
    return dict(seed=seed, primitives=primitives, tasks=tasks, latent_motifs=[], rejections={})


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scratch = tempfile.TemporaryDirectory()
        cls.root = Path(cls.scratch.name)
        cls.environment = synthetic_environment(41)
        cls.output = cls.root / 'collection'
        cls.output.mkdir()
        with patch.multiple(run, BUDGET=512, INTERVAL=128, STEPS=2):
            run.collect([cls.environment], cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_static_arms_match_original_search_and_never_repeat(self):
        records = [json.loads(line) for line in (self.output / 'searches.jsonl').read_text().splitlines()]
        tasks = {t['id']: t for family in self.environment['tasks'].values() for t in family}
        for arm in ('uniform', 'frozen'):
            order = json.loads((self.output / f'seed-41/{arm}-order.json').read_text())
            for row in (r for r in records if r['arm'] == arm):
                task = tasks[row['task_id']]
                expected = search(self.environment['primitives'], task['train'], order, 512)
                result = row['result']
                for name in ('status', 'comparisons', 'primitive_ops', 'rejected'):
                    self.assertEqual(result[name], expected[name])
                if result['status'] == 'accepted':
                    self.assertEqual(result['candidate_program'], expected['candidate_program'])
                path = self.output / f'seed-41/{arm}/{task["id"]}/candidates.jsonl'
                events = [json.loads(line) for line in path.read_text().splitlines()]
                checked = [tuple(e['program']) for e in events if e['status'] in ('rejected', 'accepted')]
                self.assertEqual(len(checked), len(set(checked)))

    def test_every_task_resets_model_and_feedback_optimizer(self):
        base = torch.load(self.output / 'seed-41/experience-2.pt', weights_only=True)
        self.assertGreater(max(map(len, base['programs'])), 1)
        changed = 0
        for arm in ('uniformizing', 'rejection'):
            for folder in (self.output / f'seed-41/{arm}').iterdir():
                initial = torch.load(folder / 'initial.pt', weights_only=True)
                compare_tensors(initial['model'], base['model'])
                self.assertEqual(initial['optimizer']['state'], {})
                final = torch.load(folder / 'final.pt', weights_only=True)
                checkpoints = list(folder.glob('after-*.pt'))
                self.assertEqual(final['steps'], 2 * len(checkpoints))
                self.assertEqual([int(s['step']) for s in final['optimizer']['state'].values()],
                                 [final['steps']] * (6 if final['steps'] else 0))
                changed += any(not torch.equal(base['model'][n], v) for n, v in final['model'].items())
        self.assertGreater(changed, 0)

    def test_hidden_outputs_do_not_change_any_model_or_update(self):
        environment = copy.deepcopy(self.environment)
        for family in environment['tasks'].values():
            for task in family:
                task['test'] = [[state, (value + 1) % 16] for state, value in task['test']]
        output = self.root / 'altered-hidden'
        output.mkdir()
        with patch.multiple(run, BUDGET=512, INTERVAL=128, STEPS=2):
            run.collect([environment], output)
        for path in self.output.rglob('*.pt'):
            compare_tensors(torch.load(path, weights_only=True),
                            torch.load(output / path.relative_to(self.output), weights_only=True))
        first = [json.loads(line) for line in (self.output / 'seed-41/updates.jsonl').read_text().splitlines()]
        second = [json.loads(line) for line in (output / 'seed-41/updates.jsonl').read_text().splitlines()]
        for row in first + second:
            row.pop('seconds')
        self.assertEqual(first, second)

    def test_round_journals_and_phases_preserve_executed_work(self):
        phases = [json.loads(line) for line in (self.output / 'phases.jsonl').read_text().splitlines()]
        results = [json.loads(line) for line in (self.output / 'searches.jsonl').read_text().splitlines()]
        for row in (r for r in results if r['arm'] != 'initial'):
            selected = [p for p in phases if p['arm'] == row['arm'] and p['task_id'] == row['task_id']]
            self.assertEqual(sum(p['primitive_ops'] for p in selected if p['stage'] == 'visible_search'),
                             row['result']['primitive_ops'])
            for phase in (p for p in selected if p['stage'] == 'feedback'):
                self.assertEqual(phase['at'] % 128, 0)
                self.assertLess(phase['at'], 512)
                self.assertEqual(phase['steps'], 2)
                self.assertGreater(phase['parameter_change_l2'], 0)
            task_dir = self.output / f'seed-41/{row["arm"]}/{row["task_id"]}'
            events = [json.loads(line) for line in (task_dir / 'candidates.jsonl').read_text().splitlines()]
            rejected = [tuple(e['program']) for e in events if e['status'] == 'rejected']
            self.assertEqual(len(rejected), len(set(rejected)))
            self.assertEqual(len(rejected), row['result']['rejected'])

    def test_delayed_journal_flushes_are_measured_separately(self):
        clock = [0.]
        original_open = Path.open

        def delay_flush(stream):
            original_flush = stream.flush

            def flush():
                clock[0] += .25
                return original_flush()

            stream.flush = flush
            return stream

        def opened(path, *args, **kwargs):
            return delay_flush(original_open(path, *args, **kwargs))

        folder = self.root / 'delayed-journal'
        folder.mkdir()
        model = run.ProgramPrior(16)
        table = local_probabilities(model).detach().tolist()
        updates, phases = delay_flush(io.StringIO()), io.StringIO()
        primitives = [[(state + shift) % 16 for state in range(16)] for shift in (1, 2, 3, 4)]
        with patch.object(Path, 'open', new=opened), patch.object(run.time, 'perf_counter', side_effect=lambda: clock[0]), \
             patch.multiple(run, BUDGET=8, INTERVAL=4, STEPS=2):
            result = run.collect_transfer(primitives, [[0, 0], [1, 0]], folder, model,
                table, None, dict(seed=41, arm='rejection', task_id='synthetic'),
                updates, phases, folder / 'active.json')
        records = [json.loads(line) for line in phases.getvalue().splitlines()]
        search_phases = [r for r in records if r['stage'] == 'visible_search']
        feedback_phase = next(r for r in records if r['stage'] == 'feedback')
        candidate_lines = (folder / 'candidates.jsonl').read_text().splitlines(keepends=True)
        self.assertEqual(result['primitive_ops'], 8)
        self.assertEqual(sum(r['journal_seconds'] for r in search_phases), .25 * len(candidate_lines))
        self.assertEqual(sum(r['journal_bytes'] for r in search_phases), sum(map(len, candidate_lines)))
        self.assertEqual(feedback_phase['journal_seconds'], .5)
        self.assertEqual(feedback_phase['journal_bytes'], len(updates.getvalue()))
        self.assertTrue(all(r['seconds'] >= r['journal_seconds'] for r in search_phases + [feedback_phase]))

    def test_failure_is_preserved_and_formal_requires_matching_certificate(self):
        data = self.root / 'synthetic-grid.json'
        data.write_text(json.dumps([synthetic_environment(seed) for seed in (41, *range(14000, 14032))]))
        failed = self.root / 'failed'
        argv = ['study_l.run', '--development', '--source-revision', 'synthetic', '--output', str(failed)]
        with patch.multiple(run, DATA_PATH=data, BUDGET=512, INTERVAL=128, STEPS=2), \
             patch.object(run, 'source_hashes', return_value={}), \
             patch.object(run.torch, '__version__', '2.8.0+cu126'), \
             patch.object(run.sys, 'argv', argv), \
             patch.object(run, 'feedback_steps', side_effect=RuntimeError('synthetic feedback failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic feedback failure'):
                run.main()
        failure = json.loads((failed / 'failure.json').read_text())
        self.assertIn('synthetic feedback failure', failure['traceback'])
        self.assertEqual(json.loads((failed / 'active-phase.json').read_text())['stage'], 'feedback')
        self.assertGreater((failed / 'searches.jsonl').stat().st_size, 0)
        self.assertGreater(len(list(failed.rglob('*.pt'))), 0)
        certificate = self.root / 'bad-certificate.json'
        certificate.write_text(json.dumps(dict(status='passed', seeds=[41], source_sha256={},
            data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), runtime={'torch': 'wrong'})))
        for name, extra, message in (
            ('missing-certificate', [], 'requires a passing'),
            ('mismatched-certificate', ['--development-audit', str(certificate)], 'must match'),
        ):
            argv = ['study_l.run', '--source-revision', 'synthetic', '--output', str(self.root / name), *extra]
            with patch.object(run, 'DATA_PATH', data), patch.object(run, 'source_hashes', return_value={}), \
                 patch.object(run.torch, '__version__', '2.8.0+cu126'), patch.object(run.sys, 'argv', argv):
                with self.assertRaisesRegex(ValueError, message):
                    run.main()


if __name__ == '__main__':
    unittest.main()
