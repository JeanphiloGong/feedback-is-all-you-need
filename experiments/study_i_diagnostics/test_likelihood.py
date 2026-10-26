import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.study_i.model import ProgramPrior
from experiments.study_i_diagnostics.likelihood import (
    empirical_prefix_loss, program_sets, summarize, token_scores)
from experiments.study_i_diagnostics.verify import verify


class LikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conflicting_continuations_have_nonzero_irreducible_loss(self):
        programs = [[0, 1], [0, 1], [0, 2]]
        expected = -(2 * math.log(2 / 3) + math.log(1 / 3)) / 6
        self.assertAlmostEqual(empirical_prefix_loss(programs), expected)
        self.assertEqual(empirical_prefix_loss([[0, 1], [0, 1]]), 0)
        self.assertAlmostEqual(empirical_prefix_loss([[0], [1, 0]]), 2 * math.log(2) / 3)

    def test_uniform_and_count_smoothing_match_hand_probabilities(self):
        scores = token_scores([3, 1, 0])
        for values in scores.values():
            for value in values:
                self.assertAlmostEqual(value, -math.log(4))
        counts = [[[0] * 4 for _ in range(5)] for _ in range(5)]
        counts[4][4][2] = 4
        counts[4][2][1] = 8
        result = token_scores([2, 1, 3], counts=counts)
        for raw, smoothed, expected in zip(result['raw_log_probabilities'],
                result['smoothed_log_probabilities'], [5 / 8, 9 / 12, 1 / 4]):
            self.assertAlmostEqual(raw, math.log(expected))
            self.assertAlmostEqual(smoothed, math.log(.9 * expected + .025))

    def test_neural_targets_use_start_then_only_previous_tokens(self):
        torch.manual_seed(47)
        model = ProgramPrior(16)
        program = [1, 3, 2, 0, 1]
        scores = token_scores(program, model)
        for i, token in enumerate(program):
            self.assertAlmostEqual(scores['raw_log_probabilities'][i],
                                   math.log(model.probabilities(program[:i])[token]), places=12)
        shorter = token_scores(program[:3], model)
        for actual, expected in zip(scores['raw_log_probabilities'][:3], shorter['raw_log_probabilities']):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_actual_corpus_uses_attempt_ids_and_stored_shuffled_targets(self):
        environment = dict(tasks=dict(experience=[dict(id=f'e{i}') for i in range(24)],
            composition=[dict(id='c', generating_program=[3, 2])],
            shifted=[dict(id='s', generating_program=[0, 0])]))
        corpus = dict(accepted_task_ids=['e1', 'e9', 'e20'],
                      programs=[[0, 1], [1, 2], [2, 3]], shuffled_programs=[[1, 0], [2, 1], [3, 2]])
        after8 = program_sets(environment, corpus, 'small_after_8')
        self.assertEqual([r for r in after8 if r[0] == 'training_corpus'],
                         [('training_corpus', 'e1', [0, 1])])
        shuffled = program_sets(environment, corpus, 'small_shuffled')
        self.assertEqual([r[2] for r in shuffled if r[0] == 'training_corpus'], corpus['shuffled_programs'])
        self.assertEqual([r[2] for r in shuffled if r[0] == 'experience_all'], corpus['programs'])

    def test_summary_weights_environments_equally_despite_token_imbalance(self):
        records = []
        for seed, loss, length in ((1, 1., 1), (2, 3., 9)):
            for arm in ('small_trained', 'small_frozen', 'uniform'):
                for dataset in ('experience_all', 'composition', 'shifted', 'training_corpus'):
                    records.append(dict(seed=seed, arm=arm, dataset=dataset, task_id='toy',
                        program=[0] * length, raw_log_probabilities=[-loss] * length,
                        smoothed_log_probabilities=[-loss] * length))
        summary = summarize(records)
        overall = [r for r in summary['aggregates'] if r['length'] is None]
        self.assertTrue(all(r['raw_loss'] == 2 for r in overall))
        self.assertTrue(all(r['mean'] == 0 for r in summary['differences']))
        self.assertTrue(all(r['zero'] == 2 for r in summary['differences']))

    def test_independent_reconstruction_rejects_incomplete_grid_and_wrong_summaries(self):
        # Synthetic constant priors and hand-built programs; no benchmark generation.
        arms = ('uniform', 'counts', 'small_frozen', 'small_trained', 'small_shuffled',
                'large_trained', 'small_after_8', 'small_after_16')
        with tempfile.TemporaryDirectory() as directory:
            folder, diagnostic = Path(directory) / 'raw', Path(directory) / 'diagnostic'
            folder.mkdir()
            diagnostic.mkdir()
            environments, rows = [], []
            for seed in range(11000, 11032):
                env = dict(seed=seed, tasks=dict(experience=[dict(id=f'e{i}') for i in range(24)],
                    composition=[dict(id='c', generating_program=[0] * 6)],
                    shifted=[dict(id='s', generating_program=[1] * 9)]))
                corpus = dict(accepted_task_ids=['e0', 'e15'], programs=[[0], [1]],
                              shuffled_programs=[[0], [1]])
                environments.append(env)
                seed_folder = folder / f'seed-{seed}'
                seed_folder.mkdir()
                (seed_folder / 'corpus.json').write_text(json.dumps(corpus))
                (seed_folder / 'counts.json').write_text(json.dumps([[[0] * 4 for _ in range(5)] for _ in range(5)]))
                for arm in arms:
                    for dataset, task_id, program in program_sets(env, corpus, arm):
                        rows.append(dict(seed=seed, arm=arm, dataset=dataset, task_id=task_id,
                                         program=program, **token_scores(program)))
            (folder / 'environments.json').write_text(json.dumps(environments))
            metadata = dict(status='completed', program_records=len(rows),
                independently_checked_neural_tokens=sum(len(r['program']) for r in rows
                                                        if r['arm'] not in ('uniform', 'counts')),
                original_input_sha256={str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in folder.rglob('*') if p.is_file()})
            for scenario in ('valid', 'missing_program', 'missing_position', 'incorrect_bound'):
                with self.subTest(scenario=scenario):
                    selected = rows[1:] if scenario == 'missing_program' else rows
                    report = summarize(selected)
                    if scenario == 'missing_position':
                        report['by_environment'].pop()
                    if scenario == 'incorrect_bound':
                        report['training'][0]['empirical_bound'] += .1
                    (diagnostic / 'programs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))
                    (diagnostic / 'summary.json').write_text(json.dumps(report))
                    metadata['program_records'] = len(selected)
                    metadata['output_sha256'] = {name: hashlib.sha256((diagnostic / name).read_bytes()).hexdigest()
                                                 for name in ('programs.jsonl', 'summary.json')}
                    (diagnostic / 'metadata.json').write_text(json.dumps(metadata))
                    if scenario == 'valid':
                        self.assertEqual(verify(folder, diagnostic)['status'], 'passed')
                    else:
                        with self.assertRaises(AssertionError):
                            verify(folder, diagnostic)


if __name__ == '__main__':
    unittest.main()
