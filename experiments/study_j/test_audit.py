"""Independent tensor, evidence, and inference checks on hand-built fixtures."""

import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.study_j import audit, run
from experiments.study_j.analyze import summarize
from experiments.study_i.model import ProgramPrior, replay_steps
from experiments.study_j.local import local_replay_steps
from experiments.study_i.search import rank_programs, search, token_counts
from experiments.study_j.test_collection import fixture, records


class IndependentMathTests(unittest.TestCase):
    def test_numpy_gru_matches_torch_and_rank_order(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        rng = random.Random(91)
        for local in (False, True):
            torch.manual_seed(9)
            model = ProgramPrior(16)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
            update = local_replay_steps if local else replay_steps
            list(update(model, optimizer, [(0, 1, 2), (3, 1), (1,)], steps=4))
            weights = {name: tensor.numpy() for name, tensor in model.state_dict().items()}
            cache = {}
            for _ in range(100):
                prefix = tuple(rng.randrange(4) for _ in range(rng.randrange(10)))
                context = prefix[-2:] if local else prefix
                np.testing.assert_allclose(audit.gru_probabilities(weights, context, cache),
                                           model.probabilities(context), rtol=1e-13, atol=1e-14)
            expected, reference = audit.reference_order(weights, budget=2000, max_length=7, local=local)
            order, actual = rank_programs(lambda p: model.probabilities(p[-2:] if local else p), budget=2000, max_length=7)
            self.assertEqual(order, expected)
            self.assertEqual(actual, {k:v for k,v in reference.items() if k != 'recurrent_input_positions'})
            predicted = order[:-1] if actual['minimum_primitive_ops'] > 2000 else order
            self.assertEqual(reference['recurrent_input_positions'],
                             sum(1 + (min(2, len(p)) if local else len(p)) for p in predicted if len(p) < 7))

    def test_full_function_oracle_checks_terminal_state_and_comparisons(self):
        primitives = [(1, 2, 3, 0), (2, 0, 3, 1), (3, 2, 1, 0), (0, 2, 1, 3)]
        examples = [(0, 2), (1, 0), (3, 1)]
        order, _ = rank_programs(lambda _: [.1, .2, .3, .4], budget=100)
        for budget in range(101):
            actual = json.loads(json.dumps(search(primitives, examples, order, budget)))
            self.assertEqual(actual, audit.reference_search(primitives, examples, order, budget, {}))
        counts = token_counts([(0, 1, 2), (0, 1, 3)])
        from experiments.study_i.search import count_probabilities
        order, detail = audit.reference_order(counts=counts, budget=200)
        self.assertEqual(detail.pop('recurrent_input_positions'), 0)
        self.assertEqual(rank_programs(lambda p: count_probabilities(counts, p), 200), (order, detail))

    def test_incomplete_formal_grid_is_rejected_before_any_replay(self):
        for seeds in ([], [12000], [12000, 12001]):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / 'metadata.json').write_text(json.dumps(dict(status='completed', seeds=seeds)))
                with self.assertRaisesRegex(AssertionError, 'Incomplete fixed seed grid'):
                    audit.audit_run(folder)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.environment = fixture()
        with patch.object(run, 'BUDGET', 64):
            run.collect([cls.environment], cls.folder)
        cls.costs = json.loads((cls.folder / 'seed-7/costs.json').read_text())
        phase_rows = records(cls.folder / 'phases.jsonl')
        cls.phases = {(r['seed'], r['stage'], r.get('arm'), r.get('attempt'), r.get('name')): r for r in phase_rows}
        cls.costs['journal_phase_seconds'] = {stage: sum(r['seconds'] for r in phase_rows if r['stage'] == stage)
                                             for stage in {r['stage'] for r in phase_rows}}
        cls.costs['skipped_replay_attempts'] = 0
        cls.costs['journal_ranking_predictions'] = sum(r['predictions'] for r in phase_rows if r['stage'] == 'ranking')
        cls.rows = records(cls.folder / 'searches.jsonl')
        evaluations = {(r['family'], r['task_id'], r['arm']): r for r in records(cls.folder / 'evaluations.jsonl')}
        for row in cls.rows:
            evaluated = evaluations[(row['family'], row['task_id'], row['arm'])]
            row.update(evaluation=evaluated['evaluation'], evaluation_seconds=evaluated['evaluation_seconds'])
        cls.searches = {(r['seed'], r['family'], r['task_id'], r['arm']): r for r in cls.rows}
        cls.updates = {(r['seed'], r['arm'], r['attempt'], r['step']): r for r in records(cls.folder / 'updates.jsonl')}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_separate_replay_reproduces_all_checkpoint_weights_and_optimizer_states(self):
        states, counts, keys = audit.replay_training(self.folder / 'seed-7', self.environment,
                                                    self.searches, self.updates, self.costs, self.phases)
        self.assertEqual(keys, self.updates.keys())
        self.assertEqual(len(keys), 512)
        self.assertEqual(counts[4][4][0], 16)
        self.assertEqual(len(states), 8)

    def test_altered_training_record_is_rejected(self):
        changed = copy.deepcopy(self.updates)
        changed[(7, 'local_trained', 1, 1)]['loss_before_step'] += 1
        with self.assertRaises(AssertionError):
            audit.replay_training(self.folder / 'seed-7', self.environment,
                                  self.searches, changed, self.costs, self.phases)

    def test_actual_costs_count_shared_training_once_and_probes_separately(self):
        result = summarize(self.rows, {7: self.costs}, dict(elapsed_seconds=1))
        self.assertFalse(result['primary']['confirmatory'])
        self.assertIsNone(result['primary']['ci95'])
        self.assertEqual(result['actual_cost']['gradient_steps'], 512)
        self.assertEqual(result['actual_cost']['replay_target_tokens'], 4352)
        self.assertEqual(result['actual_cost']['searches'], 36)
        self.assertEqual(result['actual_cost']['training_input_rows'], 4352)
        self.assertEqual(result['actual_cost']['training_input_positions'], 4352)
        by_arm = {r['arm']: r for r in result['standalone_cost_totals']}
        self.assertEqual(by_arm['uniform']['experience_tasks'], 0)
        self.assertEqual(by_arm['local_frozen']['training_steps'], 0)
        self.assertEqual(by_arm['local_after_8']['experience_tasks'], 8)
        self.assertEqual(by_arm['local_after_8']['training_steps'], 128)
        self.assertEqual(by_arm['full_trained']['training_steps'], 256)
        self.assertEqual(by_arm['local_trained']['parameters_per_environment'], 1172)
        contrasts = result['descriptive_contrasts']
        self.assertEqual(len(contrasts), 9)
        self.assertEqual(contrasts[0]['contrast'], 'local_trained minus full_trained')
        self.assertEqual({r['family'] for r in contrasts}, {'composition', 'shifted', 'balanced'})

    def test_negative_interval_disagreement_is_reported(self):
        rows, costs = [], {}
        for offset, seed in enumerate(run.FORMAL_SEEDS):
            costs[seed] = self.costs
            rows += [dict(r, seed=seed) for r in self.rows if r['family'] == 'experience']
            for arm in run.ARMS + run.PROBES:
                for family in ('composition', 'shifted'):
                    for index in range(16):
                        exact = family == 'composition' and (
                            (offset < 6 and arm == 'full_trained') or
                            (offset >= 6 and arm == 'local_trained' and index == 0))
                        rows.append(dict(seed=seed, arm=arm, family=family, task_id=f'{family}-{index}',
                            result=dict(status='accepted', primitive_ops=1, candidates=1, comparisons=8),
                            evaluation=dict(exact=exact, test_correct=8 * exact, primitive_ops=16),
                            search_seconds=0, evaluation_seconds=0))
        primary = summarize(rows, costs, dict(elapsed_seconds=1))['primary']
        self.assertEqual(primary['differences'], [-1] * 6 + [1 / 16] * 26)
        self.assertTrue(primary['confirmatory'])
        self.assertLess(primary['ci95'][0], 0)
        self.assertGreater(primary['ci95'][1], 0)
        self.assertLess(primary['bootstrap95'][1], 0)
        self.assertTrue(primary['interval_disagreement'])


class CompleteAuditTests(unittest.TestCase):
    def test_complete_fixture_passes_and_unrecorded_cost_entries_fail(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        environment = fixture()
        unique, signatures = [], set()
        for length in range(4):
            for program in product(range(4), repeat=length):
                table = list(range(16))
                for op in program:
                    table = [environment['primitives'][op][x] for x in table]
                if tuple(table) not in signatures:
                    signatures.add(tuple(table))
                    unique.append(dict(generating_program=list(program),
                        train=[[i, table[i]] for i in range(8)], test=[[i, table[i]] for i in range(8, 16)]))
        environment['tasks'] = {family: [dict(id=f'{family}-{i:02d}', **task) for i, task in enumerate(group)]
            for family, group in (('experience', unique[:16]), ('composition', unique[16:17]), ('shifted', unique[17:18]))}
        self.assertEqual(len(environment['tasks']['experience']), 16)
        self.assertEqual(len(environment['tasks']['shifted']), 1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'run'
            folder.mkdir()
            data = Path(directory) / 'fixture.json'
            data.write_text(json.dumps([environment, *[dict(environment, seed=s) for s in run.FORMAL_SEEDS]]))
            with patch.object(run, 'BUDGET', 64):
                run.collect([environment], folder)
            (folder / 'environments.json').write_text(json.dumps([environment]))
            metadata = dict(status='completed', seeds=[7], source_sha256=run.source_hashes(),
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), arms=list(run.ARMS), probes=list(run.PROBES),
                budget=64, torch=torch.__version__, threads=1, device='cpu', dtype='float64',
                deterministic_algorithms=True, elapsed_seconds=1)
            (folder / 'metadata.json').write_text(json.dumps(metadata))
            path = folder / 'seed-7/costs.json'
            original = json.loads(path.read_text())
            with patch.object(audit, 'DATA_PATH', data), patch.object(audit, 'DEVELOPMENT_SEED', 7), patch.object(
                    audit, 'BUDGET', 64), patch.object(audit, 'make_environment', return_value=environment):
                rows, costs, meta, result = audit.audit_run(folder)
                self.assertEqual(result['status'], 'passed')
                actual = summarize(rows, costs, meta)['actual_cost']
                phase_rows = records(folder / 'phases.jsonl')
                self.assertEqual(actual['initialization_seconds'],
                                 sum(r['seconds'] for r in phase_rows if r['stage'] == 'initialization'))
                for field in ('optimizer_setup_seconds', 'training_seconds', 'input_rows', 'input_positions', 'ranking'):
                    changed = copy.deepcopy(original)
                    changed[field]['unrecorded'] = ({'seconds': 1000000, 'predictions': 1234567}
                                                   if field == 'ranking' else 1000000)
                    path.write_text(json.dumps(changed))
                    with self.subTest(field=field), self.assertRaises(AssertionError):
                        audit.audit_run(folder)


if __name__ == '__main__':
    unittest.main()
