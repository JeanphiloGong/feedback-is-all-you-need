"""Independently replay continued learning and reconstruct every search/cost."""

import copy
import hashlib
from itertools import product
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.study_h.audit import reference_search as reference_experience
from experiments.study_h.tasks import make_environment, rng
from experiments.study_j.audit import (compare_tensors, gru_probabilities,
                                       reference_order, reference_search)
from experiments.study_k.run import (ARMS, BUDGET, DATA_PATH, DEVELOPMENT_SEED,
                                     FORMAL_SEEDS, source_hashes)


SEARCH_FIELDS = ('seed', 'phase', 'arm', 'order', 'attempt', 'checkpoint', 'task_id')
UPDATE_FIELDS = ('seed', 'phase', 'arm', 'order', 'attempt', 'step')
PHASE_FIELDS = ('seed', 'stage', 'phase', 'arm', 'order', 'attempt', 'name', 'prior_id', 'checkpoint')


def independent_steps(model, optimizer, programs):
    if not programs:
        return
    contexts = [[4] + list(program[max(0, i - 2):i]) for program in programs for i in range(len(program))]
    width = max(map(len, contexts))
    inputs = torch.tensor([row + [0] * (width - len(row)) for row in contexts])
    positions = torch.tensor([len(row) - 1 for row in contexts])
    labels = torch.tensor([token for program in programs for token in program])
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        recurrent, _ = model['recurrent'](F.one_hot(inputs, num_classes=5).to(torch.float64))
        logits = model['head'](recurrent)[torch.arange(len(inputs)), positions]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
                   target_tokens=len(labels), input_rows=len(inputs), input_positions=inputs.numel())


def audit_run(folder):
    if not __debug__:
        raise RuntimeError('Audit requires assertions; run without -O or PYTHONOPTIMIZE.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = dict(torch=torch.__version__, threads=1, device='cpu', dtype='float64', deterministic_algorithms=True)
    assert runtime['torch'] == '2.8.0+cu126'
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert math.isfinite(metadata['elapsed_seconds']) and metadata['elapsed_seconds'] >= 0
    assert metadata['seeds'] in ([DEVELOPMENT_SEED], list(FORMAL_SEEDS)), 'Incomplete seed grid'
    assert {name: metadata[name] for name in runtime} == runtime
    assert metadata['source_sha256'] == source_hashes()
    assert metadata['data_sha256'] == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert metadata['budget'] == BUDGET and metadata['arms'] == list(ARMS)
    assert metadata['orders'] == ['composition_first', 'shifted_first']
    assert metadata['checkpoints'] == [0, 8, 16]
    assert metadata['stream_indices'] == [0, 8, 2, 10, 4, 12, 6, 14]
    assert metadata['probe_indices'] == [1, 9, 3, 11, 5, 13, 7, 15]
    if metadata['seeds'] == [DEVELOPMENT_SEED]:
        assert metadata['development_audit_sha256'] is None
    else:
        certificate_bytes = (folder / 'development-audit.json').read_bytes()
        assert hashlib.sha256(certificate_bytes).hexdigest() == metadata['development_audit_sha256']
        certificate = json.loads(certificate_bytes)
        assert certificate['status'] == 'passed' and certificate['seeds'] == [DEVELOPMENT_SEED]
        assert certificate['source_sha256'] == metadata['source_sha256']
        assert certificate['data_sha256'] == metadata['data_sha256']
        assert certificate['runtime'] == runtime
    source = json.loads(DATA_PATH.read_text())
    assert [environment['seed'] for environment in source] == [DEVELOPMENT_SEED, *FORMAL_SEEDS]
    assert all(len(environment['tasks'][family]) == 16 for environment in source
               for family in ('composition', 'shifted'))
    environments = json.loads((folder / 'environments.json').read_text())
    assert environments == [environment for environment in source if environment['seed'] in metadata['seeds']]
    assert json.loads((folder / 'active-phase.json').read_text()) == {'stage': 'completed'}
    raw = {name: [json.loads(line) for line in (folder / (name + '.jsonl')).read_text().splitlines()]
           for name in ('searches', 'evaluations', 'updates', 'phases')}
    indexed = {}
    for name, fields in (('searches', SEARCH_FIELDS), ('evaluations', SEARCH_FIELDS),
                         ('updates', UPDATE_FIELDS), ('phases', PHASE_FIELDS)):
        indexed[name] = {tuple(row.get(field) for field in fields): row for row in raw[name]}
        assert len(indexed[name]) == len(raw[name]), ('Duplicate record', name)
    assert indexed['searches'].keys() == indexed['evaluations'].keys()
    expected_searches, expected_updates, expected_phases = [], [], []
    expected_files = {name + '.jsonl' for name in raw} | {'metadata.json', 'environments.json', 'active-phase.json'}
    if metadata['seeds'] != [DEVELOPMENT_SEED]:
        expected_files.add('development-audit.json')
    state_records = []

    def phase(expected):
        key = tuple(expected.get(field) for field in PHASE_FIELDS)
        actual = indexed['phases'][key]
        seconds = actual['seconds']
        assert math.isfinite(seconds) and seconds >= 0
        assert actual == dict(expected, seconds=seconds), key
        expected_phases.append(key)
        return seconds

    for environment in environments:
        seed, primitives = environment['seed'], environment['primitives']
        assert environment == json.loads(json.dumps(make_environment(seed))), 'Generator mismatch'
        signatures = set()
        for family, tasks in environment['tasks'].items():
            if family != 'experience':
                assert [len(task['generating_program']) for task in tasks] == [6] * 8 + [9] * 8
            for task in tasks:
                function = list(range(16))
                for operation in task['generating_program']:
                    function = [primitives[operation][state] for state in function]
                assert sorted(task['train'] + task['test']) == [[i, value] for i, value in enumerate(function)]
                assert len(task['train']) == len(task['test']) == 8
                assert tuple(function) not in signatures
                signatures.add(tuple(function))
        streams = {name: [(family, environment['tasks'][family][i]) for family in families
                          for i in (0, 8, 2, 10, 4, 12, 6, 14)]
                   for name, families in (('composition_first', ('composition', 'shifted')),
                                           ('shifted_first', ('shifted', 'composition')))}
        probes = [(family, environment['tasks'][family][i]) for family in ('composition', 'shifted')
                  for i in (1, 9, 3, 11, 5, 13, 7, 15)]
        directory = folder / f'seed-{seed}'
        expected_files.update({f'seed-{seed}/roles.json', f'seed-{seed}/checkpoints.json'})
        assert json.loads((directory / 'roles.json').read_text()) == dict(
            streams={name: [task['id'] for _, task in tasks] for name, tasks in streams.items()},
            probes=[task['id'] for _, task in probes], baseline_stream_order='shared:composition_first',
            baseline_probe_checkpoint=0)
        functions, orders, saved_states = {}, {}, {}

        def search_result(task, programs, key):
            identity = tuple(key[field] for field in SEARCH_FIELDS)
            expected_searches.append(identity)
            row, scored = indexed['searches'][identity], indexed['evaluations'][identity]
            result = (reference_experience(primitives, task['train'], [], BUDGET) if programs is None
                      else reference_search(primitives, task['train'], programs, BUDGET, functions))
            seconds = row['search_seconds']
            assert math.isfinite(seconds) and seconds >= 0
            assert row == dict(key, result=result, search_seconds=seconds), identity
            if result['status'] == 'accepted':
                predictions = list(range(16))
                for operation in result['candidate_program']:
                    predictions = [primitives[operation][state] for state in predictions]
                train = sum(predictions[state] == target for state, target in task['train'])
                test = sum(predictions[state] == target for state, target in task['test'])
                value = dict(train_correct=train, test_correct=test, exact=train + test == 16,
                             predictions=predictions, primitive_ops=16 * len(result['candidate_program']))
            else:
                value = dict(train_correct=0, test_correct=0, exact=False, predictions=None, primitive_ops=0)
            seconds = scored['evaluation_seconds']
            assert math.isfinite(seconds) and seconds >= 0
            assert scored == dict(key, evaluation=value, evaluation_seconds=seconds), identity
            return result

        def train(model, optimizer, programs, key, counters):
            steps = tokens = rows = positions = 0
            step_seconds = 0.
            for step, record in enumerate(independent_steps(model, optimizer, programs), 1):
                identity = tuple(dict(key, step=step)[field] for field in UPDATE_FIELDS)
                expected_updates.append(identity)
                actual = indexed['updates'][identity]
                seconds = actual['step_seconds']
                assert math.isfinite(seconds) and seconds >= 0
                assert actual == dict(key, step=step, step_seconds=seconds, **record), identity
                step_seconds += seconds
                steps += 1
                tokens += record['target_tokens']
                rows += record['input_rows']
                positions += record['input_positions']
            elapsed = phase(dict(stage='replay', **key, steps=steps, replay_tokens=tokens,
                                 input_rows=rows, input_positions=positions,
                                 skip_reason=None if programs else 'empty_corpus'))
            assert step_seconds <= elapsed + 1e-9, ('Replay shorter than its steps', key)
            for name, amount in (('steps', steps), ('replay_tokens', tokens), ('input_rows', rows),
                                 ('input_positions', positions), ('training_seconds', elapsed)):
                counters[name] += amount
            return steps

        def checkpoint(model, optimizer, programs, ids, counters, key, prior_id):
            path = directory / (key['name'] + '.pt')
            expected_files.add(str(path.relative_to(folder)))
            saved = torch.load(path, weights_only=True)
            expected = dict(hidden=16, model=model.state_dict(),
                optimizer=optimizer.state_dict() if optimizer is not None else None,
                programs=[tuple(p) for p in programs], accepted_task_ids=ids,
                counters=dict(counters), prior_id=prior_id, **key)
            compare_tensors(saved, expected)
            phase(dict(stage='checkpoint_save', **key, bytes=path.stat().st_size))
            saved_states[key['name']] = copy.deepcopy(model.state_dict())
            state_records.append(dict(**key, prior_id=prior_id, counters=dict(counters),
                                      programs=len(programs), corpus_tokens=sum(map(len, programs)),
                                      checkpoint_bytes=path.stat().st_size))

        def prior(weights, prior_id):
            numpy_weights = {name: tensor.numpy() for name, tensor in weights.items()} if weights is not None else None
            if weights is not None:
                phase(dict(stage='probability_table', seed=seed, prior_id=prior_id,
                           model_calls=21, recurrent_input_positions=57, probability_values=84))
                path = directory / f'table-{prior_id}.json'
                expected_files.add(str(path.relative_to(folder)))
                recorded = json.loads(path.read_text())
                contexts = [list(p) for length in range(3) for p in product(range(4), repeat=length)]
                assert [row['context'] for row in recorded] == contexts
                cache = {}
                for row in recorded:
                    assert set(row) == {'context', 'probabilities'}
                    expected = gru_probabilities(numpy_weights, tuple(row['context']), cache)
                    np.testing.assert_allclose(row['probabilities'], expected, rtol=1e-12, atol=1e-12)
                phase(dict(stage='table_archive', seed=seed, prior_id=prior_id, bytes=path.stat().st_size))
            ordered, details = reference_order(numpy_weights, budget=BUDGET, local=True)
            details.pop('recurrent_input_positions')
            phase(dict(stage='ranking', seed=seed, prior_id=prior_id, **details))
            path = directory / f'order-{prior_id}.txt'
            expected_files.add(str(path.relative_to(folder)))
            assert [tuple(map(int, line)) for line in path.read_text().splitlines()] == ordered
            phase(dict(stage='order_archive', seed=seed, prior_id=prior_id, bytes=path.stat().st_size))
            orders[prior_id] = ordered

        initialization = rng(seed, 'study_k:initial').getrandbits(63)
        torch.manual_seed(initialization)
        base_model = nn.ModuleDict(dict(recurrent=nn.GRU(5, 16, batch_first=True, dtype=torch.float64),
                                       head=nn.Linear(16, 4, dtype=torch.float64)))
        phase(dict(stage='initialization', seed=seed, initial_seed=initialization,
                   parameters=1172, parameter_bytes=9376))
        counters = dict(steps=0, replay_tokens=0, input_rows=0, input_positions=0, training_seconds=0.)
        checkpoint(base_model, None, [], [], counters,
                   dict(seed=seed, name='random', arm='initial', order='shared', attempt=0), None)
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
            weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
        phase(dict(stage='optimizer_setup', seed=seed))
        initial_programs, initial_ids = [], []
        for attempt, task in enumerate(environment['tasks']['experience'], 1):
            key = dict(seed=seed, phase='experience', family='experience', task_id=task['id'],
                       arm='initial', order='shared', attempt=attempt, checkpoint=None, prior_id=None)
            result = search_result(task, None, key)
            if result['status'] == 'accepted' and result['candidate_program']:
                initial_programs.append(tuple(result['candidate_program']))
                initial_ids.append(task['id'])
            train(base_model, base_optimizer, initial_programs,
                  dict(seed=seed, phase='experience', arm='initial', order='shared', attempt=attempt), counters)
        checkpoint(base_model, base_optimizer, initial_programs, initial_ids, counters,
                   dict(seed=seed, name='base', arm='initial', order='shared',
                        attempt=len(environment['tasks']['experience'])), 'base')
        prior(None, 'uniform')
        prior(base_model.state_dict(), 'base')
        for arm in ('uniform', 'frozen'):
            prior_id = 'uniform' if arm == 'uniform' else 'base'
            for attempt, (family, task) in enumerate(streams['composition_first'], 1):
                search_result(task, orders[prior_id], dict(seed=seed, phase='stream', family=family,
                    task_id=task['id'], arm=arm, order='shared', attempt=attempt, checkpoint=None, prior_id=prior_id))
        expected_checkpoints = []
        for order, tasks in streams.items():
            for arm in ('replay', 'online'):
                model, optimizer = copy.deepcopy((base_model, base_optimizer))
                programs, ids, counter = list(initial_programs), list(initial_ids), dict(counters)
                phase(dict(stage='clone', seed=seed, arm=arm, order=order))
                prior_id, changed = 'base', False
                for attempt, (family, task) in enumerate(tasks, 1):
                    if changed:
                        prior_id = f'{arm}-{order}-{attempt - 1}'
                        prior(model.state_dict(), prior_id)
                    key = dict(seed=seed, phase='stream', family=family, task_id=task['id'],
                               arm=arm, order=order, attempt=attempt, checkpoint=None, prior_id=prior_id)
                    result = search_result(task, orders[prior_id], key)
                    if arm == 'online' and result['status'] == 'accepted' and result['candidate_program']:
                        programs.append(tuple(result['candidate_program']))
                        ids.append(task['id'])
                    changed = bool(train(model, optimizer, programs,
                        dict(seed=seed, phase='stream', arm=arm, order=order, attempt=attempt), counter))
                    if attempt in (8, 16):
                        name = f'{arm}-{order}-{attempt}'
                        checkpoint_prior = name if changed else prior_id
                        checkpoint(model, optimizer, programs, ids, counter,
                                   dict(seed=seed, name=name, arm=arm, order=order, attempt=attempt), checkpoint_prior)
                        expected_checkpoints.append(dict(arm=arm, order=order, checkpoint=attempt,
                                                         name=name, prior_id=checkpoint_prior))
        for arm in ('uniform', 'frozen'):
            prior_id = 'uniform' if arm == 'uniform' else 'base'
            for family, task in probes:
                search_result(task, orders[prior_id], dict(seed=seed, phase='probe', family=family,
                    task_id=task['id'], arm=arm, order='shared', attempt=None, checkpoint=0, prior_id=prior_id))
        for item in expected_checkpoints:
            prior_id = item['prior_id']
            if prior_id not in orders:
                phase(dict(stage='probe_load', seed=seed, **item))
                prior(saved_states[item['name']], prior_id)
            for family, task in probes:
                search_result(task, orders[prior_id], dict(seed=seed, phase='probe', family=family,
                    task_id=task['id'], arm=item['arm'], order=item['order'], attempt=None,
                    checkpoint=item['checkpoint'], prior_id=prior_id))
        assert json.loads((directory / 'checkpoints.json').read_text()) == expected_checkpoints
        print(f'Audited environment {seed}', flush=True)
    assert expected_searches == list(indexed['searches']) == list(indexed['evaluations'])
    assert expected_updates == list(indexed['updates'])
    assert expected_phases == list(indexed['phases'])
    measured_seconds = (sum(row['seconds'] for row in raw['phases'])
                        + sum(row['search_seconds'] for row in raw['searches'])
                        + sum(row['evaluation_seconds'] for row in raw['evaluations']))
    assert measured_seconds <= metadata['elapsed_seconds'] + 1e-9, 'Collection shorter than its measured work'
    paths = sorted(path for path in folder.rglob('*') if path.is_file())
    assert {str(path.relative_to(folder)) for path in paths} == expected_files
    for identity, row in indexed['searches'].items():
        value = indexed['evaluations'][identity]
        row.update(evaluation=value['evaluation'], evaluation_seconds=value['evaluation_seconds'])
    audit = dict(status='passed', seeds=metadata['seeds'], source_sha256=metadata['source_sha256'],
        data_sha256=metadata['data_sha256'], runtime=runtime, searches=len(raw['searches']),
        gradient_steps=len(raw['updates']),
        input_sha256={str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    return raw['searches'], raw['phases'], state_records, metadata, audit
