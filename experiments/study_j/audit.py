"""Replay training and verify searches from full functions and independent GRU math."""

import copy
import hashlib
import heapq
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.study_h.audit import reference_search as reference_experience
from experiments.study_h.tasks import make_environment, rng
from experiments.study_j.run import (ARMS, BUDGET, DATA_PATH, DEVELOPMENT_SEED,
                                     FORMAL_SEEDS, LEARNED, PROBES, source_hashes)


def compare_tensors(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_tensors(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_tensors(left, right)
    else:
        assert actual == expected


def gru_probabilities(weights, prefix, cache):
    """Independent NumPy GRU equations; no project forward method is called."""
    if prefix not in cache:
        if prefix:
            gru_probabilities(weights, prefix[:-1], cache)
            previous, token = cache[prefix[:-1]], prefix[-1]
        else:
            previous, token = np.zeros(weights['head.weight'].shape[1]), 4
        inputs = weights['recurrent.weight_ih_l0'][:, token] + weights['recurrent.bias_ih_l0']
        recurrent = weights['recurrent.weight_hh_l0'] @ previous + weights['recurrent.bias_hh_l0']
        ir, iz, inn = np.split(inputs, 3)
        hr, hz, hn = np.split(recurrent, 3)
        reset = 1 / (1 + np.exp(-(ir + hr)))
        update = 1 / (1 + np.exp(-(iz + hz)))
        new = np.tanh(inn + reset * hn)
        cache[prefix] = (1 - update) * new + update * previous
    logits = weights['head.weight'] @ cache[prefix] + weights['head.bias']
    values = np.exp(logits - logits.max())
    return values / values.sum()


def reference_order(weights=None, counts=None, budget=BUDGET, max_length=9, local=False):
    frontier, cache, ordered = [(0, ())], {}, []
    used = predictions = positions = children = recurrent_positions = 0
    peak = 1
    while frontier:
        cost, program = heapq.heappop(frontier)
        ordered.append(program)
        used += len(program)
        if used > budget:
            break
        if len(program) < max_length:
            if weights is not None:
                context = program[-2:] if local else program
                probabilities = gru_probabilities(weights, context, cache)
                recurrent_positions += 1 + len(context)
            elif counts is not None:
                context = (4, 4) + program
                row = counts[context[-2]][context[-1]]
                probabilities = [(value + 1) / (sum(row) + 4) for value in row]
            else:
                probabilities = [.25] * 4
            edges = [round(1000000 * -math.log(.9 * float(p) + .025)) for p in probabilities]
            for op, edge in enumerate(edges):
                heapq.heappush(frontier, (cost + edge, program + (op,)))
            predictions += 1
            positions += 1 + len(program)
            children += 4
            peak = max(peak, len(frontier))
    return ordered, dict(candidates=len(ordered), minimum_primitive_ops=used, predictions=predictions,
                         prefix_positions=positions, children=children, maximum_heap_entries=peak,
                         recurrent_input_positions=recurrent_positions)


def reference_search(primitives, examples, programs, budget, functions):
    used = comparisons = rejected = 0
    for index, program in enumerate(programs, 1):
        if program not in functions:
            table = list(range(len(primitives[0])))
            for operation in program:
                table = [primitives[operation][state] for state in table]
            functions[program] = table
        outputs = functions[program]
        wrong = next((i for i, (start, target) in enumerate(examples) if outputs[start] != target), len(examples))
        checked = min(wrong + 1, len(examples))
        cost = len(program) * checked
        if used + cost > budget:
            complete, partial = divmod(budget - used, len(program))
            state = examples[complete][0]
            for operation in program[:partial]:
                state = primitives[operation][state]
            return dict(status='exhausted', candidate_program=list(program), candidates=index,
                rejected=rejected, comparisons=comparisons + complete, primitive_ops=budget,
                example_index=complete, operation_index=partial, terminal_state=state)
        used += cost
        comparisons += checked
        if wrong == len(examples):
            return dict(status='accepted', candidate_program=list(program), candidates=index,
                rejected=rejected, comparisons=comparisons, primitive_ops=used,
                example_index=len(examples) - 1, operation_index=len(program), terminal_state=outputs[examples[-1][0]])
        rejected += 1
    return dict(status='space_exhausted', candidate_program=None, candidates=len(programs),
                rejected=rejected, comparisons=comparisons, primitive_ops=used)


def replay_training(folder, environment, searches, updates, costs, phases):
    seed = environment['seed']
    states = {}
    initialization = rng(seed, 'study_j:initial').getrandbits(63)
    assert costs['initial_seed'] == initialization
    torch.manual_seed(initialization)
    initial_model = nn.ModuleDict(dict(recurrent=nn.GRU(5, 16, batch_first=True, dtype=torch.float64),
                                      head=nn.Linear(16, 4, dtype=torch.float64)))
    saved = torch.load(folder / 'initial.pt', weights_only=True)
    assert saved['hidden'] == 16 and saved['attempts'] == 0 and saved['optimizer'] is None
    compare_tensors(saved['model'], initial_model.state_dict())
    initial = saved['model']
    models = {arm: copy.deepcopy(initial_model) for arm in LEARNED}
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
        weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False) for arm, model in models.items()}
    programs, accepted_ids = [], []
    steps, tokens = {arm: 0 for arm in LEARNED}, {arm: 0 for arm in LEARNED}
    physical_rows = {arm: 0 for arm in LEARNED}
    physical_positions = {arm: 0 for arm in LEARNED}
    expected_keys, expected_checkpoints = set(), {}
    tasks = environment['tasks']['experience']
    for index, task in enumerate(tasks):
        result = searches[(seed, 'experience', task['id'], 'uniform')]['result']
        if result['status'] == 'accepted' and result['candidate_program']:
            program = list(result['candidate_program'])
            programs.append(program)
            accepted_ids.append(task['id'])
        for arm in LEARNED:
            corpus = programs
            phase = phases[(seed, 'replay', arm, index + 1, None)]
            assert phase['steps'] == (16 if corpus else 0)
            assert phase['replay_tokens'] == 16 * sum(map(len, corpus))
            assert phase['skip_reason'] == (None if corpus else 'empty_corpus')
            assert phase['seconds'] >= 0
            if not corpus:
                assert phase['input_rows'] == phase['input_positions'] == 0
                continue
            if arm == 'local_trained':
                contexts = [[4] + p[max(0, i - 2):i] for p in corpus for i in range(len(p))]
                positions = torch.tensor([len(p) - 1 for p in contexts])
                width = max(map(len, contexts))
                inputs = torch.tensor([p + [0] * (width - len(p)) for p in contexts])
                labels = torch.tensor([token for p in corpus for token in p])
            else:
                width = max(map(len, corpus))
                inputs = torch.tensor([[4] + p[:-1] + [0] * (width - len(p)) for p in corpus])
                labels = torch.tensor([p + [-100] * (width - len(p)) for p in corpus])
            assert phase['input_rows'] == 16 * len(inputs)
            assert phase['input_positions'] == 16 * inputs.numel()
            model, optimizer = models[arm], optimizers[arm]
            for step in range(1, 17):
                key = (seed, arm, index + 1, step)
                expected_keys.add(key)
                recorded = updates[key]
                optimizer.zero_grad(set_to_none=True)
                encoded = F.one_hot(inputs, num_classes=5).to(torch.float64)
                recurrent, _ = model['recurrent'](encoded)
                logits = model['head'](recurrent)
                selected = logits[torch.arange(len(inputs)), positions] if arm == 'local_trained' else logits.flatten(0, 1)
                loss = F.cross_entropy(selected, labels.flatten())
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                assert recorded['loss_before_step'] == loss.item(), key
                assert recorded['gradient_norm_before_clip'] == norm.item(), key
                assert recorded['target_tokens'] == sum(map(len, corpus))
                assert recorded['input_rows'] == len(inputs)
                assert recorded['input_positions'] == inputs.numel()
                assert recorded['step_seconds'] >= 0
                steps[arm] += 1
                tokens[arm] += recorded['target_tokens']
                physical_rows[arm] += len(inputs)
                physical_positions[arm] += inputs.numel()
        if index + 1 in (8, 16, len(tasks)):
            for arm in LEARNED:
                name = f'{arm}-{index + 1}.pt'
                saved = torch.load(folder / name, weights_only=True)
                compare_tensors(saved['model'], models[arm].state_dict())
                compare_tensors(saved['optimizer'], optimizers[arm].state_dict())
                assert saved['hidden'] == 16
                assert saved['attempts'] == index + 1 and saved['steps'] == steps[arm]
                assert saved['replay_tokens'] == tokens[arm] and saved['training_seconds'] >= 0
                assert saved['input_rows'] == physical_rows[arm]
                assert saved['input_positions'] == physical_positions[arm]
                assert saved['training_seconds'] == sum(phases[(seed, 'replay', arm, n, None)]['seconds']
                                                       for n in range(1, index + 2))
                details = {key: saved[key] for key in ('attempts', 'steps', 'replay_tokens', 'input_rows', 'input_positions', 'training_seconds')}
                expected_checkpoints[name] = details
                states[(arm, index + 1)] = saved['model']
    assert costs['checkpoints'] == expected_checkpoints
    assert costs['training_steps'] == steps and costs['replay_tokens'] == tokens
    assert costs['input_rows'] == physical_rows and costs['input_positions'] == physical_positions
    expected_seconds = {arm: sum(phases[(seed, 'replay', arm, n, None)]['seconds']
                                for n in range(1, len(tasks) + 1)) for arm in LEARNED}
    assert costs['training_seconds'] == expected_seconds
    corpus = json.loads((folder / 'corpus.json').read_text())
    assert corpus == dict(programs=programs, accepted_task_ids=accepted_ids)
    counts = np.zeros((5, 5, 4), dtype=np.int64)
    for program in programs:
        padded = [4, 4] + program
        for i in range(len(program)):
            counts[padded[i], padded[i + 1], padded[i + 2]] += 1
    assert json.loads((folder / 'counts.json').read_text()) == counts.tolist()
    assert costs['corpus_programs'] == len(programs) and costs['corpus_tokens'] == sum(map(len, programs))
    expected_files = {'initial.pt'} | set(expected_checkpoints)
    assert {p.name for p in folder.glob('*.pt')} == expected_files == costs['checkpoint_bytes'].keys()
    assert costs['checkpoint_bytes'] == {name: (folder / name).stat().st_size for name in expected_files}
    assert costs['corpus_bytes'] == (folder / 'corpus.json').stat().st_size
    assert costs['count_file_bytes'] == (folder / 'counts.json').stat().st_size
    assert costs['count_table_cells'] == 100
    parameter_states = {arm: model.state_dict() for arm, model in models.items()}
    parameter_states.update(full_frozen=initial, local_frozen=initial)
    expected_parameters = {arm: sum(p.numel() for p in state.values()) for arm, state in parameter_states.items()}
    assert set(expected_parameters.values()) == {1172}
    assert costs['parameters'] == expected_parameters
    assert costs['parameter_bytes'] == {arm: count * 8 for arm, count in expected_parameters.items()}
    assert costs['initialization_seconds'] >= 0
    for name in ('optimizer_setup_seconds', 'training_seconds'):
        assert all(value >= 0 for value in costs[name].values())
    assert costs['clone_seconds'] >= 0 and costs['count_seconds'] >= 0
    prior_states = {arm: states[(arm, len(tasks))] for arm in LEARNED}
    prior_states.update(full_frozen=initial, local_frozen=initial)
    prior_states.update({f'{kind}_after_{n}': states[(f'{kind}_trained', n)]
                         for kind in ('full', 'local') for n in (8, 16)})
    return prior_states, counts.tolist(), expected_keys


def audit_run(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert metadata['seeds'] in ([DEVELOPMENT_SEED], list(FORMAL_SEEDS)), 'Incomplete fixed seed grid'
    assert metadata['source_sha256'] == source_hashes()
    assert metadata['data_sha256'] == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert metadata['arms'] == list(ARMS) and metadata['probes'] == list(PROBES) and metadata['budget'] == BUDGET
    assert metadata['torch'] == torch.__version__ == '2.8.0+cu126' and metadata['threads'] == 1
    assert metadata['device'] == 'cpu' and metadata['dtype'] == 'float64' and metadata['deterministic_algorithms']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    environments = json.loads((folder / 'environments.json').read_text())
    source = json.loads(DATA_PATH.read_text())
    assert [e['seed'] for e in source] == [DEVELOPMENT_SEED, *FORMAL_SEEDS]
    assert all(len(e['tasks']['experience']) >= 16 for e in source)
    assert environments == [e for e in source if e['seed'] in metadata['seeds']]
    assert [e['seed'] for e in environments] == metadata['seeds']
    rows = [json.loads(line) for line in (folder / 'searches.jsonl').read_text().splitlines()]
    indexed = {(r['seed'], r['family'], r['task_id'], r['arm']): r for r in rows}
    scored_rows = [json.loads(line) for line in (folder / 'evaluations.jsonl').read_text().splitlines()]
    scored = {(r['seed'], r['family'], r['task_id'], r['arm']): r for r in scored_rows}
    update_rows = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines()]
    updates = {(r['seed'], r['arm'], r['attempt'], r['step']): r for r in update_rows}
    phase_rows = [json.loads(line) for line in (folder / 'phases.jsonl').read_text().splitlines()]
    phases = {(r['seed'], r['stage'], r.get('arm'), r.get('attempt'), r.get('name')): r for r in phase_rows}
    assert len(indexed) == len(rows) and len(scored) == len(scored_rows) and indexed.keys() == scored.keys()
    assert len(updates) == len(update_rows)
    assert len(phases) == len(phase_rows) and all(r['seconds'] >= 0 for r in phase_rows)
    assert json.loads((folder / 'active-phase.json').read_text()) == dict(stage='completed')
    expected_keys, expected_updates, expected_phases, costs = set(), set(), set(), {}
    for environment in environments:
        seed, primitives = environment['seed'], environment['primitives']
        assert environment == json.loads(json.dumps(make_environment(seed))), 'Frozen task generation mismatch'
        assert len(environment['tasks']['experience']) >= 16
        signatures, functions = set(), {}
        for family, tasks in environment['tasks'].items():
            for task in tasks:
                table = list(range(16))
                for operation in task['generating_program']:
                    table = [primitives[operation][state] for state in table]
                assert sorted(task['train'] + task['test']) == [[i, value] for i, value in enumerate(table)]
                assert len(task['train']) == len(task['test']) == 8
                assert tuple(table) not in signatures
                signatures.add(tuple(table))
                if family == 'experience':
                    key = (seed, family, task['id'], 'uniform')
                    expected_keys.add(key)
                    assert indexed[key]['result'] == reference_experience(primitives, task['train'], [], BUDGET), key
        seed_folder = folder / f'seed-{seed}'
        costs[seed] = json.loads((seed_folder / 'costs.json').read_text())
        states, counts, trained_keys = replay_training(seed_folder, environment, indexed, updates, costs[seed], phases)
        expected_updates.update(trained_keys)
        expected_seconds = {}
        for arm in LEARNED:
            key = (seed, 'optimizer_setup', arm, None, None)
            expected_phases.add(key)
            expected_seconds[arm] = phases[key]['seconds']
        assert costs[seed]['optimizer_setup_seconds'] == expected_seconds
        for stage, field in (('initialization', 'initialization_seconds'), ('clone', 'clone_seconds'),
                             ('counts', 'count_seconds')):
            key = (seed, stage, None, None, None)
            expected_phases.add(key)
            assert phases[key]['seconds'] == costs[seed][field]
        expected_phases.update((seed, 'replay', arm, n, None) for arm in LEARNED
                               for n in range(1, len(environment['tasks']['experience']) + 1))
        expected_phases.update((seed, 'checkpoint_save', None, None, name) for name in costs[seed]['checkpoint_bytes'])
        expected_phases.update((seed, 'probe_load', arm, None, None) for arm in PROBES)
        seed_phases = [r for r in phase_rows if r['seed'] == seed]
        costs[seed]['journal_phase_seconds'] = {
            stage: sum(r['seconds'] for r in seed_phases if r['stage'] == stage)
            for stage in {r['stage'] for r in seed_phases}}
        costs[seed]['skipped_replay_attempts'] = sum(r.get('skip_reason') == 'empty_corpus' for r in seed_phases)
        expected_ranking = {arm: {k: v for k, v in phases[(seed, 'ranking', arm, None, None)].items()
                                 if k not in ('seed', 'stage', 'arm')} for arm in ARMS + PROBES}
        assert costs[seed]['ranking'] == expected_ranking
        costs[seed]['journal_ranking_predictions'] = sum(r['predictions'] for r in expected_ranking.values())
        assert costs[seed]['order_file_bytes'] == {
            arm: (seed_folder / f'order-{arm}.txt').stat().st_size for arm in ARMS + PROBES}
        for arm in ARMS + PROBES:
            weights = {name: tensor.numpy() for name, tensor in states[arm].items()} if arm in states else None
            order, detail = reference_order(weights, counts if arm == 'counts' else None, budget=BUDGET,
                                            local=arm.startswith('local_'))
            recorded_order = [tuple(map(int, line)) for line in (seed_folder / f'order-{arm}.txt').read_text().splitlines()]
            assert order == recorded_order, (seed, arm, 'candidate ordering')
            recorded_detail = json.loads((seed_folder / f'ranking-{arm}.json').read_text())
            assert recorded_detail == costs[seed]['ranking'][arm] and recorded_detail['seconds'] >= 0
            assert {k: v for k, v in recorded_detail.items() if k != 'seconds'} == detail
            phase_key = (seed, 'ranking', arm, None, None)
            expected_phases.add(phase_key)
            assert {k: v for k, v in phases[phase_key].items() if k not in ('seed', 'stage', 'arm')} == recorded_detail
            for family in ('composition', 'shifted'):
                for task in environment['tasks'][family]:
                    key = (seed, family, task['id'], arm)
                    expected_keys.add(key)
                    expected = reference_search(primitives, task['train'], order, BUDGET, functions)
                    assert indexed[key]['result'] == expected, key
        for family, tasks in environment['tasks'].items():
            for arm in (('uniform',) if family == 'experience' else ARMS + PROBES):
                for task in tasks:
                    key = (seed, family, task['id'], arm)
                    result = indexed[key]['result']
                    if result['status'] == 'accepted':
                        predictions = list(range(16))
                        for op in result['candidate_program']:
                            predictions = [primitives[op][state] for state in predictions]
                        train = sum(predictions[x] == y for x, y in task['train'])
                        test = sum(predictions[x] == y for x, y in task['test'])
                        expected = dict(train_correct=train, test_correct=test, exact=train + test == 16,
                            predictions=predictions, primitive_ops=16 * len(result['candidate_program']))
                    else:
                        expected = dict(train_correct=0, test_correct=0, exact=False, predictions=None, primitive_ops=0)
                    assert scored[key]['evaluation'] == expected, key
                    assert indexed[key]['search_seconds'] >= 0 and scored[key]['evaluation_seconds'] >= 0
                    indexed[key].update(evaluation=expected, evaluation_seconds=scored[key]['evaluation_seconds'])
        print(f'Audited environment {seed}', flush=True)
    assert expected_keys == indexed.keys() and expected_updates == updates.keys()
    assert expected_phases == phases.keys()
    paths = sorted(p for p in folder.rglob('*') if p.is_file())
    audit = dict(status='passed', seeds=metadata['seeds'], source_sha256=metadata['source_sha256'],
        runtime={name: metadata[name] for name in ('torch', 'threads', 'device', 'dtype', 'deterministic_algorithms')},
        data_sha256=metadata['data_sha256'], searches=len(rows), gradient_steps=len(update_rows),
        input_sha256={str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return rows, costs, metadata, audit
