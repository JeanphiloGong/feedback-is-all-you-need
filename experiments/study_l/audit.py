"""Independently reconstruct gradients, partial executions, and complete archives."""

import copy
import hashlib
import heapq
from itertools import product
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.study_h.audit import reference_search as reference_experience
from experiments.study_h.tasks import make_environment, rng
from experiments.study_j.audit import compare_tensors, gru_probabilities, reference_search
from experiments.study_l.run import (ARMS, BUDGET, DATA_PATH, DEVELOPMENT_SEED,
                                     FORMAL_SEEDS, INTERVAL, STEPS,
                                     check_development_evidence, source_hashes)


def independent_initial_steps(model, optimizer, programs):
    if not programs:
        return
    contexts = [[4] + list(p[max(0, i - 2):i]) for p in programs for i in range(len(p))]
    width = max(map(len, contexts))
    inputs = torch.tensor([row + [0] * (width - len(row)) for row in contexts])
    positions = torch.tensor([len(row) - 1 for row in contexts])
    labels = torch.tensor([token for program in programs for token in program])
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        states, _ = model['recurrent'](F.one_hot(inputs, num_classes=5).to(torch.float64))
        logits = model['head'](states)[torch.arange(len(inputs)), positions]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
                   target_tokens=len(labels), input_rows=len(inputs), input_positions=inputs.numel())


def independent_feedback_steps(model, optimizer, arm, programs):
    if arm == 'rejection' and not programs:
        return
    contexts = [p for n in range(3) for p in product(range(4), repeat=n)]
    inputs = torch.tensor([[4] + list(p) + [0] * (2 - len(p)) for p in contexts])
    positions = torch.tensor([len(p) for p in contexts])
    if arm == 'rejection':
        width = max(map(len, programs))
        indices = torch.tensor([[4 * contexts.index(tuple(p[max(0, i - 2):i])) + a
                                 for i, a in enumerate(p)] + [84] * (width - len(p)) for p in programs],
                               dtype=torch.long)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        states, _ = model['recurrent'](F.one_hot(inputs, num_classes=5).to(torch.float64))
        logits = model['head'](states)[torch.arange(21), positions]
        smoothed = .9 * logits.softmax(-1) + .025
        mass = None
        if arm == 'rejection':
            factors = torch.cat([smoothed.log().reshape(-1), torch.zeros(1, dtype=torch.float64)])
            mass = (factors[indices].sum(1).exp() / 10).sum()
            loss = -torch.log1p(-mass)
        else:
            loss = -smoothed.log().mean()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        yield dict(loss_before_step=loss.item(), gradient_norm_before_clip=norm.item(),
            rejection_mass_before_step=mass.item() if mass is not None else None,
            rejected_programs=len(programs) if arm == 'rejection' else 0,
            rejected_tokens=sum(map(len, programs)) if arm == 'rejection' else 0,
            input_rows=21, input_positions=63)


def independent_table(model):
    contexts = [p for n in range(3) for p in product(range(4), repeat=n)]
    inputs = torch.tensor([[4] + list(p) + [0] * (2 - len(p)) for p in contexts])
    with torch.no_grad():
        states, _ = model['recurrent'](F.one_hot(inputs, num_classes=5).to(torch.float64))
        table = model['head'](states)[torch.arange(21), [len(p) for p in contexts]].softmax(-1).tolist()
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    independent = [gru_probabilities(weights, p, {}) for p in contexts]
    np.testing.assert_allclose(table, independent, rtol=1e-12, atol=1e-12)
    return table


def independent_order(table, excluded, budget):
    contexts = [p for n in range(3) for p in product(range(4), repeat=n)]
    edges = {p: [round(-math.log(.9 * probability + .025) * 1000000) for probability in row]
             for p, row in zip(contexts, table)}
    heap, ordered = [(0, ())], []
    used = predictions = positions = children = 0
    peak = 1
    while heap:
        cost, program = heapq.heappop(heap)
        if program not in excluded:
            ordered.append(list(program))
            used += len(program)
            if used > budget:
                break
        if len(program) < 9:
            for token, edge in enumerate(edges[program[-2:]]):
                heapq.heappush(heap, (cost + edge, program + (token,)))
            predictions += 1
            positions += len(program) + 1
            children += 4
            peak = max(peak, len(heap))
    return ordered, dict(candidates=len(ordered), minimum_primitive_ops=used,
        predictions=predictions, prefix_positions=positions, children=children,
        maximum_heap_entries=peak)


def independent_events(primitives, examples, order, state, limit, functions):
    """Use whole-function outcomes and arithmetic to reconstruct partial checks."""
    candidates = iter(order)
    state['status'] = 'running'
    while True:
        if state['pending'] is None:
            if state['primitive_ops'] == limit:
                state['status'] = 'boundary'
                yield dict(status='boundary', pending=None, primitive_ops=limit,
                           comparisons=state['comparisons'])
                return
            program = next(candidates, None)
            if program is None:
                state['status'] = 'space_exhausted'
                yield dict(status='space_exhausted', primitive_ops=state['primitive_ops'],
                           comparisons=state['comparisons'])
                return
            state['candidates'] += 1
            state['pending'] = dict(program=program, example_index=0, operation_index=0, value=examples[0][0])
        pending = state['pending']
        program = tuple(pending['program'])
        if program not in functions:
            outputs = list(range(16))
            for token in program:
                outputs = [primitives[token][value] for value in outputs]
            functions[program] = outputs
        outputs = functions[program]
        start = pending['example_index']
        wrong = next((i for i in range(start, len(examples))
                      if outputs[examples[i][0]] != examples[i][1]), len(examples))
        last = min(wrong, len(examples) - 1)
        required = (last - start + 1) * len(program) - pending['operation_index']
        allowance = limit - state['primitive_ops']
        if required <= allowance:
            state['primitive_ops'] += required
            state['comparisons'] += last - start + 1
            output = outputs[examples[last][0]]
            event = dict(status='rejected' if wrong < len(examples) else 'accepted', program=list(program),
                example_index=last, output=output, target=examples[last][1],
                primitive_ops=state['primitive_ops'], comparisons=state['comparisons'])
            if wrong < len(examples):
                state['rejected'].append(list(program))
                state['pending'] = None
            else:
                pending.update(example_index=last, operation_index=len(program), value=output)
                state['status'] = 'accepted'
            yield event
            if state['status'] == 'accepted':
                return
        else:
            complete, partial = divmod(pending['operation_index'] + allowance, len(program))
            current = start + complete
            value = examples[current][0]
            for token in program[:partial]:
                value = primitives[token][value]
            pending.update(example_index=current, operation_index=partial, value=value)
            state['primitive_ops'] = limit
            state['comparisons'] += complete
            state['status'] = 'boundary'
            yield dict(status='boundary', pending=copy.deepcopy(pending), primitive_ops=limit,
                       comparisons=state['comparisons'])
            return


def audit_run(folder):
    if not __debug__:
        raise RuntimeError('Audit requires assertions; run without -O or PYTHONOPTIMIZE.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = dict(torch=str(torch.__version__), threads=1, device='cpu', dtype='float64',
                   deterministic_algorithms=True)
    assert runtime['torch'] == '2.8.0+cu126'
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert metadata['seeds'] in ([DEVELOPMENT_SEED], list(FORMAL_SEEDS)), 'Incomplete seed grid'
    assert {name: metadata[name] for name in runtime} == runtime
    assert metadata['source_sha256'] == source_hashes()
    assert metadata['data_sha256'] == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert (metadata['budget'], metadata['interval'], metadata['steps_per_update']) == (BUDGET, INTERVAL, STEPS)
    assert metadata['arms'] == list(ARMS)
    assert math.isfinite(metadata['elapsed_seconds']) and metadata['elapsed_seconds'] >= 0
    expected_files = {'metadata.json', 'environments.json', 'active-phase.json',
                      'searches.jsonl', 'evaluations.jsonl', 'phases.jsonl'}
    if metadata['seeds'] == [DEVELOPMENT_SEED]:
        assert metadata['development_audit_sha256'] is None
    else:
        expected_files.add('development-audit.json')
        certificate_bytes = (folder / 'development-audit.json').read_bytes()
        assert hashlib.sha256(certificate_bytes).hexdigest() == metadata['development_audit_sha256']
        certificate = json.loads(certificate_bytes)
        assert certificate['status'] == 'passed' and certificate['seeds'] == [DEVELOPMENT_SEED]
        for name, value in (('source_sha256', metadata['source_sha256']),
                            ('data_sha256', metadata['data_sha256']), ('runtime', runtime)):
            assert certificate[name] == value
    frozen = json.loads(DATA_PATH.read_text())
    assert [e['seed'] for e in frozen] == [DEVELOPMENT_SEED, *FORMAL_SEEDS]
    assert all(len(e['tasks'][family]) == 16 for e in frozen for family in ('composition', 'shifted'))
    if metadata['seeds'] != [DEVELOPMENT_SEED]:
        check_development_evidence(certificate, frozen[0])
    environments = json.loads((folder / 'environments.json').read_text())
    assert environments == [e for e in frozen if e['seed'] in metadata['seeds']]
    assert json.loads((folder / 'active-phase.json').read_text()) == dict(stage='completed')
    raw = {name: (folder / (name + '.jsonl')).read_bytes().splitlines(keepends=True)
           for name in ('searches', 'evaluations', 'phases')}
    rows = [json.loads(line) for line in raw['searches']]
    phases = [json.loads(line) for line in raw['phases']]
    search_records = iter(zip(rows, raw['searches']))
    evaluation_records = iter((json.loads(line), line) for line in raw['evaluations'])
    phase_records = iter(phases)
    all_updates = []
    checkpoint_records = []
    candidate_events = 0

    def phase(expected, step_seconds=0.):
        actual = next(phase_records)
        seconds = actual['seconds']
        assert math.isfinite(seconds) and seconds >= 0
        if 'journal_bytes' in expected:
            journal_seconds = actual['journal_seconds']
            assert math.isfinite(journal_seconds) and journal_seconds >= 0
            assert seconds + 1e-9 >= journal_seconds + step_seconds
            expected = dict(expected, journal_seconds=journal_seconds)
        else:
            assert seconds + 1e-9 >= step_seconds
        assert actual == dict(expected, seconds=seconds), (actual, expected)
        return actual

    def checkpoint(path, model, optimizer, details, key):
        expected_files.add(str(path.relative_to(folder)))
        saved = torch.load(path, weights_only=True)
        compare_tensors(saved, dict(model=model.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else None, **details))
        checkpoint_records.append(dict(file=str(path.relative_to(folder)), **key, steps=details['steps'],
            model_tensor_bytes=sum(v.numel() * v.element_size() for v in saved['model'].values()),
            optimizer_tensor_bytes=sum(v.numel() * v.element_size()
                for state in saved['optimizer']['state'].values() for v in state.values()
                if isinstance(v, torch.Tensor)) if saved['optimizer'] else 0))
        phase(dict(stage='checkpoint', **key, file=path.name, bytes=path.stat().st_size))

    def result_record(result, key, family):
        actual, line = next(search_records)
        assert actual == json.loads(json.dumps(dict(**key, family=family, result=result)))
        phase(dict(stage='result_archive', **key, bytes=len(line)))
        return actual

    def score_record(primitives, task, result, key, row):
        actual, line = next(evaluation_records)
        value = dict(train_correct=0, test_correct=0, exact=False, predictions=None, primitive_ops=0)
        if result['status'] == 'accepted':
            predictions = list(range(16))
            for token in result['candidate_program']:
                predictions = [primitives[token][v] for v in predictions]
            train = sum(predictions[s] == target for s, target in task['train'])
            test = sum(predictions[s] == target for s, target in task['test'])
            value = dict(train_correct=train, test_correct=test, exact=train + test == 16,
                predictions=predictions, primitive_ops=16 * len(result['candidate_program']))
        assert actual == dict(**key, evaluation=value)
        phase(dict(stage='hidden_evaluation', **key, primitive_ops=value['primitive_ops'], journal_bytes=len(line)))
        row['evaluation'] = value

    for environment in environments:
        seed, primitives = environment['seed'], environment['primitives']
        assert environment == json.loads(json.dumps(make_environment(seed))), 'Generator mismatch'
        signatures = set()
        for family, tasks in environment['tasks'].items():
            if family != 'experience':
                assert [len(t['generating_program']) for t in tasks] == [6] * 8 + [9] * 8
            for task in tasks:
                function = list(range(16))
                for token in task['generating_program']:
                    function = [primitives[token][v] for v in function]
                assert len(task['train']) == len(task['test']) == 8
                assert sorted(task['train'] + task['test']) == [[i, v] for i, v in enumerate(function)]
                assert tuple(function) not in signatures
                signatures.add(tuple(function))
        directory = folder / f'seed-{seed}'
        update_path = directory / 'updates.jsonl'
        expected_files.add(str(update_path.relative_to(folder)))
        update_lines = update_path.read_bytes().splitlines(keepends=True)
        update_rows = [json.loads(line) for line in update_lines]
        all_updates.extend(update_rows)
        update_records = iter(zip(update_rows, update_lines))

        def replay(records, key):
            steps, seconds, bytes_used = 0, 0., 0
            for steps, expected in enumerate(records, 1):
                actual, line = next(update_records)
                duration = actual['seconds']
                assert math.isfinite(duration) and duration >= 0
                assert actual == dict(**key, step=steps, seconds=duration, **expected), key
                seconds += duration
                bytes_used += len(line)
            return steps, seconds, bytes_used

        key = dict(seed=seed, arm='initial', task_id=None)
        initial_seed = rng(seed, 'study_l:initial').getrandbits(63)
        torch.manual_seed(initial_seed)
        model = nn.ModuleDict(dict(recurrent=nn.GRU(5, 16, batch_first=True, dtype=torch.float64),
                                   head=nn.Linear(16, 4, dtype=torch.float64)))
        phase(dict(stage='initialization', **key, initial_seed=initial_seed, parameters=1172))
        checkpoint(directory / 'random.pt', model, None, dict(programs=[], accepted_ids=[], steps=0), dict(**key, at=0))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
            weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
        phase(dict(stage='initial_optimizer', **key))
        corpus, accepted_ids, initial_steps = [], [], 0
        functions = {}
        for attempt, task in enumerate(environment['tasks']['experience'], 1):
            key = dict(seed=seed, arm='initial', task_id=task['id'])
            result = reference_experience(primitives, task['train'], [], BUDGET)
            phase(dict(stage='experience_search', **key,
                primitive_ops=result['primitive_ops'], comparisons=result['comparisons']))
            row = result_record(result, key, 'experience')
            if result['status'] == 'accepted' and result['candidate_program']:
                corpus.append(list(result['candidate_program']))
                accepted_ids.append(task['id'])
            count, elapsed, byte_count = replay(independent_initial_steps(model, optimizer, corpus), dict(**key, at=attempt))
            phase(dict(stage='initial_replay', **key, steps=count,
                skip_reason=None if corpus else 'empty_corpus', journal_bytes=byte_count), elapsed)
            initial_steps += count
            checkpoint(directory / f'experience-{attempt}.pt', model, optimizer,
                dict(programs=corpus, accepted_ids=accepted_ids, steps=initial_steps), dict(**key, at=attempt))
            score_record(primitives, task, result, key, row)
        base = copy.deepcopy(model)
        base_table = independent_table(base)
        key = dict(seed=seed, arm='initial', task_id=None, at=0)
        phase(dict(stage='table', **key, model_calls=1, input_rows=21, input_positions=63, probabilities=84))
        path = directory / 'base-table.json'
        expected_files.add(str(path.relative_to(folder)))
        assert json.loads(path.read_text()) == base_table
        phase(dict(stage='table_archive', **key, bytes=path.stat().st_size))
        orders = {}
        for arm in ('uniform', 'frozen'):
            key = dict(seed=seed, arm=arm, task_id=None, at=0)
            orders[arm], detail = independent_order([[.25] * 4] * 21 if arm == 'uniform' else base_table, set(), BUDGET)
            phase(dict(stage='ranking', **key, **detail))
            path = directory / f'{arm}-order.json'
            expected_files.add(str(path.relative_to(folder)))
            assert json.loads(path.read_text()) == orders[arm]
            phase(dict(stage='order_archive', **key, bytes=path.stat().st_size))
        for arm in ARMS:
            adaptive = arm in ('uniformizing', 'rejection')
            for family in ('composition', 'shifted'):
                for task in environment['tasks'][family]:
                    key = dict(seed=seed, arm=arm, task_id=task['id'])
                    task_dir = directory / arm / task['id']
                    if adaptive:
                        model = copy.deepcopy(base)
                        optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
                            weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
                        phase(dict(stage='task_setup', **key))
                        checkpoint(task_dir / 'initial.pt', model, optimizer,
                            dict(primitive_ops=0, rejected=0, steps=0, pending=None), dict(**key, at=0))
                    path = task_dir / 'candidates.jsonl'
                    expected_files.add(str(path.relative_to(folder)))
                    event_lines = path.read_bytes().splitlines(keepends=True)
                    events = iter((json.loads(line), line) for line in event_lines)
                    candidate_events += len(event_lines)
                    state = dict(status='running', primitive_ops=0, comparisons=0, candidates=0, rejected=[], pending=None)
                    changed, steps = False, 0
                    table, table_reference = base_table, 'base-table.json'
                    limits = list(range(INTERVAL, BUDGET, INTERVAL)) + [BUDGET] if adaptive else [BUDGET]
                    for round_index, limit in enumerate(limits):
                        round_key = dict(**key, at=state['primitive_ops'])
                        if adaptive:
                            if changed:
                                table = independent_table(model)
                                phase(dict(stage='table', **round_key,
                                    model_calls=1, input_rows=21, input_positions=63, probabilities=84))
                                table_reference = f'{arm}/{task["id"]}/table-{round_index}.json'
                                path = directory / table_reference
                                expected_files.add(str(path.relative_to(folder)))
                                assert json.loads(path.read_text()) == table
                                phase(dict(stage='table_archive', **round_key, bytes=path.stat().st_size))
                            excluded = {tuple(p) for p in state['rejected']}
                            if state['pending'] is not None:
                                excluded.add(tuple(state['pending']['program']))
                            order, detail = independent_order(table, excluded, limit - state['primitive_ops'])
                            phase(dict(stage='ranking', **round_key, **detail))
                            path = task_dir / f'round-{round_index}.json'
                            expected_files.add(str(path.relative_to(folder)))
                            assert json.loads(path.read_text()) == dict(limit=limit, table=table_reference, programs=order)
                            phase(dict(stage='order_archive', **round_key, bytes=path.stat().st_size))
                        else:
                            order = orders[arm]
                        previous_ops, previous_comparisons = state['primitive_ops'], state['comparisons']
                        byte_count = 0
                        for expected in independent_events(primitives, task['train'], order, state, limit, functions):
                            actual, line = next(events)
                            assert actual == dict(round=round_index, **expected), (key, actual, expected)
                            byte_count += len(line)
                        phase(dict(stage='visible_search', **round_key,
                            primitive_ops=state['primitive_ops'] - previous_ops,
                            comparisons=state['comparisons'] - previous_comparisons, journal_bytes=byte_count))
                        if state['status'] != 'boundary' or limit == BUDGET:
                            break
                        update_key = dict(**key, at=limit)
                        before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                        count, elapsed, byte_count = replay(independent_feedback_steps(model, optimizer, arm,
                                                                                      state['rejected']), update_key)
                        after = torch.cat([p.detach().flatten() for p in model.parameters()])
                        phase(dict(stage='feedback', **update_key, steps=count,
                            rejected=len(state['rejected']), rejected_tokens=sum(map(len, state['rejected'])),
                            skip_reason='empty_rejected_set' if arm == 'rejection' and not state['rejected'] else None,
                            parameter_change_l2=(after - before).norm().item(),
                            changed_parameter_values=int((after != before).sum()), journal_bytes=byte_count), elapsed)
                        steps += count
                        changed = count > 0
                        checkpoint(task_dir / f'after-{limit}.pt', model, optimizer,
                            dict(primitive_ops=limit, rejected=len(state['rejected']), steps=steps,
                                 pending=copy.deepcopy(state['pending'])), update_key)
                    assert next(events, None) is None, 'Extra candidate events'
                    if adaptive:
                        checkpoint(task_dir / 'final.pt', model, optimizer,
                            dict(primitive_ops=state['primitive_ops'], rejected=len(state['rejected']), steps=steps,
                                 pending=copy.deepcopy(state['pending'])), dict(**key, at=state['primitive_ops']))
                    result = dict(status='exhausted' if state['status'] == 'boundary' else state['status'],
                        candidate_program=state['pending']['program'] if state['pending'] else None,
                        primitive_ops=state['primitive_ops'], comparisons=state['comparisons'],
                        candidates=state['candidates'], rejected=len(state['rejected']),
                        pending=state['pending'], feedback_steps=steps)
                    if not adaptive:
                        original = reference_search(primitives, task['train'], [tuple(p) for p in orders[arm]], BUDGET, functions)
                        for name in ('status', 'comparisons', 'primitive_ops', 'rejected'):
                            assert result[name] == original[name]
                        if result['status'] == 'accepted':
                            assert result['candidate_program'] == original['candidate_program']
                    row = result_record(result, key, family)
                    score_record(primitives, task, result, key, row)
        assert next(update_records, None) is None, 'Extra update records'
    assert next(search_records, None) is None, 'Extra search records'
    assert next(evaluation_records, None) is None, 'Extra evaluation records'
    assert next(phase_records, None) is None, 'Extra phase records'
    assert sum(p['seconds'] for p in phases) <= metadata['elapsed_seconds'] + 1e-9, 'Collection shorter than measured work'
    files = sorted(path for path in folder.rglob('*') if path.is_file())
    assert {str(path.relative_to(folder)) for path in files} == expected_files, 'Unexpected or missing archive files'
    certificate = dict(status='passed', seeds=metadata['seeds'], source_sha256=metadata['source_sha256'],
        data_sha256=metadata['data_sha256'], runtime=runtime, searches=len(rows),
        gradient_steps=len(all_updates), candidate_events=candidate_events,
        checkpoints=checkpoint_records,
        files={str(path.relative_to(folder)): dict(bytes=path.stat().st_size,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files})
    return rows, phases, all_updates, metadata, certificate
