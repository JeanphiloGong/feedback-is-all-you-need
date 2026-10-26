"""Replay action feedback, gradients, Adam state, probes, and raw-file costs."""

import copy
import hashlib
import json
import math

import torch

from experiments.study_m.replay import (independent_graph, independent_initial,
    independent_replay, reference_policy, reference_rng, reference_world)
from experiments.study_m.run import (ARMS, CHECKPOINTS, DATA_PATH, INTERACTIONS,
    check_development_evidence, required_files, source_hashes)
from experiments.study_m.tasks import DEVELOPMENT_SEED, FAMILIES, FORMAL_SEEDS


def read_records(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert raw == ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode(), path
    return rows


def read_document(path, indent):
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, indent=indent, allow_nan=False) + '\n').encode(), path
    return value


def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert (actual.dtype, actual.device, actual.layout) == (expected.dtype, expected.device, expected.layout)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for name in expected:
            same(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same(a, b)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def audit_world(world, folder, all_phases):
    seed, family = world['seed'], world['family']
    selected = [p for p in all_phases if (p['seed'], p['family']) == (seed, family)]
    phases = iter(selected)
    actions = iter(read_records(folder / 'actions.jsonl'))
    updates = iter(read_records(folder / 'updates.jsonl'))
    graphs = iter(read_records(folder / 'graphs.jsonl'))
    probes = iter(read_records(folder / 'probes.jsonl'))
    oracles = iter(read_records(folder / 'oracle.jsonl'))
    totals = dict(main_actions=0, gradient_steps=0, checkpoints=0, probe_tasks=0)
    checkpoint_records = []

    def phase(stage, key, **expected):
        row = next(phases)
        timing = {'seconds', 'journal_seconds'}
        if stage == 'action':
            timing.update(('planning_seconds', 'environment_seconds', 'observation_seconds'))
        required_timing = timing if stage in ('warmup', 'training', 'graph', 'action', 'oracle', 'probe') else {'seconds'}
        assert set(row) & timing == required_timing, (stage, set(row) & timing)
        same({k: v for k, v in row.items() if k not in timing}, dict(stage=stage, **key, **expected))
        assert math.isfinite(row['seconds']) and row['seconds'] >= 0
        for name in timing - {'seconds'}:
            if name in row:
                assert math.isfinite(row[name]) and 0 <= row[name] <= row['seconds']
        if stage == 'action':
            assert sum(row[name] for name in timing - {'seconds'}) <= row['seconds'] + 1e-9
        return row

    def checkpoint(name, parameters, optimizer, corpus, graph, steps, episode, episode_step, state, key):
        path = folder / name
        saved = torch.load(path, weights_only=True)
        names = ('input_layer.weight', 'input_layer.bias', 'output_layer.weight', 'output_layer.bias')
        tensors = dict(zip(names, parameters)) if parameters is not None else None
        observed = [[s, a, corpus[s, a]] for s, a in sorted(corpus)]
        storage = dict(model_tensor_bytes=sum(p.numel() * p.element_size() for p in parameters)
                       if parameters is not None else 0,
            adam_tensor_bytes=sum(v.numel() * v.element_size() for s in optimizer.state.values()
                for v in s.values() if isinstance(v, torch.Tensor)) if optimizer else 0,
            replay_entries=len(corpus), replay_json_bytes=len(json.dumps(observed).encode()),
            graph_json_bytes=len(json.dumps(graph).encode()))
        same(saved, dict(**key, model=tensors, optimizer=optimizer.state_dict() if optimizer else None,
            corpus=observed, graph=graph, gradient_steps=steps, episode_index=episode,
            episode_step=episode_step, current_state=state, storage=storage))
        phase('checkpoint', key, file=path.name, bytes=path.stat().st_size, **storage)
        totals['checkpoints'] += 1
        checkpoint_records.append(dict(file=f'seed-{seed}/{family}/{name}', **key,
                                       gradient_steps=steps, **storage))

    def training(parameters, optimizer, corpus, first_step, count, key):
        before = torch.cat([p.detach().flatten() for p in parameters]).clone()
        byte_count, step_seconds = 0, 0.
        for expected in independent_replay(parameters, optimizer, corpus, seed, family, first_step, count):
            actual = next(updates)
            seconds = actual['step_seconds']
            assert math.isfinite(seconds) and seconds >= 0
            same({k: v for k, v in actual.items() if k != 'step_seconds'}, dict(**key, **expected))
            byte_count += len((json.dumps(actual, allow_nan=False) + '\n').encode())
            step_seconds += seconds
        after = torch.cat([p.detach().flatten() for p in parameters])
        row = phase('training', key, gradient_steps=count, input_rows=count * 32, target_bits=count * 160,
            dense_training_macs=count * 32 * 33 * len(parameters[1]), corpus_entries=len(corpus),
            parameter_change_l2=(after - before).norm().item(), changed_parameter_values=int((after != before).sum()),
            journal_bytes=byte_count)
        assert step_seconds + row['journal_seconds'] <= row['seconds'] + 1e-9
        totals['gradient_steps'] += count

    def graph_record(parameters, graph, key):
        detail = dict(input_rows=0, dense_forward_macs=0, logits_sha256=None)
        if parameters is not None:
            graph, detail = independent_graph(parameters)
        expected = dict(**key, graph=graph, **detail)
        same(next(graphs), expected)
        phase('graph', key, journal_bytes=len((json.dumps(expected, allow_nan=False) + '\n').encode()), **detail)
        return graph

    corpus, observed, state = {}, [], world['warmup']['start']
    for action in world['warmup']['actions']:
        following = world['primitives'][action][state]
        observed.append(dict(state=state, action=action, next_state=following))
        corpus[state, action] = following
        state = following
    same(json.loads((folder / 'warmup.json').read_bytes()), observed)
    assert (folder / 'warmup.json').read_bytes() == (json.dumps(observed) + '\n').encode()
    key = dict(seed=seed, family=family, arm='warmup', at=0)
    phase('warmup', key, environment_actions=len(observed), corpus_entries=len(corpus),
          journal_bytes=len((json.dumps(observed) + '\n').encode()))
    initial = {}
    for size, hidden, steps in (('small', 32, 128), ('large', 128, 32)):
        key = dict(seed=seed, family=family, arm=f'{size}_initial', at=0)
        parameters, optimizer = independent_initial(seed, family, hidden)
        phase('initialization', key, hidden=hidden, parameters=15 * hidden + 5)
        checkpoint(f'random-{size}.pt', parameters, optimizer, corpus, None, 0, -1, 0, state, key)
        training(parameters, optimizer, corpus, 1, steps, key)
        graph = graph_record(parameters, None, key)
        checkpoint(f'initial-{size}.pt', parameters, optimizer, corpus, graph, steps, -1, 0, state, key)
        initial[size] = parameters, optimizer, graph, steps
    for arm in ARMS:
        key = dict(seed=seed, family=family, arm=arm, at=0)
        parameters = optimizer = graph = None
        steps = 0
        observed = dict(corpus) if arm != 'uniform' else {}
        size = arm.split('_')[0]
        if size in initial:
            original, original_optimizer, graph, steps = initial[size]
            parameters = [torch.nn.Parameter(p.detach().clone()) for p in original]
            optimizer = torch.optim.Adam(parameters, lr=.01, foreach=False, fused=False)
            optimizer.load_state_dict(copy.deepcopy(original_optimizer.state_dict()))
            graph = copy.deepcopy(graph)
        elif arm == 'observed_table':
            graph = [[observed.get((s, a)) for s in range(32)] for a in range(4)]
        phase('branch_setup', key)
        graph_record(None, graph, key)
        episode = episode_step = graph_at = 0
        state, goal = world['episodes'][0]['start'], world['episodes'][0]['goal']
        checkpoint(f'{arm}/0.pt', parameters, optimizer, observed, graph, steps, episode, episode_step, state, key)
        for at in range(1, INTERACTIONS + 1):
            key['at'] = at
            fallback = reference_rng(seed, f'study_m:main:{family}:{at}').randrange(4)
            decision = reference_policy(graph, state, goal, fallback)
            action = decision['action']
            predicted = graph[action][state] if graph is not None else None
            following = world['primitives'][action][state]
            added = arm != 'uniform' and (state, action) not in observed
            if arm != 'uniform':
                observed[state, action] = following
            if arm == 'observed_table':
                graph[action][state] = following
            episode_step += 1
            success = following == goal
            ended = success or episode_step == 16
            expected = dict(**key, episode_index=episode, episode_step=episode_step, state=state, goal=goal,
                next_state=following, predicted_next_state=predicted, graph_at=graph_at,
                fallback_action=fallback, **decision, corpus_added=added, corpus_entries=len(observed),
                episode_success=success, episode_ended=ended,
                bit_errors=(predicted ^ following).bit_count() if predicted is not None else None)
            same(next(actions), expected)
            phase('action', key, journal_bytes=len((json.dumps(expected) + '\n').encode()),
                environment_actions=1, expanded_states=decision['expanded_states'], edge_checks=decision['edge_checks'])
            totals['main_actions'] += 1
            if arm.endswith('immediate') or (arm.endswith('delayed') and at % 16 == 0):
                count = (8 if size == 'small' else 2) * (16 if arm.endswith('delayed') else 1)
                training(parameters, optimizer, observed, steps + 1, count, key)
                steps += count
                graph = graph_record(parameters, graph, key)
                graph_at = at
            elif arm == 'observed_table':
                graph_record(None, graph, key)
                graph_at = at
            state = following
            if ended:
                episode += 1
                episode_step = 0
                state, goal = world['episodes'][episode]['start'], world['episodes'][episode]['goal']
            if at in CHECKPOINTS:
                checkpoint(f'{arm}/{at}.pt', parameters, optimizer, observed, graph, steps,
                           episode, episode_step, state, key)
    for task in world['probes']:
        key = dict(seed=seed, family=family, arm='oracle', at=0, **task)
        distances = [999] * 32
        distances[task['goal']] = 0
        for _ in range(31):
            distances = [min([distances[s]] + [1 + distances[a[s]] for a in world['primitives']])
                         for s in range(32)]
        distance = distances[task['start']]
        state, path = task['start'], []
        while state != task['goal']:
            action = next(a for a in range(4) if distances[world['primitives'][a][state]] == distances[state] - 1)
            path.append(action)
            state = world['primitives'][action][state]
        queue, seen, edges = [task['start']], {task['start']}, 0
        found = False
        for state in queue:
            for action in world['primitives']:
                edges += 1
                following = action[state]
                if following in seen:
                    continue
                if following == task['goal']:
                    found = True
                    break
                seen.add(following)
                queue.append(following)
            if found:
                break
        expected = dict(**task, shortest_path=path, distance=distance,
                        reachable_within_budget=distance <= 8, edge_checks=edges)
        same(next(oracles), expected)
        phase('oracle', key, journal_bytes=len((json.dumps(expected) + '\n').encode()), edge_checks=edges)
    for arm in ARMS:
        points = (0,) if arm.endswith('frozen') or arm == 'uniform' else CHECKPOINTS
        for at in points:
            key = dict(seed=seed, family=family, arm=arm, at=at)
            phase('probe_checkpoint_load', key)
            # The checkpoint was already independently reconstructed above.
            graph = torch.load(folder / arm / f'{at}.pt', weights_only=True)['graph']
            for task in world['probes']:
                state, trace = task['start'], []
                for step in range(1, 9):
                    fallback = reference_rng(seed, f'study_m:probe:{family}:{task["start"]}:{task["goal"]}:{step}').randrange(4)
                    decision = reference_policy(graph, state, task['goal'], fallback)
                    action = decision['action']
                    following = world['primitives'][action][state]
                    predicted = graph[action][state] if graph is not None else None
                    trace.append(dict(step=step, state=state, next_state=following,
                        predicted_next_state=predicted, fallback_action=fallback, **decision,
                        bit_errors=(predicted ^ following).bit_count() if predicted is not None else None))
                    state = following
                    if state == task['goal']:
                        break
                expected = dict(**key, **task, success=state == task['goal'], actions=len(trace), trace=trace)
                same(next(probes), expected)
                phase('probe', dict(**key, **task), journal_bytes=len((json.dumps(expected) + '\n').encode()),
                    environment_actions=len(trace), edge_checks=sum(r['edge_checks'] for r in trace),
                    expanded_states=sum(r['expanded_states'] for r in trace))
                totals['probe_tasks'] += 1
    end = object()
    for records in (phases, actions, updates, graphs, probes, oracles):
        assert next(records, end) is end, 'Unexpected additional record'
    return dict(**totals, checkpoint_records=checkpoint_records)


def audit_run(root):
    if not __debug__:
        raise RuntimeError('Study M audit requires Python assertions; remove -O or PYTHONOPTIMIZE.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = read_document(root / 'metadata.json', 2)
    runtime = dict(torch=str(torch.__version__), threads=torch.get_num_threads(), device='cpu',
                   dtype='float64', deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    assert runtime['torch'] == '2.8.0+cu126'
    same({k: metadata[k] for k in runtime}, runtime)
    assert metadata['status'] == 'completed'
    seeds = metadata['seeds']
    assert seeds in ([DEVELOPMENT_SEED], list(FORMAL_SEEDS))
    hashes = source_hashes()
    same(metadata['source_sha256'], hashes)
    data = DATA_PATH.read_bytes()
    same(metadata['data_sha256'], hashlib.sha256(data).hexdigest())
    frozen = read_document(DATA_PATH, 2)
    same([[w['seed'], w['family']] for w in frozen],
         [[s, f] for s in (DEVELOPMENT_SEED,) + FORMAL_SEEDS for f in FAMILIES])
    worlds = [w for w in frozen if w['seed'] in seeds]
    same(read_document(root / 'worlds.json', 2), worlds)
    same(read_document(root / 'active-phase.json', None), dict(stage='completed'))
    same({name: metadata[name] for name in ('arms', 'checkpoints', 'interactions', 'families')},
         dict(arms=list(ARMS), checkpoints=list(CHECKPOINTS), interactions=INTERACTIONS, families=list(FAMILIES)))
    required = required_files(seeds)
    if seeds != [DEVELOPMENT_SEED]:
        required.add('development-audit.json')
        certificate = (root / 'development-audit.json').read_bytes()
        same(metadata['development_audit_sha256'], hashlib.sha256(certificate).hexdigest())
        development = read_document(root / 'development-audit.json', 2)
        same(development['status'], 'passed')
        same(development['seeds'], [DEVELOPMENT_SEED])
        same(development['runtime'], runtime)
        same(development['source_sha256'], hashes)
        same(development['data_sha256'], metadata['data_sha256'])
        check_development_evidence(development)
    else:
        assert metadata['development_audit_sha256'] is None
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.rglob('*') if path.is_file()}
    assert set(files) == required, (set(files) - required, required - set(files))
    phases = read_records(root / 'phases.jsonl')
    valid_worlds = {(w['seed'], w['family']) for w in worlds}
    assert all((p['seed'], p['family']) in valid_worlds for p in phases)
    # Global order is collection for every world, followed by evaluations only.
    evaluation = False
    stream_order, probe_order = [], []
    for p in phases:
        if p['stage'] in ('oracle', 'probe_checkpoint_load', 'probe'):
            evaluation = True
            order = probe_order
        else:
            assert not evaluation, 'Interaction followed evaluation'
            order = stream_order
        identity = (p['seed'], p['family'])
        if not order or order[-1] != identity:
            order.append(identity)
    same(stream_order, [(w['seed'], w['family']) for w in worlds])
    same(probe_order, stream_order)
    totals = dict(main_actions=0, gradient_steps=0, checkpoints=0, probe_tasks=0)
    checkpoints = []
    for world in worlds:
        same(world, reference_world(world['seed'], world['family']))
        folder = root / f'seed-{world["seed"]}' / world['family']
        result = audit_world(world, folder, phases)
        for name in totals:
            totals[name] += result[name]
        checkpoints.extend(result['checkpoint_records'])
    assert math.isfinite(metadata['elapsed_seconds'])
    assert sum(p['seconds'] for p in phases) <= metadata['elapsed_seconds'] + 1e-9
    same({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in root.rglob('*') if path.is_file()}, files)
    certificate = dict(status='passed', seeds=seeds, runtime=runtime, source_sha256=hashes,
        data_sha256=metadata['data_sha256'], files=files, **totals, checkpoint_records=checkpoints,
        numerical_policy='Exact Torch tensors/journals/digests; NumPy forward rtol=1e-11, atol=1e-12.')
    if seeds == [DEVELOPMENT_SEED]:
        check_development_evidence(certificate)
    return worlds, phases, metadata, certificate
