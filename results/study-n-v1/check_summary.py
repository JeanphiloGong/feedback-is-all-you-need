#!/usr/bin/env python3
"""Independently verify Study N summaries from the archived evidence.

No experiment module is imported, no world is generated, and no model is
trained or loaded. The pinned remote audit owns gradient/tensor replay.
"""

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev
import sys
import time

if not __debug__:
    raise RuntimeError('Verification refuses optimized Python (-O/-OO).')

import numpy as np
import scipy
from scipy.stats import t


CHECKPOINTS = [0, 16, 32, 64]
ENCODINGS = ['canonical', 'relabeled']
SIZES = {'small': (32, 8, 485), 'large': (128, 2, 1925)}
ARMS = ['small_online', 'large_online', 'small_frozen', 'large_frozen']
GOAL_ARMS = ARMS + ['observed_table']
JOURNALS = ['queries', 'predictions', 'updates', 'phases', 'graphs', 'direct', 'probes', 'reference']
COUNTS = dict(queries=64, gradients=1280, checkpoints=16, direct=20, probes=27776)


def require(condition, location):
    if not condition:
        raise ValueError(location)


def same(actual, expected, location):
    """Compare complete structures, preserving exact integers and booleans."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                f'{location}: different field set')
        for key in expected:
            same(actual[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f'{location}: different list length')
        for index, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f'{location}[{index}]')
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f'{location}: {actual!r} != {expected!r}')
    else:
        require(type(actual) is type(expected) and actual == expected,
                f'{location}: {actual!r} != {expected!r}')


def load(path):
    return json.loads(path.read_text())


def records(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'{path}:{number}: blank journal row')
            yield json.loads(line)


def indexed(rows, fields, expected, location):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f'{location}: duplicate cell {key}')
        result[key] = row
    require(set(result) == set(expected), f'{location}: incomplete or unexpected cell grid')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def actual_checkpoints(arm):
    return [0] if arm.endswith('_frozen') else CHECKPOINTS


def cells(arms):
    return [(encoding, arm, at) for encoding in ENCODINGS for arm in arms
            for at in actual_checkpoints(arm)]


def compare_rows(actual, expected, keys, location):
    wanted = [tuple(row[key] for key in keys) for row in expected]
    found = indexed(actual, keys, wanted, location)
    for row in expected:
        key = tuple(row[field] for field in keys)
        same(found[key], row, f'{location}{key}')


def plan(graph, start, goal):
    """BFS from the protocol, returning the path and counted search work."""
    pending = deque([start])
    paths = {start: []}
    expanded = edges = 0
    while pending:
        current = pending.popleft()
        path = paths[current]
        if len(path) >= 8:
            continue
        expanded += 1
        for action in range(4):
            edges += 1
            following = graph[action][current]
            if following is None or following in paths:
                continue
            new_path = path + [action]
            if following == goal:
                return new_path, edges, expanded
            paths[following] = new_path
            pending.append(following)
    return [], edges, expanded


def fallback_tape(seed):
    tape = {}
    for start in range(32):
        for goal in range(32):
            if start == goal:
                continue
            for step in range(1, 9):
                key = json.dumps([seed, f'study_n:probe:{start}:{goal}:{step}'],
                                 separators=(',', ':')).encode()
                seed_bytes = hashlib.sha256(key).digest()[:8]
                tape[start, goal, step] = random.Random(int.from_bytes(seed_bytes, 'little')).randrange(4)
    return tape


def references(world):
    """Filter each rule against each ordered observation independently."""
    result = []
    for at in CHECKPOINTS:
        candidates = []
        comparisons = 0
        for action in range(4):
            outputs = []
            observations = [(s, world['primitives'][a][s])
                            for s, a in world['queries'][:at] if a == action]
            for out_bit in range(5):
                surviving = []
                for in_bit in range(5):
                    for sign in (0, 1):
                        agrees = True
                        for source, target in observations:
                            comparisons += 1
                            if ((source >> in_bit) & 1) ^ sign != (target >> out_bit) & 1:
                                agrees = False
                                break
                        if agrees:
                            surviving.append([in_bit, sign])
                outputs.append(surviving)
            candidates.append(outputs)
        predictions = []
        for source, action in world['test']:
            choices = [{((source >> bit) & 1) ^ sign for bit, sign in options}
                       for options in candidates[action]]
            predictions.append(sum(next(iter(values)) << bit for bit, values in enumerate(choices))
                               if all(len(values) == 1 for values in choices) else None)
        result.append(dict(at=at, candidates=candidates, comparisons=comparisons,
            predictions=predictions, covered=sum(value is not None for value in predictions),
            exact=sum(value == world['primitives'][a][s]
                      for value, (s, a) in zip(predictions, world['test'])),
            permutation_completion_expectation=1 / (32 - at // 4)))
    return result


def verify_world(world, folder):
    seed = world['seed']
    labels = {'canonical': list(range(32)), 'relabeled': world['labels']}
    require(sorted(world['labels']) == list(range(32)), f'{seed}: labels not bijective')
    queries = world['queries']
    require(len(queries) == 64 and len(set(map(tuple, queries))) == 64, f'{seed}: queries not unique')
    require(all(sorted(a for _, a in queries[i:i + 4]) == list(range(4))
                for i in range(0, 64, 4)), f'{seed}: per-round action balance')
    same(world['test'], [[s, a] for a in range(4) for s in range(32) if [s, a] not in queries],
         f'{seed}: fixed action-major unqueried test set')
    require(len(world['test']) == 64, f'{seed}: test count')
    for action, mapping in enumerate(world['primitives']):
        require(sorted(mapping) == list(range(32)), f'{seed}: action {action} not a permutation')
        permutation = world['bit_permutations'][action]
        require(sorted(permutation) == list(range(5)), f'{seed}: bit permutation')
        same(mapping, [sum(((s >> bit) & 1) << out for out, bit in enumerate(permutation))
                       ^ world['masks'][action] for s in range(32)], f'{seed}: stored physical rule')
    require(len(world['primitives']) == 4 and len(set(map(tuple, world['primitives']))) == 4
            and all(action != list(range(32)) for action in world['primitives']), f'{seed}: action family')

    expected_queries = [dict(at=at, state=s, action=a, next_state=world['primitives'][a][s])
                        for at, (s, a) in enumerate(queries, 1)]
    same(list(records(folder / 'queries.jsonl')), expected_queries, f'{seed}: physical query journal')
    neural_cells, graph_cells = cells(ARMS), cells(GOAL_ARMS)
    graphs = indexed(records(folder / 'graphs.jsonl'), ('encoding', 'arm', 'at'), graph_cells, f'{seed}: graphs')
    direct = indexed(records(folder / 'direct.jsonl'), ('encoding', 'arm', 'at'), neural_cells, f'{seed}: direct')
    direct_counts = {}
    for key, row in graphs.items():
        encoding, arm, at = key
        b = labels[encoding]
        graph = row['graph']
        require(len(graph) == 4 and all(len(action) == 32 for action in graph), f'{seed}: graph dimensions')
        if arm == 'observed_table':
            wanted = [[None] * 32 for _ in range(4)]
            for source, action in queries[:at]:
                wanted[action][b[source]] = b[world['primitives'][action][source]]
            same(graph, wanted, f'{seed}: observed table {key}')
            same([row['input_rows'], row['dense_forward_macs'], row['logits_sha256']], [0, 0, None], f'{seed}: table work')
            require(all(graph[a][b[s]] is None for s, a in world['test']), f'{seed}: table test leakage')
            continue
        width = SIZES[arm.split('_')[0]][0]
        require(all(type(value) is int and 0 <= value < 32 for action in graph for value in action), f'{seed}: neural graph values')
        same([row['input_rows'], row['dense_forward_macs']], [128, 128 * 14 * width], f'{seed}: graph work')
        predictions = [graph[a][b[s]] for s, a in world['test']]
        targets = [b[world['primitives'][a][s]] for s, a in world['test']]
        exact = sum(p == y for p, y in zip(predictions, targets))
        bits = sum(5 - (p ^ y).bit_count() for p, y in zip(predictions, targets))
        observed = sum(graph[a][b[s]] == b[world['primitives'][a][s]] for s, a in queries[:at])
        same(direct[key], dict(encoding=encoding, arm=arm, at=at, predictions=predictions,
             targets=targets, exact=exact, bit_correct=bits, observed_count=at, observed_exact=observed),
             f'{seed}: direct prediction reconstruction {key}')
        direct_counts[key] = (exact, bits, observed)
        if arm.endswith('_frozen'):
            initial = graphs[encoding, arm.replace('_frozen', '_online'), 0]
            same([graph, row['logits_sha256']], [initial['graph'], initial['logits_sha256']], f'{seed}: frozen graph')

    phase_rows = list(records(folder / 'phases.jsonl'))
    phase_keys = [('initialization', None, size, None) for size in SIZES]
    phase_keys += [('branch_setup', e, size + '_online', None) for e in ENCODINGS for size in SIZES]
    phase_keys += [('checkpoint', e, size + '_online', at) for e in ENCODINGS for size in SIZES for at in CHECKPOINTS]
    phase_keys += [('query', None, None, at) for at in range(1, 65)]
    phase_keys += [('observation', e, None, at) for e in ENCODINGS for at in range(1, 65)]
    phase_keys += [(stage, e, size + '_online', at) for stage in ('prediction', 'training')
                   for e in ENCODINGS for size in SIZES for at in range(1, 65)]
    phase_keys += [(stage, *key) for stage in ('evaluation_graph', 'probes') for key in graph_cells]
    phase_keys += [('direct', *key) for key in neural_cells]
    phase_keys += [('symbolic_reference', None, None, at) for at in CHECKPOINTS]
    phase_keys += [('true_distances', None, None, None)]
    phases = indexed([dict(encoding=None, arm=None, at=None) | row for row in phase_rows],
                     ('stage', 'encoding', 'arm', 'at'), phase_keys, f'{seed}: phases')
    require(len(phases) == 807, f'{seed}: phase count')
    for row in phase_rows:
        require(math.isfinite(row['seconds']) and row['seconds'] >= 0, f'{seed}: phase time')
        require(0 <= row.get('journal_seconds', 0) <= row['seconds'], f'{seed}: journal time')

    prediction_keys = [(e, size + '_online', at) for e in ENCODINGS for size in SIZES for at in range(1, 65)]
    predictions = indexed(records(folder / 'predictions.jsonl'), ('encoding', 'arm', 'at'), prediction_keys,
                          f'{seed}: pre-update predictions')
    update_keys = [(e, size + '_online', at, (at - 1) * steps + step)
                  for e in ENCODINGS for size, (_, steps, _) in SIZES.items()
                  for at in range(1, 65) for step in range(1, steps + 1)]
    updates = indexed(records(folder / 'updates.jsonl'), ('encoding', 'arm', 'at', 'gradient_step'),
                      update_keys, f'{seed}: update journal')
    for (e, arm, at, step), row in updates.items():
        require(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
                and row['loss'] >= 0 and row['gradient_norm'] >= 0, f'{seed}: gradient diagnostics')
        same(row['indices_sha256'], updates['canonical', arm, at, step]['indices_sha256'],
             f'{seed}: paired sampled-index digest')
    for (encoding, arm, at), row in predictions.items():
        source, action = queries[at - 1]
        same([row['state'], row['action']], [labels[encoding][source], action], f'{seed}: prediction input')
        require(type(row['prediction']) is int and 0 <= row['prediction'] < 32, f'{seed}: prediction range')
        width, steps, parameters = SIZES[arm.split('_')[0]]
        p = phases['prediction', encoding, arm, at]
        same([p['input_rows'], p['forward_macs']], [1, 14 * width], f'{seed}: prediction work')
        p = phases['training', encoding, arm, at]
        same([p['gradient_steps'], p['input_rows'], p['target_bits'], p['training_macs']],
             [steps, 32 * steps, 160 * steps, 32 * 33 * width * steps], f'{seed}: training work')
        require(0 < p['changed_parameters'] <= parameters and math.isfinite(p['parameter_change_l2'])
                and p['parameter_change_l2'] > 0, f'{seed}: recorded weight change')
    for at in range(1, 65):
        same(phases['query', None, None, at]['environment_actions'], 1, f'{seed}: query work')
    for e in ENCODINGS:
        for size, (_, _, parameters) in SIZES.items():
            for at in CHECKPOINTS:
                p = phases['checkpoint', e, size + '_online', at]
                replay = [[labels[e][s], a, labels[e][world['primitives'][a][s]]] for s, a in queries[:at]]
                same([p['bytes'], p['model_bytes'], p['adam_bytes'], p['replay_json_bytes']],
                     [(folder / e / size / f'{at}.pt').stat().st_size, parameters * 8,
                      0 if at == 0 else parameters * 16 + 16,
                      len(json.dumps(replay, separators=(',', ':')).encode())], f'{seed}: storage {e}/{size}/{at}')
            same(sha256(folder / 'canonical' / size / '0.pt'), sha256(folder / 'relabeled' / size / '0.pt'),
                 f'{seed}: identical initial checkpoint bytes')

    probe_keys = {(e, arm, at, start, goal) for e, arm, at in graph_cells
                  for start in range(32) for goal in range(32) if start != goal}
    seen = set()
    goals = {key: dict(successes=0, actions=0, edges=0, expanded=0) for key in graph_cells}
    plan_cache, initial_probes, table_probes = {}, {}, {}
    tape = fallback_tape(seed)
    for row in records(folder / 'probes.jsonl'):
        e, arm, at, start, goal = key = tuple(row[name] for name in ('encoding', 'arm', 'at', 'start', 'goal'))
        require(key in probe_keys and key not in seen, f'{seed}: invalid or duplicate probe {key}')
        seen.add(key)
        graph_key = (e, arm, at)
        graph = graphs[graph_key]['graph']
        b = labels[e]
        physical = start
        trace = []
        for step in range(1, 9):
            cache_key = (*graph_key, b[physical], b[goal])
            if cache_key not in plan_cache:
                plan_cache[cache_key] = plan(graph, b[physical], b[goal])
            path, edges, expanded = plan_cache[cache_key]
            action = path[0] if path else tape[start, goal, step]
            following = world['primitives'][action][physical]
            trace.append(dict(state=b[physical], next_state=b[following], action=action,
                              planned_path=path, expanded_states=expanded, edge_checks=edges, fallback=not bool(path)))
            physical = following
            if physical == goal:
                break
        expected = dict(encoding=e, arm=arm, at=at, start=start, goal=goal,
                        success=physical == goal, trace=trace, actions=len(trace),
                        edge_checks=sum(step['edge_checks'] for step in trace),
                        expanded_states=sum(step['expanded_states'] for step in trace))
        same(row, expected, f'{seed}: probe {key}')
        totals = goals[graph_key]
        totals['successes'] += physical == goal
        totals['actions'] += len(trace)
        totals['edges'] += expected['edge_checks']
        totals['expanded'] += expected['expanded_states']
        if arm != 'observed_table' and at == 0:
            fingerprint = json.dumps({k: v for k, v in row.items() if k != 'arm'}, sort_keys=True)
            initial_key = (e, arm.split('_')[0], start, goal)
            if arm.endswith('_frozen'):
                same(fingerprint, initial_probes[initial_key], f'{seed}: frozen probe reuse')
            else:
                initial_probes[initial_key] = fingerprint
        if arm == 'observed_table':
            inverse = {label: state for state, label in enumerate(b)}
            normalized = dict(success=row['success'], trace=[dict(step, state=inverse[step['state']],
                              next_state=inverse[step['next_state']]) for step in row['trace']])
            table_key = (at, start, goal)
            if e == 'canonical':
                table_probes[table_key] = normalized
            else:
                same(normalized, table_probes[table_key], f'{seed}: table encoding equivariance')
    require(seen == probe_keys, f'{seed}: incomplete probe grid')
    for key, counts in goals.items():
        p = phases[('probes', *key)]
        same([p['tasks'], p['successes'], p['environment_actions'], p['edge_checks'], p['expanded_states']],
             [992, counts['successes'], counts['actions'], counts['edges'], counts['expanded']], f'{seed}: probe phase totals')
        g = graphs[key]
        p = phases[('evaluation_graph', *key)]
        same([p['input_rows'], p['dense_forward_macs'], p['logits_sha256']],
             [g['input_rows'], g['dense_forward_macs'], g['logits_sha256']], f'{seed}: graph phase totals')
    for key in neural_cells:
        same(phases[('direct', *key)]['tests'], 64, f'{seed}: direct phase count')

    expected_references = references(world)
    compare_rows(list(records(folder / 'reference.jsonl')), expected_references, ('at',), f'{seed}: references')
    for row in expected_references:
        same(phases['symbolic_reference', None, None, row['at']]['comparisons'], row['comparisons'], f'{seed}: reference work')

    distances = indexed(load(folder / 'distances.json'), ('encoding', 'start'),
                        [(e, start) for e in ENCODINGS for start in range(32)], f'{seed}: distance grid')
    all_distances = []
    for start in range(32):
        depth = {start: 0}
        pending = deque([start])
        while pending:
            current = pending.popleft()
            for action in world['primitives']:
                following = action[current]
                if following not in depth:
                    depth[following] = depth[current] + 1
                    pending.append(following)
        require(len(depth) == 32, f'{seed}: world not strongly connected')
        for e in ENCODINGS:
            same(distances[e, start], dict(encoding=e, start=start, distances=[depth[g] for g in range(32)]), f'{seed}: true distances')
        all_distances += [depth[g] for g in range(32) if g != start]

    curves = []
    for e in ENCODINGS:
        for arm in ARMS:
            points = [direct_counts[e, arm, 0 if arm.endswith('_frozen') else at] for at in CHECKPOINTS]
            exact = [point[0] / 64 for point in points]
            curves.append(dict(seed=seed, encoding=e, arm=arm, checkpoints=CHECKPOINTS,
                exact_rates=exact, bit_rates=[point[1] / 320 for point in points],
                observed_rates=[point[2] / at if at and not arm.endswith('_frozen') else None
                                for point, at in zip(points, CHECKPOINTS)],
                final_minus_initial=exact[-1] - exact[0], tests_per_checkpoint=64,
                newly_evaluated_pairs=64 * len(actual_checkpoints(arm))))
    goal_curves = [dict(seed=seed, encoding=e, arm=arm, checkpoints=CHECKPOINTS,
        success_rates=[goals[e, arm, 0 if arm.endswith('_frozen') else at]['successes'] / 992 for at in CHECKPOINTS],
        tasks_per_checkpoint=992, newly_evaluated_tasks=992 * len(actual_checkpoints(arm)))
        for e in ENCODINGS for arm in GOAL_ARMS]
    reference_curves = [dict(seed=seed, at=row['at'], exact_rate=row['exact'] / 64,
        coverage=row['covered'] / 64, comparisons=row['comparisons'],
        surviving_rules=sum(len(options) for action in row['candidates'] for options in action),
        permutation_completion_expectation=row['permutation_completion_expectation']) for row in expected_references]
    bound = dict(seed=seed, tasks=992, maximum_distance=max(all_distances),
                 reachable_within_eight=sum(d <= 8 for d in all_distances))
    costs = []
    for size, (width, steps, parameters) in SIZES.items():
        for e in ENCODINGS:
            selected = [p for p in phase_rows if
                (p['stage'] == 'query') or
                (p['stage'] == 'initialization' and p['arm'] == size) or
                (p['stage'] == 'observation' and p['encoding'] == e) or
                (p['stage'] in ('branch_setup', 'prediction', 'training')
                 and p['encoding'] == e and p['arm'] == size + '_online')]
            seconds = sum(p['seconds'] for p in selected)
            journals = sum(p.get('journal_seconds', 0) for p in selected)
            replay = [[labels[e][s], a, labels[e][world['primitives'][a][s]]] for s, a in queries]
            costs.append(dict(seed=seed, encoding=e, arm=size + '_online', environment_actions=64,
                gradient_steps=64 * steps, training_macs=64 * steps * 32 * 33 * width,
                forward_macs=64 * 14 * width, target_bits=64 * steps * 160,
                training_rows=64 * steps * 32, prediction_rows=64, update_batches=64,
                batches_changing_weights=sum(p['changed_parameters'] > 0 for p in selected if p['stage'] == 'training'),
                instrumented_phase_seconds=seconds, included_journal_seconds=journals,
                seconds_excluding_journal_io=seconds - journals, model_bytes=parameters * 8,
                adam_bytes=parameters * 16 + 16, replay_json_bytes=len(json.dumps(replay, separators=(',', ':')).encode())))
    return curves, goal_curves, reference_curves, bound, costs, phase_rows, goals


def verify(raw, summary_path):
    started = time.monotonic()
    summary = load(summary_path)
    audit_path = summary_path.parent / 'audit.json'
    audit = load(audit_path)
    metadata = load(raw / 'metadata.json')
    require(type(metadata['development']) is bool, 'metadata: development flag')
    seeds = [47] if metadata['development'] else list(range(16000, 16032))
    for location, value in [('metadata', metadata), ('audit', audit), ('summary', summary)]:
        same(value['seeds'], seeds, f'{location}: complete seed grid')
    same(metadata['status'], 'completed', 'metadata status')
    same(audit['status'], 'passed', 'remote audit status')
    same(load(raw / 'active-phase.json'), {'stage': 'completed'}, 'completed phase')
    same([metadata['runtime'], metadata['device'], metadata['dtype'], metadata['threads'], metadata['deterministic_algorithms']],
         ['2.8.0+cu126', 'cpu', 'float64', 1, True], 'recorded collection runtime')
    for field in ('source_sha256', 'source_revision', 'data_sha256', 'runtime'):
        same(metadata[field], audit[field], f'audit provenance {field}')
    files = {'metadata.json', 'worlds.json', 'active-phase.json'}
    if not metadata['development']:
        files.add('development-audit.json')
        same(sha256(raw / 'development-audit.json'), metadata['development_audit_sha256'], 'development gate certificate hash')
        gate = load(raw / 'development-audit.json')
        same([gate['status'], gate['seeds'], gate['source_sha256'], gate['data_sha256']],
             ['passed', [47], audit['source_sha256'], audit['data_sha256']], 'development gate provenance')
    for seed in seeds:
        prefix = f'seed-{seed}/'
        files.update(prefix + name + '.jsonl' for name in JOURNALS)
        files.add(prefix + 'distances.json')
        files.update(f'{prefix}{e}/{size}/{at}.pt' for e in ENCODINGS for size in SIZES for at in CHECKPOINTS)
    same(set(p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()), files, 'raw file grid')
    same(set(audit['files']), files, 'audit file grid')
    before = {name: sha256(raw / name) for name in sorted(files)}
    same(before, audit['files'], 'raw hashes before verification')
    same(audit['counts'], {name: count * len(seeds) for name, count in COUNTS.items()}, 'remote audit counts')
    worlds = load(raw / 'worlds.json')
    same([world['seed'] for world in worlds], seeds, 'world grid')

    curves, goals, refs, bounds, costs, phases = [], [], [], [], [], []
    probe_actions = probe_edges = probe_expanded = 0
    for world in worlds:
        c, g, r, b, k, p, probe_totals = verify_world(world, raw / f'seed-{world["seed"]}')
        curves.extend(c)
        goals.extend(g)
        refs.extend(r)
        bounds.append(b)
        costs.extend(k)
        phases.extend(p)
        probe_actions += sum(row['actions'] for row in probe_totals.values())
        probe_edges += sum(row['edges'] for row in probe_totals.values())
        probe_expanded += sum(row['expanded'] for row in probe_totals.values())
        print(f'Verified seed {world["seed"]}: complete raw grids, predictions, traces, references, costs.', flush=True)

    contrasts = []
    for seed in seeds:
        for size in SIZES:
            c = next(row for row in curves if (row['seed'], row['encoding'], row['arm']) == (seed, 'canonical', size + '_online'))
            r = next(row for row in curves if (row['seed'], row['encoding'], row['arm']) == (seed, 'relabeled', size + '_online'))
            contrasts.append(dict(seed=seed, size=size, canonical_gain=c['final_minus_initial'],
                relabeled_gain=r['final_minus_initial'], canonical_final=c['exact_rates'][-1], relabeled_final=r['exact_rates'][-1],
                final_difference=c['exact_rates'][-1] - r['exact_rates'][-1],
                gain_difference=c['final_minus_initial'] - r['final_minus_initial']))
    contrast_fields = ['canonical_gain', 'relabeled_gain', 'canonical_final', 'relabeled_final', 'final_difference', 'gain_difference']
    aggregate_contrasts = [dict(size=size, **{field: mean(row[field] for row in contrasts if row['size'] == size)
                                           for field in contrast_fields}) for size in SIZES]
    differences = [row['gain_difference'] for row in contrasts if row['size'] == 'small']
    primary = dict(differences=differences, environments=len(seeds), mean=mean(differences),
        confirmatory=not metadata['development'], ci95=None, bootstrap95=None, positive_primary=False,
        negative=sum(value < 0 for value in differences), zero=sum(value == 0 for value in differences),
        positive=sum(value > 0 for value in differences),
        estimand='small canonical-minus-relabeled difference in exact-accuracy learning gains',
        **{field: aggregate_contrasts[0][field] for field in contrast_fields})
    if not metadata['development']:
        standard_error = stdev(differences) / math.sqrt(len(seeds))
        half = float(t.ppf(0.975, len(seeds) - 1)) * standard_error
        interval = [primary['mean'] - half, primary['mean'] + half]
        rng = np.random.default_rng(991)
        draws = rng.integers(0, len(seeds), size=(10000, len(seeds)))
        means = np.asarray(differences)[draws].mean(axis=1)
        boot = np.quantile(means, [0.025, 0.975]).tolist()
        primary.update(ci95=interval, bootstrap95=boot, positive_primary=interval[0] > 0,
                       degenerate=standard_error == 0,
                       interval_disagreement=(interval[0] > 0) != (boot[0] > 0) or (interval[1] < 0) != (boot[1] < 0))
    direct_summaries, goal_summaries = [], []
    for e in ENCODINGS:
        for arm in ARMS:
            rows = [row for row in curves if (row['encoding'], row['arm']) == (e, arm)]
            values = {field: [None if rows[0][field][i] is None else mean(row[field][i] for row in rows)
                              for i in range(4)] for field in ('exact_rates', 'bit_rates', 'observed_rates')}
            direct_summaries.append(dict(encoding=e, arm=arm, environments=len(seeds), **values,
                final_minus_initial=mean(row['final_minus_initial'] for row in rows)))
        for arm in GOAL_ARMS:
            rows = [row for row in goals if (row['encoding'], row['arm']) == (e, arm)]
            goal_summaries.append(dict(encoding=e, arm=arm,
                success_rates=[mean(row['success_rates'][i] for row in rows) for i in range(4)]))
    symbolic = [dict(at=at, **{field: mean(row[field] for row in refs if row['at'] == at)
                for field in ('exact_rate', 'coverage', 'comparisons', 'surviving_rules', 'permutation_completion_expectation')})
                for at in CHECKPOINTS]
    phase_seconds = {stage: sum(row['seconds'] for row in phases if row['stage'] == stage)
                     for stage in sorted({row['stage'] for row in phases})}
    actual = dict(**{name: count * len(seeds) for name, count in COUNTS.items()},
        collection_seconds=metadata['elapsed_seconds'], environment_actions=64 * len(seeds) + probe_actions,
        training_macs=sum(row['training_macs'] for row in costs), forward_macs=sum(row['forward_macs'] for row in costs),
        dense_forward_macs=len(seeds) * 2 * 5 * 128 * 14 * (32 + 128),
        edge_checks=probe_edges, expanded_states=probe_expanded,
        target_bits=sum(row['target_bits'] for row in costs), prediction_rows=len(seeds) * 256,
        training_rows=sum(row['training_rows'] for row in costs), evaluation_graph_rows=len(seeds) * 20 * 128,
        update_batches=len(seeds) * 256, batches_changing_weights=sum(row['batches_changing_weights'] for row in costs),
        checkpoint_file_bytes=sum((raw / name).stat().st_size for name in files if name.endswith('.pt')),
        known_learning_journal_seconds=sum(row.get('journal_seconds', 0) for row in phases), phase_seconds=phase_seconds)
    expected = dict(status='audited encoding and unqueried-transition learning', seeds=seeds,
        checkpoints=CHECKPOINTS, primary=primary, direct_summaries=direct_summaries, goal_summaries=goal_summaries,
        symbolic_reference=symbolic, observed_table_unqueried_exact_rates=[0.0] * 4,
        descriptive_contrasts=aggregate_contrasts, seed_contrasts=contrasts, seed_direct_curves=curves,
        seed_goal_curves=goals, seed_references=refs, true_graph_bounds=bounds,
        standalone_learning_costs=costs, actual_cost=actual)
    informational = {'statistics_packages', 'interpretation_notes', 'analysis_elapsed_seconds'}
    same(set(summary), set(expected) | informational, 'summary field grid')
    row_keys = {'direct_summaries': ('encoding', 'arm'), 'goal_summaries': ('encoding', 'arm'),
        'symbolic_reference': ('at',), 'descriptive_contrasts': ('size',), 'seed_contrasts': ('seed', 'size'),
        'seed_direct_curves': ('seed', 'encoding', 'arm'), 'seed_goal_curves': ('seed', 'encoding', 'arm'),
        'seed_references': ('seed', 'at'), 'true_graph_bounds': ('seed',),
        'standalone_learning_costs': ('seed', 'encoding', 'arm')}
    for field, value in expected.items():
        if field in row_keys:
            compare_rows(summary[field], value, row_keys[field], f'summary.{field}')
        else:
            same(summary[field], value, f'summary.{field}')
    same({name: sha256(raw / name) for name in sorted(files)}, before, 'raw hashes after verification')
    return dict(status='passed', scope='Independent local reconstruction of archived Study N numerical summaries; no local pinned gradient/tensor replay.',
        verified=['Exact raw file grids and audit-linked SHA-256 before and after verification.',
            'Complete seed, encoding, branch, checkpoint, gradient-journal, prediction, phase, and probe grids.',
            'Stored world rules, encoded query mapping, fixed unqueried split, and observed-table graphs.',
            'Direct exact/bit/observed scores from graph predictions and physical targets; frozen reuse.',
            'Every goal trace, seeded fallback, BFS path and operation count; table equivariance and true distances.',
            'Signed-coordinate candidates, predictions, work counts, and exact permutation-completion expectations.',
            'All seed curves, aggregate curves, contrasts, formal world-paired t95 and 10000-bootstrap seed 991 decisions.',
            'All reported actual and standalone learning costs, recorded phase sums, and storage accounting.'],
        not_reexecuted=['World random generation and rejection history.',
            'Model forwards, logits, gradients, Adam updates, tensor/corpus checkpoint contents, and tensor-change norms.',
            'Wall-time measurements and reported package versions in the remote run.'],
        remote_replay_certificate=str(audit_path), remote_replay_certificate_sha256=sha256(audit_path),
        raw_directory=str(raw), raw_file_count=len(files), summary=str(summary_path), summary_sha256=sha256(summary_path),
        verifier_sha256=sha256(Path(__file__)), seeds=seeds,
        counts={name: actual[name] for name in COUNTS}, actual_cost=actual,
        primary=primary, verified_summary_fields=sorted(expected), informational_summary_fields=sorted(informational),
        local_runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        cross_model_review='Skipped in autonomous context; no external CLI invoked.',
        elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f'Output already exists: {args.output}')
    require(not args.output.resolve().is_relative_to(args.raw.resolve()), 'Output must be outside the immutable raw directory.')
    report = verify(args.raw, args.summary)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Passed; verification certificate: {args.output}')


if __name__ == '__main__':
    main()
