"""Reconstruct every physical observation, update, saved state, and probe."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.study_m.replay import independent_graph, reference_policy, reference_rng
from experiments.study_n.replay import (reference_initial, reference_inputs,
    reference_logits, reference_updates, reference_world)
from experiments.study_n.run import (CHECKPOINTS, DATA, DEVELOPMENT_SEED, ENCODINGS, FORMAL_SEEDS,
    JOURNALS, MANIFEST, WIDTHS, required_files, source_hashes)


NAMES = ('input_layer.weight', 'input_layer.bias', 'output_layer.weight', 'output_layer.bias')


def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert (actual.dtype, actual.layout, actual.device) == (expected.dtype, expected.layout, expected.device)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same(a, b)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def read_document(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, indent=2, allow_nan=False) + '\n').encode(), path
    return value


def records(path):
    with path.open('rb') as stream:
        for raw in stream:
            value = json.loads(raw)
            assert raw == (json.dumps(value, allow_nan=False) + '\n').encode(), path
            yield value


def file_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def audit_world(world, folder):
    seed = world['seed']
    same(world, reference_world(seed))
    journals = {name: iter(records(folder / f'{name}.jsonl')) for name in JOURNALS}
    totals = dict(queries=0, gradients=0, checkpoints=0, direct=0, probes=0)
    labels = dict(canonical=list(range(32)), relabeled=world['labels'])
    parameters, optimizers, steps, snapshots = {}, {}, {}, {}
    observed = {e: [] for e in ENCODINGS}
    phase_rows = []

    def phase(stage, **expected):
        row = next(journals['phases'])
        timed = {'seconds', 'journal_seconds'} if stage in ('prediction', 'query', 'training') else {'seconds'}
        assert all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0 for k in timed)
        if 'journal_seconds' in timed:
            assert row['journal_seconds'] <= row['seconds'] + 1e-9
        same({k: v for k, v in row.items() if k not in timed}, dict(stage=stage, **expected))
        phase_rows.append(row)

    def checkpoint(encoding, size, at):
        key = encoding, size
        path = folder / encoding / size / f'{at}.pt'
        saved = torch.load(path, weights_only=True)
        expected = dict(model=dict(zip(NAMES, parameters[key])), optimizer=optimizers[key].state_dict(),
                        observations=observed[encoding], steps=steps[key])
        same(saved, expected)
        snapshots[encoding, size, at] = copy.deepcopy(parameters[key])
        phase('checkpoint', encoding=encoding, arm=size + '_online', at=at, bytes=path.stat().st_size,
            model_bytes=sum(p.numel() * p.element_size() for p in parameters[key]),
            adam_bytes=sum(v.numel() * v.element_size() for state in optimizers[key].state.values()
                           for v in state.values() if isinstance(v, torch.Tensor)),
            replay_json_bytes=len(json.dumps(observed[encoding], separators=(',', ':')).encode()))
        totals['checkpoints'] += 1

    for size, hidden, _ in WIDTHS:
        base, _ = reference_initial(seed, hidden)
        phase('initialization', arm=size)
        for encoding in ENCODINGS:
            key = encoding, size
            parameters[key] = copy.deepcopy(base)
            optimizers[key] = torch.optim.Adam(parameters[key], lr=.01, foreach=False, fused=False)
            steps[key] = 0
            phase('branch_setup', encoding=encoding, arm=size + '_online')
            checkpoint(encoding, size, 0)
    for at, (s, a) in enumerate(world['queries'], 1):
        for encoding in ENCODINGS:
            for size, hidden, _ in WIDTHS:
                with torch.no_grad():
                    logits = reference_logits(parameters[encoding, size], reference_inputs([(labels[encoding][s], a)]))[0]
                prediction = sum(int(v >= 0) << j for j, v in enumerate(logits.tolist()))
                same(next(journals['predictions']), dict(encoding=encoding, arm=size + '_online', at=at,
                     state=labels[encoding][s], action=a, prediction=prediction))
                phase('prediction', encoding=encoding, arm=size + '_online', at=at, input_rows=1, forward_macs=14 * hidden)
        following = world['primitives'][a][s]
        same(next(journals['queries']), dict(at=at, state=s, action=a, next_state=following))
        phase('query', at=at, environment_actions=1)
        totals['queries'] += 1
        for encoding in ENCODINGS:
            observed[encoding].append([labels[encoding][s], a, labels[encoding][following]])
            phase('observation', encoding=encoding, at=at)
            for size, hidden, count in WIDTHS:
                key = encoding, size
                before = torch.cat([p.detach().flatten() for p in parameters[key]]).clone()
                for update in reference_updates(parameters[key], optimizers[key], observed[encoding], seed, steps[key] + 1, count):
                    same(next(journals['updates']), dict(encoding=encoding, arm=size + '_online', at=at, **update))
                    totals['gradients'] += 1
                after = torch.cat([p.detach().flatten() for p in parameters[key]])
                steps[key] += count
                phase('training', encoding=encoding, arm=size + '_online', at=at, gradient_steps=count,
                    input_rows=32 * count, target_bits=160 * count, training_macs=32 * 33 * hidden * count,
                    changed_parameters=int((before != after).sum()), parameter_change_l2=(after - before).norm().item())
                if at in CHECKPOINTS:
                    checkpoint(encoding, size, at)
    table_results = {}
    for encoding in ENCODINGS:
        mapping = labels[encoding]
        inverse = {label: s for s, label in enumerate(mapping)}
        for arm in ('small_online', 'large_online', 'small_frozen', 'large_frozen', 'observed_table'):
            for at in ((0,) if arm.endswith('frozen') else CHECKPOINTS):
                key = dict(encoding=encoding, arm=arm, at=at)
                prefix = observed[encoding][:at]
                if arm == 'observed_table':
                    graph = [[None] * 32 for _ in range(4)]
                    for s, a, y in prefix:
                        graph[a][s] = y
                    detail = dict(input_rows=0, dense_forward_macs=0, logits_sha256=None)
                else:
                    size = arm.split('_')[0]
                    graph, detail = independent_graph(snapshots[encoding, size, at])
                same(next(journals['graphs']), dict(**key, graph=graph, **detail))
                phase('evaluation_graph', **key, **detail)
                if arm != 'observed_table':
                    predictions = [graph[a][mapping[s]] for s, a in world['test']]
                    targets = [mapping[world['primitives'][a][s]] for s, a in world['test']]
                    same(next(journals['direct']), dict(**key, predictions=predictions, targets=targets,
                        exact=sum(p == y for p, y in zip(predictions, targets)),
                        bit_correct=sum(5 - (p ^ y).bit_count() for p, y in zip(predictions, targets)),
                        observed_count=len(prefix), observed_exact=sum(graph[a][s] == y for s, a, y in prefix)))
                    totals['direct'] += 1
                    phase('direct', **key, tests=64)
                actions = edges = expanded = successes = 0
                for start in range(32):
                    for goal in range(32):
                        if start == goal:
                            continue
                        current, target, trace = mapping[start], mapping[goal], []
                        for step in range(1, 9):
                            fallback = reference_rng(seed, f'study_n:probe:{start}:{goal}:{step}').randrange(4)
                            decision = reference_policy(graph, current, target, fallback)
                            following = mapping[world['primitives'][decision['action']][inverse[current]]]
                            trace.append(dict(state=current, next_state=following, **decision))
                            current = following
                            if current == target:
                                break
                        checked = sum(row['edge_checks'] for row in trace)
                        visited = sum(row['expanded_states'] for row in trace)
                        same(next(journals['probes']), dict(**key, start=start, goal=goal, success=current == target,
                            trace=trace, actions=len(trace), edge_checks=checked, expanded_states=visited))
                        if arm == 'observed_table':
                            result = current == target, [row['action'] for row in trace]
                            if encoding == 'canonical':
                                table_results[at, start, goal] = result
                            else:
                                same(result, table_results[at, start, goal])
                        actions += len(trace)
                        edges += checked
                        expanded += visited
                        successes += current == target
                        totals['probes'] += 1
                phase('probes', **key, tasks=992, successes=successes, environment_actions=actions,
                      edge_checks=edges, expanded_states=expanded)
    for at in CHECKPOINTS:
        options = [[[[bit, flip] for bit in range(5) for flip in (0, 1)] for _ in range(5)] for _ in range(4)]
        comparisons = 0
        for s, a in world['queries'][:at]:
            target = world['primitives'][a][s]
            for output in range(5):
                previous = options[a][output]
                comparisons += len(previous)
                options[a][output] = [pair for pair in previous if ((s >> pair[0] & 1) ^ pair[1]) == (target >> output & 1)]
        predictions = []
        for s, a in world['test']:
            bits = [{(s >> bit & 1) ^ flip for bit, flip in candidates} for candidates in options[a]]
            predictions.append(sum(next(iter(v)) << j for j, v in enumerate(bits)) if all(len(v) == 1 for v in bits) else None)
        same(next(journals['reference']), dict(at=at, candidates=options, comparisons=comparisons,
            predictions=predictions, covered=sum(p is not None for p in predictions),
            exact=sum(p == world['primitives'][a][s] for p, (s, a) in zip(predictions, world['test'])),
            permutation_completion_expectation=1 / (32 - at // 4)))
        phase('symbolic_reference', at=at, comparisons=comparisons)
    import numpy as np
    distance = np.full((32, 32), 999, dtype=np.int64)
    np.fill_diagonal(distance, 0)
    for action in world['primitives']:
        distance[np.arange(32), action] = np.minimum(distance[np.arange(32), action], 1)
    for pivot in range(32):
        distance = np.minimum(distance, distance[:, pivot, None] + distance[None, pivot, :])
    same(read_document(folder / 'distances.json'), [dict(encoding=e, start=s, distances=distance[s].tolist())
                                                 for e in ENCODINGS for s in range(32)])
    phase('true_distances')
    end = object()
    for iterator in journals.values():
        assert next(iterator, end) is end, 'Extra physical journal row'
    return totals, phase_rows


def audit_run(root):
    if not __debug__:
        raise RuntimeError('Optimized Python disables audit assertions.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.__version__ == '2.8.0+cu126'
    before = file_hashes(root)
    metadata = read_document(root / 'metadata.json')
    assert metadata['status'] == 'completed'
    assert type(metadata['development']) is bool
    expected_seeds = [DEVELOPMENT_SEED] if metadata['development'] else list(FORMAL_SEEDS)
    same(metadata['seeds'], expected_seeds)
    same(metadata['source_sha256'], source_hashes())
    frozen = read_document(MANIFEST)
    same(frozen['source_sha256'], metadata['source_sha256'])
    same(frozen['source_revision'], metadata['source_revision'])
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    same(metadata['data_sha256'], digest)
    same(frozen['data_sha256'], digest)
    data = read_document(DATA)
    same([w['seed'] for w in data], [DEVELOPMENT_SEED, *FORMAL_SEEDS])
    same({k: metadata[k] for k in ('runtime', 'device', 'dtype', 'threads', 'deterministic_algorithms')},
         dict(runtime='2.8.0+cu126', device='cpu', dtype='float64', threads=1, deterministic_algorithms=True))
    assert type(metadata['elapsed_seconds']) is float and math.isfinite(metadata['elapsed_seconds']) and metadata['elapsed_seconds'] >= 0
    assert set(before) == required_files(expected_seeds, metadata['development'])
    worlds = read_document(root / 'worlds.json')
    same(worlds, data[:1] if metadata['development'] else data[1:])
    same(read_document(root / 'active-phase.json'), dict(stage='completed'))
    if metadata['development']:
        assert metadata['development_audit_sha256'] is None
    else:
        same(metadata['development_audit_sha256'], before['development-audit.json'])
        certificate = read_document(root / 'development-audit.json')
        same(certificate['status'], 'passed')
        same(certificate['seeds'], [DEVELOPMENT_SEED])
        for field in ('source_sha256', 'source_revision', 'data_sha256', 'runtime'):
            same(certificate[field], metadata[field])
        same(certificate['counts'], dict(queries=64, gradients=1280, checkpoints=16, direct=20, probes=27776))
        assert set(certificate['files']) == required_files([DEVELOPMENT_SEED], True)
        assert all(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                   for h in certificate['files'].values())
    counts = dict(queries=0, gradients=0, checkpoints=0, direct=0, probes=0)
    all_phases = []
    for world in worlds:
        totals, phases = audit_world(world, root / f'seed-{world["seed"]}')
        for name, value in totals.items():
            counts[name] += value
        all_phases.extend(dict(seed=world['seed'], **p) for p in phases)
    assert sum(p['seconds'] for p in all_phases) <= metadata['elapsed_seconds'] + 1e-6
    same(before, file_hashes(root))
    return dict(status='passed', seeds=expected_seeds, source_sha256=metadata['source_sha256'],
                source_revision=metadata['source_revision'], data_sha256=metadata['data_sha256'],
                runtime=metadata['runtime'], counts=counts,
                raw_directory=str(root.resolve()), files=before), all_phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report, _ = audit_run(args.input)
    report['elapsed_seconds'] = time.perf_counter() - started
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'source_sha256')}))


if __name__ == '__main__':
    main()
