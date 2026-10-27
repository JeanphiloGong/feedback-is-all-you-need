"""Collect paired online updates, then evaluate fixed unqueried transitions."""

import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from experiments.study_h.tasks import rng
from experiments.study_m.model import TransitionModel, features, predicted_graph
from experiments.study_m.policy import choose_action
from experiments.study_n.learning import update_steps
from experiments.study_n.tasks import (CHECKPOINTS, DEVELOPMENT_SEED, ENCODINGS,
    FORMAL_SEEDS, conjugate, signed_candidates, signed_prediction)


DATA = Path('experiments/study_n/data/worlds-v1.json')
MANIFEST = Path('experiments/study_n/data/manifest.json')
WIDTHS = (('small', 32, 8), ('large', 128, 2))
JOURNALS = ('queries', 'predictions', 'updates', 'phases', 'graphs', 'direct', 'probes', 'reference')


def document(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def emit(stream, value):
    started = time.perf_counter()
    line = json.dumps(value, allow_nan=False) + '\n'
    stream.write(line)
    stream.flush()
    return time.perf_counter() - started


def source_hashes():
    paths = sorted(Path('experiments/study_n').glob('*.py'))
    paths += [Path(p) for p in ('experiments/study_h/tasks.py', 'experiments/study_h/search.py',
        'experiments/study_m/model.py', 'experiments/study_m/policy.py', 'experiments/study_m/tasks.py',
        'experiments/study_m/replay.py', 'notes/protocol-v14-encoding-and-unseen-transitions.md')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def required_files(seeds, development):
    names = {'metadata.json', 'worlds.json', 'active-phase.json'}
    if not development:
        names.add('development-audit.json')
    for seed in seeds:
        prefix = f'seed-{seed}/'
        names.update(prefix + name + '.jsonl' for name in JOURNALS)
        names.add(prefix + 'distances.json')
        names.update(f'{prefix}{encoding}/{size}/{at}.pt' for encoding in ENCODINGS
                     for size, _, _ in WIDTHS for at in CHECKPOINTS)
    return names


def save_checkpoint(folder, encoding, size, at, model, optimizer, observations, steps, phases):
    started = time.perf_counter()
    path = folder / encoding / size / f'{at}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = model.state_dict()
    moments = optimizer.state_dict()
    torch.save(dict(model=tensors, optimizer=moments, observations=list(observations), steps=steps), path)
    emit(phases, dict(stage='checkpoint', encoding=encoding, arm=size + '_online', at=at,
        seconds=time.perf_counter() - started, bytes=path.stat().st_size,
        model_bytes=sum(p.numel() * p.element_size() for p in tensors.values()),
        adam_bytes=sum(v.numel() * v.element_size() for s in optimizer.state.values()
                       for v in s.values() if isinstance(v, torch.Tensor)),
        replay_json_bytes=len(json.dumps(observations, separators=(',', ':')).encode())))


def collect_learning(world, folder):
    folder.mkdir()
    seed = world['seed']
    labels = dict(canonical=list(range(32)), relabeled=world['labels'])
    with ExitStack() as stack:
        journals = {name: stack.enter_context((folder / f'{name}.jsonl').open('x'))
                    for name in ('queries', 'predictions', 'updates', 'phases')}
        phases = journals['phases']
        models, optimizers, steps = {}, {}, {}
        observed = {encoding: [] for encoding in ENCODINGS}
        for size, hidden, _ in WIDTHS:
            started = time.perf_counter()
            torch.manual_seed(rng(seed, f'study_n:initial:{hidden}').randrange(2 ** 31))
            base = TransitionModel(hidden)
            emit(phases, dict(stage='initialization', arm=size, seconds=time.perf_counter() - started))
            for encoding in ENCODINGS:
                started = time.perf_counter()
                model = copy.deepcopy(base)
                optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
                key = encoding, size
                models[key], optimizers[key], steps[key] = model, optimizer, 0
                emit(phases, dict(stage='branch_setup', encoding=encoding, arm=size + '_online',
                                  seconds=time.perf_counter() - started))
                save_checkpoint(folder, encoding, size, 0, model, optimizer, [], 0, phases)
        for at, (physical_state, action) in enumerate(world['queries'], 1):
            for encoding in ENCODINGS:
                for size, hidden, _ in WIDTHS:
                    started = time.perf_counter()
                    with torch.no_grad():
                        logits = models[encoding, size](features([(labels[encoding][physical_state], action)]))[0]
                        prediction = sum((int(v >= 0) << j) for j, v in enumerate(logits.tolist()))
                    journal_seconds = emit(journals['predictions'], dict(encoding=encoding, arm=size + '_online', at=at,
                        state=labels[encoding][physical_state], action=action, prediction=prediction))
                    emit(phases, dict(stage='prediction', encoding=encoding, arm=size + '_online', at=at,
                        input_rows=1, forward_macs=14 * hidden, journal_seconds=journal_seconds,
                        seconds=time.perf_counter() - started))
            started = time.perf_counter()
            physical_next = world['primitives'][action][physical_state]
            journal_seconds = emit(journals['queries'], dict(at=at, state=physical_state, action=action, next_state=physical_next))
            emit(phases, dict(stage='query', at=at, environment_actions=1, journal_seconds=journal_seconds,
                              seconds=time.perf_counter() - started))
            for encoding in ENCODINGS:
                started = time.perf_counter()
                observed[encoding].append([labels[encoding][physical_state], action, labels[encoding][physical_next]])
                emit(phases, dict(stage='observation', encoding=encoding, at=at,
                                  seconds=time.perf_counter() - started))
                for size, hidden, count in WIDTHS:
                    key = encoding, size
                    model, optimizer = models[key], optimizers[key]
                    started = time.perf_counter()
                    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                    journal_seconds = 0.
                    for update in update_steps(model, optimizer, observed[encoding], seed, steps[key] + 1, count):
                        journal_seconds += emit(journals['updates'], dict(encoding=encoding, arm=size + '_online', at=at, **update))
                    after = torch.cat([p.detach().flatten() for p in model.parameters()])
                    steps[key] += count
                    emit(phases, dict(stage='training', encoding=encoding, arm=size + '_online', at=at,
                        gradient_steps=count, input_rows=32 * count, target_bits=160 * count,
                        training_macs=32 * 33 * hidden * count, changed_parameters=int((before != after).sum()),
                        parameter_change_l2=(after - before).norm().item(), journal_seconds=journal_seconds,
                        seconds=time.perf_counter() - started))
                    if at in CHECKPOINTS:
                        save_checkpoint(folder, encoding, size, at, model, optimizer, observed[encoding], steps[key], phases)


def goal_probe(graph, true_graph, seed, labels, start, goal):
    current, target = labels[start], labels[goal]
    trace = []
    for step in range(1, 9):
        fallback = rng(seed, f'study_n:probe:{start}:{goal}:{step}').randrange(4)
        decision = choose_action(graph, current, target, fallback)
        following = true_graph[decision['action']][current]
        trace.append(dict(state=current, next_state=following, **decision))
        current = following
        if current == target:
            break
    return dict(start=start, goal=goal, success=current == target, trace=trace,
        actions=len(trace), edge_checks=sum(r['edge_checks'] for r in trace),
        expanded_states=sum(r['expanded_states'] for r in trace))


def evaluate(world, folder):
    seed = world['seed']
    with ExitStack() as stack:
        journals = {name: stack.enter_context((folder / f'{name}.jsonl').open('x'))
                    for name in ('graphs', 'direct', 'probes', 'reference')}
        phases = stack.enter_context((folder / 'phases.jsonl').open('a'))
        for encoding in ENCODINGS:
            labels = list(range(32)) if encoding == 'canonical' else world['labels']
            true_graph = conjugate(world['primitives'], labels)
            for arm in ('small_online', 'large_online', 'small_frozen', 'large_frozen', 'observed_table'):
                for at in ((0,) if arm.endswith('frozen') else CHECKPOINTS):
                    key = dict(encoding=encoding, arm=arm, at=at)
                    prefix = [[labels[s], a, true_graph[a][labels[s]]] for s, a in world['queries'][:at]]
                    started = time.perf_counter()
                    if arm == 'observed_table':
                        graph = [[None] * 32 for _ in range(4)]
                        for s, a, following in prefix:
                            graph[a][s] = following
                        detail = dict(input_rows=0, dense_forward_macs=0, logits_sha256=None)
                    else:
                        size = arm.split('_')[0]
                        hidden = 32 if size == 'small' else 128
                        saved = torch.load(folder / encoding / size / f'{at}.pt', weights_only=True)
                        model = TransitionModel(hidden)
                        model.load_state_dict(saved['model'])
                        graph, detail = predicted_graph(model)
                    emit(journals['graphs'], dict(**key, graph=graph, **detail))
                    emit(phases, dict(stage='evaluation_graph', **key, **detail, seconds=time.perf_counter() - started))
                    if arm != 'observed_table':
                        started = time.perf_counter()
                        predictions = [graph[a][labels[s]] for s, a in world['test']]
                        targets = [true_graph[a][labels[s]] for s, a in world['test']]
                        emit(journals['direct'], dict(**key, predictions=predictions, targets=targets,
                            exact=sum(p == y for p, y in zip(predictions, targets)),
                            bit_correct=sum(5 - (p ^ y).bit_count() for p, y in zip(predictions, targets)),
                            observed_count=len(prefix), observed_exact=sum(graph[a][s] == y for s, a, y in prefix)))
                        emit(phases, dict(stage='direct', **key, tests=64, seconds=time.perf_counter() - started))
                    started = time.perf_counter()
                    actions = edges = expanded = successes = 0
                    for start in range(32):
                        for goal in range(32):
                            if start == goal:
                                continue
                            row = goal_probe(graph, true_graph, seed, labels, start, goal)
                            emit(journals['probes'], dict(**key, **row))
                            actions += row['actions']
                            edges += row['edge_checks']
                            expanded += row['expanded_states']
                            successes += row['success']
                    emit(phases, dict(stage='probes', **key, tasks=992, successes=successes,
                        environment_actions=actions, edge_checks=edges, expanded_states=expanded,
                        seconds=time.perf_counter() - started))
        for at in CHECKPOINTS:
            started = time.perf_counter()
            prefix = [(s, a, world['primitives'][a][s]) for s, a in world['queries'][:at]]
            candidates, comparisons = signed_candidates(prefix)
            predictions = [signed_prediction(candidates, s, a) for s, a in world['test']]
            emit(journals['reference'], dict(at=at, candidates=candidates, comparisons=comparisons,
                predictions=predictions, covered=sum(p is not None for p in predictions),
                exact=sum(p == world['primitives'][a][s] for p, (s, a) in zip(predictions, world['test'])),
                permutation_completion_expectation=1 / (32 - at // 4)))
            emit(phases, dict(stage='symbolic_reference', at=at, comparisons=comparisons,
                              seconds=time.perf_counter() - started))
        started = time.perf_counter()
        distances = []
        for encoding in ENCODINGS:
            labels = list(range(32)) if encoding == 'canonical' else world['labels']
            graph = conjugate(world['primitives'], labels)
            for start in range(32):
                depth, queue = {labels[start]: 0}, [labels[start]]
                for current in queue:
                    for action in graph:
                        following = action[current]
                        if following not in depth:
                            depth[following] = depth[current] + 1
                            queue.append(following)
                distances.append(dict(encoding=encoding, start=start,
                                      distances=[depth[labels[g]] for g in range(32)]))
        document(folder / 'distances.json', distances)
        emit(phases, dict(stage='true_distances', seconds=time.perf_counter() - started))


def collect(worlds, output):
    document(output / 'worlds.json', worlds)
    for world in worlds:
        document(output / 'active-phase.json', dict(stage='learning', seed=world['seed']))
        collect_learning(world, output / f'seed-{world["seed"]}')
    for world in worlds:
        document(output / 'active-phase.json', dict(stage='evaluation', seed=world['seed']))
        evaluate(world, output / f'seed-{world["seed"]}')
    document(output / 'active-phase.json', dict(stage='completed'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--development-audit', type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Optimized Python disables audit assertions.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != '2.8.0+cu126':
        raise RuntimeError('Study N requires PyTorch 2.8.0+cu126.')
    sources = source_hashes()
    frozen = json.loads(MANIFEST.read_text())
    data = DATA.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if frozen['source_sha256'] != sources or frozen['data_sha256'] != digest:
        raise ValueError('Source/data do not match the complete frozen manifest.')
    worlds = json.loads(data)
    if [w['seed'] for w in worlds] != [DEVELOPMENT_SEED, *FORMAL_SEEDS]:
        raise ValueError('Incomplete frozen seed grid.')
    selected = worlds[:1] if args.development else worlds[1:]
    if not args.development:
        if args.development_audit is None:
            raise ValueError('Formal collection requires a matching development audit.')
        certificate = json.loads(args.development_audit.read_text())
        if (certificate['status'] != 'passed' or certificate['seeds'] != [DEVELOPMENT_SEED]
                or certificate['source_sha256'] != sources or certificate['data_sha256'] != digest
                or certificate['source_revision'] != frozen['source_revision']
                or certificate['runtime'] != '2.8.0+cu126'
                or set(certificate['files']) != required_files([DEVELOPMENT_SEED], True)
                or certificate['counts'] != dict(queries=64, gradients=1280, checkpoints=16, direct=20, probes=27776)):
            raise ValueError('Incomplete or mismatched development certificate.')
        raw = Path(certificate['raw_directory'])
        for name, expected in certificate['files'].items():
            if hashlib.sha256((raw / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Development evidence changed after audit.')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', development=args.development, seeds=[w['seed'] for w in selected],
        source_sha256=sources, data_sha256=digest, source_revision=frozen['source_revision'],
        runtime='2.8.0+cu126', device='cpu', dtype='float64', threads=1,
        deterministic_algorithms=True, platform=platform.platform(), node=platform.node())
    metadata['development_audit_sha256'] = None
    if not args.development:
        document(args.output / 'development-audit.json', certificate)
        metadata['development_audit_sha256'] = hashlib.sha256((args.output / 'development-audit.json').read_bytes()).hexdigest()
    document(args.output / 'metadata.json', metadata)
    started = time.perf_counter()
    try:
        collect(selected, args.output)
    except Exception as error:
        metadata.update(status='failed', elapsed_seconds=time.perf_counter() - started)
        document(args.output / 'metadata.json', metadata)
        document(args.output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
    metadata.update(status='completed', elapsed_seconds=time.perf_counter() - started)
    document(args.output / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
