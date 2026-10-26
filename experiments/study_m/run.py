"""Collect persistent action-outcome learning before any held-out-goal probe."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

from experiments.study_h.tasks import rng
from experiments.study_m.model import TransitionModel, predicted_graph, replay_steps
from experiments.study_m.policy import choose_action
from experiments.study_m.tasks import DEVELOPMENT_SEED, FAMILIES, FORMAL_SEEDS


INTERACTIONS = 512
CHECKPOINTS = (0, 64, 128, 256, 512)
ARMS = ('small_immediate', 'small_delayed', 'small_frozen',
        'large_immediate', 'large_delayed', 'large_frozen', 'observed_table', 'uniform')
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'experiments/study_m/data/worlds-v1.json'


def source_hashes():
    paths = sorted((ROOT / 'experiments/study_m').glob('*.py'))
    paths += [ROOT / 'experiments/study_h' / name for name in ('tasks.py', 'search.py')]
    paths.append(ROOT / 'notes/protocol-v13-retained-world-model.md')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def required_files(seeds):
    paths = {'metadata.json', 'worlds.json', 'phases.jsonl', 'active-phase.json'}
    for seed in seeds:
        for family in FAMILIES:
            folder = f'seed-{seed}/{family}'
            paths.update(f'{folder}/{name}' for name in
                         ('warmup.json', 'actions.jsonl', 'updates.jsonl', 'graphs.jsonl',
                          'probes.jsonl', 'oracle.jsonl', 'random-small.pt', 'random-large.pt',
                          'initial-small.pt', 'initial-large.pt'))
            paths.update(f'{folder}/{arm}/{at}.pt' for arm in ARMS for at in CHECKPOINTS)
    return paths


def check_development_evidence(certificate):
    """Check coverage of a controlled audit attestation, not its authenticity."""
    expected = dict(checkpoints=2 * (4 + len(ARMS) * len(CHECKPOINTS)),
        gradient_steps=2 * (160 + 20 * INTERACTIONS), main_actions=2 * len(ARMS) * INTERACTIONS,
        probe_tasks=2 * (5 * len(CHECKPOINTS) + 3) * 248)
    if set(certificate['files']) != required_files([DEVELOPMENT_SEED]):
        raise ValueError('Development audit must cover every required raw file.')
    if any(certificate[name] != value for name, value in expected.items()):
        raise ValueError('Development audit must cover every action, gradient, checkpoint, and probe.')


def train_batch(model, optimizer, corpus, first_step, steps, key, updates, phases):
    started = step_started = time.perf_counter()
    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    journal_seconds, journal_bytes = 0., 0
    for record in replay_steps(model, optimizer, corpus, key['seed'], key['family'], first_step, steps):
        elapsed = time.perf_counter() - step_started
        journal_started = time.perf_counter()
        line = json.dumps(dict(**key, step_seconds=elapsed, **record), allow_nan=False) + '\n'
        updates.write(line)
        updates.flush()
        journal_seconds += time.perf_counter() - journal_started
        journal_bytes += len(line.encode())
        step_started = time.perf_counter()
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    phases.write(json.dumps(dict(stage='training', **key, seconds=time.perf_counter() - started,
        journal_seconds=journal_seconds, journal_bytes=journal_bytes, gradient_steps=steps,
        input_rows=steps * 32, target_bits=steps * 160,
        dense_training_macs=steps * 32 * 33 * model.input_layer.out_features,
        corpus_entries=len(corpus), parameter_change_l2=(after - before).norm().item(),
        changed_parameter_values=int((after != before).sum())), allow_nan=False) + '\n')
    phases.flush()


def record_graph(model, graph, key, graphs, phases):
    started = time.perf_counter()
    detail = dict(input_rows=0, dense_forward_macs=0, logits_sha256=None)
    if model is not None:
        graph, detail = predicted_graph(model)
    journal_started = time.perf_counter()
    line = json.dumps(dict(**key, graph=graph, **detail), allow_nan=False) + '\n'
    graphs.write(line)
    graphs.flush()
    journal_seconds = time.perf_counter() - journal_started
    phases.write(json.dumps(dict(stage='graph', **key, seconds=time.perf_counter() - started,
        journal_seconds=journal_seconds, journal_bytes=len(line.encode()), **detail)) + '\n')
    phases.flush()
    return graph


def save_checkpoint(path, model, optimizer, corpus, graph, steps, episode, episode_step,
                    state, key, phases):
    started = time.perf_counter()
    observed = [[s, a, corpus[s, a]] for s, a in sorted(corpus)]
    tensors = model.state_dict() if model is not None else None
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) if model else 0
    adam_bytes = (sum(v.numel() * v.element_size() for state in optimizer.state.values()
                      for v in state.values() if isinstance(v, torch.Tensor)) if optimizer else 0)
    storage = dict(model_tensor_bytes=model_bytes, adam_tensor_bytes=adam_bytes,
        replay_entries=len(corpus), replay_json_bytes=len(json.dumps(observed).encode()),
        graph_json_bytes=len(json.dumps(graph).encode()))
    torch.save(dict(**key, model=tensors, optimizer=optimizer_state, corpus=observed, graph=graph,
        gradient_steps=steps, episode_index=episode, episode_step=episode_step,
        current_state=state, storage=storage), path)
    phases.write(json.dumps(dict(stage='checkpoint', **key, file=path.name,
        seconds=time.perf_counter() - started, bytes=path.stat().st_size, **storage)) + '\n')
    phases.flush()


def probe(graph, primitives, start, goal, seed, family):
    """Only the evaluator sees true maps; graph and replay are never updated."""
    state, trace = start, []
    for step in range(1, 9):
        fallback = rng(seed, f'study_m:probe:{family}:{start}:{goal}:{step}').randrange(4)
        decision = choose_action(graph, state, goal, fallback)
        action = decision['action']
        following = primitives[action][state]
        predicted = graph[action][state] if graph is not None else None
        trace.append(dict(step=step, state=state, next_state=following,
            predicted_next_state=predicted, fallback_action=fallback, **decision,
            bit_errors=(predicted ^ following).bit_count() if predicted is not None else None))
        state = following
        if state == goal:
            break
    return dict(success=state == goal, actions=len(trace), trace=trace)


def oracle_path(primitives, start, goal):
    """Evaluate exact distances in a strongly connected true graph."""
    queue, seen, edges = [(start, [])], {start}, 0
    for state, path in queue:
        for action in range(4):
            edges += 1
            following = primitives[action][state]
            if following in seen:
                continue
            candidate = path + [action]
            if following == goal:
                return dict(shortest_path=candidate, distance=len(candidate),
                            reachable_within_budget=len(candidate) <= 8, edge_checks=edges)
            seen.add(following)
            queue.append((following, candidate))


def collect_stream(world, folder, arm, initial, warmup_corpus, actions, updates, graphs, phases):
    key = dict(seed=world['seed'], family=world['family'], arm=arm, at=0)
    started = time.perf_counter()
    model = optimizer = graph = None
    steps = 0
    corpus = dict(warmup_corpus) if arm != 'uniform' else {}
    if initial is not None:
        base_model, base_optimizer, base_graph, steps = initial
        model = copy.deepcopy(base_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
        optimizer.load_state_dict(copy.deepcopy(base_optimizer.state_dict()))
        graph = copy.deepcopy(base_graph)
    elif arm == 'observed_table':
        graph = [[corpus.get((s, a)) for s in range(32)] for a in range(4)]
    episode = episode_step = graph_at = 0
    state, goal = world['episodes'][0]['start'], world['episodes'][0]['goal']
    phases.write(json.dumps(dict(stage='branch_setup', **key, seconds=time.perf_counter() - started)) + '\n')
    phases.flush()
    branch = folder / arm
    branch.mkdir()
    record_graph(None, graph, key, graphs, phases)
    save_checkpoint(branch / '0.pt', model, optimizer, corpus, graph, steps,
                    episode, episode_step, state, key, phases)
    for at in range(1, INTERACTIONS + 1):
        key['at'] = at
        started = time.perf_counter()
        fallback = rng(world['seed'], f'study_m:main:{world["family"]}:{at}').randrange(4)
        decision = choose_action(graph, state, goal, fallback)
        planning_seconds = time.perf_counter() - started
        action = decision['action']
        predicted = graph[action][state] if graph is not None else None
        observed_started = time.perf_counter()
        following = world['primitives'][action][state]
        environment_seconds = time.perf_counter() - observed_started
        observed_started = time.perf_counter()
        added = arm != 'uniform' and (state, action) not in corpus
        if arm != 'uniform':
            corpus[state, action] = following
        if arm == 'observed_table':
            graph[action][state] = following
        episode_step += 1
        success = following == goal
        ended = success or episode_step == 16
        observation_seconds = time.perf_counter() - observed_started
        journal_started = time.perf_counter()
        line = json.dumps(dict(**key, episode_index=episode, episode_step=episode_step,
            state=state, goal=goal, next_state=following, predicted_next_state=predicted,
            graph_at=graph_at, fallback_action=fallback, **decision, corpus_added=added,
            corpus_entries=len(corpus), episode_success=success, episode_ended=ended,
            bit_errors=(predicted ^ following).bit_count() if predicted is not None else None)) + '\n'
        actions.write(line)
        actions.flush()
        journal_seconds = time.perf_counter() - journal_started
        phases.write(json.dumps(dict(stage='action', **key, seconds=time.perf_counter() - started,
            planning_seconds=planning_seconds, environment_seconds=environment_seconds,
            observation_seconds=observation_seconds, journal_seconds=journal_seconds,
            journal_bytes=len(line.encode()), environment_actions=1,
            expanded_states=decision['expanded_states'], edge_checks=decision['edge_checks'])) + '\n')
        phases.flush()
        scheduled = arm.endswith('immediate') or (arm.endswith('delayed') and at % 16 == 0)
        if scheduled:
            count = (8 if arm.startswith('small') else 2) * (16 if arm.endswith('delayed') else 1)
            train_batch(model, optimizer, corpus, steps + 1, count, key, updates, phases)
            steps += count
            graph = record_graph(model, graph, key, graphs, phases)
            graph_at = at
        elif arm == 'observed_table':
            record_graph(None, graph, key, graphs, phases)
            graph_at = at
        state = following
        if ended:
            episode += 1
            episode_step = 0
            state, goal = world['episodes'][episode]['start'], world['episodes'][episode]['goal']
        if at in CHECKPOINTS:
            save_checkpoint(branch / f'{at}.pt', model, optimizer, corpus, graph, steps,
                            episode, episode_step, state, key, phases)


def collect(worlds, output):
    active = output / 'active-phase.json'
    with (output / 'phases.jsonl').open('x') as phases:
        for world in worlds:
            seed, family = world['seed'], world['family']
            folder = output / f'seed-{seed}' / family
            folder.mkdir(parents=True)
            key = dict(seed=seed, family=family, arm='warmup', at=0)
            active.write_text(json.dumps(dict(stage='warmup', **key)))
            started = time.perf_counter()
            corpus, observed, state = {}, [], world['warmup']['start']
            for action in world['warmup']['actions']:
                following = world['primitives'][action][state]
                observed.append(dict(state=state, action=action, next_state=following))
                corpus[state, action] = following
                state = following
            journal_started = time.perf_counter()
            content = json.dumps(observed) + '\n'
            (folder / 'warmup.json').write_text(content)
            journal_seconds = time.perf_counter() - journal_started
            phases.write(json.dumps(dict(stage='warmup', **key, seconds=time.perf_counter() - started,
                journal_seconds=journal_seconds, journal_bytes=len(content.encode()),
                environment_actions=len(observed), corpus_entries=len(corpus))) + '\n')
            phases.flush()
            with ((folder / 'updates.jsonl').open('x') as updates,
                  (folder / 'actions.jsonl').open('x') as actions,
                  (folder / 'graphs.jsonl').open('x') as graphs):
                initial = {}
                for size, hidden, steps in (('small', 32, 128), ('large', 128, 32)):
                    key = dict(seed=seed, family=family, arm=f'{size}_initial', at=0)
                    active.write_text(json.dumps(dict(stage='initialization', **key)))
                    started = time.perf_counter()
                    torch.manual_seed(rng(seed, f'study_m:initial:{family}:{hidden}').randrange(2 ** 31))
                    model = TransitionModel(hidden)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False, fused=False)
                    phases.write(json.dumps(dict(stage='initialization', **key,
                        seconds=time.perf_counter() - started, hidden=hidden,
                        parameters=sum(p.numel() for p in model.parameters()))) + '\n')
                    phases.flush()
                    save_checkpoint(folder / f'random-{size}.pt', model, optimizer, corpus, None, 0,
                                    -1, 0, state, key, phases)
                    train_batch(model, optimizer, corpus, 1, steps, key, updates, phases)
                    graph = record_graph(model, None, key, graphs, phases)
                    save_checkpoint(folder / f'initial-{size}.pt', model, optimizer, corpus, graph, steps,
                                    -1, 0, state, key, phases)
                    initial[size] = model, optimizer, graph, steps
                for arm in ARMS:
                    active.write_text(json.dumps(dict(stage='stream', seed=seed, family=family, arm=arm)))
                    base = initial.get(arm.split('_')[0])
                    collect_stream(world, folder, arm, base, corpus, actions, updates, graphs, phases)
        # All streams finish before the first probe or true-map bound is inspected.
        for world in worlds:
            seed, family = world['seed'], world['family']
            folder = output / f'seed-{seed}' / family
            with ((folder / 'probes.jsonl').open('x') as probes,
                  (folder / 'oracle.jsonl').open('x') as oracle):
                for task in world['probes']:
                    key = dict(seed=seed, family=family, arm='oracle', at=0, **task)
                    started = time.perf_counter()
                    decision = oracle_path(world['primitives'], task['start'], task['goal'])
                    journal_started = time.perf_counter()
                    line = json.dumps(dict(**task, **decision)) + '\n'
                    oracle.write(line)
                    oracle.flush()
                    journal_seconds = time.perf_counter() - journal_started
                    phases.write(json.dumps(dict(stage='oracle', **key, seconds=time.perf_counter() - started,
                        journal_seconds=journal_seconds, journal_bytes=len(line.encode()),
                        edge_checks=decision['edge_checks'])) + '\n')
                    phases.flush()
                for arm in ARMS:
                    points = (0,) if arm.endswith('frozen') or arm == 'uniform' else CHECKPOINTS
                    for at in points:
                        key = dict(seed=seed, family=family, arm=arm, at=at)
                        active.write_text(json.dumps(dict(stage='probes', **key)))
                        started = time.perf_counter()
                        graph = torch.load(folder / arm / f'{at}.pt', weights_only=True)['graph']
                        phases.write(json.dumps(dict(stage='probe_checkpoint_load', **key,
                            seconds=time.perf_counter() - started)) + '\n')
                        phases.flush()
                        for task in world['probes']:
                            started = time.perf_counter()
                            value = probe(graph, world['primitives'], task['start'], task['goal'], seed, family)
                            journal_started = time.perf_counter()
                            line = json.dumps(dict(**key, **task, **value)) + '\n'
                            probes.write(line)
                            probes.flush()
                            journal_seconds = time.perf_counter() - journal_started
                            phases.write(json.dumps(dict(stage='probe', **key, **task,
                                seconds=time.perf_counter() - started, journal_seconds=journal_seconds,
                                journal_bytes=len(line.encode()), environment_actions=value['actions'],
                                edge_checks=sum(r['edge_checks'] for r in value['trace']),
                                expanded_states=sum(r['expanded_states'] for r in value['trace']))) + '\n')
                            phases.flush()
    active.write_text(json.dumps(dict(stage='completed')) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--development-audit', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        runtime = dict(torch=str(torch.__version__), threads=torch.get_num_threads(), device='cpu',
            dtype='float64', deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if runtime['torch'] != '2.8.0+cu126':
            raise ValueError('Study M requires PyTorch 2.8.0+cu126.')
        hashes = source_hashes()
        data = DATA_PATH.read_bytes()
        data_hash = hashlib.sha256(data).hexdigest()
        frozen = json.loads(data)
        grid = [[seed, family] for seed in (DEVELOPMENT_SEED,) + FORMAL_SEEDS for family in FAMILIES]
        if [[w['seed'], w['family']] for w in frozen] != grid:
            raise ValueError('Frozen data must contain the complete ordered 66-world grid.')
        certificate = None
        if not args.development:
            if args.development_audit is None:
                raise ValueError('Formal collection requires a passing development audit.')
            certificate = args.development_audit.read_bytes()
            audit = json.loads(certificate)
            if (audit['status'] != 'passed' or audit['seeds'] != [DEVELOPMENT_SEED]
                    or audit['source_sha256'] != hashes or audit['data_sha256'] != data_hash
                    or audit['runtime'] != runtime):
                raise ValueError('Development audit must match source, data, and runtime.')
            check_development_evidence(audit)
            (args.output / 'development-audit.json').write_bytes(certificate)
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        worlds = [w for w in frozen if w['seed'] in seeds]
        metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=args.source_revision, source_sha256=hashes, data_sha256=data_hash,
            seeds=seeds, families=FAMILIES, interactions=INTERACTIONS, checkpoints=CHECKPOINTS, arms=ARMS,
            python=sys.version, **runtime, platform=platform.platform(), argv=sys.argv,
            cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')),
            development_audit_sha256=hashlib.sha256(certificate).hexdigest() if certificate else None)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (args.output / 'worlds.json').write_text(json.dumps(worlds, indent=2) + '\n')
        collect(worlds, args.output)
        metadata.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
