"""Collect continual local-prior learning with separate untouched probes."""

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

from experiments.study_h.run import evaluate
from experiments.study_h.search import search as experience_search
from experiments.study_h.tasks import rng
from experiments.study_i.model import ProgramPrior
from experiments.study_i.search import rank_programs, search
from experiments.study_j.local import local_replay_steps
from experiments.study_k.stream import (ORDER_FAMILIES, PROBE_INDICES, STREAM_INDICES,
                                        local_probability_table, task_roles)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'experiments/study_k/data/environments-v1.json'
DEVELOPMENT_SEED = 37
FORMAL_SEEDS = tuple(range(13000, 13032))
BUDGET = 65536
ARMS = ('uniform', 'frozen', 'replay', 'online')
LEARNED = ('replay', 'online')


def source_hashes():
    paths = sorted((ROOT / 'experiments/study_k').glob('*.py'))
    paths += [ROOT / 'experiments/study_j' / name for name in ('local.py', 'audit.py', 'run.py')]
    paths += [ROOT / 'experiments/study_i' / name for name in ('model.py', 'search.py')]
    paths += [ROOT / 'experiments/study_h' / name for name in ('tasks.py', 'search.py', 'run.py', 'audit.py')]
    paths.append(ROOT / 'notes/protocol-v11-continual-local-feedback.md')
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def record_search(primitives, task, programs, key, searches, evaluations, active):
    active.write_text(json.dumps(dict(stage='search', **key)))
    started = time.perf_counter()
    result = (experience_search(primitives, task['train'], [], BUDGET) if programs is None
              else search(primitives, task['train'], programs, BUDGET))
    searches.write(json.dumps(dict(**key, result=result,
                                  search_seconds=time.perf_counter() - started)) + '\n')
    searches.flush()
    active.write_text(json.dumps(dict(stage='evaluation', **key)))
    started = time.perf_counter()
    value = evaluate(primitives, task, result)
    evaluations.write(json.dumps(dict(**key, evaluation=value,
                                      evaluation_seconds=time.perf_counter() - started)) + '\n')
    evaluations.flush()
    return result


def record_replay(model, optimizer, programs, key, counters, updates, phases, active):
    active.write_text(json.dumps(dict(stage='replay', **key)))
    started = step_started = time.perf_counter()
    steps = tokens = rows = positions = 0
    for step, record in enumerate(local_replay_steps(model, optimizer, programs), 1):
        updates.write(json.dumps(dict(**key, step=step,
            step_seconds=time.perf_counter() - step_started, **record), allow_nan=False) + '\n')
        updates.flush()
        steps += 1
        tokens += record['target_tokens']
        rows += record['input_rows']
        positions += record['input_positions']
        step_started = time.perf_counter()
    elapsed = time.perf_counter() - started
    phases.write(json.dumps(dict(stage='replay', **key, seconds=elapsed, steps=steps,
        replay_tokens=tokens, input_rows=rows, input_positions=positions,
        skip_reason=None if programs else 'empty_corpus')) + '\n')
    phases.flush()
    for name, amount in (('steps', steps), ('replay_tokens', tokens), ('input_rows', rows),
                         ('input_positions', positions), ('training_seconds', elapsed)):
        counters[name] += amount
    return steps


def archive_checkpoint(folder, model, optimizer, programs, accepted_ids, counters,
                       key, prior_id, phases, active):
    active.write_text(json.dumps(dict(stage='checkpoint_save', **key)))
    started = time.perf_counter()
    path = folder / (key['name'] + '.pt')
    torch.save(dict(hidden=16, model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        programs=programs, accepted_task_ids=accepted_ids, counters=dict(counters),
        prior_id=prior_id, **key), path)
    phases.write(json.dumps(dict(stage='checkpoint_save', **key,
        seconds=time.perf_counter() - started, bytes=path.stat().st_size)) + '\n')
    phases.flush()


def archive_prior(folder, model, seed, prior_id, phases, active):
    if model is None:
        probabilities = lambda prefix: [.25] * 4
    else:
        active.write_text(json.dumps(dict(stage='probability_table', seed=seed, prior_id=prior_id)))
        started = time.perf_counter()
        table = local_probability_table(model)
        table_seconds = time.perf_counter() - started
        phases.write(json.dumps(dict(stage='probability_table', seed=seed, prior_id=prior_id,
            seconds=table_seconds, model_calls=len(table),
            recurrent_input_positions=sum(1 + len(context) for context in table),
            probability_values=sum(len(row) for row in table.values()))) + '\n')
        phases.flush()
        active.write_text(json.dumps(dict(stage='table_archive', seed=seed, prior_id=prior_id)))
        started = time.perf_counter()
        path = folder / f'table-{prior_id}.json'
        path.write_text(json.dumps([dict(context=context, probabilities=row)
                                   for context, row in table.items()], indent=2, allow_nan=False) + '\n')
        phases.write(json.dumps(dict(stage='table_archive', seed=seed, prior_id=prior_id,
            seconds=time.perf_counter() - started, bytes=path.stat().st_size)) + '\n')
        phases.flush()
        probabilities = lambda prefix: table[prefix[-2:]]
    active.write_text(json.dumps(dict(stage='ranking', seed=seed, prior_id=prior_id)))
    started = time.perf_counter()
    programs, detail = rank_programs(probabilities, BUDGET)
    phases.write(json.dumps(dict(stage='ranking', seed=seed, prior_id=prior_id,
        seconds=time.perf_counter() - started, **detail)) + '\n')
    phases.flush()
    active.write_text(json.dumps(dict(stage='order_archive', seed=seed, prior_id=prior_id)))
    started = time.perf_counter()
    path = folder / f'order-{prior_id}.txt'
    path.write_text(''.join(''.join(map(str, program)) + '\n' for program in programs))
    phases.write(json.dumps(dict(stage='order_archive', seed=seed, prior_id=prior_id,
        seconds=time.perf_counter() - started, bytes=path.stat().st_size)) + '\n')
    phases.flush()
    return programs


def collect(environments, output):
    with ((output / 'searches.jsonl').open('x') as searches,
          (output / 'evaluations.jsonl').open('x') as evaluations,
          (output / 'updates.jsonl').open('x') as updates,
          (output / 'phases.jsonl').open('x') as phases):
        active = output / 'active-phase.json'
        for environment in environments:
            seed, primitives = environment['seed'], environment['primitives']
            folder = output / f'seed-{seed}'
            folder.mkdir()
            streams, probes = task_roles(environment)
            (folder / 'roles.json').write_text(json.dumps(dict(
                streams={name: [task['id'] for _, task in tasks] for name, tasks in streams.items()},
                probes=[task['id'] for _, task in probes],
                baseline_stream_order='shared:composition_first',
                baseline_probe_checkpoint=0), indent=2) + '\n')
            active.write_text(json.dumps(dict(stage='initialization', seed=seed)))
            started = time.perf_counter()
            initial_seed = rng(seed, 'study_k:initial').getrandbits(63)
            torch.manual_seed(initial_seed)
            base_model = ProgramPrior(16)
            phases.write(json.dumps(dict(stage='initialization', seed=seed,
                seconds=time.perf_counter() - started, initial_seed=initial_seed,
                parameters=sum(p.numel() for p in base_model.parameters()),
                parameter_bytes=sum(p.numel() * p.element_size() for p in base_model.parameters()))) + '\n')
            phases.flush()
            counters = dict(steps=0, replay_tokens=0, input_rows=0, input_positions=0, training_seconds=0.)
            archive_checkpoint(folder, base_model, None, [], [], counters,
                dict(seed=seed, name='random', arm='initial', order='shared', attempt=0),
                None, phases, active)
            active.write_text(json.dumps(dict(stage='optimizer_setup', seed=seed)))
            started = time.perf_counter()
            base_optimizer = torch.optim.Adam(base_model.parameters(), lr=.01, betas=(.9, .999),
                eps=1e-8, weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
            phases.write(json.dumps(dict(stage='optimizer_setup', seed=seed,
                seconds=time.perf_counter() - started)) + '\n')
            phases.flush()
            initial_programs, initial_ids = [], []
            for attempt, task in enumerate(environment['tasks']['experience'], 1):
                key = dict(seed=seed, phase='experience', family='experience', task_id=task['id'],
                           arm='initial', order='shared', attempt=attempt, checkpoint=None, prior_id=None)
                result = record_search(primitives, task, None, key, searches, evaluations, active)
                if result['status'] == 'accepted' and result['candidate_program']:
                    initial_programs.append(tuple(result['candidate_program']))
                    initial_ids.append(task['id'])
                record_replay(base_model, base_optimizer, initial_programs,
                    dict(seed=seed, phase='experience', arm='initial', order='shared', attempt=attempt),
                    counters, updates, phases, active)
            archive_checkpoint(folder, base_model, base_optimizer, initial_programs, initial_ids,
                counters, dict(seed=seed, name='base', arm='initial', order='shared',
                               attempt=len(environment['tasks']['experience'])),
                'base', phases, active)
            orders = {'uniform': archive_prior(folder, None, seed, 'uniform', phases, active),
                      'base': archive_prior(folder, base_model, seed, 'base', phases, active)}
            for arm in ('uniform', 'frozen'):
                prior_id = 'uniform' if arm == 'uniform' else 'base'
                for attempt, (family, task) in enumerate(streams['composition_first'], 1):
                    key = dict(seed=seed, phase='stream', family=family, task_id=task['id'],
                               arm=arm, order='shared', attempt=attempt, checkpoint=None, prior_id=prior_id)
                    record_search(primitives, task, orders[prior_id], key, searches, evaluations, active)
            checkpoints = []
            for order, tasks in streams.items():
                for arm in LEARNED:
                    active.write_text(json.dumps(dict(stage='clone', seed=seed, arm=arm, order=order)))
                    started = time.perf_counter()
                    model, optimizer = copy.deepcopy((base_model, base_optimizer))
                    programs, accepted_ids = list(initial_programs), list(initial_ids)
                    branch_counters = dict(counters)
                    phases.write(json.dumps(dict(stage='clone', seed=seed, arm=arm, order=order,
                        seconds=time.perf_counter() - started)) + '\n')
                    phases.flush()
                    prior_id, changed = 'base', False
                    for attempt, (family, task) in enumerate(tasks, 1):
                        if changed:
                            prior_id = f'{arm}-{order}-{attempt - 1}'
                            orders[prior_id] = archive_prior(folder, model, seed, prior_id, phases, active)
                        key = dict(seed=seed, phase='stream', family=family, task_id=task['id'],
                                   arm=arm, order=order, attempt=attempt, checkpoint=None, prior_id=prior_id)
                        result = record_search(primitives, task, orders[prior_id], key,
                                               searches, evaluations, active)
                        if arm == 'online' and result['status'] == 'accepted' and result['candidate_program']:
                            programs.append(tuple(result['candidate_program']))
                            accepted_ids.append(task['id'])
                        changed = bool(record_replay(model, optimizer, programs,
                            dict(seed=seed, phase='stream', arm=arm, order=order, attempt=attempt),
                            branch_counters, updates, phases, active))
                        if attempt in (8, 16):
                            name = f'{arm}-{order}-{attempt}'
                            checkpoint_prior = name if changed else prior_id
                            archive_checkpoint(folder, model, optimizer, programs, accepted_ids,
                                branch_counters, dict(seed=seed, name=name, arm=arm,
                                                      order=order, attempt=attempt),
                                checkpoint_prior, phases, active)
                            checkpoints.append(dict(arm=arm, order=order, checkpoint=attempt,
                                                    name=name, prior_id=checkpoint_prior))
            # Every stream branch has finished; no probe can guide any update.
            for arm in ('uniform', 'frozen'):
                prior_id = 'uniform' if arm == 'uniform' else 'base'
                for family, task in probes:
                    key = dict(seed=seed, phase='probe', family=family, task_id=task['id'],
                               arm=arm, order='shared', attempt=None, checkpoint=0, prior_id=prior_id)
                    record_search(primitives, task, orders[prior_id], key, searches, evaluations, active)
            for checkpoint in checkpoints:
                prior_id = checkpoint['prior_id']
                if prior_id not in orders:
                    active.write_text(json.dumps(dict(stage='probe_load', seed=seed, **checkpoint)))
                    started = time.perf_counter()
                    saved = torch.load(folder / (checkpoint['name'] + '.pt'), weights_only=True)
                    model = copy.deepcopy(base_model)
                    model.load_state_dict(saved['model'])
                    phases.write(json.dumps(dict(stage='probe_load', seed=seed, **checkpoint,
                        seconds=time.perf_counter() - started)) + '\n')
                    phases.flush()
                    orders[prior_id] = archive_prior(folder, model, seed, prior_id, phases, active)
                for family, task in probes:
                    key = dict(seed=seed, phase='probe', family=family, task_id=task['id'],
                               arm=checkpoint['arm'], order=checkpoint['order'], attempt=None,
                               checkpoint=checkpoint['checkpoint'], prior_id=prior_id)
                    record_search(primitives, task, orders[prior_id], key, searches, evaluations, active)
            active.write_text(json.dumps(dict(stage='checkpoint_index', seed=seed)))
            (folder / 'checkpoints.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
            print(f'Completed environment {seed}', flush=True)
        active.write_text(json.dumps(dict(stage='completed')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--development-audit', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        runtime = dict(torch=torch.__version__, threads=torch.get_num_threads(), device='cpu',
                       dtype='float64', deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if runtime['torch'] != '2.8.0+cu126':
            raise ValueError('Study K requires PyTorch 2.8.0+cu126.')
        hashes = source_hashes()
        data_bytes = DATA_PATH.read_bytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        frozen = json.loads(data_bytes)
        if [row['seed'] for row in frozen] != [DEVELOPMENT_SEED, *FORMAL_SEEDS]:
            raise ValueError('Frozen data must contain the complete ordered 33-environment grid.')
        if any(len(environment['tasks'][family]) != 16 for environment in frozen
               for family in ('composition', 'shifted')):
            raise ValueError('Every frozen transfer family must have all 16 stream/probe tasks.')
        if not args.development:
            if args.development_audit is None:
                raise ValueError('Formal collection requires a passing development audit.')
            audit = json.loads(args.development_audit.read_text())
            if (audit['status'] != 'passed' or audit['seeds'] != [DEVELOPMENT_SEED]
                    or audit['source_sha256'] != hashes or audit['data_sha256'] != data_hash
                    or audit['runtime'] != runtime):
                raise ValueError('Development audit must match current source, data, and runtime.')
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        environments = [row for row in frozen if row['seed'] in seeds]
        metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=args.source_revision, source_sha256=hashes, data_sha256=data_hash,
            seeds=seeds, budget=BUDGET, arms=ARMS, orders=list(ORDER_FAMILIES), checkpoints=[0, 8, 16],
            stream_indices=STREAM_INDICES, probe_indices=PROBE_INDICES,
            baseline_reuse='Shared stream/probe searches are evaluated once and referenced by both orders.',
            python=sys.version, **runtime, platform=platform.platform(),
            cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), argv=sys.argv,
            development_audit_sha256=(hashlib.sha256(args.development_audit.read_bytes()).hexdigest()
                                      if args.development_audit else None))
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (args.output / 'environments.json').write_text(json.dumps(environments, indent=2) + '\n')
        collect(environments, args.output)
        metadata.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    except BaseException as error:
        active = args.output / 'active-phase.json'
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            active_phase_file=active.name if active.is_file() else None,
            elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
