"""Collect task-specific failure updates without exposing hidden task outputs."""

import argparse
import copy
from contextlib import ExitStack
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
from experiments.study_i.search import rank_programs
from experiments.study_j.local import local_replay_steps
from experiments.study_l.feedback import feedback_steps, local_probabilities
from experiments.study_l.search import advance_search, rank_unchecked


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'experiments/study_l/data/environments-v1.json'
DEVELOPMENT_SEED = 41
FORMAL_SEEDS = tuple(range(14000, 14032))
BUDGET = 65536
INTERVAL = 4096
STEPS = 16
ARMS = ('uniform', 'frozen', 'uniformizing', 'rejection')


def source_hashes():
    paths = sorted((ROOT / 'experiments/study_l').glob('*.py'))
    for study, names in (
        ('study_h', ('tasks.py', 'search.py', 'run.py', 'audit.py')),
        ('study_i', ('model.py', 'search.py')),
        ('study_j', ('local.py', 'audit.py', 'run.py')),
    ):
        paths.extend(ROOT / 'experiments' / study / name for name in names)
    paths.append(ROOT / 'notes/protocol-v12-failure-conditioned-weights.md')
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def save_checkpoint(path, model, optimizer, details, key, phases, active):
    active.write_text(json.dumps(dict(stage='checkpoint', **key)))
    started = time.perf_counter()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                    **details), path)
    phases.write(json.dumps(dict(stage='checkpoint', **key, file=path.name,
        seconds=time.perf_counter() - started, bytes=path.stat().st_size)) + '\n')
    phases.flush()


def collect_transfer(primitives, examples, folder, base_model, base_table, static_order,
                     key, updates, phases, active):
    arm = key['arm']
    adaptive = arm in ('uniformizing', 'rejection')
    model = optimizer = None
    if adaptive:
        active.write_text(json.dumps(dict(stage='task_setup', **key)))
        started = time.perf_counter()
        model = copy.deepcopy(base_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
            weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
        phases.write(json.dumps(dict(stage='task_setup', **key,
                                    seconds=time.perf_counter() - started)) + '\n')
        phases.flush()
        save_checkpoint(folder / 'initial.pt', model, optimizer,
            dict(primitive_ops=0, rejected=0, steps=0, pending=None),
            dict(**key, at=0), phases, active)
    state = dict(status='running', primitive_ops=0, comparisons=0,
                 candidates=0, rejected=[], pending=None)
    table, table_reference, changed, steps = base_table, 'base-table.json', False, 0
    with (folder / 'candidates.jsonl').open('x') as candidates:
        limits = list(range(INTERVAL, BUDGET, INTERVAL)) + [BUDGET] if adaptive else [BUDGET]
        for round_index, limit in enumerate(limits):
            round_key = dict(**key, at=state['primitive_ops'])
            if adaptive:
                if changed:
                    active.write_text(json.dumps(dict(stage='table', **round_key)))
                    started = time.perf_counter()
                    with torch.no_grad():
                        table = local_probabilities(model).tolist()
                    phases.write(json.dumps(dict(stage='table', **round_key,
                        seconds=time.perf_counter() - started, model_calls=1,
                        input_rows=21, input_positions=63, probabilities=84)) + '\n')
                    phases.flush()
                    table_reference = f'{arm}/{key["task_id"]}/table-{round_index}.json'
                    active.write_text(json.dumps(dict(stage='table_archive', **round_key)))
                    started = time.perf_counter()
                    table_path = folder / f'table-{round_index}.json'
                    table_path.write_text(json.dumps(table, allow_nan=False) + '\n')
                    phases.write(json.dumps(dict(stage='table_archive', **round_key,
                        seconds=time.perf_counter() - started, bytes=table_path.stat().st_size)) + '\n')
                    phases.flush()
                active.write_text(json.dumps(dict(stage='ranking', **round_key)))
                started = time.perf_counter()
                excluded = set(state['rejected'])
                if state['pending'] is not None:
                    excluded.add(state['pending']['program'])
                programs, detail = rank_unchecked(table, excluded, limit - state['primitive_ops'])
                phases.write(json.dumps(dict(stage='ranking', **round_key,
                    seconds=time.perf_counter() - started, **detail)) + '\n')
                phases.flush()
                active.write_text(json.dumps(dict(stage='order_archive', **round_key)))
                started = time.perf_counter()
                order_path = folder / f'round-{round_index}.json'
                order_path.write_text(json.dumps(dict(limit=limit, table=table_reference,
                    programs=programs), separators=(',', ':')) + '\n')
                phases.write(json.dumps(dict(stage='order_archive', **round_key,
                    seconds=time.perf_counter() - started, bytes=order_path.stat().st_size)) + '\n')
                phases.flush()
            else:
                programs = static_order
            active.write_text(json.dumps(dict(stage='visible_search', **round_key)))
            started = time.perf_counter()
            previous_ops, previous_comparisons = state['primitive_ops'], state['comparisons']
            journal_seconds, journal_bytes = 0., 0
            for event in advance_search(primitives, examples, programs, state, limit):
                journal_started = time.perf_counter()
                line = json.dumps(dict(round=round_index, **event)) + '\n'
                candidates.write(line)
                candidates.flush()
                journal_seconds += time.perf_counter() - journal_started
                journal_bytes += len(line)
            phases.write(json.dumps(dict(stage='visible_search', **round_key,
                seconds=time.perf_counter() - started,
                journal_seconds=journal_seconds, journal_bytes=journal_bytes,
                primitive_ops=state['primitive_ops'] - previous_ops,
                comparisons=state['comparisons'] - previous_comparisons)) + '\n')
            phases.flush()
            if state['status'] != 'boundary' or limit == BUDGET:
                break
            update_key = dict(**key, at=limit)
            active.write_text(json.dumps(dict(stage='feedback', **update_key)))
            started = time.perf_counter()
            before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            step_started, count = time.perf_counter(), 0
            journal_seconds, journal_bytes = 0., 0
            for count, record in enumerate(feedback_steps(model, optimizer, arm, state['rejected'], STEPS), 1):
                elapsed = time.perf_counter() - step_started
                journal_started = time.perf_counter()
                line = json.dumps(dict(**update_key, step=count, seconds=elapsed, **record), allow_nan=False) + '\n'
                updates.write(line)
                updates.flush()
                journal_seconds += time.perf_counter() - journal_started
                journal_bytes += len(line)
                step_started = time.perf_counter()
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            phases.write(json.dumps(dict(stage='feedback', **update_key,
                seconds=time.perf_counter() - started, steps=count,
                journal_seconds=journal_seconds, journal_bytes=journal_bytes,
                rejected=len(state['rejected']), rejected_tokens=sum(map(len, state['rejected'])),
                skip_reason='empty_rejected_set' if arm == 'rejection' and not state['rejected'] else None,
                parameter_change_l2=(after - before).norm().item(),
                changed_parameter_values=int((after != before).sum())), allow_nan=False) + '\n')
            phases.flush()
            steps += count
            changed = count > 0
            save_checkpoint(folder / f'after-{limit}.pt', model, optimizer,
                dict(primitive_ops=limit, rejected=len(state['rejected']), steps=steps,
                     pending=copy.deepcopy(state['pending'])), update_key, phases, active)
    if adaptive:
        save_checkpoint(folder / 'final.pt', model, optimizer,
            dict(primitive_ops=state['primitive_ops'], rejected=len(state['rejected']), steps=steps,
                 pending=copy.deepcopy(state['pending'])),
            dict(**key, at=state['primitive_ops']), phases, active)
    return dict(status='exhausted' if state['status'] == 'boundary' else state['status'],
        candidate_program=state['pending']['program'] if state['pending'] else None,
        primitive_ops=state['primitive_ops'], comparisons=state['comparisons'],
        candidates=state['candidates'], rejected=len(state['rejected']),
        pending=state['pending'], feedback_steps=steps)


def collect(environments, output):
    with ((output / 'searches.jsonl').open('x') as searches,
          (output / 'evaluations.jsonl').open('x') as evaluations,
          (output / 'phases.jsonl').open('x') as phases,
          ExitStack() as update_files):
        active = output / 'active-phase.json'
        for environment in environments:
            seed, primitives = environment['seed'], environment['primitives']
            folder = output / f'seed-{seed}'
            folder.mkdir()
            updates = update_files.enter_context((folder / 'updates.jsonl').open('x'))
            key = dict(seed=seed, arm='initial', task_id=None)
            active.write_text(json.dumps(dict(stage='initialization', **key)))
            started = time.perf_counter()
            initial_seed = rng(seed, 'study_l:initial').getrandbits(63)
            torch.manual_seed(initial_seed)
            model = ProgramPrior(16)
            phases.write(json.dumps(dict(stage='initialization', **key,
                seconds=time.perf_counter() - started, initial_seed=initial_seed,
                parameters=sum(p.numel() for p in model.parameters()))) + '\n')
            phases.flush()
            save_checkpoint(folder / 'random.pt', model, None,
                dict(programs=[], accepted_ids=[], steps=0), dict(**key, at=0), phases, active)
            active.write_text(json.dumps(dict(stage='initial_optimizer', **key)))
            started = time.perf_counter()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8,
                weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False)
            phases.write(json.dumps(dict(stage='initial_optimizer', **key,
                                        seconds=time.perf_counter() - started)) + '\n')
            phases.flush()
            programs, accepted_ids, steps = [], [], 0
            for attempt, task in enumerate(environment['tasks']['experience'], 1):
                key = dict(seed=seed, arm='initial', task_id=task['id'])
                active.write_text(json.dumps(dict(stage='experience_search', **key)))
                started = time.perf_counter()
                result = experience_search(primitives, task['train'], [], BUDGET)
                phases.write(json.dumps(dict(stage='experience_search', **key,
                    seconds=time.perf_counter() - started, primitive_ops=result['primitive_ops'],
                    comparisons=result['comparisons'])) + '\n')
                phases.flush()
                active.write_text(json.dumps(dict(stage='result_archive', **key)))
                started = time.perf_counter()
                line = json.dumps(dict(**key, family='experience', result=result)) + '\n'
                searches.write(line)
                searches.flush()
                phases.write(json.dumps(dict(stage='result_archive', **key,
                    seconds=time.perf_counter() - started, bytes=len(line))) + '\n')
                phases.flush()
                if result['status'] == 'accepted' and result['candidate_program']:
                    programs.append(tuple(result['candidate_program']))
                    accepted_ids.append(task['id'])
                active.write_text(json.dumps(dict(stage='initial_replay', **key)))
                started = step_started = time.perf_counter()
                count = 0
                journal_seconds, journal_bytes = 0., 0
                for count, record in enumerate(local_replay_steps(model, optimizer, programs, STEPS), 1):
                    elapsed = time.perf_counter() - step_started
                    journal_started = time.perf_counter()
                    line = json.dumps(dict(**key, at=attempt, step=count,
                        seconds=elapsed, **record), allow_nan=False) + '\n'
                    updates.write(line)
                    updates.flush()
                    journal_seconds += time.perf_counter() - journal_started
                    journal_bytes += len(line)
                    step_started = time.perf_counter()
                phases.write(json.dumps(dict(stage='initial_replay', **key,
                    seconds=time.perf_counter() - started, steps=count,
                    journal_seconds=journal_seconds, journal_bytes=journal_bytes,
                    skip_reason=None if programs else 'empty_corpus')) + '\n')
                phases.flush()
                steps += count
                save_checkpoint(folder / f'experience-{attempt}.pt', model, optimizer,
                    dict(programs=programs, accepted_ids=accepted_ids, steps=steps),
                    dict(**key, at=attempt), phases, active)
                active.write_text(json.dumps(dict(stage='hidden_evaluation', **key)))
                started = time.perf_counter()
                value = evaluate(primitives, task, result)
                journal_started = time.perf_counter()
                line = json.dumps(dict(**key, evaluation=value)) + '\n'
                evaluations.write(line)
                evaluations.flush()
                journal_seconds = time.perf_counter() - journal_started
                phases.write(json.dumps(dict(stage='hidden_evaluation', **key,
                    seconds=time.perf_counter() - started, primitive_ops=value['primitive_ops'],
                    journal_seconds=journal_seconds, journal_bytes=len(line))) + '\n')
                phases.flush()
            key = dict(seed=seed, arm='initial', task_id=None, at=0)
            active.write_text(json.dumps(dict(stage='table', **key)))
            started = time.perf_counter()
            with torch.no_grad():
                base_table = local_probabilities(model).tolist()
            phases.write(json.dumps(dict(stage='table', **key, seconds=time.perf_counter() - started,
                model_calls=1, input_rows=21, input_positions=63, probabilities=84)) + '\n')
            phases.flush()
            active.write_text(json.dumps(dict(stage='table_archive', **key)))
            started = time.perf_counter()
            table_path = folder / 'base-table.json'
            table_path.write_text(json.dumps(base_table, allow_nan=False) + '\n')
            phases.write(json.dumps(dict(stage='table_archive', **key,
                seconds=time.perf_counter() - started, bytes=table_path.stat().st_size)) + '\n')
            phases.flush()
            orders = {}
            for arm in ('uniform', 'frozen'):
                key = dict(seed=seed, arm=arm, task_id=None, at=0)
                active.write_text(json.dumps(dict(stage='ranking', **key)))
                started = time.perf_counter()
                if arm == 'uniform':
                    orders[arm], detail = rank_programs(lambda _: [.25] * 4, BUDGET)
                else:
                    orders[arm], detail = rank_unchecked(base_table, set(), BUDGET)
                phases.write(json.dumps(dict(stage='ranking', **key,
                    seconds=time.perf_counter() - started, **detail)) + '\n')
                phases.flush()
                active.write_text(json.dumps(dict(stage='order_archive', **key)))
                started = time.perf_counter()
                path = folder / f'{arm}-order.json'
                path.write_text(json.dumps(orders[arm], separators=(',', ':')) + '\n')
                phases.write(json.dumps(dict(stage='order_archive', **key,
                    seconds=time.perf_counter() - started, bytes=path.stat().st_size)) + '\n')
                phases.flush()
            for arm in ARMS:
                for family in ('composition', 'shifted'):
                    for task in environment['tasks'][family]:
                        key = dict(seed=seed, arm=arm, task_id=task['id'])
                        task_folder = folder / arm / task['id']
                        task_folder.mkdir(parents=True)
                        result = collect_transfer(primitives, task['train'], task_folder, model,
                            base_table, orders.get(arm), key, updates, phases, active)
                        active.write_text(json.dumps(dict(stage='result_archive', **key)))
                        started = time.perf_counter()
                        line = json.dumps(dict(**key, family=family, result=result)) + '\n'
                        searches.write(line)
                        searches.flush()
                        phases.write(json.dumps(dict(stage='result_archive', **key,
                            seconds=time.perf_counter() - started, bytes=len(line))) + '\n')
                        phases.flush()
                        active.write_text(json.dumps(dict(stage='hidden_evaluation', **key)))
                        started = time.perf_counter()
                        value = evaluate(primitives, task, result)
                        journal_started = time.perf_counter()
                        line = json.dumps(dict(**key, evaluation=value)) + '\n'
                        evaluations.write(line)
                        evaluations.flush()
                        journal_seconds = time.perf_counter() - journal_started
                        phases.write(json.dumps(dict(stage='hidden_evaluation', **key,
                            seconds=time.perf_counter() - started, primitive_ops=value['primitive_ops'],
                            journal_seconds=journal_seconds, journal_bytes=len(line))) + '\n')
                        phases.flush()
    (output / 'active-phase.json').write_text(json.dumps(dict(stage='completed')) + '\n')


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
            raise ValueError('Study L requires PyTorch 2.8.0+cu126.')
        hashes = source_hashes()
        data_bytes = DATA_PATH.read_bytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        frozen = json.loads(data_bytes)
        if [row['seed'] for row in frozen] != [DEVELOPMENT_SEED, *FORMAL_SEEDS]:
            raise ValueError('Frozen data must contain the complete ordered 33-environment grid.')
        if any(len(e['tasks'][family]) != 16 for e in frozen for family in ('composition', 'shifted')):
            raise ValueError('Every transfer family must contain all 16 tasks.')
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
            (args.output / 'development-audit.json').write_bytes(certificate)
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        environments = [row for row in frozen if row['seed'] in seeds]
        metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=args.source_revision, source_sha256=hashes, data_sha256=data_hash,
            seeds=seeds, budget=BUDGET, interval=INTERVAL, steps_per_update=STEPS, arms=ARMS,
            python=sys.version, **runtime, platform=platform.platform(),
            cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), argv=sys.argv,
            development_audit_sha256=hashlib.sha256(certificate).hexdigest() if certificate else None)
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
