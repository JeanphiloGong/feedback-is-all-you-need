"""Compare full and local context with complete learning and search records."""

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
from experiments.study_i.model import ProgramPrior, replay_steps
from experiments.study_i.search import count_probabilities, rank_programs, search, token_counts
from experiments.study_j.local import local_replay_steps


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'experiments/study_j/data/environments-v1.json'
DEVELOPMENT_SEED = 31
FORMAL_SEEDS = tuple(range(12000, 12032))
BUDGET = 65536
ARMS = ('uniform', 'counts', 'full_frozen', 'local_frozen', 'full_trained', 'local_trained')
PROBES = ('full_after_8', 'full_after_16', 'local_after_8', 'local_after_16')
LEARNED = ('full_trained', 'local_trained')


def source_hashes():
    paths = sorted((ROOT / 'experiments/study_j').glob('*.py'))
    paths += [ROOT / 'experiments/study_i' / name for name in ('model.py', 'search.py')]
    paths += [ROOT / 'experiments/study_h' / name for name in
              ('tasks.py', 'search.py', 'run.py', 'audit.py')]
    paths.append(ROOT / 'notes/protocol-v10-local-context-prior.md')
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


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
            active.write_text(json.dumps(dict(seed=seed, stage='initialization')))
            started = time.perf_counter()
            initial_seed = rng(seed, 'study_j:initial').getrandbits(63)
            torch.manual_seed(initial_seed)
            initial = ProgramPrior(16)
            initialization_seconds = time.perf_counter() - started
            phases.write(json.dumps(dict(seed=seed, stage='initialization',
                seconds=initialization_seconds)) + '\n')
            phases.flush()
            active.write_text(json.dumps(dict(seed=seed, stage='checkpoint_save', name='initial.pt')))
            started = time.perf_counter()
            torch.save(dict(hidden=16, model=initial.state_dict(), optimizer=None, attempts=0),
                       folder / 'initial.pt')
            phases.write(json.dumps(dict(seed=seed, stage='checkpoint_save', name='initial.pt',
                seconds=time.perf_counter() - started)) + '\n')
            phases.flush()
            active.write_text(json.dumps(dict(seed=seed, stage='clone')))
            started = time.perf_counter()
            models = {'full_frozen': initial}
            models.update({arm: copy.deepcopy(initial) for arm in ('local_frozen', *LEARNED)})
            clone_seconds = time.perf_counter() - started
            phases.write(json.dumps(dict(seed=seed, stage='clone', seconds=clone_seconds)) + '\n')
            phases.flush()
            optimizers, optimizer_setup_seconds = {}, {}
            for arm in LEARNED:
                active.write_text(json.dumps(dict(seed=seed, stage='optimizer_setup', arm=arm)))
                started = time.perf_counter()
                optimizers[arm] = torch.optim.Adam(models[arm].parameters(), lr=.01,
                    betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
                    maximize=False, foreach=False, fused=False)
                optimizer_setup_seconds[arm] = time.perf_counter() - started
                phases.write(json.dumps(dict(seed=seed, stage='optimizer_setup', arm=arm,
                    seconds=optimizer_setup_seconds[arm])) + '\n')
                phases.flush()
            programs, accepted_ids = [], []
            training_seconds = {arm: 0.0 for arm in LEARNED}
            training_steps = {arm: 0 for arm in LEARNED}
            replay_tokens = {arm: 0 for arm in LEARNED}
            input_rows = {arm: 0 for arm in LEARNED}
            input_positions = {arm: 0 for arm in LEARNED}
            checkpoints = {}
            experience = environment['tasks']['experience']
            for index, task in enumerate(experience):
                key = dict(seed=seed, family='experience', task_id=task['id'], arm='uniform')
                active.write_text(json.dumps(dict(stage='search', **key)))
                started = time.perf_counter()
                result = experience_search(primitives, task['train'], [], BUDGET)
                searches.write(json.dumps(dict(**key, result=result,
                    search_seconds=time.perf_counter() - started)) + '\n')
                searches.flush()
                if result['status'] == 'accepted' and result['candidate_program']:
                    program = tuple(result['candidate_program'])
                    programs.append(program)
                    accepted_ids.append(task['id'])
                for arm in LEARNED:
                    corpus = programs
                    active.write_text(json.dumps(dict(seed=seed, stage='replay', arm=arm, attempt=index + 1)))
                    started = step_started = time.perf_counter()
                    completed_steps = target_tokens = physical_rows = physical_positions = 0
                    update = local_replay_steps if arm == 'local_trained' else replay_steps
                    for step, record in enumerate(update(models[arm], optimizers[arm], corpus), 1):
                        if arm == 'full_trained':
                            record.update(input_rows=len(corpus),
                                          input_positions=len(corpus) * max(map(len, corpus)))
                        elapsed = time.perf_counter() - step_started
                        updates.write(json.dumps(dict(seed=seed, arm=arm, attempt=index + 1,
                            step=step, step_seconds=elapsed, **record), allow_nan=False) + '\n')
                        updates.flush()
                        training_steps[arm] += 1
                        replay_tokens[arm] += record['target_tokens']
                        completed_steps += 1
                        target_tokens += record['target_tokens']
                        physical_rows += record['input_rows']
                        physical_positions += record['input_positions']
                        input_rows[arm] += record['input_rows']
                        input_positions[arm] += record['input_positions']
                        step_started = time.perf_counter()
                    elapsed = time.perf_counter() - started
                    training_seconds[arm] += elapsed
                    phases.write(json.dumps(dict(seed=seed, stage='replay', arm=arm, attempt=index + 1,
                        seconds=elapsed, steps=completed_steps, replay_tokens=target_tokens,
                        input_rows=physical_rows, input_positions=physical_positions,
                        skip_reason=None if corpus else 'empty_corpus')) + '\n')
                    phases.flush()
                if index + 1 in (8, 16, len(experience)):
                    for arm in LEARNED:
                        name = f'{arm}-{index + 1}.pt'
                        details = dict(attempts=index + 1, steps=training_steps[arm],
                                       replay_tokens=replay_tokens[arm], input_rows=input_rows[arm],
                                       input_positions=input_positions[arm],
                                       training_seconds=training_seconds[arm])
                        active.write_text(json.dumps(dict(seed=seed, stage='checkpoint_save', name=name)))
                        started = time.perf_counter()
                        torch.save(dict(model=models[arm].state_dict(),
                            optimizer=optimizers[arm].state_dict(),
                            hidden=16, **details), folder / name)
                        phases.write(json.dumps(dict(seed=seed, stage='checkpoint_save', name=name,
                            seconds=time.perf_counter() - started)) + '\n')
                        phases.flush()
                        checkpoints[name] = details
                active.write_text(json.dumps(dict(stage='evaluation', **key)))
                started = time.perf_counter()
                evaluated = evaluate(primitives, task, result)
                evaluations.write(json.dumps(dict(**key, evaluation=evaluated,
                    evaluation_seconds=time.perf_counter() - started)) + '\n')
                evaluations.flush()
            active.write_text(json.dumps(dict(seed=seed, stage='corpus_archive')))
            (folder / 'corpus.json').write_text(json.dumps(dict(
                programs=programs, accepted_task_ids=accepted_ids), indent=2) + '\n')
            active.write_text(json.dumps(dict(seed=seed, stage='counts')))
            started = time.perf_counter()
            counts = token_counts(programs)
            count_seconds = time.perf_counter() - started
            phases.write(json.dumps(dict(seed=seed, stage='counts', seconds=count_seconds)) + '\n')
            phases.flush()
            (folder / 'counts.json').write_text(json.dumps(counts) + '\n')
            ranking = {}
            for arm in ARMS + PROBES:
                if arm == 'uniform':
                    probabilities = lambda prefix: [.25] * 4
                elif arm == 'counts':
                    probabilities = lambda prefix: count_probabilities(counts, prefix)
                else:
                    if arm in PROBES:
                        attempt = int(arm.rsplit('_', 1)[1])
                        active.write_text(json.dumps(dict(seed=seed, stage='probe_load', arm=arm)))
                        started = time.perf_counter()
                        model = copy.deepcopy(initial)
                        base = arm.split('_', 1)[0] + '_trained'
                        saved = torch.load(folder / f'{base}-{attempt}.pt', weights_only=True)
                        model.load_state_dict(saved['model'])
                        phases.write(json.dumps(dict(seed=seed, stage='probe_load', arm=arm,
                            seconds=time.perf_counter() - started)) + '\n')
                        phases.flush()
                    else:
                        model = models[arm]
                    model.eval()
                    probabilities = (lambda prefix: model.probabilities(prefix[-2:])) if arm.startswith('local_') else model.probabilities
                active.write_text(json.dumps(dict(seed=seed, stage='ranking', arm=arm)))
                started = time.perf_counter()
                order, detail = rank_programs(probabilities, BUDGET)
                detail['seconds'] = time.perf_counter() - started
                predicted = order[:-1] if detail['minimum_primitive_ops'] > BUDGET else order
                detail['recurrent_input_positions'] = sum(
                    1 + (min(2, len(p)) if arm.startswith('local_') else len(p))
                    for p in predicted if len(p) < 9) if arm not in ('uniform', 'counts') else 0
                phases.write(json.dumps(dict(seed=seed, stage='ranking', arm=arm, **detail)) + '\n')
                phases.flush()
                ranking[arm] = detail
                (folder / f'order-{arm}.txt').write_text(
                    ''.join(''.join(map(str, program)) + '\n' for program in order))
                (folder / f'ranking-{arm}.json').write_text(json.dumps(detail, indent=2) + '\n')
                for family in ('composition', 'shifted'):
                    for task in environment['tasks'][family]:
                        key = dict(seed=seed, family=family, task_id=task['id'], arm=arm)
                        active.write_text(json.dumps(dict(stage='search', **key)))
                        started = time.perf_counter()
                        result = search(primitives, task['train'], order, BUDGET)
                        searches.write(json.dumps(dict(**key, result=result,
                            search_seconds=time.perf_counter() - started)) + '\n')
                        searches.flush()
                        active.write_text(json.dumps(dict(stage='evaluation', **key)))
                        started = time.perf_counter()
                        evaluated = evaluate(primitives, task, result)
                        evaluations.write(json.dumps(dict(**key, evaluation=evaluated,
                            evaluation_seconds=time.perf_counter() - started)) + '\n')
                        evaluations.flush()
            costs = dict(initial_seed=initial_seed, initialization_seconds=initialization_seconds,
                clone_seconds=clone_seconds, optimizer_setup_seconds=optimizer_setup_seconds,
                training_seconds=training_seconds, training_steps=training_steps,
                replay_tokens=replay_tokens, input_rows=input_rows, input_positions=input_positions,
                count_seconds=count_seconds, ranking=ranking,
                checkpoints=checkpoints, corpus_programs=len(programs), corpus_tokens=sum(map(len, programs)),
                parameters={arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()},
                parameter_bytes={arm: sum(p.numel() * p.element_size() for p in model.parameters())
                                 for arm, model in models.items()},
                checkpoint_bytes={p.name: p.stat().st_size for p in folder.glob('*.pt')},
                corpus_bytes=(folder / 'corpus.json').stat().st_size,
                count_table_cells=100, count_file_bytes=(folder / 'counts.json').stat().st_size,
                order_file_bytes={arm: (folder / f'order-{arm}.txt').stat().st_size for arm in ARMS + PROBES})
            active.write_text(json.dumps(dict(seed=seed, stage='cost_archive')))
            (folder / 'costs.json').write_text(json.dumps(costs, indent=2) + '\n')
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
        runtime = dict(torch=torch.__version__, threads=torch.get_num_threads(),
                       device='cpu', dtype='float64',
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if runtime['torch'] != '2.8.0+cu126':
            raise ValueError('Study J requires the frozen PyTorch 2.8.0+cu126 runtime.')
        hashes = source_hashes()
        data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        if not args.development:
            if args.development_audit is None:
                raise ValueError('Formal collection requires a passing development audit.')
            audit = json.loads(args.development_audit.read_text())
            if (audit['status'] != 'passed' or audit['seeds'] != [DEVELOPMENT_SEED]
                    or audit['source_sha256'] != hashes or audit['data_sha256'] != data_hash
                    or audit['runtime'] != runtime):
                raise ValueError('Development audit must match current source, frozen data, and runtime.')
        frozen = json.loads(DATA_PATH.read_text())
        if [row['seed'] for row in frozen] != [DEVELOPMENT_SEED, *FORMAL_SEEDS]:
            raise ValueError('Frozen data must contain the complete ordered 33-environment grid.')
        if any(len(e['tasks']['experience']) < 16 for e in frozen):
            raise ValueError('The declared intermediate checkpoints require at least 16 experience tasks in every frozen environment.')
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        environments = [row for row in frozen if row['seed'] in seeds]
        metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=args.source_revision, source_sha256=hashes, data_sha256=data_hash,
            seeds=seeds, budget=BUDGET, arms=ARMS, probes=PROBES, python=sys.version,
            **runtime, platform=platform.platform(),
            cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), argv=sys.argv,
            development_audit_sha256=(hashlib.sha256(args.development_audit.read_bytes()).hexdigest()
                                      if args.development_audit else None))
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (args.output / 'environments.json').write_text(json.dumps(environments, indent=2) + '\n')
        collect(environments, args.output)
        metadata.update(status='completed', elapsed_seconds=time.perf_counter() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
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
