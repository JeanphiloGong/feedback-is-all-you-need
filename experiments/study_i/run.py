"""Collect checked experience, persistent neural updates, and frozen transfer."""

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


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'experiments/study_i/data/environments-v1.json'
DEVELOPMENT_SEED = 29
FORMAL_SEEDS = tuple(range(11000, 11032))
BUDGET = 65536
ARMS = ('uniform', 'counts', 'small_frozen', 'small_trained', 'small_shuffled', 'large_trained')
PROBES = ('small_after_8', 'small_after_16')
LEARNED = ('small_trained', 'small_shuffled', 'large_trained')


def source_hashes():
    paths = sorted((ROOT / 'experiments/study_i').glob('*.py'))
    paths += [ROOT / 'experiments/study_h' / name for name in
              ('tasks.py', 'search.py', 'run.py', 'audit.py')]
    paths.append(ROOT / 'notes/protocol-v9-neural-prior.md')
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
            initialization, initial_seeds, models = {}, {}, {}
            for label, hidden in (('small', 16), ('large', 64)):
                active.write_text(json.dumps(dict(seed=seed, stage='initialization', arm=label)))
                started = time.perf_counter()
                initial_seeds[label] = rng(seed, f'study_i:{label}_initial').getrandbits(63)
                torch.manual_seed(initial_seeds[label])
                models[label + '_trained'] = ProgramPrior(hidden)
                initialization[label] = time.perf_counter() - started
                phases.write(json.dumps(dict(seed=seed, stage='initialization', arm=label,
                    seconds=initialization[label])) + '\n')
                phases.flush()
                active.write_text(json.dumps(dict(seed=seed, stage='checkpoint_save', name=f'{label}-initial.pt')))
                started = time.perf_counter()
                torch.save(dict(hidden=hidden, model=models[label + '_trained'].state_dict(),
                                optimizer=None, attempts=0), folder / f'{label}-initial.pt')
                phases.write(json.dumps(dict(seed=seed, stage='checkpoint_save', name=f'{label}-initial.pt',
                    seconds=time.perf_counter() - started)) + '\n')
                phases.flush()
            active.write_text(json.dumps(dict(seed=seed, stage='clone')))
            started = time.perf_counter()
            models['small_frozen'] = copy.deepcopy(models['small_trained'])
            models['small_shuffled'] = copy.deepcopy(models['small_trained'])
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
            programs, shuffled, accepted_ids = [], [], []
            training_seconds = {arm: 0.0 for arm in LEARNED}
            training_steps = {arm: 0 for arm in LEARNED}
            replay_tokens = {arm: 0 for arm in LEARNED}
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
                    control = list(program)
                    rng(seed, f'study_i:shuffle:{index}').shuffle(control)
                    shuffled.append(tuple(control))
                    accepted_ids.append(task['id'])
                for arm in LEARNED:
                    corpus = shuffled if arm == 'small_shuffled' else programs
                    active.write_text(json.dumps(dict(seed=seed, stage='replay', arm=arm, attempt=index + 1)))
                    started = step_started = time.perf_counter()
                    completed_steps = target_tokens = 0
                    for step, record in enumerate(replay_steps(models[arm], optimizers[arm], corpus), 1):
                        elapsed = time.perf_counter() - step_started
                        updates.write(json.dumps(dict(seed=seed, arm=arm, attempt=index + 1,
                            step=step, step_seconds=elapsed, **record), allow_nan=False) + '\n')
                        updates.flush()
                        training_steps[arm] += 1
                        replay_tokens[arm] += record['target_tokens']
                        completed_steps += 1
                        target_tokens += record['target_tokens']
                        step_started = time.perf_counter()
                    elapsed = time.perf_counter() - started
                    training_seconds[arm] += elapsed
                    phases.write(json.dumps(dict(seed=seed, stage='replay', arm=arm, attempt=index + 1,
                        seconds=elapsed, steps=completed_steps, replay_tokens=target_tokens,
                        skip_reason=None if corpus else 'empty_corpus')) + '\n')
                    phases.flush()
                if index + 1 in (8, 16, len(experience)):
                    for arm in LEARNED:
                        name = f'{arm}-{index + 1}.pt'
                        details = dict(attempts=index + 1, steps=training_steps[arm],
                                       replay_tokens=replay_tokens[arm],
                                       training_seconds=training_seconds[arm])
                        active.write_text(json.dumps(dict(seed=seed, stage='checkpoint_save', name=name)))
                        started = time.perf_counter()
                        torch.save(dict(model=models[arm].state_dict(),
                            optimizer=optimizers[arm].state_dict(),
                            hidden=64 if arm == 'large_trained' else 16, **details), folder / name)
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
                programs=programs, shuffled_programs=shuffled, accepted_task_ids=accepted_ids), indent=2) + '\n')
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
                        model = copy.deepcopy(models['small_frozen'])
                        saved = torch.load(folder / f'small_trained-{attempt}.pt', weights_only=True)
                        model.load_state_dict(saved['model'])
                        phases.write(json.dumps(dict(seed=seed, stage='probe_load', arm=arm,
                            seconds=time.perf_counter() - started)) + '\n')
                        phases.flush()
                    else:
                        model = models[arm]
                    model.eval()
                    probabilities = model.probabilities
                active.write_text(json.dumps(dict(seed=seed, stage='ranking', arm=arm)))
                started = time.perf_counter()
                order, detail = rank_programs(probabilities, BUDGET)
                detail['seconds'] = time.perf_counter() - started
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
            costs = dict(initial_seeds=initial_seeds, initialization_seconds=initialization,
                clone_seconds=clone_seconds, optimizer_setup_seconds=optimizer_setup_seconds,
                training_seconds=training_seconds, training_steps=training_steps,
                replay_tokens=replay_tokens, count_seconds=count_seconds, ranking=ranking,
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
        hashes = source_hashes()
        data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        if not args.development:
            if args.development_audit is None:
                raise ValueError('Formal collection requires a passing development audit.')
            audit = json.loads(args.development_audit.read_text())
            if (audit['status'] != 'passed' or audit['seeds'] != [DEVELOPMENT_SEED]
                    or audit['source_sha256'] != hashes or audit['data_sha256'] != data_hash):
                raise ValueError('Development audit must match current source and frozen data.')
        seeds = [DEVELOPMENT_SEED] if args.development else list(FORMAL_SEEDS)
        environments = [row for row in json.loads(DATA_PATH.read_text()) if row['seed'] in seeds]
        if [row['seed'] for row in environments] != seeds:
            raise ValueError('Frozen data must contain the required ordered seed grid.')
        if any(len(e['tasks']['experience']) < 16 for e in environments):
            raise ValueError('The declared intermediate checkpoints require at least 16 experience tasks.')
        metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=args.source_revision, source_sha256=hashes, data_sha256=data_hash,
            seeds=seeds, budget=BUDGET, arms=ARMS, probes=PROBES, python=sys.version,
            torch=torch.__version__, threads=torch.get_num_threads(), device='cpu', dtype='float64',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), platform=platform.platform(),
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
