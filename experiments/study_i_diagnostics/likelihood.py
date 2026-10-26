"""Describe frozen Study I priors without updating models or rerunning search."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean
import time
import traceback

import numpy as np
import torch

from experiments.study_i.audit import gru_probabilities
from experiments.study_i.model import ProgramPrior
from experiments.study_i.run import ARMS, FORMAL_SEEDS, PROBES, ROOT, source_hashes


CHECKPOINTS = {
    'small_frozen': 'small-initial.pt',
    'small_trained': 'small_trained-24.pt',
    'small_shuffled': 'small_shuffled-24.pt',
    'large_trained': 'large_trained-24.pt',
    'small_after_8': 'small_trained-8.pt',
    'small_after_16': 'small_trained-16.pt',
}
ATTEMPTS = dict(small_trained=24, small_shuffled=24, large_trained=24,
                small_after_8=8, small_after_16=16)


def empirical_prefix_loss(programs):
    counts = defaultdict(Counter)
    for program in programs:
        for index, token in enumerate(program):
            counts[tuple(program[:index])][token] += 1
    total = sum(map(len, programs))
    return -sum(n * math.log(n / sum(row.values()))
                for row in counts.values() for n in row.values()) / total


@torch.no_grad()
def token_scores(program, model=None, counts=None):
    if model is not None:
        inputs = torch.tensor([[4, *program[:-1]]], dtype=torch.long)
        logs = model(inputs)[0].log_softmax(dim=-1)
        raw = logs[torch.arange(len(program)), torch.tensor(program)].tolist()
    else:
        raw = []
        for index, token in enumerate(program):
            if counts is None:
                probability = .25
            else:
                context = [4, 4, *program[:index]]
                row = counts[context[-2]][context[-1]]
                probability = (row[token] + 1) / (sum(row) + 4)
            raw.append(math.log(probability))
    return dict(raw_log_probabilities=raw,
                smoothed_log_probabilities=[math.log(.9 * math.exp(value) + .025) for value in raw])


def program_sets(environment, corpus, arm):
    common = [('experience_all', task_id, program) for task_id, program in
              zip(corpus['accepted_task_ids'], corpus['programs'])]
    common += [(family, task['id'], task['generating_program'])
               for family in ('composition', 'shifted') for task in environment['tasks'][family]]
    if arm in ATTEMPTS:
        available = {task['id'] for task in environment['tasks']['experience'][:ATTEMPTS[arm]]}
        targets = corpus['shuffled_programs'] if arm == 'small_shuffled' else corpus['programs']
        common += [('training_corpus', task_id, program) for task_id, program in
                   zip(corpus['accepted_task_ids'], targets) if task_id in available]
    return common


def summarize(records):
    cells, corpora = defaultdict(list), defaultdict(list)
    for row in records:
        seed, arm, dataset = row['seed'], row['arm'], row['dataset']
        length = len(row['program'])
        for position, (raw, smoothed) in enumerate(zip(
                row['raw_log_probabilities'], row['smoothed_log_probabilities']), 1):
            for size, offset in ((None, None), (length, None), (length, position)):
                cells[(seed, arm, dataset, size, offset)].append((-raw, -smoothed))
        if dataset == 'training_corpus':
            corpora[(seed, arm)].append(row['program'])
    by_environment, pooled = [], defaultdict(list)
    for (seed, arm, dataset, length, position), values in cells.items():
        item = dict(seed=seed, arm=arm, dataset=dataset, length=length, position=position,
                    tokens=len(values), raw_loss=mean(v[0] for v in values),
                    smoothed_loss=mean(v[1] for v in values))
        by_environment.append(item)
        pooled[(arm, dataset, length, position)].append(item)
    aggregates = [dict(arm=arm, dataset=dataset, length=length, position=position,
        environments=len(rows), tokens=sum(r['tokens'] for r in rows),
        raw_loss=mean(r['raw_loss'] for r in rows),
        smoothed_loss=mean(r['smoothed_loss'] for r in rows))
        for (arm, dataset, length, position), rows in pooled.items()]
    overall = {(r['seed'], r['arm'], r['dataset']): r for r in by_environment
               if r['length'] is None and r['position'] is None}
    training = []
    for (seed, arm), programs in corpora.items():
        row = overall[(seed, arm, 'training_corpus')]
        bound = empirical_prefix_loss(programs)
        training.append(dict(seed=seed, arm=arm, programs=len(programs), tokens=row['tokens'],
                             raw_loss=row['raw_loss'], smoothed_loss=row['smoothed_loss'],
                             empirical_bound=bound, raw_excess=row['raw_loss'] - bound))
    training_means = [dict(arm=arm, environments=len(group),
        **{key: mean(r[key] for r in group) for key in
           ('raw_loss', 'smoothed_loss', 'empirical_bound', 'raw_excess')})
        for arm in ATTEMPTS if (group := [r for r in training if r['arm'] == arm])]
    differences = []
    seeds = sorted({r['seed'] for r in records})
    for dataset, comparator in (('experience_all', 'small_frozen'),
                                 ('composition', 'uniform'), ('shifted', 'uniform')):
        for measure in ('raw_loss', 'smoothed_loss'):
            values = [overall[(seed, 'small_trained', dataset)][measure]
                      - overall[(seed, comparator, dataset)][measure] for seed in seeds]
            differences.append(dict(dataset=dataset, comparator=comparator, measure=measure,
                seeds=seeds, differences=values, mean=mean(values),
                negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values),
                positive=sum(v > 0 for v in values)))
    return dict(status='post-hoc descriptive likelihood diagnostics',
                by_environment=by_environment, aggregates=aggregates, training=training,
                training_means=training_means, differences=differences)


def analyze(folder, audit_path, output, source_revision):
    started = time.perf_counter()
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    source_revision=source_revision, python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, threads=1,
                    device='cpu', dtype='float64', deterministic_algorithms=True)
    output.mkdir(parents=True)
    try:
        audit = json.loads(audit_path.read_text())
        metadata['original_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        assert audit['status'] == 'passed' and audit['seeds'] == list(FORMAL_SEEDS)
        actual_hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(folder.rglob('*')) if p.is_file()}
        assert len(actual_hashes) == 967 and actual_hashes == audit['input_sha256']
        original_metadata = json.loads((folder / 'metadata.json').read_text())
        assert original_metadata['torch'] == torch.__version__
        assert source_hashes() == audit['source_sha256']
        metadata['original_input_sha256'] = actual_hashes
        metadata['source_sha256'] = source_hashes()
        paths = sorted((ROOT / 'experiments/study_i_diagnostics').glob('*.py'))
        paths.append(ROOT / 'notes/study-i-likelihood-diagnostic-plan.md')
        metadata['source_sha256'].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in paths})
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        environments = json.loads((folder / 'environments.json').read_text())
        assert [e['seed'] for e in environments] == list(FORMAL_SEEDS)
        records, expected, checked_tokens = [], {}, 0
        with (output / 'programs.jsonl').open('x') as stream:
            for environment in environments:
                seed = environment['seed']
                seed_folder = folder / f'seed-{seed}'
                corpus = json.loads((seed_folder / 'corpus.json').read_text())
                for arm in ARMS + PROBES:
                    model, counts, weights, cache = None, None, None, {}
                    if arm in CHECKPOINTS:
                        checkpoint = torch.load(seed_folder / CHECKPOINTS[arm], weights_only=True)
                        model = ProgramPrior(checkpoint['hidden'])
                        model.load_state_dict(checkpoint['model'])
                        model.eval()
                        assert checkpoint['attempts'] == ATTEMPTS.get(arm, 0)
                        weights = {name: value.numpy() for name, value in checkpoint['model'].items()}
                    elif arm == 'counts':
                        counts = json.loads((seed_folder / 'counts.json').read_text())
                    for dataset, task_id, program in program_sets(environment, corpus, arm):
                        key = (seed, arm, dataset, task_id)
                        assert key not in expected
                        expected[key] = program
                        scores = token_scores(program, model, counts)
                        if weights is not None:
                            reference = [float(gru_probabilities(weights, tuple(program[:i]), cache)[token])
                                         for i, token in enumerate(program)]
                            np.testing.assert_allclose(np.exp(scores['raw_log_probabilities']),
                                                       reference, rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(scores['raw_log_probabilities'],
                                                       np.log(reference), rtol=1e-12, atol=1e-12)
                            checked_tokens += len(program)
                        row = dict(seed=seed, arm=arm, dataset=dataset, task_id=task_id,
                                   program=program, **scores)
                        records.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                print(f'Scored existing checkpoints for environment {seed}', flush=True)
        archived = [json.loads(line) for line in (output / 'programs.jsonl').read_text().splitlines()]
        assert archived == records
        keyed = {(r['seed'], r['arm'], r['dataset'], r['task_id']): r['program'] for r in archived}
        assert len(keyed) == len(archived) and keyed == expected
        assert all(len(r['program']) == len(r['raw_log_probabilities']) == len(r['smoothed_log_probabilities'])
                   for r in archived)
        summary = summarize(archived)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        metadata.update(status='completed', program_records=len(records),
                        independently_checked_neural_tokens=checked_tokens,
                        elapsed_seconds=time.perf_counter() - started)
        metadata['output_sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                     for name in ('programs.jsonl', 'summary.json')}
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return summary
    except BaseException as error:
        metadata.update(status='failed', error_type=type(error).__name__, message=str(error),
                        traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        (output / 'failure.json').write_text(json.dumps(metadata, indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    analyze(args.input, args.audit, args.output, args.source_revision)


if __name__ == '__main__':
    main()
