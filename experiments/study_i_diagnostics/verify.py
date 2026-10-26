"""Independently reconstruct likelihood summaries using only the standard library."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def verify(folder, diagnostic):
    metadata = json.loads((diagnostic / 'metadata.json').read_text())
    assert metadata['status'] == 'completed'
    assert metadata['original_input_sha256'] == {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob('*') if p.is_file()}
    for name, digest in metadata['output_sha256'].items():
        assert hashlib.sha256((diagnostic / name).read_bytes()).hexdigest() == digest
    rows = [json.loads(line) for line in (diagnostic / 'programs.jsonl').read_text().splitlines()]
    summary = json.loads((diagnostic / 'summary.json').read_text())
    environments = json.loads((folder / 'environments.json').read_text())
    assert [e['seed'] for e in environments] == list(range(11000, 11032))
    arms = ('uniform', 'counts', 'small_frozen', 'small_trained', 'small_shuffled',
            'large_trained', 'small_after_8', 'small_after_16')
    expected, count_tables = {}, {}
    for env in environments:
        seed = env['seed']
        corpus = json.loads((folder / f'seed-{seed}' / 'corpus.json').read_text())
        count_tables[seed] = json.loads((folder / f'seed-{seed}' / 'counts.json').read_text())
        ordinal = {t['id']: i for i, t in enumerate(env['tasks']['experience'], 1)}
        for arm in arms:
            for index, task_id in enumerate(corpus['accepted_task_ids']):
                expected[(seed, arm, 'experience_all', task_id)] = corpus['programs'][index]
                if arm not in ('uniform', 'counts', 'small_frozen'):
                    attempts = int(arm.rsplit('_', 1)[1]) if arm.startswith('small_after_') else 24
                    if ordinal[task_id] <= attempts:
                        target = 'shuffled_programs' if arm == 'small_shuffled' else 'programs'
                        expected[(seed, arm, 'training_corpus', task_id)] = corpus[target][index]
            for family in ('composition', 'shifted'):
                for task in env['tasks'][family]:
                    expected[(seed, arm, family, task['id'])] = task['generating_program']
    actual = {(r['seed'], r['arm'], r['dataset'], r['task_id']): r['program'] for r in rows}
    assert len(actual) == len(rows) == metadata['program_records'] and actual == expected
    groups = defaultdict(list)
    for row in rows:
        program = row['program']
        assert len(row['raw_log_probabilities']) == len(row['smoothed_log_probabilities']) == len(program)
        for index, (raw, smoothed) in enumerate(zip(row['raw_log_probabilities'], row['smoothed_log_probabilities'])):
            assert math.isfinite(raw) and raw <= 0
            assert math.isclose(smoothed, math.log(.9 * math.exp(raw) + .025), rel_tol=1e-12, abs_tol=1e-12)
            if row['arm'] == 'uniform':
                assert math.isclose(raw, -math.log(4), rel_tol=1e-12, abs_tol=1e-12)
            if row['arm'] == 'counts':
                context = ([4, 4] + program[:index])[-2:]
                counts = count_tables[row['seed']][context[0]][context[1]]
                probability = (counts[program[index]] + 1) / (sum(counts) + 4)
                assert math.isclose(raw, math.log(probability), rel_tol=1e-12, abs_tol=1e-12)
        groups[(row['seed'], row['arm'], row['dataset'])].append(row)
    assert metadata['independently_checked_neural_tokens'] == sum(
        len(r['program']) for r in rows if r['arm'] not in ('uniform', 'counts'))

    expected_cells = set()
    for base, group in groups.items():
        expected_cells.add((*base, None, None))
        for length in {len(r['program']) for r in group}:
            expected_cells.add((*base, length, None))
            expected_cells.update((*base, length, position) for position in range(1, length + 1))
    cells, aggregate_groups = {}, defaultdict(list)
    for cell in summary['by_environment']:
        base = (cell['seed'], cell['arm'], cell['dataset'])
        length, position = cell['length'], cell['position']
        key = (*base, length, position)
        assert key not in cells
        cells[key] = cell
        selected = [r for r in groups[base] if length is None or len(r['program']) == length]
        for field, reported in (('raw_log_probabilities', 'raw_loss'), ('smoothed_log_probabilities', 'smoothed_loss')):
            values = [r[field][position - 1] for r in selected] if position else [x for r in selected for x in r[field]]
            assert cell['tokens'] == len(values)
            assert math.isclose(cell[reported], -math.fsum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12)
        aggregate_groups[(cell['arm'], cell['dataset'], length, position)].append(cell)
    assert set(cells) == expected_cells
    seen = set()
    for aggregate in summary['aggregates']:
        key = (aggregate['arm'], aggregate['dataset'], aggregate['length'], aggregate['position'])
        assert key not in seen
        seen.add(key)
        group = aggregate_groups[key]
        assert aggregate['environments'] == len(group) == 32
        assert aggregate['tokens'] == sum(r['tokens'] for r in group)
        for measure in ('raw_loss', 'smoothed_loss'):
            assert math.isclose(aggregate[measure], math.fsum(r[measure] for r in group) / 32,
                                rel_tol=1e-12, abs_tol=1e-12)
    assert seen == set(aggregate_groups)

    training = {}
    for row in summary['training']:
        key = (row['seed'], row['arm'])
        assert key not in training
        training[key] = row
        programs = [r['program'] for r in groups[(*key, 'training_corpus')]]
        prefixes, continuations = Counter(), Counter()
        for program in programs:
            for i, token in enumerate(program):
                prefixes[tuple(program[:i])] += 1
                continuations[(tuple(program[:i]), token)] += 1
        tokens = sum(map(len, programs))
        bound = (math.fsum(n * math.log(n) for n in prefixes.values())
                 - math.fsum(n * math.log(n) for n in continuations.values())) / tokens
        overall = cells[(*key, 'training_corpus', None, None)]
        assert row['programs'] == len(programs) and row['tokens'] == tokens
        assert math.isclose(row['empirical_bound'], bound, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(row['raw_excess'], overall['raw_loss'] - bound, rel_tol=1e-12, abs_tol=1e-12)
        assert row['raw_excess'] >= -1e-12
        for field in ('raw_loss', 'smoothed_loss'):
            assert row[field] == overall[field]
    assert set(training) == {(s, a) for s, a, d in groups if d == 'training_corpus'}
    means = {r['arm']: r for r in summary['training_means']}
    assert len(means) == len(summary['training_means']) and set(means) == {a for s, a in training}
    for arm, row in means.items():
        group = [r for (s, a), r in training.items() if a == arm]
        assert row['environments'] == len(group) == 32
        for field in ('raw_loss', 'smoothed_loss', 'empirical_bound', 'raw_excess'):
            assert math.isclose(row[field], math.fsum(r[field] for r in group) / 32, rel_tol=1e-12, abs_tol=1e-12)
    seen = set()
    for row in summary['differences']:
        dataset, field = row['dataset'], row['measure']
        assert (dataset, field) not in seen
        seen.add((dataset, field))
        comparator = 'small_frozen' if dataset == 'experience_all' else 'uniform'
        assert row['comparator'] == comparator and row['seeds'] == list(range(11000, 11032))
        values = [cells[(s, 'small_trained', dataset, None, None)][field]
                  - cells[(s, comparator, dataset, None, None)][field] for s in row['seeds']]
        assert row['differences'] == values
        assert math.isclose(row['mean'], math.fsum(values) / 32, rel_tol=1e-12, abs_tol=1e-12)
        assert [row[k] for k in ('negative', 'zero', 'positive')] == [
            sum(x < 0 for x in values), sum(x == 0 for x in values), sum(x > 0 for x in values)]
    assert seen == {(d, f) for d in ('experience_all', 'composition', 'shifted') for f in ('raw_loss', 'smoothed_loss')}
    return dict(status='passed', raw_files=len(metadata['original_input_sha256']),
                programs=len(rows), environment_cells=len(cells), aggregates=len(summary['aggregates']),
                training_corpora=len(training), differences=len(seen),
                input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in diagnostic.iterdir() if p.is_file()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input, args.diagnostic)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
