"""Reconstruct Study L outcomes and paired inference without its implementation."""
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
from scipy.stats import t

root = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_bytes())
metadata = json.loads((root / 'metadata.json').read_bytes())
searches = [json.loads(line) for line in (root / 'searches.jsonl').read_bytes().splitlines()]
evaluations = [json.loads(line) for line in (root / 'evaluations.jsonl').read_bytes().splitlines()]
seeds = metadata['seeds']
keys = lambda r: (r['seed'], r['arm'], r['task_id'])
scored = {keys(r): r['evaluation'] for r in evaluations}
assert len(scored) == len(evaluations) == len(searches)
assert {keys(r) for r in searches} == set(scored)

records = {}
for seed in seeds:
    for arm in ('uniform', 'frozen', 'uniformizing', 'rejection'):
        for family in ('composition', 'shifted', 'balanced'):
            tasks = [r for r in searches if r['seed'] == seed and r['arm'] == arm
                     and (family == 'balanced' or r['family'] == family)]
            assert len(tasks) == (32 if family == 'balanced' else 16)
            for budget in (4096, 16384, 65536):
                accepted = [r for r in tasks if r['result']['status'] == 'accepted'
                            and r['result']['primitive_ops'] <= budget]
                correct = sum(scored[keys(r)]['exact'] for r in accepted)
                row = dict(seed=seed, arm=arm, family=family, budget=budget, tasks=len(tasks),
                           accepted=len(accepted), exact_count=correct, exact_rate=correct / len(tasks),
                           false_acceptances=len(accepted) - correct,
                           test_accuracy=sum(scored[keys(r)]['test_correct'] for r in accepted) / (8 * len(tasks)))
                records[(seed, arm, family, budget)] = row
assert len(records) == len(summary['seed_records'])
for row in summary['seed_records']:
    assert row == records[(row['seed'], row['arm'], row['family'], row['budget'])]
for row in summary['summaries']:
    group = [records[(seed, row['arm'], row['family'], row['budget'])] for seed in seeds]
    assert row['tasks'] == sum(r['tasks'] for r in group)
    assert row['exact_count'] == sum(r['exact_count'] for r in group)
    assert row['exact_rate'] == mean(r['exact_rate'] for r in group)
    assert row['false_acceptances'] == sum(r['false_acceptances'] for r in group)
for contrast in summary['descriptive_contrasts']:
    comparator = contrast['contrast'].split(' minus ')[1]
    family = contrast['family']
    differences = [records[(seed, 'rejection', family, 65536)]['exact_rate']
                   - records[(seed, comparator, family, 65536)]['exact_rate'] for seed in seeds]
    assert differences == contrast['differences']
    assert contrast['mean'] == mean(differences)
    assert [contrast[name] for name in ('negative', 'zero', 'positive')] == [
        sum(v < 0 for v in differences), sum(v == 0 for v in differences), sum(v > 0 for v in differences)]
primary = summary['primary']
values = [records[(seed, 'rejection', 'balanced', 65536)]['exact_rate']
          - records[(seed, 'uniformizing', 'balanced', 65536)]['exact_rate'] for seed in seeds]
assert primary['differences'] == values
if len(seeds) == 32:
    assert seeds == list(range(14000, 14032)) and primary['confirmatory']
    mu = sum(values) / 32
    se = math.sqrt(sum((v - mu) ** 2 for v in values) / (31 * 32))
    half = t.ppf(.975, 31) * se
    np.testing.assert_allclose(primary['ci95'], [mu - half, mu + half], rtol=0, atol=1e-14)
    indices = np.random.default_rng(977).integers(0, 32, size=(10000, 32))
    boot = np.array(values)[indices].sum(axis=1) / 32
    np.testing.assert_allclose(primary['bootstrap95'], np.quantile(boot, [.025, .975]), rtol=0, atol=1e-14)
else:
    assert seeds == [41] and not primary['confirmatory']
    assert primary['ci95'] is None and primary['bootstrap95'] is None

phases = [json.loads(line) for line in (root / 'phases.jsonl').read_bytes().splitlines()]
update_totals = {}
for seed in seeds:
    for line in (root / f'seed-{seed}/updates.jsonl').read_bytes().splitlines():
        row = json.loads(line)
        totals = update_totals.setdefault((seed, row['arm']), dict(steps=0, target_tokens=0, rejected_programs=0,
                                                                rejected_tokens=0, input_rows=0, input_positions=0))
        totals['steps'] += 1
        for name in ('target_tokens', 'rejected_programs', 'rejected_tokens', 'input_rows', 'input_positions'):
            totals[name] += row.get(name, 0)
for cost in summary['standalone_seed_costs']:
    seed, arm = cost['seed'], cost['arm']
    rows = [r for r in searches if r['seed'] == seed and (r['arm'] == arm or (arm != 'uniform' and r['arm'] == 'initial'))]
    assert cost['primitive_ops'] == sum(r['result']['primitive_ops'] for r in rows)
    learning = [value for (s, a), value in update_totals.items() if s == seed and (a == arm or (arm != 'uniform' and a == 'initial'))]
    for target, source in (('gradient_steps', 'steps'), ('initial_target_tokens', 'target_tokens'),
                           ('rejection_program_exposures', 'rejected_programs'), ('rejection_token_exposures', 'rejected_tokens'),
                           ('training_rows', 'input_rows'), ('training_positions', 'input_positions')):
        assert cost[target] == sum(row[source] for row in learning)
    needed = []
    for phase in phases:
        if phase['seed'] != seed:
            continue
        if phase['arm'] == 'initial':
            include = arm != 'uniform' and phase['stage'] in {'initialization', 'initial_optimizer', 'experience_search', 'initial_replay', 'table'}
        else:
            include = phase['arm'] == arm and phase['stage'] in {'ranking', 'visible_search', 'task_setup', 'table', 'feedback'}
        if include:
            needed.append(phase)
    assert cost['instrumented_phase_seconds'] == sum(p['seconds'] for p in needed)
    assert cost['included_journal_seconds'] == sum(p.get('journal_seconds', 0) for p in needed)
    assert cost['included_journal_bytes'] == sum(p.get('journal_bytes', 0) for p in needed)
    assert cost['table_calls'] == sum(p.get('model_calls', 0) for p in needed)
    assert cost['table_positions'] == sum(p['input_positions'] for p in needed if p['stage'] == 'table')
    assert cost['ranking_queries'] == sum(p.get('predictions', 0) for p in needed)
    assert cost['ranking_prefix_positions'] == sum(p.get('prefix_positions', 0) for p in needed)
    assert cost['maximum_heap_entries'] == max(p['maximum_heap_entries'] for p in needed if p['stage'] == 'ranking')
for total in summary['standalone_cost_totals']:
    group = [row for row in summary['standalone_seed_costs'] if row['arm'] == total['arm']]
    for name, value in total.items():
        if name != 'arm':
            assert value == (max(r[name] for r in group) if name == 'maximum_heap_entries' else sum(r[name] for r in group))
actual = summary['actual_cost']
assert actual['searches'] == len(searches)
assert actual['primitive_ops'] == sum(r['result']['primitive_ops'] for r in searches)
assert actual['hidden_primitive_ops'] == sum(r['primitive_ops'] for r in scored.values())
assert actual['gradient_steps'] == sum(r['steps'] for r in update_totals.values())
report = dict(status='matched', environments=len(seeds), outcome_cells=len(records),
              aggregates=len(summary['summaries']), contrasts=len(summary['descriptive_contrasts']),
              standalone_cost_cells=len(summary['standalone_seed_costs']), primary=primary)
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
