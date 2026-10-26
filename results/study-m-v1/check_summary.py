"""Reconstruct summary from raw traces without importing experiment modules."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_bytes())
    checkpoints = [0, 64, 128, 256, 512]
    seeds = result['seeds']
    assert seeds == json.loads((args.raw / 'metadata.json').read_text())['seeds']
    assert seeds in ([43], list(range(15000, 15032)))
    arms = ('small_immediate', 'small_delayed', 'small_frozen', 'large_immediate',
            'large_delayed', 'large_frozen', 'observed_table', 'uniform')
    families = ('structured', 'random')
    expected = [(s, f, a) for s in seeds for f in families for a in arms]
    for name in ('seed_curves', 'standalone_costs'):
        assert [(r['seed'], r['family'], r['arm']) for r in result[name]] == expected
    assert [(r['family'], r['arm']) for r in result['summaries']] == [(f, a) for f in families for a in arms]
    comparisons = [('small_immediate', 'large_immediate'), ('small_immediate', 'small_delayed'),
                   ('large_immediate', 'large_delayed'), ('small_immediate', 'small_frozen'),
                   ('large_immediate', 'large_frozen'), ('small_immediate', 'observed_table'),
                   ('small_immediate', 'uniform')]
    assert [(r['family'], r['contrast'], r['metric']) for r in result['descriptive_contrasts']] == [
        (f, left + ' minus ' + right, metric) for f in families for left, right in comparisons for metric in ('auc', 'final')]
    lookup = {}
    checked = 0
    for seed in seeds:
        for family in ('structured', 'random'):
            rows = [json.loads(line) for line in
                    (args.raw / f'seed-{seed}/{family}/probes.jsonl').read_text().splitlines()]
            for arm in ('small_immediate', 'small_delayed', 'small_frozen', 'large_immediate',
                        'large_delayed', 'large_frozen', 'observed_table', 'uniform'):
                counts = []
                for at in checkpoints:
                    source_at = 0 if arm.endswith('frozen') or arm == 'uniform' else at
                    tasks = [r for r in rows if r['arm'] == arm and r['at'] == source_at]
                    assert len(tasks) == 248
                    for row in tasks:
                        assert row['success'] == (row['trace'][-1]['next_state'] == row['goal'])
                        assert row['actions'] == len(row['trace']) <= 8
                    counts.append(sum(r['success'] for r in tasks))
                rates = np.array(counts) / 248
                auc = float(np.dot(np.diff(checkpoints), (rates[:-1] + rates[1:]) / 2) / 512)
                record = next(r for r in result['seed_curves'] if
                              (r['seed'], r['family'], r['arm']) == (seed, family, arm))
                assert record['success_counts'] == counts
                assert record['checkpoints'] == checkpoints and record['tasks_per_checkpoint'] == 248
                assert record['new_probe_evaluations'] == (248 if arm.endswith('frozen') or arm == 'uniform' else 1240)
                np.testing.assert_allclose(record['success_rates'], rates, rtol=0, atol=1e-15)
                assert abs(record['auc'] - auc) < 1e-15
                assert abs(record['final_minus_initial'] - (rates[-1] - rates[0])) < 1e-15
                lookup[seed, family, arm] = auc, rates
                checked += 1
    for row in result['summaries']:
        group = [lookup[seed, row['family'], row['arm']] for seed in seeds]
        np.testing.assert_allclose(row['success_rates'], np.mean([r[1] for r in group], axis=0), atol=1e-15)
        assert abs(row['auc'] - np.mean([r[0] for r in group])) < 1e-15
        np.testing.assert_allclose(row['success_counts'], np.sum([r[1] * 248 for r in group], axis=0), atol=1e-9)
        assert row['tasks_per_checkpoint'] == len(seeds) * 248 and row['environments'] == len(seeds)
        assert row['checkpoints'] == checkpoints
        assert abs(row['final_minus_initial'] - np.mean([r[1][-1] - r[1][0] for r in group])) < 1e-15
    for row in result['descriptive_contrasts']:
        left, right = row['contrast'].split(' minus ')
        values = [lookup[s, row['family'], left][0] - lookup[s, row['family'], right][0]
                  if row['metric'] == 'auc' else
                  lookup[s, row['family'], left][1][-1] - lookup[s, row['family'], right][1][-1] for s in seeds]
        np.testing.assert_allclose(row['differences'], values, atol=1e-15)
        assert abs(row['mean'] - np.mean(values)) < 1e-15
    delta = np.array([lookup[s, 'structured', 'small_immediate'][0] -
                      lookup[s, 'structured', 'large_immediate'][0] for s in seeds])
    primary = result['primary']
    assert (primary['family'], primary['contrast'], primary['metric']) == (
        'structured', 'small_immediate minus large_immediate', 'auc')
    np.testing.assert_allclose(primary['differences'], delta, atol=1e-15)
    assert primary['environments'] == len(seeds)
    assert primary['negative'] == int(np.sum(delta < 0))
    assert primary['zero'] == int(np.sum(delta == 0))
    assert primary['positive'] == int(np.sum(delta > 0))
    assert abs(primary['mean'] - delta.mean()) < 1e-15
    if seeds == list(range(15000, 15032)):
        assert primary['confirmatory'] is True
        half = t.ppf(.975, 31) * np.std(delta, ddof=1) / np.sqrt(32)
        np.testing.assert_allclose(primary['ci95'], [delta.mean() - half, delta.mean() + half], atol=1e-15)
        boot = np.random.default_rng(983).choice(delta, (10000, 32)).mean(axis=1)
        np.testing.assert_allclose(primary['bootstrap95'], np.quantile(boot, [.025, .975]), atol=1e-15)
        assert primary['positive_primary'] == bool(delta.mean() - half > 0)
        assert primary['degenerate'] == bool(np.std(delta, ddof=1) == 0)
        assert primary['interval_disagreement'] == (
            (primary['ci95'][0] > 0) != (primary['bootstrap95'][0] > 0)
            or (primary['ci95'][1] < 0) != (primary['bootstrap95'][1] < 0))
    else:
        assert primary['ci95'] is None and primary['bootstrap95'] is None
        assert primary['confirmatory'] is False and primary['positive_primary'] is False
    phases = [json.loads(line) for line in (args.raw / 'phases.jsonl').read_text().splitlines()]
    for cost in result['standalone_costs']:
        arm = cost['arm']
        own = [p for p in phases if (p['seed'], p['family']) == (cost['seed'], cost['family'])]
        chosen = []
        for p in own:
            owner = p['arm'] == arm
            if arm != 'uniform':
                owner |= p['arm'] == 'warmup' or p['arm'] == arm.split('_')[0] + '_initial'
            if not owner or p['stage'] not in ('branch_setup', 'initialization', 'warmup', 'training', 'graph', 'action'):
                continue
            if p['stage'] == 'graph' and p['input_rows'] == 0:
                continue
            chosen.append(p)
        for field in ('environment_actions', 'gradient_steps', 'dense_training_macs', 'dense_forward_macs',
                      'target_bits', 'edge_checks', 'expanded_states'):
            assert cost[field] == sum(p.get(field, 0) for p in chosen)
        for target, source in (('instrumented_phase_seconds', 'seconds'),
                               ('included_journal_seconds', 'journal_seconds'),
                               ('included_journal_bytes', 'journal_bytes')):
            assert abs(cost[target] - sum(p.get(source, 0) for p in chosen)) < 1e-9
        batches = [p for p in own if p['arm'] == arm and p['stage'] == 'training']
        assert cost['main_update_batches'] == len(batches)
        assert cost['batches_changing_weights'] == sum(p['changed_parameter_values'] > 0 for p in batches)
    for field in ('environment_actions', 'gradient_steps', 'dense_training_macs', 'dense_forward_macs',
                  'journal_seconds', 'journal_bytes'):
        assert abs(result['actual_cost'][field] - sum(p.get(field, 0) for p in phases)) < 1e-9
    report = dict(status='passed', seed_curves=checked, aggregates=len(result['summaries']),
        contrasts=len(result['descriptive_contrasts']), standalone_costs=len(result['standalone_costs']),
        primary=primary, note='No experiment implementation was imported; checkpoint replay remains the host audit.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
