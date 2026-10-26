"""Audit before comparing failure-conditioned adaptation and its full costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_l.audit import audit_run
from experiments.study_l.run import ARMS, BUDGET, FORMAL_SEEDS


BUDGETS = (4096, 16384, 65536)


def summarize(rows, phases, updates, metadata, audit):
    seeds = metadata['seeds']
    cells, costs = [], []
    for seed in seeds:
        selected = [r for r in rows if r['seed'] == seed]
        seed_phases = [p for p in phases if p['seed'] == seed]
        seed_updates = [u for u in updates if u['seed'] == seed]
        for arm in ARMS:
            for family in ('composition', 'shifted', 'balanced'):
                tasks = [r for r in selected if r['arm'] == arm
                         and (family == 'balanced' or r['family'] == family)]
                for budget in BUDGETS:
                    accepted = [r for r in tasks if r['result']['status'] == 'accepted'
                                and r['result']['primitive_ops'] <= budget]
                    exact = sum(r['evaluation']['exact'] for r in accepted)
                    cells.append(dict(seed=seed, arm=arm, family=family, budget=budget,
                        tasks=len(tasks), accepted=len(accepted), exact_count=exact,
                        exact_rate=exact / len(tasks), false_acceptances=len(accepted) - exact,
                        test_accuracy=sum(r['evaluation']['test_correct'] for r in accepted) / (8 * len(tasks))))
            required = [p for p in seed_phases
                if (p['arm'] == arm and p['stage'] in ('ranking', 'visible_search', 'task_setup', 'table', 'feedback'))
                or (arm != 'uniform' and p['arm'] == 'initial' and p['stage'] in
                    ('initialization', 'initial_optimizer', 'experience_search', 'initial_replay', 'table'))]
            learned = [u for u in seed_updates if u['arm'] == arm or (arm != 'uniform' and u['arm'] == 'initial')]
            costs.append(dict(seed=seed, arm=arm,
                primitive_ops=sum(p.get('primitive_ops', 0) for p in required),
                gradient_steps=len(learned),
                initial_target_tokens=sum(u.get('target_tokens', 0) for u in learned),
                rejection_program_exposures=sum(u.get('rejected_programs', 0) for u in learned),
                rejection_token_exposures=sum(u.get('rejected_tokens', 0) for u in learned),
                training_rows=sum(u['input_rows'] for u in learned),
                training_positions=sum(u['input_positions'] for u in learned),
                table_calls=sum(p.get('model_calls', 0) for p in required),
                table_positions=sum(p['input_positions'] for p in required if p['stage'] == 'table'),
                ranking_queries=sum(p.get('predictions', 0) for p in required),
                ranking_prefix_positions=sum(p.get('prefix_positions', 0) for p in required),
                maximum_heap_entries=max(p['maximum_heap_entries'] for p in required if p['stage'] == 'ranking'),
                instrumented_phase_seconds=sum(p['seconds'] for p in required),
                included_journal_seconds=sum(p.get('journal_seconds', 0.) for p in required),
                included_journal_bytes=sum(p.get('journal_bytes', 0) for p in required)))
    aggregates = []
    for arm in ARMS:
        for family in ('composition', 'shifted', 'balanced'):
            for budget in BUDGETS:
                group = [r for r in cells if (r['arm'], r['family'], r['budget']) == (arm, family, budget)]
                aggregates.append(dict(arm=arm, family=family, budget=budget, environments=len(seeds),
                    tasks=sum(r['tasks'] for r in group), exact_count=sum(r['exact_count'] for r in group),
                    exact_rate=mean(r['exact_rate'] for r in group),
                    false_acceptances=sum(r['false_acceptances'] for r in group)))
    contrasts = []
    for family in ('balanced', 'composition', 'shifted'):
        for comparator in ('uniformizing', 'frozen', 'uniform'):
            differences = []
            for seed in seeds:
                scores = {r['arm']: r['exact_rate'] for r in cells
                          if r['seed'] == seed and r['family'] == family and r['budget'] == BUDGET}
                differences.append(scores['rejection'] - scores[comparator])
            contrasts.append(dict(contrast=f'rejection minus {comparator}', family=family, budget=BUDGET,
                seeds=seeds, n=len(seeds), differences=differences, mean=mean(differences),
                negative=sum(v < 0 for v in differences), zero=sum(v == 0 for v in differences),
                positive=sum(v > 0 for v in differences)))
    primary = dict(contrasts[0], confirmatory=seeds == list(FORMAL_SEEDS), ci95=None, bootstrap95=None)
    if primary['confirmatory']:
        values = primary['differences']
        error = stdev(values) / len(values)**.5
        half = float(t.ppf(.975, len(values) - 1)) * error
        bootstrap = np.random.default_rng(977).choice(values, (10000, len(values))).mean(axis=1)
        primary.update(ci95=[primary['mean'] - half, primary['mean'] + half],
            bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
            degenerate=error == 0, positive_primary=primary['mean'] - half > 0)
        primary['interval_disagreement'] = ((primary['ci95'][0] > 0) != (primary['bootstrap95'][0] > 0)
                                           or (primary['ci95'][1] < 0) != (primary['bootstrap95'][1] < 0))
    standalone = []
    for arm in ARMS:
        group = [r for r in costs if r['arm'] == arm]
        total = {name: sum(r[name] for r in group) for name in group[0]
                 if name not in ('seed', 'arm', 'maximum_heap_entries')}
        total.update(arm=arm, maximum_heap_entries=max(r['maximum_heap_entries'] for r in group))
        standalone.append(total)
    actual = dict(searches=len(rows), primitive_ops=sum(r['result']['primitive_ops'] for r in rows),
        hidden_primitive_ops=sum(r['evaluation']['primitive_ops'] for r in rows),
        gradient_steps=len(updates), initial_target_tokens=sum(u.get('target_tokens', 0) for u in updates),
        rejection_program_exposures=sum(u.get('rejected_programs', 0) for u in updates),
        rejection_token_exposures=sum(u.get('rejected_tokens', 0) for u in updates),
        training_rows=sum(u['input_rows'] for u in updates), training_positions=sum(u['input_positions'] for u in updates),
        table_calls=sum(p.get('model_calls', 0) for p in phases),
        table_positions=sum(p['input_positions'] for p in phases if p['stage'] == 'table'),
        ranking_builds=sum(p['stage'] == 'ranking' for p in phases),
        ranking_queries=sum(p.get('predictions', 0) for p in phases),
        ranking_prefix_positions=sum(p.get('prefix_positions', 0) for p in phases),
        candidate_events=audit['candidate_events'],
        checkpoint_files=len(audit['checkpoints']),
        checkpoint_bytes=sum(p['bytes'] for p in phases if p['stage'] == 'checkpoint'),
        archived_model_tensor_bytes=sum(p['model_tensor_bytes'] for p in audit['checkpoints']),
        archived_optimizer_tensor_bytes=sum(p['optimizer_tensor_bytes'] for p in audit['checkpoints']),
        journal_seconds=sum(p.get('journal_seconds', 0.) for p in phases),
        journal_bytes=sum(p.get('journal_bytes', 0) for p in phases),
        phase_seconds={stage: sum(p['seconds'] for p in phases if p['stage'] == stage)
                       for stage in sorted({p['stage'] for p in phases})},
        collection_seconds=metadata['elapsed_seconds'])
    return dict(status='audited task-specific failure-conditioned weight updates', seeds=seeds,
        statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__), primary=primary,
        descriptive_contrasts=contrasts, summaries=aggregates, seed_records=cells,
        standalone_seed_costs=costs, standalone_cost_totals=standalone, actual_cost=actual,
        feedback_batches=[p for p in phases if p['stage'] == 'feedback'],
        interpretation_notes=[
            'Environment means are the inferential units; the two families have equal weight.',
            'Lower-budget scores follow the observed full-budget learning path without retraining or checkpoint selection.',
            'Weights and feedback optimizer reset per task; this does not measure retained cross-task learning.',
            'All arms exclude already rejected complete strings; the learned effect concerns unchecked-program ranking.',
            'Equal update opportunities do not match realized steps, gradient scales, or lifetime resources.',
            'Standalone phase sums include needed initial learning once per environment and task-specific setup, search, ranking, and updates.',
            'Explicit result/checkpoint/table/order archives and hidden evaluations are excluded from standalone sums and retained in actual collection.',
            'Included journal costs are measured separately; phase-ledger/status/file-management metadata overhead remains in total collection time.',
            'Phase sums are shared instrumented measurements, not pure-compute or independent wall-time benchmarks.',
            'Archived tensor/checkpoint bytes are not inference RAM or total system capacity.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        rows, phases, updates, metadata, audit = audit_run(args.input)
        summary = summarize(rows, phases, updates, metadata, audit)
        audit['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(primary=summary['primary'], final=[r for r in summary['summaries'] if r['budget'] == BUDGET]), indent=2))
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
