"""Audit the complete fixed grid before computing learning and resource summaries."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_i.audit import audit_run
from experiments.study_i.run import ARMS, BUDGET, FORMAL_SEEDS, LEARNED, PROBES


CHECKPOINTS = (4096, 16384, 65536)


def summarize(rows, costs, metadata):
    seeds = sorted(costs)
    records, charged = [], []
    for seed in seeds:
        experience = sorted((r for r in rows if r['seed'] == seed and r['family'] == 'experience'),
                            key=lambda r: r['task_id'])
        cost = costs[seed]
        for arm in ARMS + PROBES:
            evaluated = [r for r in rows if r['seed'] == seed and r['family'] != 'experience' and r['arm'] == arm]
            for family in ('composition', 'shifted'):
                group = [r for r in evaluated if r['family'] == family]
                for budget in CHECKPOINTS:
                    available = [r for r in group if r['result']['status'] == 'accepted'
                                 and r['result']['primitive_ops'] <= budget]
                    exact = sum(r['evaluation']['exact'] for r in available)
                    records.append(dict(seed=seed, arm=arm, family=family, budget=budget,
                        tasks=len(group), accepted=len(available), exact_count=exact,
                        false_acceptances=len(available) - exact, exact_rate=exact / len(group),
                        test_accuracy=sum(r['evaluation']['test_correct'] for r in available) / (8 * len(group))))
            trained = arm in LEARNED or arm in PROBES
            neural = trained or arm == 'small_frozen'
            attempts = (int(arm.rsplit('_', 1)[1]) if arm in PROBES else
                        len(experience) if trained or arm == 'counts' else 0)
            previous = experience[:attempts]
            base = 'small_trained' if arm in PROBES else arm
            checkpoint = cost['checkpoints'][f'{base}-{attempts}.pt'] if trained else None
            item = dict(seed=seed, arm=arm, experience_tasks=attempts,
                experience_false_acceptances=sum(r['result']['status'] == 'accepted' and not r['evaluation']['exact']
                                                 for r in previous),
                parameters=cost['parameters'][base] if neural else 0,
                parameter_bytes=cost['parameter_bytes'][base] if neural else 0,
                count_table_cells=cost['count_table_cells'] if arm == 'counts' else 0,
                training_steps=checkpoint['steps'] if trained else 0,
                replay_target_tokens=checkpoint['replay_tokens'] if trained else 0,
                experience_primitive_ops=sum(r['result']['primitive_ops'] for r in previous),
                evaluation_primitive_ops=sum(r['result']['primitive_ops'] for r in evaluated),
                standalone_search_primitive_ops=sum(r['result']['primitive_ops'] for r in previous + evaluated),
                candidates=sum(r['result']['candidates'] for r in previous + evaluated),
                comparisons=sum(r['result']['comparisons'] for r in previous + evaluated),
                prior_predictions=cost['ranking'][arm]['predictions'],
                neural_prefix_positions=cost['ranking'][arm]['prefix_positions'] if neural else 0,
                maximum_heap_entries=cost['ranking'][arm]['maximum_heap_entries'],
                order_file_bytes=cost['order_file_bytes'][arm],
                initialization_seconds=cost['initialization_seconds']['large' if base == 'large_trained' else 'small'] if neural else 0,
                optimizer_setup_seconds=cost['optimizer_setup_seconds'][base] if trained else 0,
                training_seconds=checkpoint['training_seconds'] if trained else 0,
                count_seconds=cost['count_seconds'] if arm == 'counts' else 0,
                ranking_seconds=cost['ranking'][arm]['seconds'],
                search_seconds=sum(r['search_seconds'] for r in previous + evaluated))
            item['charged_phase_seconds'] = sum(item[name] for name in (
                'initialization_seconds', 'optimizer_setup_seconds', 'training_seconds',
                'count_seconds', 'ranking_seconds', 'search_seconds'))
            charged.append(item)
    summaries = []
    for family in ('composition', 'shifted', 'balanced'):
        for arm in ARMS + PROBES:
            for budget in CHECKPOINTS:
                group = [r for r in records if r['arm'] == arm and r['budget'] == budget
                         and (family == 'balanced' or r['family'] == family)]
                summaries.append(dict(family=family, arm=arm, budget=budget, n=len(seeds),
                    tasks=sum(r['tasks'] for r in group), accepted=sum(r['accepted'] for r in group),
                    exact_count=sum(r['exact_count'] for r in group),
                    false_acceptances=sum(r['false_acceptances'] for r in group),
                    exact_rate=mean(r['exact_rate'] for r in group), test_accuracy=mean(r['test_accuracy'] for r in group)))
    endpoints = {(r['seed'], r['arm']): r['exact_rate'] for r in records
                 if r['family'] == 'composition' and r['budget'] == BUDGET}
    differences = [endpoints[(seed, 'small_trained')] - endpoints[(seed, 'uniform')] for seed in seeds]
    primary = dict(contrast='small_trained minus uniform', family='composition', budget=BUDGET,
        n=len(seeds), mean=mean(differences), differences=differences, ci95=None, bootstrap95=None,
        confirmatory=seeds == list(FORMAL_SEEDS))
    if primary['confirmatory']:
        error = stdev(differences) / len(seeds)**.5
        half = float(t.ppf(.975, len(seeds) - 1)) * error
        bootstrap = np.random.default_rng(947).choice(differences, (10000, len(seeds))).mean(axis=1)
        primary.update(ci95=[primary['mean'] - half, primary['mean'] + half],
                       bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                       degenerate=error == 0, positive_primary=primary['mean'] - half > 0)
        primary['interval_disagreement'] = (
            (primary['ci95'][0] > 0) != (primary['bootstrap95'][0] > 0)
            or (primary['ci95'][1] < 0) != (primary['bootstrap95'][1] < 0))
    totals = []
    nonsums = ('seed', 'arm', 'parameters', 'parameter_bytes', 'count_table_cells', 'maximum_heap_entries')
    for arm in ARMS + PROBES:
        group = [r for r in charged if r['arm'] == arm]
        total = {key: sum(r[key] for r in group) for key in group[0] if key not in nonsums}
        total.update(arm=arm, parameters_per_environment=group[0]['parameters'],
                     parameter_bytes_per_environment=group[0]['parameter_bytes'],
                     count_cells_per_environment=group[0]['count_table_cells'],
                     maximum_heap_entries=max(r['maximum_heap_entries'] for r in group))
        totals.append(total)
    actual = dict(searches=len(rows), primitive_ops=sum(r['result']['primitive_ops'] for r in rows),
        search_seconds=sum(r['search_seconds'] for r in rows),
        final_evaluator_primitive_ops=sum(r['evaluation']['primitive_ops'] for r in rows),
        final_evaluator_seconds=sum(r['evaluation_seconds'] for r in rows),
        gradient_steps=sum(sum(c['training_steps'].values()) for c in costs.values()),
        replay_target_tokens=sum(sum(c['replay_tokens'].values()) for c in costs.values()),
        training_seconds=sum(c['journal_phase_seconds']['replay'] for c in costs.values()),
        initialization_seconds=sum(c['journal_phase_seconds']['initialization'] for c in costs.values()),
        clone_seconds=sum(c['journal_phase_seconds']['clone'] for c in costs.values()),
        optimizer_setup_seconds=sum(c['journal_phase_seconds']['optimizer_setup'] for c in costs.values()),
        count_seconds=sum(c['journal_phase_seconds']['counts'] for c in costs.values()),
        ranking_seconds=sum(c['journal_phase_seconds']['ranking'] for c in costs.values()),
        prior_predictions=sum(c['journal_ranking_predictions'] for c in costs.values()),
        neural_prefix_positions=sum(sum(c['ranking'][arm]['prefix_positions'] for arm in ARMS + PROBES
                                         if arm not in ('uniform', 'counts')) for c in costs.values()),
        checkpoint_bytes=sum(sum(c['checkpoint_bytes'].values()) for c in costs.values()),
        checkpoint_save_seconds=sum(c['journal_phase_seconds']['checkpoint_save'] for c in costs.values()),
        probe_load_seconds=sum(c['journal_phase_seconds']['probe_load'] for c in costs.values()),
        skipped_replay_attempts=sum(c['skipped_replay_attempts'] for c in costs.values()),
        collection_elapsed_seconds=metadata['elapsed_seconds'])
    return dict(status='audited learned program-prior experiment; supplied search and checking', seeds=seeds,
        statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__), primary=primary,
        summaries=summaries, actual_cost=actual, standalone_cost_totals=totals,
        seed_records=records, seed_costs=charged,
        cost_note='Standalone phase sums reuse shared measured work; they are not independent wall-time benchmarks. Ranking is amortized over 32 transfer tasks. Actual totals include intermediate probes.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        rows, costs, metadata, audit = audit_run(args.input)
        summary = summarize(rows, costs, metadata)
        audit['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(primary=summary['primary'], endpoints=[s for s in summary['summaries']
                             if s['budget'] == BUDGET]), indent=2))
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
