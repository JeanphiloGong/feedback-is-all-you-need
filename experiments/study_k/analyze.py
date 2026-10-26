"""Audit first, then compare untouched probes and required final-only costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_k.audit import audit_run
from experiments.study_k.run import ARMS, BUDGET, FORMAL_SEEDS


CHECKPOINTS = (0, 8, 16)
BUDGETS = (4096, 16384, 65536)
ORDERS = ('composition_first', 'shifted_first')
FAMILIES = ('composition', 'shifted')


def summarize(rows, phases, states, metadata):
    seeds = metadata['seeds']
    probes, prequential, costs = [], [], []
    for seed in seeds:
        selected = [row for row in rows if row['seed'] == seed]
        seed_phases = [row for row in phases if row['seed'] == seed]
        checkpoints = {row['name']: row for row in states if row['seed'] == seed}
        for order in ORDERS:
            for arm in ARMS:
                for checkpoint in CHECKPOINTS:
                    actual_arm = ('uniform' if arm == 'uniform' else 'frozen') if checkpoint == 0 else arm
                    actual_order = 'shared' if actual_arm in ('uniform', 'frozen') else order
                    actual_checkpoint = 0 if actual_arm in ('uniform', 'frozen') else checkpoint
                    group = [row for row in selected if row['phase'] == 'probe' and row['arm'] == actual_arm
                             and row['order'] == actual_order and row['checkpoint'] == actual_checkpoint]
                    for family in FAMILIES:
                        tasks = [row for row in group if row['family'] == family]
                        for budget in BUDGETS:
                            accepted = [row for row in tasks if row['result']['status'] == 'accepted'
                                        and row['result']['primitive_ops'] <= budget]
                            exact = sum(row['evaluation']['exact'] for row in accepted)
                            probes.append(dict(seed=seed, order=order, arm=arm, checkpoint=checkpoint,
                                family=family, budget=budget, tasks=len(tasks), accepted=len(accepted),
                                exact_count=exact, exact_rate=exact / len(tasks),
                                false_acceptances=len(accepted) - exact,
                                test_accuracy=sum(row['evaluation']['test_correct'] for row in accepted) / (8 * len(tasks)),
                                actual_arm=actual_arm, actual_order=actual_order, actual_checkpoint=actual_checkpoint))
                actual_order = 'shared' if arm in ('uniform', 'frozen') else order
                for family in FAMILIES:
                    tasks = [row for row in selected if row['phase'] == 'stream' and row['arm'] == arm
                             and row['order'] == actual_order and row['family'] == family]
                    for budget in BUDGETS:
                        accepted = [row for row in tasks if row['result']['status'] == 'accepted'
                                    and row['result']['primitive_ops'] <= budget]
                        exact = sum(row['evaluation']['exact'] for row in accepted)
                        prequential.append(dict(seed=seed, order=order, arm=arm, family=family, budget=budget,
                            tasks=len(tasks), accepted=len(accepted), exact_count=exact, exact_rate=exact / len(tasks),
                            false_acceptances=len(accepted) - exact))
                final_prior = ('uniform' if arm == 'uniform' else 'base' if arm == 'frozen'
                               else checkpoints[f'{arm}-{order}-16']['prior_id'])
                final_rows = [row for row in selected if row['phase'] == 'probe' and row['arm'] == arm
                              and row['order'] == actual_order and row['checkpoint'] == (0 if actual_order == 'shared' else 16)]
                experience = [row for row in selected if row['phase'] == 'experience'] if arm != 'uniform' else []
                stream = [row for row in selected if row['phase'] == 'stream' and row['arm'] == arm
                          and row['order'] == order] if arm == 'online' else []
                required_priors = {final_prior} | {row['prior_id'] for row in stream}
                initial = [row for row in seed_phases if row['stage'] in ('initialization', 'optimizer_setup')]
                training = [row for row in seed_phases if row['stage'] == 'replay'
                            and (row['phase'] == 'experience' or (row['arm'] == arm and row['order'] == order))]
                if arm == 'uniform':
                    initial, training = [], []
                ranking = [row for row in seed_phases if row['stage'] == 'ranking' and row['prior_id'] in required_priors]
                tables = [row for row in seed_phases if row['stage'] == 'probability_table' and row['prior_id'] in required_priors]
                searches = experience + stream + final_rows
                item = dict(seed=seed, order=order, arm=arm, required_priors=sorted(required_priors),
                    experience_searches=len(experience), stream_searches=len(stream), final_probe_searches=len(final_rows),
                    experience_primitive_ops=sum(row['result']['primitive_ops'] for row in experience),
                    stream_primitive_ops=sum(row['result']['primitive_ops'] for row in stream),
                    final_probe_primitive_ops=sum(row['result']['primitive_ops'] for row in final_rows),
                    search_primitive_ops=sum(row['result']['primitive_ops'] for row in searches),
                    gradient_steps=sum(row['steps'] for row in training),
                    replay_target_tokens=sum(row['replay_tokens'] for row in training),
                    training_input_rows=sum(row['input_rows'] for row in training),
                    training_input_positions=sum(row['input_positions'] for row in training),
                    table_model_calls=sum(row['model_calls'] for row in tables),
                    table_recurrent_input_positions=sum(row['recurrent_input_positions'] for row in tables),
                    ranking_queries=sum(row['predictions'] for row in ranking),
                    logical_prefix_positions=sum(row['prefix_positions'] for row in ranking),
                    maximum_heap_entries=max(row['maximum_heap_entries'] for row in ranking),
                    initialization_seconds=sum(row['seconds'] for row in initial if row['stage'] == 'initialization'),
                    optimizer_setup_seconds=sum(row['seconds'] for row in initial if row['stage'] == 'optimizer_setup'),
                    training_seconds=sum(row['seconds'] for row in training),
                    table_seconds=sum(row['seconds'] for row in tables),
                    ranking_seconds=sum(row['seconds'] for row in ranking),
                    search_seconds=sum(row['search_seconds'] for row in searches),
                    parameters_per_model=0 if arm == 'uniform' else 1172,
                    live_probability_values=0 if arm == 'uniform' else 84)
                item['charged_phase_seconds'] = sum(item[key] for key in ('initialization_seconds', 'optimizer_setup_seconds',
                    'training_seconds', 'table_seconds', 'ranking_seconds', 'search_seconds'))
                costs.append(item)
    aggregates = []
    for order in (*ORDERS, 'both'):
        for arm in ARMS:
            for checkpoint in CHECKPOINTS:
                for family in (*FAMILIES, 'balanced'):
                    for budget in BUDGETS:
                        group = [row for row in probes if row['arm'] == arm and row['checkpoint'] == checkpoint
                                 and row['budget'] == budget and (order == 'both' or row['order'] == order)
                                 and (family == 'balanced' or row['family'] == family)]
                        conditions = sum(row['tasks'] for row in group)
                        aggregates.append(dict(order=order, arm=arm, checkpoint=checkpoint, family=family,
                            budget=budget, environments=len(seeds), condition_tasks=conditions,
                            unique_tasks=conditions // (2 if order == 'both' else 1),
                            exact_count=sum(row['exact_count'] for row in group),
                            exact_rate=mean(row['exact_rate'] for row in group),
                            false_acceptances=sum(row['false_acceptances'] for row in group)))
    contrasts = []
    for family in ('balanced', *FAMILIES):
        for comparator in ('frozen', 'replay', 'uniform'):
            differences = []
            for seed in seeds:
                rates = {arm: mean(row['exact_rate'] for row in probes if row['seed'] == seed
                                   and row['arm'] == arm and row['checkpoint'] == 16 and row['budget'] == BUDGET
                                   and (family == 'balanced' or row['family'] == family))
                         for arm in ('online', comparator)}
                differences.append(rates['online'] - rates[comparator])
            contrasts.append(dict(contrast=f'online minus {comparator}', family=family, checkpoint=16,
                budget=BUDGET, n=len(seeds), seeds=seeds, differences=differences, mean=mean(differences),
                negative=sum(value < 0 for value in differences), zero=sum(value == 0 for value in differences),
                positive=sum(value > 0 for value in differences)))
    primary = dict(contrasts[0], confirmatory=seeds == list(FORMAL_SEEDS), ci95=None, bootstrap95=None)
    if primary['confirmatory']:
        values = primary['differences']
        error = stdev(values) / len(values)**.5
        half = float(t.ppf(.975, len(values) - 1)) * error
        bootstrap = np.random.default_rng(967).choice(values, (10000, len(values))).mean(axis=1)
        primary.update(ci95=[primary['mean'] - half, primary['mean'] + half],
                       bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(), degenerate=error == 0,
                       positive_primary=primary['mean'] - half > 0)
        primary['interval_disagreement'] = ((primary['ci95'][0] > 0) != (primary['bootstrap95'][0] > 0)
                                           or (primary['ci95'][1] < 0) != (primary['bootstrap95'][1] < 0))
    standalone = []
    excluded = ('seed', 'order', 'arm', 'required_priors', 'maximum_heap_entries',
                'parameters_per_model', 'live_probability_values')
    for order in ORDERS:
        for arm in ARMS:
            group = [row for row in costs if row['order'] == order and row['arm'] == arm]
            total = {key: sum(row[key] for row in group) for key in group[0] if key not in excluded}
            total.update(order=order, arm=arm, parameters_per_model=group[0]['parameters_per_model'],
                         live_probability_values=group[0]['live_probability_values'],
                         maximum_heap_entries=max(row['maximum_heap_entries'] for row in group))
            standalone.append(total)
    training = [row for row in phases if row['stage'] == 'replay']
    tables = [row for row in phases if row['stage'] == 'probability_table']
    ranking = [row for row in phases if row['stage'] == 'ranking']
    actual = dict(searches=len(rows), search_primitive_ops=sum(row['result']['primitive_ops'] for row in rows),
        search_seconds=sum(row['search_seconds'] for row in rows),
        evaluator_primitive_ops=sum(row['evaluation']['primitive_ops'] for row in rows),
        evaluator_seconds=sum(row['evaluation_seconds'] for row in rows),
        gradient_steps=sum(row['steps'] for row in training),
        replay_target_tokens=sum(row['replay_tokens'] for row in training),
        training_input_rows=sum(row['input_rows'] for row in training),
        training_input_positions=sum(row['input_positions'] for row in training),
        table_builds=len(tables), table_model_calls=sum(row['model_calls'] for row in tables),
        table_recurrent_input_positions=sum(row['recurrent_input_positions'] for row in tables),
        ranking_builds=len(ranking), ranking_queries=sum(row['predictions'] for row in ranking),
        logical_prefix_positions=sum(row['prefix_positions'] for row in ranking),
        checkpoint_bytes=sum(row['bytes'] for row in phases if row['stage'] == 'checkpoint_save'),
        skipped_replay_attempts=sum(row['skip_reason'] == 'empty_corpus' for row in training),
        phase_seconds={stage: sum(row['seconds'] for row in phases if row['stage'] == stage)
                       for stage in sorted({row['stage'] for row in phases})},
        collection_elapsed_seconds=metadata['elapsed_seconds'])
    return dict(status='audited continued local-prior learning with untouched probes', seeds=seeds,
        statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__), primary=primary,
        descriptive_contrasts=contrasts, summaries=aggregates, probe_seed_records=probes,
        prequential_seed_records=prequential, actual_cost=actual,
        standalone_cost_totals=standalone, standalone_seed_costs=costs, checkpoint_records=states,
        interpretation_notes=[
            'Inference averages both orders and families within environments; tasks and repeated probes are not independent units.',
            'Both-order counts describe evaluation conditions; uniform/frozen and checkpoint-zero results explicitly reuse actual searches.',
            'Lower-budget scores rescore first acceptances on the unchanged 65536-budget learning path, not a retrained lower-budget agent.',
            'Standalone totals serve final probes only: replay stream searches/intermediate ranking are diagnostic; online requires stream outcomes.',
            'Standalone sums exclude explicit cloning/loading/checkpoint/table/order archive phases; actual collection retains them.',
            'Replay phase durations include per-step journal writes and flushes; reported seconds are instrumented phase times, not pure training-compute times.',
            'Shared measured phases and reused artifacts are not independent wall-time trials or equal lifetime compute.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        rows, phases, states, metadata, audit = audit_run(args.input)
        summary = summarize(rows, phases, states, metadata)
        audit['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(primary=summary['primary'], final=[row for row in summary['summaries']
            if row['order'] == 'both' and row['checkpoint'] == 16 and row['budget'] == BUDGET]), indent=2))
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
