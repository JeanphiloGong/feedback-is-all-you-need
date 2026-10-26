"""Audit before paired learning-curve inference and explicit lifetime costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_m.audit import audit_run, read_records
from experiments.study_m.run import ARMS, CHECKPOINTS, INTERACTIONS
from experiments.study_m.tasks import FAMILIES, FORMAL_SEEDS


def curve(rows, arm):
    points = (0,) if arm.endswith('frozen') or arm == 'uniform' else CHECKPOINTS
    selected = [[r for r in rows if r['arm'] == arm and r['at'] == at] for at in points]
    rates = [mean(r['success'] for r in group) for group in selected]
    counts = [sum(r['success'] for r in group) for group in selected]
    if len(points) == 1:
        rates *= len(CHECKPOINTS)
        counts *= len(CHECKPOINTS)
    auc = sum((right - left) * (rates[i] + rates[i + 1]) / 2
              for i, (left, right) in enumerate(zip(CHECKPOINTS, CHECKPOINTS[1:]))) / INTERACTIONS
    return dict(arm=arm, checkpoints=list(CHECKPOINTS), success_rates=rates, success_counts=counts,
                tasks_per_checkpoint=len(selected[0]), new_probe_evaluations=sum(map(len, selected)),
                auc=auc, final_minus_initial=rates[-1] - rates[0])


def paired_inference(differences, confirmatory):
    result = dict(differences=differences, environments=len(differences), mean=mean(differences),
        confirmatory=confirmatory, ci95=None, bootstrap95=None, positive_primary=False,
        negative=sum(d < 0 for d in differences), zero=sum(d == 0 for d in differences),
        positive=sum(d > 0 for d in differences))
    if confirmatory:
        error = stdev(differences) / len(differences) ** .5
        half = float(t.ppf(.975, len(differences) - 1)) * error
        boot = np.random.default_rng(983).choice(differences, (10000, len(differences))).mean(axis=1)
        result.update(ci95=[result['mean'] - half, result['mean'] + half],
            bootstrap95=np.quantile(boot, [.025, .975]).tolist(), degenerate=error == 0,
            positive_primary=result['mean'] - half > 0)
        result['interval_disagreement'] = ((result['ci95'][0] > 0) != (result['bootstrap95'][0] > 0)
                                           or (result['ci95'][1] < 0) != (result['bootstrap95'][1] < 0))
    return result


def standalone_cost(phases, arm):
    stages = ('branch_setup', 'initialization', 'warmup', 'training', 'graph', 'action')
    initial = arm.split('_')[0] + '_initial'
    selected = [p for p in phases if p['stage'] in stages and
                (p['arm'] == arm or p['arm'] == initial or (p['arm'] == 'warmup' and arm != 'uniform'))
                and (p['stage'] != 'graph' or p['input_rows'] > 0)]
    result = {name: sum(p.get(name, 0) for p in selected) for name in
              ('environment_actions', 'gradient_steps', 'dense_training_macs', 'dense_forward_macs',
               'target_bits', 'edge_checks', 'expanded_states')}
    result.update(training_input_rows=sum(p['input_rows'] for p in selected if p['stage'] == 'training'),
        graph_input_rows=sum(p['input_rows'] for p in selected if p['stage'] == 'graph'),
        instrumented_phase_seconds=sum(p['seconds'] for p in selected),
        included_journal_seconds=sum(p.get('journal_seconds', 0.) for p in selected),
        included_journal_bytes=sum(p.get('journal_bytes', 0) for p in selected))
    result.update({'main_action_' + name: sum(p.get(name, 0.) for p in selected) for name in
                   ('planning_seconds', 'environment_seconds', 'observation_seconds')})
    return result


def summarize(root, worlds, phases, metadata, audit):
    curves, streams, costs, bounds = [], [], [], []
    for world in worlds:
        seed, family = world['seed'], world['family']
        folder = root / f'seed-{seed}' / family
        probes = read_records(folder / 'probes.jsonl')
        actions = read_records(folder / 'actions.jsonl')
        oracle = read_records(folder / 'oracle.jsonl')
        selected = [p for p in phases if (p['seed'], p['family']) == (seed, family)]
        bounds.append(dict(seed=seed, family=family, tasks=len(oracle),
            reachable_within_budget=sum(r['reachable_within_budget'] for r in oracle),
            maximum_exact_distance=max(r['distance'] for r in oracle)))
        for arm in ARMS:
            key = dict(seed=seed, family=family, arm=arm)
            curves.append(dict(seed=seed, family=family, **curve(probes, arm)))
            rows = [r for r in actions if r['arm'] == arm]
            predicted = [r for r in rows if r['predicted_next_state'] is not None]
            initial = next(c for c in audit['checkpoint_records'] if
                (c['seed'], c['family'], c['arm'], c['at']) == (seed, family, arm, 0))
            final = next(c for c in audit['checkpoint_records'] if
                (c['seed'], c['family'], c['arm'], c['at']) == (seed, family, arm, INTERACTIONS))
            streams.append(dict(**key, actions=len(rows), episodes_ended=sum(r['episode_ended'] for r in rows),
                successful_episodes=sum(r['episode_success'] for r in rows),
                fallback_actions=sum(r['fallback'] for r in rows),
                unfinished_episode=not rows[-1]['episode_ended'],
                prediction_observations=len(predicted),
                pre_update_exact_accuracy=mean(r['bit_errors'] == 0 for r in predicted) if predicted else None,
                pre_update_bit_accuracy=1 - sum(r['bit_errors'] for r in predicted) / (5 * len(predicted)) if predicted else None,
                initial_transition_coverage=initial['replay_entries'], final_transition_coverage=final['replay_entries']))
            batches = [p for p in selected if p['arm'] == arm and p['stage'] == 'training']
            costs.append(dict(**key, **standalone_cost(selected, arm),
                main_update_batches=len(batches),
                batches_changing_weights=sum(p['changed_parameter_values'] > 0 for p in batches),
                final_model_tensor_bytes=final['model_tensor_bytes'], final_adam_tensor_bytes=final['adam_tensor_bytes'],
                final_replay_json_bytes=final['replay_json_bytes'], final_graph_json_bytes=final['graph_json_bytes']))
    aggregates = []
    for family in FAMILIES:
        for arm in ARMS:
            group = [r for r in curves if (r['family'], r['arm']) == (family, arm)]
            aggregates.append(dict(family=family, arm=arm, environments=len(group),
                checkpoints=list(CHECKPOINTS), auc=mean(r['auc'] for r in group),
                success_rates=[mean(r['success_rates'][i] for r in group) for i in range(len(CHECKPOINTS))],
                success_counts=[sum(r['success_counts'][i] for r in group) for i in range(len(CHECKPOINTS))],
                tasks_per_checkpoint=sum(r['tasks_per_checkpoint'] for r in group),
                final_minus_initial=mean(r['final_minus_initial'] for r in group)))
    contrasts = []
    for family in FAMILIES:
        for left, right in (('small_immediate', 'large_immediate'),
                            ('small_immediate', 'small_delayed'), ('large_immediate', 'large_delayed'),
                            ('small_immediate', 'small_frozen'), ('large_immediate', 'large_frozen'),
                            ('small_immediate', 'observed_table'), ('small_immediate', 'uniform')):
            for metric in ('auc', 'final'):
                differences = []
                for seed in metadata['seeds']:
                    values = {r['arm']: r['auc'] if metric == 'auc' else r['success_rates'][-1]
                              for r in curves if (r['seed'], r['family']) == (seed, family)}
                    differences.append(values[left] - values[right])
                contrasts.append(dict(family=family, contrast=f'{left} minus {right}', metric=metric,
                    differences=differences, mean=mean(differences)))
    primary = dict(family='structured', contrast='small_immediate minus large_immediate', metric='auc',
        **paired_inference(contrasts[0]['differences'], metadata['seeds'] == list(FORMAL_SEEDS)))
    actual = dict(**{name: audit[name] for name in ('main_actions', 'gradient_steps', 'checkpoints', 'probe_tasks')},
        environment_actions=sum(p.get('environment_actions', 0) for p in phases),
        dense_training_macs=sum(p.get('dense_training_macs', 0) for p in phases),
        dense_forward_macs=sum(p.get('dense_forward_macs', 0) for p in phases),
        journal_seconds=sum(p.get('journal_seconds', 0) for p in phases),
        journal_bytes=sum(p.get('journal_bytes', 0) for p in phases),
        checkpoint_file_bytes=sum(p['bytes'] for p in phases if p['stage'] == 'checkpoint'),
        collection_seconds=metadata['elapsed_seconds'],
        main_action_planning_seconds=sum(p.get('planning_seconds', 0) for p in phases),
        main_action_environment_seconds=sum(p.get('environment_seconds', 0) for p in phases),
        main_action_observation_seconds=sum(p.get('observation_seconds', 0) for p in phases),
        phase_seconds={stage: sum(p['seconds'] for p in phases if p['stage'] == stage)
                       for stage in sorted({p['stage'] for p in phases})})
    return dict(status='audited retained predictive learning', seeds=metadata['seeds'],
        statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__), primary=primary,
        summaries=aggregates, descriptive_contrasts=contrasts, seed_curves=curves,
        stream_summaries=streams, true_graph_bounds=bounds, standalone_costs=costs, actual_cost=actual,
        interpretation_notes=[
            'World-seed pairs are the inferential units; only structured AUC small immediate minus large immediate is primary.',
            'The primary changes model width and gradient allocation under matched training matrix MACs, not full compute or time.',
            'Frozen and uniform probes are executed once and reused across checkpoints.',
            'Held-out goal cues are unused during training; the same physical states can occur as observations.',
            'All agents use current-state feedback; neural contrasts concern weight adaptation.',
            'State encoding, replay memory, BFS, and resets supply capability beyond neural parameters.',
            'Standalone costs charge necessary shared warmup and width-specific initialization once per arm/world.',
            'Archive and probe costs remain in actual totals; included journal costs are separated from instrumented phases.',
            'Pure graph archives perform no inference and are excluded from standalone costs.',
            'Main-action subtimings exclude warmup/probes, whose combined phase times remain in actual totals.',
            'Phase sums are not independent wall-time benchmarks; remaining bookkeeping stays in collection elapsed time.',
            'Tensor and serialized JSON sizes are not total resident memory or inference RAM.',
            'Missing superiority does not establish equivalence; no unrestricted intelligence claim follows.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        worlds, phases, metadata, audit = audit_run(args.input)
        summary = summarize(args.input, worlds, phases, metadata, audit)
        audit['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(primary=summary['primary'], summaries=summary['summaries']), indent=2))
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
