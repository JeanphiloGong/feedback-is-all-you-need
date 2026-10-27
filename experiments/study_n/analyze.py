"""Audit first, then estimate encoding-dependent learning and its costs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t

from experiments.study_n.audit import audit_run, file_hashes, read_document, records, same
from experiments.study_n.run import CHECKPOINTS, ENCODINGS, FORMAL_SEEDS, WIDTHS, document


def paired_inference(differences, confirmatory):
    result = dict(differences=differences, environments=len(differences), mean=mean(differences),
        confirmatory=confirmatory, ci95=None, bootstrap95=None, positive_primary=False,
        negative=sum(d < 0 for d in differences), zero=sum(d == 0 for d in differences),
        positive=sum(d > 0 for d in differences))
    if confirmatory:
        error = stdev(differences) / len(differences) ** .5
        half = float(t.ppf(.975, len(differences) - 1)) * error
        boot = np.random.default_rng(991).choice(differences, (10000, len(differences))).mean(axis=1)
        result.update(ci95=[result['mean'] - half, result['mean'] + half],
            bootstrap95=np.quantile(boot, [.025, .975]).tolist(), degenerate=error == 0,
            positive_primary=result['mean'] - half > 0)
        result['interval_disagreement'] = ((result['ci95'][0] > 0) != (result['bootstrap95'][0] > 0)
                                           or (result['ci95'][1] < 0) != (result['bootstrap95'][1] < 0))
    return result


def direct_curve(rows, encoding, arm):
    selected = [next(r for r in rows if (r['encoding'], r['arm'], r['at']) ==
                    (encoding, arm, 0 if arm.endswith('frozen') else at)) for at in CHECKPOINTS]
    rates = [r['exact'] / 64 for r in selected]
    return dict(encoding=encoding, arm=arm, checkpoints=list(CHECKPOINTS), exact_rates=rates,
        bit_rates=[r['bit_correct'] / 320 for r in selected],
        observed_rates=[r['observed_exact'] / r['observed_count'] if r['observed_count'] else None for r in selected],
        final_minus_initial=rates[-1] - rates[0], tests_per_checkpoint=64,
        newly_evaluated_pairs=64 if arm.endswith('frozen') else 256)


def learning_contrast(curves, size):
    canonical = next(r for r in curves if (r['encoding'], r['arm']) == ('canonical', size + '_online'))
    relabeled = next(r for r in curves if (r['encoding'], r['arm']) == ('relabeled', size + '_online'))
    return dict(size=size, canonical_gain=canonical['final_minus_initial'], relabeled_gain=relabeled['final_minus_initial'],
        canonical_final=canonical['exact_rates'][-1], relabeled_final=relabeled['exact_rates'][-1],
        final_difference=canonical['exact_rates'][-1] - relabeled['exact_rates'][-1],
        gain_difference=canonical['final_minus_initial'] - relabeled['final_minus_initial'])


def standalone_cost(phases, encoding, size):
    arm = size + '_online'
    selected = [p for p in phases if p['stage'] == 'query'
        or (p['stage'] == 'initialization' and p['arm'] == size)
        or (p['stage'] == 'observation' and p['encoding'] == encoding)
        or (p['stage'] in ('branch_setup', 'prediction', 'training')
            and (p['encoding'], p['arm']) == (encoding, arm))]
    training = [p for p in selected if p['stage'] == 'training']
    final = next(p for p in phases if p['stage'] == 'checkpoint'
                 and (p['encoding'], p['arm'], p['at']) == (encoding, arm, 64))
    seconds = sum(p['seconds'] for p in selected)
    journals = sum(p.get('journal_seconds', 0.) for p in selected)
    return dict(encoding=encoding, arm=arm,
        **{name: sum(p.get(name, 0) for p in selected) for name in
           ('environment_actions', 'gradient_steps', 'training_macs', 'forward_macs', 'target_bits')},
        training_rows=sum(p['input_rows'] for p in training),
        prediction_rows=sum(p['input_rows'] for p in selected if p['stage'] == 'prediction'),
        update_batches=len(training), batches_changing_weights=sum(p['changed_parameters'] > 0 for p in training),
        instrumented_phase_seconds=seconds, included_journal_seconds=journals,
        seconds_excluding_journal_io=seconds - journals,
        **{name: final[name] for name in ('model_bytes', 'adam_bytes', 'replay_json_bytes')})


def summarize(root, phases, report):
    metadata = read_document(root / 'metadata.json')
    curves, goals, references, contrasts, costs, bounds = [], [], [], [], [], []
    arms = ('small_online', 'large_online', 'small_frozen', 'large_frozen')
    for seed in report['seeds']:
        folder = root / f'seed-{seed}'
        direct = list(records(folder / 'direct.jsonl'))
        selected = [p for p in phases if p['seed'] == seed]
        seed_curves = [dict(seed=seed, **direct_curve(direct, encoding, arm)) for encoding in ENCODINGS for arm in arms]
        curves.extend(seed_curves)
        for size, _, _ in WIDTHS:
            contrasts.append(dict(seed=seed, **learning_contrast(seed_curves, size)))
            costs.extend(dict(seed=seed, **standalone_cost(selected, encoding, size)) for encoding in ENCODINGS)
        for encoding in ENCODINGS:
            for arm in (*arms, 'observed_table'):
                points = [next(p for p in selected if p['stage'] == 'probes' and
                    (p['encoding'], p['arm'], p['at']) == (encoding, arm, 0 if arm.endswith('frozen') else at))
                    for at in CHECKPOINTS]
                goals.append(dict(seed=seed, encoding=encoding, arm=arm, checkpoints=list(CHECKPOINTS),
                    success_rates=[p['successes'] / p['tasks'] for p in points],
                    tasks_per_checkpoint=992, newly_evaluated_tasks=992 if arm.endswith('frozen') else 3968))
        for row in records(folder / 'reference.jsonl'):
            references.append(dict(seed=seed, at=row['at'], exact_rate=row['exact'] / 64,
                coverage=row['covered'] / 64, comparisons=row['comparisons'],
                surviving_rules=sum(len(rules) for action in row['candidates'] for rules in action),
                permutation_completion_expectation=row['permutation_completion_expectation']))
        distances = read_document(folder / 'distances.json')[:32]
        values = [distance for r in distances for goal, distance in enumerate(r['distances']) if goal != r['start']]
        bounds.append(dict(seed=seed, tasks=len(values), maximum_distance=max(values),
                           reachable_within_eight=sum(distance <= 8 for distance in values)))
    aggregates, goal_aggregates = [], []
    for encoding in ENCODINGS:
        for arm in arms:
            selected = [c for c in curves if (c['encoding'], c['arm']) == (encoding, arm)]
            averages = {name: [mean(c[name][i] for c in selected) if selected[0][name][i] is not None else None
                        for i in range(4)] for name in ('exact_rates', 'bit_rates', 'observed_rates')}
            aggregates.append(dict(encoding=encoding, arm=arm, environments=len(selected), **averages,
                                   final_minus_initial=mean(c['final_minus_initial'] for c in selected)))
        for arm in (*arms, 'observed_table'):
            selected = [c for c in goals if (c['encoding'], c['arm']) == (encoding, arm)]
            goal_aggregates.append(dict(encoding=encoding, arm=arm,
                success_rates=[mean(c['success_rates'][i] for c in selected) for i in range(4)]))
    contrast_aggregates = [dict(size=size, **{name: mean(c[name] for c in contrasts if c['size'] == size)
        for name in ('canonical_gain', 'relabeled_gain', 'canonical_final', 'relabeled_final', 'final_difference', 'gain_difference')})
        for size, _, _ in WIDTHS]
    primary = paired_inference([c['gain_difference'] for c in contrasts if c['size'] == 'small'],
                               report['seeds'] == list(FORMAL_SEEDS))
    primary.update(estimand='small canonical-minus-relabeled difference in exact-accuracy learning gains',
                   **{k: v for k, v in contrast_aggregates[0].items() if k != 'size'})
    symbolic = [dict(at=at, **{name: mean(r[name] for r in references if r['at'] == at)
               for name in ('exact_rate', 'coverage', 'comparisons', 'surviving_rules', 'permutation_completion_expectation')})
               for at in CHECKPOINTS]
    actual = dict(**report['counts'], collection_seconds=metadata['elapsed_seconds'],
        **{name: sum(p.get(name, 0) for p in phases) for name in
           ('environment_actions', 'training_macs', 'forward_macs', 'dense_forward_macs',
            'edge_checks', 'expanded_states', 'target_bits')},
        prediction_rows=sum(p['input_rows'] for p in phases if p['stage'] == 'prediction'),
        training_rows=sum(p['input_rows'] for p in phases if p['stage'] == 'training'),
        evaluation_graph_rows=sum(p['input_rows'] for p in phases if p['stage'] == 'evaluation_graph'),
        update_batches=sum(p['stage'] == 'training' for p in phases),
        batches_changing_weights=sum(p['stage'] == 'training' and p['changed_parameters'] > 0 for p in phases),
        checkpoint_file_bytes=sum(p['bytes'] for p in phases if p['stage'] == 'checkpoint'),
        known_learning_journal_seconds=sum(p.get('journal_seconds', 0.) for p in phases),
        phase_seconds={stage: sum(p['seconds'] for p in phases if p['stage'] == stage)
                       for stage in sorted({p['stage'] for p in phases})})
    return dict(status='audited encoding and unqueried-transition learning', seeds=report['seeds'],
        statistics_packages=dict(numpy=np.__version__, scipy=scipy.__version__), checkpoints=list(CHECKPOINTS),
        primary=primary, direct_summaries=aggregates, goal_summaries=goal_aggregates,
        symbolic_reference=symbolic, observed_table_unqueried_exact_rates=[0.] * 4,
        descriptive_contrasts=contrast_aggregates, seed_contrasts=contrasts, seed_direct_curves=curves,
        seed_goal_curves=goals, seed_references=references, true_graph_bounds=bounds,
        standalone_learning_costs=costs, actual_cost=actual, interpretation_notes=[
            'One primary: paired small-model difference in learning gains over world seeds; all other comparisons descriptive.',
            'Identical physical evidence and sampling indices; supplied encodings induce different neural geometry.',
            'Positive contrast alone does not imply either encoding improves: gains and final accuracies are reported.',
            'Unqueried pairs share state vocabulary and dynamics with observed pairs; no unseen-world claim follows.',
            'Frozen evaluations run once and are reused; an observed-edge table abstains on every unqueried pair.',
            'The canonical symbolic matcher has a stronger supplied family prior; permutation completion is an exact expectation.',
            'BFS and live state feedback supply goal capability beyond learned weights; learners cannot train on probes.',
            'Standalone online-learning costs charge shared width initialization, physical queries, and encoded insertions once per learner.',
            'Known journal I/O is subtracted; sampling digests and some instrumentation remain, so this is not a pure deployment benchmark.',
            'Archive, graph, and probe work appear only in actual totals; phase sums exclude uninstrumented bookkeeping.',
            'Equal training matrix MACs do not imply equal total compute or wall time; storage counts are not resident RAM.',
            'No open-ended intelligence, human-intelligence explanation, equivalence, or universal capacity conclusion is tested.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        report, phases = audit_run(args.input)
        audit_seconds = time.perf_counter() - started
        summary = summarize(args.input, phases, report)
        same(file_hashes(args.input), report['files'])
        report['elapsed_seconds'] = audit_seconds
        summary['analysis_elapsed_seconds'] = time.perf_counter() - started - audit_seconds
        document(args.output / 'audit.json', report)
        document(args.output / 'summary.json', summary)
        print(json.dumps(dict(primary=summary['primary'], direct_summaries=summary['direct_summaries']), indent=2))
    except BaseException as error:
        document(args.output / 'failure.json', dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
