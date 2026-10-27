"""Hand-computed gain contrasts, paired inference, and cost attribution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from experiments.study_n import analyze


class AnalysisTests(unittest.TestCase):
    def test_direct_curves_separate_initial_final_and_observed_fit(self):
        rows = [dict(encoding=e, arm='small_online', at=at, exact=exact, bit_correct=5 * exact,
                     observed_count=at, observed_exact=at // 2)
                for e, counts in [('canonical', [16, 24, 32, 48]), ('relabeled', [8, 16, 24, 32])]
                for at, exact in zip((0, 16, 32, 64), counts)]
        curves = [analyze.direct_curve(rows, e, 'small_online') for e in ('canonical', 'relabeled')]
        self.assertEqual(curves[0]['exact_rates'], [.25, .375, .5, .75])
        self.assertEqual(curves[0]['observed_rates'], [None, .5, .5, .5])
        contrast = analyze.learning_contrast(curves, 'small')
        self.assertEqual(contrast['canonical_gain'], .5)
        self.assertEqual(contrast['relabeled_gain'], .375)
        self.assertEqual(contrast['gain_difference'], .125)
        self.assertEqual(contrast['final_difference'], .25)
        frozen = analyze.direct_curve([dict(rows[0], arm='small_frozen')], 'canonical', 'small_frozen')
        self.assertEqual(frozen['exact_rates'], [.25] * 4)
        self.assertEqual(frozen['final_minus_initial'], 0.)
        self.assertEqual(frozen['newly_evaluated_pairs'], 64)

    def test_inference_resamples_world_differences_and_handles_degeneracy(self):
        values = [i / 64 for i in range(-16, 16)]
        actual = analyze.paired_inference(values, True)
        delta = np.array(values)
        half = t.ppf(.975, 31) * delta.std(ddof=1) / np.sqrt(32)
        np.testing.assert_allclose(actual['ci95'], [delta.mean() - half, delta.mean() + half])
        indices = np.random.default_rng(991).integers(0, 32, size=(10000, 32))
        bootstrap = delta[indices].mean(axis=1)
        np.testing.assert_allclose(actual['bootstrap95'], np.percentile(bootstrap, [2.5, 97.5]))
        self.assertEqual((actual['negative'], actual['zero'], actual['positive']), (16, 1, 15))
        for value in (0., .125, -.125):
            actual = analyze.paired_inference([value] * 32, True)
            self.assertEqual(actual['ci95'], [value, value])
            self.assertEqual(actual['positive_primary'], value > 0)
            self.assertTrue(actual['degenerate'])
        self.assertIsNone(analyze.paired_inference([.5], False)['ci95'])

    def test_learning_cost_includes_shared_work_once_and_removes_known_io(self):
        phases = [dict(stage='initialization', arm='small', seconds=2.),
            dict(stage='initialization', arm='large', seconds=200.),
            dict(stage='query', seconds=3., journal_seconds=1., environment_actions=64),
            dict(stage='observation', encoding='canonical', seconds=5.),
            dict(stage='observation', encoding='relabeled', seconds=500.)]
        for e in ('canonical', 'relabeled'):
            for size in ('small', 'large'):
                common = dict(encoding=e, arm=size + '_online')
                multiplier = 1 if (e, size) == ('canonical', 'small') else 100
                phases.extend([
                    dict(**common, stage='branch_setup', seconds=7. * multiplier),
                    dict(**common, stage='prediction', seconds=11. * multiplier, journal_seconds=2.,
                         input_rows=64, forward_macs=28672),
                    dict(**common, stage='training', seconds=13. * multiplier, journal_seconds=3.,
                         gradient_steps=512, input_rows=16384, target_bits=81920,
                         training_macs=17301504, changed_parameters=42),
                    dict(**common, stage='checkpoint', at=64, seconds=1700., model_bytes=3880,
                         adam_bytes=7776, replay_json_bytes=500),
                    dict(**common, stage='evaluation_graph', seconds=1900., dense_forward_macs=100),
                    dict(**common, stage='probes', seconds=2300., environment_actions=9999)])
        cost = analyze.standalone_cost(phases, 'canonical', 'small')
        self.assertEqual(cost['environment_actions'], 64)
        self.assertEqual(cost['gradient_steps'], 512)
        self.assertEqual(cost['training_macs'], 17301504)
        self.assertEqual(cost['forward_macs'], 28672)
        self.assertEqual(cost['instrumented_phase_seconds'], 41.)
        self.assertEqual(cost['included_journal_seconds'], 6.)
        self.assertEqual(cost['seconds_excluding_journal_io'], 35.)
        self.assertEqual(cost['batches_changing_weights'], 1)
        self.assertEqual(cost['adam_bytes'], 7776)

    def test_failed_audit_cannot_produce_a_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'analysis'
            with patch.object(sys, 'argv', ['analyze', '--input', 'unused', '--output', str(output)]), \
                    patch.object(analyze, 'audit_run', side_effect=AssertionError('invalid evidence')), \
                    patch.object(analyze, 'summarize') as summarize, self.assertRaises(AssertionError):
                analyze.main()
            summarize.assert_not_called()
            self.assertFalse((output / 'summary.json').exists())
            self.assertEqual(json.loads((output / 'failure.json').read_text())['message'], 'invalid evidence')

    def test_post_audit_raw_change_cannot_produce_a_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / 'raw', Path(temporary) / 'analysis'
            root.mkdir()
            record = root / 'metadata.json'
            record.write_text('{}\n')
            report = dict(files=analyze.file_hashes(root))
            def changed(*args):
                record.write_text('{"changed": true}\n')
                return {}
            with patch.object(sys, 'argv', ['analyze', '--input', str(root), '--output', str(output)]), \
                    patch.object(analyze, 'audit_run', return_value=(report, [])), \
                    patch.object(analyze, 'summarize', side_effect=changed), self.assertRaises(AssertionError):
                analyze.main()
            self.assertFalse((output / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
