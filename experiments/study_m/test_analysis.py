"""Hand-computed learning curves, paired inference, and standalone costs."""

import unittest

import numpy as np
from scipy.stats import t

from experiments.study_m.analyze import curve, paired_inference, standalone_cost


class AnalysisTests(unittest.TestCase):
    def test_trapezoidal_curve_uses_interaction_spacing_and_static_reuse(self):
        rows = [dict(arm='small_immediate', at=at, success=i < correct)
                for at, correct in zip((0, 64, 128, 256, 512), (0, 1, 2, 3, 4)) for i in range(4)]
        value = curve(rows, 'small_immediate')
        self.assertEqual(value['success_rates'], [0, .25, .5, .75, 1])
        self.assertEqual(value['auc'], .65625)
        frozen = curve([dict(arm='small_frozen', at=0, success=i < 3) for i in range(4)], 'small_frozen')
        self.assertEqual(frozen['success_rates'], [.75] * 5)
        self.assertEqual(frozen['auc'], .75)
        self.assertEqual(frozen['new_probe_evaluations'], 4)

    def test_paired_intervals_and_bootstrap_use_environment_differences(self):
        differences = [i / 100 for i in range(-16, 16)]
        actual = paired_inference(differences, True)
        delta = np.array(differences)
        half = t.ppf(.975, 31) * delta.std(ddof=1) / np.sqrt(32)
        np.testing.assert_allclose(actual['ci95'], [delta.mean() - half, delta.mean() + half])
        bootstrap = np.random.default_rng(983).choice(delta, (10000, 32)).mean(axis=1)
        np.testing.assert_allclose(actual['bootstrap95'], np.quantile(bootstrap, [.025, .975]))
        self.assertFalse(actual['positive_primary'])
        for value, positive in ((0., False), (.125, True), (-.125, False)):
            result = paired_inference([value] * 32, True)
            self.assertTrue(result['degenerate'])
            self.assertEqual(result['ci95'], [value, value])
            self.assertEqual(result['positive_primary'], positive)
        development = paired_inference([.75], False)
        self.assertIsNone(development['ci95'])

    def test_standalone_cost_charges_shared_initialization_once(self):
        phases = [dict(seed=-1, family='structured', arm=arm, stage=stage, seconds=1.,
                       journal_seconds=.25, journal_bytes=11, **fields)
                  for arm, stage, fields in (
                      ('warmup', 'warmup', dict(environment_actions=16)),
                      ('small_initial', 'initialization', {}),
                      ('small_initial', 'training', dict(gradient_steps=128, dense_training_macs=100, input_rows=4096)),
                      ('small_initial', 'graph', dict(dense_forward_macs=13, input_rows=128)),
                      ('small_initial', 'checkpoint', dict(bytes=99999)),
                      ('small_immediate', 'branch_setup', {}),
                      ('small_immediate', 'graph', dict(input_rows=0, dense_forward_macs=0)),
                      ('small_immediate', 'action', dict(environment_actions=1, edge_checks=4)),
                      ('small_immediate', 'training', dict(gradient_steps=8, dense_training_macs=10, input_rows=256)),
                      ('small_immediate', 'probe', dict(environment_actions=8)),
                      ('small_immediate', 'checkpoint', dict(bytes=99999)),
                      ('uniform', 'branch_setup', {}),
                      ('uniform', 'graph', dict(input_rows=0, dense_forward_macs=0)),
                      ('uniform', 'action', dict(environment_actions=1)),
                  )]
        value = standalone_cost(phases, 'small_immediate')
        self.assertEqual(value['environment_actions'], 17)
        self.assertEqual(value['gradient_steps'], 136)
        self.assertEqual(value['dense_training_macs'], 110)
        self.assertEqual(value['dense_forward_macs'], 13)
        self.assertEqual(value['instrumented_phase_seconds'], 7.)
        self.assertEqual(value['included_journal_seconds'], 1.75)
        uniform = standalone_cost(phases, 'uniform')
        self.assertEqual(uniform['environment_actions'], 1)
        self.assertEqual(uniform['gradient_steps'], 0)
        self.assertEqual(uniform['instrumented_phase_seconds'], 2.)


if __name__ == '__main__':
    unittest.main()
