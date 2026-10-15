import copy
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from strength_privacy import norms, population, scores
from joint_strength.pc1 import fit_pc1
from joint_strength.utility import assess, safe_assess
from joint_strength.runner import run, validate_report, fingerprint, write_report

ROOT = Path(__file__).resolve().parents[1]


class UtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = population.population(10101, 'original')
        cls.audit = population.population(10102, 'original')

    def test_corrected_constants_and_unchanged_limits(self):
        with (ROOT / 'fixtures/S1_reference_constants.csv').open() as f:
            for row in csv.DictReader(f):
                sex, lift = row['stratum'], row['lift']
                self.assertEqual(norms.RSD[sex][lift], float(row['residual_scale']))
                self.assertEqual(norms.COEFFICIENTS[sex][lift], tuple(float(row[k]) for k in
                    ('intercept', 'height_cm', 'age_years', 'age_squared', 'log_mass_kg', 'log_mass_squared')))
        with (ROOT / 'fixtures/S4_unchanged_preservation_limits.csv').open() as f:
            expected = {r['metric_key']: float(r['unchanged_limit']) for r in csv.DictReader(f)}
        self.assertEqual(scores.LIMITS, expected)

    def test_copy_with_unrelated_ids_passes_without_row_pairing(self):
        release = copy.deepcopy(self.reference)
        release['ids'] = [f'ARTIFICIAL-COPY-{i}' for i in range(393)]
        before = fingerprint(self.reference)
        result = assess(self.reference, release, self.audit['z'])
        self.assertTrue(result['all_passed'])
        self.assertLess(result['common_audit']['pc1_rmse'], 1e-12)
        self.assertEqual(fingerprint(self.reference), before)
        self.assertNotIn('paired_diagnostics', result)

    def test_permutation_invariance(self):
        order = np.random.default_rng(4).permutation(393)
        ref = self.reference
        release = {k: [v[i] for i in order] if isinstance(v, list) else v[order].copy()
                   for k, v in ref.items()}
        self.assertTrue(assess(ref, release, self.audit['z'])['all_passed'])

    def test_frozen_scale_detects_shift_and_boundary(self):
        from strength_privacy.scores import encode_scores
        for shift, expected in ((0.1, True), (0.100001, False), (0.4, False)):
            release = copy.deepcopy(self.reference)
            encode_scores(release, release['z'] + shift)
            result = assess(self.reference, release, self.audit['z'])
            self.assertEqual(result['checks']['all']['z_mean_max'], expected)
            self.assertAlmostEqual(result['groups']['all']['z_mean_max'], shift, places=12)
            if shift == 0.4:
                self.assertFalse(result['checks']['all']['frozen_pc1_mean'])

    def test_pc1_agrees_with_independent_svd(self):
        x = self.reference['z']
        model = fit_pc1(x)
        u = (x - x.mean(0)) / x.std(0, ddof=1)
        _, singular, vh = np.linalg.svd(u, full_matrices=False)
        weight = vh[0] * (1 if vh[0].sum() >= 0 else -1)
        raw = u @ weight
        np.testing.assert_allclose(model.transform(x), (raw - raw.mean()) / raw.std(ddof=1), atol=1e-12)
        self.assertAlmostEqual(model.variance_fraction, singular[0] ** 2 / np.sum(singular ** 2))
        outside = self.audit['z'] + 2
        self.assertGreater(abs(model.transform(outside).mean()), 1)

    def test_degenerate_pc1_rejected(self):
        with self.assertRaises(ValueError):
            fit_pc1(np.ones((30, 3)))
        with self.assertRaises(ValueError):
            fit_pc1(np.vstack([np.eye(3), -np.eye(3)]))

    def test_inconsistent_raw_strength_and_scores_rejected(self):
        release = copy.deepcopy(self.reference)
        release['z'] += 0.2
        self.assertEqual(safe_assess(self.reference, release, self.audit['z'])['status'], 'FAILED_EVALUATION')

    def test_missingness_and_invalid_audit_are_visible(self):
        release = copy.deepcopy(self.reference)
        for sex in ('female', 'male'):
            idx = np.flatnonzero(np.array(release['sex']) == sex)[:30]
            release['strength'][idx, 1] = np.nan
            release['z'][idx, 1] = np.nan
        result = assess(self.reference, release, self.audit['z'])
        self.assertFalse(result['checks']['all']['missing_fraction_change'])
        self.assertEqual(safe_assess(self.reference, self.reference, np.full((4, 3), np.nan))['status'], 'FAILED_EVALUATION')


class JointTests(unittest.TestCase):
    def test_both_branches_receive_identical_release_and_report_is_complete(self):
        calls = []
        from joint_strength import runner
        original_assess = runner.safe_assess
        def utility(reference, release, audit):
            calls.append(('utility', id(release), fingerprint(release)))
            return original_assess(reference, release, audit)
        def privacy(members, controls, release, *args):
            calls.append(('privacy', id(release), fingerprint(release)))
            # Make an explicit non-evaluable fixture, never a fabricated risk PASS.
            return {'frozen': {}, 'adapted': {}}
        with patch.object(runner, 'safe_assess', side_effect=utility), patch.object(runner.design, 'evaluate', side_effect=privacy):
            report = run(models=('original',), cohorts=1, releases=1, levels=('exact',))
        self.assertEqual(len(report['records']), 6)
        for i in range(0, len(calls), 2):
            self.assertEqual(calls[i][1:], calls[i+1][1:])
        self.assertEqual(report['privacy_acceptance'], 'NOT_DEFINED')
        self.assertTrue(all(r['privacy']['exact']['status'] == 'NOT_EVALUABLE' for r in report['records']))
        self.assertTrue(all(r['utility']['all_passed'] for r in report['records'] if r['kind'] == 'copy_positive'))
        validate_report(report)
        broken = copy.deepcopy(report)
        broken['records'].pop()
        with self.assertRaises(ValueError):
            validate_report(broken)
        broken = copy.deepcopy(report)
        broken['records'][0]['utility']['checks']['all'].pop('z_tail_max')
        with self.assertRaises(ValueError):
            validate_report(broken)
        broken = copy.deepcopy(report)
        broken['records'][0]['reference_seed'] += 1
        with self.assertRaises(ValueError):
            validate_report(broken)

    def test_changed_corrected_fixture_is_rejected(self):
        from joint_strength import runner
        with patch.dict(scores.LIMITS, {'z_mean_max': .2}):
            with self.assertRaises(ValueError):
                runner.inputs()

    def test_no_input_mutation_by_attack_allowed(self):
        from joint_strength import runner
        def bad_attack(members, controls, release, *args):
            release['z'][0, 0] += 1
            return {'frozen': {}, 'adapted': {}}
        with patch.object(runner.design, 'evaluate', side_effect=bad_attack):
            with self.assertRaises(RuntimeError):
                run(models=('original',), cohorts=1, releases=1, levels=('exact',))

    def test_invalid_config_does_not_start_generation(self):
        from joint_strength import runner
        for kwargs in ({'cohorts': True}, {'releases': 0}, {'models': ('unknown',)},
                       {'levels': ('exact', 'exact')}, {'cohorts': 41}):
            with patch.object(runner.population, 'population') as make:
                with self.assertRaises(ValueError):
                    run(**kwargs)
                make.assert_not_called()

    def test_output_cannot_overwrite_existing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'existing.json'
            target.write_text('keep')
            with self.assertRaises(FileExistsError):
                write_report({}, target)
            self.assertEqual(target.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
