import copy
import unittest
import numpy as np
from strength_privacy import population
from joint_strength import study
from joint_strength.utility import safe_assess


class StudyTests(unittest.TestCase):
    def test_bootstrap_keeps_counts_and_scores_but_uses_draw_ids(self):
        ref = population.population(3, 'original')
        sample = study.bootstrap(ref, 9)
        self.assertEqual(sample['sex'].count('female'), 197)
        self.assertEqual(sample['sex'].count('male'), 196)
        self.assertEqual(len(set(sample['ids'])), 393)
        self.assertLess(len(set(map(tuple, sample['z']))), 393)
        result = safe_assess(ref, sample, population.population(4, 'original')['z'])
        self.assertEqual(result['status'], 'ARTIFICIAL_CANDIDATE_SCREEN')

    def test_uncertainty_uses_cohort_means_not_individual_releases(self):
        values = [.1, .2, .3]
        result = study.estimate(values)
        self.assertEqual(result['n_cohorts'], 3)
        self.assertAlmostEqual(result['mean'], .2)
        self.assertAlmostEqual(result['cohort_sd'], .1)
        self.assertGreater(result['t95'][1], .4)
        self.assertEqual(study.estimate([.1])['t95'], None)

    def test_zero_pass_bootstrap_not_claimed_as_zero_upper_risk(self):
        result = study.estimate([0.] * 10, probability=True)
        self.assertEqual(result['bootstrap95'], [0., 0.])
        self.assertGreater(result['cohort_any_positive_exact95'][1], .3)
        self.assertIn('different endpoint', result['zero_boundary_note'])

    def test_utility_denominator_keeps_failed_evaluations(self):
        ref = population.population(8, 'original')
        good = safe_assess(ref, ref, population.population(9, 'original')['z'])
        bad = {'status': 'FAILED_EVALUATION', 'all_passed': False}
        result = study.utility_summary([good, bad])
        self.assertEqual(result['n_expected'], 2)
        self.assertEqual(result['failures'], 1)
        self.assertEqual(result['all_pass_fraction'], .5)
        self.assertEqual(result['criterion_pass_fractions']['z_tail_max'], .5)

    def test_primary_configuration_is_fixed_and_complete(self):
        self.assertEqual(study.COHORTS, 10)
        self.assertEqual(study.RELEASES, 100)
        self.assertEqual(study.NATURAL_REPLICAS, 100)
        self.assertEqual(study.NEGATIVES, 10)
        seeds = study.role_seeds(0)
        self.assertEqual(len(set(seeds.values())), 4)
        self.assertNotIn(91000000, seeds.values())


if __name__ == '__main__':
    unittest.main()
