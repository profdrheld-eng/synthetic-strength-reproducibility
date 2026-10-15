import unittest
import numpy as np
from diagnose import tail_rows, failure_keys, nonlinear_relationships, population, relationships


class DiagnosticsTests(unittest.TestCase):
    def test_tail_direction_and_order_invariance(self):
        a = {'sex': ['female'] * 10 + ['male'] * 10,
             'z': np.arange(60, dtype=float).reshape(20, 3)}
        b = {'sex': a['sex'][::-1], 'z': (a['z'] + .25)[::-1]}
        rows = tail_rows(a, b)
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(abs(r['delta'] - .25) < 1e-12 for r in rows))
        self.assertTrue(all(r['exceeded'] for r in rows))

    def test_exact_boundary_tolerance_and_negative_shift(self):
        a = {'sex': ['female'] * 10 + ['male'] * 10,
             'z': np.arange(60, dtype=float).reshape(20, 3)}
        b = {'sex': a['sex'], 'z': a['z'] - .2}
        self.assertTrue(all(not r['exceeded'] for r in tail_rows(a, b)))
        self.assertTrue(all(r['delta'] < 0 for r in tail_rows(a, b)))

    def test_failure_types_collapse_groups_only(self):
        checks = {'female': {'z_tail_max': False, 'z_mean_max': True},
                  'male': {'z_tail_max': False}, 'common': {'weight_max': True}}
        self.assertEqual(failure_keys(checks), {'z_tail_max'})

    def test_nonlinear_self_comparison_is_zero(self):
        a = population.population(111000000, 'original')
        rows = nonlinear_relationships(a, a)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['absolute_change'] == 0 for r in rows.values()))

    def test_known_quadratic_signal_is_detected(self):
        a = population.population(111000000, 'original')
        a['z'] = np.repeat(relationships.features(a)[:, 3, None], 3, axis=1)
        b = {**a, 'z': -a['z']}
        rows = nonlinear_relationships(a, b)
        for key, row in rows.items():
            if 'age_quadratic' in key:
                self.assertAlmostEqual(row['reference'], 1.)
                self.assertAlmostEqual(row['release'], -1.)
                self.assertAlmostEqual(row['absolute_change'], 2.)


if __name__ == '__main__':
    unittest.main()
