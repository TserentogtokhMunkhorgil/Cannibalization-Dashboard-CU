import unittest
import numpy as np
from regional_cumulative_impact import _bands, fixed_lookup, FIXED_INCOME, FIXED_CUSTOMERS


class FixedCoefficientTests(unittest.TestCase):
    def test_boundaries_are_unique_and_inclusive(self):
        np.testing.assert_array_equal(
            _bands(np.array([0., 100., 100.01, 250., 500., 1000., 1000.01])),
            ['0_100', '0_100', '100_250', '100_250', '250_500', '500_1000', ''])

    def test_user_example_and_fallback(self):
        lookup = fixed_lookup(FIXED_INCOME)
        total = sum(lookup[('City Center', band)] for band in ['0_100', '100_250', '250_500'])
        self.assertAlmostEqual(total, -.0753)
        self.assertAlmostEqual(4_000_000 * np.exp(total), 4_000_000 * np.exp(-.0753))
        self.assertEqual(lookup[('City Outskirts', '0_100')], -.1893)
        self.assertEqual(lookup[('Far / Outer Areas', '0_100')], -.3565)

    def test_positive_customer_estimate_is_preserved(self):
        beta = fixed_lookup(FIXED_CUSTOMERS)[('City Outskirts', '500_1000')]
        self.assertEqual(beta, .0042)
        self.assertGreater(np.expm1(beta), 0)
        self.assertEqual(np.expm1(min(beta, 0)), 0)
