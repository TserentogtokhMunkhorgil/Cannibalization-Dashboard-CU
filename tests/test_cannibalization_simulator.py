import math
import unittest

import pandas as pd

from cannibalization_simulator import (
    DISTANCE_BANDS,
    maximum_openings,
    regional_sales_estimates,
    simulate_mixed_scenario,
    validate_sales_inputs,
)


def _coefficient_frame(
    *,
    region="City Center",
    band="0_100",
    beta=-0.1,
    lower=-0.2,
    upper=0.0,
):
    return pd.DataFrame(
        [
            {
                "region": region,
                "outcome": "sales",
                "model_type": "cumulative",
                "distance_band": band,
                "coefficient": beta,
                "percent_effect": 100 * math.expm1(beta),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "percent_ci_95_lower": 100 * math.expm1(lower),
                "percent_ci_95_upper": 100 * math.expm1(upper),
                "outcome_scale": "log",
                "exposed_stores": 12,
                "observations": 100,
            }
        ]
    )


class CannibalizationSimulatorTests(unittest.TestCase):
    def test_four_million_baseline_three_million_minimum(self):
        beta = math.log(0.9)
        result = maximum_openings(4_000_000, 3_000_000, beta)
        self.assertEqual(result.maximum_openings, 2)
        self.assertGreaterEqual(result.predicted_at_maximum, 3_000_000)
        self.assertLess(result.predicted_at_next, 3_000_000)

    def test_minimum_equal_to_baseline_allows_zero_negative_effect_openings(self):
        result = maximum_openings(4_000_000, 4_000_000, math.log(0.9))
        self.assertEqual(result.maximum_openings, 0)
        self.assertEqual(result.predicted_at_maximum, 4_000_000)
        self.assertLess(result.predicted_at_next, 4_000_000)

    def test_missing_regional_coefficient_is_not_estimable(self):
        estimates = regional_sales_estimates(
            _coefficient_frame(region="City Center"), "Rural"
        )
        self.assertEqual(len(estimates), len(DISTANCE_BANDS))
        self.assertIsNone(estimates["0_100"].beta)
        self.assertEqual(
            maximum_openings(4_000_000, 3_000_000, estimates["0_100"].beta).status,
            "Not estimable",
        )

    def test_zero_effect_has_no_finite_maximum(self):
        result = maximum_openings(4_000_000, 3_000_000, 0.0)
        self.assertEqual(result.status, "No finite maximum under this model")
        self.assertIsNone(result.maximum_openings)

    def test_positive_effect_has_no_finite_maximum(self):
        result = maximum_openings(4_000_000, 3_000_000, 0.05)
        self.assertEqual(result.status, "No finite maximum under this model")

    def test_negative_effect_has_a_verified_finite_maximum(self):
        result = maximum_openings(4_000_000, 3_000_000, math.log(0.8))
        self.assertEqual(result.status, "Finite capacity")
        self.assertGreaterEqual(result.predicted_at_maximum, 3_000_000)
        self.assertLess(result.predicted_at_next, 3_000_000)

    def test_mixed_distance_effects_are_added_on_the_log_scale(self):
        beta_close = math.log(0.9)
        beta_far = math.log(0.95)
        result = simulate_mixed_scenario(
            4_000_000,
            3_000_000,
            {"0_100": 1, "100_250": 2},
            {"0_100": beta_close, "100_250": beta_far},
        )
        expected = 4_000_000 * math.exp(beta_close + 2 * beta_far)
        self.assertEqual(result.total_openings, 3)
        self.assertAlmostEqual(result.predicted_apsd, expected)
        self.assertAlmostEqual(result.absolute_change, expected - 4_000_000)
        self.assertAlmostEqual(result.percentage_change, (expected / 4_000_000 - 1) * 100)
        self.assertAlmostEqual(result.headroom, expected - 3_000_000)
        self.assertEqual(result.constraint_satisfied, expected >= 3_000_000)

    def test_maximum_openings_boundary_count_meets_and_next_count_fails(self):
        beta = math.log(0.9)
        result = maximum_openings(4_000_000, 3_000_000, beta)
        self.assertEqual(result.maximum_openings, 2)
        self.assertAlmostEqual(result.predicted_at_maximum, 4_000_000 * 0.9**2)
        self.assertAlmostEqual(result.predicted_at_next, 4_000_000 * 0.9**3)

    def test_minimum_above_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_sales_inputs(3_000_000, 4_000_000)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            maximum_openings(3_000_000, 4_000_000, -0.1)

    def test_missing_or_invalid_confidence_intervals_are_not_used(self):
        missing = _coefficient_frame()
        missing["ci_95_lower"] = float("nan")
        missing["ci_95_upper"] = float("nan")
        missing["percent_ci_95_lower"] = float("nan")
        missing["percent_ci_95_upper"] = float("nan")
        estimate = regional_sales_estimates(missing, "City Center")["0_100"]
        self.assertIsNotNone(estimate.beta)
        self.assertIsNone(estimate.confidence_lower)
        self.assertIsNone(estimate.confidence_upper)
        self.assertIn("confidence interval", estimate.issue)

        invalid = _coefficient_frame(lower=0.1, upper=0.2)
        invalid["percent_ci_95_lower"] = float("nan")
        invalid["percent_ci_95_upper"] = float("nan")
        estimate = regional_sales_estimates(invalid, "City Center")["0_100"]
        self.assertIsNone(estimate.confidence_lower)
        self.assertIsNone(estimate.confidence_upper)


if __name__ == "__main__":
    unittest.main()
