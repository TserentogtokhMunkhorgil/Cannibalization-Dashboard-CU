import unittest

import numpy as np
import pandas as pd

from expansion_simulation import (
    DISTANCE_BANDS,
    _pair_impact_rows,
    _portfolio_schedule,
    distance_band_codes,
    haversine_distances,
    schedule_months,
    schedule_weights,
)


class ExpansionSimulationTests(unittest.TestCase):
    def test_haversine_uses_longitude_latitude_and_returns_meters(self):
        distance = haversine_distances([0.0], [0.0], [1.0], [0.0])[0, 0]
        self.assertAlmostEqual(distance, 111_195, delta=100)

    def test_distance_band_boundaries_match_model_definitions(self):
        codes = distance_band_codes(
            np.array([0.0, 99.9, 100.0, 250.0, 500.0, 999.9, 1000.0])
        )
        np.testing.assert_array_equal(codes, [0, 0, 1, 2, 3, 3, -1])

    def test_monthly_schedule_allocates_exact_target(self):
        months = schedule_months(180, schedule_weights("monthly_even"))
        np.testing.assert_array_equal(np.bincount(months, minlength=12), [15] * 12)

    def test_quarterly_schedule_and_empty_schedule(self):
        months = schedule_months(7, schedule_weights("quarterly_even"))
        self.assertEqual(set(months), {0, 3, 6, 9})
        empty_months, empty_dates = _portfolio_schedule(
            0, 2027, "monthly_even", None, 15
        )
        self.assertEqual(len(empty_months), 0)
        self.assertTrue(isinstance(empty_dates, pd.DatetimeIndex))

    def test_pair_impacts_retain_real_candidate_to_existing_distance(self):
        coefficient_lookup = {
            (band, event_time): -0.1
            for band, _, _ in DISTANCE_BANDS
            for event_time in range(12)
        }
        beta_models = {
            outcome: {
                "global": coefficient_lookup,
                "regional": {},
                "regions": ("All Regions",),
            }
            for outcome in ("sales", "customers")
        }
        candidates = pd.DataFrame({"potential_store_id": ["P-1"]})
        existing = pd.DataFrame(
            {"store_id": ["E-1"], "analysis_region": ["All Regions"]}
        )
        distances = pd.DataFrame(
            {
                "potential_store_id": ["P-1"],
                "existing_store_id": ["E-1"],
                "distance_m": [75.0],
            }
        )
        impacts = _pair_impact_rows(
            candidates=candidates,
            existing=existing,
            selected=[0],
            dates=[pd.Timestamp("2027-01-15")],
            existing_bands=np.array([[0]]),
            existing_distances=distances,
            baseline={
                "sales": np.full((12, 1), 100.0),
                "customers": np.full((12, 1), 10.0),
            },
            active_days=np.full(12, 30.0),
            beta_models=beta_models,
            forecast_year=2027,
            scale=1.0,
            target=1,
        )
        self.assertEqual(len(impacts), 12)
        self.assertTrue(impacts["distance_m"].eq(75.0).all())
        self.assertTrue(
            np.allclose(
                impacts["APSD_income_percent_effect"],
                100.0 * np.expm1(-0.1),
            )
        )


if __name__ == "__main__":
    unittest.main()
