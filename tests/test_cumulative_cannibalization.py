import math
import unittest

import numpy as np
import pandas as pd

from cumulative_cannibalization import (
    calculate_cumulative_scenarios,
    load_model4_month_zero_coefficients,
    load_regional_month_zero_coefficients,
    prepare_candidates,
)


class CumulativeCannibalizationTests(unittest.TestCase):
    def test_candidate_order_and_duplicate_locations(self):
        workbook = pd.DataFrame(
            {
                "#": [11, 10, 12, 13],
                "Potential Point": ["duplicate", "first", "second", "invalid"],
                "Longitude": [0.0, 0.0, 0.001, 200.0],
                "Latitude": [0.0, 0.0, 0.0, 0.0],
            }
        )
        candidates = prepare_candidates(workbook)
        self.assertEqual(candidates["candidate_rank"].tolist(), [1, 2])
        self.assertEqual(
            candidates["candidate_name"].tolist(), ["first", "second"]
        )
        self.assertEqual(
            candidates["candidate_source_number"].tolist(), [10, 12]
        )

    def test_requires_exact_model_d_event_month_zero_coefficients(self):
        coefficients = pd.DataFrame(
            [
                {
                    "model": "D",
                    "term": f"exposure_{band}_event_p0",
                    "coefficient": -0.1,
                }
                for band in ("0_100", "100_250", "250_500", "500_1000")
            ]
        )
        self.assertEqual(
            set(load_model4_month_zero_coefficients(coefficients)),
            {"0_100", "100_250", "250_500", "500_1000"},
        )
        with self.assertRaisesRegex(ValueError, "one finite event-month-0"):
            load_model4_month_zero_coefficients(coefficients.iloc[:-1])

    def test_existing_regional_event_study_supports_income_and_customers(self):
        coefficients = pd.DataFrame(
            [
                {
                    "region": "City Center",
                    "model_type": "binned_event_study",
                    "event_period": "p0",
                    "distance_band": band,
                    "coefficient": -0.1,
                    "outcome_scale": "log",
                }
                for band in ("0_100", "100_250", "250_500", "500_1000")
            ]
        )
        regional = load_regional_month_zero_coefficients(
            coefficients, "sales"
        )
        self.assertEqual(regional[("City Center", "0_100")], -0.1)
        with self.assertRaisesRegex(ValueError, "not all on the log scale"):
            load_regional_month_zero_coefficients(
                coefficients.assign(outcome_scale="percent"),
                "sales",
            )

        stores = pd.DataFrame(
            [
                {
                    "existing_store_id": "S1",
                    "region": "City Center",
                    "APSD_income_baseline": 1_000_000.0,
                    "APSD_customers_baseline": 100.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "A",
                    "candidate_rank": 1,
                    "candidate_name": "near 50m",
                    "longitude": 50 / 111_195,
                    "latitude": 0.0,
                },
                {
                    "candidate_id": "B",
                    "candidate_rank": 2,
                    "candidate_name": "near 150m",
                    "longitude": 150 / 111_195,
                    "latitude": 0.0,
                },
            ]
        )
        regional_sales = {
            ("City Center", "0_100"): math.log(0.95),
            ("City Center", "100_250"): math.log(0.9),
        }
        regional_customers = {
            ("City Center", "0_100"): math.log(0.98),
            ("City Center", "100_250"): math.log(0.97),
        }
        results, _ = calculate_cumulative_scenarios(
            stores,
            candidates,
            {
                "0_100": 0.0,
                "100_250": 0.0,
                "250_500": 0.0,
                "500_1000": 0.0,
            },
            scenario_sizes=(2,),
            regional_sales_coefficients=regional_sales,
            regional_customer_coefficients=regional_customers,
        )
        result = results.iloc[0]
        self.assertAlmostEqual(
            result["regional_APSD_income_change_pct_2"],
            (0.95 * 0.9 - 1) * 100,
            places=5,
        )
        self.assertAlmostEqual(
            result["APSD_customers_change_pct_2"],
            (0.98 * 0.97 - 1) * 100,
            places=5,
        )
        self.assertTrue(result["customer_effect_supported_2"])

    def test_effects_sum_on_log_scale_and_each_candidate_counts_once(self):
        stores = pd.DataFrame(
            [
                {
                    "existing_store_id": "S1",
                    "store_name": "CU S1",
                    "region": "City Center",
                    "loc_area_new": "Street",
                    "APSD_baseline_month": "2026-10-01",
                    "APSD_income_baseline": 1_000_000.0,
                    "APSD_customers_baseline": 100.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "A",
                    "candidate_rank": 1,
                    "candidate_name": "near 50m",
                    "longitude": 50 / 111_195,
                    "latitude": 0.0,
                },
                {
                    "candidate_id": "B",
                    "candidate_rank": 2,
                    "candidate_name": "near 150m",
                    "longitude": 150 / 111_195,
                    "latitude": 0.0,
                },
            ]
        )
        coefficients = {
            "0_100": math.log(0.9),
            "100_250": math.log(0.8),
            "250_500": 0.0,
            "500_1000": 0.0,
        }
        results, contributions = calculate_cumulative_scenarios(
            stores, candidates, coefficients, scenario_sizes=(1, 2)
        )
        result = results.iloc[0]
        self.assertEqual(result["nearby_new_stores_1"], 1)
        self.assertEqual(result["nearby_new_stores_2"], 2)
        self.assertEqual(len(contributions), 2)
        self.assertAlmostEqual(
            result["APSD_income_change_pct_2"],
            (math.exp(math.log(0.9) + math.log(0.8)) - 1) * 100,
            places=5,
        )
        self.assertAlmostEqual(
            result["APSD_income_change_2"],
            1_000_000 * (math.exp(math.log(0.9) + math.log(0.8)) - 1),
            places=5,
        )
        self.assertAlmostEqual(
            result["APSD_income_expected_2"],
            1_000_000 * math.exp(math.log(0.9) + math.log(0.8)),
            places=5,
        )
        self.assertTrue(np.isnan(result["APSD_customers_change_pct_2"]))

    def test_zero_openings_has_zero_incremental_effect(self):
        stores = pd.DataFrame(
            [
                {
                    "existing_store_id": "S1",
                    "APSD_income_baseline": 10.0,
                    "APSD_customers_baseline": 2.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "A",
                    "candidate_rank": 1,
                    "candidate_name": "near",
                    "longitude": 0.0,
                    "latitude": 0.0,
                }
            ]
        )
        results, _ = calculate_cumulative_scenarios(
            stores,
            candidates,
            {band: -0.1 for band in ("0_100", "100_250", "250_500", "500_1000")},
            scenario_sizes=(0, 1),
        )
        self.assertEqual(results.iloc[0]["nearby_new_stores_0"], 0)
        self.assertEqual(results.iloc[0]["APSD_income_change_pct_0"], 0)
        self.assertEqual(results.iloc[0]["cannibalization_pct_0"], 0)


if __name__ == "__main__":
    unittest.main()
