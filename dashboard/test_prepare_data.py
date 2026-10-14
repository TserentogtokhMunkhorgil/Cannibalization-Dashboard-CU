import unittest

import pandas as pd

from prepare_data import prepare_datasets


def _sales_row(
    store_id,
    date,
    opening_date,
    apsd,
    customers,
    latitude=47.9,
    longitude=106.9,
    status="Active",
    closing_date=None,
):
    return {
        "store_id": store_id,
        "store_name": f"CU {store_id}",
        "SalesMonth": date,
        "APSD_income": apsd,
        "Customers": customers,
        "ost_dt": opening_date,
        "cls_dt": closing_date,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "Store_Status": status,
        "loc_area_new": "Test",
    }


def _distance_row(existing_id, new_id, distance, existing_open, new_open):
    return {
        "existing_store_id": existing_id,
        "new_store_id": new_id,
        "distance_m": distance,
        "existing_opening_date": existing_open,
        "new_opening_date": new_open,
        "existing_store_name": f"CU {existing_id}",
        "new_store_name": f"CU {new_id}",
        "new_latitude": 47.9,
        "new_longitude": 106.9,
    }


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.sales = pd.DataFrame(
            [
                _sales_row(1, "2025-01-01", "2024-12-01", 0, 0),
                _sales_row(1, "2025-02-01", "2024-12-01", 100, 10),
                _sales_row(1, "2025-03-01", "2024-12-01", 110, 11),
                _sales_row(2, "2025-01-01", "2025-02-15", 50, 5),
                _sales_row(2, "2025-02-01", "2025-02-15", 200, 40),
                _sales_row(2, "2025-03-01", "2025-02-15", 220, 44),
                _sales_row(3, "2025-03-01", "2025-03-10", 150, 30),
                _sales_row(
                    4,
                    "2025-01-01",
                    "2024-12-01",
                    75,
                    8,
                    latitude=None,
                    longitude=None,
                ),
            ]
        )
        self.distances = pd.DataFrame(
            [
                _distance_row(1, 2, 100, "2024-12-01", "2025-02-15"),
                _distance_row(1, 3, 250, "2024-12-01", "2025-03-10"),
                _distance_row(2, 3, 100, "2025-02-15", "2025-03-10"),
                _distance_row(2, 4, 100, "2025-02-15", "2025-02-15"),
            ]
        )

    def test_operating_period_mom_and_new_store_exposure(self):
        panel, exposure, quality = prepare_datasets(self.sales, self.distances)
        rows = panel.set_index(["store_id", "date"])

        self.assertFalse(rows.loc[("2", pd.Timestamp("2025-01-01")), "is_operating"])
        self.assertTrue(rows.loc[("2", pd.Timestamp("2025-02-01")), "is_operating"])
        self.assertTrue(rows.loc[("3", pd.Timestamp("2025-03-01")), "is_operating"])
        self.assertTrue(pd.isna(rows.loc[("2", pd.Timestamp("2025-02-01")), "previous_APSD"]))
        self.assertTrue(
            pd.isna(rows.loc[("1", pd.Timestamp("2025-02-01")), "APSD_change_pct"])
        )
        self.assertEqual(rows.loc[("1", pd.Timestamp("2025-02-01")), "APSD_change"], 100)
        self.assertEqual(
            rows.loc[("1", pd.Timestamp("2025-03-01")), "nearby_new_stores_250m"],
            2,
        )
        self.assertEqual(
            rows.loc[("2", pd.Timestamp("2025-03-01")), "nearby_new_stores_250m"],
            1,
        )
        self.assertTrue(
            pd.isna(
                rows.loc[
                    ("4", pd.Timestamp("2025-03-01")),
                    "nearby_new_stores_250m",
                ]
            )
        )

        event = exposure.set_index("new_store_id").loc["3"]
        self.assertEqual(event["potential_stores_250m"], 2)
        self.assertEqual(event["potential_apsd_exposure_250m"], 300)
        self.assertEqual(event["potential_apsd_observed_stores_250m"], 2)
        self.assertEqual(
            quality.set_index("check").loc[
                "source_rows_before_opening", "value"
            ],
            1,
        )
        self.assertEqual(
            quality.set_index("check").loc[
                "distance_pairs_not_existing_before_new_opening", "value"
            ],
            1,
        )
        self.assertNotIn("4", exposure["new_store_id"].tolist())

    def test_duplicate_store_month_records_are_rejected(self):
        duplicate = pd.concat([self.sales, self.sales.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "not unique at store-month grain"):
            prepare_datasets(duplicate, self.distances)


if __name__ == "__main__":
    unittest.main()
