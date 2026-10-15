import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from regional_cumulative_impact import (
    build_regional_pairs, aggregate_portfolio, prepare_consistent_baseline,
    fixed_lookup, FIXED_INCOME, FIXED_CUSTOMERS,
    apply_pair_mode,
)


class PlanningPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stores = pd.DataFrame({
            'existing_store_id': ['S1', 'S2', 'S3'], 'store_name': ['One', 'Two', 'Missing'],
            'region': ['City Center'] * 3, 'baseline_month_or_period': ['2026-09'] * 3,
            'baseline_APSD_income': [100., 300., 50.], 'baseline_APSD_customers': [10., 30., 5.],
            'baseline_valid': [True] * 3, 'latitude': [0., 1., np.nan], 'longitude': [0., 1., np.nan],
        })
        self.candidates = pd.DataFrame({
            'candidate_id': ['A', 'B'], 'candidate_rank': [1, 2],
            'candidate_name': ['A', 'B'], 'latitude': [0., 0.], 'longitude': [.0001, .0015],
        })
        self.selected = pd.DataFrame({'candidate_id': ['A', 'B'], 'selection_order': [1, 2]})

    def pairs(self):
        return build_regional_pairs(self.stores, self.candidates,
                                    fixed_lookup(FIXED_INCOME), fixed_lookup(FIXED_CUSTOMERS))

    def test_multiple_pairs_zero_exposure_and_weighted_identity(self):
        result, pairs, summary = aggregate_portfolio(self.stores, self.pairs(), self.selected, 2)
        first = result.iloc[0]
        self.assertEqual(first.selected_pair_count, 2)
        expected = np.expm1(-.0628 - .0069) * 100
        self.assertAlmostEqual(first.cumulative_predicted_APSD_income_effect_pct, expected)
        self.assertEqual(result.iloc[1].cumulative_predicted_APSD_income_effect_pct, 0)
        self.assertAlmostEqual(summary['APSD_weighted_network_effect_pct'], expected / 4)
        self.assertAlmostEqual(summary['network_effect_validation_difference_pct_points'], 0)
        self.assertTrue(np.isnan(result.iloc[2].cumulative_predicted_APSD_customers_effect_pct))
        self.assertEqual(result.iloc[2].customer_model_estimate_status, 'missing_or_invalid_store_coordinates')

    def test_cutoff_is_enforced_again_at_aggregation(self):
        pairs = self.pairs()
        pairs.loc[pairs.candidate_id.eq('B'), 'distance_m'] = 1001.
        result, _, _ = aggregate_portfolio(self.stores, pairs, self.selected, 2)
        self.assertEqual(result.iloc[0].selected_pair_count, 1)
        with self.assertRaises(ValueError):
            aggregate_portfolio(self.stores, pd.concat([pairs, pairs]), self.selected, 2)

    def test_missing_coefficient_is_not_zero(self):
        pairs = build_regional_pairs(self.stores, self.candidates, {}, {})
        result, _, _ = aggregate_portfolio(self.stores, pairs, self.selected, 2)
        self.assertTrue(np.isnan(result.iloc[0].cumulative_predicted_APSD_income_effect_pct))
        self.assertEqual(result.iloc[0].model_estimate_status, 'missing_regional_income_estimate')

    def test_raw_positive_customer_pair_and_negative_only_are_distinct(self):
        self.stores.loc[0, 'region'] = 'City Outskirts'
        self.candidates['longitude'] = [.006, .007]
        pairs = self.pairs()
        raw, _, _ = aggregate_portfolio(self.stores, apply_pair_mode(pairs, 'raw'), self.selected, 2)
        planning, _, _ = aggregate_portfolio(self.stores, apply_pair_mode(pairs, 'negative_only'), self.selected, 2)
        self.assertAlmostEqual(raw.iloc[0].cumulative_predicted_APSD_customers_effect_pct, 100 * np.expm1(.0084))
        self.assertEqual(planning.iloc[0].cumulative_predicted_APSD_customers_effect_pct, 0.)
        self.assertTrue(pairs.regional_customer_log_coefficient.eq(.0042).all())

    def test_saved_portfolio_identity_and_coordinates_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        original = root / 'outputs' / 'cannibalization_model4'
        frozen = root / 'Data' / 'planning_inputs' / 'accepted_v1'
        for name in ('candidates', 'selected_180', 'selected_400'):
            self.assertEqual((frozen / (name + '.csv')).read_bytes(),
                (original / ('cumulative_cannibalization_' + name + '.csv')).read_bytes())

    def test_completed_baseline_and_duplicate_rejection(self):
        panel = pd.DataFrame({
            'existing_store_id': ['1', '1'], 'SalesMonth': ['2026-09-01', '2026-10-01'],
            'APSD_income': [100., 200.], 'LATITUDE': [0., 0.], 'LONGITUDE': [0., 0.],
            'loc_area_new': ['Street', 'Street'],
        })
        master = pd.DataFrame({'store_id': ['1', '1'], 'SalesMonth': panel.SalesMonth,
            'APSD_customers': [10., 20.], 'Distance_Zone': ['Core', 'Core'], 'UB': ['UB', 'UB']})
        result = prepare_consistent_baseline(panel, master, as_of='2026-10-15')
        self.assertEqual(result.baseline_month_or_period.iloc[0], '2026-09')
        self.assertEqual(result.baseline_APSD_income.iloc[0], 100.)
        with self.assertRaises(ValueError):
            prepare_consistent_baseline(pd.concat([panel, panel]), master, as_of='2026-10-15')

    def test_preparation_imports_do_not_read_or_query(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('01 Cleaning StoreMaster.py', '02 Store distances.py', '02 gs25.py'):
            spec = importlib.util.spec_from_file_location('safe_preparation', root / name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
