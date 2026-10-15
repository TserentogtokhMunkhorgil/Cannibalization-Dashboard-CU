from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from expansion_optimization import Costs, Inputs, OptimizationConfig, Evaluator, optimize, result_tables
from expansion_optimization_inputs import kernel, safe_csv


def fixture(h=24, count=3):
    c = pd.DataFrame({'candidate_id': [f'C{i}' for i in range(count)],
        'candidate_name': [f'Site{i}' for i in range(count)], 'region': ['All Regions'] * count,
        'latitude': np.zeros(count), 'longitude': np.arange(count) / 1000})
    e = pd.DataFrame({'store_id': ['E1', 'E2'], 'store_name': ['E1', 'E2'], 'region': ['All Regions'] * 2})
    ke = np.zeros((h, count, 2)); ke[:, :, 0] = -.05
    kn = np.full((h, count, count), -.1)
    kn[:, np.arange(count), np.arange(count)] = 0.
    return Inputs(c, e, np.full((h, 2), 100.), np.full(h, 30.), np.arange(count) * 10 + 50.,
        np.ones(h), ke, kn, np.full((count, 2), 500.), np.full((count, count), 200.), {})


def config(**kwargs):
    return replace(OptimizationConfig(max_stores=3, max_iterations=2, neighborhood_size=2,
        time_budget_seconds=30, seed=7), **kwargs)


class OptimizationTests(unittest.TestCase):
    def test_zero_one_and_reconciliation(self):
        evaluator = Evaluator(fixture(), config())
        self.assertEqual(evaluator.evaluate({})[0], 0.)
        _, monthly, impacts, summary = result_tables(evaluator, {0: 2})
        self.assertEqual(monthly.loc[:1, 'incremental_network_sales'].sum(), 0.)
        self.assertAlmostEqual(summary['incremental_network_sales'], summary['new_store_sales'] + summary['existing_store_change'])
        self.assertAlmostEqual(impacts.horizon_existing_sales_change.sum(), summary['existing_store_change'])
        self.assertEqual(summary['new_to_new_change'], 0.)

    def test_manual_and_auto_counts(self):
        ev = Evaluator(fixture(), config(target=2))
        plan, _, diagnostics = optimize(ev)
        self.assertEqual(len(plan), 2)
        self.assertTrue(ev.feasible(plan))
        auto = Evaluator(fixture(), config())
        plan, _, _ = optimize(auto)
        self.assertEqual(len(plan), 3)

    def test_count_can_choose_zero_for_negative_objective(self):
        data = fixture(); data.predictions[:] = 1.; data.existing_log_kernel[:] = -2.
        plan, _, _ = optimize(Evaluator(data, config()))
        self.assertEqual(plan, {})

    def test_incremental_scores_equal_full_evaluation(self):
        ev = Evaluator(fixture(), config(opening_day=15))
        plan = {0: 3}
        scores = ev.addition_scores(plan)
        before = ev.evaluate(plan)[0]
        for c in (1, 2):
            for m in (0, 3, 11):
                self.assertAlmostEqual(scores[c, m], ev.evaluate({**plan, c: m})[0] - before, places=7)

    def test_new_to_new_and_overlapping_existing(self):
        ev = Evaluator(fixture(), config())
        _, ex, gross, nn, _, _ = ev.evaluate({0: 0, 1: 0})
        self.assertAlmostEqual(ex[0, 0], 3000 * np.expm1(-.1))
        self.assertAlmostEqual(nn.sum(), gross.sum() * np.expm1(-.1))

    def test_readiness_capacity_and_exact_counts(self):
        data = fixture(); data.candidates['earliest_opening_date'] = '2027-06-01'
        ev = Evaluator(data, config(target=2, monthly_max=(1,) * 12))
        plan, _, _ = optimize(ev)
        self.assertTrue(all(m >= 5 for m in plan.values()))
        self.assertEqual(len(set(plan.values())), 2)
        with self.assertRaises(ValueError):
            Evaluator(data, config(target=2, exact_monthly=(1,) + (0,) * 11))
        ev = Evaluator(fixture(), config(target=2, exact_monthly=(1, 1) + (0,) * 10))
        plan, _, _ = optimize(ev)
        self.assertEqual(sorted(plan.values()), [0, 1])

    def test_npv_requires_costs_and_budget(self):
        with self.assertRaises(ValueError):
            Evaluator(fixture(), config(objective='npv'))
        cfg = config(objective='npv', costs=Costs(.2, 500., 10000., .1), budget=20000.)
        ev = Evaluator(fixture(), cfg)
        self.assertFalse(ev.feasible({0: 0, 1: 0, 2: 0}))
        plan, _, _ = optimize(ev)
        self.assertTrue(ev.feasible(plan))
        self.assertEqual(plan, {})

    def test_event_index_fallback_tail_and_distance_zero(self):
        coeff = pd.DataFrame({'region': ['All Regions'] * 2, 'distance_band': ['0_100'] * 2,
                              'age': [0, 1], 'coefficient': [-.1, -.2]})
        # Supply other bands so missing model support is never silently zero.
        coeff = pd.concat([coeff.assign(distance_band=b) for b in ['0_100', '100_250', '250_500', '500_1000']])
        codes = np.array([[0], [-1]])
        values, sources = kernel(coeff, ['City Center'], codes, 3, 'hold_last', 'raw')
        np.testing.assert_allclose(values[:, 0, 0], [-.1, -.2, -.2])
        self.assertTrue((values[:, 1, 0] == 0).all())
        self.assertTrue(any('All Regions' in s for s in sources))
        with self.assertRaises(ValueError):
            kernel(coeff, ['All Regions'], codes, 3, 'error', 'raw')

    def test_reproducibility_and_no_accepted_degradation(self):
        ev = Evaluator(fixture(), config(target=2, starts=2))
        plan, _, diagnostics = optimize(ev, initial_plan={0: 10, 1: 11})
        again, _, _ = optimize(Evaluator(fixture(), config(target=2, starts=2)), initial_plan={0: 10, 1: 11})
        self.assertEqual(plan, again)
        self.assertGreaterEqual(ev.evaluate(plan)[0], ev.evaluate({0: 10, 1: 11})[0])
        self.assertTrue(all(move['after'] > move['before'] for move in diagnostics['accepted_moves']))

    def test_fixed_180_unique_and_capacity(self):
        data = fixture(h=12, count=180)
        cfg = config(target=180, max_stores=180, horizon=12, monthly_max=(15,) * 12, max_iterations=0, time_budget_seconds=60)
        ev = Evaluator(data, cfg)
        plan, _, _ = optimize(ev)
        self.assertEqual(len(plan), 180)
        self.assertTrue(ev.feasible(plan))

    def test_regional_capacity_and_latest_readiness(self):
        data = fixture()
        data.candidates['latest_opening_date'] = '2027-02-01'
        ev = Evaluator(data, config(target=2, monthly_max=(2,) * 12,
            regional_capacity=(('All Regions', 1, 1),)))
        plan, _, _ = optimize(ev)
        self.assertEqual(sorted(plan.values()), [0, 1])
        with self.assertRaises(ValueError):
            Evaluator(data, config(regional_capacity=(('Unknown', 1, 1),)))

    def test_npv_incremental_and_location_swap_improvement(self):
        ev = Evaluator(fixture(), config(objective='npv', costs=Costs(.3, 1., 10., .1)))
        scores = ev.addition_scores({0: 0})
        self.assertAlmostEqual(scores[1, 2], ev.evaluate({0: 0, 1: 2})[0] - ev.evaluate({0: 0})[0], places=6)
        data = fixture(h=12)
        data.predictions[:] = [100., 99., 98.]
        data.existing_log_kernel[:] = 0
        data.new_log_kernel[:] = 0
        data.new_log_kernel[:, 0, 1:] = -5.
        data.new_log_kernel[:, 1:, 0] = -5.
        cfg = config(horizon=12, target=2, exact_monthly=(2,) + (0,) * 11, neighborhood_size=3, max_iterations=4)
        base_ev = Evaluator(data, replace(cfg, max_iterations=0))
        base, _, _ = optimize(base_ev)
        ev = Evaluator(data, cfg)
        improved, _, diagnostics = optimize(ev)
        self.assertGreater(ev.evaluate(improved)[0], base_ev.evaluate(base)[0])
        self.assertTrue(any(move['move'] == 'location_swap' for move in diagnostics['accepted_moves']))


if __name__ == '__main__':
    unittest.main()
