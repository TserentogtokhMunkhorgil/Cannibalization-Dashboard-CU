"""Joint count/location/month search using one reconciled network evaluator.

No historical regressions or accepted portfolios are changed. All search results
are feasible heuristics, not certificates of global optimality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd

VERSION = 'joint-expansion-v2.0'


@dataclass(frozen=True)
class Costs:
    margin: float
    monthly_operating_cost: float
    investment: float
    annual_discount_rate: float

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Cost assumptions must be finite.')
        if not 0 < self.margin <= 1 or min(self.monthly_operating_cost, self.investment, self.annual_discount_rate) < 0:
            raise ValueError('Supply a margin in (0,1], nonnegative costs and discount rate.')


@dataclass(frozen=True)
class OptimizationConfig:
    as_of: str = field(default_factory=lambda: str(pd.Timestamp.now(tz='Asia/Irkutsk').date()))
    planning_year: int = 2027
    horizon: int = 24
    target: int | None = None
    min_stores: int = 0
    max_stores: int = 400
    candidate_limit: int | None = None
    monthly_min: tuple[int, ...] = (0,) * 12
    monthly_max: tuple[int, ...] = (40,) * 12
    exact_monthly: tuple[int, ...] | None = None
    regional_capacity: tuple[tuple[str, int, int], ...] = ()  # region, calendar month 1..12, cap
    objective: str = 'sales'
    costs: Costs | None = None
    budget: float | None = None
    minimum_marginal: float = 0.
    demand_scenario: str = 'central'
    coefficient_source: str = 'fixed_cumulative'
    coefficient_mode: str = 'negative_only'
    tail_assumption: str = 'error'  # error, hold_last, zero
    maturity_tail: str = 'hold_last'  # explicitly assumes age-11 maturity after month 11
    opening_day: int = 1
    max_iterations: int = 8
    time_budget_seconds: float = 60.
    neighborhood_size: int = 8
    starts: int = 1
    seed: int = 2027
    tolerance: float = 1.

    def validate(self, candidate_count: int):
        if any(isinstance(v, bool) or int(v) != v for v in (self.min_stores, self.max_stores)):
            raise ValueError('Store-count bounds must be integers.')
        if self.target is not None and (isinstance(self.target, bool) or int(self.target) != self.target):
            raise ValueError('Manual store count must be an integer.')
        for values in (self.monthly_min, self.monthly_max):
            if len(values) != 12 or any(isinstance(v, bool) or int(v) != v or v < 0 for v in values):
                raise ValueError('Monthly constraints require twelve nonnegative integer counts.')
        if any(a > b for a, b in zip(self.monthly_min, self.monthly_max)):
            raise ValueError('Monthly minimum exceeds maximum.')
        if self.exact_monthly is not None:
            if len(self.exact_monthly) != 12 or any(int(v) != v or v < 0 for v in self.exact_monthly):
                raise ValueError('Exact schedule requires twelve nonnegative integers.')
            if any(v < lo or v > hi for v, lo, hi in zip(self.exact_monthly, self.monthly_min, self.monthly_max)):
                raise ValueError('Exact schedule violates monthly bounds.')
            if self.target is not None and sum(self.exact_monthly) != self.target:
                raise ValueError('Exact allocations must sum to the manual target.')
        if not 0 <= self.min_stores <= self.max_stores <= candidate_count:
            raise ValueError('Store-count bounds exceed available eligible candidates.')
        if self.target is not None and not self.min_stores <= self.target <= self.max_stores:
            raise ValueError('Manual target is outside configured count bounds.')
        if self.horizon not in (12, 24):
            raise ValueError('Evaluation horizon must be 12 or 24 months.')
        if not 1 <= self.opening_day <= 28:
            raise ValueError('Opening day must be 1..28.')
        if self.objective not in ('sales', 'npv') or self.coefficient_mode not in ('raw', 'negative_only'):
            raise ValueError('Unsupported objective or coefficient mode.')
        if self.demand_scenario not in ('central', 'lower', 'upper') or self.coefficient_source not in ('fixed_cumulative', 'monthly_event_study'):
            raise ValueError('Unsupported demand or regression source.')
        if self.tail_assumption not in ('error', 'hold_last', 'zero') or self.maturity_tail not in ('hold_last', 'one'):
            raise ValueError('Unsupported explicit horizon assumption.')
        if self.costs:
            self.costs.validate()
        if (self.objective == 'npv' or self.budget is not None) and self.costs is None:
            raise ValueError('NPV/budget requires supplied cost and margin assumptions.')
        if self.budget is not None and (not np.isfinite(self.budget) or self.budget < 0):
            raise ValueError('Budget must be finite and nonnegative.')
        if self.starts < 1 or self.max_iterations < 0 or self.neighborhood_size < 1 or not np.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0:
            raise ValueError('Search limits must be positive (iterations may be zero).')
        if not np.isfinite(self.minimum_marginal) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('Contribution threshold and tolerance must be finite.')
        for region, month, cap in self.regional_capacity:
            if not region or not 1 <= month <= 12 or int(cap) != cap or cap < 0:
                raise ValueError('Invalid regional monthly capacity.')
        lower, upper = self.count_bounds()
        if lower > upper:
            raise ValueError('Count, capacity and budget constraints are infeasible.')

    def month_bounds(self):
        return (self.exact_monthly, self.exact_monthly) if self.exact_monthly is not None else (self.monthly_min, self.monthly_max)

    def count_bounds(self):
        lo, hi = self.month_bounds()
        lower = self.target if self.target is not None else max(self.min_stores, sum(lo))
        upper = self.target if self.target is not None else min(self.max_stores, sum(hi))
        upper = min(upper, sum(hi))
        if sum(lo) > lower and self.target is not None:
            return 1, 0
        if self.budget is not None and self.costs and self.costs.investment > 0:
            upper = min(upper, int(self.budget // self.costs.investment))
        return lower, upper


@dataclass
class Inputs:
    candidates: pd.DataFrame
    existing: pd.DataFrame
    baseline_sales: np.ndarray  # horizon × existing, MNT per active day
    active_days: np.ndarray
    predictions: np.ndarray
    maturity: np.ndarray
    existing_log_kernel: np.ndarray  # source age × candidate × existing
    new_log_kernel: np.ndarray  # source age × candidate × receiving candidate
    existing_distances: np.ndarray
    pair_distances: np.ndarray
    provenance: dict


class Evaluator:
    """Exact evaluation and incremental add/removal scores share the same kernels."""
    def __init__(self, inputs: Inputs, config: OptimizationConfig):
        self.data, self.config = inputs, config
        self.candidates = inputs.candidates.reset_index(drop=True)
        self.c = len(self.candidates)
        config.validate(self.c)
        self.h = config.horizon
        self.e = len(inputs.existing)
        if inputs.baseline_sales.shape != (self.h, self.e):
            raise ValueError('Existing baseline dimensions do not match forecast calendar.')
        for field in (inputs.baseline_sales, inputs.active_days, inputs.predictions, inputs.maturity,
                      inputs.existing_log_kernel, inputs.new_log_kernel):
            if not np.isfinite(field).all():
                raise ValueError('Forecast and coefficient arrays must have finite supported values.')
        if np.any(inputs.predictions <= 0) or np.any(inputs.baseline_sales < 0):
            raise ValueError('Invalid APSD predictions or baselines.')
        self.months = pd.date_range(f'{config.planning_year}-01-01', periods=self.h, freq='MS')
        self.base = inputs.baseline_sales * inputs.active_days[:, None]
        self.ke = np.asarray(inputs.existing_log_kernel, dtype=float)
        self.kn = np.asarray(inputs.new_log_kernel, dtype=float)
        if self.ke.shape != (self.h, self.c, self.e) or self.kn.shape != (self.h, self.c, self.c):
            raise ValueError('Coefficient kernel dimensions are invalid.')
        self.fe = np.expm1(self.ke)
        self.fn = np.expm1(self.kn)
        # Opening-month duration weights log effects. This is explicit prorating,
        # not an estimated within-month event-time model.
        self.fractions = np.array([(m.days_in_month - config.opening_day + 1) / m.days_in_month for m in self.months[:12]])
        self.gross = np.zeros((12, self.h, self.c))
        for month in range(12):
            age = np.arange(self.h - month)
            self.gross[month, month:] = inputs.predictions[None, :] * inputs.maturity[age, None] * inputs.active_days[month:, None]
            self.gross[month, month] *= self.fractions[month]
        self.weights = np.ones(self.h)
        if config.objective == 'npv':
            self.weights = config.costs.margin / (1 + config.costs.annual_discount_rate) ** (np.arange(self.h) / 12)
        self.cost_by_month = np.zeros(12)
        if config.objective == 'npv':
            discount = (1 + config.costs.annual_discount_rate) ** (np.arange(self.h) / 12)
            for m in range(12):
                operating = np.ones(self.h - m) * config.costs.monthly_operating_cost
                operating[0] *= self.fractions[m]
                self.cost_by_month[m] = (operating / discount[m:]).sum() + config.costs.investment / discount[m]
        self.ready = np.ones((self.c, 12), dtype=bool)
        self.capacity_regions = self.candidates.get('capacity_region', self.candidates['region']).astype(str)
        if any(region not in set(self.capacity_regions) for region, _, _ in config.regional_capacity):
            raise ValueError('Regional capacity label is absent from verified candidate capacity regions.')
        dates = pd.DatetimeIndex([pd.Timestamp(config.planning_year, m + 1, config.opening_day) for m in range(12)])
        for col, earliest in [('earliest_opening_date', True), ('latest_opening_date', False)]:
            if col in self.candidates:
                parsed = pd.to_datetime(self.candidates[col], errors='coerce')
                bad = self.candidates[col].notna() & parsed.isna()
                if bad.any():
                    raise ValueError(f'Invalid site readiness values: {col}.')
                for i, date in enumerate(parsed):
                    if pd.notna(date):
                        self.ready[i] &= (dates >= date) if earliest else (dates <= date)
        self.evaluations = 0

    def action_logs(self, candidate, month):
        ex = self.ke[:self.h - month, candidate].copy()
        new = self.kn[:self.h - month, candidate].copy()
        ex[0] *= self.fractions[month]
        new[0] *= self.fractions[month]
        return ex, new

    def state(self, plan):
        le, ln = np.zeros((self.h, self.e)), np.zeros((self.h, self.c))
        gross = np.zeros((self.h, self.c))
        for c, m in plan.items():
            ex, new = self.action_logs(c, m)
            le[m:] += ex
            ln[m:] += new
            gross[:, c] = self.gross[m, :, c]
        return le, ln, gross

    def feasible(self, plan, complete=True):
        if any(not 0 <= c < self.c or not 0 <= m < 12 or not self.ready[c, m] for c, m in plan.items()):
            return False
        counts = np.bincount(list(plan.values()), minlength=12)
        lo, hi = self.config.month_bounds()
        lower, upper = self.config.count_bounds()
        if len(plan) > upper or np.any(counts > hi):
            return False
        if complete and (len(plan) < lower or np.any(counts < lo)):
            return False
        for region, month, cap in self.config.regional_capacity:
            if sum(m == month - 1 and self.capacity_regions.iloc[c] == region for c, m in plan.items()) > cap:
                return False
        return True

    def evaluate(self, plan, require_feasible=True):
        if require_feasible and not self.feasible(plan):
            raise ValueError('Portfolio violates count, readiness, monthly or budget constraints.')
        le, ln, gross = self.state(plan)
        existing_change = self.base * np.expm1(le)
        net_new = gross * np.exp(ln)
        new_to_new = net_new - gross
        incremental = existing_change.sum(axis=1) + net_new.sum(axis=1)
        value = float(incremental @ self.weights - sum(self.cost_by_month[m] for m in plan.values()))
        if not np.isfinite(value) or not np.isfinite(incremental).all():
            raise ValueError('Cumulative prediction exceeds the numeric range; review raw coefficient extrapolation.')
        self.evaluations += 1
        return value, existing_change, gross, new_to_new, net_new, incremental

    def addition_scores(self, plan):
        le, ln, gross = self.state(plan)
        existing_revenue = self.base * np.exp(le) * self.weights[:, None]
        new_revenue = gross * np.exp(ln) * self.weights[:, None]
        scores = np.empty((self.c, 12))
        for month in range(12):
            count = self.h - month
            # Coefficient kernels depend on source event age, so geography and
            # lookup tensors are reused across every candidate/month proposal.
            ex_factor = self.fe[:count]
            nn_factor = self.fn[:count]
            scores[:, month] = np.einsum('hce,he->c', ex_factor, existing_revenue[month:], optimize=False)
            scores[:, month] += np.einsum('hcr,hr->c', nn_factor, new_revenue[month:], optimize=False)
            if self.fractions[month] != 1.:
                scores[:, month] += (np.expm1(self.ke[0] * self.fractions[month]) - self.fe[0]) @ existing_revenue[month]
                scores[:, month] += (np.expm1(self.kn[0] * self.fractions[month]) - self.fn[0]) @ new_revenue[month]
            scores[:, month] += (self.gross[month] * np.exp(ln) * self.weights[:, None]).sum(axis=0)
            scores[:, month] -= self.cost_by_month[month]
        scores[~self.ready] = -np.inf
        if plan:
            scores[list(plan)] = -np.inf
        _, maxima = self.config.month_bounds()
        counts = np.bincount(list(plan.values()), minlength=12)
        scores[:, counts >= maxima] = -np.inf
        for region, month, cap in self.config.regional_capacity:
            used = sum(m == month - 1 and self.capacity_regions.iloc[c] == region for c, m in plan.items())
            if used >= cap:
                scores[self.capacity_regions.eq(region), month - 1] = -np.inf
        return scores


def optimize(evaluator: Evaluator, progress: Callable | None = None, initial_plan=None):
    """Greedy initialization then exact accepted add/remove/swap/reschedule moves.

    Runtime budget is shared by starts. Neighborhood sampling and first improvement
    make this a bounded local search; termination never asserts a global optimum.
    """
    cfg = evaluator.config
    started = time.perf_counter()
    deadline = started + cfg.time_budget_seconds
    rng = np.random.default_rng(cfg.seed)
    curve, history, greedy_plans = [], [], []
    best_plan, best_value = None, -np.inf
    if initial_plan is not None and evaluator.feasible(initial_plan):
        best_plan = dict(initial_plan)
        best_value = evaluator.evaluate(best_plan)[0]
    stop = 'iteration_limit'
    lower, upper = cfg.count_bounds()
    for start in range(cfg.starts):
        plan = {}
        lo, _ = cfg.month_bounds()
        if best_plan is None and evaluator.feasible(plan):
            best_plan, best_value = {}, evaluator.evaluate({})[0]
        greedy_best = None
        while len(plan) < upper:
            if time.perf_counter() >= deadline:
                stop = 'time_budget'
                break
            scores = evaluator.addition_scores(plan)
            counts = np.bincount(list(plan.values()), minlength=12)
            deficits = counts < lo
            if deficits.any():
                scores[:, ~deficits] = -np.inf
            flat = scores.ravel()
            finite = np.flatnonzero(np.isfinite(flat))
            if not len(finite):
                stop = 'no_feasible_addition'
                break
            ordered = finite[np.argsort(-flat[finite], kind='stable')]
            selected = int(ordered[0] if start == 0 else rng.choice(ordered[:min(3, len(ordered))]))
            c, month = np.unravel_index(selected, scores.shape)
            if cfg.target is None and not deficits.any() and len(plan) >= lower and flat[selected] <= cfg.minimum_marginal:
                stop = 'marginal_threshold'
                break
            plan[int(c)] = int(month)
            value = evaluator.evaluate(plan, require_feasible=False)[0]
            curve.append({'start': start, 'store_count': len(plan), 'objective_value': value,
                          'curve_type': 'sequential_joint_greedy_prefix', 'feasible': evaluator.feasible(plan)})
            if evaluator.feasible(plan) and (greedy_best is None or value > greedy_best[0]):
                greedy_best = (value, dict(plan))
            if progress:
                progress(min(.7, .7 * len(plan) / max(upper, 1)), f'Joint greedy: {len(plan)} stores')
        if greedy_best is None and evaluator.feasible(plan):
            greedy_best = (evaluator.evaluate(plan)[0], dict(plan))
        if greedy_best is None:
            if best_plan is not None:
                greedy_best = (best_value, dict(best_plan))
            else:
                raise ValueError('No feasible complete portfolio found within constraints/search budget. Increase time or relax constraints; this is not proof of infeasibility.')
        value, plan = greedy_best
        if best_plan is not None and best_value > value:
            plan, value = dict(best_plan), best_value
        greedy_plans.append((value, dict(plan)))
        for iteration in range(cfg.max_iterations):
            if time.perf_counter() >= deadline:
                stop = 'time_budget'
                break
            proposals = []
            if cfg.target is None:
                if len(plan) < upper:
                    scores = evaluator.addition_scores(plan)
                    if np.isfinite(scores).any():
                        c, m = np.unravel_index(np.argmax(scores), scores.shape)
                        proposals.append(('add', {**plan, int(c): int(m)}))
                if len(plan) > lower:
                    for c in list(plan):
                        p = dict(plan); del p[c]
                        proposals.append(('remove', p))
            # Sample several removal neighborhoods. After removing a location,
            # its own candidacy allows pure month moves; other IDs allow swaps.
            selected_ids = list(plan)
            rng.shuffle(selected_ids)
            for c in selected_ids[:cfg.neighborhood_size]:
                if time.perf_counter() >= deadline:
                    break
                reduced = dict(plan); del reduced[c]
                scores = evaluator.addition_scores(reduced)
                lo, _ = cfg.month_bounds()
                counts = np.bincount(list(reduced.values()), minlength=12)
                if (counts < lo).any():
                    scores[:, counts >= lo] = -np.inf
                if np.isfinite(scores).any():
                    replacement, month = np.unravel_index(np.argmax(scores), scores.shape)
                    p = {**reduced, int(replacement): int(month)}
                    proposals.append(('reschedule' if int(replacement) == c else 'location_swap', p))
            for a, b in zip(selected_ids[::2], selected_ids[1::2]):
                p = dict(plan); p[a], p[b] = p[b], p[a]
                proposals.append(('month_swap', p))
            improved = False
            for move, proposal in proposals:
                if time.perf_counter() >= deadline:
                    stop = 'time_budget'
                    break
                if not evaluator.feasible(proposal):
                    continue
                proposed_value = evaluator.evaluate(proposal)[0]
                if proposed_value > value + cfg.tolerance:
                    history.append({'start': start, 'iteration': iteration, 'move': move,
                                    'before': value, 'after': proposed_value, 'store_count': len(proposal)})
                    value, plan = proposed_value, proposal
                    improved = True
            if progress:
                progress(.75 + .2 * (iteration + 1) / max(1, cfg.max_iterations), f'Local improvement iteration {iteration + 1}')
            if not improved:
                stop = 'sampled_neighborhood_no_improvement'
                break
        if value > best_value:
            best_value, best_plan = value, dict(plan)
        if time.perf_counter() >= deadline:
            break
    if best_plan is None:
        raise ValueError('No feasible solution found.')
    greedy_value, greedy_plan = max(greedy_plans, key=lambda x: x[0]) if greedy_plans else (0., {})
    return best_plan, pd.DataFrame(curve, columns=['start', 'store_count', 'objective_value', 'curve_type', 'feasible']), {
        'method': 'joint_greedy_plus_bounded_add_remove_swap_reschedule',
        'status': stop, 'runtime_seconds': time.perf_counter() - started,
        'greedy_initial_objective': greedy_value, 'best_objective': best_value,
        'absolute_improvement': best_value - greedy_value,
        'percentage_improvement': (100 * (best_value - greedy_value) / abs(greedy_value)) if greedy_value else None,
        'portfolio_changed': best_plan != greedy_plan, 'accepted_moves': history,
        'evaluations': evaluator.evaluations, 'global_optimality_proven': False,
    }


def result_tables(evaluator, plan):
    value, existing, gross, nn, net, incremental = evaluator.evaluate(plan)
    dates = evaluator.months
    opening_counts = np.bincount(list(plan.values()), minlength=evaluator.h)
    monthly = pd.DataFrame({'month': dates, 'new_openings': opening_counts,
        'cumulative_openings': opening_counts.cumsum(), 'gross_new_store_sales': gross.sum(axis=1),
        'new_to_new_change': nn.sum(axis=1), 'new_store_sales': net.sum(axis=1),
        'existing_store_change': existing.sum(axis=1), 'incremental_network_sales': incremental,
        'total_network_sales': evaluator.base.sum(axis=1) + incremental})
    selected = evaluator.candidates.iloc[list(plan)].copy()
    selected['opening_month'] = [m + 1 for m in plan.values()]
    selected['opening_date'] = [pd.Timestamp(evaluator.config.planning_year, m + 1, evaluator.config.opening_day) for m in plan.values()]
    selected['gross_new_store_sales'] = gross[:, list(plan)].sum(axis=0)
    selected['new_to_new_change'] = nn[:, list(plan)].sum(axis=0)
    selected['horizon_new_store_sales'] = net[:, list(plan)].sum(axis=0)
    # Allocate interacting existing effects by proportional log contribution,
    # which reconciles exactly even with overlapping mixed-sign contributions.
    le, _, _ = evaluator.state(plan)
    allocation = []
    for c, m in plan.items():
        ex, _ = evaluator.action_logs(c, m)
        ratio = np.ones_like(le[m:])
        np.divide(np.expm1(le[m:]), le[m:], out=ratio, where=np.abs(le[m:]) > 1e-12)
        allocation.append(float((evaluator.base[m:] * ratio * ex).sum()))
    selected['allocated_existing_store_change'] = allocation
    selected['allocated_incremental_network_sales'] = selected.horizon_new_store_sales + selected.allocated_existing_store_change
    selected['attribution_method'] = 'proportional_log_contribution; not standalone marginal ranking'
    impacts = evaluator.data.existing[['store_id', 'store_name', 'region']].copy()
    impacts['horizon_existing_sales_change'] = existing.sum(axis=0)
    impacts['baseline_sales'] = evaluator.base.sum(axis=0)
    impacts['scenario_sales'] = impacts.baseline_sales + impacts.horizon_existing_sales_change
    affected = (evaluator.data.existing_distances[list(plan)] <= 1000).any(axis=0) if plan else np.zeros(evaluator.e, bool)
    impacts['exposed_within_1000m'] = affected
    summary = {'store_count': len(plan), 'objective_value': value, 'objective': evaluator.config.objective,
        'monthly_allocation': json.dumps(opening_counts[:12].tolist()),
        'gross_new_store_sales': float(gross.sum()), 'new_to_new_change': float(nn.sum()),
        'new_store_sales': float(net.sum()), 'existing_store_change': float(existing.sum()),
        'incremental_network_sales': float(incremental.sum()), 'total_network_sales': float(evaluator.base.sum() + incremental.sum()),
        'planning_year_incremental_sales': float(incremental[:12].sum()),
        'following_year_incremental_sales': float(incremental[12:].sum()) if evaluator.h > 12 else None,
        'existing_stores_affected': int(affected.sum()), 'incremental_npv': value if evaluator.config.objective == 'npv' else None}
    if not np.isclose(summary['incremental_network_sales'], summary['gross_new_store_sales'] + summary['new_to_new_change'] + summary['existing_store_change'], rtol=1e-12, atol=0.01):
        raise AssertionError('Network components do not reconcile.')
    return selected, monthly, impacts, summary


def save_scenario(evaluator, plan, curve, diagnostics, output_root):
    selected, monthly, impacts, summary = result_tables(evaluator, plan)
    config = asdict(evaluator.config)
    fingerprint = {'version': VERSION, 'config': config, 'inputs': evaluator.data.provenance,
                   'solution': sorted((str(evaluator.candidates.iloc[c].candidate_id), m) for c, m in plan.items())}
    scenario_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:20]
    directory = Path(output_root) / scenario_id
    directory.mkdir(parents=True, exist_ok=True)
    summary.update(scenario_id=scenario_id, model_version=VERSION,
                   planning_year=evaluator.config.planning_year, horizon=evaluator.h,
                   optimization_status=diagnostics['status'],
                   optimization_method=diagnostics['method'], random_seed=evaluator.config.seed,
                   demand_scenario=evaluator.config.demand_scenario,
                   coefficient_source=evaluator.config.coefficient_source,
                   coefficient_mode=evaluator.config.coefficient_mode,
                   monthly_capacity=json.dumps(evaluator.config.monthly_max),
                   forecast_as_of=evaluator.config.as_of,
                   run_timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat())
    pd.DataFrame([summary]).to_csv(directory / 'optimization_summary.csv', index=False)
    curve.to_csv(directory / 'optimal_store_count_curve.csv', index=False)
    selected.to_csv(directory / 'optimal_selected_stores.csv', index=False)
    monthly.to_csv(directory / 'optimal_monthly_schedule.csv', index=False)
    impacts.to_parquet(directory / 'store_level_network_impacts.parquet', index=False)
    (directory / 'optimization_diagnostics.json').write_text(json.dumps({**diagnostics,
        'configuration': config, 'inputs': evaluator.data.provenance,
        'run_timestamp_utc': pd.Timestamp.now(tz='UTC').isoformat()}, indent=2, default=str), encoding='utf-8')
    return directory, summary
