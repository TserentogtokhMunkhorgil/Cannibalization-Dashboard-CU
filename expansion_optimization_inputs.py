"""Offline input adapter; reuses saved demand forecasts and existing geo helpers."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from expansion_optimization import (Inputs, Evaluator, OptimizationConfig, optimize, result_tables, save_scenario)
from expansion_simulation import haversine_distances, _maturation_profile, _greedy_portfolio
from regional_cumulative_impact import FIXED_INCOME, BANDS, _bands, _region

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs' / 'expansion_optimization_v2'


def safe_csv(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if any(line.startswith(('<<<<<<<', '=======', '>>>>>>>')) for line in text.splitlines()):
        raise ValueError(f'Merge-conflicted input: {path.name}. Supply a clean, verified coefficient file; no branch is selected automatically.')
    return pd.read_csv(path)


def kernel(coefficients, regions, codes, horizon, tail, mode):
    """Map source event age to receiver-region effects without silent tail extension."""
    result = np.zeros((horizon, *codes.shape))
    sources = set()
    for receiver, region in enumerate(regions):
        for band_index, (band, _, _) in enumerate(BANDS):
            rows = coefficients.loc[coefficients.region.eq(region) & coefficients.distance_band.eq(band)]
            used = region
            if rows.empty:
                rows = coefficients.loc[coefficients.region.eq('All Regions') & coefficients.distance_band.eq(band)]
                used = 'All Regions'
            if rows.empty:
                raise ValueError(f'No coefficient for {region}/{band}, including declared All Regions fallback.')
            ages = rows.set_index('age').coefficient.to_dict()
            for age in range(horizon):
                if age in ages:
                    beta = float(ages[age])
                elif tail == 'hold_last' and age > max(ages):
                    beta = float(ages[max(ages)])
                elif tail == 'zero' and age > max(ages):
                    beta = 0.
                else:
                    raise ValueError(f'Missing event age {age} for {used}/{band}. Choose an explicit tail assumption if beyond supported ages.')
                if not np.isfinite(beta):
                    raise ValueError('Nonfinite coefficient.')
                if mode == 'negative_only':
                    beta = min(beta, 0.)
                result[age, codes[:, receiver] == band_index, receiver] = beta
            sources.add(f'{region}/{band} <- {used}')
    return result, sorted(sources)


def prepare_inputs(base, config):
    """Read/reuse static inputs once per assumption set; never fit a model."""
    base = Path(base)
    prediction_file = base / 'outputs/expansion_simulation/potential_store_predictions.csv'
    master_file = base / 'store_master_data.xlsx'
    panel_file = base / 'outputs/cannibalization_regional/cu_regional_cannibalization_panel.parquet'
    inputs_read = [prediction_file, master_file, panel_file]
    predictions = safe_csv(prediction_file)
    candidates = predictions.loc[predictions.selection_eligible.astype(str).str.lower().eq('true')].copy()
    candidates = candidates.sort_values('potential_store_id', kind='stable').reset_index(drop=True)
    if config.candidate_limit is not None:
        if config.candidate_limit < 1:
            raise ValueError('Candidate limit must be positive.')
        candidates = candidates.head(config.candidate_limit).copy()
    candidates['candidate_id'] = candidates.potential_store_id.astype(str)
    candidates['candidate_name'] = candidates['Potential Point'].fillna(candidates.candidate_id)
    if candidates.candidate_id.duplicated().any() or candidates.duplicated(['latitude', 'longitude']).any():
        raise ValueError('Eligible forecast candidates contain duplicate IDs/coordinates.')
    if not (candidates.latitude.between(-90, 90) & candidates.longitude.between(-180, 180)).all():
        raise ValueError('Invalid eligible candidate coordinates.')
    # Existing demand forecasts only support UB/Rural candidate geography. Use
    # the original expansion new-receiver All Regions policy explicitly; do not
    # manufacture Center/Outskirts classifications for candidate sites.
    candidates['region'] = 'All Regions'
    candidates['geographic_area'] = candidates['demand_region']
    candidates['capacity_region'] = candidates['demand_region']
    candidates['coefficient_region_rule'] = 'All Regions for proposed-store receivers (original expansion policy; sub-region unverified)'
    gps_file = base / 'Data/raw/New Store GPS.xlsx'
    if gps_file.is_file():
        gps = pd.read_excel(gps_file)
        inputs_read.append(gps_file)
        readiness = [c for c in ('earliest_opening_date', 'latest_opening_date') if c in gps]
        if readiness:
            gps = gps.rename(columns={'Longitude': 'longitude', 'Latitude': 'latitude'})
            if gps.duplicated(['longitude', 'latitude']).any():
                raise ValueError('Readiness mapping has duplicate candidate coordinates.')
            candidates = candidates.merge(gps[['longitude', 'latitude', *readiness]], on=['longitude', 'latitude'], how='left', validate='one_to_one', indicator=True)
            if candidates['_merge'].ne('both').any():
                raise ValueError('Readiness coordinates do not match saved forecast locations; reconcile the source coordinates explicitly.')
            candidates = candidates.drop(columns='_merge')
    suffix = '' if config.demand_scenario == 'central' else f'_{config.demand_scenario}_95'
    prediction = pd.to_numeric(candidates['predicted_APSD_sales' + suffix], errors='coerce').to_numpy(float)
    if not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError('Saved APSD forecasts are missing/nonpositive for eligible candidates.')
    candidates['scenario_APSD_sales'] = prediction
    master = pd.read_excel(master_file)
    master['store_id'] = master.store_id.astype(str).str.replace(r'\.0$', '', regex=True)
    master['SalesMonth'] = pd.to_datetime(master.SalesMonth, errors='coerce')
    if master.duplicated(['store_id', 'SalesMonth']).any():
        raise ValueError('Duplicate historical store-month observations.')
    as_of = pd.Timestamp(config.as_of)
    if config.planning_year <= as_of.year:
        raise ValueError('Planning year must follow the analysis as-of year.')
    completed = master.loc[master.SalesMonth.dt.to_period('M') < as_of.to_period('M')].copy()
    completed['APSD_income'] = pd.to_numeric(completed.APSD_income, errors='coerce')
    completed = completed.loc[np.isfinite(completed.APSD_income) & completed.APSD_income.gt(0)]
    if completed.empty:
        raise ValueError('No completed historical month with valid baseline income.')
    latest = completed.SalesMonth.max()
    current = completed.loc[completed.SalesMonth.eq(latest)].copy()
    current['latitude'] = pd.to_numeric(current.LATITUDE, errors='coerce')
    current['longitude'] = pd.to_numeric(current.LONGITUDE, errors='coerce')
    current['open_date'] = pd.to_datetime(current.ost_dt, errors='coerce')
    current['close_date'] = pd.to_datetime(current.cls_dt, errors='coerce') if 'cls_dt' in current else pd.NaT
    valid = current.latitude.between(-90, 90) & current.longitude.between(-180, 180) & current.open_date.notna() & current.open_date.le(as_of)
    excluded = current.loc[~valid, 'store_id'].tolist()
    existing = current.loc[valid].sort_values('store_id').reset_index(drop=True)
    existing['region'] = [_region(z, u) for z, u in zip(existing.Distance_Zone, existing.UB)]
    if existing.region.eq('Unmapped').any():
        raise ValueError('Existing-store region is unmapped; provide a verified region before optimization.')
    recent = completed.sort_values('SalesMonth').groupby('store_id').tail(12)
    levels = recent.groupby('store_id').APSD_income.mean()
    months = pd.date_range(f'{config.planning_year}-01-01', periods=config.horizon, freq='MS')
    # Preserve the original mean-of-recent-observed-month APSD concept; seasonality
    # is descriptive within completed observations, not a newly fitted regression.
    seasonal = completed.assign(calendar_month=completed.SalesMonth.dt.month).groupby('calendar_month').APSD_income.mean()
    seasonal = (seasonal / completed.APSD_income.mean()).clip(.6, 1.4)
    baseline = np.array([existing.store_id.map(levels).to_numpy(float) * float(seasonal.get(m.month, 1.)) for m in months])
    for t, month in enumerate(months):
        baseline[t, existing.close_date.notna() & existing.close_date.lt(month)] = 0.
    if 'Active_Days' not in completed:
        raise ValueError('Active_Days are required; no operating-day forecast is invented.')
    day_means = completed.groupby(completed.SalesMonth.dt.month).Active_Days.median()
    days = np.array([min(float(day_means.get(m.month, np.nan)), m.days_in_month) for m in months])
    if not (np.isfinite(days) & (days > 0)).all():
        raise ValueError('Historical active-day support is missing for a forecast calendar month.')
    maturation_file = base / 'outputs/expansion_simulation/maturation_profiles.csv'
    maturations = safe_csv(maturation_file)
    inputs_read.append(maturation_file)
    lookup = maturations.loc[maturations.outcome.eq('sales')].set_index('age_month').maturation_multiplier.to_dict()
    if any(age not in lookup for age in range(12)):
        raise ValueError('Saved sales maturation profile is incomplete.')
    maturity = np.array([lookup[a] if a < 12 else (lookup[11] if config.maturity_tail == 'hold_last' else 1.) for a in range(config.horizon)])
    if config.coefficient_source == 'fixed_cumulative':
        rows = [{'region': r, 'distance_band': b, 'age': age, 'coefficient': v}
                for r, values in FIXED_INCOME.items() for (b, _, _), v in zip(BANDS, values) for age in range(config.horizon)]
        coefficients = pd.DataFrame(rows)
        coefficient_provenance = 'Exact user fixed cumulative exposure coefficients; constant log effect per opened store, all post-opening ages'
    else:
        file = base / 'outputs/cannibalization_regional/regional_sales_coefficients.csv'
        coefficients = safe_csv(file)
        inputs_read.append(file)
        coefficients = coefficients.loc[coefficients.model_type.eq('monthly_event_study') & coefficients.event_time.ge(0)].copy()
        if not coefficients.outcome_scale.eq('log').all():
            raise ValueError('Event-study outcome scale must be log.')
        coefficients = coefficients.rename(columns={'event_time': 'age'})
        if coefficients.duplicated(['region', 'distance_band', 'age']).any():
            raise ValueError('Duplicate event-time coefficients.')
        coefficient_provenance = 'Saved monthly_event_study only; event age indexed from source opening; All Regions used only for missing regional cells'
    dist = haversine_distances(candidates.longitude, candidates.latitude, existing.longitude, existing.latitude)
    pair = haversine_distances(candidates.longitude, candidates.latitude, candidates.longitude, candidates.latitude)
    def band_codes(values):
        bands = _bands(values)
        return np.array([np.where(bands == b, i, -1) for i, (b, _, _) in enumerate(BANDS)]).max(axis=0)
    ke, source_existing = kernel(coefficients, existing.region, band_codes(dist), config.horizon, config.tail_assumption, config.coefficient_mode)
    kn, source_new = kernel(coefficients, candidates.region, band_codes(pair), config.horizon, config.tail_assumption, config.coefficient_mode)
    kn[:, np.arange(len(candidates)), np.arange(len(candidates))] = 0.
    provenance = {'source_hashes': {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs_read},
        'coefficients': coefficient_provenance, 'fixed_income_values': FIXED_INCOME if config.coefficient_source == 'fixed_cumulative' else None,
        'existing_coefficient_mapping': source_existing, 'new_receiver_mapping': source_new,
        'excluded_baseline_store_ids': excluded, 'baseline_latest_month': str(latest.date()),
        'baseline_rule': 'Existing stores observed in latest completed month; last 12 observed positive monthly APSD mean with pooled historical monthly seasonality',
        'prediction_warning': 'Reused saved opening-month demand forecasts. Historical validation is conditional and does not establish profit or causal growth; source vintage is the saved forecast file.',
        'maturity_after_age_11': config.maturity_tail, 'opening_month_log_proration': 'fraction of calendar month active',
        'distance_policy': '[0,100], (100,250], (250,500], (500,1000]; zero beyond',
        'new_receiver_policy': 'Original expansion All Regions coefficients; sub-regional candidate classification unverified'}
    return Inputs(candidates, existing, baseline, days, prediction, maturity, ke, kn, dist, pair, provenance)


def original_greedy(evaluator, target):
    """Run the unchanged original greedy method, then use the common evaluator."""
    from expansion_simulation import schedule_months
    data, cfg = evaluator.data, evaluator.config
    if cfg.coefficient_source != 'fixed_cumulative' or cfg.opening_day != 1:
        raise ValueError('Comparable original-greedy bridge currently requires fixed cumulative coefficients and opening day 1.')
    def codes(distance):
        bands = _bands(distance)
        result = np.full(distance.shape, -1, dtype=int)
        for i, (b, _, _) in enumerate(BANDS):
            result[bands == b] = i
        return result
    ex, nn = codes(data.existing_distances), codes(data.pair_distances)
    np.fill_diagonal(nn, -1)
    candidates = data.candidates.assign(selection_eligible=True)
    existing = data.existing.rename(columns={'region': 'analysis_region'})
    regional = {r: {(b, age): (min(v, 0.) if cfg.coefficient_mode == 'negative_only' else v)
                    for (b, _, _), v in zip(BANDS, values) for age in range(12)} for r, values in FIXED_INCOME.items()}
    global_values = regional['All Regions'].copy()
    global_values['matrix'] = np.array([[global_values[(b, a)] for a in range(12)] for b, _, _ in BANDS])
    models = {'sales': {'global': global_values, 'regional': regional, 'regions': ('All Regions', 'City Center', 'City Outskirts', 'Far / Outer Areas', 'Rural')}}
    weights = cfg.exact_monthly if cfg.exact_monthly is not None else cfg.monthly_max
    started = time.perf_counter()
    selected, dates, _, _, _ = _greedy_portfolio(candidates, ex, nn, existing, ex,
        data.baseline_sales[:12], data.active_days[:12], data.predictions,
        {age: data.maturity[age] for age in range(12)}, data.active_days[:12], models,
        target, cfg.planning_year, 'monthly_even', weights, cfg.opening_day, 1.)
    plan = {int(c): date.month - 1 for c, date in zip(selected, dates)}
    return plan, time.perf_counter() - started


def saved_baseline(evaluator, file, frozen=False):
    """Coordinate crosswalk keeps the two original candidate-ID schemes distinct."""
    frame = safe_csv(file)
    if frozen:
        directory = Path(file).parent
        coordinates = safe_csv(directory / 'candidates.csv')
        frame = frame.merge(coordinates[['candidate_id', 'latitude', 'longitude']], on='candidate_id', validate='one_to_one')
    if len(frame) != 180:
        raise ValueError('Existing benchmark must have exactly 180 locations.')
    distances = haversine_distances(frame.longitude, frame.latitude, evaluator.candidates.longitude, evaluator.candidates.latitude)
    nearest = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(frame)), nearest] > .5) or len(set(nearest)) != 180:
        raise ValueError('Benchmark coordinates do not map one-to-one to eligible forecast candidates within 0.5 m.')
    if 'assumed_opening_date' in frame:
        months = pd.to_datetime(frame.assumed_opening_date).dt.month.to_numpy() - 1
    else:
        from expansion_simulation import schedule_months
        months = schedule_months(180, np.ones(12))
    plan = dict(zip(map(int, nearest), map(int, months)))
    return plan


def run_scenario(config, base=ROOT, prepared=None, progress=None):
    inputs = prepared if prepared is not None else prepare_inputs(base, config)
    evaluator = Evaluator(inputs, config)
    seed_plan, seed_runtime = None, 0.
    seed_count = config.target if config.target is not None else min(config.max_stores, sum(config.monthly_max))
    if config.coefficient_source == 'fixed_cumulative' and config.opening_day == 1 and seed_count and sum(config.monthly_max):
        proposed, seed_runtime = original_greedy(evaluator, seed_count)
        if evaluator.feasible(proposed):
            seed_plan = proposed
    plan, curve, diagnostics = optimize(evaluator, progress, initial_plan=seed_plan)
    diagnostics['original_greedy_initialization_seconds'] = seed_runtime
    diagnostics['total_solver_seconds'] = diagnostics['runtime_seconds'] + seed_runtime
    diagnostics['original_greedy_seed_feasible'] = seed_plan is not None
    if seed_plan is not None:
        seed_value = evaluator.evaluate(seed_plan)[0]
        diagnostics['original_greedy_common_objective'] = seed_value
        diagnostics['improvement_over_original_greedy'] = evaluator.evaluate(plan)[0] - seed_value
    directory, summary = save_scenario(evaluator, plan, curve, diagnostics, Path(base) / 'outputs/expansion_optimization_v2')
    selected, monthly, _, _ = result_tables(evaluator, plan)
    return {'directory': str(directory), 'summary': summary, 'selected': selected,
            'monthly': monthly, 'curve': curve, 'diagnostics': diagnostics, 'plan': plan}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Joint CU count, location and month optimization')
    parser.add_argument('--target', type=int, help='Omit for automatic count')
    parser.add_argument('--max-stores', type=int, default=400)
    parser.add_argument('--year', type=int, default=2027)
    parser.add_argument('--capacity', type=int, default=40)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    config = OptimizationConfig(target=args.target, max_stores=args.max_stores, planning_year=args.year,
                                monthly_max=(args.capacity,) * 12, horizon=args.horizon, time_budget_seconds=args.seconds)
    result = run_scenario(config)
    print(json.dumps(result['summary'], indent=2))
    if args.benchmark:
        benchmark(config)


def benchmark(config, base=ROOT):
    """Independent fixed-180/automatic/manual solves under identical evaluation assumptions."""
    inputs = prepare_inputs(base, config)
    rows, curve_rows = [], []
    # Baselines need not satisfy the user's new capacity limits. They remain
    # evaluated, but their feasibility is shown rather than silently changed.
    evaluator = Evaluator(inputs, config)
    for name, file, frozen in [('Existing expansion fixed-180', Path(base) / 'outputs/expansion_simulation/selected_stores_180.csv', False),
                               ('Frozen accepted fixed-180', Path(base) / 'Data/planning_inputs/accepted_v1/selected_180.csv', True)]:
        try:
            plan = saved_baseline(evaluator, file, frozen)
            relaxed = replace(config, target=None, min_stores=0, max_stores=len(inputs.candidates), monthly_min=(0,) * 12,
                              monthly_max=(len(inputs.candidates),) * 12, exact_monthly=None, regional_capacity=(), budget=None)
            common = Evaluator(inputs, relaxed)
            summary = result_tables(common, plan)[3]
            checker = Evaluator(inputs, replace(config, target=None, min_stores=0))
            rows.append({'scenario': name, 'feasible_under_requested_constraints': checker.feasible(plan), **summary})
        except (ValueError, FileNotFoundError) as error:
            rows.append({'scenario': name, 'status': str(error)})
    zero = Evaluator(inputs, replace(config, target=0, min_stores=0, monthly_min=(0,) * 12, exact_monthly=None))
    rows.append({'scenario': 'No expansion', **result_tables(zero, {})[3]})
    counts = list(dict.fromkeys([180, config.target, None]))
    results = {}
    for count in counts:
        if count is not None and count > min(config.max_stores, len(inputs.candidates)):
            continue
        cfg = replace(config, target=count)
        try:
            result = run_scenario(cfg, base, inputs)
            name = 'Automatic count' if count is None else f'Optimized manual {count}'
            results[name] = result
            rows.append({'scenario': name, 'status': result['diagnostics']['status'],
                         'runtime_seconds': result['diagnostics']['total_solver_seconds'], **result['summary']})
            curve = result['curve'].copy()
            curve['scenario'] = name
            curve_rows.append(curve)
        except ValueError as error:
            rows.append({'scenario': f'Optimized {count}', 'status': str(error)})
    if config.coefficient_source == 'fixed_cumulative' and config.opening_day == 1 and config.max_stores >= 180 and (config.exact_monthly is None or sum(config.exact_monthly) == 180) and config.min_stores <= 180:
        fixed = Evaluator(inputs, replace(config, target=180))
        original, runtime = original_greedy(fixed, 180)
        comparable = fixed.feasible(original)
        if comparable:
            old_summary = result_tables(fixed, original)[3]
            row = {'scenario': 'Original unchanged greedy (re-evaluated)', 'runtime_seconds': runtime, **old_summary}
            improved = results.get('Optimized manual 180')
            if improved:
                delta = improved['summary']['objective_value'] - old_summary['objective_value']
                row.update(improvement_over_original=delta, improvement_percent=100 * delta / abs(old_summary['objective_value']) if old_summary['objective_value'] else None,
                           portfolio_changed=improved['plan'] != original,
                           improved_runtime_seconds=improved['diagnostics']['total_solver_seconds'])
            rows.append(row)
        else:
            rows.append({'scenario': 'Original unchanged greedy', 'status': 'Infeasible under requested constraints; no improvement claim'})
    destination = Path(base) / 'outputs/expansion_optimization_v2' / 'comparisons'
    fingerprint = hashlib.sha256(json.dumps({'config': asdict(config), 'inputs': inputs.provenance}, sort_keys=True, default=str).encode()).hexdigest()[:20]
    destination = destination / fingerprint
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / 'scenario_comparison.csv', index=False)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(destination / 'optimal_store_count_curve.csv', index=False)
    return pd.DataFrame(rows), destination


if __name__ == '__main__':
    main()
