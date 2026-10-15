"""Explicit-run expansion planning UI; no optimization on page startup."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from expansion_optimization import OptimizationConfig, Costs, Evaluator
from expansion_optimization_inputs import ROOT, prepare_inputs, run_scenario, benchmark, safe_csv
from expansion_scenarios import reference_from_result


def signature():
    files = [ROOT / 'store_master_data.xlsx', ROOT / 'Data/raw/New Store GPS.xlsx',
        ROOT / 'outputs/expansion_simulation/potential_store_predictions.csv',
        ROOT / 'outputs/expansion_simulation/maturation_profiles.csv',
        ROOT / 'outputs/cannibalization_regional/regional_sales_coefficients.csv',
        ROOT / 'expansion_optimization.py', ROOT / 'expansion_optimization_inputs.py', ROOT / 'regional_cumulative_impact.py']
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())


def from_json(config_json):
    values = json.loads(config_json)
    if values.get('costs'):
        values['costs'] = Costs(**values['costs'])
    return OptimizationConfig(**values)


@st.cache_resource(max_entries=2, show_spinner=False)
def _prepared(config_json, input_signature):
    return prepare_inputs(ROOT, from_json(config_json))


def prepared(config_json, input_signature):
    values = json.loads(config_json)
    static_keys = ('as_of', 'planning_year', 'horizon', 'candidate_limit', 'demand_scenario',
                   'coefficient_source', 'coefficient_mode', 'tail_assumption', 'maturity_tail')
    static = {key: values[key] for key in static_keys}
    return _prepared(json.dumps(static, sort_keys=True), input_signature)


@st.cache_data(max_entries=12, show_spinner=False)
def completed(config_json, input_signature, _progress=None):
    return run_scenario(from_json(config_json), ROOT, prepared(config_json, input_signature), _progress)


@st.cache_data(max_entries=4, show_spinner=False)
def comparisons(config_json, input_signature):
    return benchmark(from_json(config_json), ROOT)


def download(frame, label, name):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')


def render():
    st.caption('Sales-maximizing expansion scenarios, or NPV only with supplied costs. Results are the best feasible portfolios found within the search limits; global optimality is not established.')
    try:
        file = ROOT / 'outputs/expansion_simulation/potential_store_predictions.csv'
        header = safe_csv(file)
        eligible_count = int(header.selection_eligible.astype(str).str.lower().eq('true').sum())
        if eligible_count == 0:
            st.error('No eligible forecast candidate locations are available.')
            return
    except (ValueError, FileNotFoundError, AttributeError) as error:
        st.error(f'Prepare valid saved demand forecasts first: {error}')
        return
    mode = st.segmented_control('Optimization mode', ['Automatic', 'Manual'], default='Automatic', key='expansion_mode')
    objective = st.segmented_control('Objective', ['sales', 'npv'], default='sales',
        format_func=lambda x: 'Incremental network sales' if x == 'sales' else 'Incremental NPV', key='expansion_objective')
    with st.form('expansion_settings'):
        cols = st.columns(3)
        target = cols[0].number_input('Manual store count', 0, eligible_count, min(180, eligible_count), disabled=mode != 'Manual')
        year = cols[1].number_input('Planning year', 2027, 2100, pd.Timestamp.now(tz='Asia/Irkutsk').year + 1)
        horizon = cols[2].selectbox('Evaluation horizon (months)', [24, 12])
        cols = st.columns(3)
        minimum = cols[0].number_input('Minimum new stores', 0, eligible_count, 0)
        maximum = cols[1].number_input('Maximum new stores', 0, eligible_count, min(400, eligible_count))
        pool_limit = cols[2].number_input('Candidate pool limit', 1, eligible_count, eligible_count,
            help='A smaller pool uses the saved ID order; excluded sites are not searched.')
        schedule_type = st.selectbox('Monthly allocation', ['Capacity bounds', 'Exact opening counts'])
        schedule = pd.DataFrame({'month': pd.date_range(f'{int(year)}-01-01', periods=12, freq='MS').strftime('%b'),
            'minimum': [0] * 12, 'maximum': [40] * 12, 'exact_count': [15] * 12})
        st.caption('Edit minimum, maximum, and optional exact monthly opening counts.')
        edited = st.data_editor(schedule, hide_index=True, disabled=['month'], key='expansion_schedule')
        cols = st.columns(3)
        demand = cols[0].selectbox('Saved demand forecast scenario', ['central', 'lower', 'upper'])
        coefficient_mode = cols[1].selectbox('Cannibalization assumption', ['negative_only', 'raw'])
        source = cols[2].selectbox('Regression source', ['fixed_cumulative', 'monthly_event_study'])
        cols = st.columns(3)
        tail = cols[0].selectbox('Beyond available event ages', ['error', 'hold_last', 'zero'],
            help='Explicit sensitivity assumption; never silently carry month 11 forward.')
        maturity_tail = cols[1].selectbox('Maturation after age 11', ['hold_last', 'one'])
        seconds = cols[2].number_input('Local search time budget (seconds)', 5, 1800, 60)
        cols = st.columns(3)
        iterations = cols[0].number_input('Maximum local iterations', 0, 100, 8)
        seed = cols[1].number_input('Random seed', 0, 1000000, 2027)
        starts = cols[2].number_input('Initial solutions', 1, 10, 1)
        as_of = st.date_input('Historical data as-of date', value=pd.Timestamp.now(tz='Asia/Irkutsk').date())
        threshold = st.number_input('Minimum optional marginal contribution (MNT)', value=0.)
        st.caption('Costs below are optional for sales and required for NPV. No costs or margins are inferred.')
        cols = st.columns(4)
        margin = cols[0].number_input('Contribution margin fraction', min_value=0., max_value=1., value=None)
        operating = cols[1].number_input('Monthly operating cost per new store (MNT)', min_value=0., value=None)
        investment = cols[2].number_input('Investment per new store (MNT)', min_value=0., value=None)
        rate = cols[3].number_input('Annual discount rate fraction', min_value=0., value=None)
        budget = st.number_input('Optional total investment budget (MNT)', min_value=0., value=None)
        regional_json = st.text_input('Optional regional capacity JSON', value='[]',
            help='Rows are [candidate capacity area (UB or Rural), calendar month 1..12, maximum openings]. Unverified city sub-regions are not inferred.')
        run = st.form_submit_button('Run optimization', type='primary')
    if not run and 'expansion_result' not in st.session_state:
        saved = sorted((ROOT / 'outputs/expansion_optimization_v2').glob('*/optimization_summary.csv'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        if saved:
            choices = {}
            for file in saved:
                record = pd.read_csv(file).iloc[0]
                choices[f"{int(record['store_count'])} stores · {int(record['planning_year'])} · {record['objective']} · {file.parent.name}"] = file.parent
            chosen = st.selectbox('Available completed scenarios', list(choices))
            if st.button('Load saved scenario', key='expansion_load_saved'):
                directory = choices[chosen]
                diagnostics = json.loads((directory / 'optimization_diagnostics.json').read_text(encoding='utf-8'))
                saved_config = json.dumps(diagnostics['configuration'], sort_keys=True)
                st.session_state['expansion_config'] = saved_config
                st.session_state['expansion_elapsed'] = 0.
                st.session_state['expansion_result'] = {'directory': str(directory),
                    'summary': pd.read_csv(directory / 'optimization_summary.csv').iloc[0].where(lambda x: x.notna(), None).to_dict(),
                    'selected': pd.read_csv(directory / 'optimal_selected_stores.csv'),
                    'monthly': pd.read_csv(directory / 'optimal_monthly_schedule.csv', parse_dates=['month']),
                    'curve': pd.read_csv(directory / 'optimal_store_count_curve.csv'), 'diagnostics': diagnostics}
                st.session_state['active_expansion_scenario'] = reference_from_result(st.session_state['expansion_result'])
                st.session_state.pop('active_expansion_scenario_selector', None)
    if run:
        try:
            costs = None
            if any(v is not None for v in [margin, operating, investment, rate]):
                if any(v is None for v in [margin, operating, investment, rate]):
                    raise ValueError('Supply all four cost assumptions or leave all blank.')
                costs = Costs(margin, operating, investment, rate)
            cfg = OptimizationConfig(planning_year=int(year), horizon=int(horizon), target=int(target) if mode == 'Manual' else None,
                min_stores=int(minimum), max_stores=int(maximum), candidate_limit=int(pool_limit),
                monthly_min=tuple(edited.minimum), monthly_max=tuple(edited.maximum),
                exact_monthly=tuple(edited.exact_count) if schedule_type == 'Exact opening counts' else None,
                regional_capacity=tuple(tuple(row) for row in json.loads(regional_json)), objective=objective,
                costs=costs, budget=budget, minimum_marginal=threshold, demand_scenario=demand,
                coefficient_source=source, coefficient_mode=coefficient_mode, tail_assumption=tail,
                maturity_tail=maturity_tail, time_budget_seconds=float(seconds), max_iterations=int(iterations),
                starts=int(starts), seed=int(seed), as_of=str(as_of))
            cfg.validate(min(eligible_count, int(pool_limit)))
            cfg_json = json.dumps(asdict(cfg), sort_keys=True)
            progress = st.progress(0., text='Preparing saved forecasts and geographic inputs')
            started = time.perf_counter()
            result = completed(cfg_json, signature(), _progress=lambda value, message: progress.progress(value, text=message))
            progress.progress(1., text='Optimization complete')
            st.session_state['expansion_result'] = result
            st.session_state['expansion_config'] = cfg_json
            st.session_state['expansion_elapsed'] = time.perf_counter() - started
            st.session_state['active_expansion_scenario'] = reference_from_result(result)
            st.session_state.pop('active_expansion_scenario_selector', None)
        except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
            st.error(str(error))
    result = st.session_state.get('expansion_result')
    if result is None:
        st.info('Adjust the assumptions and choose Run optimization. No historical regression or optimization runs when this page opens.')
        return
    cfg_json = st.session_state['expansion_config']
    cfg = from_json(cfg_json)
    summary = result['summary']
    st.caption(f"Last completed scenario {summary['scenario_id']} · {cfg.planning_year}, {cfg.horizon} months, {cfg.objective}, {cfg.demand_scenario} demand, {cfg.coefficient_source}/{cfg.coefficient_mode}. Controls apply only after Run optimization.")
    st.caption('Saved demand forecasts are conditional estimates; new-store receiver coefficients use the original All Regions policy because sub-regional site classification is unverified. Opening-month effects are prorated by active calendar fraction.')
    with st.container(horizontal=True):
        st.metric('Recommended / selected new stores', summary['store_count'], border=True)
        st.metric('Objective value (MNT)', f"{summary['objective_value']:,.0f}", border=True)
        st.metric('Incremental network sales', f"{summary['incremental_network_sales']:,.0f}", border=True)
        st.metric('Existing stores exposed', summary['existing_stores_affected'], border=True)
    with st.container(horizontal=True):
        st.metric('Gross new-store sales', f"{summary['gross_new_store_sales']:,.0f}")
        st.metric('Existing-store change', f"{summary['existing_store_change']:,.0f}")
        st.metric('New-to-new change', f"{summary['new_to_new_change']:,.0f}")
        if summary.get('incremental_npv') is not None:
            st.metric('Incremental NPV', f"{summary['incremental_npv']:,.0f}")
    view = st.segmented_control('Results', ['Opening plan', 'Store-count curve', 'Map & portfolio', 'Comparisons', 'Diagnostics'], default='Opening plan', key='expansion_results_view')
    if view == 'Opening plan':
        monthly = result['monthly']
        st.bar_chart(monthly, x='month', y='new_openings', alt='Planned openings by month')
        st.line_chart(monthly, x='month', y='cumulative_openings', alt='Cumulative new-store openings')
        st.bar_chart(monthly, x='month', y=['gross_new_store_sales', 'existing_store_change', 'new_to_new_change'], alt='Monthly reconciled network sales components')
        st.dataframe(monthly, hide_index=True, alt='Monthly sales, cannibalization and opening totals')
        st.write({'planning_year_incremental_sales': summary['planning_year_incremental_sales'],
                  'following_year_incremental_sales': summary['following_year_incremental_sales']})
        download(monthly, 'Download monthly plan', 'monthly_opening_plan.csv')
    elif view == 'Store-count curve':
        st.caption('The curve contains approximate sequential joint-greedy prefixes, not independently optimized portfolios for every N. The selected result is separately improved; comparison scenarios are independent solves.')
        curve = result['curve']
        if not curve.empty:
            fig = px.line(curve, x='store_count', y='objective_value', color='start', title='Approximate count search paths')
            fig.add_scatter(x=[summary['store_count']], y=[summary['objective_value']], mode='markers', name='Best feasible found', marker_size=14)
            fig.add_vline(x=180, line_dash='dot', annotation_text='180-store reference count')
            comparison = st.session_state.get('expansion_comparison')
            if comparison and comparison[0] == cfg_json:
                marks = comparison[1].dropna(subset=['store_count', 'objective_value'])
                fig.add_scatter(x=marks.store_count, y=marks.objective_value, text=marks.scenario,
                    mode='markers', name='Independently evaluated / optimized scenarios')
            st.plotly_chart(fig, width='stretch', alt='Sequential count curves and improved selected solution')
        else:
            st.info('Zero expansion or no greedy additions: no count curve is available.')
    elif view == 'Map & portfolio':
        data = prepared(cfg_json, signature())
        candidates = data.candidates.copy()
        selected = result['selected'].copy()
        candidates = candidates.merge(selected[['candidate_id', 'opening_month']], on='candidate_id', how='left', validate='one_to_one')
        candidates['status'] = np.where(candidates.opening_month.notna(), 'Selected', 'Unselected eligible')
        status = st.selectbox('Candidate status', ['All', 'Selected', 'Unselected eligible'])
        months = st.multiselect('Opening month filter', list(range(1, 13)), default=list(range(1, 13)))
        regions = st.multiselect('Candidate geography', sorted(candidates.geographic_area.unique()), default=sorted(candidates.geographic_area.unique()))
        points = candidates.loc[candidates.geographic_area.isin(regions)]
        if status != 'All':
            points = points.loc[points.status.eq(status)]
        points = points.loc[points.opening_month.isna() | points.opening_month.isin(months)]
        fig = px.scatter_map(points, lat='latitude', lon='longitude', color='status', hover_name='candidate_name',
            custom_data=['candidate_id'], hover_data=['opening_month', 'geographic_area', 'scenario_APSD_sales'],
            map_style='open-street-map', zoom=5, height=550)
        fig.add_trace(go.Scattermap(lat=data.existing.latitude, lon=data.existing.longitude, mode='markers',
            marker={'color': '#777777', 'size': 5}, name='Existing CU stores'))
        selection = st.plotly_chart(fig, width='stretch', on_select='rerun', selection_mode='points',
            key='expansion_map', alt='Existing CU stores and selected or unselected eligible candidates')
        picked = selection.selection.points if selection else []
        ids = [p.get('customdata', [None])[0] for p in picked if p.get('customdata')]
        detail_id = st.selectbox('Candidate detail', list(points.candidate_id), index=0) if not points.empty else None
        if ids:
            detail_id = ids[0]
        if detail_id is not None:
            idx = int(data.candidates.index[data.candidates.candidate_id.eq(detail_id)][0])
            near = data.existing.loc[data.existing_distances[idx] <= 1000, ['store_id', 'store_name', 'region']].copy()
            near['distance_m'] = data.existing_distances[idx][data.existing_distances[idx] <= 1000]
            st.dataframe(near, hide_index=True, alt='Existing stores within 1 km of the inspected candidate')
            st.dataframe(selected.loc[selected.candidate_id.eq(detail_id)], hide_index=True, alt='Selected candidate timing, sales and attributed network impact')
        st.dataframe(selected, hide_index=True, alt='Selected portfolio with opening dates and horizon-specific contributions')
        download(selected, 'Download selected portfolio', 'selected_portfolio.csv')
    elif view == 'Comparisons':
        st.caption('This explicitly runs independent optimized fixed-180, manual and automatic scenarios plus unchanged saved benchmarks under identical forecast assumptions. It can take several search budgets.')
        if st.button('Run scenario comparison', key='expansion_compare'):
            try:
                with st.spinner('Evaluating baseline and independently optimized scenarios'):
                    table, path = comparisons(cfg_json, signature())
                st.session_state['expansion_comparison'] = (cfg_json, table, str(path))
            except (ValueError, FileNotFoundError) as error:
                st.error(str(error))
        comparison = st.session_state.get('expansion_comparison')
        if comparison and comparison[0] == cfg_json:
            st.dataframe(comparison[1], hide_index=True, alt='Comparable no-expansion, frozen, fixed-180, automatic and manual scenarios')
            download(comparison[1], 'Download comparison', 'scenario_comparison.csv')
    else:
        st.json(result['diagnostics'])
        st.caption(f"Elapsed including preparation/cache: {st.session_state['expansion_elapsed']:.2f} seconds. Saved at {result['directory']}")
        st.json(json.loads(cfg_json))
