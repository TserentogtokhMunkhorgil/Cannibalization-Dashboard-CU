"""Four business views over saved evidence and accepted-portfolio evaluations."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cannibalization_simulator import weighted_network_effect

ROOT = Path(__file__).resolve().parent
PLANNING = ROOT / 'outputs' / 'planning'
EVIDENCE = ROOT / 'outputs' / 'cannibalization_regional'
PREFIX = 'cumulative_cannibalization_'


@st.cache_data(show_spinner=False)
def _read(path: str, version: int):
    file = Path(path)
    if file.suffix == '.parquet':
        return pd.read_parquet(file)
    if file.suffix == '.xlsx':
        return pd.read_excel(file)
    return pd.read_csv(file)


def read(file: Path):
    if not file.is_file():
        return pd.DataFrame()
    return _read(str(file), file.stat().st_mtime_ns)


def download(frame, label, name):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')


def weighted(frame, outcome):
    baseline = pd.to_numeric(frame[f'baseline_APSD_{outcome}'], errors='coerce')
    effect = pd.to_numeric(frame[f'cumulative_predicted_APSD_{outcome}_effect_pct'], errors='coerce')
    valid = np.isfinite(baseline) & baseline.gt(0) & np.isfinite(effect)
    return weighted_network_effect(baseline[valid], effect[valid]) if valid.any() else np.nan


def controls(stores):
    scenario = st.sidebar.selectbox('Accepted portfolio', sorted(stores.scenario_size.unique()))
    mode = st.sidebar.radio('Calculation', ['negative_only', 'raw'],
        format_func=lambda x: 'Negative-only planning' if x == 'negative_only' else 'Raw signed estimates')
    outcome = st.sidebar.selectbox('Outcome', ['income', 'customers'],
        format_func=lambda x: 'APSD sales' if x == 'income' else 'APSD customers')
    data = stores.loc[stores.scenario_size.eq(scenario) & stores.calculation_mode.eq(mode)].copy()
    st.caption(f"Baseline: {data.baseline_month_or_period.iloc[0]} · Exact supplied cumulative coefficients · {mode.replace('_', ' ')}")
    st.caption('Previously selected portfolio evaluated with fixed coefficients. Historical selection scores do not establish optimality under these coefficients.')
    st.caption('Direct effects are assumed zero beyond 1,000 m. This is a planning assumption. Customers are transaction-derived equivalents per active day.')
    return data, scenario, mode, outcome


def metrics(data, outcome):
    effect = data[f'cumulative_predicted_APSD_{outcome}_effect_pct']
    cols = st.columns(4)
    value = weighted(data, outcome)
    cols[0].metric('APSD-weighted network effect', f'{value:.2f}%' if np.isfinite(value) else 'Unavailable')
    cols[1].metric('Stores exposed within 1 km', int(data.selected_pair_count.gt(0).sum()))
    cols[2].metric('Stores with decline >10%', int(effect.lt(-10).sum()))
    cols[3].metric('Stores with supported effects', int(effect.notna().sum()))
    st.caption('Network weights include supported zero-exposure stores. Indicators refer to the existing network; no annualization or profitability forecast is applied.')


def evidence(outcome):
    file = 'regional_sales_coefficients.csv' if outcome == 'income' else 'regional_customer_coefficients.csv'
    refreshed = ROOT / 'outputs' / 'cumulative_evidence' / file
    frame = read(refreshed if refreshed.is_file() else EVIDENCE / file)
    return frame.loc[frame.model_type.eq('cumulative')].copy() if not frame.empty else frame


def overview(data, outcome):
    metrics(data, outcome)
    regional = pd.DataFrame([
        {'region': region, 'APSD-weighted effect (%)': weighted(group, outcome)}
        for region, group in data.groupby('region')])
    st.plotly_chart(px.bar(regional, x='region', y='APSD-weighted effect (%)', title='Planned portfolio impact by existing-store region'), width='stretch')
    from regional_cumulative_impact import FIXED_INCOME, FIXED_CUSTOMERS, BANDS
    table = FIXED_INCOME if outcome == 'income' else FIXED_CUSTOMERS
    points = pd.DataFrame([{'Region': region, 'Distance band': band, 'Raw effect (%)': 100 * np.expm1(beta)}
        for region, values in table.items() for (band, _, _), beta in zip(BANDS, values)])
    st.plotly_chart(px.bar(points, x='Distance band', y='Raw effect (%)', color='Region', barmode='group', title='One additional opening: supplied signed coefficients'), width='stretch')
    st.caption('Each coefficient is a log change per additional opening. For several openings, add the log contributions and convert with 100 × (exp(total) − 1). City Outskirts sales use All Regions as the explicit fallback.')


def regression(outcome):
    frame = evidence(outcome)
    if frame.empty:
        st.info('Saved cumulative regression evidence is unavailable.')
        return
    region = st.selectbox('Regression region', ['All regions', *frame.region.unique()])
    shown = frame if region == 'All regions' else frame.loc[frame.region.eq(region)]
    columns = [c for c in ['region', 'outcome', 'distance_band', 'coefficient', 'percent_effect', 'standard_error', 'ci_95_lower', 'ci_95_upper', 'p_value', 'observations', 'stores'] if c in shown]
    st.caption('Saved estimated cumulative regressions with store and calendar-month fixed effects and store-clustered covariance. These preserve signed estimates and uncertainty. Planning uses the separately supplied rounded coefficients.')
    st.dataframe(shown[columns], hide_index=True, width='stretch')
    st.plotly_chart(px.scatter(shown, x='distance_band', y='percent_effect', color='region',
        error_y=shown.percent_ci_95_upper - shown.percent_effect,
        error_y_minus=shown.percent_effect - shown.percent_ci_95_lower,
        title='Estimated signed effects and 95% intervals'), width='stretch')
    download(shown, 'Download cumulative regression evidence', 'cumulative_regression_evidence.csv')
    with st.expander('Model diagnostics and limitations'):
        st.write('Saved pre-opening trend tests reject at 5%. These associations do not establish causal effects. Nearby openings may respond to local demand trends; store clustering may leave shared-event dependence across stores.')
        st.dataframe(read(EVIDENCE / 'pretrend_tests.csv'), hide_index=True)
        download(read(EVIDENCE / 'model_status.csv'), 'Download detailed model diagnostics', 'model_status.csv')


def map_figure(data, candidates, outcome):
    valid = data.loc[data.latitude.between(-90, 90) & data.longitude.between(-180, 180)].copy()
    if valid.empty:
        st.info('No stores with valid coordinates in this view.')
        return
    effect = f'cumulative_predicted_APSD_{outcome}_effect_pct'
    fig = px.scatter_map(valid, lat='latitude', lon='longitude', color=effect,
        color_continuous_scale='RdYlGn', hover_name='store_name', zoom=10,
        center={'lat': valid.latitude.mean(), 'lon': valid.longitude.mean()}, map_style='open-street-map', height=500)
    if not candidates.empty:
        fig.add_trace(go.Scattermap(lat=candidates.latitude, lon=candidates.longitude,
            text=candidates.candidate_name, mode='markers', name='Selected new locations',
            marker={'size': 11, 'color': '#2864dc'}, hovertemplate='%{text}<extra>Selected candidate</extra>'))
    st.plotly_chart(fig, width='stretch')


def locations(data, scenario, mode, outcome):
    region = st.selectbox('Existing-store region', ['All regions', *sorted(data.region.unique())])
    shown = data if region == 'All regions' else data.loc[data.region.eq(region)]
    metrics(shown, outcome)
    candidates = read(PLANNING / (PREFIX + 'candidates.csv'))
    selected = candidates.loc[candidates[f'in_scenario_{scenario}'].astype(str).str.lower().eq('true')]
    st.caption('Region filters apply to existing-store metrics and tables. All accepted candidates remain visible on the map.')
    map_figure(shown, selected, outcome)
    effect = f'cumulative_predicted_APSD_{outcome}_effect_pct'
    columns = ['existing_store_id', 'store_name', 'region', 'selected_pair_count', f'baseline_APSD_{outcome}', effect, f'predicted_scenario_APSD_{outcome}']
    st.dataframe(shown.sort_values(effect)[columns], hide_index=True, width='stretch')
    download(shown, 'Download existing-store impacts', f'impacts_{scenario}_{mode}_{outcome}.csv')
    with st.expander('Accepted candidate locations'):
        st.dataframe(selected[['candidate_id', 'candidate_name', 'latitude', 'longitude']], hide_index=True)
        download(selected, 'Download accepted candidate locations', f'accepted_candidates_{scenario}.csv')


def store_detail(data, scenario, mode, outcome):
    search = st.text_input('Search store ID or name')
    matches = data.loc[data.existing_store_id.astype(str).str.contains(search, case=False, regex=False)
        | data.store_name.str.contains(search, case=False, regex=False)]
    if matches.empty:
        st.info('No matching stores.')
        return
    names = matches.set_index('existing_store_id').store_name.to_dict()
    store_id = st.selectbox('Existing store', list(names), format_func=lambda value: f'{value} · {names[value]}')
    one = matches.loc[matches.existing_store_id.eq(store_id)]
    metrics(one, outcome)
    row = one.iloc[0]
    st.write(f"{row.store_name} · {row.region}")
    st.write(f"Baseline APSD: {row[f'baseline_APSD_{outcome}']:,.2f} · Scenario APSD: {row[f'predicted_scenario_APSD_{outcome}']:,.2f}")
    # Read the columnar detail artifact only on this page, after store selection.
    all_pairs = read(PLANNING / 'contributions.parquet')
    pairs = all_pairs.loc[all_pairs.scenario_size.eq(scenario) & all_pairs.calculation_mode.eq(mode)
        & all_pairs.existing_store_id.astype(str).eq(str(store_id))]
    if not pairs.empty:
        source = 'income' if outcome == 'income' else 'customer'
        st.plotly_chart(px.bar(pairs, x='candidate_name', y=f'direct_APSD_{outcome}_effect_pct', color='distance_band', title='All selected nearby candidate contributions'), width='stretch')
        st.dataframe(pairs[['candidate_id', 'candidate_name', 'distance_m', 'distance_band', f'{source}_coefficient_source', f'regional_{source}_log_coefficient']], hide_index=True)
        points = pairs.rename(columns={'candidate_latitude': 'latitude', 'candidate_longitude': 'longitude'})
        map_figure(one, points, outcome)
    else:
        st.info('No qualifying candidate exposures; check model status for stores with missing coordinates.')
    st.caption('Scenario effect = exp(sum of selected pair log contributions) − 1; no future trend is extrapolated.')
    history = read(ROOT / 'store_master_data.xlsx')
    history['store_id'] = history.store_id.astype(str).str.replace(r'\.0$', '', regex=True)
    observed = history.loc[history.store_id.eq(str(store_id))].copy()
    observed['SalesMonth'] = pd.to_datetime(observed.SalesMonth)
    if not observed.empty:
        st.plotly_chart(px.line(observed.sort_values('SalesMonth'), x='SalesMonth', y=f'APSD_{outcome}', title='Observed historical monthly APSD'), width='stretch')
    download(one, 'Download store scenario', f'store_{store_id}_{scenario}_{mode}.csv')


def main():
    st.set_page_config(page_title='CU Cannibalization', layout='wide')
    page = st.sidebar.selectbox('View', ['Cannibalization Overview', 'Regression Results', 'Planned New Store Locations', 'Store-Level Exploration'])
    st.title(page)
    if page == 'Regression Results':
        outcome = st.sidebar.selectbox('Outcome', ['income', 'customers'])
        regression(outcome)
        return
    stores = read(PLANNING / (PREFIX + 'scenario_store_results.csv'))
    if stores.empty:
        st.info('Generate planning outputs with: python regional_cumulative_impact.py --as-of YYYY-MM-DD')
        return
    data, scenario, mode, outcome = controls(stores)
    if page == 'Cannibalization Overview':
        overview(data, outcome)
    elif page == 'Planned New Store Locations':
        locations(data, scenario, mode, outcome)
    else:
        store_detail(data, scenario, mode, outcome)


if __name__ == '__main__':
    main()
