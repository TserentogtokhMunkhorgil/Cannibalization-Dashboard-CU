"""Business-facing CU network dashboard driven by one active expansion scenario."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from expansion_scenarios import (
    ROOT, evaluate_expansion_scenario, evaluation_signature,
    scenario_discovery_issues, selectable_scenarios,
)


EVIDENCE = ROOT / "outputs" / "cannibalization_regional"


def _money(value: object) -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return "Unavailable"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} bn MNT"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f} m MNT"
    return f"{value:,.0f} MNT"


def _percent(value: object) -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{value:+.2f}%" if np.isfinite(value) else "Unavailable"


@st.cache_data(show_spinner=False)
def _read(path: str, modified_ns: int) -> pd.DataFrame:
    file = Path(path)
    if not file.is_file():
        return pd.DataFrame()
    if file.suffix == ".parquet":
        return pd.read_parquet(file)
    if file.suffix == ".xlsx":
        return pd.read_excel(file)
    return pd.read_csv(file, encoding="utf-8-sig")


def read(path: Path) -> pd.DataFrame:
    return _read(str(path), path.stat().st_mtime_ns if path.is_file() else 0)


@st.cache_data(max_entries=8, show_spinner="Loading the selected expansion scenario...")
def _evaluate_cached(reference_json: str, source_signature: tuple[tuple[str, int, int], ...]) -> dict:
    # ``source_signature`` deliberately invalidates cached results when either
    # this portfolio or its reusable model inputs have changed.
    del source_signature
    return evaluate_expansion_scenario(json.loads(reference_json), ROOT)


@st.cache_data(show_spinner=False)
def _historical_apsd(master_modified_ns: int) -> pd.DataFrame:
    path = ROOT / "store_master_data.xlsx"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_excel(path)
    wanted = [name for name in ("store_id", "store_name", "SalesMonth", "APSD_income", "APSD_customers") if name in frame]
    frame = frame[wanted].copy()
    frame["store_id"] = frame["store_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    frame["SalesMonth"] = pd.to_datetime(frame["SalesMonth"], errors="coerce")
    return frame.dropna(subset=["SalesMonth"])


def download(frame: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), file_name, "text/csv")


def _active_scenario_selector() -> dict:
    options = selectable_scenarios(ROOT)
    by_key = {(item["kind"], item["scenario_id"]): item for item in options}
    current = st.session_state.get("active_expansion_scenario")
    current_key = (current or {}).get("kind"), (current or {}).get("scenario_id")
    if current and current_key not in by_key:
        # Retain the invalid active reference until a user intentionally picks
        # another one. This prevents a hidden fallback to the 180 benchmark or
        # any other portfolio.
        unavailable = dict(current)
        unavailable["label"] = f"Unavailable active scenario · {current.get('scenario_id', 'unknown')}"
        options.insert(0, unavailable)
        default = unavailable
    else:
        default = by_key.get(current_key, options[0])
    labels = [item["label"] for item in options]
    widget_key = "active_expansion_scenario_selector"
    if widget_key not in st.session_state or st.session_state[widget_key] not in labels:
        st.session_state[widget_key] = default["label"]
    chosen_label = st.selectbox("Active Expansion Scenario", labels, key=widget_key)
    selected = next(item for item in options if item["label"] == chosen_label)
    if (current or {}).get("scenario_id") != selected["scenario_id"] or (current or {}).get("kind") != selected["kind"]:
        st.session_state["active_expansion_scenario"] = {
            key: selected.get(key) for key in ("kind", "scenario_id", "label", "directory")
        }
    return st.session_state["active_expansion_scenario"]


def _scenario_header(evaluation: dict) -> None:
    summary = evaluation["summary"]
    st.title("CU Network & Cannibalization")
    st.caption(
        "A single schedule-aware scenario evaluation. Historical APSD, regression coefficients, "
        "and forward-looking model outputs are deliberately shown as distinct concepts."
    )
    metadata = st.columns(5)
    metadata[0].metric("Selected new stores", f"{int(summary['store_count']):,}")
    metadata[1].metric("Planning year", str(summary["planning_year"]))
    metadata[2].metric("Optimization mode", summary["scenario_mode"])
    metadata[3].metric("Objective", "Network sales" if summary["objective"] == "sales" else "NPV")
    metadata[4].metric("Forecast horizon", f"{int(summary['forecast_horizon'])} months")
    st.caption(f"Scenario ID: `{summary['scenario_id']}` · demand: {summary['demand_scenario']} · "
               f"sales coefficient source: {summary['coefficient_source']} / {summary['coefficient_mode']}.")


def _network_overview(evaluation: dict) -> None:
    summary, monthly = evaluation["summary"], evaluation["monthly"]
    st.subheader("1 — Network Overview")
    st.caption(
        f"Monetary figures are cumulative MNT across the {int(summary['forecast_horizon'])}-month forecast horizon. "
        "APSD is not added to revenue."
    )
    cards = st.columns(4)
    cards[0].metric("Existing CU stores", f"{len(evaluation['existing_map']):,}")
    cards[1].metric("Selected new stores", f"{int(summary['store_count']):,}")
    cards[2].metric("Existing-network baseline sales", _money(summary["existing_network_baseline_sales"]))
    cards[3].metric("Predicted new-store sales", _money(summary["new_store_sales"]))
    cards = st.columns(4)
    cards[0].metric("Existing-store cannibalization", _money(summary["existing_store_change"]))
    cards[1].metric("New-to-new cannibalization", _money(summary["new_to_new_change"]))
    cards[2].metric("Net incremental network sales", _money(summary["incremental_network_sales"]))
    cards[3].metric("Projected network sales", _money(summary["total_network_sales"]))

    before_after = pd.DataFrame({
        "Network state": ["Before expansion", "After expansion"],
        "Existing-store sales (MNT)": [summary["existing_network_baseline_sales"], summary["existing_network_scenario_sales"]],
        "New-store sales (MNT)": [0.0, summary["new_store_sales"]],
        "Total network sales (MNT)": [summary["existing_network_baseline_sales"], summary["total_network_sales"]],
    })
    left, right = st.columns((1, 1.4))
    with left:
        st.markdown("**Before Expansion vs After Expansion**")
        st.dataframe(before_after, hide_index=True, width="stretch", column_config={
            column: st.column_config.NumberColumn(format="%,.0f") for column in before_after.columns[1:]
        })
    with right:
        trend = monthly.melt(
            id_vars="month",
            value_vars=["gross_new_store_sales", "new_to_new_change", "existing_store_change", "incremental_network_sales"],
            var_name="Component", value_name="MNT",
        )
        trend["Component"] = trend["Component"].map({
            "gross_new_store_sales": "Gross new-store sales", "new_to_new_change": "New-to-new change",
            "existing_store_change": "Existing-store change", "incremental_network_sales": "Net incremental sales",
        })
        st.plotly_chart(px.line(trend, x="month", y="MNT", color="Component", markers=True,
            title="Monthly scheduled network contribution"), width="stretch")


def _cannibalization_overview(evaluation: dict) -> None:
    summary = evaluation["summary"]
    impacts, pairs, monthly = evaluation["existing_impacts"], evaluation["pair_impacts"], evaluation["existing_monthly"]
    st.subheader("2 — Cannibalization Overview")
    st.caption(
        "Modeled effects apply the active portfolio's opening schedule and cumulative overlapping log effects. "
        "They are regression-based associations for planning, not proven causal effects. Direct modeled exposure is zero beyond 1,000 m."
    )
    weighted_effect = 100 * impacts["horizon_existing_sales_change"].sum() / impacts["baseline_sales_mnt"].sum() if impacts["baseline_sales_mnt"].sum() else np.nan
    cards = st.columns(4)
    cards[0].metric("Existing-store sales impact", _money(impacts["horizon_existing_sales_change"].sum()))
    cards[1].metric("Existing stores affected", f"{int(impacts['exposed_within_1000m'].sum()):,}")
    cards[2].metric("APSD-weighted existing-network impact", _percent(weighted_effect))
    if summary["customer_model_supported"]:
        base = monthly["baseline_APSD_customers"].sum()
        customer_effect = 100 * (monthly["scenario_APSD_customers"].sum() - base) / base if base else np.nan
        cards[3].metric("APSD customer-equivalent impact", _percent(customer_effect))
    else:
        cards[3].metric("APSD customer-equivalent impact", "Unavailable")
        st.caption(summary["customer_model_note"])

    region = impacts.groupby("region", as_index=False).agg(
        existing_stores=("store_id", "nunique"),
        baseline_sales_mnt=("baseline_sales_mnt", "sum"),
        existing_store_sales_change=("horizon_existing_sales_change", "sum"),
    )
    region["existing_sales_effect_pct"] = np.where(
        region["baseline_sales_mnt"].ne(0), 100 * region["existing_store_sales_change"] / region["baseline_sales_mnt"], np.nan
    )
    distance_order = ["0–100 m", "100–250 m", "250–500 m", "500–1,000 m"]
    band_names = {"0_100": "0–100 m", "100_250": "100–250 m", "250_500": "250–500 m", "500_1000": "500–1,000 m"}
    if pairs.empty:
        distance = pd.DataFrame({"Distance band": distance_order, "Existing-store sales change (MNT)": [0.0] * 4, "Affected existing stores": [0] * 4})
    else:
        distance = pairs.assign(**{"Distance band": pairs["distance_band"].map(band_names)}).groupby("Distance band", as_index=False).agg(
            **{"Existing-store sales change (MNT)": ("horizon_existing_sales_change", "sum"),
               "Affected existing stores": ("existing_store_id", "nunique"), "Selected pair exposures": ("candidate_id", "size")}
        )
        distance["Distance band"] = pd.Categorical(distance["Distance band"], distance_order, ordered=True)
        distance = distance.sort_values("Distance band")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(region, x="region", y="existing_sales_effect_pct", title="Existing-store sales impact by region"), width="stretch")
        st.dataframe(region, hide_index=True, width="stretch", column_config={
            "baseline_sales_mnt": st.column_config.NumberColumn(format="%,.0f"),
            "existing_store_sales_change": st.column_config.NumberColumn(format="%,.0f"),
            "existing_sales_effect_pct": st.column_config.NumberColumn(format="%.2f%%"),
        })
    with right:
        st.plotly_chart(px.bar(distance, x="Distance band", y="Existing-store sales change (MNT)",
            title="Allocated existing-store impact by distance band"), width="stretch")
        st.dataframe(distance, hide_index=True, width="stretch")


def _planned_locations(evaluation: dict) -> None:
    candidates, selected, pairs, existing = evaluation["candidates"].copy(), evaluation["selected"].copy(), evaluation["pair_impacts"], evaluation["existing_map"]
    st.subheader("3 — Planned New Store Locations")
    st.caption("Selected and unselected locations are from the active scenario's eligible candidate pool; no benchmark portfolio is substituted.")
    areas = candidates.get("geographic_area", pd.Series("Unclassified", index=candidates.index)).fillna("Unclassified").astype(str)
    regions = sorted(areas.unique())
    filters = st.columns(4)
    selected_regions = filters[0].multiselect("Candidate region", regions, default=regions, key="network_location_regions")
    status = filters[1].selectbox("Selected / unselected", ["All", "Selected", "Unselected"], key="network_location_status")
    months = filters[2].multiselect("Opening month", list(range(1, 13)), default=list(range(1, 13)), key="network_location_months")
    nearby = filters[3].selectbox("Nearby existing stores", ["All", "Within 1 km", "Beyond 1 km"], key="network_location_nearby")
    shown = candidates.loc[areas.isin(selected_regions)].copy()
    if status != "All":
        shown = shown.loc[shown["selection_status"].eq("Selected" if status == "Selected" else "Unselected candidate")]
    shown = shown.loc[shown["opening_month"].isna() | shown["opening_month"].isin(months)]
    if nearby == "Within 1 km":
        shown = shown.loc[shown["nearby_existing_store_count"].gt(0)]
    elif nearby == "Beyond 1 km":
        shown = shown.loc[shown["nearby_existing_store_count"].eq(0)]
    valid = shown.loc[shown["latitude"].between(-90, 90) & shown["longitude"].between(-180, 180)].copy()
    if not valid.empty:
        fig = px.scatter_map(valid, lat="latitude", lon="longitude", color="selection_status", hover_name="candidate_name",
            hover_data=[column for column in ("candidate_id", "geographic_area", "opening_month", "scenario_APSD_sales", "nearby_existing_store_count") if column in valid],
            map_style="open-street-map", zoom=5, height=600, title="Existing CU stores and active expansion portfolio")
        existing_valid = existing.loc[existing["latitude"].between(-90, 90) & existing["longitude"].between(-180, 180)]
        fig.add_trace(go.Scattermap(lat=existing_valid["latitude"], lon=existing_valid["longitude"], mode="markers",
            name="Existing CU store", text=existing_valid["store_name"], marker={"size": 6, "color": "#666666"},
            hovertemplate="%{text}<extra>Existing CU store</extra>"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No candidate locations match the selected filters.")

    selected = selected.merge(candidates[["candidate_id", "nearby_existing_store_count", "nearby_existing_stores_within_1km"]],
        on="candidate_id", how="left", validate="one_to_one")
    columns = [column for column in (
        "candidate_id", "candidate_name", "latitude", "longitude", "geographic_area", "opening_month", "opening_date", "scenario_APSD_sales",
        "nearby_existing_store_count", "nearby_existing_stores_within_1km", "allocated_existing_store_change", "new_to_new_change", "allocated_incremental_network_sales",
    ) if column in selected]
    st.markdown("**Selected portfolio**")
    st.dataframe(selected[columns].sort_values(["opening_month", "candidate_id"]), hide_index=True, width="stretch", column_config={
        column: st.column_config.NumberColumn(format="%,.0f") for column in ("scenario_APSD_sales", "allocated_existing_store_change", "new_to_new_change", "allocated_incremental_network_sales") if column in selected
    })
    download(selected[columns], "Download active selected portfolio", f"active_portfolio_{evaluation['summary']['scenario_id']}.csv")
    if selected.empty:
        return
    option_ids = selected["candidate_id"].astype(str).tolist()
    chosen = st.selectbox("Selected location detail", option_ids, format_func=lambda candidate_id: (
        f"{candidate_id} · {selected.loc[selected.candidate_id.astype(str).eq(candidate_id), 'candidate_name'].iloc[0]}"
    ), key="network_candidate_detail")
    detail_pairs = pairs.loc[pairs["candidate_id"].astype(str).eq(chosen)].sort_values("distance_m") if not pairs.empty else pd.DataFrame()
    if detail_pairs.empty:
        st.info("This selected location has no modeled existing-store exposure within 1 km.")
    else:
        st.dataframe(detail_pairs[[column for column in (
            "existing_store_id", "existing_store_name", "existing_region", "distance_m", "distance_band", "opening_date", "opening_month_APSD_effect_pct", "horizon_existing_sales_change"
        ) if column in detail_pairs]], hide_index=True, width="stretch")


def _store_exploration(evaluation: dict) -> None:
    impacts, forecast, pairs = evaluation["existing_impacts"], evaluation["existing_monthly"], evaluation["pair_impacts"]
    st.subheader("4 — Store-Level Exploration")
    options = impacts[["store_id", "store_name", "region"]].drop_duplicates().sort_values("store_name")
    if options.empty:
        st.info("No existing CU stores are available for this scenario.")
        return
    lookup = options.set_index("store_id").store_name.to_dict()
    selected_id = st.selectbox("Existing CU store", options["store_id"].tolist(),
        format_func=lambda store_id: f"{lookup[store_id]} · ID {store_id}", key="network_store_detail")
    one_impact = impacts.loc[impacts["store_id"].astype(str).eq(str(selected_id))].iloc[0]
    one_forecast = forecast.loc[forecast["store_id"].astype(str).eq(str(selected_id))].sort_values("month")
    historic_path = ROOT / "store_master_data.xlsx"
    history = _historical_apsd(historic_path.stat().st_mtime_ns if historic_path.is_file() else 0)
    history = history.loc[history["store_id"].eq(str(selected_id))].sort_values("SalesMonth") if not history.empty else history
    st.write(f"**Store:** {one_impact.store_name} · **Region:** {one_impact.region} · **Store ID:** {selected_id}")
    cards = st.columns(4)
    latest_sales = pd.to_numeric(history.get("APSD_income", pd.Series(dtype=float)), errors="coerce").dropna()
    latest_customers = pd.to_numeric(history.get("APSD_customers", pd.Series(dtype=float)), errors="coerce").dropna()
    cards[0].metric("Latest observed APSD sales", f"{latest_sales.iloc[-1]:,.0f}" if not latest_sales.empty else "Unavailable")
    cards[1].metric("Latest observed APSD customers", f"{latest_customers.iloc[-1]:,.1f}" if not latest_customers.empty else "Unavailable")
    cards[2].metric("Forecast APSD sales without expansion", f"{one_forecast['baseline_APSD_sales'].mean():,.0f}")
    cards[3].metric("Forecast APSD sales after expansion", f"{one_forecast['scenario_APSD_sales'].mean():,.0f}", _percent(one_impact.horizon_existing_sales_effect_pct))

    metric = st.radio("APSD series", ["Sales", "Customer-equivalents"], horizontal=True, key="network_store_metric")
    fig = go.Figure()
    if metric == "Sales":
        if not history.empty and "APSD_income" in history:
            fig.add_scatter(x=history["SalesMonth"], y=history["APSD_income"], mode="lines+markers", name="Historical APSD sales")
        fig.add_scatter(x=one_forecast["month"], y=one_forecast["baseline_APSD_sales"], mode="lines", line={"dash": "dash"}, name="Forecast APSD without expansion")
        fig.add_scatter(x=one_forecast["month"], y=one_forecast["scenario_APSD_sales"], mode="lines+markers", name="Forecast APSD after expansion")
        title = "Historical and scheduled sales APSD forecast"
    elif "scenario_APSD_customers" in one_forecast:
        if not history.empty and "APSD_customers" in history:
            fig.add_scatter(x=history["SalesMonth"], y=history["APSD_customers"], mode="lines+markers", name="Historical APSD customers")
        fig.add_scatter(x=one_forecast["month"], y=one_forecast["baseline_APSD_customers"], mode="lines", line={"dash": "dash"}, name="Forecast APSD customers without expansion")
        fig.add_scatter(x=one_forecast["month"], y=one_forecast["scenario_APSD_customers"], mode="lines+markers", name="Forecast APSD customers after expansion")
        title = "Historical and scheduled customer-equivalent APSD forecast"
    else:
        title = "Historical APSD customers"
        if not history.empty and "APSD_customers" in history:
            fig.add_scatter(x=history["SalesMonth"], y=history["APSD_customers"], mode="lines+markers", name="Historical APSD customers")
        st.caption(evaluation["summary"]["customer_model_note"])
    nearby = pairs.loc[pairs["existing_store_id"].astype(str).eq(str(selected_id))].sort_values("opening_date") if not pairs.empty else pd.DataFrame()
    for event in nearby.itertuples(index=False):
        fig.add_vline(x=event.opening_date, line_dash="dot", line_color="#B91C1C",
            annotation_text=f"{event.candidate_name} · {event.distance_m:,.0f} m", annotation_textangle=-90)
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="APSD", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Forecast impacts begin only in each selected location's planned opening month; dotted markers identify nearby selected openings.")
    if nearby.empty:
        st.info("No selected new stores are within 1 km of this existing store.")
    else:
        st.markdown("**Nearby selected new stores and cumulative modeled contribution**")
        st.dataframe(nearby[[column for column in (
            "candidate_id", "candidate_name", "opening_date", "opening_month", "distance_m", "distance_band", "opening_month_APSD_effect_pct", "horizon_existing_sales_change"
        ) if column in nearby]], hide_index=True, width="stretch")


def network_and_cannibalization_page() -> None:
    reference = _active_scenario_selector()
    for issue in scenario_discovery_issues(ROOT):
        st.warning(f"Saved expansion scenario is unavailable: {issue}")
    reference_json = json.dumps(reference, sort_keys=True)
    try:
        evaluation = _evaluate_cached(reference_json, evaluation_signature(reference, ROOT))
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as error:
        st.error(f"The active scenario could not be evaluated: {error}")
        st.info("Choose another completed scenario or create one on the Expansion Optimization page. No substitute portfolio was loaded.")
        return
    _scenario_header(evaluation)
    section = st.segmented_control(
        "Dashboard section",
        ["Network Overview", "Cannibalization Overview", "Planned New Store Locations", "Store-Level Exploration"],
        default="Network Overview",
        key="network_dashboard_section",
    )
    # Unlike st.tabs, this renders only the section being inspected. The shared
    # evaluation remains cached, while map and store history work is deferred.
    if section == "Network Overview":
        _network_overview(evaluation)
    elif section == "Cannibalization Overview":
        _cannibalization_overview(evaluation)
    elif section == "Planned New Store Locations":
        _planned_locations(evaluation)
    else:
        _store_exploration(evaluation)


def regression_results_page() -> None:
    st.title("Regression Results")
    st.caption("Saved regression evidence is distinct from the active scenario evaluation. Associations do not, by themselves, prove causality.")
    outcome = st.selectbox("Outcome", ["APSD sales", "APSD customer-equivalents"], key="regression_outcome")
    file = EVIDENCE / ("regional_sales_coefficients.csv" if outcome == "APSD sales" else "regional_customer_coefficients.csv")
    data = read(file)
    if data.empty:
        st.info("Saved regression evidence is unavailable.")
        return
    data = data.loc[data.get("model_type", pd.Series(index=data.index, dtype=str)).eq("cumulative")].copy()
    if data.empty:
        st.info("No cumulative coefficient rows are available for this outcome.")
        return
    region = st.selectbox("Existing-store region", ["All regions", *sorted(data["region"].dropna().unique())], key="regression_region")
    shown = data if region == "All regions" else data.loc[data["region"].eq(region)]
    st.plotly_chart(px.bar(shown, x="distance_band", y="percent_effect", color="region", barmode="group",
        title="Saved cumulative APSD associations by distance band"), width="stretch")
    columns = [name for name in ("region", "distance_band", "coefficient", "percent_effect", "standard_error", "ci_95_lower", "ci_95_upper", "p_value", "observations", "stores") if name in shown]
    st.dataframe(shown[columns], hide_index=True, width="stretch")
    download(shown[columns], "Download regression evidence", "cumulative_regression_evidence.csv")


def main() -> None:
    st.set_page_config(page_title="CU Network & Cannibalization", page_icon="🏪", layout="wide")
    page = st.sidebar.selectbox("View", ["CU Network & Cannibalization", "Expansion Optimization", "Regression Results"])
    if page == "Expansion Optimization":
        st.title("Expansion Optimization")
        from expansion_optimization_page import render
        render()
    elif page == "Regression Results":
        regression_results_page()
    else:
        network_and_cannibalization_page()


if __name__ == "__main__":
    main()
