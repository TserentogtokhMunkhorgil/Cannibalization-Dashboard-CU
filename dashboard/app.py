"""Interactive CU store network and performance dashboard."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORE_MONTH_FILE = DATA_DIR / "dashboard_store_month.parquet"
EXPOSURE_FILE = DATA_DIR / "potential_store_exposure.parquet"
QUALITY_FILE = BASE_DIR / "outputs" / "data_quality_report.csv"
DISTANCE_THRESHOLDS = (100, 250, 500, 1000)

METRIC_OPTIONS = {
    "Store location": None,
    "APSD": "APSD",
    "APSD change %": "APSD_change_pct",
    "Customers": "customers",
    "Customer change %": "customer_change_pct",
    "Nearby new stores <=100m": "nearby_new_stores_100m",
    "Nearby new stores <=250m": "nearby_new_stores_250m",
    "Nearby new stores <=500m": "nearby_new_stores_500m",
    "Nearby new stores <=1,000m": "nearby_new_stores_1000m",
}


st.set_page_config(
    page_title="CU Store Network & Performance",
    page_icon="🏪",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    store_month = pd.read_parquet(STORE_MONTH_FILE)
    exposure = pd.read_parquet(EXPOSURE_FILE)
    store_month["date"] = pd.to_datetime(store_month["date"])
    exposure["new_opening_date"] = pd.to_datetime(exposure["new_opening_date"])
    exposure["apsd_reference_month"] = pd.to_datetime(
        exposure["apsd_reference_month"]
    )
    return store_month, exposure
st

def _format_mnt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    amount = float(value)
    if abs(amount) >= 1_000_000_000:
        return f"₮{amount / 1_000_000_000:,.2f}b"
    if abs(amount) >= 1_000_000:
        return f"₮{amount / 1_000_000:,.2f}m"
    return f"₮{amount:,.0f}"


def _format_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):,.0f}"


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "Not available"
    return f"{float(value):+,.1f}%"


def _format_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return pd.Timestamp(value).strftime("%d-%b-%Y")


def _marker_style(frame: pd.DataFrame, metric_column: str | None) -> pd.DataFrame:
    styled = frame.copy()
    values = (
        pd.to_numeric(styled[metric_column], errors="coerce")
        if metric_column
        else pd.Series(np.nan, index=styled.index, dtype=float)
    )
    values = values.where(np.isfinite(values))

    if metric_column in {"APSD_change_pct", "customer_change_pct"}:
        def change_color(value: Any) -> list[int]:
            if pd.isna(value):
                return [145, 151, 160, 190]
            if value > 0:
                return [33, 150, 105, 220]
            if value < 0:
                return [216, 73, 73, 220]
            return [125, 132, 142, 210]

        styled["fill_color"] = [change_color(value) for value in values]
        sizes = values.abs().fillna(0)
        styled["marker_radius"] = 75 + sizes.clip(upper=30) / 30 * 360
    elif metric_column and metric_column.startswith("nearby_new_stores_"):
        max_value = max(float(values.max()) if values.notna().any() else 0, 1)
        normalized = values.fillna(0).clip(lower=0) / max_value
        styled["fill_color"] = [
            [255, int(231 - 119 * value), int(190 - 160 * value), 220]
            if pd.notna(raw)
            else [145, 151, 160, 190]
            for raw, value in zip(values, normalized)
        ]
        styled["marker_radius"] = 85 + normalized * 390
    elif metric_column in {"APSD", "customers"}:
        lower = float(values.min()) if values.notna().any() else 0
        upper = float(values.max()) if values.notna().any() else 0
        normalized = (
            (values - lower) / (upper - lower)
            if upper > lower
            else pd.Series(0.5, index=values.index)
        )
        styled["fill_color"] = [
            [int(89 - 45 * value), int(160 - 70 * value), 220, 220]
            if pd.notna(raw)
            else [145, 151, 160, 190]
            for raw, value in zip(values, normalized)
        ]
        styled["marker_radius"] = 85 + normalized.fillna(0.5) * 390
    else:
        styled["fill_color"] = [[42, 112, 184, 220] for _ in range(len(styled))]
        styled["marker_radius"] = 130.0

    styled["marker_radius"] = styled["marker_radius"].fillna(100).astype(float)
    styled["hover_apsd"] = styled["APSD"].map(_format_mnt)
    styled["hover_customers"] = styled["customers"].map(_format_count)
    styled["hover_change"] = styled["APSD_change_pct"].map(_format_percent)
    return styled


def _map_event_store_id(event: Any) -> str | None:
    if not hasattr(event, "get"):
        return None
    selection = event.get("selection", {})
    if not hasattr(selection, "get"):
        return None
    objects = selection.get("objects", {})
    if not hasattr(objects, "values"):
        return None
    for selected_objects in objects.values():
        if selected_objects:
            selected = selected_objects[0]
            if hasattr(selected, "get") and selected.get("store_id") is not None:
                return str(selected["store_id"])
    return None


def _make_history_chart(history: pd.DataFrame, store_name: str) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.68, 0.32],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["APSD"],
            mode="lines+markers",
            name="APSD",
            connectgaps=False,
            line={"color": "#2867A8", "width": 3},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    if "customers" in history.columns:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["customers"],
                mode="lines+markers",
                name="Customers",
                connectgaps=False,
                line={"color": "#E49A38", "dash": "dot"},
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    changes = history["APSD_change_pct"]
    figure.add_trace(
        go.Bar(
            x=history["date"],
            y=changes,
            name="APSD MoM %",
            marker_color=[
                "#27966A" if value >= 0 else "#D84949"
                for value in changes.fillna(0)
            ],
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="APSD (MNT / day)", row=1, col=1, secondary_y=False)
    if "customers" in history.columns:
        figure.update_yaxes(title_text="Customers", row=1, col=1, secondary_y=True)
    figure.update_yaxes(title_text="MoM change (%)", row=2, col=1)
    figure.update_layout(
        title=f"Monthly performance — {store_name}",
        height=460,
        margin={"l": 10, "r": 10, "t": 75, "b": 10},
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 0.5,
            "xanchor": "center",
        },
        hovermode="x unified",
    )
    return figure


if not STORE_MONTH_FILE.is_file() or not EXPOSURE_FILE.is_file():
    st.error(
        "Dashboard data files are missing. Run "
        "`python dashboard\\prepare_data.py` from the project folder first."
    )
    st.stop()

store_month, exposure = load_dashboard_data()

st.title("CU Store Network & Performance")
st.caption(
    "Descriptive monthly performance and geographic exploration. "
    "Observed changes are not causal estimates."
)
st.sidebar.header("Dashboard filters")

available_years = sorted(store_month["year"].dropna().astype(int).unique().tolist())
if not available_years:
    st.error("No dated store-month data is available.")
    st.stop()
latest_date = store_month["date"].max()
default_year = int(latest_date.year)
year_value = st.sidebar.selectbox(
    "Year",
    available_years,
    index=available_years.index(default_year),
)
if year_value is None:
    st.error("Select a year to continue.")
    st.stop()
year = int(year_value)
year_months = sorted(
    store_month.loc[store_month["year"].eq(year), "date"].drop_duplicates()
)
if not year_months:
    st.error(f"No monthly data is available for {year}.")
    st.stop()
month_numbers = [int(month.month) for month in year_months]
default_month = int(latest_date.month) if latest_date.year == year else month_numbers[-1]
selected_month_value = st.sidebar.selectbox(
    "Month",
    month_numbers,
    index=month_numbers.index(default_month),
    format_func=lambda month_number: pd.Timestamp(
        year=year, month=month_number, day=1
    ).strftime("%B"),
)
if selected_month_value is None:
    st.error("Select a month to continue.")
    st.stop()
selected_month_number = int(selected_month_value)
selected_date = pd.Timestamp(year=year, month=selected_month_number, day=1)

metadata = store_month.drop_duplicates("store_id")
areas = sorted(
    metadata["location_area"].dropna().astype(str).unique().tolist()
)
selected_areas = st.sidebar.multiselect("Location / area", areas)
store_options = metadata[["store_id", "store_name"]].copy()
store_options["label"] = store_options.apply(
    lambda row: (
        f"{row['store_id']} — {row['store_name']}"
        if pd.notna(row["store_name"])
        else f"CU {row['store_id']}"
    ),
    axis=1,
)
store_options = store_options.sort_values("label")
store_label_by_id = dict(zip(store_options["store_id"], store_options["label"]))
selected_store_ids = st.sidebar.multiselect(
    "Store ID / name",
    options=store_options["store_id"].tolist(),
    format_func=lambda store_id: str(store_label_by_id[store_id]),
)
categories = [
    "Strong Growth",
    "Growth",
    "Stable",
    "Decline",
    "Strong Decline",
]
selected_categories = st.sidebar.multiselect("Performance category", categories)
map_metric_label = st.selectbox("Map metric", list(METRIC_OPTIONS))
map_metric_column = METRIC_OPTIONS[map_metric_label]

month_frame = store_month.loc[
    store_month["date"].eq(selected_date) & store_month["is_operating"]
].copy()
if selected_areas:
    month_frame = month_frame.loc[
        month_frame["location_area"].astype("string").isin(selected_areas)
    ]
if selected_store_ids:
    month_frame = month_frame.loc[month_frame["store_id"].isin(selected_store_ids)]
if selected_categories:
    month_frame = month_frame.loc[
        month_frame["performance_category"].isin(selected_categories)
    ]

display_month = selected_date.strftime("%B %Y")
st.subheader(display_month)
if selected_date.year == date.today().year and selected_date.month == date.today().month:
    st.info("The selected month is still in progress; monthly totals may be partial.")

apsd_values = pd.to_numeric(month_frame["APSD"], errors="coerce").dropna()
customer_values = (
    pd.to_numeric(month_frame["customers"], errors="coerce").dropna()
    if "customers" in month_frame.columns
    else pd.Series(dtype=float)
)
apsd_change_values = pd.to_numeric(
    month_frame["APSD_change_pct"], errors="coerce"
).dropna()
growing_count = int(apsd_change_values.gt(0).sum())
declining_count = int(apsd_change_values.lt(0).sum())

kpis = st.columns(6)
kpis[0].metric("Operating stores", f"{len(month_frame):,}")
kpis[1].metric(
    "Total APSD",
    _format_mnt(apsd_values.sum()) if not apsd_values.empty else "Not available",
)
kpis[2].metric(
    "Average APSD",
    _format_mnt(apsd_values.mean()) if not apsd_values.empty else "Not available",
)
kpis[3].metric(
    "Customers",
    _format_count(customer_values.sum()) if not customer_values.empty else "Not available",
)
kpis[4].metric("Stores growing", f"{growing_count:,}")
kpis[5].metric("Stores declining", f"{declining_count:,}")

if "customers" in store_month.columns:
    st.caption(
        "Customer values use the existing `Customers` source measure "
        "(transactions less twice returned transactions); they are not "
        "unique identified people."
    )

map_column, detail_column = st.columns([2.1, 1])
with map_column:
    st.subheader("Store map")
    valid_coordinates = month_frame["latitude"].between(
        -90, 90
    ) & month_frame["longitude"].between(-180, 180)
    map_frame = month_frame.loc[valid_coordinates].copy()
    no_coordinates = int(len(month_frame) - len(map_frame))
    if no_coordinates:
        st.warning(
            f"{no_coordinates:,} operating stores have missing or invalid "
            "coordinates and cannot be placed on the map."
        )
    if map_frame.empty:
        st.info("No operating stores with valid coordinates match these filters.")
        map_event = None
    else:
        map_frame = _marker_style(map_frame, map_metric_column)
        coordinates = map_frame[["longitude", "latitude"]]
        span = max(
            float(coordinates["longitude"].max() - coordinates["longitude"].min()),
            float(coordinates["latitude"].max() - coordinates["latitude"].min()),
            0.25,
        )
        zoom = float(np.clip(np.log2(360 / (1.3 * span)), 3, 12))
        view_state = pdk.ViewState(
            latitude=float(coordinates["latitude"].median()),
            longitude=float(coordinates["longitude"].median()),
            zoom=zoom,
            pitch=0,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_frame,
            id="store_layer",
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_line_color=[35, 45, 55, 200],
            get_radius="marker_radius",
            radius_min_pixels=5,
            radius_max_pixels=22,
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {
            "html": (
                "<b>{store_name}</b><br/>Store ID: {store_id}<br/>"
                "APSD: {hover_apsd}<br/>Customers: {hover_customers}<br/>"
                "APSD MoM: {hover_change}"
            ),
            "style": {"backgroundColor": "white", "color": "#223"},
        }
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            # PyDeck supports custom tooltip mappings although its type stub says bool.
            tooltip=tooltip,  # type: ignore[arg-type]
        )
        map_event = st.pydeck_chart(
            deck,
            width="stretch",
            height=620,
            key="cu_store_map",
            on_select="rerun",
            selection_mode="single-object",
        )
        clicked_store_id = _map_event_store_id(map_event)
        if clicked_store_id is not None:
            st.session_state["selected_store_id"] = clicked_store_id
        st.caption("Click a marker to open that store's performance panel.")
        if map_metric_label in {"APSD change %", "Customer change %"}:
            st.caption("Marker color: green = positive, red = negative, gray = zero / unavailable.")
        elif map_metric_label.startswith("Nearby new stores"):
            st.caption("Larger amber markers indicate more later-opening nearby stores.")
        elif map_metric_column:
            st.caption("Larger blue markers indicate a higher selected metric.")
        else:
            st.caption("Each marker is a CU store with a verified source coordinate.")

with detail_column:
    st.subheader("Selected store")
    selected_store_id = st.session_state.get("selected_store_id")
    current_store: pd.Series | None = None
    if selected_store_id is None:
        current_store_rows = month_frame.iloc[0:0]
    else:
        current_store_rows = month_frame.loc[
            month_frame["store_id"].eq(selected_store_id)
        ]
    if current_store_rows.empty:
        st.info("Click a store marker to view its monthly details and history.")
        selected_store_id = None
        st.session_state.pop("selected_store_id", None)
    else:
        current_store = current_store_rows.iloc[0]
        store_title = (
            current_store["store_name"]
            if pd.notna(current_store["store_name"])
            else f"CU {selected_store_id}"
        )
        st.markdown(f"### {store_title}")
        st.write(f"**Store ID:** {selected_store_id}")
        st.write(f"**Opening date:** {_format_date(current_store['opening_date'])}")
        st.write(f"**Selected month:** {display_month}")
        st.metric("APSD", _format_mnt(current_store["APSD"]))
        st.write(
            f"Previous month APSD: {_format_mnt(current_store['previous_APSD'])}"
        )
        st.write(f"APSD change: {_format_mnt(current_store['APSD_change'])}")
        st.write(f"APSD change %: {_format_percent(current_store['APSD_change_pct'])}")
        if "customers" in current_store.index:
            st.metric("Customers", _format_count(current_store["customers"]))
            st.write(
                "Previous month customers: "
                f"{_format_count(current_store.get('previous_customers'))}"
            )
            st.write(
                "Customer change: "
                f"{_format_count(current_store.get('customer_change'))}"
            )
            st.write(
                "Customer change %: "
                f"{_format_percent(current_store.get('customer_change_pct'))}"
            )
        st.markdown("**Nearby later-opening stores**")
        for threshold in DISTANCE_THRESHOLDS:
            value = current_store.get(f"nearby_new_stores_{threshold}m")
            st.write(f"Within {threshold:,} m: {_format_count(value)}")
        if st.button("Clear selected store"):
            st.session_state.pop("selected_store_id", None)
            st.rerun()

st.subheader("Store performance comparison")
comparison_columns = [
    "store_id",
    "store_name",
    "APSD",
    "previous_APSD",
    "APSD_change_pct",
]
if "customers" in month_frame.columns:
    comparison_columns.extend(
        ["customers", "customer_change_pct"]
    )
comparison_columns.extend(
    [
        "opening_date",
        "nearby_new_stores_250m",
    ]
)
comparison = month_frame[comparison_columns].rename(
    columns={
        "store_id": "Store ID",
        "store_name": "Store",
        "APSD": "APSD",
        "previous_APSD": "Previous APSD",
        "APSD_change_pct": "APSD MoM %",
        "customers": "Customers",
        "customer_change_pct": "Customer MoM %",
        "opening_date": "Opening date",
        "nearby_new_stores_250m": "Nearby new stores <=250m",
    }
)
st.dataframe(comparison, hide_index=True, width="stretch", height=340)

if selected_store_id is not None and current_store is not None:
    selected_history = store_month.loc[
        store_month["store_id"].eq(selected_store_id)
        & store_month["is_operating"]
    ].sort_values("date")
    selected_name = str(current_store["store_name"])
    st.plotly_chart(
        _make_history_chart(selected_history, selected_name),
        width="stretch",
        config={"displayModeBar": False},
    )
else:
    st.info("Select a store on the map to see its APSD and customer trends.")

st.subheader(f"New-store exposure — openings in {display_month}")
month_events = exposure.loc[
    exposure["new_opening_date"].dt.year.eq(selected_date.year)
    & exposure["new_opening_date"].dt.month.eq(selected_date.month)
].copy()
if month_events.empty:
    st.caption("No new-store opening events are listed for this month.")
else:
    event_columns = [
        "new_store_id",
        "new_store_name",
        "new_opening_date",
        "potential_stores_100m",
        "potential_stores_250m",
        "potential_stores_500m",
        "potential_stores_1000m",
        "apsd_reference_month",
        "potential_apsd_exposure_250m",
        "potential_apsd_observed_stores_250m",
    ]
    month_events = month_events[event_columns].rename(
        columns={
            "new_store_id": "New store ID",
            "new_store_name": "New store",
            "new_opening_date": "Opening date",
            "potential_stores_100m": "Existing <=100m",
            "potential_stores_250m": "Existing <=250m",
            "potential_stores_500m": "Existing <=500m",
            "potential_stores_1000m": "Existing <=1,000m",
            "apsd_reference_month": "APSD reference month",
            "potential_apsd_exposure_250m": "APSD exposure <=250m",
            "potential_apsd_observed_stores_250m": "APSD stores observed <=250m",
        }
    )
    st.dataframe(month_events, hide_index=True, width="stretch")
    st.caption(
        "Potential APSD exposure sums the nearby existing stores' APSD from "
        "the latest complete calendar month before the new-store opening month. "
        "It is descriptive, not an estimate of sales lost or a causal effect."
    )

if QUALITY_FILE.is_file():
    with st.expander("Data quality summary"):
        quality = pd.read_csv(QUALITY_FILE)
        st.dataframe(quality, hide_index=True, width="stretch")
