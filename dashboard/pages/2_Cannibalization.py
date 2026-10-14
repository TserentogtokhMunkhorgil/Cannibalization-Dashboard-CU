"""Visualization page for the saved CU cannibalization model results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "cannibalization_model4"
DISTANCE_FILE = PROJECT_ROOT / "store_to_store_distance.xlsx"
REFERENCE_EVENT = -1
DISTANCE_BANDS = {
    "0_100": "0–100m",
    "100_250": "100–250m",
    "250_500": "250–500m",
    "500_1000": "500–1,000m",
}
DISTANCE_SUMMARY_LABELS = {
    "0-100m": "0–100m",
    "100-250m": "100–250m",
    "250-500m": "250–500m",
    "500-1000m": "500–1,000m",
}
EVENT_TIMES = tuple(range(-6, 7))
EXPOSURE_COLUMNS = {
    100: ("exposure_0_100_opened",),
    250: ("exposure_0_100_opened", "exposure_100_250_opened"),
    500: (
        "exposure_0_100_opened",
        "exposure_100_250_opened",
        "exposure_250_500_opened",
    ),
    1000: (
        "exposure_0_100_opened",
        "exposure_100_250_opened",
        "exposure_250_500_opened",
        "exposure_500_1000_opened",
    ),
}
MODEL_LABELS = {
    "A": "Baseline FE",
    "B": "Cumulative Distance Exposure",
    "C": "Repeated Event Study",
    "D": "Distance × Event Time",
    "D_alt_0_250": "Alternative 0–250m Specification",
}
MODEL_COLOR = {
    "0–100m": "#B23A48",
    "100–250m": "#D17A35",
    "250–500m": "#3A8D83",
    "500–1,000m": "#426A9A",
}


st.set_page_config(
    page_title="CU Store Cannibalization Analysis",
    page_icon="📊",
    layout="wide",
)


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(float(parsed)):
        return None
    return float(parsed)


def _count_text(value: Any) -> str:
    number = _number(value)
    return "Not available" if number is None else f"{number:,.0f}"


def _percent_text(value: Any, decimals: int = 1) -> str:
    number = _number(value)
    return "Not available" if number is None else f"{number:+,.{decimals}f}%"


def _format_mnt(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "Not available"
    if abs(number) >= 1_000_000_000:
        return f"₮{number / 1_000_000_000:,.2f}b"
    if abs(number) >= 1_000_000:
        return f"₮{number / 1_000_000:,.2f}m"
    return f"₮{number:,.0f}"


def _metric_map(sample_summary: pd.DataFrame | None) -> dict[str, Any]:
    if sample_summary is None or not {"metric", "value"}.issubset(
        sample_summary.columns
    ):
        return {}
    return dict(zip(sample_summary["metric"].astype(str), sample_summary["value"]))


@st.cache_data(show_spinner="Loading saved cannibalization model outputs…")
def load_cannibalization_results(results_dir: str, project_root: str) -> dict[str, Any]:
    results_path = Path(results_dir)
    root_path = Path(project_root)
    required_files = {
        "model_comparison": results_path / "model_comparison.csv",
        "model4_coefficients": results_path / "model4_coefficients.csv",
    }
    missing_required = [str(path) for path in required_files.values() if not path.is_file()]
    if missing_required:
        return {
            "missing_required": missing_required,
            "optional_missing": [],
        }

    results: dict[str, Any] = {
        key: pd.read_csv(path) for key, path in required_files.items()
    }
    optional_files = {
        "all_model_coefficients": results_path / "all_model_coefficients.csv",
        "model_sample_summary": results_path / "model_sample_summary.csv",
        "distance_exposure_summary": results_path / "distance_exposure_summary.csv",
        "event_time_exposure_summary": results_path / "event_time_exposure_summary.csv",
        "overlapping_openings": results_path / "overlapping_openings.csv",
        "new_store_openings_by_month": results_path / "new_store_openings_by_month.csv",
        "panel": results_path / "cu_cannibalization_model4_panel.parquet",
    }
    optional_missing: list[str] = []
    for key, path in optional_files.items():
        if not path.is_file():
            results[key] = None
            optional_missing.append(str(path))
        elif path.suffix == ".parquet":
            results[key] = pd.read_parquet(path)
        else:
            results[key] = pd.read_csv(path)

    distance_file = root_path / "store_to_store_distance.xlsx"
    if distance_file.is_file():
        results["store_distances"] = pd.read_excel(distance_file, sheet_name=0)
    else:
        results["store_distances"] = None
        optional_missing.append(str(distance_file))

    results["missing_required"] = []
    results["optional_missing"] = optional_missing
    return results


def prepare_model_d(
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = {
        "term",
        "percent_effect",
        "percent_ci_95_lower",
        "percent_ci_95_upper",
        "p_value",
    }
    missing = sorted(required.difference(coefficients.columns))
    if missing:
        raise ValueError(f"Model D coefficients are missing columns: {missing}")

    prepared = coefficients.copy()
    extracted = prepared["term"].astype(str).str.extract(
        r"^exposure_(?P<band>0_100|100_250|250_500|500_1000)_event_(?P<event>[mp]\d+)$"
    )
    prepared = prepared.loc[extracted["band"].notna()].copy()
    extracted = extracted.loc[prepared.index]
    prepared["distance_band_code"] = extracted["band"]
    prepared["distance_band"] = prepared["distance_band_code"].map(DISTANCE_BANDS)
    event_codes = extracted["event"].astype(str)
    prepared["event_time"] = event_codes.map(
        lambda value: -int(value[1:])
        if value.startswith("m")
        else int(value[1:])
    )
    for column in (
        "percent_effect",
        "percent_ci_95_lower",
        "percent_ci_95_upper",
        "p_value",
    ):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    expected_events = set(EVENT_TIMES).difference({REFERENCE_EVENT})
    expected_bands = set(DISTANCE_BANDS.values())
    observed_events = set(prepared["event_time"].dropna().astype(int))
    observed_bands = set(prepared["distance_band"].dropna())
    issues: list[str] = []
    if observed_events != expected_events:
        issues.append(
            "Model D event times differ from the expected -6 to +6 window "
            "with month -1 omitted."
        )
    if observed_bands != expected_bands:
        issues.append("Model D does not contain all four expected distance bands.")
    if len(prepared) != len(expected_events) * len(expected_bands):
        issues.append(
            f"Expected 48 Model D estimates; found {len(prepared)}."
        )
    if prepared["percent_effect"].isna().any():
        issues.append("One or more percentage effects are missing or non-numeric.")
    if (
        prepared["percent_ci_95_lower"].isna().any()
        or prepared["percent_ci_95_upper"].isna().any()
    ):
        issues.append("One or more 95% confidence intervals are incomplete.")
    if (
        prepared["percent_ci_95_lower"]
        > prepared["percent_ci_95_upper"]
    ).any():
        issues.append("One or more confidence intervals have reversed bounds.")
    return prepared.sort_values(["distance_band_code", "event_time"]), issues


def strongest_post_opening_effect(model_d: pd.DataFrame) -> pd.Series | None:
    post_opening = model_d.loc[
        model_d["event_time"].ge(0)
        & model_d["event_time"].ne(REFERENCE_EVENT)
        & model_d["percent_effect"].notna()
    ]
    if post_opening.empty:
        return None
    return post_opening.loc[post_opening["percent_effect"].idxmin()]


def make_event_study_chart(
    model_d: pd.DataFrame,
    selected_bands: list[str],
    post_opening_only: bool,
) -> go.Figure:
    visible = model_d.loc[model_d["distance_band"].isin(selected_bands)].copy()
    if post_opening_only:
        visible = visible.loc[visible["event_time"].ge(0)]

    figure = go.Figure()
    for band in DISTANCE_BANDS.values():
        series = visible.loc[visible["distance_band"].eq(band)].sort_values(
            "event_time"
        )
        if series.empty:
            continue
        y = series["percent_effect"].to_numpy(dtype=float)
        lower = series["percent_ci_95_lower"].to_numpy(dtype=float)
        upper = series["percent_ci_95_upper"].to_numpy(dtype=float)
        figure.add_trace(
            go.Scatter(
                x=series["event_time"],
                y=y,
                mode="lines+markers",
                name=band,
                line={"color": MODEL_COLOR[band], "width": 2.5},
                marker={"size": 7},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": upper - y,
                    "arrayminus": y - lower,
                    "color": MODEL_COLOR[band],
                    "thickness": 1.3,
                    "width": 4,
                    "visible": True,
                },
                customdata=np.column_stack(
                    [
                        lower,
                        upper,
                        series["p_value"].to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    f"<b>{band}</b><br>"
                    "Event month: %{x:+d}<br>"
                    "Estimated APSD difference: %{y:.2f}%<br>"
                    "95% CI: [%{customdata[0]:.2f}%, "
                    "%{customdata[1]:.2f}%]<br>"
                    "p-value: %{customdata[2]:.3g}<extra></extra>"
                ),
            )
        )

    figure.add_hline(y=0, line_color="#52606D", line_width=1.2)
    figure.add_vline(
        x=0,
        line_color="#243B53",
        line_dash="dash",
        line_width=1.5,
        annotation_text="New-store opening",
        annotation_position="top right",
    )
    if not post_opening_only:
        figure.add_vline(
            x=REFERENCE_EVENT,
            line_color="#7B8794",
            line_dash="dot",
            line_width=1.3,
            annotation_text="Month -1 reference",
            annotation_position="top left",
        )

    ticks = list(EVENT_TIMES)
    tick_text = [
        "-1<br>Reference" if event == REFERENCE_EVENT else f"{event:+d}"
        for event in ticks
    ]
    figure.update_xaxes(
        title_text="Months relative to new-store opening",
        tickmode="array",
        tickvals=ticks,
        ticktext=tick_text,
        range=[-6.5, 6.5],
        showgrid=True,
        gridcolor="#E8EDF2",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="Estimated APSD change (%)",
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E8EDF2",
        zeroline=False,
    )
    figure.update_layout(
        title={
            "text": (
                "Estimated APSD impact around nearby new-store openings"
                "<br><sup>Reference period: month -1</sup>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        height=560,
        margin={"l": 30, "r": 25, "t": 95, "b": 35},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.04,
            "x": 1,
            "xanchor": "right",
            "yanchor": "bottom",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return figure


def make_distance_event_heatmap(model_d: pd.DataFrame) -> go.Figure:
    band_order = list(DISTANCE_BANDS.values())
    event_order = list(EVENT_TIMES)
    table = model_d.pivot(
        index="distance_band",
        columns="event_time",
        values="percent_effect",
    ).reindex(index=band_order, columns=event_order)
    matrix = table.to_numpy(dtype=float)
    scale = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    scale = max(scale, 1.0)
    x_labels = [
        "−1<br>Reference" if event == REFERENCE_EVENT else f"{event:+d}"
        for event in event_order
    ]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=x_labels,
            y=band_order,
            zmin=-scale,
            zmax=scale,
            zmid=0,
            colorscale=[
                [0.0, "#B23A48"],
                [0.5, "#F7F8FA"],
                [1.0, "#3A8D83"],
            ],
            colorbar={"title": "Estimated<br>APSD change", "ticksuffix": "%"},
            hovertemplate=(
                "Distance: %{y}<br>Event month: %{x}<br>"
                "Estimated APSD difference: %{z:.2f}%<extra></extra>"
            ),
            xgap=2,
            ygap=3,
        )
    )
    for row_index, band in enumerate(band_order):
        for column_index, event in enumerate(event_order):
            value = matrix[row_index, column_index]
            if event == REFERENCE_EVENT or not np.isfinite(value):
                continue
            figure.add_annotation(
                x=x_labels[column_index],
                y=band,
                text=f"{value:+.1f}%",
                showarrow=False,
                font={"size": 10, "color": "#243B53"},
            )
    figure.update_layout(
        title="Distance × event-time estimates",
        height=355,
        margin={"l": 20, "r": 20, "t": 55, "b": 15},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    figure.update_xaxes(title_text="Months relative to new-store opening")
    figure.update_yaxes(title_text="")
    return figure


def make_model_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "observations",
        "stores",
        "calendar_months",
        "exposure_terms",
        "within_r_squared",
        "stores_with_exposure",
        "pair_events_in_sales_period",
        "pretrend_joint_p_value",
    ]
    missing = [column for column in columns if column not in comparison.columns]
    if missing:
        raise ValueError(f"Model comparison is missing columns: {missing}")
    table = comparison.reindex(columns=columns).copy()
    table["model"] = table["model"].astype(str).map(
        lambda model: f"{model} — {MODEL_LABELS.get(str(model), str(model))}"
    )
    table = table.rename(
        columns={
            "model": "Model",
            "observations": "Observations",
            "stores": "Stores",
            "calendar_months": "Calendar months",
            "exposure_terms": "Exposure terms",
            "within_r_squared": "Within R²",
            "stores_with_exposure": "Stores with exposure",
            "pair_events_in_sales_period": "Pair events in sales period",
            "pretrend_joint_p_value": "Pre-trend joint p-value",
        }
    )
    table["Within R²"] = pd.to_numeric(table["Within R²"], errors="coerce").map(
        lambda value: f"{value:.4f}" if pd.notna(value) else "—"
    )
    table["Pre-trend joint p-value"] = pd.to_numeric(
        table["Pre-trend joint p-value"], errors="coerce"
    ).map(lambda value: f"{value:.3g}" if pd.notna(value) else "—")
    return table


def prepare_store_exposure(
    panel: pd.DataFrame,
    store_distances: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "existing_store_id",
        "SalesMonth",
        "APSD_income",
        "LATITUDE",
        "LONGITUDE",
        *[column for columns in EXPOSURE_COLUMNS.values() for column in columns],
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Model panel is missing exposure fields: {missing}")

    latest = (
        panel.sort_values("SalesMonth")
        .groupby("existing_store_id", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    latest["store_id"] = latest["existing_store_id"].astype("string")
    latest["latest_month"] = pd.to_datetime(latest["SalesMonth"], errors="coerce")
    latest["latest_APSD"] = pd.to_numeric(
        latest["APSD_income"], errors="coerce"
    )
    latest["latitude"] = pd.to_numeric(latest["LATITUDE"], errors="coerce")
    latest["longitude"] = pd.to_numeric(latest["LONGITUDE"], errors="coerce")
    if "loc_area_new" in latest.columns:
        latest["area"] = latest["loc_area_new"].astype("string")
    else:
        latest["area"] = pd.NA

    for threshold, columns in EXPOSURE_COLUMNS.items():
        latest[f"nearby_openings_{threshold}m"] = latest.loc[
            :, list(columns)
        ].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    if store_distances is not None and {
        "existing_store_id",
        "existing_store_name",
    }.issubset(store_distances.columns):
        names = (
            store_distances[["existing_store_id", "existing_store_name"]]
            .dropna(subset=["existing_store_id"])
            .drop_duplicates("existing_store_id")
            .copy()
        )
        names["store_id"] = names["existing_store_id"].astype("string")
        latest = latest.merge(
            names[["store_id", "existing_store_name"]],
            on="store_id",
            how="left",
            validate="one_to_one",
        )
        latest["store_name"] = latest["existing_store_name"].fillna(
            "CU " + latest["store_id"]
        )
    else:
        latest["store_name"] = "CU " + latest["store_id"]

    exposure_events = pd.DataFrame()
    if store_distances is not None:
        needed = {
            "existing_store_id",
            "new_store_id",
            "distance_m",
            "existing_opening_date",
            "new_opening_date",
        }
        if needed.issubset(store_distances.columns):
            exposure_events = store_distances.copy()
            exposure_events["existing_store_id"] = exposure_events[
                "existing_store_id"
            ].astype("string")
            exposure_events["new_store_id"] = exposure_events["new_store_id"].astype(
                "string"
            )
            exposure_events["distance_m"] = pd.to_numeric(
                exposure_events["distance_m"], errors="coerce"
            )
            exposure_events["existing_opening_date"] = pd.to_datetime(
                exposure_events["existing_opening_date"], errors="coerce"
            )
            exposure_events["new_opening_date"] = pd.to_datetime(
                exposure_events["new_opening_date"], errors="coerce"
            )
            exposure_events = exposure_events.loc[
                exposure_events["distance_m"].between(0, 1000, inclusive="both")
                & exposure_events["existing_opening_date"].notna()
                & exposure_events["new_opening_date"].notna()
                & exposure_events["existing_opening_date"].lt(
                    exposure_events["new_opening_date"]
                )
            ].copy()
            bands = pd.cut(
                exposure_events["distance_m"],
                bins=[0, 100, 250, 500, 1000.000001],
                labels=["0–100m", "100–250m", "250–500m", "500–1,000m"],
                right=False,
                include_lowest=True,
            )
            exposure_events["display_distance_band"] = bands.astype("string")
            if "new_store_name" not in exposure_events.columns:
                exposure_events["new_store_name"] = pd.NA
    return latest, exposure_events


def make_exposure_summary(
    sample_metrics: dict[str, Any],
    distance_summary: pd.DataFrame | None,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    cards = [
        ("Existing stores exposed ≤100m", sample_metrics.get("existing_stores_exposed_within_100m")),
        ("Existing stores exposed ≤250m", sample_metrics.get("existing_stores_exposed_within_250m")),
        ("Existing stores exposed ≤500m", sample_metrics.get("existing_stores_exposed_within_500m")),
        ("Existing stores exposed ≤1km", sample_metrics.get("existing_stores_exposed_within_1km")),
        ("Stores with 2+ openings ≤1km", sample_metrics.get("stores_with_2plus_openings_within_1km")),
        (
            "Maximum openings for one store",
            sample_metrics.get("maximum_nearby_new_stores_for_one_existing_store"),
        ),
        (
            "Median months between openings",
            sample_metrics.get("median_months_between_consecutive_openings"),
        ),
    ]
    formatted_cards: list[tuple[str, str]] = []
    for label, value in cards:
        if value is None:
            formatted = "Not available"
        elif label == "Median months between openings":
            number = _number(value)
            formatted = "Not available" if number is None else f"{number:.2f} months"
        else:
            formatted = _count_text(value)
        formatted_cards.append((label, formatted))
    if distance_summary is None:
        return formatted_cards, pd.DataFrame()

    columns = [
        "distance_band",
        "distance_pairs",
        "pair_events_in_panel_period",
        "existing_stores_in_band",
        "unique_exposed_stores_in_panel",
        "store_month_observations_with_exposure",
    ]
    available = [column for column in columns if column in distance_summary.columns]
    table = distance_summary.loc[
        distance_summary["distance_band"].astype(str).isin(
            DISTANCE_SUMMARY_LABELS
        ),
        available,
    ].copy()
    if not table.empty:
        table["distance_band"] = table["distance_band"].astype(str).map(
            DISTANCE_SUMMARY_LABELS
        )
    return formatted_cards, table


def make_exposure_map(
    stores: pd.DataFrame,
    metric_column: str,
) -> tuple[pdk.Deck | None, pd.DataFrame]:
    valid = stores["latitude"].between(-90, 90) & stores["longitude"].between(
        -180, 180
    )
    map_data = stores.loc[valid].copy()
    if map_data.empty:
        return None, map_data
    counts = pd.to_numeric(map_data[metric_column], errors="coerce").fillna(0)
    map_data["marker_radius"] = (90 + counts * 24).clip(upper=1800)
    map_data["apsd_display"] = map_data["latest_APSD"].map(_format_mnt)
    for threshold in (100, 250, 500, 1000):
        map_data[f"openings_{threshold}_display"] = map_data[
            f"nearby_openings_{threshold}m"
        ].map(_count_text)
    data_columns = [
        "store_id",
        "store_name",
        "latest_month",
        "latest_APSD",
        "apsd_display",
        "latitude",
        "longitude",
        "marker_radius",
        *[f"nearby_openings_{distance}m" for distance in (100, 250, 500, 1000)],
        *[f"openings_{distance}_display" for distance in (100, 250, 500, 1000)],
    ]
    map_data = map_data[data_columns].replace({np.nan: None})
    spread = max(
        float(map_data["longitude"].max() - map_data["longitude"].min()),
        float(map_data["latitude"].max() - map_data["latitude"].min()),
        0.25,
    )
    zoom = float(np.clip(np.log2(360 / (1.3 * spread)), 3, 12))
    view = pdk.ViewState(
        latitude=float(map_data["latitude"].median()),
        longitude=float(map_data["longitude"].median()),
        zoom=zoom,
        pitch=0,
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        id="cannibalization_exposure",
        get_position="[longitude, latitude]",
        get_radius="marker_radius",
        get_fill_color=[58, 112, 151, 205],
        get_line_color=[33, 55, 73, 210],
        radius_min_pixels=4,
        radius_max_pixels=24,
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": (
                "<b>{store_name}</b><br>Store ID: {store_id}<br>"
                "Latest APSD: {apsd_display}<br>"
                "≤100m: {openings_100_display}<br>"
                "≤250m: {openings_250_display}<br>"
                "≤500m: {openings_500_display}<br>"
                "≤1,000m: {openings_1000_display}<br>"
                "Latitude: {latitude}<br>Longitude: {longitude}"
            ),
            "style": {"backgroundColor": "white", "color": "#243B53"},
        },  # type: ignore[arg-type]
    )
    return deck, map_data


def _selected_map_store_id(event: Any) -> str | None:
    if not hasattr(event, "get"):
        return None
    selection = event.get("selection", {})
    objects = selection.get("objects", {}) if hasattr(selection, "get") else {}
    if not hasattr(objects, "values"):
        return None
    for rows in objects.values():
        if rows and hasattr(rows[0], "get"):
            store_id = rows[0].get("store_id")
            return None if store_id is None else str(store_id)
    return None


def _exposure_events_for_store(
    events: pd.DataFrame,
    store_id: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    mask = events["existing_store_id"].eq(store_id)
    columns = [
        "new_store_id",
        "new_store_name",
        "distance_m",
        "new_opening_date",
        "display_distance_band",
    ]
    columns = [column for column in columns if column in events.columns]
    detail = events.loc[mask, columns].sort_values(
        ["distance_m", "new_opening_date", "new_store_id"]
    )
    return detail.rename(
        columns={
            "new_store_id": "New store ID",
            "new_store_name": "New store name",
            "distance_m": "Distance (m)",
            "new_opening_date": "Opening date",
            "display_distance_band": "Distance band",
        }
    )


def _render_methodology(
    sample_metrics: dict[str, Any],
    model_d_exposure_terms: Any,
) -> None:
    with st.expander("Methodology & Interpretation"):
        st.markdown(
            """
**Outcome:** `log(APSD_income)`  
**Fixed effects:** existing-store fixed effects and calendar-month fixed effects.  
**Standard errors:** clustered by existing store.  
**Distance bands:** 0–100m, 100–250m, 250–500m, and 500–1,000m.  
**Event window:** months -6 through +6; event month -1 is the omitted reference.  
**Model D:** distance × event-time exposure. Repeated nearby openings are retained rather than selecting only the nearest new store.

The chart displays `percent_effect` and the saved percentage confidence interval
bounds from the model output. These are estimates relative to the omitted event
month, not absolute MNT losses. The visual filters only subset saved estimates
or store exposure records; they never re-estimate the model.

**Limitation:** These estimates are conditional repeated-event associations.
They should not automatically be interpreted as causal effects. The
repeated-treatment structure and overlapping exposure windows require a design
specifically addressing multiple and potentially overlapping treatments for
stronger causal claims.
"""
        )
        terms = _number(model_d_exposure_terms)
        if terms is None:
            st.caption("Model D exposure-term count is unavailable.")
        else:
            st.caption(
                f"Model D contains {int(terms)} distance-by-event-time exposure "
                "terms after excluding the reference event month."
            )
        reference = sample_metrics.get("omitted_reference_event_time", REFERENCE_EVENT)
        st.caption(f"Saved model reference event time: {reference}.")


def main() -> None:
    results = load_cannibalization_results(str(RESULTS_DIR), str(PROJECT_ROOT))
    missing_required = results.get("missing_required", [])
    if missing_required:
        st.error(
            "Cannibalization model results were not found. Please run the "
            "cannibalization model first. Missing required file(s): "
            + ", ".join(missing_required)
        )
        st.stop()

    try:
        model_d, model_issues = prepare_model_d(results["model4_coefficients"])
        comparison = results["model_comparison"]
        comparison_table = make_model_comparison_table(comparison)
    except (ValueError, KeyError) as error:
        st.error(f"Saved cannibalization results could not be displayed: {error}")
        st.stop()

    if results.get("optional_missing"):
        with st.expander("Optional model outputs not available"):
            st.write("The main model chart remains available. Missing files:")
            for path in results["optional_missing"]:
                st.code(path, language=None)
    if model_issues:
        st.warning("Saved Model D output validation:")
        for issue in model_issues:
            st.write(f"- {issue}")

    sample_metrics = _metric_map(results.get("model_sample_summary"))
    model_d_row = comparison.loc[comparison["model"].astype(str).eq("D")]
    model_d_summary = model_d_row.iloc[0] if not model_d_row.empty else pd.Series(dtype=object)
    model_d_exposure_terms = model_d_summary.get("exposure_terms")
    strongest = strongest_post_opening_effect(model_d)
    distance_exposure_summary = results.get("distance_exposure_summary")
    overlap_summary = results.get("overlapping_openings")

    st.title("CU Store Cannibalization Analysis")
    st.subheader("Estimated APSD impact of nearby new-store openings")
    st.caption(
        "Existing-store and calendar-month fixed effects; standard errors "
        "clustered by existing store. Event month -1 is the reference period. "
        "These estimates are not automatically causal."
    )

    st.header("Executive summary")
    openings = _number(sample_metrics.get("all_new_store_openings"))
    if openings is None and results.get("new_store_openings_by_month") is not None:
        opening_months = results["new_store_openings_by_month"]
        if {"new_store_openings"}.issubset(opening_months.columns):
            openings = _number(opening_months["new_store_openings"].sum())
    exposed_1km = sample_metrics.get("existing_stores_exposed_within_1km")
    if exposed_1km is None and distance_exposure_summary is not None:
        row = distance_exposure_summary.loc[
            distance_exposure_summary["distance_band"].astype(str).eq("500-1000m")
        ]
        if not row.empty:
            exposed_1km = _number(row.iloc[0].get("unique_exposed_stores_in_panel"))
    stores_2plus = sample_metrics.get("stores_with_2plus_openings_within_1km")
    if stores_2plus is None and overlap_summary is not None:
        stores_2plus = int((overlap_summary["nearby_new_stores"] >= 2).sum())

    cards = st.columns(6)
    cards[0].metric("Existing stores analyzed", _count_text(model_d_summary.get("stores")))
    cards[1].metric("New-store openings", _count_text(openings))
    cards[2].metric("Stores exposed within 1 km", _count_text(exposed_1km))
    cards[3].metric("Stores with 2+ nearby openings", _count_text(stores_2plus))
    cards[4].metric(
        "Strongest distance band",
        "Not available" if strongest is None else str(strongest["distance_band"]),
    )
    cards[5].metric(
        "Most negative post-opening estimate",
        "Not available"
        if strongest is None
        else (
            f"{_percent_text(strongest['percent_effect'])} "
            f"(month {int(strongest['event_time']):+d})"
        ),
        help=(
            "Minimum saved Model D percent_effect among event months 0 to +6; "
            "month -1 is omitted."
        ),
    )

    st.header("Main Model D result")
    st.caption(
        "The chart uses the saved percentage effects and 95% confidence "
        "intervals. Display filters below only select existing estimates."
    )
    left_filter, right_filter = st.columns([1.3, 1])
    with left_filter:
        selected_bands = st.multiselect(
            "Distance bands shown",
            options=list(DISTANCE_BANDS.values()),
            default=list(DISTANCE_BANDS.values()),
            key="cannibalization_distance_bands",
        )
    with right_filter:
        event_display = st.radio(
            "Event-time display",
            ["Full window (−6 to +6)", "Post-opening (0 to +6)"],
            horizontal=True,
            key="cannibalization_event_display",
        )
    st.plotly_chart(
        make_event_study_chart(
            model_d,
            selected_bands,
            post_opening_only=event_display.startswith("Post-opening"),
        ),
        width="stretch",
        config={"displayModeBar": False},
        key="cannibalization_event_study",
    )

    st.header("Business interpretation")
    if strongest is None:
        st.info("No post-opening Model D estimates are available.")
    else:
        st.markdown(
            f"**Most negative post-opening estimate:** "
            f"{_percent_text(strongest['percent_effect'])} at "
            f"{strongest['distance_band']} in event month "
            f"{int(strongest['event_time']):+d}. This is a descriptive "
            "estimated difference relative to event month -1."
        )
    event_labels = {
        0: "At month 0",
        3: "At month +3",
        6: "At month +6",
    }
    interpretation_columns = st.columns(3)
    for column, (event_time, heading) in zip(
        interpretation_columns,
        event_labels.items(),
    ):
        column.markdown(f"**{heading}**")
        event_rows = model_d.loc[model_d["event_time"].eq(event_time)]
        lines = [
            f"- {band}: {_percent_text(row['percent_effect'])}"
            for band, row in event_rows.set_index("distance_band").iterrows()
        ]
        column.markdown("\n".join(lines) if lines else "Not available.")

    st.header("Distance × event-time heatmap")
    st.caption(
        "Red indicates a negative estimated difference; teal indicates a "
        "positive estimated difference. Month -1 is shown as the omitted reference."
    )
    st.plotly_chart(
        make_distance_event_heatmap(model_d),
        width="stretch",
        config={"displayModeBar": False},
        key="cannibalization_heatmap",
    )

    st.header("Store exposure")
    summary_cards, distance_table = make_exposure_summary(
        sample_metrics,
        distance_exposure_summary,
    )
    for row_start in range(0, len(summary_cards), 4):
        metric_row = st.columns(4)
        for column, (label, value) in zip(
            metric_row,
            summary_cards[row_start : row_start + 4],
        ):
            column.metric(label, value)
    if not distance_table.empty:
        st.markdown("**Exposure by distance band**")
        st.dataframe(
            distance_table.rename(
                columns={
                    "distance_band": "Distance band",
                    "distance_pairs": "Store pairs",
                    "pair_events_in_panel_period": "Pair events in panel period",
                    "existing_stores_in_band": "Existing stores",
                    "unique_exposed_stores_in_panel": "Exposed stores in panel",
                    "store_month_observations_with_exposure": "Exposed store-months",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    panel = results.get("panel")
    store_distances = results.get("store_distances")
    if panel is None:
        st.info("The model panel is unavailable; store-level exposure is omitted.")
        _render_model_comparison(comparison_table)
        _render_diagnostics(comparison)
        _render_methodology(sample_metrics, model_d_exposure_terms)
        return

    try:
        stores, opening_events = prepare_store_exposure(panel, store_distances)
    except (ValueError, KeyError) as error:
        st.warning(f"Store-level exposure could not be prepared: {error}")
        _render_model_comparison(comparison_table)
        _render_diagnostics(comparison)
        _render_methodology(sample_metrics, model_d_exposure_terms)
        return

    st.header("Store exposure map")
    st.caption(
        "Marker size represents the selected cumulative nearby-opening count; "
        "the map visualizes exposure and does not estimate regression effects."
    )
    map_control, area_control = st.columns([1, 1])
    with map_control:
        map_threshold = st.selectbox(
            "Map exposure metric",
            [100, 250, 500, 1000],
            index=3,
            format_func=lambda value: f"Nearby new stores ≤{value:,}m",
            key="cannibalization_map_threshold",
        )
    available_areas = sorted(
        stores["area"].dropna().astype(str).unique().tolist()
    )
    with area_control:
        selected_areas = st.multiselect(
            "Area filter",
            options=available_areas,
            key="cannibalization_area_filter",
        )
    search_column, minimum_column = st.columns([1.3, 1])
    with search_column:
        store_search = st.text_input(
            "Filter by existing store ID",
            placeholder="Leave blank to include all stores",
            key="cannibalization_store_search",
        ).strip()
    exposure_column = f"nearby_openings_{map_threshold}m"
    maximum_exposure = int(
        pd.to_numeric(stores[exposure_column], errors="coerce")
        .fillna(0)
        .max()
    )
    with minimum_column:
        minimum_exposure = st.slider(
            "Minimum nearby openings",
            min_value=0,
            max_value=max(maximum_exposure, 1),
            value=0,
            key="cannibalization_min_exposure",
        )

    visible_stores = stores.loc[
        pd.to_numeric(stores[exposure_column], errors="coerce").fillna(0).ge(
            minimum_exposure
        )
    ].copy()
    if selected_areas:
        visible_stores = visible_stores.loc[
            visible_stores["area"].astype("string").isin(selected_areas)
        ]
    if store_search:
        visible_stores = visible_stores.loc[
            visible_stores["store_id"].str.contains(
                store_search,
                case=False,
                regex=False,
                na=False,
            )
        ]
    visible_stores = visible_stores.sort_values(
        [exposure_column, "store_id"],
        ascending=[False, True],
    )
    deck, map_data = make_exposure_map(visible_stores, exposure_column)
    if deck is None:
        st.info("No stores with valid coordinates match the current filters.")
        selected_map_store_id = None
    else:
        st.caption(
            f"Showing {len(map_data):,} of {len(visible_stores):,} filtered "
            "stores with valid coordinates."
        )
        map_event = st.pydeck_chart(
            deck,
            width="stretch",
            height=560,
            key="cannibalization_exposure_map",
            on_select="rerun",
            selection_mode="single-object",
        )
        selected_map_store_id = _selected_map_store_id(map_event)

    st.header("Store detail")
    detail_stores = visible_stores.copy()
    if detail_stores.empty:
        st.info("No stores match the current exposure filters.")
    else:
        detail_stores["selector_label"] = detail_stores.apply(
            lambda row: f"{row['store_id']} — {row['store_name']}",
            axis=1,
        )
        detail_stores = detail_stores.sort_values("selector_label")
        labels = detail_stores["selector_label"].tolist()
        label_to_id = dict(
            zip(detail_stores["selector_label"], detail_stores["store_id"])
        )
        default_label = next(
            (
                label
                for label in labels
                if label_to_id[label] == selected_map_store_id
            ),
            labels[0],
        )
        chosen_label = st.selectbox(
            "Existing store",
            labels,
            index=labels.index(default_label),
            key=f"cannibalization_detail_store_{selected_map_store_id or 'none'}",
        )
        selected_store_id = label_to_id[chosen_label]
        selected_store = detail_stores.loc[
            detail_stores["store_id"].eq(selected_store_id)
        ].iloc[0]
        st.markdown(f"### {selected_store['store_name']}")
        details = st.columns(5)
        details[0].metric("Store ID", str(selected_store_id))
        details[1].metric("Latest APSD", _format_mnt(selected_store["latest_APSD"]))
        details[2].metric("Nearby openings ≤100m", _count_text(selected_store["nearby_openings_100m"]))
        details[3].metric("Nearby openings ≤250m", _count_text(selected_store["nearby_openings_250m"]))
        details[4].metric("Nearby openings ≤500m", _count_text(selected_store["nearby_openings_500m"]))
        st.caption(
            f"Latest panel observation: "
            f"{pd.Timestamp(selected_store['latest_month']):%b %Y} · "
            f"≤1,000m openings: "
            f"{_count_text(selected_store['nearby_openings_1000m'])} · "
            f"Area: {selected_store['area'] if pd.notna(selected_store['area']) else 'Not available'}"
        )
        nearby_events = _exposure_events_for_store(
            opening_events,
            str(selected_store_id),
        )
        if nearby_events.empty:
            st.caption(
                "Per-opening detail is unavailable. The model panel still "
                "provides cumulative exposure counts."
            )
        else:
            st.markdown("**Nearby new-store openings within 1 km**")
            st.dataframe(
                nearby_events,
                hide_index=True,
                width="stretch",
                height=250,
            )

    filtered_table = visible_stores[
        [
            "store_id",
            "store_name",
            "latest_month",
            "latest_APSD",
            "area",
            "nearby_openings_100m",
            "nearby_openings_250m",
            "nearby_openings_500m",
            "nearby_openings_1000m",
        ]
    ].rename(
        columns={
            "store_id": "Existing store ID",
            "store_name": "Store name",
            "latest_month": "Latest panel month",
            "latest_APSD": "Latest APSD",
            "area": "Area",
            "nearby_openings_100m": "Openings ≤100m",
            "nearby_openings_250m": "Openings ≤250m",
            "nearby_openings_500m": "Openings ≤500m",
            "nearby_openings_1000m": "Openings ≤1,000m",
        }
    )
    st.markdown("**Store-level cumulative exposure**")
    st.dataframe(
        filtered_table,
        hide_index=True,
        width="stretch",
        height=350,
    )

    _render_model_comparison(comparison_table)
    _render_diagnostics(comparison)
    _render_methodology(sample_metrics, model_d_exposure_terms)


def _render_model_comparison(
    display_table: pd.DataFrame,
) -> None:
    st.header("Model comparison")
    st.dataframe(display_table, hide_index=True, width="stretch")
    st.caption(
        "Model B is the primary business summary, while Model D provides the "
        "distance-specific event-study results. Models C and D_alt_0_250 are "
        "robustness/sensitivity specifications. R² is not the sole basis for "
        "selecting a preferred model."
    )


def _render_diagnostics(comparison: pd.DataFrame) -> None:
    st.header("Model Diagnostics")
    p_values = comparison.set_index(comparison["model"].astype(str))[
        "pretrend_joint_p_value"
    ]
    models = [
        ("C", "Model C pre-trend p-value"),
        ("D", "Model D pre-trend p-value"),
        ("D_alt_0_250", "D_alt pre-trend p-value"),
    ]
    columns = st.columns(3)
    for column, (model, label) in zip(columns, models):
        value = _number(p_values.get(model))
        column.metric(
            label,
            "Not available" if value is None else f"{value:.3g}",
        )
        if value is None:
            column.info("Pre-trend diagnostic is unavailable.")
        elif value >= 0.05:
            column.success("No statistically significant pre-opening deviation detected.")
        else:
            column.warning(
                "Pre-opening coefficients are jointly significant; causal "
                "interpretation should be treated with caution."
            )
    st.caption(
        "A non-significant pre-trend test does not by itself establish causal identification."
    )


main()
