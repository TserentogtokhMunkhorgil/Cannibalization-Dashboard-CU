"""Interactive dashboard for saved regional CU cannibalization results."""



from __future__ import annotations



import calendar
import math
import numpy as np
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import t as student_t



from cannibalization_simulator import (

    DISTANCE_BANDS,

    CapacityResult,

    CoefficientEstimate,

    ScenarioResult,

    maximum_openings,

    optimize_openings,

    predict_apsd,

    regional_sales_estimates,

    simulate_mixed_scenario,

    validate_sales_inputs,

)





BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs" / "cannibalization_regional"

INCOME_FILE = BASE_DIR / "store_master_data.xlsx"

DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"

REGIONS = ("City Center", "City Outskirts", "Far / Outer Areas", "Rural")

DISTANCE_LABELS = {

    "0_100": "Very close (0-100 m)",

    "100_250": "Close (100-250 m)",

    "250_500": "Medium (250-500 m)",

    "500_1000": "Farther nearby (500-1,000 m)",

}

EVENT_ORDER = [

    "−12 to −7",

    "−6 to −3",

    "−2",

    "0",

    "+1 to +3",

    "+4 to +6",

    "+7 to +12",

    "+13 to +18",

    "+19 to +24",

    "+25 months onward",

]





@st.cache_data(show_spinner=False)

def _read_csv(name: str, modified_ns: int) -> pd.DataFrame:

    path = OUTPUT_DIR / name

    if not path.is_file():

        return pd.DataFrame()

    return pd.read_csv(path)





@st.cache_data(show_spinner=False)

def _read_panel(modified_ns: int) -> pd.DataFrame:

    path = OUTPUT_DIR / "cu_regional_cannibalization_panel.parquet"

    if not path.is_file():

        return pd.DataFrame()

    frame = pd.read_parquet(path)

    if "SalesMonth" in frame:

        frame["SalesMonth"] = pd.to_datetime(frame["SalesMonth"], errors="coerce")

    return frame





@st.cache_data(show_spinner=False)

def _read_pair_data(modified_ns: int) -> pd.DataFrame:

    if not DISTANCE_FILE.is_file():

        return pd.DataFrame()

    pairs = pd.read_excel(DISTANCE_FILE, sheet_name="Sheet1")

    required = {

        "existing_store_id",

        "new_store_id",

        "distance_m",

        "new_opening_date",

    }

    if not required.issubset(pairs.columns):

        return pd.DataFrame()

    pairs["new_opening_date"] = pd.to_datetime(

        pairs["new_opening_date"], errors="coerce"

    )

    pairs["distance_m"] = pd.to_numeric(pairs["distance_m"], errors="coerce")
    return pairs


@st.cache_data(show_spinner=False)
def _read_store_master(modified_ns: int) -> pd.DataFrame:
    if not INCOME_FILE.is_file():
        return pd.DataFrame()
    master = pd.read_excel(INCOME_FILE, sheet_name="Sheet1")
    columns = [
        column
        for column in (
            "store_id",
            "store_name",
            "SalesMonth",
            "APSD_income",
            "APSD_customers",
            "Active_Days",
            "ost_dt",
            "cls_dt",
            "Store_Status",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
            "Distance_Zone",
            "UB",
        )
        if column in master.columns
    ]
    return master[columns].drop_duplicates().copy()


def _modified(path: Path) -> int:
    return path.stat().st_mtime_ns if path.is_file() else 0


def read_csv(name: str) -> pd.DataFrame:
    return _read_csv(name, _modified(OUTPUT_DIR / name))


def read_panel() -> pd.DataFrame:
    return _read_panel(_modified(OUTPUT_DIR / 'cu_regional_cannibalization_panel.parquet'))


def read_pair_data() -> pd.DataFrame:
    return _read_pair_data(_modified(DISTANCE_FILE))


def read_store_master() -> pd.DataFrame:
    return _read_store_master(_modified(INCOME_FILE))


def normalize_store_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _parse_dashboard_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.mask(parsed.dt.year.ge(9000))


def _month_index(values: pd.Series) -> pd.Series:
    return values.dt.year * 12 + values.dt.month


def _clustered_mean_interval(
    values: pd.Series, clusters: pd.Series
) -> tuple[float, float, float]:
    valid = values.notna() & clusters.notna()
    sample = pd.DataFrame(
        {"value": values.loc[valid].astype(float), "cluster": clusters.loc[valid]}
    )
    count = len(sample)
    cluster_count = sample["cluster"].nunique()
    mean = float(sample["value"].mean()) if count else float("nan")
    if count < 2 or cluster_count < 2:
        return mean, float("nan"), float("nan")

    centered = sample["value"] - mean
    cluster_scores = centered.groupby(sample["cluster"], sort=False).sum()
    variance = (
        cluster_count
        / (cluster_count - 1)
        * count
        / (count - 1)
        * float((cluster_scores**2).sum())
        / count**2
    )
    standard_error = math.sqrt(max(variance, 0.0))
    critical_value = float(student_t.ppf(0.975, cluster_count - 1))
    return mean, mean - critical_value * standard_error, mean + critical_value * standard_error


@st.cache_data(show_spinner="Preparing pair-level opening and event-month data...")
def prepare_dashboard_openings(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    required_pairs = {
        "existing_store_id",
        "new_store_id",
        "distance_m",
        "new_opening_date",
        "distance_band",
    }
    missing = required_pairs.difference(pair_source.columns)
    if missing:
        raise ValueError(f"Distance workbook is missing columns: {sorted(missing)}")
    required_panel = {
        "existing_store_id",
        "SalesMonth",
        "APSD_income",
        "APSD_customers",
        "ost_dt",
        "analysis_region",
    }
    missing = required_panel.difference(panel.columns)
    if missing:
        raise ValueError(f"Saved analysis panel is missing columns: {sorted(missing)}")

    source = pair_source.copy()
    source["_existing_key"] = normalize_store_id(source["existing_store_id"])
    source["_new_key"] = normalize_store_id(source["new_store_id"])
    source["new_opening_date"] = _parse_dashboard_dates(
        source["new_opening_date"]
    )
    source["_opening_key"] = (
        source["_new_key"]
        + "::"
        + source["new_opening_date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    )
    pair_key = ["_existing_key", "_opening_key"]
    duplicate_pair_rows = int(source.duplicated(pair_key, keep=False).sum())
    if duplicate_pair_rows:
        raise ValueError(
            f"Distance workbook has {duplicate_pair_rows} duplicate existing-store/"
            "new-store-opening rows (store IDs plus opening date); resolve duplicates "
            "before calculating opening impacts."
        )
    if "existing_opening_date" in source:
        source["existing_opening_date"] = _parse_dashboard_dates(
            source["existing_opening_date"]
        )
    else:
        source["existing_opening_date"] = pd.NaT
    source["distance_m"] = pd.to_numeric(source["distance_m"], errors="coerce")
    source["distance_band"] = source["distance_band"].astype("string").str.strip()
    source["distance_band"] = source["distance_band"].mask(
        source["distance_band"].eq(""), pd.NA
    )

    panel_metadata = panel.copy()
    panel_metadata["_existing_key"] = normalize_store_id(
        panel_metadata["existing_store_id"]
    )
    panel_metadata["ost_dt"] = _parse_dashboard_dates(panel_metadata["ost_dt"])
    metadata_columns = [
        "_existing_key",
        "ost_dt",
        "analysis_region",
    ]
    metadata = (
        panel_metadata.sort_values("SalesMonth")
        .drop_duplicates("_existing_key")[metadata_columns]
        .rename(
            columns={
                "ost_dt": "master_existing_opening_date",
                "analysis_region": "existing_store_region",
            }
        )
    )
    if not master_source.empty and {"store_id", "store_name"}.issubset(
        master_source.columns
    ):
        names = master_source[["store_id", "store_name"]].copy()
        names["_existing_key"] = normalize_store_id(names["store_id"])
        names["store_name"] = names["store_name"].astype("string").str.strip()
        names = names.dropna(subset=["store_name"]).drop_duplicates(
            "_existing_key"
        )
        metadata = metadata.merge(
            names[["_existing_key", "store_name"]],
            on="_existing_key",
            how="left",
            validate="one_to_one",
        )
    source = source.merge(
        metadata,
        on="_existing_key",
        how="left",
        validate="many_to_one",
    )
    source["existing_opening_date_used"] = source[
        "master_existing_opening_date"
    ].fillna(source["existing_opening_date"])
    source["new_store_opening_month"] = (
        source["new_opening_date"].dt.to_period("M").dt.to_timestamp()
    )
    source["opening_month_index"] = _month_index(
        source["new_store_opening_month"]
    )
    source["existing_opening_month"] = (
        source["existing_opening_date_used"].dt.to_period("M").dt.to_timestamp()
    )
    date_mismatch = (
        source["master_existing_opening_date"].notna()
        & source["existing_opening_date"].notna()
        & (
            source["master_existing_opening_date"].dt.normalize()
            != source["existing_opening_date"].dt.normalize()
        )
    )
    has_dates = (
        source["new_opening_date"].notna()
        & source["existing_opening_date_used"].notna()
    )
    preexisting = (
        has_dates
        & (source["existing_opening_date_used"] < source["new_opening_date"])
    )
    valid_distance = source["distance_m"].notna() & source["distance_m"].ge(0)
    source["distance_band"] = source["distance_band"].fillna("Unclassified")
    eligible = source.loc[has_dates & preexisting & valid_distance].copy()
    eligible["event_id"] = (
        eligible["_existing_key"].astype(str)
        + "::"
        + eligible["_opening_key"].astype(str)
    )
    eligible["new_store_name"] = (
        eligible.get("new_store_name", pd.Series(index=eligible.index, dtype="string"))
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    eligible["event_order"] = (
        eligible.sort_values(
            ["_existing_key", "new_opening_date", "_new_key"],
            kind="stable",
        )
        .groupby("_existing_key", sort=False)
        .cumcount()
        .add(1)
        .reindex(eligible.index)
        .astype("int64")
    )
    eligible["exposure_order"] = eligible["event_order"].map(
        lambda order: (
            "First observed opening event"
            if order == 1
            else "Second observed opening event"
            if order == 2
            else "Third or later observed opening event"
        )
    )
    eligible["previous_nearby_openings_in_pair_history"] = (
        eligible["event_order"] - 1
    )
    eligible["region"] = eligible["existing_store_region"].fillna("Unmapped")

    band_minimums = (
        source.loc[source["distance_m"].notna()]
        .groupby("distance_band", dropna=False)["distance_m"]
        .min()
        .sort_values(kind="stable")
    )
    band_order = band_minimums.index.astype(str).tolist()
    if "Unclassified" in source["distance_band"].astype(str).unique():
        band_order.append("Unclassified")

    month_data = panel.copy()
    month_data["_existing_key"] = normalize_store_id(
        month_data["existing_store_id"]
    )
    month_data["SalesMonth"] = _parse_dashboard_dates(month_data["SalesMonth"])
    month_data["SalesMonth"] = (
        month_data["SalesMonth"].dt.to_period("M").dt.to_timestamp()
    )
    if month_data.duplicated(["_existing_key", "SalesMonth"]).any():
        raise ValueError(
            "Saved analysis panel has duplicate existing-store/month keys; "
            "opening-event statistics are not calculated."
        )
    month_data["_calendar_month_index"] = _month_index(month_data["SalesMonth"])
    keep = [
        "_existing_key",
        "SalesMonth",
        "_calendar_month_index",
        "APSD_income",
        "APSD_customers",
    ]
    for optional in ("Active_Days", "operating_month_known"):
        if optional in month_data:
            keep.append(optional)
    month_data = month_data[keep]
    for outcome_column in ("APSD_income", "APSD_customers"):
        month_data[outcome_column] = pd.to_numeric(
            month_data[outcome_column], errors="coerce"
        )
        if "operating_month_known" in month_data:
            month_data.loc[
                ~month_data["operating_month_known"].fillna(False),
                outcome_column,
            ] = pd.NA
    event_columns = [
        "event_id",
        "_existing_key",
        "_new_key",
        "_opening_key",
        "existing_store_id",
        "new_store_id",
        "new_store_name",
        "new_opening_date",
        "new_store_opening_month",
        "opening_month_index",
        "distance_m",
        "distance_band",
        "event_order",
        "exposure_order",
        "previous_nearby_openings_in_pair_history",
        "region",
        "existing_opening_date_used",
    ]
    events = eligible[event_columns].copy()
    observations = month_data.merge(
        events.drop(columns="existing_store_id"),
        on="_existing_key",
        how="inner",
        validate="many_to_many",
    )
    observations["event_time_month"] = (
        observations["_calendar_month_index"] - observations["opening_month_index"]
    )
    diagnostics = {
        "source_pair_rows": int(len(source)),
        "duplicate_pair_rows": duplicate_pair_rows,
        "missing_new_opening_date": int(source["new_opening_date"].isna().sum()),
        "missing_existing_opening_date_after_fallback": int(
            source["existing_opening_date_used"].isna().sum()
        ),
        "invalid_distance_rows": int((~valid_distance).sum()),
        "missing_distance_band_rows": int(
            source["distance_band"].eq("Unclassified").sum()
        ),
        "opening_date_mismatches": int(date_mismatch.sum()),
        "pairs_not_preexisting": int((has_dates & ~preexisting).sum()),
        "eligible_opening_events": int(len(events)),
        "unique_new_store_openings": int(events["_opening_key"].nunique()),
        "distinct_existing_stores_exposed": int(events["_existing_key"].nunique()),
        "panel_start_month": str(month_data["SalesMonth"].min().date()),
        "panel_end_month": str(month_data["SalesMonth"].max().date()),
        "events_outside_observed_panel_dates": int(
            (~events["new_store_opening_month"].between(
                month_data["SalesMonth"].min(), month_data["SalesMonth"].max()
            )).sum()
        ),
    }
    return events, observations, diagnostics, band_order


def _filter_events_by_dates(
    events: pd.DataFrame, widget_key: str
) -> tuple[pd.DataFrame, tuple[Any, Any]]:
    if events.empty:
        return events.copy(), (None, None)
    start = events["new_opening_date"].min().date()
    end = events["new_opening_date"].max().date()
    selected = st.date_input(
        "New-store opening date range",
        value=(start, end),
        min_value=start,
        max_value=end,
        key=widget_key,
    )
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        return events.iloc[0:0].copy(), (None, None)
    lower, upper = selected
    if lower is None or upper is None:
        return events.iloc[0:0].copy(), (lower, upper)
    mask = events["new_opening_date"].dt.date.between(lower, upper)
    return events.loc[mask].copy(), (lower, upper)


def build_opening_event_comparisons(
    events: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    before_months: int,
    after_months: int,
    include_event_month: bool,
) -> pd.DataFrame:
    """Create paired opening-event means; event month 0 is separate by default."""
    if events.empty:
        return pd.DataFrame()
    end_post = after_months
    start_post = 0 if include_event_month else 1
    keep_ids = set(events["event_id"])
    relevant = observations.loc[
        observations["event_id"].isin(keep_ids)
        & observations["event_time_month"].between(-before_months, end_post)
    ]
    results: list[pd.DataFrame] = []
    for outcome in ("APSD_income", "APSD_customers"):
        pre = (
            relevant.loc[relevant["event_time_month"].between(-before_months, -1)]
            .groupby("event_id", as_index=False)
            .agg(
                pre_mean=(outcome, "mean"),
                pre_observed_months=(outcome, "count"),
            )
        )
        post = (
            relevant.loc[
                relevant["event_time_month"].between(start_post, end_post)
            ]
            .groupby("event_id", as_index=False)
            .agg(
                post_mean=(outcome, "mean"),
                post_observed_months=(outcome, "count"),
            )
        )
        paired = (
            events.merge(pre, on="event_id", how="inner", validate="one_to_one")
            .merge(post, on="event_id", how="inner", validate="one_to_one")
        )
        paired = paired.loc[
            paired["pre_mean"].notna() & paired["post_mean"].notna()
        ].copy()
        paired["outcome"] = outcome
        paired["absolute_change"] = paired["post_mean"] - paired["pre_mean"]
        paired["percentage_change"] = pd.NA
        valid_baseline = paired["pre_mean"] > 0
        paired.loc[valid_baseline, "percentage_change"] = (
            paired.loc[valid_baseline, "post_mean"]
            / paired.loc[valid_baseline, "pre_mean"]
            - 1
        ) * 100
        results.append(paired)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def summarize_opening_changes(
    changes: pd.DataFrame,
    outcome: str,
    *,
    minimum_events: int,
    minimum_stores: int,
) -> pd.DataFrame:
    """Summarize event-weighted paired changes with existing-store clustered CIs."""
    if changes.empty:
        return pd.DataFrame()
    selected = changes.loc[changes["outcome"] == outcome].copy()
    groups: list[tuple[str, str | None, str | None, str | None, pd.DataFrame]] = [
        ("Overall", None, None, None, selected)
    ]
    for region, rows in selected.groupby("region", dropna=False):
        groups.append(("Region", str(region), None, None, rows))
    for band, rows in selected.groupby("distance_band", dropna=False):
        groups.append(("Distance band", None, str(band), None, rows))
    for (region, band), rows in selected.groupby(
        ["region", "distance_band"], dropna=False
    ):
        groups.append(("Region × distance band", str(region), str(band), None, rows))
    for order, rows in selected.groupby("exposure_order", dropna=False):
        groups.append(("Exposure order", None, None, str(order), rows))
    for (order, region), rows in selected.groupby(
        ["exposure_order", "region"], dropna=False
    ):
        groups.append(
            ("Exposure order × region", str(region), None, str(order), rows)
        )
    for (order, band), rows in selected.groupby(
        ["exposure_order", "distance_band"], dropna=False
    ):
        groups.append(
            ("Exposure order × distance band", None, str(band), str(order), rows)
        )
    for (order, region, band), rows in selected.groupby(
        ["exposure_order", "region", "distance_band"], dropna=False
    ):
        groups.append(
            (
                "Exposure order × region × distance band",
                str(region),
                str(band),
                str(order),
                rows,
            )
        )

    summary_rows: list[dict[str, Any]] = []
    for group_type, region, band, exposure_order, rows in groups:
        deltas = pd.to_numeric(rows["absolute_change"], errors="coerce")
        valid_delta = deltas.notna()
        rows = rows.loc[valid_delta].copy()
        deltas = deltas.loc[valid_delta]
        event_count = int(rows["event_id"].nunique())
        store_count = int(rows["_existing_key"].nunique())
        mean_delta, ci_lower, ci_upper = _clustered_mean_interval(
            deltas.reset_index(drop=True),
            rows["_existing_key"].reset_index(drop=True),
        )
        pct_values = pd.to_numeric(rows["percentage_change"], errors="coerce")
        pct_mean, pct_lower, pct_upper = _clustered_mean_interval(
            pct_values.reset_index(drop=True),
            rows["_existing_key"].reset_index(drop=True),
        )
        decline_count = int(deltas.lt(0).sum())
        increase_count = int(deltas.gt(0).sum())
        summary_rows.append(
            {
                "group_type": group_type,
                "region": region if region is not None else "All Regions",
                "distance_band": band if band is not None else "All Bands",
                "exposure_order": exposure_order or "All Orders",
                "outcome": outcome,
                "existing_stores": store_count,
                "opening_events": event_count,
                "unique_new_store_openings": int(rows["_opening_key"].nunique()),
                "mean_before_per_opening_event": float(rows["pre_mean"].mean())
                if event_count else float("nan"),
                "mean_after_per_opening_event": float(rows["post_mean"].mean())
                if event_count else float("nan"),
                "mean_absolute_change_per_opening_event": mean_delta,
                "mean_percentage_change_per_valid_baseline_event": pct_mean,
                "median_absolute_change_per_opening_event": float(deltas.median())
                if event_count else float("nan"),
                "standard_deviation_event_change": float(deltas.std(ddof=1))
                if event_count > 1 else float("nan"),
                "clustered_standard_error_event_change": (
                    (ci_upper - ci_lower)
                    / (2 * float(student_t.ppf(0.975, store_count - 1)))
                    if store_count > 1 and pd.notna(ci_lower) and pd.notna(ci_upper)
                    else float("nan")
                ),
                "clustered_ci_95_lower_change": ci_lower,
                "clustered_ci_95_upper_change": ci_upper,
                "mean_percentage_ci_95_lower": pct_lower,
                "mean_percentage_ci_95_upper": pct_upper,
                "decline_events": decline_count,
                "decline_event_percentage": (
                    decline_count / event_count * 100 if event_count else float("nan")
                ),
                "increase_events": increase_count,
                "increase_event_percentage": (
                    increase_count / event_count * 100 if event_count else float("nan")
                ),
                "unchanged_events": int(deltas.eq(0).sum()),
                "mean_observed_pre_months_per_event": float(
                    rows["pre_observed_months"].mean()
                ) if event_count else float("nan"),
                "mean_observed_post_months_per_event": float(
                    rows["post_observed_months"].mean()
                ) if event_count else float("nan"),
                "support_status": (
                    "Supported"
                    if event_count >= minimum_events and store_count >= minimum_stores
                    else "Insufficient support"
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def summarize_event_time(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    selected = observations.loc[
        observations["event_id"].isin(set(events["event_id"]))
        & observations["event_time_month"].between(-12, 12)
    ]
    selected = selected.loc[selected[outcome].notna()]
    rows: list[dict[str, Any]] = []
    for event_time, sample in selected.groupby("event_time_month", sort=True):
        mean, lower, upper = _clustered_mean_interval(
            sample[outcome].reset_index(drop=True),
            sample["_existing_key"].reset_index(drop=True),
        )
        rows.append(
            {
                "event_time_month": int(str(event_time)),
                "mean_outcome": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "pair_month_observations": int(len(sample)),
                "opening_events": int(sample["event_id"].nunique()),
                "existing_stores": int(sample["_existing_key"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _event_order_label(order: object) -> str:
    if order == 1:
        return "First observed opening event"
    if order == 2:
        return "Second observed opening event"
    return "Third or later observed opening event"


def _metric_label(outcome: str) -> str:
    return (
        "APSD sales (MNT per active sales day)"
        if outcome == "APSD_income"
        else "APSD customers per active sales day"
    )


@st.cache_data(show_spinner="Preparing store-network statistics...")
def prepare_store_statistics_data(
    panel: pd.DataFrame,
    master_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_panel = {
        "existing_store_id",
        "SalesMonth",
        "APSD_income",
        "APSD_customers",
        "ost_dt",
        "analysis_region",
    }
    missing = required_panel.difference(panel.columns)
    if missing:
        raise ValueError(f"Saved analysis panel is missing columns: {sorted(missing)}")
    if master_source.empty or "store_id" not in master_source:
        raise ValueError("Store master data is unavailable or missing store_id.")

    panel_columns = [
        column
        for column in (
            "existing_store_id",
            "SalesMonth",
            "APSD_income",
            "APSD_customers",
            "ost_dt",
            "cls_dt",
            "Store_Status",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
            "Distance_Zone",
            "UB",
            "analysis_region",
            "Active_Days",
            "operating_month_known",
        )
        if column in panel
    ]
    source = panel[panel_columns].copy()
    source["_store_key"] = normalize_store_id(source["existing_store_id"])
    source["SalesMonth"] = _parse_dashboard_dates(source["SalesMonth"])
    source["SalesMonth"] = (
        source["SalesMonth"].dt.to_period("M").dt.to_timestamp()
    )
    source = source.dropna(subset=["_store_key", "SalesMonth"])
    duplicate_rows = int(
        source.duplicated(["_store_key", "SalesMonth"], keep=False).sum()
    )
    if duplicate_rows:
        raise ValueError(
            f"The saved monthly panel contains {duplicate_rows} duplicate "
            "store-month rows; resolve these before calculating statistics."
        )
    source["_month_index"] = _month_index(source["SalesMonth"])

    master = master_source.copy()
    master["_store_key"] = normalize_store_id(master["store_id"])
    if "SalesMonth" in master:
        master["SalesMonth"] = _parse_dashboard_dates(master["SalesMonth"])
    if "ost_dt" in master:
        master["ost_dt"] = _parse_dashboard_dates(master["ost_dt"])
    else:
        master["ost_dt"] = pd.NaT
    for column in ("cls_dt",):
        if column in master:
            master[column] = _parse_dashboard_dates(master[column])

    metadata_columns = [
        column
        for column in (
            "store_id",
            "store_name",
            "ost_dt",
            "cls_dt",
            "Store_Status",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
            "Distance_Zone",
            "UB",
        )
        if column in master
    ]
    sort_columns = ["_store_key"]
    if "SalesMonth" in master:
        sort_columns.append("SalesMonth")
    master = master.sort_values(sort_columns, kind="stable")
    master_meta = (
        master.dropna(subset=["_store_key"])
        .groupby("_store_key", as_index=False, sort=False)[metadata_columns]
        .first()
    )
    if "Store_Status" in master and "SalesMonth" in master:
        latest_status = (
            master.dropna(subset=["_store_key", "Store_Status"])
            .drop_duplicates("_store_key", keep="last")
            [["_store_key", "Store_Status"]]
        )
        master_meta = master_meta.drop(columns="Store_Status", errors="ignore").merge(
            latest_status,
            on="_store_key",
            how="left",
            validate="one_to_one",
        )
    panel_metadata_columns = [
        column
        for column in (
            "existing_store_id",
            "analysis_region",
            "Store_Status",
            "ost_dt",
            "cls_dt",
            "LATITUDE",
            "LONGITUDE",
            "Distance_Zone",
            "UB",
        )
        if column in source
    ]
    panel_meta = (
        source.sort_values("SalesMonth")
        .drop_duplicates("_store_key")[["_store_key", *panel_metadata_columns]]
        .drop(columns="existing_store_id", errors="ignore")
    )
    stores = master_meta.merge(
        panel_meta,
        on="_store_key",
        how="outer",
        suffixes=("", "_panel"),
        validate="one_to_one",
    )
    if "analysis_region" not in stores:
        stores["analysis_region"] = pd.NA
    if "analysis_region_panel" in stores:
        stores["analysis_region"] = stores["analysis_region"].fillna(
            stores["analysis_region_panel"]
        )
        stores = stores.drop(columns="analysis_region_panel")
    for column in ("ost_dt", "cls_dt", "Store_Status", "LATITUDE", "LONGITUDE",
                   "Distance_Zone", "UB"):
        panel_column = f"{column}_panel"
        if panel_column in stores:
            if column in stores:
                stores[column] = stores[column].fillna(stores[panel_column])
            else:
                stores[column] = stores[panel_column]
            stores = stores.drop(columns=panel_column)
    stores["ost_dt"] = _parse_dashboard_dates(stores["ost_dt"])
    stores["cls_dt"] = _parse_dashboard_dates(stores["cls_dt"])
    stores["analysis_region"] = (
        stores["analysis_region"].astype("string").fillna("Unmapped")
    )
    if "store_name" not in stores:
        stores["store_name"] = pd.NA
    for column in ("Store_Status", "loc_area_new", "Distance_Zone", "UB"):
        if column not in stores:
            stores[column] = pd.NA
    for column in ("LATITUDE", "LONGITUDE"):
        if column not in stores:
            stores[column] = np.nan
        stores[column] = pd.to_numeric(stores[column], errors="coerce")
    opening_month_dates = pd.Series(
        stores["ost_dt"].dt.to_period("M").dt.to_timestamp(),
        index=stores.index,
    )
    stores["_opening_month_index"] = _month_index(opening_month_dates)
    stores["opening_year"] = stores["ost_dt"].dt.year.astype("Int64")
    stores["opening_month"] = stores["ost_dt"].dt.month.astype("Int64")
    stores["opening_cohort_month"] = (
        stores["ost_dt"].dt.to_period("M").astype("string")
    )
    stores = stores.rename(columns={"store_id": "existing_store_id"})

    monthly = source.merge(
        stores[
            [
                "_store_key",
                "store_name",
                "analysis_region",
                "Store_Status",
                "ost_dt",
                "cls_dt",
                "LATITUDE",
                "LONGITUDE",
                "Distance_Zone",
                "UB",
                "_opening_month_index",
                "opening_year",
                "opening_month",
                "opening_cohort_month",
            ]
        ],
        on="_store_key",
        how="left",
        suffixes=("", "_store"),
        validate="many_to_one",
    )
    for outcome_column in ("APSD_income", "APSD_customers"):
        monthly[outcome_column] = pd.to_numeric(
            monthly[outcome_column], errors="coerce"
        )
    monthly["panel_row_present"] = True
    monthly["store_age_month"] = (
        monthly["_month_index"] - monthly["_opening_month_index"]
    )
    monthly["store_age_group"] = pd.cut(
        monthly["store_age_month"],
        bins=[-np.inf, 3, 12, 24, np.inf],
        labels=["0-3 months", "4-12 months", "13-24 months", "25+ months"],
    )

    opening_values = stores.loc[stores["ost_dt"].notna()].copy()
    opening_values = opening_values.merge(
        monthly[
            [
                "_store_key",
                "_month_index",
                "APSD_income",
                "APSD_customers",
                "panel_row_present",
            ]
        ],
        left_on=["_store_key", "_opening_month_index"],
        right_on=["_store_key", "_month_index"],
        how="left",
        validate="one_to_one",
    ).drop(columns="_month_index")
    opening_values["opening_month_panel_row"] = (
        opening_values["panel_row_present"].astype("boolean").fillna(False).astype(bool)
    )
    opening_values = opening_values.drop(columns="panel_row_present")

    for outcome_column, label in (
        ("APSD_income", "sales"),
        ("APSD_customers", "customers"),
    ):
        first_observed = (
            monthly.loc[monthly[outcome_column].notna()]
            .sort_values(["SalesMonth", "_store_key"])
            .drop_duplicates("_store_key")
            [["_store_key", "SalesMonth", outcome_column]]
            .rename(
                columns={
                    "SalesMonth": f"first_observed_month_{label}",
                    outcome_column: f"first_observed_{outcome_column}",
                }
            )
        )
        opening_values = opening_values.merge(
            first_observed,
            on="_store_key",
            how="left",
            validate="one_to_one",
        )
    latest_values: list[pd.DataFrame] = []
    for outcome_column in ("APSD_income", "APSD_customers"):
        latest = (
            monthly.loc[monthly[outcome_column].notna()]
            .sort_values(["SalesMonth", "_store_key"])
            .drop_duplicates("_store_key", keep="last")
            [["_store_key", "SalesMonth", outcome_column]]
            .rename(
                columns={
                    "SalesMonth": f"latest_month_{outcome_column}",
                    outcome_column: f"latest_{outcome_column}",
                }
            )
        )
        latest_values.append(latest)
    stores = stores.merge(
        opening_values[
            [
                "_store_key",
                "APSD_income",
                "APSD_customers",
                "opening_month_panel_row",
                "first_observed_month_sales",
                "first_observed_APSD_income",
                "first_observed_month_customers",
                "first_observed_APSD_customers",
            ]
        ].rename(
            columns={
                "APSD_income": "opening_month_APSD_income",
                "APSD_customers": "opening_month_APSD_customers",
            }
        ),
        on="_store_key",
        how="left",
        validate="one_to_one",
    )
    for latest in latest_values:
        stores = stores.merge(
            latest, on="_store_key", how="left", validate="one_to_one"
        )

    raw_opening = master_source.get(
        "ost_dt", pd.Series(index=master_source.index, dtype="object")
    )
    parsed_opening = _parse_dashboard_dates(raw_opening)
    raw_has_opening = raw_opening.notna()
    invalid_keys = set(
        normalize_store_id(master_source.loc[raw_has_opening, "store_id"])
        .loc[parsed_opening.loc[raw_has_opening].isna()]
        .dropna()
    )
    metadata_conflicts = {
        column: int(
            master.groupby("_store_key")[column]
            .nunique(dropna=True)
            .gt(1)
            .sum()
        )
        for column in ("store_name", "ost_dt")
        if column in master
    }
    latest_panel_month = monthly["SalesMonth"].max()
    diagnostics = {
        "store_count": int(stores["_store_key"].nunique()),
        "unique_store_names": int(stores["store_name"].dropna().nunique()),
        "stores_with_missing_name": int(stores["store_name"].isna().sum()),
        "stores_missing_opening_date": int(stores["ost_dt"].isna().sum()),
        "stores_with_invalid_opening_date": len(invalid_keys),
        "stores_missing_region": int(stores["analysis_region"].eq("Unmapped").sum()),
        "stores_missing_coordinates": int(
            stores[["LATITUDE", "LONGITUDE"]].isna().any(axis=1).sum()
        ),
        "duplicate_store_month_rows": duplicate_rows,
        "panel_start_month": monthly["SalesMonth"].min(),
        "panel_end_month": latest_panel_month,
        "distinct_panel_months": int(monthly["SalesMonth"].nunique()),
        "master_rows": int(len(master_source)),
        "master_store_id_conflicts": int(
            master.groupby("_store_key")["store_id"].nunique(dropna=True).gt(1).sum()
        ),
        "metadata_conflicts": metadata_conflicts,
        "stores_opened_after_panel_end": int(
            stores["ost_dt"].gt(latest_panel_month).sum()
        ),
    }
    return stores, monthly, diagnostics


def _month_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    month = int(value)
    return calendar.month_name[month] if 1 <= month <= 12 else "Unknown"


def _longest_missing_month_gap(values: Any) -> int:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return 0
    ordered = sorted(int(value) for value in values if pd.notna(value))
    return max(
        (right - left - 1 for left, right in zip(ordered, ordered[1:])),
        default=0,
    )


def statistics_page(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master_source: pd.DataFrame,
    outcome: str,
) -> None:
    st.header("Statistics")
    st.caption(
        "Descriptive statistics for CU store openings and APSD performance. "
        "The shared Outcome selector controls the primary performance outcome; "
        "annual tables and selected data-quality checks include both outcomes."
    )
    try:
        stores, monthly, diagnostics = prepare_store_statistics_data(
            panel, master_source
        )
    except ValueError as error:
        st.error(f"Cannot prepare store statistics: {error}")
        return
    if stores.empty or monthly.empty:
        st.warning("The store master or monthly panel has no usable records.")
        return

    month_names = list(calendar.month_name)[1:]
    opening_years = sorted(
        stores["opening_year"].dropna().astype(int).unique().tolist()
    )
    region_values = sorted(stores["analysis_region"].dropna().astype(str).unique())
    status_values = sorted(
        stores["Store_Status"].dropna().astype(str).unique().tolist()
    )
    available_months = monthly["SalesMonth"].dropna()
    panel_start = available_months.min()
    panel_end = available_months.max()
    if pd.isna(panel_start) or pd.isna(panel_end):
        st.warning("The monthly panel does not contain valid calendar months.")
        return

    st.subheader("Analysis filters")
    filter_columns = st.columns(5)
    year_choice = filter_columns[0].multiselect(
        "New-store opening year",
        options=opening_years,
        default=opening_years,
        key="statistics_opening_years",
        help="Filters opening cohorts and the stores included in performance summaries.",
    )
    month_choice = filter_columns[1].multiselect(
        "New-store opening month",
        options=month_names,
        default=month_names,
        key="statistics_opening_months",
    )
    region_choice = filter_columns[2].multiselect(
        "Region",
        options=region_values,
        default=region_values,
        key="statistics_regions",
    )
    status_choice = filter_columns[3].multiselect(
        "Latest store status",
        options=status_values,
        default=status_values,
        key="statistics_store_status",
        help="Uses the latest status recorded in the store master.",
    )
    age_groups = [
        "0-3 months",
        "4-12 months",
        "13-24 months",
        "25+ months",
        "Unknown opening age",
    ]
    age_choice = filter_columns[4].multiselect(
        "Store age group (monthly data)",
        options=["All age groups", *age_groups],
        default=["All age groups"],
        key="statistics_store_age_groups",
    )
    period_value = st.date_input(
        "Monthly outcome analysis period",
        value=(panel_start.date(), panel_end.date()),
        min_value=panel_start.date(),
        max_value=panel_end.date(),
        key="statistics_analysis_period",
        help="Applies to monthly trends, annual/seasonal performance, and cohort trajectories.",
    )
    if isinstance(period_value, (tuple, list)) and len(period_value) == 2:
        period_start, period_end = map(pd.Timestamp, period_value)
    else:
        period_start = pd.Timestamp(period_value[0] if period_value else panel_start.date())
        period_end = pd.Timestamp(period_value[-1] if period_value else panel_end.date())
    period_start = period_start.to_period("M").to_timestamp()
    period_end = period_end.to_period("M").to_timestamp()
    minimum_group_size = int(
        st.number_input(
            "Minimum stores for descriptive group comparisons",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            key="statistics_minimum_group_size",
        )
    )

    base_stores = stores.copy()
    if region_choice:
        base_stores = base_stores.loc[
            base_stores["analysis_region"].isin(region_choice)
        ]
    if status_choice:
        base_stores = base_stores.loc[
            base_stores["Store_Status"].astype("string").isin(status_choice)
        ]
    opening_stores = base_stores.loc[base_stores["ost_dt"].notna()].copy()
    if year_choice:
        opening_stores = opening_stores.loc[
            opening_stores["opening_year"].isin(year_choice)
        ]
    if month_choice:
        month_numbers = [month_names.index(month) + 1 for month in month_choice]
        opening_stores = opening_stores.loc[
            opening_stores["opening_month"].isin(month_numbers)
        ]
    has_cohort_filter = (
        (bool(year_choice) and len(year_choice) < len(opening_years))
        or (bool(month_choice) and len(month_choice) < len(month_names))
    )
    monthly_stores = (
        base_stores.loc[
            base_stores["_store_key"].isin(opening_stores["_store_key"])
        ]
        if has_cohort_filter
        else base_stores
    )
    selected_monthly = monthly.loc[
        monthly["_store_key"].isin(monthly_stores["_store_key"])
        & monthly["SalesMonth"].between(period_start, period_end)
    ].copy()
    if age_choice and "All age groups" not in age_choice:
        selected_monthly = selected_monthly.loc[
            selected_monthly["store_age_group"]
            .astype("string")
            .fillna("Unknown opening age")
            .isin(age_choice)
        ]

    tabs = st.tabs(
        [
            "Opening overview",
            "Opening-month performance",
            "Trends & regions",
            "Distance statistics",
            "Opening cohorts",
            "Data quality",
        ]
    )

    with tabs[0]:
        st.subheader("Store-opening overview")
        selected_year_count = int(opening_stores["opening_year"].nunique())
        selected_month_count = len(month_choice) if month_choice else 12
        openings_by_year = (
            opening_stores.groupby("opening_year")["_store_key"]
            .nunique()
            .sort_index()
        )
        kpis = st.columns(4)
        kpis[0].metric("Unique stores in network", f"{stores['_store_key'].nunique():,}")
        kpis[1].metric("Unique openings in selected cohorts", f"{len(opening_stores):,}")
        kpis[2].metric("Opening years represented", f"{selected_year_count:,}")
        kpis[3].metric(
            "Regions represented",
            f"{opening_stores['analysis_region'].nunique():,}",
        )
        second_kpis = st.columns(4)
        second_kpis[0].metric(
            "Average openings per represented year",
            f"{openings_by_year.mean():,.1f}" if selected_year_count else "Unavailable",
        )
        month_denominator = selected_year_count * selected_month_count
        second_kpis[1].metric(
            "Average openings per selected year-month",
            f"{len(opening_stores) / month_denominator:,.1f}"
            if month_denominator else "Unavailable",
        )
        second_kpis[2].metric(
            "Stores with valid opening dates",
            f"{int(base_stores['ost_dt'].notna().sum()):,}",
        )
        second_kpis[3].metric(
            "Stores with missing/invalid opening dates",
            f"{int(base_stores['ost_dt'].isna().sum()):,}",
        )
        st.caption(
            "Opening counts use unique store IDs with actual recorded opening dates. "
            "Average openings per year uses represented opening years; monthly average "
            "uses represented years × selected opening months."
        )

        chart_columns = st.columns(2)
        annual = openings_by_year.rename("openings").reset_index()
        if not annual.empty:
            fig = px.bar(
                annual,
                x="opening_year",
                y="openings",
                text="openings",
                title="Unique store openings by year",
                labels={"opening_year": "Opening year", "openings": "Stores opened"},
            )
            fig.update_traces(textposition="outside")
            chart_columns[0].plotly_chart(fig, width="stretch")
        monthly_counts = (
            opening_stores.groupby(["opening_month", "analysis_region"])["_store_key"]
            .nunique()
            .rename("openings")
            .reset_index()
        )
        if not monthly_counts.empty:
            monthly_counts["opening_month_name"] = monthly_counts[
                "opening_month"
            ].map(_month_label)
            fig = px.line(
                monthly_counts,
                x="opening_month_name",
                y="openings",
                color="analysis_region",
                markers=True,
                title="Openings by calendar month across selected years",
                labels={
                    "opening_month_name": "Opening month",
                    "openings": "Unique stores opened",
                    "analysis_region": "Region",
                },
                category_orders={"opening_month_name": month_names},
            )
            fig.update_traces(
                customdata=monthly_counts[["opening_month"]].to_numpy(),
                hovertemplate="%{x}<br>Openings: %{y}<extra></extra>",
            )
            chart_columns[1].plotly_chart(fig, width="stretch")
        trend = (
            opening_stores.groupby("opening_cohort_month")["_store_key"]
            .nunique()
            .rename("openings")
            .reset_index()
            .sort_values("opening_cohort_month")
        )
        if not trend.empty:
            fig = px.bar(
                trend,
                x="opening_cohort_month",
                y="openings",
                title="Monthly store-opening trend",
                labels={
                    "opening_cohort_month": "Calendar month",
                    "openings": "Unique stores opened",
                },
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")
        seasonality = opening_stores.pivot_table(
            index="opening_year",
            columns="opening_month",
            values="_store_key",
            aggfunc="nunique",
            fill_value=0,
        ).reindex(columns=range(1, 13), fill_value=0)
        if not seasonality.empty:
            fig = go.Figure(
                go.Heatmap(
                    z=seasonality.to_numpy(dtype=float),
                    x=month_names,
                    y=seasonality.index.astype(str),
                    text=seasonality.to_numpy(dtype=int).astype(str),
                    texttemplate="%{text}",
                    colorbar={"title": "Stores"},
                    hovertemplate="Year %{y}<br>%{x}: %{z} openings<extra></extra>",
                )
            )
            fig.update_layout(
                title="Opening seasonality: counts by year and month",
                xaxis_title="Opening month",
                yaxis_title="Opening year",
            )
            st.plotly_chart(fig, width="stretch")
        download_table(
            opening_stores,
            "Download filtered store openings",
            "statistics_store_openings.csv",
        )

    with tabs[1]:
        st.subheader(f"Opening-month performance — {_metric_label(outcome)}")
        outcome_openings = opening_stores.copy()
        opening_values = f"opening_month_{outcome}"
        first_values = f"first_observed_{outcome}"
        valid_opening = outcome_openings.loc[
            outcome_openings[opening_values].notna()
        ]
        valid_first = outcome_openings.loc[
            outcome_openings[first_values].notna()
        ]
        opening_n = int(valid_opening["_store_key"].nunique())
        kpis = st.columns(4)
        kpis[0].metric(
            "Mean opening-month APSD",
            f"{valid_opening[opening_values].mean():,.2f}" if opening_n else "Unavailable",
        )
        kpis[1].metric(
            "Median opening-month APSD",
            f"{valid_opening[opening_values].median():,.2f}" if opening_n else "Unavailable",
        )
        kpis[2].metric(
            "Opening-month standard deviation",
            f"{valid_opening[opening_values].std():,.2f}"
            if opening_n > 1 else "Unavailable",
        )
        kpis[3].metric("Stores with valid opening-month APSD", f"{opening_n:,}")
        second_kpis = st.columns(4)
        second_kpis[0].metric(
            "Opening-month coverage",
            f"{opening_n / len(opening_stores) * 100:.1f}%"
            if not opening_stores.empty else "Unavailable",
        )
        second_kpis[1].metric(
            "Opening-month minimum",
            f"{valid_opening[opening_values].min():,.2f}"
            if opening_n else "Unavailable",
        )
        second_kpis[2].metric(
            "Opening-month maximum",
            f"{valid_opening[opening_values].max():,.2f}"
            if opening_n else "Unavailable",
        )
        second_kpis[3].metric(
            "Mean first-observed APSD (separate)",
            f"{valid_first[first_values].mean():,.2f}"
            if not valid_first.empty else "Unavailable",
        )
        first_month_column = (
            "first_observed_month_sales"
            if outcome == "APSD_income"
            else "first_observed_month_customers"
        )
        first_month_stores = outcome_openings.loc[
            outcome_openings[first_month_column].notna()
        ].copy()
        first_matches = (
            first_month_stores[first_month_column].dt.to_period("M")
            == first_month_stores["ost_dt"].dt.to_period("M")
        )
        third_kpis = st.columns(2)
        third_kpis[0].metric(
            "First observation in opening month",
            f"{int(first_matches.sum()):,} / {len(first_month_stores):,}",
        )
        third_kpis[1].metric(
            "Share of first observations matching opening month",
            f"{first_matches.mean() * 100:.1f}%"
            if not first_month_stores.empty else "Unavailable",
        )
        opening_store_count = int(opening_stores["_store_key"].nunique())
        opening_row_count = int(
            opening_stores["opening_month_panel_row"]
            .astype("boolean")
            .fillna(False)
            .sum()
        )
        st.caption(
            f"Opening-month values are matched to the actual opening calendar month. "
            f"A panel row with missing APSD is not counted as a valid value. "
            f"Opening-month panel rows are available for {opening_row_count:,} of "
            f"{opening_store_count:,} stores; valid {_metric_label(outcome)} values "
            f"are available for {opening_n:,}. First-observed APSD uses each store's "
            "earliest valid value in the panel and is not substituted for a missing "
            "opening-month value."
        )
        if not outcome_openings.empty:
            outcome_openings["opening_month_name"] = outcome_openings[
                "opening_month"
            ].map(_month_label)
            chart_specs = [
                ("opening_year", "Opening year", "Opening-month APSD by opening year"),
                ("opening_month_name", "Opening month", "Opening-month APSD by month"),
                ("analysis_region", "Region", "Opening-month APSD by region"),
            ]
            for group_column, label, title in chart_specs:
                valid = outcome_openings.loc[
                    outcome_openings[opening_values].notna()
                ].copy()
                if group_column == "opening_month_name":
                    valid[group_column] = pd.Categorical(
                        valid[group_column], categories=month_names, ordered=True
                    )
                grouped = (
                    valid.groupby(group_column, observed=True)[opening_values]
                    .agg(mean="mean", median="median", stores="count")
                    .reset_index()
                    .sort_values(group_column)
                )
                if grouped.empty:
                    st.info(f"No valid opening-month outcomes are available by {label.lower()}.")
                    continue
                fig = go.Figure()
                for stat, name in (("mean", "Mean"), ("median", "Median")):
                    fig.add_trace(
                        go.Scatter(
                            x=grouped[group_column].astype(str),
                            y=grouped[stat],
                            mode="lines+markers",
                            name=name,
                            customdata=grouped[["stores"]].to_numpy(),
                            hovertemplate=(
                                f"{label}: %{{x}}<br>{name}: %{{y:.2f}}"
                                "<br>Stores: %{customdata[0]}<extra></extra>"
                            ),
                        )
                    )
                fig.update_layout(
                    title=title,
                    xaxis_title=label,
                    yaxis_title=_metric_label(outcome),
                )
                st.plotly_chart(fig, width="stretch")
                grouped["support_status"] = np.where(
                    grouped["stores"] >= minimum_group_size,
                    "Meets minimum sample",
                    "Below minimum sample",
                )
                st.dataframe(grouped, width="stretch", hide_index=True)

        paired_opening = outcome_openings.loc[
            outcome_openings["opening_month_APSD_income"].notna()
            & outcome_openings["opening_month_APSD_customers"].notna()
        ].copy()
        paired_n = len(paired_opening)
        if paired_n:
            paired_opening["_sales_outlier"] = False
            paired_opening["_customer_outlier"] = False
            for column, flag in (
                ("opening_month_APSD_income", "_sales_outlier"),
                ("opening_month_APSD_customers", "_customer_outlier"),
            ):
                q1, q3 = paired_opening[column].quantile([0.25, 0.75])
                iqr = q3 - q1
                paired_opening[flag] = (
                    paired_opening[column].lt(q1 - 1.5 * iqr)
                    | paired_opening[column].gt(q3 + 1.5 * iqr)
                )
            paired_opening["potential_outlier"] = (
                paired_opening["_sales_outlier"]
                | paired_opening["_customer_outlier"]
            ).map({True: "Potential IQR outlier", False: "Within IQR rule"})
            correlation = paired_opening[
                ["opening_month_APSD_income", "opening_month_APSD_customers"]
            ].corr().iloc[0, 1]
            fig = px.scatter(
                paired_opening,
                x="opening_month_APSD_income",
                y="opening_month_APSD_customers",
                color="analysis_region",
                hover_data=[
                    "store_name",
                    "existing_store_id",
                    "ost_dt",
                    "analysis_region",
                    "opening_month_APSD_income",
                    "opening_month_APSD_customers",
                    "potential_outlier",
                ],
                title="Opening-month APSD sales versus APSD customers",
                labels={
                    "opening_month_APSD_income": "APSD sales (MNT/active day)",
                    "opening_month_APSD_customers": "APSD customers/active day",
                },
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"{paired_n:,} stores contribute both opening-month outcomes; "
                f"Pearson correlation = {correlation:.3f}. Potential outliers are "
                "flagged using 1.5×IQR and retained in the chart."
            )
        else:
            st.info("No stores have both sales and customer APSD in the actual opening month.")
        download_table(
            outcome_openings,
            "Download opening-month and first-observed outcomes",
            f"statistics_opening_performance_{outcome}.csv",
        )

    with tabs[2]:
        st.subheader(f"Monthly and yearly performance — {_metric_label(outcome)}")
        if selected_monthly.empty:
            st.info("No monthly observations match the selected filters and period.")
        else:
            trend_data = (
                selected_monthly.loc[selected_monthly[outcome].notna()]
                .groupby(["SalesMonth", "analysis_region"], dropna=False)
                .agg(
                    mean=(outcome, "mean"),
                    median=(outcome, "median"),
                    stores=("_store_key", "nunique"),
                    observations=(outcome, "count"),
                )
                .reset_index()
            )
            trend_metric = st.selectbox(
                "Monthly trend statistic",
                ["Mean", "Median"],
                key="statistics_monthly_trend_stat",
            )
            trend_value = "mean" if trend_metric == "Mean" else "median"
            fig = px.line(
                trend_data,
                x="SalesMonth",
                y=trend_value,
                color="analysis_region",
                markers=True,
                custom_data=["stores", "observations"],
                title=f"Monthly {trend_metric.lower()} APSD by region",
                labels={
                    "SalesMonth": "Calendar month",
                    trend_value: _metric_label(outcome),
                    "analysis_region": "Region",
                },
            )
            fig.update_traces(
                hovertemplate=(
                    "%{x|%Y-%m}<br>APSD: %{y:.2f}"
                    "<br>Stores: %{customdata[0]}"
                    "<br>Store-month observations: %{customdata[1]}<extra></extra>"
                )
            )
            st.plotly_chart(fig, width="stretch")
            annual_frames: list[pd.DataFrame] = []
            for outcome_column in ("APSD_income", "APSD_customers"):
                annual = selected_monthly.loc[
                    selected_monthly[outcome_column].notna()
                ].copy()
                annual["calendar_year"] = annual["SalesMonth"].dt.year
                stats = (
                    annual.groupby("calendar_year")[outcome_column]
                    .agg(mean="mean", median="median", observations="count")
                    .reset_index()
                )
                store_counts = annual.groupby("calendar_year")["_store_key"].nunique()
                stats["stores_observed"] = stats["calendar_year"].map(store_counts)
                previous_year_mean = stats["mean"].shift(1)
                consecutive_year = stats["calendar_year"].diff().eq(1)
                stats["change_from_previous_year_pct"] = (
                    (stats["mean"] / previous_year_mean - 1) * 100
                ).where(previous_year_mean.gt(0) & consecutive_year)
                stats["outcome"] = _metric_label(outcome_column)
                annual_frames.append(stats)
            annual_table = pd.concat(annual_frames, ignore_index=True)
            st.subheader("Year-over-year APSD summary")
            st.dataframe(annual_table, width="stretch", hide_index=True)
            st.caption(
                "Annual means are pooled across valid store-month observations, not "
                "company totals. Store counts and observation counts are shown because "
                "network coverage changes across years."
            )
            download_table(
                annual_table,
                "Download annual APSD summary",
                "statistics_annual_apsd.csv",
            )
            seasonal = (
                selected_monthly.loc[selected_monthly[outcome].notna()]
                .assign(calendar_month=lambda frame: frame["SalesMonth"].dt.month)
                .groupby("calendar_month")[outcome]
                .agg(mean="mean", median="median", observations="count")
                .reindex(range(1, 13))
                .reset_index()
            )
            seasonal["month_name"] = seasonal["calendar_month"].map(_month_label)
            fig = go.Figure()
            for stat in ("mean", "median"):
                fig.add_trace(
                    go.Scatter(
                        x=seasonal["month_name"],
                        y=seasonal[stat],
                        mode="lines+markers",
                        name=stat.title(),
                        customdata=seasonal[["observations"]].fillna(0).to_numpy(),
                        hovertemplate=(
                            "%{x}<br>" + stat.title() + ": %{y:.2f}"
                            "<br>Store-month observations: %{customdata[0]}<extra></extra>"
                        ),
                    )
                )
            fig.update_layout(
                title=f"Calendar-month seasonality — {_metric_label(outcome)}",
                xaxis={"categoryorder": "array", "categoryarray": month_names},
                xaxis_title="Calendar month across selected years",
                yaxis_title=_metric_label(outcome),
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(seasonal, width="stretch", hide_index=True)

            age_summary = (
                selected_monthly.loc[
                    selected_monthly[outcome].notna()
                    & selected_monthly["store_age_group"].notna()
                ]
                .groupby("store_age_group", observed=True)[outcome]
                .agg(mean="mean", median="median", std="std", observations="count")
                .reset_index()
            )
            if not age_summary.empty:
                fig = px.bar(
                    age_summary,
                    x="store_age_group",
                    y="mean",
                    error_y="std",
                    custom_data=["observations"],
                    title="APSD by store age (calendar-month age)",
                    labels={
                        "store_age_group": "Calendar months since opening",
                        "mean": _metric_label(outcome),
                    },
                )
                fig.update_traces(
                    hovertemplate=(
                        "%{x}<br>Mean APSD: %{y:.2f}"
                        "<br>Store-month observations: %{customdata[0]}<extra></extra>"
                    )
                )
                st.plotly_chart(fig, width="stretch")
                st.dataframe(age_summary, width="stretch", hide_index=True)
            st.caption(
                "Store age is calculated as calendar-month index minus the actual "
                "opening-month index. Groups are 0–3, 4–12, 13–24, and 25+ months."
            )

        st.subheader("Regional summary")
        region_rows: list[dict[str, Any]] = []
        filtered_opening_metrics = opening_stores.copy()
        for region, group in base_stores.groupby("analysis_region", dropna=False):
            keys = set(group["_store_key"])
            open_group = filtered_opening_metrics.loc[
                filtered_opening_metrics["_store_key"].isin(keys)
            ]
            obs_group = selected_monthly.loc[
                selected_monthly["_store_key"].isin(keys)
            ]
            row: dict[str, Any] = {
                "region": region,
                "unique_stores": len(keys),
                "openings_in_selected_cohorts": len(open_group),
                "mean_opening_APSD_sales": open_group[
                    "opening_month_APSD_income"
                ].mean(),
                "median_opening_APSD_sales": open_group[
                    "opening_month_APSD_income"
                ].median(),
                "mean_opening_APSD_customers": open_group[
                    "opening_month_APSD_customers"
                ].mean(),
                "median_opening_APSD_customers": open_group[
                    "opening_month_APSD_customers"
                ].median(),
                "average_store_age_months_at_panel_end": group[
                    "_opening_month_index"
                ].map(
                    lambda value: (
                        int(panel_end.year * 12 + panel_end.month - value)
                        if pd.notna(value) else np.nan
                    )
                ).mean(),
            }
            for column, label in (
                ("APSD_income", "sales"),
                ("APSD_customers", "customers"),
            ):
                valid = obs_group.loc[obs_group[column].notna()]
                latest_by_store = (
                    valid.sort_values("SalesMonth")
                    .drop_duplicates("_store_key", keep="last")
                )
                row[f"valid_{label}_stores"] = int(valid["_store_key"].nunique())
                row[f"latest_mean_APSD_{label}"] = latest_by_store[column].mean()
                row[f"latest_median_APSD_{label}"] = latest_by_store[column].median()
                row[f"missing_{label}_observation_pct"] = (
                    100 * (1 - len(valid) / len(obs_group))
                    if len(obs_group) else np.nan
                )
            region_rows.append(row)
        regional_table = pd.DataFrame(region_rows).sort_values("region")
        regional_table["group_support"] = np.where(
            regional_table["unique_stores"] >= minimum_group_size,
            "Meets minimum sample",
            "Below minimum sample",
        )
        st.dataframe(regional_table, width="stretch", hide_index=True)
        fig = px.bar(
            regional_table,
            x="region",
            y="unique_stores",
            text="unique_stores",
            title="Unique stores by region",
            labels={"region": "Region", "unique_stores": "Unique store IDs"},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Opening APSD columns summarize actual opening-month observations. "
            "Latest APSD statistics use each store's latest valid outcome in the "
            "selected analysis period; missing values are excluded, not set to zero."
        )
        download_table(
            regional_table,
            "Download regional statistics",
            "statistics_regional_summary.csv",
        )
        region_openings = (
            opening_stores.groupby(["analysis_region", "opening_year"])["_store_key"]
            .nunique()
            .rename("openings")
            .reset_index()
        )
        if not region_openings.empty:
            fig = px.bar(
                region_openings,
                x="opening_year",
                y="openings",
                color="analysis_region",
                barmode="group",
                title="Annual openings by region",
                labels={
                    "opening_year": "Opening year",
                    "openings": "Unique stores opened",
                    "analysis_region": "Region",
                },
            )
            st.plotly_chart(fig, width="stretch")
        regional_observations = selected_monthly.loc[
            selected_monthly[outcome].notna()
        ]
        if not regional_observations.empty:
            fig = px.box(
                regional_observations,
                x="analysis_region",
                y=outcome,
                points="outliers",
                title=f"Monthly APSD distribution by region — {_metric_label(outcome)}",
                labels={
                    "analysis_region": "Region",
                    outcome: _metric_label(outcome),
                },
            )
            st.plotly_chart(fig, width="stretch")

    with tabs[3]:
        st.subheader("Existing-store exposure by new-store distance")
        st.caption(
            "Distance bands describe an existing-store/new-store pair. A new store "
            "can appear in several bands because it can have several nearby existing "
            "stores; this page does not assign one arbitrary band to the new store."
        )
        if pair_source.empty:
            st.info("The existing-store/new-store distance workbook is unavailable.")
        else:
            try:
                events, event_observations, event_diagnostics, band_order = (
                    prepare_dashboard_openings(panel, pair_source, master_source)
                )
            except ValueError as error:
                st.error(f"Cannot calculate distance statistics: {error}")
                events = pd.DataFrame()
                event_observations = pd.DataFrame()
                event_diagnostics = {}
                band_order = []
            if not events.empty:
                chosen_bands = st.multiselect(
                    "Existing-store/new-store distance bands",
                    options=band_order,
                    default=band_order,
                    key="statistics_distance_bands",
                )
                allowed_keys = set(base_stores["_store_key"])
                selected_pair_events = events.loc[
                    events["_existing_key"].isin(allowed_keys)
                    & events["distance_band"].isin(chosen_bands)
                ].copy()
                if year_choice:
                    selected_pair_events = selected_pair_events.loc[
                        selected_pair_events["new_opening_date"].dt.year.isin(year_choice)
                    ]
                if month_choice:
                    selected_pair_events = selected_pair_events.loc[
                        selected_pair_events["new_opening_date"].dt.month.isin(
                            [month_names.index(month) + 1 for month in month_choice]
                        )
                    ]
                event_ids = set(selected_pair_events["event_id"])
                selected_pair_observations = event_observations.loc[
                    event_observations["event_id"].isin(event_ids)
                ]
                pair_changes = build_opening_event_comparisons(
                    selected_pair_events,
                    selected_pair_observations,
                    before_months=6,
                    after_months=6,
                    include_event_month=False,
                )
                pair_base = (
                    selected_pair_events.groupby("distance_band", observed=True)
                    .agg(
                        unique_existing_new_pairs=("event_id", "nunique"),
                        unique_new_store_openings=("_opening_key", "nunique"),
                        unique_existing_stores=("_existing_key", "nunique"),
                        mean_distance_m=("distance_m", "mean"),
                    )
                    .reindex(chosen_bands)
                    .fillna(
                        {
                            "unique_existing_new_pairs": 0,
                            "unique_new_store_openings": 0,
                            "unique_existing_stores": 0,
                        }
                    )
                    .reset_index()
                )
                distance_rows: list[dict[str, Any]] = []
                for band in chosen_bands:
                    row: dict[str, Any] = {
                        str(key): value
                        for key, value in pair_base.loc[
                            pair_base["distance_band"] == band
                        ].iloc[0].to_dict().items()
                    }
                    for outcome_column, label in (
                        ("APSD_income", "sales"),
                        ("APSD_customers", "customers"),
                    ):
                        changes = pair_changes.loc[
                            (pair_changes["distance_band"] == band)
                            & (pair_changes["outcome"] == outcome_column)
                        ] if not pair_changes.empty else pd.DataFrame()
                        deltas = pd.to_numeric(
                            changes.get("absolute_change", pd.Series(dtype=float)),
                            errors="coerce",
                        ).dropna()
                        percentages = pd.to_numeric(
                            changes.get("percentage_change", pd.Series(dtype=float)),
                            errors="coerce",
                        ).dropna()
                        mean_change, change_ci_low, change_ci_high = (
                            _clustered_mean_interval(
                                deltas.reset_index(drop=True),
                                changes.loc[
                                    deltas.index, "_existing_key"
                                ].reset_index(drop=True)
                                if not changes.empty else pd.Series(dtype="string"),
                            )
                        )
                        percent_rows = pd.to_numeric(
                            changes.get("percentage_change", pd.Series(dtype=float)),
                            errors="coerce",
                        )
                        valid_percent = percent_rows.notna()
                        mean_percent, percent_ci_low, percent_ci_high = (
                            _clustered_mean_interval(
                                percent_rows.loc[valid_percent].reset_index(drop=True),
                                changes.loc[
                                    valid_percent, "_existing_key"
                                ].reset_index(drop=True)
                                if not changes.empty else pd.Series(dtype="string"),
                            )
                        )
                        delta_cluster_count = (
                            changes.loc[deltas.index, "_existing_key"].nunique()
                            if not changes.empty else 0
                        )
                        row.update(
                            {
                                f"valid_{label}_paired_events": int(changes["event_id"].nunique())
                                if not changes.empty else 0,
                                f"mean_pre_APSD_{label}": changes["pre_mean"].mean()
                                if not changes.empty else np.nan,
                                f"mean_post_APSD_{label}": changes["post_mean"].mean()
                                if not changes.empty else np.nan,
                                f"mean_absolute_change_{label}": mean_change,
                                f"median_absolute_change_{label}": deltas.median(),
                                f"standard_deviation_change_{label}": deltas.std(ddof=1),
                                f"clustered_standard_error_change_{label}": (
                                    (change_ci_high - change_ci_low)
                                    / (
                                        2
                                        * float(
                                            student_t.ppf(
                                                0.975, delta_cluster_count - 1
                                            )
                                        )
                                    )
                                    if delta_cluster_count > 1
                                    and pd.notna(change_ci_low)
                                    and pd.notna(change_ci_high)
                                    else np.nan
                                ),
                                f"clustered_ci_95_lower_change_{label}": change_ci_low,
                                f"clustered_ci_95_upper_change_{label}": change_ci_high,
                                f"mean_percentage_change_{label}": mean_percent,
                                f"clustered_ci_95_lower_pct_{label}": percent_ci_low,
                                f"clustered_ci_95_upper_pct_{label}": percent_ci_high,
                                f"valid_percentage_events_{label}": len(percentages),
                                f"support_status_{label}": (
                                    "Meets minimum sample"
                                    if not changes.empty
                                    and changes["event_id"].nunique() >= minimum_group_size
                                    else "Below minimum sample"
                                ),
                            }
                        )
                    distance_rows.append(row)
                distance_table = pd.DataFrame(distance_rows)
                distance_table["distance_band"] = pd.Categorical(
                    distance_table["distance_band"],
                    categories=band_order,
                    ordered=True,
                )
                st.dataframe(
                    distance_table.sort_values("distance_band"),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Pre period = months −6 to −1; post period = months +1 to +6. "
                    "Each valid result is one existing-store/new-store opening event "
                    "with observed outcomes on both sides; means are event-weighted. "
                    "Month 0 is excluded. Percentage changes are averaged per event "
                    "only for positive pre-opening baselines. Error bars use "
                    "existing-store-clustered 95% intervals where supported. "
                    "These descriptive changes are not causal."
                )
                bar_columns = st.columns(2)
                for column, metric, title, y_title in (
                    (
                        bar_columns[0],
                        "unique_existing_stores",
                        "Existing stores represented by pair events",
                        "Unique existing stores",
                    ),
                    (
                        bar_columns[1],
                        "unique_new_store_openings",
                        "Unique new-store openings represented by pair events",
                        "Unique new-store opening events",
                    ),
                ):
                    fig = px.bar(
                        distance_table.sort_values("distance_band"),
                        x="distance_band",
                        y=metric,
                        text=metric,
                        title=title,
                        labels={"distance_band": "Source distance band", metric: y_title},
                    )
                    fig.update_traces(textposition="outside")
                    column.plotly_chart(fig, width="stretch")
                change_columns = st.columns(2)
                for column, metric, title in (
                    (
                        change_columns[0],
                        "mean_absolute_change_sales",
                        "Mean paired APSD sales change",
                    ),
                    (
                        change_columns[1],
                        "mean_absolute_change_customers",
                        "Mean paired APSD customer change",
                    ),
                ):
                    metric_label = "sales" if metric.endswith("sales") else "customers"
                    distance_plot = distance_table.sort_values("distance_band")
                    ci_low = distance_plot[
                        f"clustered_ci_95_lower_change_{metric_label}"
                    ]
                    ci_high = distance_plot[
                        f"clustered_ci_95_upper_change_{metric_label}"
                    ]
                    mean_values = distance_plot[metric]
                    fig = px.bar(
                        distance_plot,
                        x="distance_band",
                        y=metric,
                        custom_data=[
                            f"valid_{metric_label}_paired_events"
                        ],
                        title=title,
                        labels={"distance_band": "Source distance band", metric: "Post − pre APSD"},
                    )
                    fig.update_traces(
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": (ci_high - mean_values).clip(lower=0),
                            "arrayminus": (mean_values - ci_low).clip(lower=0),
                        },
                        hovertemplate=(
                            "%{x}<br>Mean change: %{y:.2f}"
                            "<br>Valid paired events: %{customdata[0]}<extra></extra>"
                        )
                    )
                    column.plotly_chart(fig, width="stretch")
                pct_columns = st.columns(2)
                for column, metric, title in (
                    (
                        pct_columns[0],
                        "mean_percentage_change_sales",
                        "Mean event-level APSD sales change (%)",
                    ),
                    (
                        pct_columns[1],
                        "mean_percentage_change_customers",
                        "Mean event-level APSD customer change (%)",
                    ),
                ):
                    metric_label = "sales" if metric.endswith("sales") else "customers"
                    distance_plot = distance_table.sort_values("distance_band")
                    ci_low = distance_plot[
                        f"clustered_ci_95_lower_pct_{metric_label}"
                    ]
                    ci_high = distance_plot[
                        f"clustered_ci_95_upper_pct_{metric_label}"
                    ]
                    mean_values = distance_plot[metric]
                    fig = px.bar(
                        distance_plot,
                        x="distance_band",
                        y=metric,
                        title=title,
                        labels={"distance_band": "Source distance band", metric: "Mean percentage change"},
                    )
                    fig.update_traces(
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": (ci_high - mean_values).clip(lower=0),
                            "arrayminus": (mean_values - ci_low).clip(lower=0),
                        }
                    )
                    column.plotly_chart(fig, width="stretch")
                download_table(
                    distance_table.sort_values("distance_band"),
                    "Download distance-band statistics",
                    "statistics_distance_band_summary.csv",
                )
                st.caption(
                    f"Eligible source pair events available before filters: "
                    f"{event_diagnostics['eligible_opening_events']:,}. "
                    "Unique-store counts and opening-event counts are distinct "
                    "denominators; events outside observed panel follow-up may have "
                    "no paired APSD changes."
                )

    with tabs[4]:
        st.subheader(f"Opening-cohort performance — {_metric_label(outcome)}")
        granularity = st.radio(
            "Opening cohort definition",
            ["Opening year", "Opening month"],
            horizontal=True,
            key="statistics_cohort_granularity",
        )
        cohort_column = (
            "opening_year" if granularity == "Opening year" else "opening_cohort_month"
        )
        cohort_options = sorted(
            opening_stores[cohort_column].dropna().astype(str).unique().tolist()
        )
        default_cohorts = cohort_options[-6:] if granularity == "Opening year" else cohort_options[-12:]
        chosen_cohorts = st.multiselect(
            "Cohorts to compare",
            cohort_options,
            default=default_cohorts,
            key="statistics_selected_cohorts",
        )
        max_observed_age = pd.to_numeric(
            selected_monthly["store_age_month"], errors="coerce"
        ).max()
        age_max = max(12, min(60, int(max_observed_age) if pd.notna(max_observed_age) else 12))
        selected_age_max = st.slider(
            "Months since opening shown",
            min_value=0,
            max_value=age_max,
            value=min(24, age_max),
            key="statistics_cohort_age_max",
        )
        cohort_store_rows = opening_stores.loc[
            opening_stores[cohort_column].astype("string").isin(chosen_cohorts)
        ]
        cohort_keys = set(cohort_store_rows["_store_key"])
        cohort_data = selected_monthly.loc[
            selected_monthly["_store_key"].isin(cohort_keys)
            & selected_monthly["store_age_month"].between(0, selected_age_max)
            & selected_monthly[outcome].notna()
        ].copy()
        cohort_data["cohort"] = cohort_data[cohort_column].astype("string")
        cohort_summary = (
            cohort_data.groupby(["cohort", "store_age_month"])[outcome]
            .agg(mean="mean", median="median", stores="count")
            .reset_index()
        )
        if cohort_summary.empty:
            st.info("No valid outcome observations match the selected opening cohorts.")
        else:
            fig = px.line(
                cohort_summary,
                x="store_age_month",
                y="mean",
                color="cohort",
                markers=True,
                custom_data=["stores"],
                title="Mean APSD trajectory by actual opening cohort",
                labels={
                    "store_age_month": "Calendar months since opening",
                    "mean": _metric_label(outcome),
                    "cohort": "Opening cohort",
                },
            )
            fig.update_traces(
                hovertemplate=(
                    "Store age month %{x}<br>Mean APSD: %{y:.2f}"
                    "<br>Stores: %{customdata[0]}<extra></extra>"
                )
            )
            st.plotly_chart(fig, width="stretch")
            milestones = [age for age in (0, 1, 2, 3, 6, 12) if age <= selected_age_max]
            cohort_wide = cohort_summary.loc[
                cohort_summary["store_age_month"].isin(milestones)
            ].pivot(index="cohort", columns="store_age_month", values="mean")
            cohort_counts = cohort_summary.loc[
                cohort_summary["store_age_month"].isin(milestones)
            ].pivot(index="cohort", columns="store_age_month", values="stores")
            cohort_table = pd.DataFrame(index=cohort_wide.index)
            cohort_table["cohort_stores"] = cohort_store_rows.groupby(
                cohort_store_rows[cohort_column].astype("string")
            )["_store_key"].nunique()
            for age in milestones:
                cohort_table[f"mean_APSD_month_{age}"] = cohort_wide.get(age)
                cohort_table[f"stores_month_{age}"] = cohort_counts.get(age)
            if 0 in cohort_wide:
                for age in (3, 6, 12):
                    if age in cohort_wide:
                        change = cohort_wide[age] - cohort_wide[0]
                        cohort_table[f"absolute_change_month_{age}"] = change
                        cohort_table[f"percentage_change_month_{age}"] = (
                            (cohort_wide[age] / cohort_wide[0] - 1) * 100
                        ).where(cohort_wide[0] > 0)
            cohort_table = cohort_table.reset_index()
            cohort_table = cohort_table.rename(columns={"index": "cohort"})
            st.subheader("Cohort milestone comparison")
            st.dataframe(cohort_table, width="stretch", hide_index=True)
            download_table(
                cohort_table,
                "Download opening-cohort comparison",
                f"statistics_cohorts_{outcome}.csv",
            )
            means = cohort_summary.pivot(
                index="cohort", columns="store_age_month", values="mean"
            )
            counts = cohort_summary.pivot(
                index="cohort", columns="store_age_month", values="stores"
            )
            if not means.empty:
                age_columns = list(means.columns)
                fig = go.Figure(
                    go.Heatmap(
                        z=means.to_numpy(dtype=float),
                        x=age_columns,
                        y=means.index.astype(str),
                        customdata=counts.reindex_like(means).to_numpy(dtype=float),
                        colorbar={"title": _metric_label(outcome)},
                        hovertemplate=(
                            "Cohort %{y}<br>Age month %{x}<br>Mean APSD: %{z:.2f}"
                            "<br>Stores: %{customdata}<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(
                    title="Cohort-by-store-age heatmap",
                    xaxis_title="Calendar months since opening",
                    yaxis_title="Opening cohort",
                )
                st.plotly_chart(fig, width="stretch")
            st.caption(
                "Cohorts use actual opening dates; store age is measured in calendar "
                "months from the opening month. Later cohorts have less follow-up by "
                "construction. Differences may reflect location choices, expansion "
                "strategy, economic conditions, and data coverage, not age alone."
            )

    with tabs[5]:
        st.subheader("Data quality and dataset structure")
        if diagnostics["metadata_conflicts"]:
            conflicts = ", ".join(
                f"{key}: {value}" for key, value in diagnostics["metadata_conflicts"].items()
            )
        else:
            conflicts = "None"
        quality_rows = [
            ("Unique store IDs", diagnostics["store_count"]),
            ("Unique nonmissing store names", diagnostics["unique_store_names"]),
            ("Stores without a name", diagnostics["stores_with_missing_name"]),
            ("Stores with missing opening dates", diagnostics["stores_missing_opening_date"]),
            ("Stores with invalid opening dates", diagnostics["stores_with_invalid_opening_date"]),
            ("Stores without a mapped region", diagnostics["stores_missing_region"]),
            ("Stores missing at least one coordinate", diagnostics["stores_missing_coordinates"]),
            ("Store-month duplicate rows", diagnostics["duplicate_store_month_rows"]),
            ("Conflicting master metadata values (name/opening date)", conflicts),
            ("Monthly panel start", diagnostics["panel_start_month"].strftime("%Y-%m")),
            ("Monthly panel end", diagnostics["panel_end_month"].strftime("%Y-%m")),
            ("Distinct panel months", diagnostics["distinct_panel_months"]),
            ("Stores opening after panel end", diagnostics["stores_opened_after_panel_end"]),
        ]
        st.dataframe(
            pd.DataFrame(
                [(label, str(value)) for label, value in quality_rows],
                columns=["Data check", "Count / value"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Repeated store IDs across monthly master rows are expected. Duplicate "
            "store-month keys are checked separately. Historical APSD observations "
            "before the panel start are unavailable, not assumed to be zero."
        )
        month_quality = (
            monthly.groupby("SalesMonth")
            .agg(
                panel_rows=("_store_key", "size"),
                stores_observed=("_store_key", "nunique"),
                valid_sales=("APSD_income", "count"),
                valid_customers=("APSD_customers", "count"),
            )
            .reset_index()
        )
        month_quality["sales_missing_pct"] = (
            100 * (1 - month_quality["valid_sales"] / month_quality["panel_rows"])
        )
        month_quality["customers_missing_pct"] = (
            100 * (1 - month_quality["valid_customers"] / month_quality["panel_rows"])
        )
        st.subheader("Panel completeness by calendar month")
        st.dataframe(month_quality, width="stretch", hide_index=True)
        download_table(
            month_quality,
            "Download monthly coverage checks",
            "statistics_monthly_data_coverage.csv",
        )
        preopening = monthly.loc[
            monthly["store_age_month"].lt(0)
            & (monthly["APSD_income"].notna() | monthly["APSD_customers"].notna())
        ]
        st.metric(
            "Valid outcome rows before recorded opening month (flag for review)",
            f"{len(preopening):,}",
        )
        st.caption(
            "This count is flagged for review rather than automatically excluded; "
            "the saved panel's operating-month rules and source history should be "
            "checked before treating pre-opening records as invalid."
        )
        for outcome_column in ("APSD_income", "APSD_customers"):
            valid = monthly[outcome_column].dropna()
            if valid.empty:
                continue
            q10, q25, q75, q90, q99 = valid.quantile(
                [0.10, 0.25, 0.75, 0.90, 0.99]
            )
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = int(
                (valid.lt(q1 - 1.5 * iqr) | valid.gt(q3 + 1.5 * iqr)).sum()
            )
            store_month = monthly.loc[monthly[outcome_column].notna()]
            observation_counts = store_month.groupby("_store_key").size()
            month_indices = (
                store_month.sort_values(["_store_key", "_month_index"])
                .groupby("_store_key")["_month_index"]
                .agg(list)
            )
            longest_gaps = month_indices.map(_longest_missing_month_gap)
            distribution = pd.DataFrame(
                [
                    {
                        "outcome": _metric_label(outcome_column),
                        "valid_store_month_observations": len(valid),
                        "mean": valid.mean(),
                        "median": valid.median(),
                        "standard_deviation": valid.std(),
                        "P10": q10,
                        "P25": q25,
                        "P75": q75,
                        "P90": q90,
                        "P99": q99,
                        "zero_observations": int(valid.eq(0).sum()),
                        "negative_observations": int(valid.lt(0).sum()),
                        "potential_1_5_IQR_outliers": outlier_count,
                        "stores_with_only_one_valid_month": int(
                            observation_counts.eq(1).sum()
                        ),
                        "stores_with_gap_over_3_months": int(longest_gaps.gt(3).sum()),
                        "longest_missing_month_gap": int(longest_gaps.max())
                        if not longest_gaps.empty else 0,
                    }
                ]
            )
            st.subheader(f"APSD distribution — {_metric_label(outcome_column)}")
            st.dataframe(distribution, width="stretch", hide_index=True)
            fig = px.box(
                monthly.loc[monthly[outcome_column].notna()],
                y=outcome_column,
                points="outliers",
                title=f"Observed APSD distribution — {_metric_label(outcome_column)}",
                labels={outcome_column: _metric_label(outcome_column)},
            )
            st.plotly_chart(fig, width="stretch")

        comparable = monthly.loc[
            monthly["APSD_income"].notna() & monthly["APSD_customers"].notna()
        ].copy()
        if not comparable.empty:
            comparable = comparable.loc[comparable["APSD_customers"] > 0]
            correlation = comparable[
                ["APSD_income", "APSD_customers"]
            ].corr().iloc[0, 1] if len(comparable) > 1 else np.nan
            comparable["sales_per_customer_day"] = (
                comparable["APSD_income"] / comparable["APSD_customers"]
            )
            st.metric(
                "Paired store-month sales/customer-equivalent relationship",
                f"{correlation:.3f}" if pd.notna(correlation) else "Unavailable",
                f"{len(comparable):,} valid same-month pairs",
            )
            st.metric(
                "Mean sales per customer-equivalent per active day",
                f"{comparable['sales_per_customer_day'].mean():,.2f}"
                if not comparable.empty else "Unavailable",
                "Calculated only where both APSDs exist and APSD customers > 0",
            )
        else:
            st.info("No paired sales/customer observations support a same-month ratio.")

        try:
            opening_events, _, event_diagnostics, _ = prepare_dashboard_openings(
                panel, pair_source, master_source
            ) if not pair_source.empty else (pd.DataFrame(), pd.DataFrame(), {}, [])
            multiple_exposure_stores = (
                int(
                    opening_events.groupby("_existing_key")["event_id"]
                    .nunique()
                    .gt(1)
                    .sum()
                )
                if not opening_events.empty else 0
            )
        except ValueError:
            multiple_exposure_stores = 0
            event_diagnostics = {}
        st.metric(
            "Existing stores with multiple recorded nearby openings",
            f"{multiple_exposure_stores:,}",
            "Based on pair-workbook history; not necessarily lifetime history",
        )


def regional_distance_statistics_page(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master: pd.DataFrame,
    outcome: str,
    outcome_label: str,
) -> None:
    st.header("Regional & Distance Statistics")
    st.caption(
        "Descriptive paired changes around nearby new-store opening events. "
        "These before-and-after comparisons are not causal estimates; other "
        "openings, seasonality, promotions, and local demand changes may also affect outcomes."
    )
    if pair_source.empty:
        st.error("The pairwise distance workbook is unavailable or has no usable rows.")
        return
    try:
        all_events, observations, diagnostics, band_order = prepare_dashboard_openings(
            panel, pair_source, master
        )
    except ValueError as error:
        st.error(f"Cannot calculate opening statistics: {error}")
        return
    if all_events.empty:
        st.warning("No qualifying existing-store/new-store opening pairs were found.")
        return

    with st.expander("Data checks and exclusions"):
        st.write(
            "Pair events are unique existing-store/new-store pairs. Existing-store "
            "opening dates from the saved panel take precedence; the pair workbook "
            "date is only a fallback. Missing or invalid dates, unknown existing "
            "opening dates, distances, and events before the existing store opened "
            "are not used."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Check": label, "Count / value": value}
                    for label, value in (
                        ("Pair rows in source workbook", diagnostics["source_pair_rows"]),
                        ("Eligible pair opening events", diagnostics["eligible_opening_events"]),
                        ("Distinct new-store openings", diagnostics["unique_new_store_openings"]),
                        ("Existing stores exposed", diagnostics["distinct_existing_stores_exposed"]),
                        ("Missing new-opening dates", diagnostics["missing_new_opening_date"]),
                        ("Missing existing-opening dates after fallback", diagnostics["missing_existing_opening_date_after_fallback"]),
                        ("Invalid distances", diagnostics["invalid_distance_rows"]),
                        ("Unclassified distance bands", diagnostics["missing_distance_band_rows"]),
                        ("Existing-opening date discrepancies", diagnostics["opening_date_mismatches"]),
                        ("Pairs where existing store was not yet open", diagnostics["pairs_not_preexisting"]),
                        ("Opening events outside outcome-panel dates", diagnostics["events_outside_observed_panel_dates"]),
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Observed outcomes cover {diagnostics['panel_start_month']} through "
            f"{diagnostics['panel_end_month']}. Event orders are based on all "
            "qualifying opening pairs available in the pair workbook, not only "
            "events within the selected date range. Therefore, labels say "
            "'first observed' rather than 'first ever'."
        )

    region_values = sorted(
        all_events["region"].dropna().astype(str).unique().tolist()
    )
    region_options = ["All Regions", *region_values]
    distance_options = ["All Bands", *band_order]
    order_options = [
        "First observed opening event",
        "Second observed opening event",
        "Third or later observed opening event",
    ]
    filter_cols = st.columns(3)
    selected_regions = filter_cols[0].multiselect(
        "Existing-store region",
        options=region_options,
        default=["All Regions"],
        key="regional_stats_regions",
    )
    selected_bands = filter_cols[1].multiselect(
        "Distance band (source workbook)",
        options=distance_options,
        default=["All Bands"],
        key="regional_stats_distance_bands",
        help="Uses the workbook's labels; bands are not re-binned.",
    )
    selected_orders = filter_cols[2].multiselect(
        "Opening order",
        options=order_options,
        default=order_options,
        key="regional_stats_opening_order",
    )
    selected_events, _ = _filter_events_by_dates(
        all_events, "regional_stats_opening_date_range"
    )
    window_cols = st.columns(4)
    before_months = window_cols[0].slider(
        "Pre-opening months",
        min_value=1,
        max_value=12,
        value=6,
        key="regional_stats_pre_window",
    )
    after_months = window_cols[1].slider(
        "Post-opening months",
        min_value=1,
        max_value=12,
        value=6,
        key="regional_stats_post_window",
    )
    include_event_month = window_cols[2].checkbox(
        "Include opening month (0) in post period",
        value=False,
        key="regional_stats_include_month_zero",
        help="Month 0 may be only partly exposed; it is separate by default.",
    )
    minimum_events = window_cols[3].number_input(
        "Minimum opening events per group",
        min_value=1,
        max_value=500,
        value=5,
        step=1,
        key="regional_stats_min_events",
    )
    minimum_stores = st.number_input(
        "Minimum distinct existing stores per group",
        min_value=1,
        max_value=250,
        value=3,
        step=1,
        key="regional_stats_min_stores",
    )

    if selected_regions and "All Regions" not in selected_regions:
        selected_events = selected_events.loc[
            selected_events["region"].isin(selected_regions)
        ]
    if selected_bands and "All Bands" not in selected_bands:
        selected_events = selected_events.loc[
            selected_events["distance_band"].isin(selected_bands)
        ]
    selected_events = selected_events.loc[
        selected_events["exposure_order"].isin(selected_orders)
    ]
    event_ids = set(selected_events["event_id"])
    filtered_observations = observations.loc[
        observations["event_id"].isin(event_ids)
    ]
    changes = build_opening_event_comparisons(
        selected_events,
        filtered_observations,
        before_months=before_months,
        after_months=after_months,
        include_event_month=include_event_month,
    )
    summary = summarize_opening_changes(
        changes,
        outcome,
        minimum_events=minimum_events,
        minimum_stores=minimum_stores,
    )

    st.subheader(f"Opening impact summary — {outcome_label}")
    all_event_count = int(selected_events["_opening_key"].nunique())
    exposed_store_count = int(selected_events["_existing_key"].nunique())
    overall = summary.loc[summary["group_type"] == "Overall"] if not summary.empty else pd.DataFrame()
    paired = overall.iloc[0] if not overall.empty else pd.Series(dtype=object)
    kpis = st.columns(4)
    kpis[0].metric("Unique new-store openings", f"{all_event_count:,}")
    kpis[1].metric("Existing stores exposed", f"{exposed_store_count:,}")
    kpis[2].metric(
        "Opening events with paired valid outcomes",
        f"{int(paired.get('opening_events', 0)):,}",
    )
    kpis[3].metric(
        "Average APSD before opening",
        f"{paired.get('mean_before_per_opening_event', float('nan')):,.2f}",
    )
    second_kpis = st.columns(4)
    second_kpis[0].metric(
        "Average APSD after opening",
        f"{paired.get('mean_after_per_opening_event', float('nan')):,.2f}",
    )
    second_kpis[1].metric(
        "Mean absolute change",
        f"{paired.get('mean_absolute_change_per_opening_event', float('nan')):+,.2f}",
    )
    second_kpis[2].metric(
        "Mean percentage change",
        f"{paired.get('mean_percentage_change_per_valid_baseline_event', float('nan')):+.2f}%",
    )
    second_kpis[3].metric(
        "Median absolute change",
        f"{paired.get('median_absolute_change_per_opening_event', float('nan')):+,.2f}",
    )
    third_kpis = st.columns(4)
    third_kpis[0].metric(
        "Opening events with a decline",
        f"{int(paired.get('decline_events', 0)):,} "
        f"({paired.get('decline_event_percentage', float('nan')):.1f}%)",
    )
    third_kpis[1].metric(
        "Opening events with an increase",
        f"{int(paired.get('increase_events', 0)):,} "
        f"({paired.get('increase_event_percentage', float('nan')):.1f}%)",
    )
    third_kpis[2].metric(
        "Mean observed pre-period months/event",
        f"{paired.get('mean_observed_pre_months_per_event', float('nan')):.1f}",
    )
    third_kpis[3].metric(
        "Mean observed post-period months/event",
        f"{paired.get('mean_observed_post_months_per_event', float('nan')):.1f}",
    )
    post_start = 0 if include_event_month else 1
    st.caption(
        f"Each paired observation is one existing-store/new-store opening event. "
        f"Pre period: months −{before_months} to −1; post period: +{post_start} "
        f"to +{after_months}. Means are averages of each event's observed monthly "
        "APSD means, not averages over all raw store-month rows. Event month 0 is "
        "excluded unless selected. Percent changes are calculated only for events "
        "with a positive pre-opening APSD baseline."
    )
    st.caption(
        "Confidence intervals in the group table use a store-clustered paired-mean "
        "standard error, allowing repeated opening events for the same existing "
        "store to be dependent. Groups below the selected support thresholds are "
        "flagged; no p-values or causal claims are reported."
    )

    st.subheader("Paired comparison by region and distance band")
    if summary.empty:
        st.info("No opening events have observed APSD values in both selected periods.")
    else:
        group_options = [
            "Overall",
            "Region",
            "Distance band",
            "Region × distance band",
            "Exposure order",
            "Exposure order × region",
            "Exposure order × distance band",
            "Exposure order × region × distance band",
        ]
        chosen_groups = st.multiselect(
            "Summary breakdowns",
            options=group_options,
            default=["Region", "Distance band", "Region × distance band"],
            key="regional_stats_summary_breakdowns",
        )
        table = summary.loc[summary["group_type"].isin(chosen_groups)].copy()
        table = table.sort_values(
            ["group_type", "region", "distance_band", "exposure_order"],
            kind="stable",
        )
        st.dataframe(table, width="stretch", hide_index=True)
        download_table(
            table,
            "Download paired summary CSV",
            f"opening_summary_{outcome}.csv",
        )
        download_table(
            changes.loc[changes["outcome"] == outcome],
            "Download event-level paired changes CSV",
            f"opening_event_changes_{outcome}.csv",
        )

    st.subheader("Distance gradient")
    distance_summary = summary.loc[
        (summary["group_type"] == "Distance band")
        & (summary["support_status"] == "Supported")
    ].copy() if not summary.empty else pd.DataFrame()
    if distance_summary.empty:
        st.info("No distance band meets the selected event and store support thresholds.")
    else:
        distance_summary["distance_band"] = pd.Categorical(
            distance_summary["distance_band"],
            categories=band_order,
            ordered=True,
        )
        gradient_cols = st.columns(2)
        for column, value, low, high, title, y_label in (
            (
                gradient_cols[0],
                "mean_absolute_change_per_opening_event",
                "clustered_ci_95_lower_change",
                "clustered_ci_95_upper_change",
                "Mean APSD change by distance",
                "Mean APSD change",
            ),
            (
                gradient_cols[1],
                "mean_percentage_change_per_valid_baseline_event",
                "mean_percentage_ci_95_lower",
                "mean_percentage_ci_95_upper",
                "Mean percentage APSD change by distance",
                "Mean percentage change (%)",
            ),
        ):
            fig = go.Figure(
                go.Bar(
                    x=distance_summary["distance_band"].astype(str),
                    y=distance_summary[value],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": distance_summary[high] - distance_summary[value],
                        "arrayminus": distance_summary[value] - distance_summary[low],
                    },
                    customdata=distance_summary[
                        ["existing_stores", "opening_events"]
                    ].to_numpy(),
                    hovertemplate=(
                        "%{x}<br>Mean: %{y:.2f}"
                        "<br>Existing stores: %{customdata[0]}"
                        "<br>Opening events: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Source distance band",
                yaxis_title=y_label,
                xaxis={"categoryorder": "array", "categoryarray": band_order},
            )
            fig.add_hline(y=0, line_color="black", line_width=1)
            column.plotly_chart(fig, width="stretch")

    st.subheader("Region × distance heatmaps")
    heatmap_outcome = st.selectbox(
        "Heatmap outcome",
        ["Both outcomes", "APSD sales", "APSD customers"],
        index=0,
        key="regional_stats_heatmap_outcome",
    )
    map_outcomes = (
        [("APSD sales", "APSD_income"), ("APSD customers", "APSD_customers")]
        if heatmap_outcome == "Both outcomes"
        else [
            (
                heatmap_outcome,
                "APSD_income" if heatmap_outcome == "APSD sales" else "APSD_customers",
            )
        ]
    )
    heatmap_regions = (
        region_values
        if not selected_regions or "All Regions" in selected_regions
        else [region for region in selected_regions if region in region_values]
    )
    heatmap_bands = (
        band_order
        if not selected_bands or "All Bands" in selected_bands
        else [band for band in band_order if band in selected_bands]
    )
    heatmap_changes = changes
    for heatmap_label, heatmap_outcome_column in map_outcomes:
        outcome_changes = heatmap_changes.loc[
            heatmap_changes["outcome"] == heatmap_outcome_column
        ]
        if outcome_changes.empty:
            outcome_summary = pd.DataFrame()
        elif heatmap_outcome_column == outcome:
            outcome_summary = summary
        else:
            outcome_summary = summarize_opening_changes(
                heatmap_changes,
                heatmap_outcome_column,
                minimum_events=minimum_events,
                minimum_stores=minimum_stores,
            )
        cells = outcome_summary.loc[
            outcome_summary["group_type"] == "Region × distance band"
        ].copy() if not outcome_summary.empty else pd.DataFrame()
        cell_grid = pd.MultiIndex.from_product(
            [heatmap_regions, heatmap_bands], names=["region", "distance_band"]
        ).to_frame(index=False)
        if not cells.empty:
            cell_grid = cell_grid.merge(
                cells,
                on=["region", "distance_band"],
                how="left",
                validate="one_to_one",
            )
        for metric_column, title, percent in (
            (
                "mean_absolute_change_per_opening_event",
                f"{heatmap_label}: mean absolute APSD change",
                False,
            ),
            (
                "mean_percentage_change_per_valid_baseline_event",
                f"{heatmap_label}: mean percentage APSD change",
                True,
            ),
        ):
            if cell_grid.empty:
                st.info(f"No region-by-distance groups are available for {heatmap_label}.")
                continue
            supported = (
                cell_grid["support_status"].eq("Supported")
                if "support_status" in cell_grid
                else pd.Series(False, index=cell_grid.index)
            )
            z_values = cell_grid[metric_column].where(supported)
            pivot = pd.DataFrame(
                {
                    "region": cell_grid["region"],
                    "distance_band": cell_grid["distance_band"],
                    "value": z_values,
                    "opening_events": cell_grid.get(
                        "opening_events", pd.Series(0, index=cell_grid.index)
                    ).fillna(0),
                    "existing_stores": cell_grid.get(
                        "existing_stores", pd.Series(0, index=cell_grid.index)
                    ).fillna(0),
                    "support_status": cell_grid.get(
                        "support_status", pd.Series("Insufficient support", index=cell_grid.index)
                    ).fillna("Insufficient support"),
                }
            )
            z_matrix = pivot.pivot(
                index="region", columns="distance_band", values="value"
            ).reindex(index=heatmap_regions, columns=heatmap_bands)
            custom_matrix = []
            text_matrix = []
            for region_value in heatmap_regions:
                custom_row = []
                text_row = []
                for band_value in heatmap_bands:
                    cell = pivot.loc[
                        (pivot["region"] == region_value)
                        & (pivot["distance_band"] == band_value)
                    ]
                    if cell.empty:
                        event_n, store_n, status = 0, 0, "Insufficient support"
                    else:
                        item = cell.iloc[0]
                        event_n = int(item["opening_events"])
                        store_n = int(item["existing_stores"])
                        status = str(item["support_status"])
                    custom_row.append([event_n, store_n, status])
                    cell_value = z_matrix.at[region_value, band_value]
                    if pd.isna(cell_value):
                        text_row.append(f"Low support<br>n={event_n} events")
                    else:
                        text_row.append(
                            f"{cell_value:+.1f}{'%' if percent else ''}"
                            f"<br>n={event_n}"
                        )
                custom_matrix.append(custom_row)
                text_matrix.append(text_row)
            fig = go.Figure(
                go.Heatmap(
                    z=z_matrix.to_numpy(dtype=float),
                    x=heatmap_bands,
                    y=heatmap_regions,
                    text=text_matrix,
                    texttemplate="%{text}",
                    customdata=custom_matrix,
                    zmid=0,
                    colorscale="RdYlGn",
                    colorbar={"title": "Change (%)" if percent else "APSD change"},
                    hovertemplate=(
                        "Region: %{y}<br>Distance band: %{x}"
                        "<br>Change: %{z:.2f}"
                        "<br>Opening events: %{customdata[0]}"
                        "<br>Existing stores: %{customdata[1]}"
                        "<br>Status: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Source distance band (categorical)",
                yaxis_title="Existing-store region",
            )
            st.plotly_chart(fig, width="stretch")
    st.caption(
        "Heatmap cells below the selected minimum event/store support are visibly "
        "flagged rather than colored as stable estimates. Counts in each cell are "
        "unique paired opening events and existing stores."
    )

    st.subheader("Event-time trajectory")
    compare_by = st.selectbox(
        "Compare event trajectories by",
        ["Exposure order", "Region", "Distance band", "No split"],
        key="regional_stats_trajectory_split",
    )
    if compare_by == "Exposure order":
        compare_column = "exposure_order"
    elif compare_by == "Region":
        compare_column = "region"
    elif compare_by == "Distance band":
        compare_column = "distance_band"
    else:
        compare_column = ""
    trajectory_groups = (
        [(str(value), group) for value, group in selected_events.groupby(compare_column)]
        if compare_column
        else [("Selected events", selected_events)]
    )
    trajectory_rows: list[pd.DataFrame] = []
    for group_label, group_events in trajectory_groups:
        stats = summarize_event_time(filtered_observations, group_events, outcome)
        if stats.empty:
            continue
        stats = stats.set_index("event_time_month").reindex(range(-12, 13))
        stats.index.name = "event_time_month"
        stats = stats.reset_index()
        for count_column in (
            "pair_month_observations",
            "opening_events",
            "existing_stores",
        ):
            stats[count_column] = stats[count_column].fillna(0).astype(int)
        supported = (
            stats["opening_events"].ge(minimum_events)
            & stats["existing_stores"].ge(minimum_stores)
        )
        stats.loc[~supported, ["mean_outcome", "ci_95_lower", "ci_95_upper"]] = np.nan
        stats["support_status"] = np.where(
            supported, "Supported", "Insufficient support"
        )
        stats["series"] = group_label
        trajectory_rows.append(stats)
    trajectories = (
        pd.concat(trajectory_rows, ignore_index=True)
        if trajectory_rows
        else pd.DataFrame()
    )
    if trajectories.empty:
        st.info("No event-time outcome observations are available for this selection.")
    else:
        fig = go.Figure()
        for group_label, group in trajectories.groupby("series", sort=False):
            custom = group[
                ["pair_month_observations", "opening_events", "existing_stores"]
            ].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=group["event_time_month"],
                    y=group["mean_outcome"],
                    mode="lines+markers",
                    name=group_label,
                    connectgaps=False,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": group["ci_95_upper"] - group["mean_outcome"],
                        "arrayminus": group["mean_outcome"] - group["ci_95_lower"],
                        "visible": True,
                    },
                    customdata=custom,
                    hovertemplate=(
                        "Event month %{x}<br>Mean APSD: %{y:.2f}"
                        "<br>Pair-event observations: %{customdata[0]}"
                        "<br>Unique opening events: %{customdata[1]}"
                        "<br>Existing stores: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
        fig.add_vrect(
            x0=-12.5, x1=-0.5, fillcolor="#dbeafe", opacity=0.25, line_width=0
        )
        fig.add_vrect(
            x0=0.5, x1=12.5, fillcolor="#dcfce7", opacity=0.18, line_width=0
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#b91c1c",
            annotation_text="Opening month (0; may be partial)",
        )
        fig.update_layout(
            title=f"Observed mean {_metric_label(outcome)} around opening events",
            xaxis_title="Calendar months from new-store opening month",
            yaxis_title=_metric_label(outcome),
            hovermode="x unified",
            legend_title=compare_by,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each point averages available event-month observations; the number of "
            "pair-months, unique openings, and existing stores appears on hover. "
            "The sample composition can change across event months. Missing months "
            "remain missing, not zero. Shaded intervals separate pre-opening from "
            "post-opening months; clustered 95% intervals account for repeated "
            "observations from the same existing store."
        )
        st.dataframe(
            trajectories,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Do stores with previous nearby openings respond differently?")
    order_summary = summary.loc[
        summary["group_type"] == "Exposure order"
    ].copy() if not summary.empty else pd.DataFrame()
    repeated = changes.loc[
        (changes["outcome"] == outcome) & changes["event_order"].ge(2)
    ].copy() if not changes.empty else pd.DataFrame()
    if order_summary.empty:
        st.info("No supported paired events by opening order are available.")
    else:
        st.dataframe(order_summary, width="stretch", hide_index=True)
        st.caption(
            "Opening order reflects prior qualifying nearby pair events in the "
            "available distance workbook, including events before the outcome panel "
            "starts. `previous_nearby_openings_in_pair_history` counts earlier "
            "opening pairs for that existing store; it is not the number of new "
            "stores opened in the selected before/after window."
        )
    if not repeated.empty:
        st.write(
            f"Repeated-exposure sample: {repeated['_existing_key'].nunique():,} "
            f"existing stores and {repeated['event_id'].nunique():,} paired opening "
            "events with valid before/after outcomes."
        )
        repeat_event_summary = summarize_opening_changes(
            repeated,
            outcome,
            minimum_events=minimum_events,
            minimum_stores=minimum_stores,
        )
        repeat_table = repeat_event_summary.loc[
            repeat_event_summary["group_type"].isin(
                [
                    "Exposure order × region",
                    "Exposure order × distance band",
                    "Exposure order × region × distance band",
                ]
            )
        ]
        st.dataframe(repeat_table, width="stretch", hide_index=True)
    else:
        st.info("No repeated-opening events with paired valid outcomes in this selection.")


def download_table(data: pd.DataFrame, label: str, file_name: str) -> None:

    st.download_button(

        label=label,

        data=data.to_csv(index=False).encode("utf-8-sig"),

        file_name=file_name,

        mime="text/csv",

    )


@st.cache_data(show_spinner=False)
def read_expansion_output(name: str, modified_ns: int) -> pd.DataFrame:
    path = BASE_DIR / "outputs" / "expansion_simulation" / name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def expansion_simulation_page(master: pd.DataFrame) -> None:
    st.header("Expansion Simulation")
    st.caption(
        "Forward-looking 2027 scenarios using historical demand validation and "
        "saved cannibalization estimates. These are conditional model estimates, "
        "not guaranteed forecasts, profit, or ROI."
    )
    def read_output(name: str) -> pd.DataFrame:
        path = BASE_DIR / "outputs" / "expansion_simulation" / name
        modified_ns = path.stat().st_mtime_ns if path.is_file() else 0
        return read_expansion_output(name, modified_ns)

    summaries = read_output("scenario_summaries.csv")
    rankings = read_output("potential_store_rankings.csv")
    monthly = read_output("monthly_network_simulation_2027.csv")
    existing_impact = read_output("existing_store_impact_2027.csv")
    regional = read_output("regional_scenario_comparison.csv")
    validation = read_output("model_validation_report.csv")
    sensitivity = read_output("scenario_sensitivity.csv")
    robustness = read_output("candidate_robustness.csv")

    if summaries.empty or rankings.empty:
        st.info(
            "No expansion simulation outputs are available yet. Run "
            "`python expansion_simulation.py` from the project root, then "
            "refresh this page."
        )
        return

    targets = sorted(pd.to_numeric(summaries["scenario_target"], errors="coerce").dropna().astype(int).unique())
    target = st.selectbox(
        "Expansion scenario",
        targets,
        format_func=lambda value: "Baseline — no new stores" if value == 0 else f"{value} selected stores",
    )
    summary_rows = summaries.loc[
        pd.to_numeric(summaries["scenario_target"], errors="coerce").eq(target)
    ]
    if summary_rows.empty:
        st.warning(f"No saved results are available for the {target}-store scenario.")
        return
    summary = summary_rows.iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected locations", f"{int(summary['selected_count']):,}")
    metric_columns[1].metric(
        "New-store sales, 2027",
        f"{summary['new_store_sales_2027_mnt']:,.0f} MNT",
        help="APSD demand multiplied by forecast active days; opening-month days are prorated.",
    )
    metric_columns[2].metric(
        "Existing-store sales change",
        f"{summary['existing_sales_change_2027_mnt']:,.0f} MNT",
    )
    metric_columns[3].metric(
        "Net network sales change",
        f"{summary['network_sales_change_2027_mnt']:,.0f} MNT",
        help="Scenario sales change relative to the no-expansion baseline; not profit or ROI.",
    )
    st.caption(
        f"Assumptions: {summary['opening_schedule']} opening schedule, day "
        f"{int(summary['opening_day'])} of each month, {int(summary['forecast_year'])}. "
        f"Candidate shortfall: {int(summary['shortfall'])}."
    )

    scenario_monthly = monthly.loc[
        pd.to_numeric(monthly["scenario_target"], errors="coerce").eq(target)
    ].copy()
    candidate_col = f"selected_{target}"
    if candidate_col in rankings:
        rankings[candidate_col] = rankings[candidate_col].astype(str).str.casefold().eq("true")
    selected_tab, impact_tab, region_tab, validation_tab, sensitivity_tab = st.tabs(
        [
            "Locations",
            "Existing-store impact",
            "Regional comparison",
            "Validation",
            "Sensitivity",
        ]
    )
    with selected_tab:
        map_candidates = rankings.loc[
            pd.to_numeric(rankings.get("latitude"), errors="coerce").between(-90, 90)
            & pd.to_numeric(rankings.get("longitude"), errors="coerce").between(-180, 180)
        ].copy()
        map_filters = st.columns(3)
        area_values = sorted(
            map_candidates.get("Area Type", pd.Series(dtype="string"))
            .dropna()
            .astype(str)
            .unique()
        )
        selected_areas = map_filters[0].multiselect(
            "Candidate area",
            area_values,
            default=area_values,
        )
        map_status = map_filters[1].selectbox(
            "Candidate status", ["All", "Selected", "Unselected"]
        )
        rank_limit = map_filters[2].selectbox(
            "Candidate ranking",
            ["All candidates", "Top 100 standalone"],
        )
        if "Area Type" in map_candidates:
            map_candidates = map_candidates.loc[
                map_candidates["Area Type"].astype(str).isin(selected_areas)
            ]
        selected_mask = (
            map_candidates[candidate_col].astype(bool)
            if candidate_col in map_candidates
            else pd.Series(False, index=map_candidates.index)
        )
        if map_status == "Selected":
            map_candidates = map_candidates.loc[selected_mask]
        elif map_status == "Unselected":
            map_candidates = map_candidates.loc[~selected_mask]
        if rank_limit == "Top 100 standalone":
            map_candidates = map_candidates.loc[
                pd.to_numeric(
                    map_candidates["standalone_ranking"], errors="coerce"
                ).le(100)
            ]
        map_candidates["site_type"] = "Unselected candidate"
        standalone_rank = pd.to_numeric(
            map_candidates.get("standalone_ranking"), errors="coerce"
        )
        map_candidates.loc[
            standalone_rank.le(20), "site_type"
        ] = "High standalone potential (top 20)"
        downside_column = "largest_existing_store_sales_change_2027_mnt"
        if downside_column in map_candidates:
            downside = pd.to_numeric(map_candidates[downside_column], errors="coerce")
            negative_downside = downside.loc[downside.lt(0)]
            if len(negative_downside):
                downside_threshold = negative_downside.quantile(0.05)
                map_candidates.loc[
                    downside.le(downside_threshold), "site_type"
                ] = "Highest modeled downside (bottom 5%)"
        map_candidates.loc[selected_mask.reindex(map_candidates.index, fill_value=False), "site_type"] = (
            "Selected candidate"
        )
        st.caption(
            "Map layers distinguish selected locations, the top 20 standalone-ranked "
            "candidates, and the bottom 5% of candidates with negative modeled "
            "existing-store impact."
        )
        map_data = map_candidates.copy()
        existing_points = master.drop_duplicates("store_id").copy()
        if {"LATITUDE", "LONGITUDE"}.issubset(existing_points.columns):
            existing_points["latitude"] = pd.to_numeric(
                existing_points["LATITUDE"], errors="coerce"
            )
            existing_points["longitude"] = pd.to_numeric(
                existing_points["LONGITUDE"], errors="coerce"
            )
            existing_points = existing_points.loc[
                existing_points["latitude"].between(-90, 90)
                & existing_points["longitude"].between(-180, 180)
            ].copy()
            existing_points["site_type"] = "Existing CU store"
            existing_points["Potential Point"] = existing_points.get(
                "store_name", existing_points["store_id"]
            )
            existing_points["potential_store_id"] = existing_points["store_id"].astype(str)
            existing_points["predicted_APSD_sales"] = np.nan
            map_data = pd.concat(
                [map_data, existing_points[map_data.columns.intersection(existing_points.columns)]],
                ignore_index=True,
                sort=False,
            )
        if not map_data.empty:
            map_figure = px.scatter_map(
                map_data,
                lat="latitude",
                lon="longitude",
                color="site_type",
                hover_name="Potential Point" if "Potential Point" in map_data else "potential_store_id",
                hover_data=[
                    column
                    for column in (
                        "potential_store_id",
                        "Area Type",
                        "predicted_APSD_sales",
                        f"portfolio_rank_{target}",
                    )
                    if column in map_data
                ],
                zoom=4.2,
                center={"lat": 46.8, "lon": 103.8},
                map_style="open-street-map",
                height=620,
                title="Existing stores and 2027 candidate portfolio",
            )
            st.plotly_chart(map_figure, width="stretch")
        if not scenario_monthly.empty:
            chart = px.line(
                scenario_monthly,
                x="month",
                y=[
                    "new_store_sales_mnt",
                    "existing_sales_change_mnt",
                    "network_sales_change_mnt",
                ],
                title="2027 monthly sales change (MNT)",
                labels={"value": "Sales change (MNT)", "month": "Month", "variable": "Component"},
            )
            st.plotly_chart(chart, width="stretch")
        display_columns = [
            column
            for column in (
                "portfolio_rank_" + str(target),
                "potential_store_id",
                "Potential Point",
                "Area Type",
                "latitude",
                "longitude",
                "predicted_APSD_sales",
                "predicted_APSD_customers",
                "standalone_net_network_sales_2027_mnt_jan_open",
                "greedy_marginal_value_" + str(target) + "_mnt",
                candidate_col,
            )
            if column in rankings
        ]
        shown = rankings.loc[:, display_columns].sort_values(
            "greedy_marginal_value_" + str(target) + "_mnt",
            ascending=False,
            na_position="last",
        ) if "greedy_marginal_value_" + str(target) + "_mnt" in display_columns else rankings.loc[:, display_columns]
        st.dataframe(shown, width="stretch", hide_index=True)
        download_table(shown, "Download candidate ranking", f"candidate_ranking_{target}.csv")

    with impact_tab:
        scenario_impact = (
            existing_impact.loc[
                pd.to_numeric(
                    existing_impact["scenario_target"], errors="coerce"
                ).eq(target)
            ].sort_values("sales_change_2027_mnt")
            if "scenario_target" in existing_impact
            else pd.DataFrame()
        )
        impact_columns = [
            column
            for column in (
                "negative_impact_rank",
                "store_id",
                "store_name",
                "analysis_region",
                "mean_APSD_income_change_2027",
                "sales_change_2027_mnt",
                "sales_change_pct_2027",
                "mean_APSD_customers_change_2027",
                "customer_equivalent_change_2027",
            )
            if column in scenario_impact
        ]
        st.dataframe(
            scenario_impact.loc[:, impact_columns].head(100),
            width="stretch",
            hide_index=True,
        )
        download_table(
            scenario_impact.loc[:, impact_columns],
            "Download existing-store impact",
            f"existing_store_impact_{target}.csv",
        )

    with region_tab:
        if "scenario_target" in regional:
            regional_scenario = regional.loc[
                pd.to_numeric(regional["scenario_target"], errors="coerce").eq(
                    target
                )
            ]
            st.dataframe(regional_scenario, width="stretch", hide_index=True)
        else:
            st.info("No regional comparison output was saved.")

    with validation_tab:
        if validation.empty:
            st.info("No demand validation report was saved.")
        else:
            st.dataframe(validation, width="stretch", hide_index=True)
        st.warning(
            "The saved regional event-study diagnostics report significant pre-trends. "
            "Cannibalization estimates are therefore shown as conditional scenario "
            "associations rather than cleanly identified causal effects."
        )

    with sensitivity_tab:
        if "scenario_target" in sensitivity:
            scenario_sensitivity = sensitivity.loc[
                pd.to_numeric(sensitivity["scenario_target"], errors="coerce").eq(
                    target
                )
            ]
            st.dataframe(
                scenario_sensitivity,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "Sensitivity outputs are not available. Re-run "
                "`python expansion_simulation.py` to generate them."
            )
        if "scenario_target" in robustness:
            scenario_robustness = robustness.loc[
                pd.to_numeric(robustness["scenario_target"], errors="coerce").eq(
                    target
                )
            ].sort_values(
                ["selection_frequency", "selected_in_central_case"],
                ascending=[False, False],
            )
            st.subheader("Portfolio selection robustness")
            st.dataframe(
                scenario_robustness.merge(
                    rankings[
                        [
                            "potential_store_id",
                            "Potential Point",
                            "predicted_APSD_sales",
                            "predicted_APSD_customers",
                        ]
                    ].drop_duplicates("potential_store_id"),
                    on="potential_store_id",
                    how="left",
                ),
                width="stretch",
                hide_index=True,
            )


@st.cache_data(show_spinner=False)
def read_cumulative_output(name: str, modified_ns: int) -> pd.DataFrame:
    path = BASE_DIR / "outputs" / "cannibalization_model4" / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def cumulative_cannibalization_page() -> None:
    """Dashboard page for saved regional cumulative APSD scenarios."""
    result_dir = BASE_DIR / "outputs" / "cannibalization_model4"

    def load(name: str) -> pd.DataFrame:
        path = result_dir / name
        return read_cumulative_output(name, path.stat().st_mtime_ns if path.is_file() else 0)

    stores = load("cumulative_cannibalization_scenario_store_results.csv")
    summary = load("cumulative_cannibalization_scenario_summary.csv")
    regional = load("cumulative_cannibalization_regional_summary.csv")
    contributions = load("cumulative_cannibalization_contributions.csv")
    candidates = load("cumulative_cannibalization_candidates.csv")
    sensitivity = load("cumulative_cannibalization_sensitivity_summary.csv")
    validation = load("cumulative_cannibalization_validation_report.csv")
    st.header("Cumulative Cannibalization Impact")
    if stores.empty or summary.empty:
        st.info("No saved cumulative outputs were found. Run python cumulative_cannibalization.py, then refresh.")
        return
    mode = st.radio('Coefficient calculation', ['negative_only', 'raw'],
                    format_func=lambda value: 'Negative-only cannibalization assumption' if value == 'negative_only' else 'Raw fixed regression estimates')
    st.caption('Exact supplied cumulative exposure coefficients. City Outskirts sales use the explicitly documented All Regions fallback. Raw customer estimates preserve +0.0042 in the 500–1,000 m band; negative-only caps each positive pair coefficient at zero.')
    for frame in (stores, summary, regional, contributions):
        if 'calculation_mode' in frame:
            frame.drop(frame.index[frame['calculation_mode'].ne(mode)], inplace=True)
    st.caption("Direct cannibalization effects are assumed to be zero beyond 1,000 meters. This is a modeling assumption, not an empirically established universal threshold.")
    st.caption("This is a baseline-APSD-weighted existing-network indicator, not an annual revenue, profit, demand, or ROI forecast.")
    scenarios = sorted(pd.to_numeric(summary["scenario_stores"], errors="coerce").dropna().astype(int).unique())
    scenario = st.selectbox("Scenario", scenarios, format_func=lambda value: f"{value} selected new-store candidates")
    row = summary.loc[pd.to_numeric(summary["scenario_stores"], errors="coerce").eq(scenario)].iloc[0]
    data = stores.loc[pd.to_numeric(stores["scenario_size"], errors="coerce").eq(scenario)].copy()
    st.info(f"Fixed baseline period: {row['baseline_month_or_period']}. It is the same for the 180- and 400-store comparison.")

    controls = st.columns(4)
    available_regions = sorted(data["region"].dropna().astype(str).unique())
    region = controls[0].selectbox("Region", ["All regions", *available_regions])
    search = controls[1].text_input("Store ID or name")
    sort_option = controls[2].selectbox("Sort", ["Percentage decline", "Absolute APSD decline", "Baseline APSD"])
    compare = controls[3].checkbox("Compare 180 vs 400")
    if region != "All regions":
        data = data.loc[data["region"].astype(str).eq(region)]
    if search:
        search = search.casefold()
        data = data.loc[
            data["existing_store_id"].astype(str).str.casefold().str.contains(search, na=False)
            | data["store_name"].astype(str).str.casefold().str.contains(search, na=False)
        ]
    sort_map = {
        "Percentage decline": ("cumulative_predicted_APSD_income_effect_pct", True),
        "Absolute APSD decline": ("cumulative_predicted_APSD_income_change", True),
        "Baseline APSD": ("baseline_APSD_income", False),
    }
    sort_column, sort_ascending = sort_map[sort_option]
    data = data.sort_values(sort_column, ascending=sort_ascending, na_position="last")

    cards = st.columns(4)
    cards[0].metric("Selected candidates", f"{int(row['selected_unique_candidates']):,}")
    cards[1].metric("APSD-weighted network effect", f"{row['APSD_weighted_network_effect_pct']:.2f}%")
    cards[2].metric("Existing stores affected", f"{int(row['existing_stores_with_nearby_opening']):,}")
    cards[3].metric("Affected stores", f"{row['pct_existing_stores_with_nearby_opening']:.1f}%")
    cards = st.columns(4)
    cards[0].metric("Declines >10%", f"{int(row['stores_decline_gt_10_pct']):,}")
    cards[1].metric("Declines >20%", f"{int(row['stores_decline_gt_20_pct']):,}")
    cards[2].metric("Median effect", f"{row['median_store_level_effect_pct']:.2f}%")
    worst = data.loc[data["cumulative_predicted_APSD_income_effect_pct"].idxmin(), "store_name"] if not data.empty else "N/A"
    cards[3].metric("Most affected in view", str(worst))

    overview, table_tab, detail_tab, map_tab, compare_tab, sensitivity_tab, validation_tab = st.tabs(
        ["Distribution & regions", "Existing-store table", "Store detail", "Map", "180 vs 400", "Sensitivity", "Validation"]
    )
    with overview:
        effect = pd.to_numeric(data["cumulative_predicted_APSD_income_effect_pct"], errors="coerce")
        data["effect_bucket"] = np.select(
            [effect.gt(0), effect.eq(0), effect.lt(0) & effect.gt(-5), effect.le(-5) & effect.gt(-10), effect.le(-10) & effect.gt(-20), effect.le(-20) & effect.gt(-30), effect.le(-30)],
            ["Positive effect", "0% effect", "0% to -5%", "-5% to -10%", "-10% to -20%", "-20% to -30%", "Below -30%"],
            default="Unavailable",
        )
        order = ["Positive effect", "0% effect", "0% to -5%", "-5% to -10%", "-10% to -20%", "-20% to -30%", "Below -30%", "Unavailable"]
        distribution = data["effect_bucket"].value_counts().reindex(order, fill_value=0).rename_axis("Effect bucket").reset_index(name="Existing stores")
        st.plotly_chart(px.bar(distribution, x="Effect bucket", y="Existing stores", title="Cumulative predicted APSD income-effect distribution"), width="stretch")
        regional_data = regional.loc[pd.to_numeric(regional["scenario_stores"], errors="coerce").eq(scenario)]
        st.caption("Regional weighted effects use baseline APSD as weights; mean and median are simple store-level statistics.")
        st.dataframe(regional_data, width="stretch", hide_index=True)
        download_table(regional_data, "Download regional summary", f"cumulative_regional_{scenario}.csv")

    columns = [
        "existing_store_id", "store_name", "region", "baseline_month_or_period",
        "baseline_APSD_income", "baseline_APSD_customers", "number_within_100m",
        "number_within_250m", "number_within_500m", "number_within_1000m",
        "number_of_contributing_candidate_stores", "cumulative_predicted_APSD_income_effect_pct",
        "cumulative_predicted_APSD_income_change", "predicted_scenario_APSD_income",
        "cumulative_predicted_APSD_customers_effect_pct", "predicted_scenario_APSD_customers",
        "rank_by_percentage_decline", "rank_by_absolute_APSD_income_decline",
        "main_contributing_candidate_locations", "model_estimate_status",
    ]
    columns = [column for column in columns if column in data]
    with table_tab:
        st.caption("Zero-exposure stores with a valid baseline remain in the table and network denominator; missing data are separately flagged.")
        view = data[columns].style.background_gradient(
            subset=["cumulative_predicted_APSD_income_effect_pct"], cmap="RdYlGn", vmin=-30, vmax=10
        )
        st.dataframe(view, width="stretch", hide_index=True)
        download_table(data[columns], "Download filtered store-level results", f"cumulative_stores_{scenario}.csv")

    with detail_tab:
        choices = {
            f"{record.existing_store_id} — {record.store_name}": record.existing_store_id
            for record in data[["existing_store_id", "store_name"]].drop_duplicates().itertuples()
        }
        if not choices:
            st.info("No stores match the filters.")
        else:
            store_id = choices[st.selectbox("Existing CU store", list(choices))]
            detail = data.loc[data["existing_store_id"].astype(str).eq(str(store_id))].iloc[0]
            st.json({
                "store": detail["store_name"], "region": detail["region"],
                "baseline_APSD_income": detail["baseline_APSD_income"],
                "cumulative_effect_pct": detail["cumulative_predicted_APSD_income_effect_pct"],
                "scenario_APSD_income": detail["predicted_scenario_APSD_income"],
                "cumulative_log_effect": detail["cumulative_income_log_effect"],
                "model_status": detail["model_estimate_status"],
            })
            detail_pairs = contributions.loc[
                pd.to_numeric(contributions["scenario_size"], errors="coerce").eq(scenario)
                & contributions["existing_store_id"].astype(str).eq(str(store_id))
            ].sort_values("regional_income_log_coefficient")
            if detail_pairs.empty:
                st.info("No selected candidate is within 1 km for this existing store.")
            else:
                st.plotly_chart(px.bar(detail_pairs, x="candidate_name", y="direct_APSD_income_effect_pct", color="distance_band", title="Every selected pair-level contribution"), width="stretch")
                st.dataframe(detail_pairs[[
                    column for column in ["candidate_id", "candidate_name", "selection_order", "distance_m", "distance_band", "regional_income_log_coefficient", "direct_APSD_income_effect_pct", "income_model_status"] if column in detail_pairs
                ]], width="stretch", hide_index=True)

    with map_tab:
        mapped = data.loc[
            pd.to_numeric(data["latitude"], errors="coerce").between(-90, 90)
            & pd.to_numeric(data["longitude"], errors="coerce").between(-180, 180)
        ]
        selected_flag = candidates.get(f"in_scenario_{scenario}", pd.Series(False, index=candidates.index))
        selected_candidates = candidates.loc[selected_flag.astype(str).str.casefold().eq("true") | selected_flag.eq(True)]
        if not mapped.empty:
            figure = px.scatter_map(mapped, lat="latitude", lon="longitude", color="cumulative_predicted_APSD_income_effect_pct", color_continuous_scale="RdYlGn", hover_name="store_name", hover_data=["existing_store_id", "region", "number_of_contributing_candidate_stores"], zoom=9, center={"lat": mapped["latitude"].mean(), "lon": mapped["longitude"].mean()}, map_style="open-street-map", height=650, title="Existing stores colored by modeled cumulative effect")
            if not selected_candidates.empty:
                candidates_figure = px.scatter_map(selected_candidates, lat="latitude", lon="longitude", hover_name="candidate_name", hover_data=["candidate_id", "selection_order"], map_style="open-street-map")
                for trace in candidates_figure.data:
                    trace.name = "Selected candidate"
                    trace.marker.color = "#1f77b4"
                    trace.marker.size = 11
                    figure.add_trace(trace)
            st.plotly_chart(figure, width="stretch")
        st.caption("No pair farther than 1,000 m is represented as a direct modeled effect.")

    with compare_tab:
        st.dataframe(summary.loc[pd.to_numeric(summary["scenario_stores"], errors="coerce").isin([180, 400])], width="stretch", hide_index=True)
        if compare:
            comparison_data = stores.loc[pd.to_numeric(stores["scenario_size"], errors="coerce").isin([180, 400])]
            st.plotly_chart(px.box(comparison_data, x="scenario_size", y="cumulative_predicted_APSD_income_effect_pct", points=False, title="Store-level cumulative effect distribution"), width="stretch")

    with sensitivity_tab:
        st.caption("The 1 km scenario is primary. A 1.5 km estimate is not fabricated when the saved model lacks a 1,000-1,500 m coefficient.")
        st.dataframe(sensitivity, width="stretch", hide_index=True)

    with validation_tab:
        st.dataframe(validation, width="stretch", hide_index=True)
        st.caption("The regional estimates are associational model outputs, not guaranteed causal effects or forecasts.")


def effect_chart(data: pd.DataFrame, x: str, color: str | None = None) -> None:

    if data.empty:

        st.info("No estimates are available for this selection.")

        return

    fig = go.Figure()

    data = data.copy()

    for column in ("observations", "stores", "exposed_stores"):

        if column not in data:

            data[column] = pd.NA

    groups = [(None, data)] if color is None else list(data.groupby(color, dropna=False))

    for group_name, group in groups:

        group = group.sort_values(x)

        fig.add_trace(

            go.Scatter(

                x=group[x],

                y=group["percent_effect"],

                error_y={

                    "type": "data",

                    "symmetric": False,

                    "array": group["percent_ci_95_upper"] - group["percent_effect"],

                    "arrayminus": group["percent_effect"] - group["percent_ci_95_lower"],

                },

                mode="lines+markers",

                name=str(group_name) if group_name is not None else "Estimate",

                customdata=group[

                    ["observations", "stores", "exposed_stores"]

                ].to_numpy(),

                hovertemplate=(

                    "%{x}<br>Association: %{y:.1f}%"

                    "<br>95% CI upper/lower shown"

                    "<br>Store-month observations: %{customdata[0]}"

                    "<br>Stores: %{customdata[1]}"

                    "<br>Exposed stores: %{customdata[2]}<extra></extra>"

                ),

            )

        )

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(

        yaxis_title="Estimated associated change (%)",

        xaxis_title=x.replace("_", " ").title(),

        hovermode="x unified",

        legend_title_text=color.replace("_", " ").title() if color else "",

    )

    st.plotly_chart(fig, width="stretch")





def add_common_warning(pretrends: pd.DataFrame, region: str, outcome: str) -> None:

    selected = pretrends.loc[

        (pretrends["region"] == region)

        & (pretrends["outcome"] == outcome)

        & (pretrends["model_type"] == "binned_event_study")

    ] if not pretrends.empty else pd.DataFrame()

    if selected.empty:

        st.warning(

            "Pre-trend diagnostic is unavailable. Interpret these estimates as "

            "associations, not established causal effects."

        )

        return

    p_value = pd.to_numeric(

        pd.Series([selected.iloc[0].get("joint_p_value")]), errors="coerce"

    ).iloc[0]

    if pd.notna(p_value) and p_value < 0.05:

        st.error(

            f"Pre-opening leads are jointly significant (p={p_value:.3g}). "

            "double check needed."

        )

    elif pd.notna(p_value):

        st.info(

            f"No joint pre-trend rejection at 5% (p={p_value:.3g}); this does not "

            "by itself establish causal identification."

        )

    else:

        st.warning(

            "The joint pre-trend test could not be estimated. Treat results as "

            "associations."

        )





def _format_mnt(value: float) -> str:

    return f"MNT {value:,.0f}"





def _format_beta_percent(beta: float) -> str:

    try:

        return f"{100 * math.expm1(beta):.1f}%"

    except OverflowError:

        return "Outside numeric range"





def _capacity_cell(result: CapacityResult) -> str:

    if result.status == "Finite capacity":

        return f"{int(result.maximum_openings or 0):,}"

    return result.status





def _show_capacity_diagnostics(

    estimates: dict[str, CoefficientEstimate],

    region: str,

    pretrends: pd.DataFrame,

) -> None:

    diagnostic = pretrends.loc[

        (pretrends["region"] == region)

        & (pretrends["outcome"] == "sales")

        & (pretrends["model_type"] == "binned_event_study")

    ] if not pretrends.empty else pd.DataFrame()

    if diagnostic.empty:

        st.warning(

            "The regional sales pre-trend diagnostic is unavailable; interpret "

            "the estimates as associations."

        )

    else:

        p_value = pd.to_numeric(

            pd.Series([diagnostic.iloc[0].get("joint_p_value")]), errors="coerce"

        ).iloc[0]

        if pd.notna(p_value) and p_value < 0.05:

            st.warning(

                f"The regional binned event-study pre-trend test rejects at 5% "

                f"(p={p_value:.3g}); causal interpretation is not supported."

            )

        elif pd.notna(p_value):

            st.info(

                f"No regional pre-trend rejection at 5% (p={p_value:.3g}); "

                "this does not by itself establish causal identification."

            )

        else:

            st.warning("The regional pre-trend test could not be estimated.")



    limited_support = [

        f"{dict(DISTANCE_BANDS)[band]} ({estimate.exposed_stores} exposed stores)"

        for band, estimate in estimates.items()

        if estimate.beta is not None

        and estimate.exposed_stores is not None

        and estimate.exposed_stores < 10

    ]

    if limited_support:

        st.warning(

            "Very few stores contribute exposure support for: "

            + "; ".join(limited_support)

            + ". Treat these estimates as especially uncertain."

        )

    missing_estimates = [

        dict(DISTANCE_BANDS)[band]

        for band, estimate in estimates.items()

        if estimate.beta is None

    ]

    if missing_estimates:

        st.warning(

            "No valid cumulative sales coefficient is available for: "

            + ", ".join(missing_estimates)

            + ". Those bands are not treated as zero effect."

        )

    uncertain_intervals = [

        dict(DISTANCE_BANDS)[band]

        for band, estimate in estimates.items()

        if estimate.confidence_lower is not None

        and estimate.confidence_upper is not None

        and (

            estimate.confidence_lower <= 0 <= estimate.confidence_upper

            or estimate.confidence_upper - estimate.confidence_lower

            > 2 * abs(estimate.beta or 0)

        )

    ]

    if uncertain_intervals:

        st.warning(

            "The 95% coefficient intervals are imprecise (include zero or are "

            "wider than twice the absolute central estimate) for: "

            + ", ".join(uncertain_intervals)

            + "."

        )



    rows = []

    for band, label in DISTANCE_BANDS:

        estimate = estimates[band]

        rows.append(

            {

                "Distance band": label,

                "Log coefficient": (

                    f"{estimate.beta:.4f}"

                    if estimate.beta is not None

                    else "Not estimable"

                ),

                "Percent effect per opening": (

                    _format_beta_percent(estimate.beta)

                    if estimate.beta is not None

                    else "Not estimable"

                ),

                "95% CI (percent)": (

                    f"{100 * math.expm1(estimate.confidence_lower):.1f}% to "

                    f"{100 * math.expm1(estimate.confidence_upper):.1f}%"

                    if estimate.confidence_lower is not None

                    and estimate.confidence_upper is not None

                    else "Unavailable"

                ),

                "Exposed stores": (

                    f"{estimate.exposed_stores:,}"

                    if estimate.exposed_stores is not None

                    else "Unavailable"

                ),

                "Observations": (

                    f"{estimate.observations:,}"

                    if estimate.observations is not None

                    else "Unavailable"

                ),

                "Status": estimate.issue or "Estimated",

            }

        )

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)





def store_opening_capacity_page(

    sales_coefficients: pd.DataFrame,

    pretrends: pd.DataFrame,

) -> None:

    st.header("Store Opening Capacity Simulator")

    st.caption(

        "The simulator estimates how many additional nearby CU openings are "

        "compatible with a user-defined minimum APSD sales threshold, using "

        "historical regional cannibalization regression estimates. Results are "

        "conditional scenarios, not guaranteed forecasts or definitive "

        "site-opening recommendations."

    )

    st.markdown(

        "The estimates describe historical associations and may not identify "

        "purely causal effects. Future openings may differ from historical ones; "

        "the calculations assume estimated effects apply to the proposed "

        "combination of openings. New stores may affect one another. Actual "

        "capacity also depends on demand, market size, operating costs, new-store "

        "sales, and geographic availability. This is not a count of stores CU can "

        "physically or commercially open."

    )



    with st.form("store_opening_capacity_form"):

        baseline = st.number_input(

            "Current APSD Sales (MNT per active sales day)",

            min_value=0.01,

            value=4_000_000.0,

            step=100_000.0,

            format="%.0f",

        )

        minimum = st.number_input(

            "Minimum Required APSD Sales (MNT per active sales day)",

            min_value=0.01,

            value=3_000_000.0,

            step=100_000.0,

            format="%.0f",

        )

        region = st.selectbox("Existing-store region", list(REGIONS))

        mode = st.radio(

            "Simulation mode",

            [

                "Single-distance-band simulation",

                "Mixed-distance simulation",

                "Maximum Combined Openings",

            ],

        )

        selected_band = st.selectbox(

            "Distance band for single-band simulation and sensitivity chart",

            list(DISTANCE_LABELS),

            format_func=lambda band: DISTANCE_LABELS[band],

        )

        proposed: dict[str, int] = {}

        upper_bounds: dict[str, int] = {}

        if mode == "Mixed-distance simulation":

            st.markdown("**Proposed additional openings by distance band**")

            cols = st.columns(4)

            for col, (band, label) in zip(cols, DISTANCE_BANDS):

                proposed[band] = col.number_input(

                    f"{label}",

                    min_value=0,

                    value=0,

                    step=1,

                    key=f"mixed_openings_{band}",

                )

        elif mode == "Maximum Combined Openings":

            st.markdown(

                "**Maximum candidate openings by distance band** "

                "(each bound is capped at 25 to keep exhaustive search bounded)"

            )

            cols = st.columns(4)

            for col, (band, label) in zip(cols, DISTANCE_BANDS):

                upper_bounds[band] = col.number_input(

                    f"{label} upper bound",

                    min_value=0,

                    max_value=25,

                    value=10,

                    step=1,

                    key=f"opening_bound_{band}",

                )

        submitted = st.form_submit_button("Run Simulation")



    estimates = regional_sales_estimates(sales_coefficients, region)

    with st.expander(f"Selected region estimates and diagnostics — {region}", expanded=True):

        _show_capacity_diagnostics(estimates, region, pretrends)



    if submitted:

        st.session_state.pop("store_opening_capacity_result", None)

        try:

            validate_sales_inputs(float(baseline), float(minimum))

            betas = {band: estimate.beta for band, estimate in estimates.items()}

            if mode == "Single-distance-band simulation":

                capacity = maximum_openings(

                    float(baseline), float(minimum), estimates[selected_band].beta

                )

                count = (

                    capacity.maximum_openings

                    if capacity.status == "Finite capacity"

                    else (1 if estimates[selected_band].beta is not None else 0)

                )

                counts = {selected_band: int(count or 0)}

                scenario = (

                    simulate_mixed_scenario(

                        float(baseline), float(minimum), counts, betas

                    )

                    if estimates[selected_band].beta is not None

                    else None

                )

                allocation = None

            elif mode == "Mixed-distance simulation":

                capacity = None

                counts = proposed.copy()

                scenario = simulate_mixed_scenario(

                    float(baseline), float(minimum), counts, betas

                )

                allocation = None

            else:

                capacity = None

                allocation, scenario = optimize_openings(

                    float(baseline), float(minimum), upper_bounds, betas

                )

                counts = allocation.copy()

            st.session_state["store_opening_capacity_result"] = {

                "baseline": float(baseline),

                "minimum": float(minimum),

                "region": region,

                "mode": mode,

                "selected_band": selected_band,

                "counts": counts,

                "capacity": capacity,

                "scenario": scenario,

                "allocation": allocation,

            }

        except (ValueError, ArithmeticError) as error:

            st.error(str(error))



    result = st.session_state.get("store_opening_capacity_result")

    if result is None:

        return



    baseline = result["baseline"]

    minimum = result["minimum"]

    result_region = result["region"]

    result_estimates = regional_sales_estimates(sales_coefficients, result_region)

    betas = {band: estimate.beta for band, estimate in result_estimates.items()}

    scenario: ScenarioResult | None = result["scenario"]

    counts: dict[str, int] = result["counts"]

    capacity: CapacityResult | None = result["capacity"]



    st.subheader("Simulation results")

    if result["mode"] == "Single-distance-band simulation":

        label = DISTANCE_LABELS[result["selected_band"]]

        st.markdown(f"**Single-band scenario — all openings within {label}.**")

        if capacity is not None:

            if capacity.status == "Finite capacity":

                st.success(

                    f"Estimated maximum: **{capacity.maximum_openings:,}** "

                    f"additional opening(s) in this single band."

                )

                st.caption(

                    f"Boundary check: APSD at the maximum is "

                    f"{_format_mnt(capacity.predicted_at_maximum or 0)}; at one "

                    f"additional opening it is "

                    f"{_format_mnt(capacity.predicted_at_next or 0)}."

                )

            elif capacity.status == "No finite maximum under this model":

                st.info(capacity.issue)

            else:

                st.warning(f"{capacity.status}: {capacity.issue}")

    elif result["mode"] == "Maximum Combined Openings":

        st.markdown(

            "**Model- and bound-constrained allocation.** This allocation is "

            "optimal only for these coefficient estimates and user-defined bounds; "

            "it is not an optimal real-world site plan."

        )

        st.dataframe(

            pd.DataFrame(

                [

                    {

                        "Distance band": DISTANCE_LABELS[band],

                        "Selected openings": result["allocation"][band],

                    }

                    for band, _ in DISTANCE_BANDS

                ]

            ),

            width="stretch",

            hide_index=True,

        )



    if scenario is not None:

        metric_cols = st.columns(4)

        metric_cols[0].metric("Total additional openings", f"{scenario.total_openings:,}")

        metric_cols[1].metric("Predicted APSD sales", _format_mnt(scenario.predicted_apsd))

        metric_cols[2].metric(

            "Predicted APSD change",

            _format_mnt(scenario.absolute_change),

            f"{scenario.percentage_change:+.2f}%",

        )

        metric_cols[3].metric(

            "Headroom above minimum",

            _format_mnt(scenario.headroom),

            "Constraint satisfied" if scenario.constraint_satisfied else "Below minimum",

        )

        if scenario.constraint_satisfied:

            st.success("The predicted APSD meets the specified minimum.")

        else:

            st.error("The predicted APSD is below the specified minimum.")

    else:

        st.warning(

            "A predicted APSD scenario is unavailable because the selected regional "

            "coefficient is not estimable."

        )



    show_conservative = st.checkbox(

        "Show conservative 95% CI sensitivity",

        value=False,

        help=(

            "Uses the lower endpoint of each available pointwise 95% coefficient "

            "interval. This is a sensitivity case, not a guaranteed worst case."

        ),

        key="show_capacity_conservative",

    )

    if show_conservative and scenario is not None:

        active_bands = [band for band, count in counts.items() if count > 0]

        missing_intervals = [

            band

            for band in active_bands

            if result_estimates[band].confidence_lower is None

        ]

        if missing_intervals:

            st.warning(

                "Conservative sensitivity is unavailable because valid 95% "

                "confidence intervals are missing for: "

                + ", ".join(DISTANCE_LABELS[band] for band in missing_intervals)

                + "."

            )

        else:

            conservative_betas = {

                band: (

                    result_estimates[band].confidence_lower

                    if result_estimates[band].confidence_lower is not None

                    else estimate.beta

                )

                for band, estimate in result_estimates.items()

            }

            conservative = simulate_mixed_scenario(

                baseline, minimum, counts, conservative_betas

            )

            st.info(

                f"Conservative sensitivity APSD: "

                f"{_format_mnt(conservative.predicted_apsd)} "

                f"({conservative.percentage_change:+.2f}% vs baseline); "

                f"minimum met: {'Yes' if conservative.constraint_satisfied else 'No'}. "

                "This uses pointwise lower confidence bounds and is not a guaranteed "

                "worst case or a joint 95% interval."

            )



    comparison_rows: list[dict[str, object]] = []

    comparison_plot_rows: list[dict[str, object]] = []

    for compare_region in REGIONS:

        compare_estimates = regional_sales_estimates(sales_coefficients, compare_region)

        row: dict[str, object] = {

            "Region": compare_region,

            "Current APSD sales": _format_mnt(baseline),

            "Minimum required APSD sales": _format_mnt(minimum),

        }

        for band, label in DISTANCE_BANDS:

            estimate = compare_estimates[band]

            result_for_band = maximum_openings(baseline, minimum, estimate.beta)

            row[f"Maximum openings, {label}"] = _capacity_cell(result_for_band)

            if result_for_band.status == "Finite capacity":

                comparison_plot_rows.append(

                    {

                        "Region": compare_region,

                        "Distance band": label,

                        "Maximum additional openings": result_for_band.maximum_openings,

                    }

                )

        comparison_rows.append(row)



    st.subheader("Regional comparison — single-band capacities")

    st.caption(

        "Each cell is a separate single-distance-band scenario. The maxima for "

        "different bands must not be added or assumed achievable simultaneously."

    )

    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    if comparison_plot_rows:

        capacity_chart = px.bar(

            pd.DataFrame(comparison_plot_rows),

            x="Region",

            y="Maximum additional openings",

            color="Distance band",

            barmode="group",

            title="Maximum additional openings by region and distance band",

            category_orders={"Region": list(REGIONS)},

        )

        capacity_chart.update_layout(yaxis_title="Additional openings (count)")

        st.plotly_chart(capacity_chart, width="stretch")

    else:

        st.info("No finite regional capacities are available to plot.")



    st.subheader("APSD sales scenario")

    predicted_value = scenario.predicted_apsd if scenario is not None else None

    apsd_chart = go.Figure()

    apsd_chart.add_trace(

        go.Bar(

            x=["Current APSD", "Minimum required APSD", "Predicted APSD"],

            y=[baseline, minimum, predicted_value],

            marker_color=["#4C78A8", "#F58518", "#54A24B"],

            hovertemplate="%{x}<br>MNT %{y:,.0f}<extra></extra>",

        )

    )

    apsd_chart.update_layout(

        yaxis_title="APSD sales (MNT per active sales day)",

        showlegend=False,

    )

    st.plotly_chart(apsd_chart, width="stretch")



    st.subheader("Proposed opening composition")

    composition = pd.DataFrame(

        [

            {

                "Distance band": DISTANCE_LABELS[band],

                "Proposed openings": counts.get(band, 0),

            }

            for band, _ in DISTANCE_BANDS

        ]

    )

    composition_chart = px.bar(

        composition,

        x="Distance band",

        y="Proposed openings",

        title="Scenario composition by distance band",

    )

    composition_chart.update_layout(yaxis_title="Proposed additional openings (count)")

    st.plotly_chart(composition_chart, width="stretch")



    st.subheader("APSD sales sensitivity — selected distance band")

    sensitivity_beta = result_estimates[result["selected_band"]].beta

    if sensitivity_beta is None:

        st.warning(

            "Sensitivity is unavailable because the selected distance-band "

            "coefficient is not estimable for this region."

        )

    else:

        selected_capacity = maximum_openings(

            baseline, minimum, sensitivity_beta

        )

        end_count = (

            min(max(selected_capacity.maximum_openings or 0, 10) + 1, 50)

            if selected_capacity.status == "Finite capacity"

            else 10

        )

        try:

            sensitivity = pd.DataFrame(

                {

                    "Additional openings": list(range(end_count + 1)),

                    "Predicted APSD sales (MNT)": [

                        predict_apsd(

                            baseline,

                            {result["selected_band"]: count},

                            {result["selected_band"]: sensitivity_beta},

                        )

                        for count in range(end_count + 1)

                    ],

                }

            )

        except ValueError as error:

            st.warning(f"APSD sensitivity cannot be plotted: {error}")

        else:

            sensitivity_chart = px.line(

                sensitivity,

                x="Additional openings",

                y="Predicted APSD sales (MNT)",

                markers=True,

                title=f"Single-band sensitivity — {DISTANCE_LABELS[result['selected_band']]}",

            )

            sensitivity_chart.add_hline(

                y=minimum,

                line_dash="dash",

                line_color="#E45756",

                annotation_text="Minimum required APSD",

            )

            sensitivity_chart.update_yaxes(tickprefix="MNT ", tickformat=",.0f")

            st.plotly_chart(sensitivity_chart, width="stretch")

            if selected_capacity.status == "Finite capacity" and selected_capacity.maximum_openings is not None and selected_capacity.maximum_openings >= 50:

                st.caption("The sensitivity chart is capped at 50 openings for readability.")



    st.markdown(

        "**Interpretation limitations:** estimates reflect historical associations "

        "and can be affected by local demand trends. The simulated combination "

        "assumes additive log-scale coefficients, while actual stores may interact. "

        "Pointwise confidence bounds cannot be combined into a joint interval "

        "because coefficient covariance is unavailable."

    )


def store_level_exploration_page(
    panel: pd.DataFrame,
    pair_source: pd.DataFrame,
    master: pd.DataFrame,
    outcome: str,
) -> None:
    st.header("Store-level exploration")
    panel_source = panel.copy()
    panel_source["_existing_key"] = normalize_store_id(
        panel_source["existing_store_id"]
    )
    name_lookup = pd.DataFrame(columns=["_existing_key", "store_name"])
    if "store_name" in panel_source:
        name_lookup = panel_source[
            ["_existing_key", "store_name"]
        ].copy()
    if not master.empty and {"store_id", "store_name"}.issubset(master.columns):
        master_names = master[["store_id", "store_name"]].copy()
        master_names["_existing_key"] = normalize_store_id(
            master_names["store_id"]
        )
        master_names = master_names[["_existing_key", "store_name"]]
        name_lookup = (
            pd.concat([name_lookup, master_names], ignore_index=True)
            if not name_lookup.empty
            else master_names.copy()
        )
    name_lookup["store_name"] = name_lookup["store_name"].astype("string").str.strip()
    name_lookup["store_name"] = name_lookup["store_name"].mask(
        name_lookup["store_name"].isin(["", "nan", "None", "<NA>"]), pd.NA
    )
    name_lookup = (
        name_lookup.dropna(subset=["_existing_key", "store_name"])
        .drop_duplicates("_existing_key")
    )
    store_options = (
        panel_source[["_existing_key", "existing_store_id"]]
        .dropna(subset=["_existing_key"])
        .drop_duplicates("_existing_key")
        .merge(name_lookup, on="_existing_key", how="left", validate="one_to_one")
    )
    store_options["display_name"] = store_options["store_name"].fillna("Unnamed store")
    duplicate_names = store_options["display_name"].duplicated(keep=False)
    store_options.loc[duplicate_names, "display_name"] = (
        store_options.loc[duplicate_names, "display_name"]
        + " (ID: "
        + store_options.loc[duplicate_names, "_existing_key"]
        + ")"
    )
    store_options = store_options.sort_values(
        "display_name", key=lambda values: values.str.lower()
    )
    if store_options.empty:
        st.warning("No existing stores are available in the analysis panel.")
        return
    store_lookup = dict(
        zip(store_options["display_name"], store_options["_existing_key"])
    )
    selected_name = st.selectbox(
        "Select store",
        store_options["display_name"].tolist(),
        help="The selected store name is mapped to its internal store ID.",
        key="store_exploration_name",
    )
    selected_key = store_lookup[selected_name]
    store_panel = panel_source.loc[
        panel_source["_existing_key"] == selected_key
    ].sort_values("SalesMonth").copy()
    if store_panel.empty:
        st.warning("No panel observations were found for this store.")
        return
    selected_id = store_panel["existing_store_id"].iloc[0]
    store_name = selected_name
    region = (
        str(store_panel["analysis_region"].dropna().iloc[0])
        if store_panel["analysis_region"].notna().any()
        else "Unmapped"
    )
    opening_date = (
        _parse_dashboard_dates(store_panel["ost_dt"]).dropna().iloc[0]
        if store_panel["ost_dt"].notna().any()
        else pd.NaT
    )
    closing_date = (
        _parse_dashboard_dates(store_panel["cls_dt"]).dropna().iloc[-1]
        if store_panel["cls_dt"].notna().any()
        else pd.NaT
    )
    if pair_source.empty:
        store_events = pd.DataFrame()
        all_observations = pd.DataFrame()
        band_order: list[str] = []
    else:
        try:
            all_events, all_observations, _, band_order = prepare_dashboard_openings(
                panel, pair_source, master
            )
        except ValueError as error:
            st.error(f"Cannot build store opening events: {error}")
            return
        store_events = all_events.loc[
            all_events["_existing_key"] == selected_key
        ].copy()
        all_observations = all_observations.loc[
            all_observations["_existing_key"] == selected_key
        ].copy()

    store_event_history = store_events.copy()
    if "new_store_name" in store_events:
        store_events["new_store_name"] = (
            store_events["new_store_name"]
            .astype("string")
            .str.strip()
            .mask(lambda names: names.eq(""), pd.NA)
            .fillna("Unnamed new store")
        )
        store_event_history["new_store_name"] = store_events["new_store_name"]

    st.write(f"**Store:** {store_name}")
    st.write(f"**Store ID:** {selected_id}")
    st.write(f"**Region:** {region}")
    st.write(
        f"**Store opening date:** "
        f"{opening_date:%Y-%m-%d}" if pd.notna(opening_date) else
        "**Store opening date:** unavailable"
    )
    st.write(
        f"**Store closing date:** "
        f"{closing_date:%Y-%m-%d}" if pd.notna(closing_date) else
        "**Store closing date:** open or unavailable"
    )
    outcome_column = "APSD_income" if outcome == "sales" else "APSD_customers"
    outcome_title = _metric_label(outcome_column)
    latest_sales = (
        store_panel.dropna(subset=["APSD_income"])
        .sort_values("SalesMonth")
        .tail(1)
    )
    latest_customers = (
        store_panel.dropna(subset=["APSD_customers"])
        .sort_values("SalesMonth")
        .tail(1)
    )

    if not store_events.empty:
        store_events, _ = _filter_events_by_dates(
            store_events, f"store_opening_dates_{selected_key}"
        )
        selected_store_bands = st.multiselect(
            "Nearby opening distance bands",
            options=band_order,
            default=band_order,
            key=f"store_distance_bands_{selected_key}",
        )
        selected_store_events = store_events.loc[
            store_events["distance_band"].isin(selected_store_bands)
        ].copy()
    else:
        selected_store_events = pd.DataFrame()
        st.info("No qualifying nearby-store opening pair is recorded for this store.")

    metrics = st.columns(4)
    metrics[0].metric(
        "Latest APSD sales",
        f"{latest_sales['APSD_income'].iloc[0]:,.2f}"
        if not latest_sales.empty else "Unavailable",
        latest_sales["SalesMonth"].iloc[0].strftime("%Y-%m")
        if not latest_sales.empty else "No valid outcome",
    )
    metrics[1].metric(
        "Latest APSD customers",
        f"{latest_customers['APSD_customers'].iloc[0]:,.2f}"
        if not latest_customers.empty else "Unavailable",
        latest_customers["SalesMonth"].iloc[0].strftime("%Y-%m")
        if not latest_customers.empty else "No valid outcome",
    )
    metrics[2].metric(
        "Nearby stores opened in selected date/band filters",
        f"{selected_store_events['_opening_key'].nunique():,}"
        if not selected_store_events.empty else "0",
    )
    metrics[3].metric(
        "Qualifying opening-pair events",
        f"{len(selected_store_events):,}",
    )

    store_events_for_plot = selected_store_events.sort_values("new_opening_date")
    timeline = store_panel[
        ["SalesMonth", outcome_column]
        + (["Active_Days"] if "Active_Days" in store_panel else [])
    ].copy()
    timeline["SalesMonth"] = pd.to_datetime(timeline["SalesMonth"], errors="coerce")
    timeline = timeline.dropna(subset=["SalesMonth"]).set_index("SalesMonth")
    if not timeline.empty:
        full_months = pd.date_range(
            timeline.index.min(), timeline.index.max(), freq="MS"
        )
        timeline = timeline.reindex(full_months)
        timeline.index.name = "SalesMonth"
        timeline = timeline.reset_index()
    else:
        timeline = pd.DataFrame(columns=["SalesMonth", outcome_column])
    fig = go.Figure()
    if not timeline.empty:
        fig.add_trace(
            go.Scatter(
                x=timeline["SalesMonth"],
                y=timeline[outcome_column],
                mode="lines+markers",
                connectgaps=False,
                name=outcome_title,
                customdata=(
                    timeline[["Active_Days"]].to_numpy()
                    if "Active_Days" in timeline
                    else None
                ),
                hovertemplate=(
                    "%{x|%Y-%m}<br>Observed APSD: %{y:.2f}"
                    "<br>Active days: %{customdata[0]}<extra></extra>"
                    if "Active_Days" in timeline
                    else "%{x|%Y-%m}<br>Observed APSD: %{y:.2f}<extra></extra>"
                ),
            )
        )
    marker_count = 0
    for event in store_events_for_plot.itertuples(index=False):
        marker_count += 1
        fig.add_shape(
            type="line",
            x0=event.new_opening_date,
            x1=event.new_opening_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"dash": "dot", "color": "#D62728", "width": 1},
            opacity=0.55,
        )
        fig.add_annotation(
            x=event.new_opening_date,
            y=1,
            xref="x",
            yref="paper",
            text=f"{event.new_store_name} ({event.distance_band})",
            textangle=-90,
            yanchor="top",
            showarrow=False,
        )
    if marker_count:
        st.caption(
            f"{marker_count:,} nearby opening-date marker(s) are overlaid. "
            "Use the distance/date filters to reduce overlap."
        )
    fig.update_layout(
        title=f"Observed {outcome_title} — {store_name}",
        xaxis_title="Observed calendar month",
        yaxis_title=outcome_title,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    if not selected_store_events.empty:
        event_table = selected_store_events.copy()
        event_table["opening_event_order"] = event_table["event_order"].map(
            _event_order_label
        )
        event_table["event_month_relative_to_opening"] = 0
        event_table["new_store_name"] = event_table["new_store_name"].fillna(
            "Unnamed new store"
        )
        table_columns = [
            column
            for column in (
                "new_store_name",
                "new_store_id",
                "new_opening_date",
                "distance_m",
                "distance_band",
                "opening_event_order",
                "previous_nearby_openings_in_pair_history",
                "event_month_relative_to_opening",
                "event_id",
            )
            if column in event_table
        ]
        st.subheader("Nearby new-store opening events")
        st.dataframe(
            event_table[table_columns].sort_values("new_opening_date"),
            width="stretch",
            hide_index=True,
        )
        download_table(
            event_table[table_columns],
            "Download nearby opening events",
            f"store_{selected_key}_nearby_openings.csv",
        )

        event_options: dict[str, str] = {}
        for event in selected_store_events.itertuples(index=False):
            event_options[
                (
                    f"{event.new_store_name or 'Unnamed new store'} "
                    f"(ID {event.new_store_id}) — {event.new_opening_date:%Y-%m-%d} — "
                    f"{event.distance_m:,.0f} m, {event.distance_band} — "
                    f"{_event_order_label(event.event_order)}"
                )
            ] = str(event.event_id)
        event_labels = list(event_options)
        selected_event_label = st.selectbox(
            "Opening event for the store-specific event study",
            event_labels,
            key=f"store_selected_event_{selected_key}",
        )
        selected_event_id = event_options[selected_event_label]
        selected_event = selected_store_events.loc[
            selected_store_events["event_id"] == selected_event_id
        ].iloc[0]
        event_window = st.slider(
            "Months shown on each side of this opening",
            min_value=1,
            max_value=12,
            value=12,
            key=f"store_event_window_{selected_key}",
        )
        event_month_index = int(selected_event["opening_month_index"])
        month_index = _month_index(
            pd.to_datetime(store_panel["SalesMonth"], errors="coerce")
        )
        store_observations = store_panel.copy()
        store_observations["_calendar_month_index"] = month_index
        event_months = pd.DataFrame(
            {
                "event_time_month": range(-event_window, event_window + 1),
            }
        )
        event_months["SalesMonth"] = pd.to_datetime(
            (
                event_month_index + event_months["event_time_month"]
            ).map(lambda value: f"{(value - 1) // 12:04d}-{(value - 1) % 12 + 1:02d}-01"),
            errors="coerce",
        )
        event_plot = event_months.merge(
            store_observations[["SalesMonth", outcome_column] + (
                ["Active_Days"] if "Active_Days" in store_observations else []
            )],
            on="SalesMonth",
            how="left",
            validate="one_to_one",
        )
        overlapping = store_event_history.loc[
            store_event_history["event_id"].ne(selected_event_id)
            & (
                (
                    store_event_history["opening_month_index"] - event_month_index
                ).between(-event_window, event_window)
            )
        ] if not store_event_history.empty else pd.DataFrame()
        st.subheader("Selected-opening event study")
        st.write(
            f"**New store:** {selected_event['new_store_name'] or 'Unnamed new store'} "
            f"(ID {selected_event['new_store_id']}); "
            f"**opened:** {selected_event['new_opening_date']:%Y-%m-%d}; "
            f"**distance:** {selected_event['distance_m']:,.0f} m "
            f"({selected_event['distance_band']}); "
            f"**order:** {_event_order_label(selected_event['event_order'])}."
        )
        if not overlapping.empty:
            st.warning(
                f"{len(overlapping)} other qualifying nearby opening(s) fall within "
                "this event-study window. The selected-store chart shows observed "
                "outcomes around the selected date, not an isolated effect of that one opening."
            )
            st.dataframe(
                overlapping[
                    [
                        "new_store_name",
                        "new_store_id",
                        "new_opening_date",
                        "distance_m",
                        "distance_band",
                        "event_order",
                    ]
                ].sort_values("new_opening_date"),
                width="stretch",
                hide_index=True,
            )
        event_fig = go.Figure(
            go.Scatter(
                x=event_plot["event_time_month"],
                y=event_plot[outcome_column],
                mode="lines+markers",
                connectgaps=False,
                name=outcome_title,
                customdata=(
                    event_plot[["Active_Days"]].to_numpy()
                    if "Active_Days" in event_plot
                    else None
                ),
                hovertemplate=(
                    "Event month %{x}<br>Observed APSD: %{y:.2f}"
                    "<br>Active days: %{customdata[0]}<extra></extra>"
                    if "Active_Days" in event_plot
                    else "Event month %{x}<br>Observed APSD: %{y:.2f}<extra></extra>"
                ),
            )
        )
        event_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#B91C1C",
            annotation_text="New-store opening month (0)",
        )
        event_fig.add_hline(y=0, line_color="#777777", line_width=1)
        event_fig.update_layout(
            title=f"{store_name}: observed {outcome_title} around selected opening",
            xaxis_title="Months relative to new-store opening month",
            yaxis_title=outcome_title,
        )
        st.plotly_chart(event_fig, width="stretch")
        st.caption(
            "Months absent from the saved panel are shown as gaps, never filled with "
            "zero. Month 0 is the calendar month containing the opening date and may "
            "be only partly exposed. Active days are shown where available, but daily "
            "records are not available to establish complete-month exposure."
        )
    else:
        st.info("No opening events match the selected distance and date filters.")

    st.subheader("Cumulative nearby-opening exposure")
    exposure_bands = st.multiselect(
        "Distance bands for cumulative opening count",
        options=band_order,
        default=band_order,
        key=f"store_exposure_bands_{selected_key}",
    )
    show_outcome_overlay = st.checkbox(
        f"Overlay {outcome_title} on secondary axis",
        value=True,
        key=f"store_exposure_overlay_{selected_key}",
    )
    if not timeline.empty:
        exposure_timeline = timeline[["SalesMonth", outcome_column]].copy()
        panel_month_index = _month_index(
            pd.to_datetime(exposure_timeline["SalesMonth"], errors="coerce")
        )
        exposure_timeline["_month_index"] = panel_month_index
        opening_counts = (
            store_events.loc[store_events["distance_band"].isin(exposure_bands)]
            .groupby(["distance_band", "opening_month_index"])
            .size()
            .rename("openings")
            .reset_index()
            if not store_events.empty
            else pd.DataFrame(columns=["distance_band", "opening_month_index", "openings"])
        )
        exposure_fig = go.Figure()
        for band in exposure_bands:
            counts = (
                opening_counts.loc[opening_counts["distance_band"] == band]
                .groupby("opening_month_index")["openings"]
                .sum()
                .sort_index()
                .cumsum()
                if not opening_counts.empty
                else pd.Series(dtype=float)
            )
            cumulative_counts = counts.reindex(
                counts.index.union(
                    pd.Index(exposure_timeline["_month_index"])
                )
            ).sort_index().ffill().fillna(0)
            exposure_timeline[f"cumulative_{band}"] = exposure_timeline[
                "_month_index"
            ].map(cumulative_counts)
            exposure_fig.add_trace(
                go.Scatter(
                    x=exposure_timeline["SalesMonth"],
                    y=exposure_timeline[f"cumulative_{band}"],
                    mode="lines",
                    name=f"Cumulative openings — {band}",
                    yaxis="y",
                )
            )
        if show_outcome_overlay:
            exposure_fig.add_trace(
                go.Scatter(
                    x=exposure_timeline["SalesMonth"],
                    y=exposure_timeline[outcome_column],
                    mode="lines+markers",
                    connectgaps=False,
                    name=outcome_title,
                    yaxis="y2",
                )
            )
        exposure_fig.update_layout(
            title=f"Cumulative new-store openings — {store_name}",
            xaxis_title="Calendar month",
            yaxis={"title": "Cumulative pair opening events"},
            yaxis2={
                "title": outcome_title,
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
            hovermode="x unified",
        )
        st.plotly_chart(exposure_fig, width="stretch")
        contributing_events = store_events.loc[
            store_events["distance_band"].isin(exposure_bands)
            & store_events["new_store_opening_month"].le(timeline["SalesMonth"].max())
        ].copy() if not store_events.empty else pd.DataFrame()
        if not contributing_events.empty:
            st.caption(
                "Exposure is counted from unique existing/new-store pair events in "
                "each source distance band. It is not a count of store-month rows."
            )
            st.dataframe(
                contributing_events[
                    [
                        "new_store_name",
                        "new_store_id",
                        "new_opening_date",
                        "distance_m",
                        "distance_band",
                        "event_order",
                        "previous_nearby_openings_in_pair_history",
                    ]
                ].sort_values("new_opening_date"),
                width="stretch",
                hide_index=True,
            )
    st.info(
        "All store-level charts show observed outcomes and nearby opening chronology. "
        "They do not estimate a store-specific causal effect."
    )
    download_table(
        store_panel.drop(columns="_existing_key"),
        "Download this store's panel",
        f"store_{selected_key}_panel.csv",
    )


def main() -> None:

    st.set_page_config(

        page_title="CU Store Cannibalization",

        page_icon="🏪",

        layout="wide",

    )

    st.title("CU store cannibalization")

    st.caption(

        "Regression-based associations between nearby CU openings and existing "

        "stores' APSD sales and customer-equivalents. Estimates are not guaranteed "

        "forecasts or automatically causal effects."

    )



    sales_coefs = read_csv("regional_sales_coefficients.csv")

    customer_coefs = read_csv("regional_customer_coefficients.csv")

    pretrends = read_csv("pretrend_tests.csv")

    sample_summary = read_csv("regional_sample_summary.csv")

    event_support = read_csv("long_term_event_support.csv")

    seasonal_effects = read_csv("seasonal_effects.csv")

    seasonal_means = read_csv("seasonal_observed_means.csv")

    established = read_csv("established_store_effects.csv")

    followup = read_csv("established_followup_support.csv")

    distance_summary = read_csv("distance_exposure_summary.csv")

    model_status = read_csv("model_status.csv")

    openings_by_month = read_csv("new_store_openings_by_month.csv")

    panel = read_panel()

    pairs = read_pair_data()



    if panel.empty:

        st.error("The saved analysis panel could not be loaded.")

        st.stop()



    st.sidebar.header("Dashboard")

    page = st.sidebar.selectbox(

        "View",

        [

            "Executive Overview",

            "Regional Comparison",

            "Distance Analysis",

            "Long-Term Effects",

            "Seasonality",
            "Established Stores",
            "Regional & Distance Statistics",
            "Statistics",
            "Store-Level Exploration",
            "Expansion Simulation",
            "Cumulative Cannibalization Impact",

            "Store Opening Capacity Simulator",

        ],

    )

    required_files = [
        "regional_model_comparison.csv",
        "regional_sales_coefficients.csv",
        "regional_customer_coefficients.csv",
        "cu_regional_cannibalization_panel.parquet",
    ]
    missing = [name for name in required_files if not (OUTPUT_DIR / name).is_file()]
    if missing and page not in {"Statistics", "Expansion Simulation", "Cumulative Cannibalization Impact"}:
        st.error(
            "Regional analysis outputs are incomplete. Run "
            "`04 Regional Cannibalization.py` first. Missing: "
            + ", ".join(missing)
        )
        st.stop()

    outcome_label = st.sidebar.selectbox(

        "Outcome",

        ["APSD sales", "APSD customers"],

        help="APSD customers are transaction-derived customer-equivalents per active day, not unique people.",

    )

    outcome = "sales" if outcome_label == "APSD sales" else "customers"

    coefficients = sales_coefs if outcome == "sales" else customer_coefs



    if page == "Statistics":
        statistics_page(
            panel,
            pairs,
            read_store_master(),
            "APSD_income" if outcome == "sales" else "APSD_customers",
        )
    elif page == "Expansion Simulation":
        expansion_simulation_page(read_store_master())
    elif page == "Cumulative Cannibalization Impact":
        cumulative_cannibalization_page()
    elif page == "Regional & Distance Statistics":
        regional_distance_statistics_page(
            panel,
            pairs,
            read_store_master(),
            "APSD_income" if outcome == "sales" else "APSD_customers",
            outcome_label,
        )
    elif page == "Store-Level Exploration":
        store_level_exploration_page(
            panel,
            pairs,
            read_store_master(),
            outcome,
        )
    elif page == "Store Opening Capacity Simulator":
        store_opening_capacity_page(sales_coefs, pretrends)

    elif page == "Executive Overview":

        st.header("Executive overview")

        region_row = (

            sample_summary.select_dtypes(include="number").sum()

            if not sample_summary.empty

            else pd.Series(dtype=object)

        )

        sales_effect = sales_coefs.loc[

            (sales_coefs["region"] == "All Regions")

            & (sales_coefs["model_type"] == "cumulative")

            & (sales_coefs["distance_band"] == "0_100")

        ]

        customer_effect = customer_coefs.loc[

            (customer_coefs["region"] == "All Regions")

            & (customer_coefs["model_type"] == "cumulative")

            & (customer_coefs["distance_band"] == "0_100")

        ]

        if region_row.empty:

            st.warning("Regional sample summary is unavailable.")

        else:

            new_opening_count = (

                int(openings_by_month["distinct_new_store_openings"].sum())

                if "distinct_new_store_openings" in openings_by_month

                else 0

            )

            cols = st.columns(4)

            cols[0].metric(

                "Existing stores in panel",

                f"{panel['existing_store_id'].nunique():,}",

            )

            cols[1].metric("Distinct new-store openings", f"{new_opening_count:,}")

            exposed_stores = panel.loc[

                panel["cum_openings_within_1km"] > 0,

                "existing_store_id",

            ].nunique()

            cols[2].metric("Stores exposed within 1 km", f"{exposed_stores:,}")

            cols[3].metric(

                "Mapped existing/new-store pair events",

                f"{int(region_row.get('new_store_pair_events', 0)):,}",

            )

            st.caption(

                f"Analysis panel: {len(panel):,} store-month observations; regional "

                "regressions use stores with a validated regional classification."

            )

        cols = st.columns(2)

        for col, data, title in (

            (cols[0], sales_effect, "APSD sales, one additional opening within 100 m"),

            (cols[1], customer_effect, "APSD customers, one additional opening within 100 m"),

        ):

            if not data.empty:

                row = data.iloc[0]

                col.metric(

                    title,

                    f"{row['percent_effect']:.1f}%",

                    help=(

                        f"95% CI: {row['percent_ci_95_lower']:.1f}% to "

                        f"{row['percent_ci_95_upper']:.1f}%; "

                        f"{int(row['observations']):,} observations, "

                        f"{int(row['exposed_stores']):,} exposed stores."

                    ),

                )

            else:

                col.metric(title, "Not estimable")

        long_term_items = []

        for outcome_name, outcome_data in (

            ("APSD sales", sales_coefs),

            ("APSD customers", customer_coefs),

        ):

            estimate = outcome_data.loc[

                (outcome_data["region"] == "All Regions")

                & (outcome_data["model_type"] == "binned_event_study")

                & (outcome_data["distance_band"] == "0_100")

                & (outcome_data["event_period_label"] == "+19 to +24")

            ]

            support = event_support.loc[

                (event_support["region"] == "All Regions")

                & (event_support["outcome"] == ("sales" if outcome_name == "APSD sales" else "customers"))

                & (event_support["time_kind"] == "binned")

                & (event_support["event_period"] == "+19 to +24")

                & (event_support["distance_band"] == "0_100")

                & (event_support["sample"] == "all qualifying openings")

            ] if not event_support.empty else pd.DataFrame()

            long_term_items.append((outcome_name, estimate, support))

        st.subheader("Long-term association, 19–24 months after opening")

        long_term_cols = st.columns(2)

        for col, (title, estimate, support) in zip(long_term_cols, long_term_items):

            if estimate.empty or support.empty or int(support.iloc[0]["distinct_pair_events"]) == 0:

                col.metric(title, "Not estimable")

                col.caption("There is no qualifying follow-up support for this estimate.")

            else:

                row = estimate.iloc[0]

                support_row = support.iloc[0]

                col.metric(

                    title,

                    f"{row['percent_effect']:.1f}%",

                    help=(

                        f"95% CI: {row['percent_ci_95_lower']:.1f}% to "

                        f"{row['percent_ci_95_upper']:.1f}%."

                    ),

                )

                col.caption(

                    f"{int(support_row['distinct_pair_events']):,} pair events; "

                    f"{int(support_row['exposed_existing_stores']):,} stores; "

                    f"{int(support_row['store_month_observations']):,} store-months."

                )

        st.markdown(

            "**What this measures:** the estimated change in the existing store's "

            "average daily sales or transaction-derived customer-equivalents after "

            "new stores open nearby, conditional on store and calendar-month effects. "

            "The main cumulative coefficient is per additional qualifying opening."

        )

        add_common_warning(pretrends, "All Regions", outcome)

        if not model_status.empty:

            st.subheader("Model availability")

            statuses = model_status.groupby(

                ["model_type", "status"], dropna=False

            ).size().rename("models").reset_index()

            st.dataframe(statuses, width="stretch", hide_index=True)



    elif page == "Regional Comparison":

        st.header("Regional comparison")

        region = st.selectbox("Existing-store region", list(REGIONS))

        add_common_warning(pretrends, region, outcome)

        primary = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["distance_band"].notna())

            & (coefficients["region"] == region)

        ].copy()

        st.subheader(f"Distance effects in {region}")

        effect_chart(primary, "distance_band_label")

        regional_compare = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["distance_band"] == "0_100")

            & (coefficients["region"].isin(REGIONS))

        ].copy()

        st.subheader("Very-close-opening comparison across regions")

        effect_chart(regional_compare, "region")

        if not sample_summary.empty:

            st.subheader("Regional sample and event counts")

            st.dataframe(sample_summary, width="stretch", hide_index=True)

        download_table(primary, "Download regional estimates", "regional_estimates.csv")



    elif page == "Distance Analysis":

        st.header("Distance analysis")

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band",

            ["All distance bands", *DISTANCE_LABELS.values()],

        )

        primary = coefficients.loc[

            (coefficients["model_type"] == "cumulative")

            & (coefficients["region"] == region)

        ].copy()

        primary["distance_band_label"] = primary["distance_band"].map(

            DISTANCE_LABELS

        )

        if distance_choice != "All distance bands":

            primary = primary.loc[

                primary["distance_band_label"] == distance_choice

            ]

        primary["distance_band_label"] = pd.Categorical(

            primary["distance_band_label"],

            categories=list(DISTANCE_LABELS.values()),

            ordered=True,

        )

        effect_chart(primary, "distance_band_label")

        if not distance_summary.empty:

            st.subheader("Pair-event support")

            st.dataframe(

                distance_summary.loc[distance_summary["region"] == region],

                width="stretch",

                hide_index=True,

            )

        download_table(primary, "Download distance estimates", "distance_estimates.csv")



    elif page == "Long-Term Effects":

        st.header("Long-term event-study estimates")

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band", list(DISTANCE_LABELS.values())

        )

        horizon_options = {

            "All observed periods": EVENT_ORDER,

            "Pre-opening": ["−12 to −7", "−6 to −3", "−2"],

            "Opening through 3 months": ["0", "+1 to +3"],

            "4–6 months": ["+4 to +6"],

            "7–12 months": ["+7 to +12"],

            "13–18 months": ["+13 to +18"],

            "19–24 months": ["+19 to +24"],

            "25+ months": ["+25 months onward"],

        }

        horizon_choice = st.selectbox("Event-time horizon", list(horizon_options))

        selected = coefficients.loc[

            (coefficients["region"] == region)

            & (coefficients["model_type"] == "binned_event_study")

            & (coefficients["distance_band"] == {

                label: band for band, label in DISTANCE_LABELS.items()

            }[distance_choice])

            & coefficients["event_period_label"].notna()

            & (coefficients["event_period_label"] != "−1 (reference)")

        ].copy()

        selected = selected.loc[

            selected["event_period_label"].isin(horizon_options[horizon_choice])

        ]

        selected["event_period_label"] = pd.Categorical(

            selected["event_period_label"], categories=EVENT_ORDER, ordered=True

        )

        effect_chart(selected, "event_period_label", "distance_band_label")

        add_common_warning(pretrends, region, outcome)

        support = event_support.loc[

            (event_support["region"] == region)

            & (event_support["outcome"] == outcome)

            & (event_support["time_kind"] == "binned")

            & (

                event_support["distance_band"]

                == {

                    label: band for band, label in DISTANCE_LABELS.items()

                }[distance_choice]

            )

            & (event_support["sample"] == "all qualifying openings")

        ].copy() if not event_support.empty else pd.DataFrame()

        if not support.empty:

            support = support.loc[

                support["event_period"].isin(horizon_options[horizon_choice])

            ]

            support = support.sort_values(

                "event_period",

                key=lambda column: column.map(

                    {label: i for i, label in enumerate(EVENT_ORDER)}

                ),

            )

            st.subheader("Events and observations contributing to each period")

            st.dataframe(support, width="stretch", hide_index=True)

            if not support.loc[

                support["event_period"].isin(["+19 to +24", "+25 months onward"]),

                "distinct_pair_events",

            ].gt(0).any():

                st.warning(

                    "There are no qualifying 19-24 month observations; a two-year "

                    "effect is not supported for this selection."

                )

        download_table(selected, "Download event-study estimates", "event_study_estimates.csv")



    elif page == "Seasonality":

        st.header("Seasonal heterogeneity")

        st.caption(

            "The chart separates descriptive observed seasonal patterns from "

            "regression-based exposure interactions. No unvalidated winter, summer, "

            "or holiday calendar is assumed."

        )

        if not seasonal_means.empty:

            region = st.selectbox("Observed pattern region", ["All Regions", *REGIONS])

            means = seasonal_means.loc[seasonal_means["region"] == region]

            col1, col2 = st.columns(2)

            for col, metric, title in (

                (col1, "mean_APSD_income", "Observed mean APSD sales"),

                (col2, "mean_APSD_customers", "Observed mean APSD customers"),

            ):

                if metric in means:

                    fig = px.line(

                        means,

                        x="month_of_year",

                        y=metric,

                        markers=True,

                        title=f"{title} (descriptive)",

                    )

                    fig.update_xaxes(dtick=1, title="Month of year")

                    col.plotly_chart(fig, width="stretch")

        if not seasonal_effects.empty:

            st.subheader("Exposure association by month of year")

            effects = seasonal_effects.loc[

                (seasonal_effects["outcome"] == outcome)

                & (seasonal_effects["effect_scope"]

                   == "One additional new-store opening within 1 km")

            ]

            effect_chart(effects, "month_label", "region")

            interaction_effects = seasonal_effects.loc[

                (seasonal_effects["outcome"] == outcome)

                & (seasonal_effects["effect_scope"]

                   == "Exposure interaction difference vs December")

                & (seasonal_effects["region"] != "All Regions")

            ]

            st.subheader("Seasonal difference relative to December")

            st.caption(

                "These interaction estimates show how the exposure association "

                "differs from December; they are not standalone monthly effects."

            )

            effect_chart(interaction_effects, "month_label", "region")

            download_table(

                seasonal_effects,

                "Download seasonal effects",

                "seasonal_effects.csv",

            )



    elif page == "Established Stores":

        st.header("Established stores (24+ months before opening)")

        st.caption(

            "Eligible stores had at least 24 calendar months of operating tenure "

            "before a nearby new-store opening. The repeated-opening exposure model "

            "retains overlapping qualifying openings."

        )

        region = st.selectbox("Existing-store region", ["All Regions", *REGIONS])

        distance_choice = st.selectbox(

            "Distance band", ["All distance bands", *DISTANCE_LABELS.values()]

        )

        horizon_choice = st.selectbox(

            "Event-time horizon",

            ["All observed periods", *EVENT_ORDER],

        )

        selected = established.loc[

            (established["region"] == region)

            & (established["outcome"] == outcome)

            & (established["model_type"] == "established_binned_event_study")

        ].copy() if not established.empty else pd.DataFrame()

        if distance_choice != "All distance bands":

            selected = selected.loc[

                selected["distance_band_label"] == distance_choice

            ]

        if horizon_choice != "All observed periods":

            selected = selected.loc[

                selected["event_period_label"] == horizon_choice

            ]

        effect_chart(selected, "event_period_label", "distance_band_label")

        if not followup.empty:

            st.subheader("Observed operating and outcome follow-up")

            support = followup.loc[

                (followup["region"] == region)

                & (followup["outcome"] == outcome)

            ]

            if distance_choice != "All distance bands":

                support = support.loc[

                    support["distance_band"] == distance_choice

                ]

            st.dataframe(support, width="stretch", hide_index=True)

        download_table(selected, "Download established-store effects", "established_effects.csv")



if __name__ == "__main__":
    import os
    if os.environ.get('CU_ANALYST_MODE') == '1':
        main()
    else:
        from business_dashboard import main as business_main
        business_main()
