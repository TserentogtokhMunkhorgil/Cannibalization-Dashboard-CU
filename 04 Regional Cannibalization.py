"""Regional and long-horizon CU cannibalization analysis.

Reads the existing Excel workbooks. Default cumulative evidence goes to
outputs/cumulative_evidence; --full-diagnostics reproduces the historical suite
under outputs/cannibalization_regional. No database connection is required.
"""

from __future__ import annotations

import json
import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "cannibalization_regional"
INCOME_FILE = BASE_DIR / "store_master_data.xlsx"
DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"
SHEET_NAME = "Sheet1"

REGION_ORDER = (
    "City Center",
    "City Outskirts",
    "Far / Outer Areas",
    "Rural",
)
ZONE_TO_REGION = {
    "Core": "City Center",
    "Inner City": "City Outskirts",
    "Outer City": "Far / Outer Areas",
    "Far": "Far / Outer Areas",
    "Rural": "Rural",
}
ZONE_DEFINITIONS = {
    "Core": "0 to 1.5 km from Sukhbaatar Square",
    "Inner City": "1.5 to 3.5 km from Sukhbaatar Square",
    "Outer City": "3.5 to 7 km from Sukhbaatar Square",
    "Far": "More than 7 km from Sukhbaatar Square",
    "Rural": "UB source field equals Rural; explicit override in supplied zone code",
}

DISTANCE_BANDS = (
    ("0_100", "Very close (0-100 m)", 0.0, 100.0),
    ("100_250", "Close (100-250 m)", 100.0, 250.0),
    ("250_500", "Medium (250-500 m)", 250.0, 500.0),
    ("500_1000", "Farther nearby (500-1,000 m)", 500.0, 1000.0),
)
EVENT_MIN = -12
EVENT_MAX_DIAGNOSTIC = 24
REFERENCE_EVENT = -1
EVENT_BINS = (
    ("m12_m7", "−12 to −7", -12, -7),
    ("m6_m3", "−6 to −3", -6, -3),
    ("m2", "−2", -2, -2),
    ("reference_m1", "−1 (reference)", -1, -1),
    ("p0", "0", 0, 0),
    ("p1_p3", "+1 to +3", 1, 3),
    ("p4_p6", "+4 to +6", 4, 6),
    ("p7_p12", "+7 to +12", 7, 12),
    ("p13_p18", "+13 to +18", 13, 18),
    ("p19_p24", "+19 to +24", 19, 24),
    ("p25_plus", "+25 months onward", 25, None),
)
MODEL_REGIONS = ("All Regions", *REGION_ORDER)
OUTCOMES = {
    "sales": {
        "source": "APSD_income",
        "log": "log_APSD_income",
        "asinh": "asinh_APSD_income",
        "label": "APSD sales (MNT per active sales day)",
    },
    "customers": {
        "source": "APSD_customers",
        "log": "log_APSD_customers",
        "asinh": "asinh_APSD_customers",
        "label": "APSD customer-equivalents per active sales day",
    },
}


class UnsupportedDesign(RuntimeError):
    """Raised when an exposure design is not identified after fixed effects."""


def progress(message: str) -> None:
    print(f"\n[Regional CU analysis] {message}", flush=True)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{name} is missing required columns {missing}. "
            f"Available columns: {frame.columns.tolist()}"
        )


def _distance_band(distance: float) -> str | None:
    for name, _, lower, upper in DISTANCE_BANDS:
        if lower <= distance < upper:
            return name
    return None


def _month_index(month: pd.Series) -> pd.Series:
    return month.dt.year.astype("int64") * 12 + month.dt.month.astype("int64")


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.mask(parsed.dt.year.ge(9000))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    progress("Reading store-month and pairwise-distance workbooks")
    for path in (INCOME_FILE, DISTANCE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required workbook not found: {path}")
    income = pd.read_excel(INCOME_FILE, sheet_name=SHEET_NAME)
    distances = pd.read_excel(DISTANCE_FILE, sheet_name=SHEET_NAME)
    return income, distances


def classify_regions(income: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the supplied Distance_Zone definitions without inventing zones."""
    _require_columns(income, {"store_id", "Distance_Zone", "UB"}, "Income data")
    result = income.copy()
    zone = result["Distance_Zone"].astype("string").str.strip()
    ub = result["UB"].astype("string").str.strip()

    unknown = sorted(set(zone.dropna().unique()).difference(ZONE_TO_REGION))
    if unknown:
        raise ValueError(
            "Unrecognized Distance_Zone values; provide a validated mapping "
            f"before estimating regions: {unknown}"
        )
    mapped = zone.map(ZONE_TO_REGION)
    rural_override = ub.eq("Rural")
    mapped = mapped.mask(rural_override, "Rural")
    result["analysis_region"] = mapped

    mapping_rows: list[dict[str, Any]] = [
        {
            "source_field": "Distance_Zone",
            "source_value": value,
            "analysis_region": region,
            "source_definition": ZONE_DEFINITIONS[value],
            "mapping_rule": "Applied the user-provided Distance_Zone derivation",
        }
        for value, region in ZONE_TO_REGION.items()
    ]
    mapping_rows.append(
        {
            "source_field": "UB",
            "source_value": "Rural",
            "analysis_region": "Rural",
            "source_definition": ZONE_DEFINITIONS["Rural"],
            "mapping_rule": "Overrides distance ring as in supplied zone-construction code",
        }
    )
    mapping_rows.append(
        {
            "source_field": "Distance_Zone",
            "source_value": "Missing or unmapped",
            "analysis_region": pd.NA,
            "source_definition": "Not assigned to a requested region",
            "mapping_rule": "Retained in source panel; excluded from regional fits",
        }
    )
    mapping = pd.DataFrame(mapping_rows)
    return result, mapping


def validate_data(
    income: pd.DataFrame, distances: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    progress("Validating source columns, keys, outcome values, and region labels")
    _require_columns(
        income,
        {
            "SalesMonth",
            "store_id",
            "APSD_income",
            "APSD_customers",
            "ost_dt",
            "cls_dt",
            "Store_Status",
            "loc_area_new",
            "UB",
            "LATITUDE",
            "LONGITUDE",
            "Distance_Zone",
        },
        "store_master_data.xlsx",
    )
    _require_columns(
        distances,
        {
            "existing_store_id",
            "new_store_id",
            "distance_m",
            "existing_opening_date",
            "new_opening_date",
        },
        "store_to_store_distance.xlsx",
    )

    income = income.copy()
    distances = distances.copy()
    for column in ("SalesMonth", "ost_dt", "cls_dt"):
        income[column] = _parse_dates(income[column])
    for column in ("existing_opening_date", "new_opening_date"):
        distances[column] = _parse_dates(distances[column])
    for column in (
        "APSD_income",
        "APSD_customers",
        "LATITUDE",
        "LONGITUDE",
    ):
        income[column] = pd.to_numeric(income[column], errors="coerce")
    distances["distance_m"] = pd.to_numeric(distances["distance_m"], errors="coerce")
    income["SalesMonth"] = (
        income["SalesMonth"].dt.to_period("M").dt.to_timestamp()
    )

    duplicate_store_month = int(
        income.duplicated(["store_id", "SalesMonth"]).sum()
    )
    duplicate_pair = int(
        distances.duplicated(["existing_store_id", "new_store_id"]).sum()
    )
    if duplicate_store_month or duplicate_pair:
        raise ValueError(
            "Duplicate source keys detected and not removed: "
            f"store-month={duplicate_store_month}, pair={duplicate_pair}. "
            "Resolve duplicates before continuing."
        )
    if income[["store_id", "SalesMonth"]].isna().any().any():
        raise ValueError("Income file contains missing store IDs or invalid months.")
    if distances[
        ["existing_store_id", "new_store_id", "distance_m", "new_opening_date"]
    ].isna().any().any():
        raise ValueError("Distance file contains invalid keys, distances, or dates.")
    if (distances["distance_m"] < 0).any():
        raise ValueError("Negative pairwise distances are invalid.")

    income, mapping = classify_regions(income)
    if income.groupby("store_id")["analysis_region"].nunique(dropna=True).gt(1).any():
        raise ValueError("A store has conflicting regional classifications by month.")
    existing_date_conflicts = int(
        (distances.groupby("existing_store_id")["existing_opening_date"].nunique() > 1).sum()
    )
    new_date_conflicts = int(
        (distances.groupby("new_store_id")["new_opening_date"].nunique() > 1).sum()
    )
    if existing_date_conflicts or new_date_conflicts:
        raise ValueError(
            "Conflicting opening dates in pair metadata: "
            f"{existing_date_conflicts} existing IDs, {new_date_conflicts} new IDs."
        )

    distances["distance_group"] = distances["distance_m"].map(_distance_band)
    distances["new_opening_month"] = distances["new_opening_date"].dt.to_period("M")
    distances["existing_opening_month"] = (
        distances["existing_opening_date"].dt.to_period("M")
    )
    distances["event_pair_id"] = np.arange(len(distances), dtype=np.int64)
    distances["new_opening_month_index"] = (
        distances["new_opening_month"].dt.year * 12
        + distances["new_opening_month"].dt.month
    )
    distances["existing_opening_month_index"] = (
        distances["existing_opening_month"].dt.year * 12
        + distances["existing_opening_month"].dt.month
    )

    zone_conflicts = int(
        (income.groupby("store_id")["Distance_Zone"].nunique(dropna=True) > 1).sum()
    )
    outcome_stats = {}
    for outcome in ("APSD_income", "APSD_customers"):
        numeric = pd.to_numeric(income[outcome], errors="coerce")
        outcome_stats[f"{outcome}_missing"] = int(numeric.isna().sum())
        outcome_stats[f"{outcome}_zero"] = int(numeric.eq(0).sum())
        outcome_stats[f"{outcome}_negative"] = int(numeric.lt(0).sum())
    valid_coordinates = (
        income["LATITUDE"].between(-90, 90)
        & income["LONGITUDE"].between(-180, 180)
    )
    panel_min = income["SalesMonth"].min().to_period("M")
    panel_max = income["SalesMonth"].max().to_period("M")
    opening_in_panel = distances["new_opening_month"].between(panel_min, panel_max)
    valid_events = distances["distance_group"].notna()
    master_open_dates = (
        income[["store_id", "ost_dt"]]
        .drop_duplicates("store_id")
        .rename(columns={"store_id": "existing_store_id", "ost_dt": "master_ost_dt"})
    )
    date_check = distances[
        ["existing_store_id", "existing_opening_date", "new_opening_date"]
    ].merge(master_open_dates, on="existing_store_id", how="left", validate="many_to_one")
    opening_date_mismatch_rows = int(
        (
            date_check["existing_opening_date"].notna()
            & date_check["master_ost_dt"].notna()
            & (
                date_check["existing_opening_date"].dt.normalize()
                != date_check["master_ost_dt"].dt.normalize()
            )
        ).sum()
    )
    opening_date_used = date_check["master_ost_dt"].fillna(
        date_check["existing_opening_date"]
    )
    qualifies_as_existing = (
        opening_date_used.notna()
        & (opening_date_used < date_check["new_opening_date"])
    )

    validation_rows: list[dict[str, Any]] = [
        {"check": "income_rows", "value": len(income), "detail": ""},
        {"check": "existing_stores", "value": income["store_id"].nunique(), "detail": ""},
        {
            "check": "sales_month_range",
            "value": f"{income['SalesMonth'].min():%Y-%m} to {income['SalesMonth'].max():%Y-%m}",
            "detail": "",
        },
        {"check": "distance_pair_rows", "value": len(distances), "detail": ""},
        {
            "check": "distance_existing_stores",
            "value": distances["existing_store_id"].nunique(),
            "detail": "",
        },
        {
            "check": "distinct_new_store_openings",
            "value": distances["new_store_id"].nunique(),
            "detail": "Distinct new-store IDs",
        },
        {
            "check": "duplicate_store_month_keys",
            "value": duplicate_store_month,
            "detail": "Extra key rows; fatal if nonzero",
        },
        {
            "check": "duplicate_existing_new_pairs",
            "value": duplicate_pair,
            "detail": "Extra pair rows; fatal if nonzero",
        },
        {
            "check": "stores_with_missing_open_date",
            "value": income.loc[income["ost_dt"].isna(), "store_id"].nunique(),
            "detail": "Store IDs",
        },
        {
            "check": "rows_with_missing_closing_date",
            "value": int(income["cls_dt"].isna().sum()),
            "detail": "Sentinel dates >= year 9000 treated as open-ended",
        },
        {
            "check": "rows_with_invalid_coordinates",
            "value": int((~valid_coordinates).sum()),
            "detail": "Retained; coordinates are not used to recalculate Distance_Zone",
        },
        {
            "check": "unmapped_region_stores",
            "value": income.loc[income["analysis_region"].isna(), "store_id"].nunique(),
            "detail": "Retained in source panel; excluded only from regional estimates",
        },
        {
            "check": "conflicting_zone_store_ids",
            "value": zone_conflicts,
            "detail": "Store IDs with multiple non-null Distance_Zone values",
        },
        {
            "check": "new_opening_pairs_inside_observed_panel",
            "value": int(opening_in_panel.sum()),
            "detail": "Pair-events",
        },
        {
            "check": "distance_pairs_within_1km",
            "value": int(valid_events.sum()),
            "detail": "All preserved existing-store/new-store pairs",
        },
        {
            "check": "nearby_pairs_where_existing_store_preexists",
            "value": int((valid_events & qualifies_as_existing).sum()),
            "detail": "Pairs eligible as treatments",
        },
        {
            "check": "nearby_pairs_excluded_existing_not_preexisting",
            "value": int((valid_events & ~qualifies_as_existing).sum()),
            "detail": "New opening on/before existing-store opening or unknown date",
        },
        {
            "check": "existing_opening_date_mismatch_pairs",
            "value": opening_date_mismatch_rows,
            "detail": (
                "When available, store_master_data.ost_dt is authoritative for "
                "eligibility and tenure; distance-file date is fallback only."
            ),
        },
        {
            "check": "pairs_within_1km_opening_in_observed_month_range",
            "value": int((valid_events & opening_in_panel & qualifies_as_existing).sum()),
            "detail": "Events opening within observed panel months",
        },
        *[
            {
                "check": key,
                "value": value,
                "detail": "Outcome rows",
            }
            for key, value in outcome_stats.items()
        ],
    ]
    for status, count in income["Store_Status"].value_counts(dropna=False).items():
        validation_rows.append(
            {
                "check": f"store_status_rows_{status}",
                "value": int(count),
                "detail": "Current source status repeated over monthly rows",
            }
        )
    for band_name, band_label, _, _ in DISTANCE_BANDS:
        validation_rows.append(
            {
                "check": f"distance_pairs_{band_name}",
                "value": int((distances["distance_group"] == band_name).sum()),
                "detail": band_label,
            }
        )
    nearby_event_counts = (
        distances.loc[valid_events & qualifies_as_existing]
        .groupby("existing_store_id")["new_store_id"]
        .nunique()
    )
    validation_rows.extend(
        [
            {
                "check": "existing_stores_with_2plus_nearby_openings",
                "value": int(nearby_event_counts.ge(2).sum()),
                "detail": "Qualifying distinct new-store openings within 1 km",
            },
            {
                "check": "maximum_nearby_openings_for_one_existing_store",
                "value": int(nearby_event_counts.max()) if not nearby_event_counts.empty else 0,
                "detail": "Qualifying distinct new-store openings within 1 km",
            },
            {
                "check": "distance_pairs_over_1km_control",
                "value": int((distances["distance_m"] >= 1000).sum()),
                "detail": "Not included as treated exposure; potential comparison group",
            },
        ]
    )
    summary = pd.DataFrame(validation_rows)

    info: dict[str, Any] = {
        "income_rows": int(len(income)),
        "income_stores": int(income["store_id"].nunique()),
        "income_month_min": str(income["SalesMonth"].min().date()),
        "income_month_max": str(income["SalesMonth"].max().date()),
        "distance_rows": int(len(distances)),
        "distinct_new_store_openings": int(distances["new_store_id"].nunique()),
        "duplicate_store_month_keys": duplicate_store_month,
        "duplicate_pair_keys": duplicate_pair,
        "invalid_coordinate_rows": int((~valid_coordinates).sum()),
        "missing_open_date_rows": int(income["ost_dt"].isna().sum()),
        "missing_or_unmapped_region_stores": int(
            income.loc[income["analysis_region"].isna(), "store_id"].nunique()
        ),
        "distance_events_within_1km": int(valid_events.sum()),
        "qualifying_treatment_pairs": int((valid_events & qualifies_as_existing).sum()),
        "panel_start_month": str(panel_min),
        "panel_end_month": str(panel_max),
        **outcome_stats,
    }
    return income, distances, summary, info, mapping


def prepare_store_month_panel(
    income: pd.DataFrame, distances: pd.DataFrame
) -> pd.DataFrame:
    """Keep observed store-month records, derive outcome transforms and tenure."""
    progress("Preparing unique observed store-month panel")
    static_columns = [
        "store_id",
        "ost_dt",
        "cls_dt",
        "Store_Status",
        "loc_area_new",
        "UB",
        "LATITUDE",
        "LONGITUDE",
        "Distance_Zone",
        "analysis_region",
    ]
    static = (
        income[["store_id", "SalesMonth", *static_columns[1:]]]
        .sort_values("SalesMonth")
        .groupby("store_id", as_index=False, dropna=False)
        .agg({column: "last" for column in static_columns[1:]})
    )

    pair_open_dates = (
        distances[["existing_store_id", "existing_opening_date"]]
        .dropna(subset=["existing_opening_date"])
        .drop_duplicates("existing_store_id")
        .rename(
            columns={
                "existing_store_id": "store_id",
                "existing_opening_date": "pair_existing_opening_date",
            }
        )
    )
    static = static.merge(
        pair_open_dates,
        on="store_id",
        how="left",
        validate="one_to_one",
    )
    fill_opening = static["ost_dt"].isna() & static["pair_existing_opening_date"].notna()
    static.loc[fill_opening, "ost_dt"] = static.loc[
        fill_opening, "pair_existing_opening_date"
    ]
    static = static.drop(columns="pair_existing_opening_date")

    panel = income[
        [
            "store_id",
            "SalesMonth",
            "APSD_income",
            "APSD_customers",
            "Sales_Income_New",
            "Customers",
            "Active_Days",
        ]
    ].merge(static, on="store_id", how="left", validate="many_to_one")
    panel = panel.rename(
        columns={"store_id": "existing_store_id", "SalesMonth": "SalesMonth"}
    )
    panel["calendar_month"] = panel["SalesMonth"].dt.to_period("M")
    panel["month_index"] = _month_index(panel["SalesMonth"])
    panel["month_of_year"] = panel["SalesMonth"].dt.month
    panel["log_APSD_income"] = np.where(
        panel["APSD_income"] > 0,
        np.log(panel["APSD_income"].where(panel["APSD_income"] > 0)),
        np.nan,
    )
    panel["log_APSD_customers"] = np.where(
        panel["APSD_customers"] > 0,
        np.log(panel["APSD_customers"].where(panel["APSD_customers"] > 0)),
        np.nan,
    )
    panel["asinh_APSD_income"] = np.arcsinh(panel["APSD_income"])
    panel["asinh_APSD_customers"] = np.arcsinh(panel["APSD_customers"])
    panel["opening_month"] = panel["ost_dt"].dt.to_period("M")
    panel["tenure_months"] = (
        panel["month_index"]
        - (panel["ost_dt"].dt.year * 12 + panel["ost_dt"].dt.month)
    )
    panel["closing_month"] = panel["cls_dt"].dt.to_period("M")
    panel["operating_month_known"] = (
        panel["opening_month"].notna()
        & panel["calendar_month"].ge(panel["opening_month"])
        & (
            panel["closing_month"].isna()
            | panel["calendar_month"].le(panel["closing_month"])
        )
    )
    panel["continuity_status_known"] = panel["opening_month"].notna() & (
        panel["closing_month"].notna()
        | panel["Store_Status"].astype("string").str.strip().eq("Active")
    )

    if panel.duplicated(["existing_store_id", "SalesMonth"]).any():
        raise AssertionError("Panel is not unique at existing store-month grain.")
    if panel["calendar_month"].isna().any() or panel["month_index"].isna().any():
        raise AssertionError("Calendar-month indexing generated missing values.")
    print(
        f"Observed panel: {len(panel):,} rows, "
        f"{panel['existing_store_id'].nunique():,} stores, "
        f"{panel['calendar_month'].nunique():,} months"
    )
    print(
        f"APSD_income zero/negative: "
        f"{int((panel['APSD_income'] <= 0).sum()):,}; "
        f"APSD_customers zero/negative: "
        f"{int((panel['APSD_customers'] <= 0).sum()):,}"
    )
    print(
        "APSD definitions verified in 01 Cleaning StoreMaster.py: "
        "income / active days and transaction-derived customers / active days."
    )
    print(
        "Customer source caveat: Customers = transaction count minus twice "
        "returned-transaction count; not deduplicated people."
    )
    return panel


def prepare_pair_level_events(
    distances: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Retain all distinct qualifying pair openings, never nearest-only."""
    progress("Preparing all qualifying pair-level opening events")
    master_openings = (
        panel[["existing_store_id", "ost_dt", "analysis_region"]]
        .drop_duplicates("existing_store_id")
        .rename(columns={"ost_dt": "master_existing_opening_date"})
    )
    pairs = distances.merge(
        master_openings,
        on="existing_store_id",
        how="left",
        validate="many_to_one",
    )
    pairs["existing_opening_date_used"] = pairs[
        "existing_opening_date"
    ].fillna(pairs["master_existing_opening_date"])
    both_dates = pairs["existing_opening_date_used"].notna()
    date_conflict = (
        pairs["existing_opening_date"].notna()
        & pairs["master_existing_opening_date"].notna()
        & (
            pairs["existing_opening_date"].dt.normalize()
            != pairs["master_existing_opening_date"].dt.normalize()
        )
    )
    if date_conflict.any():
        print(
            f"Opening-date discrepancy: {int(date_conflict.sum())} pair rows; "
            "using store_master_data.ost_dt where available, pair metadata otherwise."
        )
    qualifies = (
        pairs["distance_group"].notna()
        & both_dates
        & (
            pairs["master_existing_opening_date"]
            .fillna(pairs["existing_opening_date"])
            < pairs["new_opening_date"]
        )
    )
    events = pairs.loc[qualifies].copy()
    events["existing_opening_date_used"] = events[
        "master_existing_opening_date"
    ].fillna(events["existing_opening_date"])
    events["existing_opening_month"] = (
        events["existing_opening_date_used"].dt.to_period("M")
    )
    events["existing_opening_month_index"] = (
        events["existing_opening_month"].dt.year * 12
        + events["existing_opening_month"].dt.month
    )
    events["tenure_months_at_new_opening"] = (
        events["new_opening_month_index"]
        - events["existing_opening_month_index"]
    )
    events["established_at_opening"] = (
        events["tenure_months_at_new_opening"] >= 24
    )
    events["pair_event_key"] = (
        events["existing_store_id"].astype(str)
        + "::"
        + events["new_store_id"].astype(str)
    )
    if events["pair_event_key"].duplicated().any():
        raise AssertionError("Duplicate pair-event keys remain after validation.")
    print(
        f"Qualifying existing-store/new-store opening events: {len(events):,}; "
        f"excluded pairs (not within 1 km or store not pre-existing): "
        f"{len(distances) - len(events):,}"
    )
    print(
        f"Eligible established-store opening events (24+ months tenure): "
        f"{int(events['established_at_opening'].sum()):,}"
    )
    return events


def construct_monthly_exposure(
    panel: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build repeated event-time and cumulative exposures at store-month grain."""
    progress("Constructing repeated-opening monthly exposure features")
    panel = panel.copy()
    cumulative_columns = [
        f"cum_openings_{band}_opened" for band, _, _, _ in DISTANCE_BANDS
    ]
    monthly_times = range(EVENT_MIN, EVENT_MAX_DIAGNOSTIC + 1)
    monthly_feature_names = {
        (band, k): f"exposure_{band}_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        for band, _, _, _ in DISTANCE_BANDS
        for k in monthly_times
    }
    event_bin_names = {
        (band, key): f"exposure_{band}_bin_{key}"
        for band, _, _, _ in DISTANCE_BANDS
        for key, _, _, _ in EVENT_BINS
        if key != "reference_m1"
    }
    initialized_columns = [
        *cumulative_columns,
        *monthly_feature_names.values(),
        *event_bin_names.values(),
        "cum_openings_within_1km",
        "established_cum_openings_within_1km",
    ]
    if set(initialized_columns).intersection(panel.columns):
        raise ValueError("Exposure feature columns already exist on the input panel.")
    panel = pd.concat(
        [
            panel,
            pd.DataFrame(
                0,
                index=panel.index,
                columns=initialized_columns,
                dtype=np.int64,
            ),
        ],
        axis=1,
    )

    if events.empty:
        return panel, pd.DataFrame()

    pair_columns = [
        "event_pair_id",
        "pair_event_key",
        "existing_store_id",
        "new_store_id",
        "distance_group",
        "new_opening_month_index",
        "established_at_opening",
        "analysis_region",
    ]
    pair_data = events[pair_columns].copy()
    pair_months = panel[
        [
            "existing_store_id",
            "SalesMonth",
            "month_index",
            "analysis_region",
            *[config["log"] for config in OUTCOMES.values()],
        ]
    ].merge(
        pair_data,
        on="existing_store_id",
        how="inner",
        suffixes=("_panel", "_event"),
        validate="many_to_many",
    )
    pair_months["event_time"] = (
        pair_months["month_index"] - pair_months["new_opening_month_index"]
    )
    pair_months["event_bin"] = pd.NA
    for key, _, lower, upper in EVENT_BINS:
        if upper is None:
            selected = pair_months["event_time"] >= lower
        else:
            selected = pair_months["event_time"].between(lower, upper)
        pair_months.loc[selected, "event_bin"] = key

    observed_event_months = pair_months.loc[
        pair_months["event_time"].between(EVENT_MIN, EVENT_MAX_DIAGNOSTIC)
    ]
    monthly_counts = (
        observed_event_months.groupby(
            ["existing_store_id", "SalesMonth", "distance_group", "event_time"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    for band, _, _, _ in DISTANCE_BANDS:
        for k in monthly_times:
            target = monthly_feature_names[(band, k)]
            values = monthly_counts.loc[
                (monthly_counts["distance_group"] == band)
                & (monthly_counts["event_time"] == k),
                ["existing_store_id", "SalesMonth", "n"],
            ]
            if not values.empty:
                panel = panel.drop(columns=target).merge(
                    values.rename(columns={"n": target}),
                    on=["existing_store_id", "SalesMonth"],
                    how="left",
                    validate="one_to_one",
                )
                panel[target] = panel[target].fillna(0).astype(int)

    binned = pair_months.loc[
        pair_months["event_bin"].notna()
        & pair_months["event_bin"].ne("reference_m1")
    ]
    bin_counts = (
        binned.groupby(
            ["existing_store_id", "SalesMonth", "distance_group", "event_bin"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    for band, _, _, _ in DISTANCE_BANDS:
        for event_bin in EVENT_BINS:
            key = event_bin[0]
            if key == "reference_m1":
                continue
            target = event_bin_names[(band, key)]
            values = bin_counts.loc[
                (bin_counts["distance_group"] == band)
                & (bin_counts["event_bin"] == key),
                ["existing_store_id", "SalesMonth", "n"],
            ]
            if not values.empty:
                panel = panel.drop(columns=target).merge(
                    values.rename(columns={"n": target}),
                    on=["existing_store_id", "SalesMonth"],
                    how="left",
                    validate="one_to_one",
                )
                panel[target] = panel[target].fillna(0).astype(int)

    opened = pair_months.loc[pair_months["event_time"] >= 0]
    cum_counts = (
        opened.groupby(
            ["existing_store_id", "SalesMonth", "distance_group"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    for band, _, _, _ in DISTANCE_BANDS:
        target = f"cum_openings_{band}_opened"
        values = cum_counts.loc[
            cum_counts["distance_group"] == band,
            ["existing_store_id", "SalesMonth", "n"],
        ]
        if not values.empty:
            panel = panel.drop(columns=target).merge(
                values.rename(columns={"n": target}),
                on=["existing_store_id", "SalesMonth"],
                how="left",
                validate="one_to_one",
            )
            panel[target] = panel[target].fillna(0).astype(int)
    within_counts = (
        opened.groupby(["existing_store_id", "SalesMonth"], observed=True)
        .size()
        .rename("cum_openings_within_1km")
        .reset_index()
    )
    panel = panel.drop(columns="cum_openings_within_1km").merge(
        within_counts,
        on=["existing_store_id", "SalesMonth"],
        how="left",
        validate="one_to_one",
    )
    panel["cum_openings_within_1km"] = (
        panel["cum_openings_within_1km"].fillna(0).astype(int)
    )
    established_opened = opened.loc[opened["established_at_opening"]]
    established_counts = (
        established_opened.groupby(
            ["existing_store_id", "SalesMonth"], observed=True
        )
        .size()
        .rename("established_cum_openings_within_1km")
        .reset_index()
    )
    panel = panel.drop(columns="established_cum_openings_within_1km").merge(
        established_counts,
        on=["existing_store_id", "SalesMonth"],
        how="left",
        validate="one_to_one",
    )
    panel["established_cum_openings_within_1km"] = (
        panel["established_cum_openings_within_1km"].fillna(0).astype(int)
    )

    if panel.duplicated(["existing_store_id", "SalesMonth"]).any():
        raise AssertionError("Exposure construction changed panel key uniqueness.")
    print(
        f"Pair-month rows: {len(pair_months):,}; "
        f"event-month rows in monthly diagnostic window: "
        f"{len(observed_event_months):,}"
    )
    return panel, pair_months


def _group_means(
    values: np.ndarray, codes: np.ndarray, group_count: int
) -> np.ndarray:
    totals = np.zeros((group_count, values.shape[1]), dtype=float)
    np.add.at(totals, codes, values)
    counts = np.bincount(codes, minlength=group_count).astype(float)
    return totals[codes] / counts[codes, None]


def _absorb_two_way(
    values: np.ndarray,
    store_codes: np.ndarray,
    month_codes: np.ndarray,
    n_stores: int,
    n_months: int,
) -> np.ndarray:
    residuals = np.asarray(values, dtype=float)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    residuals = residuals.copy()
    for _ in range(2000):
        previous = residuals.copy()
        residuals -= _group_means(residuals, store_codes, n_stores)
        residuals -= _group_means(residuals, month_codes, n_months)
        if np.max(np.abs(residuals - previous)) < 1e-9:
            return residuals
    raise UnsupportedDesign("Two-way fixed-effect absorption failed to converge.")


def _fit_fixed_effects(
    sample: pd.DataFrame,
    outcome_column: str,
    regressors: list[str],
    *,
    region: str,
    outcome: str,
    model_type: str,
    model_label: str,
    term_metadata: dict[str, dict[str, Any]],
    pretrend_terms: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    use = sample.loc[
        sample[outcome_column].notna()
        & np.isfinite(sample[outcome_column].astype(float))
    ].copy()
    if len(use) < 20:
        raise UnsupportedDesign(f"Only {len(use)} valid outcome observations.")
    if use["existing_store_id"].nunique() < 3:
        raise UnsupportedDesign("Fewer than three existing stores in model sample.")
    if use["calendar_month"].nunique() < 2:
        raise UnsupportedDesign("Fewer than two calendar months in model sample.")
    for column in regressors:
        use[column] = pd.to_numeric(use[column], errors="coerce").fillna(0.0)
    unsupported_terms = [
        column for column in regressors if not use[column].ne(0).any()
    ]
    regressors = [
        column for column in regressors if column not in unsupported_terms
    ]
    if not regressors:
        raise UnsupportedDesign(
            "No exposure terms have observed support after sample selection."
        )

    store_codes, store_levels = pd.factorize(use["existing_store_id"])
    month_codes, month_levels = pd.factorize(use["calendar_month"])
    y_within = _absorb_two_way(
        use[outcome_column].to_numpy(dtype=float),
        store_codes,
        month_codes,
        len(store_levels),
        len(month_levels),
    )[:, 0]
    within_total = float(np.dot(y_within, y_within))
    if regressors:
        x_within = _absorb_two_way(
            use[regressors].to_numpy(dtype=float),
            store_codes,
            month_codes,
            len(store_levels),
            len(month_levels),
        )
        rank = int(np.linalg.matrix_rank(x_within))
        if rank != x_within.shape[1]:
            raise UnsupportedDesign(
                f"Exposure matrix is rank deficient after fixed effects: "
                f"rank {rank} of {x_within.shape[1]} terms."
            )
        if len(use) <= rank:
            raise UnsupportedDesign("No residual degrees of freedom.")
        fit = sm.OLS(y_within, x_within, hasconst=False).fit(
            cov_type="cluster",
            cov_kwds={"groups": use["existing_store_id"].to_numpy()},
        )
        residual_ss = float(np.dot(fit.resid, fit.resid))
        confidence = fit.conf_int()
        rows: list[dict[str, Any]] = []
        for index, term in enumerate(regressors):
            beta = float(fit.params[index])
            lower = float(confidence[index, 0])
            upper = float(confidence[index, 1])
            metadata = term_metadata.get(term, {})
            exposure = use[term] > 0
            rows.append(
                {
                    "region": region,
                    "outcome": outcome,
                    "model_type": model_type,
                    "model_label": model_label,
                    "term": term,
                    "distance_band": metadata.get("distance_band"),
                    "distance_band_label": metadata.get("distance_band_label"),
                    "event_period": metadata.get("event_period"),
                    "event_period_label": metadata.get("event_period_label"),
                    "event_time": metadata.get("event_time"),
                    "calendar_month_number": metadata.get("calendar_month_number"),
                    "coefficient": beta,
                    "standard_error": float(fit.bse[index]),
                    "p_value": float(fit.pvalues[index]),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "percent_effect": float(100 * np.expm1(beta))
                    if outcome_column.startswith("log_")
                    else np.nan,
                    "percent_ci_95_lower": float(100 * np.expm1(lower))
                    if outcome_column.startswith("log_")
                    else np.nan,
                    "percent_ci_95_upper": float(100 * np.expm1(upper))
                    if outcome_column.startswith("log_")
                    else np.nan,
                    "outcome_scale": "log" if outcome_column.startswith("log_") else "asinh",
                    "observations": int(len(use)),
                    "stores": int(use["existing_store_id"].nunique()),
                    "exposed_stores": int(
                        use.loc[exposure, "existing_store_id"].nunique()
                    ),
                    "pair_month_exposure_count": int(use[term].sum()),
                    "within_r_squared": (
                        1 - residual_ss / within_total if within_total > 0 else np.nan
                    ),
                    "cluster_count": int(use["existing_store_id"].nunique()),
                }
            )
        coef = pd.DataFrame(rows)
        pretrend_stat = np.nan
        pretrend_p = np.nan
        pretrend_df = 0
        tested = [term for term in (pretrend_terms or []) if term in regressors]
        if tested:
            restrictions = np.zeros((len(tested), len(regressors)))
            for row_idx, term in enumerate(tested):
                restrictions[row_idx, regressors.index(term)] = 1
            test = fit.wald_test(restrictions, use_f=True, scalar=True)
            pretrend_stat = float(np.asarray(test.statistic).item())
            pretrend_p = float(np.asarray(test.pvalue).item())
            pretrend_df = len(tested)
    else:
        fit = None
        residual_ss = within_total
        coef = pd.DataFrame()
        rank = 0
        pretrend_stat = np.nan
        pretrend_p = np.nan
        pretrend_df = 0

    model_summary = {
        "region": region,
        "outcome": outcome,
        "model_type": model_type,
        "model_label": model_label,
        "status": "estimated",
        "observations": int(len(use)),
        "stores": int(use["existing_store_id"].nunique()),
        "calendar_months": int(use["calendar_month"].nunique()),
        "exposure_terms": len(regressors),
        "omitted_unsupported_terms": ";".join(unsupported_terms),
        "design_rank": rank,
        "within_r_squared": (
            1 - residual_ss / within_total if within_total > 0 else np.nan
        ),
        "pretrend_joint_f_statistic": pretrend_stat,
        "pretrend_joint_p_value": pretrend_p,
        "pretrend_terms_tested": pretrend_df,
        "clustered_by": "existing_store_id",
        "inference_note": (
            "Store-clustered covariance; common-event dependence across stores "
            "may remain."
        ),
        "failure_reason": "",
    }
    return model_summary, coef, fit


def _region_sample(panel: pd.DataFrame, region: str) -> pd.DataFrame:
    if region == "All Regions":
        return panel.loc[panel["analysis_region"].isin(REGION_ORDER)].copy()
    return panel.loc[panel["analysis_region"] == region].copy()


def _term_for_event_bin(band: str, key: str) -> str:
    return f"exposure_{band}_bin_{key}"


def _term_for_month(band: str, event_time: int) -> str:
    suffix = f"m{abs(event_time)}" if event_time < 0 else f"p{event_time}"
    return f"exposure_{band}_event_{suffix}"


def _distance_term_metadata(band: str, period: str) -> dict[str, Any]:
    label = next(label for name, label, _, _ in DISTANCE_BANDS if name == band)
    bin_lookup = {key: label for key, label, _, _ in EVENT_BINS}
    return {
        "distance_band": band,
        "distance_band_label": label,
        "event_period": period,
        "event_period_label": bin_lookup.get(period, period),
    }


def _run_recorded_model(
    *,
    panel: pd.DataFrame,
    region: str,
    outcome: str,
    outcome_column: str,
    model_type: str,
    model_label: str,
    regressors: list[str],
    term_metadata: dict[str, dict[str, Any]],
    pretrend_terms: list[str] | None,
    status_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, pd.DataFrame, Any]:
    sample = _region_sample(panel, region)
    try:
        summary, coefficients, fit = _fit_fixed_effects(
            sample,
            outcome_column,
            regressors,
            region=region,
            outcome=outcome,
            model_type=model_type,
            model_label=model_label,
            term_metadata=term_metadata,
            pretrend_terms=pretrend_terms,
        )
    except Exception as error:
        summary = {
            "region": region,
            "outcome": outcome,
            "model_type": model_type,
            "model_label": model_label,
            "status": "failed",
            "observations": int(sample[outcome_column].notna().sum()),
            "stores": int(sample.loc[sample[outcome_column].notna(), "existing_store_id"].nunique()),
            "calendar_months": int(sample.loc[sample[outcome_column].notna(), "calendar_month"].nunique()),
            "exposure_terms": len(regressors),
            "design_rank": np.nan,
            "within_r_squared": np.nan,
            "pretrend_joint_f_statistic": np.nan,
            "pretrend_joint_p_value": np.nan,
            "pretrend_terms_tested": 0,
            "clustered_by": "existing_store_id",
            "inference_note": "No estimates produced.",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
        status_rows.append(summary)
        return None, pd.DataFrame(), None
    status_rows.append(summary)
    return summary, coefficients, fit


def construct_long_term_event_time(
    pair_months: pd.DataFrame,
) -> pd.DataFrame:
    """Create event support by exact month and prespecified long-term bins."""
    progress("Summarizing monthly and binned event-time support by region/outcome")
    support_rows: list[dict[str, Any]] = []
    diagnostic = pair_months.loc[
        pair_months["event_time"].between(EVENT_MIN, EVENT_MAX_DIAGNOSTIC)
    ].copy()

    def add_support(
        rows: pd.DataFrame,
        label: str,
        time_kind: str,
        band: str,
        region: str,
        outcome: str,
        outcome_column: str,
        eligible_only: bool,
    ) -> None:
        if region == "All Regions":
            rows = rows.loc[rows["analysis_region_panel"].isin(REGION_ORDER)]
        else:
            rows = rows.loc[rows["analysis_region_panel"] == region]
        if eligible_only:
            rows = rows.loc[rows["established_at_opening"]]
        rows = rows.loc[rows[outcome_column].notna()]
        if band != "All distance bands":
            rows = rows.loc[rows["distance_group"] == band]
        support_rows.append(
            {
                "region": region,
                "outcome": outcome,
                "sample": "established" if eligible_only else "all qualifying openings",
                "time_kind": time_kind,
                "event_period": label,
                "distance_band": band,
                "distinct_pair_events": int(rows["event_pair_id"].nunique()),
                "distinct_new_store_openings": int(rows["new_store_id"].nunique()),
                "exposed_existing_stores": int(rows["existing_store_id"].nunique()),
                "pair_month_observations": int(len(rows)),
                "store_month_observations": int(
                    rows[["existing_store_id", "SalesMonth"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )

    for region in MODEL_REGIONS:
        for outcome, details in OUTCOMES.items():
            outcome_column = details["log"]
            for band, _, _, _ in DISTANCE_BANDS:
                for event_time in range(EVENT_MIN, EVENT_MAX_DIAGNOSTIC + 1):
                    add_support(
                        diagnostic.loc[diagnostic["event_time"] == event_time],
                        str(event_time),
                        "monthly",
                        band,
                        region,
                        outcome,
                        outcome_column,
                        False,
                    )
                for event_bin in EVENT_BINS:
                    label, lower, upper = event_bin[1], event_bin[2], event_bin[3]
                    if upper is None:
                        selection = pair_months["event_time"] >= lower
                    else:
                        selection = pair_months["event_time"].between(lower, upper)
                    add_support(
                        pair_months.loc[selection],
                        label,
                        "binned",
                        band,
                        region,
                        outcome,
                        outcome_column,
                        False,
                    )
                    add_support(
                        pair_months.loc[selection],
                        label,
                        "binned",
                        band,
                        region,
                        outcome,
                        outcome_column,
                        True,
                    )
    return pd.DataFrame(support_rows)


def _model_regressors(
    model_type: str, bands: list[tuple[str, str, float, float]]
) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    regressors: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    leads: list[str] = []
    if model_type == "cumulative":
        for band, label, _, _ in bands:
            term = f"cum_openings_{band}_opened"
            regressors.append(term)
            metadata[term] = {
                "distance_band": band,
                "distance_band_label": label,
            }
    elif model_type == "binned_event_study":
        for band, label, _, _ in bands:
            for event_bin in EVENT_BINS:
                key, event_label, upper = (
                    event_bin[0],
                    event_bin[1],
                    event_bin[3],
                )
                if key == "reference_m1":
                    continue
                term = _term_for_event_bin(band, key)
                regressors.append(term)
                metadata[term] = {
                    "distance_band": band,
                    "distance_band_label": label,
                    "event_period": key,
                    "event_period_label": event_label,
                }
                if upper is not None and upper < 0:
                    leads.append(term)
    elif model_type == "monthly_event_study":
        for band, label, _, _ in bands:
            for event_time in [
                *range(EVENT_MIN, REFERENCE_EVENT),
                *range(0, EVENT_MAX_DIAGNOSTIC + 1),
            ]:
                term = _term_for_month(band, event_time)
                regressors.append(term)
                metadata[term] = {
                    "distance_band": band,
                    "distance_band_label": label,
                    "event_time": event_time,
                    "event_period": str(event_time),
                    "event_period_label": f"{event_time:+d}",
                }
                if event_time < REFERENCE_EVENT:
                    leads.append(term)
    return regressors, metadata, leads


def run_regional_models(
    panel: pd.DataFrame,
    model_types: tuple[str, ...] = ('cumulative', 'monthly_event_study', 'binned_event_study'),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run cumulative, monthly and binned repeated-event models by region/outcome."""
    progress("Estimating regional fixed-effects model families")
    status_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    pretrend_rows: list[dict[str, Any]] = []

    for region in MODEL_REGIONS:
        for outcome, details in OUTCOMES.items():
            outcome_column = details["log"]
            for model_type, label in (
                ("cumulative", "Cumulative distance exposure"),
                ("monthly_event_study", "Monthly event study, -12 to +24"),
                ("binned_event_study", "Binned event study, -12 to 25+"),
            ):
                if model_type not in model_types:
                    continue
                regressors, term_metadata, leads = _model_regressors(
                    model_type, list(DISTANCE_BANDS)
                )
                summary, coefficients, _ = _run_recorded_model(
                    panel=panel,
                    region=region,
                    outcome=outcome,
                    outcome_column=outcome_column,
                    model_type=model_type,
                    model_label=label,
                    regressors=regressors,
                    term_metadata=term_metadata,
                    pretrend_terms=leads,
                    status_rows=status_rows,
                )
                if summary and not coefficients.empty:
                    coefficient_frames.append(coefficients)
                if model_type in ("monthly_event_study", "binned_event_study"):
                    current = summary or status_rows[-1]
                    pretrend_rows.append(
                        {
                            "region": region,
                            "outcome": outcome,
                            "model_type": model_type,
                            "joint_f_statistic": current.get(
                                "pretrend_joint_f_statistic", np.nan
                            ),
                            "joint_p_value": current.get(
                                "pretrend_joint_p_value", np.nan
                            ),
                            "lead_terms_tested": current.get(
                                "pretrend_terms_tested", 0
                            ),
                            "observations": current.get("observations", 0),
                            "stores": current.get("stores", 0),
                            "status": current.get("status", "failed"),
                            "warning": (
                                "Joint pre-opening coefficients are not all zero at 5%."
                                if pd.notna(
                                    current.get("pretrend_joint_p_value", np.nan)
                                )
                                and current["pretrend_joint_p_value"] < 0.05
                                else ""
                            ),
                        }
                    )

    coefficients_all = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    comparison = pd.DataFrame(status_rows)
    pretrends = pd.DataFrame(pretrend_rows)
    return comparison, coefficients_all, pretrends


def run_seasonal_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate month-of-year exposure heterogeneity with calendar-month FE."""
    progress("Estimating month-of-year exposure interactions")
    seasonal_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    means = (
        panel.loc[panel["analysis_region"].isin(REGION_ORDER)]
        .groupby(["analysis_region", "month_of_year"], observed=True)
        .agg(
            mean_APSD_income=("APSD_income", "mean"),
            mean_APSD_customers=("APSD_customers", "mean"),
            sales_observations=("APSD_income", "count"),
            customer_observations=("APSD_customers", "count"),
            stores=("existing_store_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"analysis_region": "region"})
    )
    panel = panel.copy()
    base_term = "cum_openings_within_1km"
    interaction_terms = []
    metadata: dict[str, dict[str, Any]] = {
        base_term: {"calendar_month_number": 12}
    }
    for month in range(1, 12):
        term = f"exposure_within_1km_x_month_{month:02d}"
        panel[term] = panel[base_term] * panel["month_of_year"].eq(month).astype(int)
        interaction_terms.append(term)
        metadata[term] = {"calendar_month_number": month}
    regressors = [base_term, *interaction_terms]

    for region in MODEL_REGIONS:
        for outcome, details in OUTCOMES.items():
            sample = _region_sample(panel, region)
            try:
                summary, seasonal_coefficients, fit = _fit_fixed_effects(
                    sample,
                    details["log"],
                    regressors,
                    region=region,
                    outcome=outcome,
                    model_type="seasonal_interaction",
                    model_label="Cumulative <=1km exposure x month of year",
                    term_metadata=metadata,
                )
            except Exception as error:
                status_rows.append(
                    {
                        "region": region,
                        "outcome": outcome,
                        "model_type": "seasonal_interaction",
                        "status": "failed",
                        "observations": int(sample[details["log"]].notna().sum()),
                        "stores": int(
                            sample.loc[sample[details["log"]].notna(), "existing_store_id"].nunique()
                        ),
                        "failure_reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            status_rows.append(summary)
            if fit is None:
                continue
            for _, interaction in seasonal_coefficients.iterrows():
                if interaction["term"] == base_term:
                    continue
                seasonal_rows.append(
                    {
                        "region": region,
                        "outcome": outcome,
                        "month_of_year": int(
                            interaction["calendar_month_number"]
                        ),
                        "month_label": pd.Timestamp(
                            2000,
                            int(interaction["calendar_month_number"]),
                            1,
                        ).strftime("%B"),
                        "effect_scope": "Exposure interaction difference vs December",
                        "coefficient_log": interaction["coefficient"],
                        "standard_error": interaction["standard_error"],
                        "ci_95_lower_log": interaction["ci_95_lower"],
                        "ci_95_upper_log": interaction["ci_95_upper"],
                        "percent_effect": interaction["percent_effect"],
                        "percent_ci_95_lower": interaction["percent_ci_95_lower"],
                        "percent_ci_95_upper": interaction["percent_ci_95_upper"],
                        "p_value": interaction["p_value"],
                        "observations": summary["observations"],
                        "stores": summary["stores"],
                        "exposed_stores": int(
                            sample.loc[
                                sample[base_term] > 0, "existing_store_id"
                            ].nunique()
                        ),
                        "reference_month": "December",
                    }
                )
            position = {term: idx for idx, term in enumerate(regressors)}
            covariance = np.asarray(fit.cov_params())
            beta = np.asarray(fit.params)
            for month in range(1, 13):
                contrast = np.zeros(len(regressors))
                contrast[position[base_term]] = 1
                if month != 12:
                    contrast[position[f"exposure_within_1km_x_month_{month:02d}"]] = 1
                estimate = float(contrast @ beta)
                variance = float(contrast @ covariance @ contrast)
                se = float(np.sqrt(max(variance, 0)))
                crit = 1.959963984540054
                lower, upper = estimate - crit * se, estimate + crit * se
                seasonal_rows.append(
                    {
                        "region": region,
                        "outcome": outcome,
                        "month_of_year": month,
                        "month_label": pd.Timestamp(2000, month, 1).strftime("%B"),
                        "effect_scope": "One additional new-store opening within 1 km",
                        "coefficient_log": estimate,
                        "standard_error": se,
                        "ci_95_lower_log": lower,
                        "ci_95_upper_log": upper,
                        "percent_effect": float(100 * np.expm1(estimate)),
                        "percent_ci_95_lower": float(100 * np.expm1(lower)),
                        "percent_ci_95_upper": float(100 * np.expm1(upper)),
                        "observations": summary["observations"],
                        "stores": summary["stores"],
                        "exposed_stores": int(
                            sample.loc[
                                sample[base_term] > 0, "existing_store_id"
                            ].nunique()
                        ),
                        "reference_month": "December",
                    }
                )
    return pd.DataFrame(seasonal_rows), pd.DataFrame(status_rows), means


def prepare_established_store_sample(
    panel: pd.DataFrame,
    pair_months: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add exposures from events whose existing stores had 24+ months tenure."""
    progress("Constructing established-store (24+ months pre-event tenure) sample")
    established_pairs = pair_months.loc[pair_months["established_at_opening"]].copy()
    panel = panel.copy()
    term_map = {
        (band, key): f"established_exposure_{band}_bin_{key}"
        for band, _, _, _ in DISTANCE_BANDS
        for key, _, _, _ in EVENT_BINS
        if key != "reference_m1"
    }
    for term in term_map.values():
        panel[term] = 0
    if not established_pairs.empty:
        binned = established_pairs.loc[
            established_pairs["event_bin"].notna()
            & established_pairs["event_bin"].ne("reference_m1")
        ]
        counts = (
            binned.groupby(
                [
                    "existing_store_id",
                    "SalesMonth",
                    "distance_group",
                    "event_bin",
                ],
                observed=True,
            )
            .size()
            .rename("n")
            .reset_index()
        )
        for band, _, _, _ in DISTANCE_BANDS:
            for key, _, _, _ in EVENT_BINS:
                if key == "reference_m1":
                    continue
                term = term_map[(band, key)]
                subset = counts.loc[
                    (counts["distance_group"] == band)
                    & (counts["event_bin"] == key),
                    ["existing_store_id", "SalesMonth", "n"],
                ]
                if not subset.empty:
                    panel = panel.drop(columns=term).merge(
                        subset.rename(columns={"n": term}),
                        on=["existing_store_id", "SalesMonth"],
                        how="left",
                        validate="one_to_one",
                    )
                    panel[term] = panel[term].fillna(0).astype(int)

    follow_rows: list[dict[str, Any]] = []
    operating = panel.set_index(["existing_store_id", "month_index"])
    horizon_months = (0, 3, 6, 12, 18, 24)
    for outcome, details in OUTCOMES.items():
        for band, label, _, _ in DISTANCE_BANDS:
            events = established_pairs.loc[
                established_pairs["distance_group"] == band
            ].drop_duplicates("event_pair_id")
            for horizon in horizon_months:
                complete: set[int] = set()
                for event in events.itertuples(index=False):
                    event_id_value = event.event_pair_id
                    store_id_value = event.existing_store_id
                    open_month_index_value = event.new_opening_month_index
                    if not isinstance(event_id_value, (int, np.integer)):
                        raise TypeError("Event identifiers must be integer values.")
                    if not isinstance(store_id_value, (int, np.integer)):
                        raise TypeError("Store identifiers must be integer values.")
                    if not isinstance(open_month_index_value, (int, np.integer)):
                        raise TypeError("New-store opening month indexes must be integers.")
                    event_id = int(event_id_value)
                    store_id = int(store_id_value)
                    opening_month_index = int(open_month_index_value)
                    target_months = [
                        opening_month_index + k for k in range(0, horizon + 1)
                    ]
                    rows: list[Any] = [
                        operating.loc[(store_id, month)]
                        for month in target_months
                        if (store_id, month) in operating.index
                    ]
                    if len(rows) != horizon + 1:
                        continue
                    outcome_values = [row[details["log"]] for row in rows]
                    status_values = [bool(row["continuity_status_known"]) for row in rows]
                    operating_values = [bool(row["operating_month_known"]) for row in rows]
                    if (
                        all(pd.notna(value) for value in outcome_values)
                        and all(status_values)
                        and all(operating_values)
                    ):
                        complete.add(event_id)
                follow_rows.append(
                    {
                        "region": "All Regions",
                        "outcome": outcome,
                        "distance_band": label,
                        "followup_horizon_months": horizon,
                        "eligible_established_opening_events": int(len(events)),
                        "events_open_and_observed_through_horizon": int(len(complete)),
                        "eligible_stores": int(events["existing_store_id"].nunique()),
                        "stores_open_and_observed_through_horizon": int(
                            events.loc[
                                events["event_pair_id"].isin(complete),
                                "existing_store_id",
                            ].nunique()
                        ),
                    }
                )
                for region in REGION_ORDER:
                    region_events = events.loc[
                        events["analysis_region_panel"] == region
                    ]
                    region_complete = complete.intersection(
                        set(region_events["event_pair_id"].astype(int))
                    )
                    follow_rows.append(
                        {
                            "region": region,
                            "outcome": outcome,
                            "distance_band": label,
                            "followup_horizon_months": horizon,
                            "eligible_established_opening_events": int(
                                len(region_events)
                            ),
                            "events_open_and_observed_through_horizon": int(
                                len(region_complete)
                            ),
                            "eligible_stores": int(
                                region_events["existing_store_id"].nunique()
                            ),
                            "stores_open_and_observed_through_horizon": int(
                                region_events.loc[
                                    region_events["event_pair_id"].isin(region_complete),
                                    "existing_store_id",
                                ].nunique()
                            ),
                        }
                    )
    return panel, pd.DataFrame(follow_rows)


def run_established_store_models(
    established_panel: pd.DataFrame,
    status_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for region in MODEL_REGIONS:
        for outcome, details in OUTCOMES.items():
            regressors: list[str] = []
            term_metadata: dict[str, dict[str, Any]] = {}
            leads: list[str] = []
            for band, label, _, _ in DISTANCE_BANDS:
                for key, event_label, _, upper in EVENT_BINS:
                    if key == "reference_m1":
                        continue
                    term = f"established_exposure_{band}_bin_{key}"
                    regressors.append(term)
                    term_metadata[term] = {
                        "distance_band": band,
                        "distance_band_label": label,
                        "event_period": key,
                        "event_period_label": event_label,
                    }
                    if upper is not None and upper < 0:
                        leads.append(term)
            summary, coefficients, _ = _run_recorded_model(
                panel=established_panel,
                region=region,
                outcome=outcome,
                outcome_column=details["log"],
                model_type="established_binned_event_study",
                model_label="24+ months tenure before each new opening",
                regressors=regressors,
                term_metadata=term_metadata,
                pretrend_terms=leads,
                status_rows=status_rows,
            )
            if summary and not coefficients.empty:
                frames.append(coefficients)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        pd.DataFrame(status_rows),
    )


def run_robustness_checks(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Compare distance aggregation and sensitivity to stores with repeated events."""
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    repeated_counts = panel.groupby("existing_store_id")[
        "cum_openings_within_1km"
    ].max()
    single_store_ids = repeated_counts.index[repeated_counts <= 1]

    panel_alt = panel.copy()
    panel_alt["cum_openings_0_250m"] = (
        panel_alt["cum_openings_0_100_opened"]
        + panel_alt["cum_openings_100_250_opened"]
    )
    alt_terms = [
        "cum_openings_0_250m",
        "cum_openings_250_500_opened",
        "cum_openings_500_1000_opened",
    ]
    alt_metadata = {
        term: {"distance_band": term.replace("cum_openings_", "")}
        for term in alt_terms
    }
    for region in MODEL_REGIONS:
        for outcome, details in OUTCOMES.items():
            for label, model_panel, regressors, model_type in (
                (
                    "Alternative distance: combine 0-100m and 100-250m",
                    panel_alt,
                    alt_terms,
                    "robustness_alt_distance",
                ),
                (
                    "Single-opening stores only",
                    panel.loc[
                        panel["existing_store_id"].isin(single_store_ids)
                    ],
                    [
                        f"cum_openings_{band}_opened"
                        for band, _, _, _ in DISTANCE_BANDS
                    ],
                    "robustness_single_event_stores",
                ),
                (
                    "Asinh outcome; all qualifying openings",
                    panel,
                    [
                        f"cum_openings_{band}_opened"
                        for band, _, _, _ in DISTANCE_BANDS
                    ],
                    "robustness_asinh",
                ),
            ):
                outcome_column = (
                    details["asinh"]
                    if model_type == "robustness_asinh"
                    else details["log"]
                )
                metadata = (
                    alt_metadata
                    if model_type == "robustness_alt_distance"
                    else {
                        term: _distance_term_metadata(band, "")
                        for term, (band, _, _, _) in zip(
                            regressors, DISTANCE_BANDS
                        )
                    }
                )
                summary, coefficients, _ = _run_recorded_model(
                    panel=model_panel,
                    region=region,
                    outcome=outcome,
                    outcome_column=outcome_column,
                    model_type=model_type,
                    model_label=label,
                    regressors=regressors,
                    term_metadata=metadata,
                    pretrend_terms=None,
                    status_rows=status_rows,
                )
                rows.append(
                    {
                        "region": region,
                        "outcome": outcome,
                        "robustness_specification": label,
                        "model_type": model_type,
                        "status": summary["status"] if summary else "failed",
                        "observations": summary["observations"] if summary else 0,
                        "stores": summary["stores"] if summary else 0,
                        "pretrend_joint_p_value": np.nan,
                        "failure_reason": summary["failure_reason"] if summary else "",
                        "coefficient_count": len(coefficients),
                    }
                )
                if not coefficients.empty:
                    coefficient_frames.append(coefficients)
    return (
        pd.DataFrame(rows),
        status_rows,
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(),
    )


def create_regional_summaries(
    panel: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        region_panel = panel.loc[panel["analysis_region"] == region]
        region_events = events.loc[events["analysis_region"] == region]
        rows.append(
            {
                "region": region,
                "existing_stores": int(region_panel["existing_store_id"].nunique()),
                "store_month_observations": int(len(region_panel)),
                "stores_with_positive_sales": int(
                    region_panel.loc[
                        region_panel["log_APSD_income"].notna(),
                        "existing_store_id",
                    ].nunique()
                ),
                "stores_with_positive_customers": int(
                    region_panel.loc[
                        region_panel["log_APSD_customers"].notna(),
                        "existing_store_id",
                    ].nunique()
                ),
                "exposed_stores_within_1km": int(
                    region_events["existing_store_id"].nunique()
                ),
                "new_store_pair_events": int(len(region_events)),
                "new_store_openings": int(region_events["new_store_id"].nunique()),
                "established_pair_events": int(
                    region_events["established_at_opening"].sum()
                ),
            }
        )
    regional = pd.DataFrame(rows)
    distance_rows: list[dict[str, Any]] = []
    for region in MODEL_REGIONS:
        for band, label, _, _ in DISTANCE_BANDS:
            subset = events.loc[events["distance_group"] == band]
            if region != "All Regions":
                subset = subset.loc[subset["analysis_region"] == region]
            distance_rows.append(
                {
                    "region": region,
                    "distance_band": band,
                    "distance_band_label": label,
                    "pair_events": int(len(subset)),
                    "distinct_new_store_openings": int(subset["new_store_id"].nunique()),
                    "existing_stores": int(subset["existing_store_id"].nunique()),
                    "established_pair_events": int(
                        subset["established_at_opening"].sum()
                    ),
                }
            )
    return regional, pd.DataFrame(distance_rows)


def create_plots(
    coefficients: pd.DataFrame,
    seasonal_effects: pd.DataFrame,
    seasonal_means: pd.DataFrame,
) -> None:
    progress("Drawing saved analysis figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for outcome, file_name, title in (
        ("sales", "event_study_regional_sales.png", "Regional event study: APSD sales"),
        (
            "customers",
            "event_study_regional_customers.png",
            "Regional event study: APSD customers",
        ),
    ):
        data = coefficients.loc[
            (coefficients["outcome"] == outcome)
            & (coefficients["model_type"] == "binned_event_study")
            & (coefficients["distance_band"] == "0_100")
        ]
        fig, axis = plt.subplots(figsize=(12, 6))
        for region in REGION_ORDER:
            series = data.loc[data["region"] == region].copy()
            series["_order"] = series["event_period"].map(
                {key: idx for idx, (key, _, _, _) in enumerate(EVENT_BINS)}
            )
            series = series.sort_values("_order")
            axis.errorbar(
                series["event_period_label"],
                series["percent_effect"],
                yerr=[
                    series["percent_effect"] - series["percent_ci_95_lower"],
                    series["percent_ci_95_upper"] - series["percent_effect"],
                ],
                fmt="o-",
                capsize=3,
                label=region,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(3.5, color="gray", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Associated percentage change per opening")
        axis.set_title(f"{title}, 0-100 m; event -1 is reference")
        axis.tick_params(axis="x", rotation=20)
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / file_name, dpi=170)
        plt.close(fig)

    cumulative = coefficients.loc[
        (coefficients["model_type"] == "cumulative")
        & (coefficients["region"] != "All Regions")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, outcome in zip(axes, ("sales", "customers")):
        data = cumulative.loc[cumulative["outcome"] == outcome]
        for region in REGION_ORDER:
            series = data.loc[data["region"] == region].copy()
            series["_band_order"] = series["distance_band"].map(
                {band: idx for idx, (band, _, _, _) in enumerate(DISTANCE_BANDS)}
            )
            series = series.sort_values("_band_order")
            axis.errorbar(
                series["distance_band_label"],
                series["percent_effect"],
                yerr=[
                    series["percent_effect"] - series["percent_ci_95_lower"],
                    series["percent_ci_95_upper"] - series["percent_effect"],
                ],
                fmt="o",
                capsize=3,
                label=region,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"APSD {outcome}")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Associated percentage per additional opening")
    axes[1].legend()
    fig.suptitle("Regional cumulative-exposure estimates")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regional_distance_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    month_labels = [
        pd.Timestamp(2000, month, 1).strftime("%b") for month in range(1, 13)
    ]
    for axis, _, measure, title in (
        (axes[0, 0], "sales", "mean_APSD_income", "Observed mean APSD sales"),
        (
            axes[0, 1],
            "customers",
            "mean_APSD_customers",
            "Observed mean APSD customers",
        ),
    ):
        means = seasonal_means.copy()
        for region in REGION_ORDER:
            series = means.loc[means["region"] == region].sort_values(
                "month_of_year"
            )
            axis.plot(
                [month_labels[int(x) - 1] for x in series["month_of_year"]],
                series[measure],
                marker="o",
                label=region,
            )
        axis.set_title(title + " (descriptive)")
        axis.grid(alpha=0.25)
    for axis, outcome in zip(axes[1], ("sales", "customers")):
        data = seasonal_effects.loc[
            (seasonal_effects["region"] != "All Regions")
            & (seasonal_effects["outcome"] == outcome)
        ]
        for region in REGION_ORDER:
            series = data.loc[data["region"] == region].sort_values(
                "month_of_year"
            )
            axis.errorbar(
                series["month_of_year"],
                series["percent_effect"],
                yerr=[
                    series["percent_effect"] - series["percent_ci_95_lower"],
                    series["percent_ci_95_upper"] - series["percent_effect"],
                ],
                marker="o",
                capsize=2,
                label=region,
            )
        axis.set_xticks(range(1, 13), month_labels)
        axis.set_title(f"Exposure association by month: APSD {outcome}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
    axes[1, 0].set_ylabel("Associated change per additional opening (%)")
    axes[1, 1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "seasonal_effects.png", dpi=170)
    plt.close(fig)


def save_outputs(
    *,
    panel: pd.DataFrame,
    validation: pd.DataFrame,
    mapping: pd.DataFrame,
    regional_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    event_support: pd.DataFrame,
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
    seasonal_effects: pd.DataFrame,
    established_effects: pd.DataFrame,
    pretrends: pd.DataFrame,
    robustness: pd.DataFrame,
    model_status: pd.DataFrame,
    seasonal_means: pd.DataFrame,
    followup_support: pd.DataFrame,
    validation_info: dict[str, Any],
    opening_month_summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data_validation_summary.csv": validation,
        "region_mapping.csv": mapping,
        "regional_sample_summary.csv": regional_summary,
        "distance_exposure_summary.csv": distance_summary,
        "long_term_event_support.csv": event_support,
        "regional_model_comparison.csv": comparison,
        "regional_sales_coefficients.csv": coefficients.loc[
            coefficients["outcome"] == "sales"
        ],
        "regional_customer_coefficients.csv": coefficients.loc[
            coefficients["outcome"] == "customers"
        ],
        "seasonal_effects.csv": seasonal_effects,
        "established_store_effects.csv": established_effects,
        "pretrend_tests.csv": pretrends,
        "robustness_checks.csv": robustness,
        "model_status.csv": model_status,
        "seasonal_observed_means.csv": seasonal_means,
        "established_followup_support.csv": followup_support,
        "new_store_openings_by_month.csv": opening_month_summary,
    }
    for file_name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / file_name, index=False)
    panel.to_parquet(
        OUTPUT_DIR / "cu_regional_cannibalization_panel.parquet", index=False
    )
    with (OUTPUT_DIR / "analysis_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "distance_bands_m": [
                    {"code": band, "label": label, "lower_inclusive": lower, "upper_exclusive": upper}
                    for band, label, lower, upper in DISTANCE_BANDS
                ],
                "event_time_monthly_diagnostic": [EVENT_MIN, EVENT_MAX_DIAGNOSTIC],
                "event_time_reference": REFERENCE_EVENT,
                "event_time_bins": [
                    {
                        "key": key,
                        "label": label,
                        "lower": lower,
                        "upper": upper,
                    }
                    for key, label, lower, upper in EVENT_BINS
                ],
                "region_mapping": ZONE_TO_REGION,
                "region_definitions": ZONE_DEFINITIONS,
                "winter_summer_holiday_models": (
                    "Not estimated: no validated winter/summer/holiday date calendar supplied. "
                    "Seasonal heterogeneity is month-of-year exposure interaction only."
                ),
                "causal_identification": (
                    "Associational fixed-effects design; nearby openings may respond to "
                    "local demand trends; repeated events overlap."
                ),
                "estimator": (
                    "Two-way store and calendar-month fixed effects; store-clustered "
                    "standard errors."
                ),
                "outcome_definitions": {
                    "APSD_income": "Sales_Income_New / Active_Days; MNT per active sales day.",
                    "APSD_customers": (
                        "Customers / Active_Days; Customers = Transaction_Count - "
                        "2 * Returned_Transaction_Count, not distinct people."
                    ),
                },
                "inference_note": (
                    "Store-clustered SE permit serial dependence within store, not "
                    "arbitrary shared opening-event dependence across stores."
                ),
                "validation_summary": validation_info,
            },
            file,
            indent=2,
        )
    create_plots(coefficients, seasonal_effects, seasonal_means)


def _print_region_readiness(regional_summary: pd.DataFrame, support: pd.DataFrame) -> None:
    print("\nRegional sample summary:")
    print(regional_summary.to_string(index=False))
    two_year = support.loc[
        (support["time_kind"] == "binned")
        & (support["event_period"].isin(["+19 to +24", "+25 months onward"]))
        & (support["distance_band"] == "0_100")
        & (support["sample"] == "all qualifying openings")
        & (support["outcome"] == "sales")
    ]
    print("\nLong-term support, nearest distance band:")
    print(
        two_year[
            [
                "region",
                "event_period",
                "distinct_pair_events",
                "exposed_existing_stores",
                "store_month_observations",
            ]
        ].to_string(index=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-diagnostics', action='store_true', help='Reproduce the original complete model suite and outputs.')
    args = parser.parse_args()
    if not args.full_diagnostics:
        # Keep historical model artifacts intact; primary-only refresh is separate.
        destination = BASE_DIR / 'outputs' / 'cumulative_evidence'
        income, distances = load_data()
        income, distances, validation, info, mapping = validate_data(income, distances)
        panel = prepare_store_month_panel(income, distances)
        events = prepare_pair_level_events(distances, panel)
        panel, _ = construct_monthly_exposure(panel, events)
        comparison, coefficients, _ = run_regional_models(panel, model_types=('cumulative',))
        destination.mkdir(parents=True, exist_ok=True)
        for outcome, name in [('sales', 'regional_sales_coefficients.csv'), ('customers', 'regional_customer_coefficients.csv')]:
            coefficients.loc[coefficients['outcome'].eq(outcome)].to_csv(destination / name, index=False)
        coefficients.to_csv(destination / 'cumulative_coefficients.csv', index=False)
        comparison.to_csv(destination / 'model_status.csv', index=False)
        validation.to_csv(destination / 'data_validation_summary.csv', index=False)
        mapping.to_csv(destination / 'region_mapping.csv', index=False)
        panel.to_parquet(destination / 'store_month_panel.parquet', index=False)
        print(f'Saved primary cumulative evidence to {destination}')
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    income, distances = load_data()
    income, distances, validation, info, mapping = validate_data(income, distances)
    panel = prepare_store_month_panel(income, distances)
    events = prepare_pair_level_events(distances, panel)
    panel, pair_months = construct_monthly_exposure(panel, events)
    regional_summary, distance_summary = create_regional_summaries(panel, events)
    event_support = construct_long_term_event_time(pair_months)
    _print_region_readiness(regional_summary, event_support)

    comparison, coefficients, pretrends = run_regional_models(panel)
    seasonal_effects, seasonal_status, seasonal_means = run_seasonal_models(panel)
    established_panel, followup_support = prepare_established_store_sample(
        panel, pair_months
    )
    established_status_rows: list[dict[str, Any]] = []
    established_effects, established_status = run_established_store_models(
        established_panel, established_status_rows
    )
    robustness, robustness_status, robustness_coefficients = run_robustness_checks(
        panel
    )
    if not robustness_coefficients.empty:
        coefficients = pd.concat(
            [coefficients, robustness_coefficients],
            ignore_index=True,
            sort=False,
        )
    event_counts = events.groupby("analysis_region").agg(
        distinct_pair_events=("event_pair_id", "nunique"),
        distinct_new_store_openings=("new_store_id", "nunique"),
        stores_with_pair_events=("existing_store_id", "nunique"),
    )
    all_regions_events = events.loc[
        events["analysis_region"].isin(REGION_ORDER)
    ]
    event_counts.loc["All Regions"] = {
        "distinct_pair_events": int(all_regions_events["event_pair_id"].nunique()),
        "distinct_new_store_openings": int(
            all_regions_events["new_store_id"].nunique()
        ),
        "stores_with_pair_events": int(
            all_regions_events["existing_store_id"].nunique()
        ),
    }
    comparison = comparison.merge(
        event_counts.reset_index().rename(columns={"analysis_region": "region"}),
        on="region",
        how="left",
    )
    model_status = pd.concat(
        [comparison, seasonal_status, established_status, pd.DataFrame(robustness_status)],
        ignore_index=True,
        sort=False,
    )

    # Robustness estimates remain in the general coefficient file so dashboard
    # users can filter them without changing the primary model outputs.
    if not seasonal_effects.empty:
        season_coefficients = seasonal_effects.rename(
            columns={
                "coefficient_log": "coefficient",
                "ci_95_lower_log": "ci_95_lower",
                "ci_95_upper_log": "ci_95_upper",
                "standard_error": "standard_error",
            }
        )
        season_coefficients["model_type"] = "seasonal_marginal_effect"
        season_coefficients["model_label"] = "Exposure effect by month of year"
        season_coefficients["term"] = season_coefficients["month_label"]
        coefficients = pd.concat(
            [coefficients, season_coefficients], ignore_index=True, sort=False
        )
    if not established_effects.empty:
        coefficients = pd.concat(
            [coefficients, established_effects], ignore_index=True, sort=False
        )

    print("\nOutcome definitions and scope:")
    print(
        "- APSD_income = monthly Sales_Income_New / Active_Days (MNT per "
        "active sales day), confirmed in 01 Cleaning StoreMaster.py."
    )
    print(
        "- APSD_customers = transaction-derived Customers / Active_Days; "
        "Customers subtracts twice returned transactions and are not "
        "deduplicated people."
    )
    print(
        "- Main log models exclude only nonpositive/missing outcomes; the "
        "asinh robustness preserves observed zero and negative values."
    )
    print(
        "- Seasonality uses month-of-year exposure interactions only. No "
        "winter/summer or holiday effect is labeled without a validated calendar."
    )
    print(
        "- Regions classify existing stores. New-store region analysis is not "
        "reported because the supplied regional crosswalk is for existing stores "
        "and no validated mapping was supplied for all new-store IDs."
    )
    print(
        "- Store-clustered standard errors do not address all dependence caused "
        "by a single new opening affecting several existing stores."
    )
    print(
        "\nModel statuses:\n"
        + model_status.groupby(
            ["model_type", "status"], dropna=False
        ).size().rename("models").reset_index().to_string(index=False)
    )

    save_outputs(
        panel=panel,
        validation=validation,
        mapping=mapping,
        regional_summary=regional_summary,
        distance_summary=distance_summary,
        event_support=event_support,
        comparison=comparison,
        coefficients=coefficients,
        seasonal_effects=seasonal_effects,
        established_effects=established_effects,
        pretrends=pretrends,
        robustness=robustness,
        model_status=model_status,
        seasonal_means=seasonal_means,
        followup_support=followup_support,
        validation_info=info,
        opening_month_summary=(
            distances[["new_store_id", "new_opening_month"]]
            .drop_duplicates("new_store_id")
            .groupby("new_opening_month")["new_store_id"]
            .nunique()
            .rename("distinct_new_store_openings")
            .reset_index()
        ),
    )
    print(f"\nSaved regional analysis outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        sys.exit(main())
