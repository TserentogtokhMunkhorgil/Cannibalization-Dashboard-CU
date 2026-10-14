"""CU store cannibalization analysis with repeated nearby-store openings.

Inputs are the two Excel workbooks in this script's directory. Outputs are
written to outputs/cannibalization_model4; input workbooks are never modified.
"""

from __future__ import annotations

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
OUTPUT_DIR = BASE_DIR / "outputs" / "cannibalization_model4"
INCOME_FILE = BASE_DIR / "store_master_data.xlsx"
DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"
SHEET_NAME = "Sheet1"

DISTANCE_BANDS = [
    ("0_100", "0-100m", 0.0, 100.0),
    ("100_250", "100-250m", 100.0, 250.0),
    ("250_500", "250-500m", 250.0, 500.0),
    ("500_1000", "500-1000m", 500.0, 1000.0),
]
EVENT_TIMES = tuple(range(-6, 7))
REFERENCE_EVENT_TIME = -1
PAIR_KEY = ["existing_store_id", "new_store_id"]


class InsufficientModelSupport(RuntimeError):
    """Raised when the requested fixed-effect design is not estimable."""


def progress(message: str) -> None:
    print(f"\n[CU cannibalization] {message}", flush=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the supplied Excel inputs without dropping or rewriting rows."""
    for path in (INCOME_FILE, DISTANCE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input workbook was not found: {path}")

    income = pd.read_excel(INCOME_FILE, sheet_name=SHEET_NAME)
    distances = pd.read_excel(DISTANCE_FILE, sheet_name=SHEET_NAME)
    return income, distances


def _require_columns(
    frame: pd.DataFrame, required: set[str], frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}. "
            f"Available columns: {frame.columns.tolist()}"
        )


def _distance_band(distance: float) -> str | None:
    for name, _, lower, upper in DISTANCE_BANDS:
        if lower <= distance < upper:
            return name
    if distance >= 1000:
        return "gt_1000"
    return None


def validate_data(
    income: pd.DataFrame, distances: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate source structure, report duplicates/missingness, and normalize types."""
    progress("Validating source rows, keys, dates, and distance coverage")

    income_required = {
        "SalesMonth",
        "store_id",
        "APSD_income",
        "ost_dt",
        "loc_area_new",
        "LATITUDE",
        "LONGITUDE",
    }
    distance_required = {
        "existing_store_id",
        "new_store_id",
        "distance_m",
        "existing_opening_date",
        "new_opening_date",
        "distance_band",
    }
    _require_columns(income, income_required, "store_master_data.xlsx")
    _require_columns(distances, distance_required, "store_to_store_distance.xlsx")

    income = income.copy()
    distances = distances.copy()
    for column in ("SalesMonth", "ost_dt"):
        income[column] = pd.to_datetime(income[column], errors="coerce")
    for column in ("existing_opening_date", "new_opening_date"):
        distances[column] = pd.to_datetime(distances[column], errors="coerce")
    for column in ("APSD_income", "LATITUDE", "LONGITUDE"):
        income[column] = pd.to_numeric(income[column], errors="coerce")
    distances["distance_m"] = pd.to_numeric(distances["distance_m"], errors="coerce")
    income["SalesMonth"] = income["SalesMonth"].dt.to_period("M").dt.to_timestamp()

    duplicate_store_month_rows = int(
        income.duplicated(["store_id", "SalesMonth"], keep=False).sum()
    )
    duplicate_store_month_keys = int(
        income.duplicated(["store_id", "SalesMonth"]).sum()
    )
    duplicate_pair_rows = int(distances.duplicated(PAIR_KEY, keep=False).sum())
    duplicate_pair_keys = int(distances.duplicated(PAIR_KEY).sum())

    progress(
        "Source counts: "
        f"income={len(income):,} rows / {income['store_id'].nunique():,} stores; "
        f"distances={len(distances):,} rows / "
        f"{distances['existing_store_id'].nunique():,} existing IDs / "
        f"{distances['new_store_id'].nunique():,} new IDs"
    )
    print(
        "Duplicate keys: "
        f"store-month rows={duplicate_store_month_rows:,} "
        f"(extra keys={duplicate_store_month_keys:,}); "
        f"store-pair rows={duplicate_pair_rows:,} "
        f"(extra keys={duplicate_pair_keys:,})"
    )
    print(
        "Income dates: "
        f"{income['SalesMonth'].min()} to {income['SalesMonth'].max()}; "
        "new opening dates: "
        f"{distances['new_opening_date'].min()} to "
        f"{distances['new_opening_date'].max()}"
    )
    print(
        "Missing values in required fields:\n"
        + pd.concat(
            {
                "income": income[list(income_required)].isna().sum(),
                "distance": distances[list(distance_required)].isna().sum(),
            },
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    if duplicate_store_month_keys or duplicate_pair_keys:
        raise ValueError(
            "Duplicate source keys were found and have not been removed. "
            "Resolve the reported store-month or existing/new-store pair "
            "duplicates, then rerun."
        )
    if income[["store_id", "SalesMonth"]].isna().any().any():
        raise ValueError("Income data has missing store IDs or invalid SalesMonth dates.")
    if distances[PAIR_KEY + ["distance_m", "new_opening_date"]].isna().any().any():
        raise ValueError(
            "Distance data has missing pair IDs, invalid distances, or invalid "
            "new-store opening dates."
        )
    if (distances["distance_m"] < 0).any():
        raise ValueError("Negative pair distances are invalid.")

    distances["distance_group"] = distances["distance_m"].map(_distance_band)
    distances["distance_group_label"] = distances["distance_group"].map(
        {name: label for name, label, _, _ in DISTANCE_BANDS}
        | {"gt_1000": ">1000m / control"}
    )
    distances["new_opening_month"] = (
        distances["new_opening_date"].dt.to_period("M")
    )
    distances["event_pair_id"] = np.arange(len(distances), dtype=np.int64)

    invalid_bands = int(distances["distance_group"].isna().sum())
    print(
        "Source distance_band labels:\n"
        + distances["distance_band"].value_counts(dropna=False).to_string()
    )
    print(
        "Recomputed requested distance bands:\n"
        + distances["distance_group_label"]
        .value_counts(dropna=False)
        .to_string()
    )
    if invalid_bands:
        print(
            f"Distance pairs outside the modeled 0-1000m bands: {invalid_bands:,}; "
            "these are retained as >1000m controls, not included in treated bands."
        )

    pair_date_counts = distances.groupby("existing_store_id")[
        "existing_opening_date"
    ].nunique(dropna=True)
    pair_date_conflicts = int((pair_date_counts > 1).sum())
    new_date_counts = distances.groupby("new_store_id")["new_opening_date"].nunique()
    new_date_conflicts = int((new_date_counts > 1).sum())
    if pair_date_conflicts or new_date_conflicts:
        raise ValueError(
            "Conflicting opening dates found for "
            f"{pair_date_conflicts} existing stores or {new_date_conflicts} new stores."
        )

    info: dict[str, Any] = {
        "income_rows": len(income),
        "income_stores": int(income["store_id"].nunique()),
        "income_months": int(income["SalesMonth"].nunique()),
        "distance_rows": len(distances),
        "existing_ids_in_distance_file": int(
            distances["existing_store_id"].nunique()
        ),
        "new_ids_in_distance_file": int(distances["new_store_id"].nunique()),
        "duplicate_store_month_rows": duplicate_store_month_rows,
        "duplicate_store_pair_rows": duplicate_pair_rows,
        "invalid_distance_band_rows": invalid_bands,
        "missing_open_date_rows": int(income["ost_dt"].isna().sum()),
        "nonpositive_income_rows": int((income["APSD_income"] <= 0).sum()),
        "stores_with_time_varying_loc_area": int(
            (income.groupby("store_id")["loc_area_new"].nunique(dropna=True) > 1).sum()
        ),
        "income_start": income["SalesMonth"].min(),
        "income_end": income["SalesMonth"].max(),
    }
    return income, distances, info


def prepare_store_month_panel(
    income: pd.DataFrame, distances: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Build exactly one observation per existing store and calendar month."""
    progress("Building the existing-store monthly panel")
    panel = income[
        [
            "store_id",
            "SalesMonth",
            "APSD_income",
            "ost_dt",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
        ]
    ].copy()
    panel = panel.rename(
        columns={
            "store_id": "existing_store_id",
            "ost_dt": "existing_opening_date",
        }
    )

    opening_dates = (
        distances[["existing_store_id", "existing_opening_date"]]
        .dropna(subset=["existing_opening_date"])
        .drop_duplicates("existing_store_id")
    )
    panel = panel.merge(
        opening_dates,
        on="existing_store_id",
        how="left",
        suffixes=("", "_distance"),
        validate="many_to_one",
    )
    missing_before_fill = panel["existing_opening_date"].isna()
    panel.loc[missing_before_fill, "existing_opening_date"] = panel.loc[
        missing_before_fill, "existing_opening_date_distance"
    ]
    filled_opening_dates = int(
        (missing_before_fill & panel["existing_opening_date"].notna()).sum()
    )
    panel = panel.drop(columns=["existing_opening_date_distance"])

    panel["calendar_month"] = panel["SalesMonth"].dt.to_period("M")
    panel["log_APSD_income"] = np.where(
        panel["APSD_income"] > 0,
        np.log(panel["APSD_income"].where(panel["APSD_income"] > 0)),
        np.nan,
    )
    panel["valid_coordinates"] = (
        panel["LATITUDE"].between(-90, 90)
        & panel["LONGITUDE"].between(-180, 180)
    )

    if panel.duplicated(["existing_store_id", "SalesMonth"]).any():
        raise AssertionError("Store-month panel is not unique after construction.")
    if panel["calendar_month"].isna().any():
        raise AssertionError("Calendar-month conversion produced missing values.")

    print(
        f"Panel: {len(panel):,} store-month rows, "
        f"{panel['existing_store_id'].nunique():,} stores, "
        f"{panel['SalesMonth'].nunique():,} calendar months"
    )
    print(
        f"Positive APSD_income: {panel['log_APSD_income'].notna().sum():,}; "
        f"zero/negative: {panel['log_APSD_income'].isna().sum():,} "
        "(kept in panel, excluded from log-outcome regressions)"
    )
    print(
        f"Opening dates recovered from pair metadata: {filled_opening_dates:,}; "
        f"still missing: {panel['existing_opening_date'].isna().sum():,}"
    )
    print(
        f"Stores with valid coordinates: "
        f"{panel.loc[panel['valid_coordinates'], 'existing_store_id'].nunique():,}"
    )
    return panel, filled_opening_dates


def _make_event_rows(
    panel: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Expand each pair to observed panel months at each requested event time."""
    observed_keys = panel[["existing_store_id", "SalesMonth"]]
    outputs: list[pd.DataFrame] = []
    for event_time in EVENT_TIMES:
        shifted = pairs[
            [
                "event_pair_id",
                "existing_store_id",
                "new_store_id",
                "distance_group",
                "new_opening_month",
            ]
        ].copy()
        shifted["SalesMonth"] = (
            shifted["new_opening_month"] + event_time
        ).dt.to_timestamp()
        shifted["event_time"] = event_time
        outputs.append(
            shifted.merge(
                observed_keys,
                on=["existing_store_id", "SalesMonth"],
                how="inner",
                validate="many_to_one",
            )
        )
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _make_event_cell_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band_name, band_label, _, _ in DISTANCE_BANDS:
        for event_time in EVENT_TIMES:
            cell = event_rows.loc[
                (event_rows["distance_group"] == band_name)
                & (event_rows["event_time"] == event_time)
            ]
            rows.append(
                {
                    "event_time": event_time,
                    "distance_band": band_label,
                    "pair_events": int(cell["event_pair_id"].nunique()),
                    "exposure_pair_months": int(len(cell)),
                    "store_month_observations": int(
                        cell[["existing_store_id", "SalesMonth"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "existing_stores": int(cell["existing_store_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _add_cumulative_exposures(
    panel: pd.DataFrame, distances: pd.DataFrame
) -> pd.DataFrame:
    """Count all distance-band openings that have occurred by each month."""
    result = panel.copy()
    result["month_index"] = (
        result["SalesMonth"].dt.year * 12 + result["SalesMonth"].dt.month
    )
    result["all_dist_event_count"] = 0

    for band_name, _, _, _ in DISTANCE_BANDS:
        result[f"exposure_{band_name}_opened"] = 0

    modeled_bands = [name for name, _, _, _ in DISTANCE_BANDS]
    treated_pairs = distances.loc[distances["distance_group"].isin(modeled_bands)]
    events_by_store: dict[Any, pd.DataFrame] = {
        store_id: group
        for store_id, group in treated_pairs.groupby("existing_store_id")
    }
    for store_id, row_indexes in result.groupby("existing_store_id").groups.items():
        store_events = events_by_store.get(store_id)
        if store_events is None:
            continue
        month_indexes = result.loc[row_indexes, "month_index"].to_numpy()
        for band_name, _, _, _ in DISTANCE_BANDS:
            opening_months = store_events.loc[
                store_events["distance_group"] == band_name,
                "new_opening_month",
            ]
            opening_indexes = (
                opening_months.dt.year * 12 + opening_months.dt.month
            ).to_numpy(dtype=np.int64)
            opening_indexes.sort()
            result.loc[row_indexes, f"exposure_{band_name}_opened"] = np.searchsorted(
                opening_indexes, month_indexes, side="right"
            )
        result.loc[row_indexes, "all_dist_event_count"] = result.loc[
            row_indexes,
            [f"exposure_{name}_opened" for name, _, _, _ in DISTANCE_BANDS],
        ].sum(axis=1)
    return result.drop(columns=["month_index"])


def create_exposure_variables(
    panel: pd.DataFrame, distances: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate repeated pair-level openings into one store-month exposure panel."""
    progress("Creating cumulative and distance-by-event-time exposures")
    modeled_bands = [name for name, _, _, _ in DISTANCE_BANDS]
    event_pairs = distances.loc[
        distances["distance_group"].isin(modeled_bands)
    ].copy()
    event_rows = _make_event_rows(panel, event_pairs)
    event_summary = _make_event_cell_summary(event_rows)

    if len(event_rows):
        exposure_counts = (
            event_rows.groupby(
                [
                    "existing_store_id",
                    "SalesMonth",
                    "distance_group",
                    "event_time",
                ],
                observed=True,
            )
            .size()
            .unstack(["distance_group", "event_time"], fill_value=0)
        )
        desired_columns = pd.MultiIndex.from_product(
            [
                [name for name, _, _, _ in DISTANCE_BANDS],
                EVENT_TIMES,
            ],
            names=["distance_group", "event_time"],
        )
        exposure_counts = exposure_counts.reindex(
            columns=desired_columns, fill_value=0
        )
        exposure_counts.columns = [
            f"exposure_{band}_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
            for band, _, _, _ in DISTANCE_BANDS
            for k in EVENT_TIMES
        ]
        exposure_counts = exposure_counts.reset_index()
        panel = panel.merge(
            exposure_counts,
            on=["existing_store_id", "SalesMonth"],
            how="left",
            validate="one_to_one",
        )

    event_feature_columns = [
        f"exposure_{band}_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        for band, _, _, _ in DISTANCE_BANDS
        for k in EVENT_TIMES
    ]
    for column in event_feature_columns:
        if column not in panel.columns:
            panel[column] = 0
    panel[event_feature_columns] = panel[event_feature_columns].fillna(0).astype(int)

    for event_time in EVENT_TIMES:
        event_column = (
            f"all_dist_event_{'m' + str(abs(event_time)) if event_time < 0 else 'p' + str(event_time)}"
        )
        distance_columns = [
            f"exposure_{band}_event_{'m' + str(abs(event_time)) if event_time < 0 else 'p' + str(event_time)}"
            for band, _, _, _ in DISTANCE_BANDS
        ]
        panel[event_column] = panel[distance_columns].sum(axis=1)

    panel = _add_cumulative_exposures(panel, distances)
    if panel.duplicated(["existing_store_id", "SalesMonth"]).any():
        raise AssertionError("Exposure construction duplicated a store-month row.")

    distance_summary = create_distance_exposure_summary(panel, distances)
    print(
        f"Pair-level events within 1 km: {len(event_pairs):,}; "
        f"pair-event/month exposures observed: {len(event_rows):,}"
    )
    return panel, distance_summary, event_summary


def create_distance_exposure_summary(
    panel: pd.DataFrame, distances: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_bands = [
        (name, label, lower, upper) for name, label, lower, upper in DISTANCE_BANDS
    ] + [("gt_1000", ">1000m / control", 1000.0, np.inf)]
    for band_name, band_label, _, _ in all_bands:
        pairs = distances.loc[distances["distance_group"] == band_name]
        exposure_column = f"exposure_{band_name}_opened"
        opened = (
            panel[exposure_column] > 0
            if exposure_column in panel.columns
            else pd.Series(False, index=panel.index)
        )
        rows.append(
            {
                "distance_band": band_label,
                "distance_pairs": int(len(pairs)),
                "pair_events_in_panel_period": int(
                    pairs["new_opening_month"].between(
                        panel["calendar_month"].min(),
                        panel["calendar_month"].max(),
                    ).sum()
                ),
                "new_stores_in_band": int(pairs["new_store_id"].nunique()),
                "existing_stores_in_band": int(
                    pairs["existing_store_id"].nunique()
                ),
                "existing_stores_with_opening_in_panel_period": int(
                    pairs.loc[
                        pairs["new_opening_month"].between(
                            panel["calendar_month"].min(),
                            panel["calendar_month"].max(),
                        ),
                        "existing_store_id",
                    ].nunique()
                ),
                "store_month_observations_with_exposure": int(opened.sum()),
                "unique_exposed_stores_in_panel": int(
                    panel.loc[opened, "existing_store_id"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def _store_support_counts(
    panel: pd.DataFrame, distances: pd.DataFrame
) -> dict[str, int]:
    valid_panel = panel.loc[panel["log_APSD_income"].notna()]
    valid_months: dict[Any, set[Any]] = {
        store_id: set(group["calendar_month"])
        for store_id, group in valid_panel.groupby("existing_store_id")
    }
    pairs = distances.loc[distances["distance_m"] < 1000]
    pair_months = pairs.groupby("existing_store_id")["new_opening_month"].apply(
        list
    )
    exposed_ids = set(pair_months.index)

    pre_6: set[Any] = set()
    post_6: set[Any] = set()
    both: set[Any] = set()
    for store_id, opening_months in pair_months.items():
        observed = valid_months.get(store_id, set())
        has_pre = any(
            all(opening - offset in observed for offset in range(1, 7))
            for opening in opening_months
        )
        has_post = any(
            all(opening + offset in observed for offset in range(0, 6))
            for opening in opening_months
        )
        if has_pre:
            pre_6.add(store_id)
        if has_post:
            post_6.add(store_id)
        if has_pre and has_post:
            both.add(store_id)
    return {
        "total_existing_stores": int(panel["existing_store_id"].nunique()),
        "existing_stores_with_valid_coordinates": int(
            panel.loc[panel["valid_coordinates"], "existing_store_id"].nunique()
        ),
        "existing_stores_with_valid_sales": int(
            valid_panel["existing_store_id"].nunique()
        ),
        "existing_stores_with_6_valid_pre_months": len(pre_6),
        "existing_stores_with_6_valid_post_months": len(post_6),
        "existing_stores_with_6_pre_and_post_months": len(both),
        "existing_stores_exposed_within_1km": len(exposed_ids),
        "existing_stores_exposed_within_100m": int(
            distances.loc[distances["distance_m"] < 100, "existing_store_id"].nunique()
        ),
        "existing_stores_exposed_within_250m": int(
            distances.loc[distances["distance_m"] < 250, "existing_store_id"].nunique()
        ),
        "existing_stores_exposed_within_500m": int(
            distances.loc[distances["distance_m"] < 500, "existing_store_id"].nunique()
        ),
    }


def create_overlap_summary(
    distances: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Describe repeated openings by existing store; preserve all pair events."""
    nearby = distances.loc[distances["distance_m"] < 1000].copy()
    nearby = nearby.drop_duplicates(["existing_store_id", "new_store_id"])
    grouped = nearby.groupby("existing_store_id")
    summary = grouped.agg(
        nearby_new_stores=("new_store_id", "nunique"),
        minimum_distance_m=("distance_m", "min"),
        opening_dates=("new_opening_date", "nunique"),
    ).reset_index()
    summary = summary.loc[summary["nearby_new_stores"] >= 2].copy()

    gaps: list[float] = []
    for _, group in nearby.groupby("existing_store_id"):
        opening_dates = sorted(group["new_opening_date"].tolist())
        if len(opening_dates) > 1:
            gaps.extend(
                float((right - left).days / 30.4375)
                for left, right in zip(opening_dates, opening_dates[1:])
            )
    details: dict[str, Any] = {
        "stores_with_2plus_openings_within_1km": int(len(summary)),
        "maximum_nearby_new_stores_for_one_existing_store": int(
            nearby.groupby("existing_store_id")["new_store_id"].nunique().max()
        )
        if len(nearby)
        else 0,
        "median_months_between_consecutive_openings": float(np.median(gaps))
        if gaps
        else np.nan,
        "minimum_months_between_consecutive_openings": float(min(gaps))
        if gaps
        else np.nan,
    }
    return summary.sort_values(
        ["nearby_new_stores", "existing_store_id"], ascending=[False, True]
    ), details


def create_diagnostics(
    panel: pd.DataFrame,
    distances: pd.DataFrame,
    info: dict[str, Any],
    distance_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    filled_opening_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Print complete design support and save diagnostics before fitting models."""
    progress("Reviewing sample, overlap, and event-time support before estimation")
    support = _store_support_counts(panel, distances)
    overlap_table, overlap_metrics = create_overlap_summary(distances)
    info.update(support)
    info.update(overlap_metrics)
    info["existing_opening_dates_filled_from_pairs"] = filled_opening_dates
    info["stores_with_2plus_nearby_openings"] = overlap_metrics[
        "stores_with_2plus_openings_within_1km"
    ]
    info["all_new_store_openings"] = int(distances["new_store_id"].nunique())
    info["all_existing_new_store_pair_events"] = len(distances)
    modeled_bands = [name for name, _, _, _ in DISTANCE_BANDS]
    info["distance_pair_events_within_1km"] = int(
        distances["distance_group"].isin(modeled_bands).sum()
    )
    info["stores_with_any_nearby_opening"] = int(
        distances.loc[
            distances["distance_group"].isin(modeled_bands), "existing_store_id"
        ].nunique()
    )
    info["valid_log_outcome_observations"] = int(
        panel["log_APSD_income"].notna().sum()
    )
    info["nonpositive_income_observations"] = int(
        (panel["APSD_income"] <= 0).sum()
    )
    info["window_min_event_time"] = min(EVENT_TIMES)
    info["window_max_event_time"] = max(EVENT_TIMES)
    info["omitted_reference_event_time"] = REFERENCE_EVENT_TIME

    summary_rows = [
        {"metric": key, "value": value}
        for key, value in info.items()
    ]
    sample_summary = pd.DataFrame(summary_rows)
    sample_summary["value"] = sample_summary["value"].astype(str)

    print("Sample summary:")
    print(sample_summary.to_string(index=False))
    print("\nDistance exposure summary:")
    print(distance_summary.to_string(index=False))
    print("\nPair-event / event-time support (each cell):")
    print(
        event_summary.pivot(
            index="event_time",
            columns="distance_band",
            values="pair_events",
        ).to_string()
    )
    print("\nStore-month observations with nonzero exposure (each cell):")
    print(
        event_summary.pivot(
            index="event_time",
            columns="distance_band",
            values="store_month_observations",
        ).to_string()
    )
    print("\nOpening-month counts (distinct new-store IDs):")
    openings_by_month = (
        distances[["new_store_id", "new_opening_month"]]
        .drop_duplicates("new_store_id")
        .groupby("new_opening_month")["new_store_id"]
        .nunique()
        .rename("new_store_openings")
        .reset_index()
    )
    print(openings_by_month.to_string(index=False))
    print(
        "\nRepeated openings: "
        f"{len(overlap_table):,} existing stores have 2+ new stores within 1 km; "
        f"maximum={overlap_metrics['maximum_nearby_new_stores_for_one_existing_store']}; "
        f"median gap between consecutive opening events="
        f"{overlap_metrics['median_months_between_consecutive_openings']:.1f} months."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_summary.to_csv(OUTPUT_DIR / "model_sample_summary.csv", index=False)
    distance_summary.to_csv(
        OUTPUT_DIR / "distance_exposure_summary.csv", index=False
    )
    event_summary.to_csv(
        OUTPUT_DIR / "event_time_exposure_summary.csv", index=False
    )
    overlap_table.to_csv(OUTPUT_DIR / "overlapping_openings.csv", index=False)
    openings_by_month.to_csv(
        OUTPUT_DIR / "new_store_openings_by_month.csv", index=False
    )
    return sample_summary, overlap_table


def _group_means(
    values: np.ndarray, codes: np.ndarray, group_count: int
) -> np.ndarray:
    sums = np.zeros((group_count, values.shape[1]), dtype=float)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=group_count).astype(float)
    return sums[codes] / counts[codes, None]


def absorb_store_and_month_effects(
    values: np.ndarray,
    store_codes: np.ndarray,
    month_codes: np.ndarray,
    store_count: int,
    month_count: int,
) -> np.ndarray:
    """Absorb two-way fixed effects by alternating projections."""
    residualized = np.asarray(values, dtype=float).copy()
    if residualized.ndim == 1:
        residualized = residualized[:, None]
    for _ in range(2000):
        previous = residualized.copy()
        residualized -= _group_means(residualized, store_codes, store_count)
        residualized -= _group_means(residualized, month_codes, month_count)
        if np.max(np.abs(residualized - previous)) < 1e-9:
            return residualized
    raise InsufficientModelSupport(
        "Two-way fixed-effect absorption did not converge after 2,000 "
        "alternating-projection iterations."
    )


def _fit_twfe(
    model_name: str,
    panel: pd.DataFrame,
    regressors: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit absorbed store/month FE with store-clustered standard errors."""
    sample = panel.loc[panel["log_APSD_income"].notna()].copy()
    if sample["existing_store_id"].nunique() < 2:
        raise InsufficientModelSupport(f"{model_name} has fewer than two stores.")
    if sample["calendar_month"].nunique() < 2:
        raise InsufficientModelSupport(f"{model_name} has fewer than two months.")

    store_codes, store_levels = pd.factorize(sample["existing_store_id"])
    month_codes, month_levels = pd.factorize(sample["calendar_month"])
    y = sample["log_APSD_income"].to_numpy(dtype=float)
    y_within = absorb_store_and_month_effects(
        y, store_codes, month_codes, len(store_levels), len(month_levels)
    )[:, 0]
    total_sum_squares = float(np.dot(y_within, y_within))
    pretrend_statistic = np.nan
    pretrend_p_value = np.nan
    pretrend_degrees = 0

    if regressors:
        x = sample[regressors].to_numpy(dtype=float)
        x_within = absorb_store_and_month_effects(
            x, store_codes, month_codes, len(store_levels), len(month_levels)
        )
        rank = int(np.linalg.matrix_rank(x_within))
        if rank != x_within.shape[1]:
            raise InsufficientModelSupport(
                f"{model_name} is not identified after store/month fixed effects: "
                f"{x_within.shape[1]} requested exposure terms but rank={rank}. "
                "Reduce or combine unsupported distance/event-time cells; "
                "no terms were silently dropped."
            )
        if len(sample) <= rank:
            raise InsufficientModelSupport(
                f"{model_name} has no residual degrees of freedom."
            )
        fitted = sm.OLS(y_within, x_within, hasconst=False).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["existing_store_id"].to_numpy()},
        )
        pretrend_indexes = [
            index
            for index, term in enumerate(regressors)
            if _event_time_from_term(term) in {-6, -5, -4, -3, -2}
        ]
        if pretrend_indexes:
            restrictions = np.zeros((len(pretrend_indexes), len(regressors)))
            restrictions[np.arange(len(pretrend_indexes)), pretrend_indexes] = 1
            pretrend_test = fitted.wald_test(
                restrictions, use_f=True, scalar=True
            )
            pretrend_statistic = float(np.asarray(pretrend_test.statistic).item())
            pretrend_p_value = float(np.asarray(pretrend_test.pvalue).item())
            pretrend_degrees = len(pretrend_indexes)
        residual_sum_squares = float(np.dot(fitted.resid, fitted.resid))
        coefficient_rows = []
        for index, variable in enumerate(regressors):
            estimate = float(fitted.params[index])
            standard_error = float(fitted.bse[index])
            lower, upper = fitted.conf_int()[index]
            coefficient_rows.append(
                {
                    "model": model_name,
                    "term": variable,
                    "coefficient": estimate,
                    "standard_error": standard_error,
                    "p_value": float(fitted.pvalues[index]),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "percent_effect": float(100 * np.expm1(estimate)),
                    "percent_ci_95_lower": float(100 * np.expm1(lower)),
                    "percent_ci_95_upper": float(100 * np.expm1(upper)),
                    "treated_stores": int(
                        sample.loc[sample[variable] > 0, "existing_store_id"].nunique()
                    ),
                    "exposure_pair_months": int(sample[variable].sum()),
                }
            )
        coefficients = pd.DataFrame(coefficient_rows)
        rank_value = rank
    else:
        residual_sum_squares = total_sum_squares
        coefficients = pd.DataFrame()
        rank_value = 0

    within_r2 = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    model_summary = {
        "model": model_name,
        "observations": len(sample),
        "stores": int(sample["existing_store_id"].nunique()),
        "calendar_months": int(sample["calendar_month"].nunique()),
        "exposure_terms": len(regressors),
        "absorbed_design_rank": rank_value,
        "within_r_squared": within_r2,
        "within_residual_sum_squares": residual_sum_squares,
        "pretrend_joint_f_statistic": pretrend_statistic,
        "pretrend_joint_p_value": pretrend_p_value,
        "pretrend_restrictions": pretrend_degrees,
        "clustered_by": "existing_store_id",
        "estimator": "Two-way FE absorbed by alternating projections; OLS with store-clustered covariance",
        "pair_events_in_sales_period": int(sample["all_dist_event_p0"].sum())
        if "all_dist_event_p0" in sample.columns
        else 0,
        "stores_with_exposure": int(
            sample.loc[
                sample[regressors].gt(0).any(axis=1), "existing_store_id"
            ].nunique()
        )
        if regressors
        else 0,
    }
    return model_summary, coefficients


def run_model_2(panel: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate distance-specific cumulative counts of opened new stores."""
    regressors = [
        f"exposure_{band}_opened" for band, _, _, _ in DISTANCE_BANDS
    ]
    return _fit_twfe("B", panel, regressors)


def run_model_3(panel: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate an all-distance multiple-event event study, omitting event -1."""
    regressors = [
        f"all_dist_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        for k in EVENT_TIMES
        if k != REFERENCE_EVENT_TIME
    ]
    return _fit_twfe("C", panel, regressors)


def run_model_4(panel: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate distance-by-event-time effects, omitting event -1 per band."""
    regressors = [
        f"exposure_{band}_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        for band, _, _, _ in DISTANCE_BANDS
        for k in EVENT_TIMES
        if k != REFERENCE_EVENT_TIME
    ]
    return _fit_twfe("D", panel, regressors)


def run_alternative_distance_model(
    panel: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Combine the two nearest bands into a 0-250m robustness specification."""
    alt_bands = ("0_250", "250_500", "500_1000")
    for event_time in EVENT_TIMES:
        suffix = "m" + str(abs(event_time)) if event_time < 0 else "p" + str(event_time)
        nearest_100 = f"exposure_0_100_event_{suffix}"
        nearest_250 = f"exposure_100_250_event_{suffix}"
        panel[f"exposure_0_250_event_{suffix}"] = (
            panel[nearest_100] + panel[nearest_250]
        )
    regressors = [
        f"exposure_{band}_event_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        for band in alt_bands
        for k in EVENT_TIMES
        if k != REFERENCE_EVENT_TIME
    ]
    return _fit_twfe("D_alt_0_250", panel, regressors)


def run_robustness_checks(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit A-D on the same positive-income sample for comparable diagnostics."""
    progress("Estimating A-D fixed-effect specifications")
    baseline, baseline_coefficients = _fit_twfe("A", panel, [])
    model_b, coefficients_b = run_model_2(panel)
    model_c, coefficients_c = run_model_3(panel)
    model_d, coefficients_d = run_model_4(panel)
    model_d_alt, coefficients_d_alt = run_alternative_distance_model(panel)
    comparison = pd.DataFrame(
        [baseline, model_b, model_c, model_d, model_d_alt]
    )
    coefficients = pd.concat(
        [
            baseline_coefficients,
            coefficients_b,
            coefficients_c,
            coefficients_d,
            coefficients_d_alt,
        ],
        ignore_index=True,
    )
    return comparison, coefficients


def _event_time_from_term(term: str) -> int | None:
    if "_event_" not in term:
        return None
    suffix = term.rsplit("_event_", 1)[1]
    if suffix.startswith("m"):
        return -int(suffix[1:])
    if suffix.startswith("p"):
        return int(suffix[1:])
    return None


def create_plots(coefficients: pd.DataFrame) -> None:
    """Create event-study plots with store-clustered 95% confidence intervals."""
    progress("Creating event-study figures")
    model_c = coefficients.loc[coefficients["model"] == "C"].copy()
    model_c["event_time"] = model_c["term"].map(_event_time_from_term)
    model_c = model_c.sort_values("event_time")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.errorbar(
        model_c["event_time"],
        model_c["coefficient"],
        yerr=[
            model_c["coefficient"] - model_c["ci_95_lower"],
            model_c["ci_95_upper"] - model_c["coefficient"],
        ],
        fmt="o-",
        capsize=3,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="red", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Months relative to new-store opening",
        ylabel="Change in log APSD income",
        title="Model C: repeated-event study, all distances within 1 km",
        xticks=list(EVENT_TIMES),
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model4_event_study_plot.png", dpi=180)
    plt.close(fig)

    model_d = coefficients.loc[coefficients["model"] == "D"].copy()
    model_d["event_time"] = model_d["term"].map(_event_time_from_term)
    fig, axis = plt.subplots(figsize=(10, 6))
    for band_name, band_label, _, _ in DISTANCE_BANDS:
        prefix = f"exposure_{band_name}_event_"
        series = model_d.loc[model_d["term"].str.startswith(prefix)].sort_values(
            "event_time"
        )
        axis.errorbar(
            series["event_time"],
            series["coefficient"],
            yerr=[
                series["coefficient"] - series["ci_95_lower"],
                series["ci_95_upper"] - series["coefficient"],
            ],
            fmt="o-",
            capsize=2,
            label=band_label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="red", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Months relative to new-store opening",
        ylabel="Change in log APSD income per opening",
        title="Model D: distance x event time (event -1 omitted)",
        xticks=list(EVENT_TIMES),
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model4_distance_event_plot.png", dpi=180)
    plt.close(fig)


def save_outputs(
    panel: pd.DataFrame,
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    progress("Saving analysis dataset and model results")
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(
        OUTPUT_DIR / "all_model_coefficients.csv", index=False
    )
    coefficients.loc[coefficients["model"] == "D"].to_csv(
        OUTPUT_DIR / "model4_coefficients.csv", index=False
    )
    panel.to_parquet(
        OUTPUT_DIR / "cu_cannibalization_model4_panel.parquet", index=False
    )
    create_plots(coefficients)


def _explain_model_comparison(comparison: pd.DataFrame) -> None:
    print("\nModel comparison:")
    print(
        comparison[
            [
                "model",
                "observations",
                "stores",
                "calendar_months",
                "exposure_terms",
                "within_r_squared",
                "stores_with_exposure",
                "pair_events_in_sales_period",
                "pretrend_joint_f_statistic",
                "pretrend_joint_p_value",
            ]
        ].to_string(index=False)
    )
    print(
        "\nInterpretation and estimator limits:\n"
        "- Model B is the cumulative distance-exposure business summary; "
        "coefficients are per additional opened store in each distance band.\n"
        "- Models C and D are repeated-event studies; each nearby opening is "
        "counted, rather than selecting only the nearest store.\n"
        "- Model D coefficients are relative to event month -1, conditional on "
        "store and calendar-month fixed effects. Percent effect is "
        "100*(exp(beta)-1), per additional pair-event exposure.\n"
        "- Use Model B as the primary business summary; use Model D as the "
        "distance-specific research specification, with Model C and "
        "D_alt_0_250 as event-study and distance sensitivity checks.\n"
        "- Store-clustered confidence intervals allow arbitrary serial "
        "correlation within each existing store.\n"
        "- loc_area_new is retained in the analysis panel. It does not receive "
        "a separate coefficient when it is time-invariant because store fixed "
        "effects absorb it.\n"
        "- Models C, D, and D_alt report joint tests that the -6 through -2 "
        "lead coefficients are all zero; inspect these p-values alongside "
        "the plotted leads.\n"
        "- Individual Model D p-values are not adjusted for testing 48 "
        "distance-by-event-time coefficients; treat them as exploratory.\n"
        "- D_alt_0_250 combines the 0-100m and 100-250m bands as a distance "
        "definition sensitivity check; it does not replace the prespecified "
        "four-band Model D.\n"
        "- Sun-Abraham single-adoption estimators do not directly fit this "
        "repeated-treatment setup. Use these as conditional repeated-event "
        "associations, not as a causal estimator that automatically resolves "
        "staggered adoption or heterogeneous effects. A causal extension "
        "requires a design that explicitly handles multiple exposures and "
        "overlapping event windows."
    )
    pretrend_rows = comparison.loc[
        comparison["model"].isin(["C", "D", "D_alt_0_250"])
        & comparison["pretrend_joint_p_value"].notna()
    ]
    rejected = pretrend_rows.loc[
        pretrend_rows["pretrend_joint_p_value"] < 0.05, "model"
    ].tolist()
    if rejected:
        print(
            "WARNING: the joint pre-trend test rejects zero leads at 5% for "
            f"{', '.join(rejected)}. Parallel trends are not supported by these "
            "data; do not interpret the post-opening estimates as causal "
            "cannibalization effects without a stronger identification design."
        )
    else:
        print(
            "The joint pre-trend test does not reject zero leads at 5%; this "
            "does not by itself establish causal identification."
        )
    d_terms = comparison.loc[comparison["model"] == "D", "exposure_terms"]
    if len(d_terms) and int(d_terms.iloc[0]) < 4 * (len(EVENT_TIMES) - 1):
        print(
            "WARNING: not all requested Model D exposure terms entered the "
            "fit; review rank/support diagnostics before interpretation."
        )


def main() -> int:
    progress("Starting Excel-based analysis (no ClickHouse connection or embedded credentials)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    income, distances = load_data()
    income, distances, info = validate_data(income, distances)
    panel, filled_dates = prepare_store_month_panel(income, distances)
    panel, distance_summary, event_summary = create_exposure_variables(
        panel, distances
    )
    panel.to_parquet(
        OUTPUT_DIR / "cu_cannibalization_model4_panel.parquet", index=False
    )
    create_diagnostics(
        panel,
        distances,
        info,
        distance_summary,
        event_summary,
        filled_dates,
    )

    progress("Assessing distance-band and event-window support")
    sparse_cells = event_summary.loc[
        event_summary["pair_events"] < 10,
        ["distance_band", "event_time", "pair_events"],
    ]
    if len(sparse_cells):
        print(
            "Sparse cells (fewer than 10 distinct existing/new-store pair-events); "
            "the threshold is a warning, not an automatic deletion rule:\n"
            + sparse_cells.to_string(index=False)
        )
        print(
            "The 0-100m band has far fewer events than wider bands. Keep the "
            "requested bands for diagnostics, but treat its estimates as "
            "exploratory; consider combining it with 100-250m only as a "
            "prespecified sensitivity analysis."
        )
    else:
        print("All requested band/event-time cells have at least 10 pair-events.")

    try:
        comparison, coefficients = run_robustness_checks(panel)
    except InsufficientModelSupport as error:
        print(f"\nSTOP: requested model is not supported as specified. {error}")
        print(
            "Diagnostic CSVs and the full constructed panel have been saved. "
            "Change the distance/event-time specification and rerun; no "
            "unsupported regression results were written."
        )
        return 2

    save_outputs(panel, comparison, coefficients)
    _explain_model_comparison(comparison)
    print(f"\nAll outputs are in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        sys.exit(main())
