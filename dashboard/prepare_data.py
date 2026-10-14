"""Prepare dashboard-ready store-month and new-store exposure datasets."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INCOME_FILE = BASE_DIR / "store_master_data.xlsx"
DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DISTANCE_THRESHOLDS = (100, 250, 500, 1000)
STRONG_GROWTH_THRESHOLD_PCT = 10
GROWTH_THRESHOLD_PCT = 3
STABLE_LOWER_THRESHOLD_PCT = -3
STRONG_DECLINE_THRESHOLD_PCT = -10
METADATA_SOURCE_COLUMNS = {
    "store_name": "store_name",
    "opening_date": "ost_dt",
    "closing_date": "cls_dt",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "store_status": "Store_Status",
    "location_area": "loc_area_new",
    "Distance_Zone": "Distance_Zone",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized.mask(normalized.eq(""))


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    # Excel uses 9999-12-31 as an open-ended closing date.
    return parsed.mask(parsed.dt.year.ge(9000))


def _parse_coordinate(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    decimal_text = text.replace(",", ".") if text.count(",") == 1 else text
    try:
        return float(decimal_text)
    except ValueError:
        pass

    dms_text = re.sub(r"[°º\uFFFD'\"′″]", " ", text)
    parts = re.findall(r"[-+]?\d+(?:\.\d*)?", dms_text)
    if len(parts) != 3:
        return np.nan
    degrees, minutes, seconds = map(float, parts)
    if not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return np.nan
    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + minutes / 60 + seconds / 3600)


def _first_non_null(values: pd.Series) -> Any:
    available = values.dropna()
    return available.iloc[0] if not available.empty else pd.NA


def _master_conflicts(
    sales: pd.DataFrame,
    metadata_source_columns: dict[str, str],
) -> int:
    conflicts = 0
    for column in metadata_source_columns.values():
        if column not in sales.columns:
            continue
        distinct = sales.groupby("store_id", dropna=True)[column].nunique(
            dropna=True
        )
        conflicts += int(distinct.gt(1).sum())
    return conflicts


def _format_quality_report(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["check", "value", "detail"])


def _make_monthly_panel(
    sales_source: pd.DataFrame,
    distances_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    _require_columns(
        sales_source,
        {"SalesMonth", "store_id", "APSD_income"},
        "store_master_data.xlsx",
    )
    _require_columns(
        distances_source,
        {
            "existing_store_id",
            "new_store_id",
            "distance_m",
            "existing_opening_date",
            "new_opening_date",
        },
        "store_to_store_distance.xlsx",
    )

    sales = sales_source.copy()
    distances = distances_source.copy()
    records: list[dict[str, Any]] = []

    def add_check(check: str, value: Any, detail: str = "") -> None:
        records.append({"check": check, "value": value, "detail": detail})

    source_rows = len(sales)
    sales["store_id"] = _normalize_ids(sales["store_id"])
    missing_sales_ids = int(sales["store_id"].isna().sum())
    sales["date"] = _parse_dates(sales["SalesMonth"]).dt.to_period("M").dt.to_timestamp()
    invalid_sales_months = int(sales["date"].isna().sum())
    sales = sales.loc[sales["store_id"].notna() & sales["date"].notna()].copy()

    sales["APSD"] = pd.to_numeric(sales["APSD_income"], errors="coerce")
    if "Customers" in sales.columns:
        sales["customers"] = pd.to_numeric(sales["Customers"], errors="coerce")
    if "APSD_customers" in sales.columns:
        sales["APSD_customers"] = pd.to_numeric(
            sales["APSD_customers"], errors="coerce"
        )

    for output_column, source_column in METADATA_SOURCE_COLUMNS.items():
        if source_column in sales.columns:
            if output_column in {"opening_date", "closing_date"}:
                sales[output_column] = _parse_dates(sales[source_column])
            elif output_column in {"latitude", "longitude"}:
                sales[output_column] = sales[source_column].map(_parse_coordinate)
            else:
                sales[output_column] = sales[source_column]
        else:
            sales[output_column] = pd.NA

    duplicate_store_months = int(sales.duplicated(["store_id", "date"]).sum())
    master_conflicts = _master_conflicts(sales, METADATA_SOURCE_COLUMNS)

    add_check("source_store_month_rows", source_rows)
    add_check("valid_store_month_rows", len(sales))
    add_check("missing_store_id_rows", missing_sales_ids)
    add_check("invalid_sales_month_rows", invalid_sales_months)
    add_check("duplicate_store_month_rows", duplicate_store_months)
    add_check("master_metadata_conflicts", master_conflicts)
    add_check(
        "store_count",
        sales["store_id"].nunique(),
        "Stores present in the monthly sales workbook.",
    )
    add_check(
        "sales_month_range",
        (
            f"{sales['date'].min():%Y-%m} to {sales['date'].max():%Y-%m}"
            if not sales.empty
            else "unavailable"
        ),
    )

    if duplicate_store_months:
        raise ValueError(
            "Monthly sales data is not unique at store-month grain "
            f"({duplicate_store_months:,} duplicate rows)."
        )
    if master_conflicts:
        raise ValueError(
            "Store master fields conflict between monthly rows "
            f"({master_conflicts:,} store-field conflicts)."
        )
    if sales.empty:
        raise ValueError("No valid store-month observations were found.")

    metadata_columns = [
        "store_id",
        *METADATA_SOURCE_COLUMNS.keys(),
    ]
    metadata = (
        sales[metadata_columns]
        .groupby("store_id", as_index=False, sort=False)
        .agg({column: _first_non_null for column in metadata_columns[1:]})
    )
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")

    invalid_latitude = metadata["latitude"].notna() & ~metadata["latitude"].between(
        -90, 90
    )
    invalid_longitude = metadata["longitude"].notna() & ~metadata[
        "longitude"
    ].between(-180, 180)
    invalid_coordinates = invalid_latitude | invalid_longitude
    missing_coordinates = metadata["latitude"].isna() | metadata["longitude"].isna()

    sales_months = pd.date_range(
        sales["date"].min(),
        sales["date"].max(),
        freq="MS",
    )
    month_frame = pd.DataFrame({"date": sales_months})
    metadata["_join_key"] = 1
    month_frame["_join_key"] = 1
    panel = metadata.merge(month_frame, on="_join_key").drop(columns="_join_key")

    sales_columns = ["store_id", "date", "APSD"]
    for column in ("customers", "APSD_customers"):
        if column in sales.columns:
            sales_columns.append(column)
    panel = panel.merge(
        sales[sales_columns],
        on=["store_id", "date"],
        how="left",
        validate="one_to_one",
    )
    panel["has_sales_observation"] = panel["APSD"].notna()
    panel["year"] = panel["date"].dt.year.astype("int16")
    panel["month"] = panel["date"].dt.month.astype("int8")

    month_end = panel["date"] + pd.offsets.MonthEnd(0)
    open_date_known = panel["opening_date"].notna()
    opened_by_month_end = open_date_known & panel["opening_date"].le(month_end)
    not_closed_before_month = (
        panel["closing_date"].isna()
        | panel["closing_date"].gt(panel["date"])
    )
    inactive_without_closing_date = (
        panel["store_status"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("inactive")
        .fillna(False)
        & panel["closing_date"].isna()
    )
    panel["is_operating"] = (
        opened_by_month_end
        & not_closed_before_month
        & ~inactive_without_closing_date
    )

    source_month_end = sales["date"] + pd.offsets.MonthEnd(0)
    before_opening = (
        sales["opening_date"].notna()
        & source_month_end.lt(sales["opening_date"])
    )
    after_closing = (
        sales["closing_date"].notna()
        & sales["date"].ge(sales["closing_date"])
    )
    missing_opening_ids = int(
        metadata.loc[metadata["opening_date"].isna(), "store_id"].nunique()
    )
    inactive_missing_close_ids = int(
        metadata.loc[
            metadata["store_status"]
            .astype("string")
            .str.strip()
            .str.casefold()
            .eq("inactive")
            .fillna(False)
            & metadata["closing_date"].isna(),
            "store_id",
        ].nunique()
    )
    missing_apsd_rows = int(pd.to_numeric(sales["APSD"], errors="coerce").isna().sum())
    negative_apsd_rows = int(sales["APSD"].lt(0).sum())
    zero_apsd_rows = int(sales["APSD"].eq(0).sum())
    missing_customer_rows = (
        int(sales["customers"].isna().sum())
        if "customers" in sales.columns
        else None
    )

    add_check(
        "duplicate_store_master_records",
        int(metadata["store_id"].duplicated().sum()),
        "Unique master is reduced from repeated monthly rows after conflict checks.",
    )
    add_check("store_ids_missing_opening_date", missing_opening_ids)
    add_check(
        "store_ids_missing_coordinates",
        int(missing_coordinates.sum()),
    )
    add_check("invalid_coordinate_store_ids", int(invalid_coordinates.sum()))
    add_check("missing_APSD_rows", missing_apsd_rows)
    add_check("negative_APSD_rows", negative_apsd_rows)
    add_check("zero_APSD_rows", zero_apsd_rows)
    add_check(
        "missing_customer_rows",
        missing_customer_rows,
        "Customer metric is unavailable only if the source column is absent.",
    )
    add_check(
        "source_rows_before_opening",
        int(before_opening.sum()),
        f"{sales.loc[before_opening, 'store_id'].nunique():,} affected stores.",
    )
    add_check(
        "source_rows_after_closing",
        int(after_closing.sum()),
        f"{sales.loc[after_closing, 'store_id'].nunique():,} affected stores.",
    )
    add_check(
        "inactive_stores_without_closing_date",
        inactive_missing_close_ids,
        "Conservatively excluded from the operating network because history cannot be dated.",
    )
    add_check(
        "store_month_grid_rows",
        len(panel),
        "Complete monthly grid across the observed source date range.",
    )

    non_operating_sales = panel.loc[
        panel["has_sales_observation"] & ~panel["is_operating"]
    ]
    add_check(
        "sales_observations_outside_operating_period",
        len(non_operating_sales),
        "APSD and customer metrics are nulled for these store-months.",
    )

    performance_columns = [
        column
        for column in ("APSD", "customers", "APSD_customers")
        if column in panel.columns
    ]
    panel.loc[~panel["is_operating"], performance_columns] = np.nan

    for source_column, previous_column, change_column, pct_column in (
        ("APSD", "previous_APSD", "APSD_change", "APSD_change_pct"),
        (
            "customers",
            "previous_customers",
            "customer_change",
            "customer_change_pct",
        ),
    ):
        if source_column not in panel.columns:
            continue
        previous = panel.groupby("store_id", sort=False)[source_column].shift(1)
        panel[previous_column] = previous
        panel[change_column] = panel[source_column] - previous
        valid_pct = (
            panel[source_column].notna()
            & previous.notna()
            & previous.ne(0)
        )
        panel[pct_column] = np.nan
        panel.loc[valid_pct, pct_column] = (
            panel.loc[valid_pct, source_column] / previous.loc[valid_pct] - 1
        ) * 100
        panel[pct_column] = panel[pct_column].replace([np.inf, -np.inf], np.nan)

    if "APSD_change_pct" in panel.columns:
        change = panel["APSD_change_pct"]
        panel["performance_category"] = np.select(
            [
                change.ge(STRONG_GROWTH_THRESHOLD_PCT),
                change.ge(GROWTH_THRESHOLD_PCT),
                change.ge(STABLE_LOWER_THRESHOLD_PCT),
                change.gt(STRONG_DECLINE_THRESHOLD_PCT),
            ],
            [
                "Strong Growth",
                "Growth",
                "Stable",
                "Decline",
            ],
            default="Strong Decline",
        )
        panel.loc[change.isna(), "performance_category"] = pd.NA
    else:
        panel["performance_category"] = pd.NA

    distances["existing_store_id"] = _normalize_ids(
        distances["existing_store_id"]
    )
    distances["new_store_id"] = _normalize_ids(distances["new_store_id"])
    distances["distance_m"] = pd.to_numeric(distances["distance_m"], errors="coerce")
    distances["existing_opening_date"] = _parse_dates(
        distances["existing_opening_date"]
    )
    distances["new_opening_date"] = _parse_dates(distances["new_opening_date"])

    duplicate_pairs = int(
        distances.duplicated(["existing_store_id", "new_store_id"]).sum()
    )
    invalid_distance = distances["distance_m"].isna() | distances["distance_m"].lt(0)
    invalid_pair = (
        distances["existing_store_id"].isna()
        | distances["new_store_id"].isna()
        | distances["existing_opening_date"].isna()
        | distances["new_opening_date"].isna()
    )
    chronology_violation = (
        distances["existing_opening_date"].notna()
        & distances["new_opening_date"].notna()
        & distances["existing_opening_date"].ge(distances["new_opening_date"])
    )
    add_check("distance_pair_rows", len(distances))
    add_check("duplicate_distance_pairs", duplicate_pairs)
    add_check("missing_or_invalid_distance_rows", int(invalid_distance.sum()))
    add_check("distance_pairs_missing_ids_or_dates", int(invalid_pair.sum()))
    add_check(
        "distance_pairs_not_existing_before_new_opening",
        int(chronology_violation.sum()),
    )

    if duplicate_pairs:
        raise ValueError(
            f"Distance data contains {duplicate_pairs:,} duplicate store pairs."
        )

    valid_pairs = distances.loc[
        ~invalid_distance & ~invalid_pair & ~chronology_violation
    ].copy()
    valid_pairs = valid_pairs.loc[valid_pairs["distance_m"].le(1500)].copy()

    exposure = _build_new_store_exposure(valid_pairs, panel)
    panel = _add_nearby_new_store_counts(panel, valid_pairs, metadata)
    apsd_reference_coverage = int(
        exposure["potential_apsd_exposure_250m"].notna().sum()
    )
    records.append(
        {
            "check": "new_store_events_with_APSD_reference_data_250m",
            "value": apsd_reference_coverage,
            "detail": (
                "Events with at least one observed existing-store APSD in the "
                "previous complete month."
            ),
        }
    )
    missing_previous_apsd = int(
        (
            panel["is_operating"]
            & panel["APSD"].notna()
            & panel["previous_APSD"].isna()
        ).sum()
    )
    records.append(
        {
            "check": "operating_store_months_without_previous_APSD",
            "value": missing_previous_apsd,
            "detail": "No valid APSD observation exists for the immediately previous month.",
        }
    )
    if "customers" in panel.columns:
        missing_previous_customers = int(
            (
                panel["is_operating"]
                & panel["customers"].notna()
                & panel["previous_customers"].isna()
            ).sum()
        )
        records.append(
            {
                "check": "operating_store_months_without_previous_customers",
                "value": missing_previous_customers,
                "detail": "No valid customer observation exists for the immediately previous month.",
            }
        )
    panel = panel.sort_values(["store_id", "date"]).reset_index(drop=True)
    exposure = exposure.sort_values(
        ["new_opening_date", "new_store_id"]
    ).reset_index(drop=True)

    add_check("valid_distance_pairs_used", len(valid_pairs))
    add_check("new_store_events_with_distance_data", len(exposure))
    add_check(
        "distance_pair_rows_over_1500m",
        int((~invalid_distance & distances["distance_m"].gt(1500)).sum()),
        "Not used; distance workbook is expected to contain pairs up to 1,500 m.",
    )

    return panel, exposure, records


def _build_new_store_exposure(
    pairs: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "new_store_id",
        "new_store_name",
        "new_opening_date",
        "new_latitude",
        "new_longitude",
        "apsd_reference_month",
    ]
    for threshold in DISTANCE_THRESHOLDS:
        columns.extend(
            [
                f"potential_stores_{threshold}m",
                f"potential_apsd_exposure_{threshold}m",
                f"potential_apsd_observed_stores_{threshold}m",
            ]
        )
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    info_columns = {
        "new_store_name": "new_store_name",
        "new_opening_date": "new_opening_date",
        "new_latitude": "new_latitude",
        "new_longitude": "new_longitude",
    }
    for column in info_columns.values():
        if column not in pairs.columns:
            pairs[column] = pd.NA
    for output_column, source_column in (
        ("new_latitude", "new_latitude"),
        ("new_longitude", "new_longitude"),
    ):
        pairs[source_column] = pd.to_numeric(
            pairs[source_column], errors="coerce"
        )

    apsd_by_month = {
        date: group.set_index("store_id")["APSD"]
        for date, group in panel.loc[
            panel["is_operating"] & panel["APSD"].notna()
        ].groupby("date")
    }
    records: list[dict[str, Any]] = []
    for new_store_id, group in pairs.groupby("new_store_id", sort=False):
        row: dict[str, Any] = {"new_store_id": new_store_id}
        for output_column, source_column in info_columns.items():
            row[output_column] = _first_non_null(group[source_column])

        opening_date = row["new_opening_date"]
        reference_month = (
            opening_date.to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
        )
        row["apsd_reference_month"] = reference_month
        month_apsd = apsd_by_month.get(reference_month, pd.Series(dtype=float))

        for threshold in DISTANCE_THRESHOLDS:
            neighbors = group.loc[group["distance_m"].le(threshold)]
            neighbor_ids = neighbors["existing_store_id"].drop_duplicates()
            row[f"potential_stores_{threshold}m"] = len(neighbor_ids)
            values = month_apsd.reindex(neighbor_ids).dropna()
            row[f"potential_apsd_observed_stores_{threshold}m"] = len(values)
            row[f"potential_apsd_exposure_{threshold}m"] = (
                float(values.sum()) if not values.empty else np.nan
            )
        records.append(row)

    return pd.DataFrame.from_records(records, columns=columns)


def _add_nearby_new_store_counts(
    panel: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    for threshold in DISTANCE_THRESHOLDS:
        panel[f"nearby_new_stores_{threshold}m"] = 0.0

    known_sequence = (
        metadata["opening_date"].notna()
        & metadata["latitude"].between(-90, 90)
        & metadata["longitude"].between(-180, 180)
    )
    unknown_ids = set(metadata.loc[~known_sequence, "store_id"])

    for date in panel["date"].drop_duplicates():
        month_end = date + pd.offsets.MonthEnd(0)
        active_pairs = pairs.loc[pairs["new_opening_date"].le(month_end)]
        month_mask = panel["date"].eq(date)
        for threshold in DISTANCE_THRESHOLDS:
            counts = (
                active_pairs.loc[active_pairs["distance_m"].le(threshold)]
                .groupby("existing_store_id")["new_store_id"]
                .nunique()
            )
            panel.loc[month_mask, f"nearby_new_stores_{threshold}m"] = (
                panel.loc[month_mask, "store_id"]
                .map(counts)
                .fillna(0)
                .to_numpy()
            )

    if unknown_ids:
        unknown = panel["store_id"].isin(unknown_ids)
        for threshold in DISTANCE_THRESHOLDS:
            panel.loc[unknown, f"nearby_new_stores_{threshold}m"] = np.nan
    return panel


def prepare_datasets(
    sales_source: pd.DataFrame,
    distances_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, exposure, records = _make_monthly_panel(
        sales_source,
        distances_source,
    )
    return panel, exposure, _format_quality_report(records)


def main() -> None:
    if not INCOME_FILE.is_file():
        raise FileNotFoundError(INCOME_FILE)
    if not DISTANCE_FILE.is_file():
        raise FileNotFoundError(DISTANCE_FILE)

    sales = pd.read_excel(INCOME_FILE, sheet_name=0)
    distances = pd.read_excel(DISTANCE_FILE, sheet_name=0)
    panel, exposure, quality = prepare_datasets(sales, distances)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(DATA_DIR / "dashboard_store_month.parquet", index=False)
    exposure.to_parquet(DATA_DIR / "potential_store_exposure.parquet", index=False)
    quality.to_csv(
        OUTPUT_DIR / "data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Dashboard data preparation complete.")
    print(f"Store-month rows: {len(panel):,}")
    print(f"New-store exposure rows: {len(exposure):,}")
    print(f"Quality report: {OUTPUT_DIR / 'data_quality_report.csv'}")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
