"""Apply saved Model 4 event-month estimates to planned GPS candidate sites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "cannibalization_model4"
GPS_FILE = PROJECT_ROOT / "Data" / "raw" / "New Store GPS.xlsx"
MASTER_FILE = PROJECT_ROOT / "store_master_data.xlsx"
PANEL_FILE = RESULTS_DIR / "cu_cannibalization_model4_panel.parquet"
COEFFICIENT_FILE = RESULTS_DIR / "model4_coefficients.csv"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
REGIONAL_SALES_FILE = (
    PROJECT_ROOT / "outputs" / "cannibalization_regional" / "regional_sales_coefficients.csv"
)
REGIONAL_CUSTOMER_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "cannibalization_regional"
    / "regional_customer_coefficients.csv"
)
REGIONAL_STATUS_FILE = (
    PROJECT_ROOT / "outputs" / "cannibalization_regional" / "model_status.csv"
)
DISTANCE_BANDS = (
    ("0_100", 0.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1000.0),
)
DEFAULT_SCENARIOS = (180, 400)
EARTH_RADIUS_M = 6_371_000.0


def _require_columns(
    frame: pd.DataFrame, columns: set[str], frame_name: str
) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _normalize_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized.mask(normalized.eq(""))


def _valid_coordinates(frame: pd.DataFrame) -> pd.Series:
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    return (
        latitude.between(-90, 90)
        & longitude.between(-180, 180)
        & latitude.notna()
        & longitude.notna()
    )


def prepare_candidates(workbook: pd.DataFrame) -> pd.DataFrame:
    """Keep first unique valid location in workbook order as candidate rank."""
    _require_columns(
        workbook,
        {"Potential Point", "Longitude", "Latitude"},
        "New Store GPS.xlsx",
    )
    candidates = workbook.copy().reset_index(drop=True)
    candidates["_workbook_order"] = np.arange(len(candidates), dtype=np.int64)
    candidates["longitude"] = pd.to_numeric(
        candidates["Longitude"], errors="coerce"
    )
    candidates["latitude"] = pd.to_numeric(candidates["Latitude"], errors="coerce")
    if "#" in candidates.columns:
        candidates["_candidate_order"] = pd.to_numeric(
            candidates["#"], errors="coerce"
        )
    else:
        candidates["_candidate_order"] = np.nan
    candidates["_candidate_order"] = candidates["_candidate_order"].fillna(
        candidates["_workbook_order"] + 1
    )
    candidates = candidates.loc[_valid_coordinates(candidates)].copy()
    candidates = candidates.sort_values(
        ["_candidate_order", "_workbook_order"], kind="stable"
    ).drop_duplicates(["longitude", "latitude"], keep="first")
    candidates = candidates.reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
    candidates["candidate_id"] = candidates["candidate_rank"].map(
        lambda rank: f"GPS-{rank:04d}"
    )
    candidates["candidate_source_number"] = (
        pd.to_numeric(candidates["#"], errors="coerce")
        if "#" in candidates.columns
        else candidates["_workbook_order"] + 1
    )
    candidates["candidate_name"] = candidates["Potential Point"].astype("string")
    candidates["expansion"] = (
        candidates["Expansion"].astype("string")
        if "Expansion" in candidates.columns
        else pd.NA
    )
    candidates["area_type"] = (
        candidates["Area Type"].astype("string")
        if "Area Type" in candidates.columns
        else pd.NA
    )
    return candidates[
        [
            "candidate_id",
            "candidate_rank",
            "candidate_source_number",
            "candidate_name",
            "expansion",
            "area_type",
            "longitude",
            "latitude",
        ]
    ].reset_index(drop=True)


def load_model4_month_zero_coefficients(
    coefficients: pd.DataFrame,
) -> dict[str, float]:
    """Read the four event-month-0 log coefficients from Model D."""
    _require_columns(
        coefficients,
        {"model", "term", "coefficient"},
        "model4_coefficients.csv",
    )
    model_d = coefficients.loc[coefficients["model"].astype(str).eq("D")].copy()
    parsed = model_d["term"].astype(str).str.extract(
        r"^exposure_(0_100|100_250|250_500|500_1000)_event_p0$"
    )
    selected = model_d.loc[parsed[0].notna()].copy()
    selected["distance_band"] = parsed.loc[selected.index, 0]
    selected["coefficient"] = pd.to_numeric(
        selected["coefficient"], errors="coerce"
    )
    result = dict(
        zip(selected["distance_band"], selected["coefficient"].astype(float))
    )
    expected = {band for band, _, _ in DISTANCE_BANDS}
    if set(result) != expected or not np.isfinite(list(result.values())).all():
        raise ValueError(
            "Model D must contain one finite event-month-0 coefficient for "
            f"each distance band; found {sorted(result)}."
        )
    return result


def load_regional_month_zero_coefficients(
    coefficients: pd.DataFrame,
    outcome: str,
) -> dict[tuple[str, str], float]:
    """Read existing regional binned event-study log coefficients for month 0."""
    required = {
        "region",
        "model_type",
        "event_period",
        "distance_band",
        "coefficient",
        "outcome_scale",
    }
    _require_columns(coefficients, required, f"regional {outcome} coefficients")
    selected = coefficients.loc[
        coefficients["model_type"].astype(str).eq("binned_event_study")
        & coefficients["event_period"].astype(str).eq("p0")
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No regional binned event-study month-0 coefficients for {outcome}."
        )
    if not selected["outcome_scale"].astype(str).eq("log").all():
        raise ValueError(
            f"Regional month-0 {outcome} coefficients are not all on the log scale."
        )
    selected["coefficient"] = pd.to_numeric(
        selected["coefficient"], errors="coerce"
    )
    if selected["coefficient"].isna().any():
        raise ValueError(
            f"Regional month-0 {outcome} coefficients contain missing estimates."
        )
    if selected.duplicated(["region", "distance_band"]).any():
        raise ValueError(
            f"Regional month-0 {outcome} coefficients are not unique by region/band."
        )
    coefficient_values = selected["coefficient"].to_numpy(dtype=np.float64)
    return {
        (str(region), str(band)): float(coefficient)
        for region, band, coefficient in zip(
            selected["region"],
            selected["distance_band"],
            coefficient_values,
        )
    }


def _analysis_region(distance_zone: Any, ub: Any) -> str:
    zone = "" if pd.isna(distance_zone) else str(distance_zone).strip()
    ub_value = "" if pd.isna(ub) else str(ub).strip().casefold()
    if ub_value == "rural" or zone == "Rural":
        return "Rural"
    return {
        "Core": "City Center",
        "Inner City": "City Outskirts",
        "Outer City": "Far / Outer Areas",
        "Far": "Far / Outer Areas",
    }.get(zone, "Unmapped")


def prepare_store_baselines(
    panel: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Use each modeled store's latest panel month as the shared baseline."""
    _require_columns(
        panel,
        {
            "existing_store_id",
            "SalesMonth",
            "APSD_income",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
        },
        "cu_cannibalization_model4_panel.parquet",
    )
    _require_columns(
        master,
        {"store_id", "SalesMonth", "APSD_customers"},
        "store_master_data.xlsx",
    )
    latest = panel.copy()
    latest["existing_store_id"] = _normalize_ids(latest["existing_store_id"])
    latest["SalesMonth"] = pd.to_datetime(latest["SalesMonth"], errors="coerce")
    latest = (
        latest.dropna(subset=["existing_store_id", "SalesMonth"])
        .sort_values(["existing_store_id", "SalesMonth"])
        .drop_duplicates("existing_store_id", keep="last")
        .copy()
    )
    latest["APSD_income_baseline"] = pd.to_numeric(
        latest["APSD_income"], errors="coerce"
    )
    latest["latitude"] = pd.to_numeric(latest["LATITUDE"], errors="coerce")
    latest["longitude"] = pd.to_numeric(latest["LONGITUDE"], errors="coerce")

    source = master.copy()
    source["store_id"] = _normalize_ids(source["store_id"])
    source["SalesMonth"] = pd.to_datetime(source["SalesMonth"], errors="coerce")
    metadata_columns = [
        column
        for column in (
            "store_id",
            "SalesMonth",
            "store_name",
            "APSD_customers",
            "Distance_Zone",
            "UB",
        )
        if column in source.columns
    ]
    source = source[metadata_columns].drop_duplicates(
        ["store_id", "SalesMonth"], keep="last"
    )
    latest = latest.merge(
        source,
        left_on=["existing_store_id", "SalesMonth"],
        right_on=["store_id", "SalesMonth"],
        how="left",
        validate="one_to_one",
    )
    latest["existing_store_id"] = latest["existing_store_id"].astype("string")
    latest["store_name"] = latest.get(
        "store_name", pd.Series(pd.NA, index=latest.index)
    ).fillna("CU " + latest["existing_store_id"])
    latest["APSD_customers_baseline"] = pd.to_numeric(
        latest["APSD_customers"], errors="coerce"
    )
    latest["region"] = [
        _analysis_region(zone, ub)
        for zone, ub in zip(
            latest.get("Distance_Zone", pd.Series(pd.NA, index=latest.index)),
            latest.get("UB", pd.Series(pd.NA, index=latest.index)),
        )
    ]
    latest["loc_area_new"] = latest["loc_area_new"].astype("string")
    return latest[
        [
            "existing_store_id",
            "store_name",
            "region",
            "loc_area_new",
            "SalesMonth",
            "APSD_income_baseline",
            "APSD_customers_baseline",
            "latitude",
            "longitude",
        ]
    ].rename(columns={"SalesMonth": "APSD_baseline_month"})


def _haversine_matrix(stores: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    store_lat = np.radians(stores["latitude"].to_numpy(dtype=float))[:, None]
    store_lon = np.radians(stores["longitude"].to_numpy(dtype=float))[:, None]
    candidate_lat = np.radians(candidates["latitude"].to_numpy(dtype=float))[None, :]
    candidate_lon = np.radians(candidates["longitude"].to_numpy(dtype=float))[None, :]
    delta_lat = candidate_lat - store_lat
    delta_lon = candidate_lon - store_lon
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(store_lat) * np.cos(candidate_lat) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def _distance_band_codes(distances: np.ndarray) -> np.ndarray:
    codes = np.full(distances.shape, "", dtype=object)
    for band, lower, upper in DISTANCE_BANDS:
        codes[(distances >= lower) & (distances < upper)] = band
    return codes


def calculate_cumulative_scenarios(
    stores: pd.DataFrame,
    candidates: pd.DataFrame,
    coefficients: dict[str, float],
    scenario_sizes: tuple[int, ...] = DEFAULT_SCENARIOS,
    regional_sales_coefficients: dict[tuple[str, str], float] | None = None,
    regional_customer_coefficients: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate all selected nearby candidate log effects, not nearest-only."""
    _require_columns(
        stores,
        {
            "existing_store_id",
            "latitude",
            "longitude",
            "APSD_income_baseline",
            "APSD_customers_baseline",
        },
        "store baselines",
    )
    _require_columns(
        candidates,
        {
            "candidate_id",
            "candidate_rank",
            "candidate_name",
            "longitude",
            "latitude",
        },
        "candidate locations",
    )
    if stores["existing_store_id"].duplicated().any():
        raise ValueError("Store baselines must contain one row per existing store.")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique.")
    if any(size < 0 for size in scenario_sizes):
        raise ValueError("Scenario sizes cannot be negative.")
    expected_bands = {band for band, _, _ in DISTANCE_BANDS}
    if set(coefficients) != expected_bands:
        raise ValueError("A Model 4 coefficient is required for each distance band.")

    valid_stores = stores.loc[_valid_coordinates(stores)].copy()
    if candidates.empty or valid_stores.empty:
        pairs = pd.DataFrame()
    else:
        distance_matrix = _haversine_matrix(valid_stores, candidates)
        band_matrix = _distance_band_codes(distance_matrix)
        store_positions, candidate_positions = np.where(
            (distance_matrix < 1000.0) & (band_matrix != "")
        )
        pairs = pd.DataFrame(
            {
                "existing_store_id": valid_stores.iloc[
                    store_positions
                ]["existing_store_id"].to_numpy(),
                "candidate_id": candidates.iloc[candidate_positions][
                    "candidate_id"
                ].to_numpy(),
                "candidate_rank": candidates.iloc[candidate_positions][
                    "candidate_rank"
                ].to_numpy(),
                "distance_m": distance_matrix[store_positions, candidate_positions],
                "distance_band": band_matrix[store_positions, candidate_positions],
            }
        )
        candidate_details = candidates.set_index("candidate_id")
        pairs["candidate_name"] = pairs["candidate_id"].map(
            candidate_details["candidate_name"]
        )
        pairs["candidate_longitude"] = pairs["candidate_id"].map(
            candidate_details["longitude"]
        )
        pairs["candidate_latitude"] = pairs["candidate_id"].map(
            candidate_details["latitude"]
        )
        for column in ("expansion", "area_type", "candidate_source_number"):
            if column in candidate_details.columns:
                pairs[column] = pairs["candidate_id"].map(
                    candidate_details[column]
                )
        pairs["coefficient_log_effect"] = pairs["distance_band"].map(coefficients)
        pairs["individual_APSD_effect_pct"] = np.expm1(
            pairs["coefficient_log_effect"]
        ) * 100
        store_regions = stores[["existing_store_id"]].copy()
        store_regions["region"] = (
            stores["region"] if "region" in stores.columns else "Unmapped"
        )
        store_regions = store_regions.set_index("existing_store_id")["region"]
        pairs["region"] = pairs["existing_store_id"].map(store_regions).fillna(
            "Unmapped"
        )
        if regional_sales_coefficients is not None:
            pairs["regional_sales_coefficient"] = [
                regional_sales_coefficients.get(
                    (str(region), str(band)),
                    regional_sales_coefficients.get(("All Regions", str(band))),
                )
                for region, band in zip(pairs["region"], pairs["distance_band"])
            ]
            pairs["individual_regional_APSD_effect_pct"] = np.expm1(
                pd.to_numeric(pairs["regional_sales_coefficient"], errors="coerce")
            ) * 100
        if regional_customer_coefficients is not None:
            pairs["regional_customer_coefficient"] = [
                regional_customer_coefficients.get(
                    (str(region), str(band)),
                    regional_customer_coefficients.get(("All Regions", str(band))),
                )
                for region, band in zip(pairs["region"], pairs["distance_band"])
            ]
            pairs["individual_APSD_customers_effect_pct"] = np.expm1(
                pd.to_numeric(pairs["regional_customer_coefficient"], errors="coerce")
            ) * 100

    for size in scenario_sizes:
        pairs[f"in_scenario_{size}"] = pairs.get(
            "candidate_rank", pd.Series(dtype=int)
        ).le(size)

    results = stores.copy()
    if "region" not in results.columns:
        results["region"] = "Unmapped"
    for size in scenario_sizes:
        selected = pairs.loc[pairs[f"in_scenario_{size}"]].copy()
        if selected.empty:
            grouped = pd.DataFrame(
                columns=["existing_store_id", "nearby_count", "log_effect"]
            )
            band_counts = pd.DataFrame(columns=["existing_store_id"])
        else:
            grouped = (
                selected.groupby("existing_store_id", as_index=False)
                .agg(
                    nearby_count=("candidate_id", "nunique"),
                    log_effect=("coefficient_log_effect", "sum"),
                )
            )
            band_counts = (
                selected.groupby(
                    ["existing_store_id", "distance_band"]
                )["candidate_id"]
                .nunique()
                .unstack(fill_value=0)
                .reset_index()
            )
        results = results.merge(
            grouped,
            on="existing_store_id",
            how="left",
            validate="one_to_one",
        )
        results = results.merge(
            band_counts,
            on="existing_store_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_band"),
        )
        regional_fields: dict[str, tuple[str, Any]] = {
            "pair_count": ("candidate_id", "nunique"),
        }
        if "regional_sales_coefficient" in selected.columns:
            regional_fields.update(
                {
                    "regional_sales_log": (
                        "regional_sales_coefficient",
                        lambda values: values.sum(min_count=1),
                    ),
                    "regional_sales_terms": (
                        "regional_sales_coefficient",
                        "count",
                    ),
                }
            )
        if "regional_customer_coefficient" in selected.columns:
            regional_fields.update(
                {
                    "regional_customer_log": (
                        "regional_customer_coefficient",
                        lambda values: values.sum(min_count=1),
                    ),
                    "regional_customer_terms": (
                        "regional_customer_coefficient",
                        "count",
                    ),
                }
            )
        if not selected.empty and len(regional_fields) > 1:
            regional_grouped = selected.groupby("existing_store_id").agg(
                **regional_fields
            )
            if "regional_customer_log" in regional_grouped:
                regional_grouped.loc[
                    regional_grouped["regional_customer_terms"]
                    < regional_grouped["pair_count"],
                    "regional_customer_log",
                ] = np.nan
            if "regional_sales_log" in regional_grouped:
                regional_grouped.loc[
                    regional_grouped["regional_sales_terms"]
                    < regional_grouped["pair_count"],
                    "regional_sales_log",
                ] = np.nan
            regional_grouped = regional_grouped.reset_index()
            results = results.merge(
                regional_grouped,
                on="existing_store_id",
                how="left",
                validate="one_to_one",
            )
        results[f"nearby_new_stores_{size}"] = (
            pd.to_numeric(results.pop("nearby_count"), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        log_effect = pd.to_numeric(results.pop("log_effect"), errors="coerce").fillna(0)
        net_change = np.expm1(log_effect) * 100
        results[f"APSD_income_change_pct_{size}"] = net_change
        results[f"cannibalization_pct_{size}"] = np.maximum(-net_change, 0.0)
        results[f"APSD_income_change_{size}"] = (
            results["APSD_income_baseline"] * np.expm1(log_effect)
        )
        results[f"APSD_income_expected_{size}"] = (
            results["APSD_income_baseline"] + results[f"APSD_income_change_{size}"]
        )
        has_selected_openings = results[f"nearby_new_stores_{size}"].gt(0)
        if "regional_sales_log" in results.columns:
            regional_sales_log = pd.to_numeric(
                results.pop("regional_sales_log"), errors="coerce"
            ).where(has_selected_openings, 0.0)
            regional_sales_change = np.expm1(regional_sales_log) * 100
            results[f"regional_APSD_income_change_pct_{size}"] = (
                regional_sales_change
            )
            results[f"regional_cannibalization_pct_{size}"] = np.maximum(
                -regional_sales_change, 0.0
            )
            results[f"regional_APSD_income_change_{size}"] = (
                results["APSD_income_baseline"] * np.expm1(regional_sales_log)
            )
            results[f"regional_income_effect_supported_{size}"] = (
                ~has_selected_openings | results["regional_sales_terms"].eq(
                    results[f"nearby_new_stores_{size}"]
                )
            )
            results = results.drop(
                columns=["regional_sales_terms", "pair_count"],
                errors="ignore",
            )
        if "regional_customer_log" in results.columns:
            regional_customer_log = pd.to_numeric(
                results.pop("regional_customer_log"), errors="coerce"
            ).where(has_selected_openings, 0.0)
            customer_change = np.expm1(regional_customer_log) * 100
            results[f"APSD_customers_change_pct_{size}"] = customer_change
            results[f"APSD_customers_change_{size}"] = (
                results["APSD_customers_baseline"]
                * np.expm1(regional_customer_log)
            )
            results[f"APSD_customers_expected_{size}"] = (
                results["APSD_customers_baseline"]
                + results[f"APSD_customers_change_{size}"]
            )
            results[f"customer_effect_supported_{size}"] = (
                ~has_selected_openings | results["regional_customer_terms"].eq(
                    results[f"nearby_new_stores_{size}"]
                )
            )
            results = results.drop(
                columns=["regional_customer_terms", "pair_count"],
                errors="ignore",
            )
        else:
            results[f"APSD_customers_change_pct_{size}"] = np.nan
            results[f"customer_effect_supported_{size}"] = False
        for band, _, _ in DISTANCE_BANDS:
            count_column = band
            output_column = f"nearby_{band}_{size}"
            if count_column in results.columns:
                results[output_column] = (
                    pd.to_numeric(results.pop(count_column), errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
            else:
                results[output_column] = 0
        negative_contributors = selected.loc[
            selected["coefficient_log_effect"] < 0
        ].sort_values(
            ["coefficient_log_effect", "distance_m", "candidate_rank"],
            ascending=[True, True, True],
        )
        top_openings: dict[Any, str] = {}
        for store_id, opening_rows in (
            negative_contributors.groupby("existing_store_id").head(5).groupby(
                "existing_store_id"
            )
        ):
            top_openings[store_id] = "; ".join(
                f"{row.candidate_name} "
                f"({row.distance_m:.0f}m, "
                f"{row.individual_APSD_effect_pct:+.2f}%)"
                for row in opening_rows.itertuples()
            )
        results[f"main_nearby_openings_{size}"] = (
            results["existing_store_id"].map(top_openings).fillna("")
        )
        results[f"event_time_months_{size}"] = 0

    return results, pairs


def summarize_scenarios(
    stores: pd.DataFrame,
    scenario_sizes: tuple[int, ...] = DEFAULT_SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for size in scenario_sizes:
        reduction = pd.to_numeric(
            stores[f"cannibalization_pct_{size}"], errors="coerce"
        ).fillna(0)
        income_change = pd.to_numeric(
            stores[f"APSD_income_change_pct_{size}"], errors="coerce"
        )
        affected = stores[f"nearby_new_stores_{size}"].gt(0)
        losses = income_change.lt(0)
        regional_reduction = pd.to_numeric(
            stores.get(
                f"regional_cannibalization_pct_{size}",
                pd.Series(np.nan, index=stores.index),
            ),
            errors="coerce",
        )
        customer_change = pd.to_numeric(
            stores.get(
                f"APSD_customers_change_pct_{size}",
                pd.Series(np.nan, index=stores.index),
            ),
            errors="coerce",
        )
        negative_sorted = stores.assign(_net_change=income_change).sort_values(
            ["_net_change", "existing_store_id"], ascending=[True, True]
        )
        worst = negative_sorted.iloc[0]
        scenario_rows.append(
            {
                "scenario_stores": size,
                "existing_stores_analyzed": int(len(stores)),
                "selected_unique_candidates": int(
                    stores.attrs.get(f"selected_candidate_count_{size}", 0)
                ),
                "existing_stores_with_nearby_opening": int(affected.sum()),
                "pct_existing_stores_with_nearby_opening": float(
                    affected.mean() * 100
                )
                if len(stores)
                else np.nan,
                "mean_reduction_pct_all_stores": float(reduction.mean()),
                "median_reduction_pct_all_stores": float(reduction.median()),
                "mean_reduction_pct_affected_stores": float(
                    reduction.loc[affected].mean()
                )
                if affected.any()
                else 0.0,
                "median_reduction_pct_affected_stores": float(
                    reduction.loc[affected].median()
                )
                if affected.any()
                else 0.0,
                "existing_stores_with_net_negative_effect": int(losses.sum()),
                "pct_existing_stores_with_net_negative_effect": float(
                    losses.mean() * 100
                )
                if len(stores)
                else np.nan,
                "stores_reduction_gt_5_pct": int(reduction.gt(5).sum()),
                "stores_reduction_gt_10_pct": int(reduction.gt(10).sum()),
                "stores_reduction_gt_20_pct": int(reduction.gt(20).sum()),
                "stores_reduction_gt_30_pct": int(reduction.gt(30).sum()),
                "most_affected_store_id": worst["existing_store_id"],
                "most_affected_store_name": worst["store_name"],
                "most_affected_APSD_change_pct": float(worst["_net_change"]),
                "mean_net_APSD_change_pct": float(income_change.mean()),
                "median_net_APSD_change_pct": float(income_change.median()),
                "mean_regional_model_reduction_pct_all_stores": float(
                    regional_reduction.mean()
                ),
                "median_regional_model_reduction_pct_all_stores": float(
                    regional_reduction.median()
                ),
                "mean_APSD_customers_change_pct_all_stores": float(
                    customer_change.mean()
                ),
                "median_APSD_customers_change_pct_all_stores": float(
                    customer_change.median()
                ),
            }
        )
        for region, group in stores.groupby("region", dropna=False, sort=True):
            region_reduction = pd.to_numeric(
                group[f"cannibalization_pct_{size}"], errors="coerce"
            ).fillna(0)
            region_affected = group[f"nearby_new_stores_{size}"].gt(0)
            regional_group_reduction = pd.to_numeric(
                group.get(
                    f"regional_cannibalization_pct_{size}",
                    pd.Series(np.nan, index=group.index),
                ),
                errors="coerce",
            )
            customer_group_change = pd.to_numeric(
                group.get(
                    f"APSD_customers_change_pct_{size}",
                    pd.Series(np.nan, index=group.index),
                ),
                errors="coerce",
            )
            region_rows.append(
                {
                    "scenario_stores": size,
                    "region": region if pd.notna(region) else "Unmapped",
                    "existing_stores": int(len(group)),
                    "existing_stores_with_nearby_opening": int(
                        region_affected.sum()
                    ),
                    "mean_reduction_pct_all_stores": float(
                        region_reduction.mean()
                    ),
                    "median_reduction_pct_all_stores": float(
                        region_reduction.median()
                    ),
                    "mean_reduction_pct_affected_stores": float(
                        region_reduction.loc[region_affected].mean()
                    )
                    if region_affected.any()
                    else 0.0,
                    "net_negative_stores": int(
                        pd.to_numeric(
                            group[f"APSD_income_change_pct_{size}"],
                            errors="coerce",
                        ).lt(0).sum()
                    ),
                    "mean_regional_model_reduction_pct_all_stores": float(
                        regional_group_reduction.mean()
                    ),
                    "median_regional_model_reduction_pct_all_stores": float(
                        regional_group_reduction.median()
                    ),
                    "mean_regional_model_reduction_pct_affected": float(
                        regional_group_reduction.loc[region_affected].mean()
                    )
                    if region_affected.any()
                    else 0.0,
                    "mean_APSD_customers_change_pct": float(
                        customer_group_change.mean()
                    ),
                }
            )
    return pd.DataFrame(scenario_rows), pd.DataFrame(region_rows)


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The regional implementation is kept separate so the legacy public helper
    # functions below remain available to existing tests and scripts.
    from regional_cumulative_impact import run_regional_analysis

    return run_regional_analysis()


if __name__ == "__main__":
    store_results, candidate_rows, contribution_rows, summary = run_analysis()
    print(f"Eligible unique GPS candidates: {len(candidate_rows):,}")
    print(f"Existing stores analyzed: {len(store_results):,}")
    print(summary.to_string(index=False))
