"""CU 2027 expansion-demand and network-impact simulation.

The module reuses saved CU panels and event-study estimates. It does not refit
the existing cannibalization models. APSD values are converted to monthly sales
only by multiplying by explicitly forecast active days.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


MODEL_VERSION = "cu-expansion-2027-v1"
OUTCOMES = ("sales", "customers")
DISTANCE_BANDS = (
    ("0_100", 0.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1000.0),
)
NUMERIC_FEATURES = (
    "stores_0_100m",
    "stores_100_250m",
    "stores_250_500m",
    "stores_500_1000m",
    "nearest_store_km",
    "neighbor_sales_1km",
    "neighbor_customers_1km",
)


@dataclass(frozen=True)
class SimulationConfig:
    forecast_year: int = 2027
    targets: tuple[int, ...] = (180, 400)
    schedule: str = "monthly_even"
    opening_day: int = 15
    cannibalization_scale: float = 1.0


def haversine_distances(
    longitude: Sequence[float] | np.ndarray,
    latitude: Sequence[float] | np.ndarray,
    other_longitude: Sequence[float] | np.ndarray,
    other_latitude: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return pairwise great-circle distances in meters (coordinates are lon, lat)."""
    lon_a = np.radians(np.asarray(longitude, dtype=float))[:, None]
    lat_a = np.radians(np.asarray(latitude, dtype=float))[:, None]
    lon_b = np.radians(np.asarray(other_longitude, dtype=float))[None, :]
    lat_b = np.radians(np.asarray(other_latitude, dtype=float))[None, :]
    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def distance_band_codes(distances: np.ndarray) -> np.ndarray:
    """Map distances to Model 4 bands; -1 denotes the unmodeled >1 km control."""
    values = np.asarray(distances, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int8)
    for code, (_, lower, upper) in enumerate(DISTANCE_BANDS):
        result[(values >= lower) & (values < upper)] = code
    return result


def schedule_months(
    target: int, weights: Sequence[float], forecast_year: int = 2027
) -> np.ndarray:
    """Allocate an exact target across weighted months using largest remainders."""
    if target < 0:
        raise ValueError("The target number of stores cannot be negative.")
    weight_values = np.asarray(weights, dtype=float)
    if weight_values.shape != (12,):
        raise ValueError("Opening schedule must contain exactly 12 monthly weights.")
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("Opening schedule weights must be finite and nonnegative.")
    if weight_values.sum() <= 0:
        raise ValueError("At least one opening schedule weight must be positive.")
    exact = target * weight_values / weight_values.sum()
    counts = np.floor(exact).astype(int)
    remainder = target - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
    return np.repeat(np.arange(12, dtype=int), counts)


def schedule_weights(schedule: str) -> np.ndarray:
    if schedule == "monthly_even":
        return np.ones(12, dtype=float)
    if schedule == "quarterly_even":
        weights = np.zeros(12, dtype=float)
        weights[[1, 4, 7, 10]] = 1.0
        return weights
    raise ValueError(
        "Schedule must be 'monthly_even', 'quarterly_even', or custom weights."
    )


def _coordinate_mask(frame: pd.DataFrame) -> pd.Series:
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    return (
        longitude.between(-180.0, 180.0)
        & latitude.between(-90.0, 90.0)
    )


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _candidate_frame(
    workbook: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Longitude", "Latitude"}
    missing = required.difference(workbook.columns)
    if missing:
        raise ValueError(
            f"Potential-location workbook is missing coordinates: {sorted(missing)}"
        )
    candidates = workbook.copy()
    candidates["longitude"] = pd.to_numeric(
        candidates["Longitude"], errors="coerce"
    )
    candidates["latitude"] = pd.to_numeric(candidates["Latitude"], errors="coerce")
    if "#" in candidates.columns and candidates["#"].notna().all():
        candidates["potential_store_id"] = candidates["#"].map(
            lambda value: f"GPS-{value}"
        )
    else:
        candidates["potential_store_id"] = [
            f"GPS-ROW-{index + 1}" for index in range(len(candidates))
        ]
    candidates["source_identifier"] = (
        candidates["#"] if "#" in candidates.columns else pd.NA
    )
    candidates["coordinate_valid"] = _coordinate_mask(candidates)
    candidates["duplicate_candidate_id"] = candidates[
        "potential_store_id"
    ].duplicated(keep=False)

    candidates["_coordinate_key"] = (
        candidates["longitude"].round(7).astype("string")
        + ":"
        + candidates["latitude"].round(7).astype("string")
    )
    coordinate_groups = candidates.loc[candidates["coordinate_valid"]].groupby(
        "_coordinate_key", sort=False
    )
    candidates["duplicate_coordinate_count"] = 0
    candidates.loc[candidates["coordinate_valid"], "duplicate_coordinate_count"] = (
        coordinate_groups["_coordinate_key"].transform("size").astype(int)
    )
    candidates["duplicate_coordinate"] = candidates[
        "duplicate_coordinate_count"
    ].gt(1)

    existing_coords = existing.loc[existing["coordinate_valid"]]
    if len(existing_coords) and candidates["coordinate_valid"].any():
        distances = haversine_distances(
            candidates.loc[candidates["coordinate_valid"], "longitude"].to_numpy(),
            candidates.loc[candidates["coordinate_valid"], "latitude"].to_numpy(),
            existing_coords["longitude"].to_numpy(),
            existing_coords["latitude"].to_numpy(),
        )
        candidate_rows = candidates.index[candidates["coordinate_valid"]]
        nearest_positions = distances.argmin(axis=1)
        nearest_distances = distances.min(axis=1)
        candidates["nearest_existing_store_m"] = np.nan
        candidates["nearest_existing_store_id"] = pd.NA
        candidates.loc[candidate_rows, "nearest_existing_store_m"] = nearest_distances
        candidates.loc[candidate_rows, "nearest_existing_store_id"] = (
            existing_coords.iloc[nearest_positions]["store_id"].astype(str).to_numpy()
        )
        candidates["matches_existing_location"] = (
            candidates["nearest_existing_store_m"].le(5.0)
        )
    else:
        candidates["nearest_existing_store_m"] = np.nan
        candidates["nearest_existing_store_id"] = pd.NA
        candidates["matches_existing_location"] = False

    candidates["area_type"] = (
        candidates["Area Type"].astype("string").str.strip()
        if "Area Type" in candidates.columns
        else pd.Series("Unknown", index=candidates.index, dtype="string")
    )
    candidates["demand_region"] = np.where(
        candidates["area_type"].str.casefold().eq("ub"),
        "UB",
        np.where(
            candidates["area_type"].str.casefold().eq("rural"),
            "Rural",
            "Rural",
        ),
    )
    candidates["region_mapping_rule"] = np.where(
        candidates["demand_region"].eq("UB"),
        "Source Area Type is UB; no validated sub-region assignment is supplied.",
        "Non-UB source area (Rural or named province) grouped as Rural for demand-model support.",
    )
    candidates = candidates.sort_values(
        ["potential_store_id"], kind="stable"
    ).reset_index(drop=True)
    candidates["selection_eligible"] = (
        candidates["coordinate_valid"]
        & ~candidates["duplicate_candidate_id"]
        & ~candidates["matches_existing_location"]
    )
    first_coordinate = ~candidates.duplicated("_coordinate_key", keep="first")
    candidates.loc[candidates["duplicate_coordinate"], "selection_eligible"] &= (
        first_coordinate
    )
    return candidates, coordinate_groups.size().rename("candidate_count").reset_index()


def _existing_store_frame(
    master: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    required = {"store_id", "LATITUDE", "LONGITUDE", "ost_dt"}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(
            f"Store master is missing required columns: {sorted(missing)}"
        )
    rows = master.copy()
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["store_id"] = rows["store_id"].astype("string")
    if rows.duplicated(["store_id", "SalesMonth"]).any():
        duplicates = int(rows.duplicated(["store_id", "SalesMonth"]).sum())
        raise ValueError(
            f"Store master contains {duplicates} duplicate store-month keys; "
            "the expansion simulation will not aggregate them silently."
        )
    metadata_columns = [
        column
        for column in (
            "store_id",
            "store_name",
            "LATITUDE",
            "LONGITUDE",
            "ost_dt",
            "cls_dt",
            "Store_Status",
            "loc_area_new",
            "UB",
            "Distance_Zone",
        )
        if column in rows.columns
    ]
    metadata = rows[metadata_columns].drop_duplicates("store_id").copy()
    metadata["longitude"] = pd.to_numeric(metadata["LONGITUDE"], errors="coerce")
    metadata["latitude"] = pd.to_numeric(metadata["LATITUDE"], errors="coerce")
    metadata["open_date"] = pd.to_datetime(metadata["ost_dt"], errors="coerce")
    metadata["close_date"] = (
        pd.to_datetime(metadata["cls_dt"], errors="coerce")
        if "cls_dt" in metadata.columns
        else pd.NaT
    )
    metadata["coordinate_valid"] = _coordinate_mask(metadata)
    if "analysis_region" in panel.columns:
        regions = (
            panel[["existing_store_id", "analysis_region"]]
            .drop_duplicates()
            .drop_duplicates("existing_store_id")
        )
        regions["existing_store_id"] = regions["existing_store_id"].astype("string")
        metadata = metadata.merge(
            regions.rename(columns={"existing_store_id": "store_id"}),
            on="store_id",
            how="left",
            validate="one_to_one",
        )
    else:
        metadata["analysis_region"] = pd.NA
    metadata["demand_region"] = np.where(
        metadata.get("UB", pd.Series(index=metadata.index, dtype="string"))
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("rural"),
        "Rural",
        "UB",
    )
    return metadata.reset_index(drop=True)


def _make_monthly_outcome_matrices(
    master: pd.DataFrame,
) -> tuple[pd.PeriodIndex, dict[str, pd.DataFrame]]:
    rows = master.copy()
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["store_id"] = rows["store_id"].astype("string")
    rows = rows.dropna(subset=["SalesMonth"])
    rows["_period"] = rows["SalesMonth"].dt.to_period("M")
    months = pd.period_range(rows["_period"].min(), rows["_period"].max(), freq="M")
    matrices: dict[str, pd.DataFrame] = {}
    for outcome, column in (
        ("sales", "APSD_income"),
        ("customers", "APSD_customers"),
    ):
        values = pd.to_numeric(rows[column], errors="coerce")
        wide = rows.assign(_value=values).pivot(
            index="_period", columns="store_id", values="_value"
        )
        matrices[outcome] = wide.reindex(months).sort_index().ffill()
    return months, matrices


def _location_features(
    longitude: float,
    latitude: float,
    open_date: pd.Timestamp,
    demand_region: str,
    store_meta: pd.DataFrame,
    periods: pd.PeriodIndex,
    matrices: dict[str, pd.DataFrame],
) -> dict[str, float | str]:
    month = pd.Timestamp(open_date).to_period("M")
    prior_position = int(periods.searchsorted(month, side="left") - 1)
    eligible = (
        store_meta["coordinate_valid"]
        & store_meta["open_date"].notna()
        & store_meta["open_date"].lt(month.start_time)
        & (
            store_meta["close_date"].isna()
            | store_meta["close_date"].ge(month.start_time)
        )
    )
    if not eligible.any():
        distances = np.array([], dtype=float)
        neighbor_ids = pd.Index([], dtype="string")
        neighbor_positions = np.array([], dtype=int)
    else:
        eligible_stores = store_meta.loc[eligible].reset_index(drop=True)
        distances = haversine_distances(
            [longitude],
            [latitude],
            eligible_stores["longitude"].to_numpy(),
            eligible_stores["latitude"].to_numpy(),
        )[0]
        neighbor_ids = eligible_stores["store_id"]
        neighbor_positions = np.flatnonzero(distances < 1000.0)

    result: dict[str, float | str] = {"demand_region": demand_region}
    lower = 0.0
    for feature, upper in (
        ("stores_0_100m", 100.0),
        ("stores_100_250m", 250.0),
        ("stores_250_500m", 500.0),
        ("stores_500_1000m", 1000.0),
    ):
        if feature == "stores_0_100m":
            result[feature] = float(np.count_nonzero(distances < upper))
        else:
            result[feature] = float(
                np.count_nonzero((distances >= lower) & (distances < upper))
            )
        lower = upper
    result["nearest_store_km"] = (
        float(distances.min() / 1000.0) if len(distances) else np.nan
    )
    for outcome, feature in (
        ("sales", "neighbor_sales_1km"),
        ("customers", "neighbor_customers_1km"),
    ):
        if prior_position >= 0 and len(neighbor_positions):
            neighbor_values = matrices[outcome].iloc[prior_position]
            ids = neighbor_ids.iloc[neighbor_positions]
            valid_values = pd.to_numeric(
                neighbor_values.reindex(ids).to_numpy(), errors="coerce"
            )
            valid_values = valid_values[np.isfinite(valid_values)]
            result[feature] = (
                float(np.mean(valid_values)) if len(valid_values) else np.nan
            )
        else:
            result[feature] = np.nan
    return result


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.reindex(columns=NUMERIC_FEATURES).apply(
        pd.to_numeric, errors="coerce"
    )
    category = frame.get(
        "demand_region", pd.Series("Rural", index=frame.index)
    ).astype("string")
    categorical = pd.DataFrame(
        {
            "region_UB": category.eq("UB").astype(float),
            "region_Rural": category.eq("Rural").astype(float),
        },
        index=frame.index,
    )
    return pd.concat([numeric, categorical], axis=1)


def _fit_ridge(
    train_x: pd.DataFrame, train_y: np.ndarray, alpha: float = 10.0
) -> dict[str, Any]:
    medians = train_x.median().fillna(0.0)
    imputed = train_x.fillna(medians)
    means = imputed.mean()
    scales = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    standardized = (imputed - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized.to_numpy()])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return {
        "medians": medians,
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
    }


def _predict_ridge(model: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    imputed = features.fillna(model["medians"])
    standardized = (imputed - model["means"]) / model["scales"]
    design = np.column_stack([np.ones(len(standardized)), standardized.to_numpy()])
    return design @ model["coefficients"]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2)) if len(actual) else 0.0
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))) if len(actual) else np.nan,
        "rmse": float(np.sqrt(np.mean(error**2))) if len(actual) else np.nan,
        "out_of_sample_r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if len(actual) > 1 and denominator > 0
            else np.nan
        ),
        "prediction_bias": float(np.mean(error)) if len(actual) else np.nan,
    }


def _fit_demand_models(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    features = _feature_matrix(training)
    candidate_x = _feature_matrix(candidate_features)
    train_mask = training["open_month"].lt(pd.Period("2025-01", freq="M"))
    test_mask = ~train_mask
    if train_mask.sum() < 30 or test_mask.sum() < 20:
        raise ValueError(
            "Time-based demand validation requires at least 30 training and "
            f"20 test stores; found {int(train_mask.sum())} and {int(test_mask.sum())}."
        )

    validation_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for outcome, target_column in (
        ("sales", "APSD_income"),
        ("customers", "APSD_customers"),
    ):
        target = pd.to_numeric(training[target_column], errors="coerce").to_numpy()
        valid_target = np.isfinite(target) & (target > 0)
        train_valid = train_mask.to_numpy() & valid_target
        test_valid = test_mask.to_numpy() & valid_target
        target_transform = np.log(target[train_valid])
        model = _fit_ridge(features.loc[train_valid], target_transform)
        validation_log = _predict_ridge(model, features.loc[test_valid])
        validation_prediction = np.exp(validation_log)
        actual = target[test_valid]
        baseline_by_region = (
            training.loc[train_valid]
            .assign(_actual=target[train_valid])
            .groupby("demand_region")["_actual"]
            .median()
        )
        validation_regions = training.loc[test_valid, "demand_region"].astype("string")
        validation_baseline = validation_regions.map(baseline_by_region).fillna(
            float(np.median(target[train_valid]))
        ).to_numpy(dtype=float)
        for model_name, predicted in (
            ("Training median baseline", validation_baseline),
            ("Regularized location-context model", validation_prediction),
        ):
            metric = _metrics(actual, predicted)
            validation_rows.append(
                {
                    "record_type": "time_holdout",
                    "outcome": outcome,
                    "model": model_name,
                    "train_opening_period": "Before 2025-01",
                    "test_opening_period": "2025-01 onward",
                    **metric,
                }
            )
        validation_frame = training.loc[test_valid].copy()
        validation_frame["actual"] = actual
        validation_frame["predicted"] = validation_prediction
        validation_frame["opening_cohort"] = (
            validation_frame["open_month"].astype(str).str[:4]
        )
        for grouping, column in (
            ("region", "demand_region"),
            ("opening_cohort", "opening_cohort"),
        ):
            for group_name, group in validation_frame.groupby(column, dropna=False):
                metric = _metrics(
                    group["actual"].to_numpy(), group["predicted"].to_numpy()
                )
                validation_rows.append(
                    {
                        "record_type": f"holdout_by_{grouping}",
                        "outcome": outcome,
                        "model": "Regularized location-context model",
                        "group": str(group_name),
                        **metric,
                    }
                )

        all_valid = valid_target
        final_model = _fit_ridge(features.loc[all_valid], np.log(target[all_valid]))
        final_log = _predict_ridge(final_model, candidate_x)
        final_prediction = np.exp(final_log)
        residuals = validation_log - np.log(actual)
        residual_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        lower = np.exp(final_log - 1.96 * residual_sd)
        upper = np.exp(final_log + 1.96 * residual_sd)
        predictions[outcome] = {
            "prediction": final_prediction,
            "lower": lower,
            "upper": upper,
            "validation_log_residual_sd": residual_sd,
            "model": final_model,
        }
    candidate_predictions = candidates.copy()
    for outcome in OUTCOMES:
        candidate_predictions[f"predicted_APSD_{outcome}"] = predictions[outcome][
            "prediction"
        ]
        candidate_predictions[f"predicted_APSD_{outcome}_lower_95"] = predictions[
            outcome
        ]["lower"]
        candidate_predictions[f"predicted_APSD_{outcome}_upper_95"] = predictions[
            outcome
        ]["upper"]
    return (
        candidate_predictions,
        pd.DataFrame(validation_rows),
        predictions,
    )


def _opening_training_data(
    master: pd.DataFrame,
    store_meta: pd.DataFrame,
    periods: pd.PeriodIndex,
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = master.copy()
    rows["store_id"] = rows["store_id"].astype("string")
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["_period"] = rows["SalesMonth"].dt.to_period("M")
    rows["ost_dt"] = pd.to_datetime(rows["ost_dt"], errors="coerce")
    rows["_opening_period"] = rows["ost_dt"].dt.to_period("M")
    target = rows.loc[
        rows["_period"].eq(rows["_opening_period"]),
        ["store_id", "ost_dt", "_opening_period", "APSD_income", "APSD_customers"],
    ].drop_duplicates(["store_id", "_opening_period"])
    metadata = store_meta[
        ["store_id", "longitude", "latitude", "coordinate_valid", "demand_region"]
    ]
    target = target.merge(
        metadata, on="store_id", how="left", validate="many_to_one"
    )
    target = target.loc[
        target["coordinate_valid"]
        & target["ost_dt"].notna()
        & target["APSD_income"].gt(0)
        & target["APSD_customers"].gt(0)
    ].copy()
    features = [
        _location_features(
            float(row.longitude),
            float(row.latitude),
            row.ost_dt,
            str(row.demand_region),
            store_meta,
            periods,
            matrices,
        )
        for row in target.itertuples(index=False)
    ]
    feature_frame = pd.DataFrame(features, index=target.index)
    result = pd.concat([target, feature_frame.drop(columns="demand_region")], axis=1)
    result["open_month"] = result["_opening_period"]
    result = result.rename(
        columns={
            "APSD_income": "APSD_income",
            "APSD_customers": "APSD_customers",
        }
    )
    return result


def _candidate_context(
    candidates: pd.DataFrame,
    store_meta: pd.DataFrame,
    periods: pd.PeriodIndex,
    matrices: dict[str, pd.DataFrame],
    latest_period: pd.Period,
) -> pd.DataFrame:
    reference_date = (latest_period + 1).start_time
    feature_rows = [
        _location_features(
            float(row.longitude),
            float(row.latitude),
            reference_date,
            str(row.demand_region),
            store_meta,
            periods,
            matrices,
        )
        for row in candidates.itertuples(index=False)
    ]
    return pd.DataFrame(feature_rows, index=candidates.index)


def _maturation_profile(master: pd.DataFrame) -> dict[str, dict[int, float]]:
    rows = master.copy()
    rows["store_id"] = rows["store_id"].astype("string")
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["ost_dt"] = pd.to_datetime(rows["ost_dt"], errors="coerce")
    rows["_month_index"] = rows["SalesMonth"].dt.year * 12 + rows["SalesMonth"].dt.month
    rows["_open_index"] = rows["ost_dt"].dt.year * 12 + rows["ost_dt"].dt.month
    rows["age_month"] = rows["_month_index"] - rows["_open_index"]
    opening = rows.loc[
        rows["age_month"].eq(0),
        ["store_id", "APSD_income", "APSD_customers"],
    ].drop_duplicates("store_id").set_index("store_id")
    rows = rows.loc[rows["age_month"].between(0, 11)].join(
        opening, on="store_id", rsuffix="_opening"
    )
    result: dict[str, dict[int, float]] = {}
    for outcome, column in (
        ("sales", "APSD_income"),
        ("customers", "APSD_customers"),
    ):
        ratio = pd.to_numeric(rows[column], errors="coerce") / pd.to_numeric(
            rows[f"{column}_opening"], errors="coerce"
        )
        valid = ratio.where(np.isfinite(ratio) & ratio.gt(0))
        grouped = valid.groupby(rows["age_month"]).agg(["median", "count"])
        result[outcome] = {0: 1.0}
        for age in range(1, 12):
            if age in grouped.index and grouped.loc[age, "count"] >= 20:
                result[outcome][age] = float(grouped.loc[age, "median"])
            else:
                result[outcome][age] = 1.0
    return result


def _seasonality(
    panel: pd.DataFrame,
    outcome_column: str,
) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    rows = panel.copy()
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["month_number"] = rows["SalesMonth"].dt.month
    rows["analysis_region"] = rows["analysis_region"].fillna("All Regions").astype(str)
    rows[outcome_column] = pd.to_numeric(rows[outcome_column], errors="coerce")
    rows = rows.loc[rows[outcome_column].gt(0)]
    overall = float(rows[outcome_column].mean()) if len(rows) else 1.0
    region_mean = rows.groupby("analysis_region")[outcome_column].mean()
    seasonal: dict[tuple[str, int], float] = {}
    for (region, month), value in rows.groupby(
        ["analysis_region", "month_number"]
    )[outcome_column].mean().items():
        denominator = float(region_mean.get(region, overall))
        seasonal[(str(region), int(month))] = (
            float(np.clip(value / denominator, 0.6, 1.4))
            if denominator > 0
            else 1.0
        )
    global_by_month = rows.groupby("month_number")[outcome_column].mean()
    global_mean = float(rows[outcome_column].mean()) if len(rows) else 1.0
    for month in range(1, 13):
        denominator = global_mean
        seasonal[("All Regions", month)] = (
            float(np.clip(global_by_month.get(month, denominator) / denominator, 0.6, 1.4))
            if denominator > 0
            else 1.0
        )
    return seasonal, dict(region_mean.astype(float))


def _existing_baselines(
    master: pd.DataFrame,
    panel: pd.DataFrame,
    store_meta: pd.DataFrame,
    forecast_year: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DatetimeIndex]:
    rows = master.copy()
    rows["SalesMonth"] = pd.to_datetime(rows["SalesMonth"], errors="coerce")
    rows["store_id"] = rows["store_id"].astype("string")
    rows = rows.loc[rows["SalesMonth"].le(pd.Timestamp("2026-10-31"))].copy()
    rows = rows.sort_values("SalesMonth")
    latest_period = rows["SalesMonth"].max().to_period("M")
    months = pd.date_range(
        f"{forecast_year}-01-01", f"{forecast_year}-12-01", freq="MS"
    )
    store_ids = store_meta["store_id"].astype("string").tolist()
    baselines: dict[str, np.ndarray] = {}

    days = pd.to_numeric(rows.get("Active_Days"), errors="coerce")
    days_by_month = rows.assign(_active_days=days).groupby(
        rows["SalesMonth"].dt.month
    )["_active_days"].median()
    active_days = np.array(
        [float(days_by_month.get(month.month, month.days_in_month)) for month in months],
        dtype=float,
    )
    active_days = np.minimum(active_days, np.array([m.days_in_month for m in months]))

    for outcome, column in (
        ("sales", "APSD_income"),
        ("customers", "APSD_customers"),
    ):
        valid = rows.loc[
            pd.to_numeric(rows[column], errors="coerce").gt(0),
            ["store_id", "SalesMonth", column],
        ].copy()
        valid["_rank"] = valid.groupby("store_id")["SalesMonth"].rank(
            method="first", ascending=False
        )
        levels = valid.loc[valid["_rank"].le(12)].groupby("store_id")[column].mean()
        seasonal, _ = _seasonality(panel, column)
        base = np.full((12, len(store_ids)), np.nan, dtype=float)
        for store_position, store in enumerate(store_meta.itertuples(index=False)):
            level = float(levels.get(str(store.store_id), np.nan))
            if not np.isfinite(level):
                continue
            region = (
                str(store.analysis_region)
                if pd.notna(store.analysis_region)
                else "All Regions"
            )
            for month_position, month in enumerate(months):
                base[month_position, store_position] = level * seasonal.get(
                    (region, month.month),
                    seasonal.get(("All Regions", month.month), 1.0),
                )
                close_date = getattr(store, "close_date")
                if pd.notna(close_date) and pd.Timestamp(close_date).to_period(
                    "M"
                ) < month.to_period("M"):
                    base[month_position, store_position] = 0.0
        baselines[outcome] = base

    return baselines, active_days, months


def _load_event_betas(
    outputs: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    betas: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    regions = (
        "All Regions",
        "City Center",
        "City Outskirts",
        "Far / Outer Areas",
        "Rural",
    )
    for outcome, filename in (
        ("sales", "regional_sales_coefficients.csv"),
        ("customers", "regional_customer_coefficients.csv"),
    ):
        path = outputs / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Saved {outcome} event-study coefficients were not found: {path}"
            )
        coefficients = pd.read_csv(path)
        selected = coefficients.loc[
            coefficients["model_type"].eq("monthly_event_study")
            & coefficients["event_time"].between(0, 11)
        ].copy()
        all_regions = selected.loc[selected["region"].eq("All Regions")]
        if all_regions.empty:
            raise ValueError(
                f"No all-region monthly event coefficients are available for {outcome}."
            )
        global_lookup = {
            (str(row.distance_band), int(row.event_time)): float(row.coefficient)
            for row in all_regions.itertuples(index=False)
        }
        if len(global_lookup) < len(DISTANCE_BANDS) * 12:
            raise ValueError(
                f"Incomplete all-region {outcome} Model 4 support for months 0-11."
            )
        by_region: dict[str, dict[tuple[str, int], float]] = {}
        for region in regions[1:]:
            subset = selected.loc[selected["region"].eq(region)]
            if subset.empty:
                continue
            region_lookup: dict[tuple[str, int], float] = {}
            for row in subset.itertuples(index=False):
                if int(row.exposed_stores) >= 20 and int(row.cluster_count) >= 10:
                    region_lookup[(str(row.distance_band), int(row.event_time))] = float(
                        row.coefficient
                    )
            for band, _, _ in DISTANCE_BANDS:
                supported = all((band, event) in region_lookup for event in range(12))
                if not supported:
                    region_lookup = {
                        key: value
                        for key, value in region_lookup.items()
                        if key[0] != band
                    }
                diagnostics.append(
                    {
                        "record_type": "regional_effect_support",
                        "outcome": outcome,
                        "region": region,
                        "distance_band": band,
                        "regional_months_used": (
                            12 if supported else 0
                        ),
                        "fallback": "All Regions" if not supported else "",
                        "minimum_exposed_stores": 20,
                        "minimum_clusters": 10,
                    }
                )
            by_region[region] = region_lookup
        betas[outcome] = {
            "global": global_lookup,
            "regional": by_region,
            "regions": regions,
        }
    return betas, pd.DataFrame(diagnostics)


def _beta_tensor(
    beta_model: dict[str, Any],
    receiver_regions: Sequence[str],
) -> np.ndarray:
    tensor = np.zeros((len(receiver_regions), len(DISTANCE_BANDS), 12), dtype=float)
    global_lookup = beta_model["global"]
    for region_index, region in enumerate(receiver_regions):
        regional = beta_model["regional"].get(str(region), {})
        for band_index, (band, _, _) in enumerate(DISTANCE_BANDS):
            for event_time in range(12):
                tensor[region_index, band_index, event_time] = regional.get(
                    (band, event_time),
                    global_lookup[(band, event_time)],
                )
    return tensor


def _candidate_band_matrices(
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    existing_valid = existing.loc[existing["coordinate_valid"]].copy().reset_index(
        drop=True
    )
    candidate_distances = haversine_distances(
        candidates["longitude"].to_numpy(dtype=float),
        candidates["latitude"].to_numpy(dtype=float),
        existing_valid["longitude"].to_numpy(dtype=float),
        existing_valid["latitude"].to_numpy(dtype=float),
    )
    candidate_bands = distance_band_codes(candidate_distances)
    pair_distances = haversine_distances(
        candidates["longitude"].to_numpy(dtype=float),
        candidates["latitude"].to_numpy(dtype=float),
        candidates["longitude"].to_numpy(dtype=float),
        candidates["latitude"].to_numpy(dtype=float),
    )
    pair_bands = distance_band_codes(pair_distances)
    np.fill_diagonal(pair_bands, -1)
    existing_rows = pd.DataFrame(
        {
            "potential_store_id": np.repeat(
                candidates["potential_store_id"].to_numpy(), len(existing_valid)
            ),
            "existing_store_id": np.tile(
                existing_valid["store_id"].astype(str).to_numpy(), len(candidates)
            ),
            "distance_m": candidate_distances.ravel(),
        }
    )
    band_labels = (
        "0-100m",
        "100-250m",
        "250-500m",
        "500-1,000m",
        ">1,000m / control",
    )
    existing_rows["distance_band"] = pd.Categorical(
        [band_labels[int(code)] for code in candidate_bands.ravel()],
        categories=band_labels,
        ordered=True,
    )
    rows, columns = np.triu_indices(len(candidates), k=1)
    pair_rows = pd.DataFrame(
        {
            "potential_store_id_a": candidates.iloc[rows]["potential_store_id"].to_numpy(),
            "potential_store_id_b": candidates.iloc[columns]["potential_store_id"].to_numpy(),
            "distance_m": pair_distances[rows, columns],
            "distance_band": [
                band_labels[int(code)] for code in pair_bands[rows, columns]
            ],
        }
    )
    return candidate_bands, pair_bands, existing_rows, pair_rows


def _portfolio_schedule(
    target: int,
    forecast_year: int,
    schedule: str,
    custom_weights: Sequence[float] | None,
    opening_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    if opening_day < 1 or opening_day > 28:
        raise ValueError("Opening day must be between 1 and 28.")
    weights = (
        np.asarray(custom_weights, dtype=float)
        if custom_weights is not None
        else schedule_weights(schedule)
    )
    months = schedule_months(target, weights, forecast_year)
    dates = pd.to_datetime(
        [
            pd.Timestamp(forecast_year, int(month) + 1, opening_day)
            for month in months
        ]
    )
    counts = np.bincount(months, minlength=12)
    return months, dates


def _greedy_portfolio(
    candidates: pd.DataFrame,
    candidate_bands: np.ndarray,
    pair_bands: np.ndarray,
    existing: pd.DataFrame,
    existing_bands: np.ndarray,
    baseline_sales: np.ndarray,
    existing_active_days: np.ndarray,
    candidate_sales: np.ndarray,
    sales_maturation: dict[int, float],
    new_active_days: np.ndarray,
    beta_model: dict[str, Any],
    target: int,
    forecast_year: int,
    schedule: str,
    custom_weights: Sequence[float] | None,
    opening_day: int,
    cannibalization_scale: float,
) -> tuple[list[int], list[pd.Timestamp], list[float], np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(candidates["selection_eligible"].to_numpy())
    requested = min(target, len(eligible))
    schedule_month, schedule_dates = _portfolio_schedule(
        requested, forecast_year, schedule, custom_weights, opening_day
    )
    month_numbers = np.arange(12)
    event_model = beta_model["sales"]
    existing_regions = existing["analysis_region"].fillna("All Regions").astype(str)
    region_values = event_model["regions"]
    region_codes = pd.Categorical(existing_regions, categories=region_values).codes
    region_codes = np.where(region_codes < 0, 0, region_codes)
    beta_by_store = _beta_tensor(event_model, existing_regions)
    baseline_monthly_revenue = np.nan_to_num(
        baseline_sales * existing_active_days[:, None], nan=0.0
    )
    active_candidate = np.where(
        existing_bands >= 0,
        existing_bands,
        -1,
    )
    existing_log_effect = np.zeros_like(baseline_sales, dtype=float)
    candidate_log_effect = np.zeros((12, len(candidates)), dtype=float)
    selected: list[int] = []
    selected_months: list[int] = []
    selected_dates: list[pd.Timestamp] = []
    marginal_values: list[float] = []

    for step in range(requested):
        open_month = int(schedule_month[step])
        choices = np.setdiff1d(eligible, np.asarray(selected, dtype=int), assume_unique=False)
        existing_delta = np.zeros(len(choices), dtype=float)
        own_revenue = np.zeros(len(choices), dtype=float)
        selected_new_delta = np.zeros(len(choices), dtype=float)
        for month in range(open_month, 12):
            event_time = month - open_month
            candidate_beta = np.zeros((len(choices), existing_bands.shape[1]), dtype=float)
            valid = existing_bands[choices] >= 0
            candidate_beta[valid] = beta_by_store[
                np.broadcast_to(
                    region_codes[None, :], existing_bands[choices].shape
                )[valid],
                existing_bands[choices][valid],
                event_time,
            ]
            candidate_beta *= cannibalization_scale
            existing_delta += (
                baseline_monthly_revenue[month][None, :]
                * np.exp(existing_log_effect[month])[None, :]
                * np.expm1(candidate_beta)
            ).sum(axis=1)

            ages = month - open_month
            fraction = (
                (pd.Timestamp(forecast_year, open_month + 1, opening_day).days_in_month
                 - opening_day + 1)
                / pd.Timestamp(forecast_year, open_month + 1, opening_day).days_in_month
                if month == open_month
                else 1.0
            )
            own_revenue += (
                candidate_sales[choices]
                * sales_maturation.get(ages, 1.0)
                * new_active_days[month]
                * (fraction if month == open_month else 1.0)
                * np.exp(candidate_log_effect[month, choices])
            )

            if selected:
                chosen = np.asarray(selected, dtype=int)
                selected_band_codes = pair_bands[np.ix_(choices, chosen)]
                valid_selected = selected_band_codes >= 0
                selected_beta = np.zeros(selected_band_codes.shape, dtype=float)
                selected_beta[valid_selected] = event_model["global"][
                    "matrix"
                ][selected_band_codes[valid_selected], event_time]
                selected_beta *= cannibalization_scale
                selected_open_months = np.asarray(selected_months)
                open_selected = selected_open_months <= month
                if open_selected.any():
                    selected_age = month - selected_open_months
                    selected_revenue = (
                        candidate_sales[chosen]
                        * np.array(
                            [
                                sales_maturation.get(int(age), 1.0)
                                for age in selected_age
                            ]
                        )
                        * new_active_days[month]
                    )
                    selected_fraction = np.where(
                        selected_open_months == month,
                        np.array(
                            [
                                (pd.Timestamp(forecast_year, int(m) + 1, opening_day).days_in_month
                                 - opening_day + 1)
                                / pd.Timestamp(forecast_year, int(m) + 1, opening_day).days_in_month
                                for m in selected_open_months
                            ]
                        ),
                        1.0,
                    )
                    selected_revenue *= selected_fraction
                    selected_delta = (
                        selected_revenue[None, :]
                        * np.exp(candidate_log_effect[month, chosen])[None, :]
                        * np.expm1(selected_beta)
                    )
                    selected_new_delta += selected_delta[:, open_selected].sum(axis=1)

        score = existing_delta + own_revenue + selected_new_delta
        winner_position = int(np.argmax(score))
        winner = int(choices[winner_position])
        selected.append(winner)
        selected_months.append(open_month)
        selected_dates.append(schedule_dates[step])
        marginal_values.append(float(score[winner_position]))

        for month in range(open_month, 12):
            event_time = month - open_month
            band_to_stores = existing_bands[winner]
            mask = band_to_stores >= 0
            if mask.any():
                existing_log_effect[month, mask] += (
                    beta_by_store[
                        region_codes[mask], band_to_stores[mask], event_time
                    ]
                    * cannibalization_scale
                )
            to_candidates = pair_bands[:, winner]
            candidate_mask = to_candidates >= 0
            if candidate_mask.any():
                candidate_log_effect[month, candidate_mask] += (
                    event_model["global"]["matrix"][
                        to_candidates[candidate_mask], event_time
                    ]
                    * cannibalization_scale
                )
    return (
        selected,
        selected_dates,
        marginal_values,
        existing_log_effect,
        candidate_log_effect,
    )


def _standalone_value(
    candidates: pd.DataFrame,
    candidate_bands: np.ndarray,
    existing_bands: np.ndarray,
    existing: pd.DataFrame,
    baseline_sales: np.ndarray,
    active_days: np.ndarray,
    candidate_sales: np.ndarray,
    maturity: dict[int, float],
    beta_model: dict[str, Any],
    forecast_year: int,
    cannibalization_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_model = beta_model["sales"]
    existing_regions = existing["analysis_region"].fillna("All Regions").astype(str)
    tensor = _beta_tensor(event_model, existing_regions)
    region_codes = pd.Categorical(
        existing_regions, categories=event_model["regions"]
    ).codes
    region_codes = np.where(region_codes < 0, 0, region_codes)
    baseline_revenue = baseline_sales * active_days[:, None]
    gross = np.zeros(len(candidates), dtype=float)
    displaced = np.zeros(len(candidates), dtype=float)
    max_store_loss = np.zeros(len(candidates), dtype=float)
    for event_time in range(12):
        fraction = 1.0
        new_revenue = (
            candidate_sales
            * maturity.get(event_time, 1.0)
            * active_days[event_time]
            * fraction
        )
        gross += new_revenue
        beta = np.zeros_like(existing_bands, dtype=float)
        valid = existing_bands >= 0
        if valid.any():
            beta[valid] = tensor[
                np.broadcast_to(region_codes[None, :], existing_bands.shape)[valid],
                existing_bands[valid],
                event_time,
            ]
        beta *= cannibalization_scale
        deltas = baseline_revenue[event_time][None, :] * np.expm1(beta)
        displaced += deltas.sum(axis=1)
        max_store_loss = np.minimum(max_store_loss, deltas.min(axis=1))
    net = gross + displaced
    return gross, displaced, max_store_loss


def _scenario_outputs(
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
    candidate_bands: np.ndarray,
    pair_bands: np.ndarray,
    existing_bands: np.ndarray,
    existing_distances: pd.DataFrame,
    baseline: dict[str, np.ndarray],
    active_days: np.ndarray,
    new_active_days: np.ndarray,
    months: pd.DatetimeIndex,
    maturity: dict[str, dict[int, float]],
    beta_models: dict[str, dict[str, Any]],
    target: int,
    forecast_year: int,
    schedule: str,
    custom_weights: Sequence[float] | None,
    opening_day: int,
    cannibalization_scale: float,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sales_matrix = baseline["sales"]
    selected, dates, marginal, existing_log_sales, candidate_log = _greedy_portfolio(
        candidates,
        candidate_bands,
        pair_bands,
        existing,
        existing_bands,
        sales_matrix,
        active_days,
        predictions["sales"],
        maturity["sales"],
        new_active_days,
        beta_models,
        target,
        forecast_year,
        schedule,
        custom_weights,
        opening_day,
        cannibalization_scale,
    )
    open_month_by_candidate = dict(zip(selected, [date.month - 1 for date in dates]))
    selected_set = set(selected)
    selected_output = candidates.iloc[selected].copy().reset_index(drop=True)
    selected_output["selection_rank"] = np.arange(1, len(selected) + 1)
    selected_output["assumed_opening_date"] = dates
    selected_output["greedy_marginal_network_sales_2027_mnt"] = marginal
    selected_output["scenario_target"] = target

    baseline_active = np.where(np.isfinite(sales_matrix), active_days[:, None], 0.0)
    simulated_sales_apsd = sales_matrix * np.exp(existing_log_sales)
    simulated_sales_apsd = np.where(baseline_active > 0, simulated_sales_apsd, 0.0)
    simulated_customers_apsd = baseline["customers"].copy()
    customer_log = np.zeros_like(existing_log_sales)
    for candidate, opening_date in zip(selected, dates):
        open_month = opening_date.month - 1
        for month in range(open_month, 12):
            event_time = month - open_month
            valid = existing_bands[candidate] >= 0
            if valid.any():
                bands = existing_bands[candidate, valid]
                regions = existing.loc[valid, "analysis_region"].fillna(
                    "All Regions"
                ).astype(str)
                tensor = _beta_tensor(beta_models["customers"], regions)
                codes = pd.Categorical(
                    regions, categories=beta_models["customers"]["regions"]
                ).codes
                codes = np.where(codes < 0, 0, codes)
                customer_log[month, valid] += (
                    tensor[codes, bands, event_time] * cannibalization_scale
                )
    simulated_customers_apsd *= np.exp(customer_log)
    simulated_customers_apsd = np.where(
        baseline_active > 0, simulated_customers_apsd, 0.0
    )

    existing_rows: list[pd.DataFrame] = []
    existing_summary = existing[["store_id", "store_name", "analysis_region"]].copy()
    existing_summary["scenario_target"] = target
    existing_summary["baseline_APSD_income_mean_2027"] = np.nanmean(sales_matrix, axis=0)
    existing_summary["simulated_APSD_income_mean_2027"] = np.nanmean(
        simulated_sales_apsd, axis=0
    )
    existing_summary["mean_APSD_income_change_2027"] = (
        existing_summary["simulated_APSD_income_mean_2027"]
        - existing_summary["baseline_APSD_income_mean_2027"]
    )
    existing_summary["baseline_APSD_customers_mean_2027"] = np.nanmean(
        baseline["customers"], axis=0
    )
    existing_summary["simulated_APSD_customers_mean_2027"] = np.nanmean(
        simulated_customers_apsd, axis=0
    )
    existing_summary["mean_APSD_customers_change_2027"] = (
        existing_summary["simulated_APSD_customers_mean_2027"]
        - existing_summary["baseline_APSD_customers_mean_2027"]
    )
    existing_summary["baseline_sales_2027_mnt"] = np.nansum(
        sales_matrix * active_days[:, None], axis=0
    )
    existing_summary["simulated_sales_2027_mnt"] = np.nansum(
        simulated_sales_apsd * active_days[:, None], axis=0
    )
    existing_summary["sales_change_2027_mnt"] = (
        existing_summary["simulated_sales_2027_mnt"]
        - existing_summary["baseline_sales_2027_mnt"]
    )
    existing_summary["customer_equivalent_change_2027"] = np.nansum(
        (simulated_customers_apsd - baseline["customers"]) * active_days[:, None],
        axis=0,
    )
    existing_summary = existing_summary.sort_values(
        "sales_change_2027_mnt", kind="stable"
    ).reset_index(drop=True)
    existing_summary["negative_impact_rank"] = np.arange(
        1, len(existing_summary) + 1
    )

    monthly_rows: list[dict[str, Any]] = []
    selected_month = np.zeros(12, dtype=int)
    candidate_revenue = np.zeros(12, dtype=float)
    candidate_customer_equiv = np.zeros(12, dtype=float)
    candidate_sales_apsd = np.full((12, len(candidates)), np.nan, dtype=float)
    candidate_customer_apsd = np.full((12, len(candidates)), np.nan, dtype=float)
    for index, (candidate, opening_date) in enumerate(zip(selected, dates)):
        open_month = opening_date.month - 1
        selected_month[open_month:] += 1
        for month in range(open_month, 12):
            age = month - open_month
            fraction = 1.0
            if month == open_month:
                fraction = (
                    pd.Timestamp(forecast_year, open_month + 1, opening_day).days_in_month
                    - opening_day
                    + 1
                ) / pd.Timestamp(
                    forecast_year, open_month + 1, opening_day
                ).days_in_month
            candidate_sales_apsd[month, candidate] = (
                predictions["sales"][candidate]
                * maturity["sales"].get(age, 1.0)
                * np.exp(candidate_log[month, candidate])
            )
            candidate_customer_apsd[month, candidate] = (
                predictions["customers"][candidate]
                * maturity["customers"].get(age, 1.0)
                * np.exp(_candidate_customer_log(
                    candidate, month, selected, dates, pair_bands,
                    beta_models["customers"], cannibalization_scale
                ))
            )
            candidate_revenue[month] += (
                candidate_sales_apsd[month, candidate]
                * new_active_days[month]
                * fraction
            )
            candidate_customer_equiv[month] += (
                candidate_customer_apsd[month, candidate]
                * new_active_days[month]
                * fraction
            )

    baseline_existing_revenue = np.nansum(
        sales_matrix * active_days[:, None], axis=1
    )
    simulated_existing_revenue = np.nansum(
        simulated_sales_apsd * active_days[:, None], axis=1
    )
    baseline_customer_equiv = np.nansum(
        baseline["customers"] * active_days[:, None], axis=1
    )
    simulated_existing_customer_equiv = np.nansum(
        simulated_customers_apsd * active_days[:, None], axis=1
    )
    for month in range(12):
        monthly_rows.append(
            {
                "scenario_target": target,
                "month": months[month],
                "selected_open_stores_to_date": int(selected_month[month]),
                "baseline_existing_sales_mnt": baseline_existing_revenue[month],
                "simulated_existing_sales_mnt": simulated_existing_revenue[month],
                "existing_sales_change_mnt": (
                    simulated_existing_revenue[month]
                    - baseline_existing_revenue[month]
                ),
                "new_store_sales_mnt": candidate_revenue[month],
                "baseline_network_sales_mnt": baseline_existing_revenue[month],
                "simulated_network_sales_mnt": (
                    simulated_existing_revenue[month] + candidate_revenue[month]
                ),
                "network_sales_change_mnt": (
                    simulated_existing_revenue[month]
                    - baseline_existing_revenue[month]
                    + candidate_revenue[month]
                ),
                "existing_customer_equivalent_change": (
                    simulated_existing_customer_equiv[month]
                    - baseline_customer_equiv[month]
                ),
                "new_store_customer_equivalents": candidate_customer_equiv[month],
                "network_customer_equivalent_change": (
                    simulated_existing_customer_equiv[month]
                    - baseline_customer_equiv[month]
                    + candidate_customer_equiv[month]
                ),
            }
        )

    monthly = pd.DataFrame(monthly_rows)
    chosen_ranks = dict(zip(selected, range(1, len(selected) + 1)))
    selected_output["scenario_name"] = f"{target}_selected"
    selected_output["scenario_opening_month"] = selected_output[
        "assumed_opening_date"
    ].dt.month

    # Event-level contributions are allocated in the greedy selection order.
    pair_impacts = _pair_impact_rows(
        candidates,
        existing,
        selected,
        dates,
        existing_bands,
        baseline,
        active_days,
        beta_models,
        forecast_year,
        cannibalization_scale,
        target,
    )

    # Persist monthly detail so APSD changes are never presented as total revenue.
    for month in range(12):
        ids = existing["store_id"].astype(str).to_numpy()
        detail = pd.DataFrame(
            {
                "scenario_target": target,
                "existing_store_id": ids,
                "month": months[month],
                "baseline_APSD_income": sales_matrix[month],
                "simulated_APSD_income": simulated_sales_apsd[month],
                "APSD_income_change": (
                    simulated_sales_apsd[month] - sales_matrix[month]
                ),
                "baseline_APSD_customers": baseline["customers"][month],
                "simulated_APSD_customers": simulated_customers_apsd[month],
                "APSD_customers_change": (
                    simulated_customers_apsd[month] - baseline["customers"][month]
                ),
                "forecast_active_days": active_days[month],
                "sales_revenue_change_mnt": (
                    simulated_sales_apsd[month] - sales_matrix[month]
                )
                * active_days[month],
                "customer_equivalent_change": (
                    simulated_customers_apsd[month] - baseline["customers"][month]
                )
                * active_days[month],
            }
        )
        existing_rows.append(detail)
    existing_monthly = pd.concat(existing_rows, ignore_index=True)

    candidate_summary = candidates.copy()
    candidate_summary["scenario_target"] = target
    candidate_summary["portfolio_selection_rank"] = candidate_summary.index.map(
        chosen_ranks
    )
    candidate_summary["selected"] = candidate_summary.index.isin(selected)
    candidate_summary["assumed_opening_date"] = candidate_summary.index.map(
        dict(zip(selected, dates))
    )
    candidate_summary["greedy_marginal_network_sales_2027_mnt"] = (
        candidate_summary.index.map(dict(zip(selected, marginal)))
    )
    candidate_summary["scenario_gross_sales_2027_mnt"] = np.nan
    candidate_summary.loc[selected, "scenario_gross_sales_2027_mnt"] = [
        float(candidate_revenue.sum() * 0 + np.nansum(
            candidate_sales_apsd[:, candidate] * new_active_days
        ))
        for candidate in selected
    ]

    summary = {
        "scenario_target": target,
        "selected_count": len(selected),
        "eligible_candidate_count": int(candidates["selection_eligible"].sum()),
        "shortfall": max(0, target - len(selected)),
        "opening_schedule": schedule,
        "opening_day": opening_day,
        "forecast_year": forecast_year,
        "cannibalization_scale": cannibalization_scale,
        "network_sales_change_2027_mnt": float(monthly["network_sales_change_mnt"].sum()),
        "existing_sales_change_2027_mnt": float(monthly["existing_sales_change_mnt"].sum()),
        "new_store_sales_2027_mnt": float(monthly["new_store_sales_mnt"].sum()),
        "customer_equivalent_change_2027": float(
            monthly["network_customer_equivalent_change"].sum()
        ),
        "portfolio_net_marginal_values_sum_mnt": float(sum(marginal)),
    }
    return (
        candidate_summary,
        selected_output,
        existing_summary,
        monthly,
        {
            "summary": summary,
            "existing_monthly": existing_monthly,
            "pair_impacts": pair_impacts,
            "existing_log_sales": existing_log_sales,
            "candidate_log_sales": candidate_log,
        },
    )


def _candidate_customer_log(
    candidate: int,
    month: int,
    selected: list[int],
    dates: list[pd.Timestamp],
    pair_bands: np.ndarray,
    beta_model: dict[str, Any],
    scale: float,
) -> float:
    total = 0.0
    for source, opening_date in zip(selected, dates):
        event_time = month - (opening_date.month - 1)
        band = int(pair_bands[candidate, source])
        if 0 <= event_time <= 11 and band >= 0:
            total += (
                beta_model["global"]["matrix"][band, event_time] * scale
            )
    return total


def _pair_impact_rows(
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
    selected: list[int],
    dates: list[pd.Timestamp],
    existing_bands: np.ndarray,
    baseline: dict[str, np.ndarray],
    active_days: np.ndarray,
    beta_models: dict[str, dict[str, Any]],
    forecast_year: int,
    scale: float,
    target: int,
) -> pd.DataFrame:
    sales_model = beta_models["sales"]
    customer_model = beta_models["customers"]
    store_regions = existing["analysis_region"].fillna("All Regions").astype(str)
    sales_tensor = _beta_tensor(sales_model, store_regions)
    customer_tensor = _beta_tensor(customer_model, store_regions)
    region_codes = pd.Categorical(
        store_regions, categories=sales_model["regions"]
    ).codes
    region_codes = np.where(region_codes < 0, 0, region_codes)
    current_log_sales = np.zeros((12, len(existing)), dtype=float)
    current_log_customers = np.zeros((12, len(existing)), dtype=float)
    rows: list[pd.DataFrame] = []
    for candidate, opening_date in zip(selected, dates):
        open_month = opening_date.month - 1
        affected = np.flatnonzero(existing_bands[candidate] >= 0)
        if not len(affected):
            continue
        for month in range(open_month, 12):
            event_time = month - open_month
            band_codes = existing_bands[candidate, affected]
            sales_beta = (
                sales_tensor[region_codes[affected], band_codes, event_time] * scale
            )
            customer_beta = (
                customer_tensor[region_codes[affected], band_codes, event_time] * scale
            )
            sales_before = baseline["sales"][month, affected] * np.exp(
                current_log_sales[month, affected]
            )
            customers_before = baseline["customers"][month, affected] * np.exp(
                current_log_customers[month, affected]
            )
            days = active_days[month]
            rows.append(
                pd.DataFrame(
                    {
                        "scenario_target": target,
                        "potential_store_id": candidates.iloc[candidate][
                            "potential_store_id"
                        ],
                        "existing_store_id": existing.iloc[affected][
                            "store_id"
                        ].astype(str).to_numpy(),
                        "assumed_opening_date": opening_date,
                        "month": pd.Timestamp(forecast_year, month + 1, 1),
                        "event_time_month": event_time,
                        "distance_m": np.nan,
                        "distance_band": [
                            "0-100m",
                            "100-250m",
                            "250-500m",
                            "500-1,000m",
                        ][band_codes],
                        "log_APSD_income_effect": sales_beta,
                        "APSD_income_percent_effect": 100.0 * np.expm1(sales_beta),
                        "APSD_income_change": sales_before * np.expm1(sales_beta),
                        "sales_revenue_change_mnt": (
                            sales_before * np.expm1(sales_beta) * days
                        ),
                        "log_APSD_customers_effect": customer_beta,
                        "APSD_customers_percent_effect": 100.0
                        * np.expm1(customer_beta),
                        "APSD_customers_change": (
                            customers_before * np.expm1(customer_beta)
                        ),
                        "customer_equivalent_change": (
                            customers_before * np.expm1(customer_beta) * days
                        ),
                        "effect_allocation": (
                            "Sequential marginal allocation in portfolio selection order"
                        ),
                    }
                )
            )
            current_log_sales[month, affected] += sales_beta
            current_log_customers[month, affected] += customer_beta
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_and_validate_inputs(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    master_path = base_dir / "store_master_data.xlsx"
    pair_path = base_dir / "store_to_store_distance.xlsx"
    gps_path = base_dir / "Data" / "raw" / "New Store GPS.xlsx"
    regional_panel_path = (
        base_dir
        / "outputs"
        / "cannibalization_regional"
        / "cu_regional_cannibalization_panel.parquet"
    )
    for path in (master_path, pair_path, gps_path, regional_panel_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required expansion input was not found: {path}")
    master = pd.read_excel(master_path, sheet_name="Sheet1")
    pair_data = pd.read_excel(pair_path, sheet_name="Sheet1")
    candidates_source = pd.read_excel(gps_path, sheet_name=0)
    panel = pd.read_parquet(regional_panel_path)
    if "SalesMonth" not in panel:
        raise ValueError("The saved regional panel has no SalesMonth column.")
    panel["SalesMonth"] = pd.to_datetime(panel["SalesMonth"], errors="coerce")
    if panel.duplicated(["existing_store_id", "SalesMonth"]).any():
        raise ValueError("Saved regional panel contains duplicate store-month rows.")
    existing = _existing_store_frame(master, panel)
    candidates, coordinate_counts = _candidate_frame(candidates_source, existing)
    pair_keys = (
        pair_data[["existing_store_id", "new_store_id"]]
        if {"existing_store_id", "new_store_id"}.issubset(pair_data.columns)
        else pd.DataFrame()
    )
    data_quality = {
        "master_rows": int(len(master)),
        "master_unique_stores": int(master["store_id"].nunique()),
        "master_duplicate_store_month_keys": int(
            master.duplicated(["store_id", "SalesMonth"]).sum()
        ),
        "pair_rows": int(len(pair_data)),
        "pair_unique_existing_new_pairs": int(len(pair_keys.drop_duplicates()))
        if len(pair_keys)
        else 0,
        "pair_duplicate_existing_new_pairs": int(pair_keys.duplicated().sum())
        if len(pair_keys)
        else 0,
        "candidate_rows": int(len(candidates)),
        "candidate_unique_source_ids": int(candidates["potential_store_id"].nunique()),
        "candidate_missing_or_invalid_coordinates": int(
            (~candidates["coordinate_valid"]).sum()
        ),
        "candidate_unique_valid_coordinate_pairs": int(
            candidates.loc[candidates["coordinate_valid"], "_coordinate_key"].nunique()
        ),
        "candidate_duplicate_coordinate_rows": int(
            candidates["duplicate_coordinate"].sum()
        ),
        "candidate_exact_existing_location_matches": int(
            candidates["matches_existing_location"].sum()
        ),
        "candidate_eligible_unique_locations": int(
            candidates["selection_eligible"].sum()
        ),
        "candidate_missing_names": int(
            candidates["Potential Point"].isna().sum()
            if "Potential Point" in candidates
            else len(candidates)
        ),
        "candidate_area_types": json.dumps(
            {
                str(key): int(value)
                for key, value in candidates["area_type"].value_counts(
                    dropna=False
                ).items()
            },
            ensure_ascii=True,
        ),
        "existing_unique_stores": int(existing["store_id"].nunique()),
        "existing_stores_with_valid_coordinates": int(
            existing["coordinate_valid"].sum()
        ),
        "historical_panel_months": int(panel["SalesMonth"].nunique()),
        "historical_panel_start": str(panel["SalesMonth"].min().date()),
        "historical_panel_end": str(panel["SalesMonth"].max().date()),
    }
    return master, pair_data, panel, candidates, data_quality


def run_expansion_simulation(
    base_dir: str | Path,
    config: SimulationConfig = SimulationConfig(),
    custom_weights: Sequence[float] | None = None,
    export: bool = True,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Build candidate forecasts and independently optimize each requested portfolio."""
    base = Path(base_dir).resolve()
    output_dir = base / "outputs" / "expansion_simulation"
    model_output_dir = base / "outputs" / "cannibalization_regional"
    master, pair_data, panel, candidates, quality = _load_and_validate_inputs(base)
    store_meta = _existing_store_frame(master, panel)
    periods, matrices = _make_monthly_outcome_matrices(master)
    latest_period = periods[-1]

    training = _opening_training_data(master, store_meta, periods, matrices)
    candidate_features = _candidate_context(
        candidates, store_meta, periods, matrices, latest_period
    )
    candidate_predictions, validation, demand_models = _fit_demand_models(
        training, candidates, candidate_features
    )
    validation["train_opening_month_count"] = int(
        training["open_month"].lt(pd.Period("2025-01", freq="M")).sum()
    )
    validation["test_opening_month_count"] = int(
        training["open_month"].ge(pd.Period("2025-01", freq="M")).sum()
    )
    validation["historical_demand_rows"] = len(training)
    validation["demand_model_features"] = ", ".join(
        (*NUMERIC_FEATURES, "demand_region")
    )
    validation["validation_note"] = (
        "Time holdout: stores opened before 2025 train the model; stores opened "
        "during 2025-2026 are held out. This predicts opening-month APSD, not "
        "guaranteed future performance."
    )

    beta_models, effect_support = _load_event_betas(model_output_dir)
    model_status_path = model_output_dir / "model_status.csv"
    statuses = pd.read_csv(model_status_path)
    for outcome in OUTCOMES:
        all_region = statuses.loc[
            statuses["region"].eq("All Regions")
            & statuses["outcome"].eq(outcome)
            & statuses["model_type"].eq("monthly_event_study")
        ]
        if len(all_region):
            validation = pd.concat(
                [
                    validation,
                    pd.DataFrame(
                        [
                            {
                                "record_type": "cannibalization_model_diagnostic",
                                "outcome": outcome,
                                "model": "Saved all-region distance-by-event-time model",
                                "observations": int(all_region.iloc[0]["observations"]),
                                "stores": int(all_region.iloc[0]["stores"]),
                                "within_r_squared": float(
                                    all_region.iloc[0]["within_r_squared"]
                                ),
                                "pretrend_joint_p_value": float(
                                    all_region.iloc[0]["pretrend_joint_p_value"]
                                ),
                                "source": (
                                    "outputs/cannibalization_regional/"
                                    f"regional_{'sales' if outcome == 'sales' else 'customer'}_coefficients.csv"
                                ),
                                "interpretation": (
                                    "Saved fixed-effect event-study association; "
                                    "significant pre-trends weaken causal interpretation."
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    validation = pd.concat([validation, effect_support], ignore_index=True, sort=False)

    eligible_existing = store_meta.loc[store_meta["coordinate_valid"]].copy()
    candidates_for_distance = candidate_predictions.copy()
    candidate_bands, pair_bands, candidate_existing_distances, pair_distances = (
        _candidate_band_matrices(candidates_for_distance, eligible_existing)
    )
    valid_existing_indices = eligible_existing.index.to_numpy()
    # Keep exactly the columns that align with distance-matrix target stores.
    existing_for_sim = eligible_existing.reset_index(drop=True)
    existing_bands = candidate_bands.copy()
    # Recompute candidate-to-existing codes with the validated target ordering.
    existing_bands = distance_band_codes(
        haversine_distances(
            candidates_for_distance["longitude"].to_numpy(),
            candidates_for_distance["latitude"].to_numpy(),
            existing_for_sim["longitude"].to_numpy(),
            existing_for_sim["latitude"].to_numpy(),
        )
    )
    candidate_bands = existing_bands
    baseline_all, active_days, forecast_months = _existing_baselines(
        master, panel, store_meta, config.forecast_year
    )
    baseline = {
        outcome: baseline_all[outcome][:, valid_existing_indices]
        for outcome in OUTCOMES
    }
    active_days_by_existing = np.tile(active_days[:, None], (1, len(existing_for_sim)))
    candidate_active_days = active_days.copy()
    maturations = _maturation_profile(master)
    active_days_open_month = candidate_active_days.copy()
    beta_models["sales"]["global"]["matrix"] = np.array(
        [
            [
                beta_models["sales"]["global"][(band, event)]
                for event in range(12)
            ]
            for band, _, _ in DISTANCE_BANDS
        ]
    )
    beta_models["customers"]["global"]["matrix"] = np.array(
        [
            [
                beta_models["customers"]["global"][(band, event)]
                for event in range(12)
            ]
            for band, _, _ in DISTANCE_BANDS
        ]
    )
    sales_predictions = candidate_predictions[
        "predicted_APSD_sales"
    ].to_numpy(dtype=float)
    customer_predictions = candidate_predictions[
        "predicted_APSD_customers"
    ].to_numpy(dtype=float)
    predictions = {"sales": sales_predictions, "customers": customer_predictions}

    standalone_gross, standalone_existing_delta, standalone_largest_loss = (
        _standalone_value(
            candidate_predictions,
            candidate_bands,
            existing_bands,
            existing_for_sim,
            baseline["sales"],
            active_days,
            sales_predictions,
            maturations["sales"],
            beta_models,
            config.forecast_year,
            config.cannibalization_scale,
        )
    )
    candidate_predictions["standalone_gross_sales_2027_mnt_jan_open"] = (
        standalone_gross
    )
    candidate_predictions["standalone_existing_sales_change_2027_mnt_jan_open"] = (
        standalone_existing_delta
    )
    candidate_predictions["standalone_net_network_sales_2027_mnt_jan_open"] = (
        standalone_gross + standalone_existing_delta
    )
    candidate_predictions["largest_existing_store_sales_change_2027_mnt"] = (
        standalone_largest_loss
    )
    candidate_predictions["affected_existing_stores_within_1km"] = (
        (candidate_bands >= 0).sum(axis=1)
    )
    for band_index, band_label in enumerate(
        ("0_100", "100_250", "250_500", "500_1000")
    ):
        candidate_predictions[f"existing_stores_{band_label}"] = (
            candidate_bands == band_index
        ).sum(axis=1)
    candidate_predictions["standalone_ranking"] = (
        candidate_predictions["standalone_net_network_sales_2027_mnt_jan_open"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    scenario_summaries: list[dict[str, Any]] = []
    selection_by_target: dict[int, pd.DataFrame] = {}
    existing_impact_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    detailed_pair_rows: list[pd.DataFrame] = []
    ranking_rows: list[pd.DataFrame] = []
    results_by_target: dict[int, dict[str, Any]] = {}
    for target in config.targets:
        candidate_result, selection, existing_summary, monthly, details = (
            _scenario_outputs(
                candidate_predictions,
                existing_for_sim,
                candidate_bands,
                pair_bands,
                existing_bands,
                candidate_existing_distances,
                baseline,
                active_days_by_existing[:, 0],
                active_days_open_month,
                forecast_months,
                maturations,
                beta_models,
                target,
                config.forecast_year,
                config.schedule,
                custom_weights,
                config.opening_day,
                config.cannibalization_scale,
                predictions,
            )
        )
        selection_by_target[target] = selection
        existing_impact_rows.append(existing_summary)
        monthly_rows.append(monthly)
        detailed_pair_rows.append(details["pair_impacts"])
        candidate_result["selected"] = candidate_result["selected"].astype(bool)
        candidate_result["standalone_ranking"] = candidate_predictions[
            "standalone_ranking"
        ].to_numpy()
        ranking_rows.append(candidate_result)
        summary = details["summary"]
        scenario_summaries.append(summary)
        results_by_target[target] = {
            "selection": selection,
            "candidate_results": candidate_result,
            "existing_summary": existing_summary,
            "monthly": monthly,
            "details": details,
        }

    ranking = candidate_predictions.copy()
    for target, result in results_by_target.items():
        candidate_result = result["candidate_results"].set_index("potential_store_id")
        mapping = candidate_result["portfolio_selection_rank"].to_dict()
        selected_map = candidate_result["selected"].to_dict()
        marginal_map = candidate_result[
            "greedy_marginal_network_sales_2027_mnt"
        ].to_dict()
        date_map = candidate_result["assumed_opening_date"].to_dict()
        ranking[f"portfolio_rank_{target}"] = ranking["potential_store_id"].map(mapping)
        ranking[f"selected_{target}"] = (
            ranking["potential_store_id"].map(selected_map).fillna(False).astype(bool)
        )
        ranking[f"greedy_marginal_value_{target}_mnt"] = ranking[
            "potential_store_id"
        ].map(marginal_map)
        ranking[f"assumed_opening_date_{target}"] = ranking[
            "potential_store_id"
        ].map(date_map)
        ranking[f"rank_within_area_type_{target}"] = ranking.groupby(
            "area_type", dropna=False
        )[f"greedy_marginal_value_{target}_mnt"].rank(
            method="first", ascending=False
        )
    existing_impacts = pd.concat(existing_impact_rows, ignore_index=True)
    monthly_network = pd.concat(monthly_rows, ignore_index=True)
    pair_impacts = pd.concat(detailed_pair_rows, ignore_index=True)
    selected_180 = selection_by_target.get(
        180, pd.DataFrame(columns=ranking.columns)
    )
    selected_400 = selection_by_target.get(
        400, pd.DataFrame(columns=ranking.columns)
    )
    candidate_existing_distances["distance_band"] = (
        candidate_existing_distances["distance_band"].astype("string")
    )
    regional_rows = (
        ranking.loc[ranking["selected_180"] | ranking["selected_400"]]
        .melt(
            id_vars=["potential_store_id", "area_type"],
            value_vars=["selected_180", "selected_400"],
            var_name="scenario",
            value_name="selected",
        )
        .loc[lambda frame: frame["selected"]]
        .groupby(["scenario", "area_type"], dropna=False)
        .agg(selected_candidates=("potential_store_id", "nunique"))
        .reset_index()
    )
    regional_scenario = regional_rows.merge(
        ranking.melt(
            id_vars=["potential_store_id", "area_type"],
            value_vars=[
                "predicted_APSD_sales",
                "predicted_APSD_customers",
                "standalone_net_network_sales_2027_mnt_jan_open",
            ],
            var_name="metric",
            value_name="value",
        )
        .merge(
            regional_rows[["scenario", "area_type"]],
            on="area_type",
            how="inner",
        )
        .groupby(["scenario", "area_type", "metric"])["value"]
        .mean()
        .unstack("metric")
        .reset_index(),
        on=["scenario", "area_type"],
        how="left",
    )

    quality_rows = [
        {"metric": key, "value": value} for key, value in quality.items()
    ]
    quality_rows.extend(
        [
            {
                "metric": "demand_training_rows",
                "value": len(training),
            },
            {
                "metric": "demand_time_holdout_rows",
                "value": int(
                    training["open_month"].ge(pd.Period("2025-01", freq="M")).sum()
                ),
            },
            {
                "metric": "existing_store_apsd_forecast_basis",
                "value": "Mean of last 12 available positive store-month APSD observations",
            },
            {
                "metric": "new_store_active_day_assumption",
                "value": "Median observed Active_Days by calendar month; opening month prorated by assumed opening day",
            },
            {
                "metric": "regional_candidate_mapping",
                "value": "Area Type UB remains UB; all Rural/province labels map to Rural for demand features; original labels retained.",
            },
        ]
    )
    data_quality = pd.DataFrame(quality_rows)
    assumptions = {
        "model_version": MODEL_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_year": config.forecast_year,
        "targets": list(config.targets),
        "schedule": config.schedule,
        "custom_monthly_weights": (
            list(map(float, custom_weights)) if custom_weights is not None else None
        ),
        "opening_day": config.opening_day,
        "cannibalization_scale": config.cannibalization_scale,
        "candidate_source": "Data/raw/New Store GPS.xlsx",
        "historical_source": "store_master_data.xlsx",
        "distance_source": "store_to_store_distance.xlsx",
        "saved_model_source": "outputs/cannibalization_regional/regional_*_coefficients.csv",
        "selection_objective": (
            "Greedy marginal increase in modeled 2027 network sales in MNT: "
            "new-store APSD x forecast active days plus changes to existing and "
            "already selected new-store revenue. Each opening is a single event; "
            "log effects are accumulated before exponentiation."
        ),
        "standalone_ranking_assumption": (
            "Standalone benchmark assumes a January 1 opening to provide a common "
            "12-month comparison; selected portfolio dates follow the configured schedule."
        ),
        "demand_model": (
            "Separate ridge regression on historical opening-month APSD, using "
            "pre-opening local store counts, nearest distance, nearby observed APSD, "
            "and a UB/Rural group. Validation holds out 2025-2026 opening cohorts."
        ),
        "new_store_region_mapping": (
            "Raw candidate Area Type values are preserved. UB maps to a coarse UB "
            "training category; Rural and province labels map to Rural because the "
            "candidate sheet does not contain the project's Distance_Zone classification."
        ),
        "cannibalization_note": (
            "Uses saved all-region monthly distance-by-event-time coefficients for "
            "sales and customers. Model status reports significant pre-trend joint "
            "tests; results are conditional scenario estimates, not causal guarantees."
        ),
        "event_time": (
            "calendar month minus candidate opening month; Model 4 event -1 is "
            "omitted and represented as zero. Forecast months use event times 0-11."
        ),
        "unknown_effects_beyond_event_window": (
            "Not extrapolated; no simulated 2027 event time exceeds month +11."
        ),
        "apsd_revenue_conversion": (
            "Monthly sales MNT = APSD_income (MNT per active sales day) x forecast "
            "Active_Days. Customer APSD times active days is customer-equivalents, "
            "not unique people. New-store open-month active days are prorated."
        ),
        "costs_and_margin": "Not supplied; results are sales value, not profit or ROI.",
        "data_quality": quality,
        "scenario_summaries": scenario_summaries,
    }

    result: dict[str, pd.DataFrame | dict[str, Any]] = {
        "potential_store_predictions": candidate_predictions,
        "potential_store_rankings": ranking.sort_values(
            "standalone_ranking", kind="stable"
        ),
        "selected_stores_180": selected_180,
        "selected_stores_400": selected_400,
        "existing_store_impact_2027": existing_impacts,
        "monthly_network_simulation_2027": monthly_network,
        "regional_scenario_comparison": regional_scenario,
        "candidate_pair_distances": pair_distances,
        "candidate_existing_distances": candidate_existing_distances,
        "candidate_existing_impacts": pair_impacts,
        "model_validation_report": validation,
        "data_quality_report": data_quality,
        "simulation_assumptions": assumptions,
        "scenario_summaries": pd.DataFrame(scenario_summaries),
        "maturation_profiles": pd.DataFrame(
            [
                {
                    "outcome": outcome,
                    "age_month": age,
                    "maturation_multiplier": multiplier,
                }
                for outcome, profile in maturations.items()
                for age, multiplier in profile.items()
            ]
        ),
    }
    if export:
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in (
            ("potential_store_predictions", "potential_store_predictions.csv"),
            ("potential_store_rankings", "potential_store_rankings.csv"),
            ("selected_stores_180", "selected_stores_180.csv"),
            ("selected_stores_400", "selected_stores_400.csv"),
            ("existing_store_impact_2027", "existing_store_impact_2027.csv"),
            ("monthly_network_simulation_2027", "monthly_network_simulation_2027.csv"),
            ("regional_scenario_comparison", "regional_scenario_comparison.csv"),
            ("candidate_pair_distances", "candidate_pair_distances.csv"),
            ("candidate_existing_distances", "candidate_existing_distances.csv"),
            ("candidate_existing_impacts", "candidate_existing_impacts.csv"),
            ("model_validation_report", "model_validation_report.csv"),
            ("data_quality_report", "data_quality_report.csv"),
            ("scenario_summaries", "scenario_summaries.csv"),
            ("maturation_profiles", "maturation_profiles.csv"),
        ):
            frame = result[key]
            if isinstance(frame, pd.DataFrame):
                frame.to_csv(output_dir / filename, index=False)
        (output_dir / "simulation_assumptions.json").write_text(
            json.dumps(assumptions, indent=2, ensure_ascii=True, default=_safe_value),
            encoding="utf-8",
        )
    return result


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results = run_expansion_simulation(base_dir)
    summaries = results["scenario_summaries"]
    print("Expansion simulation outputs:", base_dir / "outputs" / "expansion_simulation")
    if isinstance(summaries, pd.DataFrame):
        print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
