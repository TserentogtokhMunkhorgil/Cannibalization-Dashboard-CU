"""Evaluate frozen accepted portfolios using exact supplied log-APSD coefficients.

City Outskirts sales explicitly use the supplied All Regions fallback. Signed
raw estimates and negative-only pair assumptions remain separate. No selection
or regression is rerun by the production entry point.
"""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cannibalization_simulator import weighted_network_effect


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "planning"
INPUTS = ROOT / 'Data' / 'planning_inputs' / 'accepted_v1'
COEFFICIENT_VERSION = 'user_fixed_cumulative_v1'
GPS = ROOT / "Data" / "raw" / "New Store GPS.xlsx"
MASTER = ROOT / "store_master_data.xlsx"
PANEL = ROOT / 'outputs' / 'cannibalization_model4' / "cu_cannibalization_model4_panel.parquet"
REGIONAL_SALES = ROOT / "outputs" / "cannibalization_regional" / "regional_sales_coefficients.csv"
REGIONAL_CUSTOMERS = ROOT / "outputs" / "cannibalization_regional" / "regional_customer_coefficients.csv"
CUTOFF_M = 1_000.0
BANDS = (
    ("0_100", 0.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1_000.0),
)
SCENARIOS = (180, 400)
FIXED_INCOME = {
    'All Regions': (-0.1893, -0.0740, -0.0278, -0.0110),
    'Rural': (-0.4639, -0.2106, -0.1878, -0.1362),
    'Far / Outer Areas': (-0.3565, -0.1253, -0.0492, -0.0202),
    'City Center': (-0.0628, -0.0069, -0.0056, -0.0086),
}
FIXED_CUSTOMERS = {
    'All Regions': (-0.1493, -0.0632, -0.0201, -0.0082),
    'City Center': (-0.0666, -0.0220, -0.0006, -0.0125),
    'City Outskirts': (-0.1046, -0.0706, -0.0150, 0.0042),
    'Far / Outer Areas': (-0.2734, -0.1026, -0.0393, -0.0107),
    'Rural': (-0.3812, -0.1812, -0.1547, -0.1144),
}


def fixed_lookup(table: dict) -> dict:
    regions = set(FIXED_INCOME) | set(FIXED_CUSTOMERS)
    return {(region, band): table.get(region, table['All Regions'])[i]
            for region in regions for i, (band, _, _) in enumerate(BANDS)}
EARTH_RADIUS_M = 6_371_000.0


def _ids(values: pd.Series) -> pd.Series:
    values = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return values.mask(values.eq(""))


def _valid_coordinates(frame: pd.DataFrame) -> pd.Series:
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    return latitude.between(-90, 90) & longitude.between(-180, 180)


def _region(zone: Any, ub: Any) -> str:
    zone = "" if pd.isna(zone) else str(zone).strip()
    rural = "" if pd.isna(ub) else str(ub).strip().casefold()
    if rural == "rural" or zone == "Rural":
        return "Rural"
    return {
        "Core": "City Center",
        "Inner City": "City Outskirts",
        "Outer City": "Far / Outer Areas",
        "Far": "Far / Outer Areas",
    }.get(zone, "Unmapped")


def _bands(distance_m: np.ndarray) -> np.ndarray:
    codes = np.full(distance_m.shape, "", dtype=object)
    for position, (band, lower, upper) in enumerate(BANDS):
        lower_test = distance_m >= lower if position == 0 else distance_m > lower
        codes[lower_test & (distance_m <= upper)] = band
    return codes


def _distance_matrix(stores: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    store_latitude = np.radians(stores["latitude"].to_numpy(float))[:, None]
    store_longitude = np.radians(stores["longitude"].to_numpy(float))[:, None]
    candidate_latitude = np.radians(candidates["latitude"].to_numpy(float))[None, :]
    candidate_longitude = np.radians(candidates["longitude"].to_numpy(float))[None, :]
    a = (
        np.sin((candidate_latitude - store_latitude) / 2) ** 2
        + np.cos(store_latitude)
        * np.cos(candidate_latitude)
        * np.sin((candidate_longitude - store_longitude) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare_consistent_baseline(panel: pd.DataFrame, master: pd.DataFrame, as_of=None) -> pd.DataFrame:
    required = {"existing_store_id", "SalesMonth", "APSD_income", "LATITUDE", "LONGITUDE", "loc_area_new"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Model panel missing: {sorted(missing)}")
    panel = panel.copy()
    panel["existing_store_id"] = _ids(panel["existing_store_id"])
    panel["SalesMonth"] = pd.to_datetime(panel["SalesMonth"], errors="coerce")
    panel["APSD_income"] = pd.to_numeric(panel["APSD_income"], errors="coerce")
    if panel.duplicated(['existing_store_id', 'SalesMonth']).any():
        raise ValueError('Duplicate store-month rows in baseline panel.')
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now(tz='Asia/Irkutsk').date())
    panel = panel.loc[panel['SalesMonth'].dt.to_period('M') < as_of.to_period('M')].copy()
    panel['APSD_income'] = panel['APSD_income'].where(np.isfinite(panel['APSD_income']))
    coverage = panel.groupby("SalesMonth")["APSD_income"].apply(lambda x: x.gt(0).sum())
    if coverage.empty or coverage.max() == 0:
        raise ValueError('No completed month has a valid positive income baseline.')
    month = pd.Timestamp(coverage.loc[coverage.ge(np.ceil(coverage.max() * 0.95))].index.max())
    stores = panel.loc[panel["SalesMonth"].eq(month)].drop_duplicates("existing_store_id", keep="last").copy()
    stores["latitude"] = pd.to_numeric(stores["LATITUDE"], errors="coerce")
    stores["longitude"] = pd.to_numeric(stores["LONGITUDE"], errors="coerce")
    stores["baseline_APSD_income"] = stores["APSD_income"]

    metadata = master.copy()
    metadata["store_id"] = _ids(metadata["store_id"])
    metadata["SalesMonth"] = pd.to_datetime(metadata["SalesMonth"], errors="coerce")
    if metadata.duplicated(['store_id', 'SalesMonth']).any():
        raise ValueError('Duplicate store-month rows in store master.')
    wanted = [c for c in ["store_id", "SalesMonth", "store_name", "APSD_customers", "Distance_Zone", "UB"] if c in metadata]
    metadata = metadata.loc[metadata["SalesMonth"].eq(month), wanted]
    metadata = metadata.drop_duplicates(["store_id", "SalesMonth"], keep="last")
    stores = stores.merge(
        metadata,
        left_on=["existing_store_id", "SalesMonth"],
        right_on=["store_id", "SalesMonth"],
        how="left",
        validate="one_to_one",
    )
    stores["store_name"] = stores.get("store_name", pd.Series(pd.NA, index=stores.index)).fillna(
        "CU " + stores["existing_store_id"].astype(str)
    )
    stores["baseline_APSD_customers"] = pd.to_numeric(
        stores.get("APSD_customers", pd.Series(np.nan, index=stores.index)), errors="coerce"
    )
    stores["region"] = [
        _region(zone, ub)
        for zone, ub in zip(
            stores.get("Distance_Zone", pd.Series(pd.NA, index=stores.index)),
            stores.get("UB", pd.Series(pd.NA, index=stores.index)),
        )
    ]
    stores["baseline_valid"] = stores["baseline_APSD_income"].gt(0)
    stores["baseline_month_or_period"] = month.strftime("%Y-%m")
    stores["baseline_status"] = np.select(
        [
            ~stores["baseline_valid"],
            ~_valid_coordinates(stores),
            stores["region"].eq("Unmapped"),
        ],
        [
            "invalid_or_nonpositive_income_APSD",
            "invalid_store_coordinates",
            "unmapped_region",
        ],
        default="valid",
    )
    return stores[
        [
            "existing_store_id", "store_name", "region", "loc_area_new",
            "baseline_month_or_period", "baseline_APSD_income",
            "baseline_APSD_customers", "latitude", "longitude",
            "baseline_valid", "baseline_status",
        ]
    ].reset_index(drop=True)


def build_regional_pairs(
    stores: pd.DataFrame,
    candidates: pd.DataFrame,
    income_coefficients: dict[tuple[str, str], float],
    customer_coefficients: dict[tuple[str, str], float],
    cutoff_m: float = CUTOFF_M,
) -> pd.DataFrame:
    if not 0 < cutoff_m <= CUTOFF_M:
        raise ValueError('Planning cutoff must be greater than zero and at most 1,000 m.')
    if stores['existing_store_id'].duplicated().any() or candidates['candidate_id'].duplicated().any():
        raise ValueError('Store and candidate IDs must be unique.')
    if not _valid_coordinates(candidates).all():
        raise ValueError('Candidate coordinates must be valid.')
    valid_stores = stores.loc[_valid_coordinates(stores)].copy()
    distances = _distance_matrix(valid_stores, candidates)
    store_position, candidate_position = np.where(distances <= cutoff_m)
    pairs = pd.DataFrame(
        {
            "existing_store_id": valid_stores.iloc[store_position]["existing_store_id"].to_numpy(),
            "region": valid_stores.iloc[store_position]["region"].to_numpy(),
            "candidate_id": candidates.iloc[candidate_position]["candidate_id"].to_numpy(),
            "candidate_rank": candidates.iloc[candidate_position]["candidate_rank"].to_numpy(),
            "candidate_name": candidates.iloc[candidate_position]["candidate_name"].to_numpy(),
            "candidate_longitude": candidates.iloc[candidate_position]["longitude"].to_numpy(),
            "candidate_latitude": candidates.iloc[candidate_position]["latitude"].to_numpy(),
            "distance_m": distances[store_position, candidate_position],
        }
    )
    pairs["distance_band"] = _bands(pairs["distance_m"].to_numpy())
    pairs['income_coefficient_source'] = np.where(
        pairs['region'].isin(FIXED_INCOME), pairs['region'], np.where(pairs['region'].eq('City Outskirts'), 'All Regions (explicit fallback)', 'Unavailable region'))
    pairs['customer_coefficient_source'] = np.where(
        pairs['region'].isin(FIXED_CUSTOMERS), pairs['region'], 'Unavailable region')
    pairs["regional_income_log_coefficient"] = [
        income_coefficients.get((str(region), str(band)), np.nan) if band else np.nan
        for region, band in zip(pairs["region"], pairs["distance_band"])
    ]
    pairs["income_model_status"] = np.where(
        pairs["regional_income_log_coefficient"].notna(),
        "supported_regional_estimate",
        "missing_regional_income_estimate",
    )
    pairs["direct_APSD_income_effect_pct"] = np.expm1(
        pairs["regional_income_log_coefficient"]
    ) * 100
    pairs["regional_customer_log_coefficient"] = [
        customer_coefficients.get((str(region), str(band)), np.nan) if band else np.nan
        for region, band in zip(pairs["region"], pairs["distance_band"])
    ]
    pairs["customer_model_status"] = np.where(
        pairs["regional_customer_log_coefficient"].notna(),
        "supported_regional_estimate",
        "missing_regional_customer_estimate",
    )
    pairs["direct_APSD_customers_effect_pct"] = np.expm1(
        pairs["regional_customer_log_coefficient"]
    ) * 100
    if pairs.duplicated(["existing_store_id", "candidate_id"]).any():
        raise AssertionError("Duplicate candidate/store exposure pair.")
    return pairs


def select_greedy_portfolio(
    stores: pd.DataFrame, candidates: pd.DataFrame, pairs: pd.DataFrame, maximum: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy re-evaluated marginal-loss selector, not a static score ranking."""
    network = stores.loc[
        stores["baseline_valid"] & stores["region"].ne("Unmapped") & _valid_coordinates(stores)
    ].copy()
    store_ids = network["existing_store_id"].astype(str).tolist()
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    store_index = {store_id: position for position, store_id in enumerate(store_ids)}
    candidate_index = {candidate_id: position for position, candidate_id in enumerate(candidate_ids)}
    contribution = np.zeros((len(store_ids), len(candidate_ids)))
    unsupported = set(
        pairs.loc[
            pairs["income_model_status"].ne("supported_regional_estimate"), "candidate_id"
        ].astype(str)
    )
    for row in pairs.loc[pairs["income_model_status"].eq("supported_regional_estimate")].itertuples():
        store_position = store_index.get(str(row.existing_store_id))
        if store_position is not None:
            contribution[store_position, candidate_index[str(row.candidate_id)]] = row.regional_income_log_coefficient
    eligibility = candidates[["candidate_id", "candidate_rank", "candidate_name"]].copy()
    eligibility["selection_eligible"] = ~eligibility["candidate_id"].astype(str).isin(unsupported)
    eligibility["selection_ineligibility_reason"] = np.where(
        eligibility["selection_eligible"],
        "",
        "missing_regional_income_estimate_for_qualifying_pair",
    )
    remaining = np.flatnonzero(eligibility["selection_eligible"].to_numpy()).tolist()
    current_log = np.zeros(len(network))
    weights = network["baseline_APSD_income"].to_numpy(float)
    selected = []
    for order in range(1, min(maximum, len(remaining)) + 1):
        positions = np.array(remaining, dtype=int)
        next_log = current_log[:, None] + contribution[:, positions]
        objective = 100 * (
            weights[:, None] * np.maximum(-np.expm1(next_log), 0)
        ).sum(axis=0) / weights.sum()
        prior = 100 * (weights * np.maximum(-np.expm1(current_log), 0)).sum() / weights.sum()
        source_rank = candidates.iloc[positions]["candidate_rank"].to_numpy()
        best_offset = int(np.lexsort((source_rank, objective))[0])
        best_position = int(positions[best_offset])
        current_log += contribution[:, best_position]
        candidate = candidates.iloc[best_position]
        selected.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_rank": candidate["candidate_rank"],
                "candidate_name": candidate["candidate_name"],
                "selection_order": order,
                "marginal_weighted_network_loss_pct": float(objective[best_offset] - prior),
                "portfolio_weighted_network_loss_pct": float(objective[best_offset]),
                "selection_method": "greedy_recalculated_marginal_regional_log_effect",
            }
        )
        remaining.remove(best_position)
    return pd.DataFrame(selected), eligibility


def apply_pair_mode(pairs: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Preserve the source and cap positive pair betas only in planning mode."""
    if mode not in ('raw', 'negative_only'):
        raise ValueError('Unknown calculation mode.')
    output = pairs.copy()
    for coefficient_outcome, effect_outcome in [('income', 'income'), ('customer', 'customers')]:
        column = f'regional_{coefficient_outcome}_log_coefficient'
        if mode == 'negative_only':
            output[column] = output[column].clip(upper=0)
        output[f'direct_APSD_{effect_outcome}_effect_pct'] = np.expm1(output[column]) * 100
    return output


def aggregate_portfolio(
    stores: pd.DataFrame,
    pairs: pd.DataFrame,
    selected: pd.DataFrame,
    scenario_size: int,
    cutoff_m: float = CUTOFF_M,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 < cutoff_m <= CUTOFF_M:
        raise ValueError('Unsupported direct cutoff.')
    if selected['candidate_id'].duplicated().any() or selected['selection_order'].duplicated().any():
        raise ValueError('Portfolio IDs and selection order must be unique.')
    if pairs.duplicated(['existing_store_id', 'candidate_id']).any():
        raise ValueError('Duplicate candidate/store pair.')
    selected_pairs = pairs.loc[pairs["candidate_id"].isin(selected["candidate_id"]) & pairs['distance_m'].between(0, cutoff_m)].copy()
    selected_pairs["selection_order"] = selected_pairs["candidate_id"].map(
        selected.set_index("candidate_id")["selection_order"]
    )
    output = stores.copy()
    output['baseline_valid'] = output['baseline_valid'] & np.isfinite(output['baseline_APSD_income']) & output['baseline_APSD_income'].gt(0)
    output["scenario_size"] = scenario_size
    output["direct_cannibalization_cutoff_m"] = cutoff_m
    output["number_selected_candidates"] = len(selected)
    count = selected_pairs.groupby("existing_store_id")["candidate_id"].nunique()
    output["number_of_contributing_candidate_stores"] = (
        output["existing_store_id"].map(count).fillna(0).astype(int)
    )
    # These are cumulative radius counts, not mutually exclusive band counts.
    # The distance band remains available separately in the pair-level output.
    for threshold in (100, 250, 500, 1000):
        count = selected_pairs.loc[selected_pairs["distance_m"].le(threshold)].groupby(
            "existing_store_id"
        )["candidate_id"].nunique()
        output[f"number_within_{threshold}m"] = (
            output["existing_store_id"].map(count).fillna(0).astype(int)
        )
    coverage = selected_pairs.groupby("existing_store_id").agg(
        selected_pair_count=("candidate_id", "nunique"),
        income_terms=("income_model_status", lambda x: int(x.eq("supported_regional_estimate").sum())),
        customer_terms=("customer_model_status", lambda x: int(x.eq("supported_regional_estimate").sum())),
    )
    output = output.merge(coverage, left_on="existing_store_id", right_index=True, how="left")
    for column in ["selected_pair_count", "income_terms", "customer_terms"]:
        output[column] = output[column].fillna(0).astype(int)
    income_log = selected_pairs.loc[
        selected_pairs["income_model_status"].eq("supported_regional_estimate")
    ].groupby("existing_store_id")["regional_income_log_coefficient"].sum()
    customer_log = selected_pairs.loc[
        selected_pairs["customer_model_status"].eq("supported_regional_estimate")
    ].groupby("existing_store_id")["regional_customer_log_coefficient"].sum()
    output["cumulative_income_log_effect"] = output["existing_store_id"].map(income_log)
    output["cumulative_customer_log_effect"] = output["existing_store_id"].map(customer_log)
    output["model_estimate_status"] = np.select(
        [
            ~output["baseline_valid"],
            ~_valid_coordinates(output),
            output["selected_pair_count"].eq(0),
            output["income_terms"].eq(output["selected_pair_count"]),
        ],
        [
            "insufficient_or_invalid_baseline",
            "missing_or_invalid_store_coordinates",
            "valid_zero_exposure",
            "estimated_with_regional_income_model",
        ],
        default="missing_regional_income_estimate",
    )
    income_ok = output["model_estimate_status"].isin(
        ["valid_zero_exposure", "estimated_with_regional_income_model"]
    )
    output.loc[output["model_estimate_status"].eq("valid_zero_exposure"), "cumulative_income_log_effect"] = 0.0
    output["cumulative_predicted_APSD_income_effect_pct"] = np.where(
        income_ok, np.expm1(output["cumulative_income_log_effect"].fillna(0)) * 100, np.nan
    )
    output["cumulative_predicted_APSD_income_change"] = (
        output["baseline_APSD_income"] * output["cumulative_predicted_APSD_income_effect_pct"] / 100
    )
    output["predicted_scenario_APSD_income"] = (
        output["baseline_APSD_income"] + output["cumulative_predicted_APSD_income_change"]
    )
    output["customer_model_estimate_status"] = np.select(
        [
            ~(output["baseline_APSD_customers"].gt(0) & np.isfinite(output['baseline_APSD_customers'])),
            ~_valid_coordinates(output),
            output["selected_pair_count"].eq(0),
            output["customer_terms"].eq(output["selected_pair_count"]),
        ],
        [
            "customer_baseline_unavailable",
            "missing_or_invalid_store_coordinates",
            "valid_zero_exposure",
            "estimated_with_regional_customer_model",
        ],
        default="missing_regional_customer_estimate",
    )
    customer_ok = output["customer_model_estimate_status"].isin(
        ["valid_zero_exposure", "estimated_with_regional_customer_model"]
    )
    output.loc[output["customer_model_estimate_status"].eq("valid_zero_exposure"), "cumulative_customer_log_effect"] = 0.0
    output["cumulative_predicted_APSD_customers_effect_pct"] = np.where(
        customer_ok, np.expm1(output["cumulative_customer_log_effect"].fillna(0)) * 100, np.nan
    )
    output["cumulative_predicted_APSD_customers_change"] = (
        output["baseline_APSD_customers"] * output["cumulative_predicted_APSD_customers_effect_pct"] / 100
    )
    output["predicted_scenario_APSD_customers"] = (
        output["baseline_APSD_customers"] + output["cumulative_predicted_APSD_customers_change"]
    )
    contributors = selected_pairs.loc[
        selected_pairs["income_model_status"].eq("supported_regional_estimate")
    ].sort_values(["regional_income_log_coefficient", "distance_m", "selection_order"])
    labels: dict[str, str] = {}
    for store_id, group in contributors.groupby("existing_store_id"):
        labels[str(store_id)] = "; ".join(
            f"{row.candidate_name} ({row.distance_m:.0f}m, {row.direct_APSD_income_effect_pct:+.2f}%)"
            for row in group.head(5).itertuples()
        )
    output["main_contributing_candidate_locations"] = output["existing_store_id"].astype(str).map(labels).fillna("")
    output["rank_by_percentage_decline"] = output[
        "cumulative_predicted_APSD_income_effect_pct"
    ].rank(method="dense", ascending=True, na_option="bottom").astype("Int64")
    output["rank_by_absolute_APSD_income_decline"] = output[
        "cumulative_predicted_APSD_income_change"
    ].rank(method="dense", ascending=True, na_option="bottom").astype("Int64")
    network = output["baseline_APSD_income"].gt(0) & income_ok
    denominator = output.loc[network, "baseline_APSD_income"].sum()
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('No valid baseline population for network weighting.')
    weighted_effect = weighted_network_effect(
        output.loc[network, 'baseline_APSD_income'],
        output.loc[network, 'cumulative_predicted_APSD_income_effect_pct'])
    scenario_change = 100 * (
        output.loc[network, "predicted_scenario_APSD_income"].sum() - denominator
    ) / denominator
    summary = {
        "scenario_stores": scenario_size,
        "selected_unique_candidates": len(selected),
        "baseline_month_or_period": output["baseline_month_or_period"].iloc[0],
        "direct_cannibalization_cutoff_m": cutoff_m,
        "network_population_stores": int(network.sum()),
        "existing_stores_with_nearby_opening": int(output["selected_pair_count"].gt(0).sum()),
        "pct_existing_stores_with_nearby_opening": float(output["selected_pair_count"].gt(0).mean() * 100),
        "existing_stores_with_predicted_decline": int(output["cumulative_predicted_APSD_income_effect_pct"].lt(0).sum()),
        "pct_existing_stores_with_predicted_decline": float(output["cumulative_predicted_APSD_income_effect_pct"].lt(0).mean() * 100),
        "APSD_weighted_network_effect_pct": float(weighted_effect),
        "network_APSD_weighted_change_pct": float(scenario_change),
        "network_effect_validation_difference_pct_points": float(weighted_effect - scenario_change),
        "mean_store_level_effect_pct": float(output["cumulative_predicted_APSD_income_effect_pct"].mean()),
        "median_store_level_effect_pct": float(output["cumulative_predicted_APSD_income_effect_pct"].median()),
        "total_absolute_APSD_income_change": float(output.loc[network, "cumulative_predicted_APSD_income_change"].sum()),
        "median_absolute_APSD_income_change": float(output["cumulative_predicted_APSD_income_change"].median()),
        "stores_decline_gt_5_pct": int(output["cumulative_predicted_APSD_income_effect_pct"].lt(-5).sum()),
        "stores_decline_gt_10_pct": int(output["cumulative_predicted_APSD_income_effect_pct"].lt(-10).sum()),
        "stores_decline_gt_20_pct": int(output["cumulative_predicted_APSD_income_effect_pct"].lt(-20).sum()),
        "stores_decline_gt_30_pct": int(output["cumulative_predicted_APSD_income_effect_pct"].lt(-30).sum()),
        "stores_with_positive_modeled_effect": int(output["cumulative_predicted_APSD_income_effect_pct"].gt(0).sum()),
    }
    selected_pairs["scenario_size"] = scenario_size
    selected_pairs["direct_cannibalization_cutoff_m"] = cutoff_m
    selected_pairs["cumulative_calculation"] = "sum regional log contributions then exp(sum)-1"
    return output, selected_pairs, summary


def regional_summary(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for region, group in results.groupby("region", dropna=False):
        valid = group["model_estimate_status"].isin(
            ["valid_zero_exposure", "estimated_with_regional_income_model"]
        ) & group["baseline_APSD_income"].gt(0)
        denominator = group.loc[valid, "baseline_APSD_income"].sum()
        weighted = 100 * (
            group.loc[valid, "baseline_APSD_income"]
            * group.loc[valid, "cumulative_predicted_APSD_income_effect_pct"]
            / 100
        ).sum() / denominator if denominator else np.nan
        records.append(
            {
                "scenario_stores": group["scenario_size"].iloc[0],
                "region": region,
                "existing_stores": len(group),
                "network_population_stores": int(valid.sum()),
                "existing_stores_affected": int(group["selected_pair_count"].gt(0).sum()),
                "pct_existing_stores_affected": float(group["selected_pair_count"].gt(0).mean() * 100),
                "APSD_weighted_network_effect_pct": float(weighted),
                "mean_store_level_effect_pct": float(group["cumulative_predicted_APSD_income_effect_pct"].mean()),
                "median_store_level_effect_pct": float(group["cumulative_predicted_APSD_income_effect_pct"].median()),
                "stores_decline_gt_10_pct": int(group["cumulative_predicted_APSD_income_effect_pct"].lt(-10).sum()),
                "stores_decline_gt_20_pct": int(group["cumulative_predicted_APSD_income_effect_pct"].lt(-20).sum()),
                "stores_decline_gt_30_pct": int(group["cumulative_predicted_APSD_income_effect_pct"].lt(-30).sum()),
            }
        )
    return pd.DataFrame(records)


def _sensitivity(stores: pd.DataFrame, candidates: pd.DataFrame, income: dict, customers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, portfolios = [], []
    for cutoff in (500.0, 1_000.0, 1_500.0):
        if cutoff > CUTOFF_M:
            summaries.append(
                {
                    "cutoff_m": cutoff,
                    "scenario_stores": 180,
                    "status": "not_estimable",
                    "reason": "No saved regional coefficient supports the 1,000-1,500 m band.",
                }
            )
            continue
        pairs = build_regional_pairs(stores, candidates, income, customers, cutoff)
        selected, _ = select_greedy_portfolio(stores, candidates, pairs, 180)
        _, _, summary = aggregate_portfolio(stores, pairs, selected, 180, cutoff)
        summaries.append({"cutoff_m": cutoff, "status": "estimated", "reason": "", **summary})
        selected["cutoff_m"] = cutoff
        portfolios.append(selected)
    return pd.DataFrame(summaries), pd.concat(portfolios, ignore_index=True)


def _compatibility_wide(results: list[pd.DataFrame]) -> pd.DataFrame:
    base = results[0][[
        "existing_store_id", "store_name", "region", "loc_area_new",
        "baseline_month_or_period", "baseline_APSD_income", "baseline_APSD_customers",
        "latitude", "longitude",
    ]].copy()
    base = base.rename(columns={
        "baseline_month_or_period": "APSD_baseline_month",
        "baseline_APSD_income": "APSD_income_baseline",
        "baseline_APSD_customers": "APSD_customers_baseline",
    })
    fields = {
        "nearby_new_stores": "number_of_contributing_candidate_stores",
        "APSD_income_change_pct": "cumulative_predicted_APSD_income_effect_pct",
        "APSD_income_change": "cumulative_predicted_APSD_income_change",
        "APSD_income_expected": "predicted_scenario_APSD_income",
        "APSD_customers_change_pct": "cumulative_predicted_APSD_customers_effect_pct",
        "APSD_customers_change": "cumulative_predicted_APSD_customers_change",
        "APSD_customers_expected": "predicted_scenario_APSD_customers",
        "main_nearby_openings": "main_contributing_candidate_locations",
    }
    for frame in results:
        size = int(frame["scenario_size"].iloc[0])
        right = frame[["existing_store_id", *fields.values()]].rename(
            columns={source: f"{target}_{size}" for target, source in fields.items()}
        )
        base = base.merge(right, on="existing_store_id", how="left", validate="one_to_one")
        base[f"cannibalization_pct_{size}"] = np.maximum(
            -base[f"APSD_income_change_pct_{size}"].fillna(0), 0
        )
    return base


def run_regional_analysis(output_dir=None, input_dir=None, as_of=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    destination = Path(output_dir) if output_dir is not None else OUT
    input_dir = Path(input_dir) if input_dir is not None else INPUTS
    for path in [MASTER, PANEL, input_dir / 'candidates.csv', input_dir / 'selected_180.csv']:
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis input not found: {path}")
    # Preserve saved candidate identities and coordinates rather than rebuilding
    # IDs from a potentially reordered GPS workbook.
    candidates = pd.read_csv(input_dir / 'candidates.csv')
    candidates = candidates.drop(columns=[c for c in (
        'selection_order', 'marginal_weighted_network_loss_pct',
        'portfolio_weighted_network_loss_pct', 'selection_eligible',
        'selection_ineligibility_reason', 'in_scenario_180', 'in_scenario_400'
    ) if c in candidates])
    stores = prepare_consistent_baseline(
        pd.read_parquet(PANEL), pd.read_excel(MASTER, sheet_name="Sheet1"), as_of=as_of
    )
    income = fixed_lookup(FIXED_INCOME)
    customers = fixed_lookup(FIXED_CUSTOMERS)
    pairs = build_regional_pairs(stores, candidates, income, customers)
    # Freeze the actual saved portfolio. Recalculation never reselects sites.
    selected_180 = pd.read_csv(input_dir / 'selected_180.csv')
    selected_400 = pd.read_csv(input_dir / 'selected_400.csv') if (input_dir / 'selected_400.csv').is_file() else selected_180.copy()
    sizes = (180, 400) if (input_dir / 'selected_400.csv').is_file() else (180,)
    if candidates['candidate_id'].duplicated().any() or candidates.duplicated(['latitude', 'longitude']).any():
        raise ValueError('Candidate IDs and coordinate pairs must be unique.')
    for size in sizes:
        selected = selected_180 if size == 180 else selected_400
        if len(selected) != size or selected['selection_order'].duplicated().any():
            raise ValueError(f'Expected {size} distinct selection orders.')
        if selected['candidate_id'].duplicated().any():
            raise ValueError('Saved portfolio contains duplicate candidate IDs.')
        if not selected['candidate_id'].isin(candidates['candidate_id']).all():
            raise ValueError('Saved portfolio references unknown candidate IDs.')
    eligibility = candidates[['candidate_id']].assign(selection_eligible=True, selection_ineligibility_reason='')

    scenario_results, contribution_results, summaries, regional_results, portfolios = [], [], [], [], {}
    for size in sizes:
        portfolio = (selected_180 if size == 180 else selected_400).copy()
        portfolios[size] = portfolio
        for mode in ('negative_only', 'raw'):
            calculation_pairs = apply_pair_mode(pairs, mode)
            result, contributions, summary = aggregate_portfolio(stores, calculation_pairs, portfolio, size)
            result['calculation_mode'] = mode
            contributions['calculation_mode'] = mode
            summary['calculation_mode'] = mode
            summary['coefficient_version'] = COEFFICIENT_VERSION
            result['coefficient_version'] = COEFFICIENT_VERSION
            contributions['coefficient_version'] = COEFFICIENT_VERSION
            region_result = regional_summary(result)
            region_result['calculation_mode'] = mode
            scenario_results.append(result)
            contribution_results.append(contributions)
            summaries.append(summary)
            regional_results.append(region_result)
    store_results = pd.concat(scenario_results, ignore_index=True)
    contributions = pd.concat(contribution_results, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary["candidate_shortfall"] = (
        summary["scenario_stores"] - summary["selected_unique_candidates"]
    ).clip(lower=0)
    regional = pd.concat(regional_results, ignore_index=True)

    candidates = candidates.merge(
        selected_400[[
            "candidate_id", "selection_order", "marginal_weighted_network_loss_pct",
            "portfolio_weighted_network_loss_pct",
        ]],
        on="candidate_id", how="left", validate="one_to_one",
    ).merge(
        eligibility[["candidate_id", "selection_eligible", "selection_ineligibility_reason"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    for size in sizes:
        candidates[f"in_scenario_{size}"] = candidates['candidate_id'].isin(portfolios[size]['candidate_id'])
    validation = pd.DataFrame(
        [
            {
                "check": "all_primary_pairs_at_or_below_1000m",
                "passed": bool(pairs["distance_m"].le(CUTOFF_M).all()),
                "detail": "The cutoff is applied before aggregation.",
            },
            {
                "check": "no_duplicate_primary_pairs",
                "passed": not pairs.duplicated(["existing_store_id", "candidate_id"]).any(),
                "detail": "One candidate/store pair can contribute once.",
            },
            {
                "check": "portfolio_180_is_unique",
                "passed": not portfolios[180]["candidate_id"].duplicated().any(),
                "detail": "The portfolio contains 180 unique candidate IDs.",
            },
            {
                "check": "portfolio_400_is_unique",
                "passed": 400 not in portfolios or (len(portfolios[400]) == 400 and not portfolios[400]["candidate_id"].duplicated().any()),
                "detail": "Optional 400 portfolio is unique when supplied.",
            },
            {
                "check": "manual_log_aggregation",
                "passed": bool(np.isclose(np.expm1(np.log(.9) + np.log(.8)), -.28)),
                "detail": "log(.9) plus log(.8) equals a -28% cumulative effect.",
            },
        ]
    )
    for row in summary.itertuples():
        validation.loc[len(validation)] = {
            "check": f"weighted_network_identity_{int(row.scenario_stores)}_{row.calculation_mode}",
            "passed": abs(row.network_effect_validation_difference_pct_points) < 1e-9,
            "detail": "Weighted effect agrees with baseline-weighted scenario APSD change.",
        }
    quality = pd.DataFrame(
        [
            {"metric": "baseline_stores", "value": len(stores)},
            {"metric": "valid_income_baselines", "value": int(stores["baseline_valid"].sum())},
            {"metric": "unmapped_regions", "value": int(stores["region"].eq("Unmapped").sum())},
            {"metric": "pairs_missing_regional_income_estimate", "value": int(pairs["income_model_status"].ne("supported_regional_estimate").sum())},
            {"metric": "candidates_excluded_for_missing_income_estimate", "value": int((~eligibility["selection_eligible"]).sum())},
            {"metric": "direct_cannibalization_cutoff_m", "value": CUTOFF_M},
        ]
    )
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "cumulative_cannibalization_pair_effects.csv": pairs,
        "cumulative_cannibalization_contributions.csv": contributions,
        "cumulative_cannibalization_scenario_store_results.csv": store_results,
        "cumulative_cannibalization_store_results.csv": _compatibility_wide([r for r in scenario_results if r['calculation_mode'].iloc[0] == 'negative_only']),
        "cumulative_cannibalization_scenario_summary.csv": summary,
        "cumulative_cannibalization_network_indicators.csv": summary[[
            "scenario_stores", "calculation_mode", "coefficient_version", "baseline_month_or_period", "direct_cannibalization_cutoff_m",
            "network_population_stores", "APSD_weighted_network_effect_pct",
            "network_APSD_weighted_change_pct", "network_effect_validation_difference_pct_points",
        ]],
        "cumulative_cannibalization_regional_summary.csv": regional,
        "cumulative_cannibalization_candidates.csv": candidates,
        "cumulative_cannibalization_selected_180.csv": portfolios[180],
        "cumulative_cannibalization_validation_report.csv": validation,
        "cumulative_cannibalization_data_quality_report.csv": quality,
    }
    fixed_coefficients = pd.DataFrame([
        {'coefficient_version': COEFFICIENT_VERSION, 'outcome': outcome, 'region': region,
         'distance_band': band, 'coefficient': beta, 'percent_effect': 100 * np.expm1(beta),
         'source': 'Exact user-supplied cumulative regression coefficients',
         'uncertainty_status': 'Not supplied for fixed rounded coefficients'}
        for outcome, table in [('income', FIXED_INCOME), ('customers', FIXED_CUSTOMERS)]
        for region, values in table.items() for (band, _, _), beta in zip(BANDS, values)
    ])
    files['fixed_planning_coefficients.csv'] = fixed_coefficients
    previous_path = ROOT / 'outputs' / 'cannibalization_model4' / 'cumulative_cannibalization_scenario_summary.csv'
    if previous_path.is_file():
        previous = pd.read_csv(previous_path)
        comparison = summary.merge(previous, on=['scenario_stores', 'calculation_mode'], suffixes=('_new', '_previous'), validate='one_to_one')
        comparison['weighted_effect_difference_pct_points'] = comparison['APSD_weighted_network_effect_pct_new'] - comparison['APSD_weighted_network_effect_pct_previous']
        comparison['difference_reason'] = 'Latest completed month replaces in-progress month; accepted sites and exact coefficients unchanged.'
        files['previous_result_comparison.csv'] = comparison
    if 400 in portfolios:
        files['cumulative_cannibalization_selected_400.csv'] = portfolios[400]
    # Historical marginal scores belong to the frozen selection, not this evaluation.
    historical_columns = [c for c in candidates if c.startswith(('marginal_', 'portfolio_weighted_'))]
    candidates.drop(columns=historical_columns, inplace=True)
    for name, value in files.items():
        if name.startswith('cumulative_cannibalization_selected_'):
            value = value.rename(columns={c: 'historical_' + c for c in value if c.startswith(('marginal_', 'portfolio_weighted_', 'selection_method'))})
            value['evaluation_label'] = 'Previously selected portfolio evaluated with fixed coefficients'
        value.to_csv(destination / name, index=False)
    contributions.to_parquet(destination / 'contributions.parquet', index=False)
    with (destination / "cumulative_cannibalization_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "cutoff_disclosure": "Direct cannibalization effects are assumed to be zero beyond 1,000 meters. This is a modeling assumption, not an empirically established universal threshold.",
                "baseline": "Latest completed calendar month with at least 95% of peak valid positive income coverage; current month excluded.",
                "as_of": str(as_of if as_of is not None else pd.Timestamp.now(tz='Asia/Irkutsk').date()),
                "coefficient_version": COEFFICIENT_VERSION,
                "portfolio_label": "Previously selected portfolio evaluated with fixed coefficients",
                "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_dir.glob('*.csv')},
                "aggregation": "Exact user-supplied cumulative exposure-count log coefficients; exp(sum)-1. Bands are [0,100], (100,250], (250,500], (500,1000].",
                "selection": "Existing saved portfolios and candidate coordinates are reused; no reselection or regression fitting.",
                "regional_fallback": "City Outskirts income uses All Regions explicitly. Customer estimates are region-specific. Other unavailable regions use All Regions explicitly.",
                "calculation_modes": "Raw preserves all supplied coefficients; negative_only caps each positive pair contribution at zero before aggregation.",
                "income_coefficients": FIXED_INCOME,
                "customer_coefficients": FIXED_CUSTOMERS,
            },
            file,
            indent=2,
        )
    return store_results, candidates, contributions, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', help='Reproducible analysis date, YYYY-MM-DD')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    run_regional_analysis(output_dir=args.output_dir, as_of=args.as_of)
