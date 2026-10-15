"""Scenario registry and shared, schedule-aware expansion evaluation.

This module deliberately does not run an optimizer.  It loads a completed
portfolio (or the explicit frozen benchmark), rebuilds the already-saved
forecast inputs, and uses :class:`expansion_optimization.Evaluator` for every
network calculation shown in the business dashboard.  Keeping that logic here
prevents dashboard sections from accidentally selecting different portfolios
or applying all openings at the start of the year.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from expansion_optimization import Costs, Evaluator, OptimizationConfig, result_tables
from expansion_optimization_inputs import ROOT, prepare_inputs, saved_baseline
from regional_cumulative_impact import BANDS, FIXED_CUSTOMERS, _bands


SCENARIO_ROOT = ROOT / "outputs" / "expansion_optimization_v2"
BENCHMARK_FILE = ROOT / "Data" / "planning_inputs" / "accepted_v1" / "selected_180.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _json_config(values: dict[str, Any]) -> OptimizationConfig:
    """Restore the immutable configuration written by ``save_scenario``."""
    values = dict(values)
    if values.get("costs"):
        values["costs"] = Costs(**values["costs"])
    for key in ("monthly_min", "monthly_max", "exact_monthly", "regional_capacity"):
        if values.get(key) is not None:
            values[key] = tuple(tuple(item) if key == "regional_capacity" else item for item in values[key])
    return OptimizationConfig(**values)


def _input_signature(base: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [
        base / "store_master_data.xlsx",
        base / "Data" / "raw" / "New Store GPS.xlsx",
        base / "outputs" / "expansion_simulation" / "potential_store_predictions.csv",
        base / "outputs" / "expansion_simulation" / "maturation_profiles.csv",
        base / "outputs" / "cannibalization_regional" / "regional_sales_coefficients.csv",
    ]
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths if path.is_file())


@lru_cache(maxsize=8)
def _prepared_inputs_cached(base_name: str, static_config_json: str, signature: tuple[tuple[str, int, int], ...]):
    """Reuse immutable geographic/model inputs across scenario switches."""
    del signature
    return prepare_inputs(Path(base_name), _json_config(json.loads(static_config_json)))


def _prepared_inputs(base: Path, config: OptimizationConfig):
    static_keys = (
        "as_of", "planning_year", "horizon", "candidate_limit", "demand_scenario",
        "coefficient_source", "coefficient_mode", "tail_assumption", "maturity_tail",
    )
    values = asdict(config)
    static = {key: values[key] for key in static_keys}
    return _prepared_inputs_cached(str(base), json.dumps(static, sort_keys=True), _input_signature(base))


def _descriptor(
    kind: str,
    scenario_id: str,
    label: str,
    directory: Path | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "scenario_id": scenario_id,
        "label": label,
        "directory": str(directory) if directory else None,
        "configuration": asdict(config) if config else None,
    }


def validate_completed_result(result: dict[str, Any]) -> None:
    """Validate an in-memory completed solve before it can become active."""
    summary = result.get("summary") or {}
    selected = result.get("selected")
    monthly = result.get("monthly")
    if not summary.get("scenario_id") or not isinstance(selected, pd.DataFrame):
        raise ValueError("Completed optimization result is missing its scenario identifier or selected portfolio.")
    required = {"candidate_id", "opening_month"}
    if not required.issubset(selected.columns):
        raise ValueError("Completed optimization portfolio is missing candidate IDs or opening months.")
    ids = selected["candidate_id"].astype(str)
    months = pd.to_numeric(selected["opening_month"], errors="coerce")
    if ids.duplicated().any() or ids.eq("").any() or not months.between(1, 12).all():
        raise ValueError("Completed optimization portfolio has duplicate IDs or invalid opening months.")
    if len(selected) != int(summary.get("store_count", -1)):
        raise ValueError("Completed optimization portfolio count does not match the scenario summary.")
    if isinstance(monthly, pd.DataFrame) and "new_openings" in monthly:
        if int(pd.to_numeric(monthly["new_openings"], errors="coerce").fillna(0).sum()) != len(selected):
            raise ValueError("Monthly opening counts do not reconcile to the completed portfolio.")


def reference_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a small session-state-safe reference to a validated result."""
    validate_completed_result(result)
    summary = result["summary"]
    directory = Path(result["directory"])
    return _descriptor(
        "optimization",
        str(summary["scenario_id"]),
        f"Completed optimization · {int(summary['store_count'])} stores · {summary['scenario_id']}",
        directory,
    )


def discover_scenarios(base: Path = ROOT) -> list[dict[str, Any]]:
    """List only complete, internally consistent saved optimizer scenarios."""
    root = Path(base) / "outputs" / "expansion_optimization_v2"
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return records
    for directory in root.iterdir():
        required = [
            directory / "optimization_summary.csv",
            directory / "optimization_diagnostics.json",
            directory / "optimal_selected_stores.csv",
            directory / "optimal_monthly_schedule.csv",
        ]
        if not directory.is_dir() or not all(path.is_file() for path in required):
            continue
        try:
            summary = _read_csv(required[0]).iloc[0].to_dict()
            diagnostics = json.loads(required[1].read_text(encoding="utf-8"))
            selected = _read_csv(required[2])
            monthly = _read_csv(required[3])
            validate_completed_result({"summary": summary, "selected": selected, "monthly": monthly})
            scenario_id = str(summary["scenario_id"])
            if scenario_id != directory.name:
                raise ValueError("scenario ID does not match its result directory")
            config = _json_config(diagnostics["configuration"])
            mode = "Automatic" if config.target is None else "Manual"
            records.append({
                **_descriptor(
                    "optimization", scenario_id,
                    f"{mode} optimization · {len(selected)} stores · {scenario_id}", directory,
                ),
                "mode": mode,
                "store_count": len(selected),
                "planning_year": config.planning_year,
                "horizon": config.horizon,
                "objective": config.objective,
                "modified_ns": max(path.stat().st_mtime_ns for path in required),
            })
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
            # Invalid directories are intentionally omitted. The UI makes this
            # visible as an unavailable saved scenario rather than falling back.
            continue
    return sorted(records, key=lambda row: row["modified_ns"], reverse=True)


def scenario_discovery_issues(base: Path = ROOT) -> list[str]:
    """Return explicit diagnostics for saved directories that cannot be selected."""
    root = Path(base) / "outputs" / "expansion_optimization_v2"
    if not root.is_dir():
        return []
    issues: list[str] = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name == "comparisons":
            continue
        required = [
            directory / "optimization_summary.csv",
            directory / "optimization_diagnostics.json",
            directory / "optimal_selected_stores.csv",
            directory / "optimal_monthly_schedule.csv",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            issues.append(f"{directory.name}: missing " + ", ".join(missing))
            continue
        try:
            summary = _read_csv(required[0]).iloc[0].to_dict()
            diagnostics = json.loads(required[1].read_text(encoding="utf-8"))
            selected = _read_csv(required[2])
            monthly = _read_csv(required[3])
            validate_completed_result({"summary": summary, "selected": selected, "monthly": monthly})
            _json_config(diagnostics["configuration"])
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
            issues.append(f"{directory.name}: {error}")
    return issues


def selectable_scenarios(base: Path = ROOT) -> list[dict[str, Any]]:
    """Build selector choices, with the fixed portfolio available only explicitly."""
    saved = discover_scenarios(base)
    records = [
        _descriptor("none", "no_expansion", "No Expansion"),
        _descriptor("benchmark", "original_180_benchmark", "Original 180-Store Benchmark"),
    ]
    latest_auto = next((row for row in saved if row["mode"] == "Automatic"), None)
    latest_manual = next((row for row in saved if row["mode"] == "Manual"), None)
    used: set[str] = set()
    for prefix, row in (("Latest Automatic Optimization", latest_auto), ("Latest Manual Optimization", latest_manual)):
        if row:
            item = dict(row)
            item["label"] = f"{prefix} · {item['store_count']} stores · {item['scenario_id']}"
            records.append(item)
            used.add(item["scenario_id"])
    for row in saved:
        if row["scenario_id"] not in used:
            records.append(row)
    return records


def _reference_configuration(reference: dict[str, Any], base: Path) -> OptimizationConfig:
    if reference.get("kind") == "optimization":
        directory = Path(str(reference.get("directory", "")))
        diagnostic_file = directory / "optimization_diagnostics.json"
        if not diagnostic_file.is_file():
            raise ValueError("The selected optimization scenario is missing its saved configuration.")
        return _json_config(json.loads(diagnostic_file.read_text(encoding="utf-8"))["configuration"])
    # No-expansion and benchmark evaluations use the newest saved solve's
    # configuration when it is available, so their horizon and baseline match
    # the portfolio a user has just been comparing.  Otherwise this is an
    # explicit 2027 / 24-month baseline, never a fixed selected portfolio.
    latest = discover_scenarios(base)
    if latest:
        return _reference_configuration(latest[0], base)
    return OptimizationConfig(as_of="2026-10-15", planning_year=2027, horizon=24)


def _loaded_plan(evaluator: Evaluator, selected: pd.DataFrame) -> dict[int, int]:
    ids = evaluator.candidates["candidate_id"].astype(str)
    index = dict(zip(ids, range(len(ids))))
    plan: dict[int, int] = {}
    for row in selected.itertuples(index=False):
        candidate_id = str(getattr(row, "candidate_id"))
        if candidate_id not in index:
            raise ValueError(f"Selected candidate {candidate_id!r} is not available in this scenario's forecast input.")
        position = index[candidate_id]
        if position in plan:
            raise ValueError("Selected portfolio maps more than one candidate to one eligible location.")
        month = int(getattr(row, "opening_month")) - 1
        if not 0 <= month < 12:
            raise ValueError("Selected portfolio contains an opening month outside January–December.")
        plan[position] = month
    return plan


def _customer_log_effects(evaluator: Evaluator, plan: dict[int, int]) -> tuple[np.ndarray | None, str | None]:
    """Apply only the supported fixed customer model; never reuse sales betas."""
    if evaluator.config.coefficient_source != "fixed_cumulative":
        return None, "Customer forecast is unavailable for a sales-only event-study scenario."
    values = evaluator.data.existing.get("APSD_customers")
    if values is None or not pd.to_numeric(values, errors="coerce").gt(0).any():
        return None, "Historical APSD customer-equivalent baselines are unavailable."
    output = np.zeros((evaluator.h, evaluator.e))
    bands = _bands(evaluator.data.existing_distances)
    for candidate, opening_month in plan.items():
        for existing_position, region in enumerate(evaluator.data.existing["region"].astype(str)):
            beta = 0.0
            band = bands[candidate, existing_position]
            if band:
                try:
                    beta = FIXED_CUSTOMERS.get(region, FIXED_CUSTOMERS["All Regions"])[
                        next(i for i, item in enumerate(BANDS) if item[0] == band)
                    ]
                except (KeyError, StopIteration):
                    beta = 0.0
            if evaluator.config.coefficient_mode == "negative_only":
                beta = min(beta, 0.0)
            if beta:
                output[opening_month:, existing_position] += beta
                output[opening_month, existing_position] -= beta * (1 - evaluator.fractions[opening_month])
    return output, None


def _pairs_and_allocations(evaluator: Evaluator, plan: dict[int, int], log_effects: np.ndarray) -> pd.DataFrame:
    """Allocate the exact overlapping existing-store change to source pairs."""
    records: list[dict[str, Any]] = []
    if not plan:
        return pd.DataFrame()
    ratio = np.ones_like(log_effects)
    np.divide(np.expm1(log_effects), log_effects, out=ratio, where=np.abs(log_effects) > 1e-12)
    candidates, existing = evaluator.candidates, evaluator.data.existing
    for candidate, opening_month in plan.items():
        log_action, _ = evaluator.action_logs(candidate, opening_month)
        changes = (evaluator.base[opening_month:] * ratio[opening_month:] * log_action).sum(axis=0)
        distance = evaluator.data.existing_distances[candidate]
        for existing_position in np.flatnonzero(distance <= 1000):
            band = _bands(np.array([distance[existing_position]]))[0]
            records.append({
                "candidate_id": str(candidates.iloc[candidate].candidate_id),
                "candidate_name": candidates.iloc[candidate].candidate_name,
                "opening_month": opening_month + 1,
                "opening_date": pd.Timestamp(evaluator.config.planning_year, opening_month + 1, evaluator.config.opening_day),
                "existing_store_id": str(existing.iloc[existing_position].store_id),
                "existing_store_name": existing.iloc[existing_position].store_name,
                "existing_region": existing.iloc[existing_position].region,
                "distance_m": float(distance[existing_position]),
                "distance_band": band,
                "opening_month_APSD_effect_pct": float(np.expm1(log_action[0, existing_position]) * 100),
                "horizon_existing_sales_change": float(changes[existing_position]),
            })
    return pd.DataFrame(records)


def evaluate_expansion_scenario(reference: dict[str, Any], base: Path = ROOT) -> dict[str, Any]:
    """Evaluate one active scenario with the same event-time engine as optimization.

    The returned data are deliberately shared by every business-dashboard
    section. No refitting and no portfolio search occurs here.
    """
    base = Path(base)
    kind = reference.get("kind")
    config = _reference_configuration(reference, base)
    # Count/capacity constraints govern search feasibility, not how an already
    # chosen portfolio is evaluated.  A no-expansion comparison and the frozen
    # benchmark must therefore retain demand/coefficient assumptions while
    # explicitly relaxing a previous solve's count requirement.
    if kind == "none":
        config = replace(
            config, target=0, min_stores=0, max_stores=400, candidate_limit=None,
            monthly_min=(0,) * 12, monthly_max=(40,) * 12, exact_monthly=None,
        )
    elif kind == "benchmark":
        config = replace(
            config, target=None, min_stores=0, max_stores=400, candidate_limit=None,
            monthly_min=(0,) * 12, monthly_max=(40,) * 12, exact_monthly=None,
            regional_capacity=(), budget=None,
        )
    inputs = _prepared_inputs(base, config)
    evaluator = Evaluator(inputs, config)
    source_selected = pd.DataFrame()
    if kind == "optimization":
        directory = Path(str(reference.get("directory", "")))
        required = [directory / "optimization_summary.csv", directory / "optimal_selected_stores.csv", directory / "optimal_monthly_schedule.csv"]
        if not all(path.is_file() for path in required):
            raise ValueError("The selected saved scenario is incomplete; it was not substituted with another portfolio.")
        stored_summary = _read_csv(required[0]).iloc[0].to_dict()
        source_selected = _read_csv(required[1])
        stored_monthly = _read_csv(required[2])
        validate_completed_result({"summary": stored_summary, "selected": source_selected, "monthly": stored_monthly})
        if str(stored_summary["scenario_id"]) != str(reference.get("scenario_id")):
            raise ValueError("The selected scenario reference does not match the saved scenario ID.")
        plan = _loaded_plan(evaluator, source_selected)
        scenario_id = str(stored_summary["scenario_id"])
        scenario_mode = "Automatic" if config.target is None else "Manual"
    elif kind == "benchmark":
        if not BENCHMARK_FILE.is_file():
            raise ValueError("The original 180-store benchmark file is unavailable.")
        plan = saved_baseline(evaluator, BENCHMARK_FILE, frozen=True)
        scenario_id = "original_180_benchmark"
        scenario_mode = "Benchmark"
    elif kind == "none":
        plan = {}
        scenario_id = "no_expansion"
        scenario_mode = "No Expansion"
    else:
        raise ValueError("Unknown active expansion scenario.")

    # Benchmark schedules are intentionally unconstrained historical inputs;
    # completed optimizations already passed their own feasibility validation.
    selected, monthly, existing_impacts, summary = result_tables(evaluator, plan)
    logs, _, _ = evaluator.state(plan)
    monthly["month"] = pd.to_datetime(monthly["month"])
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    pair_impacts = _pairs_and_allocations(evaluator, plan, logs)

    existing = evaluator.data.existing.copy()
    existing["store_id"] = existing["store_id"].astype(str)
    base_apsd = evaluator.data.baseline_sales
    existing_monthly = pd.DataFrame({
        "month": np.repeat(evaluator.months.to_numpy(), evaluator.e),
        "store_id": np.tile(existing["store_id"].to_numpy(), evaluator.h),
        "store_name": np.tile(existing["store_name"].to_numpy(), evaluator.h),
        "region": np.tile(existing["region"].to_numpy(), evaluator.h),
        "baseline_APSD_sales": base_apsd.reshape(-1),
        "modeled_APSD_sales_effect_pct": (np.expm1(logs) * 100).reshape(-1),
        "scenario_APSD_sales": (base_apsd * np.exp(logs)).reshape(-1),
        "baseline_sales_mnt": evaluator.base.reshape(-1),
        "scenario_sales_mnt": (evaluator.base * np.exp(logs)).reshape(-1),
    })

    customer_logs, customer_note = _customer_log_effects(evaluator, plan)
    customer_supported = customer_logs is not None
    if customer_supported:
        customer_base = pd.to_numeric(existing["APSD_customers"], errors="coerce").to_numpy(float)
        existing_monthly["baseline_APSD_customers"] = np.tile(customer_base, evaluator.h)
        existing_monthly["modeled_APSD_customers_effect_pct"] = (np.expm1(customer_logs) * 100).reshape(-1)
        existing_monthly["scenario_APSD_customers"] = (customer_base[None, :] * np.exp(customer_logs)).reshape(-1)

    existing_impacts["store_id"] = existing_impacts["store_id"].astype(str)
    by_store = existing_monthly.groupby("store_id", as_index=False).agg(
        baseline_sales_mnt=("baseline_sales_mnt", "sum"),
        scenario_sales_mnt=("scenario_sales_mnt", "sum"),
    )
    existing_impacts = existing_impacts.drop(columns=["baseline_sales", "scenario_sales"], errors="ignore").merge(
        by_store, on="store_id", how="left", validate="one_to_one"
    )
    existing_impacts["horizon_existing_sales_change"] = (
        existing_impacts["scenario_sales_mnt"] - existing_impacts["baseline_sales_mnt"]
    )
    existing_impacts["horizon_existing_sales_effect_pct"] = np.where(
        existing_impacts["baseline_sales_mnt"].ne(0),
        100 * existing_impacts["horizon_existing_sales_change"] / existing_impacts["baseline_sales_mnt"], np.nan,
    )
    if customer_supported:
        customer_end = existing_monthly.sort_values("month").groupby("store_id", as_index=False).tail(1)
        existing_impacts = existing_impacts.merge(
            customer_end[["store_id", "baseline_APSD_customers", "scenario_APSD_customers", "modeled_APSD_customers_effect_pct"]],
            on="store_id", how="left", validate="one_to_one",
        )

    candidates = evaluator.candidates.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidates = candidates.merge(
        selected[[column for column in (
            "candidate_id", "opening_month", "opening_date", "gross_new_store_sales", "new_to_new_change",
            "horizon_new_store_sales", "allocated_existing_store_change", "allocated_incremental_network_sales",
        ) if column in selected]], on="candidate_id", how="left", validate="one_to_one"
    )
    candidates["selection_status"] = np.where(candidates["opening_month"].notna(), "Selected", "Unselected candidate")
    if pair_impacts.empty:
        candidates["nearby_existing_store_count"] = 0
        candidates["nearby_existing_stores_within_1km"] = ""
    else:
        pair_impacts["_label"] = pair_impacts.apply(
            lambda row: f"{row.existing_store_id} · {row.existing_store_name} ({row.distance_m:,.0f} m)", axis=1
        )
        nearby = pair_impacts.groupby("candidate_id").agg(
            nearby_existing_store_count=("existing_store_id", "nunique"),
            nearby_existing_stores_within_1km=("_label", lambda values: "; ".join(values)),
        )
        candidates = candidates.merge(nearby, on="candidate_id", how="left", validate="one_to_one")
        candidates["nearby_existing_store_count"] = candidates["nearby_existing_store_count"].fillna(0).astype(int)
        candidates["nearby_existing_stores_within_1km"] = candidates["nearby_existing_stores_within_1km"].fillna("")

    summary.update({
        "scenario_id": scenario_id,
        "scenario_mode": scenario_mode,
        "planning_year": config.planning_year,
        "forecast_horizon": config.horizon,
        "objective": config.objective,
        "demand_scenario": config.demand_scenario,
        "coefficient_source": config.coefficient_source,
        "coefficient_mode": config.coefficient_mode,
        "customer_model_supported": customer_supported,
        "customer_model_note": customer_note,
        "existing_network_baseline_sales": float(existing_monthly["baseline_sales_mnt"].sum()),
        "existing_network_scenario_sales": float(existing_monthly["scenario_sales_mnt"].sum()),
    })
    return {
        "summary": summary,
        "monthly": monthly,
        "selected": selected,
        "candidates": candidates,
        "pair_impacts": pair_impacts.drop(columns="_label", errors="ignore"),
        "existing_impacts": existing_impacts,
        "existing_monthly": existing_monthly,
        "existing_map": existing[[column for column in ("store_id", "store_name", "region", "latitude", "longitude") if column in existing]],
    }


def evaluation_signature(reference: dict[str, Any], base: Path = ROOT) -> tuple[tuple[str, int, int], ...]:
    """Scenario-aware cache key for saved results and the shared model inputs."""
    base = Path(base)
    paths = [
        base / "store_master_data.xlsx",
        base / "Data" / "raw" / "New Store GPS.xlsx",
        base / "outputs" / "expansion_simulation" / "potential_store_predictions.csv",
        base / "outputs" / "expansion_simulation" / "maturation_profiles.csv",
        base / "outputs" / "cannibalization_regional" / "regional_sales_coefficients.csv",
        BENCHMARK_FILE,
    ]
    if reference.get("directory"):
        directory = Path(str(reference["directory"]))
        paths.extend(directory / name for name in (
            "optimization_summary.csv", "optimization_diagnostics.json", "optimal_selected_stores.csv", "optimal_monthly_schedule.csv"
        ))
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths if path.is_file())
