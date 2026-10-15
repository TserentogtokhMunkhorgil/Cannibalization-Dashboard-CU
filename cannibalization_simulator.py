"""Pure calculations for the regional CU store-opening capacity simulator."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import sys
from typing import Mapping

import pandas as pd


DISTANCE_BANDS = (
    ("0_100", "0–100 m"),
    ("100_250", "100–250 m"),
    ("250_500", "250–500 m"),
    ("500_1000", "500–1,000 m"),
)
MAX_OPTIMIZATION_CANDIDATES = 500_000
MAX_EXACT_FLOAT_INTEGER = 2**53 - 1


def weighted_network_effect(baselines, effects_pct) -> float:
    """Average supported percentage effects, retaining valid zero exposures.

    Inputs are APSD weights and percentage-point effects, not revenue totals.
    Missing effects are unavailable, not zero. Raises for an empty population.
    """
    weights = pd.Series(baselines, dtype=float).reset_index(drop=True)
    effects = pd.Series(effects_pct, dtype=float).reset_index(drop=True)
    if len(weights) != len(effects):
        raise ValueError('Baseline and effect lengths must match.')
    valid = weights.map(math.isfinite) & weights.gt(0) & effects.map(math.isfinite)
    if not valid.any():
        raise ValueError('No supported positive APSD weights.')
    return float((weights[valid] * effects[valid]).sum() / weights[valid].sum())


@dataclass(frozen=True)
class CoefficientEstimate:
    beta: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    exposed_stores: int | None
    observations: int | None
    issue: str | None = None


@dataclass(frozen=True)
class CapacityResult:
    status: str
    maximum_openings: int | None = None
    predicted_at_maximum: float | None = None
    predicted_at_next: float | None = None
    issue: str | None = None


@dataclass(frozen=True)
class ScenarioResult:
    total_openings: int
    predicted_apsd: float
    absolute_change: float
    percentage_change: float
    headroom: float
    constraint_satisfied: bool


def validate_sales_inputs(baseline: float, minimum: float) -> None:
    """Require positive APSD inputs with the minimum no higher than baseline."""
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Current APSD sales must be a finite amount greater than zero.")
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("Minimum required APSD sales must be a finite amount greater than zero.")
    if minimum > baseline:
        raise ValueError("Minimum required APSD sales cannot exceed current APSD sales.")


def _number(value: object) -> float | None:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    result = float(converted)
    return result if math.isfinite(result) else None


def _confidence_interval(row: pd.Series, beta: float) -> tuple[float | None, float | None]:
    lower = _number(row.get("ci_95_lower"))
    upper = _number(row.get("ci_95_upper"))
    if (
        lower is not None
        and upper is not None
        and lower <= beta <= upper
        and lower <= upper
    ):
        return lower, upper

    percent_lower = _number(row.get("percent_ci_95_lower"))
    percent_upper = _number(row.get("percent_ci_95_upper"))
    if (
        percent_lower is None
        or percent_upper is None
        or percent_lower <= -100
        or percent_upper <= -100
        or percent_lower > percent_upper
    ):
        return None, None
    lower = math.log1p(percent_lower / 100)
    upper = math.log1p(percent_upper / 100)
    if lower <= beta <= upper:
        return lower, upper
    return None, None


def regional_sales_estimates(
    coefficients: pd.DataFrame, region: str
) -> dict[str, CoefficientEstimate]:
    """Read only this region's cumulative log-APSD-sales estimates."""
    required = {"region", "model_type", "distance_band"}
    estimates: dict[str, CoefficientEstimate] = {}
    if not required.issubset(coefficients.columns):
        issue = "The coefficient file is missing required columns."
        return {
            band: CoefficientEstimate(None, None, None, None, None, issue)
            for band, _ in DISTANCE_BANDS
        }

    for band, _ in DISTANCE_BANDS:
        rows = coefficients.loc[
            (coefficients["region"] == region)
            & (coefficients["model_type"] == "cumulative")
            & (coefficients["distance_band"] == band)
        ]
        issue: str | None = None
        if "outcome" in rows and not rows.empty:
            rows = rows.loc[rows["outcome"] == "sales"]
        if rows.empty:
            estimates[band] = CoefficientEstimate(
                None, None, None, None, None, "No cumulative sales estimate for this region and band."
            )
            continue
        if len(rows) != 1:
            estimates[band] = CoefficientEstimate(
                None, None, None, None, None, "Duplicate coefficient records make this estimate ambiguous."
            )
            continue

        row = rows.iloc[0]
        scale = row.get("outcome_scale")
        beta = _number(row.get("coefficient"))
        if scale is not None and not pd.isna(scale) and str(scale).lower() != "log":
            beta = None
            issue = f"Unsupported outcome scale: {scale}."
        if beta is None and (scale is None or str(scale).lower() == "log"):
            percent_effect = _number(row.get("percent_effect"))
            if percent_effect is not None and percent_effect > -100:
                beta = math.log1p(percent_effect / 100)
        if beta is None:
            issue = issue or "Missing or invalid log-scale coefficient and percent effect."

        lower: float | None = None
        upper: float | None = None
        if beta is not None:
            lower, upper = _confidence_interval(row, beta)
        if beta is not None and (lower is None or upper is None):
            issue = issue or "A valid 95% confidence interval is unavailable."

        exposed = _number(row.get("exposed_stores"))
        observations = _number(row.get("observations"))
        estimates[band] = CoefficientEstimate(
            beta,
            lower,
            upper,
            int(exposed) if exposed is not None and exposed >= 0 else None,
            int(observations) if observations is not None and observations >= 0 else None,
            issue,
        )
    return estimates


def _validate_counts(openings: Mapping[str, int]) -> None:
    unknown = set(openings).difference(band for band, _ in DISTANCE_BANDS)
    if unknown:
        raise ValueError(f"Unsupported distance bands: {sorted(unknown)}")
    for band, count in openings.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Opening count for {band} must be a non-negative integer.")


def _log_prediction(
    baseline: float,
    openings: Mapping[str, int],
    coefficients: Mapping[str, float | None],
) -> float:
    _validate_counts(openings)
    effect = 0.0
    for band, count in openings.items():
        if count == 0:
            continue
        beta = coefficients.get(band)
        if beta is None or not math.isfinite(beta):
            raise ValueError(f"No valid coefficient is available for {band}.")
        effect += count * beta
    return math.log(baseline) + effect


def predict_apsd(
    baseline: float,
    openings: Mapping[str, int],
    coefficients: Mapping[str, float | None],
) -> float:
    """Apply cumulative log-scale effects to baseline APSD sales."""
    log_prediction = _log_prediction(baseline, openings, coefficients)
    if log_prediction > math.log(sys.float_info.max):
        raise ValueError("Predicted APSD exceeds the numeric range and cannot be displayed.")
    return math.exp(log_prediction)


def maximum_openings(
    baseline: float,
    minimum: float,
    beta: float | None,
) -> CapacityResult:
    """Find a single-band integer capacity and verify both sides of its boundary."""
    validate_sales_inputs(baseline, minimum)
    if beta is None or not math.isfinite(beta):
        return CapacityResult("Not estimable", issue="The selected coefficient is missing or invalid.")
    if beta >= 0:
        return CapacityResult(
            "No finite maximum under this model",
            issue="The estimated effect is zero or positive, so the APSD constraint does not impose a finite capacity.",
        )
    if minimum == baseline:
        predicted_at_next = predict_apsd(
            baseline, {DISTANCE_BANDS[0][0]: 1}, {DISTANCE_BANDS[0][0]: beta}
        )
        if predicted_at_next >= minimum:
            raise ArithmeticError("The next-opening boundary check failed.")
        return CapacityResult(
            "Finite capacity",
            0,
            baseline,
            predicted_at_next,
        )

    threshold = math.log(minimum)
    ratio = math.log(minimum / baseline) / beta
    if not math.isfinite(ratio) or ratio > MAX_EXACT_FLOAT_INTEGER:
        return CapacityResult("Not estimable", issue="The finite capacity is outside the supported numeric range.")
    count = max(0, math.floor(ratio))
    while count > 0 and math.log(baseline) + count * beta < threshold:
        count -= 1
    while math.log(baseline) + (count + 1) * beta >= threshold:
        count += 1
    single_band = DISTANCE_BANDS[0][0]
    predicted_at_maximum = predict_apsd(
        baseline, {single_band: count}, {single_band: beta}
    )
    predicted_at_next = predict_apsd(
        baseline, {single_band: count + 1}, {single_band: beta}
    )
    if predicted_at_maximum < minimum or predicted_at_next >= minimum:
        raise ArithmeticError("The calculated single-band capacity failed its boundary check.")
    return CapacityResult(
        "Finite capacity",
        count,
        predicted_at_maximum,
        predicted_at_next,
    )


def simulate_mixed_scenario(
    baseline: float,
    minimum: float,
    openings: Mapping[str, int],
    coefficients: Mapping[str, float | None],
) -> ScenarioResult:
    validate_sales_inputs(baseline, minimum)
    _validate_counts(openings)
    predicted = predict_apsd(baseline, openings, coefficients)
    total = sum(openings.get(band, 0) for band, _ in DISTANCE_BANDS)
    return ScenarioResult(
        total_openings=total,
        predicted_apsd=predicted,
        absolute_change=predicted - baseline,
        percentage_change=(predicted / baseline - 1) * 100,
        headroom=predicted - minimum,
        constraint_satisfied=predicted >= minimum,
    )


def optimize_openings(
    baseline: float,
    minimum: float,
    upper_bounds: Mapping[str, int],
    coefficients: Mapping[str, float | None],
) -> tuple[dict[str, int], ScenarioResult]:
    """Exhaustively maximize total openings within small user-specified bounds."""
    validate_sales_inputs(baseline, minimum)
    _validate_counts(upper_bounds)
    bands = [band for band, _ in DISTANCE_BANDS]
    bounds = [upper_bounds.get(band, 0) for band in bands]
    candidate_count = math.prod(bound + 1 for bound in bounds)
    if candidate_count > MAX_OPTIMIZATION_CANDIDATES:
        raise ValueError(
            f"Candidate bounds create {candidate_count:,} allocations; "
            f"the exhaustive-search limit is {MAX_OPTIMIZATION_CANDIDATES:,}."
        )
    missing = [
        band
        for band, bound in zip(bands, bounds)
        if bound > 0 and (coefficients.get(band) is None or not math.isfinite(coefficients[band]))
    ]
    if missing:
        raise ValueError(
            "Optimization is not estimable for candidate bands with missing coefficients: "
            + ", ".join(missing)
        )

    threshold_effect = math.log(minimum / baseline)
    best_counts: tuple[int, ...] | None = None
    best_total = -1
    best_effect = -math.inf
    for candidate in product(*(range(bound + 1) for bound in bounds)):
        effect = sum(
            count * float(coefficients[band])
            for band, count in zip(bands, candidate)
            if count
        )
        if effect < threshold_effect:
            continue
        total = sum(candidate)
        if total > best_total or (total == best_total and effect > best_effect):
            best_counts = candidate
            best_total = total
            best_effect = effect

    if best_counts is None:
        raise ValueError("No feasible allocation was found within the supplied bounds.")
    allocation = dict(zip(bands, best_counts))
    result = simulate_mixed_scenario(baseline, minimum, allocation, coefficients)
    return allocation, result
