"""
CU STORE CANNIBALIZATION ANALYSIS
=================================

Main model:
    Store FE + Month FE
    + Distance × Event-Time exposure

Outcome:
    log(APSD_income)

Treatment:
    Nearby new-store openings

Distance bands:
    0-100m
    100-250m
    250-500m
    500-1000m

Event window:
    -6 to +6 months

Reference month:
    -1

Standard errors:
    Clustered by existing store

Inputs:
    store_master_data.xlsx
    store_to_store_distance.xlsx

Outputs:
    outputs/cannibalization_model4/
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INCOME_FILE = BASE_DIR / "store_master_data.xlsx"
DISTANCE_FILE = BASE_DIR / "store_to_store_distance.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs" / "cannibalization_model4"

SHEET_NAME = "Sheet1"


# Distance bands
DISTANCE_BANDS = {
    "0_100": (0, 100),
    "100_250": (100, 250),
    "250_500": (250, 500),
    "500_1000": (500, 1000),
}


# Event-study window
EVENT_TIMES = range(-6, 7)

# Omitted reference period
REFERENCE_MONTH = -1


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def log(message):
    print(f"\n[CU CANNIBALIZATION] {message}", flush=True)


def event_suffix(event_time):
    """Convert -6 -> m6, 0 -> p0, 6 -> p6."""
    if event_time < 0:
        return f"m{abs(event_time)}"
    return f"p{event_time}"


def distance_band(distance):
    """Assign distance to analytical band."""

    if 0 <= distance < 100:
        return "0_100"

    if 100 <= distance < 250:
        return "100_250"

    if 250 <= distance < 500:
        return "250_500"

    if 500 <= distance < 1000:
        return "500_1000"

    if distance >= 1000:
        return "control"

    return None


def check_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    log("Loading input files")

    if not INCOME_FILE.exists():
        raise FileNotFoundError(INCOME_FILE)

    if not DISTANCE_FILE.exists():
        raise FileNotFoundError(DISTANCE_FILE)

    income = pd.read_excel(
        INCOME_FILE,
        sheet_name=SHEET_NAME
    )

    distance = pd.read_excel(
        DISTANCE_FILE,
        sheet_name=SHEET_NAME
    )

    print(
        f"Income rows: {len(income):,}"
    )

    print(
        f"Distance rows: {len(distance):,}"
    )

    return income, distance


# ============================================================
# 4. VALIDATE DATA
# ============================================================

def validate_data(income, distance):

    log("Validating data")

    check_columns(
        income,
        [
            "SalesMonth",
            "store_id",
            "APSD_income",
            "ost_dt",
            "loc_area_new",
            "LATITUDE",
            "LONGITUDE",
        ],
        "Income data"
    )

    check_columns(
        distance,
        [
            "existing_store_id",
            "new_store_id",
            "distance_m",
            "existing_opening_date",
            "new_opening_date",
        ],
        "Distance data"
    )

    income = income.copy()
    distance = distance.copy()

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    income["SalesMonth"] = pd.to_datetime(
        income["SalesMonth"],
        errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()

    income["ost_dt"] = pd.to_datetime(
        income["ost_dt"],
        errors="coerce"
    )

    distance["existing_opening_date"] = pd.to_datetime(
        distance["existing_opening_date"],
        errors="coerce"
    )

    distance["new_opening_date"] = pd.to_datetime(
        distance["new_opening_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric variables
    # --------------------------------------------------------

    income["APSD_income"] = pd.to_numeric(
        income["APSD_income"],
        errors="coerce"
    )

    distance["distance_m"] = pd.to_numeric(
        distance["distance_m"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Check store-month uniqueness
    # --------------------------------------------------------

    duplicate_income = income.duplicated(
        ["store_id", "SalesMonth"]
    ).sum()

    duplicate_pairs = distance.duplicated(
        ["existing_store_id", "new_store_id"]
    ).sum()

    print(
        f"Duplicate store-month rows: {duplicate_income:,}"
    )

    print(
        f"Duplicate store-pair rows: {duplicate_pairs:,}"
    )

    if duplicate_income:
        raise ValueError(
            "Income data is not unique at store-month level."
        )

    if duplicate_pairs:
        raise ValueError(
            "Distance data contains duplicate existing/new-store pairs."
        )

    # --------------------------------------------------------
    # Distance bands
    # --------------------------------------------------------

    distance["distance_band"] = (
        distance["distance_m"]
        .map(distance_band)
    )

    distance["new_opening_month"] = (
        distance["new_opening_date"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # Basic diagnostics
    # --------------------------------------------------------

    print(
        f"Existing stores: "
        f"{income['store_id'].nunique():,}"
    )

    print(
        f"New stores: "
        f"{distance['new_store_id'].nunique():,}"
    )

    print(
        f"Sales period: "
        f"{income['SalesMonth'].min().date()} "
        f"to "
        f"{income['SalesMonth'].max().date()}"
    )

    print("\nDistance bands:")

    print(
        distance["distance_band"]
        .value_counts(dropna=False)
        .sort_index()
    )

    return income, distance


# ============================================================
# 5. BUILD STORE-MONTH PANEL
# ============================================================

def build_panel(income, distance):

    log("Building store-month panel")

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

    # --------------------------------------------------------
    # Log outcome
    # --------------------------------------------------------

    panel["log_income"] = np.where(
        panel["APSD_income"] > 0,
        np.log(panel["APSD_income"]),
        np.nan
    )

    panel["month"] = (
        panel["SalesMonth"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # Coordinate validity
    # --------------------------------------------------------

    panel["valid_coordinates"] = (
        panel["LATITUDE"].between(-90, 90)
        &
        panel["LONGITUDE"].between(-180, 180)
    )

    print(
        f"Panel observations: {len(panel):,}"
    )

    print(
        f"Stores: "
        f"{panel['existing_store_id'].nunique():,}"
    )

    print(
        f"Months: "
        f"{panel['month'].nunique():,}"
    )

    print(
        f"Positive income observations: "
        f"{panel['log_income'].notna().sum():,}"
    )

    return panel


# ============================================================
# 6. CREATE EVENT-TIME EXPOSURE VARIABLES
# ============================================================

def create_event_exposures(panel, distance):

    log("Creating distance × event-time exposures")

    # Only model stores within 1 km
    events = distance.loc[
        distance["distance_band"].isin(
            DISTANCE_BANDS.keys()
        )
    ].copy()

    exposure_tables = []

    # --------------------------------------------------------
    # For every event month
    # --------------------------------------------------------

    for event_time in EVENT_TIMES:

        temp = events[
            [
                "existing_store_id",
                "new_store_id",
                "distance_band",
                "new_opening_month",
            ]
        ].copy()

        # Month relative to opening
        temp["SalesMonth"] = (
            temp["new_opening_month"]
            + event_time
        ).dt.to_timestamp()

        temp["event_time"] = event_time

        # Keep only months observed in panel
        temp = temp.merge(
            panel[
                [
                    "existing_store_id",
                    "SalesMonth",
                ]
            ],
            on=[
                "existing_store_id",
                "SalesMonth",
            ],
            how="inner"
        )

        exposure_tables.append(temp)

    events_long = pd.concat(
        exposure_tables,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Count exposure events
    # --------------------------------------------------------

    exposure = (
        events_long
        .groupby(
            [
                "existing_store_id",
                "SalesMonth",
                "distance_band",
                "event_time",
            ]
        )
        .size()
        .rename("exposure")
        .reset_index()
    )

    # --------------------------------------------------------
    # Convert long → wide
    # --------------------------------------------------------

    exposure = exposure.pivot_table(
        index=[
            "existing_store_id",
            "SalesMonth",
        ],
        columns=[
            "distance_band",
            "event_time",
        ],
        values="exposure",
        fill_value=0
    )

    exposure.columns = [
        f"exp_{band}_{event_suffix(event)}"
        for band, event in exposure.columns
    ]

    exposure = exposure.reset_index()

    # --------------------------------------------------------
    # Merge with panel
    # --------------------------------------------------------

    panel = panel.merge(
        exposure,
        on=[
            "existing_store_id",
            "SalesMonth",
        ],
        how="left",
        validate="one_to_one"
    )

    exposure_columns = [
        column
        for column in panel.columns
        if column.startswith("exp_")
    ]

    panel[exposure_columns] = (
        panel[exposure_columns]
        .fillna(0)
        .astype(int)
    )

    print(
        f"Distance-event exposure variables: "
        f"{len(exposure_columns):,}"
    )

    print(
        f"Pair-event observations: "
        f"{len(events_long):,}"
    )

    return panel, events_long


# ============================================================
# 7. MODEL VARIABLE LISTS
# ============================================================

def model_b_variables():

    return [
        f"exp_{band}_p0"
        for band in DISTANCE_BANDS
    ]


def model_c_variables():

    return [
        f"exp_all_{event_suffix(event)}"
        for event in EVENT_TIMES
        if event != REFERENCE_MONTH
    ]


def model_d_variables():

    return [
        f"exp_{band}_{event_suffix(event)}"
        for band in DISTANCE_BANDS
        for event in EVENT_TIMES
        if event != REFERENCE_MONTH
    ]


# ============================================================
# 8. CREATE ALL-DISTANCE EVENT VARIABLES
# ============================================================

def create_all_distance_events(panel):

    for event in EVENT_TIMES:

        columns = [
            f"exp_{band}_{event_suffix(event)}"
            for band in DISTANCE_BANDS
        ]

        panel[
            f"exp_all_{event_suffix(event)}"
        ] = panel[columns].sum(axis=1)

    return panel


# ============================================================
# 9. TWO-WAY FIXED EFFECTS
# ============================================================

def residualize(values, store_codes, month_codes):

    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values[:, None]

    result = values.copy()

    for _ in range(1000):

        previous = result.copy()

        # Store FE
        store_means = (
            pd.DataFrame(result)
            .groupby(store_codes)
            .transform("mean")
            .to_numpy()
        )

        result -= store_means

        # Month FE
        month_means = (
            pd.DataFrame(result)
            .groupby(month_codes)
            .transform("mean")
            .to_numpy()
        )

        result -= month_means

        if np.max(
            np.abs(result - previous)
        ) < 1e-9:

            return result

    raise RuntimeError(
        "Fixed-effect residualization did not converge."
    )


# ============================================================
# 10. ESTIMATE MODEL
# ============================================================

def estimate_model(
    panel,
    regressors,
    model_name
):

    log(
        f"Estimating {model_name}"
    )

    sample = panel.loc[
        panel["log_income"].notna()
    ].copy()

    # --------------------------------------------------------
    # Fixed-effect identifiers
    # --------------------------------------------------------

    store_codes, stores = pd.factorize(
        sample["existing_store_id"]
    )

    month_codes, months = pd.factorize(
        sample["month"]
    )

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    y = residualize(
        sample["log_income"].values,
        store_codes,
        month_codes
    ).ravel()

    # --------------------------------------------------------
    # Treatment variables
    # --------------------------------------------------------

    X = sample[
        regressors
    ].values

    X = residualize(
        X,
        store_codes,
        month_codes
    )

    # --------------------------------------------------------
    # Rank check
    # --------------------------------------------------------

    rank = np.linalg.matrix_rank(X)

    if rank < X.shape[1]:

        raise RuntimeError(
            f"{model_name}: treatment variables are not "
            f"fully identified. Rank={rank}, "
            f"variables={X.shape[1]}."
        )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    model = sm.OLS(
        y,
        X,
        hasconst=False
    )

    results = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
            sample["existing_store_id"]
        }
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    output = []

    for i, variable in enumerate(regressors):

        beta = results.params[i]
        se = results.bse[i]

        ci_low, ci_high = (
            results.conf_int()[i]
        )

        output.append(
            {
                "model": model_name,
                "term": variable,
                "coefficient": beta,
                "standard_error": se,
                "p_value": results.pvalues[i],

                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,

                # Convert log coefficient to %
                "percent_effect":
                    100 * np.expm1(beta),

                "percent_ci_lower":
                    100 * np.expm1(ci_low),

                "percent_ci_upper":
                    100 * np.expm1(ci_high),

                "stores_exposed":
                    sample.loc[
                        sample[variable] > 0,
                        "existing_store_id"
                    ].nunique(),

                "exposure_count":
                    sample[variable].sum(),
            }
        )

    coefficients = pd.DataFrame(output)

    # --------------------------------------------------------
    # Model statistics
    # --------------------------------------------------------

    residuals = results.resid

    r2 = (
        1 -
        np.sum(residuals ** 2)
        /
        np.sum(y ** 2)
    )

    summary = {
        "model": model_name,
        "observations": len(sample),
        "stores": sample[
            "existing_store_id"
        ].nunique(),

        "months":
            sample["month"].nunique(),

        "treatment_variables":
            len(regressors),

        "within_R2":
            r2,

        "cluster":
            "existing_store_id",

        "reference_event":
            REFERENCE_MONTH,
    }

    return summary, coefficients


# ============================================================
# 11. MODEL B
# ============================================================

def run_model_b(panel):

    """
    Model B:
        Number of nearby new stores opened
        by distance band.

    Interpretation:
        Effect of one additional nearby opening.
    """

    regressors = []

    for band in DISTANCE_BANDS:

        column = f"exp_{band}_p0"

        # p0 = opening month
        # cumulative exposure can be created separately
        regressors.append(column)

    return estimate_model(
        panel,
        regressors,
        "Model B"
    )


# ============================================================
# 12. MODEL C
# ============================================================

def run_model_c(panel):

    """
    Model C:
        Event study across all distances < 1km.
    """

    regressors = model_c_variables()

    return estimate_model(
        panel,
        regressors,
        "Model C"
    )


# ============================================================
# 13. MODEL D — MAIN MODEL
# ============================================================

def run_model_d(panel):

    """
    MAIN MODEL

    log(APSD_income_it)
        =
        store FE
        + month FE
        + distance × event-time exposure
        + error_it

    Reference:
        event month -1

    Interpretation:
        beta = change in log income associated with
        an additional nearby opening at a specific
        distance and event month.
    """

    regressors = model_d_variables()

    return estimate_model(
        panel,
        regressors,
        "Model D"
    )


# ============================================================
# 14. PRE-TREND TEST
# ============================================================

def test_pretrends(
    panel,
    coefficients,
    model_name
):

    model_results = coefficients.loc[
        coefficients["model"] == model_name
    ].copy()

    model_results["event_time"] = (
        model_results["term"]
        .str.extract(r"(m\d+|p\d+)$")[0]
    )

    model_results["event_time"] = (
        model_results["event_time"]
        .map(
            lambda x:
            -int(x[1:])
            if x.startswith("m")
            else int(x[1:])
        )
    )

    leads = model_results.loc[
        model_results["event_time"]
        .between(-6, -2)
    ]

    print(
        f"\n{model_name} pre-treatment coefficients:"
    )

    print(
        leads[
            [
                "term",
                "coefficient",
                "p_value",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 15. CREATE EVENT-STUDY PLOT
# ============================================================

def plot_event_study(
    coefficients,
    model_name="Model D"
):

    log(
        f"Creating {model_name} event-study plot"
    )

    data = coefficients.loc[
        coefficients["model"] == model_name
    ].copy()

    data["event_time"] = (
        data["term"]
        .str.extract(r"(m\d+|p\d+)$")[0]
        .map(
            lambda x:
            -int(x[1:])
            if x.startswith("m")
            else int(x[1:])
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for band in DISTANCE_BANDS:

        prefix = f"exp_{band}_"

        temp = data.loc[
            data["term"].str.startswith(prefix)
        ].sort_values(
            "event_time"
        )

        if temp.empty:
            continue

        ax.errorbar(
            temp["event_time"],
            temp["coefficient"],
            yerr=[
                temp["coefficient"]
                - temp["ci_95_lower"],

                temp["ci_95_upper"]
                - temp["coefficient"],
            ],
            marker="o",
            capsize=3,
            label=band,
        )

    # Zero effect
    ax.axhline(
        0,
        linewidth=1
    )

    # Opening month
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    # Reference month
    ax.axvline(
        -1,
        linestyle=":",
        linewidth=1
    )

    ax.set_xlabel(
        "Months relative to nearby store opening"
    )

    ax.set_ylabel(
        "Effect on log APSD income"
    )

    ax.set_title(
        "CU Store Cannibalization: "
        "Distance × Event Time"
    )

    ax.set_xticks(
        list(EVENT_TIMES)
    )

    ax.legend(
        title="Distance band"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "model4_distance_event_study.png",
        dpi=200
    )

    plt.close(fig)


# ============================================================
# 16. DIAGNOSTICS
# ============================================================

def create_diagnostics(
    panel,
    distance,
    events
):

    log("Creating analytical diagnostics")

    diagnostics = {
        "store_month_observations":
            len(panel),

        "existing_stores":
            panel[
                "existing_store_id"
            ].nunique(),

        "months":
            panel["month"].nunique(),

        "positive_income_observations":
            panel["log_income"].notna().sum(),

        "nearby_pair_events":
            len(events),

        "existing_stores_with_nearby_opening":
            events[
                "existing_store_id"
            ].nunique(),

        "new_stores":
            distance[
                "new_store_id"
            ].nunique(),

        "minimum_distance_m":
            distance["distance_m"].min(),

        "median_distance_m":
            distance["distance_m"].median(),

        "maximum_distance_m":
            distance["distance_m"].max(),
    }

    result = pd.DataFrame(
        diagnostics.items(),
        columns=[
            "metric",
            "value"
        ]
    )

    print(
        "\nAnalytical sample:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# 17. SAVE RESULTS
# ============================================================

def save_results(
    panel,
    diagnostics,
    model_summaries,
    coefficients
):

    log("Saving results")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    diagnostics.to_csv(
        OUTPUT_DIR /
        "sample_diagnostics.csv",
        index=False
    )

    pd.DataFrame(
        model_summaries
    ).to_csv(
        OUTPUT_DIR /
        "model_comparison.csv",
        index=False
    )

    coefficients.to_csv(
        OUTPUT_DIR /
        "all_model_coefficients.csv",
        index=False
    )

    coefficients.loc[
        coefficients["model"] == "Model D"
    ].to_csv(
        OUTPUT_DIR /
        "model4_coefficients.csv",
        index=False
    )

    panel.to_parquet(
        OUTPUT_DIR /
        "model4_panel.parquet",
        index=False
    )


# ============================================================
# 18. MAIN ANALYSIS
# ============================================================

def main():

    log("Starting CU cannibalization analysis")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    income, distance = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    income, distance = validate_data(
        income,
        distance
    )

    # --------------------------------------------------------
    # Store-month panel
    # --------------------------------------------------------

    panel = build_panel(
        income,
        distance
    )

    # --------------------------------------------------------
    # Event exposure
    # --------------------------------------------------------

    panel, events = create_event_exposures(
        panel,
        distance
    )

    # --------------------------------------------------------
    # All-distance event exposure
    # --------------------------------------------------------

    panel = create_all_distance_events(
        panel
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = create_diagnostics(
        panel,
        distance,
        events
    )

    # --------------------------------------------------------
    # Estimate models
    # --------------------------------------------------------

    model_summaries = []
    all_coefficients = []

    # Model C
    summary_c, coef_c = run_model_c(
        panel
    )

    model_summaries.append(
        summary_c
    )

    all_coefficients.append(
        coef_c
    )

    # Model D — MAIN MODEL
    summary_d, coef_d = run_model_d(
        panel
    )

    model_summaries.append(
        summary_d
    )

    all_coefficients.append(
        coef_d
    )

    coefficients = pd.concat(
        all_coefficients,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Pre-trend diagnostics
    # --------------------------------------------------------

    test_pretrends(
        panel,
        coefficients,
        "Model C"
    )

    test_pretrends(
        panel,
        coefficients,
        "Model D"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        panel,
        diagnostics,
        model_summaries,
        coefficients
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_event_study(
        coefficients,
        "Model D"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=================================================="
    )

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


# ============================================================
# 19. RUN
# ============================================================

if __name__ == "__main__":

    with warnings.catch_warnings():

        warnings.simplefilter(
            "default"
        )

        main()