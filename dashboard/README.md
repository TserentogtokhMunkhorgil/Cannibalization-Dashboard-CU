# CU Store Network & Performance Dashboard

The dashboard reuses the existing monthly sales/master workbook and store-pair
distance workbook. It does not connect to ClickHouse or alter the econometric
analysis.

## Prepare and run

From the project folder, run:

```powershell
python dashboard\prepare_data.py
streamlit run dashboard\main.py
```

The navigation provides **Store Performance** and **Cannibalization Analysis**.
The existing Store Performance app remains available at `dashboard\app.py`.
The cannibalization page reads the existing saved outputs in
`outputs\cannibalization_model4\`; it never runs or changes the regression.

Preparation writes:

- `data\dashboard_store_month.parquet`
- `data\potential_store_exposure.parquet`
- `outputs\data_quality_report.csv`

The app reads those precomputed files and does not reload or reaggregate the
Excel sources when filters change.

## Source measures and interpretation

- `APSD` is the source `APSD_income` measure; it is not recalculated.
- `Customers` is the existing pipeline measure derived from transaction counts
  after subtracting twice the returned-transaction count. It is not a count of
  uniquely identified people.
- Store-month records are expanded over the observed month range, then masked
  outside the store's dated operating period. Missing opening dates and inactive
  stores without a closing date cannot be reliably located in time and are not
  included in the operating network.
- Numeric, decimal-comma, and DMS coordinates are parsed from the source values;
  coordinates that remain missing or invalid are not inferred.
- New-store event exposure counts only pairs where the existing store opened
  strictly before the new store. APSD exposure uses the previous complete
  calendar month's source APSD for those nearby stores where data exists.
- Nearby-new-store counts on the map count later-opening stores already opened
  by the selected month. These descriptive measures are not causal estimates.

The map uses the source latitude and longitude only. Stores without valid source
coordinates remain in the table and KPI totals but cannot be plotted.

## Cannibalization results page

The separate cannibalization page visualizes the saved Model D percentage
effects and confidence intervals, the distance-by-event-time heatmap, exposure
summaries, and model diagnostics. Store-level exposure comes from the saved
model panel; when `store_to_store_distance.xlsx` is available, it is used only
to show descriptive nearby-opening details.
