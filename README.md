# CU Mongolia cannibalization analysis

The business dashboard has four pages: Cannibalization Overview, Regression
Results, Planned New Store Locations, and Store-Level Exploration. It reads saved
results; it does not fit regressions or reselect the accepted portfolio on interaction.

## Run locally

Use Python 3.11 or newer from the project directory. The existing Windows environment
can be invoked explicitly:

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
.\.conda\python.exe -m streamlit run "05 Cannibalization Dashboard.py"
```

The root is resolved from the source files, including after moving the project.
No code depends on the previous OneDrive project directory.

## Reproduce planning results

```powershell
.\.conda\python.exe regional_cumulative_impact.py --as-of 2026-10-15
.\.conda\python.exe -B -m unittest discover -s tests -v
```

The thin compatibility command `python cumulative_cannibalization.py` also evaluates
planning, using today's local date. For reproducible runs use the explicit as-of date.
`--output-dir PATH` can direct evaluation to a separate validation directory.

- Accepted candidate identities, coordinates and 180/400 selections are frozen under
  `Data/planning_inputs/accepted_v1/`. Do not edit these to run a comparison.
- New evaluation outputs are under `outputs/planning/`. The historical Model4 and
  regional outputs remain intact. Source input hashes are recorded in planning metadata.
- Baseline: latest **completed** calendar month with at least 95% of peak valid income
  coverage among completed months. For October 15, 2026 this selects September 2026.
- Exact user-supplied fixed coefficients are preserved in source and exported as
  `fixed_planning_coefficients.csv`. City Outskirts income uses the All Regions fallback.
- All unique exposures at distances [0,100], (100,250], (250,500], (500,1000] m contribute.
  Beyond 1 km direct contributions are zero. Log contributions sum, then `expm1` converts
  to proportional effects. Income/customer coefficients remain separate.
- Negative-only planning caps positive **pair** coefficients at zero; raw mode preserves
  all signs, including the +0.0042 Outskirts customer coefficient.
- Previously selected sites are evaluated without reselection. Original optimization
  scores are historical and do not establish optimality under the fixed coefficients.
- Weighted network indicators use positive valid baseline APSD and supported effects,
  including zero-exposure stores. Missing coordinates are unavailable for both outcomes.
  These indicators are not annual revenue, new-site demand, or profit forecasts.

`previous_result_comparison.csv` explains the difference from historical October results.
No historical inference is attached to rounded fixed inputs without a verified source.

## Data and estimated evidence

The extraction/preparation scripts are manual entry points and safe to import.
Install `requirements-extraction.txt` only if database extraction is required. Configure
`CU_STORE_METADATA` to the external StoreMaster workbook. Configure ClickHouse through
`CU_CLICKHOUSE_HOST`, `CU_CLICKHOUSE_PORT`, `CU_CLICKHOUSE_DATABASE`,
`CU_CLICKHOUSE_USER`, and either `CU_CLICKHOUSE_PASSWORD` or `CU_CLICKHOUSE_PASSWORD_FILE`.
Do not place secrets in source or IDE connection settings. Extraction preserves the
existing active-day and transaction-derived customer definitions; review their analytical
limitations in the audit before changing them. Optional GS25 KMZ processing needs
GeoPandas and a GDAL build with LIBKML support.

For offline regressions install `requirements-modeling.txt`:

```powershell
.\.conda\python.exe "03 Modelling.py"
.\.conda\python.exe "04 Regional Cannibalization.py"
```

Default runs estimate cumulative models only and write separate
`outputs/national_cumulative/` and `outputs/cumulative_evidence/` results.
Use `--full-diagnostics` on either script to reproduce its original complete suite and
original output paths. This can overwrite that suite's outputs; retain a snapshot first.
The business evidence page prefers newly saved cumulative evidence if available and
otherwise reads the original regional results. The planning scenario always uses the
approved fixed source. Event studies remain diagnostic; observed pretrends limit causal claims.

## Optional analyst views

Legacy diagnostics, statistics, capacity and conditional demand/revenue simulation remain
available without adding main business pages:

```powershell
$env:CU_ANALYST_MODE = '1'
.\.conda\python.exe -m streamlit run "05 Cannibalization Dashboard.py"
```

Unset `CU_ANALYST_MODE` to return to business navigation. `expansion_simulation.py` remains
an offline research branch with a different objective and candidate identity scheme;
it does not replace the frozen planning portfolio. No reliable profitability model is claimed.

See `PYTHON_PROJECT_AUDIT.md` for the original audit and `IMPLEMENTATION_NOTES.md` for
the approved cleanup, validation and numerical comparisons.
