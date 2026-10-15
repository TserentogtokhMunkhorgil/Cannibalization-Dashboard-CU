# Approved cleanup implementation

Implemented following approval of `PYTHON_PROJECT_AUDIT.md`. Existing raw files,
historical fitted results, accepted portfolio files and their coordinates were preserved.

## Changes

- Four main business pages in `business_dashboard.py`, reached through the unchanged
  Streamlit entry filename. Original analyst functions remain accessible using
  `CU_ANALYST_MODE=1`. Optional research outputs were retained, not deleted.
- Page-specific cached IO with file-modification keys; pair details stored in Parquet
  and read only for store exploration. Main dashboard no longer reads the entire
  historical modeling suite before selecting a page.
- One shared APSD-weighted effect utility, separate fixed/raw pair-mode function,
  checked input IDs and orders, duplicate-pair and cutoff protection, corrected
  missing-coordinate customer status, nonfinite/empty network guards, mode/version
  keys in network exports. Positive signed customer estimates remain available.
- Accepted candidate and portfolio CSVs frozen byte-for-byte under
  `Data/planning_inputs/accepted_v1`. New results write to `outputs/planning`, not
  historical input/output filenames. Input SHA-256 hashes are recorded in metadata.
- Historical selection scores were removed from the current candidate evaluation
  table and explicitly renamed `historical_*` on exported portfolio copies. No
  reselection was performed. Greedy research helpers remain callable for a future
  explicit selection exercise; current sites are not claimed optimal under new inputs.
- Unreachable legacy runner body removed. Public legacy calculation helpers remain
  for compatibility and their existing tests; they are not used by fixed planning.
- Database/geographic preparation scripts now have guarded entry points. Removed
  the duplicate DMS function and unnecessary ClickHouse import in distance preparation;
  configured external paths and database credentials through `project_config.py`.
  Existing SQL definitions, metric formulas and historical exposure conventions remain.
- National/regional modeling commands default to cumulative-only refreshes in new
  output folders; `--full-diagnostics` retains original suites and outputs. No
  regression coefficients were refitted as part of this cleanup.
- Declared matplotlib and separated modeling/extraction requirements; Python 3.11+
  CI matrix, explicit test directory and environment exclusions. Quoted the container
  launch filename and removed CORS/XSRF disabling overrides. Package publishing is
  manual and requires packaging metadata; it no longer triggers on releases.
- Removed saved password/username fields from ignored local IDE connection settings.
  This did not change database credentials or permissions. The owner should review
  prior sharing/history independently; no credential values are in reports or source.

## Numerical comparison

For an explicit as-of date of **2026-10-15**, the new completed-calendar-month rule
selects **September 2026** rather than the in-progress October month. This does not
prove September's extraction is complete; it additionally passes the 95% coverage rule.

| Sales indicator | Previous October baseline | September baseline |
|---|---:|---:|
| 180-store APSD-weighted effect | -10.573603% | -10.201714% |
| 400-store APSD-weighted effect | -17.725012% | -17.290410% |
| Valid-coordinate weighted population | 651 | 650 |
| Existing stores exposed, 180 | 346 | 346 |
| Existing stores exposed, 400 | 438 | 438 |

Sales results are identical between raw/negative-only modes because all supplied
sales coefficients are negative. Customer modes differ by the supplied +0.0042
Outskirts coefficient. Missing-coordinate customer rows now remain unavailable.
The accepted 180/400 selections were not changed. The generated
`previous_result_comparison.csv` contains the mode-specific comparison.

## Validation and retained limits

Calculation tests cover completed-month baselines, duplicate rejection, all distance
boundaries, enforced cutoff, multiple exposures, zero exposure, missing estimates,
coordinate exclusions for both outcomes, weighted identity, fixed signed/customer
mode distinction and import safety. Frozen input bytes are checked against original
accepted files. Streamlit smoke tests cover four pages, customer/raw switching,
400-store selection and a literal punctuation search.

The completed suite passed **31 tests**, including the Streamlit integration test.
All 18 present first-party Python files parse successfully. Original frozen portfolio
and candidate files compare byte-for-byte with the historical accepted inputs.
Network export keys now distinguish scenario and mode, and the two weighted-effect
formulas agree within floating-point tolerance. Timing improvements are not claimed
without a controlled before/after benchmark; the directly removed work is the
unconditional historical dataset loading and unrequested offline diagnostic fits.
An additional full evaluation with the same frozen inputs and as-of date reproduced
all **14 planning CSV files byte-for-byte**.

No claim of global portfolio optimality, causal identification, site profitability,
new validated forecasting model, or revised regression uncertainty is introduced.
Historic lower-inclusive estimation bands remain distinct from the approved
upper-inclusive planning bands. Full database extraction and model-suite runs were
not needed to verify this cleanup and would require their configured external inputs.
No raw or historical output files were removed. Automatic cleanup of optional
report artifacts was deliberately not performed because manual reporting consumers
have not been established.
