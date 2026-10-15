# Python Project Audit

Audit date: 2026-10-15 (Asia/Irkutsk). Project root: `C:\MO files\Sales Income Prediction`.

Implementation status: the user subsequently approved this audit. The original
findings below are retained as the review snapshot; see `IMPLEMENTATION_NOTES.md`
for implemented changes, preserved artifacts, verification and numerical differences.

Scope: all 14 first-party Python files currently present (10 application scripts/modules and 4 test modules). The bundled `.conda` interpreter and third-party packages are dependencies, not project source to refactor. CSV/Excel/Parquet outputs and configuration were inspected only to trace Python behavior and reproducibility. No source code, raw data, selected locations, coefficients, or analytical outputs were changed for this audit. Existing staged changes, deleted tracked files, and untracked analytical artifacts predate the audit and must be preserved.

## A. Executive Summary

The project contains a usable analytical pipeline and correctly implements the central log-additive planning calculation with the supplied fixed coefficients. Its main maintenance problem is competing implementations and sources of truth: historical econometric estimation, an older event-study cumulative calculator, the current fixed-coefficient scenario, and a separate demand/revenue expansion simulation coexist behind a very large dashboard.

The current fixed-coefficient results are evaluations of previously selected sites. They are not evidence that those same sites minimize loss under the current coefficients. The saved selection scores are inherited from an earlier calculation. This distinction needs to be prominent before partners use a portfolio ranking.

The five most important improvements, subject to approval:

1. **Make the coefficient source and scenario identity explicit everywhere.** Preserve signed saved cumulative regression estimates and the exact supplied planning coefficients as separate sources. Do not attach inference from unrounded estimates to fixed rounded inputs without a verified provenance link. Every result needs scenario, calculation mode, baseline period, and coefficient version.
2. **Freeze portfolio inputs separately from generated evaluation outputs.** Preserve the current 180 IDs and coordinates, mark existing scores as historical, and evaluate them reproducibly. Any reselection under fixed coefficients requires separate approval and a comparison, not overwriting the accepted portfolio.
3. **Repair reliability issues before simplifying the UI.** The cumulative dashboard currently fails in this environment because gradient formatting requires an undeclared dependency. Other confirmed issues include missing-coordinate customer estimates being labeled zero exposure, missing calculation-mode keys in network exports, and incomplete validation of saved portfolios.
4. **Consolidate to four business pages and load data after page selection.** The dashboard is 6,249 lines with 12 navigation choices. Existing charts can be reused; detailed dynamics, seasonality, capacity, and robustness can move to optional evidence sections rather than separate main pages.
5. **Separate active code from legacy paths and complete the runtime configuration.** Remove unreachable code only after approval, centralize a small amount of shared calculation/configuration logic, correct launch configuration, and distinguish dashboard versus modeling/extraction dependencies. Review locally stored IDE credentials with the owner without exposing values.

### Verification and limits

- All 14 project Python files were inventoried and parsed; source functions, imports, entry points, model definitions, dashboard branches, and output producers/consumers were inspected.
- `& '.\.conda\python.exe' -B -m unittest discover -s tests -v`: **23 tests passed**, test-run time reported by unittest **0.179 seconds**. Passing tests do not cover the complete fixed-scenario pipeline or dashboard.
- Streamlit `AppTest`: default overview completed without an application exception, **3.46 seconds** in this single local run. Switching to cumulative impact failed after **1.23 seconds** with missing `matplotlib` during Styler gradient rendering. These are one-run observations, not a performance benchmark or browser map validation. Streamlit temporary-directory cleanup also encountered a sandbox permission error at process exit; that is separate from the application dependency failure.
- No extraction/database queries, regression fitting, portfolio reselection, or analytical export commands were executed. Existing saved outputs were checked read-only.
- No precise total-model-runtime or memory improvement percentage is claimed. Model runtimes were not profiled because doing so would execute expensive estimation and potentially overwrite results.

## B. Python File Inventory

Line counts include comments and blank lines; size alone is not a reason to remove a file. All current first-party Python files are listed below.

| File (lines) | Main purpose | Inputs | Outputs | Project dependencies / execution | Business relevance | Recommendation |
|---|---|---|---|---|---|---|
| `01 Cleaning StoreMaster.py` (287) | Extract monthly sales, transaction-derived customer equivalents and active days; join store metadata; calculate APSD | Two ClickHouse queries; externally located StoreMaster workbook and password file | `store_master_data.xlsx` | Manual script; top-level database/file operations on import; no project imports | High | **Keep, simplify**: guarded entry point, configuration, validated joins, reconcile active-day definition |
| `02 Store distances.py` (497) | Clean coordinates, compute historical existing-to-later-opening distances within 1.5 km and nearest diagnostic | External StoreMaster workbook | `store_to_store_distance.xlsx`, `nearest_new_store.xlsx` | Manual; top-level execution; redundant ClickHouse import but no database query | High | **Keep, simplify**: remove duplicate coordinate parser, guard entry point, root-relative outputs |
| `02 gs25.py` (156) | Extract competitor coordinates from KMZ layers | `Data/raw/GS25 260903 7.kmz` | `Data/cleaned/GS25/GS25_store_coordinates.csv` | Manual GeoPandas/GDAL/LIBKML side branch; top-level execution | Low for four current objectives | **Investigate / optional**: no active first-party Python consumer found; retain until manual use confirmed |
| `03 Modelling.py` (1,208) | National sales TWFE cumulative and event-study analyses, support and overlap diagnostics | Master and historical distance workbooks | Model-4 directory panel, coefficients, model comparisons, diagnostics and PNGs | Manual offline; no project imports; guarded main | High for evidence, supporting for planning | **Keep, simplify**: B cumulative primary; other fits optional diagnostics; shared panel construction candidate |
| `04 Regional Cannibalization.py` (2,301) | National/regional sales and customers cumulative, dynamic, seasonal, established-store and robustness models | Same master/distance workbooks | Regional panel, coefficient CSVs, statuses, evidence tables, metadata, plots | Manual offline; standalone guarded main; no dependency on running 03 | High | **Keep, simplify**: cumulative main run; named optional diagnostic groups |
| `05 Cannibalization Dashboard.py` (6,249) | All interactive business/evidence/statistics/expansion/capacity views | Regional outputs, master/distance workbooks, saved cumulative and expansion CSVs | UI and CSV downloads; no regression fitting | `streamlit run "05 Cannibalization Dashboard.py"`; imports `cannibalization_simulator` | High | **Keep, consolidate**: four main pages; optional supporting evidence |
| `cannibalization_simulator.py` (326) | Pure cumulative log prediction, minimum-APSD capacity and bounded allocation search | Cumulative coefficient DataFrame and user-entered counts/baselines | Typed in-memory results | Imported by dashboard and tests; no file IO | Medium | **Keep core; investigate optional capacity UI**. Capacity maximizes counts under a per-store floor; it is not GPS portfolio selection |
| `cumulative_cannibalization.py` (949) | Public entry point plus legacy Model-D/p0 cumulative helpers and workbook candidate preparation | Active entry delegates; legacy helpers accept panels, GPS and national/regional event-study coefficients | Active entry writes through regional module; old export block is unreachable | Manual entry; imports regional implementation inside `run_analysis`; helpers imported by regional module/tests | High entry point; Low legacy body | **Simplify / merge carefully**: preserve callable compatibility until tests/consumers migrate |
| `regional_cumulative_impact.py` (685) | Actual fixed-coefficient, saved-portfolio planning evaluation; raw and negative-only modes | Model-4 panel, master workbook, saved candidates and 180/400 portfolios; fixed tables in source; also checks unused GPS/regional-estimate files | 16 cumulative CSV/JSON outputs in Model-4 directory | Via cumulative entry or own guarded main; imports helpers from cumulative module | High | **Keep, repair and simplify**: provenance, immutable inputs, mode keys, validation, stale score handling |
| `expansion_simulation.py` (2,487) | Ridge opening-month demand models and conditional 2027 monthly network simulation; separate revenue-based greedy portfolios | Master, historical distances, GPS workbook, regional panel and monthly event-study coefficients | 21 expansion CSV/JSON/HTML artifacts | Manual CLI; NumPy ridge, pandas and optional Plotly; imported by its tests | Medium, conditional on demand validation | **Investigate / optional branch**: keep reproducible artifacts; hide from primary fixed-coefficient planning until reliability accepted |
| `tests/test_cannibalization_simulator.py` (135) | Cumulative capacity/prediction, missing estimates and interval handling tests | Synthetic data | Test assertions | unittest; imports simulator | High | **Keep** |
| `tests/test_cumulative_cannibalization.py` (239) | Legacy Model-D and regional-p0 readers, deduplication and aggregation tests | Synthetic data | Test assertions | unittest; imports legacy cumulative helpers | Medium | **Keep then migrate**: currently exercises a different calculator than production fixed evaluation |
| `tests/test_expansion_simulation.py` (94) | Geodesic units, historical band convention, schedule allocation and pair distance retention | Synthetic data | Test assertions | unittest; imports expansion | Medium | **Keep if branch retained**; test records an intentional historical convention differing from planning |
| `tests/test_fixed_cumulative.py` (24) | Exact fixed coefficients, user example, planning band boundaries, signed customer coefficient | Synthetic arrays and fixed tables | Test assertions | unittest; imports regional cumulative module | High | **Keep and extend**: currently 3 small tests, not pipeline regression coverage |

No present first-party Python file is wholly unused merely because it has no import: several are manual entry points. No standalone temporary/copy script remains in the working tree. Git reports previously deleted `03 Modelling B analytic.py`, its copy, and the old `dashboard/` Python files; do not restore, remove, or classify those deleted versions without checking the user's existing work. `.conda` should be excluded from first-party inventory, linting, and version control scope.

## C. Current Workflow

The actual workflow is branched, rather than one sequential dependency chain:

```text
ClickHouse sales + external StoreMaster.xlsx
  -> 01 Cleaning StoreMaster.py -> store_master_data.xlsx
External StoreMaster.xlsx
  -> 02 Store distances.py -> historical pair distances + nearest diagnostic

Master + historical pair distances
  -> 03 Modelling.py -> national sales panel / B cumulative / C,D diagnostics
  -> 04 Regional Cannibalization.py -> regional panel / sales + customer evidence

03 panel + master + saved candidate table + saved 180/400 portfolio IDs
  + exact fixed planning coefficient dictionaries
  -> cumulative_cannibalization.run_analysis()
  -> regional_cumulative_impact.run_regional_analysis()
  -> raw / negative-only store, pair, regional and network outputs

Master + regional panel + GPS + regional monthly event-study estimates
  -> expansion_simulation.py -> demand forecasts / different portfolios / 2027 revenue scenarios

Regional evidence + saved planning outputs + optional expansion outputs
  -> 05 Cannibalization Dashboard.py

GS25 KMZ -> 02 gs25.py -> competitor coordinates (no active Python consumer found)
```

03 and 04 independently load and prepare the same source workbooks; 04 does not import 03. Current fixed planning still depends on the **03 panel**, even though it no longer uses Model-D coefficients. The national modeling stage currently also fits C/D before its normal completion; the fixed calculator should eventually be able to consume a validated common prepared panel without requiring that suite.

Most jobs are manual; no Python scheduling/orchestration entry point was found. GitHub workflows run CI and attempt package publishing, rather than executing the production data pipeline on a schedule.

## D. Unnecessary or Duplicate Work

| Repeated work | Evidence | Recommendation and constraint |
|---|---|---|
| Coordinate parsing twice in one script | `02 Store distances.py` defines `dms_to_decimal` twice and applies conversion in successive sections | One tested conversion pass; verify signed DMS handling before broadening accepted formats |
| Cleaning/ID/date/schema checks and panel preparation | 03, 04, dashboard preparation, expansion and both cumulative modules | Share validated primitives first; do not merge panels blindly because samples and exposure definitions differ |
| Haversine distances | Historical pair builder; legacy cumulative matrix; fixed cumulative matrix; expansion candidate-to-existing/candidate-to-candidate matrices; per-location demand features | One reusable distance primitive plus separate historical and proposed-site artifacts keyed by coordinates; candidate-pair distances are a different required geometry |
| Distance classification and regional mapping | Historical lower-inclusive bands vs fixed planning upper-inclusive bands; repeated mappings in dashboard and cumulative | Central policy with explicit historical/planning convention. Preserve historic fit definition until approved refit |
| Log-effect conversion / weighted summaries | Simulator, both cumulative calculators, expansion and dashboard | Common tested proportional-effect utilities, with explicit outcome scale and mode |
| Estimation overlap | 03 B national sales vs 04 All Regions sales cumulative | Retain one canonical regional evidence table, but keep 03's earlier sample/results for validation; estimates are not numerically identical because samples differ |
| Panel written twice in 03 | Main writes panel before estimation and save routine writes it again | Keep early diagnostic save if needed on failure; avoid identical second write when unchanged |
| Legacy cumulative runner | `cumulative_cannibalization.py:794–799` immediately returns the new runner; remainder of old runner through old exports cannot run | Remove unreachable body after compatibility review; no numerical change expected from removing unreachable statements |
| Imports no longer needed by production | Regional module imports regional-p0 reader and candidate preparer although active run uses fixed tables and saved candidates | Remove after any helper migration; avoid a two-way module dependency |
| Fixed run treats generated outputs as inputs | Reads saved candidates and portfolios, then writes those same filenames; cannot bootstrap from raw inputs after those files disappear | Freeze versioned portfolio input and evaluation output separately; retain current IDs/coordinates |
| Sensitivity helper exists but production emits placeholder | `_sensitivity` remains defined; active run writes `not_recomputed` and an effectively empty portfolio CSV | Hide tab when unavailable; optional sensitivity should evaluate fixed accepted sites if later requested, not silently reselect |
| Repeated dashboard presentation | Overview/regional/distance pages; statistics/regional-distance statistics; observed-store exploration vs cumulative detail | Consolidate presentation around four questions; keep observed before/after statistics labeled descriptive |

The code contains similar implementations, rather than a simple exact-text duplicate block that can safely be deleted wholesale. Historical opening-pair distances cannot replace proposed-candidate distances just because both are measured in meters.

## E. Econometric Model Review

### Model inventory and proposed role

All log models below are estimated with store and calendar-month fixed effects via iterative alternating demeaning followed by OLS. Covariance is clustered by existing store. Repeated nearby openings enter as counts, not a nearest-store indicator; overlapping openings are retained at store-month grain. Event studies omit event month -1. No coefficients should be dropped merely because they are positive or insignificant.

| Source / model | Outcome and regressors | Sample / saved support | Outputs and dashboard consumer | Proposed role |
|---|---|---|---|---|
| 03 A | log income; fixed effects without exposure terms | 22,246 valid store-months, 703 stores, 52 months | `model_comparison.csv` | Supporting fit benchmark |
| 03 B | log income; four cumulative opened-store distance counts | Same 22,246 / 703; within R² 0.022856 | `all_model_coefficients.csv`; not the fixed planning input | Primary national historical evidence, or supporting cross-check once 04 standardized |
| 03 C | log income; pooled repeated event counts -6 through +6, reference -1 | 12 exposure terms; same sample | National event plot/comparison | Supporting pretrend/dynamics diagnostic |
| 03 D | log income; distance-by-event counts -6 through +6, reference -1 | 48 terms; same sample | Model4 coefficients/plots; legacy helper reads p0 | Supporting only; not fixed planning source |
| 03 D_alt_0_250 | Same as D with first two distance bands combined | 36 terms; same sample | General coefficients/comparison | Optional sparse-band robustness |
| 04 cumulative, 5 regions × 2 outcomes | log income or log customers; four cumulative opened counts | All Regions 22,213 / 698; Center 4,122 / 110; Outskirts 8,032 / 213; Outer 7,529 / 248; Rural 2,530 / 127. Counts are the same for the two outcomes in saved main cumulative results | Regional coefficient tables, comparison/status files; overview, regional/distance and capacity views | **Primary estimated evidence** |
| 04 monthly event study | log outcome; distance × monthly event counts -12 through +24, reference -1 | Outcome/region support recorded in statuses; unsupported all-zero terms omitted; rank deficiency fails explicitly | Regional coefficient tables, pretrend table; long-term views and expansion simulation | Supporting diagnostics; required by optional expansion branch |
| 04 binned event study | log outcome; distance × bins -12 to -7, -6 to -3, -2, 0, +1–3, +4–6, +7–12, +13–18, +19–24, +25 onward | Reference -1 omitted; support tables record long-horizon coverage | Same coefficient tables; long-term pages; legacy p0 helper | Supporting compact dynamics; p0 must not replace cumulative planning |
| 04 seasonal | log outcome; cumulative 1 km exposure × month-of-year, with regional fits | Recorded in seasonal/status tables | Seasonal coefficients/observed means, seasonality page | Optional; no validated holiday calendar, so no holiday causal claim |
| 04 established stores | log outcome; repeated binned openings among stores eligible at 24+ months tenure | Eligibility/follow-up tables; model-specific sample | Established-store effects, support, page | Optional tenure/selection diagnostic |
| 04 alternative distance | log outcome; combine first two cumulative bands | Model-specific status | Robustness results / general coefficient files | Supporting sparse-band check |
| 04 single-opening stores | log outcome; cumulative counts restricted to at most one opening | Different sample; failures recorded | Robustness/status | Supporting sensitivity with explicit selection limitation |
| 04 asinh robustness | asinh income/customers; cumulative counts | Preserves observed zero/nonpositive outcomes where usable | Robustness/general coefficient files | Supporting transform sensitivity; cannot use exp(beta)-1 interpretation |
| Fixed scenario | Supplied rounded coefficients; log exposure-count interpretation applied to selected GPS pairs | No estimation/inference performed | Cumulative artifacts and planning page | **Primary planning source**, separate from estimated evidence |
| Expansion demand ridge | Log opening-month APSD predicted from neighboring-store features and UB/Rural context | Time holdout before 2025 vs 2025 onward; final fit uses available valid training rows | Demand artifacts/predictions/validation, Expansion Simulation page | Optional forecast research; not cannibalization regression |

### Findings about correctness and interpretation

1. **Functional form is verified.** 03 constructs `log_APSD_income`; 04 constructs both log outcomes and `cum_openings_{band}_opened` from counts of eligible opened pairs. Summing count × beta, then using `100 * expm1(total_log)`, is the correct conditional proportional change for the additive log specification. The current fixed calculator follows it. Income and customer coefficients are separate; negative-only caps each pair beta before summation, preserving raw mode separately.
2. **Region definitions:** Core = Center (0–1.5 km from Sukhbaatar Square); Inner City = Outskirts (1.5–3.5 km); Outer City/Far = Far / Outer Areas (3.5–7 km / >7 km); UB Rural overrides ring. These definitions are supplied metadata, not recomputed candidate-region forecasts. Effects use the existing receiver store's region. Outskirts is not Outer Areas.
3. **Fixed versus saved coefficients:** supplied regional fixed values agree with saved cumulative estimates within approximately 0.00005 where supplied (rounding). Saved 04 additionally has an Outskirts sales estimate, but the explicitly requested fixed scenario intentionally uses All Regions sales there. Do not automatically replace this approved fallback. Fixed tables contain no standard errors or covariance; report inference as unavailable for the fixed version unless linked to a verified matching estimated result.
4. **Boundary conventions differ:** historical 03/04 and expansion use [0,100), [100,250), [250,500), [500,1000), so exactly 1000 m is outside. Current fixed planning uses [0,100], (100,250], (250,500], (500,1000]. This is documented user policy, not permission to relabel historical estimates. Test and show the distinction; quantifying affected exact-boundary rows should precede any historical refit.
5. **Causal interpretation is limited.** All 20 saved main regional monthly/binned pretrend tests reject at 5%; national C p≈1.98e-10 and D p≈1.58e-7 also reject. Shared local demand trends, anticipatory openings and endogenous placement remain plausible. Removing event-study pages does not remove this limitation. Preserve diagnostics offline and a short business caveat.
6. **Store clustering does not handle arbitrary shared-opening dependence across stores.** Neither OLS call supplies absorbed fixed-effect degrees of freedom explicitly; the finite-sample correction and normal/t inference should be validated against an established absorbed-FE estimator on the same sample before claiming SEs are fully verified. Do not silently modify coefficients or intervals during cleanup.
7. **Repeated-event assumptions:** cumulative models impose a common marginal log effect per opening within a band, constant over time and additive over exposures. This handles counts mechanically but is a substantive extrapolation for dense portfolios. Historical dynamic analyses have finite windows/reference restrictions; omitted outside-window effects can contaminate dynamic interpretation. Established-store restrictions alter the estimand/sample and may compound repeated-event eligibility selection. Keep these concerns distinct from programming bugs.
8. **Zero/nonpositive APSD and population:** log models exclude invalid outcomes; fixed baseline uses positive income. Current saved baseline is October 2026, 652 stores, with 651 valid-coordinate network estimates. Regional evidence is 698 stores over history. These populations are not interchangeable. Coverage alone does not prove October is a completed month as of October 15.
9. **Log retransformation:** baseline × exp(log scenario change) is a proportional conditional scenario indicator, not a new forecast of unconditional expected currency revenue. No annualization is needed. Retaining signed positive estimates is correct in raw mode.
10. **Demand branch evidence is mixed:** saved sales holdout n=230, R²≈0.113 overall; UB subgroup R²≈-0.201. This supports keeping it as explicitly conditional research, not a proven profitability model. Holdout intervals use residual dispersion and clipping; they are not validated site-specific prediction intervals. No cost/profit model was found.

Stopping routine C/D/seasonal fits would reduce fresh diagnostics, not change already saved cumulative estimates. Archiving code/output would reduce reproducibility unless a runnable archived command and inputs are retained. Keep historical results; defer changes to estimation samples/inference for a separate approved analytical change.

## F. Dashboard Simplification Proposal

| Main page | Reuse existing functionality | Essential content | Consolidation |
|---|---|---|---|
| Cannibalization Overview | Executive overview + fixed cumulative summary + regional/distance charts | Selected scenario and coefficient source; weighted sales impact, exposed-store count, >10% decline count; one distance chart and one regional comparison with sales/customer selector | Merge Executive Overview, Regional Comparison, Distance Analysis presentation |
| Regression Results | Saved cumulative rows, intervals, diagnostics downloads | Region/outcome/band table; signed beta, expm1 interpretation, SE/CI/p/sample counts; optional evidence expander | Put long-term/event studies, seasonal and established-store diagnostics here as optional evidence; never relabel fixed assumptions as fitted inference |
| Planned New Store Locations | Cumulative map, candidates and scenario table | Saved 180 default; 400 optional; raw/negative-only control; coefficient/fallback badge; network impact, exposed-store table, selected candidates and contribution download | Name future reselected portfolio **Lowest modeled cannibalization portfolio**; current fixed saved sites labeled **Previously selected portfolio evaluated with fixed coefficients** |
| Store-Level Exploration | Existing observed trend code + cumulative detail | Search one store; observed monthly sales/customer trends; scenario APSD; all selected pairs within 1 km; distance/source/coefficient table, contribution chart and local map | Merge historical store exploration with cumulative detail; distinguish observed and modeled quantities |

Optional exceptions: broad Statistics/Regional & Distance Statistics are analyst aids and can move to one expandable analyst view or separate analyst launch, retaining CSV downloads. Capacity simulator can be an optional tool under evidence or planning; it optimizes count bounds, not GPS locations. Expansion Simulation should remain separately labeled research unless approved forecast inputs/validation justify a main-page link. No code/page removal is authorized by this proposal alone.

Current UI issues to correct after approval:

- Fixed cumulative main charts, KPIs and pair detail are sales-centric even when the global outcome sidebar says customers; customer effects are mostly table fields. Make outcome controls page-specific and effective.
- Cumulative region/search filters alter tables/distribution but most KPIs remain global; most-affected card alone is filtered. Label scope consistently or recompute small filtered summaries.
- Literal store search currently uses regex-enabled `str.contains`; punctuation can act as regex or raise an error. Use literal matching.
- An all-missing filtered effect column can make `idxmin()`/`.loc` fail; guard on valid effect rows.
- Distribution label 'Below -30%' currently includes exactly -30%; boundary labels need precise, mutually exclusive meaning.
- Cumulative detail uses a JSON block and does not integrate observed trends/customer contribution charts, despite equivalent observed-store capabilities elsewhere.
- Streamlit executes all ordinary tab bodies. Putting heavy map/table code in a tab does not make it lazy.

## G. Performance Improvements

### Observed evidence

- Dashboard main unconditionally loads approximately 12 regional CSVs plus regional panel and historical pair Excel **before** page dispatch. Root loaders use cache without file modification keys; cumulative/expansion loaders do include modification times. Cold startup pays for unrelated data and warm sessions can retain stale evidence after offline updates.
- Cumulative contributions CSV is **7,968,906 bytes** and loaded in full for both modes/scenarios before filtering to one store. All candidate existing distances in expansion are **16,344,893 bytes**, pair impacts **35,624,275 bytes**, and candidate-pair distances **5,821,864 bytes**. These are measurable file sizes, not RAM measurements.
- Current geometry is about 652 baseline stores × 459 candidates (~299,000 distances), well within vectorized NumPy scale. There is no evidence that a spatial database/new geospatial dependency is necessary for this task.
- No model fitting runs inside the dashboard. Retain this separation.

### Recommended changes and expected benefit

| Change | Benefit | Evidence level / risk |
|---|---|---|
| Dispatch first, load only page inputs | Lower cold-start IO, fewer mandatory unrelated files | Directly supported by main flow; speedup must be measured after change |
| Add file modification/version keys to common cached loaders | Correct refresh after new evidence; avoids manual cache clearing | Reliability improvement; negligible numerical impact |
| Load/precompute selected-store pair detail on demand; use Parquet for large analytical pairs | Less CSV parsing and string-object memory; column selection | File-size evidence only; no claimed percentage saving |
| Reuse a coordinate-keyed candidate distance artifact across raw/negative-only evaluations | Avoid repeated geometry when portfolios/modes change | Safe if keyed by full store/candidate ID-coordinate set and distance policy |
| Make diagnostics optional in offline model command | Fewer unnecessary full regressions per refresh | 04 primary loop alone requests 30 main region/outcome/model fits before robustness/seasonal/established work; exact runtime unmeasured |
| Avoid repeated Excel reads and repeated panel saves in offline stages | Lower IO and clearer execution | Retain source workbooks and failure diagnostics |
| Keep current vectorized greedy method for any approved future reselection initially | Avoid unvalidated algorithm change | At 459 candidates, first resolve objective/provenance before algorithm replacement |

Do not convert all outputs to new formats immediately. Keep lightweight CSV summaries/downloads; add Parquet only for measured heavy artifacts already supported by pyarrow. Profile stages before allocating more optimization work.

## H. Code Quality and Reliability Issues

### Confirmed defects / inconsistencies (report first; no corrections made)

| Severity | Evidence | Recommended correction | Could numerical results change? |
|---|---|---|---|
| High | Cumulative page `Styler.background_gradient`, dashboard ~3602; smoke test raises missing matplotlib | Declare/install required visualization dependency or use dependency-free formatting; modeling also needs matplotlib/statsmodels | No, presentation/environment |
| High | Saved selected-180 final `portfolio_weighted_network_loss_pct` = **0.0**, while current fixed sales weighted loss magnitude = **10.573603%** | Preserve selection provenance and scores as historical; recompute evaluation scores under current source if approved; do not claim current optimality | Evaluation labels/scores change; selected sites stay frozen |
| High | Baseline `prepare_consistent_baseline` ~105 chooses month by 95% of peak positive-row coverage; October has 652 rows vs September 653 | Establish completed-month/extraction-as-of rule; test duplicate keys before counting; report excluded/not-observed stores | Yes, weights/population/rankings can change; approval needed |
| High | `aggregate_portfolio` checks coordinates for income but customer status only checks positive customer baseline and pair count | Missing-coordinate stores must have unavailable customer effects rather than valid zero exposure; 4 saved rows (one store × two scenarios × two modes) currently show this mismatch | Customer coverage/status and any customer aggregates can change |
| High | `cumulative_cannibalization_network_indicators.csv` selection omits `calculation_mode`, despite writing raw and negative-only rows | Include mode and coefficient version in all export keys. File currently contains indistinguishable duplicate 180/400 rows | No values need change; identity/schema changes |
| High | Fixed run reads/writes same candidate/portfolio artifacts; requires both 180 and 400 even for 180 | Immutable portfolio inputs with explicit validation, input hashes; allow 180-only evaluation independently | No if frozen inputs preserved |
| Medium | Fixed runner validates duplicate selected IDs and membership but not exact size, candidate-table coordinate/ID uniqueness, finite coefficients, order uniqueness or all join multiplicities | Add input contract and integration tests; current 180 has 180 unique valid IDs and saved candidate table has 459 unique coordinate pairs | May turn silent bad future inputs into errors |
| Medium | Fixed `aggregate_portfolio(...cutoff_m)` labels cutoff but does not filter pair input to cutoff internally | Enforce cutoff at public boundary or require/assert prefiltered input; direct caller can pass distant pairs | Yes for invalid caller inputs |
| Medium | Fixed lookup enumerates known table regions; source labels say All Regions fallback for other regions while actual lookup can remain missing | Central explicit fallback policy; unmapped geography should be separately flagged, not misleadingly labeled | Potentially; missing vs fallback decisions need approval |
| Medium | `number_of_contributing_candidate_stores` counts exposures even when coefficients are missing | Separate exposed, supported and nonzero-contributing counts | Labels/count interpretation |
| Medium | Legacy cumulative helper defaults missing region/band to All Regions and excludes exactly 1000 m; active fixed helper has different policy | Limit legacy helper to documented compatibility; use one production planning engine | Yes if callers are migrated without policy tests |
| Medium | Fixed runner still checks GPS and regional coefficient files without using their contents | Validate real dependencies; remove misleading prerequisites | No effect on valid calculations |
| Medium | Unconditional eligibility=True and stale marginal table exports; validation strings assert 180/400 while actual checks test uniqueness only | Count/membership/coordinate/version checks; distinguish inherited scores from current evaluation | Usually validation/schema; ranking claim materially changes |
| Medium | Empty sensitivity portfolio export is two bytes; fixed mode writes placeholder summary | Omit unsupported artifacts or emit explicit status schema; hide empty UI | No primary result change |
| Medium | Weighted calculation has no explicit zero-denominator guard; baseline uses positive test without finite check; duplicate store months dropped silently | Reject nonfinite baselines, validate uniqueness, handle empty network explicitly | Potentially on invalid inputs |
| Medium | `.drop_duplicates(...keep='last')` resolves inconsistent metadata silently; aggregate joins lack explicit validate in places | Fail/report duplicates before joins; deterministic metadata policy | Potentially on inconsistent input |
| Low | Stale docstrings mention p0/no fallback; mixed fixed and legacy result directories; coarse type annotations | Update documentation/source metadata and keep output compatibility map | No |

### Data-preparation and econometric concerns, separate from cleanup

- Active-day comment says non-zero-sales day, but `query_active_days` counts distinct dates with qualifying rows and no explicit nonzero daily-sales condition. Decide definition with owner before changing APSD. Query logic also scans the sales source separately from receipt totals; combine only after proving equivalent day/return/product filtering.
- Customers are `Transaction_Count - 2 * Returned_Transaction_Count`, not distinct people. Receipt counts are distinct receipt identifiers at monthly/store grain; investigate whether identifiers are unique across dates. Do not alter the established metric in a refactor.
- Master join in cleaning does not validate uniqueness; duplicate metadata could multiply monthly rows. Both early scripts execute reads/writes on import and use current-working-directory output paths.
- Historical distance builder uses strictly earlier opening date and positive distance, excluding same-day openings and distinct colocated stores. These are substantive eligibility choices to document and assess, not automatically 'fix'. Its intermediate bands split 500–750/750–1000; models reclassify numeric distance into the four modeled bands.
- Finite-sample FE covariance, pretrends, identification, fixed rounded coefficient inference, and extrapolation are econometric review items in E; code cleanup must not silently change estimates.

### Paths, configuration, dependencies and security

- Search found **no exact old project-root reference** in first-party Python/config sources inspected. Most analytical modules already use `Path(__file__).resolve().parent`, so moving them is supported without replacing every path with the new absolute root.
- `01 Cleaning StoreMaster.py` and `02 Store distances.py` still reference external OneDrive Power BI metadata; cleaning also references an external password file. Those are external dependencies, not references to the former project root. Configure them explicitly; do not assume those files moved with the project.
- `02 gs25.py` and early Excel exporters use working-directory-relative paths. Use a small central root/configuration helper after approval.
- **Sensitive local configuration:** `.vscode/settings.json` contains plaintext database connection credentials. No values are reproduced here. It is ignored by current `.gitignore`; that does not prove it was never tracked/shared. Review tracking/history with the owner, move secrets to an appropriate local secret mechanism, and let the owner decide credential rotation. Do not change production permissions or credentials in this task.
- Python extraction reads a password file rather than embedding a password value; network connection/account settings remain machine-specific. No database connection was attempted in this audit.
- `requirements.txt` lacks `matplotlib`, `statsmodels`, `clickhouse-connect`, and GeoPandas/GDAL runtime needs. Split optional extraction/geospatial/modeling requirements if necessary rather than forcing every dashboard install to require KMZ drivers.
- CI includes Python 3.9, while code uses modern union/type syntax and version-sensitive pandas APIs (`include_groups`); dependency floors and interpreter matrix need an explicit supported-version decision and fresh-env tests. Not every deferred annotation alone proves a 3.9 import failure, so test actual compatibility rather than assume it.
- `.devcontainer` launch does not quote the dashboard filename with spaces and disables Streamlit CORS/XSRF protections. Correct launch quoting and review those overrides; no configuration changed.
- PyPI workflow calls a package build but repository has no observed packaging metadata. Disable/retire that workflow only after owner confirms no packaging purpose. It has no clear role in a Streamlit analytics repository.
- `.conda` is absent from the ignore patterns; lint/test discovery should exclude it explicitly. README currently provides almost no runbook.

## I. Proposed Simplified Project Structure

Prefer minimal changes around existing entry scripts:

```text
01 Cleaning StoreMaster.py          # guarded extraction/preparation
02 Store distances.py              # historical distance preparation
03 Modelling.py                    # retained national evidence / optional diagnostics
04 Regional Cannibalization.py     # canonical estimated cumulative evidence
cumulative_cannibalization.py       # thin supported CLI after legacy migration
regional_cumulative_impact.py       # fixed planning evaluation and future approved selection
cannibalization_simulator.py        # shared pure log predictions; optional capacity
project_config.py                  # one small justified root/external-path helper (proposed)
05 Cannibalization Dashboard.py     # four business pages, initially same entry filename
tests/                             # existing tests + fixed-pipeline contracts
Data/raw/                          # preserved raw GPS and other inputs
Data/planning_inputs/<version>/    # proposed immutable accepted IDs and coordinates
outputs/cannibalization_regional/   # preserved estimated evidence
outputs/cannibalization_model4/     # preserve historical results and compatibility
outputs/planning/<version>/        # optional future separation; migrate with manifest
expansion_simulation.py            # optional research branch; retained until decision
02 gs25.py                         # optional competitor preparation
```

Do not create a package hierarchy or migrate outputs immediately. First consolidate shared band/region/log/weighted logic into existing pure modules where practical. A new config helper is justified by external-path setup; a common panel artifact can replace duplicated preparation only after row/key/sample equivalence is established. Maintain a manifest recording source files, timestamps/hashes, coefficient source, policy, baseline, modes and portfolio IDs.

## J. Prioritized Improvement Plan

Implementation is **pending user approval**. 'Results change' includes labels/coverage as indicated, not permission to recompute or replace selected sites.

| Priority | Affected files | Current problem / proposed change | Expected benefit | Risk | Existing results could change? | Complexity |
|---|---|---|---|---|---|---|
| High | requirements; dashboard; model scripts | Complete runtime dependencies and fix launch quoting; test clean environment | Cumulative page opens; reproducible setup | Low | No numerical change | Small |
| High | fixed module; cumulative outputs; dashboard | Separate historical scores from fixed evaluation; freeze selected 180/coordinates; version source/mode | Prevents unsupported optimality and source claims | Medium | Scores/labels yes; sites no | Medium |
| High | fixed baseline preparation | Approve completed-month/as-of rule and excluded-store report | Reliable, comparable weights | High analytical sensitivity | Yes; explain differences before adopting | Medium |
| High | fixed aggregation/export | Correct coordinate-status asymmetry, mode key omission, finite/empty handling, cutoff assertions | Prevents false zeros and ambiguous exports | Medium | Coverage/schema, possibly customer results | Small–Medium |
| High | `.vscode` config; extraction config | Owner review of local secret storage; configure external paths without exposing values | Portable and safer local setup | Medium operational | No analytical change | Small–Medium |
| High | tests; fixed module | Add fixed-pipeline and saved-output reproducibility tests before refactor | Protects exact coefficients and accepted sites | Low | No | Medium |
| Medium | cumulative and regional modules | Remove unreachable old runner and migrate required candidate helpers | One supported calculation path | Medium compatibility | No intended differences; compare fixtures | Medium |
| Medium | dashboard | Consolidate navigation to four pages; effective outcome controls and filtered-scope labels | Clearer partner experience | Medium UI | Presentation only | Medium–Large |
| Medium | common loading/dashboard | Page-specific reads, consistent cache invalidation, demand-load details | Lower cold startup and stale-data risk | Low–Medium | Only refreshes to actual input version | Medium |
| Medium | 03/04 | Canonical cumulative evidence table, optional diagnostics command groups | Fewer repeated fits; clear evidence source | Medium | No if same fits/sample; omission affects fresh diagnostics | Medium |
| Medium | cleaning/distance/config | Guard early scripts, validate metadata joins, one coordinate parser | Safe imports and reproducible preprocessing | Medium | Possible on invalid metadata; compare explicitly | Medium |
| Medium | fixed selector/expansion | Keep current sites; evaluate objective/provenance and demand reliability before any future reselection | Meaningful portfolio labels | High if objective changes | Yes for future approved reselection only | Medium–Large |
| Low | output writers/consumers | Add column-selective Parquet for large pair artifacts after profiling; retain CSV summaries | Reduced IO/memory | Low | No intended value changes | Small–Medium |
| Low | GS25 and optional research pages | Confirm manual users before archiving optional branches | Smaller default workflow | Medium information-loss risk | No if artifacts retained | Small |
| Low | CI/README | Supported runtime, exclude environment files, replace unrelated package workflow after confirmation, runbook | Maintainability | Low | No | Small |

### Output dependency map and retention classification

Classes: **Essential** for current production/reproducibility; **Validation** for evidence; **Optional** for analyst/research/manual reporting; **Potentially obsolete** where no active Python consumer was found; **Investigate** where provenance/use is unresolved. Lack of a Python consumer never authorizes deletion.

| Output / family | Producer -> current consumer | Class and retention decision |
|---|---|---|
| `store_master_data.xlsx` | 01 -> 03, 04, dashboard, fixed and expansion | Essential; preserve |
| `store_to_store_distance.xlsx` | 02 distances -> 03, 04, dashboard, expansion | Essential; preserve historical inputs |
| `nearest_new_store.xlsx` | 02 distances -> no active Python reader found | Potentially obsolete / manual reporting investigation; never substitute for all-pair exposure |
| `Data/cleaned/GS25/GS25_store_coordinates.csv` | 02 gs25 -> no active Python reader found | Optional / investigate manual competitor use |
| Model4 panel parquet | 03 -> fixed evaluation; legacy cumulative helpers | Essential under current wiring; future common panel migration only with comparison |
| `all_model_coefficients.csv`, `model_comparison.csv`, `model4_coefficients.csv` | 03 -> legacy helpers; current fixed runner no longer consumes these estimates | Validation; Model4-only subset overlaps all-model file, retain for compatibility |
| Model4 sample/distance/event/overlap/opening summaries and two PNGs | 03 -> diagnostic/manual evidence; old unreachable runner references some | Validation / optional PNGs; no deletion now |
| Regional panel parquet | 04 -> dashboard, expansion | Essential |
| Regional sales/customer coefficients | 04 -> dashboard evidence/capacity, expansion monthly effects; fixed runner checks existence only | Essential; filter by model_type rather than merge estimates invisibly |
| Regional comparison/status/pretrend tables | 04 -> dashboard; expansion diagnostics; historical evidence | Validation, essential for reliable interpretation |
| Region mapping, regional sample and distance summaries, opening summaries, analysis metadata, data validation | 04 -> dashboard partly; mapping/provenance/manual evidence | Essential provenance + Validation |
| Long-term event support; established effects/follow-up | 04 -> Long-Term / Established pages | Validation; keep if pages consolidated |
| Seasonal effects/observed means; robustness checks | 04 -> seasonality/evidence pages or manual evidence | Validation / Optional; preservation needed if fits become opt-in |
| Regional event, distance and seasonal PNGs | 04 -> manual/reporting, dashboard uses interactive tables/charts | Optional; no active direct PNG reader found |
| Cumulative `candidates.csv`, `selected_180.csv`, `selected_400.csv` (full prefix `cumulative_cannibalization_`) | Fixed runner **reads then rewrites**; dashboard consumes candidate flags | Essential accepted portfolio inputs currently mixed with outputs; freeze before cleanup |
| Cumulative `scenario_store_results.csv`, `scenario_summary.csv`, `regional_summary.csv`, `contributions.csv` | Fixed runner -> cumulative dashboard | Essential; preserve raw and negative-only modes |
| Cumulative `pair_effects.csv` | Fixed runner -> audit/manual detail; no direct dashboard reader | Useful validation; reusable geometry/coefficient evidence |
| Cumulative `network_indicators.csv` | Fixed runner -> no active dashboard reader (dashboard uses scenario summary) | Optional summary, but mode identity needs repair |
| Cumulative wide `store_results.csv` | Fixed runner -> compatibility/manual reader potential | Investigate before removal; legacy wide result differs from new long/mode schema |
| Cumulative `candidate_marginals.csv` | Copies historical saved 400 selection scores | Investigate / historical provenance; not a current fixed-coefficient optimization result |
| Cumulative validation/data-quality reports and metadata JSON | Fixed runner -> validation tab partly; manual audit | Validation / essential provenance; extend coverage rather than remove |
| Cumulative sensitivity summary/portfolios | Fixed runner emits placeholder/empty artifact -> sensitivity tab reads summary | Potentially obsolete placeholders; hide availability until approved recomputation |
| Expansion rankings, scenario summaries, monthly network, existing impact, regional comparison, validation, sensitivity, robustness | Expansion -> Expansion Simulation page | Optional research branch; preserve if hidden |
| Expansion `selected_stores_180.csv`, `selected_stores_400.csv`, predictions | Expansion -> manual analysis; dashboard primarily reads rankings | Investigate/manual outputs; **different portfolio IDs/objective** from fixed branch |
| Expansion candidate-existing distances, candidate-pair distances, candidate-existing impacts | Expansion -> exported validation/manual evidence; rerun computes its own matrices | Optional heavy artifacts; candidate geometry can be reusable only after identity/policy reconciliation |
| Expansion assumptions/model artifacts JSON and maturation CSV | Expansion -> provenance/manual validation | Validation for optional branch |
| Expansion two HTML plots | Expansion -> manual/browser reporting | Optional; dashboard recreates interactive charts |
| `outputs/data_quality_report.csv` | Existing artifact without a clear current writer/reader located | Investigate provenance; do not delete |
| `Data/dashboard_store_month.parquet`, `Data/potential_store_exposure.parquet` | No active present Python consumer found; likely older dashboard pipeline | Potentially obsolete; verify deleted-dashboard/manual workflows first |

### Proposed acceptance checks after approval

1. Hash/freeze the existing 180 IDs and coordinates; assert unchanged through refactoring. Keep the 400 input independent and optional.
2. Integration fixture with hand-calculated multi-opening effects; exact boundaries 0/100/250/500/1000/>1000, and one candidate near a second store while distant from the first.
3. All fixed regional coefficients and the explicit Outskirts sales fallback; preserve raw +0.0042 and verify pairwise capping only in negative-only mode.
4. Unknown region/band, invalid/nonfinite baselines, duplicate store/months and pairs, invalid/missing coordinates, zero exposures, zero denominator and empty portfolio.
5. Same baseline keys/weights across modes/scenarios; manual weighted example and scenario-APSD reconstruction; separate income/customer coverage. Export key uniqueness includes mode/source/baseline/scenario.
6. Reproduce saved outputs from immutable inputs in a temporary output directory, compare values/tolerances and hashes; do not overwrite business files to run tests.
7. Dashboard smoke coverage for four pages, raw/negative-only/outcome/scenario switching, empty filters, literal searches, map data and downloads; verify table values against saved outputs.
8. Retain historical-versus-planning band convention tests; compare samples and signed coefficients before any model-stage consolidation. New uncertainty calculations need estimator validation, not merely snapshot tests.

**Approval boundary:** approve a set of recommendations before implementation. No deletion, archiving, model removal, portfolio reselection, baseline replacement, analytical correction or structural rewrite has been performed in this audit.
