# Dynamic expansion optimization v2

## Existing system and integration

The project root is now `C:\MO files\Cannibalization analyses`. Existing
`expansion_simulation.py` fits opening-month demand models offline, loads monthly
event-study coefficients for ages 0–11, assigns a fixed weighted monthly calendar,
and independently selects candidates through `_greedy_portfolio`. It considers
existing-store effects and competition between selected new stores. That code and
its original outputs remain unchanged.

The new system reuses its saved demand predictions, saved maturation profile,
candidate eligibility and geodesic-distance function. Expensive historical model
fitting never runs from the new dashboard. Existing four business pages remain;
**Expansion Optimization** is an additional page.

Files created: `expansion_optimization.py` (constraints, shared evaluator, joint
greedy/local search, reconciliation/outputs), `expansion_optimization_inputs.py`
(saved-input adapter, original-greedy bridge, benchmark/CLI),
`expansion_optimization_page.py` (interactive page), and two optimization test
modules. `business_dashboard.py` adds routing; its existing smoke test expects
five pages. `.gitignore` excludes the new sensitive generated optimization data.
`project_config.py` now imports the optional ClickHouse client only when a database
connection is requested, so offline analysis and tests do not require that client.
Existing connection settings and credentials were not changed.

## Objective and interpretation

Sales objective maximizes horizon incremental network sales in MNT:

`gross new-store sales + new-to-new change + existing-store change`.

New-store sales after new-to-new competition equal the first two components.
Do not add competition losses again to that net new-store total. Ordinary planning
APSD indicators remain separate; this branch explicitly multiplies APSD forecasts
by forecast active days to form horizon-specific sales.

NPV is available only when the user supplies margin, monthly new-store operating
cost, initial investment, and annual discount rate. The same supplied margin is
applied to incremental existing and new sales; investment occurs in the opening
month, operating costs run while open, and all flows are discounted by calendar
month. No costs, terminal value, tax benefits, or financing assumptions are invented.
The finite 12/24-month NPV is conditional on these explicit assumptions and horizon.

The current saved demand estimates have limited out-of-sample validation; they
are neither guaranteed site forecasts nor evidence of profitability. A sales
objective can favor the maximum allowed count. That is a conditional constrained
sales result, not a profit-optimal expansion recommendation.

## Coefficient sources and horizon

Default: **exact supplied cumulative income coefficients**, with explicit
All Regions sales fallback for City Outskirts. These log coefficients persist
after an opening because the regressors count cumulatively opened stores. This
is a distinct specification from a month-0 event study. Raw mode preserves all
signs; negative-only mode caps each pair's positive log contribution before summing.

Optional: **saved monthly event study** only. The loader indexes the actual
post-opening event age and uses available ages, including longer horizons if
present. When a required age exceeds the last supplied estimate, the user must
choose `error`, `hold_last`, or `zero`. Interior gaps remain errors. No month-11
effect is carried forward implicitly. Compare separate runs with different tail
assumptions; the scenario configuration and input hashes distinguish them.

**Current input issue:** the regional sales/customer coefficient CSVs contain
literal Git merge-conflict markers. The new loader rejects these files and does
not choose a branch or modify them. The fixed cumulative source works independently.
Event-study scenarios require a verified, clean input file before they can run.
The regional cumulative evidence refreshed elsewhere is not substituted for
event-study coefficients.

Effects use the existing receiver's established Center/Outskirts/Outer/Rural
classification. Proposed-site forecasts supply only UB/Rural demand geography;
their new-to-new receiving coefficients retain the original expansion **All Regions**
policy, explicitly recorded, rather than inventing city sub-regions. Regional
monthly operational capacity uses the available UB/Rural candidate capacity areas;
an unknown capacity label raises an error.

Planning distance intervals are [0,100], (100,250], (250,500], (500,1000] meters;
all more-distant direct contributions are zero. Every selected unique source
contributes after it opens. New-store receiver sales begin only at their own
opening. Prior planned sources may affect a later new receiver because the saved
demand forecasts do not include future planned expansion. Each source/receiver
pair enters once; self effects are zero. Default openings are the first of each
month. API opening days 1–28 prorate gross sales and the source's opening-month log
effect by the fraction active; this is an explicit within-month assumption.

Maturation ages 0–11 reuse the saved profile. For older ages the user explicitly
chooses `hold_last` (default) or `one`; this is a sensitivity assumption, not a
newly estimated profile. The existing baseline uses stores observed in the latest
completed month, their last 12 observed positive APSD months and pooled historical
calendar-month seasonality. Forecast active days use completed-history monthly
medians, bounded by actual calendar lengths. Missing coordinates/dates and source
hashes are recorded; required unsupported forecasts or regions fail visibly.

## Optimization method

1. The original greedy algorithm remains a reproducible comparator. For compatible
   fixed-coefficient scenarios its feasible result also supplies an initial objective
   floor, re-evaluated by the shared evaluator. Original search initialization time
   is reported separately from the bounded improvement search budget.
2. Joint greedy initialization scores candidate **and opening month** against the
   current network. Monthly lower bounds are filled first. Automatic mode considers
   feasible greedy prefixes and stops optional additions at the configured marginal
   threshold. Manual targets independently initialize and optimize; they are never
   truncations of an unrelated 180/400 solution.
3. Bounded local search tries additions/removals (automatic counts), replacement
   sites, relocation to other months, and pairwise opening-month swaps. Accepted
   proposals are re-evaluated using full network equations and must improve by the
   numerical tolerance. A sampled neighborhood, iteration limit and wall-clock
   budget bound work; multiple reproducible seeded starts are available.
4. The returned plan is the best feasible found. A failed constructive search is
   reported as failure to find feasibility, not proof that no feasible plan exists.
   A time-limited run can stop before all neighborhoods/counts are explored.

Readiness columns `earliest_opening_date` / `latest_opening_date`, when supplied in
the GPS workbook, map by validated unique coordinates. Constraints include count
bounds, 12 monthly lower/upper bounds, optional exact monthly counts, optional
UB/Rural monthly limits and an investment budget with supplied costs. Each candidate
can open once, in the planning year. Zero openings are allowed when lower bounds allow.

Static candidate-to-existing and candidate-to-candidate distances and age-indexed
coefficient kernels are prepared once. Batched additions use vectorized incremental
scores; accepted changes are checked by the full evaluator. The dashboard caches
static preparation across target/count/search-limit changes and caches completed
scenarios with complete configuration, input signatures and code signatures.

The count curve is labeled **sequential joint-greedy prefixes**. It does not claim
that every count on the curve has been independently improved. The selected best
solution and independent benchmark/manual/automatic scenarios are separate points.

## Run

```powershell
cd 'C:\MO files\Cannibalization analyses'
.\.conda\python.exe expansion_optimization_inputs.py --target 180 --year 2027 --capacity 40 --horizon 24 --seconds 120
.\.conda\python.exe expansion_optimization_inputs.py --max-stores 400 --year 2027 --capacity 40 --horizon 24 --seconds 120
.\.conda\python.exe expansion_optimization_inputs.py --target 220 --seconds 60 --benchmark
.\.conda\python.exe -B -m unittest discover -s tests -v
.\.conda\python.exe -m streamlit run '05 Cannibalization Dashboard.py'
```

The CLI uses sales/fixed-cumulative/negative-only defaults. Programmatic use of
`OptimizationConfig` exposes NPV costs, exact allocations, regional capacities,
readiness, sources/tails, multiple starts and the remaining constraints. UI costs
remain blank until supplied. Exact monthly counts must sum to the manual target.
On the dashboard choose **Run optimization**; changing controls alone does not
compute a portfolio. Results are explicitly labeled with their last completed
configuration. The comparison button performs additional independent solves.

## Outputs and preservation

New outputs only: `outputs/expansion_optimization_v2/<scenario_id>/`.
The identifier includes model version, every scenario parameter, input content
hashes and the resulting site/month assignment. Each folder contains:

- `optimization_summary.csv`
- `optimal_store_count_curve.csv`
- `optimal_selected_stores.csv`
- `optimal_monthly_schedule.csv`
- `store_level_network_impacts.parquet`
- `optimization_diagnostics.json` (configuration, constraints, runtime, status,
  moves, coefficient provenance, seed, inputs and run timestamp)

Independent comparisons write under `comparisons/<configuration-and-input-hash>/`:
`scenario_comparison.csv` and a labeled combined approximate count curve. They
include no expansion, original expansion 180, frozen accepted 180, independently
optimized 180, manual target and automatic count where feasible. Candidate identity
schemes differ in old files, so benchmark crosswalks use one-to-one coordinate
matches within 0.5 m; mapping failures are reported instead of guessing IDs.
An original calendar that violates new constraints is marked infeasible, not
silently reassigned or used for an improvement claim.
Saved baseline calendar months are retained and evaluated at the same configured
opening day as new scenarios; original nominal day-15 totals are not compared
directly with day-1 totals.

Candidate portfolio tables show horizon gross/net sales, new-to-new changes and
allocated existing-store effects. Existing overlap attribution uses proportional
log contributions and reconciles; it is not a standalone candidate ranking. No
claim of unique causal attribution is made.

No original regression, cleaned dataset, accepted planning portfolio, fixed
planning result or `outputs/expansion_simulation/selected_stores_180.csv` is
overwritten. New generated optimization files are ignored by Git; no company
data is committed. The other working-tree data/configuration changes found at
the start of this task belong to the user and were left intact.

## Validation

Tests exercise zero/one/fixed-180/manual/automatic counts, exact and bounded
monthly capacities, regional limits, readiness dates, overlapping store effects,
new-to-new effects, coefficient fallback/tail indexing, cutoff behavior, signed
versus negative-only kernels, NPV cost requirements, objective/component
reconciliation, full-versus-incremental score equality, deterministic seeded
search and strict improvement of accepted location swaps. UI tests ensure no
startup optimization and visible rejection of missing NPV assumptions.

Geographic/demand preparation measured approximately 29 seconds locally; a batched
459-candidate × 12-month proposal scoring step measured approximately 0.12 seconds
after preparation. These are single-run observations, not deployment guarantees.
Benchmark result files provide actual baseline/improved objectives and runtimes;
improvements are reported only after common-evaluator reconciliation.

### Local benchmark completed

Inputs: 459 eligible saved forecasts, 651 existing stores, planning year 2027,
24-month horizon, opening day 1, maximum 40 openings per month, maximum count 400,
central demand, fixed cumulative coefficients, negative-only effects, seed 2027,
one start and a 60-second improvement search budget. No costs were supplied.

| Scenario | New stores | Incremental 24-month network sales (billion MNT) |
|---|---:|---:|
| No expansion | 0 | 0.000 |
| Existing saved expansion portfolio, common assumptions | 180 | 2.961 |
| Frozen accepted planning portfolio, common assumptions | 180 | 105.968 |
| Original greedy rerun, common evaluator | 180 | 495.601 |
| Jointly optimized fixed 180 | 180 | 566.492 |
| Independently optimized manual 220 | 220 | 601.546 |
| Best feasible automatic count found within budget | 348 | 647.737 |

Fixed-180 improvement over the unchanged greedy rerun: **70.892 billion MNT,
14.304%**. Greedy rerun took 15.33 seconds. Improved fixed-180 search took 50.01
seconds, plus 14.58 seconds for its greedy initializer (approximately 64.59 seconds
solver total, excluding shared input preparation). The fixed-180 result stopped
without an improvement in the sampled final neighborhood. Manual 220 and automatic
348 stopped on the search budget; neither is certified globally optimal. The
automatic count can change with budget, seed, source, demand or cost assumptions.
This particular fixed-180 gain came from the joint greedy initialization; its
local-search phase accepted no further move. The larger scenarios each accepted
one local move. No artificial improvement is attributed to the local search.

The existing saved expansion portfolio was originally selected under a different
event-study source and opening schedule. Its re-evaluated value above must not be
confused with its historical stored forecast. Comparisons here use one common
evaluator/source/calendar policy, not raw totals from incompatible output files.

Validation completed: **45 tests passed**. An additional real dashboard form test
optimized 3 stores from a 5-candidate pool and displayed reconciled KPIs without
an exception. CSV merge-conflict rejection remains an explicit input limitation
for optional event-study runs. Streamlit map-selection browser events are not
simulated by AppTest; a candidate detail dropdown provides the same inspection path.
