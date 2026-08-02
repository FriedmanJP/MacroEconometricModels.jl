# [Data Management](@id data_page)

**MacroEconometricModels.jl** provides typed data containers that carry metadata alongside the numbers, validate inputs, transform series to stationarity, and dispatch directly to every estimation function. The data module removes the manual bookkeeping between a raw data file and a fitted model.

- **Containers**: `TimeSeriesData`, `PanelData`, and `CrossSectionData` wrap a numeric matrix with variable names, frequency, FRED transformation codes, descriptions, and bibliographic references
- **Built-in datasets**: twelve curated datasets --- the two FRED databases, three research panels, and seven textbook teaching sets --- load with a single `load_example` call
- **Transformations**: FRED transformation codes 1--7 map raw levels to stationary series; `inverse_tcode` reconstructs the original levels
- **Validation**: `diagnose` reports NaN, Inf, constant columns, and short samples; `fix`, `dropna`, and `keeprows` repair or subset the sample
- **Panel operations**: Stata-style `xtset`, within-group lag/lead/difference, group extraction, balance detection, and DFM-based gap filling
- **Filtering**: `apply_filter` applies HP, Hamilton, Beveridge-Nelson, Baxter-King, or Boosted HP per variable and returns a container
- **Estimation dispatch**: every estimator accepts the matching container directly --- no manual conversion
- **Interoperability**: `DataFrame(result)`, `long_table`, and `write_csv` turn any estimate into a tidy table or CSV; `set_log_level` and `with_min_level` control library logging
- **Reproducibility**: randomized results carry a `ReproManifest`; `reproduce` re-runs and verifies bit-for-bit, and `save_model`/`load_model` persist a fitted model or container

```@setup data
using MacroEconometricModels, DataFrames, Random
```

## Quick Start

**Recipe 1: Load FRED-MD and inspect a subset**

```@example data
fred = load_example(:fred_md)                        # 126 variables, 804 months
describe_data(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]])
```

**Recipe 2: Transform to stationarity, then clean**

```@example data
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

d = apply_tcode(sub)   # applies the per-variable codes stored in sub.tcode
d = fix(d)             # drop rows left non-finite by differencing
describe_data(d)
```

**Recipe 3: Estimate straight from a container**

```@example data
model = estimate_var(d, 2)   # no to_matrix() call needed
report(model)
```

**Recipe 4: Panel data with the Penn World Table**

```@example data
pwt = load_example(:pwt)
panel_summary(pwt)
```

**Recipe 5: Subset rows the supported way**

```@example data
usa = group_data(pwt, "USA")             # one country as TimeSeriesData
postwar = keeprows(usa, collect(41:74))  # 1990-2023
describe_data(postwar[:, ["rgdpna", "emp"]])
```

**Recipe 6: Within-group panel transformations**

```@example data
ddcg = load_example(:ddcg)

lag1_y  = panel_lag(ddcg, :y, 1)    # L.y
lead1_y = panel_lead(ddcg, :y, 1)   # F.y
d_dem   = panel_diff(ddcg, :dem)    # D.dem
count(==(1.0), filter(isfinite, d_dem))   # democratizations in the sample
```

---

## Data Containers

All three containers subtype `AbstractMacroData` and store a numeric matrix plus metadata. They correspond to the three data structures of applied econometrics: a single entity observed over time, many entities observed over time, and many entities observed once.

### TimeSeriesData

`TimeSeriesData{T}` holds a ``T_{obs} \times n`` matrix with variable names, a frequency, FRED transformation codes, an integer time index, optional date labels, dataset and per-variable descriptions, and reference keys. Four constructors cover the common inputs.

```@example data
# From a matrix with full metadata
d_ts = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"],
    frequency=Quarterly,
    tcode=[5, 6, 2],
    time_index=collect(1:200))

# From a vector (univariate)
d_uni_demo = TimeSeriesData(randn(200); varname="GDP", frequency=Monthly)

# From a DataFrame --- numeric columns only, missing becomes NaN
d_df = TimeSeriesData(DataFrame(gdp=randn(100), cpi=randn(100)); frequency=Quarterly)
```

Integer and other non-float inputs convert to `Float64`. `tcode` entries must lie in `1:7`, and `varnames`, `tcode`, `time_index`, and `dates` must all match the matrix dimensions or the constructor throws an `ArgumentError`.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Matrix{T}` | ``T_{obs} \times n`` data matrix |
| `varnames` | `Vector{String}` | Variable names (default `["x1", ...]`) |
| `frequency` | `Frequency` | Data frequency (default `Other`; informational metadata) |
| `tcode` | `Vector{Int}` | FRED transformation code per variable (default all `1`) |
| `time_index` | `Vector{Int}` | Integer time identifiers (default `1:T_obs`) |
| `T_obs` | `Int` | Number of observations |
| `n_vars` | `Int` | Number of variables |
| `desc` | `Vector{String}` | Dataset description, stored as a length-1 vector so it stays mutable |
| `vardesc` | `Dict{String,String}` | Per-variable descriptions keyed by variable name |
| `source_refs` | `Vector{Symbol}` | Reference keys resolved by `refs()` |
| `dates` | `Vector{String}` | Date labels (default empty; set with `set_dates!`) |

### PanelData

`PanelData{T}` stores a stacked longitudinal matrix with a group and a time identifier on every row. Build it with `xtset` rather than the raw constructor --- see [Panel Data](@ref panel_data_section).

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Matrix{T}` | Stacked data matrix, ``T_{obs} \times n`` over all groups |
| `varnames` | `Vector{String}` | Variable names |
| `frequency` | `Frequency` | Data frequency |
| `tcode` | `Vector{Int}` | FRED transformation code per variable |
| `group_id` | `Vector{Int}` | Group identifier per row |
| `time_id` | `Vector{Int}` | Time identifier per row |
| `cohort_id` | `Union{Vector{Int}, Nothing}` | Treatment cohort per row, `nothing` when unused |
| `group_names` | `Vector{String}` | Unique group labels |
| `n_groups` | `Int` | Number of groups |
| `n_vars` | `Int` | Number of variables |
| `T_obs` | `Int` | Total number of rows |
| `balanced` | `Bool` | True when every group has the same number of rows |
| `desc`, `vardesc`, `source_refs` | --- | As for `TimeSeriesData` |

### CrossSectionData

`CrossSectionData{T}` stores cross-sectional observations, with an observation identifier in place of a time index. It has no `tcode` and no `frequency`, because neither means anything without a time dimension.

```@example data
mroz = load_example(:mroz)
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Matrix{T}` | ``N_{obs} \times n`` data matrix |
| `varnames` | `Vector{String}` | Variable names |
| `obs_id` | `Vector{Int}` | Observation identifiers (default `1:N_obs`) |
| `N_obs` | `Int` | Number of observations |
| `n_vars` | `Int` | Number of variables |
| `desc`, `vardesc`, `source_refs` | --- | As for `TimeSeriesData` |

### Frequency

```julia
@enum Frequency Daily Monthly Quarterly Yearly Mixed Other
```

`frequency` is metadata. It labels summary displays and plot axes; it never changes an estimate. Filter bandwidths and smoothing parameters are always passed explicitly --- `lambda=129600.0` for a monthly HP filter, for instance.

---

## Indexing and Accessors

The containers implement a deliberately small indexing surface: column selection by name or index, and --- for a `TimeSeriesData` carrying date labels --- row selection by date string. **They are not `AbstractArray`s.** Anything outside the table below throws a `MethodError`.

| Expression | `TimeSeriesData` | `PanelData` | `CrossSectionData` |
|---|---|---|---|
| `d[:, "GDP"]` | `Vector{T}` | not supported | `Vector{T}` |
| `d[:, ["GDP", "CPI"]]` | sub-container | not supported | not supported |
| `d[:, 2]` | `Vector{T}` | not supported | not supported |
| `d["2020Q1", :]` | `Vector{T}` (one row) | not supported | not supported |
| `d[["2020Q1", "2020Q2"], :]` | sub-container | not supported | not supported |
| `d[10:60, :]` | **not supported** | **not supported** | **not supported** |
| `d[5, 2]` | **not supported** | **not supported** | **not supported** |

!!! warning "Integer row ranges do not index a container"
    `d[end-59:end, :]` throws `MethodError: no method matching axes(::TimeSeriesData{Float64}, ::Int64)`
    --- `end` needs `axes`, which the containers do not define. To take a row window, either
    use `keeprows` (which keeps the metadata aligned) or drop to the matrix with `to_matrix`
    first and rebuild. Both idioms appear below.

```@example data
# Column extraction
ip  = fred[:, "INDPRO"]                              # Vector{Float64}, length 804
sub2 = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]   # new TimeSeriesData, metadata sliced

# Row selection --- keeprows takes a mask or explicit indices and keeps metadata aligned
gm = keeprows(fred, collect(337:588))                # 1987:01-2007:12

# Or drop to the raw matrix and rebuild (metadata must be re-supplied)
gm2 = TimeSeriesData(to_matrix(fred)[337:588, :];
    varnames=varnames(fred), frequency=Monthly, tcode=fred.tcode)
```

Dimensions and metadata come from accessor functions rather than field access, so the same call works across all three container types.

```@example data
(nobs(fred), nvars(fred), size(fred), frequency(fred))
```

`to_matrix` and `to_vector` reach the raw arrays. The one-argument `to_vector(d)` requires a single variable; the two-argument forms select one column by name or by index.

```@example data
(size(to_matrix(sub2)), length(to_vector(sub2, "INDPRO")), length(to_vector(sub2, 1)))
```

### Descriptions

Containers carry a dataset description and a per-variable description dictionary. Built-in datasets arrive with both populated, and both survive subsetting, `apply_tcode`, `fix`, and `group_data`.

```@example data
(desc(fred), vardesc(fred, "INDPRO"), vardesc(fred, "CPIAUCSL"))
```

`set_desc!`, `set_vardesc!`, and `rename_vars!` mutate metadata in place. `rename_vars!` moves the matching `vardesc` key along with the name, so a description never goes stale.

```@example data
d_rn = TimeSeriesData(randn(50, 2); varnames=["a", "b"],
    vardesc=Dict("a" => "Output gap", "b" => "Core inflation"))
rename_vars!(d_rn, "a" => "GAP")          # single rename; the description follows
set_desc!(d_rn, "Quarterly US macro data")
vardesc(d_rn, "GAP")
```

### Date Labels

`set_dates!` attaches string labels to rows. Once set they enable date-string row indexing and label the x-axis of `plot_result`.

```@example data
d_dt = TimeSeriesData(randn(4, 2); varnames=["GDP", "CPI"])
set_dates!(d_dt, ["2020Q1", "2020Q2", "2020Q3", "2020Q4"])

d_dt[["2020Q1", "2020Q2"], :]   # sub-container; d_dt["2020Q1", :] returns one row as a Vector
```

---

## Built-in Datasets

`load_example(name)` returns a fully populated container --- data, variable names, transformation codes, descriptions, and reference keys --- read from a TOML file shipped with the package. Twelve datasets are available; an unknown name throws an `ArgumentError` listing the valid set.

| Key | Returns | Size | Content |
|-----|---------|------|---------|
| `:fred_md` | `TimeSeriesData` | 804 × 126, Monthly | FRED-MD, January 2026 vintage (McCracken & Ng 2016) |
| `:fred_qd` | `TimeSeriesData` | 268 × 245, Quarterly | FRED-QD, January 2026 vintage (McCracken & Ng 2021) |
| `:pwt` | `PanelData` | 2812 × 42, 38 countries × 74 years | Penn World Table 10.01, OECD countries (Feenstra, Inklaar & Timmer 2015) |
| `:ddcg` | `PanelData` | 9384 × 2, 184 countries × 51 years | Democracy and growth (Acemoglu, Naidu, Restrepo & Robinson 2019) |
| `:mpdta` | `PanelData` | 2500 × 3, 500 counties × 5 years | County minimum wage and employment (Callaway & Sant'Anna 2021) |
| `:grunfeld` | `PanelData` | 200 × 3, 10 firms × 20 years | Firm investment, 1935-1954 (Grunfeld 1958) --- the canonical SUR/3SLS panel |
| `:mroz` | `CrossSectionData` | 753 × 22 | Married women's labor supply (Mroz 1987) --- the canonical selection-model extract |
| `:stackloss` | `CrossSectionData` | 21 × 4 | Ammonia plant stack loss (Brownlee 1965) --- the canonical robust-regression outlier set |
| `:nile` | `TimeSeriesData` | 100 × 1, Yearly | Nile flow at Aswan, 1871-1970 (Durbin & Koopman 2012) --- the local-level state-space series |
| `:gnp_hamilton` | `TimeSeriesData` | 135 × 1, Quarterly | US real GNP growth, 1951Q2-1984Q4 (Hamilton 1989) --- the Markov-switching business-cycle series |
| `:denmark` | `TimeSeriesData` | 55 × 5, Quarterly | Danish money demand, 1974Q1-1987Q3 (Johansen & Juselius 1990) --- the cointegration set |
| `:wiot` | `IOData` | 2 sectors | Hypothetical two-sector input-output table (Miller & Blair 2009) |

The first five are the working datasets for the macro methods on this site; the rest are the textbook sets the univariate, cross-sectional, and cointegration pages estimate against. `:wiot` returns an `IOData` rather than one of the three containers --- see [Input-Output Analysis](@ref io_page).

### The FRED Databases

FRED-MD and FRED-QD ship as the January 2026 vintage with the St. Louis Fed variable descriptions and the recommended transformation code for every series. Because the codes travel with the data, `apply_tcode(d)` with no second argument reproduces the transformations used in the source papers.

```@example data
qd = load_example(:fred_qd)
(desc(qd), vardesc(qd, "GDPC1"), qd.tcode[findfirst(==("GDPC1"), varnames(qd))])
```

Real GDP carries code 5, the log first difference --- quarterly real growth. `refs` resolves the reference keys stored on a container into a formatted bibliography in `:text`, `:latex`, `:bibtex`, or `:html`. The one-argument form **returns** a `String` instead of printing it, so wrap it in `print` to display it.

```@example data
print(refs(qd))
```

```@example data
print(refs(:fred_md; format=:bibtex))
```

### Research Panels

The Penn World Table loads as a balanced `PanelData`: 38 OECD countries observed every year from 1950 to 2023, with 42 national-accounts variables. Individual countries come out as `TimeSeriesData` through `group_data`.

```@example data
(nobs(pwt), nvars(pwt), ngroups(pwt), isbalanced(pwt), vardesc(pwt, "rgdpna"))
```

The DDCG panel of Acemoglu, Naidu, Restrepo & Robinson (2019) covers 184 countries over 1960-2010 with two variables: log GDP per capita (`y`) and a binary democracy indicator (`dem`). It is the reference dataset for LP-DiD and event-study local projections. The mpdta panel of Callaway & Sant'Anna (2021) covers 500 US counties over 2003-2007 with log employment (`lemp`), log population (`lpop`), and `first_treat`, the calendar year of a county's first minimum-wage increase (`0` for never-treated counties).

```@example data
mpdta = load_example(:mpdta)
(varnames(mpdta), ngroups(mpdta), isbalanced(mpdta))
```

!!! warning "`refs` fails on `:ddcg` and `:mpdta`"
    Both TOML files record reference keys the bibliography does not define
    (`Acemoglu2019_DDCG`, `DubeGirardiJordaTaylor2025`, `callaway_santanna_2021`), so
    `refs(load_example(:ddcg))` throws `ArgumentError: Unknown reference key`. Cite these two
    datasets from the reference list at the foot of this page until the keys are reconciled.

---

## FRED Transformation Codes

FRED-MD and FRED-QD assign every series an integer code recording the transformation that renders it stationary (McCracken & Ng 2016). `apply_tcode` implements all seven. Writing ``x_t`` for the raw level and ``\Delta`` for the first-difference operator:

```math
\begin{aligned}
&\text{1: } x_t
&&\text{2: } \Delta x_t
&&\text{3: } \Delta^2 x_t
&&\text{4: } \ln x_t \\
&\text{5: } \Delta \ln x_t
&&\text{6: } \Delta^2 \ln x_t
&&\text{7: } \Delta\!\left(\frac{x_t}{x_{t-1}} - 1\right)
\end{aligned}
```

where:
- ``x_t`` is the raw series value in period ``t``
- ``\Delta x_t = x_t - x_{t-1}`` and ``\Delta^2 x_t = \Delta x_t - \Delta x_{t-1}``
- ``\ln`` is the natural logarithm, so ``\Delta \ln x_t`` is the continuously compounded growth rate
- code 7 differences the simple percentage change, not the level

| Code | Transformation | Economic reading | Rows lost (vector) | Rows lost (container) |
|------|----------------|------------------|--------------------|-----------------------|
| 1 | Level | The series itself | 0 | 0 |
| 2 | First difference | Change in the level | 1 | 1 |
| 3 | Second difference | Change in the change | 2 | 2 |
| 4 | Log level | Log of the series | 0 | 1 |
| 5 | Log first difference | Growth rate | 1 | 1 |
| 6 | Log second difference | **Change in the growth rate** | 2 | 2 |
| 7 | Difference of percent change | Change in the percentage growth rate | 2 | 2 |

!!! warning "Code 6 is not inflation"
    `CPIAUCSL` carries code 6, so `apply_tcode` returns ``\Delta^2 \ln \text{CPI}_t`` --- the
    *change* in monthly inflation, a series whose mean is essentially zero. Monthly inflation
    itself is code 5. The same holds for `M2SL`. Reading a code-6 column as a rate rather than
    as a change in a rate inverts the sign of every interpretation built on it.

!!! note "Code 4 costs a row inside a container"
    The vector method `apply_tcode(y, 4)` returns `log.(y)` at full length, but the container
    method budgets one lost row for code 4 and trims the sample accordingly. A `TimeSeriesData`
    transformed with code 4 therefore has ``T_{obs} - 1`` rows even though the log transform
    discards nothing.

### Applying Transformations

Three call forms cover the useful cases: the codes stored on the container, an explicit per-variable vector, or one code for every variable.

```@example data
y = [100.0, 105.0, 110.0, 108.0, 115.0]
apply_tcode(y, 5)      # log first differences of a raw vector
```

```@example data
d_meta = apply_tcode(sub)              # uses sub.tcode = [5, 6, 2]
d_expl = apply_tcode(sub, [5, 5, 2])   # override: growth rates for both quantity and price
d_all  = apply_tcode(sub, 5)           # one code for every variable
(nobs(sub), nobs(d_meta), d_meta.tcode)
```

Under mixed codes, rows are trimmed to the **shortest** transformed series and aligned to the end of the sample, so every column stays on one calendar. `sub` mixes codes 5, 6, and 2; code 6 costs two rows, so all three columns come back with ``804 - 2 = 802`` observations and a `time_index` starting at 3.

Codes 4-7 require strictly positive data. When a log-based code meets a non-positive value, `apply_tcode` warns, falls back to code 2 for that variable alone, and records the substitution in the returned `tcode` field --- so comparing `d.tcode` against the codes you requested tells you exactly which columns were demoted.

### Inverse Transformations

`inverse_tcode` reconstructs levels from a transformed series. Difference-based codes need anchor values, supplied through `x_prev` as original levels rather than transformed values.

```@example data
y_inv = [100.0, 105.0, 110.0, 108.0]
yd = apply_tcode(y_inv, 5)
inverse_tcode(yd, 5; x_prev=[y_inv[1]])   # recovers y_inv[2:end]
```

| Code | Required `x_prev` |
|------|-------------------|
| 1, 4 | None |
| 2, 5 | One value --- the last pre-sample level |
| 3, 6, 7 | Two values --- the last two pre-sample levels |

Round trips are exact to machine precision: the reconstruction above matches the original levels to within ``4.3 \times 10^{-14}``, the accumulated rounding of the `log`/`exp` pair.

---

## Validation and Cleaning

### Diagnosing

`diagnose` scans a container for the four problems that silently corrupt an estimate: NaN, Inf, zero-variance columns, and a sample too short to estimate anything. It returns a `DataDiagnostic`, which prints only the offending variables.

```@example data
d_raw = apply_tcode(sub)
diagnose(d_raw)
```

The transformed panel is not clean. `CPIAUCSL` carries three NaN, all traceable to a single missing level: the January 2026 vintage has one gap in the raw CPI series near the end of the sample, and a second log difference spreads one missing level across three transformed rows. Nothing is infinite, no column is constant, and 802 observations is far from short, so listwise deletion of those three rows is the right repair.

| Field | Type | Description |
|-------|------|-------------|
| `n_nan` | `Vector{Int}` | NaN count per variable |
| `n_inf` | `Vector{Int}` | Inf count per variable |
| `is_constant` | `Vector{Bool}` | True when the variable has zero variance |
| `is_short` | `Bool` | True when the sample has fewer than 10 observations |
| `varnames` | `Vector{String}` | Variable names |
| `is_clean` | `Bool` | True when none of the above fired |

### Repairing

`fix` returns a clean copy --- the input is never modified. Inf becomes NaN first, then the chosen method runs, then constant columns are dropped with a warning.

```@example data
(nobs(fix(d_raw)),                       # :listwise (default) --- 3 rows dropped
 nobs(fix(d_raw; method=:interpolate)),  # interior NaN interpolated, edges filled
 nobs(fix(d_raw; method=:mean)))         # NaN replaced by the column mean
```

Only `:listwise` changes the row count. `:interpolate` fills interior gaps by linear interpolation and carries the first and last finite values outward to the edges; `:mean` substitutes the column mean of the finite values. Both keep all 802 rows, which matters when the container has to stay aligned with an external series.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:listwise` | Repair strategy: `:listwise`, `:interpolate`, or `:mean` |

For row-level control without the constant-column logic, `dropna` removes every row holding a NaN or Inf --- optionally restricted to selected variables through `vars=` --- and `keeprows` subsets to an explicit `BitVector` mask or index vector. Both preserve `time_index`, `dates`, and the description metadata.

!!! note "Technical Note"
    After `fix`, `diagnose(d).is_clean` is `true` unless every column was constant, in which
    case `fix` throws rather than return an empty container. `fix` on a `PanelData`
    interpolates and averages *within* each group, never across a group boundary.

### Model Compatibility

`validate_for_model` checks the dimensionality a model family requires and throws an `ArgumentError` naming the mismatch. It costs nothing and turns a confusing downstream failure into a clear one.

```@example data
validate_for_model(d, :var)                    # passes: 3 variables
validate_for_model(d[:, ["INDPRO"]], :arima)   # passes: 1 variable
```

| Category | Requirement | Model types |
|----------|-------------|-------------|
| Multivariate | ``n \geq 2`` | `:var`, `:vecm`, `:bvar`, `:factors`, `:dynamic_factors`, `:gdfm` |
| Univariate | ``n = 1`` | `:arima`, `:ar`, `:ma`, `:arma`, `:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`, `:hp_filter`, `:hamilton_filter`, `:beveridge_nelson`, `:baxter_king`, `:boosted_hp`, `:adf`, `:kpss`, `:pp`, `:za`, `:ngperron` |
| Flexible | any | `:lp`, `:lp_iv`, `:smooth_lp`, `:state_lp`, `:propensity_lp`, `:gmm` |

---

## [Panel Data](@id panel_data_section)

### Stata-Style xtset

`xtset` turns a `DataFrame` into a `PanelData`, mirroring Stata's command of the same name. It takes every numeric column except the group, time, and cohort columns, sorts by (group, time), rejects duplicate (group, time) pairs, and records whether the panel is balanced.

```@example data
Random.seed!(20260802)
df_xt = DataFrame(
    firm       = repeat(1:50, inner=20),
    year       = repeat(2001:2020, 50),
    investment = randn(1000),
    output     = randn(1000))

xtset(df_xt, :firm, :year; frequency=Yearly)
```

For difference-in-differences, a `cohort` column encodes treatment timing on the container itself, which every DiD estimator reads in preference to deriving cohorts from a treatment indicator.

```@example data
df_did = DataFrame(
    firm    = repeat(1:6, inner=10),
    year    = repeat(2001:2010, 6),
    revenue = randn(60),
    cohort  = repeat([0, 0, 2004, 2004, 2007, 2007], inner=10))

pd_did = xtset(df_did, :firm, :year; cohort=:cohort)
sort(unique(pd_did.cohort_id))
```

!!! warning "`xtset` stores cohort ranks, not cohort values"
    The cohort column is mapped to 1-based ranks over its sorted unique values, so
    `[0, 2004, 2007]` is stored as `[1, 2, 3]`. DiD estimators read `cohort_id` as an adoption
    *period* and compare it against `time_id`, so calendar-year cohorts do not survive the
    round trip. Pass cohorts already expressed on the `time_id` scale, or let the estimators
    derive timing from the treatment column. See [Difference-in-Differences](@ref did_page).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `varnames` | `Union{Vector{String},Nothing}` | `nothing` | Override variable names (default: the column names) |
| `frequency` | `Frequency` | `Other` | Frequency metadata |
| `tcode` | `Union{Vector{Int},Nothing}` | `nothing` | Transformation code per variable (default: all `1`) |
| `desc` | `String` | `""` | Dataset description |
| `vardesc` | `Union{Dict{String,String},Nothing}` | `nothing` | Per-variable descriptions |
| `cohort` | `Union{Symbol,Nothing}` | `nothing` | Column identifying treatment cohort membership |

### Structure and Extraction

```@example data
(ngroups(pwt), isbalanced(pwt), groups(pwt)[1:3], groups(pwt)[end])
```

`group_data` extracts one group as a `TimeSeriesData`, by name or by index, carrying the panel's `time_id` through as the `time_index`. It is the bridge from panel storage to every univariate and multivariate time series method.

```@example data
group_data(pwt, "USA")
```

### Lag, Lead, and Difference

`panel_lag`, `panel_lead`, and `panel_diff` compute within-group transformations that respect both group boundaries and gaps in `time_id`. Each returns a plain `Vector` of length ``T_{obs}``, aligned with the container's rows, carrying `NaN` wherever the operation is undefined.

```@example data
lag4_y = panel_lag(ddcg, :y, 4)
(count(isfinite, lag1_y), count(isfinite, lag4_y), length(lag1_y))
```

Of 9384 country-years, 6979 have a one-period lag of log GDP available. The 2405 missing values are not merely the 184 first observations: the DDCG panel is rectangular in storage but has genuine gaps in `y`, and a lag exists only when the previous *year* is present with a finite value. Requesting a four-period lag shrinks the usable sample further, which is why LP-DiD specifications on this panel lose observations quickly as the horizon grows.

`add_panel_lag`, `add_panel_lead`, and `add_panel_diff` append those same vectors as new columns and return a new `PanelData`, named `lag{k}_{var}`, `lead{k}_{var}`, and `d_{var}`.

```@example data
varnames(add_panel_lag(ddcg, :y, 1))
```

### Filling Gaps

`balance_panel` fills NaN by running a dynamic factor model on each group that has missing data and substituting the Kalman-smoothed estimates. It fills *values*; it does not add rows, so a panel that is ragged in row count stays ragged.

```@example data
Random.seed!(11)
df_bal = DataFrame(firm   = repeat(1:4, inner=12),
                   year   = repeat(2001:2012, 4),
                   output = randn(48),
                   hours  = randn(48))
df_bal.output[[5, 17, 33]] .= NaN
pd_in = xtset(df_bal, :firm, :year; frequency=Yearly)

pd_filled = balance_panel(pd_in; r=1, p=1)
(count(isnan, to_matrix(pd_in)), count(isnan, to_matrix(pd_filled)))
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:dfm` | Fill method (only `:dfm` is implemented) |
| `r` | `Int` | `3` | Number of factors in the DFM |
| `p` | `Int` | `2` | VAR lag order in the factor dynamics |

The factor count is clamped internally to what each group can support, so `r=3` on a two-variable panel silently becomes `r=1`. Cost scales with the number of groups holding missing data: filling all 38 Penn World Table countries across 42 variables takes about a minute, while the four-group example above completes in seconds.

---

## Summary Statistics

`describe_data` returns a `DataSummary` and prints a per-variable table. For a `PanelData` it prints the panel structure as well. NaN values are excluded from every statistic, so `n` can differ across columns.

```@example data
describe_data(d)
```

All three series are transformed, so the level information is gone and what remains is the shape of the shock distribution. Industrial production growth averages 0.0019 per month --- about 2.3% a year --- with a standard deviation of 0.0099 and an excess kurtosis of 59.0, the signature of the 2020 shutdown outliers. `CPIAUCSL` is a code-6 series, so its mean is zero by construction: the *change* in inflation has no drift even though inflation does. The federal funds rate in first differences averages a near-zero 0.0022 with excess kurtosis of 50.9, reflecting long stretches of no change punctuated by large policy moves.

| Field | Type | Description |
|-------|------|-------------|
| `varnames` | `Vector{String}` | Variable names |
| `n` | `Vector{Int}` | Count of finite observations per variable |
| `mean` | `Vector{Float64}` | Mean of the finite values |
| `std` | `Vector{Float64}` | Sample standard deviation (divisor ``n-1``) |
| `min`, `max` | `Vector{Float64}` | Extremes |
| `p25`, `median`, `p75` | `Vector{Float64}` | Quartiles by linear interpolation |
| `skewness` | `Vector{Float64}` | Population (method-of-moments) skewness |
| `kurtosis` | `Vector{Float64}` | Population excess kurtosis |
| `T_obs`, `n_vars` | `Int` | Container dimensions |
| `frequency` | `Frequency` | Data frequency |

The `std` column uses the sample convention (divisor ``n-1``), while `skewness` and `kurtosis` use the population convention so numerator and denominator share one divisor. With ``m_k = \frac{1}{n}\sum (x_i - \bar x)^k``, the reported values are ``\text{skew} = m_3 / m_2^{3/2}`` and ``\text{kurt} = m_4 / m_2^2 - 3``. A series with fewer than three finite observations reports zero for both.

---

## Filtering

`apply_filter` runs a time series filter over the variables of a container and returns a container of the same type, so a filtered panel stays estimable. When filters disagree about how many observations they consume, the output is trimmed to the intersection of the valid ranges. For the mathematics of each filter, see [Time Series Filters](@ref filters_page).

```@example data
d_fl = fix(TimeSeriesData(log.(to_matrix(fred[:, ["INDPRO", "PAYEMS", "HOUST"]]));
    varnames=["INDPRO", "PAYEMS", "HOUST"], frequency=Monthly))

d_hp  = apply_filter(d_fl, :hp; component=:cycle, lambda=129600.0)
d_ham = apply_filter(d_fl, :hamilton; component=:cycle, h=24, p=12)
(nobs(d_fl), nobs(d_hp), nobs(d_ham))
```

The HP filter is two-sided and returns a cycle for every observation, so the sample is unchanged at 802 months. The Hamilton (2018) regression filter conditions on ``p = 12`` lags and projects ``h = 24`` months ahead, consuming 35 observations and returning 767. The available filter symbols are `:hp`, `:hamilton`, `:bn`, `:bk`, and `:boosted_hp`; keyword arguments other than `component` and `vars` are forwarded to the underlying filter.

```@example data
# One filter per variable; `nothing` passes a column through untouched
d_pv = apply_filter(d_fl, [(:hp, :trend), (:hamilton, :cycle), nothing])

# Filter a subset by name; the rest pass through
d_sel = apply_filter(d_fl, :hp; vars=["INDPRO", "PAYEMS"], component=:cycle)
(nobs(d_pv), nobs(d_sel))
```

Mixing an HP trend with a Hamilton cycle costs the sample the Hamilton startup: the result is trimmed to 791 rows, the intersection of a full-length and a shortened valid range. Selective filtering leaves the untouched columns at full length, so nothing is trimmed.

On a `PanelData`, `apply_filter` runs group by group and reassembles the result. The `vars=` keyword is how panel variables get selected, since a `PanelData` has no column indexing.

```@example data
pwt_hp = apply_filter(pwt, :hp; vars=["rgdpna", "rconna"], component=:cycle)
isbalanced(pwt_hp)
```

!!! note "Technical Note"
    Filters that shorten their output (Hamilton, Baxter-King) trim each group against that
    group's own valid range. If the groups differ in length, a balanced input panel can come
    back unbalanced. The HP, Beveridge-Nelson, and Boosted HP filters preserve the row count
    and cannot unbalance a panel.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `component` | `Symbol` | `:cycle` | Component to extract: `:cycle` or `:trend` |
| `vars` | `Union{Nothing, Vector{String}, Vector{Int}}` | `nothing` | Variables to filter (default: all) |

---

## Estimation Dispatch

Every estimator has a container method. Multivariate estimators call `to_matrix` and forward `varnames`, so equation labels in the output carry the names from the container; univariate estimators call `to_vector` and therefore require exactly one variable.

```@example data
d_uni = d[:, ["INDPRO"]]
report(estimate_ar(d_uni, 2))
```

Industrial production growth is close to an AR(1): the first lag loads at 0.3028 with a t-ratio above 8, the second at ``-0.0656`` is marginal, and the model accounts for 8.5% of the monthly variance. A unit root test on the same container confirms the transformation did its job.

```@example data
report(adf_test(d_uni))
```

The ADF statistic of ``-12.42`` on 795 observations rejects the unit root null far beyond the 1% critical value of ``-3.433``. Code 5 delivered a stationary series, which is exactly what the VAR in Recipe 3 requires.

The dispatch surface covers the multivariate families (`estimate_var`, `estimate_vecm`, `estimate_bvar`, `estimate_factors`, `estimate_dynamic_factors`, `estimate_gdfm`, the five local-projection variants, `johansen_test`), the univariate families (the ARIMA and GARCH families, `estimate_sv`, the five filters, the five unit root tests), the nowcasting entry points, `estimate_dsge_bayes`, and the spectral and autocorrelation functions. `CrossSectionData` dispatches to `estimate_reg`, `estimate_logit`, `estimate_probit`, and `estimate_iv` with symbol arguments naming the dependent and independent variables.

---

## Visualization

`plot_result` renders each container through a family of views selected by the `view` keyword; an unrecognized view throws an `ArgumentError` listing the valid set. `vars` accepts names or 1-based indices everywhere.

| Container | Views |
|-----------|-------|
| `TimeSeriesData` | `:line` (default), `:scatter`, `:hist`, `:density`, `:corr`, `:growth`, `:binscatter` |
| `PanelData` | `:lines` (default), `:quantiles`, `:spaghetti`, `:groups`, `:scatter`, `:binscatter` |
| `CrossSectionData` | `:hist` (default), `:density`, `:scatter`, `:corr`, `:pairs`, `:binscatter` |

The default line view draws one panel per variable, labelling the x-axis with `dates(d)` when they are set and with `time_index` otherwise.

```julia
gm2 = TimeSeriesData(to_matrix(fred)[337:588, :];   # 1987:01-2007:12
    varnames=varnames(fred), frequency=Monthly, tcode=fred.tcode)
plot_result(gm2[:, ["INDPRO", "UNRATE", "CPIAUCSL"]])
```

```@raw html
<iframe src="../assets/plots/data_timeseries.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `:corr` view replaces the panels with a Pearson correlation heatmap on a symmetric ``[-1, 1]`` diverging scale, capped at twelve variables. It answers the question that precedes every VAR ordering decision: which of these series actually move together once they are stationary?

```julia
cvars = ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL", "GS10", "PAYEMS", "TB3MS"]
plot_result(fix(apply_tcode(gm2[:, cvars])); view=:corr)
```

```@raw html
<iframe src="../assets/plots/data_timeseries_corr.html" width="100%" height="480" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

A `PanelData` draws one panel per variable with a line per group, capped at ten groups by default.

```julia
plot_result(pwt; vars=["rgdpna", "pop", "emp", "hc"])
```

```@raw html
<iframe src="../assets/plots/data_panel.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The remaining views --- quantile fans, spaghetti plots with highlighted units, small-multiple grids, scatters with within- or between-group demeaning, and binscatters --- are catalogued with examples on the [Plotting](@ref plotting_page) page.

---

## Interoperability: DataFrames and CSV

Every coefficient-bearing result and every array-valued result exposes a programmatic tabular view, so an estimate reaches a `DataFrame`, a CSV file, or a co-author's R or Python session without hand-scraping fields. Coefficient models implement the Tables.jl source interface, so `DataFrame(model)` returns exactly the numbers `report(model)` prints.

```@example data
coef_table = DataFrame(model)
first(coef_table, 5)
```

Array-valued results --- impulse responses, variance decompositions, forecasts --- have no canonical rectangular shape, so `long_table` returns a tidy table with one row per cell and explicit index columns.

```@example data
irf_var = irf(model, 12; method=:cholesky)
tidy_irf = long_table(irf_var)
first(tidy_irf, 5)
```

Twelve horizons across three variables and three shocks give 108 rows. The `lower` and `upper` columns are `missing` here because a Cholesky IRF computed without bootstrap replications carries no uncertainty; a bootstrapped IRF fills them.

`write_csv` exports either form --- coefficient models write their coefficient table, array-valued results write their `long_table` --- through the standard-library `DelimitedFiles`, with no CSV.jl dependency.

```@example data
write_csv(model, joinpath(tempdir(), "var_coefficients.csv"))
write_csv(irf_var, joinpath(tempdir(), "var_irf.csv"))
nothing # hide
```

The column sets are uniform across result families so downstream scripts stay generic.

| Result | Columns |
|--------|---------|
| Coefficient models (`RegModel`, `LogitModel`, panel/ordered/multinomial, `VARModel`, `MarginalEffects`, `DIDResult`) | `term, estimate, std_error, stat, p_value, ci_lower, ci_upper`, plus a block discriminator (`equation`, `alternative`) where applicable |
| `ImpulseResponse` / `BayesianImpulseResponse` | `horizon, variable, shock, value, lower, upper` |
| `FEVD` | `horizon, variable, shock, value` |
| `LPImpulseResponse` | `horizon, variable, shock, value, se, lower, upper` |
| Forecasts (`VARForecast`, BVAR/VECM/LP) | `horizon, variable, value, lower, upper` |

---

## Reproducibility and Model Persistence

Randomized results --- bootstrap IRF bands, BVAR posteriors --- carry a `ReproManifest` recording how they were produced: the RNG seed, the thread count, the Julia and package versions, the operating system, a UTC timestamp, and the package git revision. Passing `seed` to a randomized estimator makes the draw reproducible and lets `reproduce` re-run the computation and confirm the numbers match bit for bit.

```@example data
post = estimate_bvar(d, 2; n_draws=200, seed=20260802)
show(IOContext(stdout, :repro => false), post.manifest)
```

A bare `post.manifest` adds a `git` line carrying the package revision the result
was produced under, and marks it `(dirty)` when the working tree had uncommitted
changes. `:repro => false` drops that one line, which is what keeps a rebuilt
page from changing on every commit; the revision is still stored and readable as
`post.manifest.git_sha`. The same key controls the one-line reproducibility
footer on `report`, which is off by default for the same reason --- pass
`report(post; repro = true)` to see it.

```@example data
reproduce(post)
```

The report compares every stored draw array against a fresh run from the recorded seed and settings, and states the thread count under which each was produced. A seed cannot be recovered from an `AbstractRNG` after the fact, so the estimator has to own it: pass `seed=N` and it seeds a fresh generator, records `N`, and reproduces exactly, independently of thread count. Without a seed the manifest still captures the environment but marks the result as not seed-reproducible. A bootstrap IRF carries a manifest too, reproduced with `reproduce(ir, model)` because the source model is not retained on the IRF.

`save_model` and `load_model` persist a fitted model --- or a data container --- to a versioned, self-describing file backed by the optional `JLD2` package. Coverage spans the VAR, regression, panel, volatility, factor, ARIMA, local-projection, and GMM families plus `TimeSeriesData`, `PanelData`, `CrossSectionData`, and `IOData`. The file records its format version, the package and Julia versions, and any reproducibility manifest; a file whose `format_version` the running build does not recognize is rejected with a `SerializationError` naming the expected version rather than silently misread. Only public fields are stored, so cached factorizations recompute on load and a state-space `builder` closure returns as `nothing`.

```julia
using JLD2                                       # loads the disk backend
save_model(post, "bvar_posterior.jld2")
post_reloaded = load_model("bvar_posterior.jld2")
reproduce(post_reloaded)                         # still reproduces from the persisted seed
```

| Function | Description |
|----------|-------------|
| `capture_manifest(; seed)` | Capture the current environment as a `ReproManifest` |
| `reproduce(result)` | Re-run from the recorded seed and return a `ReproReport` (`matched` true/false/`missing`) |
| `save_model(model, path)` | Persist to a versioned container (requires `using JLD2`) |
| `load_model(path)` | Reconstruct a saved model, validating `format_version` |

---

## Controlling Logging

Library diagnostics flow through the `Logging` standard library, and the package is quiet by default: solver iteration traces are emitted at `@debug`, which the default logger hides. `with_min_level` scopes a minimum severity to one computation, which mutes per-draw warnings inside a bootstrap or Monte Carlo loop while still surfacing a genuine error.

```@example data
using Logging
with_min_level(Logging.Error) do
    estimate_var(to_matrix(d), 2)
end
nothing # hide
```

`set_log_level` moves the global threshold for the rest of the session --- raise it to `:debug` to watch solver iterations.

```julia
set_log_level(:debug)     # show @debug solver iteration traces
set_log_level(:info)      # back to the default verbosity
```

| Function | Effect |
|----------|--------|
| `set_log_level(level)` | Set the global minimum log level (`:debug`, `:info`, `:warn`, `:error`, or a `Logging.LogLevel`) |
| `with_min_level(f, level)` | Run `f()` at a scoped minimum level, restoring the previous logger afterwards |

---

## Complete Example

The full pipeline --- load, transform, diagnose, repair, summarize, validate, estimate, identify --- with the container carrying names and metadata the whole way.

```@example data
# Step 1: load and select
sub_ce = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# Step 2: transform with the codes shipped in the metadata
d_ce = apply_tcode(sub_ce)

# Step 3: diagnose before trusting anything
diagnose(d_ce)
```

```@example data
# Step 4: repair, then confirm the repair
d_clean = fix(d_ce)
(nobs(d_ce), nobs(d_clean), diagnose(d_clean).is_clean)
```

```@example data
# Step 5: describe the stationary sample
describe_data(d_clean)
```

```@example data
# Step 6: check the model requirement, then estimate from the container
validate_for_model(d_clean, :var)
model_ce = estimate_var(d_clean, 2)
report(model_ce)
```

The VAR(2) uses 797 effective observations and is comfortably stationary, with a largest eigenvalue modulus of 0.4970. Industrial production growth is strongly autoregressive at the first lag (0.2765, ``t = 7.72``) and responds positively to the lagged change in inflation. The funds-rate equation is a policy rule in reduced form: it loads on lagged output growth (4.6257, ``t = 2.73``) and on the lagged change in inflation (11.5841, ``t = 1.91``), the Taylor-rule pattern one expects from a monthly sample dominated by the post-1980 period. Residual correlations stay below 0.16, so the Cholesky ordering is not doing much work here.

```julia
# Step 7: structural analysis on the fitted model
irfs = irf(model_ce, 20; method=:cholesky)
plot_result(irfs)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example data
# Step 8: the same pipeline on panel data, one country at a time
for country in ["USA", "GBR", "JPN"]
    gd = group_data(pwt, country)
    log_gdp = filter(isfinite, log.(gd[:, "rgdpna"]))
    report(hp_filter(log_gdp))
end
```

Filtering log real GDP for three countries over 1950-2023 gives cyclical standard deviations of 0.0269 for the United States, 0.0294 for the United Kingdom, and 0.0404 for Japan. Japan's cycle is half again as volatile as the United States', the standard finding for an economy whose post-war catch-up growth and subsequent stagnation leave large low-frequency swings that ``\lambda = 1600`` assigns to the cycle rather than the trend.

---

## Common Pitfalls

1. **Integer row ranges do not index a container.** `d[end-59:end, :]` and `d[1:100, :]` both throw a `MethodError` --- containers are not `AbstractArray`s and define no `axes`. Use `keeprows(d, idx)` or `keeprows(d, mask)` to keep metadata aligned, or take `to_matrix(d)[a:b, :]` and rebuild a container when the metadata no longer matters.

2. **`apply_tcode` leaves NaN behind.** Codes 2, 3, 5, 6, and 7 consume leading observations, and any missing value in the raw series propagates into one to three transformed rows. Always follow `apply_tcode` with `fix` or `dropna`, and check `diagnose(d).is_clean` before estimating --- a NaN reaching an estimator produces a silently invalid result rather than an error.

3. **Code 6 is a change in a growth rate.** `CPIAUCSL` and `M2SL` carry code 6, so `apply_tcode` returns the second log difference, not inflation or money growth. Read `d.tcode` before interpreting any coefficient on a FRED series.

4. **Log codes fall back silently.** Codes 4-7 require strictly positive data. A non-positive value demotes that column to code 2 with a warning and records the substitution in the returned `tcode`. Compare `d.tcode` against the codes you requested to see which columns were demoted.

5. **`PanelData` has no column indexing.** `pd[:, "gdp"]` and `pd[:, ["gdp", "pop"]]` both throw. Select panel variables through the `vars=` keyword of `apply_filter` and `plot_result`, extract one group with `group_data(pd, "USA")` and index that, or work with `to_matrix(pd)` and the positions in `varnames(pd)`.

6. **Panel containers do not dispatch to time series estimators.** `estimate_var(pd, 2)` has no method: a VAR needs a single entity's history. Extract a group first with `group_data`, or use the panel-native estimators on the [Panel Regression](@ref panel_reg_page) page.

7. **`balance_panel` fills values, not rows.** It replaces NaN with DFM-smoothed estimates; it never appends missing periods. `isbalanced` therefore reports the same value before and after when the input already had equal row counts per group, and stays `false` when the input was genuinely ragged.

8. **Filters can unbalance a panel.** Hamilton and Baxter-King trim each group against that group's own valid range, so a balanced panel of unequal-length groups comes back unbalanced. Check `isbalanced` after filtering, or use HP, Beveridge-Nelson, or Boosted HP, which preserve the row count.

9. **`refs` returns a `String`.** The one-argument form builds the bibliography and hands it back rather than printing it, so a bare `refs(d)` displays an escaped string literal. Use `print(refs(d))`, or `refs(stdout, d)`, to render it.

---

## References

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574--589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- McCracken, M. W., & Ng, S. (2021). FRED-QD: A Quarterly Database for Macroeconomic Research.
  *Federal Reserve Bank of St. Louis Review*, 103(1), 1--44. [DOI](https://doi.org/10.20955/r.103.1-44)
  Circulated as Working Paper 2020-005, the version `refs(:fred_qd)` prints.

- Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table.
  *American Economic Review*, 105(10), 3150--3182. [DOI](https://doi.org/10.1257/aer.20130954)

- Acemoglu, D., Naidu, S., Restrepo, P., & Robinson, J. A. (2019). Democracy Does Cause Growth.
  *Journal of Political Economy*, 127(1), 47--100. [DOI](https://doi.org/10.1086/700936)

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods.
  *Journal of Econometrics*, 225(2), 200--230. [DOI](https://doi.org/10.1016/j.jeconom.2020.12.001)

- Grunfeld, Y. (1958). *The Determinants of Corporate Investment*.
  Ph.D. thesis, University of Chicago.

- Mroz, T. A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to
  Economic and Statistical Assumptions. *Econometrica*, 55(4), 765--799.
  [DOI](https://doi.org/10.2307/1911029)

- Brownlee, K. A. (1965). *Statistical Theory and Methodology in Science and Engineering*, 2nd ed.
  Wiley, 491--500.

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed.
  Oxford University Press. ISBN 978-0-19-964117-8.
  [DOI](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)

- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and
  the Business Cycle. *Econometrica*, 57(2), 357--384. [DOI](https://doi.org/10.2307/1912559)

- Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
  *Review of Economics and Statistics*, 100(5), 831--843. [DOI](https://doi.org/10.1162/rest_a_00706)

- Johansen, S., & Juselius, K. (1990). Maximum Likelihood Estimation and Inference on Cointegration
  --- with Applications to the Demand for Money. *Oxford Bulletin of Economics and Statistics*,
  52(2), 169--210. [DOI](https://doi.org/10.1111/j.1468-0084.1990.mp52002003.x)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions*, 2nd ed.
  Cambridge University Press. ISBN 978-0-521-51713-3.
  [DOI](https://doi.org/10.1017/CBO9780511626982)
