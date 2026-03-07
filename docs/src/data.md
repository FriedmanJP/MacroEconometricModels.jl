# [Data Management](@id data_page)

**MacroEconometricModels.jl** provides typed data containers that track metadata, validate inputs, transform series to stationarity, and dispatch directly to estimation functions. The data module eliminates manual bookkeeping between loading raw data and fitting models.

- **Containers**: `TimeSeriesData`, `PanelData`, and `CrossSectionData` wrap numeric matrices with variable names, frequency, transformation codes, and bibliographic references
- **Built-in Datasets**: Five curated datasets --- FRED-MD, FRED-QD, Penn World Table, DDCG democracy panel, and Callaway & Sant'Anna minimum wage panel --- load with a single call
- **Transformations**: FRED transformation codes 1--7 convert raw levels to stationary series; `inverse_tcode` recovers original levels
- **Validation**: `diagnose` detects NaN, Inf, and constant columns; `fix` repairs them via listwise deletion, interpolation, or mean imputation
- **Filtering**: `apply_filter` applies HP, Hamilton, BN, BK, or Boosted HP filters per-variable to extract trend or cycle components
- **Panel Operations**: Stata-style `xtset` for panel construction, within-group lag/lead/diff, group extraction, and balance detection
- **Estimation Dispatch**: All estimators accept `TimeSeriesData` and `PanelData` directly --- no manual conversion required

```@setup data
using MacroEconometricModels, DataFrames
```

## Quick Start

**Recipe 1: Load FRED-MD and explore**

```@example data
# Load the January 2026 vintage (126 variables, 804 months)
fred = load_example(:fred_md)
describe_data(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]])
```

**Recipe 2: Transform to stationarity and clean**

```@example data
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# Apply recommended FRED transformation codes
d = apply_tcode(sub)

# Differencing introduces NaN --- fix by dropping those rows
d_clean = fix(d)
describe_data(d_clean)
```

**Recipe 3: Estimate directly from a data container**

```@example data
d = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))

# No manual to_matrix() needed --- dispatch handles it
model = estimate_var(d, 2)
report(model)
```

**Recipe 4: Panel data with Penn World Table**

```@example data
pwt = load_example(:pwt)
panel_summary(pwt)

# Extract a single country as TimeSeriesData
usa = group_data(pwt, "USA")
describe_data(usa[:, ["rgdpna", "rconna"]])
```

**Recipe 5: Apply filters to data containers**

```@example data
d_filt = TimeSeriesData(
    log.(to_matrix(fred[:, ["INDPRO", "PAYEMS"]]));
    varnames=["INDPRO", "PAYEMS"], frequency=Monthly)
d_filt = fix(d_filt)

# HP cycle extraction for all variables
d_cycle = apply_filter(d_filt, :hp; component=:cycle, lambda=129600.0)
describe_data(d_cycle)
```

**Recipe 6: Panel lag/lead/diff operations**

```@example data
ddcg = load_example(:ddcg)

# Stata-style within-group transformations
lag1_y = panel_lag(ddcg, :y, 1)     # L.y
d_dem = panel_diff(ddcg, :dem)      # D.dem
lead1_y = panel_lead(ddcg, :y, 1)   # F.y
nothing # hide
```

---

## Data Containers

All containers inherit from `AbstractMacroData` and carry metadata alongside the numeric data matrix. The three container types correspond to the three fundamental data structures in applied econometrics: time series, panel, and cross-sectional.

### TimeSeriesData

`TimeSeriesData{T}` is the primary container for single-entity time series. It stores a ``T_{obs} \times n`` data matrix together with variable names, frequency, FRED transformation codes, an integer time index, optional date labels, dataset and per-variable descriptions, and bibliographic references.

```@example data
# From a built-in dataset (recommended)
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# From a matrix with metadata
d_ts = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"],
    frequency=Quarterly,
    tcode=[5, 5, 1],
    time_index=collect(1:200))

# From a vector (univariate)
d_uni = TimeSeriesData(randn(200); varname="GDP", frequency=Monthly)

# From a DataFrame (auto-selects numeric columns, missing becomes NaN)
df = DataFrame(gdp=randn(100), cpi=randn(100), date=1:100)
d_df = TimeSeriesData(df; frequency=Quarterly)
```

Non-float inputs are automatically converted to `Float64`. Missing values in DataFrames become `NaN`.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Matrix{T}` | ``T_{obs} \times n`` data matrix |
| `varnames` | `Vector{String}` | Variable names |
| `frequency` | `Frequency` | Data frequency (informational metadata) |
| `tcode` | `Vector{Int}` | FRED transformation codes per variable (default: all 1) |
| `time_index` | `Vector{Int}` | Integer time identifiers (default: `1:T`) |
| `desc` | `Vector{String}` | Dataset description (length-1 vector for mutability) |
| `vardesc` | `Dict{String,String}` | Per-variable descriptions keyed by variable name |
| `source_refs` | `Vector{Symbol}` | Reference keys for bibliographic citations |
| `dates` | `Vector{String}` | Date labels (default: empty) |

### PanelData

`PanelData{T}` stores stacked panel (longitudinal) data with group and time identifiers. The preferred constructor is `xtset()`, described in the [Panel Data](@ref panel_data_section) section below.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Matrix{T}` | Stacked data matrix (total rows ``\times n``) |
| `varnames` | `Vector{String}` | Variable names |
| `frequency` | `Frequency` | Data frequency |
| `group_id` | `Vector{Int}` | Group identifier per row |
| `time_id` | `Vector{Int}` | Time identifier per row |
| `cohort_id` | `Union{Vector{Int}, Nothing}` | Treatment cohort per row (for DiD) |
| `group_names` | `Vector{String}` | Unique group labels |
| `n_groups` | `Int` | Number of groups |
| `balanced` | `Bool` | True if all groups have the same number of observations |

### CrossSectionData

`CrossSectionData{T}` stores cross-sectional observations (single time point):

```@example data
d_cs = CrossSectionData(randn(500, 4);
    varnames=["income", "education", "age", "hours"])
```

### Frequency Enum

```julia
@enum Frequency Daily Monthly Quarterly Yearly Mixed Other
```

The `frequency` field is informational metadata used in summary displays. It does not affect estimation.

---

## Accessors and Indexing

All data types support a common interface for inspecting dimensions, metadata, and extracting subsets.

```@example data
# Dimensions
nobs(fred)      # 804
nvars(fred)     # 126
size(fred)      # (804, 126)

# Metadata
varnames(fred)     # ["RPI", "W875RX1", ..., "CONSPI"]
frequency(fred)    # Monthly
time_index(fred)   # 1:804

# Column extraction
ip = fred[:, "INDPRO"]                              # Vector{Float64}
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]   # new TimeSeriesData

# Conversion to raw arrays
to_matrix(sub)               # raw T x n matrix
to_vector(sub[:, ["INDPRO"]])  # raw vector (univariate only)
to_vector(sub, "INDPRO")      # single column by name
to_vector(sub, 1)              # single column by index
```

### Renaming Variables

```@example data
d_rn = TimeSeriesData(randn(50, 2); varnames=["a", "b"])
rename_vars!(d_rn, "a" => "GDP")       # single rename
rename_vars!(d_rn, ["output", "prices"])  # replace all
```

`rename_vars!` also updates `vardesc` keys automatically.

### Descriptions

Data containers carry optional metadata descriptions --- one for the dataset itself, and per-variable descriptions accessible by name. Built-in datasets come with descriptions pre-populated.

```@example data
desc(fred)              # "FRED-MD Monthly Database, January 2026 Vintage ..."
vardesc(fred, "INDPRO")  # "IP Index"
vardesc(fred)            # Dict with all variable descriptions

# Set descriptions on custom data
d_desc = TimeSeriesData(randn(100, 2); varnames=["GDP", "CPI"],
    desc="US macroeconomic quarterly data",
    vardesc=Dict("GDP" => "Real GDP growth", "CPI" => "Consumer prices"))

# Modify after construction
set_desc!(d_desc, "Updated description")
set_vardesc!(d_desc, "GDP", "Real Gross Domestic Product")
set_vardesc!(d_desc, Dict("GDP" => "Real GDP", "CPI" => "CPI inflation"))
```

Descriptions propagate through subsetting (`d[:, ["GDP"]]`), transformations (`apply_tcode`), cleaning (`fix`), and panel extraction (`group_data`).

### Date Labels

```@example data
d_dt = TimeSeriesData(randn(4, 2); varnames=["GDP", "CPI"])
set_dates!(d_dt, ["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
dates(d_dt)  # ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]

# Date-based indexing
d_dt["2020Q1", :]                      # row values as vector
d_dt[["2020Q1", "2020Q2"], :]          # sub-TimeSeriesData
```

---

## Visualization

`plot_result()` renders `TimeSeriesData` as multi-panel line charts and `PanelData` as multi-panel charts with one line per group.

### TimeSeriesData Plot

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
d = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
p = plot_result(d)                               # All variables
p = plot_result(d; vars=["INDPRO", "CPIAUCSL"])  # Subset
```

```@raw html
<iframe src="../assets/plots/data_timeseries.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### PanelData Plot

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
p = plot_result(pwt; vars=["rgdpna", "pop", "emp", "hc"])
```

```@raw html
<iframe src="../assets/plots/data_panel.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Panel data plots show each variable in a separate panel with one line per group.

---

## Validation

### Diagnosing Issues

`diagnose()` scans for NaN, Inf, constant columns, and very short series. It returns a `DataDiagnostic` struct summarizing per-variable issues.

```@example data
d_diag = apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]])

diag = diagnose(d_diag)
diag.is_clean      # false --- NaN rows from differencing
diag.n_nan         # NaN count per variable
diag.is_constant   # [false, false, false]
diag.is_short      # false
```

| Field | Type | Description |
|-------|------|-------------|
| `n_nan` | `Vector{Int}` | NaN count per variable |
| `n_inf` | `Vector{Int}` | Inf count per variable |
| `is_constant` | `Vector{Bool}` | True if variable has zero variance |
| `is_short` | `Bool` | True if fewer than 10 observations |
| `is_clean` | `Bool` | True if no issues detected |

### Fixing Issues

`fix()` returns a clean copy using one of three methods:

```@example data
# Drop rows with any NaN/Inf (default)
d_clean2 = fix(d_diag; method=:listwise)

# Linear interpolation for interior NaN, forward-fill edges
d_interp = fix(d_diag; method=:interpolate)

# Replace NaN with column mean of finite values
d_mean = fix(d_diag; method=:mean)
nothing # hide
```

All methods replace Inf with NaN first, then apply the chosen method. Constant columns are dropped automatically with a warning.

!!! note "Technical Note"
    `fix()` always returns a new `TimeSeriesData` object. The original is never modified. After fixing, `diagnose(d_clean).is_clean` is guaranteed to be `true` (unless all columns are constant).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:listwise` | Fix strategy: `:listwise`, `:interpolate`, or `:mean` |

### Model Compatibility

`validate_for_model()` checks dimensionality requirements before estimation:

```@example data
d_multi = TimeSeriesData(randn(100, 3))
d_uni2 = TimeSeriesData(randn(100))

validate_for_model(d_multi, :var)    # OK
validate_for_model(d_uni2, :arima)    # OK
# validate_for_model(d_uni2, :var)   # throws ArgumentError
# validate_for_model(d_multi, :garch)  # throws ArgumentError
nothing # hide
```

| Model Category | Requirement | Model Types |
|----------------|-------------|-------------|
| Multivariate | ``n \geq 2`` | `:var`, `:vecm`, `:bvar`, `:factors`, `:dynamic_factors`, `:gdfm` |
| Univariate | ``n = 1`` | `:arima`, `:ar`, `:ma`, `:arma`, `:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`, `:hp_filter`, `:hamilton_filter`, `:beveridge_nelson`, `:baxter_king`, `:boosted_hp`, `:adf`, `:kpss`, `:pp`, `:za`, `:ngperron` |
| Flexible | any | `:lp`, `:lp_iv`, `:smooth_lp`, `:state_lp`, `:propensity_lp`, `:gmm` |

---

## FRED Transformation Codes

The FRED-MD and FRED-QD databases use integer codes to specify how each series should be transformed to achieve stationarity (McCracken & Ng 2016). `apply_tcode()` implements all seven codes:

```math
\text{tcode 1: } x_t, \quad \text{tcode 2: } \Delta x_t, \quad \text{tcode 3: } \Delta^2 x_t, \quad \text{tcode 4: } \ln x_t
```

```math
\text{tcode 5: } \Delta \ln x_t, \quad \text{tcode 6: } \Delta^2 \ln x_t, \quad \text{tcode 7: } \Delta(x_t / x_{t-1} - 1)
```

where:
- ``x_t`` is the raw series value at time ``t``
- ``\Delta`` is the first-difference operator
- ``\Delta^2`` is the second-difference operator
- ``\ln`` is the natural logarithm

| Code | Transformation | Observations Lost |
|------|---------------|-------------------|
| 1 | Level (no transformation) | 0 |
| 2 | First difference | 1 |
| 3 | Second difference | 2 |
| 4 | Log level | 0 |
| 5 | Log first difference (growth rate) | 1 |
| 6 | Log second difference | 2 |
| 7 | Delta percent change | 2 |

Codes 4--7 require strictly positive data. If a series contains non-positive values with a log-based code, `apply_tcode` falls back to code 2 (first difference) with a warning.

### Applying Transformations

```@example data
# Univariate
y_tc = [100.0, 105.0, 110.0, 108.0, 115.0]
growth = apply_tcode(y_tc, 5)   # log first differences (approx growth rates)

# Apply recommended FRED codes stored in metadata
sub_tc = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]
d_tc = apply_tcode(sub_tc)   # uses per-variable tcode from metadata

# Specify codes explicitly
d2 = apply_tcode(sub_tc, [5, 5, 1])   # log-diff IP and CPI, level FFR

# Same code for all variables
d3 = apply_tcode(sub_tc, 5)
nothing # hide
```

When applying per-variable codes to a `TimeSeriesData`, rows are trimmed consistently to the shortest transformed series, aligning to the end of the sample. For example, if one variable uses code 6 (losing 2 observations) and another uses code 1 (losing none), the output has ``T - 2`` rows for both variables.

### Inverse Transformations

`inverse_tcode()` undoes a transformation given initial values needed to anchor the reconstruction:

```@example data
y_inv = [100.0, 105.0, 110.0, 108.0]
yd = apply_tcode(y_inv, 5)

# Recover original levels
recovered = inverse_tcode(yd, 5; x_prev=[y_inv[1]])
# recovered approx [105.0, 110.0, 108.0]
```

| Code | Required `x_prev` |
|------|-------------------|
| 1, 4 | None |
| 2, 5 | 1 value (last pre-sample level) |
| 3, 6, 7 | 2 values (last two pre-sample levels) |

!!! note "Technical Note"
    Round-trip accuracy (`inverse_tcode(apply_tcode(y, c), c; x_prev=...)`) is exact to machine precision for all codes.

---

## [Panel Data](@id panel_data_section)

### Stata-Style xtset

`xtset()` converts a DataFrame into a `PanelData` container, analogous to Stata's `xtset` command. It extracts all numeric columns (excluding group, time, and cohort columns), sorts by (group, time), validates no duplicate (group, time) pairs, and detects balanced vs unbalanced panels.

```@example data
df_xt = DataFrame(
    firm = repeat(1:50, inner=20),
    year = repeat(2001:2020, 50),
    investment = randn(1000),
    output = randn(1000)
)

pd_xt = xtset(df_xt, :firm, :year; frequency=Yearly)
```

For difference-in-differences estimation, specify a cohort column to encode treatment timing:

```@example data
df_did = DataFrame(
    firm = repeat(1:6, inner=10),
    year = repeat(2001:2010, 6),
    revenue = randn(60),
    treatment_cohort = repeat([1, 1, 2, 2, 0, 0], inner=10)
)
pd_did = xtset(df_did, :firm, :year; cohort=:treatment_cohort)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `varnames` | `Union{Vector{String},Nothing}` | `nothing` | Override variable names (default: column names) |
| `frequency` | `Frequency` | `Other` | Data frequency metadata |
| `tcode` | `Union{Vector{Int},Nothing}` | `nothing` | Transformation codes per variable |
| `cohort` | `Union{Symbol,Nothing}` | `nothing` | Column identifying treatment cohort membership |

### Panel Operations

```@example data
# Structure summary
isbalanced(pwt)       # true
ngroups(pwt)          # 38
groups(pwt)           # ["AUS", "AUT", ..., "USA"]
panel_summary(pwt)    # printed summary table

# Extract single entity as TimeSeriesData
usa2 = group_data(pwt, "USA")       # by name
usa2 = group_data(pwt, 38)          # by index
nothing # hide
```

### Panel Lag, Lead, and Diff

`panel_lag`, `panel_lead`, and `panel_diff` compute within-group transformations that respect panel structure. They return vectors of length ``T_{obs}`` with `NaN` where the operation is unavailable (first observations per group, or time gaps).

```@example data
lag1_y  = panel_lag(ddcg, :y, 1)     # L.y --- one-period lag of GDP
lag4_y  = panel_lag(ddcg, :y, 4)     # L4.y --- four-period lag
lead1_y = panel_lead(ddcg, :y, 1)    # F.y --- one-period lead
d_dem   = panel_diff(ddcg, :dem)     # D.dem --- first difference of democracy

# Append as new columns (returns new PanelData)
ddcg2 = add_panel_lag(ddcg, :y, 1)   # adds "lag1_y" column
ddcg3 = add_panel_diff(ddcg, :dem)   # adds "d_dem" column
nothing # hide
```

### Balance Panel

`balance_panel` fills missing values (NaN) using DFM-based nowcasting (Kalman smoothing) to produce a complete panel:

```@example data
pd_bal = balance_panel(pwt; r=2, p=1)
isbalanced(pd_bal)   # true
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:dfm` | Fill method (currently only `:dfm`) |
| `r` | `Int` | `3` | Number of factors for DFM |
| `p` | `Int` | `2` | VAR lags in DFM factor dynamics |

---

## Summary Statistics

`describe_data()` computes per-variable descriptive statistics and displays them via PrettyTables. For `PanelData`, it additionally prints panel dimensions.

```@example data
d_ss = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
s = describe_data(d_ss)
```

The output table shows N, Mean, Std, Min, P25, Median, P75, Max, Skewness, and Kurtosis for each variable. For INDPRO (log first difference), a positive mean indicates trend growth in industrial production. The skewness and excess kurtosis columns reveal departures from normality common in macroeconomic data.

| Field | Type | Description |
|-------|------|-------------|
| `varnames` | `Vector{String}` | Variable names |
| `n` | `Vector{Int}` | Non-NaN observation count per variable |
| `mean` | `Vector{Float64}` | Mean of finite values |
| `std` | `Vector{Float64}` | Standard deviation |
| `min` | `Vector{Float64}` | Minimum |
| `p25` | `Vector{Float64}` | 25th percentile |
| `median` | `Vector{Float64}` | 50th percentile |
| `p75` | `Vector{Float64}` | 75th percentile |
| `max` | `Vector{Float64}` | Maximum |
| `skewness` | `Vector{Float64}` | Skewness |
| `kurtosis` | `Vector{Float64}` | Excess kurtosis |

---

## Estimation Dispatch

All estimation functions accept `TimeSeriesData` directly via thin dispatch wrappers. This avoids manual conversion and preserves variable names through to the output:

```@example data
d_ed = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))

# Multivariate --- automatically extracts to_matrix(d)
model = estimate_var(d_ed, 2)
post = estimate_bvar(d_ed, 2)
fm = estimate_factors(d_ed, 2)
lp = estimate_lp(d_ed, 1, 20)

# Univariate --- automatically extracts to_vector(d) (requires n_vars == 1)
d_uni_ed = d_ed[:, ["INDPRO"]]
ar = estimate_ar(d_uni_ed, 2)
adf = adf_test(d_uni_ed)
nothing # hide
```

Explicit conversion is also available when working with raw arrays:

```@example data
to_matrix(d_ed)             # Matrix{Float64}
to_vector(d_ed[:, ["INDPRO"]])   # Vector{Float64} (n_vars == 1 only)
to_vector(d_ed, "INDPRO")   # single column by name
to_vector(d_ed, 2)           # single column by index
```

---

## Filtering

`apply_filter()` applies time series filters to variables in a `TimeSeriesData` or `PanelData`, extracting trend or cycle components. When filters produce different-length outputs (e.g., Hamilton drops initial observations), the result is trimmed to the common valid range. For mathematical details on each filter, see [Time Series Filters](@ref filters_page).

### Basic Usage

```@example data
d_fl = TimeSeriesData(
    log.(to_matrix(fred[:, ["INDPRO", "PAYEMS", "HOUST"]]));
    varnames=["INDPRO", "PAYEMS", "HOUST"], frequency=Monthly)
d_fl = fix(d_fl)

# HP cycle for all variables (monthly lambda)
d_hp = apply_filter(d_fl, :hp; component=:cycle, lambda=129600.0)

# HP trend for all variables
d_trend = apply_filter(d_fl, :hp; component=:trend, lambda=129600.0)

# Hamilton filter (output is shorter --- drops initial observations)
d_ham = apply_filter(d_fl, :hamilton; component=:cycle, h=24, p=12)
nothing # hide
```

Available filter symbols: `:hp`, `:hamilton`, `:bn`, `:bk`, `:boosted_hp`.

### Per-Variable Specifications

```@example data
# Different filters per variable (nothing = pass-through)
d2_pv = apply_filter(d_fl, [:hp, :hamilton, nothing]; component=:cycle)

# Per-variable component overrides via tuples
d3_pv = apply_filter(d_fl, [(:hp, :trend), (:hamilton, :cycle), nothing])
nothing # hide
```

### Selective and Panel Filtering

```@example data
# Filter only selected variables (others pass through unchanged)
d_sel = apply_filter(d_fl, :hp; vars=["INDPRO", "PAYEMS"], component=:cycle)

# Panel data: filters applied group-by-group
pd_hp = apply_filter(pwt[:, ["rgdpna", "rconna"]], :hp; component=:cycle)
nothing # hide
```

!!! note "Technical Note"
    Filters that produce shorter output (Hamilton, Baxter-King) trim each group independently. If groups have different lengths, the resulting panel may become unbalanced.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `component` | `Symbol` | `:cycle` | Component to extract: `:cycle` or `:trend` |
| `vars` | `Union{Nothing, Vector{String}, Vector{Int}}` | `nothing` | Variables to filter (default: all) |

Additional keyword arguments are forwarded to the underlying filter functions (e.g., `lambda` for HP, `h` and `p` for Hamilton).

---

## Example Datasets

Five built-in datasets are included, stored as TOML files in the `data/` directory:

| Dataset | Function | Type | Variables | Observations | Frequency |
|---------|----------|------|-----------|--------------|-----------|
| FRED-MD | `load_example(:fred_md)` | `TimeSeriesData` | 126 | 804 months (1959--2025) | Monthly |
| FRED-QD | `load_example(:fred_qd)` | `TimeSeriesData` | 245 | 268 quarters (1959--2025) | Quarterly |
| PWT | `load_example(:pwt)` | `PanelData` | 42 | 38 countries ``\times`` 74 years (1950--2023) | Yearly |
| DDCG | `load_example(:ddcg)` | `PanelData` | 2 | 184 countries ``\times`` 51 years (1960--2010) | Yearly |
| mpdta | `load_example(:mpdta)` | `PanelData` | 3 | 500 counties ``\times`` 5 years (2003--2007) | Yearly |

### FRED Databases

FRED-MD and FRED-QD are January 2026 vintage and include per-variable descriptions and recommended transformation codes from McCracken & Ng (2016, 2020).

```@example data
# Load FRED-MD
md = load_example(:fred_md)
desc(md)                        # "FRED-MD Monthly Database, January 2026 Vintage ..."
vardesc(md, "INDPRO")           # "IP Index"
refs(md)                        # McCracken & Ng (2016)

# Apply recommended FRED transformations to achieve stationarity
md_stationary = apply_tcode(md)

# Estimate a VAR on a subset
sub_md = fix(md_stationary[:, ["INDPRO", "UNRATE", "CPIAUCSL", "FEDFUNDS"]])
model_md = estimate_var(sub_md, 4)
report(model_md)

# Load FRED-QD
qd = load_example(:fred_qd)
desc(qd)                        # "FRED-QD Quarterly Database, January 2026 Vintage ..."
vardesc(qd, "GDPC1")            # "Real Gross Domestic Product, 3 Decimal ..."
refs(qd)                        # McCracken & Ng (2020)
```

### Penn World Table

The Penn World Table (PWT) 10.01 provides a balanced panel of 38 OECD countries over 1950--2023 (Feenstra, Inklaar & Timmer 2015). It loads as `PanelData`, giving access to panel-specific functions.

```@example data
nobs(pwt)                       # 2812 (38 x 74)
nvars(pwt)                      # 42
ngroups(pwt)                    # 38
groups(pwt)                     # ["AUS", "AUT", ..., "USA"]
isbalanced(pwt)                 # true
vardesc(pwt, "rgdpna")          # "Real GDP at constant 2021 national prices ..."
refs(pwt)                       # Feenstra, Inklaar & Timmer (2015)

# Extract a single country as TimeSeriesData
usa_pwt = group_data(pwt, "USA")
nobs(usa_pwt)                   # 74 (years 1950-2023)
panel_summary(pwt)
```

### DDCG Democracy Panel

The DDCG panel from Acemoglu, Naidu, Restrepo & Robinson (2019) contains 184 countries over 1960--2010 with two variables: log GDP per capita (`y`) and a binary democracy indicator (`dem`). It is the standard test dataset for LP-DiD and event study LP methods.

```@example data
nobs(ddcg)      # 9384
ngroups(ddcg)   # 184
panel_summary(ddcg)
```

### Callaway & Sant'Anna Panel (mpdta)

The mpdta dataset from Callaway & Sant'Anna (2021) contains 500 US counties over 2003--2007 with county-level employment (`lemp`, log), population (`lpop`, log), and a staggered treatment indicator for minimum wage changes. It is the reference dataset for the Callaway-Sant'Anna DiD estimator.

```@example data
mpdta = load_example(:mpdta)
nobs(mpdta)      # 2500
ngroups(mpdta)   # 500
panel_summary(mpdta)
```

### Bibliographic References

Each loaded dataset carries bibliographic references accessible via `refs()`, supporting `:text`, `:latex`, `:bibtex`, and `:html` output formats:

```@example data
refs(md; format=:bibtex)   # BibTeX entry for McCracken & Ng (2016)
refs(:fred_md)             # same via symbol dispatch
refs(:pwt)                 # Feenstra, Inklaar & Timmer (2015)
```

---

## Complete Example

This example demonstrates a full data pipeline: loading FRED-MD, diagnosing and cleaning, summarizing, estimating a VAR, and performing structural analysis --- all using data containers without manual conversion.

```@example data
# Step 1: Load FRED-MD and select key macro variables
sub_ce = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# Step 2: Apply FRED transformation codes to achieve stationarity
d_ce = apply_tcode(sub_ce)

# Step 3: Diagnose --- differencing introduces NaN in early rows
diag_ce = diagnose(d_ce)
diag_ce.is_clean   # false

# Step 4: Fix by dropping NaN rows
d_clean_ce = fix(d_ce)
diagnose(d_clean_ce).is_clean   # true

# Step 5: Summary statistics
describe_data(d_clean_ce)

# Step 6: Validate for VAR estimation
validate_for_model(d_clean_ce, :var)   # OK --- multivariate

# Step 7: Estimate VAR directly from container
model_ce = estimate_var(d_clean_ce, 2)
report(model_ce)
```

```julia
# Step 8: Structural analysis --- Cholesky-identified IRFs
irfs = irf(model_ce, 20; method=:cholesky)
plot_result(irfs)
```

```@example data
# Step 9: Panel workflow with Penn World Table
panel_summary(pwt)

# Extract and analyze per country
for country in ["USA", "GBR", "JPN"]
    gd = group_data(pwt, country)
    y_hp = filter(isfinite, log.(gd[:, "rgdpna"]))
    hp = hp_filter(y_hp)
    report(hp)
end
```

The pipeline starts with `load_example(:fred_md)`, which returns a `TimeSeriesData` with 126 variables, transformation codes, and descriptions pre-loaded. `apply_tcode(sub)` applies the per-variable codes stored in `tcode` (code 5 for INDPRO and CPIAUCSL, code 1 for FEDFUNDS), producing log growth rates for the first two and leaving the federal funds rate in levels. The differencing step introduces NaN in the first two rows, which `fix(d)` removes via listwise deletion. The cleaned container passes directly to `estimate_var`, which extracts the data matrix and variable names automatically. The Penn World Table loop demonstrates extracting individual countries from a panel container for univariate analysis.

---

## Common Pitfalls

1. **NaN from differencing**: `apply_tcode` with codes 2, 3, 5, 6, or 7 produces NaN in the first 1--2 rows because differencing requires prior values. Always follow `apply_tcode` with `fix(d)` or manually drop NaN rows before estimation. Passing data with NaN to `estimate_var` or other estimators produces invalid results silently.

2. **Log of non-positive values**: Codes 4--7 require strictly positive data. If a series contains zeros or negatives, `apply_tcode` falls back to code 2 (first difference) with a warning. Check `d.tcode` after transformation to verify which codes were actually applied.

3. **Panel vs time series dispatch**: Passing a `PanelData` to `estimate_var` does not work --- VAR estimators expect `TimeSeriesData` or raw matrices. Extract a single group with `group_data(pd, "USA")` first, then estimate.

4. **Forgetting to filter NaN rows**: `diagnose(d).is_clean` returns `false` if any NaN exists. Do not assume `apply_tcode` produces clean output. The `fix` step is not optional for downstream estimation.

5. **tcode metadata mismatch after subsetting**: When subsetting columns with `d[:, ["INDPRO", "FEDFUNDS"]]`, the `tcode` vector is automatically sliced to match. However, if you construct `TimeSeriesData` manually from a subset of columns, you must provide the correct `tcode` vector yourself.

6. **Unbalanced panels from filtering**: Applying Hamilton or Baxter-King filters to `PanelData` via `apply_filter` trims each group independently. If groups have different time spans, the resulting panel becomes unbalanced even if the input was balanced.

---

## References

- McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
  *Journal of Business & Economic Statistics*, 34(4), 574--589. [DOI](https://doi.org/10.1080/07350015.2015.1086655)

- McCracken, M. W., & Ng, S. (2020). FRED-QD: A Quarterly Database for Macroeconomic Research.
  *Federal Reserve Bank of St. Louis Working Paper*, 2020-005. [DOI](https://doi.org/10.20955/wp.2020.005)

- Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table.
  *American Economic Review*, 105(10), 3150--3182. [DOI](https://doi.org/10.1257/aer.20130954)

- Acemoglu, D., Naidu, S., Restrepo, P., & Robinson, J. A. (2019). Democracy Does Cause Growth.
  *Journal of Political Economy*, 127(1), 47--100. [DOI](https://doi.org/10.1086/700936)

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods.
  *Journal of Econometrics*, 225(2), 200--230. [DOI](https://doi.org/10.1016/j.jeconom.2020.12.001)
