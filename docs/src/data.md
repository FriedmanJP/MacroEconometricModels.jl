# Data Management

Applied macroeconometric research begins with data. This module provides typed data containers that track metadata (frequency, variable names, panel structure, transformation codes), validate inputs, compute summary statistics, and guarantee clean data for estimation.

| Feature | Function | Description |
|---------|----------|-------------|
| **Containers** | `TimeSeriesData`, `PanelData`, `CrossSectionData` | Typed wrappers with metadata |
| **Validation** | `diagnose`, `fix` | Detect and repair NaN, Inf, constant columns |
| **Transforms** | `apply_tcode`, `inverse_tcode` | FRED transformation codes 1--7 |
| **Filtering** | `apply_filter` | Apply HP/Hamilton/BN/BK/Boosted HP per-variable |
| **Panel** | `xtset`, `group_data` | Stata-style panel setup and slicing |
| **Summary** | `describe_data` | Per-variable descriptive statistics |
| **Dispatch** | `estimate_var(d, p)` | All estimators accept `TimeSeriesData` directly |

## Quick Start

```julia
using MacroEconometricModels

# Load FRED-MD and select key macro variables
fred = load_example(:fred_md)
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# Apply FRED transformation codes to achieve stationarity
d = apply_tcode(sub)

# Diagnose — apply_tcode introduces NaN from differencing
diag = diagnose(d)

# Fix by dropping NaN rows
d_clean = fix(d)

# Summary statistics
describe_data(d_clean)

# Estimate directly from data container
model = estimate_var(d_clean, 2)

# Panel data — Penn World Table
pwt = load_example(:pwt)
panel_summary(pwt)
usa = group_data(pwt, "USA")   # extract single country
```

---

## Data Containers

All containers inherit from `AbstractMacroData` and carry metadata alongside the numeric data matrix.

### TimeSeriesData

`TimeSeriesData{T}` is the primary container for single-entity time series:

```julia
# Load built-in dataset (recommended)
fred = load_example(:fred_md)   # 804 obs × 126 vars (Monthly)
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# From matrix with metadata
d = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"],
    frequency=Quarterly,
    tcode=[5, 5, 1],
    time_index=collect(1959:2158))

# From vector (univariate)
d = TimeSeriesData(randn(200); varname="GDP", frequency=Monthly)

# From DataFrame (auto-selects numeric columns)
using DataFrames
df = DataFrame(gdp=randn(100), cpi=randn(100), date=1:100)
d = TimeSeriesData(df; frequency=Quarterly)
```

Non-float inputs are automatically converted to `Float64`. Missing values in DataFrames become `NaN`.

### Frequency Enum

```julia
@enum Frequency Daily Monthly Quarterly Yearly Mixed Other
```

The `frequency` field is informational metadata used in summary displays. It does not affect estimation.

### PanelData

`PanelData{T}` stores stacked panel (longitudinal) data with group and time identifiers. Constructed via `xtset()`:

```julia
# Load Penn World Table (balanced panel, 38 OECD countries × 74 years)
pwt = load_example(:pwt)

# Or construct from a DataFrame via xtset
using DataFrames
df = DataFrame(
    country = repeat(["US", "UK", "JP"], inner=50),
    quarter = repeat(1:50, 3),
    gdp = randn(150),
    cpi = randn(150)
)
pd = xtset(df, :country, :quarter; frequency=Quarterly)
```

### CrossSectionData

`CrossSectionData{T}` stores cross-sectional observations (single time point):

```julia
d = CrossSectionData(randn(500, 4);
    varnames=["income", "education", "age", "hours"])
```

---

## Descriptions

Data containers carry optional metadata descriptions — one for the dataset itself, and per-variable descriptions accessible by name. These are empty by default and only populated if the user provides them.

### Setting at Construction

```julia
# Built-in datasets already carry descriptions
fred = load_example(:fred_md)
desc(fred)              # "FRED-MD Monthly Database, January 2026 Vintage ..."
vardesc(fred, "INDPRO")  # "IP Index"

# Or set manually at construction
d = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"],
    frequency=Quarterly,
    desc="US macroeconomic quarterly data 1959-2024",
    vardesc=Dict(
        "GDP" => "Real Gross Domestic Product, seasonally adjusted annual rate",
        "CPI" => "Consumer Price Index for All Urban Consumers",
        "FFR" => "Effective Federal Funds Rate"))
```

### Accessing Descriptions

```julia
fred = load_example(:fred_md)
desc(fred)              # "FRED-MD Monthly Database, January 2026 Vintage ..."
vardesc(fred, "INDPRO")  # "IP Index"
vardesc(fred)            # Dict with all variable descriptions
```

### Setting After Construction

```julia
d = TimeSeriesData(randn(100, 2); varnames=["GDP", "CPI"])
set_desc!(d, "Updated dataset description")
set_vardesc!(d, "GDP", "Real GDP growth rate")
set_vardesc!(d, Dict("GDP" => "Real GDP", "CPI" => "Consumer prices"))
```

Descriptions propagate through subsetting (`d[:, ["GDP"]]`), transformations (`apply_tcode`), cleaning (`fix`), and panel extraction (`group_data`). Renaming variables (`rename_vars!`) automatically updates `vardesc` keys.

---

## Accessors and Indexing

All data types support a common interface:

```julia
fred = load_example(:fred_md)

# Dimensions
nobs(fred)      # 804
nvars(fred)     # 126
size(fred)      # (804, 126)

# Metadata
varnames(fred)     # ["RPI", "W875RX1", ..., "CONSPI"]
frequency(fred)    # Monthly
time_index(fred)   # 1:804

# Column extraction
ip = fred[:, "INDPRO"]                          # Vector{Float64}
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]  # new TimeSeriesData with 3 variables

# Conversion
to_matrix(sub)    # raw T x n matrix
to_vector(fred[:, ["INDPRO"]])   # raw vector (univariate only)
```

### Renaming Variables

```julia
d = TimeSeriesData(randn(50, 2); varnames=["a", "b"])
rename_vars!(d, "a" => "GDP")
rename_vars!(d, ["output", "prices"])
```

### Time Index

```julia
d = TimeSeriesData(randn(50, 1))
set_time_index!(d, collect(1970:2019))
time_index(d)  # [1970, 1971, ..., 2019]
```

---

## Validation

### Diagnosing Issues

`diagnose()` scans for NaN, Inf, constant columns, and very short series:

```julia
# Transforming FRED-MD introduces NaN from differencing and log of non-positive values
fred = load_example(:fred_md)
d = apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]])

diag = diagnose(d)
diag.is_clean     # false — NaN rows from differencing
diag.n_nan        # number of NaN per variable
diag.is_constant  # [false, false, false]
diag.is_short     # false
```

### Fixing Issues

`fix()` returns a clean copy using one of three methods:

```julia
# Drop rows with any NaN/Inf (default)
d_clean = fix(d; method=:listwise)

# Linear interpolation for interior NaN, forward-fill edges
d_clean = fix(d; method=:interpolate)

# Replace NaN with column mean of finite values
d_clean = fix(d; method=:mean)
```

All methods replace Inf with NaN first, then apply the chosen method. Constant columns are dropped automatically with a warning.

!!! note "Technical Note"
    `fix()` always returns a new `TimeSeriesData` object. The original is never modified. After fixing, `diagnose(d_clean).is_clean` is guaranteed to be `true` (unless all columns are constant).

### Model Compatibility

`validate_for_model()` checks dimensionality requirements:

```julia
d_multi = TimeSeriesData(randn(100, 3))
d_uni = TimeSeriesData(randn(100))

validate_for_model(d_multi, :var)    # OK
validate_for_model(d_uni, :arima)    # OK
validate_for_model(d_uni, :var)      # throws ArgumentError
validate_for_model(d_multi, :garch)  # throws ArgumentError
```

| Model Category | Requirement | Model Types |
|----------------|-------------|-------------|
| Multivariate | ``n \geq 2`` | `:var`, `:vecm`, `:bvar`, `:factors`, `:dynamic_factors`, `:gdfm` |
| Univariate | ``n = 1`` | `:arima`, `:ar`, `:ma`, `:arma`, `:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`, `:hp_filter`, `:hamilton_filter`, `:beveridge_nelson`, `:baxter_king`, `:boosted_hp`, `:adf`, `:kpss`, `:pp`, `:za`, `:ngperron` |
| Flexible | any | `:lp`, `:lp_iv`, `:smooth_lp`, `:state_lp`, `:propensity_lp`, `:gmm` |

---

## FRED Transformation Codes

The FRED-MD database uses integer codes to specify how each series should be transformed to achieve stationarity. `apply_tcode()` implements all seven codes:

| Code | Transformation | Formula | Observations Lost |
|------|---------------|---------|-------------------|
| 1 | Level | ``x_t`` | 0 |
| 2 | First difference | ``\Delta x_t`` | 1 |
| 3 | Second difference | ``\Delta^2 x_t`` | 2 |
| 4 | Log | ``\ln x_t`` | 0 |
| 5 | Log first difference | ``\Delta \ln x_t`` | 1 |
| 6 | Log second difference | ``\Delta^2 \ln x_t`` | 2 |
| 7 | Delta percent change | ``\Delta(x_t / x_{t-1} - 1)`` | 2 |

Codes 4--7 require strictly positive data.

### Applying Transformations

```julia
# Univariate
y = [100.0, 105.0, 110.0, 108.0, 115.0]
growth = apply_tcode(y, 5)   # log first differences

# Apply recommended FRED codes to data container
fred = load_example(:fred_md)
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]
d = apply_tcode(sub)   # uses per-variable tcode from metadata

# Or specify codes explicitly
d2 = apply_tcode(sub, [5, 5, 1])   # log-diff IP and CPI, level FFR

# Same code for all variables
d3 = apply_tcode(sub, 5)
```

When applying per-variable codes to a `TimeSeriesData`, rows are trimmed consistently to the shortest transformed series, aligning to the end of the sample.

### Inverse Transformations

`inverse_tcode()` undoes a transformation given initial values:

```julia
y = [100.0, 105.0, 110.0, 108.0]
yd = apply_tcode(y, 5)

# Recover original levels
recovered = inverse_tcode(yd, 5; x_prev=[y[1]])
# recovered ≈ [105.0, 110.0, 108.0]
```

The `x_prev` argument provides the initial values needed to anchor the reconstruction:

| Code | Required `x_prev` |
|------|-------------------|
| 1, 4 | None |
| 2, 5 | 1 value (last pre-sample level) |
| 3, 6, 7 | 2 values (last two pre-sample levels) |

!!! note "Technical Note"
    Round-trip accuracy (`inverse_tcode(apply_tcode(y, c), c; x_prev=...)`) is exact to machine precision for all codes.

---

## Panel Data

### Stata-style xtset

`xtset()` converts a DataFrame into a `PanelData` container, analogous to Stata's `xtset` command:

```julia
# The preferred way to get panel data is load_example(:pwt)
# For custom DataFrames, use xtset:
using DataFrames

df = DataFrame(
    firm = repeat(1:50, inner=20),
    year = repeat(2001:2020, 50),
    investment = randn(1000),
    output = randn(1000)
)

pd = xtset(df, :firm, :year; frequency=Yearly)
```

The function:
- Extracts all numeric columns (excluding group and time columns)
- Sorts by (group, time)
- Validates no duplicate (group, time) pairs
- Detects balanced vs unbalanced panels

### Panel Operations

```julia
pwt = load_example(:pwt)

# Structure summary
isbalanced(pwt)       # true
ngroups(pwt)          # 38
groups(pwt)           # ["AUS", "AUT", ..., "USA"]
panel_summary(pwt)    # printed summary table

# Extract single entity as TimeSeriesData
usa = group_data(pwt, "USA")       # by name
usa = group_data(pwt, 38)          # by index
```

---

## Summary Statistics

`describe_data()` computes per-variable descriptive statistics displayed via PrettyTables:

```julia
fred = load_example(:fred_md)
d = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
s = describe_data(d)
```

The returned `DataSummary` object contains fields: `varnames`, `n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `max`, `skewness`, `kurtosis`.

For `PanelData`, `describe_data()` additionally prints panel dimensions.

---

## Estimation Dispatch

All estimation functions accept `TimeSeriesData` directly via thin dispatch wrappers. This avoids manual conversion:

```julia
fred = load_example(:fred_md)
d = fix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))

# Multivariate — automatically calls to_matrix(d)
model = estimate_var(d, 2)
post = estimate_bvar(d, 2)
fm = estimate_factors(d, 2)
lp = estimate_lp(d, 1, 20)

# Univariate — automatically calls to_vector(d) (requires n_vars == 1)
d_uni = d[:, ["INDPRO"]]  # select single variable
ar = estimate_ar(d_uni, 2)
adf = adf_test(d_uni)
```

Explicit conversion is also available:

```julia
to_matrix(d)             # Matrix{Float64}
to_vector(d)             # Vector{Float64} (n_vars == 1 only)
to_vector(d, "INDPRO")   # single column by name
to_vector(d, 2)          # single column by index
```

---

## Example Datasets

Three built-in datasets are included, stored as TOML files in the `data/` directory:

| Dataset | Function | Type | Variables | Observations | Frequency |
|---------|----------|------|-----------|--------------|-----------|
| FRED-MD | `load_example(:fred_md)` | `TimeSeriesData` | 126 | 804 months (1959--2025) | Monthly |
| FRED-QD | `load_example(:fred_qd)` | `TimeSeriesData` | 245 | 268 quarters (1959--2025) | Quarterly |
| PWT | `load_example(:pwt)` | `PanelData` | 42 | 38 countries × 74 years (1950--2023) | Yearly |

FRED-MD and FRED-QD are January 2026 vintage and include per-variable descriptions and recommended transformation codes from McCracken and Ng. PWT 10.01 covers 38 OECD countries with national accounts, productivity, and price level data from Feenstra, Inklaar, and Timmer.

### FRED Databases

```julia
# Load FRED-MD
md = load_example(:fred_md)
md                              # 804 obs × 126 vars (Monthly)
desc(md)                        # "FRED-MD Monthly Database, January 2026 Vintage ..."
vardesc(md, "INDPRO")           # "IP Index"
refs(md)                        # McCracken & Ng (2016)

# Apply recommended FRED transformations to achieve stationarity
md_stationary = apply_tcode(md, md.tcode)

# Estimate a VAR on a subset
sub = md_stationary[:, ["INDPRO", "UNRATE", "CPIAUCSL", "FEDFUNDS"]]
model = estimate_var(sub, 4)

# Load FRED-QD
qd = load_example(:fred_qd)
desc(qd)                        # "FRED-QD Quarterly Database, January 2026 Vintage ..."
vardesc(qd, "GDPC1")            # "Real Gross Domestic Product, 3 Decimal ..."
refs(qd)                        # McCracken & Ng (2020)
```

### Penn World Table

The Penn World Table (PWT) 10.01 provides a balanced panel of 38 OECD countries over 1950--2023. It loads as `PanelData`, giving access to panel-specific functions like `group_data`, `groups`, and `panel_summary`.

```julia
# Load PWT
pwt = load_example(:pwt)
nobs(pwt)                       # 2812 (38 × 74)
nvars(pwt)                      # 42
ngroups(pwt)                    # 38
groups(pwt)                     # ["AUS", "AUT", ..., "USA"]
isbalanced(pwt)                 # true
vardesc(pwt, "rgdpna")          # "Real GDP at constant 2021 national prices ..."
refs(pwt)                       # Feenstra, Inklaar & Timmer (2015)

# Extract a single country as TimeSeriesData
usa = group_data(pwt, "USA")
nobs(usa)                       # 74 (years 1950–2023)

# Run a VAR on US real GDP, consumption, and investment
rgdpna = usa[:, "rgdpna"]
rconna = usa[:, "rconna"]
# Panel summary
panel_summary(pwt)
```

### References

Each loaded dataset carries bibliographic references accessible via `refs()`, supporting `:text`, `:latex`, `:bibtex`, and `:html` output formats:

```julia
refs(md; format=:bibtex)   # BibTeX entry for McCracken & Ng (2016)
refs(:fred_md)             # same via symbol dispatch
refs(:pwt)                 # Feenstra, Inklaar & Timmer (2015)
```

---

## Filtering

`apply_filter()` applies time series filters (HP, Hamilton, BN, BK, Boosted HP) to variables in a `TimeSeriesData` or `PanelData`, extracting trend or cycle components. When filters produce different-length outputs (e.g., Hamilton drops initial observations), the result is trimmed to the common valid range.

### Basic Usage

```julia
# Log levels from FRED-MD (I(1) series, suitable for trend-cycle decomposition)
fred = load_example(:fred_md)
d = TimeSeriesData(
    log.(to_matrix(fred[:, ["INDPRO", "PAYEMS", "HOUST"]]));
    varnames=["INDPRO", "PAYEMS", "HOUST"], frequency=Monthly)
# Drop any NaN from log of non-positive values
d = fix(d)

# HP cycle for all variables (monthly lambda)
d_hp = apply_filter(d, :hp; component=:cycle, lambda=129600.0)

# HP trend for all variables
d_trend = apply_filter(d, :hp; component=:trend, lambda=129600.0)

# Hamilton filter (output is shorter — drops initial observations)
d_ham = apply_filter(d, :hamilton; component=:cycle, h=24, p=12)
```

Available filter symbols: `:hp`, `:hamilton`, `:bn`, `:bk`, `:boosted_hp`.

### Per-Variable Specifications

```julia
# Different filters per variable (nothing = pass-through)
d2 = apply_filter(d, [:hp, :hamilton, nothing]; component=:cycle)

# Per-variable component overrides via tuples
d3 = apply_filter(d, [(:hp, :trend), (:hamilton, :cycle), nothing])
```

### Selective Filtering

```julia
# Filter only selected variables (others pass through unchanged)
d_sel = apply_filter(d, :hp; vars=["INDPRO", "PAYEMS"], component=:cycle)
d_sel = apply_filter(d, :hp; vars=[1, 2], component=:cycle)  # by index
```

### Pre-Computed Results

```julia
# Use a pre-computed filter result
r = hp_filter(d[:, "INDPRO"]; lambda=129600.0)
d2 = apply_filter(d, [r, :hp, nothing]; component=:cycle)
```

### Forwarding Filter Parameters

Additional keyword arguments are forwarded to the filter functions:

```julia
# Custom HP lambda
d_smooth = apply_filter(d, :hp; component=:cycle, lambda=100.0)

# Custom Hamilton horizon and lags
d_ham = apply_filter(d, :hamilton; component=:cycle, h=24, p=12)
```

### Panel Data

`apply_filter` applies filters group-by-group to `PanelData`, reassembling the results:

```julia
# Penn World Table — real GDP and consumption for 38 OECD countries
pwt = load_example(:pwt)

# HP cycle for all variables, applied per-group
pd_hp = apply_filter(pwt[:, ["rgdpna", "rconna"]], :hp; component=:cycle)

# Filter only rgdpna, pass through rconna
pd_sel = apply_filter(pwt[:, ["rgdpna", "rconna"]], :hp; vars=["rgdpna"], component=:cycle)
```

!!! note "Technical Note"
    Filters that produce shorter output (Hamilton, Baxter-King) trim each group independently. If groups have different lengths, the resulting panel may become unbalanced.

---

## Complete Example

```julia
using MacroEconometricModels

# === Step 1: Load FRED-MD and select variables ===
fred = load_example(:fred_md)
sub = fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]

# === Step 2: Apply FRED transformation codes ===
d = apply_tcode(sub)

# === Step 3: Diagnose — differencing introduces NaN ===
diag = diagnose(d)
println("Clean: ", diag.is_clean)   # false

# === Step 4: Fix by dropping NaN rows ===
d_clean = fix(d)
println("Clean: ", diagnose(d_clean).is_clean)   # true

# === Step 5: Summary statistics ===
describe_data(d_clean)

# === Step 6: Validate for VAR ===
validate_for_model(d_clean, :var)   # OK — multivariate

# === Step 7: Estimate VAR directly from container ===
model = estimate_var(d_clean, 2)

# === Step 8: Structural analysis ===
irfs = irf(model, 20; method=:cholesky)

# === Step 9: Panel workflow with Penn World Table ===
pwt = load_example(:pwt)
panel_summary(pwt)

# Extract and estimate per country
for country in ["USA", "GBR", "JPN"]
    gd = group_data(pwt, country)
    y = filter(isfinite, log.(gd[:, "rgdpna"]))
    hp = hp_filter(y)
    println("$country: trend length = ", length(trend(hp)))
end
```

### See Also

- [Time Series Filters](filters.md) -- HP, Hamilton, BN, BK, and boosted HP filters used by `apply_filter`
- [Examples](examples.md) -- Complete worked examples including FRED-MD data pipeline
- [API Reference](api_functions.md) -- Complete function signatures

## References

- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574--589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- McCracken, Michael W., and Serena Ng. 2020. "FRED-QD: A Quarterly Database for Macroeconomic Research." *Federal Reserve Bank of St. Louis Working Paper* 2020-005. [https://doi.org/10.20955/wp.2020.005](https://doi.org/10.20955/wp.2020.005)
