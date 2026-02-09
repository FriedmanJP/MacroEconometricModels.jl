# Data Management

Applied macroeconometric research begins with data. This module provides typed data containers that track metadata (frequency, variable names, panel structure, transformation codes), validate inputs, compute summary statistics, and guarantee clean data for estimation.

| Feature | Function | Description |
|---------|----------|-------------|
| **Containers** | `TimeSeriesData`, `PanelData`, `CrossSectionData` | Typed wrappers with metadata |
| **Validation** | `diagnose`, `fix` | Detect and repair NaN, Inf, constant columns |
| **Transforms** | `apply_tcode`, `inverse_tcode` | FRED transformation codes 1--7 |
| **Panel** | `xtset`, `group_data` | Stata-style panel setup and slicing |
| **Summary** | `describe_data` | Per-variable descriptive statistics |
| **Dispatch** | `estimate_var(d, p)` | All estimators accept `TimeSeriesData` directly |

## Quick Start

```julia
using MacroEconometricModels, Random, DataFrames

Random.seed!(42)

# Create time series data with metadata
d = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"], frequency=Quarterly)

# Diagnose and fix
diag = diagnose(d)
diag.is_clean  # true

# FRED transformations (log-diff GDP and CPI, leave FFR in levels)
d2 = apply_tcode(d, [5, 5, 1])

# Summary statistics
describe_data(d2)

# Estimate directly from data container
model = estimate_var(d2, 2)

# Panel data
df = DataFrame(id=repeat(1:3, inner=50), t=repeat(1:50, 3),
               x=randn(150), y=randn(150))
pd = xtset(df, :id, :t)
panel_summary(pd)
g1 = group_data(pd, 1)   # extract first entity
```

---

## Data Containers

All containers inherit from `AbstractMacroData` and carry metadata alongside the numeric data matrix.

### TimeSeriesData

`TimeSeriesData{T}` is the primary container for single-entity time series:

```julia
# From matrix (auto-generates variable names)
d = TimeSeriesData(randn(100, 3))

# With full metadata
d = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"],
    frequency=Quarterly,
    tcode=[5, 5, 1],
    time_index=collect(1959:2158))

# From vector (univariate)
d = TimeSeriesData(randn(200); varname="GDP", frequency=Monthly)

# From DataFrame (auto-selects numeric columns)
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
desc(d)            # "US macroeconomic quarterly data 1959-2024"
vardesc(d, "GDP")  # "Real Gross Domestic Product, seasonally adjusted annual rate"
vardesc(d)         # Dict with all variable descriptions
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
d = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "FFR"])

# Dimensions
nobs(d)      # 100
nvars(d)     # 3
size(d)      # (100, 3)

# Metadata
varnames(d)     # ["GDP", "CPI", "FFR"]
frequency(d)    # Other
time_index(d)   # 1:100

# Column extraction
gdp = d[:, "GDP"]          # Vector{Float64}
sub = d[:, ["GDP", "FFR"]] # new TimeSeriesData with 2 variables

# Conversion
Matrix(d)    # raw T x n matrix
Vector(d)    # raw vector (univariate only)
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
mat = randn(100, 3)
mat[5, 1] = NaN
mat[10, 2] = Inf
d = TimeSeriesData(mat; varnames=["GDP", "CPI", "FFR"])

diag = diagnose(d)
diag.is_clean     # false
diag.n_nan        # [1, 0, 0]
diag.n_inf        # [0, 1, 0]
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
y = [100.0, 105.0, 110.0, 108.0, 115.0]

# Univariate
growth = apply_tcode(y, 5)   # log first differences

# Per-variable on data container
d = TimeSeriesData(rand(200, 3) .+ 1.0; varnames=["GDP", "CPI", "FFR"])
d2 = apply_tcode(d, [5, 5, 1])   # log-diff GDP and CPI, level FFR

# Same code for all variables
d3 = apply_tcode(d, 5)
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
# Structure summary
isbalanced(pd)       # true/false
ngroups(pd)          # number of entities
groups(pd)           # entity labels
panel_summary(pd)    # printed summary table

# Extract single entity as TimeSeriesData
firm1 = group_data(pd, 1)       # by index
firm1 = group_data(pd, "1")     # by name
estimate_ar(firm1, 2)            # estimate AR(2) for firm 1
```

---

## Summary Statistics

`describe_data()` computes per-variable descriptive statistics displayed via PrettyTables:

```julia
d = TimeSeriesData(randn(200, 3);
    varnames=["GDP", "CPI", "FFR"], frequency=Quarterly)
s = describe_data(d)
```

```
Summary Statistics — Quarterly (200 × 3)
 Variable     N    Mean     Std      Min    P25   Median    P75     Max   Skew   Kurt
 GDP        200  -0.014   0.985  -2.914  -0.712  -0.060   0.619   2.543  0.032 -0.099
 CPI        200   0.040   1.019  -2.545  -0.619   0.067   0.689   2.781  0.005  0.004
 FFR        200  -0.011   0.989  -2.879  -0.675  -0.002   0.650   3.063  0.063  0.076
```

The returned `DataSummary` object contains fields: `varnames`, `n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `max`, `skewness`, `kurtosis`.

For `PanelData`, `describe_data()` additionally prints panel dimensions.

---

## Estimation Dispatch

All estimation functions accept `TimeSeriesData` directly via thin dispatch wrappers. This avoids manual conversion:

```julia
d = TimeSeriesData(randn(200, 3); varnames=["y1", "y2", "y3"])

# Multivariate — automatically calls to_matrix(d)
model = estimate_var(d, 2)
vecm = estimate_vecm(d, 2; rank=:auto)
post = estimate_bvar(d, 2)
fm = estimate_factors(d, 2)
lp = estimate_lp(d, 1, 20)

# Univariate — automatically calls to_vector(d) (requires n_vars == 1)
d_uni = d[:, ["y1"]]  # select single variable
ar = estimate_ar(d_uni, 2)
hp = hp_filter(d_uni)
adf = adf_test(d_uni)
```

Explicit conversion is also available:

```julia
to_matrix(d)         # Matrix{Float64}
to_vector(d)         # Vector{Float64} (n_vars == 1 only)
to_vector(d, "y1")   # single column by name
to_vector(d, 2)      # single column by index
```

---

## Example Datasets

Two FRED databases are included as built-in example datasets, stored as TOML files in the `data/` directory:

| Dataset | Function | Variables | Observations | Frequency |
|---------|----------|-----------|--------------|-----------|
| FRED-MD | `load_example(:fred_md)` | 126 | 804 months (1959--2025) | Monthly |
| FRED-QD | `load_example(:fred_qd)` | 245 | 268 quarters (1959--2025) | Quarterly |

Both datasets are January 2026 vintage and include per-variable descriptions and recommended transformation codes from McCracken and Ng.

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

Each loaded dataset carries bibliographic references accessible via `refs()`, supporting `:text`, `:latex`, `:bibtex`, and `:html` output formats:

```julia
refs(md; format=:bibtex)   # BibTeX entry for McCracken & Ng (2016)
refs(:fred_md)             # same via symbol dispatch
```

---

## Complete Example

```julia
using MacroEconometricModels, Random, DataFrames

Random.seed!(42)

# === Step 1: Create data with metadata ===
Y = randn(200, 3)
for t in 2:200
    Y[t, :] = 0.6 * Y[t-1, :] + 0.3 * randn(3)
end
d = TimeSeriesData(Y; varnames=["GDP", "INF", "FFR"], frequency=Quarterly)

# === Step 2: Diagnose ===
diag = diagnose(d)
println("Clean: ", diag.is_clean)   # true

# === Step 3: Summary statistics ===
describe_data(d)

# === Step 4: Validate for VAR ===
validate_for_model(d, :var)   # OK — multivariate

# === Step 5: Estimate VAR directly from container ===
model = estimate_var(d, 2)

# === Step 6: Structural analysis ===
irfs = irf(model, 20; method=:cholesky)

# === Step 7: Panel workflow ===
df = DataFrame(
    id = repeat(1:3, inner=50),
    t  = repeat(1:50, 3),
    x  = randn(150),
    y  = randn(150)
)
pd = xtset(df, :id, :t; frequency=Quarterly)
panel_summary(pd)

# Extract and estimate per entity
for g in 1:ngroups(pd)
    gd = group_data(pd, g)
    ar = estimate_ar(gd[:, ["x"]], 2)
    println("Group $g: AR(2) coefs = ", round.(coef(ar)[2:3], digits=3))
end
```

## References

- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574--589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- McCracken, Michael W., and Serena Ng. 2020. "FRED-QD: A Quarterly Database for Macroeconomic Research." *Federal Reserve Bank of St. Louis Working Paper* 2020-005. [https://doi.org/10.20955/wp.2020.005](https://doi.org/10.20955/wp.2020.005)
