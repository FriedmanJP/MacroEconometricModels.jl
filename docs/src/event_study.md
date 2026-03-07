# [Event Study LP](@id event_study_page)

**MacroEconometricModels.jl** provides two LP-based event study estimators for causal inference in panel settings: the **Event Study LP** (Jorda 2005; Acemoglu, Naidu, Restrepo & Robinson 2019) and the **LP-DiD** estimator (Dube, Girardi, Jorda & Taylor 2025) with clean control samples, switching indicator treatment, and time-only fixed effects. The package achieves full parity with Stata `lpdid` v1.0.2.

- **Event Study LP**: Horizon-by-horizon local projections with switching indicator treatment and time-only FE
- **LP-DiD**: Clean control sample restrictions (absorbing/non-absorbing/one-off), pre-mean differencing, IPW reweighting, pooled estimates
- **DDCG dataset**: Built-in Acemoglu et al. (2019) democracy-GDP panel (184 countries, 1960--2010)
- **Panel utilities**: `panel_lag`, `panel_lead`, `panel_diff` for within-group transformations
- **Diagnostics**: Pre-trend tests and HonestDiD sensitivity analysis

## Quick Start

**Recipe 1: Standard Event Study LP**

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd = xtset(df, :group, :time)
eslp = estimate_event_study_lp(pd, :outcome, :treat, 5; leads=3, lags=2)
report(eslp)
```

**Recipe 2: LP-DiD with absorbing treatment**

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd = xtset(df, :group, :time)

r = estimate_lp_did(pd, :outcome, :treat, 5; pre_window=3, ylags=2)
report(r)
```

**Recipe 3: LP-DiD on DDCG dataset**

```julia
using MacroEconometricModels

# Democracy and GDP: Acemoglu et al. (2019)
ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, ylags=1)
report(r)
plot_result(r; title="Democracy -> GDP (DDCG)")
```

**Recipe 4: LP-DiD with PMD and IPW reweighting**

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)

# Pre-mean differencing + inverse probability weighting
r_pmd = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, pmd=:max, reweight=true)
report(r_pmd)
```

**Recipe 5: Pooled estimates**

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)

r = estimate_lp_did(ddcg, :y, :dem, 10;
    post_pooled=(0, 5),   # Average effect over h=0,...,5
    pre_pooled=(1, 5)     # Pre-treatment placebo over h=-5,...,-1
)
report(r)
```

---

## Model Specification

Both estimators run separate regressions for each event-time horizon ``h \in \{-K, \ldots, -1, 0, 1, \ldots, H\}``:

```math
Y_{i,t+h} - Y_{i,t-1} = \gamma_t^h + \beta_h \, \Delta D_{it} + \mathbf{X}_{it}'\boldsymbol{\delta}^h + \varepsilon_{i,t+h}
```

where:
- ``Y_{i,t+h} - Y_{i,t-1}`` is the long-differenced outcome (absorbs unit fixed effects)
- ``\Delta D_{it} = D_{it} - D_{i,t-1}`` is the **switching indicator** (equals 1 only at treatment onset)
- ``\gamma_t^h`` is a time fixed effect (absorbed by within-time demeaning)
- ``\mathbf{X}_{it}`` includes lagged outcomes ``L_1.Y, \ldots, L_k.Y``, differenced lags ``\Delta Y_{t-l}``, and optional covariates
- ``\beta_h`` is the **dynamic treatment effect** at horizon ``h``

The reference period ``h = -1`` is normalized to zero.

!!! note "Time-Only Fixed Effects"
    Long differencing ``Y_{i,t+h} - Y_{i,t-1}`` absorbs unit fixed effects, so only time FE remain. This is consistent with both the Acemoglu et al. (2019) specification and the Stata `lpdid` package.

!!! note "Switching Indicator vs Treatment Level"
    The treatment regressor is the first difference ``\Delta D_{it}``, not the treatment level ``D_{it}``. This ensures that only the treatment onset contributes to identification. Already-treated observations with ``\Delta D = 0`` and ``D = 1`` are excluded from the sample.

---

## Event Study LP

The standard estimator uses all switching (``\Delta D = 1``) and control (``D = 0``) observations at each horizon:

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd = xtset(df, :group, :time)

eslp = estimate_event_study_lp(pd, :outcome, :treat, 5;
    leads=3,             # Pre-treatment horizons K
    lags=2,              # Lagged outcome controls
    cluster=:unit,       # :unit, :time, or :twoway
    conf_level=0.95
)
report(eslp)
plot_result(eslp)
```

The `estimate_event_study_lp` function runs ``K + H + 1`` separate OLS regressions (one per event-time horizon), each with cluster-robust standard errors. The resulting coefficients ``\beta_h`` trace out the dynamic treatment effect path.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `leads` | `Int` | `3` | Pre-treatment horizons ``K`` |
| `lags` | `Int` | `0` | Lagged outcome controls |
| `covariates` | `Vector{String}` | `String[]` | Additional control variables |
| `cluster` | `Symbol` | `:unit` | SE clustering: `:unit`, `:time`, or `:twoway` |
| `conf_level` | `Real` | `0.95` | Confidence level |

### Return Value (`EventStudyLP{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Vector{T}` | Treatment effect ``\beta_h`` at each event-time |
| `se` | `Vector{T}` | Cluster-robust standard errors |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid ``[-K, \ldots, H]`` |
| `reference_period` | `Int` | Omitted period (``-1``) |
| `B` | `Vector{Matrix{T}}` | Full coefficient vectors per horizon |
| `residuals_per_h` | `Vector{Matrix{T}}` | OLS residuals per horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices per horizon |
| `T_eff` | `Vector{Int}` | Effective sample size per horizon |
| `n_obs` | `Int` | Total panel observations |
| `lags` | `Int` | Number of lagged controls |
| `leads` | `Int` | Pre-treatment window |
| `horizon` | `Int` | Maximum horizon ``H`` |
| `cluster` | `Symbol` | Clustering level |

---

## LP-DiD (Dube et al. 2025)

The LP-DiD estimator adds **clean control sample** (CCS) restrictions. At each horizon ``h``, the control group contains only units whose treatment status does not change between ``t`` and ``t + h``. This prevents already-treated units from contaminating the control group under heterogeneous treatment effects.

### Clean Control Samples

Three CCS specifications match the Stata `lpdid` package:

**Absorbing treatment** (default): A ``(i, t)`` pair belongs to CCS at horizon ``h`` if the unit is switching (``\Delta D_{it} = 1``) or treatment status remains at zero through ``t + h``:

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10)  # Absorbing is default
report(r)
```

**Non-absorbing treatment**: Treatment may reverse. A pair belongs to CCS if no switches occurred in the stabilization window of ``L`` periods before ``t``:

```julia
r = estimate_lp_did(ddcg, :y, :dem, 10; nonabsorbing=5)
```

**One-off treatment**: Treatment lasts exactly one period. Requires `nonabsorbing`:

```julia
r = estimate_lp_did(ddcg, :y, :dem, 10; nonabsorbing=3, oneoff=true)
```

### Pre-Mean Differencing (PMD)

Instead of long differencing ``Y_{t+h} - Y_{t-1}``, PMD uses the average of pre-treatment outcomes as baseline. This reduces noise from a single pre-treatment period:

```math
Y_{i,t+h} - \bar{Y}_{i,\text{pre}} = \gamma_t^h + \beta_h \, \Delta D_{it} + \mathbf{X}_{it}'\boldsymbol{\delta}^h + \varepsilon_{i,t+h}
```

where:
- ``\bar{Y}_{i,\text{pre}}`` is the average of ``Y_{i,t-1}, Y_{i,t-2}, \ldots`` over a window of pre-treatment periods

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)

# Use cumulative pre-treatment mean
r = estimate_lp_did(ddcg, :y, :dem, 10; pmd=:max)

# Use moving average of k pre-treatment periods
r = estimate_lp_did(ddcg, :y, :dem, 10; pmd=3)
```

### IPW Reweighting

Inverse probability weights ensure equally weighted ATE across time periods, correcting for compositional changes in the treatment-control balance:

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10; reweight=true)
```

### Pooled Estimates

Pooled regressions average the left-hand side over a window of horizons, producing a single average treatment effect:

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10;
    post_pooled=(0, 5),   # Average effect over h=0,...,5
    pre_pooled=(1, 3)     # Pre-treatment placebo over h=-3,...,-1
)
report(r)
```

The pooled estimates are stored in `r.pooled_post` and `r.pooled_pre` as named tuples with fields `coef`, `se`, `ci_lower`, `ci_upper`, and `nobs`.

### Full Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `pre_window` | `Int` | `3` | Pre-treatment event-time ``K`` |
| `post_window` | `Int` | `H` | Post-treatment event-time |
| `ylags` | `Int` | `0` | Outcome lags (``L_1.Y, \ldots, L_k.Y``) |
| `dylags` | `Int` | `0` | Differenced outcome lags (``L_1.\Delta Y, \ldots``) |
| `covariates` | `Vector{String}` | `String[]` | Additional covariates |
| `nonabsorbing` | `Union{Int,Nothing}` | `nothing` | Stabilization window ``L`` for non-absorbing CCS |
| `oneoff` | `Bool` | `false` | One-off treatment (requires `nonabsorbing`) |
| `notyet` | `Bool` | `false` | Restrict to not-yet-treated controls |
| `nevertreated` | `Bool` | `false` | Restrict to never-treated controls |
| `firsttreat` | `Bool` | `false` | Use only first treatment event per unit |
| `pmd` | `Union{Symbol,Int,Nothing}` | `nothing` | Pre-mean differencing (`:max` or integer ``k``) |
| `reweight` | `Bool` | `false` | IPW for equally weighted ATE across time |
| `nocomp` | `Bool` | `false` | Restrict to obs in CCS at all horizons |
| `cluster` | `Symbol` | `:unit` | SE clustering: `:unit`, `:time`, or `:twoway` |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `post_pooled` | `Union{Tuple,Nothing}` | `nothing` | ``(start, end)`` for pooled post-treatment |
| `pre_pooled` | `Union{Tuple,Nothing}` | `nothing` | ``(start, end)`` for pooled pre-treatment |
| `only_pooled` | `Bool` | `false` | Skip event study, compute only pooled |
| `only_event` | `Bool` | `false` | Skip pooled, compute only event study |

### Return Value (`LPDiDResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Vector{T}` | Treatment effect ``\beta_h`` at each event-time |
| `se` | `Vector{T}` | Cluster-robust standard errors |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid ``[-K, \ldots, H]`` |
| `reference_period` | `Int` | Omitted period (``-1``) |
| `nobs_per_horizon` | `Vector{Int}` | Effective sample size per horizon |
| `pooled_post` | `NamedTuple` or `nothing` | Pooled post-treatment estimate |
| `pooled_pre` | `NamedTuple` or `nothing` | Pooled pre-treatment estimate |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices per horizon |
| `specification` | `Symbol` | `:absorbing`, `:nonabsorbing`, or `:oneoff` |
| `pmd` | varies | PMD specification (`nothing`, `:max`, or `Int`) |
| `reweight` | `Bool` | IPW reweighting flag |
| `cluster` | `Symbol` | Clustering level |

---

## DDCG Dataset

The built-in DDCG dataset contains 184 countries from 1960--2010 with log GDP per capita and a binary democracy indicator from Acemoglu, Naidu, Restrepo & Robinson (2019):

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)

r = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=1, post_pooled=(0, 25))
report(r)
```

| Variable | Description |
|----------|-------------|
| `y` | Log GDP per capita |
| `dem` | Democracy indicator (0/1) |

The dataset is organized as a balanced panel with country-year observations. The `dem` variable records democratic transitions (0 to 1) and reversals (1 to 0), making this a non-absorbing treatment setting suitable for `nonabsorbing` CCS.

---

## Panel Utilities

Within-group lag, lead, and difference operations for `PanelData`:

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)

# Compute lag/lead/diff vectors
l1 = panel_lag(ddcg, :y, 1)     # L1.y
f1 = panel_lead(ddcg, :y, 1)    # F1.y
dy = panel_diff(ddcg, :y)       # delta y = y - L1.y

# Add as new columns (returns new PanelData)
ddcg2 = add_panel_lag(ddcg, :y, 1)    # adds "lag1_y"
ddcg3 = add_panel_lead(ddcg, :y, 1)   # adds "lead1_y"
ddcg4 = add_panel_diff(ddcg, :y)      # adds "d_y"
```

These functions respect panel group boundaries --- lags, leads, and differences do not cross from one unit to another.

---

## Clustering

Both estimators support three clustering options for standard error computation:

- **`:unit`** (default) --- accounts for serial correlation within units
- **`:time`** --- accounts for cross-sectional correlation within periods
- **`:twoway`** --- two-way clustering (Cameron, Gelbach & Miller 2011): ``V_{\text{twoway}} = V_{\text{unit}} + V_{\text{time}} - V_{\text{het}}``

```julia
using MacroEconometricModels

ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10; cluster=:twoway)
report(r)
```

Two-way clustering is recommended when both serial correlation (within units) and cross-sectional correlation (across units within periods) are present, as in macroeconomic panels where common shocks affect all countries simultaneously.

---

## Diagnostics

### Pre-Trend Test

Joint Wald test that all pre-treatment coefficients are zero:

```math
H_0: \beta_{-K} = \beta_{-K+1} = \cdots = \beta_{-2} = 0
```

where:
- ``\beta_k`` is the LP coefficient at event-time ``k``

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd = xtset(df, :group, :time)
eslp = estimate_event_study_lp(pd, :outcome, :treat, 5; leads=3, lags=2)

pt = pretrend_test(eslp)
```

A high p-value indicates no evidence against parallel trends at the given sample size.

### HonestDiD Sensitivity Analysis

Rambachan & Roth (2023) robust confidence intervals under bounded violations of parallel trends:

```julia
h = honest_did(eslp; Mbar=1.0)
plot_result(h)
```

The breakdown value ``\bar{M}^*`` reports the smallest violation magnitude at which the robust confidence interval includes zero. See [Difference-in-Differences](@ref did_page) for detailed documentation of `bacon_decomposition`, `negative_weight_check`, and HonestDiD methodology.

---

## Visualization

`plot_result` produces interactive D3.js event study plots for both `EventStudyLP` and `LPDiDResult`:

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd = xtset(df, :group, :time)
eslp = estimate_event_study_lp(pd, :outcome, :treat, 5; leads=3, lags=2)

p = plot_result(eslp)
save_plot(p, "event_study.html")
```

```@raw html
<iframe src="../assets/plots/eslp_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

---

## Complete Example

```julia
using MacroEconometricModels

# Load DDCG democracy-GDP dataset (Acemoglu et al. 2019)
ddcg = load_example(:ddcg)

# LP-DiD: effect of democracy on log GDP per capita
r = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5,
    ylags=1,
    post_pooled=(0, 25),
    pre_pooled=(1, 5)
)
report(r)

# Robustness: PMD + reweighting
r_pmd = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=1, pmd=:max, reweight=true)
report(r_pmd)

# Plot
plot_result(r; title="Democracy -> GDP (LP-DiD, DDCG)")
```

The baseline LP-DiD specification estimates the causal effect of democratic transitions on log GDP per capita using a switching indicator and clean control samples. The pooled post-treatment estimate averages the dynamic treatment effect over horizons 0 through 25, providing a single summary measure of democracy's long-run GDP impact. The pre-treatment pooled estimate serves as a placebo --- a value near zero supports the parallel trends assumption. The PMD + IPW robustness check uses pre-mean differencing to reduce noise from a single baseline period and inverse probability weighting to ensure equal representation across time periods, confirming that the baseline result is not driven by compositional changes in the sample.

---

## Common Pitfalls

1. **Treatment column format**: `estimate_lp_did` auto-detects binary (0/1) vs timing (year values). Mixing formats (e.g., 0, 1, 2019) causes misclassification. Ensure the treatment column is consistently encoded.

2. **Small effective samples at long horizons**: CCS restrictions reduce the sample at each horizon as more units switch treatment status. Monitor `r.nobs_per_horizon` and reduce ``H`` if counts drop below approximately 30 observations.

3. **Combining `notyet` and `nevertreated`**: These are mutually exclusive. `notyet` uses units not yet treated at ``t+h`` as controls; `nevertreated` uses only units with ``G_i = 0``. Specifying both raises an error.

4. **`oneoff` requires `nonabsorbing`**: One-off treatment is a special case of non-absorbing treatment where the treatment indicator lasts exactly one period. Calling `oneoff=true` without setting `nonabsorbing` raises an error.

5. **PMD with short pre-treatment windows**: `pmd=:max` uses all available pre-treatment data. With few pre-treatment periods, the average baseline may be noisy. Consider `pmd=k` with a small ``k`` to use a fixed window.

---

## References

- Acemoglu, Daron, Suresh Naidu, Pascual Restrepo, and James A. Robinson. 2019. "Democracy Does Cause Growth."
  *Journal of Political Economy* 127 (1): 47--100. [DOI](https://doi.org/10.1086/700936)

- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2011. "Robust Inference with Multiway Clustering."
  *Journal of Business & Economic Statistics* 29 (2): 238--249. [DOI](https://doi.org/10.1198/jbes.2010.07136)

- Dube, Arindrajit, Daniele Girardi, Oscar Jorda, and Alan M. Taylor. 2025. "A Local Projections Approach to Difference-in-Differences."
  *Journal of Applied Econometrics*. [DOI](https://doi.org/10.1002/jae.3117)

- Jorda, Oscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
  *American Economic Review* 95 (1): 161--182. [DOI](https://doi.org/10.1257/0002828053828518)

- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends."
  *Review of Economic Studies* 90 (5): 2555--2591. [DOI](https://doi.org/10.1093/restud/rdad018)
