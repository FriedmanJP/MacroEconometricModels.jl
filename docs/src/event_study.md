# Event Study LP

This page documents the **Event Study Local Projections** implementation in **MacroEconometricModels.jl**, providing LP-based event study estimation with panel fixed effects (Jorda 2005) and the LP-DiD estimator with clean controls (Dube et al. 2023).

## Quick Start

```julia
using MacroEconometricModels

# Synthetic staggered treatment panel: 50 units, 20 periods
using Random; Random.seed!(42)
N, T_periods = 50, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)
treat_timing = [i <= 25 ? 10 : (i <= 40 ? 14 : 0) for i in 1:N]
treatment = [treat_timing[g] for g in group_id]
y = randn(N * T_periods) .+ 0.5 .* [t >= treat_timing[g] && treat_timing[g] > 0 ? 1.0 : 0.0
    for (g, t) in zip(group_id, time_id)]

using DataFrames
df = DataFrame(group=group_id, time=time_id, outcome=y, treat_time=Float64.(treatment))
pd = xtset(df, :group, :time)

# Standard event study LP
eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5; leads=3, lags=2)

# LP-DiD with clean controls (Dube et al. 2023)
lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5; leads=3, lags=2)
```

---

## Model Specification

The event study LP estimates separate regressions for each horizon ``h \in \{-K, \ldots, -1, 0, 1, \ldots, H\}``:

```math
y_{i,t+h} - y_{i,t-1} = \alpha_i^h + \gamma_t^h + \beta_h D_{it} + \mathbf{X}_{it}'\boldsymbol{\delta}^h + \varepsilon_{i,t+h}
```

where:
- ``y_{i,t+h}`` is the outcome for unit ``i`` at time ``t+h``
- ``\alpha_i^h`` is a unit fixed effect (absorbed by double-demeaning)
- ``\gamma_t^h`` is a time fixed effect (absorbed by double-demeaning)
- ``D_{it} = \mathbf{1}[t \geq g_i]`` is the treatment indicator (``g_i`` = treatment timing for unit ``i``)
- ``\mathbf{X}_{it}`` includes lagged outcomes and optional covariates
- ``\beta_h`` is the **dynamic treatment effect** at horizon ``h``

The dependent variable is differenced from the baseline period ``t-1``, so ``\beta_h`` measures the cumulative effect of treatment ``h`` periods after (or before) the event. The reference period is ``h = -1`` (normalized to zero).

!!! note "Double-Demeaning for Fixed Effects"
    Unit and time fixed effects are absorbed by iterative double-demeaning (within-transformation along both dimensions). This avoids estimating a potentially large number of dummy coefficients while producing numerically identical estimates to explicit FE inclusion.

---

## Standard Event Study LP

The standard estimator runs horizon-by-horizon OLS on the full panel, using all untreated and already-treated units as potential controls:

```julia
eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
    leads=3,                      # pre-treatment horizons (for pre-trends)
    lags=2,                       # lagged outcome controls
    covariates=String[],          # additional covariates
    cluster=:unit,                # clustering level
    conf_level=0.95               # confidence level
)

# Access results
eslp.coefficients    # dynamic treatment effects (beta_h)
eslp.se              # standard errors
eslp.event_times     # event-time grid [-3, -2, -1, 0, 1, ..., 5]
eslp.T_eff           # effective observations per horizon
```

### Return Values

| Field | Type | Description |
|:------|:-----|:------------|
| `coefficients` | `Vector{T}` | Treatment effect ``\beta_h`` at each event-time |
| `se` | `Vector{T}` | Cluster-robust standard errors |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid (e.g., ``[-3, -2, 0, 1, \ldots, H]``) |
| `reference_period` | `Int` | Omitted period (``-1`` by default) |
| `B` | `Vector{Matrix{T}}` | Full coefficient vectors per horizon |
| `residuals_per_h` | `Vector{Matrix{T}}` | OLS residuals per horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices per horizon |
| `T_eff` | `Vector{Int}` | Effective sample size per horizon |
| `outcome_var` | `String` | Outcome variable name |
| `treatment_var` | `String` | Treatment timing variable name |
| `n_obs` | `Int` | Total panel observations |
| `n_groups` | `Int` | Number of panel units |
| `lags` | `Int` | Number of lagged outcome controls |
| `leads` | `Int` | Number of pre-treatment leads |
| `horizon` | `Int` | Maximum post-treatment horizon ``H`` |
| `clean_controls` | `Bool` | `false` for standard LP, `true` for LP-DiD |
| `cluster` | `Symbol` | Clustering level used |
| `conf_level` | `T` | Confidence level |
| `data` | `PanelData{T}` | Original panel data |

---

## LP-DiD (Dube et al. 2023)

The `estimate_lp_did` function implements the **clean control** restriction from Dube, Girardi, Jorda, and Taylor (2023). In a standard event study LP, units that are already treated at ``t+h`` may serve as controls for newly treated units, introducing **contamination bias** when treatment effects are heterogeneous over time.

LP-DiD solves this by restricting the control group at each horizon ``h`` to units that are either:
1. **Never-treated** (``g_i = 0``), or
2. **Not-yet-treated at ``t+h``** (``g_i > t + h``)

This ensures that no already-treated unit contaminates the control group.

```julia
lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5;
    leads=3, lags=2, cluster=:unit, conf_level=0.95
)

# The only difference from estimate_event_study_lp is clean_controls=true
lpdid.clean_controls  # true
```

!!! warning "Sample Size Reduction"
    Clean controls can substantially reduce the effective sample at longer horizons, since more units have been treated by ``t+h``. Check `lpdid.T_eff` to monitor the effective sample size at each horizon. If `T_eff` drops too low, consider reducing the maximum horizon `H`.

---

## Clustering

Both estimators support three clustering options for robust standard error computation via the `cluster` keyword:

- **`:unit`** (default) -- cluster at the unit (entity) level. Accounts for serial correlation within units. Appropriate for most panel DiD settings.
- **`:time`** -- cluster at the time level. Accounts for cross-sectional correlation within periods. Useful when shocks are common across units within a period.
- **`:twoway`** -- two-way clustering (Cameron, Gelbach, and Miller 2011). Computes ``V_{\text{twoway}} = V_{\text{unit}} + V_{\text{time}} - V_{\text{het}}``, accounting for both serial and cross-sectional dependence simultaneously.

```julia
# Two-way clustering for robustness
eslp_tw = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
    leads=3, lags=2, cluster=:twoway
)
```

!!! note "Cluster Choice"
    Unit-level clustering is the standard default for panel event studies. Two-way clustering is more conservative and may be preferred when both dimensions of dependence are plausible, but requires sufficient clusters in both dimensions for reliable inference.

---

## Diagnostics

### Pre-Trend Test

The `pretrend_test` function performs a joint Wald test that all pre-treatment coefficients are zero, providing a formal check of the **parallel trends** assumption:

```julia
pt = pretrend_test(eslp)
pt.statistic    # Wald chi-squared statistic
pt.pvalue       # p-value (high = no evidence against parallel trends)
pt.df           # degrees of freedom
```

### HonestDiD Sensitivity Analysis

The `honest_did` function implements the Rambachan and Roth (2023) sensitivity analysis, constructing robust confidence intervals under bounded violations of parallel trends:

```julia
h = honest_did(eslp; Mbar=1.0, conf_level=0.95)
h.robust_ci_lower     # lower bounds accounting for trend violations
h.robust_ci_upper     # upper bounds
h.breakdown_value     # smallest Mbar overturning significance
```

The `Mbar` parameter controls the maximum allowed violation magnitude per period. The **breakdown value** is the smallest `Mbar` at which the robust confidence interval for at least one post-treatment period includes zero.

See [Difference-in-Differences](@ref did_page) for additional diagnostics including `bacon_decomposition` and `negative_weight_check`.

---

## Visualization

`plot_result` produces an interactive D3.js event study plot with point estimates, confidence bands, and a reference line at zero:

```julia
p = plot_result(eslp)
save_plot(p, "event_study.html")
```

@raw html
<iframe src="../assets/plots/eslp_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>

---

## Complete Example

```julia
using MacroEconometricModels
using Random; Random.seed!(42)

# --- Synthetic staggered treatment panel ---
N, T_periods = 50, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)

# Staggered adoption: cohort 1 at t=10, cohort 2 at t=14, never-treated
treat_timing = [i <= 25 ? 10 : (i <= 40 ? 14 : 0) for i in 1:N]
treatment = [treat_timing[g] for g in group_id]

# Outcome with treatment effect = 1.0
y = randn(N * T_periods) .+ [t >= treat_timing[g] && treat_timing[g] > 0 ? 1.0 : 0.0
    for (g, t) in zip(group_id, time_id)]

using DataFrames
df = DataFrame(group=group_id, time=time_id, outcome=y, treat_time=Float64.(treatment))
pd = xtset(df, :group, :time)

# Standard event study LP
eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5; leads=3, lags=2)
println("Standard LP coefficients: ", round.(eslp.coefficients, digits=3))

# LP-DiD with clean controls
lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5; leads=3, lags=2)
println("LP-DiD coefficients:      ", round.(lpdid.coefficients, digits=3))

# Pre-trend test
pt = pretrend_test(eslp)
println("Pre-trend test: p = ", round(pt.pvalue, digits=3))

# HonestDiD sensitivity
h = honest_did(eslp; Mbar=0.5)
println("Breakdown value: ", round(h.breakdown_value, digits=3))

# Visualization
p = plot_result(eslp)
```

---

## References

- Jorda, Oscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161--182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Dube, Arindrajit, Daniele Girardi, Oscar Jorda, and Alan M. Taylor. 2023. "A Local Projections Approach to Difference-in-Differences Event Studies." NBER Working Paper 31184. [https://doi.org/10.3386/w31184](https://doi.org/10.3386/w31184)
