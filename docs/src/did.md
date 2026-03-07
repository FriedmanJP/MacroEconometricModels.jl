# [Difference-in-Differences](@id did_page)

**MacroEconometricModels.jl** provides a comprehensive Difference-in-Differences (DiD) toolkit for staggered treatment designs. The package implements five heterogeneity-robust estimators, Bacon decomposition diagnostics, pre-trend testing, negative weight checks, and HonestDiD sensitivity analysis.

- **TWFE**: Traditional two-way fixed effects event study regression
- **Callaway & Sant'Anna (2021)**: Group-time ATTs via outcome regression with cohort-size aggregation
- **Sun & Abraham (2021)**: Interaction-weighted estimator avoiding forbidden comparisons
- **Borusyak, Jaravel & Spiess (2024)**: Imputation estimator using only untreated subsample
- **de Chaisemartin & D'Haultfoeuille (2020)**: First-difference DiD with bootstrap inference
- **Diagnostics**: Bacon decomposition, pre-trend test, negative weight check, HonestDiD sensitivity

## Quick Start

**Recipe 1: TWFE event study**

```julia
using MacroEconometricModels

# Built-in Callaway & Sant'Anna (2021) minimum wage dataset
pd = load_example(:mpdta)

# TWFE event study: teen employment and minimum wage
did = estimate_did(pd, "lemp", "first_treat"; method=:twfe, leads=3, horizon=3)
report(did)
```

**Recipe 2: Callaway-Sant'Anna (heterogeneity-robust)**

```julia
using MacroEconometricModels

pd = load_example(:mpdta)

# Group-time ATTs with never-treated controls
did_cs = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                      leads=3, horizon=3, control_group=:never_treated)
report(did_cs)
plot_result(did_cs)
```

**Recipe 3: Sun-Abraham interaction-weighted estimator**

```julia
using MacroEconometricModels

pd = load_example(:mpdta)

did_sa = estimate_did(pd, "lemp", "first_treat"; method=:sun_abraham,
                      leads=3, horizon=3)
report(did_sa)
```

**Recipe 4: Bacon decomposition diagnostics**

```julia
using MacroEconometricModels

pd = load_example(:mpdta)

# Decompose the TWFE estimate into 2x2 comparisons
bd = bacon_decomposition(pd, "lemp", "first_treat")
report(bd)
plot_result(bd)
```

**Recipe 5: HonestDiD sensitivity analysis**

```julia
using MacroEconometricModels

pd = load_example(:mpdta)

did = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                   leads=3, horizon=3)
h = honest_did(did; Mbar=1.0, conf_level=0.95)
plot_result(h)
```

---

## Model Specification

The potential outcomes framework for staggered DiD considers units ``i = 1, \ldots, N`` observed over periods ``t = 1, \ldots, T``. Each unit has a treatment adoption date ``G_i \in \{2, 3, \ldots, T\} \cup \{\infty\}`` where ``G_i = \infty`` denotes never-treated units. The observed outcome is:

```math
Y_{it} = Y_{it}(0) + \left(Y_{it}(G_i) - Y_{it}(0)\right) \cdot \mathbf{1}\{t \geq G_i\}
```

where:
- ``Y_{it}(0)`` is the untreated potential outcome
- ``Y_{it}(g)`` is the potential outcome under treatment adopted at time ``g``
- ``\mathbf{1}\{t \geq G_i\}`` is the treatment indicator

The **group-time average treatment effect** is:

```math
\text{ATT}(g, t) = \mathbb{E}\left[Y_{it}(g) - Y_{it}(0) \mid G_i = g\right], \quad t \geq g
```

where:
- ``g`` is the treatment cohort (adoption period)
- ``t`` is the calendar period
- The expectation conditions on membership in cohort ``g``

Event-time ATTs aggregate across cohorts at each relative time ``e = t - g``:

```math
\text{ATT}(e) = \sum_{g} \frac{N_g}{N_{\text{treated}}} \cdot \text{ATT}(g, g + e)
```

where:
- ``N_g`` is the size of cohort ``g``
- ``N_{\text{treated}}`` is the total number of ever-treated units

The **parallel trends** assumption requires:

```math
\mathbb{E}\left[Y_{it}(0) - Y_{it-1}(0) \mid G_i = g\right] = \mathbb{E}\left[Y_{it}(0) - Y_{it-1}(0) \mid G_i = \infty\right] \quad \forall\, g, t
```

where:
- ``Y_{it}(0)`` is the untreated potential outcome for unit ``i`` at time ``t``
- ``G_i = g`` denotes membership in treatment cohort ``g``
- ``G_i = \infty`` denotes the never-treated group

This states that absent treatment, the average outcome change for cohort ``g`` equals the average outcome change for never-treated units in every period.

!!! note "Treatment Timing Encoding"
    The treatment column contains the **period of first treatment** for each unit (e.g., `2004` if treatment begins in 2004). Use `0` or `NaN` for never-treated units. The value must be constant within each panel unit across time. Do not pass a binary treatment indicator (0/1).

---

## Data Preparation

DiD estimation requires a `PanelData` object with an outcome variable and a treatment timing variable. The treatment column records **when** each unit first receives treatment (not a binary indicator).

### Built-in Dataset: mpdta

The `mpdta` dataset from Callaway & Sant'Anna (2021) contains county-level minimum wage data for 500 US counties over 2003--2007. Three treatment cohorts (2004, 2006, 2007) and 309 never-treated counties:

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did = estimate_did(pd, "lemp", "first_treat";
                   method=:callaway_santanna, leads=3, horizon=3)
report(did)
```

| Variable | Description |
|----------|-------------|
| `lemp` | Log of county-level teen employment (outcome) |
| `lpop` | Log of county population |
| `first_treat` | Year state first raised minimum wage; 0 = never-treated |

### Synthetic Data

For simulation studies, construct a staggered adoption panel with known treatment effects:

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

N, T_periods = 200, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)

# Treatment timing: cohort 1 at t=8, cohort 2 at t=12, never-treated = 0
treat_time = [i <= 60 ? 8 : i <= 140 ? 12 : 0 for i in 1:N]
treat_col = Float64[treat_time[g] for g in group_id]

# Heterogeneous treatment effects
fe_i = randn(N); fe_t = 0.5 * randn(T_periods)
y = [fe_i[g] + fe_t[t] +
     (treat_time[g] > 0 && t >= treat_time[g] ? 2.0 + 0.3*(t - treat_time[g]) : 0.0) +
     randn()
     for (g, t) in zip(group_id, time_id)]

df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col)
pd = xtset(df, :group, :time)
```

### Custom Cohort Specification

By default, DiD methods derive cohorts from the treatment timing column. For custom cohort definitions (e.g., geographic clusters, pre-treatment characteristics), specify a `cohort` column in `xtset`:

```julia
df.region_cohort = [g <= 60 ? 1 : g <= 140 ? 2 : 0 for g in group_id]
pd = xtset(df, :group, :time; cohort=:region_cohort)

# DiD methods use region_cohort instead of deriving from treatment timing
did = estimate_did(pd, :gdp, :reform; method=:callaway_santanna)
```

When `cohort_id` is `nothing` (the default), cohorts are inferred from the treatment column.

---

## TWFE Event Study

The traditional Two-Way Fixed Effects (TWFE) event-study regression estimates:

```math
Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}\{t - G_i = k\} + \mathbf{X}_{it}'\boldsymbol{\delta} + \varepsilon_{it}
```

where:
- ``\alpha_i`` and ``\gamma_t`` are unit and time fixed effects
- ``\beta_k`` is the event-time coefficient at relative time ``k``
- ``\mathbf{X}_{it}`` is a vector of covariates
- The period ``k = -1`` is normalized to zero (reference period)

!!! warning "TWFE Bias under Heterogeneity"
    When treatment effects vary across cohorts or over time, the TWFE estimator implicitly uses already-treated units as controls, which introduces negative weights on some group-time ATTs (Goodman-Bacon 2021). Use the heterogeneity-robust estimators below when treatment timing is staggered.

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did = estimate_did(pd, "lemp", "first_treat"; method=:twfe, leads=3, horizon=3)
report(did)
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:twfe` | Estimation method (see below) |
| `leads` | `Int` | `3` | Pre-treatment event-time window ``K`` |
| `horizon` | `Int` | `5` | Post-treatment horizon ``H`` |
| `control_group` | `Symbol` | `:never_treated` | `:never_treated` or `:not_yet_treated` |
| `cluster` | `Symbol` | `:unit` | SE clustering: `:unit`, `:time`, or `:twoway` |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `base_period` | `Symbol` | `:varying` | `:varying` or `:universal` (Callaway-Sant'Anna only) |
| `n_boot` | `Int` | `200` | Bootstrap replications (de Chaisemartin-D'Haultfoeuille only) |

### Return Value (`DIDResult{T}`)

| Field | Type | Description |
|-------|------|-------------|
| `att` | `Vector{T}` | ATT coefficients by event-time |
| `se` | `Vector{T}` | Standard errors |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid ``[-K, \ldots, H]`` |
| `reference_period` | `Int` | Omitted period (typically ``-1``) |
| `group_time_att` | `Union{Matrix{T}, Nothing}` | Cohort ``\times`` period ATT matrix (Callaway-Sant'Anna only) |
| `cohorts` | `Union{Vector{Int}, Nothing}` | Treatment cohort identifiers |
| `overall_att` | `T` | Aggregate ATT (weighted average across post-periods) |
| `overall_se` | `T` | Standard error of aggregate ATT |
| `n_obs` | `Int` | Total observations |
| `n_groups` | `Int` | Number of panel units |
| `n_treated` | `Int` | Number of ever-treated units |
| `n_control` | `Int` | Number of never-treated units |
| `method` | `Symbol` | Estimation method used |

---

## Heterogeneity-Robust Estimators

All robust estimators share the `estimate_did` interface and return `DIDResult{T}`:

### Callaway & Sant'Anna (2021)

The estimator computes group-time ATTs via outcome regression, then aggregates with cohort-size weights:

1. Identify treatment cohorts ``G = \{g_1, g_2, \ldots\}``
2. For each ``(g, t)``: compute ``\text{ATT}(g, t) = \mathbb{E}[\Delta Y \mid G = g] - \mathbb{E}[\Delta Y \mid C]``
3. Aggregate to event-time: ``\text{ATT}(e) = \sum_g w_g \cdot \text{ATT}(g, g+e)``

The `control_group` keyword controls the comparison group ``C``:
- `:never_treated` (default) --- only units with ``G_i = \infty``
- `:not_yet_treated` --- units not yet treated at time ``t``

The `base_period` keyword controls the reference period for ``\Delta Y``:
- `:varying` (default) --- pre-treatment: ``\Delta Y = Y_t - Y_{t-1}`` (adjacent periods); post-treatment: ``\Delta Y = Y_t - Y_{g-1}``
- `:universal` --- always ``\Delta Y = Y_t - Y_{g-1}``, normalizing ``e = -1`` to zero by construction

!!! note "Control Group Choice"
    Using `:not_yet_treated` increases the effective control sample but requires a stronger parallel trends assumption (across all cohorts, not just vs never-treated). When there are few never-treated units, `:not_yet_treated` may be necessary for precision.

```julia
using MacroEconometricModels

pd = load_example(:mpdta)

# Varying base (default): adjacent-period pre-treatment comparisons
did_cs = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                      leads=3, horizon=3, base_period=:varying)
report(did_cs)

# Universal base period (forces e=-1 to zero)
did_univ = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                        leads=3, horizon=3, base_period=:universal)
```

The `group_time_att` field stores the full ``n_{\text{cohorts}} \times n_{\text{periods}}`` matrix of ``\text{ATT}(g,t)`` estimates.

### Sun & Abraham (2021)

The interaction-weighted estimator runs per-cohort TWFE regressions (each cohort vs the control group) with event-time dummies for **all** relative periods, then aggregates with cohort-size weights:

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did_sa = estimate_did(pd, "lemp", "first_treat"; method=:sun_abraham,
                      leads=3, horizon=3)
report(did_sa)
```

This avoids the "forbidden comparisons" (using already-treated units as controls) that bias TWFE when treatment effects are heterogeneous across cohorts.

### Borusyak, Jaravel & Spiess (2024)

The imputation estimator follows a two-step procedure:

1. Estimate unit and time fixed effects on the **untreated subsample** only
2. Impute counterfactual ``\hat{Y}_{it}(0)`` for treated observations
3. Compute cell-level treatment effects ``\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)``
4. Aggregate to event-time ATTs with cohort-size weights

!!! note "Efficiency and Precision"
    The BJS imputation estimator is efficient under homoskedasticity and uses all available pre-treatment data for imputation. It naturally handles unbalanced panels and does not require specifying a control group explicitly.

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did_bjs = estimate_did(pd, "lemp", "first_treat"; method=:bjs,
                       leads=3, horizon=3)
report(did_bjs)
```

### de Chaisemartin & D'Haultfoeuille (2020)

The `did_multiplegt` estimator uses first-differences and bootstrap inference:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

pd = load_example(:mpdta)
did_dcdh = estimate_did(pd, "lemp", "first_treat"; method=:did_multiplegt,
                        leads=3, horizon=3, n_boot=200)
report(did_dcdh)
```

For each cohort and event-time, it computes a first-difference DiD effect and aggregates with cohort-size weights. Standard errors are obtained via unit-level block bootstrap.

---

## Diagnostics

### Bacon Decomposition

The Goodman-Bacon (2021) decomposition reveals the TWFE estimator as a weighted average of all possible 2x2 DiD comparisons. Three types of comparisons arise:

- **Treated vs Untreated**: a treated cohort vs never-treated units (clean identification)
- **Earlier vs Later**: an earlier-treated cohort vs a later-treated cohort before the later cohort's treatment (valid comparison)
- **Later vs Earlier**: a later-treated cohort vs an already-treated earlier cohort (problematic --- uses treated units as controls)

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
bd = bacon_decomposition(pd, "lemp", "first_treat")
report(bd)
plot_result(bd)
```

| Field | Type | Description |
|-------|------|-------------|
| `estimates` | `Vector{T}` | 2x2 DiD estimates |
| `weights` | `Vector{T}` | Corresponding weights (sum to 1) |
| `comparison_type` | `Vector{Symbol}` | `:treated_vs_untreated`, `:earlier_vs_later`, or `:later_vs_earlier` |
| `cohort_i` | `Vector{Int}` | First cohort in each 2x2 comparison |
| `cohort_j` | `Vector{Int}` | Second cohort (0 for never-treated) |
| `overall_att` | `T` | Weighted average (equals the TWFE estimate) |

!!! warning "Later vs Earlier Comparisons"
    The "later vs earlier" comparisons use already-treated units as controls. If treatment effects evolve over time, these comparisons are contaminated and can flip the sign of the overall estimate. Large weights on these comparisons signal that TWFE is unreliable.

### Pre-Trend Test

The `pretrend_test` function performs a joint Wald test of the null hypothesis that all pre-treatment event-time coefficients are zero:

```math
H_0: \beta_{-K} = \beta_{-K+1} = \cdots = \beta_{-2} = 0
```

where:
- ``\beta_k`` is the event-time coefficient at relative time ``k``
- ``K`` is the number of pre-treatment leads

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                   leads=3, horizon=3)
pt = pretrend_test(did)
```

A high p-value indicates no evidence against parallel trends. A low p-value suggests the parallel trends assumption may be violated.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Wald chi-squared (or F) statistic |
| `pvalue` | `T` | P-value |
| `df` | `Int` | Degrees of freedom (number of pre-treatment periods) |
| `pre_coefficients` | `Vector{T}` | Pre-treatment event-time coefficients |
| `pre_se` | `Vector{T}` | Standard errors of pre-treatment coefficients |
| `test_type` | `Symbol` | `:f_test` or `:wald` |

!!! note "Pre-testing Bias (Roth 2022)"
    Conditioning on passing a pre-trend test can bias post-treatment estimates. A non-rejection does not prove parallel trends hold --- it only means the data cannot reject them at the given sample size. Complement pre-trend testing with HonestDiD sensitivity analysis.

### Negative Weight Check

The de Chaisemartin & D'Haultfoeuille (2020) diagnostic checks whether the TWFE estimator assigns **negative weights** to some group-time ATTs:

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
nw = negative_weight_check(pd, "first_treat")
nw.has_negative_weights   # true if any weights are negative
nw.n_negative             # count of negative-weight cells
nw.total_negative_weight  # sum of all negative weights
```

Negative weights mean the TWFE estimate can have the opposite sign of every underlying ``\text{ATT}(g, t)``. When negative weights are detected, switch to one of the heterogeneity-robust estimators.

| Field | Type | Description |
|-------|------|-------------|
| `has_negative_weights` | `Bool` | `true` if any TWFE weights are negative |
| `n_negative` | `Int` | Number of group-time cells with negative weights |
| `total_negative_weight` | `T` | Sum of negative weights |
| `weights` | `Vector{T}` | All TWFE weights |
| `cohort_time_pairs` | `Vector{Tuple{Int,Int}}` | (cohort, time) for each weight |

---

## HonestDiD Sensitivity Analysis

The Rambachan & Roth (2023) HonestDiD framework constructs **robust confidence intervals** that remain valid even if parallel trends are violated by a bounded amount. The key parameter ``\bar{M}`` controls the maximum allowed violation magnitude per period:

```math
\left|\delta_{t+1} - \delta_t\right| \leq \bar{M}
```

where:
- ``\delta_t = \mathbb{E}[Y_{it}(0) \mid G_i = g] - \mathbb{E}[Y_{it}(0) \mid G_i = \infty]`` is the trend violation at time ``t``

Under this relative magnitudes restriction, the worst-case bias at post-treatment event-time ``e \geq 0`` accumulates as ``\bar{M} \cdot (e + 1)``, widening the confidence interval:

```math
\text{Robust CI}_e = \left[\hat{\beta}_e - \bar{M}(e+1) - z_{\alpha/2} \cdot \text{SE}_e, \quad \hat{\beta}_e + \bar{M}(e+1) + z_{\alpha/2} \cdot \text{SE}_e\right]
```

where:
- ``\hat{\beta}_e`` is the point estimate at event-time ``e``
- ``\text{SE}_e`` is the standard error
- ``z_{\alpha/2}`` is the critical value for confidence level ``1 - \alpha``

The **breakdown value** ``\bar{M}^*`` is the smallest violation bound at which the robust confidence interval for at least one post-treatment period includes zero:

```math
\bar{M}^* = \min_e \frac{\max\left(|\hat{\beta}_e| - z_{\alpha/2} \cdot \text{SE}_e,\; 0\right)}{e + 1}
```

where:
- ``\hat{\beta}_e`` is the point estimate at post-treatment event-time ``e``
- ``\text{SE}_e`` is the standard error of the event-time estimate
- ``z_{\alpha/2}`` is the critical value for the chosen confidence level

A large breakdown value indicates that the result is robust to substantial departures from parallel trends.

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                   leads=3, horizon=3)

h = honest_did(did; Mbar=1.0, conf_level=0.95)
plot_result(h)
```

!!! note "Choosing Mbar"
    ``\bar{M} = 0`` recovers the original (unconditional) confidence interval. ``\bar{M} = 1`` allows violations equal to the pre-trend slope. Start with ``\bar{M} \in \{0.5, 1.0, 2.0\}`` to explore sensitivity. If the breakdown value exceeds the magnitude of pre-treatment coefficient fluctuations, the result is credibly robust.

| Field | Type | Description |
|-------|------|-------------|
| `Mbar` | `T` | Violation bound used |
| `robust_ci_lower` | `Vector{T}` | Robust CI lower bounds per post-period |
| `robust_ci_upper` | `Vector{T}` | Robust CI upper bounds per post-period |
| `original_ci_lower` | `Vector{T}` | Original CIs for comparison |
| `original_ci_upper` | `Vector{T}` | Original CIs for comparison |
| `breakdown_value` | `T` | Smallest ``\bar{M}`` that overturns significance |
| `post_event_times` | `Vector{Int}` | Post-treatment event-time grid |
| `post_att` | `Vector{T}` | Post-treatment ATT point estimates |
| `conf_level` | `T` | Confidence level |

---

## Visualization

All DiD result types support `plot_result` for interactive D3.js visualization.

### Event Study Plot

```julia
using MacroEconometricModels

pd = load_example(:mpdta)
did = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                   leads=3, horizon=3)
plot_result(did)
```

```@raw html
<iframe src="../assets/plots/did_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

### Bacon Decomposition Plot

```julia
bd = bacon_decomposition(pd, "lemp", "first_treat")
plot_result(bd)
```

```@raw html
<iframe src="../assets/plots/did_bacon.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

### HonestDiD Sensitivity Plot

```julia
h = honest_did(did; Mbar=1.0)
plot_result(h)
```

```@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

---

## Complete Example

```julia
using MacroEconometricModels, DataFrames, Random
Random.seed!(42)

# Simulate staggered adoption panel with heterogeneous effects
N, T_periods = 200, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)
treat_time = [i <= 60 ? 8 : i <= 140 ? 12 : 0 for i in 1:N]
treat_col = Float64[treat_time[g] for g in group_id]

# Early cohort: +2.0 with +0.3/period dynamics; late cohort: +3.5 with +0.1/period
fe_i = randn(N); fe_t = 0.5 * randn(T_periods)
y = [fe_i[g] + fe_t[t] +
     (treat_time[g] == 8  && t >= 8  ? 2.0 + 0.3*(t - 8)  : 0.0) +
     (treat_time[g] == 12 && t >= 12 ? 3.5 + 0.1*(t - 12) : 0.0) +
     randn()
     for (g, t) in zip(group_id, time_id)]

df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col)
pd = xtset(df, :group, :time)

# Diagnostics: check for TWFE problems
bd = bacon_decomposition(pd, :gdp, :reform)
report(bd)

nw = negative_weight_check(pd, :reform)

# Estimate with multiple methods
did_twfe = estimate_did(pd, :gdp, :reform; method=:twfe, leads=3, horizon=5)
did_cs   = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
did_sa   = estimate_did(pd, :gdp, :reform; method=:sun_abraham, leads=3, horizon=5)
did_bjs  = estimate_did(pd, :gdp, :reform; method=:bjs, leads=3, horizon=5)

report(did_cs)

# Pre-trend test
pt = pretrend_test(did_cs)

# HonestDiD sensitivity
h = honest_did(did_cs; Mbar=1.0, conf_level=0.95)

# Visualize
plot_result(did_cs)
plot_result(bd)
plot_result(h)
```

The TWFE estimator produces biased event-study coefficients when treatment effects are heterogeneous across the early and late cohorts. The Bacon decomposition reveals the source of this bias by decomposing the TWFE estimate into treated-vs-untreated, earlier-vs-later, and later-vs-earlier 2x2 comparisons --- the last category uses already-treated units as controls and is contaminated. The Callaway-Sant'Anna, Sun-Abraham, and BJS estimators all avoid this problem by restricting comparisons to clean control groups, producing consistent ATT estimates. The pre-trend test checks whether pre-treatment event-study coefficients are jointly zero, and the HonestDiD analysis quantifies how much violation of parallel trends the result can withstand before significance is overturned.

---

## Common Pitfalls

1. **Parallel trends is untestable**: Pre-trend tests evaluate whether pre-treatment coefficients are jointly zero, but non-rejection does not prove parallel trends hold in the post-treatment period. Conditioning on passing a pre-trend test introduces pre-testing bias (Roth 2022). Always complement with HonestDiD sensitivity analysis.

2. **Negative weights in TWFE**: With staggered adoption, the TWFE estimator assigns negative weights to some group-time ATTs, potentially flipping the sign of the overall estimate. Run `negative_weight_check` before interpreting TWFE results. When negative weights are detected, switch to Callaway-Sant'Anna, Sun-Abraham, or BJS.

3. **Staggered adoption requires robust estimators**: The standard TWFE event study is only valid when all units adopt treatment simultaneously. With staggered adoption timing and heterogeneous effects, TWFE produces biased estimates. The four robust estimators (CS, SA, BJS, dCDH) are designed for staggered settings.

4. **Never-treated group requirement**: Callaway-Sant'Anna and Sun-Abraham with `control_group=:never_treated` require a sufficient number of never-treated units. When all units eventually receive treatment, use `control_group=:not_yet_treated` (at the cost of a stronger parallel trends assumption) or the BJS imputation estimator.

5. **Treatment column format**: The treatment variable must contain the **period number** when treatment first occurs, not a binary 0/1 indicator. Passing a binary indicator causes the package to misidentify cohorts. Use `0` or `NaN` for never-treated units.

---

## References

- Borusyak, Kirill, Xavier Jaravel, and Jann Spiess. 2024. "Revisiting Event-Study Designs: Robust and Efficient Estimation."
  *Review of Economic Studies* 91 (6): 3253--3285. [DOI](https://doi.org/10.1093/restud/rdae007)

- Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods."
  *Journal of Econometrics* 225 (2): 200--230. [DOI](https://doi.org/10.1016/j.jeconom.2020.12.001)

- de Chaisemartin, Clement, and Xavier D'Haultfoeuille. 2020. "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
  *American Economic Review* 110 (9): 2964--2996. [DOI](https://doi.org/10.1257/aer.20181169)

- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing."
  *Journal of Econometrics* 225 (2): 254--277. [DOI](https://doi.org/10.1016/j.jeconom.2021.03.014)

- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends."
  *Review of Economic Studies* 90 (5): 2555--2591. [DOI](https://doi.org/10.1093/restud/rdad018)

- Roth, Jonathan. 2022. "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends."
  *American Economic Review: Insights* 4 (3): 305--322. [DOI](https://doi.org/10.1257/aeri.20210236)

- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects."
  *Journal of Econometrics* 225 (2): 175--199. [DOI](https://doi.org/10.1016/j.jeconom.2020.09.006)
