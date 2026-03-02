# [Difference-in-Differences](@id did_page)

This page documents the Difference-in-Differences (DiD) implementation in **MacroEconometricModels.jl**, providing five heterogeneity-robust estimators for staggered treatment designs, Bacon decomposition diagnostics, pre-trend testing, and HonestDiD sensitivity analysis.

## Quick Start

```julia
using MacroEconometricModels

# Synthetic staggered adoption panel: 200 units, 20 periods, 3 cohorts
using Random; Random.seed!(42)
N, T_periods = 200, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)
treat_time = [i <= 60 ? 8 : i <= 140 ? 12 : 0 for i in 1:N]  # cohorts at t=8, t=12, never
treat_col = Float64[treat_time[g] for g in group_id]
fe_i = randn(N); fe_t = randn(T_periods)
y = [fe_i[g] + fe_t[t] + (treat_time[g] > 0 && t >= treat_time[g] ? 2.0 + 0.3*(t - treat_time[g]) : 0.0) + randn()
     for (g, t) in zip(group_id, time_id)]
using DataFrames
df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col)
pd = xtset(df, :group, :time)

# TWFE event study
did = estimate_did(pd, :gdp, :reform; method=:twfe, leads=3, horizon=5)

# Callaway-Sant'Anna (heterogeneity-robust)
did_cs = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)

# Visualize the event study
plot_result(did_cs)
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

Event-time ATTs aggregate across cohorts at each relative time ``e = t - g``:

```math
\text{ATT}(e) = \sum_{g} \frac{N_g}{N_{\text{treated}}} \cdot \text{ATT}(g, g + e)
```

where ``N_g`` is the size of cohort ``g``. The **parallel trends** assumption requires:

```math
\mathbb{E}\left[Y_{it}(0) - Y_{it-1}(0) \mid G_i = g\right] = \mathbb{E}\left[Y_{it}(0) - Y_{it-1}(0) \mid G_i = \infty\right] \quad \forall\, g, t
```

!!! note "Treatment Timing Encoding"
    The treatment column should contain the **period of first treatment** for each unit (e.g., `8` if treatment begins in period 8). Use `0` or `NaN` for never-treated units. The value must be constant within each panel unit across time.

---

## Data Preparation

DiD estimation requires a `PanelData` object with an outcome variable and a treatment timing variable. The treatment column records **when** each unit first receives treatment (not a binary indicator).

```julia
using MacroEconometricModels
using Random; Random.seed!(42)

# Create synthetic staggered treatment panel
N, T_periods = 200, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)

# Treatment timing: cohort 1 at t=8, cohort 2 at t=12, never-treated = 0
treat_time = [i <= 60 ? 8 : i <= 140 ? 12 : 0 for i in 1:N]
treat_col = Float64[treat_time[g] for g in group_id]

# Generate outcome with heterogeneous treatment effects
fe_i = randn(N)
fe_t = 0.5 * randn(T_periods)
y = [fe_i[g] + fe_t[t] + (treat_time[g] > 0 && t >= treat_time[g] ? 2.0 + 0.3*(t - treat_time[g]) : 0.0) + randn()
     for (g, t) in zip(group_id, time_id)]

using DataFrames
df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col)
pd = xtset(df, :group, :time)
```

The `xtset` function creates a `PanelData` object (Stata-style panel declaration). All DiD functions accept `PanelData` directly.

### Custom Cohort Specification

By default, DiD methods derive cohorts from the treatment timing column. For custom cohort definitions (e.g., geographic clusters, pre-treatment characteristics), specify a `cohort` column in `xtset`:

```julia
df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col,
               region_cohort=[g <= 60 ? 1 : g <= 140 ? 2 : 0 for g in group_id])
pd = xtset(df, :group, :time; cohort=:region_cohort)

# DiD methods will use region_cohort instead of deriving from reform timing
did = estimate_did(pd, :gdp, :reform; method=:callaway_santanna)
```

When `cohort_id` is `nothing` (the default), behavior is unchanged --- cohorts are inferred from the treatment column.

!!! warning "Binary vs Timing Encoding"
    Do **not** pass a binary treatment indicator (0/1) as the treatment variable. The treatment column must contain the **period number** when treatment first occurs. The package internally constructs event-time dummies from this timing information.

---

## TWFE Event Study

The traditional Two-Way Fixed Effects (TWFE) event-study regression estimates:

```math
Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}\{t - G_i = k\} + \mathbf{X}_{it}'\boldsymbol{\delta} + \varepsilon_{it}
```

where ``\alpha_i`` and ``\gamma_t`` are unit and time fixed effects, and the event-time coefficients ``\beta_k`` trace out the dynamic treatment effect path. The period ``k = -1`` is normalized to zero.

```julia
did = estimate_did(pd, :gdp, :reform; method=:twfe, leads=3, horizon=5)
report(did)
```

The `estimate_did` dispatcher routes to the appropriate internal estimator based on the `method` keyword.

### Return Values

| Field | Type | Description |
|:---|:---|:---|
| `att` | `Vector{T}` | ATT coefficients by event-time |
| `se` | `Vector{T}` | Standard errors |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid (e.g., `[-3,-2,-1,0,1,...,H]`) |
| `reference_period` | `Int` | Omitted period (typically `-1`) |
| `group_time_att` | `Union{Matrix{T}, Nothing}` | Cohort x period ATT matrix (Callaway-Sant'Anna only) |
| `cohorts` | `Union{Vector{Int}, Nothing}` | Treatment cohort identifiers |
| `overall_att` | `T` | Aggregate ATT (weighted average across post-periods) |
| `overall_se` | `T` | Standard error of aggregate ATT |
| `n_obs` | `Int` | Total observations |
| `n_groups` | `Int` | Number of panel units |
| `n_treated` | `Int` | Number of ever-treated units |
| `n_control` | `Int` | Number of never-treated units |
| `method` | `Symbol` | Estimation method used |
| `outcome_var` | `String` | Outcome variable name |
| `treatment_var` | `String` | Treatment variable name |
| `control_group` | `Symbol` | Control group definition |
| `cluster` | `Symbol` | SE clustering level |
| `conf_level` | `T` | Confidence level |

!!! warning "TWFE Bias under Heterogeneity"
    When treatment effects vary across cohorts or over time, the TWFE estimator can be severely biased. It implicitly uses already-treated units as controls, which introduces negative weights on some group-time ATTs (Goodman-Bacon 2021). Use the heterogeneity-robust estimators below when treatment timing is staggered.

---

## Heterogeneity-Robust Estimators

All robust estimators share the `estimate_did` interface and return `DIDResult{T}`:

```julia
estimate_did(pd, outcome, treatment; method=:method_name, leads=3, horizon=5,
             control_group=:never_treated, cluster=:unit, conf_level=0.95)
```

### Callaway & Sant'Anna (2021)

Estimates group-time ATTs via outcome regression, then aggregates with cohort-size weights:

```julia
did_cs = estimate_did(pd, :gdp, :reform; method=:callaway_santanna,
                      leads=3, horizon=5, control_group=:never_treated)
```

The algorithm:
1. Identify treatment cohorts ``G = \{g_1, g_2, \ldots\}``
2. For each ``(g, t)``: compute ``\text{ATT}(g, t) = \mathbb{E}[Y_t - Y_{g-1} \mid G = g] - \mathbb{E}[Y_t - Y_{g-1} \mid C]``
3. Aggregate to event-time: ``\text{ATT}(e) = \sum_g w_g \cdot \text{ATT}(g, g+e)``

The `control_group` keyword controls the comparison group ``C``:
- `:never_treated` (default) -- only units with ``G_i = \infty``
- `:not_yet_treated` -- units not yet treated at time ``t``

The `group_time_att` field of the result stores the full ``n_{\text{cohorts}} \times n_{\text{periods}}`` matrix of ATT(g,t) estimates.

!!! note "Control Group Choice"
    Using `:not_yet_treated` increases the effective control sample but requires a stronger parallel trends assumption (across all cohorts, not just vs never-treated). When there are few never-treated units, `:not_yet_treated` may be necessary for precision.

### Sun & Abraham (2021)

The interaction-weighted estimator runs per-cohort TWFE regressions (each cohort vs the control group) with event-time dummies for **all** relative periods, then aggregates with cohort-size weights:

```julia
did_sa = estimate_did(pd, :gdp, :reform; method=:sun_abraham,
                      leads=3, horizon=5)
```

This avoids the "forbidden comparisons" (using already-treated units as controls) that bias TWFE when treatment effects are heterogeneous across cohorts.

### Borusyak, Jaravel & Spiess (2024)

The imputation estimator follows a two-step procedure:

```julia
did_bjs = estimate_did(pd, :gdp, :reform; method=:bjs, leads=3, horizon=5)
```

1. Estimate unit and time fixed effects on the **untreated subsample** only
2. Impute counterfactual ``\hat{Y}_{it}(0)`` for treated observations
3. Compute cell-level treatment effects ``\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)``
4. Aggregate to event-time ATTs with cohort-size weights

!!! note "Efficiency and Precision"
    The BJS imputation estimator is efficient under homoskedasticity and uses all available pre-treatment data for imputation. It naturally handles unbalanced panels and does not require specifying a control group explicitly.

### de Chaisemartin & D'Haultfoeuille (2020)

The `did_multiplegt` estimator uses first-differences and bootstrap inference:

```julia
using Random; Random.seed!(42)
did_dcdh = estimate_did(pd, :gdp, :reform; method=:did_multiplegt,
                        leads=3, horizon=5, n_boot=200)
```

For each cohort and event-time, it computes a first-difference DiD effect and aggregates with cohort-size weights. Standard errors are obtained via unit-level block bootstrap.

---

## Diagnostics

### Bacon Decomposition

The Goodman-Bacon (2021) decomposition reveals the TWFE estimator as a weighted average of all possible 2x2 DiD comparisons. Three types of comparisons arise:

- **Treated vs Untreated**: a treated cohort vs never-treated units
- **Earlier vs Later**: an earlier-treated cohort vs a later-treated cohort (before the later cohort's treatment)
- **Later vs Earlier**: a later-treated cohort vs an already-treated earlier cohort (the problematic comparison)

```julia
bd = bacon_decomposition(pd, :gdp, :reform)
report(bd)
```

| Field | Type | Description |
|:---|:---|:---|
| `estimates` | `Vector{T}` | 2x2 DiD estimates |
| `weights` | `Vector{T}` | Corresponding weights (sum to 1) |
| `comparison_type` | `Vector{Symbol}` | `:treated_vs_untreated`, `:earlier_vs_later`, or `:later_vs_earlier` |
| `cohort_i` | `Vector{Int}` | First cohort in each 2x2 comparison |
| `cohort_j` | `Vector{Int}` | Second cohort (0 for never-treated) |
| `overall_att` | `T` | Weighted average (equals the TWFE estimate) |

!!! warning "Later vs Earlier Comparisons"
    The "later vs earlier" comparisons use already-treated units as controls. If treatment effects evolve over time, these comparisons are contaminated and can flip the sign of the overall estimate. Large weights on these comparisons signal that TWFE may be unreliable.

### Pre-Trend Test

The `pretrend_test` function performs a joint Wald test of the null hypothesis that all pre-treatment event-time coefficients are zero:

```math
H_0: \beta_{-K} = \beta_{-K+1} = \cdots = \beta_{-2} = 0
```

```julia
did = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
pt = pretrend_test(did)
pt.statistic   # Wald chi-squared statistic
pt.pvalue      # p-value (high = no evidence against parallel trends)
```

| Field | Type | Description |
|:---|:---|:---|
| `statistic` | `T` | Wald chi-squared (or F) statistic |
| `pvalue` | `T` | p-value |
| `df` | `Int` | Degrees of freedom (number of pre-treatment periods) |
| `pre_coefficients` | `Vector{T}` | Pre-treatment event-time coefficients |
| `pre_se` | `Vector{T}` | Standard errors of pre-treatment coefficients |
| `test_type` | `Symbol` | `:f_test` or `:wald` |

!!! note "Pre-testing Bias (Roth 2022)"
    Conditioning on passing a pre-trend test can bias post-treatment estimates. A non-rejection does not prove parallel trends hold -- it only means the data cannot reject them at the given sample size. Consider HonestDiD sensitivity analysis (below) as a complement to pre-trend testing.

### Negative Weight Check

The de Chaisemartin & D'Haultfoeuille (2020) diagnostic checks whether the TWFE estimator assigns **negative weights** to some group-time ATTs:

```julia
nw = negative_weight_check(pd, :reform)
nw.has_negative_weights   # true if any weights are negative
nw.n_negative             # count of negative-weight cells
nw.total_negative_weight  # sum of all negative weights
```

| Field | Type | Description |
|:---|:---|:---|
| `has_negative_weights` | `Bool` | `true` if any TWFE weights are negative |
| `n_negative` | `Int` | Number of group-time cells with negative weights |
| `total_negative_weight` | `T` | Sum of negative weights |
| `weights` | `Vector{T}` | All TWFE weights |
| `cohort_time_pairs` | `Vector{Tuple{Int,Int}}` | (cohort, time) for each weight |

Negative weights mean the TWFE estimate can have the opposite sign of every underlying ``\text{ATT}(g, t)``. When negative weights are detected, switch to one of the heterogeneity-robust estimators.

---

## HonestDiD Sensitivity Analysis

The Rambachan & Roth (2023) HonestDiD framework constructs **robust confidence intervals** that remain valid even if parallel trends are violated by a bounded amount. The key parameter ``\bar{M}`` controls the maximum allowed violation magnitude per period:

```math
\left|\delta_{t+1} - \delta_t\right| \leq \bar{M}
```

where ``\delta_t = \mathbb{E}[Y_{it}(0) \mid G_i = g] - \mathbb{E}[Y_{it}(0) \mid G_i = \infty]`` is the trend violation at time ``t``. Under this relative magnitudes restriction, the worst-case bias at post-treatment event-time ``e \geq 0`` accumulates as ``\bar{M} \cdot (e + 1)``, widening the confidence interval:

```math
\text{Robust CI}_e = \left[\hat{\beta}_e - \bar{M}(e+1) - z_{\alpha/2} \cdot \text{SE}_e, \quad \hat{\beta}_e + \bar{M}(e+1) + z_{\alpha/2} \cdot \text{SE}_e\right]
```

```julia
did = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
h = honest_did(did; Mbar=1.0, conf_level=0.95)

h.robust_ci_lower     # robust lower bounds per post-period
h.robust_ci_upper     # robust upper bounds per post-period
h.breakdown_value     # smallest Mbar that overturns significance
```

The **breakdown value** ``\bar{M}^*`` is the smallest violation bound at which the robust confidence interval for at least one post-treatment period includes zero:

```math
\bar{M}^* = \min_e \frac{\max\left(|\hat{\beta}_e| - z_{\alpha/2} \cdot \text{SE}_e,\; 0\right)}{e + 1}
```

A large breakdown value indicates that the result is robust to substantial departures from parallel trends.

| Field | Type | Description |
|:---|:---|:---|
| `Mbar` | `T` | Violation bound used |
| `robust_ci_lower` | `Vector{T}` | Robust CI lower bounds per post-period |
| `robust_ci_upper` | `Vector{T}` | Robust CI upper bounds per post-period |
| `original_ci_lower` | `Vector{T}` | Original CIs for comparison |
| `original_ci_upper` | `Vector{T}` | Original CIs for comparison |
| `breakdown_value` | `T` | Smallest ``\bar{M}`` that overturns significance |
| `post_event_times` | `Vector{Int}` | Post-treatment event-time grid |
| `post_att` | `Vector{T}` | Post-treatment ATT point estimates |
| `conf_level` | `T` | Confidence level |

!!! note "Choosing Mbar"
    ``\bar{M} = 0`` recovers the original (unconditional) confidence interval. ``\bar{M} = 1`` allows violations equal to the pre-trend slope. Start with ``\bar{M} \in \{0.5, 1.0, 2.0\}`` to explore sensitivity. If the breakdown value exceeds the magnitude of pre-treatment coefficient fluctuations, the result is credibly robust.

---

## Visualization

All DiD result types support `plot_result` for interactive D3.js visualization.

### Event Study Plot

```julia
did = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
plot_result(did)
```

@raw html
<iframe src="../assets/plots/did_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>

### Bacon Decomposition Plot

```julia
bd = bacon_decomposition(pd, :gdp, :reform)
plot_result(bd)
```

@raw html
<iframe src="../assets/plots/did_bacon.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>

### HonestDiD Sensitivity Plot

```julia
h = honest_did(did; Mbar=1.0)
plot_result(h)
```

@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>

---

## Complete Example

```julia
using MacroEconometricModels
using Random; Random.seed!(42)

# ── 1. Simulate staggered adoption panel ──────────────────────────
N, T_periods = 200, 20
group_id = repeat(1:N, inner=T_periods)
time_id = repeat(1:T_periods, outer=N)
treat_time = [i <= 60 ? 8 : i <= 140 ? 12 : 0 for i in 1:N]
treat_col = Float64[treat_time[g] for g in group_id]

# Heterogeneous treatment effects: early cohort +2.0, late cohort +3.5
fe_i = randn(N); fe_t = 0.5 * randn(T_periods)
y = [fe_i[g] + fe_t[t] +
     (treat_time[g] == 8  && t >= 8  ? 2.0 + 0.3*(t - 8)  : 0.0) +
     (treat_time[g] == 12 && t >= 12 ? 3.5 + 0.1*(t - 12) : 0.0) +
     randn()
     for (g, t) in zip(group_id, time_id)]

using DataFrames
df = DataFrame(group=group_id, time=time_id, gdp=y, reform=treat_col)
pd = xtset(df, :group, :time)

# ── 2. Diagnostics: check for TWFE problems ──────────────────────
bd = bacon_decomposition(pd, :gdp, :reform)
println("Bacon decomposition: $(length(bd.estimates)) 2x2 comparisons")

nw = negative_weight_check(pd, :reform)
println("Negative weights: $(nw.has_negative_weights) (n=$(nw.n_negative))")

# ── 3. Estimate with multiple methods ────────────────────────────
did_twfe = estimate_did(pd, :gdp, :reform; method=:twfe, leads=3, horizon=5)
did_cs   = estimate_did(pd, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
did_sa   = estimate_did(pd, :gdp, :reform; method=:sun_abraham, leads=3, horizon=5)
did_bjs  = estimate_did(pd, :gdp, :reform; method=:bjs, leads=3, horizon=5)

println("Overall ATT -- TWFE: $(round(did_twfe.overall_att, digits=3)), ",
        "CS: $(round(did_cs.overall_att, digits=3)), ",
        "SA: $(round(did_sa.overall_att, digits=3)), ",
        "BJS: $(round(did_bjs.overall_att, digits=3))")

# ── 4. Pre-trend test ────────────────────────────────────────────
pt = pretrend_test(did_cs)
println("Pre-trend test: stat=$(round(pt.statistic, digits=3)), p=$(round(pt.pvalue, digits=3))")

# ── 5. HonestDiD sensitivity ─────────────────────────────────────
h = honest_did(did_cs; Mbar=1.0, conf_level=0.95)
println("Breakdown value: $(round(h.breakdown_value, digits=3))")

# ── 6. Visualize ─────────────────────────────────────────────────
plot_result(did_cs)
plot_result(bd)
plot_result(h)
```

---

## References

- Borusyak, Kai, Xavier Jaravel, and Jann Spiess. 2024. "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies* 91 (6): 3253--3285. [https://doi.org/10.1093/restud/rdae007](https://doi.org/10.1093/restud/rdae007)
- Callaway, Brantly, and Pedro H. C. Sant'Anna. 2021. "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225 (2): 200--230. [https://doi.org/10.1016/j.jeconom.2020.12.001](https://doi.org/10.1016/j.jeconom.2020.12.001)
- de Chaisemartin, Clement, and Xavier D'Haultfoeuille. 2020. "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review* 110 (9): 2964--2996. [https://doi.org/10.1257/aer.20181169](https://doi.org/10.1257/aer.20181169)
- Goodman-Bacon, Andrew. 2021. "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics* 225 (2): 254--277. [https://doi.org/10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)
- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90 (5): 2555--2591. [https://doi.org/10.1093/restud/rdad018](https://doi.org/10.1093/restud/rdad018)
- Roth, Jonathan. 2022. "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights* 4 (3): 305--322. [https://doi.org/10.1257/aeri.20210236](https://doi.org/10.1257/aeri.20210236)
- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics* 225 (2): 175--199. [https://doi.org/10.1016/j.jeconom.2020.09.006](https://doi.org/10.1016/j.jeconom.2020.09.006)
