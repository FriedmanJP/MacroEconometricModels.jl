# Event Study & LP-DiD

This page documents two LP-based event study estimators: the **Event Study LP** (Jordà 2005; Acemoglu et al. 2019) and the **LP-DiD** estimator (Dube, Girardi, Jordà, and Taylor 2025) with clean control samples, switching indicator treatment, and time-only fixed effects.

- **Event Study LP** — horizon-by-horizon local projections with switching indicator treatment and time-only FE
- **LP-DiD** — full-featured engine matching Stata `lpdid` v1.0.2: absorbing/non-absorbing/one-off CCS, PMD, IPW reweighting, pooled estimates
- **DDCG dataset** — built-in Acemoglu et al. (2019) democracy-GDP panel (184 countries, 1960–2010)
- **Panel utilities** — `panel_lag`, `panel_lead`, `panel_diff` for within-group transformations

## Quick Start

```julia
using MacroEconometricModels, DataFrames, Random

# --- Recipe 1: Standard Event Study LP ---
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

# --- Recipe 2: LP-DiD with absorbing treatment ---
r = estimate_lp_did(pd, :outcome, :treat, 5; pre_window=3, ylags=2)
report(r)

# --- Recipe 3: LP-DiD on DDCG dataset ---
ddcg = load_example(:ddcg)
r_ddcg = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, ylags=1)
plot_result(r_ddcg; title="Democracy → GDP (DDCG)")

# --- Recipe 4: LP-DiD with PMD + reweighting ---
r_pmd = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, pmd=:max, reweight=true)

# --- Recipe 5: Pooled estimates ---
r_pool = estimate_lp_did(ddcg, :y, :dem, 10; post_pooled=(0, 5), pre_pooled=(1, 5))
r_pool.pooled_post   # (coef=..., se=..., ci_lower=..., ci_upper=..., nobs=...)
```

---

## Model Specification

Both estimators run separate regressions for each event-time horizon ``h \in \{-K, \ldots, -1, 0, 1, \ldots, H\}``:

```math
Y_{i,t+h} - Y_{i,t-1} = \gamma_t^h + \beta_h \Delta D_{it} + \mathbf{X}_{it}'\boldsymbol{\delta}^h + \varepsilon_{i,t+h}
```

where:
- ``\Delta D_{it} = D_{it} - D_{i,t-1}`` is the **switching indicator** (equals 1 only at treatment onset)
- ``\gamma_t^h`` is a time fixed effect (absorbed by within-time demeaning)
- ``\mathbf{X}_{it}`` includes lagged outcomes ``L_1.Y, \ldots, L_k.Y``, differenced lags ``\Delta Y_{t-l}``, and optional covariates
- ``\beta_h`` is the **dynamic treatment effect** at horizon ``h``

The reference period ``h = -1`` is normalized to zero.

!!! note "Time-Only Fixed Effects"
    Long differencing ``Y_{i,t+h} - Y_{i,t-1}`` absorbs unit fixed effects, so only time FE remain. This is consistent with both the ANRR (2019) specification and the Stata `lpdid` package.

!!! note "Switching Indicator vs Treatment Level"
    The treatment regressor is the first difference ``\Delta D_{it}``, not the treatment level ``D_{it}``. This ensures that only the treatment onset contributes to identification. Already-treated observations with ``\Delta D = 0`` and ``D = 1`` are excluded from the sample.

---

## Event Study LP

The standard estimator uses all switching (``\Delta D = 1``) and control (``D = 0``) observations:

```julia
eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
    leads=3,             # pre-treatment horizons K
    lags=2,              # lagged outcome controls
    covariates=String[], # additional controls
    cluster=:unit,       # :unit, :time, or :twoway
    conf_level=0.95
)
```

### Return Values (`EventStudyLP{T}`)

| Field | Type | Description |
|:------|:-----|:------------|
| `coefficients` | `Vector{T}` | Treatment effect ``\beta_h`` at each event-time |
| `se` | `Vector{T}` | Cluster-robust standard errors |
| `ci_lower`, `ci_upper` | `Vector{T}` | Confidence interval bounds |
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

### Full API

```julia
r = estimate_lp_did(pd, outcome, treatment, H;
    # Window
    pre_window=3,              # pre-treatment event-time K
    post_window=H,             # post-treatment event-time
    # Controls
    ylags=0,                   # outcome lags (L1.Y, ..., Lk.Y)
    dylags=0,                  # differenced outcome lags (L1.ΔY, ...)
    covariates=String[],       # additional covariates
    # CCS specification
    nonabsorbing=nothing,      # stabilization window L (Int) for non-absorbing
    oneoff=false,              # one-off treatment (requires nonabsorbing)
    # Control group restrictions
    notyet=false,              # restrict to not-yet-treated controls
    nevertreated=false,        # restrict to never-treated controls
    firsttreat=false,          # use only first treatment event per unit
    # Baseline
    pmd=nothing,               # pre-mean differencing (:max or Int k)
    # Weighting
    reweight=false,            # IPW for equally weighted ATE across time
    nocomp=false,              # restrict to obs in CCS at all horizons
    # Inference
    cluster=:unit,             # :unit, :time, :twoway
    conf_level=0.95,
    # Pooled estimates
    post_pooled=nothing,       # (start, end) tuple for pooled post-treatment
    pre_pooled=nothing,        # (start, end) tuple for pooled pre-treatment
    only_pooled=false,         # skip event study
    only_event=false           # skip pooled
)
```

### Clean Control Samples

Three CCS specifications match the Stata `lpdid` package:

**Absorbing treatment** (default): A ``(g, t)`` pair belongs to CCS at horizon ``h`` if the unit is switching (``\Delta D_{it} = 1``) or treatment status does not change through ``t + h`` (i.e., ``D_{i,t+h} = 0``).

```julia
r = estimate_lp_did(pd, :y, :treat, 10)  # absorbing is default
```

**Non-absorbing treatment**: Treatment may reverse. A pair belongs to CCS if no switches occurred in the stabilization window of ``L`` periods before ``t``:

```julia
r = estimate_lp_did(pd, :y, :treat, 10; nonabsorbing=5)
```

**One-off treatment**: Treatment lasts exactly one period. Requires `nonabsorbing`:

```julia
r = estimate_lp_did(pd, :y, :treat, 10; nonabsorbing=3, oneoff=true)
```

### Pre-Mean Differencing (PMD)

Instead of long differencing ``Y_{t+h} - Y_{t-1}``, PMD uses the average of pre-treatment outcomes as baseline:

```julia
# Use cumulative pre-treatment mean
r = estimate_lp_did(pd, :y, :treat, 10; pmd=:max)

# Use moving average of k pre-treatment periods
r = estimate_lp_did(pd, :y, :treat, 10; pmd=3)
```

### IPW Reweighting

Inverse probability weights ensure equally weighted ATE across time periods:

```julia
r = estimate_lp_did(pd, :y, :treat, 10; reweight=true)
```

### Pooled Estimates

Pooled regressions average the LHS over a window of horizons:

```julia
r = estimate_lp_did(pd, :y, :treat, 10;
    post_pooled=(0, 5),   # average effect over h=0,...,5
    pre_pooled=(1, 3)     # pre-treatment placebo over h=-3,...,-1
)

r.pooled_post   # (coef, se, ci_lower, ci_upper, nobs)
r.pooled_pre    # (coef, se, ci_lower, ci_upper, nobs)
```

### Return Values (`LPDiDResult{T}`)

| Field | Type | Description |
|:------|:-----|:------------|
| `coefficients` | `Vector{T}` | Treatment effect ``\beta_h`` at each event-time |
| `se` | `Vector{T}` | Cluster-robust standard errors |
| `ci_lower`, `ci_upper` | `Vector{T}` | Confidence interval bounds |
| `event_times` | `Vector{Int}` | Event-time grid ``[-K, \ldots, H]`` |
| `reference_period` | `Int` | Omitted period (``-1``) |
| `nobs_per_horizon` | `Vector{Int}` | Effective sample size per horizon |
| `pooled_post` | `NamedTuple` or `nothing` | Pooled post-treatment estimate |
| `pooled_pre` | `NamedTuple` or `nothing` | Pooled pre-treatment estimate |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices per horizon |
| `specification` | `Symbol` | `:absorbing`, `:nonabsorbing`, or `:oneoff` |
| `pmd` | type varies | PMD specification (`nothing`, `:max`, or `Int`) |
| `reweight` | `Bool` | IPW reweighting flag |
| `nocomp` | `Bool` | Nocomp restriction flag |
| `ylags`, `dylags` | `Int` | Lag specifications |
| `cluster` | `Symbol` | Clustering level |

---

## DDCG Dataset

The built-in DDCG dataset contains 184 countries from 1960–2010 with log GDP per capita and a binary democracy indicator from Acemoglu, Naidu, Restrepo, and Robinson (2019):

```julia
ddcg = load_example(:ddcg)

# Variables: y (log GDP per capita), dem (democracy 0/1)
r = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=1, post_pooled=(0, 25))
report(r)
```

---

## Panel Utilities

Within-group lag, lead, and difference operations for `PanelData`:

```julia
ddcg = load_example(:ddcg)

# Compute lag/lead/diff vectors
l1 = panel_lag(ddcg, :y, 1)     # L1.y
f1 = panel_lead(ddcg, :y, 1)    # F1.y
dy = panel_diff(ddcg, :y)       # Δy = y - L1.y

# Add as new columns (returns new PanelData)
ddcg2 = add_panel_lag(ddcg, :y, 1)    # adds "lag1_y"
ddcg3 = add_panel_lead(ddcg, :y, 1)   # adds "lead1_y"
ddcg4 = add_panel_diff(ddcg, :y)      # adds "d_y"
```

---

## Clustering

Both estimators support three clustering options:

- **`:unit`** (default) — accounts for serial correlation within units
- **`:time`** — accounts for cross-sectional correlation within periods
- **`:twoway`** — two-way clustering (Cameron, Gelbach, and Miller 2011): ``V_{\text{twoway}} = V_{\text{unit}} + V_{\text{time}} - V_{\text{het}}``

```julia
r = estimate_lp_did(ddcg, :y, :dem, 10; cluster=:twoway)
```

---

## Diagnostics

### Pre-Trend Test

Joint Wald test that all pre-treatment coefficients are zero:

```julia
pt = pretrend_test(eslp)
pt.statistic    # Wald chi-squared
pt.pvalue       # high = no evidence against parallel trends
```

### HonestDiD Sensitivity Analysis

Rambachan and Roth (2023) robust confidence intervals under bounded violations of parallel trends:

```julia
h = honest_did(eslp; Mbar=1.0)
h.robust_ci_lower     # lower bounds
h.robust_ci_upper     # upper bounds
h.breakdown_value     # smallest Mbar overturning significance
```

See [Difference-in-Differences](@ref did_page) for `bacon_decomposition` and `negative_weight_check`.

---

## Visualization

`plot_result` produces interactive D3.js event study plots for both `EventStudyLP` and `LPDiDResult`:

```julia
p = plot_result(eslp)
save_plot(p, "event_study.html")

p2 = plot_result(r; title="LP-DiD: Democracy → GDP")
save_plot(p2, "lpdid_ddcg.html")
```

@raw html
<iframe src="../assets/plots/eslp_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>

---

## Complete Example

```julia
using MacroEconometricModels, DataFrames

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

# Pooled estimates
println("Post-treatment pooled (h=0:25): β = ", round(r.pooled_post.coef, digits=3),
        " (SE = ", round(r.pooled_post.se, digits=3), ")")
println("Pre-treatment pooled (h=-5:-1): β = ", round(r.pooled_pre.coef, digits=3),
        " (SE = ", round(r.pooled_pre.se, digits=3), ")")

# Robustness: PMD + reweighting
r_pmd = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=1, pmd=:max, reweight=true)
println("PMD + IPW coefficients: ", round.(r_pmd.coefficients, digits=3))

# Plot
p = plot_result(r; title="Democracy → GDP (LP-DiD, DDCG)")
```

---

## Common Pitfalls

1. **Treatment column format**: `estimate_lp_did` auto-detects binary (0/1) vs timing (year values). Mixing formats (e.g., 0, 1, 2019) causes misclassification.

2. **Small effective samples at long horizons**: CCS restrictions reduce the sample at each horizon. Monitor `r.nobs_per_horizon` and reduce ``H`` if counts drop below ~30.

3. **Combining `notyet` and `nevertreated`**: These are mutually exclusive. `notyet` uses units not yet treated at ``t+h`` as controls; `nevertreated` uses only units with ``g_i = 0``.

4. **`oneoff` requires `nonabsorbing`**: One-off treatment is a special case of non-absorbing treatment where the treatment indicator lasts exactly one period.

5. **PMD with short pre-treatment windows**: `pmd=:max` uses all available pre-treatment data. With few pre-treatment periods, the average baseline may be noisy. Consider `pmd=k` with a small ``k``.

---

## References

- Jordà, Óscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161–182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Acemoglu, Daron, Suresh Naidu, Pascual Restrepo, and James A. Robinson. 2019. "Democracy Does Cause Growth." *Journal of Political Economy* 127 (1): 47–100. [https://doi.org/10.1086/700936](https://doi.org/10.1086/700936)
- Dube, Arindrajit, Daniele Girardi, Óscar Jordà, and Alan M. Taylor. 2025. "A Local Projections Approach to Difference-in-Differences." *Journal of Applied Econometrics*. [https://doi.org/10.1002/jae.3117](https://doi.org/10.1002/jae.3117)
- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90 (5): 2555–2591. [https://doi.org/10.1093/restud/rdad018](https://doi.org/10.1093/restud/rdad018)
- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2011. "Robust Inference with Multiway Clustering." *Journal of Business & Economic Statistics* 29 (2): 238–249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)
