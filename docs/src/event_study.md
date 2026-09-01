# [Event Study LP](@id event_study_page)

**MacroEconometricModels.jl** provides two LP-based event study estimators for causal inference in panel settings: the **Event Study LP** (Jorda 2005; Acemoglu, Naidu, Restrepo & Robinson 2019) and the **LP-DiD** estimator (Dube, Girardi, Jorda & Taylor 2025) with clean control samples, switching indicator treatment, and time-only fixed effects. The package achieves full parity with Stata `lpdid` v1.0.2.

- **Event Study LP**: Horizon-by-horizon local projections with switching indicator treatment and time-only FE
- **LP-DiD**: Clean control sample restrictions (absorbing/non-absorbing/one-off), pre-mean differencing, pooled estimates
- **DDCG dataset**: Built-in Acemoglu et al. (2019) democracy-GDP panel (184 countries, 1960--2010)
- **Panel utilities**: `panel_lag`, `panel_lead`, `panel_diff` for within-group transformations
- **Diagnostics**: Pre-trend tests and HonestDiD sensitivity analysis

Both estimators here identify dynamic treatment effects one horizon at a time. The cohort-aggregation approach to the same designs — Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, and the Bacon and negative-weight diagnostics — lives on [Difference-in-Differences](@ref did_page), which also carries a table comparing the two families. For panel models identified from covariates rather than treatment timing, see [Panel Regression](@ref panel_reg_page) and [Panel VAR](@ref pvar_page).

```@setup event
using MacroEconometricModels, Random, DataFrames, Statistics
Random.seed!(42)
ddcg = load_example(:ddcg)
```

## Quick Start

**Recipe 1: Standard Event Study LP**

```@example event
# 50 units, 20 periods; half receive treatment at t = 10 with a true effect of 1.0
Random.seed!(101)
N, T_per = 50, 20
df = DataFrame(
    group = repeat(1:N, inner=T_per),
    time = repeat(1:T_per, outer=N),
    outcome = randn(N * T_per) .+ [i <= 25 && t >= 10 ? 1.0 : 0.0
        for i in 1:N for t in 1:T_per],
    treat = Float64.([i <= 25 ? 10 : 0 for i in 1:N for _ in 1:T_per])
)
pd_sim = xtset(df, :group, :time)

eslp = estimate_event_study_lp(pd_sim, :outcome, :treat, 5; leads=3, lags=2)
report(eslp)
```

**Recipe 2: LP-DiD with absorbing treatment**

```@example event
r_sim = estimate_lp_did(pd_sim, :outcome, :treat, 5; pre_window=3, ylags=2)
report(r_sim)
```

**Recipe 3: LP-DiD on the DDCG democracy panel**

```@example event
# Democracy and GDP: Acemoglu et al. (2019). Democratic status reverses, so the
# clean control sample must be non-absorbing.
ddcg = load_example(:ddcg)
r = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, ylags=4, nonabsorbing=5)
report(r)
```

```julia
plot_result(r; title="Democracy -> GDP (DDCG)")
```

**Recipe 4: Pooled estimates**

```@example event
r_pool = estimate_lp_did(ddcg, :y, :dem, 10;
    pre_window=5, ylags=4, nonabsorbing=5,
    post_pooled=(0, 10),   # Average effect over h = 0,...,10
    pre_pooled=(1, 5)      # Pre-treatment placebo over h = -5,...,-1
)
report(r_pool)
```

**Recipe 5: Pre-trend test**

```@example event
pt = pretrend_test(eslp)
report(pt)
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

```@example event
eslp = estimate_event_study_lp(pd_sim, :outcome, :treat, 5;
    leads=3,             # Pre-treatment horizons K
    lags=2,              # Lagged outcome controls
    cluster=:unit,       # :unit, :time, or :twoway
    conf_level=0.95
)
report(eslp)
```

The simulation puts a true effect of 1.0 from ``t = 10`` onward, and the estimator recovers it: post-treatment coefficients average close to unity (0.4684, 0.8327, 0.7893, 1.4007, 0.7464, 0.8755), all but the impact effect significant at 5%. The pre-treatment coefficients behave as a placebo should — 0.3590 at ``h = -3`` with a standard error of 0.2780. The ``h = -2`` row prints as numerically zero because with `lags=2` the two-period pre-treatment long difference is an exact linear combination of the included outcome lags, a mechanical consequence discussed under Diagnostics below.

The effective sample shrinks with the horizon — `eslp.T_eff` runs 650, 625, 600, 575, 550, 525 from ``h = 0`` to ``h = 5`` — because each additional horizon requires one more observed future period:

```@example event
(event_times = eslp.event_times, T_eff = eslp.T_eff,
 n_obs = eslp.n_obs, n_groups = eslp.n_groups)
```

```julia
plot_result(eslp)
```

```@raw html
<iframe src="../assets/plots/eslp_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

`estimate_event_study_lp` runs ``K + H + 1`` separate OLS regressions, one per event-time horizon, each with cluster-robust standard errors. The resulting coefficients ``\beta_h`` trace out the dynamic treatment effect path. Because the regressions are separate, the estimator stores no cross-horizon covariance, which is why `honest_did` falls back to a diagonal covariance on an `EventStudyLP` result.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `leads` | `Int` | `3` | Pre-treatment horizons ``K`` |
| `lags` | `Int` | `4` | Lagged outcome controls |
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
| `outcome_var` / `treatment_var` | `String` | Column names the fit used |
| `n_obs` | `Int` | Total panel observations |
| `n_groups` | `Int` | Number of panel units |
| `lags` | `Int` | Number of lagged controls |
| `leads` | `Int` | Pre-treatment window |
| `horizon` | `Int` | Maximum horizon ``H`` |
| `clean_controls` | `Bool` | Whether clean-control restrictions were applied |
| `cluster` | `Symbol` | Clustering level |
| `conf_level` | `T` | Confidence level of the reported intervals |
| `data` | `PanelData{T}` | The panel the model was fitted on |

---

## LP-DiD (Dube et al. 2025)

The LP-DiD estimator adds **clean control sample** (CCS) restrictions. At each horizon ``h``, the control group contains only units whose treatment status does not change between ``t`` and ``t + h``. This prevents already-treated units from contaminating the control group under heterogeneous treatment effects.

### Clean Control Samples

Three CCS specifications match the Stata `lpdid` package.

**Absorbing treatment** (default): a ``(i, t)`` pair belongs to the CCS at horizon ``h`` if the unit is switching (``\Delta D_{it} = 1``) or treatment status remains at zero through ``t + h``. This is the right choice when treatment, once received, is permanent.

**Non-absorbing treatment**: treatment may reverse. A pair belongs to the CCS if no switches occurred in the stabilization window of ``L`` periods before ``t``. Democratic status in the DDCG panel reverses, so this is the correct specification for it:

```@example event
r_abs = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, ylags=4)
r_non = estimate_lp_did(ddcg, :y, :dem, 10; pre_window=5, ylags=4, nonabsorbing=5)

(absorbing    = round.(r_abs.coefficients[end-3:end], digits=3),
 nonabsorbing = round.(r_non.coefficients[end-3:end], digits=3))
```

The last four horizons differ substantially — 2.560, 3.959, 4.299 under the absorbing rule against 2.109, 3.590, 5.156 under the non-absorbing one — because the absorbing rule treats countries that democratized and then reverted as permanently treated, putting genuinely untreated country-years on the wrong side of the comparison. The gap is the cost of the wrong CCS, and it grows with the horizon as more reversals accumulate.

!!! warning "Match the CCS to the treatment process"
    The `:absorbing` default is silently wrong for reversible treatments, and nothing in the
    output flags it — only the `Treatment` line of the `report` header records which rule
    was used. Check whether the treatment indicator ever falls from 1 back to 0 before
    accepting the default.

**One-off treatment**: treatment lasts exactly one period. Requires `nonabsorbing`:

```@example event
r_oo = estimate_lp_did(ddcg, :y, :dem, 10; nonabsorbing=5, ylags=4, oneoff=true)
(specification = r_oo.specification, nobs = r_oo.nobs_per_horizon[1:4])
```

### Pre-Mean Differencing (PMD)

Instead of long differencing ``Y_{t+h} - Y_{t-1}``, PMD uses the average of pre-treatment outcomes as baseline. This reduces noise from a single pre-treatment period:

```math
Y_{i,t+h} - \bar{Y}_{i,\text{pre}} = \gamma_t^h + \beta_h \, \Delta D_{it} + \mathbf{X}_{it}'\boldsymbol{\delta}^h + \varepsilon_{i,t+h}
```

where:
- ``\bar{Y}_{i,\text{pre}}`` is the average of ``Y_{i,t-1}, Y_{i,t-2}, \ldots`` over a window of pre-treatment periods

```@example event
# :max uses the cumulative pre-treatment mean; an integer uses a k-period window
r_pmd = estimate_lp_did(ddcg, :y, :dem, 10;
                        pre_window=5, ylags=4, nonabsorbing=5, pmd=:max)
round.(r_pmd.coefficients[end-3:end], digits=3)
```

!!! note "PMD and outcome lags overlap"
    With `pmd=k` and `ylags` at least as large as ``k``, the pre-treatment mean is already
    an exact linear combination of the included outcome lags, so it is partialled out and
    the estimates are numerically identical to the long-difference specification. PMD only
    changes the answer when it reaches further back than the controls do — `pmd=:max` with a
    small `ylags` is the configuration where it matters.

### IPW Reweighting

Dube et al. (2025) propose inverse probability weights that equalize the weight each calendar period receives in the pooled average treatment effect, correcting for compositional change in the treatment-control balance. With `reweight=true` every observation in the clean control sample carries

```math
w_{it} = \frac{\Delta D_{it}}{p_t} + \frac{1 - \Delta D_{it}}{1 - p_t}
```

where:
- ``\Delta D_{it}`` is the switching indicator
- ``p_t`` is the period-specific treatment propensity, the share of switchers among the observations that survive the clean control sample at that horizon

```@example event
r_rw = estimate_lp_did(ddcg, :y, :dem, 10;
                       pre_window=5, ylags=4, nonabsorbing=5, reweight=true)
(unweighted = round.(r_non.coefficients[end-3:end], digits=3),
 reweighted = round.(r_rw.coefficients[end-3:end], digits=3))
```

The weights enter the fit twice: the time fixed effects are partialled out with weighted period means, and the horizon regression is then WLS in ``\sqrt{w_{it}}``. Both steps are needed — demeaning with unweighted period means and only then weighting the regression does not produce the weighted within estimator. Because each period contributes total weight ``2 n_t`` and a weighted treated share of one half, reweighting strips out the ``p_t (1 - p_t)`` variance weighting that the default OLS fit applies: unweighted LP-DiD loads on the calendar periods where the treated share is closest to one half, while the reweighted fit lets every period count in proportion to its sample size, which is the equally weighted ATE of Dube et al. (2025).

On DDCG the two answers separate at long horizons — 5.156 unweighted against 2.161 reweighted at ``h = 10`` — because democratic transitions cluster in a handful of calendar years that the variance weighting favours. Standard errors fall as well (2.97 against 4.00 at ``h = 10``), and the per-horizon sample sizes are unchanged, since weighting reweights observations rather than dropping them. The `report` header records the choice as `Reweighted: Yes (IPW)`.

### Pooled Estimates

Pooled regressions average the left-hand side over a window of horizons, producing a single average treatment effect and a single pre-treatment placebo:

```@example event
r_pool = estimate_lp_did(ddcg, :y, :dem, 10;
    pre_window=5, ylags=4, nonabsorbing=5,
    post_pooled=(0, 10),   # Average effect over h = 0,...,10
    pre_pooled=(1, 5)      # Pre-treatment placebo over h = -5,...,-1
)
(post = r_pool.pooled_post, pre = r_pool.pooled_pre)
```

Averaged over the first eleven years, a democratic transition raises log GDP per capita by 1.52 points — the DDCG outcome is scaled by 100, so this is about 1.5% — with a standard error of 2.48 and therefore no significance at conventional levels. The pre-treatment placebo is ``-0.028`` with a standard error of 0.136, indistinguishable from zero, which is the result the specification is supposed to deliver: no differential pre-trend between countries that democratize and those that do not. The pooled estimates are stored in `r.pooled_post` and `r.pooled_pre` as named tuples with fields `coef`, `se`, `ci_lower`, `ci_upper`, and `nobs`.

### Full Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `pre_window` | `Int` | `3` | Pre-treatment event-time ``K`` |
| `post_window` | `Int` | `H` | Post-treatment event-time |
| `ylags` | `Int` | `0` | Outcome lags (``L_1.Y, \ldots, L_k.Y``) |
| `dylags` | `Int` | `0` | Differenced outcome lags (``L_1.\Delta Y, \ldots``) |
| `covariates` | `Vector{String}` | `String[]` | Additional covariates |
| `nonabsorbing` | `Union{Nothing,Int}` | `nothing` | Stabilization window ``L`` for non-absorbing CCS |
| `oneoff` | `Bool` | `false` | One-off treatment (requires `nonabsorbing`) |
| `notyet` | `Bool` | `false` | Restrict to not-yet-treated controls |
| `nevertreated` | `Bool` | `false` | Restrict to never-treated controls |
| `firsttreat` | `Bool` | `false` | Use only first treatment event per unit |
| `pmd` | `Union{Nothing,Symbol,Int}` | `nothing` | Pre-mean differencing (`:max` or integer ``k``) |
| `reweight` | `Bool` | `false` | IPW reweighting toward the equally weighted ATE |
| `nocomp` | `Bool` | `false` | Restrict to obs in CCS at all horizons |
| `cluster` | `Symbol` | `:unit` | SE clustering: `:unit`, `:time`, or `:twoway` |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `post_pooled` | `Union{Nothing,Tuple{Int,Int}}` | `nothing` | ``(start, end)`` for pooled post-treatment |
| `pre_pooled` | `Union{Nothing,Tuple{Int,Int}}` | `nothing` | ``(start, end)`` for pooled pre-treatment |
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
| `nobs_per_horizon` | `Vector{Int}` | Effective sample size per horizon (0 at the reference) |
| `pooled_post` | `NamedTuple` or `nothing` | Pooled post-treatment estimate |
| `pooled_pre` | `NamedTuple` or `nothing` | Pooled pre-treatment estimate |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices per horizon |
| `outcome_var` / `treatment_var` | `String` | Column names the fit used |
| `T_obs` | `Int` | Total panel observations |
| `n_groups` | `Int` | Number of panel units |
| `specification` | `Symbol` | `:absorbing`, `:nonabsorbing`, or `:oneoff` |
| `pmd` | `Union{Nothing,Symbol,Int}` | PMD specification |
| `reweight` / `nocomp` | `Bool` | Flags as passed |
| `ylags` / `dylags` | `Int` | Outcome and differenced-outcome lag counts |
| `pre_window` / `post_window` | `Int` | Event-time windows |
| `cluster` | `Symbol` | Clustering level |
| `conf_level` | `T` | Confidence level of the reported intervals |
| `data` | `PanelData{T}` | The panel the model was fitted on |

---

## DDCG Dataset

The built-in DDCG dataset contains 184 countries from 1960--2010 with log GDP per capita and a binary democracy indicator from Acemoglu, Naidu, Restrepo & Robinson (2019):

| Variable | Description |
|----------|-------------|
| `y` | Log GDP per capita, scaled by 100 (so coefficients read as percent) |
| `dem` | Democracy indicator (0/1) |

The `dem` variable records democratic transitions (0 to 1) **and reversals** (1 to 0), which makes this a non-absorbing treatment setting: every example on this page passes `nonabsorbing=5`. Acemoglu et al. control for four lags of GDP, which is the `ylags=4` in the baseline specification. Extending the horizon to 25 years traces the full long-run path:

```@example event
r_long = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=4, nonabsorbing=5, post_pooled=(0, 25))
(h10 = round(r_long.coefficients[findfirst(==(10), r_long.event_times)], digits=3),
 h25 = round(r_long.coefficients[end], digits=3),
 pooled = round(r_long.pooled_post.coef, digits=3),
 pooled_se = round(r_long.pooled_post.se, digits=3))
```

The path is hump-shaped rather than monotone: the effect builds from 5.2% at ten years to a peak of 14.34% at fifteen (standard error 6.82, significant at 5%), stays significant through ``h = 17``, and then recedes to 6.9% by twenty-five years, where the standard error of 9.77 leaves it indistinguishable from zero. Horizons 12 through 17 are the ones the data speak to; beyond that the clean control sample has thinned enough that the intervals swamp the point estimates. The 25-year pooled average of 2.93% (standard error 3.73) averages the strong middle horizons with the noisy late ones and is therefore insignificant, which is why the per-horizon path is the more informative summary here.

---

## Panel Utilities

Within-group lag, lead, and difference operations for `PanelData`. These respect panel group boundaries: lags, leads, and differences never cross from one unit to another, and cells with no within-group predecessor are `NaN`:

```@example event
l1 = panel_lag(ddcg, :y, 1)     # L1.y
f1 = panel_lead(ddcg, :y, 1)    # F1.y
dy = panel_diff(ddcg, :y)       # delta y = y - L1.y

(observed = length(dy), finite = count(isfinite, dy),
 mean_growth = round(mean(filter(isfinite, dy)), digits=3))
```

Of 9384 country-year cells, 6968 have a within-country predecessor and therefore a defined first difference; the remainder are first observations or sit after a gap. The average annual change in the scaled log GDP series is 1.835, i.e. about 1.8% growth per year.

The `add_panel_*` variants return a new `PanelData` with the derived series appended under a conventional name:

```@example event
(lagged  = add_panel_lag(ddcg, :y, 1).varnames,
 led     = add_panel_lead(ddcg, :y, 1).varnames,
 diffed  = add_panel_diff(ddcg, :y).varnames)
```

---

## Clustering

Both estimators support three clustering options for standard error computation:

- **`:unit`** (default) --- accounts for serial correlation within units
- **`:time`** --- accounts for cross-sectional correlation within periods
- **`:twoway`** --- two-way clustering (Cameron, Gelbach & Miller 2011): ``V_{\text{twoway}} = V_{\text{unit}} + V_{\text{time}} - V_{\text{het}}``

```@example event
r_tw = estimate_lp_did(ddcg, :y, :dem, 10;
                       pre_window=5, ylags=4, nonabsorbing=5, cluster=:twoway)
(unit_se    = round(r_non.se[end], digits=4),
 twoway_se  = round(r_tw.se[end], digits=4))
```

At the ten-year horizon two-way clustering gives a standard error of 3.6217 against 4.0024 for unit clustering — slightly *smaller* here, which happens when the subtracted heteroskedasticity term outweighs the added time-cluster term. Two-way clustering is the right default when both serial correlation within countries and cross-sectional correlation across countries within a year are present, as in macroeconomic panels where global shocks hit every country at once. It requires enough clusters in *both* dimensions; with few time periods the time-cluster component is unreliable.

---

## Diagnostics

### Pre-Trend Test

Joint Wald test that all pre-treatment coefficients are zero:

```math
H_0: \beta_{-K} = \beta_{-K+1} = \cdots = \beta_{-2} = 0
```

where:
- ``\beta_k`` is the LP coefficient at event-time ``k``

```@example event
pt = pretrend_test(eslp)
report(pt)
```

The Wald statistic of 2.7246 on 2 degrees of freedom gives ``p = 0.2561``, so there is no evidence against parallel trends in the simulated panel — correctly, since the data were generated with none. The test excludes the reference period and uses the per-horizon standard errors, since separate horizon regressions supply no cross-horizon covariance.

!!! warning "Outcome lags can make placebos mechanically zero"
    A pre-treatment coefficient at horizon ``h < 0`` regresses ``Y_{t+h} - Y_{t-1}`` on the
    switching indicator while controlling for ``L_1.Y, \ldots, L_k.Y``. Whenever
    ``|h| \leq k`` that long difference is an exact linear combination of the controls, so
    the coefficient and its standard error are numerically zero and the horizon carries no
    information. With `ylags=4` only ``h = -5`` and beyond are informative placebos. Set
    `pre_window` greater than `ylags` for the pre-trend test to test anything.

### HonestDiD Sensitivity Analysis

Rambachan & Roth (2023) robust confidence intervals under bounded violations of parallel trends:

```@example event
h = honest_did(eslp; Mbar=1.0)
report(h)
```

Allowing post-treatment violations as large as the worst observed pre-trend widens every interval past zero: the ``h = 3`` effect, conventionally ``1.4007`` with a CI of ``[0.8821, 1.9193]``, becomes ``[-2.2755, 5.0769]``. The reported breakdown value is 0 because the impact effect at ``h = 0`` is already insignificant conventionally, and the breakdown scan stops at the first post-treatment period that admits zero. Event-study LP results are handled with a **diagonal** covariance — the estimator stores no cross-horizon covariance, and `honest_did` emits a warning to that effect — which makes these robust intervals conservative relative to the joint-covariance version available for the Callaway-Sant'Anna results on [Difference-in-Differences](@ref did_page).

```julia
plot_result(h)
```

```@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

See [Difference-in-Differences](@ref did_page) for the full HonestDiD methodology, including the ``\Delta^{SD}`` smoothness restriction and the `bacon_decomposition` and `negative_weight_check` diagnostics.

---

## Visualization

`plot_result` produces interactive D3.js event study plots for both `EventStudyLP` and `LPDiDResult`:

```julia
p = plot_result(eslp)
save_plot(p, "eslp_event_study.html")
```

```@raw html
<iframe src="../assets/plots/eslp_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

---

## Complete Example

```@example event
# LP-DiD: effect of democracy on log GDP per capita, Acemoglu et al. (2019) design
r_full = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5,
    ylags=4,
    nonabsorbing=5,        # democratic status reverses
    post_pooled=(0, 25),
    pre_pooled=(1, 5)
)
report(r_full)
```

```@example event
# Robustness: cumulative pre-treatment mean as the baseline
r_pmd_full = estimate_lp_did(ddcg, :y, :dem, 25;
    pre_window=5, ylags=4, nonabsorbing=5, pmd=:max)

(baseline_h25 = round(r_full.coefficients[end], digits=3),
 pmd_h25      = round(r_pmd_full.coefficients[end], digits=3))
```

```julia
plot_result(r_full; title="Democracy -> GDP (LP-DiD, DDCG)")
```

The baseline specification estimates the causal effect of democratic transitions on log GDP per capita using a switching indicator, four lags of the outcome, and a non-absorbing clean control sample with a five-year stabilization window. The dynamic path is positive from the second year onward, peaks at 14.34% around fifteen years — significant at 5%, as are horizons 14 and 17 — and then recedes to 6.9% by twenty-five years as the clean control sample thins and standard errors widen. The pre-treatment pooled placebo of ``-0.028`` (standard error 0.136) is the specification's main credential: countries that democratize were not on a different GDP trajectory beforehand. The 25-year pooled average of 2.93% is insignificant because it averages the well-identified middle horizons with the very noisy late ones, so the defensible summary is the medium-run effect rather than the pooled number.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example event
path = joinpath(mktempdir(), "event_study.jld2")
save_model(eslp, path)
eslp2 = load_model(path)
typeof(eslp2)
```

---

## Common Pitfalls

1. **Leaving the CCS absorbing when treatment reverses**: `:absorbing` is the default and is silently wrong for reversible treatments such as democratic status, because it treats reverted units as permanently treated. Check whether the indicator ever falls from 1 to 0 and pass `nonabsorbing=L` when it does.

2. **Reading mechanically-zero placebos as evidence**: pre-treatment coefficients at horizons within `ylags` of the reference are exact linear combinations of the outcome-lag controls, so they are numerically zero regardless of the data. Only horizons beyond `ylags` test anything; set `pre_window > ylags`.

3. **Treatment column format**: `estimate_lp_did` auto-detects binary (0/1) versus timing (year values), unlike the estimators on [Difference-in-Differences](@ref did_page), which require timing. Mixing formats in one column (0, 1, 2019) causes misclassification.

4. **Small effective samples at long horizons**: CCS restrictions reduce the sample at each horizon as more units switch treatment status. Monitor `r.nobs_per_horizon` and reduce ``H`` if counts drop below roughly 30 observations. The reference period always reports 0.

5. **Combining `notyet` and `nevertreated`**: these are mutually exclusive. `notyet` uses units not yet treated at ``t+h`` as controls; `nevertreated` uses only units that are never treated. Specifying both raises an error.

6. **`oneoff` requires `nonabsorbing`**: one-off treatment is a special case of non-absorbing treatment where the indicator lasts exactly one period. Calling `oneoff=true` without `nonabsorbing` raises an error.

7. **Comparing weighted and unweighted estimates as if they targeted the same quantity**: `reweight=true` estimates the equally weighted ATE, the default estimates the variance-weighted one. A gap between them is compositional, not a bug — it says the effect differs across the calendar periods where treatment is unbalanced.

8. **Redundant PMD**: `pmd=k` has no effect when `ylags` is at least ``k``, because the pre-treatment mean is then spanned by the controls. Use `pmd=:max` with a small `ylags` if pre-mean differencing is the point.

---

## References

- Acemoglu, Daron, Suresh Naidu, Pascual Restrepo, and James A. Robinson. 2019. "Democracy Does Cause Growth."
  *Journal of Political Economy* 127 (1): 47--100. [DOI](https://doi.org/10.1086/700936)

- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2011. "Robust Inference with Multiway Clustering."
  *Journal of Business & Economic Statistics* 29 (2): 238--249. [DOI](https://doi.org/10.1198/jbes.2010.07136)

- Dube, Arindrajit, Daniele Girardi, Oscar Jorda, and Alan M. Taylor. 2025. "A Local Projections Approach to Difference-in-Differences."
  *Journal of Applied Econometrics* 40 (7): 741--758. [DOI](https://doi.org/10.1002/jae.70000)

- Jorda, Oscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
  *American Economic Review* 95 (1): 161--182. [DOI](https://doi.org/10.1257/0002828053828518)

- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends."
  *Review of Economic Studies* 90 (5): 2555--2591. [DOI](https://doi.org/10.1093/restud/rdad018)
