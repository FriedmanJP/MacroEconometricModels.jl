# [Difference-in-Differences](@id did_page)

**MacroEconometricModels.jl** provides a comprehensive Difference-in-Differences (DiD) toolkit for staggered treatment designs. The package implements five heterogeneity-robust estimators, Bacon decomposition diagnostics, pre-trend testing, negative weight checks, and HonestDiD sensitivity analysis.

- **TWFE**: Traditional two-way fixed effects event study regression
- **Callaway & Sant'Anna (2021)**: Group-time ATTs via outcome regression with cohort-size aggregation
- **Sun & Abraham (2021)**: Interaction-weighted estimator avoiding forbidden comparisons
- **Borusyak, Jaravel & Spiess (2024)**: Imputation estimator using only untreated subsample
- **de Chaisemartin & D'Haultfoeuille (2020)**: First-difference DiD with bootstrap inference
- **Diagnostics**: Bacon decomposition, pre-trend test, negative weight check, HonestDiD sensitivity

This page covers **static and event-time treatment effects** identified from treatment timing. For the local-projection formulation of the same design — horizon-by-horizon regressions with clean control samples — see [Event Study LP](@ref event_study_page); the two pages are complementary, and the choice between them is discussed in [Relation to Event Study LP](@ref). For panel models identified from covariates rather than treatment timing, see [Panel Regression](@ref panel_reg_page) and [Panel VAR](@ref pvar_page).

```@setup did
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)
pd = load_example(:mpdta)
```

## Quick Start

**Recipe 1: TWFE event study**

```@example did
# Built-in Callaway & Sant'Anna (2021) minimum wage dataset
pd = load_example(:mpdta)

# TWFE event study: teen employment and minimum wage
did = estimate_did(pd, "lemp", "first_treat"; method=:twfe, leads=3, horizon=3)
report(did)
```

**Recipe 2: Callaway-Sant'Anna (heterogeneity-robust)**

```@example did
# Group-time ATTs with never-treated controls
did_cs = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                      leads=3, horizon=3, control_group=:never_treated)
report(did_cs)
```

```julia
plot_result(did_cs)
```

```@raw html
<iframe src="../assets/plots/did_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

**Recipe 3: Sun-Abraham interaction-weighted estimator**

```@example did
did_sa = estimate_did(pd, "lemp", "first_treat"; method=:sun_abraham,
                      leads=3, horizon=3)
report(did_sa)
```

**Recipe 4: Bacon decomposition diagnostics**

```@example did
# Decompose the static TWFE estimate into 2x2 comparisons
bd = bacon_decomposition(pd, "lemp", "first_treat")
report(bd)
```

```julia
plot_result(bd)
```

```@raw html
<iframe src="../assets/plots/did_bacon.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

**Recipe 5: HonestDiD sensitivity analysis**

```@example did
h = honest_did(did_cs; Mbar=1.0, conf_level=0.95)
report(h)
```

```julia
plot_result(h)
```

```@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
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

---

## Data Preparation

DiD estimation requires a `PanelData` object with an outcome variable and a treatment timing variable. The treatment column records **when** each unit first receives treatment (not a binary indicator).

### Built-in Dataset: mpdta

The `mpdta` dataset from Callaway & Sant'Anna (2021) contains county-level minimum wage data for 500 US counties over 2003--2007. Three treatment cohorts (2004, 2006, 2007) and 309 never-treated counties:

```@example did
did = estimate_did(pd, "lemp", "first_treat";
                   method=:callaway_santanna, leads=3, horizon=3)
report(did)
```

The panel is 500 counties observed 2003-2007, of which 191 are ever treated and 309 never are. Employment falls after a minimum-wage increase and the effect grows: ``-0.0199`` on impact, ``-0.0510`` after one year, and ``-0.1373`` after two, on a log outcome, so the two-year effect is roughly a 13% employment decline. The aggregate ATT of ``-0.0772`` (SE 0.0200) is the cohort-size-weighted average across post-treatment event times.

| Variable | Description |
|----------|-------------|
| `lemp` | Log of county-level teen employment (outcome) |
| `lpop` | Log of county population |
| `first_treat` | Year state first raised minimum wage; 0 = never-treated |

### Synthetic Data

For simulation studies, construct a staggered adoption panel with known treatment effects. The **Complete Example** below builds one with cohort-specific dynamics and uses it to show TWFE failing where the robust estimators succeed.

### Custom Cohort Specification

By default, DiD methods derive cohorts from the treatment timing column. For custom cohort definitions (e.g. geographic clusters or pre-treatment characteristics), pass a `cohort` column to `xtset`:

```julia
df.region_cohort = [g <= 60 ? 1 : g <= 140 ? 2 : 0 for g in df.group]
pd_cohort = xtset(df, :group, :time; cohort=:region_cohort)

# DiD methods use region_cohort instead of deriving from treatment timing
did_grouped = estimate_did(pd_cohort, :gdp, :reform; method=:callaway_santanna)
```

When `cohort_id` is `nothing` (the default), cohorts are inferred from the treatment column. `PanelData.cohort_id` takes precedence wherever it is set: `estimate_did`, `bacon_decomposition`, and `negative_weight_check` all rebuild their cohort assignment from it rather than from the treatment timing.

Cohort values are stored in the time index's own encoding — calendar years stay calendar years, non-integer times are ranked ``1, \ldots, T`` — with `0` and `missing` reserved for never-treated units and every other value read as an adoption period. A value matching no period in the sample is kept verbatim and warned about once: it still serves as a categorical grouping, which is what `absorb=:cohort` on the [panel regression](@ref panel_reg_page) estimators consumes, but no DiD estimator forms a treatment group from it. Labels that happen to coincide with sample periods are indistinguishable from adoption dates, so on a panel indexed ``1, \ldots, T`` give geographic cohorts labels outside that range.

!!! note "Treatment column is a period, not a flag"
    Every function on this page reads the treatment column as the **period of first
    treatment** (`2004`), with `0` or `NaN` for never-treated units, and requires the value
    to be constant within a unit. Passing a binary 0/1 indicator silently produces cohorts
    `{0, 1}` and meaningless event times. `estimate_lp_did` on the
    [Event Study LP](@ref event_study_page) page is the exception: it auto-detects either
    encoding.

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

```@example did
did_twfe = estimate_did(pd, "lemp", "first_treat"; method=:twfe, leads=3, horizon=3)
report(did_twfe)
```

On this dataset TWFE and Callaway-Sant'Anna happen to agree closely — aggregate ATTs of ``-0.0719`` and ``-0.0772`` — because the treatment effects are similar across the three cohorts. The pre-treatment coefficients are the warning sign: TWFE reports ``0.0231`` at ``e = -3`` and ``0.0221`` at ``e = -2``, both marginally significant, while Callaway-Sant'Anna puts ``e = -2`` at ``-0.0006``. Apparent pre-trends that vanish under a heterogeneity-robust estimator are a hallmark of contamination from already-treated comparison units rather than genuine violations of parallel trends. The Complete Example at the end of this page shows a design where the two diverge dramatically.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:twfe` | Estimation method (see below) |
| `leads` | `Int` | `0` | Pre-treatment event-time window ``K`` |
| `horizon` | `Int` | `5` | Post-treatment horizon ``H`` |
| `control_group` | `Symbol` | `:never_treated` | `:never_treated` or `:not_yet_treated` |
| `cluster` | `Symbol` | `:unit` | SE clustering: `:unit`, `:time`, or `:twoway` |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `base_period` | `Symbol` | `:varying` | `:varying` or `:universal` (Callaway-Sant'Anna only) |
| `n_boot` | `Int` | `200` | Bootstrap replications (de Chaisemartin-D'Haultfoeuille only) |
| `covariates` | `Vector{String}` | `String[]` | Additional controls (TWFE only) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Generator for the dCDH bootstrap |

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
| `outcome_var` / `treatment_var` | `String` | Column names the fit used |
| `control_group` | `Symbol` | `:never_treated` or `:not_yet_treated` |
| `cluster` | `Symbol` | Clustering level applied |
| `conf_level` | `T` | Confidence level of `ci_lower`/`ci_upper` |
| `att_vcov` | `Union{Matrix{T}, Nothing}` | Joint cross-horizon covariance of `att`; `nothing` when the estimator does not supply one |
| `base_period` | `Symbol` | `:varying` or `:universal`; every estimator other than Callaway-Sant'Anna reports `:universal` |

!!! warning "`att` includes the reference period"
    `att` and `event_times` are the same length and both contain the reference index, so
    `att[findfirst(==(-1), event_times)]` is **not** guaranteed to be zero. Under
    `base_period=:universal` — TWFE, Sun-Abraham, BJS, dCDH, and Callaway-Sant'Anna when
    asked for it — the cell is a structural `0.0` with a zero standard error, printed as
    `(ref) —`. Under `base_period=:varying` it holds the estimated placebo ATT(g, g−1)
    (``-0.0245`` on `mpdta`), which `report` prints as an ordinary row. Aggregating `att`
    yourself therefore requires filtering on `base_period`, not on `event_times`.

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

```@example did
# Universal base period: every comparison is against g-1
did_univ = estimate_did(pd, "lemp", "first_treat"; method=:callaway_santanna,
                        leads=3, horizon=3, base_period=:universal)
report(did_univ)
```

The base period changes only the pre-treatment coefficients — post-treatment estimates are identical to the `:varying` fit at ``-0.0199``, ``-0.0510``, ``-0.1373``, ``-0.1008``, because both definitions compare ``Y_t`` to ``Y_{g-1}`` after treatment. Before treatment they differ: `:varying` reports adjacent-period changes (``0.0305`` at ``e=-3``, significant at 5%) while `:universal` reports cumulative deviations from ``g-1`` (``0.0250``, insignificant). Neither is more correct; `:varying` is more sensitive to a one-off blip in a single year and `:universal` to a slow drift, so disagreement between them localizes where a pre-trend lives.

The two definitions also differ over ``e = -1``. Under `:universal` that cell *is* the normalization and prints as `(ref) —`. Under `:varying` it is the estimable placebo ``\text{ATT}(g, g-1)``, the adjacent change from ``g-2`` to ``g-1``, which the table above reports as an ordinary row: ``-0.0245`` with a standard error of 0.0142, matching the R `did` package. Reading it as omitted throws away the pre-treatment period closest to treatment — the one most likely to reveal anticipation — which is why `pretrend_test` includes it and tests three coefficients rather than two here. The `:universal` fit puts the same information in its ``e = -2`` coefficient of ``0.0245``, the sign-flipped counterpart, because it accumulates from ``g-1`` backwards.

The `group_time_att` field stores the full ``n_{\text{cohorts}} \times n_{\text{periods}}`` matrix of ``\text{ATT}(g,t)`` estimates, with `NaN` in cells that no comparison identifies:

```@example did
round.(did_cs.group_time_att, digits=4)
```

Rows are the cohorts in `did_cs.cohorts` (2004, 2006, 2007) and columns are calendar years 2003-2007. The 2004 cohort, observed for four post-treatment years, shows the effect building from ``-0.0105`` to ``-0.1373``; the 2007 cohort has only one post-period. Event-time aggregation averages down the diagonals of this matrix with cohort-size weights, which is why the ``e=3`` estimate rests entirely on the 2004 cohort.

### Sun & Abraham (2021)

The interaction-weighted estimator runs per-cohort TWFE regressions (each cohort vs the control group) with event-time dummies for **all** relative periods, then aggregates with cohort-size weights:

```@example did
did_sa = estimate_did(pd, "lemp", "first_treat"; method=:sun_abraham,
                      leads=3, horizon=3)
report(did_sa)
```

Sun-Abraham reproduces Callaway-Sant'Anna's post-treatment path almost exactly (``-0.0199``, ``-0.0510``, ``-0.1373``, ``-0.1008``) and the same aggregate ATT of ``-0.0772``. That is expected: with a common never-treated control group and no covariates, the interaction-weighted estimator and the group-time aggregation are algebraically very close, and they differ from each other mainly in how standard errors are computed. Both avoid the "forbidden comparisons" — using already-treated units as controls — that bias TWFE when treatment effects are heterogeneous across cohorts.

### Borusyak, Jaravel & Spiess (2024)

The imputation estimator follows a two-step procedure:

1. Estimate unit and time fixed effects on the **untreated subsample** only
2. Impute counterfactual ``\hat{Y}_{it}(0)`` for treated observations
3. Compute cell-level treatment effects ``\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)``
4. Aggregate to event-time ATTs with cohort-size weights

!!! note "Efficiency and Precision"
    The BJS imputation estimator is efficient under homoskedasticity and uses all available pre-treatment data for imputation. It naturally handles unbalanced panels and does not require specifying a control group explicitly.

```@example did
did_bjs = estimate_did(pd, "lemp", "first_treat"; method=:bjs,
                       leads=3, horizon=3)
report(did_bjs)
```

BJS gives the largest aggregate ATT of the four robust estimators, ``-0.0810``, and a slightly larger impact effect (``-0.0311`` against ``-0.0199``). Its pre-treatment coefficients are exactly zero with zero standard errors, and this is mechanical rather than an empirical finding: the fixed effects are fitted on the untreated subsample, so an untreated cell's imputed counterfactual equals its own fitted value and ``\hat{\tau}_{it} = 0`` identically. BJS therefore offers **no pre-trend test** — use `pretrend_test` on a Callaway-Sant'Anna or Sun-Abraham fit instead, and read the BJS result as the efficient estimate conditional on parallel trends holding.

### de Chaisemartin & D'Haultfoeuille (2020)

The `did_multiplegt` estimator uses first-differences and bootstrap inference:

```@example did
Random.seed!(99)
did_dcdh = estimate_did(pd, "lemp", "first_treat"; method=:did_multiplegt,
                        leads=3, horizon=3, n_boot=50)
report(did_dcdh)
```

The point estimates match Sun-Abraham to four decimals because both build from the same clean cohort-by-event-time comparisons; only the standard errors differ, since dCDH obtains them from a unit-level block bootstrap rather than a closed-form cluster formula. The bootstrap is the reason this estimator alone accepts `n_boot` and `rng` — 50 replications keeps the documentation build fast, but applied work should use several hundred, and results are only reproducible if the generator is seeded.

---

## Diagnostics

### Bacon Decomposition

The Goodman-Bacon (2021) decomposition reveals the TWFE estimator as a weighted average of all possible 2x2 DiD comparisons. Three types of comparisons arise:

- **Treated vs Untreated**: a treated cohort vs never-treated units (clean identification)
- **Earlier vs Later**: an earlier-treated cohort vs a later-treated cohort before the later cohort's treatment (valid comparison)
- **Later vs Earlier**: a later-treated cohort vs an already-treated earlier cohort (problematic --- uses treated units as controls)

```@example did
bd = bacon_decomposition(pd, "lemp", "first_treat")
report(bd)
```

Three cohorts generate nine 2x2 comparisons. The three clean treated-vs-untreated comparisons carry 86% of the total weight (0.0818 + 0.2453 + 0.5357) and all point the same way, which is why TWFE is not badly misleading on this dataset. The problematic later-vs-earlier comparisons carry only 2.8% of the weight, but note their sign: 2006-vs-2004 returns ``+0.0543`` against a true negative effect, exactly the sign flip the decomposition exists to expose. The overall ATT of ``-0.0365`` is the weighted average of all nine.

```julia
plot_result(bd)
```

```@raw html
<iframe src="../assets/plots/did_bacon.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

| Field | Type | Description |
|-------|------|-------------|
| `estimates` | `Vector{T}` | 2x2 DiD estimates |
| `weights` | `Vector{T}` | Corresponding weights (sum to 1) |
| `comparison_type` | `Vector{Symbol}` | `:treated_vs_untreated`, `:earlier_vs_later`, or `:later_vs_earlier` |
| `cohort_i` | `Vector{Int}` | First cohort in each 2x2 comparison |
| `cohort_j` | `Vector{Int}` | Second cohort (0 for never-treated) |
| `overall_att` | `T` | Weighted average of the 2x2 estimates |

The weights sum to 1 and `overall_att` reproduces the **static** two-way fixed-effects coefficient — the regression of the outcome on a binary treated-and-post indicator with unit and time effects — to machine precision (``-0.036549`` here). It is *not* the same number as `estimate_did(...; method=:twfe).overall_att` (``-0.071894``), which averages the event-study coefficients over post-treatment horizons. The two are different estimands, and the decomposition applies to the static one.

!!! warning "Later vs Earlier Comparisons"
    The "later vs earlier" comparisons use already-treated units as controls. If treatment effects evolve over time, these comparisons are contaminated and can flip the sign of the overall estimate. Large weights on these comparisons signal that TWFE is unreliable.

### Pre-Trend Test

The `pretrend_test` function performs a joint Wald test of the null hypothesis that all pre-treatment event-time coefficients are zero:

```math
H_0: \beta_{-K} = \beta_{-K+1} = \cdots = \beta_{-1} = 0
```

where:
- ``\beta_k`` is the event-time coefficient at relative time ``k``
- ``K`` is the number of pre-treatment leads
- ``\beta_{-1}`` enters only under `base_period=:varying`, where it is estimated; under `:universal` it is the normalization and the null stops at ``\beta_{-2}``

```@example did
pt = pretrend_test(did_cs)
report(pt)
```

The joint Wald statistic is 7.3418 on 3 degrees of freedom, ``p = 0.0618``, so the three pre-treatment coefficients are not jointly distinguishable from zero at 5%, though the margin is thin enough that the design deserves the sensitivity analysis below rather than a clean bill of health. Note that this is a *joint* test: the individual ``e=-3`` coefficient of ``0.0305`` is significant at 5% on its own, and ``e=-1`` at ``-0.0245`` is significant at 10%, while ``e=-2`` is essentially zero. The same fit with `base_period=:universal` tests two coefficients instead of three and returns 3.0425 with ``p = 0.2184`` — a reminder that the two base periods pose different null hypotheses, not two versions of one. The test uses the full cross-horizon covariance when the estimator supplies `att_vcov` (Callaway-Sant'Anna does) and falls back to a diagonal form otherwise; coefficients with a zero standard error, which is how every estimator other than Callaway-Sant'Anna stores its reference period, are dropped.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Wald chi-squared (or F) statistic |
| `pvalue` | `T` | P-value |
| `df` | `Int` | Degrees of freedom (number of pre-treatment coefficients tested) |
| `pre_coefficients` | `Vector{T}` | Pre-treatment event-time coefficients |
| `pre_se` | `Vector{T}` | Standard errors of pre-treatment coefficients |
| `test_type` | `Symbol` | `:f_test` or `:wald` |

!!! note "Pre-testing Bias (Roth 2022)"
    Conditioning on passing a pre-trend test can bias post-treatment estimates. A non-rejection does not prove parallel trends hold --- it only means the data cannot reject them at the given sample size. Complement pre-trend testing with HonestDiD sensitivity analysis.

### Negative Weight Check

The de Chaisemartin & D'Haultfoeuille (2020) diagnostic checks whether the TWFE estimator assigns **negative weights** to some group-time ATTs:

```@example did
nw = negative_weight_check(pd, "first_treat")
(has_negative_weights = nw.has_negative_weights,
 n_negative           = nw.n_negative,
 total_negative_weight = round(nw.total_negative_weight, digits=4),
 n_cells              = length(nw.weights))
```

None of the seven identified group-time cells receives a negative weight here, so on `mpdta` the TWFE estimand is a genuine convex combination of underlying ATTs. That is consistent with the Bacon decomposition, which put only 2.8% of the weight on contaminated comparisons. Negative weights mean the TWFE estimate can carry the opposite sign of *every* underlying ``\text{ATT}(g, t)``; when the check reports any, switch to one of the heterogeneity-robust estimators rather than trying to interpret the TWFE coefficient. Pair the weights with `cohort_time_pairs` to see which cells are responsible.

| Field | Type | Description |
|-------|------|-------------|
| `has_negative_weights` | `Bool` | `true` if any TWFE weights are negative |
| `n_negative` | `Int` | Number of group-time cells with negative weights |
| `total_negative_weight` | `T` | Sum of negative weights |
| `weights` | `Vector{T}` | All TWFE weights |
| `cohort_time_pairs` | `Vector{Tuple{Int,Int}}` | (cohort, time) for each weight |

---

## Honest DiD Sensitivity Analysis

The Rambachan & Roth (2023) framework constructs **robust confidence sets** that remain valid under bounded violations of parallel trends. The event-study coefficients satisfy ``\hat{\beta} \sim N(\tau + \delta, \Sigma)`` with ``\tau_{\text{pre}} = 0``, so the observed pre-treatment coefficients reveal ``\delta_{\text{pre}}`` and discipline the possible post-treatment bias ``\delta_{\text{post}}``. Two restriction sets are supported.

**Relative magnitudes** ``\Delta^{RM}(\bar{M})`` (`restriction=:rm`, default) bounds post-treatment trend violations by ``\bar{M}`` times the largest observed pre-treatment violation:

```math
\left|\delta_{t+1} - \delta_t\right| \leq \bar{M} \cdot \max_{s < 0}\left|\delta_{s+1} - \delta_s\right|, \quad t \geq 0
```

where:
- ``\delta_t = \mathbb{E}[Y_{it}(0) \mid G_i = g] - \mathbb{E}[Y_{it}(0) \mid G_i = \infty]`` is the trend violation at time ``t``
- ``\bar{M}`` is dimensionless: ``\bar{M} = 1`` allows post-treatment violations as large as the worst pre-trend

The identified set for the period-``e`` effect follows in closed form from the observed pre-period first differences, and the robust CI widens the identified set by delta-method standard errors of its endpoints. Both the set and the CI scale one-for-one with the outcome and depend on the pre-treatment coefficients — perturbing a pre-period estimate moves the robust CI.

**Second differences** ``\Delta^{SD}(M)`` (`restriction=:sd`) bounds the change in the slope of ``\delta`` by ``M`` (in outcome units per period squared):

```math
\left|\delta_{t+1} - 2\delta_t + \delta_{t-1}\right| \leq M
```

Inference uses the Armstrong & Kolesár (2018) **fixed-length confidence interval** (FLCI): an affine estimator ``\hat{\theta} = a'\hat{\beta}_{\text{pre}} + l'\hat{\beta}_{\text{post}}`` whose pre-period weights cancel the unbounded linear-trend direction, with the worst-case bias over ``\Delta^{SD}(M)`` traded off against variance and the half-length ``cv_\alpha(\text{bias}/\text{sd}) \cdot \text{sd}`` minimized over the bias–variance frontier (``cv_\alpha(b)`` is the ``1-\alpha`` quantile of ``|N(b,1)|``).

The **breakdown value** is the smallest bound (``\bar{M}^*`` or ``M^*``) at which the robust confidence interval for at least one post-treatment period includes zero. A large breakdown value indicates that the result is robust to substantial departures from parallel trends.

`honest_did` uses the joint event-study covariance stored by the estimator (`att_vcov`) when available and falls back to a diagonal covariance from the per-period standard errors otherwise (with a warning).

```@example did
h = honest_did(did_cs; restriction=:rm, Mbar=1.0, conf_level=0.95)
report(h)
```

Allowing post-treatment trend violations as large as the worst observed pre-trend destroys every result: the ``e = 2`` effect, conventionally ``-0.1373`` with a CI of ``[-0.2087, -0.0658]``, widens to ``[-0.3705, 0.1019]`` and now includes zero. The reported breakdown value of 0 follows from the definition used here — the smallest bound at which *at least one* post-treatment period admits zero — and the ``e = 0`` effect is already insignificant conventionally, so no positive bound is required to overturn it. Read the per-period rows rather than the summary whenever the earliest post-period is the weakest one.

The smoothness restriction is selected with `restriction=:sd` and its bound `M`, which is in outcome units per period squared rather than dimensionless:

```@example did
h_sd = honest_did(did_cs; restriction=:sd, M=0.01)
report(h_sd)
```

Bounding the *curvature* of the trend violation at 0.01 log points per period squared is a far weaker restriction than ``\Delta^{RM}(1)``, and the robust intervals are correspondingly tighter — ``[-0.2537, 0.0288]`` at ``e = 2`` against ``[-0.3705, 0.1019]``. The two restrictions answer different questions: ``\Delta^{RM}`` asks how large a violation the data can absorb relative to what is already visible, while ``\Delta^{SD}`` asks how nonlinear the counterfactual trend could be. Report both when the pre-treatment window is short.

```julia
plot_result(h)
```

```@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `restriction` | `Symbol` | `:rm` | `:rm` (relative magnitudes) or `:sd` (second differences) |
| `Mbar` | `Real` | `1.0` | Relative-magnitudes bound (used when `restriction=:rm`) |
| `M` | `Real` | `0.0` | Smoothness bound in outcome units per period squared (used when `restriction=:sd`) |
| `conf_level` | `Real` | `0.95` | Confidence level |

The core method `honest_did(betahat, sigma; num_pre, num_post, ...)` takes the same keywords plus `num_pre`, `num_post`, and `l_vec`, the linear combination of post-period effects to target.

!!! note "Choosing Mbar"
    ``\bar{M} = 0`` recovers the original (conventional) confidence interval. ``\bar{M} = 1`` allows post-treatment violations equal to the worst pre-treatment violation. Start with ``\bar{M} \in \{0.5, 1.0, 2.0\}`` to explore sensitivity; the result is credibly robust when the breakdown value exceeds the violation magnitudes considered plausible.

The core method accepts raw event-study coefficients (pre-periods first, reference period omitted) with their joint covariance, mirroring the R `HonestDiD` package.

| Field | Type | Description |
|-------|------|-------------|
| `Mbar` | `T` | Relative-magnitudes bound (active for `:rm`) |
| `robust_ci_lower` | `Vector{T}` | Robust CI lower bounds per post-period |
| `robust_ci_upper` | `Vector{T}` | Robust CI upper bounds per post-period |
| `original_ci_lower` | `Vector{T}` | Conventional CIs for comparison |
| `original_ci_upper` | `Vector{T}` | Conventional CIs for comparison |
| `breakdown_value` | `T` | Smallest bound that overturns significance |
| `post_event_times` | `Vector{Int}` | Post-treatment event-time grid |
| `post_att` | `Vector{T}` | Post-treatment ATT point estimates |
| `conf_level` | `T` | Confidence level |
| `restriction` | `Symbol` | `:rm` (relative magnitudes) or `:sd` (second differences) |
| `M` | `T` | Smoothness bound (active for `:sd`) |
| `method` | `Symbol` | `:flci` (``\Delta^{SD}``) or `:delta_id` (``\Delta^{RM}``) |

---

## Visualization

All DiD result types support `plot_result` for interactive D3.js visualization. Plot calls stay in static blocks; the rendered output appears in the iframe beneath each one.

### Event Study Plot

```julia
plot_result(did_cs)
```

```@raw html
<iframe src="../assets/plots/did_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

### Bacon Decomposition Plot

```julia
plot_result(bd)
```

```@raw html
<iframe src="../assets/plots/did_bacon.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

### HonestDiD Sensitivity Plot

```julia
plot_result(h)
```

```@raw html
<iframe src="../assets/plots/did_honest.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

---

## Complete Example

The `mpdta` results above are reassuring because the three cohorts happen to have similar treatment effects. This simulation makes them differ — an early cohort with a +2.0 effect growing at +0.3 per period, a late cohort with +3.5 growing at +0.1 — which is precisely the setting where TWFE breaks.

```@example did
Random.seed!(2025)
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
pd_sim = xtset(df, :group, :time)

# Diagnostics first: is TWFE trustworthy here?
bd_sim = bacon_decomposition(pd_sim, :gdp, :reform)
report(bd_sim)
```

```@example did
# Estimate with every method
did_twfe_sim = estimate_did(pd_sim, :gdp, :reform; method=:twfe, leads=3, horizon=5)
did_cs_sim   = estimate_did(pd_sim, :gdp, :reform; method=:callaway_santanna, leads=3, horizon=5)
did_sa_sim   = estimate_did(pd_sim, :gdp, :reform; method=:sun_abraham, leads=3, horizon=5)
did_bjs_sim  = estimate_did(pd_sim, :gdp, :reform; method=:bjs, leads=3, horizon=5)

(twfe = round(did_twfe_sim.overall_att, digits=4),
 cs   = round(did_cs_sim.overall_att, digits=4),
 sa   = round(did_sa_sim.overall_att, digits=4),
 bjs  = round(did_bjs_sim.overall_att, digits=4))
```

```@example did
report(did_twfe_sim)
```

```@example did
report(did_cs_sim)
```

```@example did
# Pre-trend test on the robust estimate
pt_sim = pretrend_test(did_cs_sim)
report(pt_sim)
```

```@example did
# How much parallel-trends violation can the result withstand?
h_sim = honest_did(did_cs_sim; Mbar=1.0, conf_level=0.95)
report(h_sim)
```

```julia
plot_result(did_cs_sim)
plot_result(bd_sim)
plot_result(h_sim)
```

```@raw html
<iframe src="../assets/plots/did_event_study.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

The three robust estimators agree — 3.3365 for Callaway-Sant'Anna and Sun-Abraham, 3.3093 for BJS — while TWFE returns 1.0999, less than a third of the true average effect. The TWFE event-study path shows why the failure is not subtle: it reports pre-treatment coefficients of ``-1.2503`` and ``-1.2043``, large and precisely estimated, in a simulation where the untreated potential outcomes satisfy parallel trends by construction. Those "pre-trends" are an artifact of the late cohort being used as a control for the early one after its own treatment begins.

The Bacon decomposition isolates the mechanism: the later-vs-earlier comparison of cohort 12 against cohort 8 returns 1.8530 against a truth above 3, and it carries 15.6% of the total weight. The pre-trend test on the Callaway-Sant'Anna fit does not reject (``p = 0.7911`` across all three pre-periods, ``e = -1`` included), correctly, and the HonestDiD analysis reports a breakdown value of 1.4482 — the result survives post-treatment trend violations up to 1.45 times the largest observed pre-treatment violation, which is a genuinely robust finding rather than one propped up by the parallel-trends assumption.

---

## Relation to Event Study LP

Both this page and [Event Study LP](@ref event_study_page) estimate dynamic treatment effects from staggered adoption, and on clean designs they broadly agree. They differ in what they condition on and what they assume:

| | This page (`estimate_did`) | [Event Study LP](@ref event_study_page) (`estimate_lp_did`) |
|---|---|---|
| Unit of estimation | Group-time ATTs aggregated to event time | One regression per horizon |
| Treatment variable | Period of first treatment | Switching indicator ``\Delta D_{it}``, auto-detected encoding |
| Unit effects | Explicit unit fixed effects | Absorbed by long-differencing ``Y_{t+h} - Y_{t-1}`` |
| Reversible treatment | Not supported — cohorts are absorbing | `nonabsorbing` and `oneoff` clean control samples |
| Covariate lags | Contemporaneous covariates (TWFE only) | `ylags`, `dylags`, and arbitrary covariates per horizon |
| Reference baseline | Period ``g-1`` | Period ``t-1``, or a pre-treatment mean via `pmd` |

Choose the estimators here when treatment is absorbing and the question is a cohort-weighted ATT; choose LP-DiD when treatment can switch off, when the baseline should average several pre-periods, or when horizon-specific controls matter. `pretrend_test` and `honest_did` accept results from either.

---

## Common Pitfalls

1. **Parallel trends is untestable**: Pre-trend tests evaluate whether pre-treatment coefficients are jointly zero, but non-rejection does not prove parallel trends hold in the post-treatment period. Conditioning on passing a pre-trend test introduces pre-testing bias (Roth 2022). Always complement with HonestDiD sensitivity analysis.

2. **Negative weights in TWFE**: With staggered adoption, the TWFE estimator can assign negative weights to some group-time ATTs, potentially flipping the sign of the overall estimate. Run `negative_weight_check` before interpreting TWFE results. When negative weights are detected, switch to Callaway-Sant'Anna, Sun-Abraham, or BJS.

3. **Staggered adoption requires robust estimators**: The standard TWFE event study is only valid when all units adopt treatment simultaneously or effects are homogeneous. With staggered timing and heterogeneous effects, TWFE produces biased estimates and can manufacture pre-trends that do not exist, as the Complete Example on this page demonstrates.

4. **Never-treated group requirement**: Callaway-Sant'Anna and Sun-Abraham with `control_group=:never_treated` require a sufficient number of never-treated units. When all units eventually receive treatment, use `control_group=:not_yet_treated` (at the cost of a stronger parallel trends assumption) or the BJS imputation estimator.

5. **Treatment column format**: The treatment variable must contain the **period number** when treatment first occurs, not a binary 0/1 indicator. Passing a binary indicator causes the package to misidentify cohorts. Use `0` or `NaN` for never-treated units, and keep the value constant within each unit.

6. **Reading `att` at the reference index**: `att` has the same length as `event_times` and includes the reference period. Under `base_period=:universal` that cell is an exact zero; under Callaway-Sant'Anna's default `:varying` it is the estimated placebo ATT(g, g−1) and `report` prints it as an ordinary row. Check `result.base_period` before deciding whether to drop `event_times .== reference_period` — dropping it unconditionally discards a real estimate.

7. **Expecting a pre-trend test from BJS**: The imputation estimator fits its fixed effects on untreated cells only, so pre-treatment effects are identically zero with zero standard errors. Run `pretrend_test` on a Callaway-Sant'Anna or Sun-Abraham fit instead.

8. **Confusing the two TWFE estimands**: `bacon_decomposition(...).overall_att` reproduces the static two-way FE coefficient on a binary treated-and-post indicator; `estimate_did(...; method=:twfe).overall_att` averages event-study coefficients over post-treatment horizons. They differ (``-0.0365`` against ``-0.0719`` on `mpdta`) and the decomposition explains only the former.

9. **Unseeded dCDH bootstrap**: `method=:did_multiplegt` draws its standard errors from a block bootstrap. Pass `rng` or call `Random.seed!` for reproducible standard errors, and raise `n_boot` well above the default 200 for publication.

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

- Armstrong, Timothy B., and Michal Kolesár. 2018. "Optimal Inference in a Class of Regression Models."
  *Econometrica* 86 (2): 655--683. [DOI](https://doi.org/10.3982/ECTA14434)

- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends."
  *Review of Economic Studies* 90 (5): 2555--2591. [DOI](https://doi.org/10.1093/restud/rdad018)

- Roth, Jonathan. 2022. "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends."
  *American Economic Review: Insights* 4 (3): 305--322. [DOI](https://doi.org/10.1257/aeri.20210236)

- Sun, Liyang, and Sarah Abraham. 2021. "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects."
  *Journal of Econometrics* 225 (2): 175--199. [DOI](https://doi.org/10.1016/j.jeconom.2020.09.006)
