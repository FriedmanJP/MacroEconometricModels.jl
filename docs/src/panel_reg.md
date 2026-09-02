# [Panel Regression](@id panel_reg_page)

**MacroEconometricModels.jl** provides a comprehensive panel regression module following Stata's `xtreg`/`xtivreg`/`xtlogit`/`xtprobit` conventions. The module covers linear panel models, panel instrumental variables, panel discrete choice, and six specification tests with five covariance estimators.

- **Linear panel** (`estimate_xtreg`): Fixed Effects, Random Effects (Swamy-Arora), First-Difference, Between, Correlated Random Effects (Mundlak 1978), Arellano-Bond, Blundell-Bond
- **High-dimensional FE** (`absorb`): Any number of categorical dimensions by alternating projections (Guimaraes & Portugal 2010; Correia 2016)
- **Panel IV** (`estimate_xtiv`): FE-IV, RE-IV/EC2SLS (Baltagi 1981), FD-IV, Hausman-Taylor (1981)
- **Panel logit** (`estimate_xtlogit`): Pooled, FE conditional (Chamberlain 1980), RE (Gauss-Hermite quadrature), CRE
- **Panel probit** (`estimate_xtprobit`): Pooled, RE, CRE (no FE — incidental parameters problem)
- **Panel marginal effects**: AME with delta-method SEs for panel logit/probit
- **Specification tests**: Hausman, Breusch-Pagan LM, F-test for FE, Pesaran CD, Wooldridge AR, Modified Wald
- **Covariance estimators**: Classical, entity-cluster (Arellano 1987), two-way cluster (Cameron-Gelbach-Miller 2011), Driscoll-Kraay (1998) HAC, Beck-Katz (1995) panel-corrected SE
- **Serial correlation**: Prais-Winsten AR(1) FGLS (`ar1=:common`/`:panel_specific`)

Panel regression estimates the *level* relationship between an outcome and its covariates. Three sibling pages cover the dynamic and causal extensions: [Panel VAR](@ref pvar_page) for multivariate feedback among several endogenous panel series, [Difference-in-Differences](@ref did_page) for static and event-time treatment effects under staggered adoption, and [Event Study LP](@ref event_study_page) for the local-projection version of the same design. Panel unit-root and cointegration pretests live on [Panel Tests](@ref tests_panel_page).

```@setup preg
using MacroEconometricModels, Random, DataFrames, Statistics
Random.seed!(42)

# ---- PWT: growth regression panel ----
pwt = load_example(:pwt)
df_pwt = DataFrame(pwt.data, pwt.varnames)
df_pwt.country = pwt.group_names[pwt.group_id]
df_pwt.year = pwt.time_id
# Filter valid observations and create log variables
valid = .!isnan.(df_pwt.rgdpna) .& .!isnan.(df_pwt.rkna) .& .!isnan.(df_pwt.emp) .&
        .!isnan.(df_pwt.pop) .& .!isnan.(df_pwt.hc) .& .!isnan.(df_pwt.labsh) .&
        .!isnan.(df_pwt.csh_i) .&
        (df_pwt.emp .> 0) .& (df_pwt.pop .> 0) .& (df_pwt.rgdpna .> 0) .& (df_pwt.rkna .> 0)
df_pwt = df_pwt[valid, :]
df_pwt.lngdppc = log.(df_pwt.rgdpna ./ df_pwt.pop)  # log GDP per capita
df_pwt.lnk = log.(df_pwt.rkna ./ df_pwt.emp)         # log capital per worker
pd_pwt = xtset(df_pwt, :country, :year)

# ---- DDCG: democracy and growth panel ----
ddcg = load_example(:ddcg)
df_ddcg = DataFrame(ddcg.data, ddcg.varnames)
df_ddcg.country = ddcg.group_names[ddcg.group_id]
df_ddcg.year = ddcg.time_id
valid_ddcg = .!isnan.(df_ddcg.y) .& .!isnan.(df_ddcg.dem)
df_ddcg = df_ddcg[valid_ddcg, :]
df_ddcg.lngdppc = df_ddcg.y ./ 100   # DDCG stores log GDP per capita x 100
pd_ddcg = xtset(df_ddcg, :country, :year)
```

## Quick Start

**Recipe 1: Fixed effects — growth regression**

```@example preg
# PWT: log GDP per capita on human capital and log capital per worker
m_fe = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk])
report(m_fe)
```

**Recipe 2: Random effects**

```@example preg
m_re = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:re)
report(m_re)
```

**Recipe 3: Hausman test (FE vs RE)**

```@example preg
ht = hausman_test(m_fe, m_re)
report(ht)
```

**Recipe 4: Panel IV (FE-IV with simulated endogeneity)**

```@example preg
Random.seed!(1)
# Synthetic panel: one endogenous regressor, two excluded instruments
N, T_p = 50, 20
n = N * T_p
df_iv = DataFrame(id=repeat(1:N, inner=T_p), t=repeat(1:T_p, N),
                  x=randn(n), z=randn(n), z2=randn(n))
alpha_i = repeat(randn(N), inner=T_p)
u = randn(n)                                        # shared error component (endogeneity source)
df_iv.x_endog = 0.5 .* df_iv.z .+ 0.4 .* df_iv.z2 .+ u .+ randn(n)
df_iv.wage = alpha_i .+ 1.5 .* df_iv.x .+ 2.0 .* df_iv.x_endog .+ u .+ randn(n)
pd_iv = xtset(df_iv, :id, :t)
m_iv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z, :z2])
report(m_iv)
```

**Recipe 5: Panel logit — democracy and development**

```@example preg
# DDCG: democracy (0/1) on log GDP per capita — Lipset modernization hypothesis
m_logit = estimate_xtlogit(pd_ddcg, :dem, [:lngdppc])
report(m_logit)
```

**Recipe 6: Specification test battery**

```@example preg
bp = breusch_pagan_test(m_re)
report(bp)
```

---

## Linear Panel Models

### Fixed Effects (Within Estimator)

The **within estimator** eliminates time-invariant unobserved heterogeneity by demeaning within each panel unit (Baltagi 2021):

```math
\tilde{y}_{it} = \tilde{x}_{it}' \beta + \tilde{e}_{it}
```

where:
- ``\tilde{y}_{it} = y_{it} - \bar{y}_i`` is the within-demeaned outcome
- ``\tilde{x}_{it} = x_{it} - \bar{x}_i`` is the within-demeaned regressor
- ``\beta`` is estimated by OLS on demeaned data
- Entity effects ``\hat{\alpha}_i = \bar{y}_i - \bar{x}_i' \hat{\beta}`` are recovered after estimation

The model reports three R-squared variants: **within** (variation within entities), **between** (variation of entity means), and **overall** (total variation).

```@example preg
# PWT: within-country variation in GDP per capita
m_fe = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; cov_type=:cluster)
report(m_fe)
```

Both elasticities are positive and precisely estimated: a one-unit rise in the human-capital index raises log GDP per capita by 0.4169 within a country, and a one-log-point rise in capital per worker raises it by 0.5161, close to the capital share a Cobb-Douglas production function would imply. The within R-squared of 0.9376 says these two inputs track almost all of the within-country movement in output per head once the country effect is removed. The contrast between `rho = 0.9789` and the between R-squared of 0.1294 is the substantive point: 98% of the residual variance is the permanent country effect ``\alpha_i``, so cross-country level differences dwarf anything the regressors explain in the cross-section, and only the within variation identifies ``\beta``.

**Two-way fixed effects** absorb both entity and time effects:

```@example preg
m_twoway = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; twoway=true)
report(m_twoway)
```

Adding year effects absorbs the global technology trend that all countries share, which is why the within R-squared drops from 0.9376 to 0.6034 while the capital coefficient barely moves (0.5161 to 0.5137). The human-capital coefficient rises to 0.5550 with a standard error nearly twice as large, since only the *deviation* of a country's human capital from the world average in that year now identifies it. `twoway=true` is exactly `absorb=[:entity, :time]` from the [High-Dimensional Fixed Effects](@ref) section: it removes entity and time effects by alternating projections rather than by the additive identity ``y_{it} - \bar{y}_i - \bar{y}_t + \bar{y}``, which is the two-way within transformation only on a balanced panel. For more than two dimensions, use `absorb`.

### Random Effects (GLS)

The **random effects** estimator treats entity effects as random draws from a distribution and uses GLS with the Swamy-Arora (1972) variance component estimator:

```math
y_{it} - \hat{\theta} \bar{y}_i = (x_{it} - \hat{\theta} \bar{x}_i)' \beta + (1 - \hat{\theta}) \mu + \text{error}
```

where:
- ``\hat{\theta} = 1 - \sqrt{\hat{\sigma}_e^2 / (\hat{\sigma}_e^2 + T_i \hat{\sigma}_u^2)}`` is the quasi-demeaning parameter
- ``\hat{\sigma}_e^2`` is from the within regression
- ``\hat{\sigma}_u^2`` is from the between regression (Swamy-Arora 1972)

```@example preg
m_re = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:re)
report(m_re)
```

The quasi-demeaning weight is ``\hat{\theta} = 0.9566``, so close to 1 that RE removes almost the whole entity mean and its slopes sit near the FE ones — 0.4808 versus 0.4169 for human capital, 0.4791 versus 0.5161 for capital. What RE buys is a smaller ``\hat{\sigma}_u`` (0.3731 against the FE value of 0.8444, since the entity effect is now a drawn component rather than a free parameter) and coefficients that remain identified for time-invariant regressors. RE is efficient only under ``E[\alpha_i \mid x_{it}] = 0``; the [Hausman Test (FE vs RE)](@ref) evaluates exactly that assumption.

### First-Difference

The **first-difference** estimator removes entity effects by differencing consecutive observations:

```math
\Delta y_{it} = \Delta x_{it}' \beta + \Delta e_{it}
```

```@example preg
m_fd = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:fd)
report(m_fd)
```

First-differencing gives a very different answer here: 0.2216 for human capital and 0.1463 for capital, roughly a third of the within estimates, with a within R-squared of 0.0203. FD and FE are both consistent under strict exogeneity and differ only in efficiency — FD is the better choice when ``e_{it}`` is close to a random walk, FE when it is close to white noise. The gap on PWT is a symptom, not a contradiction: annual differences of log GDP per capita are dominated by measurement error and business-cycle noise relative to the slow-moving levels that identify the within estimator, which attenuates the differenced slopes.

### Between Estimator

The **between estimator** regresses entity means on entity-mean regressors:

```math
\bar{y}_i = \bar{x}_i' \beta + \bar{\alpha}_i + \bar{e}_i
```

```@example preg
m_be = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:between, cov_type=:ols)
report(m_be)
```

The between regression collapses the 2375 country-year observations to 38 country means, so it answers a different question from FE: it compares *levels* across countries rather than changes within them. Human capital carries the cross-country variation (0.5715, t = 4.19) while capital per worker is insignificant (0.0378, t = 0.98) — the mirror image of the within estimates, where both matter. Because the regression is one observation per entity, there is nothing left to cluster on: `estimate_xtreg` reports classical OLS standard errors on the group-means regression and warns if a different `cov_type` is requested.

### Correlated Random Effects (Mundlak)

The **CRE** approach (Mundlak 1978) relaxes the RE exogeneity assumption by augmenting the RE model with group means of time-varying regressors:

```math
y_{it} = x_{it}' \beta + \bar{x}_i' \gamma + \alpha_i + e_{it}
```

```@example preg
m_cre = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:cre)
report(m_cre)
```

The slope estimates ``\hat{\beta}`` from CRE reproduce FE exactly — 0.4169 and 0.5161, matching the within regression to four decimals — which is the Mundlak equivalence result. The interesting coefficients are the group means: `hc_mean` at 1.2984 and `lnk_mean` at ``-0.3442``, both significant at 1%. Their joint significance is a direct test of ``E[\alpha_i \mid x_{it}] = 0``, and rejecting it says the country effect is systematically correlated with average human capital and capital intensity, so RE is inconsistent and FE (or CRE itself) is the right specification.

---

## High-Dimensional Fixed Effects

The `absorb` keyword removes any number of categorical fixed-effect dimensions by the **method of alternating projections** (Guimarães & Portugal 2010; Correia 2016), the algorithm behind Stata's `reghdfe`. Writing ``D`` for the stacked dummy design of all absorbed dimensions, the within transformation is the orthogonal projection

```math
M = I - D(D'D)^{-}D'
```

Forming ``D`` is impossible once the level counts are large — worker × firm, or firm × year × product. Alternating projections compute ``M v`` by cycling the single-dimension demeaning operators ``M_d = I - P_d``, each one O(n):

```math
v \leftarrow M_D \cdots M_2 M_1 v \qquad \text{(repeat until } v \text{ stops moving)}
```

where:
- ``P_d`` projects onto dimension ``d``'s dummy span, so ``M_d v`` subtracts each level's mean
- ``M_d`` is an orthogonal projection, so by von Neumann–Halperin the cycle converges to the projection onto ``\bigcap_d \operatorname{range}(M_d) = \operatorname{range}(M)``

OLS on the absorbed data is therefore the within-all-fixed-effects estimator, with no dummy ever materialized.

Dimension names resolve to panel variables, or to the reserved indices `:entity` (`:id`/`:unit`/`:group`), `:time` (`:period`), and `:cohort`:

```@example preg
m_hdfe = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; absorb=[:entity, :time])
report(m_hdfe)
```

This is the same estimator as `twoway=true` above — that keyword routes here. Absorbing entity and time reports 107 fixed-effect parameters: 38 countries plus 70 years minus the one collinearity implied by a single **mobility group** (see below). The distinction matters because the PWT panel is unbalanced, where the additive identity is not the two-way within transformation; it put the capital-deepening coefficient at 0.297 against the dummy-OLS truth of 0.514 that alternating projections recover to 1e-14.

Any number of dimensions is allowed, and they need not be nested. A common applied specification adds **group × time** effects, which control for shocks common to countries at a similar development level:

```@example preg
# Income tercile from each country's first observed GDP per capita
y0 = Dict(g.country[1] => g.lngdppc[1] for g in groupby(df_pwt, :country))
cuts = quantile(collect(values(y0)), [1/3, 2/3])
df_pwt.income_group = [y0[c] <= cuts[1] ? 1.0 : y0[c] <= cuts[2] ? 2.0 : 3.0
                       for c in df_pwt.country]
df_pwt.group_year = 10_000 .* df_pwt.income_group .+ df_pwt.year   # interacted dimension

pd_gy = xtset(df_pwt, :country, :year)
m_gy = estimate_xtreg(pd_gy, :lngdppc, [:hc, :lnk]; absorb=[:entity, :group_year])
report(m_gy)
```

**Degrees of freedom.** The absorbed-parameter count is the rank of the dummy design, and getting it right is where naive implementations fail. For two dimensions the rank is ``G_1 + G_2 - C``, where ``C`` is the number of **connected components** (Abowd, Creecy & Kramarz 2002 "mobility groups") of the bipartite graph linking the two dimensions' levels — not ``G_1 + G_2 - 1``. The example above reports ``C = 3``: countries never change income tercile, so the country ↔ group-year graph splits into exactly one component per tercile, and the design absorbs ``38 + 210 - 3 = 245`` parameters rather than 247.

!!! note "Three or more dimensions"
    No closed form for the rank exists beyond two dimensions. Each dimension past the second is charged one collinearity, `G_d - 1`, which is an **upper bound** on its contribution. Absorbed parameters are therefore never understated, so the residual degrees of freedom are never overstated and the small-sample correction errs conservative. Estimated coefficients are unaffected — they depend only on the range of the dummy design, not on the bookkeeping, and are invariant to the order the dimensions are listed in.

**Cluster-robust standard errors** charge only the fixed-effect dimensions *not* nested within the clustering variable. Entity fixed effects clustered on entity — the default panel setup — contribute nothing, because the ``G/(G-1)`` cluster factor already accounts for them. This is why `absorb=[:entity]` reproduces the plain one-way `estimate_xtreg` standard errors exactly, while `absorb=[:entity, :time]` — and equivalently `twoway=true` — additionally charges the ``T-1`` non-nested time parameters.

**Convergence.** Plain alternating projections converge linearly at a rate set by the angle between the dummy subspaces, which is punishing when the dimensions are weakly connected (sparse worker–firm mobility). `hdfe_accel=true` (the default) applies Irons–Tuck vector extrapolation, which is exact for a single geometric mode. The `MAP converged` line in `report` and the `hdfe.converged` field are authoritative — a `NO` there means the coefficients are still moving, and `hdfe_maxiter` should be raised:

```@example preg
m_hdfe.hdfe.converged, m_hdfe.hdfe.iterations, m_hdfe.hdfe.n_absorbed
```

Use [`absorb_fe`](@ref) directly when you want the residualized data rather than a fitted model:

```@example preg
y_raw = pd_pwt.data[:, findfirst(==("lngdppc"), pd_pwt.varnames)]
X_raw = pd_pwt.data[:, [findfirst(==(v), pd_pwt.varnames) for v in ("hc", "lnk")]]
a = absorb_fe(y_raw, X_raw, [pd_pwt.group_id, pd_pwt.time_id])
a.X \ a.y      # identical to coef(m_hdfe)
```

---

## Dynamic Panels: Arellano-Bond and Blundell-Bond

**Dynamic panel** estimators handle a lagged dependent variable using GMM with internal instruments. Both `:ab` and `:bb` add the lag of the outcome to the regressor list themselves — pass only the exogenous covariates in `indepvars` and the reported table gains an `L.growth` row.

**Arellano-Bond** (1991) differences the equation, which sweeps out ``\alpha_i`` at the cost of correlating ``\Delta y_{i,t-1}`` with ``\Delta e_{it}``, and instruments the differenced lag with lagged **levels** ``y_{i,t-2}, y_{i,t-3}, \ldots``:

```@example preg
Random.seed!(11)
# Synthetic dynamic panel: growth follows an AR(1) with persistence 0.5
N_d, T_d = 100, 10
n_d = N_d * T_d
alpha_d = randn(N_d)
invest, output = randn(n_d), randn(n_d)
growth = zeros(n_d)
for i in 1:N_d, s in 1:T_d
    k = (i - 1) * T_d + s
    prev = s == 1 ? alpha_d[i] / (1 - 0.5) : growth[k-1]   # start at the unit's own mean
    growth[k] = alpha_d[i] + 0.5 * prev + 0.8 * invest[k] - 0.3 * output[k] + randn()
end
df_dyn = DataFrame(id=repeat(1:N_d, inner=T_d), t=repeat(1:T_d, N_d),
                   invest=invest, output=output, growth=growth)
pd_dyn = xtset(df_dyn, :id, :t)
m_ab = estimate_xtreg(pd_dyn, :growth, [:invest, :output]; model=:ab)
report(m_ab)
```

Arellano-Bond recovers the design closely: the autoregressive coefficient is 0.4913 against a true 0.5, investment 0.8276 against 0.8, and output ``-0.2323`` against ``-0.3``. The diagnostics block is where a dynamic panel is judged. AR(1) rejects at ``p < 0.001`` and AR(2) does not (``p = 0.959``), which is exactly the signature of a correctly specified model: differencing makes the idiosyncratic error MA(1) by construction, so first-order correlation is expected while second-order correlation would indicate that ``e_{it}`` was serially correlated in levels and the lagged-level instruments are invalid. The Hansen J does not reject (``p = 0.491``) with 36 moment conditions against 100 groups, comfortably inside the instrument budget discussed below.

**Blundell-Bond** (1998) adds the level equations back, instrumented by lagged **differences**. The extra moments matter when the autoregressive parameter approaches unity, where lagged levels become weak instruments for differences:

```@example preg
m_bb = estimate_xtreg(pd_dyn, :growth, [:invest, :output]; model=:bb)
report(m_bb)
```

Retrieve the AR tests programmatically with [`arellano_bond_ar_test`](@ref), which reads them off `m.dynamic_diagnostics`:

```@example preg
arellano_bond_ar_test(m_ab; order=2)
```

The coefficient covariance (`vcov(m)`) is the full Windmeijer (2005) corrected GMM covariance, so joint Wald tests across coefficients use the cross-coefficient (off-diagonal) terms rather than the reported standard errors alone.

!!! warning "Instrument count grows with T squared"
    The block-diagonal instrument matrix contributes one column per available lag at
    every period, so the moment count grows as ``O(T^2)`` per equation. Once it exceeds
    the number of groups the Hansen J stops rejecting anything (`p > 0.999` is the
    tell-tale) and the two-step weighting matrix overfits. `estimate_xtreg` emits a
    warning naming the ratio; pass `collapse=true` or a finite `max_lag_endo` to
    `estimate_xtreg` itself, which accepts both for `:ab` and `:bb`, or shorten the panel.

---

## Panel-Corrected Standard Errors and Prais-Winsten AR(1)

For **time-series-cross-section (TSCS)** panels — a modest number of units observed
over many periods (``T \gtrsim N``) with **contemporaneous cross-section
correlation** — Beck & Katz (1995) recommend **panel-corrected standard errors
(PCSE)**. PCSE forms the ``N \times N`` contemporaneous residual covariance
``\hat{\Sigma}`` and sandwiches it as ``(X'X)^{-1} [\sum_t X_t' \hat{\Sigma} X_t] (X'X)^{-1}``.
Point estimates are unchanged; only the covariance differs:

Unbalanced panels choose how ``\hat{\Sigma}`` is accumulated with `pcse_unbalanced`:
`:casewise` (default — only fully-observed periods enter ``\hat{\Sigma}``) or
`:pairwise` (``\hat{\Sigma}_{ij}`` over the overlapping periods of ``i`` and ``j``).
PWT has 38 countries but only 30 periods in which all of them are observed, so the
casewise ``\hat{\Sigma}`` is rank-deficient and `:pairwise` is the right choice here:

```@example preg
m_pcse = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk];
                        cov_type=:pcse, pcse_unbalanced=:pairwise)
report(m_pcse)
```

The coefficients are identical to the cluster-robust fit — 0.4169 and 0.5161 — because
PCSE changes only the covariance. The standard errors, however, fall by roughly a factor
of three (0.0521 to 0.0163 on capital), which is the usual TSCS trade: PCSE models the
cross-section correlation parametrically through a single ``N \times N`` matrix instead
of letting entity clusters absorb arbitrary within-unit dependence, so it is far more
efficient when the Beck-Katz assumptions hold and far too optimistic when they do not.

When the idiosyncratic error is serially correlated, add a **Prais-Winsten AR(1)**
FGLS transform via `ar1`. The pipeline is **PW transform ``\to`` estimate ``\to``
PCSE**: each unit's series is quasi-differenced (``x_{it} - \hat{\rho}\, x_{i,t-1}``),
with the first observation weighted by ``\sqrt{1-\hat{\rho}^2}`` (dropping that
weight silently reverts to Cochrane-Orcutt). Use `ar1=:common` for one pooled
``\hat{\rho}`` or `ar1=:panel_specific` for per-unit ``\hat{\rho}_i``. Unlike `:pcse`
alone, the AR(1) FGLS **does** change the point estimates:

```@example preg
m_ar1 = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk];
                       cov_type=:pcse, pcse_unbalanced=:pairwise, ar1=:common)
report(m_ar1)
```

The pooled ``\hat{\rho}`` is 0.9472 — log GDP per capita is very nearly a unit root —
and quasi-differencing at that value changes the estimand completely: human capital jumps
to 2.7123 and capital turns negative at ``-0.6903``. This is not a bug but the known
hazard of Prais-Winsten on highly persistent data. With ``\hat{\rho} \to 1`` the
transform approaches first-differencing, so the reported coefficients are short-run
responses of *growth* rather than the level elasticities the untransformed regression
estimates, and the AR(1) parameter is itself badly biased in a fixed-effects panel.
Check `m.ar1_rho` before interpreting: values near unity mean the model should be
respecified in differences (`model=:fd`) rather than quasi-differenced.

**Which covariance for which panel?**

| Situation | Recommended | Why |
|-----------|-------------|-----|
| ``T \gtrsim N``, contemporaneous cross-section correlation | `:pcse` | Beck-Katz (1995) TSCS standard |
| ``T \gtrsim N``, serial **and** contemporaneous correlation | `:pcse` + `ar1=:common` | PW purges AR(1); PCSE handles cross-section |
| Large ``T``, spatial + serial dependence | `:driscoll_kraay` | HAC across time, robust to cross-section |
| Many entities, within-entity serial correlation | `:cluster` | Entity clusters absorb arbitrary within-``i`` dependence |
| Two-dimensional clustering | `:twoway` | Cameron-Gelbach-Miller (2011) |

!!! note "T ≥ N for casewise PCSE"
    The casewise ``\hat{\Sigma}`` is full rank only when the number of
    fully-observed periods is at least ``N``. With ``T < N`` it is singular;
    `estimate_xtreg` warns and you should prefer `pcse_unbalanced=:pairwise` or a
    longer panel (``\hat{\Sigma}`` is never inverted, so no garbage is returned).

---

## `estimate_xtreg` Reference

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | `Symbol` | `:fe` | Estimator: `:fe`, `:re`, `:fd`, `:between`, `:cre`, `:ab`, `:bb` |
| `twoway` | `Bool` | `false` | Entity + time fixed effects (FE only); equivalent to `absorb=[:entity, :time]` |
| `absorb` | `Vector{Symbol}` | `Symbol[]` | High-dimensional FE to absorb by alternating projections (`:fe` only) |
| `hdfe_tol` | `Real` | `1e-8` | Absorption convergence tolerance |
| `hdfe_maxiter` | `Int` | `1000` | Maximum alternating-projection iterations |
| `hdfe_accel` | `Bool` | `true` | Irons-Tuck acceleration of the projection loop |
| `cov_type` | `Symbol` | `:cluster` | Covariance: `:ols`, `:cluster`, `:twoway`, `:driscoll_kraay`, `:pcse` |
| `bandwidth` | `Union{Nothing,Int}` | `nothing` | Driscoll-Kraay bandwidth (`nothing` = Newey-West optimal) |
| `pcse_unbalanced` | `Symbol` | `:casewise` | `:pcse` unbalanced handling: `:casewise` or `:pairwise` |
| `ar1` | `Symbol` | `:none` | Prais-Winsten AR(1): `:none`, `:common`, `:panel_specific` (`:fe`/`:re`/`:cre`) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Vector{T}` | Estimated coefficients |
| `vcov_mat` | `Matrix{T}` | Variance-covariance matrix |
| `r2_within` | `T` | Within R-squared |
| `r2_between` | `T` | Between R-squared |
| `r2_overall` | `T` | Overall R-squared |
| `sigma_u` | `T` | Between-group standard deviation |
| `sigma_e` | `T` | Within-group standard deviation |
| `rho` | `T` | Fraction of variance due to ``\alpha_i`` |
| `theta` | `Union{Nothing,T}` | Quasi-demeaning parameter (`:re`/`:cre`; `nothing` otherwise) |
| `group_effects` | `Union{Nothing,Vector{T}}` | Estimated entity effects (`:fe`; `nothing` otherwise) |
| `f_stat` / `f_pval` | `T` | Model F-statistic and its p-value |
| `loglik` / `aic` / `bic` | `T` | Log-likelihood and information criteria |
| `method` | `Symbol` | Estimation method used |
| `cov_type` | `Symbol` | Covariance estimator actually applied |
| `n_obs` / `n_groups` | `Int` | Effective observations and panel units |
| `n_periods_avg` | `T` | Average periods per unit |
| `dynamic_diagnostics` | `Union{Nothing,NamedTuple}` | AR(1)/AR(2) and Hansen J (`:ab`/`:bb`; `nothing` otherwise) |
| `ar1_rho` | `Union{Nothing,T,Vector{T}}` | Prais-Winsten ``\hat{\rho}`` (scalar for `:common`, per-unit for `:panel_specific`) |
| `hdfe` | `Union{Nothing,NamedTuple}` | Absorption diagnostics (`absorb` only; `nothing` otherwise) |
| `residuals` / `fitted` | `Vector{T}` | Residuals and fitted values |
| `data` | `PanelData{T}` | The panel the model was fitted on |

`dynamic_diagnostics` carries `ar1`, `ar1_p`, `ar2`, `ar2_p`, `hansen`, `hansen_df`,
`hansen_p`, and `n_instruments`.

When `absorb` is used, `hdfe` carries:

| Field | Type | Description |
|-------|------|-------------|
| `absorb` | `Vector{Symbol}` | Absorbed dimension names |
| `n_absorbed` | `Int` | Absorbed FE parameters (rank of the dummy design) |
| `n_levels` | `Vector{Int}` | Levels per dimension |
| `n_components` | `Int` | Mobility groups of the first two dimensions |
| `marginal` | `Vector{Int}` | Per-dimension contribution to `n_absorbed` |
| `n_absorbed_cluster` | `Int` | Non-nested parameters charged to the cluster dof |
| `converged` | `Bool` | Alternating projections reached `hdfe_tol` |
| `iterations` | `Int` | Iterations run |
| `sweeps` | `Int` | Total demeaning sweeps |
| `change` | `T` | Final relative movement |
| `tol` | `T` | Tolerance the fit used |
| `accel` | `Bool` | Whether Irons-Tuck acceleration was on |

---

## Panel Instrumental Variables

The `estimate_xtiv` function handles endogeneity in panel data through four IV strategies. Each transforms the panel to remove ``\alpha_i`` and then runs 2SLS on the transformed data, so the *same* transformation is applied to the outcome, the regressors, and the instruments.

The synthetic panel from Recipe 4 makes the problem concrete. There, `x_endog` and `wage` share the error component `u`, so plain fixed effects are inconsistent:

```@example preg
m_naive = estimate_xtreg(pd_iv, :wage, [:x, :x_endog])
report(m_naive)
```

FE puts the coefficient on the endogenous regressor at 2.4157 against a true value of 2.0 — a 21% upward bias that the tight standard error (0.0256) does nothing to signal. The exogenous coefficient on `x` is fine at 1.5585 because `x` is uncorrelated with `u`. Every IV variant below recovers the truth.

### FE-IV

Within-transform all variables, then apply 2SLS on the demeaned data:

```@example preg
m_feiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z, :z2], model=:fe)
report(m_feiv)
```

FE-IV returns 2.0372 for the endogenous coefficient and 1.5731 for the exogenous one, both within a standard error of the truth, and the bias visible in the naive fit is gone. The price is precision: the standard error on `x_endog` more than doubles (0.0256 to 0.0584), because only the instrumented part of the variation identifies the coefficient.

### RE-IV (EC2SLS)

The Baltagi (1981) EC2SLS estimator quasi-demeans all variables, then uses ``[\tilde{Z}, \bar{Z}_i]`` as instruments:

```@example preg
m_reiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z, :z2], model=:re)
report(m_reiv)
```

EC2SLS lands on essentially the same estimate (2.0293) with a marginally larger standard error. Doubling the instrument set by adding the group means ``\bar{Z}_i`` is what makes EC2SLS more efficient than FE-IV in theory; here the entity effects are independent of the regressors by construction, so there is little for the extra moments to add.

### FD-IV

First-difference all variables, then apply 2SLS:

```@example preg
m_fdiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z, :z2], model=:fd)
report(m_fdiv)
```

FD-IV gives 1.9781, again unbiased, on 950 rather than 1000 observations — differencing costs one period per unit. Its standard error is the largest of the three (0.0825) because differencing white-noise errors doubles the residual variance.

### Weak Instruments and Overidentification

The specification block of every `estimate_xtiv` fit carries the instrument diagnostics, and they should be read before the coefficients. Two of them do the work. The **first-stage F** tests whether the excluded instruments explain the endogenous regressor after partialling out the included exogenous ones; when several regressors are endogenous, `first_stage_f` reports the *minimum* across them. The conventional threshold is 10 (Staiger & Stock 1997), and the example clears it by an order of magnitude.

The **Sargan-Hansen J** tests the overidentifying restrictions: with more instruments than endogenous regressors, the surplus moments are testable, and a rejection says at least one instrument fails the exclusion restriction. The statistic is the classical homoskedastic Sargan ``J = e'P_Z e / \hat{\sigma}^2`` when `cov_type=:ols` and the entity-clustered Hansen J otherwise, so it inherits the robustness of the covariance you chose:

```@example preg
(first_stage_F = round(m_feiv.first_stage_f, digits=2),
 sargan_J      = round(m_feiv.sargan_stat, digits=4),
 sargan_p      = round(m_feiv.sargan_pval, digits=4))
```

The J-statistic of 0.2189 against a ``\chi^2(1)`` gives ``p = 0.6399``, so the data cannot reject the exclusion restriction — as they should not, since `z` and `z2` were drawn independently of the error. Both fields are `nothing` when the model is exactly identified (one instrument per endogenous regressor), because there is then no overidentifying restriction to test; the Hausman-Taylor example below is exactly that case.

!!! warning "Non-rejection is not validation"
    A large Sargan p-value means the overidentifying moments are mutually consistent, not
    that they are individually valid. If *all* instruments share the same violation the
    statistic has no power against it, and adding weak instruments inflates the degrees of
    freedom faster than the statistic, pushing p toward 1. Report the instrument count and
    first-stage F alongside every J-test.

### Hausman-Taylor

The **Hausman-Taylor** (1981) estimator handles endogenous time-invariant regressors by using within-deviations of time-varying exogenous variables as instruments. Time-invariant regressors are wiped out by the within transformation, so FE cannot estimate them at all and RE only can if they are exogenous; HT occupies the middle ground:

```@example preg
Random.seed!(2)
# Time-invariant variables (e.g., education level, geographic endowment)
N_ht, T_ht = 50, 20
n_ht = N_ht * T_ht
df_ht = DataFrame(id=repeat(1:N_ht, inner=T_ht), t=repeat(1:T_ht, N_ht),
                  experience=randn(n_ht))
df_ht.region = repeat(randn(N_ht), inner=T_ht)        # time-invariant exogenous
df_ht.education = repeat(randn(N_ht), inner=T_ht)     # time-invariant endogenous
alpha_ht = repeat(randn(N_ht) .+ 0.5 .* randn(N_ht), inner=T_ht)
df_ht.earnings = alpha_ht .+ 1.5 .* df_ht.experience .+ 0.3 .* df_ht.region .+ 0.8 .* df_ht.education .+ randn(n_ht)
pd_ht = xtset(df_ht, :id, :t)
m_ht = estimate_xtiv(pd_ht, :earnings, [:experience], Symbol[];
                      model=:hausman_taylor,
                      time_invariant_exog=[:region],
                      time_invariant_endog=[:education])
report(m_ht)
```

Experience, the time-varying regressor, is recovered precisely at 1.5459 against a true 1.5. The time-invariant coefficients are the ones HT exists to deliver, and they are far noisier: `region` (exogenous, true 0.3) comes in at 0.4287 with a standard error of 0.2116, and `education` (endogenous, true 0.8) at 0.5266 with a standard error of 0.9637, indistinguishable from zero. That is the honest cost of identifying a time-invariant effect from 50 entities using within-deviations as instruments — the first-stage F of 20.47 clears the weak-instrument threshold but leaves little precision.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `instruments` | `Vector{Symbol}` | `Symbol[]` | Excluded instruments |
| `model` | `Symbol` | `:fe` | Method: `:fe`, `:re`, `:fd`, `:hausman_taylor` |
| `cov_type` | `Symbol` | `:cluster` | Covariance: `:ols`, `:cluster`, `:twoway`, `:driscoll_kraay` |
| `time_invariant_exog` | `Vector{Symbol}` | `Symbol[]` | Time-invariant exogenous (HT only) |
| `time_invariant_endog` | `Vector{Symbol}` | `Symbol[]` | Time-invariant endogenous (HT only) |

`estimate_xtiv` does not accept `:pcse`; panel-corrected standard errors are available for
`estimate_xtreg` only.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Vector{T}` | Estimated coefficients |
| `vcov_mat` | `Matrix{T}` | Variance-covariance matrix |
| `first_stage_f` | `T` | Minimum first-stage F-statistic |
| `cragg_donald_f` | `Union{Nothing,T}` | Cragg-Donald weak-instrument F on the transformed design |
| `kleibergen_paap_f` | `Union{Nothing,T}` | Kleibergen-Paap rk Wald F, heteroskedasticity-robust |
| `stock_yogo_10pct` | `Union{Nothing,T}` | Stock-Yogo 10% maximal-size critical value |
| `sargan_stat` | `Union{Nothing,T}` | Sargan-Hansen J statistic (`nothing` if exactly identified) |
| `sargan_pval` | `Union{Nothing,T}` | J-test p-value (`nothing` if exactly identified) |
| `r2_within` / `r2_between` / `r2_overall` | `T` | R-squared variants |
| `sigma_u` / `sigma_e` / `rho` | `T` | Variance components |
| `endog_names` | `Vector{String}` | Endogenous regressor names |
| `instrument_names` | `Vector{String}` | Excluded instrument names |
| `method` | `Symbol` | IV method used |
| `cov_type` | `Symbol` | Covariance estimator applied |
| `n_obs` / `n_groups` | `Int` | Effective observations and panel units |
| `Z` | `Matrix{T}` | Instrument matrix actually used |

The three weak-instrument fields are computed on the *transformed* design — demeaned for FE,
differenced for FD, quasi-demeaned for RE and Hausman-Taylor — so they measure the strength of
the moments that actually identify ``\beta``. The FE-IV fit above reports a Cragg-Donald F of
103.02 and a Kleibergen-Paap F of 94.06 against a Stock-Yogo 10% critical value of 19.93, all
consistent with its first-stage F of 108.46. The Cragg-Donald denominator degrees of freedom
subtract the fixed effects the transformation absorbed, matching `xtivreg2, fe`. They are
`nothing` when the model is underidentified or the statistic fails to evaluate.

!!! warning "Kleibergen-Paap is robust to heteroskedasticity, not to clustering"
    The reported rk Wald F uses a heteroskedasticity-robust meat in every branch; a
    cluster-robust rk statistic is not implemented. Under the default `cov_type=:cluster`
    it therefore ignores within-entity dependence and overstates instrument strength
    whenever the first stage is serially correlated within units.

---

## Panel Discrete Choice

For cross-sectional (non-panel) discrete choice models, see [Binary Choice Models](@ref binary_choice_page).

The examples use the DDCG panel of Acemoglu, Naidu, Restrepo & Robinson (2019): 175 countries observed annually, a binary `dem` indicator, and log GDP per capita. Regressing democracy on income is the classic **Lipset (1959) modernization hypothesis** — richer countries are more likely to be democracies.

### Panel Logit

The `estimate_xtlogit` function estimates panel logistic regression with four methods:

- **Pooled**: Standard logit ignoring panel structure (with optional cluster-robust SEs)
- **FE (conditional)**: Chamberlain (1980) conditional likelihood — eliminates entity effects by conditioning on the within-unit count of successes, a sufficient statistic for ``\alpha_i``. Only within-entity variation identifies coefficients.
- **RE**: Adaptive Gauss-Hermite quadrature integration over the random effect distribution
- **CRE**: Mundlak-style augmentation of the RE model with group means

```@example preg
m_pooled = estimate_xtlogit(pd_ddcg, :dem, [:lngdppc])
report(m_pooled)
```

Pooled logit puts the log-odds slope at 0.6879 with a cluster-robust standard error of 0.0858, so a one-log-point rise in GDP per capita (roughly a 170% income gain) raises the log odds of democracy by 0.69 — an odds ratio near 2. The pseudo R-squared of 0.1581 is modest, as expected when a single covariate is asked to explain regime type. Pooling ignores the panel structure entirely, so this coefficient mixes the cross-country association (rich countries *are* democracies) with the within-country one (countries democratize *as* they get richer), and those are the two things the estimators below separate.

```@example preg
# FE conditional logit — within-country variation only
m_fe_logit = estimate_xtlogit(pd_ddcg, :dem, [:lngdppc]; model=:fe)
report(m_fe_logit)
```

Conditioning on each country's number of democratic years drops the 87 countries that never switch regime, leaving 88 countries and 3589 observations — the sample where within-country identification is even possible. The slope rises to 1.8515, well above the pooled 0.6879, so within the switching countries income and democratization move together more strongly than the pooled fit suggests. The conditional log-likelihood is globally concave and the Newton step uses its exact information matrix, computed by central-differencing the dynamic-programming score, so the fit converges in five iterations from a zero start.

```@example preg
# RE logit — integrates over country-level heterogeneity
m_re_logit = estimate_xtlogit(pd_ddcg, :dem, [:lngdppc]; model=:re, tol=1e-12)
report(m_re_logit)
```

Random effects keeps all 175 countries and integrates over a normal country effect with estimated ``\hat{\sigma}_u = 4.8225``, implying ``\rho = 0.8761``: 88% of the latent-variable residual variance is permanent country heterogeneity. Absorbing that heterogeneity raises the pseudo R-squared from 0.1581 to 0.5233 and the income slope to 1.8483 — essentially the conditional-FE 1.8515, which is exactly what a correctly specified RE model should deliver: both estimators isolate the within-country association, they just weight the switching countries differently. The entity-clustered standard error of 0.5217 puts ``z = 3.54``, in line with the clustered conditional-FE ``z`` of 3.44 below. All three estimates share the sign that supports the Lipset hypothesis; they differ in which variation identifies it.

!!! note "Two defects used to corrupt this fit (#542)"
    Earlier releases reported `Converged: No` with standard errors near ``10^{-10}``
    on this example. Two distinct defects, both fixed in
    [#542](https://github.com/FriedmanJP/MacroEconometricModels.jl/issues/542): the
    inner posterior-mode search of the adaptive quadrature ran a fixed number of Newton
    steps, leaving the likelihood *value* jagged — optimizers stopped on noise-dependent
    pseudo-optima 20+ log-likelihood units above the true optimum (which is why older
    documentation quoted a slope of 1.10) — and the covariance differentiated through
    that truncated search, inflating the observed information by orders of magnitude.
    The mode search now converges to machine precision, and the covariance uses the
    Louis (1982) score/information identities computed by posterior quadrature. The fit
    converges in 14 iterations, the optimum is unique across starting values, and the
    reported standard errors respect the complete-data information bound.

!!! note "The fit itself no longer aborts"
    Before
    [#600](https://github.com/FriedmanJP/MacroEconometricModels.jl/issues/600) this call
    threw an `AssertionError` from inside the line search: the adaptive Gauss-Hermite
    likelihood formed `exp(-eta)` and `exp(2 log sigma_u)` directly, both of which overflow
    on this panel and put a `NaN` in the ForwardDiff gradient, and `HagerZhang` asserts that
    the trial value and its directional derivative are finite. The likelihood is now written
    so that neither the value nor its gradient can go non-finite anywhere the optimizer can
    reach.

!!! note "Which standard errors you get"
    `cov_type` applies to every estimator. The FE fit inverts the conditional information
    for `:ols` and sandwiches it with the per-group conditional scores for `:cluster`, the
    default — 0.5379 against 0.1490 on `lngdppc` here, a ``z`` of 3.44 rather than 12.42,
    since the clustered version prices in the within-country dependence the conditional
    likelihood does not model. The RE and CRE fits build the bread from the Louis
    observed information of the marginal likelihood and, under `:cluster`, sandwich it
    with the per-group marginal scores using the same ``G/(G-1)\cdot(n-1)/(n-k)``
    finite-sample correction as the pooled path.

### Panel Probit

The `estimate_xtprobit` function supports pooled, RE, and CRE models. **No FE probit** is available because there is no conditioning trick analogous to the logit case — the incidental parameters problem biases FE probit coefficients (Wooldridge 2010, §15.8).

```@example preg
m_probit = estimate_xtprobit(pd_ddcg, :dem, [:lngdppc])
report(m_probit)
```

Probit coefficients are on the standard-normal scale, so they are not directly comparable to logit ones; the usual rule of thumb ``\beta_{\text{logit}} \approx 1.6 \, \beta_{\text{probit}}`` holds here (0.6879 against 0.4185). Marginal effects, computed next, are the scale-free way to compare the two link functions. `model=:re` and `model=:cre` integrate over a normal country effect, but with *fixed* (non-adaptive) Gauss-Hermite nodes and a likelihood-change stopping rule — on panels with large ``\sigma_u`` like this one the probit quadrature under-resolves where the adaptive logit version does not, so read its RE output with more caution than the logit's.

### Panel Marginal Effects

`marginal_effects` computes average marginal effects with delta-method standard errors for panel logit and probit models. For RE and CRE models the marginal effects integrate over the random effect distribution using Gauss-Hermite quadrature.

```@example preg
me = marginal_effects(m_pooled)
report(me)
```

The AME of 0.1368 is the interpretable quantity: averaged over the observed distribution of income, one additional log point of GDP per capita raises the probability of being a democracy by about 13.7 percentage points. Unlike the log-odds coefficient this is directly comparable across logit and probit and across specifications.

### Keywords (Logit/Probit)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | `Symbol` | `:pooled` | Method: `:pooled`, `:fe`, `:re`, `:cre` (logit); `:pooled`, `:re`, `:cre` (probit) |
| `cov_type` | `Symbol` | `:cluster` | Covariance estimator: `:ols`, `:cluster` (pooled and FE) |
| `maxiter` | `Int` | `2000` (logit), `200` (probit) | Maximum iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance; score tolerance on `:fe`, relative function tolerance on `:re`/`:cre` |
| `n_quadrature` | `Int` | `12` | Gauss-Hermite quadrature points (RE/CRE) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Vector{T}` | Estimated coefficients (intercept first, except `:fe`) |
| `vcov_mat` | `Matrix{T}` | Variance-covariance matrix |
| `sigma_u` / `rho` | `Union{Nothing,T}` | Random-effect standard deviation and intra-class correlation (RE/CRE; `nothing` otherwise) |
| `loglik` / `loglik_null` | `T` | Fitted and null log-likelihood |
| `pseudo_r2` | `T` | McFadden pseudo R-squared |
| `aic` / `bic` | `T` | Information criteria |
| `converged` | `Bool` | Whether the optimizer met its tolerance |
| `iterations` | `Int` | Iterations run |
| `method` | `Symbol` | `:pooled`, `:fe`, `:re`, or `:cre` |
| `n_obs` / `n_groups` | `Int` | Observations and groups actually used (`:fe` drops non-varying groups) |

---

## Specification Tests

Six specification tests help choose between estimators and diagnose violations of model assumptions. Each returns a [`PanelTestResult`](@ref) carrying the test statistic, p-value, and degrees of freedom.

### Hausman Test (FE vs RE)

Tests whether the RE assumption ``E[\alpha_i \mid x_{it}] = 0`` holds (Hausman 1978). Rejection favors FE.

```@example preg
m_fe2 = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk])
m_re2 = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:re)
ht = hausman_test(m_fe2, m_re2)
report(ht)
```

The statistic uses the Moore-Penrose generalized inverse of ``V_{FE} - V_{RE}`` with
degrees of freedom equal to its numerical rank. In finite samples this difference is often
not positive semidefinite; when that happens the reported statistic can be **negative**
(the test emits a warning and cannot reject ``H_0``) and the degrees of freedom may fall
below the number of coefficients — matching Stata's `hausman`. A negative statistic
indicates the asymptotic Hausman assumption is violated in the sample, not evidence for RE.

### Breusch-Pagan LM Test

Tests for the presence of random effects: ``H_0: \sigma_u^2 = 0`` (Breusch & Pagan 1980). Rejection suggests pooled OLS is inefficient and RE or FE is preferred.

```@example preg
bp = breusch_pagan_test(m_re2)
report(bp)
```

The LM statistic of 39791.76 against a ``\chi^2(1)`` rejects overwhelmingly, so ``\sigma_u^2 > 0`` and pooled OLS — which assumes a single common intercept — is both inefficient and has understated standard errors. This is the expected verdict on any country panel with persistent level differences; it says use RE or FE, but not which one.

### F-Test for Fixed Effects

Tests joint significance of all entity fixed effects: ``H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_N``.

```@example preg
ft = f_test_fe(m_fe2)
report(ft)
```

``F(37, 2335) = 622.56`` rejects the null that all 38 country intercepts are equal. Together with the Breusch-Pagan result this settles that entity heterogeneity is real; the Hausman test above then decides whether it can be treated as random.

### Pesaran CD Test

Tests for cross-sectional dependence in panel residuals (Pesaran 2004). Under ``H_0``, residuals are uncorrelated across entities.

```@example preg
cd = pesaran_cd_test(m_fe2)
report(cd)
```

The CD statistic of 19.87 is standard normal under the null and rejects decisively: residuals are correlated *across* countries, which is exactly what global business cycles and common technology shocks produce. Entity-clustered standard errors assume independence across entities and are therefore too small here — this is the diagnostic that argues for `cov_type=:driscoll_kraay` or the two-way fixed effects fitted earlier.

### Wooldridge AR Test

Tests for first-order serial correlation in first-differenced residuals (Wooldridge 2010). Under ``H_0``, no serial correlation.

```@example preg
ar = wooldridge_ar_test(m_fe2)
report(ar)
```

``F(1, 37) = 721.76`` rejects the null of no first-order serial correlation in the first-differenced residuals. Log GDP per capita is highly persistent, so this is unsurprising; it means the classical `:ols` covariance is invalid and either cluster-robust standard errors (which absorb arbitrary within-country dependence) or an explicit AR(1) correction is required.

### Modified Wald Test

Tests for groupwise heteroskedasticity: ``H_0: \sigma_i^2 = \sigma^2`` for all ``i`` (Greene 2012). Rejection suggests entity-specific error variances.

```@example preg
mw = modified_wald_test(m_fe2)
report(mw)
```

The modified Wald statistic of 3331.65 on 38 degrees of freedom rejects a common error variance across countries. All four diagnostics point the same way for this panel: keep the entity fixed effects, and use a covariance estimator that tolerates heteroskedasticity, serial correlation, and cross-sectional dependence at once — `:driscoll_kraay` here.

### Recommended Workflow

| Question | Test | If rejected |
|----------|------|-------------|
| FE or RE? | `hausman_test` | Use FE |
| Random effects present? | `breusch_pagan_test` | Use RE or FE, not pooled OLS |
| Entity effects significant? | `f_test_fe` | Entity heterogeneity matters |
| Cross-sectional dependence? | `pesaran_cd_test` | Use Driscoll-Kraay SEs |
| Serial correlation? | `wooldridge_ar_test` | Use cluster-robust SEs or FD |
| Groupwise heteroskedasticity? | `modified_wald_test` | Use cluster-robust SEs |

---

## Covariance Estimators

`estimate_xtreg` supports five covariance estimators via the `cov_type` keyword. Point estimates never depend on the choice (with the sole exception of `ar1`, which is an FGLS transform rather than a covariance); only the standard errors do:

| `cov_type` | Formula | When to use |
|------------|---------|-------------|
| `:ols` | ``\hat{\sigma}^2 (X'X)^{-1}`` | Homoskedastic, no correlation |
| `:cluster` | Entity-cluster robust (Arellano 1987) | Default, heteroskedasticity + within-entity correlation |
| `:twoway` | Two-way cluster (Cameron et al. 2011) | Cross-sectional + serial dependence |
| `:driscoll_kraay` | HAC across both dimensions (Driscoll & Kraay 1998) | Large ``T``, spatial dependence |
| `:pcse` | Beck-Katz (1995) panel-corrected | ``T \gtrsim N`` TSCS panels |

`estimate_xtiv` accepts the first four; `:pcse` is `estimate_xtreg`-only.

```@example preg
# Driscoll-Kraay standard errors for PWT growth regression
m_dk = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; cov_type=:driscoll_kraay)
report(m_dk)
```

The coefficients are the cluster-robust ones to four decimals — 0.4169 and 0.5161 — while the standard error on capital falls from 0.0521 to 0.0198. Driscoll-Kraay averages the moment contributions *across* entities within each period before applying a Newey-West kernel over time, so it buys robustness to the cross-sectional dependence the Pesaran CD test detected at the cost of relying on a long time dimension. With 70 years and 38 countries PWT is in the regime where that trade is favourable.

Visualize the coefficients and the residual diagnostics with `plot_result`, which accepts `view=:coef` (default) for a coefficient plot with 95% intervals and `view=:diagnostics` for the four-panel residual display:

```julia
plot_result(m_fe)
plot_result(m_fe; view=:diagnostics)
```

```@raw html
<iframe src="../assets/plots/panel_reg_coef.html" width="100%" height="380" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The coefficient view omits the intercept — a within estimator's constant is an artefact of the demeaning, not a quantity to interpret — and draws each slope against a zero reference line. Both PWT slopes sit far from it: human capital at 0.417 on ``[0.261, 0.573]`` and capital per worker at 0.516 on ``[0.414, 0.618]``. The whiskers are ``\hat\beta \pm 1.96\,\mathrm{SE}`` taken from `vcov_mat`, so they inherit whichever covariance estimator the fit used — cluster-robust here, since that is the `estimate_xtreg` default. A figure drawn from the `cov_type=:driscoll_kraay` refit above puts the same two points on a visibly shorter whisker for capital per worker, whose standard error falls from 0.0521 to 0.0198.

---

## Complete Example

A full panel analysis workflow using the Penn World Table: estimate, test, and then choose a specification the diagnostics can support.

```@example preg
# Growth regression: log GDP per capita on human capital and capital deepening
m_fe_full = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk])
m_re_full = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:re)

# Hausman test: FE vs RE
ht = hausman_test(m_fe_full, m_re_full)
report(ht)
```

```@example preg
# Breusch-Pagan: test for random effects
bp = breusch_pagan_test(m_re_full)
report(bp)
```

```@example preg
# Diagnostics on FE model
cd = pesaran_cd_test(m_fe_full)
report(cd)
```

```@example preg
ar = wooldridge_ar_test(m_fe_full)
report(ar)
```

```@example preg
# CRE as robustness check
m_cre_full = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:cre)
report(m_cre_full)
```

```@example preg
# Final specification: entity + time effects, cross-section-robust SEs
m_final = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk];
                         absorb=[:entity, :time], cov_type=:driscoll_kraay)
report(m_final)
```

The chain of tests narrows the specification. Breusch-Pagan rejects a common intercept, so pooled OLS is out. The Hausman statistic is negative here — the finite-sample ``V_{FE} - V_{RE}`` is not positive semidefinite — which is uninformative rather than evidence for RE, so the CRE fit is the usable version of the same test: its group-mean coefficients (`hc_mean` at 1.2984, `lnk_mean` at ``-0.3442``, both significant at 1%) reject ``E[\alpha_i \mid x_{it}] = 0`` and settle the question in favour of fixed effects. Pesaran CD then rejects cross-sectional independence and Wooldridge AR rejects serial independence, so entity-clustered standard errors are not enough. The final specification absorbs entity and time effects and reports Driscoll-Kraay standard errors, leaving the capital elasticity at 0.5137 — essentially unchanged across every consistent specification on this page, which is the strongest evidence that it is a real feature of the data rather than an artefact of one estimator.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example preg
path = joinpath(mktempdir(), "xtreg.jld2")
save_model(m_fe, path)
m_fe2 = load_model(path)
typeof(m_fe2)
```

---

## Common Pitfalls

1. **Forgetting `xtset`.** All panel estimators require `PanelData` created via `xtset(df, :group, :time)`. Passing a raw DataFrame throws an error.

2. **Including time-invariant regressors in FE.** The within transformation eliminates all time-invariant variables. Use RE, CRE, or Hausman-Taylor to estimate their effects.

3. **Using FE probit.** There is no conditioning trick for probit (unlike logit). The package correctly excludes `:fe` from `estimate_xtprobit` — use `:re` or `:cre` instead.

4. **Weak instruments in panel IV.** Check `first_stage_f` in the `PanelIVModel` output. Values below 10 indicate weak instruments (Staiger & Stock 1997), and 2SLS is then more biased than the OLS it was meant to repair.

5. **Reading a Sargan p-value as instrument validation.** A non-rejection means the overidentifying moments agree with each other, not that they are valid. `sargan_stat` is `nothing` when the model is exactly identified, because there is nothing to test.

6. **Ignoring cross-sectional dependence.** Standard cluster-robust SEs assume independence across entities. Run `pesaran_cd_test` and switch to `cov_type=:driscoll_kraay` if rejected.

7. **Letting the AB/BB instrument count exceed ``N``.** The block-diagonal instrument matrix grows as ``O(T^2)``, and once the moment count passes the number of groups the Hansen J stops rejecting (`p > 0.999`) and standard errors turn unreliable. Rein it in where the model is fitted: `estimate_xtreg` takes `collapse`, `min_lag_endo`, `max_lag_endo`, `pca_instruments`, and `pca_max_components` for `:ab` and `:bb`.

8. **Trusting a dynamic-panel fit without the AR(2) test.** A correctly specified Arellano-Bond model *rejects* AR(1) — the differenced error is MA(1) by construction — but must *not* reject AR(2). Rejection at order 2 means the level errors are serially correlated and the lagged-level instruments are invalid.

9. **Assuming additive demeaning is the two-way within transformation.** ``y_{it} - \bar{y}_i - \bar{y}_t + \bar{y}`` equals it only on a balanced panel — on the unbalanced PWT panel that identity puts the capital-deepening coefficient at 0.297 against a truth of 0.514. `twoway=true` and `absorb=[:entity, :time]` both use alternating projections and are exact either way, so this is a trap only if you hand-demean your own data.

10. **Ignoring the `hdfe.converged` flag.** Alternating projections converge slowly when fixed-effect dimensions are weakly connected. A `MAP converged: NO` line means the coefficients have not settled — raise `hdfe_maxiter` rather than reporting them.

11. **Reporting a non-converged discrete-choice fit.** The quadrature-based RE and CRE panel logit and probit likelihoods routinely stop short of a stationary point at the default `tol=1e-8`. Read the `Converged` line and tighten `tol` before interpreting the coefficients. The conditional-FE logit is the exception: it is globally concave and settles in a handful of Newton steps.

12. **Prais-Winsten on near-unit-root data.** With ``\hat{\rho}`` close to 1 the quasi-difference approaches a first difference and the coefficients become short-run growth responses, not level elasticities. Inspect `m.ar1_rho` and prefer `model=:fd` when it is near unity.

---

## References

- Abowd, J. M., Creecy, R. H. & Kramarz, F. (2002). Computing Person and Firm Effects Using Linked Longitudinal Employer-Employee Data. *US Census Bureau Technical Paper* TP-2002-06.
- Acemoglu, D., Naidu, S., Restrepo, P. & Robinson, J. A. (2019). Democracy Does Cause Growth. *Journal of Political Economy* 127(1), 47-100. [DOI](https://doi.org/10.1086/700936)
- Arellano, M. (1987). Computing Robust Standard Errors for Within-Groups Estimators. *Oxford Bulletin of Economics and Statistics* 49(4), 431-434. [DOI](https://doi.org/10.1111/j.1468-0084.1987.mp49004006.x)
- Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations. *Review of Economic Studies* 58(2), 277-297. [DOI](https://doi.org/10.2307/2297968)
- Baltagi, B. H. (1981). Simultaneous Equations with Error Components. *Journal of Econometrics* 17(2), 189-200. [DOI](https://doi.org/10.1016/0304-4076(81)90026-9)
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer. ISBN 978-3-030-53952-8.
- Beck, N. & Katz, J. N. (1995). What To Do (and Not to Do) with Time-Series Cross-Section Data. *American Political Science Review* 89(3), 634-647. [DOI](https://doi.org/10.2307/2082979)
- Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions in Dynamic Panel Data Models. *Journal of Econometrics* 87(1), 115-143. [DOI](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Breusch, T. S. & Pagan, A. R. (1980). The Lagrange Multiplier Test and Its Applications to Model Specification in Econometrics. *Review of Economic Studies* 47(1), 239-253. [DOI](https://doi.org/10.2307/2297111)
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). Robust Inference with Multiway Clustering. *Journal of Business & Economic Statistics* 29(2), 238-249. [DOI](https://doi.org/10.1198/jbes.2010.07136)
- Cameron, A. C. & Miller, D. L. (2015). A Practitioner's Guide to Cluster-Robust Inference. *Journal of Human Resources* 50(2), 317-372. [DOI](https://doi.org/10.3368/jhr.50.2.317)
- Correia, S. (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper (`reghdfe`). [PDF](http://scorreia.com/research/hdfe.pdf)
- Chamberlain, G. (1980). Analysis of Covariance with Qualitative Data. *Review of Economic Studies* 47(1), 225-238. [DOI](https://doi.org/10.2307/2297110)
- Driscoll, J. C. & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data. *Review of Economics and Statistics* 80(4), 549-560. [DOI](https://doi.org/10.1162/003465398557825)
- Feenstra, R. C., Inklaar, R. & Timmer, M. P. (2015). The Next Generation of the Penn World Table. *American Economic Review* 105(10), 3150-3182. [DOI](https://doi.org/10.1257/aer.20130954)
- Gaure, S. (2013). OLS with Multiple High Dimensional Category Variables. *Computational Statistics & Data Analysis* 66, 8-18. [DOI](https://doi.org/10.1016/j.csda.2013.03.024)
- Greene, W. H. (2012). *Econometric Analysis*. 7th ed. Prentice Hall. ISBN 978-0-131-39538-1.
- Guimaraes, P. & Portugal, P. (2010). A Simple Feasible Procedure to Fit Models with High-Dimensional Fixed Effects. *The Stata Journal* 10(4), 628-649. [DOI](https://doi.org/10.1177/1536867X1101000406)
- Hausman, J. A. (1978). Specification Tests in Econometrics. *Econometrica* 46(6), 1251-1271. [DOI](https://doi.org/10.2307/1913827)
- Hausman, J. A. & Taylor, W. E. (1981). Panel Data and Unobservable Individual Effects. *Econometrica* 49(6), 1377-1398. [DOI](https://doi.org/10.2307/1911406)
- Lipset, S. M. (1959). Some Social Requisites of Democracy: Economic Development and Political Legitimacy. *American Political Science Review* 53(1), 69-105. [DOI](https://doi.org/10.2307/1951731)
- Mundlak, Y. (1978). On the Pooling of Time Series and Cross Section Data. *Econometrica* 46(1), 69-85. [DOI](https://doi.org/10.2307/1913646)
- Pesaran, M. H. (2004). General Diagnostic Tests for Cross Section Dependence in Panels. CESifo Working Paper No. 1229. [DOI](https://doi.org/10.2139/ssrn.572504)
- Staiger, D. & Stock, J. H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica* 65(3), 557-586. [DOI](https://doi.org/10.2307/2171753)
- Swamy, P. A. V. B. & Arora, S. S. (1972). The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models. *Econometrica* 40(2), 261-275. [DOI](https://doi.org/10.2307/1909405)
- Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators. *Journal of Econometrics* 126(1), 25-51. [DOI](https://doi.org/10.1016/j.jeconom.2004.02.005)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. ISBN 978-0-262-23258-6.
