# [Panel VAR](@id pvar_page)

**MacroEconometricModels.jl** provides a complete Panel VAR (PVAR) implementation for dynamic panel data analysis. The package supports GMM estimation via Arellano-Bond first-difference and Blundell-Bond system instruments, fixed-effects OLS, and a full suite of structural analysis and specification tests.

- **GMM estimation**: First-difference (Arellano & Bond 1991) and system (Blundell & Bond 1998) with Windmeijer (2005) corrected standard errors
- **FE-OLS**: Within estimator with cluster-robust standard errors for large-``T`` panels
- **Transformations**: First-differencing and forward orthogonal deviations (Arellano & Bover 1995)
- **Structural analysis**: Orthogonalized IRF (Cholesky), generalized IRF (Pesaran & Shin 1998), FEVD, and stability diagnostics
- **Bootstrap inference**: Group-level block bootstrap for IRF confidence intervals
- **Specification tests**: Hansen (1982) J-test, Andrews-Lu (2001) MMSC, and lag selection

Panel VAR treats several panel series as jointly endogenous. For single-equation panel models — fixed effects, random effects, panel IV, and the Arellano-Bond estimator with one dependent variable — see [Panel Regression](@ref panel_reg_page). For causal designs built on treatment timing rather than lag structure, see [Difference-in-Differences](@ref did_page) and [Event Study LP](@ref event_study_page). Panel unit-root pretests, which decide whether the series need differencing before any of this, live on [Panel Tests](@ref tests_panel_page).

```@setup pvar
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)
pwt = load_example(:pwt)
_pd_raw = apply_tcode(pwt, 5)
_dep_vars = ["rgdpna", "emp", "hc"]
_dep_idx = [findfirst(==(v), _pd_raw.varnames) for v in _dep_vars]
_df = DataFrame(_pd_raw.data[:, _dep_idx], _dep_vars)
_df.id = _pd_raw.group_id
_df.time = _pd_raw.time_id
_mask = [all(isfinite, row) for row in eachrow(Matrix(_df[:, _dep_vars]))]
_df = _df[_mask, :]
_df = _df[_df.time .> maximum(_df.time) - 30, :]
pd = xtset(_df, :id, :time)
```

## Quick Start

**Recipe 1: FD-GMM with two-step estimation**

```@example pvar
# PWT growth panel (pre-loaded in setup: log first-differenced, NaN-filtered)
dep_vars = ["rgdpna", "emp", "hc"]

# Arellano-Bond two-step GMM; collapse the instruments because N is only 38
model = estimate_pvar(pd, 1; dependent_vars=dep_vars, steps=:twostep,
                      collapse=true, max_lag_endo=6)
report(model)
```

**Recipe 2: System GMM (Blundell-Bond)**

```@example pvar
# System GMM adds level equations instrumented by lagged differences
model_sys = estimate_pvar(pd, 1; dependent_vars=dep_vars, system_instruments=true,
                          steps=:twostep, collapse=true, max_lag_endo=6)
report(model_sys)
```

**Recipe 3: Fixed-effects OLS**

```@example pvar
# Within estimator with cluster-robust SEs
model_fe = estimate_pvar_feols(pd, 1; dependent_vars=dep_vars)
report(model_fe)
```

**Recipe 4: Specification tests and lag selection**

```@example pvar
# Hansen J-test for overidentifying restrictions
j = pvar_hansen_j(model)
report(j)
```

```@example pvar
# Andrews-Lu MMSC across candidate lag orders
sel = pvar_lag_selection(pd, 4; dependent_vars=dep_vars,
                         collapse=true, max_lag_endo=6)
(best_bic = sel.best_bic, best_aic = sel.best_aic, best_hqic = sel.best_hqic)
```

**Recipe 5: Structural analysis with bootstrap CIs**

```@example pvar
# Orthogonalized IRFs and FEVD
irfs = pvar_oirf(model, 10)
decomp = pvar_fevd(model, 10)

# Impact matrix: rows are responses, columns are shocks
round.(irfs[1, :, :], digits=4)
```

```julia
plot_result(model; view=:oirf, H=10)
```

```@raw html
<iframe src="../assets/plots/pvar_irf.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

---

## Model Specification

The Panel VAR(p) model for entity ``i`` at time ``t`` is:

```math
\mathbf{y}_{i,t} = \boldsymbol{\mu}_i + \sum_{l=1}^{p} \mathbf{A}_l \, \mathbf{y}_{i,t-l} + \boldsymbol{\varepsilon}_{i,t}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T_i
```

where:
- ``\mathbf{y}_{i,t} \in \mathbb{R}^m`` is the ``m \times 1`` vector of endogenous variables for entity ``i``
- ``\boldsymbol{\mu}_i \in \mathbb{R}^m`` is an entity-specific **fixed effect**
- ``\mathbf{A}_l`` is the ``m \times m`` coefficient matrix for lag ``l``
- ``\boldsymbol{\varepsilon}_{i,t} \sim (0, \Sigma)`` are i.i.d. innovations
- ``N`` is the number of panel units and ``T_i`` is the time dimension for unit ``i``

The fixed effect ``\boldsymbol{\mu}_i`` is correlated with ``\mathbf{y}_{i,t-l}`` by construction, making OLS on the level equation inconsistent. Two strategies address this:

1. **Transform away the fixed effect** (first-differencing or forward orthogonal deviations) and estimate by GMM using lagged levels as instruments
2. **Demean within groups** (within estimator) and estimate by OLS --- consistent for large ``T`` but subject to Nickell (1981) bias when ``T`` is small relative to ``N``

!!! note "Fixed Effects and Nickell Bias"
    The within estimator (FE-OLS) is biased of order ``O(1/T)`` in dynamic panels (Nickell 1981). For panels with small ``T`` (e.g., ``T < 20``), GMM estimation is strongly preferred. For larger ``T``, FE-OLS and GMM converge to the same estimates.

---

## Panel Data Preparation

Panel VAR estimation requires a `PanelData` object. The built-in Penn World Table provides a balanced panel of 38 OECD countries with annual macroeconomic indicators:

```@example pvar
# Load PWT --- already a PanelData object
pwt_demo = load_example(:pwt)

# Convert to growth rates for stationarity
pd_demo = apply_tcode(pwt_demo, 5)  # tcode 5 = log first difference

(n_groups = pd_demo.n_groups, n_obs = pd_demo.T_obs, variables = length(pd_demo.varnames))
```

The examples below work with the last 30 periods of the three-variable subpanel, which after dropping non-finite rows leaves a balanced 38-country panel — small enough to build quickly and, more importantly, small enough that instrument proliferation is a live concern rather than a footnote.

All numeric columns are treated as potential endogenous variables. Use the `dependent_vars` keyword to select a subset — the examples on this page use real GDP growth, employment growth, and human-capital growth.

For custom panel data, construct a `PanelData` object via `xtset`. Never build a `PanelData` from a raw matrix; `xtset` is the supported constructor and it derives the group and time indices the estimators rely on:

```@example pvar
Random.seed!(11)
df = DataFrame(country=repeat(1:20, inner=30), year=repeat(1:30, outer=20),
               gdp=randn(600), inv=randn(600), cons=randn(600))
pd_custom = xtset(df, :country, :year)
model_custom = estimate_pvar(pd_custom, 1; dependent_vars=["gdp", "inv", "cons"],
                             collapse=true, max_lag_endo=6)
(n_groups = model_custom.n_groups, n_obs = model_custom.n_obs,
 n_instruments = model_custom.n_instruments)
```

### Data Cleaning

`apply_tcode` removes rows lost to differencing but does **not** drop rows with pre-existing NaN values. If your raw data contains missing observations, clean the transformed panel before estimation:

```julia
pd = apply_tcode(pwt, 5)

# Option 1: drop all rows with NaN/Inf
pd_clean = dropna(pd)

# Option 2: drop only rows where specific variables are missing
pd_clean = dropna(pd; vars=["rgdpna", "emp"])

# Option 3: full data cleaning (listwise deletion + constant-column removal)
pd_clean = fix(pd)
```

---

## GMM Estimation

### First-Difference GMM (Arellano-Bond)

The default estimator transforms the model by first-differencing to remove ``\boldsymbol{\mu}_i``:

```math
\Delta \mathbf{y}_{i,t} = \sum_{l=1}^{p} \mathbf{A}_l \, \Delta \mathbf{y}_{i,t-l} + \Delta \boldsymbol{\varepsilon}_{i,t}
```

where:
- ``\Delta \mathbf{y}_{i,t} = \mathbf{y}_{i,t} - \mathbf{y}_{i,t-1}`` is the first-differenced endogenous vector
- ``\Delta \boldsymbol{\varepsilon}_{i,t}`` is the first-differenced error (MA(1) by construction)

Lagged **levels** ``\mathbf{y}_{i,t-2}, \mathbf{y}_{i,t-3}, \ldots`` serve as instruments for ``\Delta \mathbf{y}_{i,t-l}`` (Holtz-Eakin, Newey & Rosen 1988; Arellano & Bond 1991). The instrument matrix is block-diagonal, with the number of instruments growing with ``t``.

!!! note "One-Step vs Two-Step"
    The two-step estimator is asymptotically efficient but its naive standard errors are severely downward-biased in finite samples. The package automatically applies the Windmeijer (2005) correction for two-step GMM, which restores proper inference.

```@example pvar
# One-step GMM (heteroskedasticity-robust SEs)
m1 = estimate_pvar(pd, 1; dependent_vars=dep_vars, steps=:onestep,
                   collapse=true, max_lag_endo=6)

# Forward orthogonal deviations (Arellano & Bover 1995)
m3 = estimate_pvar(pd, 1; dependent_vars=dep_vars, transformation=:fod,
                   steps=:twostep, collapse=true, max_lag_endo=6)

(onestep = round(m1.Phi[1, 1], digits=4),
 twostep = round(model.Phi[1, 1], digits=4),
 fod     = round(m3.Phi[1, 1], digits=4))
```

The three estimates of the own-persistence coefficient on GDP growth bracket each other closely: 0.3164 one-step, 0.2513 two-step, and 0.2695 under forward orthogonal deviations. One-step and two-step differ only in the weighting matrix, so a gap of this size is ordinary sampling variation rather than a specification difference; the two-step version is the efficient one and is the default. The forward orthogonal deviations (FOD) transformation subtracts the mean of all *future* observations instead of the previous one, which preserves orthogonality of the transformed errors and — unlike first-differencing — does not lose an observation at every internal gap, making it the better choice on unbalanced panels.

### System GMM (Blundell-Bond)

System GMM adds level equations instrumented by lagged **differences**, improving efficiency when the data are persistent (Blundell & Bond 1998):

```math
\underbrace{\begin{pmatrix} \Delta \mathbf{y}_{i,t} \\ \mathbf{y}_{i,t} \end{pmatrix}}_{\text{stacked}} = \mathbf{X}_{i,t} \, \boldsymbol{\Phi} + \begin{pmatrix} \Delta \boldsymbol{\varepsilon}_{i,t} \\ \boldsymbol{\varepsilon}_{i,t} \end{pmatrix}
```

where:
- The top block uses lagged levels ``\mathbf{y}_{i,t-2}, \ldots`` as instruments (as in FD-GMM)
- The bottom block uses lagged differences ``\Delta \mathbf{y}_{i,t-1}`` as instruments for the level equation

```@example pvar
m_sys = estimate_pvar(pd, 1; dependent_vars=dep_vars, system_instruments=true,
                      steps=:twostep, collapse=true, max_lag_endo=6)
report(m_sys)
```

Adding the level equations doubles the effective sample from 1064 to 2128 observations and raises the instrument count only from 15 to 19, which is why system GMM is the standard remedy for weak instruments under persistence. Its estimates diverge sharply from FD-GMM on the human-capital equation: own-persistence rises from 0.2804 (insignificant) to 0.9092 (``z = 14.71``), and the GDP equation's loading on lagged human capital jumps from 1.0148 to 3.0713. That divergence is itself diagnostic — human-capital growth is highly persistent, exactly the case where lagged levels are weak instruments for differences and the extra level moments carry most of the identification. It is also the case where the additional assumption bites hardest: system GMM requires that first differences be uncorrelated with the fixed effects (mean stationarity), and a large FD/system gap should prompt a Hansen J on the system moments before the level results are believed.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dependent_vars` | `Union{Vector{String},Nothing}` | `nothing` | Endogenous variable names (`nothing` = all columns) |
| `predet_vars` | `Vector{String}` | `String[]` | Predetermined variable names |
| `exog_vars` | `Vector{String}` | `String[]` | Strictly exogenous variable names |
| `transformation` | `Symbol` | `:fd` | `:fd` (first-difference) or `:fod` (forward orthogonal deviations) |
| `steps` | `Symbol` | `:twostep` | `:onestep`, `:twostep`, or `:mstep` (iterated) |
| `system_instruments` | `Bool` | `false` | Use System GMM (Blundell-Bond) |
| `system_constant` | `Bool` | `true` | Include constant in level equation (System GMM) |
| `min_lag_endo` | `Int` | `2` | Minimum instrument lag for endogenous variables |
| `max_lag_endo` | `Int` | `99` | Maximum instrument lag (99 = all available) |
| `collapse` | `Bool` | `false` | Collapse instruments to limit proliferation |
| `pca_instruments` | `Bool` | `false` | Apply PCA reduction to instruments |
| `pca_max_components` | `Int` | `0` | Maximum PCA components (0 = auto) |
| `max_iter` | `Int` | `100` | Maximum iterations for iterated GMM |

### Return Value

| Field | Type | Description |
|-------|------|-------------|
| `Phi` | `Matrix{T}` | ``m \times K`` coefficient matrix |
| `Sigma` | `Matrix{T}` | ``m \times m`` residual covariance |
| `se` | `Matrix{T}` | Robust standard errors (same shape as `Phi`) |
| `pvalues` | `Matrix{T}` | P-values (same shape as `Phi`) |
| `m` | `Int` | Number of endogenous variables |
| `p` | `Int` | Number of lags |
| `varnames` | `Vector{String}` | Endogenous variable names |
| `method` | `Symbol` | `:fd_gmm`, `:system_gmm`, or `:fe_ols` |
| `transformation` | `Symbol` | `:fd`, `:fod`, or `:demean` |
| `steps` | `Symbol` | `:onestep`, `:twostep`, or `:mstep` |
| `n_predet` / `n_exog` | `Int` | Counts of predetermined and strictly exogenous regressors |
| `predet_names` / `exog_names` | `Vector{String}` | Their names |
| `system_constant` | `Bool` | Whether the level equation carries a constant |
| `n_groups` | `Int` | Number of panel groups |
| `n_periods` | `Int` | Number of distinct time periods |
| `n_obs` | `Int` | Total effective observations |
| `obs_per_group` | `NamedTuple` | `(min, avg, max)` observations per group |
| `n_instruments` | `Int` | Number of moment conditions |
| `instruments` | `Vector{Matrix{T}}` | Per-group instrument blocks |
| `weighting_matrix` | `Matrix{T}` | GMM weighting matrix used |
| `coef_vcov` | `Vector{Matrix{T}}` | Per-equation ``K \times K`` coefficient covariance |
| `data` | `PanelData{T}` | The panel the model was fitted on |

`StatsAPI` accessors are defined: `coef(m)` returns `vec(m.Phi)`, `stderror(m)` returns
`vec(m.se)`, and `vcov(m)` assembles the block-diagonal coefficient covariance from
`coef_vcov`.

---

## Fixed-Effects OLS

For panels with large ``T``, the within (FE-OLS) estimator provides a simpler alternative. The estimator demeans each entity's data (removing ``\boldsymbol{\mu}_i``) and runs pooled OLS on the stacked system with cluster-robust standard errors at the group level:

```@example pvar
m_fe = estimate_pvar_feols(pd, 1; dependent_vars=dep_vars)
report(m_fe)
```

On this panel FE-OLS and two-step FD-GMM agree closely on the GDP equation — own-persistence 0.2486 against 0.2513 — which is what the theory predicts once ``T`` is large enough for the ``O(1/T)`` Nickell bias to be small. With 29 periods per country the bias is on the order of a few percent, so the within estimator is a reasonable cross-check and its standard errors are noticeably tighter because it uses no instruments. The two disagree more on the human-capital equation (0.3495 against 0.2804), the most persistent series, where GMM's weak-instrument problem and FE-OLS's Nickell bias both bite hardest.

The FE-OLS estimator accepts the same `dependent_vars`, `predet_vars`, and `exog_vars` keywords as the GMM estimator, but none of the instrument keywords, since it uses no instruments. Standard errors are clustered at the group level.

---

## Instrument Management

When the number of instruments is large relative to ``N``, standard errors become unreliable and the Hansen J-test loses power. Several options control instrument proliferation:

```@example pvar
# The moment count does not depend on `steps`, so compare with the cheap one-step fit
counts = map(kw -> estimate_pvar(pd, 1; dependent_vars=dep_vars,
                                 steps=:onestep, kw...).n_instruments,
             [NamedTuple(),                            # defaults: every available lag
              (min_lag_endo=2, max_lag_endo=4),        # shallow lag window
              (collapse=true,),                        # one column per lag distance
              (collapse=true, max_lag_endo=6),         # collapse + shallow window
              (pca_instruments=true,)])                # PCA reduction
(defaults = counts[1], lag_window = counts[2], collapse = counts[3],
 collapse_window = counts[4], pca = counts[5], n_groups = pd.n_groups)
```

The defaults generate 1218 moment conditions from 38 countries — 32 instruments per group, an extreme case of the proliferation problem. Restricting the lag window to 2-4 cuts this to 243, still far too many. `collapse=true` replaces the block-diagonal design with one column per (variable, lag distance) pair, giving 84; combining it with `max_lag_endo=6` gives the 15 used throughout this page, comfortably below ``N``. PCA reduction lands at 27. The four routes are not equivalent — collapsing preserves the moment interpretation while PCA does not — but any of them is preferable to the default on a panel this narrow.

!!! warning "Instrument Proliferation"
    Keep the instrument count below ``N``. When it exceeds ``N`` the sample moment
    covariance has rank at most ``N`` and its inverse is a pseudo-inverse, so the two-step
    weighting matrix overfits the moment conditions and the Hansen J degenerates: on this
    panel the default specification returns ``J = 38.0000`` — exactly the number of groups
    — with 1215 degrees of freedom and ``p = 1.0``, a result that carries no information.
    `report` flags the condition with a `⚠ too many` marker next to the instrument count.

---

## Structural Analysis

### Impulse Response Functions

**Orthogonalized IRFs** use the Cholesky decomposition of the residual covariance ``\Sigma = PP'``. The impulse responses are computed from the companion form ``\Phi_h = J A^h J'`` and the Cholesky factor ``P``:

```math
\Psi_h = \Phi_h \cdot P
```

where:
- ``\Phi_h`` is the ``m \times m`` moving-average coefficient at horizon ``h``
- ``P`` is the lower-triangular Cholesky factor of ``\Sigma``
- ``J = [I_m \mid 0 \cdots 0]`` is the ``m \times mp`` selection matrix

`pvar_oirf` returns an ``(H+1) \times m \times m`` array indexed as `[horizon, response, shock]`:

```@example pvar
irfs = pvar_oirf(model, 10)
round.(irfs[1, :, :], digits=4)     # impact matrix (h = 0)
```

The impact matrix is lower triangular by construction — that is what the Cholesky ordering imposes. With the ordering `rgdpna, emp, hc`, a one-standard-deviation GDP-growth shock raises GDP growth by 3.14 percentage points on impact and employment growth by 1.22, while employment and human-capital shocks are defined to have no contemporaneous effect on GDP. The ordering is an identifying assumption, not a property of the data; reverse it and the impact matrix changes.

```@example pvar
round.(irfs[2, :, :], digits=4)     # one period later (h = 1)
```

Responses collapse within one period: the own-response of GDP growth falls from 0.0314 to 0.0067, and by ``h = 5`` every entry rounds to zero at four decimals. This is exactly what the stability analysis below predicts from companion moduli near 0.27 — growth rates, unlike levels, carry almost no propagation, so the interesting dynamics here are contemporaneous rather than persistent.

**Generalized IRFs** (Pesaran & Shin 1998) do not depend on variable ordering:

```math
\text{GIRF}_h(\mathbf{e}_j) = \frac{\Phi_h \, \Sigma \, \mathbf{e}_j}{\sqrt{\sigma_{jj}}}
```

where:
- ``\mathbf{e}_j`` is the ``j``-th unit vector
- ``\sigma_{jj} = \Sigma[j,j]`` is the variance of the ``j``-th variable

```@example pvar
girfs = pvar_girf(model, 10)
round.(girfs[1, :, :], digits=4)
```

The generalized impact matrix is not triangular: an employment shock now moves GDP growth by 0.0208 on impact, because GIRFs condition on the historically observed correlation between shocks rather than zeroing it out. GIRFs sidestep the ordering choice at the cost of not corresponding to any single structural experiment — the columns are responses to *typical* correlated shocks, so they do not sum to a variance decomposition.

### Forecast Error Variance Decomposition

FEVD quantifies the share of forecast error variance of variable ``l`` attributable to shock ``k`` at horizon ``h``:

```math
\Omega_{l,k,h} = \frac{\sum_{j=0}^{h} (\Psi_j)_{l,k}^2}{\text{MSE}_{h,ll}}
```

where:
- ``\Psi_j`` is the orthogonalized impulse response at horizon ``j``
- ``\text{MSE}_{h,ll} = \sum_{j=0}^{h} (\Phi_j \Sigma \Phi_j')_{ll}`` is the forecast error variance of variable ``l``

Each row sums to 1 (100% of forecast error variance accounted for).

```@example pvar
decomp = pvar_fevd(model, 10)
round.(decomp[11, :, :], digits=4)   # horizon 10
```

At a ten-year horizon GDP growth is 98.2% own-shock, human-capital growth 97.8% own-shock, and employment growth splits 51.8/48.1 between GDP shocks and its own. The employment row is the substantive result: half of the unpredictable variation in employment growth is attributable to output shocks, consistent with employment responding to demand rather than driving it. Because the responses die out so fast, the decomposition at ``h = 10`` is nearly identical to the one at ``h = 1``.

### Stability Analysis

The system is stable if all eigenvalues of the companion matrix lie inside the unit circle:

```@example pvar
stab = pvar_stability(model)
report(stab)
```

All three moduli — 0.2651, 0.2651, and 0.1585 — sit well inside the unit circle, so the estimated PVAR is stationary and its impulse responses converge to zero. This is expected: the panel was differenced with `apply_tcode(pwt, 5)`, so the variables are growth rates rather than levels. A modulus at or above 1 would mean the growth-rate system is itself explosive and would invalidate every IRF and FEVD on this page.

```julia
plot_result(model; view=:stability)
```

```@raw html
<iframe src="../assets/plots/pvar_stability.html" style="width:100%;height:420px;border:1px solid #eee;border-radius:8px;" loading="lazy"></iframe>
```

| Field | Type | Description |
|-------|------|-------------|
| `eigenvalues` | `Vector{Complex{T}}` | Eigenvalues of companion matrix |
| `moduli` | `Vector{T}` | Moduli sorted in descending order |
| `is_stable` | `Bool` | `true` if all moduli are strictly less than 1 |

---

## Bootstrap Confidence Intervals

Group-level block bootstrap preserves the within-group time structure. For each bootstrap draw, ``N`` groups are resampled with replacement, the PVAR is re-estimated, and IRFs are computed. Quantile-based confidence intervals are constructed from the bootstrap distribution:

!!! warning "The bootstrap does not inherit the instrument controls"
    `pvar_bootstrap_irf` re-estimates each draw carrying over `transformation`, `steps`,
    `system_instruments`, and `system_constant` — but **not** `collapse`, `min_lag_endo`,
    `max_lag_endo`, or `pca_instruments`. A model fitted with collapsed instruments is
    therefore resampled with the full default instrument set, which changes the estimator
    being bootstrapped and, on a panel like this one, makes each draw hundreds of times
    more expensive. Until this is addressed, bootstrap the FE-OLS estimator, which uses no
    instruments and is unaffected.

```@example pvar
Random.seed!(7)
# n_draws=200 is fast for FE-OLS; use 500+ for publication
boot = pvar_bootstrap_irf(model_fe, 10;
    irf_type=:oirf,   # or :girf
    n_draws=200,
    ci=0.90
)

(impact = round(boot.irf[1, 1, 1], digits=4),
 lower  = round(boot.lower[1, 1, 1], digits=4),
 upper  = round(boot.upper[1, 1, 1], digits=4))
```

The own-impact response of GDP growth is 0.0308 with a 90% interval of ``[0.0281, 0.0338]``, comfortably away from zero and closely matching the 0.0314 the GMM fit reports. Because groups rather than observations are resampled, the interval accounts for arbitrary serial correlation within a country while treating countries as exchangeable. The percentile interval is not bias-corrected, so at horizons where the response is near zero the point estimate can fall outside the band — at ``h = 2`` the point estimate rounds to 0.0000 against an interval of ``[-0.0007, 0.0012]``, which is a statement about bootstrap skewness rather than about the data.

Pass the result to `plot_result` through the `ci` keyword to draw the bands on the IRF panel:

```julia
plot_result(model_fe; view=:oirf, H=10, ci=boot)
```

The returned named tuple contains `boot.irf` (point estimate), `boot.lower` and `boot.upper` (CI bounds), and `boot.draws` (all bootstrap draws). All arrays have shape ``(H+1) \times m \times m``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `irf_type` | `Symbol` | `:oirf` | `:oirf` (orthogonalized) or `:girf` (generalized) |
| `n_draws` | `Int` | `500` | Number of bootstrap replications |
| `ci` | `Real` | `0.95` | Confidence level |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Generator for the per-draw seeds |

| Field | Type | Description |
|-------|------|-------------|
| `irf` | `Array{T,3}` | Point estimate ``(H+1) \times m \times m`` |
| `lower` | `Array{T,3}` | Lower CI bound |
| `upper` | `Array{T,3}` | Upper CI bound |
| `draws` | `Array{T,4}` | All bootstrap draws ``n_{\text{draws}} \times (H+1) \times m \times m`` |

---

## Specification Tests

### Hansen J-Test

The Hansen (1982) J-test evaluates whether the overidentifying restrictions (moment conditions) are valid. Under ``H_0``: all moment conditions are correctly specified.

```math
J = N \cdot \bar{g}(\hat{\theta})' \, \hat{W} \, \bar{g}(\hat{\theta}) \sim \chi^2(c - b)
```

where:
- ``\bar{g}(\hat{\theta})`` is the ``c \times 1`` vector of sample moment conditions evaluated at the GMM estimate
- ``\hat{W}`` is the optimal weighting matrix
- ``c`` is the number of instruments and ``b`` is the number of estimated parameters

```@example pvar
j = pvar_hansen_j(model)
report(j)
```

The J-statistic of 18.61 on 12 degrees of freedom gives ``p = 0.0983``, so the overidentifying restrictions survive at the 5% level but not comfortably — a p-value in the 0.05-0.10 band is a signal to check the specification rather than a clean pass. The degrees of freedom are ``c - b`` computed per equation (15 instruments against 3 parameters), which is why `df` is 12 rather than ``15 - 9``. Rejection suggests instrument invalidity or model misspecification; non-rejection does not validate the instruments, it only means the data cannot reject the moment conditions at this sample size.

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `String` | `"Hansen J-test"` |
| `statistic` | `T` | J-statistic |
| `pvalue` | `T` | P-value (``\chi^2`` distribution) |
| `df` | `Int` | Degrees of freedom ``c - b`` |
| `n_instruments` | `Int` | Number of moment conditions ``c`` |
| `n_params` | `Int` | Number of estimated parameters ``b`` |

`pvar_hansen_j` throws an `ArgumentError` on an `estimate_pvar_feols` model — the within estimator uses no moment conditions, so there is nothing to overidentify.

### Andrews-Lu MMSC

Andrews & Lu (2001) Model and Moment Selection Criteria extend information criteria to GMM settings:

```math
\text{MMSC-BIC} = J - (c - b) \ln(n), \quad
\text{MMSC-AIC} = J - 2(c - b), \quad
\text{MMSC-HQIC} = J - \kappa (c - b) \ln \ln(n)
```

where:
- ``J`` is the Hansen J-statistic
- ``c`` is the number of instruments, ``b`` is the number of parameters, ``n`` is the number of observations
- ``\kappa > 2`` is the Hannan-Quinn constant, set by the `hq_criterion` keyword (default 2.1)

Lower values are preferred. Unlike an ordinary information criterion these penalize *overidentification* rather than parameter count, so a specification with more valid instruments scores better at equal fit:

```@example pvar
mmsc = pvar_mmsc(model)
(bic = round(mmsc.bic, digits=3), aic = round(mmsc.aic, digits=3),
 hqic = round(mmsc.hqic, digits=3))
```

All three criteria are strongly negative because the overidentification term ``(c - b) \ln(n)`` dominates a J-statistic of 18.61. The absolute values carry no meaning; only comparisons across specifications estimated on the same sample do.

---

## Lag Selection

The `pvar_lag_selection` function estimates a PVAR at each candidate lag order, computes the MMSC criteria for each, and reports the minimizing lag. Any keyword accepted by `estimate_pvar` is forwarded, so the instrument controls must be repeated here to keep the comparison meaningful:

```@example pvar
sel = pvar_lag_selection(pd, 4; dependent_vars=dep_vars,
                         collapse=true, max_lag_endo=6)
sel.table
```

```@example pvar
(best_bic = sel.best_bic, best_aic = sel.best_aic, best_hqic = sel.best_hqic)
```

All three criteria select ``p = 1``, and the BIC column deteriorates monotonically from ``-65.02`` at one lag to ``-18.49`` at four. Annual growth rates carry little serial dependence beyond one lag, so the extra coefficients buy no fit while each additional lag consumes instruments. If estimation fails at some candidate lag the corresponding criteria are set to infinity and the row prints as `—`, so a table with dashes means those specifications could not be estimated, not that they scored badly.

The returned named tuple carries `table` (the formatted comparison), `best_bic`, `best_aic`, `best_hqic` (the minimizing lag orders), and `models` (the fitted `PVARModel` at each lag).

---

## Complete Example

```@example pvar
# Lag selection across candidate orders
sel = pvar_lag_selection(pd, 3; dependent_vars=dep_vars,
                         collapse=true, max_lag_endo=6)
sel.table
```

```@example pvar
# Estimate at the selected lag via two-step FD-GMM
model = estimate_pvar(pd, sel.best_bic; dependent_vars=dep_vars, steps=:twostep,
                      collapse=true, max_lag_endo=6)
report(model)
```

```@example pvar
# Instrument validity
j = pvar_hansen_j(model)
report(j)
```

```@example pvar
# Stationarity of the estimated system
stab = pvar_stability(model)
report(stab)
```

```@example pvar
# Structural analysis
irfs = pvar_oirf(model, 10)
decomp = pvar_fevd(model, 10)
round.(decomp[11, :, :], digits=4)
```

```@example pvar
# Bootstrap confidence intervals (FE-OLS: see the caveat in the bootstrap section)
Random.seed!(7)
boot = pvar_bootstrap_irf(model_fe, 10; n_draws=200, ci=0.90)
(impact = round(boot.irf[1, 1, 1], digits=4),
 ci = (round(boot.lower[1, 1, 1], digits=4), round(boot.upper[1, 1, 1], digits=4)))
```

```@example pvar
# Academic references for the estimator
print(refs(model))
```

The workflow runs in one direction: choose the lag order, estimate, validate, then interpret. All three MMSC criteria select one lag, so the model is a PVAR(1) in growth rates estimated by two-step FD-GMM with 15 collapsed instruments against 38 countries. The Hansen J of 18.61 on 12 degrees of freedom does not reject the overidentifying restrictions at 5%, and all companion moduli lie near 0.27, so the system is comfortably stationary and its impulse responses converge. The FEVD then delivers the economic content: output and human capital are essentially own-shock driven at every horizon, while roughly half of the forecast error variance in employment growth traces back to output shocks. The FE-OLS block bootstrap confirms that the contemporaneous own-effect of an output shock, 0.0308, is estimated precisely enough to exclude zero at the 90% level.

!!! warning "`refs` returns a String"
    Call `print(refs(model))`, not `refs(model)`. The single-argument form builds the
    bibliography into a `String` and returns it, so evaluating it bare renders the escaped
    literal (`"Holtz-Eakin, Douglas...\n"`) instead of the formatted list.

---

## Common Pitfalls

1. **Nickell bias in FE-OLS**: The within estimator is biased of order ``O(1/T)`` in dynamic panels (Nickell 1981). For panels with ``T < 20``, FE-OLS overestimates the persistence of the lagged dependent variable. Use GMM estimation (`estimate_pvar`) instead of `estimate_pvar_feols` when the time dimension is short relative to the cross-section.

2. **Accepting the default instrument set**: The block-diagonal design generates one instrument per available lag at every period, which on this 38-country, 30-period panel means 1218 moment conditions. Always pass `collapse=true` and a finite `max_lag_endo` unless ``N`` is large; check `model.n_instruments` against `model.n_groups` before reading anything else.

3. **Instrument count exceeding N**: Once the moment count passes ``N`` the moment covariance is rank-deficient, the Hansen J degenerates to ``J = N`` with ``p = 1``, and standard errors become unreliable. `report` marks this with `⚠ too many`.

4. **Hansen J-test interpretation**: A high p-value (non-rejection) does not prove instrument validity — it only means the data cannot reject the moment conditions. Conversely, with many instruments the J-test almost never rejects even when some instruments are invalid. Always report the instrument count alongside the J-statistic.

5. **Comparing MMSC across different instrument sets**: The criteria penalize ``c - b``, so a specification with more instruments scores differently for reasons unrelated to lag length. Hold `collapse`, `min_lag_endo`, and `max_lag_endo` fixed across the candidates being compared, and pass them to `pvar_lag_selection` explicitly.

6. **Unbalanced panels**: First-differencing loses one observation per gap in the panel. Forward orthogonal deviations (`transformation=:fod`) handle unbalanced panels more efficiently by preserving orthogonality of the transformed errors. Use `:fod` when the panel has missing periods or unequal group sizes.

7. **System GMM stationarity assumption**: Blundell-Bond system GMM requires that first differences are uncorrelated with fixed effects — a mean stationarity condition. If the data exhibit trending behavior or structural breaks, the additional level-equation instruments are invalid and FD-GMM is preferred. A large gap between FD and system estimates on the most persistent equation is the warning sign.

8. **Reading a Cholesky IRF as ordering-free**: The impact matrix of `pvar_oirf` is lower triangular by assumption. Reorder `dependent_vars` and the impact responses change. Use `pvar_girf` when no defensible ordering exists, accepting that generalized responses do not decompose variance.

---

## References

- Arellano, Manuel, and Stephen Bond. 1991. "Some Tests of Specification for Panel Data."
  *Review of Economic Studies* 58 (2): 277--297. [DOI](https://doi.org/10.2307/2297968)

- Arellano, Manuel, and Olympia Bover. 1995. "Another Look at the Instrumental Variable Estimation of Error-Components Models."
  *Journal of Econometrics* 68 (1): 29--51. [DOI](https://doi.org/10.1016/0304-4076(94)01642-D)

- Andrews, Donald W. K., and Biao Lu. 2001. "Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models."
  *Journal of Econometrics* 101 (1): 123--164. [DOI](https://doi.org/10.1016/S0304-4076(00)00077-4)

- Blundell, Richard, and Stephen Bond. 1998. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models."
  *Journal of Econometrics* 87 (1): 115--143. [DOI](https://doi.org/10.1016/S0304-4076(98)00009-8)

- Feenstra, Robert C., Robert Inklaar, and Marcel P. Timmer. 2015. "The Next Generation of the Penn World Table."
  *American Economic Review* 105 (10): 3150--3182. [DOI](https://doi.org/10.1257/aer.20130954)

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators."
  *Econometrica* 50 (4): 1029--1054. [DOI](https://doi.org/10.2307/1912775)

- Holtz-Eakin, Douglas, Whitney Newey, and Harvey S. Rosen. 1988. "Estimating Vector Autoregressions with Panel Data."
  *Econometrica* 56 (6): 1371--1395. [DOI](https://doi.org/10.2307/1913103)

- Nickell, Stephen. 1981. "Biases in Dynamic Models with Fixed Effects."
  *Econometrica* 49 (6): 1417--1426. [DOI](https://doi.org/10.2307/1911408)

- Pesaran, M. Hashem, and Yongcheol Shin. 1998. "Generalized Impulse Response Analysis in Linear Multivariate Models."
  *Economics Letters* 58 (1): 17--29. [DOI](https://doi.org/10.1016/S0165-1765(97)00214-0)

- Windmeijer, Frank. 2005. "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators."
  *Journal of Econometrics* 126 (1): 25--51. [DOI](https://doi.org/10.1016/j.jeconom.2004.02.005)
