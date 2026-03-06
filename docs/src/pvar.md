# [Panel VAR](@id pvar_page)

**MacroEconometricModels.jl** provides a complete Panel VAR (PVAR) implementation for dynamic panel data analysis. The package supports GMM estimation via Arellano-Bond first-difference and Blundell-Bond system instruments, fixed-effects OLS, and a full suite of structural analysis and specification tests.

- **GMM estimation**: First-difference (Arellano & Bond 1991) and system (Blundell & Bond 1998) with Windmeijer (2005) corrected standard errors
- **FE-OLS**: Within estimator with cluster-robust standard errors for large-``T`` panels
- **Transformations**: First-differencing and forward orthogonal deviations (Arellano & Bover 1995)
- **Structural analysis**: Orthogonalized IRF (Cholesky), generalized IRF (Pesaran & Shin 1998), FEVD, and stability diagnostics
- **Bootstrap inference**: Group-level block bootstrap for IRF confidence intervals
- **Specification tests**: Hansen (1982) J-test, Andrews-Lu (2001) MMSC, and lag selection

## Quick Start

**Recipe 1: FD-GMM with two-step estimation**

```julia
using MacroEconometricModels

# Load Penn World Table: 38 OECD countries, 1950-2023
pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)  # Log first difference for stationarity

# Arellano-Bond two-step GMM
model = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)
report(model)
```

**Recipe 2: System GMM (Blundell-Bond)**

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)

# System GMM adds level equations instrumented by lagged differences
model_sys = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                          system_instruments=true, steps=:twostep)
report(model_sys)
```

**Recipe 3: Fixed-effects OLS**

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)

# Within estimator with cluster-robust SEs
model_fe = estimate_pvar_feols(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
report(model_fe)
```

**Recipe 4: Specification tests and lag selection**

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
dep_vars = ["rgdpna", "emp", "hc"]

model = estimate_pvar(pd, 2; dependent_vars=dep_vars, steps=:twostep)

# Hansen J-test for overidentifying restrictions
j = pvar_hansen_j(model)
report(j)

# Andrews-Lu MMSC for model selection
mmsc = pvar_mmsc(model)

# Lag selection across candidate models
sel = pvar_lag_selection(pd, 4; dependent_vars=dep_vars)
```

**Recipe 5: Structural analysis with bootstrap CIs**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
dep_vars = ["rgdpna", "emp", "hc"]

model = estimate_pvar(pd, 1; dependent_vars=dep_vars, steps=:twostep)

# Orthogonalized IRFs and FEVD
irfs = pvar_oirf(model, 10)
decomp = pvar_fevd(model, 10)

# Bootstrap confidence intervals
boot = pvar_bootstrap_irf(model, 10; n_draws=200, ci=0.90)
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

```julia
using MacroEconometricModels

# Load PWT --- already a PanelData object
pwt = load_example(:pwt)

# Convert to growth rates for stationarity
pd = apply_tcode(pwt, 5)  # tcode 5 = log first difference
```

All numeric columns are treated as potential endogenous variables. Use the `dependent_vars` keyword to select a subset:

```julia
model = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
```

For custom panel data, construct a `PanelData` object via `xtset`:

```julia
using DataFrames
df = DataFrame(country=repeat(1:20, inner=30), year=repeat(1:30, outer=20),
               gdp=randn(600), inv=randn(600), cons=randn(600))
pd = xtset(df, :country, :year)
model = estimate_pvar(pd, 2; dependent_vars=["gdp", "inv", "cons"])
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

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
dep_vars = ["rgdpna", "emp", "hc"]

# One-step GMM (heteroskedasticity-robust SEs)
m1 = estimate_pvar(pd, 2; dependent_vars=dep_vars, steps=:onestep)

# Two-step GMM (Windmeijer-corrected SEs)
m2 = estimate_pvar(pd, 2; dependent_vars=dep_vars, steps=:twostep)

# Forward orthogonal deviations (Arellano & Bover 1995)
m3 = estimate_pvar(pd, 2; dependent_vars=dep_vars,
                   transformation=:fod, steps=:twostep)
```

The forward orthogonal deviations (FOD) transformation preserves orthogonality of the transformed errors, making the initial weighting matrix more efficient than first-differencing when the panel is unbalanced.

### System GMM (Blundell-Bond)

System GMM adds level equations instrumented by lagged **differences**, improving efficiency when the data are persistent (Blundell & Bond 1998):

```math
\underbrace{\begin{pmatrix} \Delta \mathbf{y}_{i,t} \\ \mathbf{y}_{i,t} \end{pmatrix}}_{\text{stacked}} = \mathbf{X}_{i,t} \, \boldsymbol{\Phi} + \begin{pmatrix} \Delta \boldsymbol{\varepsilon}_{i,t} \\ \boldsymbol{\varepsilon}_{i,t} \end{pmatrix}
```

where:
- The top block uses lagged levels ``\mathbf{y}_{i,t-2}, \ldots`` as instruments (as in FD-GMM)
- The bottom block uses lagged differences ``\Delta \mathbf{y}_{i,t-1}`` as instruments for the level equation

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)

m_sys = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                      system_instruments=true, steps=:twostep)
report(m_sys)
```

The system estimator exploits additional moment conditions but requires the assumption that first differences are uncorrelated with fixed effects.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dependent_vars` | `Vector{String}` | `nothing` | Endogenous variable names (default: all columns) |
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
| `n_groups` | `Int` | Number of panel groups |
| `n_obs` | `Int` | Total effective observations |
| `n_instruments` | `Int` | Number of moment conditions |

---

## Fixed-Effects OLS

For panels with large ``T``, the within (FE-OLS) estimator provides a simpler alternative. The estimator demeans each entity's data (removing ``\boldsymbol{\mu}_i``) and runs pooled OLS on the stacked system with cluster-robust standard errors at the group level:

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)

m_fe = estimate_pvar_feols(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
report(m_fe)
```

The FE-OLS estimator accepts the same `dependent_vars`, `predet_vars`, and `exog_vars` keywords as the GMM estimator. Standard errors are clustered at the group level by default.

---

## Instrument Management

When the number of instruments is large relative to ``N``, standard errors become unreliable and the Hansen J-test loses power. Several options control instrument proliferation:

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
dep_vars = ["rgdpna", "emp", "hc"]

# Restrict instrument lags to avoid proliferation
m = estimate_pvar(pd, 2; dependent_vars=dep_vars,
                  min_lag_endo=2, max_lag_endo=4)

# Collapse instruments (one column per lag distance)
m = estimate_pvar(pd, 2; dependent_vars=dep_vars, collapse=true)

# PCA instrument reduction
m = estimate_pvar(pd, 2; dependent_vars=dep_vars, pca_instruments=true)
```

!!! warning "Instrument Proliferation"
    A rule of thumb: the number of instruments should not exceed ``N`` (the number of groups). When it does, the two-step GMM weighting matrix overfits the moment conditions, inflating the J-statistic and producing misleadingly small standard errors. Consider collapsing instruments or restricting lag depth.

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

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
model = estimate_pvar(pd, 1; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)

irfs = pvar_oirf(model, 20)   # H+1 x m x m array
```

**Generalized IRFs** (Pesaran & Shin 1998) do not depend on variable ordering:

```math
\text{GIRF}_h(\mathbf{e}_j) = \frac{\Phi_h \, \Sigma \, \mathbf{e}_j}{\sqrt{\sigma_{jj}}}
```

where:
- ``\mathbf{e}_j`` is the ``j``-th unit vector
- ``\sigma_{jj} = \Sigma[j,j]`` is the variance of the ``j``-th variable

```julia
girfs = pvar_girf(model, 20)   # H+1 x m x m array
```

### Forecast Error Variance Decomposition

FEVD quantifies the share of forecast error variance of variable ``l`` attributable to shock ``k`` at horizon ``h``:

```math
\Omega_{l,k,h} = \frac{\sum_{j=0}^{h} (\Psi_j)_{l,k}^2}{\text{MSE}_{h,ll}}
```

where:
- ``\Psi_j`` is the orthogonalized impulse response at horizon ``j``
- ``\text{MSE}_{h,ll} = \sum_{j=0}^{h} (\Phi_j \Sigma \Phi_j')_{ll}`` is the forecast error variance of variable ``l``

Each row sums to 1 (100% of forecast error variance accounted for).

```julia
decomp = pvar_fevd(model, 20)   # H+1 x m x m array
```

### Stability Analysis

The system is stable if all eigenvalues of the companion matrix lie inside the unit circle:

```julia
stab = pvar_stability(model)
stab.is_stable      # true if all |lambda| < 1
stab.moduli          # moduli of eigenvalues (sorted descending)
report(stab)
```

| Field | Type | Description |
|-------|------|-------------|
| `eigenvalues` | `Vector{Complex{T}}` | Eigenvalues of companion matrix |
| `moduli` | `Vector{T}` | Moduli sorted in descending order |
| `is_stable` | `Bool` | `true` if all moduli are strictly less than 1 |

---

## Bootstrap Confidence Intervals

Group-level block bootstrap preserves the within-group time structure. For each bootstrap draw, ``N`` groups are resampled with replacement, the PVAR is re-estimated, and IRFs are computed. Quantile-based confidence intervals are constructed from the bootstrap distribution:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
model = estimate_pvar(pd, 1; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)

boot = pvar_bootstrap_irf(model, 20;
    irf_type=:oirf,   # or :girf
    n_draws=500,
    ci=0.95
)
```

The returned named tuple contains `boot.irf` (point estimate), `boot.lower` and `boot.upper` (CI bounds), and `boot.draws` (all bootstrap draws). All arrays have shape ``(H+1) \times m \times m``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `irf_type` | `Symbol` | `:oirf` | `:oirf` (orthogonalized) or `:girf` (generalized) |
| `n_draws` | `Int` | `500` | Number of bootstrap replications |
| `ci` | `Real` | `0.95` | Confidence level |

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

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)
model = estimate_pvar(pd, 1; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)

j = pvar_hansen_j(model)
report(j)
```

Rejection suggests instrument invalidity or model misspecification. Non-rejection does not validate the instruments --- it only means the data cannot reject the moment conditions at the given sample size.

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `String` | `"Hansen J-test"` |
| `statistic` | `T` | J-statistic |
| `pvalue` | `T` | P-value (``\chi^2`` distribution) |
| `df` | `Int` | Degrees of freedom ``c - b`` |
| `n_instruments` | `Int` | Number of moment conditions ``c`` |
| `n_params` | `Int` | Number of estimated parameters ``b`` |

### Andrews-Lu MMSC

Andrews & Lu (2001) Model and Moment Selection Criteria extend information criteria to GMM settings:

```math
\text{MMSC-BIC} = J - (c - b) \ln(n), \quad
\text{MMSC-AIC} = J - 2(c - b)
```

where:
- ``J`` is the Hansen J-statistic
- ``c`` is the number of instruments, ``b`` is the number of parameters, ``n`` is the number of observations

Lower values are preferred. These criteria penalize overidentification, balancing model fit against instrument proliferation.

```julia
mmsc = pvar_mmsc(model)
mmsc.bic     # MMSC-BIC
mmsc.aic     # MMSC-AIC
mmsc.hqic    # MMSC-HQIC
```

---

## Lag Selection

The `pvar_lag_selection` function compares MMSC criteria across candidate lag orders to select the optimal specification:

```julia
using MacroEconometricModels

pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)

sel = pvar_lag_selection(pd, 4; dependent_vars=["rgdpna", "emp", "hc"])
sel.best_bic    # optimal lag by BIC
sel.best_aic    # optimal lag by AIC
sel.best_hqic   # optimal lag by HQIC
```

The function estimates PVAR models for lags 1 through the maximum candidate, computes MMSC criteria for each, and returns the lag order that minimizes each criterion.

---

## Complete Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load Penn World Table and convert to growth rates
pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)  # log first difference
dep_vars = ["rgdpna", "emp", "hc"]

# Lag selection
sel = pvar_lag_selection(pd, 3; dependent_vars=dep_vars)

# Estimate via two-step FD-GMM
model = estimate_pvar(pd, 1; dependent_vars=dep_vars, steps=:twostep)
report(model)

# Specification tests
j = pvar_hansen_j(model)
report(j)

# Stability check
stab = pvar_stability(model)
report(stab)

# Structural analysis
irfs = pvar_oirf(model, 10)
decomp = pvar_fevd(model, 10)

# Bootstrap confidence intervals
boot = pvar_bootstrap_irf(model, 10; n_draws=200, ci=0.90)

# Academic references
refs(model)
```

The `model` object stores the GMM coefficient estimates with Windmeijer-corrected standard errors, per-equation coefficient tables, and all GMM internals needed for specification testing. The Hansen J-test evaluates instrument validity, while the stability analysis confirms that the companion matrix eigenvalues lie inside the unit circle. The orthogonalized IRFs trace out the dynamic transmission of shocks through the GDP-employment-human capital system, and the bootstrap provides pointwise confidence intervals that account for estimation uncertainty. The FEVD quantifies how much of the forecast error variance in each variable is attributable to each structural shock at different horizons.

---

## Common Pitfalls

1. **Nickell bias in FE-OLS**: The within estimator is biased of order ``O(1/T)`` in dynamic panels (Nickell 1981). For panels with ``T < 20``, FE-OLS overestimates the persistence of the lagged dependent variable. Use GMM estimation (`estimate_pvar`) instead of `estimate_pvar_feols` when the time dimension is short relative to the cross-section.

2. **Instrument count exceeding N**: When the number of instruments exceeds ``N`` (the number of groups), the Hansen J-test loses power and standard errors become unreliable. Use `collapse=true`, restrict `max_lag_endo`, or apply `pca_instruments=true` to reduce instrument count. A rule of thumb: keep the instrument count below ``N``.

3. **Hansen J-test interpretation**: A high p-value (non-rejection) does not prove instrument validity --- it only means the data cannot reject the moment conditions. Conversely, with many instruments, the J-test almost never rejects even when some instruments are invalid. Always report the instrument count alongside the J-statistic.

4. **Unbalanced panels**: First-differencing loses one observation per gap in the panel. Forward orthogonal deviations (`transformation=:fod`) handle unbalanced panels more efficiently by preserving orthogonality of the transformed errors. Use `:fod` when the panel has missing periods or unequal group sizes.

5. **System GMM stationarity assumption**: Blundell-Bond system GMM requires that first differences are uncorrelated with fixed effects --- a mean stationarity condition. If the data exhibit trending behavior or structural breaks, the additional level-equation instruments are invalid and FD-GMM is preferred.

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
