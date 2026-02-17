# Panel VAR

This page documents the Panel VAR (PVAR) implementation in **MacroEconometricModels.jl**, providing estimation via GMM (Arellano-Bond first-difference and Blundell-Bond system) and fixed-effects OLS.

## Quick Start

```julia
using MacroEconometricModels

# Load Penn World Table: 38 OECD countries, 1950–2023
pwt = load_example(:pwt)

# Convert to growth rates (log first difference) for stationarity
pd = apply_tcode(pwt, 5)

# FD-GMM (Arellano-Bond) — GDP, employment, human capital growth spillovers
model = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)

# System GMM (Blundell-Bond)
model_sys = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                          system_instruments=true, steps=:twostep)

# Fixed-effects OLS
model_fe = estimate_pvar_feols(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
```

---

## Model Specification

The Panel VAR(p) model for entity ``i`` at time ``t`` is:

```math
\mathbf{y}_{i,t} = \boldsymbol{\mu}_i + \sum_{l=1}^{p} \mathbf{A}_l \mathbf{y}_{i,t-l} + \boldsymbol{\varepsilon}_{i,t}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T_i
```

where:
- ``\mathbf{y}_{i,t} \in \mathbb{R}^m`` is the vector of endogenous variables for entity ``i``
- ``\boldsymbol{\mu}_i \in \mathbb{R}^m`` is an entity-specific **fixed effect**
- ``\mathbf{A}_l`` is the ``m \times m`` coefficient matrix for lag ``l``
- ``\boldsymbol{\varepsilon}_{i,t} \sim (0, \Sigma)`` are i.i.d. innovations

The key econometric challenge is that ``\boldsymbol{\mu}_i`` is correlated with ``\mathbf{y}_{i,t-l}`` by construction. OLS on the level equation is inconsistent. Two strategies are available:

1. **Transform away the fixed effect** (first-differencing or forward orthogonal deviations) and estimate by GMM using lagged levels as instruments.
2. **Demean within groups** (within estimator) and estimate by OLS. This is consistent for large ``T`` but Nickell-biased when ``T`` is small relative to ``N``.

!!! note "Fixed Effects and Nickell Bias"
    The within estimator (FE-OLS) is biased of order ``O(1/T)`` in dynamic panels (Nickell, 1981). For panels with small ``T`` (e.g., ``T < 20``), GMM estimation is strongly preferred. For larger ``T``, FE-OLS and GMM converge.

---

## Panel Data Preparation

Panel VAR estimation requires a `PanelData` object. The built-in Penn World Table provides a balanced panel of 38 OECD countries:

```julia
using MacroEconometricModels

# Load PWT — already a PanelData object
pwt = load_example(:pwt)
println(ngroups(pwt), " countries × ", nvars(pwt), " variables")

# Convert to growth rates for stationarity
pd = apply_tcode(pwt, 5)  # tcode 5 = log first difference
```

All numeric columns are treated as potential endogenous variables. Use the `dependent_vars` keyword to select a subset:

```julia
model = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
```

---

## GMM Estimation

### First-Difference GMM (Arellano-Bond)

The default estimator transforms the model by first-differencing to remove ``\boldsymbol{\mu}_i``:

```math
\Delta \mathbf{y}_{i,t} = \sum_{l=1}^{p} \mathbf{A}_l \Delta \mathbf{y}_{i,t-l} + \Delta \boldsymbol{\varepsilon}_{i,t}
```

Lagged **levels** ``\mathbf{y}_{i,t-2}, \mathbf{y}_{i,t-3}, \ldots`` serve as instruments for ``\Delta \mathbf{y}_{i,t-l}`` (Holtz-Eakin, Newey & Rosen, 1988; Arellano & Bond, 1991). The instrument matrix is block-diagonal, with the number of instruments growing with ``t``.

```julia
# One-step GMM (heteroskedasticity-robust SEs)
m1 = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], steps=:onestep)

# Two-step GMM (Windmeijer-corrected SEs)
m2 = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], steps=:twostep)

# Forward orthogonal deviations (Arellano & Bover, 1995)
m3 = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                   transformation=:fod, steps=:twostep)
```

!!! note "One-Step vs Two-Step"
    The two-step estimator is asymptotically efficient but its naive standard errors are severely downward-biased in finite samples. The package automatically applies the Windmeijer (2005) correction for two-step GMM, which restores proper inference.

### System GMM (Blundell-Bond)

System GMM adds level equations instrumented by lagged **differences**, improving efficiency when the data are persistent (Blundell & Bond, 1998):

```julia
m_sys = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                      system_instruments=true, steps=:twostep)
```

The system estimator stacks transformed equations (instrumented by lagged levels) with level equations (instrumented by lagged differences). This exploits additional moment conditions but requires the assumption that first differences are uncorrelated with fixed effects.

### Instrument Management

When the number of instruments is large relative to ``N``, standard errors can be unreliable. Several options control instrument proliferation:

```julia
# Restrict instrument lags
m = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"],
                  min_lag_endo=2, max_lag_endo=4)

# Collapse instruments (one column per lag distance)
m = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], collapse=true)

# PCA instrument reduction
m = estimate_pvar(pd, 2; dependent_vars=["rgdpna", "emp", "hc"], pca_instruments=true)
```

!!! warning "Instrument Proliferation"
    A rule of thumb: the number of instruments should not exceed ``N`` (the number of groups). When it does, consider collapsing instruments or restricting lag depth.

---

## Fixed-Effects OLS

For panels with large ``T``, the within (FE-OLS) estimator provides a simpler alternative:

```julia
m_fe = estimate_pvar_feols(pd, 2; dependent_vars=["rgdpna", "emp", "hc"])
```

The estimator demeans each country's data (removing ``\boldsymbol{\mu}_i``) and runs pooled OLS on the stacked system. Standard errors are clustered at the group level.

---

## Structural Analysis

### Orthogonalized Impulse Response Functions

OIRF uses the Cholesky decomposition of the residual covariance ``\Sigma = PP'``:

```julia
irfs = pvar_oirf(model, 20)   # 20-period horizon
# irfs[h+1] is m × m: response of variable i to shock j at horizon h
```

The impulse responses are computed from the companion form ``\Phi_h = J A^h J'`` and the Cholesky factor ``P``:

```math
\Psi_h = \Phi_h P
```

### Generalized Impulse Response Functions

GIRF (Pesaran & Shin, 1998) does not depend on variable ordering:

```julia
girfs = pvar_girf(model, 20)
```

```math
\text{GIRF}_h(\mathbf{e}_j) = \frac{\Phi_h \Sigma \mathbf{e}_j}{\sqrt{\sigma_{jj}}}
```

### Forecast Error Variance Decomposition

FEVD based on the orthogonalized IRF:

```julia
decomp = pvar_fevd(model, 20)
# decomp[h+1] is m × m: share of FEV of variable i due to shock j at horizon h
```

Each row sums to 1 (100% of forecast error variance accounted for).

### Stability Analysis

Check whether all eigenvalues of the companion matrix lie inside the unit circle:

```julia
stab = pvar_stability(model)
stab.is_stable      # true if all |λ| < 1
stab.moduli          # moduli of eigenvalues
```

---

## Bootstrap Confidence Intervals

Group-level block bootstrap preserves the within-group time structure:

```julia
boot = pvar_bootstrap_irf(model, 20;
    irf_type=:oirf,   # or :girf
    n_draws=500,
    ci=0.95
)
# boot.lower[h+1], boot.upper[h+1] are m × m CI bounds
```

For each bootstrap draw, ``N`` groups are resampled with replacement, the PVAR is re-estimated, and IRFs are computed. Quantile-based confidence intervals are constructed from the bootstrap distribution.

---

## Specification Tests

### Hansen J-Test

The Hansen (1982) J-test evaluates whether the overidentifying restrictions (moment conditions) are valid:

```julia
j = pvar_hansen_j(model)
j.statistic     # J-statistic
j.pvalue        # p-value (χ² distribution)
j.df            # degrees of freedom = instruments - parameters
```

Under ``H_0``: all moment conditions are valid. Rejection suggests instrument invalidity or model misspecification.

!!! warning "J-Test and Instrument Count"
    The J-test has low power when the number of instruments is large relative to ``N``. A non-rejection does not necessarily validate the instruments.

### Andrews-Lu MMSC

Andrews-Lu (2001) Model and Moment Selection Criteria extend information criteria to GMM settings:

```julia
mmsc = pvar_mmsc(model)
mmsc.bic     # MMSC-BIC
mmsc.aic     # MMSC-AIC
mmsc.hqic    # MMSC-HQIC
```

```math
\text{MMSC-BIC} = J - (c - b) \ln(n), \quad
\text{MMSC-AIC} = J - 2(c - b)
```

where ``c`` = number of instruments, ``b`` = number of parameters, ``n`` = observations. Lower values are preferred.

---

## Lag Selection

Select the optimal lag order by comparing MMSC criteria across candidate models:

```julia
sel = pvar_lag_selection(pd, 4; dependent_vars=["rgdpna", "emp", "hc"])
sel.best_bic    # optimal lag by BIC
sel.best_aic    # optimal lag by AIC
sel.best_hqic   # optimal lag by HQIC
sel.table       # comparison table
```

---

## Complete Example

```julia
using MacroEconometricModels

# Load Penn World Table and convert to growth rates
pwt = load_example(:pwt)
pd = apply_tcode(pwt, 5)  # log first difference → growth rates
dep_vars = ["rgdpna", "emp", "hc"]

# Estimate via two-step FD-GMM: cross-country growth spillovers
model = estimate_pvar(pd, 1; dependent_vars=dep_vars, steps=:twostep)

# Specification tests
j = pvar_hansen_j(model)
println("Hansen J: stat=$(round(j.statistic, digits=3)), p=$(round(j.pvalue, digits=3))")

# Stability
stab = pvar_stability(model)
println("Stable: $(stab.is_stable)")

# Structural analysis: orthogonalized IRFs for GDP → employment → human capital
irfs = pvar_oirf(model, 10)
decomp = pvar_fevd(model, 10)

# Bootstrap confidence intervals
using Random; Random.seed!(42)
boot = pvar_bootstrap_irf(model, 10; n_draws=200, ci=0.90)

# Lag selection
sel = pvar_lag_selection(pd, 3; dependent_vars=dep_vars)
println("Best lag (BIC): $(sel.best_bic)")

# References
refs(model)
```

---

## References

- Arellano, Manuel, and Stephen Bond. 1991. "Some Tests of Specification for Panel Data." *Review of Economic Studies* 58 (2): 277--297. [https://doi.org/10.2307/2297968](https://doi.org/10.2307/2297968)
- Andrews, Donald W. K., and Biao Lu. 2001. "Consistent Model and Moment Selection Procedures for GMM Estimation." *Journal of Econometrics* 101 (1): 123--164. [https://doi.org/10.1016/S0304-4076(00)00077-4](https://doi.org/10.1016/S0304-4076(00)00077-4)
- Blundell, Richard, and Stephen Bond. 1998. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics* 87 (1): 115--143. [https://doi.org/10.1016/S0304-4076(98)00009-8](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029--1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Holtz-Eakin, Douglas, Whitney Newey, and Harvey S. Rosen. 1988. "Estimating Vector Autoregressions with Panel Data." *Econometrica* 56 (6): 1371--1395. [https://doi.org/10.2307/1913103](https://doi.org/10.2307/1913103)
- Pesaran, M. Hashem, and Yongcheol Shin. 1998. "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters* 58 (1): 17--29. [https://doi.org/10.1016/S0165-1765(97)00214-0](https://doi.org/10.1016/S0165-1765(97)00214-0)
- Windmeijer, Frank. 2005. "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics* 126 (1): 25--51. [https://doi.org/10.1016/j.jeconom.2004.02.005](https://doi.org/10.1016/j.jeconom.2004.02.005)
