# Design: Panel Regression (#66) + Ordered/Multinomial Models (#77)

**Date:** 2026-03-07
**Issues:** #66 (Panel regression — FE/RE, panel IV, panel Probit/Logit), #77 (Ordered/Multinomial logit/probit)
**Status:** Approved

---

## 1. Ordered and Multinomial Models (#77)

### New Files

- `src/reg/ordered.jl` — `OrderedLogitModel{T}`, `OrderedProbitModel{T}`, `estimate_ologit`, `estimate_oprobit`
- `src/reg/multinomial.jl` — `MultinomialLogitModel{T}`, `estimate_mlogit`

### Types

```julia
struct OrderedLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}              # outcome categories (1, 2, ..., J)
    X::Matrix{T}                # regressors (no intercept — absorbed into cutpoints)
    beta::Vector{T}             # slope coefficients (K)
    cutpoints::Vector{T}        # J-1 threshold parameters (α₁ < α₂ < ... < αⱼ₋₁)
    vcov_mat::Matrix{T}         # (K + J-1) × (K + J-1) joint vcov
    loglik::T
    loglik_null::T
    pseudo_r2::T                # McFadden
    aic::T
    bic::T
    n_categories::Int
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end
# OrderedProbitModel{T} — identical structure, Φ link instead of Λ

struct MultinomialLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{Int}              # outcome categories (1, ..., J)
    X::Matrix{T}                # regressors (includes intercept)
    beta::Matrix{T}             # K × (J-1) coefficients (base category = 0)
    vcov_mat::Matrix{T}         # K(J-1) × K(J-1) joint vcov
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    n_categories::Int
    base_category::Int
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end
```

### Estimation

- **Ordered Logit/Probit:** Newton-Raphson MLE on cumulative link model.
  `P(y ≤ j | x) = F(αⱼ - x'β)` where F is logistic (logit) or Φ (probit).
  Parameterize cutpoints to enforce ordering: `α₁ < α₂ < ... < αⱼ₋₁`.
- **Multinomial Logit:** Newton-Raphson MLE on softmax.
  `P(y = j | x) = exp(x'βⱼ) / Σₖ exp(x'βₖ)` with β_base = 0.
- Robust/cluster SEs via sandwich estimator (reuse `_reg_vcov` pattern).

### Post-Estimation

- `marginal_effects(m::OrderedLogitModel; type=:ame)` — per-category AME matrix (K × J)
- `marginal_effects(m::MultinomialLogitModel; type=:ame)` — per-alternative AME matrix (K × J)
- `predict(m, X_new)` — n × J matrix of predicted probabilities
- `brant_test(m::OrderedLogitModel)` — parallel regression assumption test
- `hausman_iia(m::MultinomialLogitModel)` — IIA test (Hausman-McFadden 1984)

### Display

- `report(m::OrderedLogitModel)` — coefficient table + cutpoint table
- `report(m::MultinomialLogitModel)` — per-alternative coefficient blocks
- `refs()` dispatches return McFadden (1974), Brant (1990), etc.

---

## 2. Panel Regression (#66)

### New Directory: `src/panel_reg/`

### Files

- `types.jl` — `PanelRegModel{T}`, `PanelIVModel{T}`, `PanelLogitModel{T}`, `PanelProbitModel{T}`, `PanelTestResult{T}`
- `estimation.jl` — `estimate_xtreg()` (FE/RE/FD/Between/CRE)
- `iv.jl` — `estimate_xtiv()` (FE-IV, RE-IV, FD-IV, Hausman-Taylor)
- `logit.jl` — `estimate_xtlogit()` (pooled/FE conditional/RE/CRE)
- `probit.jl` — `estimate_xtprobit()` (pooled/RE/CRE)
- `tests.jl` — specification tests
- `covariance.jl` — panel cluster SEs, Driscoll-Kraay
- `margins.jl` — panel-aware marginal effects
- `predict.jl` — within/between/overall predictions
- `display.jl` — `report()`, `refs()` dispatches

### Key Types

```julia
struct PanelRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T                             # between-group std dev
    sigma_e::T                             # within-group std dev
    rho::T                                 # σ²ᵤ / (σ²ᵤ + σ²ₑ)
    theta::Union{Nothing,T}                # quasi-demeaning parameter (RE)
    f_stat::T
    f_pval::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    method::Symbol                         # :fe, :re, :fd, :between, :cre
    twoway::Bool
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    n_periods_avg::T
    group_effects::Union{Nothing,Vector{T}}  # estimated αᵢ (FE only)
    data::PanelData{T}
end

struct PanelIVModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}                           # instrument matrix
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T
    sigma_e::T
    rho::T
    first_stage_f::T
    sargan_stat::Union{Nothing,T}
    sargan_pval::Union{Nothing,T}
    varnames::Vector{String}
    endog_names::Vector{String}
    instrument_names::Vector{String}
    method::Symbol                         # :fe_iv, :re_iv, :fd_iv, :hausman_taylor
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

struct PanelLogitModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::Vector{T}
    vcov_mat::Matrix{T}
    y::Vector{T}
    X::Matrix{T}
    fitted::Vector{T}                      # predicted probabilities
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    sigma_u::Union{Nothing,T}             # RE std dev (nothing for FE/pooled)
    rho::Union{Nothing,T}
    varnames::Vector{String}
    method::Symbol                         # :pooled, :fe, :re, :cre
    cov_type::Symbol
    converged::Bool
    iterations::Int
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

# PanelProbitModel{T} — identical to PanelLogitModel{T}, Φ link

struct PanelTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    test_name::String
    statistic::T
    pvalue::T
    df::Union{Int,Tuple{Int,Int}}          # chi² df or (df1, df2) for F
    description::String
end
```

### Phase A: Linear Panel + Tests + Display

**Estimators:**
- **FE (within):** Demean y and X by group (and time if `twoway=true`), OLS on demeaned data. Entity effects recovered as ᾱᵢ = ȳᵢ - x̄ᵢ'β.
- **RE (GLS):** Swamy-Arora variance components: σ²ₑ from FE residuals, σ²ᵤ from between-within comparison. θ = 1 - √(σ²ₑ / (Tᵢσ²ᵤ + σ²ₑ)). Quasi-demean and OLS.
- **FD:** First-difference y and X within groups, OLS on differences.
- **Between:** Group-mean y and X, OLS on group means.
- **CRE (Mundlak):** Add group means of time-varying regressors to RE specification.

**Specification tests:**
- `hausman_test(fe, re)` — H₀: RE consistent. χ² = (β_FE - β_RE)' (V_FE - V_RE)⁻¹ (β_FE - β_RE).
- `breusch_pagan_test(re)` — H₀: σ²ᵤ = 0 (pooled OLS adequate). LM = nT/2(T-1) [Σᵢ(ΣₜêᵢₜX)² / ΣᵢΣₜêᵢₜ² - 1]².
- `f_test_fe(fe)` — H₀: all αᵢ = 0. F-test on joint significance of entity dummies.
- `pesaran_cd_test(m)` — Cross-sectional dependence. CD = √(2T/N(N-1)) Σᵢ<ⱼ ρ̂ᵢⱼ.
- `wooldridge_ar_test(fe)` — H₀: no AR(1) in FE errors. Regression of FD residuals on lagged FD residuals.
- `modified_wald_test(fe)` — H₀: σ²ᵢ = σ² for all i. groupwise heteroskedasticity.

**Covariance:**
- Entity cluster: Arellano (1987), G/(G-1) correction
- Time cluster: analogous
- Two-way: Cameron-Gelbach-Miller (2011) — V_entity + V_time - V_entity×time
- Driscoll-Kraay: Newey-West on cross-sectional averages of moment conditions

**API:**
```julia
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe, twoway=true)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fd)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:between)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:cre)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe, cov_type=:cluster)  # entity cluster (default)
m = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe, cov_type=:twoway)

hausman_test(fe_model, re_model)
breusch_pagan_test(re_model)
pesaran_cd_test(fe_model)
wooldridge_ar_test(fe_model)
```

### Phase B: Panel IV

**Estimators:**
- **FE-IV:** Within-transform then 2SLS on demeaned data.
- **RE-IV (EC2SLS):** Baltagi (1981) — use within and between instruments.
- **FD-IV:** First-difference then 2SLS (Anderson-Hsiao style for dynamic panels).
- **Hausman-Taylor:** Partition regressors into (exogenous, endogenous) × (time-varying, time-invariant). Use within residuals and group means as instruments.

**API:**
```julia
m = estimate_xtiv(pd, :y, [:x1_exog], [:x2_endog]; instruments=[:z1, :z2], model=:fe)
m = estimate_xtiv(pd, :y, [:x1_exog], [:x2_endog]; instruments=[:z1, :z2], model=:re)
m = estimate_xtiv(pd, :y, [:x1_exog], [:x2_endog]; instruments=[:z1], model=:fd)
m = estimate_xtiv(pd, :y, [:x1_tv_exog], [:x2_tv_endog];
                   time_invariant_exog=[:z1], time_invariant_endog=[:z2],
                   model=:hausman_taylor)
```

### Phase C: Panel Probit/Logit

**Estimators:**
- **Pooled:** Reuse cross-sectional MLE + entity-clustered SEs.
- **FE Logit (conditional):** Chamberlain (1980) — condition on sufficient statistic Σₜyᵢₜ. Only within-group variation identified. No incidental parameters problem.
- **RE Probit/Logit:** yᵢₜ | αᵢ ~ Bernoulli(F(x'β + αᵢ)), αᵢ ~ N(0, σ²ᵤ). Integrate out αᵢ with Gauss-Hermite quadrature (existing infrastructure).
- **CRE:** Mundlak augmentation in RE specification.

**API:**
```julia
m = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:fe)
m = estimate_xtlogit(pd, :y, [:x1, :x2]; model=:re)
m = estimate_xtprobit(pd, :y, [:x1, :x2]; model=:re)
m = estimate_xtprobit(pd, :y, [:x1, :x2]; model=:cre)
marginal_effects(m)
predict(m, pd_new)
```

### Phase D: Dynamic Panel

Thin wrappers around existing PVAR GMM infrastructure:
- `estimate_xtreg(pd, :y, [:x1]; model=:ab)` → Arellano-Bond (dispatches to `estimate_pvar` FD-GMM)
- `estimate_xtreg(pd, :y, [:x1]; model=:bb)` → Blundell-Bond (dispatches to `estimate_pvar` System GMM)
- Inherits: Windmeijer correction, Hansen J-test, AR(1)/AR(2) tests, instrument collapse

---

## 3. Testing Strategy

- `test/reg/test_ordered.jl` — ordered logit/probit estimation, marginal effects, Brant test
- `test/reg/test_multinomial.jl` — multinomial logit, marginal effects, IIA test
- `test/panel_reg/test_panel_reg.jl` — FE/RE/FD/Between/CRE estimation, R² variants
- `test/panel_reg/test_panel_tests.jl` — Hausman, BP-LM, Pesaran CD, Wooldridge, Modified Wald
- `test/panel_reg/test_panel_iv.jl` — FE-IV, RE-IV, FD-IV, Hausman-Taylor
- `test/panel_reg/test_panel_nonlinear.jl` — panel logit/probit (pooled, FE, RE, CRE)

Verify against Stata/R reference values where possible.

---

## 4. References

- Wooldridge (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
- Baltagi (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer.
- Cameron & Trivedi (2005). *Microeconometrics*. Cambridge.
- McFadden (1974). "Conditional Logit Analysis." *Frontiers in Econometrics*.
- Chamberlain (1980). "Analysis of Covariance with Qualitative Data." *RES*.
- Mundlak (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*.
- Hausman & Taylor (1981). "Panel Data and Unobservable Individual Effects." *Econometrica*.
- Cameron, Gelbach & Miller (2011). "Robust Inference with Multiway Clustering." *JBES*.
- Windmeijer (2005). "Finite Sample Correction for 2-Step GMM." *JoE*.
- Brant (1990). "Assessing Proportionality in the Proportional Odds Model." *Biometrics*.
- Hausman & McFadden (1984). "Specification Tests for the Multinomial Logit Model." *Econometrica*.
- Driscoll & Kraay (1998). "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data." *RESTAT*.
- Swamy & Arora (1972). "The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models." *Econometrica*.
- Butler & Moffitt (1982). "A Computationally Efficient Quadrature Procedure for the One-Factor Multinomial Probit Model." *Econometrica*.
