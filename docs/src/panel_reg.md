# [Panel Regression](@id panel_reg_page)

**MacroEconometricModels.jl** provides a comprehensive panel regression module following Stata's `xtreg`/`xtiverg`/`xtlogit`/`xtprobit` conventions. The module covers linear panel models, panel instrumental variables, panel discrete choice, and six specification tests with four covariance estimators.

- **Linear panel** (`estimate_xtreg`): Fixed Effects, Random Effects (Swamy-Arora), First-Difference, Between, Correlated Random Effects (Mundlak 1978), Arellano-Bond, Blundell-Bond
- **Panel IV** (`estimate_xtiv`): FE-IV, RE-IV/EC2SLS (Baltagi 1981), FD-IV, Hausman-Taylor (1981)
- **Panel logit** (`estimate_xtlogit`): Pooled, FE conditional (Chamberlain 1980), RE (Gauss-Hermite quadrature), CRE
- **Panel probit** (`estimate_xtprobit`): Pooled, RE, CRE (no FE — incidental parameters problem)
- **Panel marginal effects**: AME with delta-method SEs for panel logit/probit
- **Specification tests**: Hausman, Breusch-Pagan LM, F-test for FE, Pesaran CD, Wooldridge AR, Modified Wald
- **Covariance estimators**: Entity-cluster (Arellano 1987), time-cluster, two-way cluster (Cameron-Gelbach-Miller 2011), Driscoll-Kraay (1998) HAC

```@setup preg
using MacroEconometricModels, Random, DataFrames
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
# Synthetic panel with endogenous regressor and instrument
N, T_p = 50, 20
n = N * T_p
df_iv = DataFrame(id=repeat(1:N, inner=T_p), t=repeat(1:T_p, N),
                  x=randn(n), z=randn(n))
alpha_i = repeat(randn(N), inner=T_p)
df_iv.x_endog = 0.5 .* df_iv.z .+ randn(n)
df_iv.wage = alpha_i .+ 1.5 .* df_iv.x .+ 2.0 .* df_iv.x_endog .+ randn(n)
pd_iv = xtset(df_iv, :id, :t)
m_iv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z])
report(m_iv)
```

**Recipe 5: Panel logit — democracy and development**

```@example preg
# DDCG: democracy (0/1) on log GDP — Lipset modernization hypothesis
m_logit = estimate_xtlogit(pd_ddcg, :dem, [:y]; model=:re)
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

The within R-squared measures how well human capital and capital deepening explain within-country GDP growth after removing country fixed effects. A high rho (fraction of variance due to ``\alpha_i``) indicates persistent cross-country differences dominate total variation.

**Two-way fixed effects** absorb both entity and time effects:

```@example preg
m_twoway = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; twoway=true)
report(m_twoway)
```

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

RE is efficient under the assumption ``E[\alpha_i \mid x_{it}] = 0``. The Hausman test evaluates this assumption.

### First-Difference

The **first-difference** estimator removes entity effects by differencing consecutive observations:

```math
\Delta y_{it} = \Delta x_{it}' \beta + \Delta e_{it}
```

```@example preg
m_fd = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:fd)
report(m_fd)
```

FD is consistent under the same assumptions as FE but may be more efficient when errors follow a random walk.

### Between Estimator

The **between estimator** regresses entity means on entity-mean regressors:

```math
\bar{y}_i = \bar{x}_i' \beta + \bar{\alpha}_i + \bar{e}_i
```

```@example preg
m_be = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:between)
report(m_be)
```

The between estimator exploits cross-country variation. Countries with higher average human capital and capital per worker have higher average GDP per capita.

### Correlated Random Effects (Mundlak)

The **CRE** approach (Mundlak 1978) relaxes the RE exogeneity assumption by augmenting the RE model with group means of time-varying regressors:

```math
y_{it} = x_{it}' \beta + \bar{x}_i' \gamma + \alpha_i + e_{it}
```

```@example preg
m_cre = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:cre)
report(m_cre)
```

The slope estimates ``\hat{\beta}`` from CRE are numerically equivalent to FE. The ``\hat{\gamma}`` coefficients on the group means test whether the RE assumption holds — significant ``\gamma`` indicates correlation between regressors and entity effects.

### Arellano-Bond and Blundell-Bond

**Dynamic panel** estimators handle lagged dependent variables using GMM with internal instruments.

**Arellano-Bond** (1991) differences the equation and uses lagged levels as instruments:

```@example preg
# Synthetic dynamic panel (lagged DV requires careful construction)
N_d, T_d = 50, 20
n_d = N_d * T_d
df_dyn = DataFrame(id=repeat(1:N_d, inner=T_d), t=repeat(1:T_d, N_d),
                   invest=randn(n_d), output=randn(n_d))
alpha_d = repeat(randn(N_d), inner=T_d)
df_dyn.growth = alpha_d .+ 0.8 .* df_dyn.invest .- 0.3 .* df_dyn.output .+ randn(n_d)
# Create lagged variable
df_dyn.growth_lag = vcat(missing, df_dyn.growth[1:end-1])
for i in 1:N_d; df_dyn.growth_lag[(i-1)*T_d + 1] = missing; end
df_dyn = dropmissing(df_dyn)
pd_dyn = xtset(df_dyn, :id, :t)
m_ab = estimate_xtreg(pd_dyn, :growth, [:invest, :output]; model=:ab)
report(m_ab)
```

**Blundell-Bond** (1998) adds level equations with lagged differences as instruments, improving efficiency when the autoregressive parameter is close to unity:

```@example preg
m_bb = estimate_xtreg(pd_dyn, :growth, [:invest, :output]; model=:bb)
report(m_bb)
```

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | `Symbol` | `:fe` | Estimator: `:fe`, `:re`, `:fd`, `:between`, `:cre`, `:ab`, `:bb` |
| `twoway` | `Bool` | `false` | Include time fixed effects (FE only) |
| `cov_type` | `Symbol` | `:cluster` | Covariance: `:ols`, `:cluster`, `:twoway`, `:driscoll_kraay` |
| `bandwidth` | `Int` | auto | Driscoll-Kraay bandwidth (auto = Newey-West optimal) |

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
| `theta` | `T` | Quasi-demeaning parameter (RE only) |
| `group_effects` | `Vector{T}` | Estimated entity effects (FE only) |
| `method` | `Symbol` | Estimation method used |

---

## Panel Instrumental Variables

The `estimate_xtiv` function handles endogeneity in panel data through four IV strategies.

### FE-IV

Within-transform all variables, then apply 2SLS on the demeaned data:

```@example preg
m_feiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z], model=:fe)
report(m_feiv)
```

### RE-IV (EC2SLS)

The Baltagi (1981) EC2SLS estimator quasi-demeans all variables, then uses ``[\tilde{Z}, \bar{Z}_i]`` as instruments:

```@example preg
m_reiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z], model=:re)
report(m_reiv)
```

### FD-IV

First-difference all variables, then apply 2SLS:

```@example preg
m_fdiv = estimate_xtiv(pd_iv, :wage, [:x], [:x_endog]; instruments=[:z], model=:fd)
report(m_fdiv)
```

### Hausman-Taylor

The **Hausman-Taylor** (1981) estimator handles endogenous time-invariant regressors by using within-deviations of time-varying exogenous variables as instruments:

```@example preg
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

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `instruments` | `Vector{Symbol}` | `Symbol[]` | Excluded instruments |
| `model` | `Symbol` | `:fe` | Method: `:fe`, `:re`, `:fd`, `:hausman_taylor` |
| `cov_type` | `Symbol` | `:cluster` | Covariance estimator |
| `time_invariant_exog` | `Vector{Symbol}` | `Symbol[]` | Time-invariant exogenous (HT only) |
| `time_invariant_endog` | `Vector{Symbol}` | `Symbol[]` | Time-invariant endogenous (HT only) |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Vector{T}` | Estimated coefficients |
| `first_stage_f` | `T` | Minimum first-stage F-statistic |
| `sargan_stat` | `T` | Sargan overidentification statistic |
| `sargan_pval` | `T` | Sargan test p-value |
| `method` | `Symbol` | IV method used |

---

## Panel Discrete Choice

### Panel Logit

The `estimate_xtlogit` function estimates panel logistic regression with four methods:

- **Pooled**: Standard logit ignoring panel structure (with optional cluster-robust SEs)
- **FE (conditional)**: Chamberlain (1980) conditional likelihood — eliminates entity effects by conditioning on sufficient statistics. Only within-entity variation identifies coefficients.
- **RE**: Gauss-Hermite quadrature integration over the random effect distribution
- **CRE**: Mundlak-style augmentation of the RE model with group means

```@example preg
# DDCG: democracy transition on log GDP (Lipset hypothesis)
m_pooled = estimate_xtlogit(pd_ddcg, :dem, [:y])
report(m_pooled)
```

```@example preg
# FE conditional logit — within-country variation only
m_fe_logit = estimate_xtlogit(pd_ddcg, :dem, [:y]; model=:fe)
report(m_fe_logit)
```

```@example preg
# RE logit — integrates over country-level heterogeneity
m_re_logit = estimate_xtlogit(pd_ddcg, :dem, [:y]; model=:re)
report(m_re_logit)
```

The RE logit coefficient on log GDP captures the effect of economic development on the probability of democracy, integrating over unobserved country-level heterogeneity. Positive and significant coefficients support the Lipset (1959) modernization hypothesis.

!!! warning "FE logit drops non-varying groups"
    The conditional logit estimator drops entities with no within-group variation in the outcome (all 0 or all 1). This is not a bug — these entities contribute no information to the conditional likelihood.

### Panel Probit

The `estimate_xtprobit` function supports pooled, RE, and CRE models. **No FE probit** is available because there is no conditioning trick analogous to the logit case — the incidental parameters problem biases FE probit coefficients (Wooldridge 2010, §15.8).

```@example preg
m_probit = estimate_xtprobit(pd_ddcg, :dem, [:y]; model=:re)
report(m_probit)
```

### Panel Marginal Effects

`marginal_effects` computes AMEs with delta-method standard errors for panel logit and probit models. For RE and CRE models, the marginal effects integrate over the random effect distribution using Gauss-Hermite quadrature.

```@example preg
me = marginal_effects(m_re_logit)
report(me)
```

### Keywords (Logit/Probit)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | `Symbol` | `:pooled` | Method: `:pooled`, `:fe`, `:re`, `:cre` (logit); `:pooled`, `:re`, `:cre` (probit) |
| `cov_type` | `Symbol` | `:cluster` | Covariance estimator: `:ols`, `:cluster` |
| `maxiter` | `Int` | `200` | Maximum iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |
| `n_quadrature` | `Int` | `12` | Gauss-Hermite quadrature points (RE/CRE) |

---

## Specification Tests

Six specification tests help choose between estimators and diagnose violations of model assumptions.

### Hausman Test (FE vs RE)

Tests whether the RE assumption ``E[\alpha_i \mid x_{it}] = 0`` holds (Hausman 1978). Rejection favors FE.

```@example preg
m_fe2 = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk])
m_re2 = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; model=:re)
ht = hausman_test(m_fe2, m_re2)
report(ht)
```

### Breusch-Pagan LM Test

Tests for the presence of random effects: ``H_0: \sigma_u^2 = 0`` (Breusch & Pagan 1980). Rejection suggests pooled OLS is inefficient and RE or FE is preferred.

```@example preg
bp = breusch_pagan_test(m_re2)
report(bp)
```

### F-Test for Fixed Effects

Tests joint significance of all entity fixed effects: ``H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_N``.

```@example preg
ft = f_test_fe(m_fe2)
report(ft)
```

### Pesaran CD Test

Tests for cross-sectional dependence in panel residuals (Pesaran 2004). Under ``H_0``, residuals are uncorrelated across entities.

```@example preg
cd = pesaran_cd_test(m_fe2)
report(cd)
```

### Wooldridge AR Test

Tests for first-order serial correlation in first-differenced residuals (Wooldridge 2010). Under ``H_0``, no serial correlation.

```@example preg
ar = wooldridge_ar_test(m_fe2)
report(ar)
```

### Modified Wald Test

Tests for groupwise heteroskedasticity: ``H_0: \sigma_i^2 = \sigma^2`` for all ``i`` (Greene 2012). Rejection suggests entity-specific error variances.

```@example preg
mw = modified_wald_test(m_fe2)
report(mw)
```

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

All linear panel models support four covariance estimators via the `cov_type` keyword:

| `cov_type` | Formula | When to use |
|------------|---------|-------------|
| `:ols` | ``\hat{\sigma}^2 (X'X)^{-1}`` | Homoskedastic, no correlation |
| `:cluster` | Entity-cluster robust (Arellano 1987) | Default, heteroskedasticity + within-entity correlation |
| `:twoway` | Two-way cluster (Cameron et al. 2011) | Cross-sectional + serial dependence |
| `:driscoll_kraay` | HAC across both dimensions (Driscoll & Kraay 1998) | Large T, spatial dependence |

```@example preg
# Driscoll-Kraay standard errors for PWT growth regression
m_dk = estimate_xtreg(pd_pwt, :lngdppc, [:hc, :lnk]; cov_type=:driscoll_kraay)
report(m_dk)
```

---

## Complete Example

A full panel analysis workflow using the Penn World Table:

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

The CRE group-mean coefficients test the RE exogeneity assumption. Significant group means indicate that countries with higher average human capital or capital intensity differ systematically in ways correlated with the entity effects — favoring FE over RE.

---

## Common Pitfalls

1. **Forgetting `xtset`.** All panel estimators require `PanelData` created via `xtset(df, :group, :time)`. Passing a raw DataFrame throws an error.

2. **Including time-invariant regressors in FE.** The within-transformation eliminates all time-invariant variables. Use RE, CRE, or Hausman-Taylor to estimate their effects.

3. **Using FE probit.** There is no conditioning trick for probit (unlike logit). The package correctly excludes `:fe` from `estimate_xtprobit` — use `:re` or `:cre` instead.

4. **Weak instruments in panel IV.** Check `first_stage_f` in the `PanelIVModel` output. Values below 10 indicate weak instruments (Staiger & Stock 1997).

5. **Ignoring cross-sectional dependence.** Standard cluster-robust SEs assume independence across entities. Run `pesaran_cd_test` and switch to `cov_type=:driscoll_kraay` if rejected.

6. **Unbalanced panels with AB/BB.** Arellano-Bond and Blundell-Bond GMM require sufficient time periods per entity for the instrument matrix. Very short panels may produce singular moment conditions.

---

## References

- Arellano, M. (1987). Computing Robust Standard Errors for Within-Groups Estimators. *Oxford Bulletin of Economics and Statistics* 49(4), 431-434. [DOI](https://doi.org/10.1111/j.1468-0084.1987.mp49004006.x)
- Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations. *Review of Economic Studies* 58(2), 277-297. [DOI](https://doi.org/10.2307/2297968)
- Baltagi, B. H. (1981). Simultaneous Equations with Error Components. *Journal of Econometrics* 17(2), 189-200. [DOI](https://doi.org/10.1016/0304-4076(81)90026-9)
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. 6th ed. Springer. ISBN 978-3-030-53952-8.
- Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions in Dynamic Panel Data Models. *Journal of Econometrics* 87(1), 115-143. [DOI](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Breusch, T. S. & Pagan, A. R. (1980). The Lagrange Multiplier Test and Its Applications to Model Specification in Econometrics. *Review of Economic Studies* 47(1), 239-253. [DOI](https://doi.org/10.2307/2297111)
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). Robust Inference with Multiway Clustering. *Journal of Business & Economic Statistics* 29(2), 238-249. [DOI](https://doi.org/10.1198/jbes.2010.07136)
- Cameron, A. C. & Miller, D. L. (2015). A Practitioner's Guide to Cluster-Robust Inference. *Journal of Human Resources* 50(2), 317-372. [DOI](https://doi.org/10.3368/jhr.50.2.317)
- Chamberlain, G. (1980). Analysis of Covariance with Qualitative Data. *Review of Economic Studies* 47(1), 225-238. [DOI](https://doi.org/10.2307/2297110)
- Driscoll, J. C. & Kraay, A. C. (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data. *Review of Economics and Statistics* 80(4), 549-560. [DOI](https://doi.org/10.1162/003465398557825)
- Feenstra, R. C., Inklaar, R. & Timmer, M. P. (2015). The Next Generation of the Penn World Table. *American Economic Review* 105(10), 3150-3182. [DOI](https://doi.org/10.1257/aer.20130954)
- Greene, W. H. (2012). *Econometric Analysis*. 7th ed. Prentice Hall. ISBN 978-0-131-39538-1.
- Hausman, J. A. (1978). Specification Tests in Econometrics. *Econometrica* 46(6), 1251-1271. [DOI](https://doi.org/10.2307/1913827)
- Hausman, J. A. & Taylor, W. E. (1981). Panel Data and Unobservable Individual Effects. *Econometrica* 49(6), 1377-1398. [DOI](https://doi.org/10.2307/1911406)
- Lipset, S. M. (1959). Some Social Requisites of Democracy: Economic Development and Political Legitimacy. *American Political Science Review* 53(1), 69-105. [DOI](https://doi.org/10.2307/1951731)
- Mundlak, Y. (1978). On the Pooling of Time Series and Cross Section Data. *Econometrica* 46(1), 69-85. [DOI](https://doi.org/10.2307/1913646)
- Pesaran, M. H. (2004). General Diagnostic Tests for Cross Section Dependence in Panels. CESifo Working Paper No. 1229. [DOI](https://doi.org/10.2139/ssrn.572504)
- Staiger, D. & Stock, J. H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica* 65(3), 557-586. [DOI](https://doi.org/10.2307/2171753)
- Swamy, P. A. V. B. & Arora, S. S. (1972). The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models. *Econometrica* 40(2), 261-275. [DOI](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. ISBN 978-0-262-23258-6.
