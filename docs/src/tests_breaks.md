# [Structural Breaks](@id tests_breaks_page)

This page covers tests for parameter instability and structural change in time series models. Structural breaks --- abrupt shifts in regression coefficients, factor loadings, or the number of latent factors --- invalidate standard inference and produce misleading forecasts if left undetected. Four complementary frameworks are provided:

- **Andrews (1993)**: Tests for a single unknown break point in a linear regression. Nine test variants combine three base statistics (Wald, LR, LM) with three functionals (supremum, exponential average, mean).
- **Bai-Perron (1998, 2003)**: Tests for multiple unknown break points. Dynamic programming finds globally optimal break dates, with sequential testing and information criteria (BIC, LWZ) for break number selection.
- **Factor model break tests**: Three methods for detecting instability in factor loadings or the number of factors --- Breitung-Eickmeier (2011) CUSUM, Chen-Dolado-Gonzalo (2014) eigenvalue ratio, and Han-Inoue (2015) sup-Wald.
- **Gregory-Hansen (1996)**: Tests for cointegration allowing a single structural break in the cointegrating relationship. Three models (level shift, level + trend, regime shift) and three test statistics (ADF*, Zt*, Za*).

## Quick Start

**Recipe 1: Andrews SupF test on a regression**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Linear regression with a break in the slope at t=100
X = hcat(ones(200), randn(200))
y = X * [1.0, 2.0] + randn(200) * 0.5
y[101:end] .+= X[101:end, 2] .* 3.0  # slope shifts from 2.0 to 5.0

result = andrews_test(y, X; test=:supwald)
report(result)
```

**Recipe 2: Bai-Perron multiple break detection**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Three regimes with different intercepts
T_len = 300
X = ones(T_len, 1)
y = vcat(ones(100) * 2.0, ones(100) * 5.0, ones(100) * 1.0) + randn(T_len) * 0.5

result = bai_perron_test(y, X; max_breaks=5)
report(result)
```

**Recipe 3: Factor break test**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Panel with 50 variables and 3 factors
X = randn(200, 50)
result = factor_break_test(X, 3; method=:breitung_eickmeier)
report(result)
```

**Recipe 4: Gregory-Hansen cointegration test with regime shift**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Two cointegrated series with a regime shift at t=100
T_len = 200
x = cumsum(randn(T_len))
y = vcat(1.0 .+ 0.8 .* x[1:100], 3.0 .+ 1.5 .* x[101:end]) + randn(T_len) * 0.5

Y = hcat(y, x)
result = gregory_hansen_test(Y; model=:CS)
report(result)
```

---

## Andrews Structural Break Test

The Andrews (1993) test detects a single structural break at an unknown date in a linear regression model. The null hypothesis is parameter constancy: all regression coefficients remain stable over the entire sample. The test searches over a trimmed range of candidate break dates and selects the date producing the most extreme test statistic.

Consider the linear model:

```math
y_t = X_t' \beta + u_t, \quad t = 1, \ldots, T
```

where:
- ``y_t`` is the dependent variable at time ``t``
- ``X_t`` is the ``k \times 1`` vector of regressors
- ``\beta`` is the ``k \times 1`` parameter vector
- ``u_t`` is the error term

Under the alternative hypothesis, the parameter vector shifts at some unknown break date ``t_b``:

```math
\beta = \begin{cases} \beta_1 & t \leq t_b \\ \beta_2 & t > t_b \end{cases}
```

The **Wald statistic** at a candidate break date ``t_b`` compares sub-sample estimates:

```math
W(t_b) = (\hat{\beta}_1 - \hat{\beta}_2)' \left[ V_1 + V_2 \right]^{-1} (\hat{\beta}_1 - \hat{\beta}_2)
```

where:
- ``\hat{\beta}_1`` and ``\hat{\beta}_2`` are OLS estimates from the two sub-samples
- ``V_1 = \hat{\sigma}_1^2 (X_1' X_1)^{-1}`` and ``V_2 = \hat{\sigma}_2^2 (X_2' X_2)^{-1}`` are their covariance matrices

The **LR statistic** compares the full-sample and split-sample residual sums of squares:

```math
\text{LR}(t_b) = T \left[ \ln(\text{SSR}_0 / T) - \ln(\text{SSR}_{split} / T) \right]
```

The **LM statistic** is a score-based test requiring only the full-sample estimates:

```math
\text{LM}(t_b) = \frac{S(t_b)' (X'X)^{-1} S(t_b)}{\hat{\sigma}^2 \cdot \tau (1 - \tau)}
```

where ``S(t_b) = \sum_{t=1}^{t_b} X_t \hat{u}_t`` is the partial sum of scores and ``\tau = t_b / T``.

Three **functionals** aggregate the base statistic over the trimmed range ``[\pi T, (1 - \pi) T]``:

| Variant | Functional | Base Statistic | Reference |
|---------|-----------|----------------|-----------|
| `:supwald` | Supremum | Wald | Andrews (1993) |
| `:suplr` | Supremum | LR | Andrews (1993) |
| `:suplm` | Supremum | LM | Andrews (1993) |
| `:expwald` | Exponential | Wald | Andrews-Ploberger (1994) |
| `:explr` | Exponential | LR | Andrews-Ploberger (1994) |
| `:explm` | Exponential | LM | Andrews-Ploberger (1994) |
| `:meanwald` | Mean | Wald | Andrews-Ploberger (1994) |
| `:meanlr` | Mean | LR | Andrews-Ploberger (1994) |
| `:meanlm` | Mean | LM | Andrews-Ploberger (1994) |

The **supremum** functional takes the maximum statistic over all candidate dates and has optimal power against a single large break. The **exponential** functional computes ``\log(\text{mean}(\exp(W/2)))`` and the **mean** functional computes ``\text{mean}(W)``; both have superior power against small or gradual parameter changes (Andrews & Ploberger 1994).

!!! note "Technical Note"
    P-values are computed by interpolation from Hansen (1997) critical value tables, which are tabulated by the number of parameters ``k`` and the functional type. Critical values depend on the trimming fraction ``\pi`` and are pre-computed for the standard ``\pi = 0.15`` setting. The asymptotic distribution under the null is a functional of a ``k``-dimensional Brownian bridge process.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data with a break in slope at t=100
T_len = 200
X = hcat(ones(T_len), randn(T_len))
beta_pre = [1.0, 2.0]
beta_post = [1.0, 5.0]
y = vcat(X[1:100, :] * beta_pre, X[101:end, :] * beta_post) + randn(T_len) * 0.5

# Sup-Wald test (highest power against a single sharp break)
result_sup = andrews_test(y, X; test=:supwald)
report(result_sup)

# Exp-Wald test (power against gradual changes)
result_exp = andrews_test(y, X; test=:expwald)
report(result_exp)
```

The sup-Wald test rejects the null of parameter constancy when the p-value falls below the chosen significance level (typically 0.05). The `break_index` field reports the estimated break location --- the observation at which the test statistic achieves its supremum. Comparing across the three functionals provides robustness: rejection by the supremum functional points to a sharp break, while rejection by the exponential or mean functional suggests a more diffuse shift.

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `test` | `Symbol` | `:supwald` | Test variant: any of the 9 combinations listed above |
| `trimming` | `Real` | `0.15` | Fraction of sample trimmed from each end |

### AndrewsResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Test statistic (functional applied to base statistic sequence) |
| `pvalue` | `T` | Approximate p-value from Hansen (1997) critical value tables |
| `break_index` | `Int` | Estimated break date (observation index of supremum) |
| `break_fraction` | `T` | Break location as fraction of sample |
| `test_type` | `Symbol` | Test variant (e.g. `:supwald`, `:explr`) |
| `critical_values` | `Dict{Int,T}` | Critical values at 1%, 5%, 10% significance levels |
| `stat_sequence` | `Vector{T}` | Full sequence of base statistics across candidate break dates |
| `trimming` | `T` | Trimming fraction used |
| `nobs` | `Int` | Number of observations |
| `n_params` | `Int` | Number of regressors (``k``) |

---

## Bai-Perron Multiple Break Test

The Bai-Perron (1998, 2003) procedure detects and dates multiple structural breaks in a linear regression model. Dynamic programming finds globally optimal break dates that minimize the total sum of squared residuals, avoiding the suboptimality of sequential single-break methods.

Consider the linear model with ``m`` breaks (``m + 1`` regimes):

```math
y_t = X_t' \beta_j + u_t, \quad t = T_{j-1} + 1, \ldots, T_j, \quad j = 1, \ldots, m + 1
```

where:
- ``T_0 = 0`` and ``T_{m+1} = T`` are the sample boundaries
- ``T_1, \ldots, T_m`` are the unknown break dates
- ``\beta_j`` is the ``k \times 1`` parameter vector for regime ``j``

The optimal break dates minimize the total SSR:

```math
(\hat{T}_1, \ldots, \hat{T}_m) = \arg\min_{T_1, \ldots, T_m} \sum_{j=1}^{m+1} \sum_{t=T_{j-1}+1}^{T_j} (y_t - X_t' \hat{\beta}_j)^2
```

The **sup-F test** for ``l`` breaks against zero breaks is:

```math
\text{sup-}F(l) = \frac{(\text{SSR}_0 - \text{SSR}_l) / (l \cdot k)}{\text{SSR}_l / (T - (l+1) \cdot k)}
```

where:
- ``\text{SSR}_0`` is the full-sample residual sum of squares (no breaks)
- ``\text{SSR}_l`` is the minimized SSR with ``l`` optimally placed breaks
- ``k`` is the number of regressors

The **sequential sup-F test** ``\text{sup-}F(l+1 | l)`` tests whether adding one more break significantly improves the fit over the ``l``-break model. Two **information criteria** provide alternative break number selection:

```math
\text{BIC}(m) = T \ln(\text{SSR}_m / T) + (m+1) k \ln T
```

```math
\text{LWZ}(m) = T \ln(\text{SSR}_m / T) + (m+1) k \cdot 0.299 (\ln T)^{2.1}
```

where:
- ``m`` is the number of breaks
- The LWZ criterion (Liu, Wu & Zidek 1997) penalizes additional breaks more heavily than BIC

!!! note "Technical Note"
    The dynamic programming algorithm has complexity ``O(T^2 \cdot m_{\max})``, where ``T`` is the sample size and ``m_{\max}`` is the maximum number of breaks. A segment SSR matrix is pre-computed and reused across all candidate break configurations. Each segment requires at least ``h = \max(k+1, \lceil \pi T \rceil)`` observations, where ``\pi`` is the trimming fraction.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data with 2 breaks: regimes at t=1-100, 101-200, 201-300
T_len = 300
X = hcat(ones(T_len), randn(T_len))

beta1 = [1.0, 2.0]
beta2 = [3.0, -1.0]
beta3 = [0.0, 4.0]

y = vcat(
    X[1:100, :] * beta1,
    X[101:200, :] * beta2,
    X[201:300, :] * beta3
) + randn(T_len) * 0.5

result = bai_perron_test(y, X; max_breaks=5, criterion=:bic)
report(result)

# Examine regime-specific coefficients
for (j, coefs) in enumerate(result.regime_coefs)
    println("Regime $j: intercept = $(round(coefs[1], digits=2)), ",
            "slope = $(round(coefs[2], digits=2))")
end

# Compare BIC and LWZ break number selection
bic_choice = argmin(result.bic_values) - 1
lwz_choice = argmin(result.lwz_values) - 1
println("BIC selects $bic_choice breaks, LWZ selects $lwz_choice breaks")
```

The `n_breaks` field reports the selected number of breaks. The `break_dates` vector gives the observation indices where breaks occur. The `regime_coefs` and `regime_ses` vectors contain coefficient estimates and standard errors for each regime. The `supf_stats` and `sequential_stats` vectors provide the test statistics for formal inference, while `bic_values` and `lwz_values` give information criteria for each candidate number of breaks (0 through `max_breaks`).

The sequential testing procedure starts from the null of zero breaks: if sup-F(1) rejects, test sup-F(2|1), and continue until the first non-rejection. BIC and LWZ provide complementary model selection. BIC tends to select more breaks in large samples, while LWZ is more conservative.

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `max_breaks` | `Int` | `5` | Maximum number of breaks to consider |
| `trimming` | `Real` | `0.15` | Minimum fraction of observations per segment |
| `criterion` | `Symbol` | `:bic` | Information criterion for break selection: `:bic` or `:lwz` |

### BaiPerronResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `n_breaks` | `Int` | Selected number of structural breaks |
| `break_dates` | `Vector{Int}` | Estimated break date indices |
| `break_cis` | `Vector{Tuple{Int,Int}}` | 95% confidence intervals for break dates |
| `regime_coefs` | `Vector{Vector{T}}` | OLS coefficient estimates for each regime |
| `regime_ses` | `Vector{Vector{T}}` | Standard errors for each regime |
| `supf_stats` | `Vector{T}` | sup-F(l) statistics for ``l = 1, \ldots, m_{\max}`` |
| `supf_pvalues` | `Vector{T}` | P-values for sup-F statistics |
| `sequential_stats` | `Vector{T}` | Sequential sup-F(l+1\|l) statistics |
| `sequential_pvalues` | `Vector{T}` | P-values for sequential statistics |
| `bic_values` | `Vector{T}` | BIC values for 0 through ``m_{\max}`` breaks |
| `lwz_values` | `Vector{T}` | LWZ values for 0 through ``m_{\max}`` breaks |
| `trimming` | `T` | Trimming fraction used |
| `nobs` | `Int` | Number of observations |

---

## Factor Model Break Tests

Factor models assume that a large cross-section of variables ``X_{it}`` loads on a small number of common factors ``F_t`` with time-invariant loadings ``\Lambda``. Structural instability in the loadings or the number of factors invalidates principal components estimation and downstream inference. Three complementary tests detect different forms of instability.

### Breitung-Eickmeier (2011)

The Breitung-Eickmeier test applies a CUSUM fluctuation principle to factor loading stability. The factor model is:

```math
X_t = \Lambda F_t + e_t
```

where:
- ``X_t`` is the ``N \times 1`` vector of observed variables at time ``t``
- ``\Lambda`` is the ``N \times r`` matrix of factor loadings
- ``F_t`` is the ``r \times 1`` vector of common factors
- ``e_t`` is the ``N \times 1`` idiosyncratic error

The test estimates loadings from the sub-sample ``[1, t]`` and compares them to the full-sample loadings ``\hat{\Lambda}_T``. The fluctuation statistic at date ``t`` is:

```math
S_t = \sqrt{t / T} \cdot \frac{\| \text{vec}(\hat{\Lambda}_t - \hat{\Lambda}_T) \|}{\hat{\sigma}}
```

where ``\hat{\sigma}^2`` is the estimated loading regression variance. The test statistic is ``\sup_t S_t`` over the trimmed range. Under the null of stable loadings, the statistic converges to the supremum of a Bessel process. Rejection indicates that the factor-variable relationships shift at some point in the sample.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Matrix dispatch
X = randn(200, 50)
result = factor_break_test(X, 3; method=:breitung_eickmeier)
report(result)

# FactorModel dispatch
fm = estimate_factors(X, 3)
result = factor_break_test(fm; method=:breitung_eickmeier)
report(result)
```

### Chen-Dolado-Gonzalo (2014)

The Chen-Dolado-Gonzalo test detects changes in the number of factors by comparing eigenvalue ratios across sub-samples. The **eigenvalue ratio** criterion identifies the number of factors as:

```math
\hat{r} = \arg\max_{k \leq k_{\max}} \frac{\lambda_k}{\lambda_{k+1}}
```

where ``\lambda_1 \geq \lambda_2 \geq \cdots`` are the ordered eigenvalues of the sample covariance matrix. The test computes eigenvalue ratios from sub-samples ``[1, t]`` and ``[t+1, T]`` for each candidate break date ``t``, then takes the supremum of the normalized maximum difference:

```math
\text{CDG}(t) = \sqrt{\frac{t(T - t)}{T}} \cdot \max_k \left| \frac{\lambda_k^{(1)}}{\lambda_{k+1}^{(1)}} - \frac{\lambda_k^{(2)}}{\lambda_{k+1}^{(2)}} \right|
```

This test does not require specifying the number of factors ``r`` in advance, making it useful as a preliminary diagnostic.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Chen-Dolado-Gonzalo does not require specifying r
X = randn(200, 50)
result = factor_break_test(X; method=:chen_dolado_gonzalo)
report(result)
```

### Han-Inoue (2015)

The Han-Inoue test aggregates individual Wald statistics for loading instability across all ``N`` cross-section units. For each variable ``i`` and candidate break date ``t``, the loading regression ``x_{it} = F_t' \lambda_i + e_{it}`` is estimated on sub-samples ``[1, t]`` and ``[t+1, T]``. The individual Wald statistic is:

```math
W_i(t) = (\hat{\lambda}_{1,i} - \hat{\lambda}_{2,i})' \left[ \hat{\sigma}_i^2 \left( (F_1'F_1)^{-1} + (F_2'F_2)^{-1} \right) \right]^{-1} (\hat{\lambda}_{1,i} - \hat{\lambda}_{2,i})
```

The test statistic aggregates across units and maximizes over break dates:

```math
\text{HI} = \sup_t \frac{1}{N} \sum_{i=1}^{N} W_i(t)
```

where:
- ``\hat{\lambda}_{1,i}`` and ``\hat{\lambda}_{2,i}`` are sub-sample loading estimates for variable ``i``
- ``\hat{\sigma}_i^2`` is the full-sample residual variance for variable ``i``
- ``F_1`` and ``F_2`` are the factor matrices for the two sub-samples

The cross-sectional averaging improves power when many loadings shift simultaneously. P-values use the Andrews (1993) sup-Wald critical values with ``k = r`` degrees of freedom.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Han-Inoue requires specifying the number of factors
X = randn(200, 50)
result = factor_break_test(X, 3; method=:han_inoue)
report(result)

# FactorModel dispatch also works
fm = estimate_factors(X, 3)
result = factor_break_test(fm; method=:han_inoue)
report(result)
```

### Method Comparison

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Loading stability check | `:breitung_eickmeier` | CUSUM with known ``r`` |
| Unknown ``r`` | `:chen_dolado_gonzalo` | Does not require ``r`` |
| Large ``N`` panels | `:han_inoue` | Cross-sectional power gains |
| Quick diagnostic | `:chen_dolado_gonzalo` | Fewest assumptions |

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `method` | `Symbol` | `:breitung_eickmeier` | Test method: `:breitung_eickmeier`, `:chen_dolado_gonzalo`, or `:han_inoue` |

!!! note "Technical Note"
    The Breitung-Eickmeier and Han-Inoue tests require the number of factors ``r`` as input. Use `ic_criteria(X, r_max)` or the Bai-Ng (2002) information criteria to determine ``r`` before applying these tests. The Chen-Dolado-Gonzalo test does not require ``r`` and automatically determines ``r_{\max} = \min(\lfloor \sqrt{\min(T, N)} \rfloor, 10)``.

### FactorBreakResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Test statistic |
| `pvalue` | `T` | Approximate p-value |
| `break_date` | `Union{Int, Nothing}` | Estimated break date (observation index), or `nothing` if insufficient data |
| `method` | `Symbol` | Test method used |
| `n_factors` | `Int` | Number of factors (or ``r_{\max}`` for Chen-Dolado-Gonzalo) |
| `nobs` | `Int` | Number of time periods (``T``) |
| `n_vars` | `Int` | Number of cross-section units (``N``) |

---

## Gregory-Hansen Cointegration Test

The Gregory-Hansen test (Gregory & Hansen 1996) extends standard residual-based cointegration tests by allowing a single structural break in the cointegrating relationship. Standard cointegration tests lose power when the equilibrium relationship shifts at an unknown date --- what appears to be no cointegration may actually be cointegration with a regime change.

The test estimates the cointegrating regression at each candidate break date and computes residual-based test statistics. The null hypothesis is no cointegration; the alternative is cointegration with a single structural break.

Three models control the form of the break:

**Level shift** (`:C`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \alpha' y_{2t} + e_t
```

**Level + trend shift** (`:CT`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \beta t + \alpha' y_{2t} + e_t
```

**Regime shift** (`:CS`):

```math
y_{1t} = \mu_1 + \mu_2 D_t + \alpha_1' y_{2t} + \alpha_2' y_{2t} D_t + e_t
```

where:
- ``y_{1t}`` is the dependent variable (first column of `Y`)
- ``y_{2t}`` is the ``m \times 1`` vector of regressors (remaining columns)
- ``D_t = \mathbf{1}(t > T_B)`` is the break dummy
- ``T_B`` is the break date, searched over the trimmed range

Three test statistics are computed from the residuals ``\hat{e}_t`` at each break date:

- **ADF***: Minimum ADF t-statistic on the residuals across all break dates
- **Zt***: Minimum Phillips-Perron ``Z_t`` statistic
- **Za***: Minimum Phillips-Perron ``Z_\alpha`` statistic

Each statistic selects its own optimal break date. The primary statistic is ADF* (most commonly reported).

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Two cointegrated series with a regime shift at t=100
T_len = 200
x = cumsum(randn(T_len))
y = vcat(1.0 .+ 0.8 .* x[1:100], 3.0 .+ 1.5 .* x[101:end]) + randn(T_len) * 0.5

Y = hcat(y, x)
result = gregory_hansen_test(Y; model=:CS)
report(result)
```

### Interpretation

**Reject** ``H_0`` (p-value < 0.05): evidence for cointegration with a structural break. The estimated break date identifies when the cointegrating relationship shifted. **Fail to reject** ``H_0`` (p-value > 0.05): cannot reject no cointegration, even allowing for a regime change. When the standard Johansen or Engle-Granger test fails to find cointegration but economic theory suggests a long-run relationship, the Gregory-Hansen test checks whether a regime shift explains the apparent lack of cointegration.

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `Symbol` | `:C` | Break model: `:C` (level shift), `:CT` (level + trend), `:CS` (regime shift) |
| `lags` | `Union{Int,Symbol}` | `:aic` | ADF augmenting lags for residual test, or `:aic`/`:bic` |
| `max_lags` | `Union{Int,Nothing}` | `nothing` | Maximum lags for IC selection |
| `trim` | `Real` | `0.15` | Trimming fraction for break search |

### GregoryHansenResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `adf_statistic` | `T` | ADF* test statistic (minimum ADF over break dates) |
| `adf_pvalue` | `T` | P-value for ADF* |
| `zt_statistic` | `T` | Zt* Phillips-Perron test statistic |
| `zt_pvalue` | `T` | P-value for Zt* |
| `za_statistic` | `T` | Za* Phillips-Perron test statistic |
| `za_pvalue` | `T` | P-value for Za* |
| `adf_break` | `Int` | Estimated break date for ADF* (observation index) |
| `zt_break` | `Int` | Estimated break date for Zt* |
| `za_break` | `Int` | Estimated break date for Za* |
| `model` | `Symbol` | Break model (`:C`, `:CT`, `:CS`) |
| `n_regressors` | `Int` | Number of regressors (``m``) |
| `adf_critical_values` | `Dict{Int,T}` | Critical values for ADF*/Zt* at 1%, 5%, 10% |
| `za_critical_values` | `Dict{Int,T}` | Critical values for Za* at 1%, 5%, 10% |
| `nobs` | `Int` | Number of observations |

!!! note "Technical Note"
    Critical values depend on the number of regressors ``m`` and the break model. The `:CS` model (regime shift) allows all slope coefficients to change at the break, providing the most flexible alternative but also the least power. Start with `:C` (level shift only) and use `:CS` only when theory suggests the entire relationship changes.

---

## Complete Example

This example demonstrates a complete structural break analysis workflow: detecting a single break with Andrews, finding multiple breaks with Bai-Perron, testing factor loading stability, and checking for cointegration with a regime shift.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# =========================================================================
# Part 1: Andrews single break test
# =========================================================================

# Generate regression data with a known break at t=150
T_len = 300
X = hcat(ones(T_len), randn(T_len), randn(T_len))
beta_pre = [1.0, 2.0, -1.0]
beta_post = [1.0, 2.0, 2.5]  # third coefficient shifts from -1.0 to 2.5

y = vcat(
    X[1:150, :] * beta_pre,
    X[151:end, :] * beta_post
) + randn(T_len) * 0.5

# Test with all three sup-type functionals
result_sup = andrews_test(y, X; test=:supwald)
result_exp = andrews_test(y, X; test=:expwald)
result_mean = andrews_test(y, X; test=:meanwald)

report(result_sup)
report(result_exp)
report(result_mean)

# =========================================================================
# Part 2: Bai-Perron multiple break test
# =========================================================================

# Generate data with 2 structural breaks
T_len2 = 450
X2 = hcat(ones(T_len2), randn(T_len2))

y2 = vcat(
    X2[1:150, :] * [2.0, 1.0],
    X2[151:300, :] * [5.0, -2.0],
    X2[301:450, :] * [0.0, 3.0]
) + randn(T_len2) * 0.5

result_bp = bai_perron_test(y2, X2; max_breaks=5, criterion=:bic)
report(result_bp)

# Display regime coefficients
for (j, (coefs, ses)) in enumerate(zip(result_bp.regime_coefs, result_bp.regime_ses))
    println("Regime $j: intercept = $(round(coefs[1], digits=2)) ",
            "(SE = $(round(ses[1], digits=2))), ",
            "slope = $(round(coefs[2], digits=2)) ",
            "(SE = $(round(ses[2], digits=2)))")
end

# =========================================================================
# Part 3: Factor model break test
# =========================================================================

# Generate factor model data with stable loadings
N, T_obs, r = 60, 200, 3
F_true = randn(T_obs, r)
Lambda_true = randn(N, r)
X_factor = F_true * Lambda_true' + randn(T_obs, N) * 0.3

# Test loading stability
result_be = factor_break_test(X_factor, r; method=:breitung_eickmeier)
result_hi = factor_break_test(X_factor, r; method=:han_inoue)
result_cdg = factor_break_test(X_factor; method=:chen_dolado_gonzalo)

report(result_be)
report(result_hi)
report(result_cdg)

# =========================================================================
# Part 4: Gregory-Hansen cointegration with regime shift
# =========================================================================

# Two cointegrated series with a level shift at t=120
T_gh = 250
x_gh = cumsum(randn(T_gh))
y_gh = vcat(2.0 .+ 1.0 .* x_gh[1:120], 5.0 .+ 1.0 .* x_gh[121:end]) + randn(T_gh) * 0.3

result_gh = gregory_hansen_test(hcat(y_gh, x_gh); model=:C)
report(result_gh)
```

---

## Common Pitfalls

1. **Trimming too aggressive or too lenient.** The `trimming` parameter controls how much of the sample is excluded from the break search. The default ``\pi = 0.15`` trims 15% from each end, leaving candidate break dates in ``[0.15T, 0.85T]``. Setting ``\pi`` too small (e.g. 0.05) includes extreme dates with few observations in one sub-sample, producing unreliable sub-sample estimates. Setting ``\pi`` too large (e.g. 0.30) excludes potential breaks near the beginning or end of the sample. The 0.15 default follows Andrews (1993) and provides a good balance.

2. **Confusing `:supwald` and `:expwald` power properties.** The supremum functional has optimal power against a single sharp break at an unknown date. The exponential and mean functionals (Andrews & Ploberger 1994) have better power against small or gradual parameter shifts, because they average information across all candidate dates rather than relying on a single maximum. When the nature of the break is unknown, run both and compare.

3. **Bai-Perron minimum segment size.** Each regime segment requires at least ``\max(k+1, \lceil \pi T \rceil)`` observations for OLS estimation. With many regressors or small samples, the maximum feasible number of breaks is reduced below `max_breaks`. The function automatically adjusts `max_breaks` downward when necessary. If the test reports zero breaks despite visual evidence of instability, reduce the number of regressors or increase the sample size.

4. **Factor break tests require sufficient cross-section dimension.** The Breitung-Eickmeier and Han-Inoue tests rely on cross-sectional averaging, which requires ``N`` large enough for the asymptotic approximation. With ``N < 30``, the CUSUM and sup-Wald statistics have poor size properties. The Chen-Dolado-Gonzalo test has similar requirements for the eigenvalue ratio computation. For small panels, Andrews or Bai-Perron tests applied to individual regression equations provide more reliable inference.

5. **Gregory-Hansen model selection affects power.** The `:CS` model (regime shift) allows all slope coefficients to change at the break, providing the most flexible alternative but also the least power due to the larger number of parameters estimated under the alternative. Start with `:C` (level shift only) unless theory suggests the entire cointegrating relationship changes at the break.

---

## References

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856. [DOI](https://doi.org/10.2307/2951764)

- Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a nuisance parameter is present only under the alternative. *Econometrica*, 62(6), 1383-1414. [DOI](https://doi.org/10.2307/2951753)

- Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78. [DOI](https://doi.org/10.2307/2998540)

- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1-22. [DOI](https://doi.org/10.1002/jae.659)

- Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic factor models. *Journal of Econometrics*, 163(1), 71-84. [DOI](https://doi.org/10.1016/j.jeconom.2010.11.008)

- Chen, L., Dolado, J. J., & Gonzalo, J. (2014). Detecting big structural breaks in large factor models. *Journal of Econometrics*, 180(1), 30-48. [DOI](https://doi.org/10.1016/j.jeconom.2014.01.006)

- Gregory, A. W., & Hansen, B. E. (1996). Residual-based tests for cointegration in models with regime shifts. *Journal of Econometrics*, 70(1), 99-126. [DOI](https://doi.org/10.1016/0304-4076(69)41685-7)

- Han, X., & Inoue, A. (2015). Tests for parameter instability in dynamic factor models. *Econometric Theory*, 31(5), 1117-1152. [DOI](https://doi.org/10.1017/S0266466614000486)

- Hansen, B. E. (1997). Approximate asymptotic p values for structural-change tests. *Journal of Business & Economic Statistics*, 15(1), 60-67. [DOI](https://doi.org/10.1080/07350015.1997.10524687)

- Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate regression. *Statistica Sinica*, 7(2), 497-525.
