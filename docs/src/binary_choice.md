# Binary Choice Models

MacroEconometricModels.jl implements logistic and probit regression for binary dependent variables, estimated via maximum likelihood using iteratively reweighted least squares (IRLS). Both models produce Stata/EViews-style output via `report()` and integrate with the package's D3.js visualization and StatsAPI interface.

- **Logit** (logistic regression) with MLE via IRLS/Fisher scoring (McCullagh & Nelder 1989)
- **Probit** with standard normal CDF link (Wooldridge 2010)
- **Marginal effects**: average (AME), at-mean (MEM), and at-representative (MER) with delta-method SEs (Cameron & Trivedi 2005)
- **Odds ratios** with delta-method confidence intervals (Agresti 2002)
- **Classification table**: confusion matrix, accuracy, sensitivity, specificity, precision, F1 score
- **Robust inference**: HC0--HC3 heteroskedasticity-robust and cluster-robust standard errors
- **CrossSectionData dispatch** for symbol-based formula-like syntax
- **StatsAPI interface**: `coef`, `vcov`, `predict`, `confint`, `stderror`, `nobs`, `loglikelihood`

## Quick Start

**Recipe 1: Logit estimation**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Simulate binary outcome data
n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
prob = 1.0 ./ (1.0 .+ exp.(-eta))
y = Float64.(rand(n) .< prob)
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
report(m)
```

**Recipe 2: Probit estimation**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.0, -0.8]
using Distributions
prob = cdf.(Normal(), eta)
y = Float64.(rand(n) .< prob)
m = estimate_probit(y, X; varnames=["(Intercept)", "x1", "x2"])
report(m)
```

**Recipe 3: Average marginal effects**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
me = marginal_effects(m)
report(me)
```

**Recipe 4: Odds ratios**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
or = odds_ratio(m)
println("OR(x1) = ", round(or.or[2], digits=3),
        " [", round(or.ci_lower[2], digits=3), ", ",
        round(or.ci_upper[2], digits=3), "]")
```

**Recipe 5: Classification diagnostics**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
ct = classification_table(m)
println("Accuracy:    ", round(ct["accuracy"], digits=3))
println("Sensitivity: ", round(ct["sensitivity"], digits=3))
println("Specificity: ", round(ct["specificity"], digits=3))
println("F1 Score:    ", round(ct["f1_score"], digits=3))
```

**Recipe 6: CrossSectionData dispatch**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Symbol-based API with automatic intercept
n = 500
data = hcat(Float64.(rand(n) .< 0.5), randn(n), randn(n))
d = CrossSectionData(data; varnames=["outcome", "x1", "x2"])
m = estimate_logit(d, :outcome, [:x1, :x2])
report(m)
```

---

## Logit Model

### Model Specification

The **logistic regression model** relates a binary outcome ``y_i \in \{0, 1\}`` to a ``k \times 1`` regressor vector ``x_i`` through the logistic cumulative distribution function (McCullagh & Nelder 1989):

```math
P(y_i = 1 \mid x_i) = \Lambda(x_i' \beta) = \frac{1}{1 + \exp(-x_i' \beta)}
```

where:
- ``y_i`` is the binary dependent variable (0 or 1)
- ``x_i`` is the ``k \times 1`` vector of regressors (including a constant if desired)
- ``\beta`` is the ``k \times 1`` vector of coefficients
- ``\Lambda(\cdot)`` is the logistic CDF

The logistic function maps the linear index ``x_i' \beta \in (-\infty, \infty)`` to probabilities in ``(0, 1)``. The coefficients ``\beta`` do not have a direct marginal effect interpretation because the relationship between ``x_j`` and ``P(y = 1)`` is nonlinear -- see [Marginal Effects](@ref marginal-effects) below.

### Log-Likelihood

The log-likelihood for the logit model is:

```math
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log \Lambda(x_i' \beta) + (1 - y_i) \log (1 - \Lambda(x_i' \beta)) \right]
```

where:
- ``n`` is the number of observations
- ``\Lambda(x_i' \beta)`` is the predicted probability for observation ``i``

The log-likelihood is globally concave, guaranteeing a unique maximum.

### IRLS Algorithm

!!! note "Technical Note"
    The estimator uses **iteratively reweighted least squares** (IRLS), also known as Fisher scoring. At each iteration, the algorithm solves a weighted least squares problem with weight matrix ``W = \text{diag}(\hat{\mu}_i (1 - \hat{\mu}_i))`` and working response ``z_i = \hat{\eta}_i + (y_i - \hat{\mu}_i) / (\hat{\mu}_i (1 - \hat{\mu}_i))``. The update is ``\hat{\beta}^{(t+1)} = (X' W^{(t)} X)^{-1} X' W^{(t)} z^{(t)}``. Convergence is declared when ``|\ell^{(t+1)} - \ell^{(t)}| < \texttt{tol} \cdot (|\ell^{(t)}| + 1)``.

### McFadden's Pseudo R-squared

The logit model does not have a natural ``R^2``. McFadden (1974) proposes a likelihood-ratio index:

```math
R^2_{\text{McF}} = 1 - \frac{\ell(\hat{\beta})}{\ell(\hat{\beta}_0)}
```

where:
- ``\ell(\hat{\beta})`` is the maximized log-likelihood of the full model
- ``\ell(\hat{\beta}_0)`` is the log-likelihood of the intercept-only (null) model

Values between 0.2 and 0.4 indicate excellent fit (McFadden 1974). Unlike the linear ``R^2``, the McFadden pseudo ``R^2`` never equals 1 in practice.

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data from a known logistic DGP
n = 1000
X = hcat(ones(n), randn(n, 2))
beta_true = [0.0, 1.5, -1.0]
eta = X * beta_true
prob = 1.0 ./ (1.0 .+ exp.(-eta))
y = Float64.(rand(n) .< prob)

# Estimate logit with HC1 robust standard errors
m = estimate_logit(y, X; cov_type=:hc1, varnames=["(Intercept)", "x1", "x2"])
report(m)

# Access individual fields
println("Coefficients: ", round.(m.beta, digits=3))
println("Pseudo R-sq:  ", round(m.pseudo_r2, digits=3))
println("Converged:    ", m.converged, " in ", m.iterations, " iterations")
```

The estimated coefficients recover the true DGP values ``(\beta_0, \beta_1, \beta_2) = (0, 1.5, -1.0)``. With ``n = 1000`` observations, the standard errors are small enough to reject ``H_0: \beta_j = 0`` for both slope coefficients at the 1% level. The McFadden pseudo ``R^2`` reflects the strong signal in the data.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (information matrix), `:hc0`, `:hc1`, `:hc2`, `:hc3`, `:cluster` |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names (auto-generated if `nothing`) |
| `clusters` | `Union{Nothing,AbstractVector}` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `100` | Maximum IRLS iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance for log-likelihood change |

### Return Values

`estimate_logit` returns a `LogitModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Binary dependent variable (0/1) |
| `X` | `Matrix{T}` | ``n \times k`` regressor matrix |
| `beta` | `Vector{T}` | ``k \times 1`` estimated coefficients |
| `vcov_mat` | `Matrix{T}` | ``k \times k`` variance-covariance matrix |
| `residuals` | `Vector{T}` | Deviance residuals |
| `fitted` | `Vector{T}` | Predicted probabilities ``\hat{P}(y_i = 1 \mid x_i)`` |
| `loglik` | `T` | Maximized log-likelihood |
| `loglik_null` | `T` | Null model log-likelihood |
| `pseudo_r2` | `T` | McFadden's pseudo R-squared |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `varnames` | `Vector{String}` | Coefficient names |
| `converged` | `Bool` | Whether IRLS converged |
| `iterations` | `Int` | Number of IRLS iterations |
| `cov_type` | `Symbol` | Covariance estimator used |

---

## Probit Model

### Model Specification

The **probit model** replaces the logistic CDF with the standard normal CDF ``\Phi(\cdot)`` (Wooldridge 2010, Chapter 15):

```math
P(y_i = 1 \mid x_i) = \Phi(x_i' \beta) = \int_{-\infty}^{x_i' \beta} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{t^2}{2}\right) dt
```

where:
- ``\Phi(\cdot)`` is the standard normal CDF
- ``x_i`` and ``\beta`` are defined as in the logit model

The probit model arises naturally from a latent variable framework: ``y_i^* = x_i' \beta + \varepsilon_i`` with ``\varepsilon_i \sim N(0, 1)``, and ``y_i = \mathbf{1}(y_i^* > 0)``. The IRLS algorithm for probit uses Fisher scoring weights ``w_i = \phi(\hat{\eta}_i)^2 / [\hat{\mu}_i (1 - \hat{\mu}_i)]``, where ``\phi(\cdot)`` is the standard normal PDF.

### Logit-Probit Relationship

!!! note "Technical Note"
    The logistic and normal CDFs are nearly identical after rescaling. The approximation ``\beta_{\text{probit}} \approx \beta_{\text{logit}} / 1.6`` holds well across the range of the linear index (Amemiya 1981). Both models produce similar predicted probabilities and marginal effects when the sample is large. The logit model is preferred when odds ratios are the quantity of interest; the probit model when latent variable interpretation or normality assumptions are natural.

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate data and estimate both models
n = 1000
X = hcat(ones(n), randn(n, 2))
beta_true = [0.0, 1.5, -1.0]
eta = X * beta_true
prob = 1.0 ./ (1.0 .+ exp.(-eta))
y = Float64.(rand(n) .< prob)

m_logit = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
m_probit = estimate_probit(y, X; varnames=["(Intercept)", "x1", "x2"])

# Compare coefficients: probit ~ logit / 1.6
println("            Logit     Probit    Probit*1.6")
for j in 1:3
    name = m_logit.varnames[j]
    println(rpad(name, 12),
            round(m_logit.beta[j], digits=3), "     ",
            round(m_probit.beta[j], digits=3), "     ",
            round(m_probit.beta[j] * 1.6, digits=3))
end

# Compare log-likelihoods (both maximize the same binary log-likelihood)
println("\nLog-likelihood (logit):  ", round(m_logit.loglik, digits=2))
println("Log-likelihood (probit): ", round(m_probit.loglik, digits=2))
```

The probit coefficients are approximately ``1/1.6`` times the logit coefficients. The log-likelihoods are close, reflecting the near-identical shapes of the logistic and normal CDFs. Both models yield similar predicted probabilities and marginal effects.

---

## Marginal Effects (@id marginal-effects)

In nonlinear binary choice models, the coefficient ``\beta_j`` does not equal the marginal effect of ``x_j`` on ``P(y = 1)``. The **marginal effect** of a continuous regressor ``x_j`` is (Cameron & Trivedi 2005, Chapter 14):

```math
\frac{\partial P(y_i = 1 \mid x_i)}{\partial x_j} = f(x_i' \beta) \cdot \beta_j
```

where:
- ``f(\cdot)`` is the density function of the link: ``f(\eta) = \Lambda(\eta)(1 - \Lambda(\eta))`` for logit, ``f(\eta) = \phi(\eta)`` for probit
- The marginal effect varies across observations because ``f(x_i' \beta)`` depends on ``x_i``

### Three Types

**Average Marginal Effects** (AME) average over the sample distribution of ``x_i``:

```math
\text{AME}_j = \frac{1}{n} \sum_{i=1}^{n} f(x_i' \hat{\beta}) \cdot \hat{\beta}_j
```

**Marginal Effects at the Mean** (MEM) evaluate at the sample mean of the regressors:

```math
\text{MEM}_j = f(\bar{x}' \hat{\beta}) \cdot \hat{\beta}_j
```

**Marginal Effects at Representative values** (MER) evaluate at user-specified covariate values ``x_0``:

```math
\text{MER}_j = f(x_0' \hat{\beta}) \cdot \hat{\beta}_j
```

where:
- ``\bar{x}`` is the ``k \times 1`` vector of sample means
- ``x_0`` is a user-specified evaluation point (defaults to ``\bar{x}`` with selected values overridden via the `at` argument)

AME is the most commonly reported in applied work because it does not depend on an arbitrary evaluation point. MEM and MER are useful for interpreting effects at specific covariate profiles.

### Delta-Method Standard Errors

!!! note "Technical Note"
    Standard errors for marginal effects use the **delta method** (Oehlert 1992). Let ``g(\hat{\beta})`` denote the vector of marginal effects and ``G = \partial g / \partial \beta'`` its Jacobian. The asymptotic covariance matrix is ``\text{Var}(\hat{g}) \approx G \, \hat{V} \, G'``, where ``\hat{V}`` is the estimated covariance matrix of ``\hat{\beta}``. For AME, the Jacobian row for variable ``j`` has element ``G_{j,l} = \frac{1}{n} \sum_{i=1}^{n} [\mathbf{1}(j = l) \cdot f_i + f'_i \cdot \hat{\beta}_j \cdot x_{il}]``, where ``f_i = f(x_i' \hat{\beta})`` and ``f'_i`` is its derivative with respect to the linear index.

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Estimate a logit model
n = 1000
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])

# Average marginal effects (default)
me_ame = marginal_effects(m)
report(me_ame)

# Marginal effects at the mean
me_mem = marginal_effects(m; type=:mem)
report(me_mem)

# Marginal effects at representative values: x1 = 0.0 (column 2)
me_mer = marginal_effects(m; type=:mer, at=Dict(2 => 0.0))
report(me_mer)
```

The AME for `x1` represents the average change in the predicted probability of ``y = 1`` for a one-unit increase in `x1`, averaged over the observed distribution of all regressors. The MEM evaluates this same derivative at the sample means. The MER evaluates at a profile where `x1 = 0` (with other variables at their means), which is useful for reporting effects at a baseline covariate value.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | `Symbol` | `:ame` | Type of marginal effect: `:ame`, `:mem`, or `:mer` |
| `at` | `Union{Nothing,Dict}` | `nothing` | Evaluation point for `:mer` (Dict mapping column index to value) |
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |

### Return Values

`marginal_effects` returns a `MarginalEffects{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `effects` | `Vector{T}` | Marginal effects for each regressor |
| `se` | `Vector{T}` | Delta-method standard errors |
| `z_stat` | `Vector{T}` | z-statistics (effect / SE) |
| `p_values` | `Vector{T}` | Two-sided p-values (normal distribution) |
| `ci_lower` | `Vector{T}` | Lower confidence interval bounds |
| `ci_upper` | `Vector{T}` | Upper confidence interval bounds |
| `varnames` | `Vector{String}` | Variable names |
| `type` | `Symbol` | `:ame`, `:mem`, or `:mer` |
| `conf_level` | `T` | Confidence level used |

---

## Odds Ratios

The **odds ratio** (OR) provides a multiplicative interpretation of logit coefficients (Agresti 2002, Chapter 5). For regressor ``x_j``, the odds ratio is:

```math
\text{OR}_j = \exp(\hat{\beta}_j)
```

where:
- ``\text{OR}_j > 1`` means a one-unit increase in ``x_j`` increases the odds of ``y = 1``
- ``\text{OR}_j < 1`` means a one-unit increase in ``x_j`` decreases the odds
- ``\text{OR}_j = 1`` means no association

The standard error uses the delta method: ``\text{SE}(\text{OR}_j) = \text{OR}_j \cdot \text{SE}(\hat{\beta}_j)``. Confidence intervals are constructed on the log scale and exponentiated for correct coverage:

```math
\text{CI} = \left[ \exp(\hat{\beta}_j - z_{\alpha/2} \cdot \text{SE}(\hat{\beta}_j)), \;\; \exp(\hat{\beta}_j + z_{\alpha/2} \cdot \text{SE}(\hat{\beta}_j)) \right]
```

!!! note "Technical Note"
    Odds ratios are defined only for logit models. For probit models, marginal effects are the standard way to quantify the impact of regressors. The `odds_ratio` function accepts only `LogitModel` inputs.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 1000
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])

or = odds_ratio(m)
for j in 1:length(or.or)
    println(rpad(or.varnames[j], 14),
            "OR = ", round(or.or[j], digits=3),
            "  95% CI [", round(or.ci_lower[j], digits=3), ", ",
            round(or.ci_upper[j], digits=3), "]")
end
```

An odds ratio of ``\exp(1.5) \approx 4.48`` for `x1` means that a one-unit increase in `x1` multiplies the odds of ``y = 1`` by approximately 4.5. The confidence interval excludes 1, confirming statistical significance. The odds ratio for `x2` is below 1, indicating that higher values of `x2` reduce the odds of the positive outcome.

---

## Classification Table

The **classification table** evaluates the predictive performance of a fitted binary choice model by comparing predicted classes (based on a probability threshold) to observed outcomes (Agresti 2002, Chapter 5). The confusion matrix cross-tabulates actual versus predicted labels:

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | TN (true negative) | FP (false positive) |
| **Actual 1** | FN (false negative) | TP (true positive) |

Summary metrics derived from the confusion matrix:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | ``(\text{TP} + \text{TN}) / n`` | Overall correct classification rate |
| **Sensitivity** | ``\text{TP} / (\text{TP} + \text{FN})`` | True positive rate (recall) |
| **Specificity** | ``\text{TN} / (\text{TN} + \text{FP})`` | True negative rate |
| **Precision** | ``\text{TP} / (\text{TP} + \text{FP})`` | Positive predictive value |
| **F1 Score** | ``2 \cdot \text{Prec} \cdot \text{Sens} / (\text{Prec} + \text{Sens})`` | Harmonic mean of precision and sensitivity |

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 1000
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])

# Default threshold = 0.5
ct = classification_table(m)
println("Confusion matrix:")
println("  TN = ", Int(ct["confusion"][1,1]),
        "  FP = ", Int(ct["confusion"][1,2]))
println("  FN = ", Int(ct["confusion"][2,1]),
        "  TP = ", Int(ct["confusion"][2,2]))
println("Accuracy:    ", round(ct["accuracy"], digits=3))
println("Sensitivity: ", round(ct["sensitivity"], digits=3))
println("Specificity: ", round(ct["specificity"], digits=3))
println("Precision:   ", round(ct["precision"], digits=3))
println("F1 Score:    ", round(ct["f1_score"], digits=3))

# Try a different threshold
ct_30 = classification_table(m; threshold=0.3)
println("\nWith threshold = 0.3:")
println("Sensitivity: ", round(ct_30["sensitivity"], digits=3),
        "  (higher — more positives classified correctly)")
println("Specificity: ", round(ct_30["specificity"], digits=3),
        "  (lower — more false positives)")
```

The default threshold of 0.5 balances sensitivity and specificity. Lowering the threshold increases sensitivity (catches more true positives) at the cost of specificity (more false positives). The optimal threshold depends on the relative costs of false positives and false negatives in the application.

---

## CrossSectionData Dispatch

The `CrossSectionData` wrapper provides a symbol-based API for logit and probit estimation. The dispatch automatically extracts the dependent variable by name, constructs the regressor matrix with an `(Intercept)` column prepended, and passes variable names through to the estimator.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Create a CrossSectionData container with binary outcome
n = 500
x1 = randn(n)
x2 = randn(n)
eta = 0.5 .+ 1.2 * x1 .- 0.8 * x2
prob = 1.0 ./ (1.0 .+ exp.(-eta))
outcome = Float64.(rand(n) .< prob)

data = hcat(outcome, x1, x2)
d = CrossSectionData(data; varnames=["outcome", "x1", "x2"])

# Logit via symbols — intercept added automatically
m_logit = estimate_logit(d, :outcome, [:x1, :x2])
report(m_logit)

# Probit via the same interface
m_probit = estimate_probit(d, :outcome, [:x1, :x2])
report(m_probit)

# Marginal effects from the CrossSectionData-dispatched model
me = marginal_effects(m_logit)
report(me)
```

The `CrossSectionData` dispatch accepts the same keyword arguments as the matrix-based API (`cov_type`, `clusters`, `maxiter`, `tol`). The intercept is always added automatically, so the user should not include a constant column in the data.

---

## Visualization

The `plot_result` function generates D3.js diagnostic plots for binary choice models:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))

m_logit = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
p = plot_result(m_logit)
save_plot(p, "logit_diagnostics.html")
```

```@raw html
<iframe src="../assets/plots/reg_logit.html" style="width:100%;height:420px;border:none;"></iframe>
```

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))

m_probit = estimate_probit(y, X; varnames=["(Intercept)", "x1", "x2"])
p = plot_result(m_probit)
save_plot(p, "probit_diagnostics.html")
```

```@raw html
<iframe src="../assets/plots/reg_probit.html" style="width:100%;height:420px;border:none;"></iframe>
```

Marginal effects produce a horizontal coefficient plot with confidence interval error bars:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

n = 500
X = hcat(ones(n), randn(n, 2))
eta = X * [0.0, 1.5, -1.0]
y = Float64.(rand(n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
m = estimate_logit(y, X; varnames=["(Intercept)", "x1", "x2"])
me = marginal_effects(m)
p = plot_result(me)
save_plot(p, "marginal_effects.html")
```

```@raw html
<iframe src="../assets/plots/reg_marginal_effects.html" style="width:100%;height:380px;border:none;"></iframe>
```

---

## Complete Example

This example demonstrates a full binary choice modeling workflow: data generation, logit and probit estimation, comparison, marginal effects, odds ratios, classification diagnostics, and visualization.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Generate synthetic binary outcome data
# ──────────────────────────────────────────────────────────────────────
n = 1000
x1 = randn(n)                          # Education (standardized)
x2 = randn(n)                          # Experience (standardized)
X = hcat(ones(n), x1, x2)
beta_true = [-0.5, 1.5, 0.8]           # True coefficients
eta = X * beta_true
prob_true = 1.0 ./ (1.0 .+ exp.(-eta))
y = Float64.(rand(n) .< prob_true)
println("Prevalence: ", round(mean(y), digits=3))

# ──────────────────────────────────────────────────────────────────────
# Step 2: Logit estimation with robust SEs
# ──────────────────────────────────────────────────────────────────────
m_logit = estimate_logit(y, X; cov_type=:hc1,
                          varnames=["(Intercept)", "education", "experience"])
report(m_logit)

# ──────────────────────────────────────────────────────────────────────
# Step 3: Probit estimation for comparison
# ──────────────────────────────────────────────────────────────────────
m_probit = estimate_probit(y, X; cov_type=:hc1,
                            varnames=["(Intercept)", "education", "experience"])

# Compare: probit coefficients ~ logit / 1.6
println("\n=== Coefficient Comparison ===")
println("Variable        Logit    Probit   Probit*1.6   True")
for j in 1:3
    name = m_logit.varnames[j]
    println(rpad(name, 16),
            rpad(round(m_logit.beta[j], digits=3), 9),
            rpad(round(m_probit.beta[j], digits=3), 9),
            rpad(round(m_probit.beta[j] * 1.6, digits=3), 13),
            round(beta_true[j], digits=3))
end

# ──────────────────────────────────────────────────────────────────────
# Step 4: Marginal effects (all three types)
# ──────────────────────────────────────────────────────────────────────
me_ame = marginal_effects(m_logit)
me_mem = marginal_effects(m_logit; type=:mem)
me_mer = marginal_effects(m_logit; type=:mer, at=Dict(2 => 1.0, 3 => 0.0))

println("\n=== Marginal Effects Comparison ===")
println("Variable        AME      MEM      MER(educ=1,exp=0)")
for j in 2:3
    name = m_logit.varnames[j]
    println(rpad(name, 16),
            rpad(round(me_ame.effects[j], digits=4), 9),
            rpad(round(me_mem.effects[j], digits=4), 9),
            round(me_mer.effects[j], digits=4))
end

# ──────────────────────────────────────────────────────────────────────
# Step 5: Odds ratios
# ──────────────────────────────────────────────────────────────────────
or = odds_ratio(m_logit)
println("\n=== Odds Ratios ===")
for j in 1:length(or.or)
    println(rpad(or.varnames[j], 16),
            "OR = ", rpad(round(or.or[j], digits=3), 8),
            "95% CI [", round(or.ci_lower[j], digits=3), ", ",
            round(or.ci_upper[j], digits=3), "]")
end

# ──────────────────────────────────────────────────────────────────────
# Step 6: Classification diagnostics
# ──────────────────────────────────────────────────────────────────────
ct = classification_table(m_logit)
println("\n=== Classification (threshold = 0.5) ===")
println("Accuracy:    ", round(ct["accuracy"], digits=3))
println("Sensitivity: ", round(ct["sensitivity"], digits=3))
println("Specificity: ", round(ct["specificity"], digits=3))
println("F1 Score:    ", round(ct["f1_score"], digits=3))

# ──────────────────────────────────────────────────────────────────────
# Step 7: Visualization
# ──────────────────────────────────────────────────────────────────────
p_logit = plot_result(m_logit)
p_me = plot_result(me_ame)
```

The complete workflow estimates both logit and probit specifications on the same data, confirming that the probit coefficients are approximately ``1/1.6`` of the logit coefficients and both yield similar predicted probabilities. The AME for education indicates that a one-standard-deviation increase in education raises the probability of the positive outcome by the reported percentage points, averaged over the sample. The odds ratio for education exceeds 1, confirming the positive association. The classification table shows strong predictive performance with balanced sensitivity and specificity at the default threshold.

---

## Common Pitfalls

1. **Forgetting the intercept column.** `estimate_logit` and `estimate_probit` require the user to include a column of ones in `X` for the intercept. If omitted, the model is estimated without a constant, which biases all coefficients. The `CrossSectionData` dispatch adds the intercept automatically.

2. **Interpreting logit coefficients as marginal effects.** Unlike linear regression, the coefficient ``\hat{\beta}_j`` in a logit or probit model does not equal the marginal effect of ``x_j`` on ``P(y = 1)``. Always compute `marginal_effects(m)` for the correct partial derivative interpretation.

3. **Comparing logit and probit coefficients directly.** Logit and probit coefficients are on different scales because the logistic and normal distributions have different variances (``\pi^2/3`` vs. 1). Use the rule ``\hat{\beta}_{\text{probit}} \approx \hat{\beta}_{\text{logit}} / 1.6`` for approximate comparison, or compare marginal effects or predicted probabilities instead.

4. **Using `odds_ratio` with probit models.** The odds ratio interpretation is specific to the logistic link function. The `odds_ratio` function accepts only `LogitModel` inputs. For probit models, report marginal effects.

5. **Choosing the classification threshold arbitrarily.** The default threshold of 0.5 is not optimal for all applications. When the cost of false negatives differs from the cost of false positives (e.g., medical screening, fraud detection), use a threshold that reflects the relative costs. Evaluate performance across a range of thresholds.

6. **Small-sample bias in pseudo R-squared.** McFadden's ``R^2_{\text{McF}}`` is bounded well below 1 even for perfectly specified models with finite samples. Do not compare pseudo ``R^2`` values across different dependent variables or samples. Use the likelihood ratio test, AIC, or BIC for formal model comparison.

---

## References

- Agresti, Alan. 2002. *Categorical Data Analysis*. 2nd ed. New York: Wiley. ISBN 978-0-471-36093-3.

- Amemiya, Takeshi. 1981. "Qualitative Response Models: A Survey." *Journal of Economic Literature* 19 (4): 1483--1536.

- Cameron, A. Colin, and Pravin K. Trivedi. 2005. *Microeconometrics: Methods and Applications*. Cambridge: Cambridge University Press. ISBN 978-0-521-84805-3.

- McCullagh, Peter, and John A. Nelder. 1989. *Generalized Linear Models*. 2nd ed. London: Chapman & Hall. ISBN 978-0-412-31760-6.

- McFadden, Daniel. 1974. "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*, edited by Paul Zarembka, 105--142. New York: Academic Press.

- Oehlert, Gary W. 1992. "A Note on the Delta Method." *The American Statistician* 46 (1): 27--29. [DOI](https://doi.org/10.1080/00031305.1992.10475842)

- Wooldridge, Jeffrey M. 2010. *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.
