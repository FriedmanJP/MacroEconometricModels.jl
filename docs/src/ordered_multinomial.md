# [Ordered & Multinomial Models](@id ordered_multinomial_page)

**MacroEconometricModels.jl** provides ordered logit/probit and multinomial logit regression for categorical dependent variables with three or more outcomes, estimated via Newton-Raphson maximum likelihood. All models produce Stata-style output and integrate with the package's StatsAPI interface, marginal effects infrastructure, and specification testing.

- **Ordered logit** (cumulative logistic link) for ordinal outcomes (McCullagh 1980)
- **Ordered probit** (cumulative normal link) for ordinal outcomes
- **Multinomial logit** (softmax) for unordered categorical outcomes (McFadden 1974)
- **Average marginal effects** (AME) for all three models with proper K × J output (Cameron & Trivedi 2005)
- **Brant test** of the proportional odds assumption for ordered logit (Brant 1990)
- **Hausman-McFadden IIA test** for multinomial logit (Hausman & McFadden 1984)
- **Robust inference**: MLE, HC0, HC1, and cluster-robust standard errors
- **StatsAPI interface**: `coef`, `vcov`, `predict`, `confint`, `stderror`, `nobs`, `loglikelihood`

```@setup ordmult
using MacroEconometricModels, Random, Distributions
Random.seed!(42)
# Helper: generate ordinal outcome from cumulative logistic model
function _gen_ordered_logit(rng, n, X, beta, cuts)
    xb = X * beta
    y = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(rng)
        y[i] = length(cuts) + 1  # default = highest category
        for (j, c) in enumerate(cuts)
            if u < 1.0 / (1.0 + exp(-(c - xb[i])))
                y[i] = j; break
            end
        end
    end
    y
end
# Helper: generate ordinal outcome from cumulative probit model
function _gen_ordered_probit(rng, n, X, beta, cuts)
    xb = X * beta
    d = Normal()
    y = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(rng)
        y[i] = length(cuts) + 1
        for (j, c) in enumerate(cuts)
            if u < cdf(d, c - xb[i])
                y[i] = j; break
            end
        end
    end
    y
end
```

## Quick Start

**Recipe 1: Ordered logit**

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
report(m)
```

**Recipe 2: Ordered probit**

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_probit(Random.default_rng(), n, X, [0.8, -0.5], [0.0, 1.0])
m = estimate_oprobit(y, X; varnames=["income", "education"])
report(m)
```

**Recipe 3: Multinomial logit**

```@example ordmult
n = 1000
X = hcat(ones(n), randn(n, 2))
# True coefficients: K=3 covariates × (J-1)=2 alternatives
beta_true = [0.5 -0.3; 1.0 -0.5; -0.5 0.8]
V = X * beta_true
eV = exp.(V)
P = hcat(ones(n), eV) ./ (1.0 .+ sum(eV, dims=2))
# Draw from categorical distribution
y = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:n]
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
report(m)
```

**Recipe 4: Marginal effects for ordered logit**

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
me = marginal_effects(m)
# K × J matrix: each row sums to zero across categories
round.(me.effects, digits=4)
```

**Recipe 5: Brant test (proportional odds)**

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
bt = brant_test(m)
round(bt.pvalue, digits=4)
```

**Recipe 6: Hausman IIA test**

```@example ordmult
# Need J >= 4 so that omitting one category leaves >= 3
n = 1500
X = hcat(ones(n), randn(n, 2))
beta_true = [0.3 -0.2 0.1; 0.8 -0.4 0.5; -0.3 0.6 -0.2]  # K=3 × (J-1)=3 → J=4
V = X * beta_true
eV = exp.(V)
P = hcat(ones(n), eV) ./ (1.0 .+ sum(eV, dims=2))
y = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:n]
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
# Test IIA by omitting category 4
iia = hausman_iia(m; omit_category=4)
round(iia.pvalue, digits=4)
```

---

## Ordered Logit

The **ordered logit** (proportional odds) model relates an ordinal outcome ``y_i \in \{1, 2, \ldots, J\}`` to regressors ``x_i`` through the cumulative logistic distribution (McCullagh 1980):

```math
P(y_i \leq j \mid x_i) = \Lambda(\alpha_j - x_i' \beta), \quad j = 1, \ldots, J-1
```

where:
- ``y_i`` is the ordinal dependent variable with ``J`` categories
- ``x_i`` is the ``K \times 1`` regressor vector (**no intercept** — absorbed by cutpoints)
- ``\beta`` is the ``K \times 1`` slope coefficient vector
- ``\alpha_1 < \alpha_2 < \cdots < \alpha_{J-1}`` are the **cutpoints** (thresholds)
- ``\Lambda(\cdot)`` is the logistic CDF

The category probabilities follow from differencing:

```math
P(y_i = j \mid x_i) = F(\alpha_j - x_i' \beta) - F(\alpha_{j-1} - x_i' \beta)
```

with the convention ``F(\alpha_0) = 0`` and ``F(\alpha_J) = 1``.

!!! warning "No intercept in X"
    The intercept is absorbed into the cutpoints. Including a column of ones in X causes identification failure.

### Estimation

The model is estimated by Newton-Raphson MLE using the BHHH (outer product of gradients) approximation to the Hessian. The optimizer enforces cutpoint ordering ``\alpha_1 < \alpha_2 < \cdots < \alpha_{J-1}`` at each iteration.

```@example ordmult
n = 1000
X = randn(n, 3)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5, 0.3], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education", "age"], cov_type=:hc1)
report(m)
```

The output reports slope coefficients and cutpoints separately. A positive ``\beta_k`` shifts probability mass toward higher categories. McFadden's pseudo ``R^2 = 1 - \ell(\hat{\beta}) / \ell_0`` measures improvement over the null (cutpoints-only) model.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (MLE), `:hc0`, `:hc1`, `:cluster` |
| `varnames` | `Vector{String}` | auto | Coefficient names |
| `clusters` | `AbstractVector` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `200` | Maximum Newton-Raphson iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Vector{T}` | Slope coefficients (K) |
| `cutpoints` | `Vector{T}` | Estimated cutpoints (J-1) |
| `vcov_mat` | `Matrix{T}` | Joint vcov of ``[\beta; \alpha]`` |
| `fitted` | `Matrix{T}` | Predicted probabilities (n × J) |
| `loglik` | `T` | Maximized log-likelihood |
| `pseudo_r2` | `T` | McFadden's pseudo R-squared |
| `categories` | `Vector` | Original category values |
| `converged` | `Bool` | Convergence flag |

---

## Ordered Probit

The **ordered probit** model uses the standard normal CDF ``\Phi(\cdot)`` as the link function:

```math
P(y_i \leq j \mid x_i) = \Phi(\alpha_j - x_i' \beta)
```

The API is identical to ordered logit. The choice between logit and probit is largely a matter of convention — both produce similar marginal effects in practice.

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_probit(Random.default_rng(), n, X, [0.8, -0.5], [0.0, 1.0])
m = estimate_oprobit(y, X; varnames=["income", "education"])
report(m)
```

---

## Multinomial Logit

The **multinomial logit** model relates an unordered categorical outcome ``y_i \in \{1, 2, \ldots, J\}`` to regressors through the softmax function (McFadden 1974):

```math
P(y_i = j \mid x_i) = \frac{\exp(x_i' \beta_j)}{\sum_{k=1}^{J} \exp(x_i' \beta_k)}
```

where:
- ``\beta_1 = 0`` (base category normalization)
- ``\beta_j`` is the ``K \times 1`` coefficient vector for alternative ``j = 2, \ldots, J``
- The model estimates ``K \times (J-1)`` free parameters

!!! note "Technical Note"
    The implementation uses the log-sum-exp trick for numerical stability and computes the analytical Hessian (not BHHH) for fast convergence. The `coef(m)` function returns `vec(beta)` of length ``K(J-1)``.

### Estimation

```@example ordmult
n = 1000
X = hcat(ones(n), randn(n, 2))
beta_true = [0.5 -0.3; 1.0 -0.5; -0.5 0.8]
V = X * beta_true
eV = exp.(V)
P = hcat(ones(n), eV) ./ (1.0 .+ sum(eV, dims=2))
y = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:n]
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
report(m)
```

The output displays one coefficient table per alternative (relative to the base category). Positive ``\beta_{j,k}`` means higher ``x_k`` increases the probability of choosing alternative ``j`` relative to the base.

### Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (MLE), `:hc0`, `:hc1`, `:cluster` |
| `varnames` | `Vector{String}` | auto | Coefficient names |
| `clusters` | `AbstractVector` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `200` | Maximum Newton-Raphson iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `beta` | `Matrix{T}` | Coefficient matrix (K × (J-1)), base = category 1 |
| `vcov_mat` | `Matrix{T}` | Vcov of `vec(beta)`, K(J-1) × K(J-1) |
| `fitted` | `Matrix{T}` | Predicted probabilities (n × J) |
| `loglik` | `T` | Maximized log-likelihood |
| `pseudo_r2` | `T` | McFadden's pseudo R-squared |
| `categories` | `Vector` | Original category values |

---

## Marginal Effects

Coefficients in ordered and multinomial models do not have direct marginal effect interpretations because the probability functions are nonlinear. **Average marginal effects** (AME) compute the mean derivative across all observations (Cameron & Trivedi 2005).

### Ordered Models

For the ordered logit, the AME of variable ``k`` on outcome ``j`` is:

```math
\text{AME}_{k,j} = \frac{1}{n} \sum_{i=1}^{n} \left[ f(\alpha_{j-1} - x_i' \beta) - f(\alpha_j - x_i' \beta) \right] \beta_k
```

where ``f(\cdot)`` is the logistic PDF. The AMEs sum to zero across categories for each variable — increasing probability in one category must decrease it elsewhere.

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
me = marginal_effects(m)
# Row k = variable, column j = category
round.(me.effects, digits=4)
```

### Multinomial Logit

For the multinomial logit, the AME formula accounts for the softmax structure:

```math
\text{AME}_{k,j} = \frac{1}{n} \sum_{i=1}^{n} p_{ij} \left( \beta_{j,k} - \sum_{m=1}^{J} p_{im} \beta_{m,k} \right)
```

```@example ordmult
n = 1000
X = hcat(ones(n), randn(n, 2))
beta_true = [0.5 -0.3; 1.0 -0.5; -0.5 0.8]
V = X * beta_true
eV = exp.(V)
P = hcat(ones(n), eV) ./ (1.0 .+ sum(eV, dims=2))
y = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:n]
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
me = marginal_effects(m)
round.(me.effects, digits=4)
```

---

## Specification Tests

### Brant Test (Proportional Odds)

The **Brant test** evaluates the proportional odds (parallel regression) assumption underlying ordered logit (Brant 1990). Under the null, the slope coefficients are equal across all ``J-1`` binary logits that split the outcome at each cutpoint.

The test fits ``J-1`` separate binary logits (``y \leq j`` vs ``y > j``) and constructs a Wald statistic comparing the binary logit coefficients to each other. Rejection suggests the ordered logit model is misspecified — different cutpoints produce different slope estimates.

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
bt = brant_test(m)
# Overall test
println("Chi-squared: ", round(bt.statistic, digits=3), ", p-value: ", round(bt.pvalue, digits=4))
# Per-variable p-values
round.(bt.per_variable, digits=4)
```

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Overall Wald statistic |
| `pvalue` | `T` | P-value (chi-squared with ``K(J-2)`` df) |
| `df` | `Int` | Degrees of freedom |
| `per_variable` | `Vector{T}` | Per-variable p-values (``J-2`` df each) |
| `binary_coefs` | `Matrix{T}` | K × (J-1) matrix of binary logit coefficients |

### Hausman-McFadden IIA Test

The **independence of irrelevant alternatives** (IIA) assumption states that the odds ratio between any two alternatives is independent of other alternatives. The Hausman-McFadden test (1984) re-estimates the multinomial logit excluding one category and compares the coefficients to the full model.

```@example ordmult
# Need J >= 4 so that omitting one leaves >= 3
n = 1500
X = hcat(ones(n), randn(n, 2))
beta_true = [0.3 -0.2 0.1; 0.8 -0.4 0.5; -0.3 0.6 -0.2]
V = X * beta_true
eV = exp.(V)
P = hcat(ones(n), eV) ./ (1.0 .+ sum(eV, dims=2))
y = [findfirst(cumsum(P[i, :]) .>= rand()) for i in 1:n]
m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
iia = hausman_iia(m; omit_category=4)
println("Hausman stat: ", round(iia.statistic, digits=3), ", p-value: ", round(iia.pvalue, digits=4))
nothing # hide
```

Failure to reject supports the IIA assumption. If IIA is rejected, consider nested logit or mixed logit alternatives.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Hausman test statistic |
| `pvalue` | `T` | P-value (chi-squared) |
| `df` | `Int` | Degrees of freedom |
| `omitted_category` | any | Label of the omitted category |

---

## Out-of-Sample Prediction

All three models support `predict(m, X_new)` for computing predicted probabilities on new data:

```@example ordmult
n = 1000
X = randn(n, 2)
y = _gen_ordered_logit(Random.default_rng(), n, X, [1.0, -0.5], [0.0, 1.5])
m = estimate_ologit(y, X; varnames=["income", "education"])
# Predict on new observations
X_new = randn(5, 2)
probs = predict(m, X_new)
round.(probs, digits=3)
```

Each row sums to 1. For multinomial logit, the same interface applies via `predict(m, X_new)`.

---

## Complete Example

A full workflow estimating ordered and multinomial models on the same simulated survey data:

```@example ordmult
# Simulate ordinal satisfaction data (5 categories)
n = 2000
X = randn(n, 3)
xb = X * [0.8, -0.4, 0.3]
cuts = [-1.5, -0.5, 0.5, 1.5]
d = Normal()
F = hcat([cdf.(d, c .- xb) for c in cuts]...)
u = rand(n)
y = ones(Int, n) .* 5
for j in 4:-1:1
    y[u .< F[:, j]] .= j
end

# Ordered logit
m_ologit = estimate_ologit(y, X; varnames=["income", "education", "age"])
report(m_ologit)
```

```@example ordmult
# Marginal effects
me = marginal_effects(m_ologit)
round.(me.effects, digits=4)
```

```@example ordmult
# Brant test
bt = brant_test(m_ologit)
println("Brant test p-value: ", round(bt.pvalue, digits=4))
nothing # hide
```

```@example ordmult
# Ordered probit on same data
m_oprobit = estimate_oprobit(y, X; varnames=["income", "education", "age"])
report(m_oprobit)
```

```@example ordmult
# Multinomial logit on unordered version
m_mlogit = estimate_mlogit(y, hcat(ones(n), X); varnames=["const", "income", "education", "age"])
report(m_mlogit)
```

---

## Common Pitfalls

1. **Including an intercept in ordered models.** The cutpoints absorb the intercept. Adding a column of ones to X causes multicollinearity and estimation failure.

2. **Fewer than 3 categories.** Both ordered and multinomial models require ``J \geq 3``. For binary outcomes, use `estimate_logit` or `estimate_probit` instead.

3. **Ignoring the proportional odds assumption.** Always run `brant_test` after ordered logit. If rejected, the coefficients vary across cutpoints and a generalized ordered logit or multinomial logit is more appropriate.

4. **Interpreting multinomial logit coefficients as marginal effects.** The ``\beta_{j,k}`` coefficients measure log-odds ratios relative to the base category, not marginal changes in probability. Always compute `marginal_effects(m)` for substantive interpretation.

5. **IIA violations in multinomial logit.** If `hausman_iia` rejects for any omitted category, the multinomial logit model is misspecified. The relative odds between remaining alternatives should not change when an alternative is removed.

---

## References

- Agresti, A. (2010). *Analysis of Ordinal Categorical Data*. 2nd ed. Wiley. ISBN 978-0-470-08289-8.
- Brant, R. (1990). Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression. *Biometrics* 46(4), 1171-1178. [DOI](https://doi.org/10.2307/2532457)
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. ISBN 978-0-521-84805-3.
- Greene, W. H. (2012). *Econometric Analysis*. 7th ed. Prentice Hall. ISBN 978-0-131-39538-1.
- Hausman, J. A. & McFadden, D. (1984). Specification Tests for the Multinomial Logit Model. *Econometrica* 52(5), 1219-1240. [DOI](https://doi.org/10.2307/1910997)
- McCullagh, P. (1980). Regression Models for Ordinal Data. *Journal of the Royal Statistical Society: Series B* 42(2), 109-142. [DOI](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x)
- McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105-142). Academic Press.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. 2nd ed. Cambridge University Press. ISBN 978-0-521-74738-7.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press. ISBN 978-0-262-23258-6.
