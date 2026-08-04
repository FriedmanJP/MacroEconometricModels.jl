# [Ordered & Multinomial Models](@id ordered_multinomial_page)

**MacroEconometricModels.jl** estimates ordered logit, ordered probit, and multinomial logit models for categorical dependent variables with three or more outcomes, all by Newton-Raphson maximum likelihood with analytic derivatives. The models produce Stata-style output through `report()` and carry the package's marginal-effects, prediction, residual, and specification-testing infrastructure. For two-category outcomes see [Binary Choice Models](@ref binary_choice_page); for panel versions of discrete choice see [Panel Regression](@ref panel_reg_page).

- **Ordered logit** (cumulative logistic link) for ordinal outcomes (McCullagh 1980)
- **Ordered probit** (cumulative normal link) for the same design
- **Multinomial logit** (softmax) for unordered alternatives (McFadden 1974)
- **Average marginal effects** for all three models, as a ``K \times J`` variables-by-categories matrix (Cameron & Trivedi 2005)
- **Brant test** of the proportional-odds assumption, overall and per variable (Brant 1990)
- **Hausman-McFadden IIA test** for multinomial logit (Hausman & McFadden 1984)
- **Residuals**: per-category response, Pearson, and deviance matrices, plus generalized residuals for the ordered models (Chesher & Irish 1987)
- **Robust inference**: observed-information, HC0, HC1, and cluster-robust standard errors
- **StatsAPI interface**: `coef`, `vcov`, `predict`, `confint`, `stderror`, `nobs`, `loglikelihood`, `residuals`

```@setup ordmult
using MacroEconometricModels, Statistics
mroz = load_example(:mroz)
```

Every example models labour-supply intensity in the Mroz (1987) extract: 753 married women observed in 1975, with annual hours banded into four ordered categories — no work (325 women), part time under 1000 hours (155), 1000 to 1999 hours (201), and 2000 hours or more (72). The covariates are non-wife household income in thousands (`nwifeinc`), years of schooling (`educ`), labour-market experience (`exper`), `age`, and the number of children under six (`kidslt6`).

## Quick Start

**Recipe 1: Ordered logit**

```@example ordmult
hours = mroz[:, "hours"]
supply = [h == 0 ? 1 : h < 1000 ? 2 : h < 2000 ? 3 : 4 for h in hours]
Xo = hcat(mroz[:, "nwifeinc"], mroz[:, "educ"], mroz[:, "exper"],
          mroz[:, "age"], mroz[:, "kidslt6"])
onames = ["nwifeinc", "educ", "exper", "age", "kidslt6"]

m_ologit = estimate_ologit(supply, Xo; varnames=onames)
report(m_ologit)
```

**Recipe 2: Ordered probit on the same design**

```@example ordmult
m_oprobit = estimate_oprobit(supply, Xo; varnames=onames)
report(m_oprobit)
```

**Recipe 3: Multinomial logit (needs its own intercept)**

```@example ordmult
Xm = hcat(ones(mroz.N_obs), Xo)
mnames = ["(Intercept)"; onames]

m_mlogit = estimate_mlogit(supply, Xm; varnames=mnames)
report(m_mlogit)
```

**Recipe 4: Average marginal effects**

```@example ordmult
me = marginal_effects(m_ologit)
round.(me.effects, digits=4)   # rows = variables, columns = categories
```

**Recipe 5: Brant test of proportional odds**

```@example ordmult
bt = brant_test(m_ologit)
(chi2 = round(bt.statistic, digits=3), df = bt.df, pvalue = round(bt.pvalue, digits=4))
```

**Recipe 6: Hausman-McFadden IIA test**

```@example ordmult
iia = hausman_iia(m_mlogit; omit_category=4)
(chi2 = round(iia.statistic, digits=3), df = iia.df, pvalue = round(iia.pvalue, digits=4))
```

---

## Ordered Logit

The **ordered logit**, or proportional-odds model, relates an ordinal outcome ``y_i \in \{1, \ldots, J\}`` to regressors through the cumulative logistic distribution (McCullagh 1980):

```math
P(y_i \leq j \mid x_i) = \Lambda(\alpha_j - x_i' \beta), \quad j = 1, \ldots, J-1
```

where:
- ``y_i`` is the ordinal outcome with ``J`` ordered categories
- ``x_i`` is the ``K \times 1`` regressor vector, **without an intercept**
- ``\beta`` is the ``K \times 1`` slope vector, common to every cutpoint
- ``\alpha_1 < \alpha_2 < \cdots < \alpha_{J-1}`` are the cutpoints on the latent index
- ``\Lambda(\cdot)`` is the logistic CDF

Category probabilities follow by differencing the cumulative probabilities:

```math
P(y_i = j \mid x_i) = F(\alpha_j - x_i' \beta) - F(\alpha_{j-1} - x_i' \beta)
```

where:
- ``F`` is the link CDF, logistic here and standard normal for ordered probit
- ``F(\alpha_0 - x_i'\beta) \equiv 0`` and ``F(\alpha_J - x_i'\beta) \equiv 1`` close the system

A positive ``\beta_k`` raises the latent index and therefore shifts probability mass toward higher categories. Because a single ``\beta`` governs every cutpoint, the log-odds of ``y \leq j`` shift in parallel — the assumption the [Brant test](@ref specification-tests) checks.

!!! warning "No intercept in X"
    The cutpoints absorb the intercept. A column of ones in `X` makes the parameter vector unidentified and the Hessian singular. Only the multinomial logit on this page takes an explicit intercept column.

!!! note "Technical Note"
    Estimation is true Newton-Raphson on ``\theta = [\beta; \alpha]`` using the **analytic observed-information Hessian**, not the BHHH outer-product approximation: the curvature block ``(\partial^2 p / \partial\theta^2)/p`` that BHHH drops is computed from the density derivative ``f'``. Cutpoint ordering is enforced after every step, and iteration stops when ``|\ell^{(t+1)} - \ell^{(t)}| < \texttt{tol}\,(|\ell^{(t)}| + 1)``. With `cov_type=:ols` the reported covariance is ``(-H)^{-1}``, which matches Stata's `vce(oim)`.

```@example ordmult
report(estimate_ologit(supply, Xo; varnames=onames, cov_type=:hc1))
```

Every slope is significant at the 5% level or better. An extra year of schooling raises the latent labour-supply index by 0.158 and an extra year of experience by 0.116, while each additional year of age lowers it by 0.082 and each preschool child by 1.328 — the dominant force in the model, as in the participation decision on the [binary choice page](@ref binary_choice_page). The three cutpoints ``(-1.30, -0.20, 1.81)`` are spaced unevenly: the gap between the second and third is far wider than between the first and second, which says that crossing from part time into the 1000-1999 hour band takes much more of a push in the latent index than crossing from no work into part time. The sandwich standard errors here differ from the observed-information ones by only a few percent, so misspecification of the conditional variance is not a live concern in this sample.

### Keyword Arguments

`estimate_ologit` and `estimate_oprobit` share this signature.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (observed information), `:hc0`, `:hc1`, `:cluster` |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names (`"x1"`, `"x2"`, … if `nothing`) |
| `clusters` | `Union{Nothing,AbstractVector}` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `200` | Maximum Newton-Raphson iterations |
| `tol` | `Real` | ``10^{-8}`` | Relative convergence tolerance on the log-likelihood |

### Return Values

`estimate_ologit` returns an `OrderedLogitModel{T}` and `estimate_oprobit` an `OrderedProbitModel{T}`, with identical fields:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{Int}` | Outcome remapped to ``1:J`` in sorted category order |
| `X` | `Matrix{T}` | ``n \times K`` regressor matrix (no intercept) |
| `beta` | `Vector{T}` | ``K \times 1`` slope coefficients |
| `cutpoints` | `Vector{T}` | ``J-1`` estimated cutpoints |
| `vcov_mat` | `Matrix{T}` | Joint covariance of ``[\beta; \alpha]``, ``(K + J - 1)`` square |
| `fitted` | `Matrix{T}` | ``n \times J`` fitted category probabilities |
| `loglik` | `T` | Maximized log-likelihood |
| `loglik_null` | `T` | Log-likelihood of the sample-frequency model |
| `pseudo_r2` | `T` | McFadden's pseudo ``R^2`` |
| `aic`, `bic` | `T` | Information criteria on ``K + J - 1`` parameters |
| `varnames` | `Vector{String}` | Coefficient names |
| `categories` | `Vector` | Original category labels, sorted |
| `converged` | `Bool` | Whether Newton-Raphson met the tolerance |
| `iterations` | `Int` | Iterations performed |
| `cov_type` | `Symbol` | Covariance estimator used |

---

## Ordered Probit

The **ordered probit** replaces the logistic CDF with the standard normal:

```math
P(y_i \leq j \mid x_i) = \Phi(\alpha_j - x_i' \beta)
```

where:
- ``\Phi(\cdot)`` is the standard normal CDF
- ``\alpha_j`` and ``\beta`` carry the same meaning as under the logistic link

The latent-variable reading is the sharper one here: ``y_i^* = x_i'\beta + \varepsilon_i`` with ``\varepsilon_i \sim N(0,1)``, and ``y_i = j`` when ``\alpha_{j-1} < y_i^* \le \alpha_j``. The API is identical to ordered logit.

```@example ordmult
round.(coef(m_ologit) ./ coef(m_oprobit), digits=2)
```

The slope ratios run from 1.70 to 1.79, the familiar logistic-to-normal scale factor, so the two links tell the same story on different rulers. The ordered probit log-likelihood is ``-831.54`` against the ordered logit's ``-828.85``, and its McFadden index is 0.127 against 0.130 — a difference well inside sampling noise, which is why the choice between the links is conventionally made on convenience. Cutpoints rescale by the same factor: the probit's ``(-0.81, -0.15, 0.98)`` map onto the logit's ``(-1.30, -0.20, 1.81)``.

---

## Multinomial Logit

The **multinomial logit** treats the outcomes as unordered alternatives and models them with the softmax (McFadden 1974):

```math
P(y_i = j \mid x_i) = \frac{\exp(x_i' \beta_j)}{\sum_{l=1}^{J} \exp(x_i' \beta_l)}
```

where:
- ``\beta_1 = 0`` normalizes the first sorted category as the base
- ``\beta_j`` is the ``K \times 1`` coefficient vector of alternative ``j = 2, \ldots, J``
- the model has ``K(J-1)`` free parameters, and ``x_i`` must include an explicit intercept

Each ``\beta_{j,k}`` is the effect of ``x_k`` on the log-odds of alternative ``j`` against the base, ``\log[P(y=j)/P(y=1)]``. Nothing constrains those effects to move monotonically across alternatives, which is exactly the flexibility the ordered models give up.

!!! note "Technical Note"
    The likelihood is evaluated with the log-sum-exp trick and maximized by Newton-Raphson on the analytic Hessian, so convergence takes six iterations here. `coef(m)` returns `vec(m.beta)` of length ``K(J-1)``, stacked alternative by alternative, and `vcov(m)` is the matching ``K(J-1)`` square matrix — index block ``j`` as rows `(j-1)K+1 : jK`.

```@example ordmult
report(m_mlogit)
```

Reading the blocks against the base category of not working: schooling raises the log-odds of every working state by a similar amount (0.230, 0.216, 0.236), so education mostly drives the participation margin rather than the choice of hours. Experience does the opposite, rising monotonically from 0.076 for part time to 0.174 for full-time work, so experience is what sorts working women into longer hours. Preschool children push against all three working states, most strongly against the 1000-1999 hour band (``-1.963``). The multinomial fit costs 18 parameters against the ordered logit's 8 and buys 8.4 log-likelihood points (``-820.41`` against ``-828.85``), which the BIC (1760.06 against 1710.70) judges a poor trade.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (observed information), `:hc0`, `:hc1`, `:cluster` |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names, one per column of `X` |
| `clusters` | `Union{Nothing,AbstractVector}` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `200` | Maximum Newton-Raphson iterations |
| `tol` | `Real` | ``10^{-8}`` | Relative convergence tolerance on the log-likelihood |

### Return Values

`estimate_mlogit` returns a `MultinomialLogitModel{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{Int}` | Outcome remapped to ``1:J`` in sorted category order |
| `X` | `Matrix{T}` | ``n \times K`` regressor matrix, intercept included |
| `beta` | `Matrix{T}` | ``K \times (J-1)`` coefficients; column ``j`` belongs to category ``j+1`` |
| `vcov_mat` | `Matrix{T}` | Covariance of `vec(beta)`, ``K(J-1)`` square |
| `fitted` | `Matrix{T}` | ``n \times J`` fitted probabilities |
| `loglik` | `T` | Maximized log-likelihood |
| `loglik_null` | `T` | Constants-only log-likelihood (Stata-comparable) |
| `pseudo_r2` | `T` | McFadden's pseudo ``R^2`` |
| `aic`, `bic` | `T` | Information criteria on ``K(J-1)`` parameters |
| `varnames` | `Vector{String}` | Coefficient names |
| `categories` | `Vector` | Original category labels; the first is the base |
| `converged` | `Bool` | Whether Newton-Raphson met the tolerance |
| `iterations` | `Int` | Iterations performed |
| `cov_type` | `Symbol` | Covariance estimator used |

---

## Marginal Effects

Neither model's coefficients are marginal effects: the probabilities are nonlinear in the index, and in the multinomial case a coefficient can even carry the opposite sign to the effect on its own category's probability. `marginal_effects` returns average marginal effects — the sample mean of the observation-level derivative (Cameron & Trivedi 2005, ch. 15) — as a ``K \times J`` matrix whose rows are variables and columns are categories.

For an ordered model the derivative of the category probability is a difference of link densities:

```math
\text{AME}_{k,j} = \frac{1}{n} \sum_{i=1}^{n} \left[ f(\alpha_{j-1} - x_i' \beta) - f(\alpha_j - x_i' \beta) \right] \beta_k
```

where:
- ``f(\cdot)`` is the logistic or standard normal density
- ``f(\alpha_0 - x_i'\beta) \equiv 0`` and ``f(\alpha_J - x_i'\beta) \equiv 0`` at the open ends

Each row sums to zero across categories: probability moved into one category must leave another.

```@example ordmult
me_o = marginal_effects(m_ologit)
round.(me_o.effects, digits=4)
```

Row five is the preschool-child effect: one more young child raises the probability of not working by 25.1 points and takes 13.5 points off the 1000-1999 hour band and 10.2 points off full-time work. Education works in the opposite direction and at a tenth of the size, lifting the two full-time bands by 1.6 and 1.2 points at the expense of a 3.0-point fall in non-participation. The part-time column is nearly inert for every regressor — the middle band's two density terms almost cancel — so the covariates here move women between not working and working long hours, rather than into part-time work. Rows sum to zero by construction, which is the check to run on any hand-rolled marginal-effect code.

For the multinomial logit the derivative involves the probability-weighted average coefficient:

```math
\text{AME}_{k,j} = \frac{1}{n} \sum_{i=1}^{n} p_{ij} \left( \beta_{j,k} - \sum_{l=1}^{J} p_{il} \beta_{l,k} \right)
```

where:
- ``p_{ij}`` is observation ``i``'s fitted probability of alternative ``j``
- ``\beta_{1,k} = 0`` for the base category

```@example ordmult
me_m = marginal_effects(m_mlogit)
report(me_m)
```

The multinomial marginal effects agree closely with the ordered ones despite the far looser parameterization: an extra preschool child moves 21.5 points out of the 1000-1999 hour band and 5.4 points out of full-time work, against the ordered model's 13.5 and 10.2. Schooling again raises every working state, most strongly part time (1.9 points). The standard errors sort these into what the data can and cannot resolve: the 21.5-point effect on the middle band carries a standard error of 4.4 points, while the 0.7-point effect on part-time work carries 3.2 and is indistinguishable from zero. The intercept row is dropped when the effects are built — the marginal effect of a constant regressor is a numerical artefact — so `me_m.effects` has one row per genuine covariate, and `report` prints one panel per non-base alternative.

!!! note "Standard errors for marginal effects"
    Both families report delta-method standard errors. For the multinomial logit the Jacobian of the AME is taken by finite differences over **every** free coefficient, intercepts included, and propagated through the full coefficient covariance, because each coefficient enters the AME through the fitted probabilities; restricting the covariance to the reported rows understates the variance. For the ordered models the Jacobian runs over ``[\beta; \alpha]``, so cutpoint uncertainty enters the interval. `MultinomialMarginalEffects.se` is `nothing` only when the model covariance is unavailable, and `plot_result` then draws points without whiskers.

---

## [Specification Tests](@id specification-tests)

### Brant Test

The **Brant test** examines the proportional-odds assumption behind ordered logit (Brant 1990). It fits the ``J-1`` binary logits that split the outcome at each cutpoint (``y \leq j`` against ``y > j``) and asks whether their slope vectors agree. Under the null they estimate the same ``\beta``; systematic differences mean the ordered logit forces one slope on relationships that vary with the threshold.

The overall statistic stacks the contrasts ``\hat\beta^{(j)} - \hat\beta^{(J-1)}`` for ``j = 1, \ldots, J-2`` and is ``\chi^2`` with ``K(J-2)`` degrees of freedom; the per-variable statistics use the same contrasts one coefficient at a time, with ``J-2`` degrees of freedom each.

```@example ordmult
bt = brant_test(m_ologit)
(statistic = round(bt.statistic, digits=3), df = bt.df, pvalue = round(bt.pvalue, digits=4))
```

```@example ordmult
NamedTuple{Tuple(Symbol.(onames))}(Tuple(round.(bt.per_variable, digits=4)))
```

The overall statistic of 15.574 on 10 degrees of freedom gives ``p = 0.113``, so the joint proportional-odds restriction survives at conventional levels. The per-variable breakdown is the more informative half: `educ` rejects on its own (``p = 0.033``) and `age` sits just outside the 5% line (``p = 0.058``), while the other three regressors are nowhere near it. Reading the binary-logit coefficients in `bt.binary_coefs` shows why — the education slope is ``-0.223`` at the first split, ``-0.120`` at the second and ``-0.083`` at the third (the split logits model ``y \leq j``, so their signs are the mirror of the ordered slopes), so schooling separates non-participants from workers roughly three times as sharply as it separates full-time workers from everyone below them. A single rejecting variable is a warning rather than a verdict: refit with `educ` interacted with the threshold, or move to the multinomial specification, and check whether the substantive conclusions move.

!!! note "Joint covariance across the binary fits"
    The ``J-1`` binary logits share one sample, so the contrast variance uses the joint sandwich ``\text{Cov}(\hat\beta^{(a)}, \hat\beta^{(b)}) = B_a \left( \sum_i s_{a,i} s_{b,i}' \right) B_b`` rather than the independence approximation ``\text{Var}(\hat\beta^{(a)}) + \text{Var}(\hat\beta^{(b)})``, which is Brant's (1990) full covariance form. Bread and meat are formed over the split logits' complete ``(K+1)``-dimensional parameter vector and restricted to the slope block afterwards, so the intercepts are partialled out rather than discarded — dropping them before inverting the bread leaves the statistic dependent on where the regressors are centred.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Overall Wald statistic |
| `pvalue` | `T` | Overall p-value, ``\chi^2`` with ``K(J-2)`` df |
| `df` | `Int` | Degrees of freedom, ``K(J-2)`` |
| `per_variable` | `Vector{T}` | Per-variable p-values, length ``K``, ``J-2`` df each |
| `binary_coefs` | `Matrix{T}` | ``K \times (J-1)`` slopes from the binary logits |

### Hausman-McFadden IIA Test

**Independence of irrelevant alternatives** requires that the odds between any two alternatives not depend on what other alternatives exist. The Hausman-McFadden test (1984) re-estimates the model on the subsample that excludes one category and compares the two coefficient vectors, after renormalizing the restricted fit onto the full model's base category:

```math
H = (\hat\beta_r - \hat\beta_f)' (\hat V_r - \hat V_f)^{-1} (\hat\beta_r - \hat\beta_f)
```

where:
- ``\hat\beta_r`` and ``\hat V_r`` come from the restricted model, estimated without the omitted category
- ``\hat\beta_f`` and ``\hat V_f`` are the corresponding full-model quantities, base-adjusted
- ``H`` is ``\chi^2`` with ``K`` times the number of comparable non-base alternatives degrees of freedom

Under IIA the restricted estimator stays consistent and the difference is noise; a large ``H`` says dropping an alternative reshuffles the remaining odds. The test needs at least three categories left after the omission, so it requires ``J \geq 4``.

```@example ordmult
iia_tests = [(omitted = j,
              statistic = round(hausman_iia(m_mlogit; omit_category=j).statistic, digits=3),
              pvalue = round(hausman_iia(m_mlogit; omit_category=j).pvalue, digits=4))
             for j in 2:4]
```

None of the three admissible omissions comes close to rejecting: the statistics are 3.70, 5.22, and 0.28 on 12 degrees of freedom, with p-values of 0.99, 0.95, and 1.00. Dropping any single hours band leaves the odds among the remaining ones essentially unchanged, which is what IIA asserts and what makes the multinomial logit defensible here. Rejection would call for nested logit or mixed logit, which relax the independence of the alternative-specific errors. The statistic is clamped at zero because ``\hat V_r - \hat V_f`` can fail to be positive definite in finite samples, a well-known feature of Hausman-type tests.

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `T` | Hausman statistic, clamped at zero |
| `pvalue` | `T` | ``\chi^2`` p-value |
| `df` | `Int` | Degrees of freedom, ``K`` times the comparable non-base alternatives |
| `omitted_category` | any | Label of the omitted category, from `m.categories` |

---

## Prediction

`predict(m, X_new)` returns an ``n_{\text{new}} \times J`` matrix of category probabilities whose rows sum to one, for all three models. The columns of `X_new` must match the estimation matrix — no intercept for the ordered models, an intercept column for the multinomial.

```@example ordmult
profile = vec(mean(Xo, dims=1))          # kidslt6 is the last column
kids = permutedims(hcat([[profile[1:4]; k] for k in 0.0:2.0]...))
round.(predict(m_ologit, kids), digits=3)
```

Each row holds income, schooling, experience, and age at their sample means and varies only the number of preschool children. With no young child the woman is more likely to work than not: 34.2% chance of no work against a combined 65.8% across the three working bands, with the 1000-1999 hour band at 31.0%. One young child flips this to 66.3% non-participation, and two leaves 88.1%. The probability of full-time work falls from 7.9% to 2.2% to 0.6% — the ordered structure forces this monotone collapse across all higher categories at once, which is precisely the restriction the multinomial logit relaxes.

---

## Residuals

A ``J``-category response has no single scalar residual, so these models expose two distinct quantities. Which one to use is a question of what the diagnostic needs, not of convenience.

**Per-category residuals.** `residuals` returns an ``n \times J`` matrix — one column per outcome category — for ordered logit, ordered probit, and multinomial logit alike. With the indicator ``d_{ij} = 1\{y_i = j\}``:

```math
r_{ij} = d_{ij} - \hat P_{ij}
```

where:
- ``d_{ij}`` is 1 when observation ``i`` falls in category ``j`` and 0 otherwise
- ``\hat P_{ij}`` is the fitted probability of category ``j`` for observation ``i``

Rows sum to exactly zero. The `kind` keyword selects `:response` (the default, above), `:pearson` (``r_{ij}/\sqrt{\hat P_{ij}(1-\hat P_{ij})}``), or `:deviance`, whose total sum of squares equals the model deviance ``-2\hat\ell``.

!!! warning "The shape differs from the binary models"
    `residuals(::LogitModel)` and `residuals(::ProbitModel)` return a length-``n`` **vector** of deviance residuals. The ordered and multinomial versions return an ``n \times J`` **matrix**, because a ``J``-category response genuinely has ``J`` residuals per observation. Code written generically over binary models must handle this rather than assume a vector.

```@example ordmult
rd = residuals(m_ologit; kind=:deviance)
round.([sum(abs2, rd), -2 * loglikelihood(m_ologit)], digits=6)
```

**Generalized residuals.** For the ordered models, `generalized_residuals` returns the length-``n`` vector

```math
e_i = \frac{f(\alpha_{j-1} - x_i'\beta) - f(\alpha_j - x_i'\beta)}{P(y_i = j \mid x_i)}, \qquad j = y_i
```

where:
- ``\alpha_0 = -\infty`` and ``\alpha_J = +\infty``, so the boundary categories drop one term
- ``f`` is the logistic or standard normal density

Equivalently ``e_i = \partial \ell_i / \partial (x_i'\beta)``, the score of observation ``i``'s log-likelihood with respect to its own index — and for the probit case exactly ``E[\varepsilon_i \mid y_i, x_i]``. This is the vector that outer-product-of-gradients LM specification tests are built on (Chesher & Irish 1987; Gourieroux, Monfort, Renault & Trognon 1987). Because the ordered score with respect to ``\beta`` is ``X'e``, it is orthogonal to the regressors at the optimum:

```@example ordmult
e = generalized_residuals(m_ologit)
round(maximum(abs, m_ologit.X' * e), sigdigits=2)
```

The largest element of ``X'e`` is 4.5e-7, zero to the Newton convergence tolerance rather than to machine precision, which is the honest way to read a first-order condition at a numerical optimum. On a two-category fit ``e_i`` collapses to ``y_i - \hat p_i``, the familiar binary score residual — the sense in which it, and not the residual matrix, is the true analogue of the binary case.

`generalized_residuals` is deliberately **not** defined for multinomial logit: an unordered response has no single latent index, so no length-``n`` scalar score exists. Its score is ``X'(d_j - \hat P_j)`` per alternative, which is precisely the `:response` residual matrix above.

---

## Visualization

`plot_result` renders both model families as horizontal dot-and-whisker plots at 95% intervals, and the multinomial marginal effects as one facet per non-base alternative:

```julia
plot_result(m_ologit)     # slopes and cutpoints, two panels
plot_result(m_mlogit)     # one coefficient facet per non-base alternative
plot_result(me_m)         # marginal-effect facets with delta-method whiskers
```

The ordered figure separates the two parameter blocks because they live on different scales: the slopes measure how a regressor shifts the latent index, the cutpoints are positions *on* that index.

```@raw html
<iframe src="../assets/plots/ordered_coef.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The preschool-child coefficient of ``-1.328`` sits far to the left of every other slope, on an interval — ``[-1.672, -0.985]`` — five times wider than the next widest. At the opposite extreme `nwifeinc` is the one slope whose interval nearly touches zero, reaching ``-0.003`` at its upper end. The cutpoint panel shows the three thresholds ordered and well separated at ``-1.303``, ``-0.196`` and 1.811, on intervals roughly 2.4 units wide: the four bands are distinguishable, but the thresholds themselves are the least precisely estimated parameters in the model.

The multinomial figure draws one panel per alternative, each labelled with the band code and its base category:

```@raw html
<iframe src="../assets/plots/mlogit_coef.html" width="100%" height="620" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Reading a regressor across the three panels is the visual form of the parallel-regression question. Experience behaves exactly as the ordered model assumes, rising monotonically with the hours band (0.076, 0.137, 0.174), and schooling is flat across the three at 0.216 to 0.236 — a single common slope would lose nothing. Preschool children do not: the coefficient is ``-0.994`` for part-time work but ``-1.963`` for the 1000--1999 hour band and ``-1.813`` for full-time work, so the effect peaks in the middle of the ordering rather than growing with it. That non-monotonicity is what a single ordered slope cannot represent, and it is why the [Specification Tests](@ref specification-tests) above matter before the ordered fit is reported.

---

## Complete Example

An end-to-end labour-supply study: fit the ordered model, test the assumption it rests on, refit without that assumption, and compare what the two say.

```@example ordmult
# Step 1 — ordered logit on the four hours bands
fit_o = estimate_ologit(supply, Xo; varnames=onames)
report(fit_o)
```

```@example ordmult
# Step 2 — does the proportional-odds restriction hold?
bt_full = brant_test(fit_o)
(overall_p = round(bt_full.pvalue, digits=4),
 worst_variable = onames[argmin(bt_full.per_variable)],
 worst_p = round(minimum(bt_full.per_variable), digits=4))
```

```@example ordmult
# Step 3 — drop the ordering restriction entirely
fit_m = estimate_mlogit(supply, Xm; varnames=mnames)
(ordered = (loglik = round(fit_o.loglik, digits=2), params = length(coef(fit_o)) + length(fit_o.cutpoints),
            bic = round(fit_o.bic, digits=2)),
 multinomial = (loglik = round(fit_m.loglik, digits=2), params = length(coef(fit_m)),
                bic = round(fit_m.bic, digits=2)))
```

```@example ordmult
# Step 4 — is the multinomial fit internally consistent?
iia_4 = hausman_iia(fit_m; omit_category=4)
(statistic = round(iia_4.statistic, digits=3), pvalue = round(iia_4.pvalue, digits=4))
```

```@example ordmult
# Step 5 — the preschool-child effect from both models (row 5 in each:
# the multinomial effects carry no intercept row)
(ordered = round.(marginal_effects(fit_o).effects[5, :], digits=4),
 multinomial = round.(marginal_effects(fit_m).effects[5, :], digits=4))
```

The ordered logit buys its parsimony with the proportional-odds restriction, and the Brant test finds that restriction acceptable overall (``p = 0.113``) though strained by `educ` (``p = 0.033``). Relaxing it costs ten extra parameters for 8.4 log-likelihood points, and the BIC prefers the ordered model by roughly 49 points, so the restriction pays for itself here. The IIA test then clears the multinomial specification on its own terms (``p = 1.00`` when the full-time band is dropped), meaning the disagreement between the two models is about the ordering assumption alone, not about a failure of the softmax. Step 5 shows how little that disagreement amounts to for the quantity most often reported: the two models put the preschool-child effect on non-participation at 25.1 and 27.5 points, and differ mainly in how they split the remaining mass between the part-time and full-time bands.

---

## Common Pitfalls

1. **Putting an intercept in an ordered model's `X`.** The cutpoints already play that role. A constant column leaves the index unidentified, and the Newton step works against a singular Hessian. The multinomial logit is the opposite case — it needs an explicit intercept column, as `Xm` above supplies.

2. **Fewer than three categories.** Both families require ``J \geq 3`` and throw an `ArgumentError` otherwise. Use `estimate_logit` or `estimate_probit` from the [binary choice page](@ref binary_choice_page) for two-category outcomes.

3. **Assuming the Brant test blesses the specification.** The overall test can pass while a single covariate violates proportional odds, exactly as `educ` does here. Read `per_variable` before concluding: the joint statistic averages one genuine violation against four well-behaved regressors.

4. **Reading multinomial coefficients as effects on a probability.** ``\beta_{j,k}`` is the effect on the log-odds of alternative ``j`` against the base. A positive coefficient can coexist with a negative marginal effect on ``P(y = j)`` when a competing alternative rises faster. Always compute `marginal_effects(m)` before interpreting.

5. **Building marginal-effect standard errors from the reported coefficients alone.** Every coefficient enters an AME through the fitted probabilities, intercepts included, so a delta method restricted to the covariance of the reported rows understates the variance. `marginal_effects` propagates the full covariance for the multinomial and the joint ``[\beta; \alpha]`` covariance for the ordered models. `se` is `nothing` only when the model covariance is unavailable.

6. **Running the IIA test with too few categories.** `hausman_iia` re-estimates on the subsample without the omitted category and needs at least three categories to remain, so a four-category outcome is the minimum. It also silently drops every observation in the omitted category, so a large omitted category leaves a much smaller restricted sample.

7. **Treating an ordered outcome's category codes as cardinal.** The models use only the ordering of `categories`, which is the sorted vector of the original labels. Banding a continuous variable, as `supply` bands hours, discards within-band variation; when the underlying variable is observed, a model for the level (or a Tobit for a corner solution at zero) uses more of it.

---

## References

- Agresti, A. (2010). *Analysis of Ordinal Categorical Data*. 2nd ed.
  Hoboken, NJ: Wiley. ISBN 978-0-470-08289-8.

- Brant, R. (1990). Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression.
  *Biometrics*, 46(4), 1171--1178. [DOI](https://doi.org/10.2307/2532457)

- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*.
  Cambridge: Cambridge University Press. ISBN 978-0-521-84805-3.

- Chesher, A., & Irish, M. (1987). Residual Analysis in the Grouped and Censored Normal Linear Model.
  *Journal of Econometrics*, 34(1--2), 33--61. [DOI](https://doi.org/10.1016/0304-4076(87)90066-2)

- Gourieroux, C., Monfort, A., Renault, E., & Trognon, A. (1987). Generalised Residuals.
  *Journal of Econometrics*, 34(1--2), 5--32. [DOI](https://doi.org/10.1016/0304-4076(87)90065-0)

- Greene, W. H. (2012). *Econometric Analysis*. 7th ed.
  Boston: Prentice Hall. ISBN 978-0-131-39538-1.

- Hausman, J. A., & McFadden, D. (1984). Specification Tests for the Multinomial Logit Model.
  *Econometrica*, 52(5), 1219--1240. [DOI](https://doi.org/10.2307/1910997)

- McCullagh, P. (1980). Regression Models for Ordinal Data.
  *Journal of the Royal Statistical Society: Series B*, 42(2), 109--127. [DOI](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x)

- McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior.
  In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105--142). New York: Academic Press.

- Mroz, T. A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions.
  *Econometrica*, 55(4), 765--799. [DOI](https://doi.org/10.2307/1911029)

- Train, K. E. (2009). *Discrete Choice Methods with Simulation*. 2nd ed.
  Cambridge: Cambridge University Press. ISBN 978-0-521-74738-7.

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed.
  Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.
