# [Binary Choice Models](@id binary_choice_page)

**MacroEconometricModels.jl** estimates logit and probit models for binary dependent variables by maximum likelihood, using iteratively reweighted least squares (IRLS). Both estimators produce Stata/EViews-style coefficient tables through `report()` and feed the package's marginal-effects, odds-ratio, classification, and D3.js visualization infrastructure. For ordinal and unordered categorical outcomes, see [Ordered & Multinomial Models](@ref ordered_multinomial_page); for the linear-probability alternative, see [Linear Regression](@ref regression_page); for panel versions with fixed or random effects, see [Panel Regression](@ref panel_reg_page).

- **Logit** (logistic link) estimated by IRLS/Fisher scoring (McCullagh & Nelder 1989)
- **Probit** (standard normal link) with latent-variable interpretation (Wooldridge 2010)
- **Marginal effects**: average (AME), at-mean (MEM), and at-representative (MER) with delta-method standard errors (Cameron & Trivedi 2005)
- **Discrete change** for ``\{0,1\}`` regressors, matching Stata's `margins, dydx(*)` treatment of factor variables
- **Odds ratios** with confidence intervals built on the log-odds scale (Agresti 2002)
- **Classification table**: confusion matrix, accuracy, sensitivity, specificity, precision, F1
- **Robust inference**: HC0--HC3 and cluster-robust sandwich covariance
- **Separation detection**: automatic warning when the MLE does not exist (Albert & Anderson 1984)
- **StatsAPI interface**: `coef`, `vcov`, `predict`, `confint`, `stderror`, `nobs`, `loglikelihood`

```@setup binary
using MacroEconometricModels, Statistics
mroz = load_example(:mroz)
```

All examples use the Mroz (1987) extract shipped with the package: 753 married women from the 1976 Panel Study of Income Dynamics, of whom 428 participated in the labour force in 1975. The outcome `inlf` equals 1 for participants; the regressors are non-wife household income in thousands (`nwifeinc`), years of schooling (`educ`), labour-market experience and its square (`exper`, `expersq`), `age`, and the counts of children under six (`kidslt6`) and aged six to eighteen (`kidsge6`).

## Quick Start

**Recipe 1: Logit from a `CrossSectionData` container**

```@example binary
m_logit = estimate_logit(mroz, :inlf,
                         [:nwifeinc, :educ, :exper, :expersq, :age, :kidslt6, :kidsge6])
report(m_logit)
```

**Recipe 2: Probit on the same specification**

```@example binary
m_probit = estimate_probit(mroz, :inlf,
                           [:nwifeinc, :educ, :exper, :expersq, :age, :kidslt6, :kidsge6])
report(m_probit)
```

**Recipe 3: Average marginal effects**

```@example binary
report(marginal_effects(m_logit))
```

**Recipe 4: Odds ratios**

```@example binary
report(odds_ratio(m_logit))
```

**Recipe 5: Classification performance**

```@example binary
ct = classification_table(m_logit)
(accuracy = round(ct["accuracy"], digits=3),
 sensitivity = round(ct["sensitivity"], digits=3),
 specificity = round(ct["specificity"], digits=3),
 f1 = round(ct["f1_score"], digits=3))
```

**Recipe 6: Heteroskedasticity-robust standard errors**

```@example binary
m_hc1 = estimate_logit(mroz, :inlf,
                       [:nwifeinc, :educ, :exper, :expersq, :age, :kidslt6, :kidsge6];
                       cov_type=:hc1)
round.(stderror(m_hc1) ./ stderror(m_logit), digits=3)
```

---

## Logit Model

The **logistic regression model** relates a binary outcome ``y_i \in \{0, 1\}`` to a ``k \times 1`` regressor vector ``x_i`` through the logistic cumulative distribution function (McCullagh & Nelder 1989):

```math
P(y_i = 1 \mid x_i) = \Lambda(x_i' \beta) = \frac{1}{1 + \exp(-x_i' \beta)}
```

where:
- ``y_i`` is the binary dependent variable (0 or 1)
- ``x_i`` is the ``k \times 1`` vector of regressors, including a constant
- ``\beta`` is the ``k \times 1`` vector of coefficients
- ``\Lambda(\cdot)`` is the logistic CDF

The logistic function maps the linear index ``x_i' \beta \in (-\infty, \infty)`` into ``(0, 1)``. Because the map is nonlinear, ``\beta_j`` is not the effect of ``x_j`` on ``P(y = 1)`` — see [Marginal Effects](@ref marginal-effects).

The log-likelihood is

```math
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log \Lambda(x_i' \beta) + (1 - y_i) \log (1 - \Lambda(x_i' \beta)) \right]
```

where:
- ``n`` is the number of observations
- ``\Lambda(x_i' \beta)`` is the fitted probability for observation ``i``

This function is globally concave, so the maximum is unique whenever it exists.

The logit model has no natural ``R^2``. McFadden (1974) proposes the likelihood-ratio index

```math
R^2_{\text{McF}} = 1 - \frac{\ell(\hat{\beta})}{\ell(\hat{\beta}_0)}
```

where:
- ``\ell(\hat{\beta})`` is the maximized log-likelihood of the full model
- ``\ell(\hat{\beta}_0)`` is the log-likelihood of the intercept-only model, ``n[\bar{y}\log\bar{y} + (1-\bar{y})\log(1-\bar{y})]``

Values of 0.2 to 0.4 signal a strong fit (McFadden 1974); the index never reaches 1 in finite samples.

!!! note "Technical Note"
    The estimator is **iteratively reweighted least squares** (Fisher scoring). Each iteration solves a weighted least squares problem with weights ``W = \text{diag}(\hat{\mu}_i (1 - \hat{\mu}_i))`` and working response ``z_i = \hat{\eta}_i + (y_i - \hat{\mu}_i) / (\hat{\mu}_i (1 - \hat{\mu}_i))``, updating ``\hat{\beta}^{(t+1)} = (X' W^{(t)} X)^{-1} X' W^{(t)} z^{(t)}``. Iteration stops when ``|\ell^{(t+1)} - \ell^{(t)}| < \texttt{tol} \cdot (|\ell^{(t)}| + 1)``. Starting from ``\beta = 0``, six iterations suffice for the Mroz data.

The matrix interface takes the response vector and a regressor matrix that must already contain the constant column:

```@example binary
y = mroz[:, "inlf"]
X = hcat(ones(mroz.N_obs), mroz[:, "nwifeinc"], mroz[:, "educ"], mroz[:, "exper"],
         mroz[:, "expersq"], mroz[:, "age"], mroz[:, "kidslt6"], mroz[:, "kidsge6"])
xnames = ["(Intercept)", "nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]

m = estimate_logit(y, X; varnames=xnames)
report(m)
```

The fit reproduces Wooldridge (2010, Example 17.1) and is numerically identical to the symbol-based fit in Recipe 1. Each additional year of schooling raises the log-odds of participation by 0.221 and each additional year of experience by 0.206, with experience entering concavely (`expersq` = ``-0.0032``, ``p = 0.002``). A child under six cuts the log-odds by 1.443 — by far the largest effect in the model — whereas school-age children carry no detectable association (``p = 0.42``). The McFadden index of 0.220 sits in the range McFadden (1974) associates with a strong fit, and the null log-likelihood of ``-514.87`` reflects the 56.8% participation rate that an intercept-only model can already match.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `cov_type` | `Symbol` | `:ols` | Covariance estimator: `:ols` (information matrix), `:hc0`, `:hc1`, `:hc2`, `:hc3`, `:cluster` |
| `varnames` | `Union{Nothing,Vector{String}}` | `nothing` | Coefficient names (`"x1"`, `"x2"`, … if `nothing`) |
| `clusters` | `Union{Nothing,AbstractVector}` | `nothing` | Cluster assignments (required for `:cluster`) |
| `maxiter` | `Int` | `100` | Maximum IRLS iterations |
| `tol` | `Real` | ``10^{-8}`` | Relative convergence tolerance on the log-likelihood |

### Return Values

`estimate_logit` returns a `LogitModel{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Vector{T}` | Binary dependent variable (0/1) |
| `X` | `Matrix{T}` | ``n \times k`` regressor matrix |
| `beta` | `Vector{T}` | ``k \times 1`` estimated coefficients |
| `vcov_mat` | `Matrix{T}` | ``k \times k`` variance-covariance matrix |
| `residuals` | `Vector{T}` | Deviance residuals |
| `fitted` | `Vector{T}` | Fitted probabilities ``\hat{P}(y_i = 1 \mid x_i)`` |
| `loglik` | `T` | Maximized log-likelihood |
| `loglik_null` | `T` | Intercept-only log-likelihood |
| `pseudo_r2` | `T` | McFadden's pseudo ``R^2`` |
| `aic` | `T` | Akaike information criterion |
| `bic` | `T` | Bayesian information criterion |
| `varnames` | `Vector{String}` | Coefficient names |
| `converged` | `Bool` | Whether IRLS met the tolerance |
| `iterations` | `Int` | IRLS iterations performed |
| `cov_type` | `Symbol` | Covariance estimator used |

---

## Probit Model

The **probit model** replaces the logistic CDF with the standard normal CDF ``\Phi(\cdot)`` (Wooldridge 2010, ch. 15):

```math
P(y_i = 1 \mid x_i) = \Phi(x_i' \beta) = \int_{-\infty}^{x_i' \beta} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{t^2}{2}\right) dt
```

where:
- ``\Phi(\cdot)`` is the standard normal CDF
- ``x_i`` and ``\beta`` are defined as in the logit model

The model follows from a latent-variable specification: ``y_i^* = x_i' \beta + \varepsilon_i`` with ``\varepsilon_i \sim N(0, 1)``, observed as ``y_i = \mathbf{1}(y_i^* > 0)``. Fisher scoring uses weights ``w_i = \phi(\hat{\eta}_i)^2 / [\hat{\mu}_i (1 - \hat{\mu}_i)]``, where ``\phi(\cdot)`` is the standard normal density.

!!! note "Technical Note"
    Logit and probit coefficients differ by a scale factor because the logistic distribution has variance ``\pi^2/3`` and the standard normal has variance 1. The rule of thumb ``\beta_{\text{probit}} \approx \beta_{\text{logit}} / 1.6`` holds well over the range of the linear index (Amemiya 1981). Marginal effects and fitted probabilities from the two links are nearly identical; choose logit when odds ratios are the target and probit when the latent-variable or normality structure is substantive.

```@example binary
m_p = estimate_probit(y, X; varnames=xnames)
report(m_p)
```

```@example binary
round.(coef(m) ./ coef(m_p), digits=2)
```

Every slope ratio lands between 1.66 and 1.78, bracketing the 1.6 rule of thumb; the intercept ratio (1.58) is the least stable because the intercept is imprecisely estimated under both links. The two log-likelihoods differ by less than half a point (``-401.77`` against ``-401.30``), and the probit McFadden index (0.221) edges out the logit one (0.220) by too little to select between the links. The substantive conclusions are identical: schooling and experience raise participation, while income, age, and preschool children lower it.

`estimate_probit` accepts the same keyword arguments as `estimate_logit` and returns a `ProbitModel{T}` whose fields match `LogitModel{T}` one for one.

---

## [Marginal Effects](@id marginal-effects)

The coefficient ``\beta_j`` is not the marginal effect of ``x_j`` on ``P(y = 1)``. For a continuous regressor the marginal effect is (Cameron & Trivedi 2005, ch. 14)

```math
\frac{\partial P(y_i = 1 \mid x_i)}{\partial x_j} = f(x_i' \beta) \cdot \beta_j
```

where:
- ``f(\cdot)`` is the density of the link: ``f(\eta) = \Lambda(\eta)(1 - \Lambda(\eta))`` for logit, ``f(\eta) = \phi(\eta)`` for probit
- the effect varies across observations, because ``f(x_i' \beta)`` depends on ``x_i``

**Average marginal effects** (AME) average the derivative over the sample:

```math
\text{AME}_j = \frac{1}{n} \sum_{i=1}^{n} f(x_i' \hat{\beta}) \cdot \hat{\beta}_j
```

where:
- ``n`` is the number of observations
- ``f(x_i' \hat{\beta})`` is the link density at observation ``i``'s fitted index

**Marginal effects at the mean** (MEM) evaluate at the mean regressor vector, and **marginal effects at representative values** (MER) at a user-chosen point:

```math
\text{MEM}_j = f(\bar{x}' \hat{\beta}) \cdot \hat{\beta}_j, \qquad
\text{MER}_j = f(x_0' \hat{\beta}) \cdot \hat{\beta}_j
```

where:
- ``\bar{x}`` is the ``k \times 1`` vector of sample means
- ``x_0`` equals ``\bar{x}`` with the entries named in `at` overridden

AME dominates modern applied work because it needs no arbitrary evaluation point; MEM and MER answer questions about one specific covariate profile.

!!! note "Technical Note"
    Standard errors come from the **delta method** (Oehlert 1992). With ``g(\hat{\beta})`` the vector of marginal effects and ``G = \partial g / \partial \beta'`` its Jacobian, ``\text{Var}(\hat{g}) \approx G \hat{V} G'``. For AME the Jacobian is ``G_{j,l} = \frac{1}{n} \sum_{i=1}^{n} [\mathbf{1}(j = l) f_i + f'_i \hat{\beta}_j x_{il}]`` with ``f_i = f(x_i' \hat{\beta})``. The intercept row is identically zero, so its reported effect, standard error, and interval are `NaN`.

```@example binary
report(marginal_effects(m))
```

```@example binary
report(marginal_effects(m; type=:mem))
```

```@example binary
report(marginal_effects(m; type=:mer, at=Dict(7 => 1.0)))
```

An additional preschool child lowers the participation probability by 25.8 percentage points on average (AME), by 35.1 points for the woman at the sample mean (MEM), and by 31.3 points for a woman who already has exactly one preschool child (MER, column 7 set to 1). The three numbers differ because the logistic density peaks at ``\hat{P} = 0.5``: the mean woman sits near that peak and shows the steepest slope, while a mother of a young child has a fitted probability near 0.32 and a flatter one. An extra year of schooling raises participation by 4.0 points and an extra year of experience by 3.7 points on average, both significant at the 0.1% level, while `kidsge6` stays insignificant under all three conventions.

For a ``\{0,1\}`` regressor, `marginal_effects` reports the **discrete change** ``F(\eta \mid x_j = 1) - F(\eta \mid x_j = 0)`` rather than a derivative, matching Stata's factor-variable convention. Replacing the count of preschool children with an indicator makes the difference visible:

```@example binary
anykids = Float64.(mroz[:, "kidslt6"] .> 0)
X_d = hcat(X[:, 1:6], anykids, X[:, 8])
m_d = estimate_logit(y, X_d; varnames=[xnames[1:6]; "anykids"; "kidsge6"])
report(marginal_effects(m_d))
```

Having any child under six lowers the participation probability by 31.5 points — the average gap between mothers of preschoolers and otherwise identical women, not a per-child derivative. The count specification above reports 25.8 points per child; the two answer different questions, and the indicator version is the one to quote when the policy margin is "any young child at home".

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | `Symbol` | `:ame` | `:ame`, `:mem`, or `:mer` |
| `at` | `Union{Nothing,Dict}` | `nothing` | Evaluation point for `:mer`, mapping column index to value (required when `type=:mer`) |
| `conf_level` | `Real` | `0.95` | Confidence level for the reported intervals |

### Return Values

`marginal_effects` returns a `MarginalEffects{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `effects` | `Vector{T}` | Marginal effect per regressor (`NaN` for the intercept) |
| `se` | `Vector{T}` | Delta-method standard errors |
| `z_stat` | `Vector{T}` | z-statistics, effect divided by standard error |
| `p_values` | `Vector{T}` | Two-sided normal p-values |
| `ci_lower` | `Vector{T}` | Lower confidence bounds |
| `ci_upper` | `Vector{T}` | Upper confidence bounds |
| `varnames` | `Vector{String}` | Variable names |
| `type` | `Symbol` | `:ame`, `:mem`, or `:mer` |
| `conf_level` | `T` | Confidence level used |

---

## Odds Ratios

The **odds ratio** gives logit coefficients a multiplicative reading (Agresti 2002, ch. 5):

```math
\text{OR}_j = \exp(\hat{\beta}_j)
```

where:
- ``\hat{\beta}_j`` is the estimated log-odds coefficient for regressor ``j``
- ``\text{OR}_j > 1`` means a one-unit rise in ``x_j`` raises the odds of ``y = 1``, ``\text{OR}_j < 1`` lowers them, and ``\text{OR}_j = 1`` means no association

The delta method gives ``\text{SE}(\text{OR}_j) = \text{OR}_j \cdot \text{SE}(\hat{\beta}_j)``, but the interval is built on the log-odds scale and then exponentiated, which keeps coverage correct and the bounds positive:

```math
\text{CI} = \left[ \exp(\hat{\beta}_j - z_{\alpha/2} \, \text{SE}(\hat{\beta}_j)), \;\; \exp(\hat{\beta}_j + z_{\alpha/2} \, \text{SE}(\hat{\beta}_j)) \right]
```

where:
- ``z_{\alpha/2}`` is the standard normal critical value at the chosen `conf_level`
- ``\text{SE}(\hat{\beta}_j)`` is the standard error of the log-odds coefficient

!!! note "Technical Note"
    Odds ratios are defined for the logistic link alone, so `odds_ratio` accepts only `LogitModel` inputs; the count-model analogue is `incidence_rate_ratio`. For probit models, report marginal effects instead. The printed table omits the intercept, whose odds at ``x = 0`` are rarely interpretable.

```@example binary
or = odds_ratio(m)
report(or)
```

A year of schooling multiplies the odds of participation by 1.248 and a year of experience by 1.229, both intervals sitting comfortably above 1. Each preschool child multiplies the odds by 0.236 — a 76% reduction — with an interval of ``[0.158, 0.352]`` that never approaches 1. Non-wife income and age each shave a little off the odds per unit (0.979 per thousand dollars, 0.916 per year), while `kidsge6` spans 1 and stays insignificant. Odds ratios are multiplicative in the odds, never in the probability: 0.236 does not mean the participation probability falls by 76%.

### Return Values

`odds_ratio` returns an `OddsRatio{T}`, which both `report` and `plot_result` accept:

| Field | Type | Description |
|-------|------|-------------|
| `or` | `Vector{T}` | Odds ratios ``\exp(\hat{\beta}_j)`` |
| `se` | `Vector{T}` | Delta-method standard errors |
| `ci_lower` | `Vector{T}` | Lower bounds ``\exp(\hat\beta_j - z_{\alpha/2}\,\text{SE})`` |
| `ci_upper` | `Vector{T}` | Upper bounds ``\exp(\hat\beta_j + z_{\alpha/2}\,\text{SE})`` |
| `varnames` | `Vector{String}` | Variable names |
| `conf_level` | `T` | Confidence level used (keyword `conf_level`, default `0.95`) |

---

## Classification Table

The **classification table** compares predicted classes, obtained by thresholding the fitted probabilities, with the observed outcomes (Agresti 2002, ch. 5):

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | TN (true negative) | FP (false positive) |
| **Actual 1** | FN (false negative) | TP (true positive) |

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | ``(\text{TP} + \text{TN}) / n`` | Overall correct classification rate |
| **Sensitivity** | ``\text{TP} / (\text{TP} + \text{FN})`` | True positive rate (recall) |
| **Specificity** | ``\text{TN} / (\text{TN} + \text{FP})`` | True negative rate |
| **Precision** | ``\text{TP} / (\text{TP} + \text{FP})`` | Positive predictive value |
| **F1 Score** | ``2 \cdot \text{Prec} \cdot \text{Sens} / (\text{Prec} + \text{Sens})`` | Harmonic mean of precision and sensitivity |

```@example binary
classification_table(m)["confusion"]
```

```@example binary
function threshold_row(t)
    c = classification_table(m; threshold=t)
    (threshold = t,
     accuracy = round(c["accuracy"], digits=3),
     sensitivity = round(c["sensitivity"], digits=3),
     specificity = round(c["specificity"], digits=3))
end
threshold_row.((0.3, 0.5, 0.7))
```

At the default threshold the model classifies 554 of 753 women correctly (73.6%), splitting its errors between 118 non-participants predicted to work and 81 participants predicted not to. The threshold trades the two error types against each other: at 0.3 sensitivity climbs to 0.928 while specificity collapses to 0.385, and at 0.7 specificity reaches 0.855 at the cost of missing 44% of the actual participants. Accuracy peaks near 0.5 here only because the sample is nearly balanced; with a rare outcome the 0.5 rule is beaten by always predicting zero, which is why sensitivity and specificity must be read alongside accuracy.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `threshold` | `Real` | `0.5` | Probability above which an observation is classified as 1 |

### Return Values

`classification_table` returns a `Dict{String,Any}`:

| Key | Type | Description |
|-----|------|-------------|
| `"confusion"` | `Matrix{T}` | ``2 \times 2`` matrix ``[\text{TN}\;\text{FP}; \text{FN}\;\text{TP}]`` |
| `"accuracy"` | `T` | Overall correct classification rate |
| `"sensitivity"` | `T` | True positive rate (recall) |
| `"specificity"` | `T` | True negative rate |
| `"precision"` | `T` | Positive predictive value |
| `"f1_score"` | `T` | Harmonic mean of precision and sensitivity |
| `"n"` | `Int` | Number of observations |
| `"threshold"` | `T` | Threshold used |

---

## Robust and Cluster-Robust Inference

With `cov_type=:ols` the covariance is the inverse information matrix ``(X'WX)^{-1}``, efficient when the model is correctly specified. The other options build the sandwich

```math
\hat{V} = (X'WX)^{-1} \, \hat{S} \, (X'WX)^{-1}
```

where:
- ``W`` is the diagonal matrix of IRLS weights at the optimum
- ``\hat{S}`` is a meat matrix built from score residuals: ``y_i - \hat\mu_i`` for logit, and ``\phi(\hat\eta_i)(y_i - \hat\mu_i) / [\hat\mu_i(1-\hat\mu_i)]`` for probit
- HC1 rescales by ``n/(n-k)``; HC2 and HC3 divide the squared score by ``(1 - h_{ii})`` and ``(1 - h_{ii})^2`` with leverages ``h_{ii} = w_i x_i'(X'WX)^{-1}x_i``; `:cluster` sums scores within each group before taking outer products

!!! warning "Cluster count, not sample size, drives cluster-robust inference"
    The Mroz extract records only seven distinct county unemployment rates, so clustering on the local labour market leaves ``G = 7`` groups. Cluster-robust asymptotics need many clusters; with `G` this small the standard errors are themselves noisy and can fall well below the independent-observation values, as they do here. Read the example as a demonstration of the API, and use a wild cluster bootstrap when clusters are few (Cameron & Miller 2015).

```@example binary
unem = mroz[:, "unem"]   # county unemployment rate: seven local labour markets
ses = (ols = stderror(m),
       hc1 = stderror(estimate_logit(y, X; cov_type=:hc1, varnames=xnames)),
       hc3 = stderror(estimate_logit(y, X; cov_type=:hc3, varnames=xnames)),
       cluster = stderror(estimate_logit(y, X; cov_type=:cluster, clusters=unem, varnames=xnames)))
map(v -> round.(v, digits=4), ses)
```

The heteroskedasticity-robust standard errors barely move: HC1 raises the `educ` standard error from 0.0434 to 0.0447, and HC3, which inflates high-leverage observations most, only reaches 0.0451. That is the expected pattern when a binary model is well specified, because the conditional variance ``\mu_i(1-\mu_i)`` is then correct by construction. Clustering on the seven labour markets moves the numbers much further — `educ` rises by half to 0.0655 while `kidslt6` falls from 0.2036 to 0.1198 — which is exactly the instability the warning above describes.

---

## Data Interfaces

Both estimators accept either a response vector with a regressor matrix, or a `CrossSectionData` container with symbol arguments. The container dispatch looks the columns up by name, **prepends the intercept automatically**, and labels it `(Intercept)`; the matrix interface leaves the constant column to the caller. Keyword arguments pass through unchanged.

```@example binary
m_sym = estimate_logit(mroz, :inlf, [:educ, :exper, :age])
p_hat = predict(m_sym, [1.0 12.0 10.0 42.0; 1.0 16.0 10.0 42.0])
round.(p_hat, digits=3)
```

`predict(m, X_new)` returns fitted probabilities for new rows, whose columns must line up with the estimation matrix — including the leading intercept. Holding experience at 10 years and age at 42, this three-regressor fit gives a high-school graduate a participation probability of 0.562 and a college graduate 0.701, a gap of 13.9 points across four years of schooling.

---

## Visualization

`plot_result` renders a logit or probit fit as a two-panel adequacy screen --- the fitted probabilities sorted and coloured by the observed outcome, above their distribution by outcome group --- marginal effects as a horizontal coefficient plot with confidence bars, and odds ratios as a forest plot on a log axis with its reference line at 1. Every figure below is the Mroz fit `m` estimated above; the marginal-effects panel omits the intercept, whose effect is `NaN`.

```julia
p = plot_result(m)
save_plot(p, "reg_logit.html")
```

```@raw html
<iframe src="../assets/plots/reg_logit.html" style="width:100%;height:420px;border:none;"></iframe>
```

```julia
p = plot_result(m_p)
save_plot(p, "reg_probit.html")
```

```@raw html
<iframe src="../assets/plots/reg_probit.html" style="width:100%;height:420px;border:none;"></iframe>
```

```julia
p = plot_result(marginal_effects(m))
save_plot(p, "reg_marginal_effects.html")
```

```@raw html
<iframe src="../assets/plots/reg_marginal_effects.html" style="width:100%;height:380px;border:none;"></iframe>
```

```julia
p = plot_result(odds_ratio(m))
save_plot(p, "odds_ratio_forest.html")
```

```@raw html
<iframe src="../assets/plots/odds_ratio_forest.html" style="width:100%;height:440px;border:none;"></iframe>
```

---

## Complete Example

A full participation study: estimate both links with robust standard errors, compare their marginal effects, read the odds, check classification performance, and predict for two schooling profiles.

```@example binary
# Step 1 — estimate both links on the Mroz specification
fit_logit  = estimate_logit(y, X; cov_type=:hc1, varnames=xnames)
fit_probit = estimate_probit(y, X; cov_type=:hc1, varnames=xnames)
report(fit_logit)
```

```@example binary
# Step 2 — the two links agree on what matters: the marginal effects
ame_logit  = marginal_effects(fit_logit)
ame_probit = marginal_effects(fit_probit)
(logit = round.(ame_logit.effects[2:end], digits=4),
 probit = round.(ame_probit.effects[2:end], digits=4))
```

```@example binary
# Step 3 — multiplicative reading of the logit fit
report(odds_ratio(fit_logit))
```

```@example binary
# Step 4 — in-sample classification at the default threshold
ct_full = classification_table(fit_logit)
(accuracy = round(ct_full["accuracy"], digits=3),
 sensitivity = round(ct_full["sensitivity"], digits=3),
 specificity = round(ct_full["specificity"], digits=3),
 precision = round(ct_full["precision"], digits=3))
```

```@example binary
# Step 5 — predicted participation for two schooling profiles, all else at the mean
profile = vec(mean(X, dims=1))
high_school = copy(profile); high_school[3] = 12.0
college     = copy(profile); college[3]     = 16.0
round.(predict(fit_logit, permutedims(hcat(high_school, college))), digits=3)
```

```julia
plot_result(fit_logit)
plot_result(ame_logit)
```

The logit and probit average marginal effects agree to the third decimal on every regressor — 0.0395 against 0.0394 for schooling, ``-0.2578`` against ``-0.2612`` for preschool children — which is the practical justification for treating the choice of link as a matter of taste. The odds-ratio table restates the same fit multiplicatively: schooling multiplies the participation odds by 1.25 per year, each preschool child by 0.24. In sample the model calls 73.6% of the participation decisions correctly, with sensitivity 0.811 and specificity 0.637. The profile predictions put a college graduate 19.3 points above a high-school graduate (0.760 against 0.567) with every other characteristic held at the sample mean — a gap much larger than four times the 4.0-point AME, because the four extra years are compounded at a point where the logistic density is near its peak.

---

## Common Pitfalls

1. **Perfect or quasi-perfect separation.** When a linear combination of the regressors separates the outcomes, the MLE does not exist and the coefficients diverge. `estimate_logit` and `estimate_probit` run the Albert & Anderson (1984) check after IRLS and warn when the linear index perfectly orders the outcomes, a fitted probability is pinned to 0 or 1, or a coefficient exceeds ``10^3`` in absolute value. Do not report a fit that warns: drop the offending regressor, pool categories, or use penalized estimation (Firth 1993).

2. **Missing values in the regressors.** The estimators reject any `NaN` or `Inf` with an `ArgumentError` naming the offending rows. In the Mroz extract `wage` and `lwage` are `NaN` for the 325 non-participants, so a specification using either must first drop those rows with `dropna(mroz; vars=["lwage"])` — and doing so conditions the sample on participation, which is exactly the selection problem the Heckman model addresses.

3. **Forgetting the intercept in the matrix interface.** `estimate_logit(y, X)` uses `X` verbatim. Without a column of ones the index is forced through the origin and every coefficient absorbs the omitted constant. The `CrossSectionData` dispatch adds the intercept for you; never supply one as well, or the design matrix becomes singular.

4. **Reading coefficients as marginal effects.** ``\hat\beta_j`` is a log-odds effect under logit and a latent-index effect under probit. Neither is a change in probability. Call `marginal_effects(m)` whenever the quantity of interest is a probability.

5. **Confusing AME with MEM.** MEM evaluates the derivative at ``\bar{x}``, a point that may describe nobody in the sample — a household with 0.24 preschool children. AME averages across the observed covariate distribution and is the modern default. The two agree only where ``f(x_i'\beta)`` is linear in ``x``, which never holds exactly.

6. **Comparing logit and probit coefficients directly.** The scales differ by roughly 1.6. Compare marginal effects or fitted probabilities instead, as the Complete Example does.

7. **Interpreting an odds ratio as a probability ratio.** An odds ratio of 2 doubles the odds, not the probability: from a baseline probability of 0.5 it moves the probability to 0.67, and from 0.9 to 0.95. Report marginal effects when the audience thinks in probabilities.

8. **Comparing pseudo ``R^2`` across samples.** McFadden's index is bounded well below 1 and depends on the null log-likelihood, which changes with the outcome frequency. Use likelihood-ratio tests, AIC, or BIC to compare specifications on the same data.

9. **Taking the default classification threshold as optimal.** The 0.5 rule is optimal only when false positives and false negatives cost the same and the classes are balanced. Sweep the threshold, as above, and pick the operating point matching the loss the application actually faces.

---

## References

- Agresti, A. (2002). *Categorical Data Analysis*. 2nd ed.
  New York: Wiley. ISBN 978-0-471-36093-3.

- Albert, A., & Anderson, J. A. (1984). On the Existence of Maximum Likelihood Estimates in Logistic Regression Models.
  *Biometrika*, 71(1), 1--10. [DOI](https://doi.org/10.1093/biomet/71.1.1)

- Amemiya, T. (1981). Qualitative Response Models: A Survey.
  *Journal of Economic Literature*, 19(4), 1483--1536.

- Cameron, A. C., & Miller, D. L. (2015). A Practitioner's Guide to Cluster-Robust Inference.
  *Journal of Human Resources*, 50(2), 317--372. [DOI](https://doi.org/10.3368/jhr.50.2.317)

- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*.
  Cambridge: Cambridge University Press. ISBN 978-0-521-84805-3.

- Firth, D. (1993). Bias Reduction of Maximum Likelihood Estimates.
  *Biometrika*, 80(1), 27--38. [DOI](https://doi.org/10.1093/biomet/80.1.27)

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*. 2nd ed.
  London: Chapman & Hall. ISBN 978-0-412-31760-6.

- McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior.
  In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105--142). New York: Academic Press.

- Mroz, T. A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions.
  *Econometrica*, 55(4), 765--799. [DOI](https://doi.org/10.2307/1911029)

- Oehlert, G. W. (1992). A Note on the Delta Method.
  *The American Statistician*, 46(1), 27--29. [DOI](https://doi.org/10.1080/00031305.1992.10475842)

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed.
  Cambridge, MA: MIT Press. ISBN 978-0-262-23258-6.
