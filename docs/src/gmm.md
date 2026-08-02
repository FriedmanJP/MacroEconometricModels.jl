# [Generalized & Simulated Method of Moments](@id gmm_page)

**MacroEconometricModels.jl** provides a flexible framework for **Generalized Method of Moments (GMM)** estimation (Hansen 1982) and its simulation-based counterpart, the **Simulated Method of Moments (SMM)** (Lee & Ingram 1991; Ruge-Murcia 2012). Any model that implies a set of population moment conditions ``\mathbb{E}[g(\theta_0)] = 0`` can be estimated by minimizing a quadratic form in the sample moments.

- **`estimate_gmm`**: one-step, optimal, two-step, and iterated (continuously-updated) GMM
- **Optimal weighting**: HAC (Newey-West/Bartlett) long-run covariance for serially-correlated moments
- **Hansen's J-test**: overidentification test with a valid ``\chi^2`` limit under efficient weighting
- **Model & moment selection**: Andrews-Lu (2001) MMSC criteria
- **`estimate_smm`**: parameter estimation when moments are available only through simulation
- **Linear GMM utilities**: closed-form solvers and robust sandwich covariances for IV-type models
- **StatsAPI interface**: `coef`, `vcov`, `stderror`, `confint`, `nobs`, and `report` for both estimators

This page documents the general-purpose GMM/SMM surface. For Local Projections estimated by GMM see [Local Projections](@ref lp_page); for DSGE parameter estimation with `method=:smm` see [DSGE Estimation](@ref dsge_estimation).

```@setup gmm
using MacroEconometricModels, Random, LinearAlgebra, Statistics
```

## Quick Start

**Recipe 1: Overidentified instrumental-variables GMM**

```@example gmm
Random.seed!(400)
n = 500
Z = randn(n, 3)                        # three instruments
u = randn(n)                           # structural error
X = Z * [0.5, 0.3, 0.2] .+ 0.5 .* u    # endogenous regressor (correlated with u)
y = 2.0 .* X .+ u                      # true slope β = 2
data = hcat(y, X, Z)

# Moment conditions: E[Z_t (y_t − X_t β)] = 0  →  3 moments, 1 parameter
iv_moments(theta, d) = d[:, 3:5] .* (d[:, 1] .- d[:, 2] .* theta[1])

m_iv = estimate_gmm(iv_moments, [0.0], data; weighting=:two_step)
report(m_iv)
```

```julia
plot_result(m_iv)
```

```@raw html
<iframe src="../assets/plots/gmm_moment_fit.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The estimate recovers the true slope ``\beta = 2`` despite the regressor being correlated with the error — OLS on the same sample would be biased upward, since ``X`` loads on ``u`` with coefficient 0.5. Three instruments for one parameter leave two over-identifying restrictions, and the Hansen J-test does not reject them: the instruments are valid, as they are by construction here. The plot renders the moment discrepancies as bars with the J-test annotated, which is the quickest way to see *which* moment a rejection comes from.

**Recipe 2: Nonlinear Euler-equation GMM**

```@example gmm
Random.seed!(7)
Tn = 500
γ0, β0 = 2.0, 0.97
R  = exp.(0.03 .+ 0.05 .* randn(Tn))            # gross asset return
sw = 0.03
w  = 0.5 * γ0 * sw^2 .+ sw .* randn(Tn)         # consumption-growth noise
gc = (β0 .* R).^(1 / γ0) .* exp.(w)             # consumption growth C_{t+1}/C_t
edata = hcat(gc, R)

# Consumption-based Euler equation: E[Z_t (β R_t g_t^{−γ} − 1)] = 0
# Instruments Z_t = [1, R_t, R_t²] → 3 moments, 2 parameters (β, γ)
euler_moments(theta, d) = begin
    β, γ = theta
    perr = β .* d[:, 2] .* d[:, 1] .^ (-γ) .- 1.0
    hcat(perr, d[:, 2] .* perr, (d[:, 2] .^ 2) .* perr)
end

m_euler = estimate_gmm(euler_moments, [0.95, 1.5], edata; weighting=:two_step)
report(m_euler)
```

``\theta[1]`` is the discount factor ``\beta`` and ``\theta[2]`` the coefficient of relative risk aversion ``\gamma``; both are recovered near their data-generating values of 0.97 and 2.0. Three moments against two parameters leave a single over-identifying restriction. This is the Hansen & Singleton (1982) consumption-based asset-pricing setup in miniature — note that no model is ever solved, only the Euler residual evaluated on the data.

**Recipe 3: Simulated Method of Moments on an AR(1)**

```@example gmm
Random.seed!(11)
Tobs = 300
ρ0, σ0 = 0.7, 0.5
y = zeros(Tobs)
for t in 2:Tobs
    y[t] = ρ0 * y[t-1] + σ0 * randn()
end
ar1data = reshape(y, :, 1)

# Simulator: T_periods observations after discarding `burn` transients
sim_ar1(theta, T_periods, burn; rng=Random.default_rng()) = begin
    ρ, σ = theta
    s = zeros(T_periods + burn)
    for t in 2:(T_periods + burn)
        s[t] = ρ * s[t-1] + abs(σ) * randn(rng)
    end
    reshape(s[(burn+1):end], :, 1)
end

bounds = ParameterTransform([-1.0, 0.0], [1.0, Inf])   # ρ ∈ (−1,1), σ > 0
m_smm = estimate_smm(sim_ar1, d -> autocovariance_moments(d; lags=2),
                     [0.4, 0.4], ar1data;
                     sim_ratio=3, burn=100, max_iter=300,
                     contributions_fn=d -> autocovariance_moment_contributions(d; lags=2),
                     bounds=bounds, rng=Random.MersenneTwister(99))
report(m_smm)
```

``\theta[1]`` is the persistence ``\rho`` and ``\theta[2]`` the innovation standard deviation ``\sigma``, against data-generating values of 0.7 and 0.5. Three moments — the variance plus two autocovariance lags of a univariate series — identify two parameters, leaving one over-identifying restriction. `bounds` keeps the search inside ``\rho \in (-1, 1)`` and ``\sigma > 0`` without penalty terms, and the explicit `rng` makes the simulated moments a deterministic function of ``\theta``.

---

## The GMM Objective

Given ``q`` moment conditions and ``k`` parameters (``q \geq k``), GMM chooses ``\theta`` to minimize the quadratic form

```math
Q(\theta) = g(\theta)' \, W \, g(\theta),
\qquad
g(\theta) = \frac{1}{n} \sum_{i=1}^{n} g_i(\theta),
```

where:
- ``g(\theta)`` is the ``q \times 1`` vector of sample moment conditions
- ``g_i(\theta)`` is the moment contribution of observation ``i``
- ``W`` is a ``q \times q`` positive-definite weighting matrix
- ``n`` is the number of observations

The estimator is consistent for any fixed ``W``, but the **asymptotically efficient** choice sets ``W = \Omega^{-1}``, the inverse long-run covariance of the moments (Hansen 1982). With efficient weighting the asymptotic covariance collapses to the sandwich-free form ``V = (G'WG)^{-1}/n``, where ``G = \partial g / \partial \theta'`` is the moment Jacobian (computed by central differences). For any other ``W`` the full sandwich ``V = (G'WG)^{-1} G'W \Omega W G (G'WG)^{-1}/n`` is used.

!!! note "Time-series moments and HAC"
    When moment conditions are serially correlated (the typical macro case), ``\Omega`` is estimated with a Newey-West/Bartlett HAC kernel. Pass `hac=true` (the default) and optionally a fixed `bandwidth`; `bandwidth=0` selects the bandwidth automatically.

The moment function has signature `moment_fn(theta, data)` and must return an ``n \times q`` matrix — one row per observation, one column per moment. `estimate_gmm` reads ``n`` and ``q`` from an initial evaluation at `theta0`. The scalar criterion ``Q(\theta)`` itself is exposed as [`gmm_objective`](@ref)`(theta, moment_fn, data, W)` for users who want to evaluate or plot the objective surface directly.

---

## Weighting Schemes

The `weighting` keyword selects how ``W`` is constructed:

| Weighting | Description |
|-----------|-------------|
| `:identity` | One-step GMM, ``W = I``. Consistent but inefficient; the J-statistic is **not** ``\chi^2`` |
| `:optimal` | Efficient weighting ``W = \hat\Omega^{-1}`` evaluated at the initial guess |
| `:two_step` | Step 1 with ``W = I``, then re-weight by ``\hat\Omega^{-1}`` from the step-1 estimate (default) |
| `:iterated` | Continuously-updated GMM: alternate ``\hat\theta`` and ``\hat\Omega^{-1}`` until convergence |

The same choices are wrapped in the `GMMWeighting` specification, stored on every fitted model:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:two_step` | One of `:identity`, `:optimal`, `:two_step`, `:iterated` |
| `max_iter` | `Int` | `100` | Maximum iterations for iterated GMM |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |

!!! warning "The J-test requires efficient weighting"
    Under `:identity` weighting the Hansen J-statistic is a weighted sum of ``\chi^2(1)`` variables, not ``\chi^2(q-k)``. `estimate_gmm` returns `J_pvalue = NaN` in that case. Use `:two_step`, `:optimal`, or `:iterated` whenever the J-test is needed.

---

## Estimating with `estimate_gmm`

```julia
estimate_gmm(moment_fn, theta0, data;
             weighting=:two_step, max_iter=100, tol=1e-8,
             hac=true, bandwidth=0, bounds=nothing)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `weighting` | `Symbol` | `:two_step` | Weighting scheme (see above) |
| `max_iter` | `Int` | `100` | Maximum optimizer / iterated-GMM iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |
| `hac` | `Bool` | `true` | Use HAC correction when building the optimal weighting matrix |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic Newey-West selection) |
| `bounds` | `ParameterTransform` or `nothing` | `nothing` | Optional box constraints via bijective transforms; SEs corrected by the delta method |

The optimizer is LBFGS with a Nelder-Mead fallback. When `bounds` are supplied the search runs in an unconstrained space via `ParameterTransform`, so parameters such as variances or probabilities can be constrained without penalty terms. `hac` governs only the weighting matrix; the ``\Omega`` entering the identity-weighting sandwich is always the Bartlett long-run covariance.

`estimate_gmm` returns a `GMMModel`:

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vector{T}` | Parameter estimates |
| `vcov` | `Matrix{T}` | Asymptotic covariance matrix |
| `n_moments` | `Int` | Number of moment conditions ``q`` |
| `n_params` | `Int` | Number of parameters ``k`` |
| `n_obs` | `Int` | Number of observations ``n`` |
| `weighting` | `GMMWeighting{T}` | Weighting specification used |
| `W` | `Matrix{T}` | Final weighting matrix |
| `g_bar` | `Vector{T}` | Sample moment vector at the solution |
| `J_stat` | `T` | Hansen J-statistic |
| `J_pvalue` | `T` | J-test p-value (`NaN` under identity weighting) |
| `converged` | `Bool` | Optimizer convergence flag |
| `iterations` | `Int` | Total iterations |

The `report` method prints the specification, a Stata-style coefficient table, and — when overidentified — the Hansen J-test. `coef`, `vcov`, `stderror`, `confint`, `nobs`, and `dof` follow the StatsAPI convention. Both `GMMModel` and `SMMModel` subtype the abstract [`AbstractGMMModel`](@ref), on which the shared `report` and StatsAPI accessors dispatch.

---

## Overidentification and Model Selection

For an overidentified system (``q > k``) the J-statistic

```math
J = n \, g(\hat\theta)' \, \hat\Omega^{-1} \, g(\hat\theta) \; \xrightarrow{d} \; \chi^2(q - k)
```

tests the joint validity of the moment conditions. A large p-value indicates the overidentifying restrictions are not rejected. `gmm_summary` collects the coefficient statistics and the J-test into a single NamedTuple:

```@example gmm
s = gmm_summary(m_iv)
(theta = round.(s.theta, digits=4),
 se = round.(s.se, digits=4),
 J = round(s.j_test.J_stat, digits=4),
 J_pvalue = round(s.j_test.p_value, digits=4))
```

The J-statistic is small relative to its two degrees of freedom and the p-value is comfortably interior, so the three instrument moments are mutually consistent. `gmm_summary` returns the same numbers `report` prints, in a NamedTuple that also carries `t_stats`, `p_values`, `n_moments`, `n_params`, `n_obs`, `weighting`, `converged`, and `iterations` — use it when the numbers feed further computation rather than a table.

To compare non-nested moment specifications, `andrews_lu_mmsc` computes the Andrews-Lu (2001) Model and Moment Selection Criteria from the J-statistic. Lower values indicate a better-specified moment set:

```math
\text{MMSC}_{\text{BIC}} = J - (q - k)\log n,
\quad
\text{MMSC}_{\text{AIC}} = J - 2(q - k),
\quad
\text{MMSC}_{\text{HQIC}} = J - Q\,(q - k)\log\log n.
```

```@example gmm
andrews_lu_mmsc(m_iv.J_stat, m_iv.n_moments, m_iv.n_params, m_iv.n_obs)
```

All three criteria are strongly negative because the J-statistic is small while the penalty term rewards the two extra moments: with ``n = 500`` the BIC penalty alone is ``2\log 500 \approx 12.4``. The absolute levels carry no meaning — only differences across competing moment sets estimated on the same data do, and the criteria disagree in general, with BIC the most and AIC the least willing to add moments.

The `hq_criterion` keyword (default `2.1`) sets the HQIC penalty ``Q``.

---

## Linear GMM Utilities

For linear IV models — including those inside [Panel VAR](@ref pvar_page) — the moment conditions ``\mathbb{E}[Z'(y - X\beta)] = 0`` admit a closed-form solution given the aggregated cross-products ``S_{ZX} = \sum_i Z_i'X_i`` and ``S_{Zy} = \sum_i Z_i'y_i``:

```math
\hat\beta = (S_{ZX}' \, W \, S_{ZX})^{-1} \, S_{ZX}' \, W \, S_{Zy}.
```

`linear_gmm_solve` implements this directly, and `gmm_sandwich_vcov` returns the robust one-step covariance ``V = (S_{ZX}'WS_{ZX})^{-1} S_{ZX}'W D_e W S_{ZX} (S_{ZX}'WS_{ZX})^{-1}``, where ``D_e = \sum_i (Z_i e_i)(Z_i e_i)'``:

```@example gmm
Zmat, Xmat, yvec = data[:, 3:5], data[:, 2:2], data[:, 1]
S_ZX = Zmat' * Xmat
S_Zy = Zmat' * yvec
W    = inv(Zmat' * Zmat)                      # 2SLS-type weighting

beta_hat = linear_gmm_solve(S_ZX, S_Zy, W)
e   = yvec .- Xmat * beta_hat
Ze  = Zmat .* e
V   = gmm_sandwich_vcov(S_ZX, W, Ze' * Ze)

(beta = round.(beta_hat, digits=4), se = round.(sqrt.(diag(V)), digits=4))
```

This is the 2SLS estimator written out in GMM form: with ``W = (Z'Z)^{-1}`` the quadratic form has a closed-form minimizer, so no optimizer runs at all. The slope lands within a fraction of a standard error of the two-step `estimate_gmm` fit above — the two differ only through the weighting matrix, and the standard error differs again because `gmm_sandwich_vcov` is the one-step robust sandwich rather than the efficient two-step form. These building blocks bypass the numerical optimizer entirely, which matters for the inner loop of panel and system estimators.

---

## Simulated Method of Moments

When model moments have no closed form but the model can be simulated, SMM replaces ``m_{\text{sim}}(\theta)`` (moments of simulated data) for the analytic moments and minimizes

```math
Q(\theta) = \big(m_{\text{data}} - m_{\text{sim}}(\theta)\big)' \, W \, \big(m_{\text{data}} - m_{\text{sim}}(\theta)\big).
```

`estimate_smm` takes a `simulator_fn(theta, T_periods, burn; rng)` and a `moments_fn(data)`:

```julia
estimate_smm(simulator_fn, moments_fn, theta0, data;
             sim_ratio=5, burn=100, weighting=:two_step,
             contributions_fn=nothing, bounds=nothing,
             hac=true, bandwidth=0, max_iter=1000, tol=1e-8,
             rng=Random.default_rng())
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `sim_ratio` | `Int` | `5` | Simulation-to-data length ratio ``\tau = T_{\text{sim}}/T`` |
| `burn` | `Int` | `100` | Burn-in periods discarded by the simulator |
| `weighting` | `Symbol` | `:two_step` | `:identity` or `:two_step` |
| `contributions_fn` | `Function` or `nothing` | `nothing` | Per-observation moment contributions for a valid ``\Omega`` |
| `bounds` | `ParameterTransform` or `nothing` | `nothing` | Optional box constraints |
| `hac` | `Bool` | `true` | HAC long-run covariance for the weighting matrix |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic) |
| `max_iter` | `Int` | `1000` | Maximum optimizer iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance |
| `rng` | `AbstractRNG` | `default_rng()` | RNG; copied inside the objective so simulated moments are a deterministic function of ``\theta`` |

The asymptotic covariance carries the simulation-noise inflation factor ``(1 + 1/\tau)``, so a larger `sim_ratio` shrinks the variance at the cost of more computation.

`autocovariance_moments` supplies the standard DSGE moment vector — the upper-triangle variance-covariance elements followed by diagonal autocovariances at each lag (``k(k{+}1)/2 + kL`` moments for ``k`` variables and ``L`` lags):

```@example gmm
autocovariance_moments(ar1data; lags=2)
```

For the single-variable AR(1) sample this is three numbers: the variance, then the lag-1 and lag-2 autocovariances. Their ratios are the sample autocorrelations — the second entry over the first is near the data-generating ``\rho = 0.7``, and the third continues the geometric decay. These are exactly the features SMM matches to pin down persistence, and the geometric pattern is why adding lags beyond two buys little extra identification for an AR(1).

!!! note "Why `contributions_fn` matters"
    A demeaning moment function evaluated one row at a time produces identically-zero rows and a degenerate ``\Omega``. `autocovariance_moment_contributions` returns the ``n \times q`` matrix whose column means equal `autocovariance_moments` exactly, giving a well-defined optimal weighting matrix and sandwich standard errors. Supply it for two-step SMM; without it, two-step weighting silently falls back to identity.

`estimate_smm` returns an `SMMModel`, which shares every field of `GMMModel` and adds `sim_ratio`. It supports the same StatsAPI methods and `report`. Its optimizer ordering is the reverse of `estimate_gmm`'s: Nelder-Mead first, with LBFGS as the fallback, because a simulation-based objective is only piecewise smooth even when the RNG is held fixed.

Unlike `estimate_gmm`, `estimate_smm` reports a ``\chi^2`` p-value for the J-statistic under **every** weighting scheme, including `:identity`. That p-value is only interpretable when the weighting is efficient — under identity weighting read the statistic as a descriptive measure of moment fit, not as a test.

---

## Complete Example

The following workflow estimates an AR(1) by SMM, inspects the overidentifying restrictions, and compares the SMM point estimates against the OLS benchmark for the same series.

```@example gmm
Random.seed!(14)
Tobs = 300
ρ_true, σ_true = 0.6, 0.4
z = zeros(Tobs)
for t in 2:Tobs
    z[t] = ρ_true * z[t-1] + σ_true * randn()
end
zdata = reshape(z, :, 1)

simulate_ar1(theta, T_periods, burn; rng=Random.default_rng()) = begin
    ρ, σ = theta
    s = zeros(T_periods + burn)
    for t in 2:(T_periods + burn)
        s[t] = ρ * s[t-1] + abs(σ) * randn(rng)
    end
    reshape(s[(burn+1):end], :, 1)
end

fit = estimate_smm(simulate_ar1, d -> autocovariance_moments(d; lags=2),
                   [0.3, 0.3], zdata;
                   sim_ratio=3, burn=100, max_iter=300,
                   contributions_fn=d -> autocovariance_moment_contributions(d; lags=2),
                   bounds=ParameterTransform([-1.0, 0.0], [1.0, Inf]),
                   rng=Random.MersenneTwister(7))
report(fit)
```

The SMM estimator recovers the persistence and innovation-standard-deviation parameters by matching the variance and the first two autocovariances of the simulated process to their empirical counterparts; both land near the data-generating ``\rho = 0.6`` and ``\sigma = 0.4``. Because the moment vector has three elements for two parameters, the Hansen J-test provides a specification check on the single remaining restriction, and it does not reject at conventional levels — the fitted AR(1) reproduces the sample second moments, as it should given the data-generating process. Simulation noise alone moves this statistic noticeably between seeds at ``\tau = 3``; raise `sim_ratio` before reading a marginal p-value as evidence of misspecification.

```@example gmm
ρ_ols = (z[1:end-1]' * z[2:end]) / (z[1:end-1]' * z[1:end-1])
(smm_rho = round(coef(fit)[1], digits=3),
 ols_rho = round(ρ_ols, digits=3),
 true_rho = ρ_true)
```

The SMM and OLS persistence estimates are close, as expected when both target the same low-order autocorrelation structure; SMM is the method of choice when — unlike this AR(1) — the model's moments cannot be written in closed form.

---

## Common Pitfalls

1. **Fewer moments than parameters.** GMM and SMM require ``q \geq k``; `estimate_gmm` and `estimate_smm` assert this. Add moment conditions (more instruments or more autocovariance lags) if the model is underidentified.
2. **Reading the J-test under identity weighting.** The J-statistic is ``\chi^2`` only with efficient weighting. `estimate_gmm` returns `J_pvalue = NaN` under `:identity` to make that explicit; `estimate_smm` returns a numeric p-value regardless, so under identity weighting it is on you not to read it as a test. Re-estimate with `:two_step` before interpreting the overidentification test either way.
3. **Two-step SMM without `contributions_fn`.** Omitting it makes the optimal weighting matrix degenerate; the estimator warns and falls back to identity weighting. Always pass a contributions function whose column means equal your moment vector.
4. **Non-deterministic SMM objective.** The simulator must be a deterministic function of ``\theta`` given the RNG. `estimate_smm` copies `rng` on every call for this reason — do not advance a shared global RNG inside `simulator_fn`.
5. **Moment function shape.** `moment_fn(theta, data)` must return an ``n \times q`` matrix (rows = observations), not the ``q``-vector of sample means. `estimate_gmm` averages the rows internally.

---

## References

- Andrews, D. W. K., & Lu, B. (2001). Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models.
  *Journal of Econometrics*, 101(1), 123-164. [DOI](https://doi.org/10.1016/S0304-4076(00)00077-4)
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators.
  *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)
- Hansen, L. P., & Singleton, K. J. (1982). Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models.
  *Econometrica*, 50(5), 1269-1286. [DOI](https://doi.org/10.2307/1911873)
- Lee, B.-S., & Ingram, B. F. (1991). Simulation Estimation of Time-Series Models.
  *Journal of Econometrics*, 47(2-3), 197-205. [DOI](https://doi.org/10.1016/0304-4076(91)90098-X)
- Newey, W. K., & McFadden, D. (1994). Large Sample Estimation and Hypothesis Testing.
  In *Handbook of Econometrics*, Vol. 4, 2111-2245. [DOI](https://doi.org/10.1016/S1573-4412(05)80005-4)
- Ruge-Murcia, F. (2012). Estimating Nonlinear DSGE Models by the Simulated Method of Moments.
  *Journal of Economic Dynamics and Control*, 36(6), 914-938. [DOI](https://doi.org/10.1016/j.jedc.2012.01.008)
