# [Heteroskedasticity-Based Identification](@id id_heteroskedastic_page)

Heteroskedasticity-based SVAR identification exploits time-varying second moments to recover the structural impact matrix ``B_0`` without distributional assumptions on the shocks. When structural shock variances change across regimes while ``B_0`` stays constant, each regime supplies a separate covariance equation, and two regimes are enough to pin down the rotation.

- **Markov-switching**: Hamilton (1989) filter with an EM algorithm estimates regime-specific covariances endogenously (Lanne & Lütkepohl 2008)
- **GARCH**: GARCH(1,1) conditional heteroskedasticity provides continuous time-varying identification (Normandin & Phaneuf 2004)
- **Smooth transition**: a logistic transition function allows gradual regime shifts (Lütkepohl & Netšunajev 2017)
- **External volatility**: known regime indicators (NBER recessions, financial crises) give the simplest sample-split approach (Rigobon 2003)

For an overview and method comparison, see [Statistical Identification](@ref nongaussian_page). For identification from higher moments instead of second moments, see [Non-Gaussian Methods](@ref id_nongaussian_page). For identifiability diagnostics, see [Testing](@ref id_testing_page). For schemes built on economic restrictions, see [Structural Identification](@ref structural_identification_page).

```@setup id_het
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-119:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: Markov-switching identification**

```@example id_het
ms = identify_markov_switching(model; n_regimes=2)
report(ms)
```

**Recipe 2: Smooth transition with an observable transition variable**

```@example id_het
# Industrial production growth, lagged one month, drives the variance regime
s = Y[2:end, 1]
st = identify_smooth_transition(model, s)
(gamma = round(st.gamma, digits=1),
 threshold = round(st.threshold, digits=5),
 converged = st.converged)
```

**Recipe 3: External volatility regimes**

```@example id_het
# Known regimes: split the residual sample at its midpoint
T_obs = size(model.U, 1)
regime = vcat(fill(1, T_obs ÷ 2), fill(2, T_obs - T_obs ÷ 2))
ev = identify_external_volatility(model, regime)
report(ev)
```

**Recipe 4: Inspect the identifying variance ratios**

```@example id_het
[round.(ms.Lambda[k], digits=3) for k in 1:ms.n_regimes]
```

**Recipe 5: Feed the identification into the IRF and FEVD pipeline**

```@example id_het
irfs = irf(model, 20; method=:markov_switching)
report(irfs)
```

---

## Eigendecomposition Identification

The core insight (Rigobon 2003): when ``B_0`` is constant across regimes but shock variances change, multiple covariance equations arise:

```math
\Sigma_k = B_0 \Lambda_k B_0', \quad k = 1, \ldots, K
```

where:
- ``\Sigma_k`` is the ``n \times n`` reduced-form covariance matrix in regime ``k``
- ``B_0`` is the ``n \times n`` structural impact matrix (constant across regimes)
- ``\Lambda_k = \text{diag}(\lambda_{1k}, \ldots, \lambda_{nk})`` holds the regime-specific shock variances

Given two regime covariance matrices, the symmetric generalized eigenproblem recovers the structural parameters:

```math
L_1^{-1} \Sigma_2 L_1^{-\top} = W \Lambda W'
```

where:
- ``L_1`` is the Cholesky factor of ``\Sigma_1``
- ``W`` is orthogonal, with columns the eigenvectors of the whitened second-regime covariance
- ``\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`` contains the relative variance ratios ``\lambda_j = \lambda_{j2} / \lambda_{j1}``, ordered ascending
- ``B_0 = L_1 W``

**Identification condition**: the eigenvalues ``\lambda_j`` must be distinct. With ``K \geq 2`` regimes producing distinct eigenvalues, ``B_0`` is identified up to column permutation and sign.

For ``K \ge 2`` the estimators then refine that start by **joint maximum likelihood** (Lanne, Lütkepohl & Maciejowska 2010). The structural covariances are

```math
\Sigma_k = B_0 \Lambda_k B_0', \qquad \Lambda_1 = I, \qquad k = 1,\ldots,K
```

where:
- ``B_0`` is free (equivalently a Cholesky factor times Givens angles)
- ``\Lambda_k = \mathrm{diag}(\lambda_{1k},\ldots,\lambda_{nk})`` for ``k \ge 2``
- every regime covariance enters the concentrated Gaussian likelihood, not just the first pair

Delta-method standard errors of ``B_0`` come from the observed information of that concentrated criterion and are stored as `se` and `vcov`.

!!! note "Technical Note"
    The two-regime start solves ``L_1^{-1}\Sigma_2 L_1^{-\top} = W\Lambda W'`` and sets ``B_0 = L_1 W``. Eigenvalues are ordered ascending. A positive-diagonal sign convention normalizes the result, and `Lambda` is recomputed per regime as ``\text{diag}(B_0^{-1} \Sigma_k B_0^{-\prime})``, so regime 1 always reports a vector of ones. There is no polar projection onto the orthogonal group.

---

## Markov-Switching Volatility

Markov-switching identification (Lanne & Lütkepohl 2008) estimates regime-specific covariance matrices with the Hamilton (1989) filter and an EM algorithm. The latent state ``S_t \in \{1, \ldots, K\}`` follows a first-order Markov chain with transition matrix ``P``, where ``P_{ij} = P(S_t = j \mid S_{t-1} = i)``:

```math
f(u_t \mid S_t = k) = (2\pi)^{-n/2} |\Sigma_k|^{-1/2} \exp\!\left(-\tfrac{1}{2} u_t' \Sigma_k^{-1} u_t\right)
```

where:
- ``u_t`` is the ``n \times 1`` vector of reduced-form residuals
- ``\Sigma_k`` is the ``n \times n`` covariance matrix in regime ``k``

The EM algorithm iterates:

1. **E-step**: the Hamilton (1989) forward filter computes filtered probabilities ``\xi_{t|t}(k)``; the Kim (1994) backward smoother produces smoothed probabilities ``\xi_{t|T}(k)``.
2. **M-step**: regime covariances are updated as smoothed-probability-weighted sample covariances, and the transition matrix from the Kim (1994) joint smoothed probabilities ``\xi_{t,t-1|T}(i,j)``.
3. **Identification**: joint ML for a common ``B_0`` and ``\Lambda_2,\ldots,\Lambda_K`` (``\Lambda_1 = I``) using all ``K`` covariances (Lanne, Lütkepohl & Maciejowska 2010). `n_starts` Dirichlet-random EM starts keep the filter off a single local mode; the start with the highest likelihood is returned.

!!! note "Kim (1994) Joint Smoother"
    The transition-matrix update uses ``\xi_{t,t-1|T}(i,j) = \xi_{t|T}(j) \cdot P_{ij} \cdot \xi_{t-1|t-1}(i) / \xi_{t|t-1}(j)`` rather than the naive product of marginal smoothed probabilities. This accounts for serial dependence in the regime assignments and produces unbiased transition-matrix estimates.

```@example id_het
ms = identify_markov_switching(model; n_regimes=2)
report(ms)
```

```@example id_het
(transition = round.(ms.transition_matrix, digits=3),
 lambda_2 = round.(ms.Lambda[2], digits=3),
 iterations = ms.iterations)
```

```julia
plot_result(ms; view=:regimes)
```

```@raw html
<iframe src="../assets/plots/ms_regime_probs.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The EM algorithm converges in 15 iterations to two persistent regimes: staying probabilities of ``0.838`` and ``0.915`` imply expected durations of about 6 and 12 months. Relative to regime 1, all three shock variances are *lower* in regime 2 --- ``\Lambda_2 = (0.05, 0.13, 0.647)`` --- so regime 1 is the turbulent state and regime 2 the calm one. Identification rests on those three numbers being distinct. The closest pair is ``0.05`` against ``0.13`` (a factor of about 2.6); the third ratio, ``0.647``, is farther from both, so that column is the most sharply identified. Persistent regimes with well-separated ratios strengthen identification; `ms.regime_probs` shows when each state prevailed.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_regimes` | `Int` | `2` | Number of volatility regimes |
| `max_iter` | `Int` | `500` | Maximum EM iterations |
| `tol` | `Real` | ``10^{-6}`` | Relative convergence tolerance on the log-likelihood |
| `n_starts` | `Int` | `5` | Dirichlet-random EM starts (first start is a chunk split) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random number generator for the extra starts |

**Return value** (`MarkovSwitchingSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime (regime 1 is normalized to ones) |
| `regime_probs` | `Matrix{T}` | Smoothed regime probabilities (``T_{\text{eff}} \times K``) |
| `transition_matrix` | `Matrix{T}` | Markov transition probabilities (``K \times K``) |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | EM iterations used |
| `n_regimes` | `Int` | Number of regimes |
| `se` | `Matrix{T}` | Delta-method standard errors of ``B_0`` |
| `vcov` | `Matrix{T}` | Parameter-space covariance (Givens angles and ``\log\Lambda_{2:K}``) |
| `classification_quality` | `T` | Mean of the max smoothed regime probability |
| `shocks` | `Matrix{T}` | Structural shocks ``\varepsilon_t = B_0^{-1} u_t`` |

!!! note "Every regime identifies B₀"
    With `n_regimes > 2` the EM step estimates all ``K`` covariance matrices and the joint ML step uses all of them. The two-regime generalized eigenproblem is only the starting value. Check `se` and `test_lambda_distinct` before treating extra regimes as free identifying information.

---

## GARCH-Based Identification

GARCH-based identification (Normandin & Phaneuf 2004) uses conditional heteroskedasticity in the structural shocks. Each shock follows a GARCH(1,1) process:

```math
h_{j,t} = \omega_j + \alpha_j \varepsilon_{j,t-1}^2 + \beta_j h_{j,t-1}
```

where:
- ``h_{j,t}`` is the conditional variance of shock ``j`` at time ``t``
- ``\omega_j > 0``, ``\alpha_j \geq 0``, ``\beta_j \geq 0`` with ``\alpha_j + \beta_j < 1``

The structural impact matrix is estimated by maximizing:

```math
\ell(B_0) = -\frac{1}{2} \sum_{t=1}^{T} \left[ n \ln(2\pi) + \sum_{j=1}^{n} \ln h_{j,t} + \sum_{j=1}^{n} \frac{\varepsilon_{j,t}^2}{h_{j,t}} \right] + T \ln|\det(B_0^{-1})|
```

where ``\varepsilon_t = B_0^{-1} u_t`` and each ``h_{j,t}`` follows the GARCH recursion. The estimator alternates between two blocks: with ``B_0`` fixed it fits a GARCH(1,1) to each structural shock; with the conditional variances fixed it re-optimizes the ``n(n-1)/2`` Givens angles that generate ``B_0``. Since ``|\det Q| = 1``, only the weighted residual sum ``\sum_t \sum_j \varepsilon_{j,t}^2 / h_{j,t}`` varies in that second block.

```@example id_het
garch = identify_garch(model)
report(garch)
```

```@example id_het
(persistence = round.(garch.garch_params[:, 2] .+ garch.garch_params[:, 3], digits=3),
 iterations = garch.iterations)
```

The outer loop stops after 18 iterations. Persistence ``\alpha_j+\beta_j`` is ``(0.894, 0.274, 0.970)``: all three sums sit below 1, so the GARCH(1,1) weights are a valid scheme. Shock 3 is nearly integrated (``\beta_3 \approx 0.97``), shock 2 is almost pure ARCH, and shock 1 is a standard covariance-stationary GARCH. Distinct persistence profiles are what identify the columns --- if every shock had the same ``(\alpha,\beta)``, the rotation would not be pinned down. Always compute this sum before interpreting a GARCH-identified ``B_0``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `500` | Maximum outer iterations |
| `tol` | `Real` | ``10^{-6}`` | Relative convergence tolerance on the log-likelihood |

**Return value** (`GARCHSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `garch_params` | `Matrix{T}` | ``n \times 3`` matrix: ``[\omega, \alpha, \beta]`` per shock |
| `cond_var` | `Matrix{T}` | ``T_{\text{eff}} \times n`` conditional variances |
| `shocks` | `Matrix{T}` | Structural shocks |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Outer iterations used |
| `se` | `Matrix{T}` | Delta-method standard errors of ``B_0`` |
| `vcov` | `Matrix{T}` | Parameter-space covariance (Givens angles and GARCH parameters) |

---

## Smooth Transition

Smooth-transition identification (Lütkepohl & Netšunajev 2017) allows gradual volatility shifts through a logistic transition function:

```math
\Sigma_t = B_0 \bigl[I + G(s_t)(\Lambda - I)\bigr] B_0'
```

where:
- ``G(s_t) = 1 / (1 + \exp(-\gamma(s_t - c)/\mathrm{std}(s)))`` is the logistic transition function
- ``s_t`` is an observable transition variable
- ``\gamma > 0`` controls the transition speed (large ``\gamma`` approximates a discrete switch)
- ``c`` is the threshold location
- ``\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`` holds the relative variances in the second regime

When ``G = 0`` the covariance equals ``B_0 B_0'``; when ``G = 1`` it equals ``B_0 \Lambda B_0'``. A median split of ``G(s_t)`` supplies the starting value ``(L, Q, \Lambda)``. Joint ML then estimates ``(\mathrm{vech}\, L, \theta, \log\Lambda, \gamma, c)`` together, so the ``G = 0`` pole is ``B_0 B_0'`` rather than a frozen split (or unconditional) covariance. ``\log\gamma`` is bounded so ``\gamma \in [10^{-3}, 20]/\mathrm{std}(s)``.

```@example id_het
s = Y[2:end, 1]
st = identify_smooth_transition(model, s)
report(st)
```

```@example id_het
(gamma = round(st.gamma, digits=1),
 threshold = round(st.threshold, digits=5),
 share_high_regime = round(count(>(0.5), st.G_values) / length(st.G_values), digits=3),
 n_transitional = count(g -> 0.01 < g < 0.99, st.G_values))
```

The estimated transition speed is on the order of 30 around a threshold near zero in monthly industrial-production growth, with tens of observations in the transitional region ``0.01 < G(s_t) < 0.99``. That is a genuine smooth transition, not a collapsed sample split: a frozen-split estimator that holds ``B_0`` fixed after the ``G = 0.5`` cut produces ``\gamma`` in the thousands and almost no transitional observations. Joint ML does not. If ``\gamma`` saturates the upper bound ``20/\mathrm{std}(s)``, check `identify_external_volatility` with an explicitly dated regime indicator. If `converged` is `false`, raise `max_iter`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `500` | Maximum L-BFGS iterations for the joint criterion |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

**Return value** (`SmoothTransitionSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance matrices for the two extreme regimes |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime (regime 1 is normalized to ones) |
| `gamma` | `T` | Transition speed parameter |
| `threshold` | `T` | Transition location parameter |
| `transition_var` | `Vector{T}` | The transition variable, truncated to ``T_{\text{eff}}`` |
| `G_values` | `Vector{T}` | Evaluated transition function ``G(s_t)`` |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Optimizer iterations used |
| `se` | `Matrix{T}` | Delta-method standard errors of ``B_0`` |
| `vcov` | `Matrix{T}` | Parameter-space covariance (``\mathrm{vech}\, L``, Givens angles, ``\log\Lambda``, ``\gamma``, ``c``) |

---

## External Volatility Instruments

When the volatility regimes are known a priori --- NBER recessions, financial crises, policy regime changes --- external volatility identification (Rigobon 2003) splits the sample, estimates regime-specific covariance matrices, and jointly estimates ``B_0`` and ``\Lambda_2,\ldots,\Lambda_K`` from all ``K`` of them (Lanne, Lütkepohl & Maciejowska 2010). This is the simplest heteroskedasticity method: no latent state, and the regime dates are an assumption the reader can check. A regime with fewer than ``n+1`` observations throws `ArgumentError`.

```@example id_het
T_obs = size(model.U, 1)
regime = vcat(fill(1, T_obs ÷ 2), fill(2, T_obs - T_obs ÷ 2))
ev = identify_external_volatility(model, regime)
report(ev)
```

```@example id_het
(lambda_2 = round.(ev.Lambda[2], digits=3),
 sizes = length.(ev.regime_indices))
```

The two halves of the sample carry 59 residuals each. The second-half variance ratios ``(0.406, 1.046, 1.517)`` are distinct and straddle 1, so the shocks are separately identified: the first shock is about 40% as volatile in the second half, the third about half again as volatile, and the middle shock barely moves. That middle ratio is the weak link --- a ratio of ``1.046`` is nearly the no-change value of 1, so the corresponding column of ``B_0`` is identified only weakly. Compare the resulting ``B_0`` against the Markov-switching solution, which chooses its own regime dates.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `regimes` | `Int` | `2` | Number of distinct regime labels in `regime_indicator` |

**Return value** (`ExternalVolatilitySVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime (regime 1 is normalized to ones) |
| `regime_indices` | `Vector{Vector{Int}}` | Observation indices per regime |
| `loglik` | `T` | Log-likelihood |
| `se` | `Matrix{T}` | Delta-method standard errors of ``B_0`` |
| `vcov` | `Matrix{T}` | Parameter-space covariance (Givens angles and ``\log\Lambda_{2:K}``) |

---

## Complete Example

This workflow identifies the monetary VAR from its volatility regimes, checks the conditions that make the identification credible, and pushes the result through the innovation-accounting pipeline.

```@example id_het
# --- Step 1: Estimate the regimes and inspect the identifying ratios ---
ms = identify_markov_switching(model; n_regimes=2)
(converged = ms.converged,
 transition = round.(ms.transition_matrix, digits=3),
 lambda_2 = round.(ms.Lambda[2], digits=3))
```

```@example id_het
# --- Step 2: Cross-check against externally dated regimes ---
ev = identify_external_volatility(model, regime)
cosdist(a, b) = 1 - abs(sum(a .* b)) / sqrt(sum(abs2, a) * sum(abs2, b))
[round(minimum(cosdist(ms.B0[:, i], ev.B0[:, j]) for j in 1:3), digits=4) for i in 1:3]
```

```@example id_het
# --- Step 3: Structural IRFs under the Markov-switching identification ---
irfs = irf(model, 20; method=:markov_switching)
report(irfs)
```

```@example id_het
# --- Step 4: Variance decomposition from the same identification ---
decomp = fevd(model, 20; method=:markov_switching)
report(decomp)
```

Step 2 matches each Markov-switching column to its closest external-volatility column: two agree to within ``0.01`` in angular distance (``0.002`` and ``0.009``), while the middle column is ``0.19`` away. That middle mismatch is the external-volatility ratio near 1 from the sample-split step --- a shock whose variance barely changes across the midpoint split is not the same object as the Hamilton filter's middle shock. Agreement on the well-separated columns is the evidence a heteroskedasticity-based identification can report; quantify the weak column with `test_lambda_distinct` from [Identification Testing](@ref id_testing_page) before drawing economic conclusions.

---

## Common Pitfalls

1. **Indistinct eigenvalues.** If two shocks experience the same proportional variance change, their columns in ``B_0`` are not separately identified. Check that the `Lambda` vectors show clearly distinct values --- ``0.05`` against ``0.13``, the closest pair in the Markov-switching fit above, is usable but not generous. Quantify it with `test_lambda_distinct` from the [Identification Testing](@ref id_testing_page) page.

2. **A variance ratio near 1 identifies nothing.** A shock whose variance is unchanged across regimes contributes no identifying equation. Watch for entries of `Lambda[2]` close to 1, such as the ``1.046`` in the external-volatility fit.

3. **EM converges to local optima.** The Markov-switching EM algorithm is initialized by splitting the sample into ``K`` equal blocks, so it is sensitive to where the volatility actually shifts. If `ms.converged == false`, raise `max_iter`; if the regimes look implausible, compare against `identify_external_volatility` with dated regimes.

4. **Short regimes.** Regime-specific covariance estimation needs at least ``n + 1`` observations per regime. `identify_external_volatility` throws `ArgumentError` on any undersized regime rather than silently substituting the full-sample covariance. Check `length.(ev.regime_indices)` before shrinking a dated split.

5. **GARCH stationarity.** Verify ``\alpha_j + \beta_j < 1`` in `garch.garch_params` before interpreting a GARCH-identified ``B_0``. Values summing above 1, or identical across shocks, mean the inner GARCH optimizer never left its starting point and the conditional variances are meaningless.

6. **The labelling problem.** Every method here identifies ``B_0`` up to column permutation and sign. The package normalizes to a positive diagonal, but naming the shocks requires economic reasoning that the volatility regimes cannot supply.

---

## References

- Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57 (2): 357--384. [DOI](https://doi.org/10.2307/1912559)

- Kim, Chang-Jin. 1994. "Dynamic Linear Models with Markov-Switching." *Journal of Econometrics* 60 (1--2): 1--22. [DOI](https://doi.org/10.1016/0304-4076(94)90036-1)

- Lanne, Markku, and Helmut Lütkepohl. 2008. "Identifying Monetary Policy Shocks via Changes in Volatility." *Journal of Money, Credit and Banking* 40 (6): 1131--1149. [DOI](https://doi.org/10.1111/j.1538-4616.2008.00151.x)

- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086--3124. [DOI](https://doi.org/10.1093/restud/rdab009)

- Lanne, Markku, Helmut Lütkepohl, and Katarzyna Maciejowska. 2010. "Structural Vector Autoregressions with Markov Switching." *Journal of Economic Dynamics and Control* 34 (2): 121--131. [DOI](https://doi.org/10.1016/j.jedc.2009.08.002)

- Lütkepohl, Helmut, and Aleksei Netšunajev. 2017. "Structural Vector Autoregressions with Smooth Transition in Variances." *Journal of Economic Dynamics and Control* 84: 43--57. [DOI](https://doi.org/10.1016/j.jedc.2017.09.001)

- Normandin, Michel, and Louis Phaneuf. 2004. "Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility." *Journal of Monetary Economics* 51 (6): 1217--1243. [DOI](https://doi.org/10.1016/j.jmoneco.2003.11.002)

- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777--792. [DOI](https://doi.org/10.1162/003465303772815727)
