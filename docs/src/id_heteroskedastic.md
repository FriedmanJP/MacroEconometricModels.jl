# [Heteroskedasticity-Based Identification](@id id_heteroskedastic_page)

Heteroskedasticity-based SVAR identification exploits time-varying second moments to recover the structural impact matrix ``B_0`` without distributional assumptions on shocks. When structural shock variances change across regimes while ``B_0`` remains constant, each regime provides a separate covariance equation, generating enough restrictions to identify the structural parameters.

- **Markov-switching**: Hamilton (1989) filter with EM algorithm estimates regime-specific covariances endogenously
- **GARCH**: GARCH(1,1) conditional heteroskedasticity provides continuous time-varying identification (Normandin & Phaneuf 2004)
- **Smooth transition**: Logistic transition function allows gradual regime shifts (Lutkepohl & Netsunajev 2017)
- **External volatility**: Known regime indicators (NBER recessions, financial crises) provide the simplest sample-split approach (Rigobon 2003)

## Quick Start

**Recipe 1: Markov-switching identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ms = identify_markov_switching(model; n_regimes=2)
report(ms)
```

**Recipe 2: GARCH-based identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

garch = identify_garch(model)
report(garch)
```

**Recipe 3: Smooth transition identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Lagged industrial production as the transition variable
s = Y[2:end, 1]
st = identify_smooth_transition(model, s)
report(st)
```

**Recipe 4: External volatility instruments**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Known regimes: split sample at midpoint
T_obs = size(model.U, 1)
regime = vcat(fill(1, T_obs ÷ 2), fill(2, T_obs - T_obs ÷ 2))
ev = identify_external_volatility(model, regime)
report(ev)
```

**Recipe 5: IRF via heteroskedasticity identification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

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
- ``\Lambda_k = \text{diag}(\lambda_{1k}, \ldots, \lambda_{nk})`` contains the regime-specific shock variances

Given two regime covariance matrices, the eigendecomposition of ``\Sigma_1^{-1}\Sigma_2`` recovers the structural parameters:

```math
\Sigma_1^{-1}\Sigma_2 = V D V^{-1}
```

where:
- ``V`` contains the eigenvectors identifying the columns of ``B_0`` up to scaling
- ``D = \text{diag}(d_1, \ldots, d_n)`` contains the relative variance ratios ``d_j = \lambda_{j2} / \lambda_{j1}``
- ``B_0 = \Sigma_1^{1/2} V`` after normalization

**Identification condition**: The eigenvalues ``d_j`` must be distinct. With ``K \geq 2`` regimes providing distinct eigenvalues, ``B_0`` is identified up to column permutation and sign.

!!! note "Technical Note"
    The implementation uses `safe_cholesky` for ``\Sigma_1^{1/2}`` and polar decomposition to enforce orthogonality of the rotation matrix ``Q = L^{-1} B_0``. A positive-diagonal sign convention normalizes the result.

---

## Markov-Switching Volatility

Markov-switching identification (Lanne & Lutkepohl 2008) estimates regime-specific covariance matrices via the Hamilton (1989) filter and EM algorithm. The latent state ``S_t \in \{1, \ldots, K\}`` follows a first-order Markov chain with transition matrix ``P`` where ``P_{ij} = P(S_t = j \mid S_{t-1} = i)``.

```math
f(u_t \mid S_t = k) = (2\pi)^{-n/2} |\Sigma_k|^{-1/2} \exp\!\left(-\tfrac{1}{2} u_t' \Sigma_k^{-1} u_t\right)
```

where:
- ``u_t`` is the ``n \times 1`` vector of reduced-form residuals
- ``\Sigma_k`` is the ``n \times n`` covariance matrix for regime ``k``

The EM algorithm iterates:

1. **E-step**: Hamilton (1989) forward filter computes filtered probabilities ``\xi_{t|t}(k)``. Kim (1994) backward smoother produces smoothed probabilities ``\xi_{t|T}(k)``.

2. **M-step**: Updates regime covariances as weighted sample covariances. Updates the transition matrix using Kim (1994) joint smoothed probabilities ``\xi_{t,t-1|T}(i,j)``.

!!! note "Kim (1994) Joint Smoother"
    The transition matrix update uses ``\xi_{t,t-1|T}(i,j) = \xi_{t|T}(j) \cdot P_{ij} \cdot \xi_{t-1|t-1}(i) / \xi_{t|t-1}(j)`` rather than the naive product of marginal smoothed probabilities. This accounts for serial dependence in regime assignments and produces unbiased transition matrix estimates.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ms = identify_markov_switching(model; n_regimes=2)
report(ms)
```

High persistence in the transition matrix diagonal indicates well-separated regimes, strengthening identification. Smoothed probabilities in `ms.regime_probs` reveal the timing and duration of volatility regimes.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_regimes` | `Int` | `2` | Number of volatility regimes |
| `max_iter` | `Int` | `500` | Maximum EM iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance for log-likelihood |

**Return value** (`MarkovSwitchingSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime |
| `regime_probs` | `Matrix{T}` | Smoothed regime probabilities (``T \times K``) |
| `transition_matrix` | `Matrix{T}` | Markov transition probabilities (``K \times K``) |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |
| `n_regimes` | `Int` | Number of regimes |

---

## GARCH-Based Identification

GARCH-based identification (Normandin & Phaneuf 2004) uses conditional heteroskedasticity in the structural shocks. Each shock follows a GARCH(1,1) process:

```math
h_{j,t} = \omega_j + \alpha_j \varepsilon_{j,t-1}^2 + \beta_j h_{j,t-1}
```

where:
- ``h_{j,t}`` is the conditional variance of shock ``j`` at time ``t``
- ``\omega_j > 0``, ``\alpha_j \geq 0``, ``\beta_j \geq 0`` with ``\alpha_j + \beta_j < 1``

The structural impact matrix ``B_0`` is estimated by maximizing:

```math
\ell(B_0) = -\frac{1}{2} \sum_{t=1}^{T} \left[ n \ln(2\pi) + \sum_{j=1}^{n} \ln h_{j,t} + \sum_{j=1}^{n} \frac{\varepsilon_{j,t}^2}{h_{j,t}} \right] + T \ln|\det(B_0^{-1})|
```

where ``\varepsilon_t = B_0^{-1} u_t`` and each ``h_{j,t}`` follows the GARCH recursion. The estimation iterates: (1) initialize ``B_0`` from Cholesky, (2) fit GARCH(1,1) to each shock, (3) re-estimate ``B_0`` via Givens rotation optimization, (4) repeat until convergence.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

garch = identify_garch(model)
report(garch)
```

Large ``\alpha`` values indicate strong ARCH effects, strengthening identification. The conditional variances in `garch.cond_var` trace time-varying volatility per shock.

**Return value** (`GARCHSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `garch_params` | `Matrix{T}` | ``n \times 3`` matrix: ``[\omega, \alpha, \beta]`` per shock |
| `cond_var` | `Matrix{T}` | ``T \times n`` conditional variances |
| `shocks` | `Matrix{T}` | Structural shocks |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |

---

## Smooth Transition

Smooth-transition identification (Lutkepohl & Netsunajev 2017) allows gradual volatility shifts via a logistic transition function:

```math
\Sigma_t = B_0 \bigl[I + G(s_t)(\Lambda - I)\bigr] B_0'
```

where:
- ``G(s_t) = 1 / (1 + \exp(-\gamma(s_t - c)))`` is the logistic transition function
- ``s_t`` is an observable transition variable
- ``\gamma > 0`` controls transition speed (large ``\gamma`` approximates a discrete switch)
- ``c`` is the threshold location
- ``\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`` contains relative variances in the second regime

When ``G = 0`` the covariance equals ``B_0 B_0'``; when ``G = 1`` it equals ``B_0 \Lambda B_0'``.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

s = Y[2:end, 1]
st = identify_smooth_transition(model, s)
report(st)
```

**Return value** (`SmoothTransitionSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance matrices for extreme regimes |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime |
| `gamma` | `T` | Transition speed parameter |
| `threshold` | `T` | Transition location parameter |
| `G_values` | `Vector{T}` | Evaluated transition function ``G(s_t)`` |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |

---

## External Volatility Instruments

When volatility regimes are known a priori (NBER recessions, financial crises, policy regime changes), external volatility identification (Rigobon 2003) splits the sample and estimates regime-specific covariance matrices directly. This is the simplest heteroskedasticity method --- no latent state estimation or iterative optimization.

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

T_obs = size(model.U, 1)
regime = vcat(fill(1, T_obs ÷ 2), fill(2, T_obs - T_obs ÷ 2))
ev = identify_external_volatility(model, regime)
report(ev)
```

**Return value** (`ExternalVolatilitySVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime |
| `regime_indices` | `Vector{Vector{Int}}` | Observation indices per regime |
| `loglik` | `T` | Log-likelihood |

---

## Complete Example

```julia
using MacroEconometricModels

# Three-variable monetary policy VAR
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Step 1: Markov-switching identification
ms = identify_markov_switching(model; n_regimes=2)
report(ms)

# Step 2: Check convergence and regime structure
println("Converged: ", ms.converged)
println("Transition matrix:")
println(round.(ms.transition_matrix, digits=3))

# Step 3: Inspect relative variance ratios
for k in 1:ms.n_regimes
    println("Regime $k variances: ", round.(ms.Lambda[k], digits=3))
end

# Step 4: Compare with GARCH identification
garch = identify_garch(model)
report(garch)

# Step 5: Structural IRFs and FEVD
irfs = irf(model, 20; method=:markov_switching)
report(irfs)

decomp = fevd(model, 20; method=:markov_switching)
report(decomp)
```

---

## Common Pitfalls

1. **Indistinct eigenvalues**: If two shocks experience identical proportional variance changes, their columns in ``B_0`` are not separately identified. Check that `ms.Lambda` vectors show distinct values. Use `test_identification_strength` from the [Identification Testing](@ref id_testing_page) page.

2. **EM convergence to local optima**: The Markov-switching EM algorithm is sensitive to initialization. If `ms.converged == false`, increase `max_iter` or re-run with different random seeds.

3. **Short regimes**: Regime-specific covariance estimation requires at least ``n + 1`` observations per regime. External volatility falls back to the full-sample covariance for undersized regimes.

4. **GARCH stationarity**: The constraint ``\alpha + \beta < 1`` is enforced via log-transforms. Near-IGARCH behavior can produce boundary solutions. Inspect `garch.garch_params` to verify interior estimates.

5. **Labeling problem**: All methods identify ``B_0`` up to column permutation and sign. Economic reasoning is needed to label shocks. The package normalizes by positive diagonal.

---

## References

- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777--792. [DOI](https://doi.org/10.1162/003465303772815727)

- Lanne, Markku, and Helmut Lutkepohl. 2008. "Identifying Monetary Policy Shocks via Changes in Volatility." *Journal of Money, Credit and Banking* 40 (6): 1131--1149. [DOI](https://doi.org/10.1111/j.1538-4616.2008.00151.x)

- Normandin, Michel, and Louis Phaneuf. 2004. "Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility." *Journal of Monetary Economics* 51 (6): 1217--1243. [DOI](https://doi.org/10.1016/j.jmoneco.2003.11.002)

- Lutkepohl, Helmut, and Aleksei Netsunajev. 2017. "Structural Vector Autoregressions with Smooth Transition in Variances." *Journal of Economic Dynamics and Control* 84: 43--57. [DOI](https://doi.org/10.1016/j.jedc.2017.09.001)

- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086--3124. [DOI](https://doi.org/10.1093/restud/rdab009)

- Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57 (2): 357--384. [DOI](https://doi.org/10.2307/1912559)

- Kim, Chang-Jin. 1994. "Dynamic Linear Models with Markov-Switching." *Journal of Econometrics* 60 (1--2): 1--22. [DOI](https://doi.org/10.1016/0304-4076(94)90036-1)
