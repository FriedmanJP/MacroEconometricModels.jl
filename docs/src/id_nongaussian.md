# [Non-Gaussian Methods](@id id_nongaussian_page)

Non-Gaussian structural VAR identification recovers the structural impact matrix ``B_0`` by exploiting the statistical independence and non-Gaussianity of structural shocks. The Darmois-Skitovich theorem (Comon 1994) establishes that if at most one shock is Gaussian, ``B_0`` is unique up to column permutation and sign — without imposing any economic restrictions.

This page covers two complementary approaches:

- **ICA-based methods** (nonparametric): FastICA, JADE, SOBI, distance covariance, HSIC
- **Maximum likelihood methods** (parametric): Student-t, mixture of normals, PML, skew-normal

For heteroskedasticity-based identification, see [Heteroskedasticity](@ref id_heteroskedastic_page). For identifiability diagnostics, see [Testing](@ref id_testing_page).

## Quick Start

**Recipe 1: FastICA identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Identify structural shocks via FastICA (Hyvärinen 1999)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ica = identify_fastica(model)
report(ica)
```

**Recipe 2: JADE identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Fourth-order cumulant diagonalization (Cardoso & Souloumiac 1993)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
jade = identify_jade(model)
report(jade)
```

**Recipe 3: Student-t ML identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Parametric ML with shock-specific degrees of freedom (Lanne, Meitz & Saikkonen 2017)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ml = identify_student_t(model)
report(ml)
```

**Recipe 4: Mixture of normals ML identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Two-component Gaussian mixture shocks (Lanne & Lütkepohl 2010)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
ml = identify_mixture_normal(model)
report(ml)
```

**Recipe 5: Model comparison across distributions**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Compare AIC/BIC across all non-Gaussian distributions
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
    ml = identify_nongaussian_ml(model; distribution=dist)
    println("$dist: AIC=$(round(ml.aic, digits=2)), BIC=$(round(ml.bic, digits=2))")
end
```

**Recipe 6: IRF via non-Gaussian identification**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Non-Gaussian identification integrates directly with the IRF pipeline
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)
irfs = irf(model, 20; method=:fastica)
report(irfs)
```

---

## ICA-Based Methods

Independent Component Analysis (ICA) identifies the structural impact matrix ``B_0`` by searching for the orthogonal rotation ``Q`` that makes recovered shocks maximally independent and non-Gaussian. These methods are nonparametric — they do not assume a specific distributional form for the shocks.

The model specification decomposes the reduced-form residuals as:

```math
u_t = B_0 \varepsilon_t, \quad B_0 = L \, Q
```

where:
- ``u_t`` is the ``n \times 1`` vector of reduced-form VAR residuals
- ``\varepsilon_t`` is the ``n \times 1`` vector of independent structural shocks
- ``L = \text{chol}(\Sigma)`` is the lower Cholesky factor of the residual covariance
- ``Q`` is an ``n \times n`` orthogonal rotation matrix

ICA searches over orthogonal ``Q`` to maximize a measure of non-Gaussianity or minimize a measure of statistical dependence among the recovered shocks ``\varepsilon_t = (LQ)^{-1} u_t``.

**Identification condition** (Darmois-Skitovich theorem): At most one structural shock may be Gaussian. If all shocks are non-Gaussian, ``B_0`` is unique up to column permutation and sign (Comon 1994; Lanne, Meitz & Saikkonen 2017).

### FastICA

FastICA (Hyvärinen 1999) finds the unmixing matrix by maximizing **negentropy** — a non-negative measure of non-Gaussianity — via a fixed-point iteration. The algorithm pre-whitens the residuals so that ``\text{Cov}(Z) = I``, then searches for orthogonal directions of maximum non-Gaussianity.

Three contrast functions ``G(u)`` approximate negentropy:
- `:logcosh` (default) — ``G(u) = \log\cosh(u)``, robust general-purpose choice
- `:exp` — ``G(u) = -\exp(-u^2/2)``, suited for super-Gaussian sources
- `:kurtosis` — ``G(u) = u^4/4``, classical kurtosis-based measure

Two extraction approaches control how components are recovered:
- `:deflation` — extracts components one at a time, orthogonalizing against previously found components
- `:symmetric` — extracts all components simultaneously via symmetric decorrelation ``W \leftarrow (WW')^{-1/2} W``

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Default: logcosh contrast, deflation approach
ica1 = identify_fastica(model)
report(ica1)

# Symmetric approach with exponential contrast
ica2 = identify_fastica(model; approach=:symmetric, contrast=:exp)
report(ica2)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `contrast` | `Symbol` | `:logcosh` | Non-Gaussianity measure: `:logcosh`, `:exp`, `:kurtosis` |
| `approach` | `Symbol` | `:deflation` | Extraction approach: `:deflation` or `:symmetric` |
| `max_iter` | `Int` | `200` | Maximum iterations per component |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### JADE

JADE (Joint Approximate Diagonalization of Eigenmatrices; Cardoso & Souloumiac 1993) computes fourth-order cumulant matrices and finds the orthogonal matrix ``V`` that simultaneously diagonalizes all of them via Jacobi rotations. The fourth-order cumulant matrix ``C_{ij}`` has entries:

```math
C_{ij}[k,l] = \text{cum}(z_k, z_l, z_i, z_j) = E[z_k z_l z_i z_j] - E[z_k z_l] E[z_i z_j] - E[z_k z_i] E[z_l z_j] - E[z_k z_j] E[z_l z_i]
```

where ``z_t`` are the pre-whitened residuals. Joint diagonalization minimizes the sum of squared off-diagonal elements across all cumulant matrices.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

jade = identify_jade(model; max_iter=100)
report(jade)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `100` | Maximum Jacobi rotation sweeps |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance on rotation angle |

### SOBI

SOBI (Second-Order Blind Identification; Belouchrani et al. 1997) exploits temporal structure by jointly diagonalizing autocovariance matrices at multiple lags. Unlike FastICA and JADE, SOBI uses only second-order statistics (autocovariances), making it suitable when temporal dependence is the primary source of identifiability.

The autocovariance matrix at lag ``\tau`` is:

```math
R(\tau) = \frac{1}{T - \tau} \sum_{t=1}^{T-\tau} z_{t+\tau} z_t'
```

SOBI finds the orthogonal ``V`` that simultaneously diagonalizes ``\{R(\tau)\}_{\tau \in \text{lags}}``.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

sobi = identify_sobi(model; lags=1:12)
report(sobi)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `AbstractRange` | `1:12` | Lag range for autocovariance matrices |
| `max_iter` | `Int` | `100` | Maximum Jacobi rotation sweeps |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### Distance Covariance

Distance covariance (Székely et al. 2007) measures dependence between random vectors and equals zero if and only if the variables are independent. The `identify_dcov` function minimizes the sum of pairwise distance covariances between recovered shocks by optimizing over Givens rotation angles (Matteson & Tsay 2017).

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

dcov = identify_dcov(model; max_iter=200)
report(dcov)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `200` | Maximum Nelder-Mead iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### HSIC

The Hilbert-Schmidt Independence Criterion (Gretton et al. 2005) measures dependence using kernel embeddings. With a characteristic kernel (Gaussian), HSIC equals zero if and only if variables are independent. The `identify_hsic` function minimizes pairwise HSIC between recovered shocks.

The Gaussian kernel bandwidth ``\sigma`` defaults to the median pairwise distance heuristic when set to `1.0`.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

hsic = identify_hsic(model; sigma=1.0)
report(hsic)
```

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `sigma` | `Real` | `1.0` | Gaussian kernel bandwidth (1.0 triggers median heuristic) |
| `max_iter` | `Int` | `200` | Maximum Nelder-Mead iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### ICASVARResult Fields

All five ICA methods return an `ICASVARResult{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``): ``u_t = B_0 \varepsilon_t`` |
| `W` | `Matrix{T}` | Unmixing matrix (``n \times n``): ``\varepsilon_t = W u_t`` |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `shocks` | `Matrix{T}` | Recovered structural shocks (``T \times n``) |
| `method` | `Symbol` | Method used: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic` |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Number of iterations |
| `objective` | `T` | Final objective value |

---

## Maximum Likelihood Methods

Maximum likelihood methods estimate ``B_0`` and the shock distribution parameters jointly. The non-Gaussian log-likelihood is:

```math
\ell(\theta) = \sum_{t=1}^{T} \left[ \log|\det(B_0^{-1})| + \sum_{j=1}^{n} \log f_j(\varepsilon_{j,t};\, \theta_j) \right]
```

where:
- ``\varepsilon_t = B_0^{-1} u_t`` are the structural shocks
- ``f_j(\cdot;\, \theta_j)`` is the marginal density of shock ``j`` with parameters ``\theta_j``
- ``B_0 = L Q`` is parameterized via ``n(n-1)/2`` Givens rotation angles

The optimizer searches over rotation angles and distribution parameters simultaneously using Nelder-Mead. Standard errors for ``B_0`` are computed from the numerical Hessian of the log-likelihood.

!!! note "Technical Note"
    All distribution parameters use unconstrained reparameterizations internally. Student-t degrees of freedom are ``\nu = \exp(\theta) + 2.01`` to ensure ``\nu > 2`` (finite variance). Mixture probabilities use the logistic transform ``p = 1/(1 + \exp(-\theta))``. Variance parameters in the mixture model use sigmoid bounds to guarantee positivity of both component variances under the unit variance constraint.

### Student-t

Each shock follows a standardized Student-t distribution with shock-specific degrees of freedom ``\nu_j`` (Lanne, Meitz & Saikkonen 2017):

```math
f_j(x;\, \nu_j) = \frac{\Gamma((\nu_j+1)/2)}{\sqrt{\pi \nu_j}\, \Gamma(\nu_j/2)} \left(1 + \frac{x^2}{\nu_j}\right)^{-(\nu_j+1)/2} \cdot \sqrt{\frac{\nu_j}{\nu_j - 2}}
```

The scaling factor ``\sqrt{\nu_j / (\nu_j - 2)}`` standardizes the variance to unity. Low ``\nu_j`` indicates heavy tails; as ``\nu_j \to \infty``, shock ``j`` approaches Gaussianity.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ml = identify_student_t(model)
report(ml)
```

The estimated degrees of freedom are available in `ml.dist_params[:nu]`. Values below 10 indicate substantial departure from Gaussianity.

### Mixture of Normals

Each shock follows a two-component Gaussian mixture (Lanne & Lütkepohl 2010):

```math
f_j(x;\, p_j, \sigma_{1j}, \sigma_{2j}) = p_j \, \phi(x / \sigma_{1j}) / \sigma_{1j} + (1 - p_j) \, \phi(x / \sigma_{2j}) / \sigma_{2j}
```

where ``\phi(\cdot)`` is the standard normal density. The unit variance constraint ``p_j \sigma_{1j}^2 + (1 - p_j) \sigma_{2j}^2 = 1`` reduces the free parameters to ``p_j`` and ``\sigma_{1j}`` per shock. The second variance is derived as ``\sigma_{2j}^2 = (1 - p_j \sigma_{1j}^2) / (1 - p_j)``.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ml = identify_mixture_normal(model)
report(ml)
```

The mixing probabilities are in `ml.dist_params[:p_mix]` and the first component standard deviations in `ml.dist_params[:sigma1]`. Identification requires that the two components differ (``\sigma_{1j} \neq \sigma_{2j}``).

### PML (Pearson Type IV)

Pseudo Maximum Likelihood (Herwartz 2018) uses Pearson Type IV distributions that accommodate both skewness and excess kurtosis. Each shock has two parameters: a skewness correction ``\kappa_j`` and a tail parameter ``\nu_j``.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ml = identify_pml(model)
report(ml)
```

PML parameters are in `ml.dist_params[:kappa]` (skewness) and `ml.dist_params[:nu]` (kurtosis).

### Skew-Normal

Each shock follows a skew-normal distribution (Azzalini 1985) with pdf:

```math
f_j(x;\, \alpha_j) = 2 \, \phi(x) \, \Phi(\alpha_j x)
```

where:
- ``\phi(\cdot)`` is the standard normal pdf
- ``\Phi(\cdot)`` is the standard normal cdf
- ``\alpha_j`` controls the direction and degree of skewness

When ``\alpha_j = 0``, the distribution reduces to the standard normal.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ml = identify_skew_normal(model)
report(ml)
```

The skewness parameters are in `ml.dist_params[:alpha]`. Positive ``\alpha`` produces right skew; negative ``\alpha`` produces left skew.

### Unified Dispatcher

The `identify_nongaussian_ml` function selects the distribution at runtime, enabling systematic model comparison:

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
    ml = identify_nongaussian_ml(model; distribution=dist)
    println("$dist: logL=$(round(ml.loglik, digits=2)), AIC=$(round(ml.aic, digits=2)), BIC=$(round(ml.bic, digits=2))")
end
```

Select the distribution with the lowest AIC or BIC. The `loglik_gaussian` field stores the Gaussian log-likelihood for likelihood ratio testing.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `distribution` | `Symbol` | `:student_t` | Distribution: `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal` |
| `max_iter` | `Int` | `500` | Maximum Nelder-Mead iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### NonGaussianMLResult Fields

All ML methods return a `NonGaussianMLResult{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``) |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `shocks` | `Matrix{T}` | Structural shocks (``T \times n``) |
| `distribution` | `Symbol` | Distribution used: `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal` |
| `loglik` | `T` | Log-likelihood at MLE |
| `loglik_gaussian` | `T` | Gaussian log-likelihood (for LR test) |
| `dist_params` | `Dict{Symbol,Any}` | Distribution parameters (keys vary by distribution) |
| `vcov` | `Matrix{T}` | Asymptotic covariance of all parameters |
| `se` | `Matrix{T}` | Standard errors for ``B_0`` elements |
| `converged` | `Bool` | Convergence status |
| `aic` | `T` | Akaike information criterion |
| `bic` | `T` | Bayesian information criterion |

!!! note "Moment-Based GMM Approaches"
    Keweloh (2021) and Lanne & Luoto (2021) develop GMM estimators that exploit coskewness and cokurtosis conditions directly — ``E[\varepsilon_i^2 \varepsilon_j] = 0`` and ``E[\varepsilon_i^3 \varepsilon_j] = 0`` — without specifying a parametric distribution for the shocks. These moment-based approaches offer robustness to distributional misspecification and are an important emerging direction in the literature (Lewis 2025, Section 4.3). Not yet implemented in this package.

---

## Complete Example

This workflow estimates a three-variable monetary policy VAR, identifies structural shocks using both ICA and ML approaches, compares models via information criteria, and computes impulse responses.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# --- Data: FRED-MD monetary policy model ---
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# --- Step 1: ICA identification (nonparametric) ---
ica = identify_fastica(model; contrast=:logcosh, approach=:deflation)
report(ica)

# --- Step 2: ML identification (parametric, Student-t) ---
ml_t = identify_student_t(model)
report(ml_t)

# --- Step 3: Compare distributions via AIC ---
best_dist = :student_t
best_aic = Inf
for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
    ml = identify_nongaussian_ml(model; distribution=dist)
    println("$dist: AIC=$(round(ml.aic, digits=2)), BIC=$(round(ml.bic, digits=2))")
    if ml.aic < best_aic
        best_aic = ml.aic
        best_dist = dist
    end
end
println("Best distribution by AIC: $best_dist")

# --- Step 4: Compute IRFs using preferred identification ---
irfs = irf(model, 20; method=:fastica)
report(irfs)
```

The ICA approach provides a quick nonparametric identification. The ML approach adds distributional detail — shock-specific tail parameters, standard errors for ``B_0``, and formal model comparison via AIC/BIC. Both integrate seamlessly with `irf`, `fevd`, and `historical_decomposition` through the `method` keyword.

---

## Common Pitfalls

1. **Gaussian shocks defeat identification.** Non-Gaussian methods require at most one Gaussian shock. Run `test_shock_gaussianity` or `normality_test_suite` on the VAR residuals before applying ICA or ML. If residuals are multivariate normal, consider heteroskedasticity-based methods instead.

2. **Column ordering is not structural.** Statistical identification recovers ``B_0`` up to column permutation and sign. The package normalizes signs (positive diagonal), but economic labeling of shocks still requires external information. See Lewis (2025, Section 6.4) on the labeling problem.

3. **Small samples weaken ICA.** FastICA and JADE estimate higher-order statistics (negentropy, fourth-order cumulants) that converge slowly. With fewer than 100 observations, consider ML methods that impose parametric structure, or use SOBI which relies on second-order autocovariances.

4. **Nelder-Mead can find local optima.** The ML estimator and the dcov/HSIC methods use derivative-free optimization. For high-dimensional systems (``n > 4``), run multiple initializations or compare results across ICA and ML methods. Consistent ``B_0`` estimates across methods strengthen confidence in identification.

5. **The LR test requires correct nesting.** The likelihood ratio test in `NonGaussianMLResult` compares non-Gaussian vs. Gaussian shocks. The test is valid only when the Gaussian model is nested within the non-Gaussian specification. For Student-t, this corresponds to ``\nu_j \to \infty``; for mixtures, to ``\sigma_{1j} = \sigma_{2j}``.

---

## References

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis."
  *IEEE Transactions on Neural Networks* 10 (3): 626--634. [DOI](https://doi.org/10.1109/72.761722)

- Cardoso, Jean-François, and Antoine Souloumiac. 1993. "Blind Beamforming for Non-Gaussian Signals."
  *IEE Proceedings-F* 140 (6): 362--370. [DOI](https://doi.org/10.1049/ip-f-2.1993.0054)

- Belouchrani, Adel, Karim Abed-Meraim, Jean-François Cardoso, and Eric Moulines. 1997. "A Blind Source Separation Technique Using Second-Order Statistics."
  *IEEE Transactions on Signal Processing* 45 (2): 434--444. [DOI](https://doi.org/10.1109/78.554307)

- Comon, Pierre. 1994. "Independent Component Analysis, A New Concept?"
  *Signal Processing* 36 (3): 287--314. [DOI](https://doi.org/10.1016/0165-1684(94)90029-9)

- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions."
  *Journal of Econometrics* 196 (2): 288--304. [DOI](https://doi.org/10.1016/j.jeconom.2016.06.002)

- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals."
  *Journal of Business & Economic Statistics* 28 (1): 159--168. [DOI](https://doi.org/10.1198/jbes.2009.06003)

- Herwartz, Helmut. 2018. "Hodges-Lehmann Detection of Structural Shocks: An Analysis of Macroeconomic Dynamics in the Euro Area."
  *Oxford Bulletin of Economics and Statistics* 80 (4): 736--754. [DOI](https://doi.org/10.1111/obes.12234)

- Azzalini, Adelchi. 1985. "A Class of Distributions Which Includes the Normal Ones."
  *Scandinavian Journal of Statistics* 12 (2): 171--178. [JSTOR](https://www.jstor.org/stable/4615982)

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances."
  *Annals of Statistics* 35 (6): 2769--2794. [DOI](https://doi.org/10.1214/009053607000000505)

- Gretton, Arthur, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. "Measuring Statistical Dependence with Hilbert-Schmidt Norms."
  In *Algorithmic Learning Theory*, 63--77. Berlin: Springer. [DOI](https://doi.org/10.1007/11564089_7)

- Matteson, David S., and Ruey S. Tsay. 2017. "Independent Component Analysis via Distance Covariance."
  *Journal of the American Statistical Association* 112 (518): 623--637. [DOI](https://doi.org/10.1080/01621459.2016.1150851)

- Lewis, Daniel J. 2025. "Identification Based on Higher Moments in Macroeconometrics."
  *Annual Review of Economics* 17: 665--693. [DOI](https://doi.org/10.1146/annurev-economics-070124-051419)

- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments."
  *Journal of Business & Economic Statistics* 39 (3): 772--882. [DOI](https://doi.org/10.1080/07350015.2020.1730858)
