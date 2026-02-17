# Statistical Identification via Higher Moments

This page covers identification of structural VAR models using statistical properties beyond second moments: **heteroskedasticity** (time-varying variances) and **non-Gaussianity** (higher-order moments). These methods provide identification without requiring the recursive ordering of Cholesky or the a priori sign/zero restrictions of traditional SVAR.

The classification follows Lewis (2025), which provides the definitive taxonomy of statistical identification in macroeconometrics. The key insight is that the standard reduced-form covariance ``\Sigma = B_0 B_0'`` provides only ``n(n+1)/2`` equations for ``n^2`` unknowns in ``B_0``. The two strands of statistical identification resolve this in complementary ways:

1. **Heteroskedasticity** (Section 2): exploits *multiple* covariance matrices from different volatility regimes
2. **Non-Gaussianity** (Section 3): exploits higher-moment conditions (coskewness, cokurtosis) from non-Gaussian shocks

## Quick Start

```julia
using MacroEconometricModels

# Multivariate normality tests (diagnostics)
suite = normality_test_suite(model)                # Run all 7 tests
jb = jarque_bera_test(model)                       # Multivariate Jarque-Bera

# Non-Gaussianity: ICA-based identification
ica = identify_fastica(model)                      # FastICA (Hyvärinen 1999)
jade = identify_jade(model)                        # JADE (Cardoso 1993)

# Non-Gaussianity: ML identification
ml = identify_student_t(model)                     # Student-t shocks
ml = identify_nongaussian_ml(model; distribution=:mixture_normal)

# Heteroskedasticity: regime-based identification
ms = identify_markov_switching(model; n_regimes=2) # Markov-switching (Lanne & Lütkepohl 2008)
ev = identify_external_volatility(model, regime)   # Known volatility regimes (Rigobon 2003)

# Identifiability tests
test_shock_gaussianity(ica)                        # Are shocks non-Gaussian?
test_gaussian_vs_nongaussian(model)                # LR test: Gaussian vs non-Gaussian
test_shock_independence(ica)                       # Are shocks independent?

# Integration with existing IRF pipeline
irfs = irf(model, 20; method=:fastica)             # Works automatically via compute_Q
```

---

## The SVAR Setting

The structural VAR has the decomposition:

```math
u_t = B_0 \varepsilon_t, \quad \Sigma = B_0 B_0'
```

where
- ``u_t`` is the ``n \times 1`` vector of reduced-form residuals
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks (unit variance, mutually independent)
- ``B_0`` is the ``n \times n`` structural impact matrix

The reduced-form covariance ``\Sigma = B_0 B_0'`` provides ``n(n+1)/2`` equations for ``n^2`` unknowns in ``B_0``, leaving ``n(n-1)/2`` free parameters. Traditional approaches (Cholesky, sign/zero restrictions) resolve this by imposing economic constraints. Statistical identification takes a different path:

- **Heteroskedasticity**: if shock variances change across regimes, each regime provides a *separate* covariance equation ``\Sigma_k = B_0 \Lambda_k B_0'``, generating enough equations to identify ``B_0``
- **Non-Gaussianity**: if shocks are non-Gaussian, independence imposes conditions beyond uncorrelatedness — coskewness, cokurtosis, and higher moments pin down ``B_0``

!!! note "Technical Note"
    Lewis (2025) shows that identification via non-Gaussianity can be thought of as a special case of identification based on heteroskedasticity (p. 674). The Darmois-Skitovich theorem (Comon 1994) establishes that if at most one component is Gaussian and shocks are independent, ``B_0`` is unique up to column permutation and sign.

---

## Identification via Heteroskedasticity

*"If variances of structural shocks change through time, then there is not just a single reduced-form covariance matrix to exploit."* — Lewis (2025, Section 3)

When the structural shock variances change across ``K`` regimes while ``B_0`` remains constant, we have:

```math
\Sigma_k = B_0 \Lambda_k B_0', \quad k = 1, \ldots, K
```

where ``\Lambda_k = \text{diag}(\lambda_{1k}, \ldots, \lambda_{nk})`` are regime-specific variance matrices. With ``K \geq 2`` regimes, the eigendecomposition of ``\Sigma_1^{-1} \Sigma_2`` identifies ``B_0`` up to column permutation and sign, provided eigenvalues are distinct (Rigobon 2003).

### Eigendecomposition Identification

The core idea (Rigobon 2003): given two regime covariance matrices ``\Sigma_1`` and ``\Sigma_2``, the eigendecomposition of ``\Sigma_1^{-1}\Sigma_2`` yields:

```math
\Sigma_1^{-1}\Sigma_2 = V D V^{-1}
```

where
- ``V`` contains the eigenvectors
- ``D = \text{diag}(\lambda_1, \ldots, \lambda_n)`` contains the relative variance ratios
- ``B_0 = \Sigma_1^{1/2} V`` (with normalization)

**Identification condition**: The eigenvalues ``\lambda_j`` must be distinct.

### Markov-Switching Volatility

Estimates regime-specific covariance matrices via the Hamilton (1989) filter with EM algorithm:

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy model
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

ms = identify_markov_switching(model; n_regimes=2)
println("Transition matrix:")
println(round.(ms.transition_matrix, digits=3))
println("Regime probabilities (first 5 obs):")
println(round.(ms.regime_probs[1:5, :], digits=3))
```

The EM algorithm iterates:
1. **E-step**: Hamilton filter (forward) + Kim smoother (backward) → regime probabilities
2. **M-step**: Update regime covariances and transition matrix given probabilities

**Reference**: Lanne & Lütkepohl (2008)

### GARCH-Based Identification

Uses GARCH(1,1) conditional heteroskedasticity in the structural shocks for identification:

```math
h_{j,t} = \omega_j + \alpha_j \varepsilon_{j,t-1}^2 + \beta_j h_{j,t-1}
```

```julia
garch = identify_garch(model)
println("GARCH parameters (ω, α, β):")
for j in 1:size(garch.garch_params, 1)
    println("  Shock $j: ", round.(garch.garch_params[j, :], digits=4))
end
```

**Reference**: Normandin & Phaneuf (2004)

### Smooth Transition

The covariance varies smoothly between two regimes via a logistic transition function:

```math
\Sigma_t = B_0 [I + G(s_t)(\Lambda - I)] B_0'
```

where ``G(s_t) = 1/(1 + \exp(-\gamma(s_t - c)))`` is the logistic transition function.

```julia
# Use a lagged variable as the transition variable
s = Y[2:end, 1]  # first variable, lagged
st = identify_smooth_transition(model, s)
println("Transition speed γ = $(round(st.gamma, digits=3))")
println("Threshold c = $(round(st.threshold, digits=3))")
```

**Reference**: Lütkepohl & Netšunajev (2017)

### External Volatility Instruments

When volatility regimes are known a priori (e.g., NBER recession dates, financial crisis indicators):

```julia
# Binary regime indicator
regime = vcat(fill(1, 100), fill(2, 100))  # first half = regime 1
ev = identify_external_volatility(model, regime)
```

This is the simplest heteroskedasticity method — it just splits the sample and applies eigendecomposition identification.

**Reference**: Rigobon (2003)

### Heteroskedasticity Result Fields

**Markov-Switching** (`MarkovSwitchingSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime |
| `regime_probs` | `Matrix{T}` | Smoothed regime probabilities (T × K) |
| `transition_matrix` | `Matrix{T}` | Markov transition probabilities (K × K) |
| `loglik` | `T` | Log-likelihood |
| `converged` | `Bool` | Convergence status |
| `n_regimes` | `Int` | Number of regimes |

**GARCH** (`GARCHSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `garch_params` | `Matrix{T}` | (n × 3): [ω, α, β] per shock |
| `cond_var` | `Matrix{T}` | (T × n) conditional variances |
| `shocks` | `Matrix{T}` | Structural shocks |
| `loglik` | `T` | Log-likelihood |

**Smooth Transition** (`SmoothTransitionSVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `gamma` | `T` | Transition speed parameter |
| `threshold` | `T` | Transition location parameter |
| `G_values` | `Vector{T}` | Transition function values |

**External Volatility** (`ExternalVolatilitySVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Sigma_regimes` | `Vector{Matrix{T}}` | Covariance per regime |
| `Lambda` | `Vector{Vector{T}}` | Relative variances per regime |
| `regime_indices` | `Vector{Vector{Int}}` | Observation indices per regime |

---

## Identification via Non-Gaussianity

*"Identification via non-Gaussianity can be thought of as a special case of identification based on heteroskedasticity."* — Lewis (2025, p. 674)

The Darmois-Skitovich theorem establishes that if ``\varepsilon_t`` has independent components and at most one is Gaussian, then the mixing matrix ``B_0`` is unique up to column permutation and sign (Comon 1994). This provides identification from a *single* sample without requiring volatility changes.

### ICA-Based Methods (Nonparametric)

Independent Component Analysis (ICA) identifies ``B_0`` by finding the rotation ``Q`` that makes the recovered shocks ``\varepsilon_t = (B_0)^{-1} u_t`` maximally independent and non-Gaussian.

#### Model Specification

```math
u_t = B_0 \varepsilon_t, \quad B_0 = L Q
```

where ``L = \text{chol}(\Sigma)`` and ``Q`` is orthogonal. ICA searches over orthogonal ``Q`` to maximize a measure of non-Gaussianity or independence.

**Identification condition**: At most one structural shock may be Gaussian (Lanne, Meitz & Saikkonen 2017). If all shocks are non-Gaussian, ``B_0`` is unique up to column permutation and sign.

#### FastICA

FastICA (Hyvärinen 1999) finds the unmixing matrix by maximizing a measure of non-Gaussianity (negentropy) via a fixed-point algorithm.

```julia
# Default: logcosh contrast, deflation approach
ica = identify_fastica(model)

# Symmetric approach with exponential contrast
ica = identify_fastica(model; approach=:symmetric, contrast=:exp)
```

Three contrast functions are available:
- `:logcosh` (default) — robust, good general-purpose choice: ``G(u) = \log\cosh(u)``
- `:exp` — better for super-Gaussian sources: ``G(u) = -\exp(-u^2/2)``
- `:kurtosis` — classical kurtosis-based: ``G(u) = u^4/4``

Two extraction approaches:
- `:deflation` — extracts components one at a time (deflation approach)
- `:symmetric` — extracts all components simultaneously

**Reference**: Hyvärinen (1999)

#### JADE

JADE (Joint Approximate Diagonalization of Eigenmatrices) uses fourth-order cumulant matrices and joint diagonalization via Jacobi rotations.

```julia
jade = identify_jade(model)
```

JADE computes the fourth-order cumulant matrices ``C_{ij}[k,l] = \text{cum}(z_k, z_l, z_i, z_j)`` and finds the orthogonal matrix that simultaneously diagonalizes all of them.

**Reference**: Cardoso & Souloumiac (1993)

#### SOBI

SOBI (Second-Order Blind Identification) exploits temporal structure via autocovariance matrices at multiple lags.

```julia
sobi = identify_sobi(model; lags=1:12)
```

Unlike FastICA and JADE which use higher-order statistics, SOBI only uses second-order statistics (autocovariances), making it suitable when temporal dependence is the main source of identifiability.

**Reference**: Belouchrani et al. (1997)

#### Distance Covariance

Minimizes the sum of pairwise distance covariances between recovered shocks. Distance covariance (Székely et al. 2007) is zero if and only if variables are independent.

```julia
dcov = identify_dcov(model)
```

**Reference**: Matteson & Tsay (2017)

#### HSIC

Minimizes the Hilbert-Schmidt Independence Criterion using a Gaussian kernel. Like distance covariance, HSIC with a characteristic kernel is zero iff variables are independent.

```julia
hsic = identify_hsic(model; sigma=1.0)
```

The bandwidth parameter ``\sigma`` defaults to the median pairwise distance heuristic.

**Reference**: Gretton et al. (2005)

### Maximum Likelihood Methods (Parametric)

Instead of the two-step ICA approach, ML methods estimate ``B_0`` and the shock distribution parameters jointly by maximizing the log-likelihood:

```math
\ell(\theta) = \sum_{t=1}^T \left[ \log|\det(B_0^{-1})| + \sum_{j=1}^n \log f_j(\varepsilon_{j,t}; \theta_j) \right]
```

where ``f_j(\cdot; \theta_j)`` is the marginal density of shock ``j`` and ``\theta_j`` are distribution-specific parameters.

#### Student-t Shocks

Assumes each shock follows a (standardized) Student-t distribution with shock-specific degrees of freedom ``\nu_j``:

```julia
ml = identify_student_t(model)
println("Degrees of freedom: ", ml.dist_params[:nu])
```

Low ``\nu`` indicates heavy tails. When ``\nu \to \infty``, the shock approaches Gaussianity. Identification requires that at most one shock has ``\nu = \infty``.

**Reference**: Lanne, Meitz & Saikkonen (2017)

#### Mixture of Normals

Each shock follows a mixture of two normals: ``\varepsilon_j \sim p_j N(0, \sigma_{1j}^2) + (1-p_j) N(0, \sigma_{2j}^2)`` with the unit variance constraint ``p_j \sigma_{1j}^2 + (1-p_j) \sigma_{2j}^2 = 1``.

```julia
ml = identify_mixture_normal(model)
println("Mixing probabilities: ", ml.dist_params[:p_mix])
```

**Reference**: Lanne & Lütkepohl (2010)

#### Pseudo Maximum Likelihood (PML)

Uses Pearson Type IV distributions, allowing both skewness and excess kurtosis.

```julia
ml = identify_pml(model)
```

**Reference**: Herwartz (2018)

#### Skew-Normal Shocks

Each shock follows a skew-normal distribution with pdf ``f(x) = 2\phi(x)\Phi(\alpha_j x)``.

```julia
ml = identify_skew_normal(model)
println("Skewness parameters: ", ml.dist_params[:alpha])
```

**Reference**: Azzalini (1985)

#### Unified Dispatcher

Use `identify_nongaussian_ml` to select the distribution at runtime:

```julia
for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
    ml = identify_nongaussian_ml(model; distribution=dist)
    println("$dist: logL=$(round(ml.loglik, digits=2)), AIC=$(round(ml.aic, digits=2))")
end
```

Compare AIC/BIC across distributions to select the best-fitting specification.

### Moment-Based Approaches

!!! note "Emerging Direction"
    Moment-based GMM estimators (Keweloh 2021, Lanne & Luoto 2021) exploit coskewness and cokurtosis conditions directly, without specifying a parametric distribution. These use conditions like:
    - ``E[\varepsilon_i^2 \varepsilon_j] = 0`` (coskewness)
    - ``E[\varepsilon_i^2 \varepsilon_j^2] - 1 = 0``, ``E[\varepsilon_i^3 \varepsilon_j] = 0`` (cokurtosis)

    This is an important emerging direction in the literature. See Lewis (2025, Section 4.3) for a comprehensive discussion. Not yet implemented in this package.

### ICA / ML Result Fields

**ICA** (`ICASVARResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``) |
| `W` | `Matrix{T}` | Unmixing matrix: ``\varepsilon_t = W u_t`` |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `shocks` | `Matrix{T}` | Recovered structural shocks (``T \times n``) |
| `method` | `Symbol` | Method used |
| `converged` | `Bool` | Whether the algorithm converged |
| `iterations` | `Int` | Number of iterations |
| `objective` | `T` | Final objective value |

**ML** (`NonGaussianMLResult`):

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix |
| `Q` | `Matrix{T}` | Rotation matrix |
| `shocks` | `Matrix{T}` | Structural shocks |
| `distribution` | `Symbol` | Distribution used |
| `loglik` | `T` | Log-likelihood at MLE |
| `loglik_gaussian` | `T` | Gaussian log-likelihood (for LR test) |
| `dist_params` | `Dict{Symbol,Any}` | Distribution parameters |
| `vcov` | `Matrix{T}` | Asymptotic covariance of parameters |
| `se` | `Matrix{T}` | Standard errors for ``B_0`` |
| `converged` | `Bool` | Convergence status |
| `aic` | `T` | Akaike information criterion |
| `bic` | `T` | Bayesian information criterion |

---

## Multivariate Normality Tests

Before applying non-Gaussian SVAR methods, it is essential to verify that the VAR residuals are indeed non-Gaussian. If residuals are Gaussian, non-Gaussian identification will not work (the problem is unidentified). These tests also serve as a prerequisite diagnostic for choosing between heteroskedasticity-based and non-Gaussianity-based approaches.

### Multivariate Jarque-Bera Test

The multivariate Jarque-Bera test extends the univariate JB test to vector residuals. Under the null hypothesis of multivariate normality, the test statistic is:

```math
JB = T \cdot \frac{b_{1,k}}{6} + T \cdot \frac{(b_{2,k} - k(k+2))^2}{24k}
```

where ``b_{1,k}`` is the multivariate skewness measure and ``b_{2,k}`` is the multivariate kurtosis measure (Lütkepohl 2005, §4.5).

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy model
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Joint test
jb = jarque_bera_test(model)
println("Statistic: $(round(jb.statistic, digits=4)), p-value: $(round(jb.pvalue, digits=4))")

# Component-wise test on standardized residuals
jb_comp = jarque_bera_test(model; method=:component)
println("Component p-values: ", round.(jb_comp.component_pvalues, digits=4))
```

With macroeconomic data, non-normality is common — rejecting the null supports using non-Gaussian SVAR methods below.

### Mardia's Tests

Mardia (1970) proposed separate tests for multivariate skewness and kurtosis:

```math
b_{1,k} = \frac{1}{T^2} \sum_{i,j} (u_i' \Sigma^{-1} u_j)^3 \quad \text{(skewness)}
```
```math
b_{2,k} = \frac{1}{T} \sum_i (u_i' \Sigma^{-1} u_i)^2 \quad \text{(kurtosis)}
```

Under H₀: ``T \cdot b_{1,k}/6 \sim \chi^2(k(k+1)(k+2)/6)`` and ``(b_{2,k} - k(k+2)) / \sqrt{8k(k+2)/T} \sim N(0,1)``.

```julia
skew_test = mardia_test(model; type=:skewness)
kurt_test = mardia_test(model; type=:kurtosis)
both_test = mardia_test(model; type=:both)
```

The `:both` option combines both tests into a single chi-squared statistic.

**Reference**: Mardia (1970)

### Doornik-Hansen Test

The Doornik-Hansen (2008) omnibus test applies the Bowman-Shenton transformation to each component's skewness and kurtosis, producing approximately standard normal transforms ``z_1`` and ``z_2``. The test statistic is:

```math
DH = \sum_{j=1}^k (z_{1j}^2 + z_{2j}^2) \sim \chi^2(2k)
```

```julia
dh = doornik_hansen_test(model)
```

### Henze-Zirkler Test

The Henze-Zirkler (1990) test is based on the empirical characteristic function and is consistent against all alternatives. The test statistic uses a smoothing parameter ``\beta`` that depends on the sample size and dimension.

```julia
hz = henze_zirkler_test(model)
```

### Normality Test Suite

Run all tests at once with `normality_test_suite`:

```julia
suite = normality_test_suite(model)
println(suite)
```

This runs 7 tests: multivariate JB, component-wise JB, Mardia skewness, Mardia kurtosis, Mardia combined, Doornik-Hansen, and Henze-Zirkler.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `Symbol` | Test identifier |
| `statistic` | `T` | Test statistic value |
| `pvalue` | `T` | p-value |
| `df` | `Int` | Degrees of freedom |
| `n_vars` | `Int` | Number of variables |
| `n_obs` | `Int` | Number of observations |
| `components` | `Vector{T}` or `nothing` | Per-component statistics |
| `component_pvalues` | `Vector{T}` or `nothing` | Per-component p-values |

---

## Identifiability and Specification Tests

### Shock Gaussianity Test

Tests whether recovered structural shocks are non-Gaussian using univariate Jarque-Bera tests on each shock. Non-Gaussian identification requires at most one Gaussian shock.

```julia
ica = identify_fastica(model)
result = test_shock_gaussianity(ica)
println("Number of Gaussian shocks: ", result.details[:n_gaussian])
println("Identified: ", result.identified)
```

### Gaussian vs Non-Gaussian LR Test

Likelihood ratio test: ``H_0``: Gaussian shocks vs ``H_1``: non-Gaussian shocks.

```math
LR = 2(\ell_1 - \ell_0) \sim \chi^2(p)
```

where ``p`` is the number of extra distribution parameters.

```julia
lr = test_gaussian_vs_nongaussian(model; distribution=:student_t)
println("LR statistic: $(round(lr.statistic, digits=4))")
println("p-value: $(round(lr.pvalue, digits=4))")
```

Rejecting ``H_0`` supports the use of non-Gaussian identification.

### Shock Independence Test

Tests whether recovered shocks are mutually independent using both cross-correlation (portmanteau) and distance covariance tests, combined via Fisher's method.

```julia
result = test_shock_independence(ica; max_lag=10)
println("Independent: ", result.identified)  # fail-to-reject = independent
```

### Identification Strength

Bootstrap test of identification robustness: resamples residuals and measures the stability of the estimated ``B_0``.

```julia
result = test_identification_strength(model; method=:fastica, n_bootstrap=499)
println("Median Procrustes distance: $(round(result.statistic, digits=4))")
```

Small distances indicate strong identification.

### Overidentification Test

Tests consistency of additional restrictions beyond non-Gaussianity.

```julia
result = test_overidentification(model, ica; n_bootstrap=499)
println("p-value: $(round(result.pvalue, digits=4))")
```

### Weak Identification

!!! warning "Weak Identification"
    Lewis (2022) shows that weak identification is likely in many empirical applications. When variances change little across regimes, or deviations from Gaussianity are small, the identifying information may be weak. In such cases:

    - Standard Wald tests may have poor size properties
    - Confidence intervals may be unreliable
    - Point estimates may be sensitive to specification choices

    Diagnostic checks (identification strength test, shock gaussianity test) are essential. See Lewis (2022) for robust inference procedures.

---

## Integration with IRF Pipeline

All ICA, ML, and heteroskedasticity methods integrate seamlessly with the existing `irf`, `fevd`, and `historical_decomposition` functions via `compute_Q`:

```julia
# Any statistical identification method works as an irf method
irfs_ica = irf(model, 20; method=:fastica)
irfs_ml  = irf(model, 20; method=:student_t)
irfs_ms  = irf(model, 20; method=:markov_switching)

# FEVD and HD also work
decomp = fevd(model, 20; method=:fastica)
```

Supported method symbols: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`, `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:markov_switching`, `:garch`.

---

## The Labeling Problem

Statistical identification (both heteroskedasticity and non-Gaussianity) identifies ``B_0`` only up to **column permutation and sign**. The columns of ``B_0`` represent structural shocks, but the data alone cannot tell us which column corresponds to which economic shock.

This means:
- Economic information is still needed to **label** shocks (e.g., "this is the monetary policy shock")
- Our convention: positive diagonal of ``B_0`` (sign normalization)
- Column ordering may differ across bootstrap replications — the Procrustes distance in `test_identification_strength` accounts for this

See Lewis (2025, Section 6.4) for a thorough discussion of the labeling problem.

---

## Complete Example

```julia
using MacroEconometricModels

# Load FRED-MD monetary policy model
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Step 1: Test for non-Gaussianity of VAR residuals
suite = normality_test_suite(model)
println(suite)

# Step 2: Try ICA identification (non-Gaussianity approach)
ica = identify_fastica(model)
println("\nFastICA result:")
println("  Converged: ", ica.converged)
println("  Q orthogonal: ", round(norm(ica.Q' * ica.Q - I), digits=8))

# Step 3: Verify identification
gauss = test_shock_gaussianity(ica)
println("\nShock Gaussianity Test:")
println("  Number of Gaussian shocks: ", gauss.details[:n_gaussian])
println("  JB p-values: ", round.(gauss.details[:jb_pvals], digits=4))

indep = test_shock_independence(ica; max_lag=5)
println("\nShock Independence Test:")
println("  Independent: ", indep.identified)
println("  Fisher p-value: ", round(indep.pvalue, digits=4))

# Step 4: Compare with ML approach (non-Gaussianity, parametric)
ml = identify_student_t(model)
println("\nStudent-t ML:")
println("  ν = ", round.(ml.dist_params[:nu], digits=2))
println("  AIC = $(round(ml.aic, digits=2)), BIC = $(round(ml.bic, digits=2))")

lr = test_gaussian_vs_nongaussian(model)
println("\nGaussian vs Non-Gaussian LR test:")
println("  LR = $(round(lr.statistic, digits=4)), p = $(round(lr.pvalue, digits=4))")

# Step 5: Try heteroskedasticity approach
ms = identify_markov_switching(model; n_regimes=2)
println("\nMarkov-switching identification:")
println("  Converged: ", ms.converged)
println("  Log-likelihood: ", round(ms.loglik, digits=2))

# Step 6: Compute IRFs using preferred method
irfs = irf(model, 20; method=:fastica)
println("\nIRF size: ", size(irfs.values))
```

---

### See Also

- [VAR Estimation](manual.md) -- Reduced-form VAR and traditional identification methods
- [Hypothesis Tests](hypothesis_tests.md) -- Normality tests for residual diagnostics
- [Innovation Accounting](innovation_accounting.md) -- IRF, FEVD, and historical decomposition
- [API Reference](api_functions.md) -- Complete function signatures

## References

### Survey

- Lewis, Daniel J. 2025. "Identification Based on Higher Moments in Macroeconometrics." *Annual Review of Economics* 17: 665–693. [https://doi.org/10.1146/annurev-economics-070124-051419](https://doi.org/10.1146/annurev-economics-070124-051419)

### Heteroskedasticity-Based Identification

- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)
- Sentana, Enrique, and Gabriele Fiorentini. 2001. "Identification, Estimation and Testing of Conditionally Heteroskedastic Factor Models." *Journal of Econometrics* 102 (2): 143–164. [https://doi.org/10.1016/S0304-4076(01)00051-3](https://doi.org/10.1016/S0304-4076(01)00051-3)
- Lanne, Markku, and Helmut Lütkepohl. 2008. "Identifying Monetary Policy Shocks via Changes in Volatility." *Journal of Money, Credit and Banking* 40 (6): 1131–1149. [https://doi.org/10.1111/j.1538-4616.2008.00151.x](https://doi.org/10.1111/j.1538-4616.2008.00151.x)
- Normandin, Michel, and Louis Phaneuf. 2004. "Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility." *Journal of Monetary Economics* 51 (6): 1217–1243. [https://doi.org/10.1016/j.jmoneco.2003.11.002](https://doi.org/10.1016/j.jmoneco.2003.11.002)
- Lütkepohl, Helmut, and Aleksei Netšunajev. 2017. "Structural Vector Autoregressions with Smooth Transition in Variances." *Journal of Economic Dynamics and Control* 84: 43–57. [https://doi.org/10.1016/j.jedc.2017.09.001](https://doi.org/10.1016/j.jedc.2017.09.001)
- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086–3124. [https://doi.org/10.1093/restud/rdab009](https://doi.org/10.1093/restud/rdab009)

### Non-Gaussianity — ICA (Nonparametric)

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Cardoso, Jean-François, and Antoine Souloumiac. 1993. "Blind Beamforming for Non-Gaussian Signals." *IEE Proceedings-F* 140 (6): 362–370. [https://doi.org/10.1049/ip-f-2.1993.0054](https://doi.org/10.1049/ip-f-2.1993.0054)
- Belouchrani, Adel, Karim Abed-Meraim, Jean-François Cardoso, and Eric Moulines. 1997. "A Blind Source Separation Technique Using Second-Order Statistics." *IEEE Transactions on Signal Processing* 45 (2): 434–444. [https://doi.org/10.1109/78.554307](https://doi.org/10.1109/78.554307)
- Comon, Pierre. 1994. "Independent Component Analysis, A New Concept?" *Signal Processing* 36 (3): 287–314. [https://doi.org/10.1016/0165-1684(94)90029-9](https://doi.org/10.1016/0165-1684(94)90029-9)

### Non-Gaussianity — ML (Parametric)

- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Gourieroux, Christian, Alain Monfort, and Jean-Paul Renne. 2017. "Statistical Inference for Independent Component Analysis: Application to Structural VAR Models." *Journal of Econometrics* 196 (1): 111–126. [https://doi.org/10.1016/j.jeconom.2016.09.007](https://doi.org/10.1016/j.jeconom.2016.09.007)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Herwartz, Helmut. 2018. "Hodges-Lehmann Detection of Structural Shocks: An Analysis of Macroeconomic Dynamics in the Euro Area." *Oxford Bulletin of Economics and Statistics* 80 (4): 736–754. [https://doi.org/10.1111/obes.12234](https://doi.org/10.1111/obes.12234)
- Azzalini, Adelchi. 1985. "A Class of Distributions Which Includes the Normal Ones." *Scandinavian Journal of Statistics* 12 (2): 171–178. [https://www.jstor.org/stable/4615982](https://www.jstor.org/stable/4615982)

### Non-Gaussianity — Moments (GMM)

- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments." *Journal of Business & Economic Statistics* 39 (3): 772–882. [https://doi.org/10.1080/07350015.2020.1730858](https://doi.org/10.1080/07350015.2020.1730858)
- Lanne, Markku, and Jani Luoto. 2021. "GMM Estimation of Non-Gaussian Structural Vector Autoregression." *Journal of Business & Economic Statistics* 39 (1): 69–81. [https://doi.org/10.1080/07350015.2019.1629940](https://doi.org/10.1080/07350015.2019.1629940)

### Diagnostics and Weak Identification

- Lewis, Daniel J. 2022. "Robust Inference in Models Identified via Heteroskedasticity." *Review of Economics and Statistics* 104 (3): 510–524. [https://doi.org/10.1162/rest_a_00977](https://doi.org/10.1162/rest_a_00977)

### Multivariate Normality Tests

- Jarque, Carlos M., and Anil K. Bera. 1980. "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals." *Economics Letters* 6 (3): 255–259. [https://doi.org/10.1016/0165-1765(80)90024-5](https://doi.org/10.1016/0165-1765(80)90024-5)
- Mardia, Kanti V. 1970. "Measures of Multivariate Skewness and Kurtosis with Applications." *Biometrika* 57 (3): 519–530. [https://doi.org/10.1093/biomet/57.3.519](https://doi.org/10.1093/biomet/57.3.519)
- Doornik, Jurgen A., and Henrik Hansen. 2008. "An Omnibus Test for Univariate and Multivariate Normality." *Oxford Bulletin of Economics and Statistics* 70: 927–939. [https://doi.org/10.1111/j.1468-0084.2008.00537.x](https://doi.org/10.1111/j.1468-0084.2008.00537.x)
- Henze, Norbert, and Bernhard Zirkler. 1990. "A Class of Invariant Consistent Tests for Multivariate Normality." *Communications in Statistics - Theory and Methods* 19 (10): 3595–3617. [https://doi.org/10.1080/03610929008830400](https://doi.org/10.1080/03610929008830400)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

### Independence Measures

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics* 35 (6): 2769–2794. [https://doi.org/10.1214/009053607000000505](https://doi.org/10.1214/009053607000000505)
- Gretton, Arthur, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. "Measuring Statistical Dependence with Hilbert-Schmidt Norms." In *Algorithmic Learning Theory*, edited by Sanjay Jain, Hans Ulrich Simon, and Etsuji Tomita, 63–77. Berlin: Springer. [https://doi.org/10.1007/11564089_7](https://doi.org/10.1007/11564089_7)
- Matteson, David S., and Ruey S. Tsay. 2017. "Independent Component Analysis via Distance Covariance." *Journal of the American Statistical Association* 112 (518): 623–637. [https://doi.org/10.1080/01621459.2016.1150851](https://doi.org/10.1080/01621459.2016.1150851)
