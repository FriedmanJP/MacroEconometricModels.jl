# Non-Gaussian SVAR Identification

This page covers identification of structural VAR models using non-Gaussian distributional assumptions, heteroskedasticity, and ICA methods. These methods provide identification without requiring the recursive ordering of Cholesky or the a priori sign/zero restrictions of traditional SVAR.

## Quick Start

```julia
using MacroEconometricModels

# Multivariate normality tests (diagnostics)
suite = normality_test_suite(model)                # Run all 7 tests
jb = jarque_bera_test(model)                       # Multivariate Jarque-Bera

# ICA-based SVAR identification
ica = identify_fastica(model)                      # FastICA (Hyvärinen 1999)
jade = identify_jade(model)                        # JADE (Cardoso 1993)

# Non-Gaussian ML identification
ml = identify_student_t(model)                     # Student-t shocks
ml = identify_nongaussian_ml(model; distribution=:mixture_normal)

# Heteroskedasticity identification
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

## Multivariate Normality Tests

Before applying non-Gaussian SVAR methods, it is essential to verify that the VAR residuals are indeed non-Gaussian. If residuals are Gaussian, non-Gaussian identification will not work (the problem is unidentified).

### Multivariate Jarque-Bera Test

The multivariate Jarque-Bera test extends the univariate JB test to vector residuals. Under the null hypothesis of multivariate normality, the test statistic is:

```math
JB = T \cdot \frac{b_{1,k}}{6} + T \cdot \frac{(b_{2,k} - k(k+2))^2}{24k}
```

where ``b_{1,k}`` is the multivariate skewness measure and ``b_{2,k}`` is the multivariate kurtosis measure (Lütkepohl 2005, §4.5).

```julia
using MacroEconometricModels, Random
Random.seed!(42)
Y = randn(300, 3)
model = estimate_var(Y, 2)

# Joint test
jb = jarque_bera_test(model)
println("Statistic: $(round(jb.statistic, digits=4)), p-value: $(round(jb.pvalue, digits=4))")

# Component-wise test on standardized residuals
jb_comp = jarque_bera_test(model; method=:component)
println("Component p-values: ", round.(jb_comp.component_pvalues, digits=4))
```

With Gaussian data, we expect p-values above 0.05 — failure to reject normality.

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

## ICA-based SVAR Identification

Independent Component Analysis (ICA) identifies the structural impact matrix ``B_0`` by finding the rotation ``Q`` that makes the recovered shocks ``\varepsilon_t = (B_0)^{-1} u_t`` maximally independent and non-Gaussian.

### Model Specification

The structural VAR has the decomposition:

```math
u_t = B_0 \varepsilon_t, \quad \Sigma = B_0 B_0'
```

where
- ``u_t`` is the ``n \times 1`` vector of reduced-form residuals
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks, assumed mutually independent and non-Gaussian
- ``B_0 = L Q`` where ``L = \text{chol}(\Sigma)`` and ``Q`` is orthogonal

**Identification condition**: At most one structural shock may be Gaussian (Lanne, Meitz & Saikkonen 2017). If all shocks are non-Gaussian, ``B_0`` is unique up to column permutation and sign.

### FastICA

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

### JADE

JADE (Joint Approximate Diagonalization of Eigenmatrices) uses fourth-order cumulant matrices and joint diagonalization via Jacobi rotations.

```julia
jade = identify_jade(model)
```

JADE computes the fourth-order cumulant matrices ``C_{ij}[k,l] = \text{cum}(z_k, z_l, z_i, z_j)`` and finds the orthogonal matrix that simultaneously diagonalizes all of them.

**Reference**: Cardoso & Souloumiac (1993)

### SOBI

SOBI (Second-Order Blind Identification) exploits temporal structure via autocovariance matrices at multiple lags.

```julia
sobi = identify_sobi(model; lags=1:12)
```

Unlike FastICA and JADE which use higher-order statistics, SOBI only uses second-order statistics (autocovariances), making it suitable when temporal dependence is the main source of identifiability.

**Reference**: Belouchrani et al. (1997)

### Distance Covariance

Minimizes the sum of pairwise distance covariances between recovered shocks. Distance covariance (Székely et al. 2007) is zero if and only if variables are independent.

```julia
dcov = identify_dcov(model)
```

**Reference**: Matteson & Tsay (2017)

### HSIC

Minimizes the Hilbert-Schmidt Independence Criterion using a Gaussian kernel. Like distance covariance, HSIC with a characteristic kernel is zero iff variables are independent.

```julia
hsic = identify_hsic(model; sigma=1.0)
```

The bandwidth parameter ``\sigma`` defaults to the median pairwise distance heuristic.

**Reference**: Gretton et al. (2005)

### ICA Result Fields

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

---

## Non-Gaussian Maximum Likelihood

Instead of the two-step ICA approach, ML methods estimate ``B_0`` and the shock distribution parameters jointly by maximizing the log-likelihood.

### Model Specification

The log-likelihood under non-Gaussian shocks is:

```math
\ell(\theta) = \sum_{t=1}^T \left[ \log|\det(B_0^{-1})| + \sum_{j=1}^n \log f_j(\varepsilon_{j,t}; \theta_j) \right]
```

where
- ``\varepsilon_t = B_0^{-1} u_t`` are the structural shocks
- ``f_j(\cdot; \theta_j)`` is the marginal density of shock ``j``
- ``\theta_j`` are distribution-specific parameters (e.g., degrees of freedom for Student-t)

### Student-t Shocks

Assumes each shock follows a (standardized) Student-t distribution with shock-specific degrees of freedom ``\nu_j``:

```julia
ml = identify_student_t(model)
println("Degrees of freedom: ", ml.dist_params[:nu])
```

Low ``\nu`` indicates heavy tails. When ``\nu \to \infty``, the shock approaches Gaussianity. Identification requires that at most one shock has ``\nu = \infty``.

**Reference**: Lanne, Meitz & Saikkonen (2017)

### Mixture of Normals

Each shock follows a mixture of two normals: ``\varepsilon_j \sim p_j N(0, \sigma_{1j}^2) + (1-p_j) N(0, \sigma_{2j}^2)`` with the unit variance constraint ``p_j \sigma_{1j}^2 + (1-p_j) \sigma_{2j}^2 = 1``.

```julia
ml = identify_mixture_normal(model)
println("Mixing probabilities: ", ml.dist_params[:p_mix])
```

**Reference**: Lanne & Lütkepohl (2010)

### Pseudo Maximum Likelihood (PML)

Uses Pearson Type IV distributions, allowing both skewness and excess kurtosis.

```julia
ml = identify_pml(model)
```

**Reference**: Herwartz (2018)

### Skew-Normal Shocks

Each shock follows a skew-normal distribution with pdf ``f(x) = 2\phi(x)\Phi(\alpha_j x)``.

```julia
ml = identify_skew_normal(model)
println("Skewness parameters: ", ml.dist_params[:alpha])
```

**Reference**: Azzalini (1985)

### Unified Dispatcher

Use `identify_nongaussian_ml` to select the distribution at runtime:

```julia
for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
    ml = identify_nongaussian_ml(model; distribution=dist)
    println("$dist: logL=$(round(ml.loglik, digits=2)), AIC=$(round(ml.aic, digits=2))")
end
```

Compare AIC/BIC across distributions to select the best-fitting specification.

### ML Result Fields

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

## Heteroskedasticity-Based Identification

These methods identify ``B_0`` from changes in the error covariance across volatility regimes, without requiring non-Gaussianity.

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

---

## Integration with IRF Pipeline

All ICA and ML methods integrate seamlessly with the existing `irf`, `fevd`, and `historical_decomposition` functions via `compute_Q`:

```julia
# Any non-Gaussian method works as an irf method
irfs_ica = irf(model, 20; method=:fastica)
irfs_ml  = irf(model, 20; method=:student_t)
irfs_ms  = irf(model, 20; method=:markov_switching)

# FEVD and HD also work
decomp = fevd(model, 20; method=:fastica)
```

Supported method symbols: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`, `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:markov_switching`, `:garch`.

---

## Complete Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Generate VAR data
T_obs, n = 300, 3
Y = randn(T_obs, n)
for t in 3:T_obs
    Y[t, :] = 0.5 * Y[t-1, :] + 0.2 * Y[t-2, :] + 0.8 * randn(n)
end
model = estimate_var(Y, 2)

# Step 1: Test for non-Gaussianity
suite = normality_test_suite(model)
println(suite)

# Step 2: Try ICA identification
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

# Step 4: Compare with ML approach
ml = identify_student_t(model)
println("\nStudent-t ML:")
println("  ν = ", round.(ml.dist_params[:nu], digits=2))
println("  AIC = $(round(ml.aic, digits=2)), BIC = $(round(ml.bic, digits=2))")

lr = test_gaussian_vs_nongaussian(model)
println("\nGaussian vs Non-Gaussian LR test:")
println("  LR = $(round(lr.statistic, digits=4)), p = $(round(lr.pvalue, digits=4))")

# Step 5: Compute IRFs
irfs = irf(model, 20; method=:fastica)
println("\nIRF size: ", size(irfs.values))
```

---

## References

### Multivariate Normality Tests

- Jarque, Carlos M., and Anil K. Bera. 1980. "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals." *Economics Letters* 6 (3): 255–259. [https://doi.org/10.1016/0165-1765(80)90024-5](https://doi.org/10.1016/0165-1765(80)90024-5)
- Mardia, Kanti V. 1970. "Measures of Multivariate Skewness and Kurtosis with Applications." *Biometrika* 57 (3): 519–530. [https://doi.org/10.1093/biomet/57.3.519](https://doi.org/10.1093/biomet/57.3.519)
- Doornik, Jurgen A., and Henrik Hansen. 2008. "An Omnibus Test for Univariate and Multivariate Normality." *Oxford Bulletin of Economics and Statistics* 70: 927–939. [https://doi.org/10.1111/j.1468-0084.2008.00537.x](https://doi.org/10.1111/j.1468-0084.2008.00537.x)
- Henze, Norbert, and Bernhard Zirkler. 1990. "A Class of Invariant Consistent Tests for Multivariate Normality." *Communications in Statistics - Theory and Methods* 19 (10): 3595–3617. [https://doi.org/10.1080/03610929008830400](https://doi.org/10.1080/03610929008830400)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.

### ICA-based Identification

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Cardoso, Jean-François, and Antoine Souloumiac. 1993. "Blind Beamforming for Non-Gaussian Signals." *IEE Proceedings-F* 140 (6): 362–370. [https://doi.org/10.1049/ip-f-2.1993.0054](https://doi.org/10.1049/ip-f-2.1993.0054)
- Belouchrani, Adel, Karim Abed-Meraim, Jean-François Cardoso, and Eric Moulines. 1997. "A Blind Source Separation Technique Using Second-Order Statistics." *IEEE Transactions on Signal Processing* 45 (2): 434–444. [https://doi.org/10.1109/78.554307](https://doi.org/10.1109/78.554307)

### Non-Gaussian ML

- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Herwartz, Helmut. 2018. "Hodges-Lehmann Detection of Structural Shocks: An Analysis of Macroeconomic Dynamics in the Euro Area." *Oxford Bulletin of Economics and Statistics* 80 (4): 736–754. [https://doi.org/10.1111/obes.12237](https://doi.org/10.1111/obes.12237)
- Azzalini, Adelchi. 1985. "A Class of Distributions Which Includes the Normal Ones." *Scandinavian Journal of Statistics* 12 (2): 171–178. [https://doi.org/10.1111/j.1467-9469.1985.tb01174.x](https://doi.org/10.1111/j.1467-9469.1985.tb01174.x)

### Heteroskedasticity-based Identification

- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)
- Lanne, Markku, and Helmut Lütkepohl. 2008. "Identifying Monetary Policy Shocks via Changes in Volatility." *Journal of Money, Credit and Banking* 40 (6): 1131–1149. [https://doi.org/10.1111/j.1538-4616.2008.00151.x](https://doi.org/10.1111/j.1538-4616.2008.00151.x)
- Normandin, Michel, and Louis Phaneuf. 2004. "Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility." *Journal of Monetary Economics* 51 (6): 1217–1243. [https://doi.org/10.1016/j.jmoneco.2003.11.002](https://doi.org/10.1016/j.jmoneco.2003.11.002)
- Lütkepohl, Helmut, and Aleksei Netšunajev. 2017. "Structural Vector Autoregressions with Smooth Transition in Variances." *Journal of Economic Dynamics and Control* 84: 43–57. [https://doi.org/10.1016/j.jedc.2017.09.001](https://doi.org/10.1016/j.jedc.2017.09.001)

### Independence Measures

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances." *Annals of Statistics* 35 (6): 2769–2794. [https://doi.org/10.1214/009053607000000505](https://doi.org/10.1214/009053607000000505)
- Gretton, Arthur, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. "Measuring Statistical Dependence with Hilbert-Schmidt Norms." In *Algorithmic Learning Theory*, edited by Sanjay Jain, Hans Ulrich Simon, and Etsuji Tomita, 63–77. Berlin: Springer. [https://doi.org/10.1007/11564089_7](https://doi.org/10.1007/11564089_7)
- Matteson, David S., and Ruey S. Tsay. 2017. "Independent Component Analysis via Distance Covariance." *Journal of the American Statistical Association* 112 (518): 623–637. [https://doi.org/10.1080/01621459.2016.1150851](https://doi.org/10.1080/01621459.2016.1150851)
