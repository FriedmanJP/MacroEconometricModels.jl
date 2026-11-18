# [Non-Gaussian Methods](@id id_nongaussian_page)

Non-Gaussian structural VAR identification recovers the structural impact matrix ``B_0`` by exploiting the statistical independence and non-Gaussianity of structural shocks. The Darmois-Skitovich theorem (Comon 1994) establishes that if at most one shock is Gaussian, ``B_0`` is unique up to column permutation and sign --- without imposing any economic restrictions.

This page covers three complementary approaches:

- **ICA-based methods** (nonparametric): FastICA, JADE, SOBI, distance covariance, HSIC
- **Maximum likelihood methods** (parametric): Student-t, symmetric and asymmetric mixture of normals, PML, skew-normal, plus a unified dispatcher
- **Moment-based GMM** (semiparametric): coskewness and cokurtosis conditions (Keweloh 2021; Lanne & Luoto 2021)

For an overview and method comparison, see [Statistical Identification](@ref nongaussian_page). For heteroskedasticity-based identification, see [Heteroskedasticity](@ref id_heteroskedastic_page). For GMM on time-varying-volatility moments instead of skewness and kurtosis, see [Time-Varying Volatility](@ref id_tvv_page). For the diagnostics that decide whether these methods are applicable at all, see [Testing](@ref id_testing_page). For schemes built on economic rather than statistical restrictions, see [Structural Identification](@ref structural_identification_page).

```@setup id_ng
using MacroEconometricModels, Random, LinearAlgebra
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-119:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: FastICA identification**

```@example id_ng
# Maximize non-Gaussianity of the recovered shocks (Hyvärinen 1999)
ica = identify_fastica(model; rng=MersenneTwister(11))
report(ica)
```

**Recipe 2: Student-t maximum likelihood**

```@example id_ng
# Parametric ML with shock-specific degrees of freedom (Lanne, Meitz & Saikkonen 2017)
ml = identify_student_t(model; seed=42)
report(ml)
```

**Recipe 3: Mixture of normals**

```@example id_ng
# Two-component Gaussian mixture shocks (Lanne & Lütkepohl 2010; Gouriéroux et al. 2020)
mix = identify_mixture_normal(model; asymmetric=true, seed=42)
report(mix)
```

**Recipe 4: Compare distributions by information criterion**

```@example id_ng
comparison = [(dist = d,
               logL = round(m.loglik, digits=2),
               AIC = round(m.aic, digits=2),
               BIC = round(m.bic, digits=2))
              for d in [:student_t, :mixture_normal, :mixture_normal_asym, :skew_normal]
              for m in (identify_nongaussian_ml(model; distribution=d, seed=42),)]
```

**Recipe 5: Second-order identification via SOBI**

```@example id_ng
sobi = identify_sobi(model; lags=1:12)
round.(sobi.B0, digits=5)
```

**Recipe 6: Feed the identification into the IRF pipeline**

```@example id_ng
irfs = irf(model, 20; method=:fastica, rng=MersenneTwister(11))
report(irfs)
```

---

## ICA-Based Methods

Independent Component Analysis (ICA) identifies the structural impact matrix ``B_0`` by searching for the orthogonal rotation ``Q`` that makes the recovered shocks maximally independent and non-Gaussian. These methods are nonparametric --- they assume no specific distributional form for the shocks.

The model decomposes the reduced-form residuals as:

```math
u_t = B_0 \varepsilon_t, \quad B_0 = L \, Q
```

where:
- ``u_t`` is the ``n \times 1`` vector of reduced-form VAR residuals
- ``\varepsilon_t`` is the ``n \times 1`` vector of independent structural shocks
- ``L = \text{chol}(\Sigma)`` is the lower Cholesky factor of the residual covariance
- ``Q`` is an ``n \times n`` orthogonal rotation matrix

ICA searches over orthogonal ``Q`` to maximize a measure of non-Gaussianity or minimize a measure of statistical dependence among the recovered shocks ``\varepsilon_t = (LQ)^{-1} u_t``. Every estimator on this page pre-whitens the residuals first, so the search is genuinely over the rotation group and never over the scale of the shocks.

**Identification condition** (Darmois-Skitovich theorem): at most one structural shock may be Gaussian. If all shocks are non-Gaussian, ``B_0`` is unique up to column permutation and sign (Comon 1994; Lanne, Meitz & Saikkonen 2017).

### FastICA

FastICA (Hyvärinen 1999) finds the unmixing matrix by maximizing **negentropy** --- a non-negative measure of non-Gaussianity --- via a fixed-point iteration. The algorithm pre-whitens the residuals so that ``\text{Cov}(Z) = I``, then searches for orthogonal directions of maximum non-Gaussianity.

Three contrast functions ``G(u)`` approximate negentropy:
- `:logcosh` (default) --- ``G(u) = \log\cosh(u)``, robust general-purpose choice
- `:exp` --- ``G(u) = -\exp(-u^2/2)``, suited for super-Gaussian sources
- `:kurtosis` --- ``G(u) = u^4/4``, the classical kurtosis-based measure

Two extraction approaches control how components are recovered:
- `:deflation` --- extracts components one at a time, orthogonalizing against previously found components
- `:symmetric` --- extracts all components simultaneously via symmetric decorrelation ``W \leftarrow (WW')^{-1/2} W``

```@example id_ng
# Default: logcosh contrast, deflation approach
ica1 = identify_fastica(model; rng=MersenneTwister(11))
report(ica1)
```

```@example id_ng
# Symmetric approach with the exponential contrast
ica2 = identify_fastica(model; approach=:symmetric, contrast=:exp,
                        rng=MersenneTwister(11))
round.(ica2.B0, digits=5)
```

The deflation fit converges in 14 fixed-point iterations at a negentropy objective of ``0.282``. Its impact matrix puts the large funds-rate loading (``0.131``) in the second column and a much smaller one (``-0.016``) in the first, so the column that behaves like a monetary shock is not the column ordered last --- statistical identification recovers the columns in no particular order. The symmetric fit with the exponential contrast returns the same three directions with the funds-rate column moved to position three; matched column by column, the largest angular discrepancy between the two solutions is ``0.019``. That is the reassurance to look for: contrast and extraction choices should permute the columns, not change the subspace they span.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `contrast` | `Symbol` | `:logcosh` | Non-Gaussianity measure: `:logcosh`, `:exp`, `:kurtosis` |
| `approach` | `Symbol` | `:deflation` | Extraction approach: `:deflation` or `:symmetric` |
| `max_iter` | `Int` | `200` | Maximum iterations per component |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Generator for the random starting directions |

!!! note "RNG state and reproducibility"
    FastICA starts each component from a random unit vector, and multi-start ML estimators (`n_starts > 1`) use a FastICA warm-start plus random orthogonal rotations. Pass `rng` or `seed` for reproducibility. JADE, SOBI, and dCov are deterministic given the data; HSIC uses `rng` only for its median-heuristic subsample on long samples.

### JADE

JADE (Joint Approximate Diagonalization of Eigenmatrices; Cardoso & Souloumiac 1993) computes fourth-order cumulant matrices and searches for the orthogonal matrix ``V`` that simultaneously diagonalizes all of them via Jacobi rotations. The fourth-order cumulant matrix ``C_{ij}`` has entries:

```math
C_{ij}[k,l] = \text{cum}(z_k, z_l, z_i, z_j) = E[z_k z_l z_i z_j] - E[z_k z_l] E[z_i z_j] - E[z_k z_i] E[z_l z_j] - E[z_k z_j] E[z_l z_i]
```

where:
- ``C_{ij}[k,l]`` is entry ``(k,l)`` of the ``n \times n`` fourth-order cumulant matrix associated with pair ``(i,j)``
- ``\text{cum}(\cdot)`` is the fourth-order joint cumulant function
- ``z_k, z_l, z_i, z_j`` are elements of the pre-whitened residual vector ``z_t = L^{-1} u_t``
- ``E[\cdot]`` denotes the mathematical expectation operator

Joint diagonalization minimizes the sum of squared off-diagonal elements across all ``n(n+1)/2`` cumulant matrices, and the `objective` field reports that sum at the returned rotation.

```@example id_ng
jade = identify_jade(model; max_iter=100)
report(jade)
```

```@example id_ng
(objective = round(jade.objective, digits=4),
 iterations = jade.iterations,
 converged = jade.converged)
```

The sweep exhausts its 100-iteration budget without the maximum Jacobi angle falling below `tol`, so `converged` reports `false` and the residual off-diagonal mass stays at ``34.45``. Raising `max_iter` does not change that value. Treat a JADE fit that ends on its iteration cap as a rotation to cross-check rather than to trust: compare its ``B_0`` against FastICA and the ML estimates before labelling the shocks.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `100` | Maximum Jacobi rotation sweeps |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance on the rotation angle |

### SOBI

SOBI (Second-Order Blind Identification; Belouchrani et al. 1997) exploits temporal structure by jointly diagonalizing autocovariance matrices at multiple lags. Unlike FastICA and JADE, SOBI uses only second-order statistics, which makes it the natural choice when serial dependence rather than higher-moment structure is the source of identifiability.

The autocovariance matrix of the whitened residuals at lag ``\tau`` is:

```math
R(\tau) = \frac{1}{T - \tau} \sum_{t=1}^{T-\tau} z_{t+\tau} z_t'
```

where:
- ``z_t`` is the ``n \times 1`` vector of pre-whitened residuals at time ``t``
- ``\tau`` ranges over the `lags` keyword

SOBI finds the orthogonal ``V`` that simultaneously diagonalizes ``\{R(\tau)\}_{\tau \in \text{lags}}``.

```@example id_ng
sobi = identify_sobi(model; lags=1:12)
report(sobi)
```

SOBI's residual off-diagonal mass is ``0.706`` and, like JADE, its sweep runs to the iteration cap. Two of its three columns sit within ``0.006`` of a JADE column in angular distance and the third is ``0.085`` away, so the two second- and fourth-order criteria broadly agree on this sample without agreeing everywhere. SOBI is the more trustworthy of the two when the sample is short, because autocovariances are estimated far more precisely than fourth-order cumulants.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `AbstractRange` | `1:12` | Lag range for the autocovariance matrices |
| `max_iter` | `Int` | `100` | Maximum Jacobi rotation sweeps |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### Distance Covariance

Distance covariance (Székely et al. 2007) measures dependence between random vectors and equals zero if and only if the variables are independent. `identify_dcov` minimizes the sum of pairwise distance covariances between recovered shocks over the ``n(n-1)/2`` Givens rotation angles (Matteson & Tsay 2017), using derivative-free Nelder-Mead.

```@example id_ng
dcov = identify_dcov(model; max_iter=200)
report(dcov)
```

The optimizer converges in 26 iterations to a summed pairwise distance covariance of ``0.0220``. Because the criterion is a genuine independence measure rather than a non-Gaussianity proxy, dCov remains valid when the shocks are non-Gaussian in ways that negentropy contrasts miss --- at the cost of an ``O(T^2)`` distance matrix per pair per function evaluation, which makes it the slowest estimator on this page for long samples.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `200` | Maximum Nelder-Mead iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### HSIC

The Hilbert-Schmidt Independence Criterion (Gretton et al. 2005) measures dependence through kernel embeddings. With a characteristic kernel --- here the Gaussian --- HSIC is zero if and only if the variables are independent. `identify_hsic` minimizes the summed pairwise HSIC of the recovered shocks over the same Givens parameterization.

Leaving `sigma` at its default of `1.0` triggers the median pairwise-distance heuristic for the kernel bandwidth; any other value is used as given.

```@example id_ng
hsic = identify_hsic(model; sigma=1.0)
report(hsic)
```

The criterion falls to ``7.4 \times 10^{-4}`` after 19 iterations. HSIC and dCov target the same null --- full independence, not merely zero correlation --- and the HSIC solution reproduces the FastICA directions up to a column permutation, with a maximum angular discrepancy of ``0.013``. Agreement across criteria with different objective functions is the practical substitute for the standard errors that nonparametric ICA does not provide; GMM and the ML estimators below do supply sandwich or information-matrix SEs.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `kernel` | `Symbol` | `:gaussian` | Kernel family (Gaussian is the only characteristic kernel implemented) |
| `sigma` | `Real` | `1.0` | Gaussian kernel bandwidth (`1.0` triggers the median heuristic) |
| `max_iter` | `Int` | `200` | Maximum Nelder-Mead iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance |

### ICASVARResult Fields

All five ICA methods return an `ICASVARResult{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``): ``u_t = B_0 \varepsilon_t`` |
| `W` | `Matrix{T}` | Unmixing matrix (``n \times n``): ``\varepsilon_t = W u_t`` |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `shocks` | `Matrix{T}` | Recovered structural shocks (``T_{\text{eff}} \times n``) |
| `method` | `Symbol` | Method used: `:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic` |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Number of iterations |
| `objective` | `T` | Final objective value (negentropy for FastICA, off-diagonal mass for JADE/SOBI, dependence for dCov/HSIC) |

The `objective` field is comparable across runs of the *same* method only --- the four criteria are on different scales.

---

## Maximum Likelihood Methods

Maximum likelihood methods estimate ``B_0`` and the shock distribution parameters jointly. The non-Gaussian log-likelihood is:

```math
\ell(\theta) = \sum_{t=1}^{T} \left[ \log|\det(B_0^{-1})| + \sum_{j=1}^{n} \log f_j(\varepsilon_{j,t};\, \theta_j) \right]
```

where:
- ``\varepsilon_t = B_0^{-1} u_t`` are the structural shocks
- ``f_j(\cdot;\, \theta_j)`` is the marginal density of shock ``j`` with parameters ``\theta_j``
- ``B_0 = L Q`` is parameterized by ``n(n-1)/2`` Givens rotation angles

The optimizer searches over rotation angles and distribution parameters simultaneously with multi-start `Optim.LBFGS()` and `ForwardDiff` gradients, warm-started from Cholesky, JADE, FastICA, and random orthogonal rotations (`n_starts` controls the start count; `rng`/`seed` control reproducibility). The `vcov_type` keyword selects the parameter covariance estimator: `:auto` (default) chooses the Gouriéroux, Monfort & Renne (2017) Huber–White sandwich ``H^{-1} J_{\text{OPG}} H^{-1}`` computed from per-observation `ForwardDiff` scores for pseudo-ML specifications (`:pml`, `:mixture_normal`, `:mixture_normal_asym`) and the inverse `ForwardDiff` Hessian ``H^{-1}`` for `:student_t` and `:skew_normal`. Standard errors for ``B_0`` propagate the full angle covariance block ``V_\theta`` through the `ForwardDiff` delta-method Jacobian ``J_B V_\theta J_B'``.

All distribution parameters use unconstrained reparameterizations internally, so the reported values in `dist_params` are already mapped back to their natural scales. Student-t degrees of freedom are ``\nu = \exp(\theta) + 2.01``, which enforces ``\nu > 2`` and hence finite variance. Mixture probabilities use a logistic map on ``(10^{-4}, 1 - 10^{-4})``; in the symmetric mixture (`asymmetric=false`), the first variance uses a bounded split ``w \in (10^{-4}, 1 - 10^{-4})`` with ``\sigma_1^2 = w/p`` and ``\sigma_2^2 = (1 - w)/(1 - p)``; in the asymmetric mixture (`asymmetric=true`), a hyperbolic-tangent mean shift ``r_\mu = (1 - 10^{-4}) \tanh(\theta_\mu)`` pins ``\mu_1 = \sqrt{(1-p)/p}\, r_\mu`` and ``\mu_2 = -\sqrt{p/(1-p)}\, r_\mu`` while splitting the remaining variance budget ``1 - r_\mu^2 > 0`` across ``\sigma_1^2`` and ``\sigma_2^2`` so zero mean and unit variance hold identically.

### Student-t

Each shock follows a standardized Student-t distribution with shock-specific degrees of freedom ``\nu_j`` (Lanne, Meitz & Saikkonen 2017):

```math
f_j(x;\, \nu_j) = \frac{\Gamma((\nu_j+1)/2)}{\sqrt{\pi \nu_j}\, \Gamma(\nu_j/2)} \left(1 + \frac{x^2}{\nu_j}\right)^{-(\nu_j+1)/2} \cdot \sqrt{\frac{\nu_j}{\nu_j - 2}}
```

where:
- ``\nu_j > 2`` is the degrees-of-freedom parameter of shock ``j``
- the factor ``\sqrt{\nu_j / (\nu_j - 2)}`` standardizes the variance to unity

Low ``\nu_j`` indicates heavy tails; as ``\nu_j \to \infty``, shock ``j`` approaches Gaussianity.

```@example id_ng
ml = identify_student_t(model; seed=42)
report(ml)
```

```@example id_ng
round.(ml.dist_params[:nu], digits=3)
```

All three degrees-of-freedom estimates --- ``3.32``, ``6.53`` and ``2.46`` --- sit far below the value at which a Student-t is indistinguishable from a normal, so every shock contributes to identification and the Darmois-Skitovich condition is comfortably satisfied. The likelihood improves from ``979.75`` under Gaussian shocks to ``1035.94``, an LR statistic of ``112.39`` that rejects Gaussianity at any conventional level. Values of ``\nu_j`` below 10 are the practical threshold: above it the column is nearly Gaussian and its position in ``B_0`` is weakly pinned down.

### Mixture of Normals

Each shock follows a two-component Gaussian mixture (Lanne & Lütkepohl 2010; Gouriéroux, Monfort & Renne 2020):

```math
f_j(x;\, p_j, \mu_{1j}, \sigma_{1j}, \mu_{2j}, \sigma_{2j}) = \frac{p_j}{\sigma_{1j}} \, \phi\!\left(\frac{x - \mu_{1j}}{\sigma_{1j}}\right) + \frac{1 - p_j}{\sigma_{2j}} \, \phi\!\left(\frac{x - \mu_{2j}}{\sigma_{2j}}\right)
```

where:
- ``\phi(\cdot)`` is the standard normal density
- ``p_j \in (0,1)`` is the mixing probability of the first component
- ``\mu_{1j}, \mu_{2j}`` are the component means (fixed to zero when `asymmetric=false`)
- ``\sigma_{1j}, \sigma_{2j}`` are the component standard deviations

The zero-mean constraint ``p_j \mu_{1j} + (1 - p_j)\mu_{2j} = 0`` and unit-variance constraint ``p_j(\mu_{1j}^2 + \sigma_{1j}^2) + (1 - p_j)(\mu_{2j}^2 + \sigma_{2j}^2) = 1`` leave two free parameters ``(p_j, \sigma_{1j})`` per shock in the symmetric mixture (`asymmetric=false`, `:mixture_normal`) and three free parameters ``(p_j, \mu_{1j}, \sigma_{1j})`` in the asymmetric mixture (`asymmetric=true`, `:mixture_normal_asym`), which accommodates skewness and skewed bimodal shocks. Both specifications default to `vcov_type=:auto` resolving to Gouriéroux, Monfort & Renne (2017) sandwich SEs.

```@example id_ng
mix = identify_mixture_normal(model; seed=42)
(p_mix = round.(mix.dist_params[:p_mix], digits=3),
 sigma1 = round.(mix.dist_params[:sigma1], digits=3),
 sigma2 = round.(mix.dist_params[:sigma2], digits=3),
 converged = mix.converged,
 iterations = mix.iterations)
```

```@example id_ng
mix_asym = identify_mixture_normal(model; asymmetric=true, seed=42)
(p_mix = round.(mix_asym.dist_params[:p_mix], digits=3),
 mu1 = round.(mix_asym.dist_params[:mu1], digits=3),
 sigma1 = round.(mix_asym.dist_params[:sigma1], digits=3),
 loglik = round(mix_asym.loglik, digits=2),
 vcov_type = mix_asym.dist_params[:vcov_type])
```

With multi-start L-BFGS, the symmetric mixture converges in ``55`` iterations (log-likelihood ``1034.27``), capturing high-kurtosis regime components with ``\sigma_{1} = (1.53, 2.46, 2.45)`` and ``\sigma_{2} = (0.27, 0.90, 0.66)``. Enabling `asymmetric=true` lifts the log-likelihood to ``1039.11`` by fitting a sharp negative outlier tail on the first shock (``p_1 = 0.010, \mu_{1,1} = -4.361``) and right-skewed tail on the third shock (``p_3 = 0.101, \mu_{1,3} = 0.846``). Identification requires the two components to differ within each shock: ``\sigma_{1j} = \sigma_{2j}`` with ``\mu_{1j} = \mu_{2j} = 0`` collapses that shock to Gaussianity and removes it from the identifying moments.

### PML (Pearson Type IV)

The `:pml` distribution (Herwartz 2018; the symbol is kept for API stability) fits each shock with the genuine Pearson Type IV density, standardized to zero mean and unit variance:

```math
f_j(x;\, \nu_j, m_j) = k_j \left[1 + z_j^2\right]^{-m_j} e^{-\nu_j \arctan z_j}, \qquad z_j = \frac{x - \lambda_j}{a_j}
```

where:
- ``f_j(x;\, \nu_j, m_j)`` is the standardized Pearson Type IV probability density function for shock ``j``
- ``x`` is the standardized structural shock variable
- ``z_j = (x - \lambda_j)/a_j`` is the shifted and scaled shock coordinate
- ``\nu_j`` is the parameter controlling skewness (``\nu_j = 0`` corresponds to symmetry)
- ``m_j > 2`` is the parameter controlling tail thickness
- ``a_j > 0`` is the scale parameter pinned by the unit-variance restriction
- ``\lambda_j`` is the location parameter pinned by the zero-mean restriction
- ``k_j`` is the normalizing constant involving ``|\Gamma(m_j + i\nu_j/2)|^2`` ensuring unit area

Because the density integrates to one for every parameter value, `loglik`, AIC/BIC, and the likelihood-ratio test against Gaussian shocks are all valid statistics — earlier releases used an unnormalized cubic-tilt approximation whose objective was unbounded in the skewness parameter ([#566](https://github.com/FriedmanJP/MacroEconometricModels.jl/issues/566)). Under `vcov_type=:auto`, `:pml` defaults to Gouriéroux, Monfort & Renne (2017) sandwich SEs.

```@example id_ng
pml = identify_pml(model; seed=42)
(kappa = round.(pml.dist_params[:kappa], digits=2),
 m = round.(pml.dist_params[:nu], digits=2),
 loglik = round(pml.loglik, digits=2),
 converged = pml.converged,
 iterations = pml.iterations)
```

In `dist_params`, `:kappa` holds the skewness parameters ``\nu_j`` and `:nu` the tail exponents ``m_j``. The fit converges in ``55`` L-BFGS iterations to a log-likelihood of ``1034.57`` (54.8 points above the Gaussian benchmark of ``979.75``), with heavy tails (``m_j \in [2.05, 3.82]``) and moderate skewness (``\kappa_j \in [-0.82, 0.12]``).

### Skew-Normal

Each shock follows a skew-normal distribution (Azzalini 1985) with density:

```math
f_j(x;\, \alpha_j) = 2 \, \phi(x) \, \Phi(\alpha_j x)
```

where:
- ``\phi(\cdot)`` is the standard normal pdf
- ``\Phi(\cdot)`` is the standard normal cdf
- ``\alpha_j`` controls the direction and degree of skewness

When ``\alpha_j = 0`` the distribution reduces to the standard normal.

```@example id_ng
skew = identify_skew_normal(model; seed=42)
(alpha = round.(skew.dist_params[:alpha], digits=4),
 loglik = round(skew.loglik, digits=3),
 loglik_gaussian = round(skew.loglik_gaussian, digits=3))
```

The estimated skewness parameters are indistinguishable from zero and the log-likelihood equals the Gaussian benchmark to three decimals. That is a substantive finding, not a numerical failure: the non-Gaussianity in these residuals is kurtosis, not asymmetry. It is also a warning, because at ``\alpha = 0`` every shock density is symmetric and the likelihood becomes flat in the rotation --- ``B_0`` collapses to the Cholesky factor and identification is lost. Read a near-zero ``\alpha`` vector as "skewness does not identify this system" and switch to Student-t or the mixture.

### Unified Dispatcher

`identify_nongaussian_ml` selects the distribution at runtime, which makes systematic comparison a one-liner:

```@example id_ng
comparison = [(dist = d,
               logL = round(m.loglik, digits=2),
               AIC = round(m.aic, digits=2),
               BIC = round(m.bic, digits=2),
               converged = m.converged)
              for d in [:student_t, :mixture_normal, :mixture_normal_asym, :pml, :skew_normal]
              for m in (identify_nongaussian_ml(model; distribution=d, seed=42),)]
```

Student-t wins on both criteria (AIC ``-2059.88`` against ``-2054.22`` for the asymmetric mixture, ``-2051.13`` for the Pearson IV, ``-2050.53`` for the symmetric mixture, and ``-1947.50`` for the skew-normal), consistent with the fitted degrees of freedom near 3. The asymmetric mixture improves the log-likelihood over the symmetric mixture and Pearson IV, though its extra three parameters leave AIC slightly behind Student-t. The skew-normal is the worst fit precisely because it spends parameters on pure skewness without accommodating heavy tails. The `loglik_gaussian` field carries the Gaussian benchmark for the likelihood-ratio test formalized on [Testing](@ref id_testing_page).

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `distribution` | `Symbol` | `:student_t` | Distribution: `:student_t`, `:mixture_normal`, `:mixture_normal_asym`, `:pml`, `:skew_normal` |
| `asymmetric` | `Bool` | `false` | Use asymmetric 3-parameter mixture (`identify_mixture_normal` and dispatcher when `distribution=:mixture_normal`) |
| `vcov_type` | `Symbol` | `:auto` | Covariance estimator: `:auto`, `:hessian`, `:sandwich`, `:opg` |
| `n_starts` | `Int` | `5` | Number of multi-start initializations (Cholesky, JADE, FastICA, random rotations) |
| `max_iter` | `Int` | `500` | Maximum L-BFGS iterations per start |
| `tol` | `Real` | ``10^{-6}`` | Gradient convergence tolerance |
| `rng` | `AbstractRNG` | `Random.default_rng()` | RNG for FastICA warm-start and random orthogonal starts |
| `seed` | `Union{Integer,Nothing}` | `nothing` | Optional seed overriding `rng` for reproducibility |

### NonGaussianMLResult Fields

All ML methods return a `NonGaussianMLResult{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``) |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `shocks` | `Matrix{T}` | Structural shocks (``T_{\text{eff}} \times n``) |
| `distribution` | `Symbol` | Distribution used: `:student_t`, `:mixture_normal`, `:mixture_normal_asym`, `:pml`, `:skew_normal` |
| `loglik` | `T` | Log-likelihood at the MLE |
| `loglik_gaussian` | `T` | Gaussian log-likelihood (for the LR test) |
| `dist_params` | `Dict{Symbol,Any}` | Distribution parameters (`:nu`; `:p_mix`, `:mu1`, `:sigma1`, `:mu2`, `:sigma2`; `:kappa`, `:nu`; `:alpha`) and `:vcov_type` |
| `vcov` | `Matrix{T}` | Asymptotic covariance of all parameters (angles and distribution parameters) |
| `se` | `Matrix{T}` | Full Jacobian delta-method standard errors for the ``B_0`` elements |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | L-BFGS iterations used by the winning start |
| `aic` | `T` | Akaike information criterion |
| `bic` | `T` | Bayesian information criterion |
| `shock_names` | `Vector{String}` | Labels for the ``n`` structural shocks |

---

## Moment-Based GMM

Keweloh (2021) and Lanne & Luoto (2021) identify ``B_0`` from independence restrictions on third and fourth moments, without specifying a shock density. The estimator is the distribution-robust member of the non-Gaussian family (Lewis 2025, Section 4.3) and supplies Hansen ``J`` and sandwich standard errors that ICA does not.

The structural impact is parameterised as ``B_0 = L Q(\theta)``, where ``L`` is the Cholesky factor of ``\Sigma`` and ``Q(\theta)`` is the orthogonal matrix generated by ``n(n-1)/2`` Givens angles. Covariance moments ``E[\varepsilon_t\varepsilon_t' - I] = 0`` then hold automatically. The remaining Keweloh (2021, eqs. 7–9) conditions are

```math
E[\varepsilon_{i,t}^2 \varepsilon_{j,t}] = 0, \qquad
E[\varepsilon_{i,t}^3 \varepsilon_{j,t}] = 0, \qquad
E[\varepsilon_{i,t}^2 \varepsilon_{j,t}^2] - 1 = 0, \qquad i \neq j.
```

where:
- ``\varepsilon_t = Q(\theta)' L^{-1} u_t`` are the candidate structural shocks
- ``:coskewness`` uses the third-moment block ``E[\varepsilon_i^2\varepsilon_j] = 0``
- ``:cokurtosis`` uses the fourth-moment block ``E[\varepsilon_i^3\varepsilon_j] = 0`` and ``E[\varepsilon_i^2\varepsilon_j^2] - 1 = 0``
- ``:both`` concatenates the two blocks

Two-step GMM (Hansen 1982) minimises ``g(\theta)' W g(\theta)`` with identity weighting in step 1 and the inverse HAC moment covariance in step 2. Continuously updated GMM (``:cue``) iterates that weighting to convergence. Identification still requires at most one Gaussian shock; `test_gaussian_shock_count` reports the sequential Jarque–Bera count.

```@example id_ng
gmm = identify_gmm_moments(model; moments=:cokurtosis, weighting=:two_step)
report(gmm)
```

```@example id_ng
gcount = test_gaussian_shock_count(gmm)
(n_gaussian = gcount.details[:n_gaussian], identified = gcount.identified,
 J = round(gmm.J; digits=2), J_pvalue = round(gmm.J_pvalue; digits=3))
```

Cokurtosis is the relevant block here: the Student-t ML fit already showed that these residuals are heavy-tailed rather than skewed, so third-moment conditions have little identifying power. Hansen ``J = 19.76`` (``p = 0.003``) rejects the overidentifying fourth-moment restrictions on this 120-observation sample --- independence at the kurtosis level is not a good description of these residuals, so the reported z-statistics (several of them in the hundreds) should not be read as precise. `test_gaussian_shock_count` nevertheless finds ``n_{\text{gaussian}} = 0``, so the Darmois-Skitovich count condition holds. The funds-rate column of ``B_0`` is the large one (``0.131``), matching FastICA and Student-t.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `moments` | `Symbol` | `:both` | Moment family: `:coskewness`, `:cokurtosis`, `:both` |
| `weighting` | `Symbol` | `:two_step` | GMM weighting: `:two_step` or `:cue` |
| `se` | `Symbol` | `:sandwich` | Covariance estimator for ``\theta`` (delta-method ``B_0`` SEs) |

### NonGaussianGMMResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | Structural impact matrix (``n \times n``) |
| `Q` | `Matrix{T}` | Rotation matrix: ``B_0 = L Q`` |
| `theta` | `Vector{T}` | Givens angles (``n(n-1)/2``) |
| `vcov` | `Matrix{T}` | Sandwich covariance of ``\theta`` |
| `se` | `Matrix{T}` | Delta-method standard errors for the ``B_0`` elements |
| `J` | `T` | Hansen ``J`` statistic |
| `J_pvalue` | `T` | ``J``-test p-value |
| `moments` | `Symbol` | Moment family used |
| `weighting` | `Symbol` | Weighting method used |
| `shocks` | `Matrix{T}` | Structural shocks (``T_{\text{eff}} \times n``) |
| `varnames` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock labels |

---

## Labelling Shocks

Statistical identification recovers columns up to signed permutation. [`label_shocks`](@ref) applies an economic labelling: `by=:max_impact` assigns column ``j`` to the shock with the largest ``|B_0[v_j,\cdot]|`` (divided by the column norm), `by=:restrictions` maximises satisfied impact-sign restrictions, and `by=:reference` matches a reference impact. This is a labelling, not a test of identification strength --- use `test_label_stability` for that (see [Identification Testing](@ref id_testing_page)).

```@example id_ng
labeled = label_shocks(ica; by=:max_impact, variables=1:3,
                       shock_names=["output", "price", "mp"])
report(labeled)
```

```@example id_ng
labeled.shock_names
```

`by=:max_impact` assigns each named shock to the column with the largest normalised impact on that variable. ICA results do not store variable names, so `variables` is a vector of column indices; pass a sign matrix or `SVARRestrictions` with `by=:restrictions` when the labelling comes from impact signs rather than from which variable moves most.

---

## Complete Example

This workflow identifies the same monetary VAR nonparametrically and parametrically, selects a distribution by information criterion, and pushes the preferred identification through the IRF pipeline.

```@example id_ng
# --- Step 1: ICA identification (nonparametric) ---
ica = identify_fastica(model; contrast=:logcosh, approach=:deflation,
                       rng=MersenneTwister(11))
report(ica)
```

```@example id_ng
# --- Step 2: ML identification (parametric, Student-t) ---
ml_t = identify_student_t(model)
report(ml_t)
```

```@example id_ng
# --- Step 3: Match the two impact matrices column by column ---
# Angular distance, invariant to column sign (1 = orthogonal, 0 = identical)
cosdist(a, b) = 1 - abs(a' * b) / (norm(a) * norm(b))
[round(minimum(cosdist(ica.B0[:, i], ml_t.B0[:, j]) for j in 1:3), digits=4)
 for i in 1:3]
```

```@example id_ng
# --- Step 4: Impulse responses under the preferred identification ---
irfs = irf(model, 20; method=:fastica, rng=MersenneTwister(11))
report(irfs)
```

```julia
plot_result(irfs)
```

```@raw html
<iframe src="../assets/plots/nongaussian_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Step 3 reports, for each FastICA column, the smallest angular distance to any Student-t ML column. Two of the three match to within ``0.0002`` --- the nonparametric and parametric routes recover those directions identically, differing only in ordering and sign. The remaining column is ``0.14`` away, which is the honest reading of how far identification extends here: two shocks are pinned down by both criteria and the third is not, so any economic claim about that third shock rests on the choice of objective function rather than on the data. The ML route adds what ICA cannot supply --- shock-specific tail parameters, standard errors for ``B_0``, and a formal likelihood-ratio test --- and both plug into `irf`, `fevd`, and `historical_decomposition` through the `method` keyword.

---

## Complete Identification and 1/3-Step Non-Gaussian ML (LMS 2017)

While rotation-based non-Gaussian estimators hold the reduced-form OLS slope coefficients fixed, Lanne, Meitz & Saikkonen (2017) show that joint maximum likelihood over the reduced-form coefficients and the structural impact matrix can strictly improve finite-sample and asymptotic efficiency when shocks are non-Gaussian. They also prove complete identification on a canonical parameter set ``\mathcal{B}`` with unit diagonal and unique lexicographic column ordering via the projection operator ``\Pi(B_0)`` implemented by [`lms_project_B`](@ref):

```math
y_t = \Pi X_t + B_0 u_t = \Pi X_t + B \operatorname{diag}(\sigma) \varepsilon_t, \qquad B \in \mathcal{B} \ (\operatorname{diag}(B) = \mathbf{1})
```

where:
- ``\Pi`` is the ``n \times (np + 1)`` matrix of reduced-form intercept and autoregressive slope matrices,
- ``B \in \mathcal{B}`` has unit diagonal and uniquely permutes and scales any invertible ``B_0`` into the identified parameter space,
- ``\sigma \in \mathbb{R}_{>0}^n`` collects shock scale parameters so standardized shocks ``\varepsilon_{it}`` satisfy ``E[\varepsilon_{it}] = 0`` and ``\operatorname{Var}(\varepsilon_{it}) = 1``,
- ``\theta = [\operatorname{vec}(\Pi'); \operatorname{vecd}^0(B); \sigma; \lambda]`` is the full joint parameter vector with analytical score and exact 16-block analytical Hessian.

[`estimate_lms_svar`](@ref) implements both the joint 1-step full MLE (`method=:full_ml`) and the asymptotically efficient 3-step estimator (`method=:three_step`), which refines an OLS initial estimate with a single analytical Newton-Fisher step on ``\Pi``. Heterogenous shock families (`:student_t`, `:skewed_t`, `:mixture_normal_asym`, `:pearson_iv`, `:skew_normal`, `:ged`) can be assigned globally or per shock, and `method=:lms` integrates directly into [`irf`](@ref), [`fevd`](@ref), and [`historical_decomposition`](@ref).

```@example id_ng
lms = estimate_lms_svar(model; distributions=:student_t, method=:three_step,
                        n_starts=3, seed=42)
report(lms)
```

```@example id_ng
# Verify canonical projection into the unit-diagonal LMS parameter space B
proj = lms_project_B(lms.B0)
(diag_B = round.(diag(proj.B), digits=6),
 B0_error = round(maximum(abs, lms.B0 .- proj.B * Diagonal(proj.sigma)), digits=10))
```

```@example id_ng
# Compare slope estimates from full ML vs three-step adaptive ML
lms_full = estimate_lms_svar(model; distributions=:student_t, method=:full_ml,
                             n_starts=3, seed=42)
(loglik_3step = round(lms.loglik, digits=2),
 loglik_full = round(lms_full.loglik, digits=2))
```

### Structural Hypothesis Testing & Shock Diagnostics (LMS §4.6, §5.2)

[`diagnose_nongaussian_svar`](@ref) evaluates per-shock Jarque–Bera normality, serial correlation, ARCH clustering, and pairwise independence diagnostics (issuing a `weak_id_warning` when two or more shocks appear Gaussian). Given an `LMSSVARResult`, hypothesis builders (`recursiveness_hypothesis`, `impact_zero_hypothesis`, `impact_linear_hypothesis`, `long_run_hypothesis`, `horizon_irf_hypothesis`, `mixed_svar_hypothesis`, `nonlinear_svar_hypothesis`) enforce diagonal interiority (`B \in \mathcal{B}`, rejecting ``B_{ii} = 0``) while [`test_svar_hypothesis`](@ref) evaluates Wald (propagating joint ``(\Pi, B)`` covariance), Likelihood Ratio (`:lr`), or Lagrange Multiplier (`:lm`) tests with optional residual bootstrap (`bootstrap = :iid | :wild`).

```@example id_ng
diag_lms = diagnose_nongaussian_svar(lms; lags=4)
report(diag_lms)
hyp_rec = recursiveness_hypothesis(3)
ht_wald = test_svar_hypothesis(lms, hyp_rec; method=:wald)
report(ht_wald)
```

---

## Full-System Non-Gaussian GMM and Cramér–von Mises ICA

While plug-in two-step GMM conditions on OLS reduced-form VAR estimates, **Lanne & Luoto (2021)** jointly estimate reduced-form dynamics and the structural impact matrix ``\theta = (\nu', \operatorname{vec}(A_1)', \dots, \operatorname{vec}(A_p)', \operatorname{vec}(B)')'`` by GMM under selected asymmetric and symmetric fourth-order co-kurtosis orthogonality conditions:

```math
E\left[\varepsilon_{it}^3 \varepsilon_{jt}\right] = 0 \; (i \ne j), \qquad
E\left[\varepsilon_{it}^2 \varepsilon_{jt}^2 - 1\right] = 0 \; (i < j), \qquad
Q_{\text{CUE}}(\theta) = \bar{g}_T(\theta)' \hat{S}_T(\theta)^{-1} \bar{g}_T(\theta)
```

where:
- ``\varepsilon_t = B^{-1} u_t`` are the unit-variance structural shocks implied by ``(\nu, A_1, \dots, A_p, B)``,
- ``\bar{g}_T(\theta)`` stacks the VAR score, second-moment unit-variance/orthogonality conditions, and selected fourth-order co-kurtosis moments,
- ``\hat{S}_T(\theta)`` is the HAC long-run covariance estimator (Newey–West/Bartlett/Andrews) updated continuously when `weighting=:cue`.

[`select_gmm_moments`](@ref) implements two-stage Andrews (1999) MSC and Hall et al. (2007) RMSC moment selection in [`GMMMomentSelectionResult`](@ref); [`estimate_lanne_luoto_gmm`](@ref) fits `:two_step`, `:iterated`, or genuine `:cue` estimation returning [`LanneLuotoGMMResult`](@ref); and [`identify_cvm`](@ref) provides distribution-free Cramér–von Mises mutual-independence ICA (Herwartz & Plödt 2016) returning an [`ICASVARResult`](@ref).

```@example id_ng
sim_ll = dgp_ssvarma(MersenneTwister(904); Phi=[0.4 0.1; -0.1 0.3], Theta=zeros(2, 2, 0),
                     dist=:mixture_normal_asym, T=160)
m_ll = estimate_var(sim_ll.Y, 1)
ms_ll = select_gmm_moments(m_ll)
ll_gmm = estimate_lanne_luoto_gmm(m_ll; weighting=:cue, max_iter=50, n_starts=2, seed=42)
cvm_res = identify_cvm(m_ll; max_iter=40, n_starts=2, seed=42)
report(ll_gmm)
(n_asym = length(ms_ll.asym_pairs), n_sym = length(ms_ll.sym_pairs), J_pval = round(ll_gmm.J_pvalue; digits=3), cvm_conv = cvm_res.converged)
```

---

## Non-Fundamental SSVARMA (Gouriéroux, Monfort & Renne 2020)

When observables contain foresight, news, or measurement lags, the structural VARMA moving-average polynomial ``\Theta(L) = I + \Theta_1 L + \dots + \Theta_q L^q`` has companion eigenvalues inside and/or outside the unit circle:

```math
y_t = \mu + \sum_{k=1}^{p} \Phi_k y_{t-k} + \varepsilon_t + \sum_{j=1}^{q} \Theta_j \varepsilon_{t-j}, \qquad \varepsilon_t = C \eta_t
```

where:
- ``y_t \in \mathbb{R}^n`` is the observed endogenous vector at date ``t``,
- ``\Phi_k`` are ``n \times n`` autoregressive coefficient matrices,
- ``\Theta_j`` are ``n \times n`` moving-average matrices whose companion matrix ``F_{\text{MA}}`` separates causal (`|\lambda_i| < 1`) from anti-causal (`|\lambda_i| > 1`) roots,
- ``C`` is the nonsingular impact matrix mapping mutually independent unit-variance non-Gaussian shocks ``\eta_t`` to reduced-form innovations ``\varepsilon_t``.

[`SVARMAModel`](@ref) stores non-fundamental VARMA systems alongside [`fundamentalness_regime`](@ref) (classifying Gouriéroux, Monfort & Renne 2020 Fig. 6 regions `I–V` for bivariate MA(1)), [`ssvarma_filter_shocks`](@ref) recovers ``\hat{\eta}_t`` via ordered real Schur (`ordschur`) forward/backward recursions with the exact log-Jacobian, [`blaschke_representations`](@ref) enumerates second-moment-equivalent Blaschke twins (Lippi & Reichlin 1994), and [`dgp_ssvarma`](@ref) simulates benchmark panels.

```@example id_ng
sim_sv = dgp_ssvarma(MersenneTwister(42); T=200)
sv_mod = SVARMAModel(sim_sv.Y, sim_sv.Phi, sim_sv.Theta, sim_sv.C)
report(sv_mod)
twins = blaschke_representations(sim_sv.Theta, sim_sv.C)
(regime = sv_mod.regime,
 gmr_region = sv_mod.gmr_region,
 n_twins = length(twins),
 interior_shock_err = round(maximum(abs, sv_mod.shocks[10:end-10, :] .- sim_sv.shocks[(sim_sv.p + 10):end-10, :]); digits=8))
```

### SSVARMA Estimation and Fundamentalness Diagnostics

[`estimate_ssvarma`](@ref) jointly estimates non-fundamental `SVARMA(p, q)` systems via exact Schur-filtered pseudo-maximum likelihood (`method=:pml` or `:ml`, handling interior unit-circle crossings) or instrument-based II-2SLS/GMM (`method=:gmm`). Before relying on finite-order VAR innovations, [`test_fundamentalness`](@ref) implements Chen, Choi & Escanciano's (2017) Theorem 2 bootstrap higher-moment orthogonality test between VAR/AR residuals and nonlinear transformations of past data ([`FundamentalnessTestResult`](@ref)), while [`test_sufficient_information`](@ref) implements Forni & Gambetti's (2014) orthogonal-projection test against principal components from an auxiliary macroeconomic information panel ([`SufficientInformationTestResult`](@ref)):

```math
H_0^{\text{fund}}\!: \operatorname{Cov}\!\bigl(\phi(\hat{u}_{i,t}),\, h(Y_{j,t-k})\bigr) = 0, \qquad
H_0^{\text{suff}}\!: \hat{u}_t \perp \{F_{t-1}, \dots, F_{t-p_f}\}
```

where:
- ``\hat{u}_t`` denotes the estimated reduced-form VAR/SVARMA innovations or candidate structural shocks,
- ``\phi(\cdot)`` and ``h(\cdot)`` are higher-moment/bounded transformations across lags ``1 \le k \le m_{\max}``,
- ``F_t`` extracts ``r \le r_{\max}`` principal components from the large external information panel ``X_t``.

```@example id_ng
est_sv = estimate_ssvarma(sim_sv.Y, 0, 1; method=:pml, n_starts=1, max_iter=80, seed=42)
var_sv = estimate_var(sim_sv.Y, 4)
ft_sv = test_fundamentalness(var_sv; max_lag=4, n_boot=99, seed=42)
X_info = hcat(sim_sv.Y, randn(MersenneTwister(907), size(sim_sv.Y, 1), 6))
si_sv = test_sufficient_information(var_sv, X_info; r_max=2, p_f=2)
report(ft_sv)
(fund_pval = round(ft_sv.pvalue; digits=3), suff_pval = round(si_sv.pvalue; digits=3))
```

---

## Common Pitfalls

1. **Gaussian shocks defeat identification.** Non-Gaussian methods require at most one Gaussian shock. Run `normality_test_suite` on the residuals and `test_shock_gaussianity` on the recovered shocks before interpreting anything. If the residuals are multivariate normal, use heteroskedasticity-based methods instead.

2. **Column ordering is not structural.** Statistical identification recovers ``B_0`` up to column permutation and sign. The package normalizes signs to a positive diagonal; `label_shocks` then applies the economic labelling. `test_label_stability` is a match-fraction, not a Procrustes strength test --- see Lewis (2025, Section 6.4). In the FastICA example above the funds-rate column is second, not third, until `label_shocks` moves it.

3. **Check `converged` before reading parameters.** `identify_jade` and `identify_sobi` can end on their iteration cap and report `converged = false`; cross-check their ``B_0`` against FastICA and multi-start ML rather than treating an unconverged iterate as final.

4. **Small samples weaken ICA.** FastICA and JADE estimate higher-order statistics that converge slowly. Below 100 observations, prefer the ML estimators, which impose parametric structure, or SOBI, which relies on second-order autocovariances.

5. **Non-convex rotation objectives can have local optima.** Multi-start ML uses `Optim.LBFGS()` with `ForwardDiff` gradients across Cholesky, JADE, FastICA, and random orthogonal starts (controlled by `n_starts`), whereas `identify_dcov` and `identify_hsic` remain single-start Nelder-Mead. For ``n > 4``, increase `n_starts` or compare across methods; consistent ``B_0`` estimates across objectives are the practical convergence diagnostic.

6. **The LR test requires correct nesting.** Comparing the non-Gaussian and Gaussian likelihoods is valid only when the Gaussian model is nested in the non-Gaussian one: ``\nu_j \to \infty`` for Student-t, ``\sigma_{1j} = \sigma_{2j}`` (and ``\mu_{1j} = \mu_{2j} = 0``) for the mixture, ``\alpha_j = 0`` for the skew-normal, and ``(\nu_j, m_j) \to (0, \infty)`` for the Pearson IV — approached through the scaled Student-t as ``m_j`` grows, with the tail cap at ``m \approx 10^4`` making the limit numerically reachable.

7. **Match the moment block to the source of non-Gaussianity.** `:coskewness` has no identifying power under symmetric shocks (Student-t, Laplace); `:cokurtosis` is silent when shocks are skewed but mesokurtic. Use `:both` if unsure, and read Hansen ``J`` together with `test_gaussian_shock_count`: two Gaussian recovered shocks mean ``B_0`` is not identified, regardless of how small ``J`` is.

---

## References

- Azzalini, Adelchi. 1985. "A Class of Distributions Which Includes the Normal Ones."
  *Scandinavian Journal of Statistics* 12 (2): 171--178. [JSTOR](https://www.jstor.org/stable/4615982)

- Belouchrani, Adel, Karim Abed-Meraim, Jean-François Cardoso, and Eric Moulines. 1997. "A Blind Source Separation Technique Using Second-Order Statistics."
  *IEEE Transactions on Signal Processing* 45 (2): 434--444. [DOI](https://doi.org/10.1109/78.554307)

- Cardoso, Jean-François, and Antoine Souloumiac. 1993. "Blind Beamforming for Non-Gaussian Signals."
  *IEE Proceedings-F* 140 (6): 362--370. [DOI](https://doi.org/10.1049/ip-f-2.1993.0054)

- Comon, Pierre. 1994. "Independent Component Analysis, A New Concept?"
  *Signal Processing* 36 (3): 287--314. [DOI](https://doi.org/10.1016/0165-1684(94)90029-9)

- Gouriéroux, Christian, Alain Monfort, and Jean-Paul Renne. 2017. "Statistical Inference for Independent Component Analysis: Application to Structural VAR Models."
  *Journal of Econometrics* 196 (1): 111--126. [DOI](https://doi.org/10.1016/j.jeconom.2016.09.007)

- Gouriéroux, Christian, Alain Monfort, and Jean-Paul Renne. 2020. "Identification and Estimation in Non-Fundamental Structural VARMA Models."
  *Review of Economic Studies* 87 (4): 1915--1953. [DOI](https://doi.org/10.1093/restud/rdz028)

- Gretton, Arthur, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. "Measuring Statistical Dependence with Hilbert-Schmidt Norms."
  In *Algorithmic Learning Theory*, 63--77. Berlin: Springer. [DOI](https://doi.org/10.1007/11564089_7)

- Herwartz, Helmut. 2018. "Hodges-Lehmann Detection of Structural Shocks: An Analysis of Macroeconomic Dynamics in the Euro Area."
  *Oxford Bulletin of Economics and Statistics* 80 (4): 736--754. [DOI](https://doi.org/10.1111/obes.12234)

- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis."
  *IEEE Transactions on Neural Networks* 10 (3): 626--634. [DOI](https://doi.org/10.1109/72.761722)

- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments."
  *Journal of Business & Economic Statistics* 39 (3): 772--782. [DOI](https://doi.org/10.1080/07350015.2020.1730858)

- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals."
  *Journal of Business & Economic Statistics* 28 (1): 159--168. [DOI](https://doi.org/10.1198/jbes.2009.06003)

- Lanne, Markku, and Jani Luoto. 2021. "GMM Estimation of Non-Gaussian Structural Vector Autoregression."
  *Journal of Business & Economic Statistics* 39 (1): 69--81. [DOI](https://doi.org/10.1080/07350015.2019.1629940)

- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions."
  *Journal of Econometrics* 196 (2): 288--304. [DOI](https://doi.org/10.1016/j.jeconom.2016.06.002)

- Lewis, Daniel J. 2025. "Identification Based on Higher Moments in Macroeconometrics."
  *Annual Review of Economics* 17: 665--693. [DOI](https://doi.org/10.1146/annurev-economics-070124-051419)

- Lippi, Marco, and Lucrezia Reichlin. 1994. "VAR Analysis, Nonfundamental Representations, Blaschke Matrices."
  *Journal of Econometrics* 63 (1): 307--325. [DOI](https://doi.org/10.1016/0304-4076(93)01570-C)

- Matteson, David S., and Ruey S. Tsay. 2017. "Independent Component Analysis via Distance Covariance."
  *Journal of the American Statistical Association* 112 (518): 623--637. [DOI](https://doi.org/10.1080/01621459.2016.1150851)

- Székely, Gábor J., Maria L. Rizzo, and Nail K. Bakirov. 2007. "Measuring and Testing Dependence by Correlation of Distances."
  *Annals of Statistics* 35 (6): 2769--2794. [DOI](https://doi.org/10.1214/009053607000000505)

- Andrews, Donald W. K. 1999. "Consistent Moment Selection Procedures for Generalized Method of Moments Estimation."
  *Econometrica* 67 (3): 543--564. [DOI](https://doi.org/10.1111/1468-0262.00036)

- Chen, Bin, Jinho Choi, and Juan Carlos Escanciano. 2017. "Testing for Fundamental Vector Moving Average Representations."
  *Quantitative Economics* 8 (1): 149--180. [DOI](https://doi.org/10.3982/QE393)

- Forni, Mario, and Luca Gambetti. 2014. "Sufficient Information in Structural VARs."
  *Journal of Monetary Economics* 66: 124--136. [DOI](https://doi.org/10.1016/j.jmoneco.2014.04.005)

- Hall, Alastair R., Atsushi Inoue, Kalidas Jana, and Chiho Shin. 2007. "Information in Generalized Method of Moments Estimation and Entropy-Based Moment Selection."
  *Journal of Econometrics* 138 (2): 488--512. [DOI](https://doi.org/10.1016/j.jeconom.2006.05.006)

- Herwartz, Helmut, and Martin Plödt. 2016. "The Macroeconomic Effects of Oil Price Shocks: Evidence from a Statistical Identification Approach."
  *Journal of International Money and Finance* 61: 30--44. [DOI](https://doi.org/10.1016/j.jimonfin.2015.11.001)
