# Manual

This manual provides a comprehensive theoretical background for the macroeconometric methods implemented in **MacroEconometricModels.jl**, including precise mathematical formulations and references to the literature.

## Quick Start

```julia
model = estimate_var(Y, 2)                                 # Estimate VAR(2) via OLS
sel = select_lag_order(Y, 8)                               # AIC/BIC/HQIC lag selection
irfs = irf(model, 20; method=:cholesky)                    # Cholesky-identified IRFs
decomp = fevd(model, 20)                                   # Forecast error variance decomposition
id = identify_sign(model; check_func=f, n_draws=1000)      # Sign restriction identification
hd = historical_decomposition(model, 198)                  # Historical decomposition
```

---

## Vector Autoregression (VAR)

### The Reduced-Form VAR Model

A VAR(p) model for an ``n``-dimensional vector of endogenous variables ``y_t`` is defined as:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

where:
- ``y_t`` is an ``n \times 1`` vector of endogenous variables at time ``t``
- ``c`` is an ``n \times 1`` vector of intercepts
- ``A_i`` are ``n \times n`` coefficient matrices for lag ``i = 1, \ldots, p``
- ``u_t`` is an ``n \times 1`` vector of reduced-form innovations with ``E[u_t] = 0`` and ``E[u_t u_t'] = \Sigma``

**Reference**: Sims (1980), Lütkepohl (2005, Chapter 2)

### Compact Matrix Representation

For estimation, we stack observations into matrices. Let ``T`` denote the effective sample size after accounting for lags. Define:

```math
Y = \begin{bmatrix} y_{p+1}' \\ y_{p+2}' \\ \vdots \\ y_T' \end{bmatrix}_{(T-p) \times n}, \quad
X = \begin{bmatrix} 1 & y_p' & y_{p-1}' & \cdots & y_1' \\
1 & y_{p+1}' & y_p' & \cdots & y_2' \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & y_{T-1}' & y_{T-2}' & \cdots & y_{T-p}' \end{bmatrix}_{(T-p) \times (1+np)}
```

The VAR can be written in matrix form as:

```math
Y = X B + U
```

where ``B = [c, A_1, A_2, \ldots, A_p]'`` is a ``(1+np) \times n`` coefficient matrix.

### OLS Estimation

The OLS estimator is given by:

```math
\hat{B} = (X'X)^{-1} X'Y
```

The residual covariance matrix is estimated as:

```math
\hat{\Sigma} = \frac{1}{T-p-k} \hat{U}'\hat{U}
```

where ``\hat{U} = Y - X\hat{B}`` and ``k = 1 + np`` is the number of regressors per equation.

**Reference**: Hamilton (1994, Chapter 11), Lütkepohl (2005, Section 3.2)

### VARModel Return Values

`estimate_var` returns a `VARModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original ``T \times n`` data matrix |
| `p` | `Int` | Number of lags |
| `B` | `Matrix{T}` | ``(1+np) \times n`` coefficient matrix ``[c, A_1, \ldots, A_p]'`` |
| `U` | `Matrix{T}` | ``(T-p) \times n`` residual matrix |
| `Sigma` | `Matrix{T}` | ``n \times n`` residual covariance matrix |
| `aic` | `T` | Akaike Information Criterion |
| `bic` | `T` | Bayesian Information Criterion |
| `hqic` | `T` | Hannan-Quinn Information Criterion |

!!! note "Technical Note"
    The coefficient matrix `B` stores the intercept in the first row, followed by ``A_1, A_2, \ldots, A_p`` stacked vertically. To extract lag-``i`` coefficients: `A_i = model.B[(i-1)*n+2 : i*n+1, :]`. The intercept is `model.B[1, :]`.

### Stability Condition

A VAR(p) is stable (stationary) if all eigenvalues of the companion matrix ``F`` lie inside the unit circle:

```math
F = \begin{bmatrix}
A_1 & A_2 & \cdots & A_{p-1} & A_p \\
I_n & 0 & \cdots & 0 & 0 \\
0 & I_n & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_n & 0
\end{bmatrix}_{np \times np}
```

**Stability Check**: ``|\lambda_i| < 1`` for all eigenvalues ``\lambda_i`` of ``F``.

### Information Criteria for Lag Selection

The optimal lag length can be selected using information criteria:

**Akaike Information Criterion (AIC)**:
```math
\text{AIC}(p) = \log|\hat{\Sigma}| + \frac{2}{T}(n^2 p + n)
```

**Bayesian Information Criterion (BIC)**:
```math
\text{BIC}(p) = \log|\hat{\Sigma}| + \frac{\log T}{T}(n^2 p + n)
```

**Hannan-Quinn Criterion (HQ)**:
```math
\text{HQ}(p) = \log|\hat{\Sigma}| + \frac{2 \log(\log T)}{T}(n^2 p + n)
```

Select the lag order ``p`` that minimizes the criterion.

**Reference**: Lütkepohl (2005, Section 4.3)

---

## Structural VAR (SVAR) and Identification

### From Reduced-Form to Structural Shocks

The reduced-form residuals ``u_t`` are linear combinations of structural shocks ``\varepsilon_t``:

```math
u_t = B_0 \varepsilon_t
```

where:
- ``B_0`` is the ``n \times n`` contemporaneous impact matrix
- ``\varepsilon_t`` are structural shocks with ``E[\varepsilon_t \varepsilon_t'] = I_n``

The relationship between the reduced-form and structural covariance is:

```math
\Sigma = B_0 B_0'
```

The **identification problem** is that infinitely many ``B_0`` matrices satisfy this condition. To identify structural shocks, we need ``n(n-1)/2`` additional restrictions.

**Reference**: Kilian & Lütkepohl (2017, Chapter 8)

### Cholesky Identification (Recursive)

The Cholesky decomposition imposes a lower triangular structure on ``B_0``:

```math
B_0 = \text{chol}(\Sigma)
```

This implies a recursive causal ordering where variable ``i`` responds contemporaneously only to variables ``1, 2, \ldots, i-1``.

**Economic Interpretation**: The ordering reflects assumptions about the speed of adjustment. Variables ordered first respond only to their own shocks contemporaneously.

**Reference**: Sims (1980), Christiano, Eichenbaum & Evans (1999)

### Sign Restrictions

Sign restrictions identify structural shocks by constraining the signs of impulse responses at selected horizons. Let ``\Theta_h`` denote the impulse response at horizon ``h``. The identification algorithm:

1. Compute the Cholesky decomposition: ``P = \text{chol}(\Sigma)``
2. Draw a random orthogonal matrix ``Q`` from the Haar measure (using QR decomposition of a random matrix)
3. Compute candidate impact matrix: ``B_0 = PQ``
4. Check if impulse responses ``\Theta_0 = B_0, \Theta_1, \ldots`` satisfy the sign restrictions
5. If restrictions are satisfied, keep the draw; otherwise, discard and repeat

**Implementation**: We use the algorithm of Rubio-Ramírez, Waggoner & Zha (2010).

**Reference**: Faust (1998), Uhlig (2005), Rubio-Ramírez, Waggoner & Zha (2010)

### Narrative Restrictions

Narrative restrictions combine sign restrictions with historical information about specific shocks at particular dates. Following Antolín-Díaz & Rubio-Ramírez (2018):

1. **Shock Sign Narrative**: At date ``t^*``, structural shock ``j`` was positive/negative
2. **Shock Contribution Narrative**: At date ``t^*``, shock ``j`` was the main driver of variable ``i``

The algorithm:
1. Draw orthogonal matrix ``Q`` satisfying sign restrictions
2. Recover structural shocks: ``\varepsilon = B_0^{-1} u``
3. Check if narrative constraints are satisfied
4. Weight the draw using importance sampling

**Reference**: Antolín-Díaz & Rubio-Ramírez (2018)

### Long-Run (Blanchard-Quah) Identification

Long-run restrictions constrain the cumulative effect of structural shocks. For a stationary VAR, the long-run impact matrix is:

```math
C(1) = (I_n - A_1 - A_2 - \cdots - A_p)^{-1} B_0
```

Blanchard & Quah (1989) impose that certain shocks have zero long-run effect on specific variables by requiring ``C(1)`` to be lower triangular:

```math
C(1) = \text{chol}\left( (I - A(1))^{-1} \Sigma (I - A(1)')^{-1} \right)
```

Then ``B_0 = (I - A(1)) C(1)``.

**Economic Application**: Demand shocks have no long-run effect on output (supply-driven long-run fluctuations).

**Reference**: Blanchard & Quah (1989), King, Plosser, Stock & Watson (1991)

### Arias et al. (2018) Identification

When sign restrictions alone are insufficient, one can impose **zero restrictions** on specific impulse responses in addition to sign constraints. Arias, Rubio-Ramírez & Waggoner (2018) develop an algorithm that draws orthogonal rotation matrices ``Q`` from a distribution that is uniform over the set satisfying the zero restrictions, then filters for sign satisfaction.

**Restriction Types**:

| Type | Function | Description |
|------|----------|-------------|
| Zero | `zero_restriction(var, shock; horizon=0)` | Variable `var` does not respond to `shock` at `horizon` |
| Sign | `sign_restriction(var, shock, :positive; horizon=0)` | Response has required sign at `horizon` |

**Algorithm**: For ``n`` variables with ``r_j`` zero restrictions on shock ``j``:
1. Compute MA coefficients ``\Phi_0, \ldots, \Phi_H`` and Cholesky factor ``L``
2. For each draw, construct ``Q`` column-by-column via QR decomposition in the null space of the zero restriction matrix
3. Check sign restrictions on the candidate IRF ``\Theta_h = \Phi_h L Q``
4. Correct non-uniform sampling via importance weights when zero restrictions reduce the dimension

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
Y = randn(200, 3)
for t in 2:200; Y[t,:] = 0.5*Y[t-1,:] + 0.3*randn(3); end
model = estimate_var(Y, 2)

# Define restrictions
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(3, 1; horizon=0)],   # Shock 1 has no impact on var 3
    signs = [sign_restriction(1, 1, :positive),     # Shock 1 → var 1 positive on impact
             sign_restriction(2, 1, :positive)]     # Shock 1 → var 2 positive on impact
)

# Identify
result = identify_arias(model, restrictions, 20; n_draws=1000)
println("Acceptance rate: ", round(result.acceptance_rate * 100, digits=1), "%")

# Weighted IRF percentiles
pct = irf_percentiles(result; probs=[0.16, 0.5, 0.84])
println("Median IRF(1→1, h=0): ", round(pct[1, 1, 1, 2], digits=3))

# Bayesian version
# bresult = identify_arias_bayesian(post, restrictions, 20)
```

The acceptance rate indicates what fraction of random draws satisfy all restrictions simultaneously. Low rates (below 1%) suggest the restrictions may be nearly contradictory or overly stringent. The importance weights correct for non-uniform sampling induced by zero restrictions — the weighted percentiles provide correctly calibrated credible intervals.

### AriasSVARResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `Q_draws` | `Vector{Matrix{T}}` | Accepted rotation matrices |
| `irf_draws` | `Array{T,4}` | ``n_{draws} \times H \times n \times n`` IRF draws |
| `weights` | `Vector{T}` | Importance weights (normalized to sum to 1) |
| `acceptance_rate` | `T` | Fraction of draws satisfying all restrictions |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |

**Reference**: Arias, Rubio-Ramírez & Waggoner (2018)

### Mountford-Uhlig (2009) Penalty Function Identification

When you want a **single best rotation** rather than a distribution of draws, Mountford & Uhlig (2009) provide a penalty function approach. Zero restrictions are enforced exactly via null-space projection; sign restrictions are encouraged via a penalty function optimized with Nelder-Mead.

**When to use**: Use `identify_uhlig` when you need one optimal ``Q`` satisfying sign (and optionally zero) restrictions. Use `identify_arias` when you need a distribution of valid ``Q`` matrices for credible intervals.

**Algorithm**:
1. Parameterize ``Q`` column-by-column using spherical coordinates in the null space of zero-restriction constraints
2. Define penalty: ``-\sum_{s} w_s \cdot \text{sign}_s \cdot \text{IRF}_s / \sigma_s`` where ``w_s = 100`` if the sign is satisfied, ``w_s = 1`` if violated
3. Minimize penalty via two-phase Nelder-Mead: coarse global search, then local refinement

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
Y = randn(200, 3)
for t in 2:200; Y[t,:] = 0.5*Y[t-1,:] + 0.3*randn(3); end
model = estimate_var(Y, 2)

# Define restrictions (same types as Arias)
restrictions = SVARRestrictions(3;
    zeros = [zero_restriction(3, 1; horizon=0)],
    signs = [sign_restriction(1, 1, :positive),
             sign_restriction(2, 1, :positive)]
)

# Find optimal Q
result = identify_uhlig(model, restrictions, 20)
println("Converged: ", result.converged)
println("Penalty: ", round(result.penalty, digits=2))
println("Impact IRF(1→1): ", round(result.irf[1, 1, 1], digits=3))
```

### UhligSVARResult Return Values

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `Matrix{T}` | Optimal rotation matrix |
| `irf` | `Array{T,3}` | ``H \times n \times n`` impulse responses |
| `penalty` | `T` | Total penalty at optimum (negative = better) |
| `shock_penalties` | `Vector{T}` | Per-shock penalty values |
| `restrictions` | `SVARRestrictions` | The imposed restrictions |
| `converged` | `Bool` | Whether all sign restrictions are satisfied |

**Reference**: Mountford & Uhlig (2009)

---

## Innovation Accounting

For detailed coverage of innovation accounting tools, see the dedicated [Innovation Accounting](innovation_accounting.md) chapter. This includes:

- **Impulse Response Functions (IRF)**: Dynamic effects of structural shocks
- **Forecast Error Variance Decomposition (FEVD)**: Variance contribution of each shock
- **Historical Decomposition (HD)**: Decompose observed movements into shock contributions
- **Summary Tables**: Publication-quality output with `report()`, `table()`, `print_table()`

---

## Bayesian VAR (BVAR)

For comprehensive coverage of Bayesian VAR estimation, see the dedicated [Bayesian VAR](bayesian.md) chapter. Key topics include:

- Minnesota/Litterman prior specification
- Hyperparameter optimization via marginal likelihood (Giannone, Lenza & Primiceri, 2015)
- Conjugate NIW posterior sampling
- Posterior inference and credible intervals

---

## Information Criteria and Model Selection

### Log-Likelihood

For a Gaussian VAR, the log-likelihood is:

```math
\log L = -\frac{T \cdot n}{2} \log(2\pi) - \frac{T}{2} \log|\Sigma| - \frac{1}{2} \sum_{t=1}^{T} u_t' \Sigma^{-1} u_t
```

### Marginal Likelihood (Bayesian)

For Bayesian model comparison, we use the marginal likelihood (also called evidence):

```math
p(Y | \mathcal{M}) = \int p(Y | \theta, \mathcal{M}) p(\theta | \mathcal{M}) \, d\theta
```

Models with higher marginal likelihood better balance fit and complexity.

---

## Covariance Estimation

### Newey-West HAC Estimator

For robust inference in the presence of heteroskedasticity and autocorrelation, we use the Newey-West (1987, 1994) estimator:

```math
\hat{V}_{NW} = (X'X)^{-1} \hat{S} (X'X)^{-1}
```

where the long-run covariance ``\hat{S}`` is:

```math
\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')
```

with ``\hat{\Gamma}_j = \frac{1}{T} \sum_{t=j+1}^{T} \hat{u}_t \hat{u}_{t-j}' x_t x_{t-j}'``.

### Kernel Functions

The weight function ``w_j`` depends on the kernel:

**Bartlett (Newey-West)**:
```math
w_j = 1 - \frac{j}{m+1}
```

**Parzen**:
```math
w_j = \begin{cases}
1 - 6x^2 + 6|x|^3 & |x| \leq 0.5 \\
2(1-|x|)^3 & 0.5 < |x| \leq 1
\end{cases}
```

where ``x = j/(m+1)``.

**Quadratic Spectral (Andrews, 1991)**:
```math
w_j = \frac{25}{12\pi^2 x^2} \left( \frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5) \right)
```

### Automatic Bandwidth Selection

Newey & West (1994) provide a data-driven bandwidth:

```math
m^* = 1.1447 \left( \hat{\alpha} \cdot T \right)^{1/3}
```

where ``\hat{\alpha}`` is estimated from an AR(1) fit to the residuals:

```math
\hat{\alpha} = \frac{4\hat{\rho}^2}{(1-\hat{\rho})^4}
```

### White Heteroscedasticity-Robust Estimator (HC0)

When errors are heteroscedastic but serially uncorrelated, the White (1980) estimator provides consistent standard errors without requiring bandwidth selection:

```math
\hat{V}_{W} = (X'X)^{-1} \left( \sum_{t=1}^{T} \hat{u}_t^2 x_t x_t' \right) (X'X)^{-1}
```

where
- ``\hat{u}_t`` are the OLS residuals
- ``x_t`` is the ``k \times 1`` regressor vector at time ``t``

### Driscoll-Kraay Panel-Robust Estimator

For panel data with both cross-sectional and temporal dependence, the Driscoll & Kraay (1998) estimator applies HAC estimation to the cross-sectional averages of the moment conditions. This produces standard errors robust to both heteroscedasticity, serial correlation, and cross-sectional dependence.

### Julia Implementation

```julia
using MacroEconometricModels

Y = randn(200, 3)
for t in 2:200; Y[t,:] = 0.5*Y[t-1,:] + 0.3*randn(3); end

# Construct design matrices
Y_eff, X = construct_var_matrices(Y, 2)
residuals = Y_eff - X * ((X'X) \ (X'Y_eff))

# Newey-West HAC (default: Bartlett kernel, automatic bandwidth)
V_nw = newey_west(X, residuals; bandwidth=0, kernel=:bartlett)

# White heteroscedasticity-robust (HC0)
V_w = white_vcov(X, residuals)

# Driscoll-Kraay for panel data
# V_dk = driscoll_kraay(X, residuals; bandwidth=4)

# Automatic bandwidth selection
bw = optimal_bandwidth_nw(residuals)
println("Optimal Newey-West bandwidth: ", bw)
```

The Newey-West estimator is appropriate for time series with heteroscedastic and serially correlated errors — the standard choice for LP and VAR applications. The White estimator is simpler but inconsistent when errors are autocorrelated. The Driscoll-Kraay estimator extends HAC to panel settings where cross-sectional units may be correlated (e.g., country-level macro panels).

### Comparing LP and VAR

The `compare_var_lp` function provides a structured comparison of VAR and LP impulse responses:

```julia
comparison = compare_var_lp(Y, 1, 20; lags=4)
```

This estimates both a VAR and LP model on the same data and returns the IRFs from each, facilitating visual and numerical comparison. Under correct specification, the IRFs should be close (Plagborg-Møller & Wolf 2021); substantial disagreement suggests dynamic misspecification in the VAR.

**Reference**: Newey & West (1987, 1994), Andrews (1991), Driscoll & Kraay (1998)

---

## Complete Example

This example demonstrates an end-to-end VAR workflow from lag selection through structural analysis.

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

# Generate data from a persistent VAR(1) DGP
T, n = 200, 3
Y = zeros(T, n)
A = [0.8 0.1 -0.1; 0.05 0.7 0.0; 0.1 0.2 0.75]
for t in 2:T
    Y[t, :] = A * Y[t-1, :] + 0.3 * randn(n)
end

# Step 1: Select lag order
sel = select_lag_order(Y, 8)
println("AIC lag: ", sel.p_aic, "  BIC lag: ", sel.p_bic)

# Step 2: Estimate VAR
model = estimate_var(Y, sel.p_bic)
println("AIC: ", round(model.aic, digits=2),
        "  BIC: ", round(model.bic, digits=2))

# Step 3: Check stability
stab = is_stationary(model)
println("Stationary: ", stab.is_stationary,
        "  Max modulus: ", round(stab.max_modulus, digits=4))

# Step 4: Cholesky IRF with bootstrap CI
irfs = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=500)
println("Impact of shock 1 on var 1: ", round(irfs.values[1, 1, 1], digits=3))
println("After 8 periods: ", round(irfs.values[9, 1, 1], digits=3))

# Step 5: FEVD
decomp = fevd(model, 20)
println("Var 1 explained by shock 1 at h=1: ",
        round(decomp.proportions[1, 1, 1] * 100, digits=1), "%")
println("Var 1 explained by shock 1 at h=20: ",
        round(decomp.proportions[20, 1, 1] * 100, digits=1), "%")

# Step 6: Historical decomposition
hd = historical_decomposition(model, size(model.U, 1))
verify_decomposition(hd)  # Should return true
```

The lag selection criteria typically agree when the true DGP is low-order; BIC tends to be more conservative and is preferred when parsimony matters. The stability check confirms all companion matrix eigenvalues lie inside the unit circle, validating the use of standard asymptotic inference. The FEVD at long horizons reveals the unconditional variance decomposition, showing which shocks are the dominant drivers of each variable. The historical decomposition identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` should hold exactly up to numerical precision.

---

### See Also

- [VECM Analysis](vecm.md) -- Error correction models for cointegrated systems
- [Local Projections](lp.md) -- Model-free impulse response estimation (Jorda 2005)
- [Non-Gaussian Identification](nongaussian.md) -- ICA, ML, and heteroskedasticity-based SVAR identification
- [Innovation Accounting](innovation_accounting.md) -- IRF, FEVD, and historical decomposition details
- [API Reference](api_functions.md) -- Complete function signatures

## References

### Vector Autoregression

- Christiano, Lawrence J., Martin Eichenbaum, and Charles L. Evans. 1999. "Monetary Policy Shocks: What Have We Learned and to What End?" In *Handbook of Macroeconomics*, Vol. 1, edited by John B. Taylor and Michael Woodford, 65–148. Amsterdam: Elsevier. [https://doi.org/10.1016/S1574-0048(99)01005-8](https://doi.org/10.1016/S1574-0048(99)01005-8)
- Hamilton, James D. 1994. *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1–48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)

### Structural Identification

- Arias, Jonas E., Juan F. Rubio-Ramírez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications." *Econometrica* 86 (2): 685–720. [https://doi.org/10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Antolín-Díaz, Juan, and Juan F. Rubio-Ramírez. 2018. "Narrative Sign Restrictions for SVARs." *American Economic Review* 108 (10): 2802–2829. [https://doi.org/10.1257/aer.20161852](https://doi.org/10.1257/aer.20161852)
- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655–673.
- Faust, Jon. 1998. "The Robustness of Identified VAR Conclusions about Money." *Carnegie-Rochester Conference Series on Public Policy* 49: 207–244. [https://doi.org/10.1016/S0167-2231(99)00009-3](https://doi.org/10.1016/S0167-2231(99)00009-3)
- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Mountford, Andrew, and Harald Uhlig. 2009. "What Are the Effects of Fiscal Policy Shocks?" *Journal of Applied Econometrics* 24 (6): 960–992. [https://doi.org/10.1002/jae.1079](https://doi.org/10.1002/jae.1079)
- Rubio-Ramírez, Juan F., Daniel F. Waggoner, and Tao Zha. 2010. "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies* 77 (2): 665–696. [https://doi.org/10.1111/j.1467-937X.2009.00578.x](https://doi.org/10.1111/j.1467-937X.2009.00578.x)
- Uhlig, Harald. 2005. "What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure." *Journal of Monetary Economics* 52 (2): 381–419. [https://doi.org/10.1016/j.jmoneco.2004.05.007](https://doi.org/10.1016/j.jmoneco.2004.05.007)

### Bayesian Methods

- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2015. "Bayesian VARs: Specification Choices and Forecast Accuracy." *Journal of Applied Econometrics* 30 (1): 46–73. [https://doi.org/10.1002/jae.2315](https://doi.org/10.1002/jae.2315)
- Doan, Thomas, Robert Litterman, and Christopher Sims. 1984. "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews* 3 (1): 1–100. [https://doi.org/10.1080/07474938408800053](https://doi.org/10.1080/07474938408800053)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Kadiyala, K. Rao, and Sune Karlsson. 1997. "Numerical Methods for Estimation and Inference in Bayesian VAR-Models." *Journal of Applied Econometrics* 12 (2): 99–132. [https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### Inference

- Driscoll, John C., and Aart C. Kraay. 1998. "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data." *Review of Economics and Statistics* 80 (4): 549–560. [https://doi.org/10.1162/003465398557825](https://doi.org/10.1162/003465398557825)
- Andrews, Donald W. K. 1991. "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59 (3): 817–858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. *Bayesian Data Analysis*. 3rd ed. Boca Raton, FL: CRC Press. ISBN 978-1-4398-4095-5.
- Hoffman, Matthew D., and Andrew Gelman. 2014. "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research* 15 (1): 1593–1623.
- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions." *Review of Economics and Statistics* 80 (2): 218–230. [https://doi.org/10.1162/003465398557465](https://doi.org/10.1162/003465398557465)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Newey, Whitney K., and Kenneth D. West. 1994. "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies* 61 (4): 631–653. [https://doi.org/10.2307/2297912](https://doi.org/10.2307/2297912)
