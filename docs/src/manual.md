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

---

## Innovation Accounting

For detailed coverage of innovation accounting tools, see the dedicated [Innovation Accounting](innovation_accounting.md) chapter. This includes:

- **Impulse Response Functions (IRF)**: Dynamic effects of structural shocks
- **Forecast Error Variance Decomposition (FEVD)**: Variance contribution of each shock
- **Historical Decomposition (HD)**: Decompose observed movements into shock contributions
- **Summary Tables**: Publication-quality output with `summary()`, `table()`, `print_table()`

---

## Bayesian VAR (BVAR)

For comprehensive coverage of Bayesian VAR estimation, see the dedicated [Bayesian VAR](bayesian.md) chapter. Key topics include:

- Minnesota/Litterman prior specification
- Hyperparameter optimization via marginal likelihood (Giannone, Lenza & Primiceri, 2015)
- MCMC estimation with Turing.jl
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

**Reference**: Newey & West (1987, 1994), Andrews (1991)

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

## References

### Vector Autoregression

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). "Monetary Policy Shocks: What Have We Learned and to What End?" *Handbook of Macroeconomics*, 1, 65-148.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48.

### Structural Identification

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). "Narrative Sign Restrictions for SVARs." *American Economic Review*, 108(10), 2802-2829.
- Blanchard, O. J., & Quah, D. (1989). "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review*, 79(4), 655-673.
- Faust, J. (1998). "The Robustness of Identified VAR Conclusions about Money." *Carnegie-Rochester Conference Series on Public Policy*, 49, 207-244.
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies*, 77(2), 665-696.
- Uhlig, H. (2005). "What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure." *Journal of Monetary Economics*, 52(2), 381-419.

### Bayesian Methods

- Bańbura, M., Giannone, D., & Reichlin, L. (2010). "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics*, 25(1), 71-92.
- Carriero, A., Clark, T. E., & Marcellino, M. (2015). "Bayesian VARs: Specification Choices and Forecast Accuracy." *Journal of Applied Econometrics*, 30(1), 46-73.
- Doan, T., Litterman, R., & Sims, C. (1984). "Forecasting and Conditional Projection Using Realistic Prior Distributions." *Econometric Reviews*, 3(1), 1-100.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics*, 97(2), 436-451.
- Kadiyala, K. R., & Karlsson, S. (1997). "Numerical Methods for Estimation and Inference in Bayesian VAR-Models." *Journal of Applied Econometrics*, 12(2), 99-132.
- Litterman, R. B. (1986). "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics*, 4(1), 25-38.

### Inference

- Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica*, 59(3), 817-858.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*, 15(1), 1593-1623.
- Kilian, L. (1998). "Small-Sample Confidence Intervals for Impulse Response Functions." *Review of Economics and Statistics*, 80(2), 218-230.
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.
- Newey, W. K., & West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies*, 61(4), 631-653.
