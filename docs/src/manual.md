# Manual

This manual provides a comprehensive theoretical background for the macroeconometric methods implemented in **MacroEconometricModels.jl**, including precise mathematical formulations and references to the literature.

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

## Impulse Response Functions (IRF)

### Definition

The impulse response function ``\Theta_h`` measures the effect of a one-unit structural shock at time ``t`` on the endogenous variables at time ``t+h``:

```math
\Theta_h = \frac{\partial y_{t+h}}{\partial \varepsilon_t'}
```

For a VAR, the IRF at horizon ``h`` is computed recursively:

```math
\Theta_h = \sum_{i=1}^{\min(h,p)} A_i \Theta_{h-i}
```

with ``\Theta_0 = B_0`` (the structural impact matrix).

### Companion Form Representation

Using the companion form, IRFs can be computed as:

```math
\Theta_h = J F^h J' B_0
```

where ``J = [I_n, 0, \ldots, 0]`` is an ``n \times np`` selection matrix and ``F`` is the companion matrix.

### Cumulative IRF

The cumulative impulse response up to horizon ``H`` is:

```math
\Theta^{cum}_H = \sum_{h=0}^{H} \Theta_h
```

As ``H \to \infty``, for a stable VAR:

```math
\Theta^{cum}_\infty = (I_n - A_1 - \cdots - A_p)^{-1} B_0
```

### Confidence Intervals

**Bootstrap (Frequentist)**: We use the residual bootstrap of Kilian (1998):
1. Estimate the VAR and save residuals ``\hat{u}_t``
2. Generate bootstrap sample by resampling residuals with replacement
3. Re-estimate the VAR and compute IRFs
4. Repeat ``B`` times to build the distribution

**Credible Intervals (Bayesian)**: For each MCMC draw, compute IRFs and report posterior quantiles (e.g., 16th and 84th percentiles for 68% intervals).

**Reference**: Kilian (1998), Lütkepohl (2005, Chapter 3)

---

## Forecast Error Variance Decomposition (FEVD)

### Definition

The FEVD measures the proportion of the ``h``-step ahead forecast error variance of variable ``i`` attributable to structural shock ``j``:

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n} (\Theta_s)_{ik}^2}
```

where ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the impulse response matrix at horizon ``s``.

### Properties

- ``0 \leq \text{FEVD}_{ij}(h) \leq 1`` for all ``i, j, h``
- ``\sum_{j=1}^{n} \text{FEVD}_{ij}(h) = 1`` for all ``i, h``
- As ``h \to \infty``, FEVD converges to the unconditional variance decomposition

**Reference**: Lütkepohl (2005, Section 2.3.3)

---

## Bayesian VAR (BVAR)

### Bayesian Framework

In the Bayesian approach, we treat the VAR parameters as random variables and update our beliefs using Bayes' theorem:

```math
p(B, \Sigma | Y) \propto p(Y | B, \Sigma) \cdot p(B, \Sigma)
```

where:
- ``p(Y | B, \Sigma)`` is the likelihood
- ``p(B, \Sigma)`` is the prior
- ``p(B, \Sigma | Y)`` is the posterior

### The Minnesota Prior

The Minnesota prior (Litterman, 1986; Doan, Litterman & Sims, 1984) shrinks VAR coefficients toward a random walk prior:

**Prior Mean**: Each variable follows a random walk:
```math
E[A_{1,ii}] = 1, \quad E[A_{1,ij}] = 0 \text{ for } i \neq j, \quad E[A_l] = 0 \text{ for } l > 1
```

**Prior Variance**: The prior variance for coefficient ``(i,j)`` at lag ``l`` is:

```math
\text{Var}(A_{l,ij}) = \begin{cases}
\frac{\tau^2}{l^d} & \text{if } i = j \text{ (own lag)} \\
\frac{\tau^2 \omega^2}{l^d} \cdot \frac{\sigma_i^2}{\sigma_j^2} & \text{if } i \neq j \text{ (cross lag)}
\end{cases}
```

where:
- ``\tau`` is the overall tightness (shrinkage intensity)
- ``d`` is the lag decay (typically ``d = 2``)
- ``\omega`` controls cross-variable shrinkage (typically ``\omega < 1``)
- ``\sigma_i^2`` is the residual variance from a univariate AR(1) for variable ``i``

### Dummy Observations Approach

We implement the Minnesota prior using dummy observations (Theil-Goldberger mixed estimation). The augmented data matrices are:

**Prior on coefficients** (tightness dummies):
```math
Y_d = \begin{bmatrix}
\text{diag}(\sigma_1, \ldots, \sigma_n) / \tau \\
0_{n(p-1) \times n} \\
\text{diag}(\sigma_1, \ldots, \sigma_n) \\
0_{1 \times n}
\end{bmatrix}, \quad
X_d = \begin{bmatrix}
0_{n \times 1} & J_p \otimes \text{diag}(\sigma_1, \ldots, \sigma_n) / \tau \\
0_{n(p-1) \times 1} & I_{p-1} \otimes \text{diag}(\sigma_1, \ldots, \sigma_n) \\
0_{n \times 1} & 0_{n \times np} \\
c & 0_{1 \times np}
\end{bmatrix}
```

where ``J_p = \text{diag}(1, 2^d, \ldots, p^d)``.

The posterior is then computed as OLS on the augmented data ``[Y; Y_d]`` and ``[X; X_d]``.

**Reference**: Litterman (1986), Kadiyala & Karlsson (1997), Bańbura, Giannone & Reichlin (2010)

### Hyperparameter Optimization (Giannone, Lenza & Primiceri, 2015)

Rather than selecting ``\tau`` subjectively, we can optimize it by maximizing the marginal likelihood:

```math
p(Y | \tau) = \int p(Y | B, \Sigma) p(B, \Sigma | \tau) \, dB \, d\Sigma
```

For the Normal-Inverse-Wishart prior with dummy observations, the log marginal likelihood has an analytical form:

```math
\log p(Y | \tau) = c + \frac{T-k}{2} \log|\tilde{S}^{-1}| - \frac{T_d}{2} \log|\tilde{S}_d^{-1}| + \log \frac{\Gamma_n(\frac{T+T_d - k}{2})}{\Gamma_n(\frac{T_d - k}{2})}
```

where ``\tilde{S}`` and ``\tilde{S}_d`` are the residual sum of squares from the augmented and dummy-only regressions.

**Reference**: Giannone, Lenza & Primiceri (2015), Carriero, Clark & Marcellino (2015)

### MCMC Estimation with Turing.jl

For more flexible priors or non-conjugate settings, we use MCMC via Turing.jl with the NUTS sampler:

```julia
@model function bvar_model(Y, X, prior_mean, prior_var, ν₀, S₀)
    n = size(Y, 2)
    k = size(X, 2)

    # Prior on error covariance
    Σ ~ InverseWishart(ν₀, S₀)

    # Prior on coefficients
    B ~ MatrixNormal(prior_mean, prior_var, Σ)

    # Likelihood
    for t in axes(Y, 1)
        Y[t, :] ~ MvNormal(X[t, :]' * B, Σ)
    end
end
```

**Reference**: Gelman et al. (2013), Hoffman & Gelman (2014)

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
