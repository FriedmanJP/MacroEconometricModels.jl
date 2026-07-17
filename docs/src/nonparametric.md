# [Nonparametric Regression & Density](@id nonparametric_page)

**MacroEconometricModels.jl** estimates distributions and conditional means without imposing a parametric form. Kernel density estimation reconstructs the shape of a sample — firm-growth rates, forecast errors, cross-sectional dispersion — from a smoothed sum of bumps centred at each observation. Nonparametric regression lets the data trace the shape of a conditional mean ``m(x) = E[y \mid x]``: Engel curves, Phillips-curve nonlinearities, or any relationship where a straight line is too rigid. These match EViews' kernel-fit and nonparametric-regression graphs.

- **Kernel density** — `kernel_density` estimates ``\hat f(x)`` with Gaussian, Epanechnikov, triangular, or uniform kernels and data-driven bandwidths: the Silverman rule of thumb (`bw.nrd0`) or the Sheather–Jones (1991) plug-in (`bw.SJ`)
- **Kernel regression** — `kernel_reg` fits the Nadaraya–Watson local-constant estimator, the Fan–Gijbels local-linear estimator (with automatic boundary-bias correction), or a local polynomial of arbitrary degree, with leave-one-out cross-validated bandwidths and pointwise standard-error bands
- **LOWESS** — `lowess` is Cleveland's (1979) robust locally-weighted scatterplot smoother: tricube-weighted local-linear fits with bisquare robustifying passes that discount outliers

`kernel_density` returns a [`KernelDensity`](@ref), `kernel_reg` a [`KernelRegression`](@ref), and `lowess` a [`LowessFit`](@ref). All three integrate with `report`, `refs`, and `plot_result`.

```@setup np
using MacroEconometricModels, Random, Statistics
# A fixed-seed nonlinear relationship y = sin(x) + noise, plus a skewed sample.
Random.seed!(20240717)
x = sort(4π .* rand(200))
y = sin.(x) .+ 0.3 .* randn(200)
# A right-skewed sample for the density example.
z = exp.(0.5 .* randn(400))
```

## Quick Start

**Recipe 1: Kernel density with a Silverman bandwidth**

```@example np
kd = kernel_density(z)               # Gaussian kernel, bw.nrd0 rule
report(kd)
```

**Recipe 2: Sheather–Jones plug-in bandwidth**

```@example np
kd_sj = kernel_density(z; kernel=:epanechnikov, bw=:sj)
report(kd_sj)
```

**Recipe 3: Local-linear regression with a CV bandwidth**

```@example np
kr = kernel_reg(y, x; method=:ll, bw=:cv)
report(kr)
```

**Recipe 4: Robust LOWESS smoother**

```@example np
lf = lowess(y, x; f=0.3, iter=3)
report(lf)
```

```julia
plot_result(kr)   # scatter of (x, y) with the fitted curve and SE band
```

---

## Kernel Density Estimation

The kernel density estimator smooths the empirical distribution by placing a scaled kernel at each observation:

```math
\hat f(x_0) = \frac{1}{n h} \sum_{i=1}^{n} K\!\left(\frac{x_0 - y_i}{h}\right)
```

where:
- ``K(\cdot)`` is a kernel scaled to unit variance (so a common ``h`` smooths comparably across kernels and ``\int \hat f = 1``)
- ``h`` is the bandwidth controlling the trade-off between bias and variance
- ``n`` is the sample size

The bandwidth ``h`` dominates the estimate. Two data-driven rules are available. The **Silverman rule of thumb** — identical to R's `bw.nrd0` — is fast and reliable for roughly unimodal data:

```math
h = 0.9 \cdot \min\!\left(\hat\sigma,\; \frac{\text{IQR}}{1.349}\right) \cdot n^{-1/5}
```

The **Sheather–Jones plug-in** (`bw=:sj`, matching R's `bw.SJ`) targets the AMISE-optimal bandwidth by solving a fixed-point equation for ``h`` that estimates the integrated squared density derivative from a pilot bandwidth. It is more accurate for multimodal or heavy-tailed samples at higher computational cost.

!!! note "Technical Note"
    The Sheather–Jones solve-the-equation method brackets the root on ``[0.1\,h_{\max}, h_{\max}]`` around the Silverman scale and locates it by bisection. On flat or degenerate samples the pilot functional can turn non-positive; the estimator then warns and falls back to the Silverman rule rather than erroring.

```@example np
kd = kernel_density(z; kernel=:gaussian, bw=:sj, npoints=512)
round(kd.bandwidth, digits=4)   # chosen bandwidth
```

The density is evaluated on an equally-spaced grid of `npoints`, extending `cut·h` beyond the data range on each side.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `kernel` | `Symbol` | `:gaussian` | `:gaussian`, `:epanechnikov`, `:triangular`, `:uniform` |
| `bw` | `Symbol` or `Real` | `:silverman` | `:silverman` (`bw.nrd0`), `:sj` (`bw.SJ`), or a positive value |
| `npoints` | `Int` | `512` | Number of grid points |
| `cut` | `Real` | `3.0` | Grid extends `cut·h` beyond the data range |

| Field | Type | Description |
|---|---|---|
| `x` | `Vector{T}` | Grid abscissae |
| `density` | `Vector{T}` | ``\hat f(x)`` on the grid |
| `bandwidth` | `T` | Chosen bandwidth ``h`` |
| `kernel` | `Symbol` | Kernel used |
| `bw_method` | `Symbol` | `:silverman` / `:sj` / `:user` |

---

## Kernel Regression

Nonparametric regression estimates the conditional mean ``m(x) = E[y \mid x]`` by weighted least squares in a shrinking neighbourhood of each target point. The **Nadaraya–Watson** (local-constant) estimator is a kernel-weighted average:

```math
\hat m_{\text{NW}}(x_0) = \frac{\sum_i K\!\left(\frac{x_i - x_0}{h}\right) y_i}{\sum_i K\!\left(\frac{x_i - x_0}{h}\right)}
```

The **local-polynomial** estimator (Fan–Gijbels) fits, at each ``x_0``, a weighted regression of ``y`` on the local design ``[1, (x_i - x_0), \dots, (x_i - x_0)^p]`` with kernel weights, and reports the intercept as the fit. The local-linear case (``p = 1``, `method=:ll`) carries automatic boundary-bias correction that Nadaraya–Watson lacks.

where:
- ``h`` is the bandwidth
- ``p`` is the local-polynomial degree (`:nw`⇒0, `:ll`⇒1, `:lp`⇒`degree`)

The bandwidth is selected by **leave-one-out cross-validation** (`bw=:cv`), minimising ``\sum_i (y_i - \hat m_{-i}(x_i))^2`` over a bandwidth grid, or by a Silverman rule of thumb (`bw=:rot`). Pointwise standard errors use the effective-weight sandwich form ``\operatorname{Var}(\hat m(x_0)) = \hat\sigma^2 \, \lVert \ell(x_0) \rVert^2``, where ``\hat m(x_0) = \sum_i \ell_i(x_0) y_i`` and ``\hat\sigma^2`` is the residual variance on effective degrees of freedom.

```@example np
kr = kernel_reg(y, x; method=:ll, bw=:cv, kernel=:gaussian)
report(kr)
```

The cross-validated bandwidth balances fit and smoothness; the standard-error band widens where the design is sparse. The fit is evaluated at the sorted design points.

```julia
plot_result(kr)
```

| Keyword | Type | Default | Description |
|---|---|---|---|
| `method` | `Symbol` | `:ll` | `:nw` (local constant), `:ll` (local linear), `:lp` (local polynomial) |
| `degree` | `Int` | `1` | Local-polynomial degree for `:lp` |
| `bw` | `Symbol` or `Real` | `:cv` | `:cv` (leave-one-out), `:rot` (rule of thumb), or a positive value |
| `kernel` | `Symbol` | `:gaussian` | Kernel for the local weights |

| Field | Type | Description |
|---|---|---|
| `x`, `fitted` | `Vector{T}` | Sorted design points and ``\hat m(x)`` |
| `se` | `Vector{T}` | Pointwise standard errors |
| `bandwidth` | `T` | Chosen bandwidth |
| `method`, `degree` | `Symbol`, `Int` | Estimator and local degree |
| `sigma2` | `T` | Residual variance ``\hat\sigma^2`` |

---

## LOWESS

LOWESS (Cleveland 1979) is a robust scatterplot smoother. At each point it fits a local linear regression to the nearest ``\lfloor f n \rfloor`` neighbours weighted by the tricube function

```math
w(d) = \left(1 - (d/d_{\max})^3\right)^3, \qquad d \le d_{\max}
```

where ``d`` is the distance to the target and ``d_{\max}`` the distance to the farthest neighbour in the window. After the initial fit, `iter` robustifying passes reweight each observation by the bisquare of its scaled residual, so outliers are progressively discounted. The span ``f`` (fraction of points per window) controls smoothness.

```@example np
lf = lowess(y, x; f=0.3, iter=3)
report(lf)
```

A larger span produces a smoother curve; more iterations increase robustness to outliers. Values are returned sorted by `x`.

```julia
plot_result(lf)
```

| Keyword | Type | Default | Description |
|---|---|---|---|
| `f` | `Real` | `2/3` | Span: fraction of points in each local window |
| `iter` | `Int` | `3` | Number of bisquare robustifying passes |
| `delta` | `Real` or `nothing` | `nothing` | Interpolation skip (default `0.01·range(x)`) |

---

## Complete Example

A full workflow: estimate the density of the response, fit both a local-linear regression and a robust LOWESS smoother, and compare bandwidths.

```@example np
# Density of the response variable with a Sheather–Jones bandwidth
kd = kernel_density(y; bw=:sj)

# Local-linear regression, cross-validated bandwidth
kr = kernel_reg(y, x; method=:ll, bw=:cv)

# Robust LOWESS with a moderate span
lf = lowess(y, x; f=0.3, iter=3)

report(kr)
```

```@example np
# Compare the CV bandwidth against the rule-of-thumb default
kr_rot = kernel_reg(y, x; method=:ll, bw=:rot)
(cv = round(kr.bandwidth, digits=4), rot = round(kr_rot.bandwidth, digits=4))
```

The cross-validated bandwidth adapts to the curvature of ``\sin(x)``, while the rule of thumb keys off the spread of ``x`` alone. Where the two disagree, prefer the CV bandwidth for regression and reserve the rule of thumb as a fast default.

---

## Common Pitfalls

1. **Bandwidth dominates the estimate.** Too small a bandwidth produces a spiky, high-variance fit; too large oversmooths and washes out structure. Start from the data-driven default (`:silverman`/`:cv`) and adjust deliberately.
2. **Sheather–Jones on degenerate samples.** On flat or near-constant data the SJ pilot functional can turn non-positive. The estimator warns and returns the Silverman bandwidth — check the warning rather than trusting the number blindly.
3. **Nadaraya–Watson boundary bias.** The local-constant estimator is biased near the edges of the support. Prefer `method=:ll` (local linear), which corrects this automatically.
4. **LOWESS span is a fraction, not a count.** `f` is the fraction of observations in each window (``0 < f \le 1``), not a bandwidth in ``x`` units. A span of `2/3` uses two-thirds of the sample per local fit.
5. **Results are sorted by `x`.** `kernel_reg` and `lowess` sort the data internally; the returned `x`/`fitted` are in ascending-`x` order, not the input order.

---

## References

```@example np
refs(kernel_reg(y, x); format=:text)
```

- Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing Scatterplots. *Journal of the American Statistical Association* 74(368), 829–836. [doi:10.1080/01621459.1979.10481038](https://doi.org/10.1080/01621459.1979.10481038)
- Fan, J. and Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
- Nadaraya, E. A. (1964). On Estimating Regression. *Theory of Probability and Its Applications* 9(1), 141–142. [doi:10.1137/1109020](https://doi.org/10.1137/1109020)
- Sheather, S. J. and Jones, M. C. (1991). A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation. *Journal of the Royal Statistical Society, Series B* 53(3), 683–690. [doi:10.1111/j.2517-6161.1991.tb01857.x](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x)
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
- Watson, G. S. (1964). Smooth Regression Analysis. *Sankhyā, Series A* 26(4), 359–372.
```
