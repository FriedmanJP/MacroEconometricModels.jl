# [Nonparametric Regression & Density](@id nonparametric_page)

**MacroEconometricModels.jl** estimates distributions and conditional means without imposing a parametric form. Kernel density estimation reconstructs the shape of a sample — firm-growth rates, forecast errors, cross-sectional dispersion — from a smoothed sum of bumps centred at each observation. Nonparametric regression lets the data trace the shape of a conditional mean ``m(x) = E[y \mid x]``: Engel curves, Phillips-curve nonlinearities, or any relationship where a straight line is too rigid. These match EViews' kernel-fit and nonparametric-regression graphs.

- **Kernel density** — `kernel_density` estimates ``\hat f(x)`` with Gaussian, Epanechnikov, triangular, or uniform kernels and data-driven bandwidths: the Silverman rule of thumb (`bw.nrd0`) or the Sheather-Jones (1991) plug-in (`bw.SJ`)
- **Kernel regression** — `kernel_reg` fits the Nadaraya-Watson local-constant estimator, the Fan-Gijbels local-linear estimator (with automatic boundary-bias correction), or a local polynomial of arbitrary degree, with leave-one-out cross-validated bandwidths and pointwise standard-error bands
- **LOWESS** — `lowess` is Cleveland's (1979) robust locally-weighted scatterplot smoother: tricube-weighted local-linear fits with bisquare robustifying passes that discount outliers

`kernel_density` returns a [`KernelDensity`](@ref), `kernel_reg` a [`KernelRegression`](@ref), and `lowess` a [`LowessFit`](@ref). All three integrate with `report`, `refs`, and `plot_result`.

For parametric conditional means see [Linear Regression](@ref regression_page); for nonlinear *time series* models that estimate a transition function instead of fitting locally, see [Nonlinear Time Series](@ref nonlinear_page).

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

The examples use two synthetic samples: `z`, a lognormal draw of 400 observations standing in for a right-skewed cross-section, and the pair `(x, y)` of 200 observations from ``y = \sin(x) + 0.3\varepsilon`` on ``[0, 4\pi]``, a conditional mean no polynomial of low order can follow.

**Recipe 1: Kernel density with a Silverman bandwidth**

```@example np
kd = kernel_density(z)               # Gaussian kernel, bw.nrd0 rule
report(kd)
```

**Recipe 2: Sheather-Jones plug-in bandwidth**

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

**Recipe 5: Compare estimators against the known truth**

```@example np
kr_nw = kernel_reg(y, x; method=:nw, bw=:cv)
kr_lp = kernel_reg(y, x; method=:lp, degree=2, bw=:cv)

rmse(fit, xs) = round(sqrt(mean((fit .- sin.(xs)).^2)), digits=4)

(local_linear = rmse(kr.fitted, kr.x),
 local_constant = rmse(kr_nw.fitted, kr_nw.x),
 local_quadratic = rmse(kr_lp.fitted, kr_lp.x),
 lowess = rmse(lf.fitted, lf.x))
```

```julia
plot_result(kr)   # scatter of (x, y) with the fitted curve and SE band
```

The three kernel estimators land within ``0.015`` of each other in root mean squared error against the true ``\sin(x)`` — ``0.0960`` for local linear, ``0.0896`` for local constant, ``0.0813`` for local quadratic — while LOWESS at a ``0.3`` span is nearly three times worse at ``0.2468``. The span is doing the damage, not the method: ``f = 0.3`` puts 60 of 200 points in each window, far more smoothing than the cross-validated kernel bandwidths choose.

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

The **Sheather-Jones plug-in** (`bw=:sj`, matching R's `bw.SJ`) targets the AMISE-optimal bandwidth by solving a fixed-point equation for ``h`` that estimates the integrated squared density derivative from a pilot bandwidth. It is more accurate for multimodal or heavy-tailed samples at higher computational cost: the functionals are evaluated by exact pairwise summation, so cost grows as ``O(n^2)`` per bandwidth trial.

!!! note "Technical Note"
    The Sheather-Jones solve-the-equation method brackets the root on ``[0.1\,h_{\max}, h_{\max}]`` around the Silverman scale and locates it by bisection, widening the bracket up to 99 times if the initial interval does not straddle the root. On flat or degenerate samples the pilot functional can turn non-positive; the estimator then warns and falls back to the Silverman rule rather than erroring.

```@example np
kd_silverman = kernel_density(z)
kd_sj = kernel_density(z; kernel=:gaussian, bw=:sj, npoints=512)

(silverman = round(kd_silverman.bandwidth, digits=4),
 sheather_jones = round(kd_sj.bandwidth, digits=4),
 sj_peak_density = round(maximum(kd_sj.density), digits=4),
 sj_mode = round(kd_sj.x[argmax(kd_sj.density)], digits=4))
```

The Sheather-Jones bandwidth of ``0.0869`` is 27 percent tighter than Silverman's ``0.1193``, the expected direction for a right-skewed sample: the rule of thumb keys off a global scale measure and oversmooths when the density is asymmetric. The tighter bandwidth places the mode at ``0.8649`` with density ``1.1662``. For a lognormal with ``\sigma = 0.5`` the true mode is ``e^{-0.25} \approx 0.7788``, so even the plug-in bandwidth leaves visible mode bias — the standard warning that kernel density estimates flatten and shift peaks.

The density is evaluated on an equally-spaced grid of `npoints`, extending `cut·h` beyond the data range on each side.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `kernel` | `Symbol` | `:gaussian` | `:gaussian`, `:epanechnikov`, `:triangular`, `:uniform` |
| `bw` | `Symbol` or `Real` | `:silverman` | `:silverman` (`bw.nrd0`), `:sj` (`bw.SJ`), or a positive value |
| `npoints` | `Int` | `512` | Number of grid points |
| `cut` | `Real` | `3.0` | Grid extends `cut·h` beyond the data range |

**Return value (`KernelDensity{T}`):**

| Field | Type | Description |
|---|---|---|
| `x` | `Vector{T}` | Grid abscissae |
| `density` | `Vector{T}` | ``\hat f(x)`` on the grid |
| `bandwidth` | `T` | Chosen bandwidth ``h`` |
| `kernel` | `Symbol` | Kernel used |
| `bw_method` | `Symbol` | `:silverman`, `:sj`, or `:user` |
| `data` | `Vector{T}` | Original sample |
| `nobs` | `Int` | Number of observations |

---

## Kernel Regression

Nonparametric regression estimates the conditional mean ``m(x) = E[y \mid x]`` by weighted least squares in a shrinking neighbourhood of each target point. The **Nadaraya-Watson** (local-constant) estimator is a kernel-weighted average:

```math
\hat m_{\text{NW}}(x_0) = \frac{\sum_i K\!\left(\frac{x_i - x_0}{h}\right) y_i}{\sum_i K\!\left(\frac{x_i - x_0}{h}\right)}
```

The **local-polynomial** estimator (Fan & Gijbels 1996) fits, at each ``x_0``, a weighted regression of ``y`` on the local design ``[1, (x_i - x_0), \dots, (x_i - x_0)^p]`` with kernel weights, and reports the intercept as the fit:

```math
\hat m(x_0) = e_1' (X_0' W_0 X_0)^{-1} X_0' W_0 y
```

where:
- ``h`` is the bandwidth
- ``p`` is the local-polynomial degree (`:nw` ⇒ 0, `:ll` ⇒ 1, `:lp` ⇒ `degree`)
- ``W_0`` is the diagonal matrix of kernel weights at ``x_0``
- ``e_1`` selects the intercept of the local fit

The local-linear case (``p = 1``, `method=:ll`) carries automatic boundary-bias correction that Nadaraya-Watson lacks. When the local design is rank-deficient — too few points carry weight at a given ``x_0`` — the estimator silently falls back to the local-constant weights there.

The bandwidth is selected by **leave-one-out cross-validation** (`bw=:cv`), minimising ``\sum_i (y_i - \hat m_{-i}(x_i))^2`` over a 30-point grid spanning ``0.25`` to ``3`` times the Silverman scale of ``x``, or set to that scale directly by the rule of thumb (`bw=:rot`). Pointwise standard errors use the effective-weight sandwich form ``\operatorname{Var}(\hat m(x_0)) = \hat\sigma^2 \, \lVert \ell(x_0) \rVert^2``, where ``\hat m(x_0) = \sum_i \ell_i(x_0) y_i`` and ``\hat\sigma^2`` is the residual variance on effective degrees of freedom ``n - \operatorname{tr}(H)``.

```@example np
kr = kernel_reg(y, x; method=:ll, bw=:cv, kernel=:gaussian)
report(kr)
```

```julia
plot_result(kr)
```

Cross-validation selects ``h = 0.3675``, under 3 percent of the ``4\pi \approx 12.57`` design range — narrow enough to follow two full periods of ``\sin(x)`` without flattening the turning points. The residual variance of ``0.0823`` recovers the ``0.3^2 = 0.09`` noise variance used to generate the data, slightly low because the fit absorbs part of the noise. The average pointwise standard error is ``0.0660``, but it ranges from ``0.0529`` in the interior to ``0.2039`` at the boundary: with no data on one side, the local fit extrapolates and the band widens fourfold. Read the band, not just the curve.

```@example np
kr_rot = kernel_reg(y, x; method=:ll, bw=:rot)

(cv_bandwidth = round(kr.bandwidth, digits=4),
 rot_bandwidth = round(kr_rot.bandwidth, digits=4),
 cv_sigma2 = round(kr.sigma2, digits=4),
 rot_sigma2 = round(kr_rot.sigma2, digits=4))
```

The rule-of-thumb bandwidth is ``1.0656``, nearly three times the cross-validated value, and the cost is immediate: residual variance rises from ``0.0823`` to ``0.1530``, almost double the true noise level. A window that wide averages across a full half-period of the sine and flattens it. The rule of thumb is a *density* bandwidth applied to the design points, blind to the curvature of the conditional mean; cross-validation sees the curvature because it scores prediction error. Use `:rot` only as a starting value.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `method` | `Symbol` | `:ll` | `:nw` (local constant), `:ll` (local linear), `:lp` (local polynomial) |
| `degree` | `Int` | `1` | Local-polynomial degree; used only when `method=:lp` |
| `bw` | `Symbol` or `Real` | `:cv` | `:cv` (leave-one-out), `:rot` (rule of thumb), or a positive value |
| `kernel` | `Symbol` | `:gaussian` | Kernel for the local weights |

**Return value (`KernelRegression{T}`):**

| Field | Type | Description |
|---|---|---|
| `x` | `Vector{T}` | Sorted design points (the evaluation grid) |
| `fitted` | `Vector{T}` | ``\hat m(x)`` at those points |
| `se` | `Vector{T}` | Pointwise standard errors of ``\hat m`` |
| `xdata` | `Vector{T}` | Original ``x``, sorted |
| `ydata` | `Vector{T}` | Original ``y``, sorted by ``x`` |
| `bandwidth` | `T` | Chosen bandwidth ``h`` |
| `method` | `Symbol` | `:nw`, `:ll`, or `:lp` |
| `degree` | `Int` | Local-polynomial degree actually used |
| `kernel` | `Symbol` | Kernel used for the weights |
| `bw_method` | `Symbol` | `:cv`, `:rot`, or `:user` |
| `sigma2` | `T` | Residual variance ``\hat\sigma^2`` |
| `nobs` | `Int` | Number of observations |

---

## LOWESS

LOWESS (Cleveland 1979) is a robust scatterplot smoother. At each point it fits a local linear regression to the nearest ``\lfloor f n \rfloor`` neighbours weighted by the tricube function

```math
w(d) = \left(1 - (d/d_{\max})^3\right)^3, \qquad d \le d_{\max}
```

where ``d`` is the distance to the target and ``d_{\max}`` the distance to the farthest neighbour in the window. After the initial fit, `iter` robustifying passes reweight each observation by the bisquare of its scaled residual, so outliers are progressively discounted. The span ``f`` (fraction of points per window) controls smoothness.

!!! note "Technical Note"
    The implementation is a port of R's `clowess`/`lowest` routines and matches `stats::lowess` to machine precision, including the tricube weights, the local-linear slope correction, the near-boundary tolerances, and the `delta` interpolation skip. `delta` avoids a local fit at abscissae within `delta` of the last computed point and linearly interpolates instead; the default of ``0.01 \cdot \text{range}(x)`` makes long runs of near-duplicate ``x`` cheap.

```@example np
lf = lowess(y, x; f=0.3, iter=3)
report(lf)
```

```@example np
lf_wide = lowess(y, x)              # default span f = 2/3

window(fit) = floor(Int, fit.span * fit.nobs)

(narrow_window = window(lf),
 wide_window = window(lf_wide),
 narrow_rss = round(sum(abs2, lf.ydata .- lf.fitted), digits=2),
 wide_rss = round(sum(abs2, lf_wide.ydata .- lf_wide.fitted), digits=2))
```

```julia
plot_result(lf)
```

The ``f = 0.3`` span puts 60 of the 200 observations in each window and leaves a residual sum of squares of ``26.42``; the default ``f = 2/3`` span uses 133 and more than doubles it to ``60.56``. On a sine wave with two full periods, two thirds of the sample spans more than a full period, so the local linear fit averages peaks against troughs — the classic oversmoothing failure. Nothing about the default is wrong; it is tuned for scatterplots with a monotone or gently curving mean, and this one is neither. Set `f` from the wiggliness you expect, and increase `iter` only when outliers are the concern.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `f` | `Real` | `2/3` | Span: fraction of points in each local window |
| `iter` | `Int` | `3` | Number of bisquare robustifying passes |
| `delta` | `Real` or `nothing` | `nothing` | Interpolation skip (defaults to ``0.01 \cdot \text{range}(x)``) |

**Return value (`LowessFit{T}`):**

| Field | Type | Description |
|---|---|---|
| `x` | `Vector{T}` | Sorted ``x`` |
| `fitted` | `Vector{T}` | Smoothed ``\hat y``, sorted by ``x`` |
| `ydata` | `Vector{T}` | Original ``y``, sorted by ``x`` |
| `span` | `T` | Smoother span ``f`` |
| `iter` | `Int` | Number of robustifying iterations |
| `nobs` | `Int` | Number of observations |

---

## Complete Example

A full workflow: estimate the density of the response, fit both a local-linear regression and a robust LOWESS smoother, and compare what each recovers.

```@example np
# Density of the response variable with a Sheather-Jones bandwidth
kd = kernel_density(y; bw=:sj)
report(kd)
```

```@example np
# Local-linear regression, cross-validated bandwidth
kr = kernel_reg(y, x; method=:ll, bw=:cv)
report(kr)
```

```@example np
# Robust LOWESS with a moderate span, scored against the same truth
lf = lowess(y, x; f=0.3, iter=3)

(kernel_bandwidth = round(kr.bandwidth, digits=4),
 kernel_rmse = round(sqrt(mean((kr.fitted .- sin.(kr.x)).^2)), digits=4),
 lowess_span = round(lf.span, digits=4),
 lowess_rmse = round(sqrt(mean((lf.fitted .- sin.(lf.x)).^2)), digits=4))
```

The density of ``y`` peaks at ``0.8309`` rather than at zero, the signature of a bimodal marginal: a sine wave spends most of its time near its extremes, so the unconditional distribution of ``y`` piles up near ``\pm 1`` even though ``E[y]`` is close to zero. This is why the marginal density and the conditional mean answer different questions and neither substitutes for the other. On the regression itself, the cross-validated kernel fit reaches RMSE ``0.0960`` against the true ``\sin(x)``, versus ``0.2468`` for LOWESS at a ``0.3`` span — the kernel estimator wins because CV tunes its bandwidth to the curvature while the LOWESS span was fixed by hand.

```@example np
# Bibliography for the methods a fitted object actually used
refs(kr; format=:text)
```

`refs` prints the formatted bibliography to `stdout` and returns `nothing`, exactly as `report` does, so call it bare. Pass `format=:bibtex`, `:latex`, or `:html` for the other output styles, and `sprint(io -> refs(io, kr; format=:bibtex))` to capture the text as a `String`.

---

## Common Pitfalls

1. **Bandwidth dominates the estimate.** Too small a bandwidth produces a spiky, high-variance fit; too large oversmooths and washes out structure. Start from the data-driven default (`:silverman` for density, `:cv` for regression) and adjust deliberately.

2. **The regression rule of thumb is a density bandwidth.** `bw=:rot` applies the Silverman rule to the design points ``x`` and never looks at ``y``, so it is blind to the curvature of the conditional mean. On the sine example it selects a bandwidth nearly three times the cross-validated one and nearly doubles the residual variance. Use `:cv` for regression.

3. **Sheather-Jones on degenerate samples.** On flat or near-constant data the SJ pilot functional can turn non-positive, or the bracket can fail. The estimator warns and returns the Silverman bandwidth — read the warning rather than trusting the number blindly.

4. **Nadaraya-Watson boundary bias.** The local-constant estimator is biased near the edges of the support. Prefer `method=:ll` (local linear), which corrects this automatically. Even so, standard errors widen sharply at the boundary — fourfold in the example above.

5. **LOWESS span is a fraction, not a count.** `f` is the fraction of observations in each window (``0 < f \le 1``), not a bandwidth in ``x`` units. The default of ``2/3`` uses two-thirds of the sample per local fit, which oversmooths any mean that turns more than once.

6. **Results are sorted by `x`.** `kernel_reg` and `lowess` sort the data internally; the returned `x`, `fitted`, `ydata` are in ascending-``x`` order, not the input order. Do not zip them back against the unsorted response.

---

## References

- Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing Scatterplots.
  *Journal of the American Statistical Association*, 74(368), 829--836. [DOI](https://doi.org/10.1080/01621459.1979.10481038)

- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. London: Chapman & Hall. ISBN 978-0-412-98321-4. [DOI](https://doi.org/10.1201/9780203748725)

- Nadaraya, E. A. (1964). On Estimating Regression.
  *Theory of Probability and Its Applications*, 9(1), 141--142. [DOI](https://doi.org/10.1137/1109020)

- Sheather, S. J., & Jones, M. C. (1991). A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation.
  *Journal of the Royal Statistical Society, Series B*, 53(3), 683--690. [DOI](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x)

- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. London: Chapman & Hall. ISBN 978-0-412-24620-3. [DOI](https://doi.org/10.1201/9781315140919)

- Watson, G. S. (1964). Smooth Regression Analysis.
  *Sankhyā: The Indian Journal of Statistics, Series A*, 26(4), 359--372. (No DOI assigned.)
