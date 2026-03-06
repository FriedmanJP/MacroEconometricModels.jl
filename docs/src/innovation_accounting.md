# [Innovation Accounting](@id innovation_accounting_page)

Innovation accounting decomposes the dynamics of a structural VAR into the contributions of individual structural shocks. Starting from the reduced-form residual decomposition ``u_t = B_0 \varepsilon_t``, where ``B_0`` is the structural impact matrix and ``\varepsilon_t`` are orthogonal structural shocks, the package provides three complementary tools:

- **Impulse Response Functions (IRF)**: trace the dynamic effect of a one-unit structural shock on each endogenous variable across horizons; see [Impulse Responses](@ref ia_irf_page)
- **Forecast Error Variance Decomposition (FEVD)**: measure the share of each variable's forecast uncertainty attributable to each structural shock; see [Variance Decomposition](@ref ia_fevd_page)
- **Historical Decomposition (HD)**: attribute observed variable movements to individual structural shocks over the sample period; see [Historical Decomposition](@ref ia_hd_page)

All three tools support frequentist VAR, Bayesian VAR, VECM, FAVAR, DSGE, and Local Projection estimation, with six structural identification schemes and interactive D3.js visualization via `plot_result()`.

## Quick Start

**Recipe 1: Cholesky IRF with bootstrap confidence intervals**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Recursive identification: INDPRO -> CPIAUCSL -> FEDFUNDS
irfs = irf(model, 20; ci_type=:bootstrap, reps=500)
report(irfs)
```

**Recipe 2: Forecast error variance decomposition**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

decomp = fevd(model, 20)
report(decomp)
```

**Recipe 3: Historical decomposition with verification**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

hd = historical_decomposition(model, size(model.U, 1))
verify_decomposition(hd)
report(hd)
```

**Recipe 4: Bayesian IRF with credible intervals**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
post = estimate_bvar(Y, 2; n_draws=1000)

# Posterior median IRF with 68% credible intervals
birfs = irf(post, 20)
report(birfs)
```

**Recipe 5: Sign-restricted IRF**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Demand shock: positive output, positive prices; supply shock: positive output, negative prices
irfs = irf(model, 20; method=:sign, sign_restrictions=[1 1 0; -1 0 0; 0 0 1])
report(irfs)
```

**Recipe 6: Structural LP impulse responses**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# LP-based IRF robust to VAR misspecification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
lp_irfs = irf(slp)
report(lp_irfs)
```

---

## Structural Identification Overview

Innovation accounting requires choosing an identification scheme to recover ``B_0`` from the reduced-form covariance ``\Sigma = B_0 B_0'``. The package implements six methods spanning point-identified and set-identified approaches:

| Identification Method | Function | Point/Set ID | Key Feature |
|---|---|---|---|
| Cholesky | `irf(model, H)` | Point | Recursive ordering |
| Sign restrictions | `irf(model, H; method=:sign, ...)` | Set | Agnostic about magnitudes |
| Narrative | `identify_narrative(model, H, ...)` | Set | Incorporates historical events |
| Long-run | `irf(model, H; method=:long_run)` | Point | Blanchard-Quah decomposition |
| Arias et al. | `identify_arias(model, ...)` | Set | Zero + sign restrictions |
| Uhlig penalty | `identify_uhlig(model, ...)` | Point | Penalty function approach |

**Point identification** (Cholesky, long-run, Uhlig) produces a unique ``B_0`` and hence unique IRFs. **Set identification** (sign, narrative, Arias et al.) produces a set of admissible ``B_0`` matrices; the reported IRFs are the median across the admissible set, with the range reflected in wider confidence/credible bands.

All six methods integrate seamlessly with `irf()`, `fevd()`, and `historical_decomposition()` via the `method` keyword or by passing a pre-identified rotation matrix. For statistical identification via heteroskedasticity or non-Gaussianity (18 additional methods), see [Statistical Identification](@ref nongaussian_page).

---

## Sub-Page Guide

For detailed treatment of each tool --- theory, equations, return value tables, and advanced usage:

- [Impulse Responses](@ref ia_irf_page) --- IRF definition, companion form representation, cumulative IRFs, bootstrap and Bayesian confidence intervals, stationarity filtering (Kilian 1998), LP-based IRFs
- [Variance Decomposition](@ref ia_fevd_page) --- FEVD definition, properties, LP-FEVD (Gorodnichenko & Lee 2019), Bayesian FEVD, bootstrap CIs
- [Historical Decomposition](@ref ia_hd_page) --- HD definition, decomposition identity, shock contributions, LP-based HD, display and table output

---

## Common Pitfalls

1. **Variable ordering matters for Cholesky.** The default `irf(model, H)` uses Cholesky identification, where the column ordering of the data matrix determines the recursive causal structure. Placing the federal funds rate last assumes monetary policy does not contemporaneously affect output or prices.

2. **Confidence bands require explicit activation.** The `ci_lower` and `ci_upper` fields contain zeros unless `ci_type=:bootstrap` is set (frequentist) or a Bayesian posterior is passed. Always check `irfs.ci_type` before interpreting bands.

3. **Sign restrictions produce set-identified IRFs.** The median response across admissible rotations is a summary statistic, not a point estimate. Report the full credible set, not just the median, to avoid overstating precision (Uhlig 2005).

4. **HD verification should always pass.** After computing `hd = historical_decomposition(model, T)`, call `verify_decomposition(hd)` to confirm the additive identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` holds to numerical precision. A failure indicates a bug, not a data issue.

5. **LP-based IRFs are wider than VAR-based IRFs.** Each horizon is estimated independently without cross-horizon restrictions, producing larger standard errors. This is a feature (robustness to dynamic misspecification), not a deficiency (Kilian and Lütkepohl 2017, Chapter 12).

---

## References

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions. *Econometrica*, 86(2), 685--720. [DOI: 10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions. *Review of Economics and Statistics*, 80(2), 218--230. [DOI: 10.1162/003465398557465](https://doi.org/10.1162/003465398557465)
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press. [DOI: 10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. ISBN 978-3-540-40172-8.
- Uhlig, H. (2005). What are the effects of monetary policy on output? *Journal of Monetary Economics*, 52(2), 381--419. [DOI: 10.1016/j.jmoneco.2004.05.007](https://doi.org/10.1016/j.jmoneco.2004.05.007)
