# [Innovation Accounting](@id innovation_accounting_page)

Innovation accounting decomposes the dynamics of a structural VAR into the contributions of individual structural shocks. Starting from the reduced-form residual decomposition ``u_t = B_0 \varepsilon_t``, where ``B_0`` is the structural impact matrix and ``\varepsilon_t`` are orthogonal structural shocks, the package provides three complementary tools: impulse responses trace the dynamic effect of a shock, variance decompositions measure how much forecast uncertainty each shock explains, and historical decompositions attribute realized movements in the data to specific shocks. One child page owns each tool.

All three support frequentist VAR, Bayesian VAR, VECM, FAVAR, DSGE, and Local Projection estimation, with six structural identification schemes and interactive D3.js visualization via `plot_result()`.

```@setup ia
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
model = estimate_var(Y, 2; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

Identify an estimated VAR recursively (INDPRO → CPIAUCSL → FEDFUNDS) and trace the impulse responses with bootstrap confidence intervals:

```@example ia
irfs = irf(model, 20; ci_type=:bootstrap, reps=50)
report(irfs)
```

---

## Choosing a Method

The three tools answer different questions about the same identified system:

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Dynamic effect of a shock over time | [Impulse Responses](@ref ia_irf_page) | Traces the moving-average representation |
| Share of forecast uncertainty per shock | [Variance Decomposition](@ref ia_fevd_page) | Normalized forecast-error MSE contributions |
| Which shocks drove a specific episode | [Historical Decomposition](@ref ia_hd_page) | Attributes realized data to shocks |
| Robustness to dynamic misspecification | [Impulse Responses](@ref ia_irf_page), `structural_lp` | Each horizon estimated separately |
| Uncertainty bands on variance shares | [Variance Decomposition](@ref ia_fevd_page), Bayesian or LP | Frequentist FEVD is a point estimate |
| Sign, narrative, or long-run restrictions | [Structural Identification](@ref structural_identification_page) | Fixes ``B_0`` before any tool runs |
| Identification from higher moments | [Statistical Identification](@ref nongaussian_page) | No economic restrictions required |

---

## Child Pages

- [Impulse Responses](@ref ia_irf_page) --- IRF definition, companion form representation, cumulative IRFs, bootstrap and Bayesian confidence intervals, stationarity filtering (Kilian & Lütkepohl 2017), LP-based IRFs
- [Variance Decomposition](@ref ia_fevd_page) --- FEVD definition, properties, generalized FEVD, LP-FEVD (Gorodnichenko & Lee 2019), Bayesian FEVD, bootstrap CIs
- [Historical Decomposition](@ref ia_hd_page) --- HD definition, decomposition identity, shock contributions, Bayesian HD, display and table output

---

## Structural Identification Overview

Innovation accounting requires choosing an identification scheme to recover ``B_0`` from the reduced-form covariance ``\Sigma = B_0 B_0'``. The package implements six methods spanning point-identified and set-identified approaches --- Cholesky (recursive), sign restrictions, narrative restrictions, long-run (Blanchard-Quah), Arias et al. (zero + sign), and Uhlig penalty. These schemes, together with their functions and keyword arguments, are documented in full on the [Structural Identification](@ref structural_identification_page) page.

**Point identification** (Cholesky, long-run, Uhlig) produces a unique ``B_0`` and hence unique IRFs. **Set identification** (sign, narrative, Arias et al.) produces a set of admissible ``B_0`` matrices; the reported IRFs are the median across the admissible set, with the range reflected in wider confidence/credible bands.

All six methods integrate seamlessly with `irf()`, `fevd()`, and `historical_decomposition()` via the `method` keyword or by passing a pre-identified rotation matrix. For statistical identification via heteroskedasticity or non-Gaussianity (13 additional methods: 5 ICA + 4 ML + 4 heteroskedasticity), see [Statistical Identification](@ref nongaussian_page).

---

## Common Pitfalls

1. **Confidence bands require explicit activation.** The `ci_lower` and `ci_upper` fields contain zeros unless `ci_type=:bootstrap` is set (frequentist) or a Bayesian posterior is passed. Always check `irfs.ci_type` before interpreting bands.

2. **Sign restrictions produce set-identified results.** The median response across admissible rotations is a summary statistic, not a point estimate. Report the full credible set, not just the median, to avoid overstating precision (Uhlig 2005).

3. **HD verification should always pass.** The additive identity ``y_t = \sum_j \text{HD}_j(t) + \text{initial}(t)`` holds by construction — the initial-conditions component is defined as the residual — so `verify_decomposition(hd)` can fail only on non-finite values. Treat it as a sanity check for numerical corruption, not as evidence the decomposition is economically valid; see [Historical Decomposition](@ref ia_hd_page) for what the identity does and does not guarantee.

4. **LP-based results are wider than VAR-based results.** Each horizon is estimated independently without cross-horizon restrictions, producing larger standard errors. This is a feature (robustness to dynamic misspecification), not a deficiency (Kilian and Lütkepohl 2017, Chapter 12).

---

## References

- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions. *Econometrica*, 86(2), 685--720. [DOI: 10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions. *Review of Economics and Statistics*, 80(2), 218--230. [DOI: 10.1162/003465398557465](https://doi.org/10.1162/003465398557465)
- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press. [DOI: 10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. ISBN 978-3-540-40172-8.
- Uhlig, H. (2005). What are the effects of monetary policy on output? *Journal of Monetary Economics*, 52(2), 381--419. [DOI: 10.1016/j.jmoneco.2004.05.007](https://doi.org/10.1016/j.jmoneco.2004.05.007)
