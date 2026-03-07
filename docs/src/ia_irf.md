# [Impulse Responses](@id ia_irf_page)

The **impulse response function** (IRF) traces the dynamic effect of a one-unit structural shock on each endogenous variable over time. MacroEconometricModels.jl computes IRFs from VAR, BVAR, and Local Projection models with bootstrap confidence intervals, Bayesian credible bands, cumulation for growth-rate variables, and stationarity-filtered inference.

```@setup ia_irf
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
model = estimate_var(Y, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post = estimate_bvar(Y, 4; n_draws=100, varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
```

## Quick Start

**Recipe 1: Basic Cholesky IRF**

```@example ia_irf
# Cholesky IRF with recursive ordering
result = irf(model, 20)
report(result)
```

**Recipe 2: IRF with bootstrap confidence intervals**

```@example ia_irf
# Residual bootstrap with 95% pointwise intervals
result = irf(model, 20; ci_type=:bootstrap, reps=50, conf_level=0.95)
report(result)
```

**Recipe 3: Cumulative IRF**

```@example ia_irf
# Cumulate IRF for growth-rate variables (level effect)
result = irf(model, 20; ci_type=:bootstrap, reps=50)
cum_result = cumulative_irf(result)
report(cum_result)
```

**Recipe 4: Bayesian IRF (BVAR)**

```@example ia_irf
# Bayesian IRF with 68% credible bands
result = irf(post, 20)
report(result)
```

**Recipe 5: Sign-restricted IRF**

```@example ia_irf
# Demand shock: positive output and positive prices on impact
check_demand = irf_array -> irf_array[1, 1, 1] > 0 && irf_array[1, 2, 1] > 0
result = irf(model, 20; method=:sign, check_func=check_demand)
report(result)
```

**Recipe 6: Structural LP IRF**

```@example ia_irf
# Structural LP: Cholesky identification + LP estimation
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
result = irf(slp)
report(result)
```

---

## Frequentist IRF

The impulse response function ``\Theta_h`` measures the dynamic causal effect of a one-unit structural shock at time ``t`` on the endogenous variables at time ``t+h``. Under a recursive (Cholesky) identification, the ordering of variables determines the contemporaneous causal structure.

```math
\Theta_h = \frac{\partial y_{t+h}}{\partial \varepsilon_t'}
```

where:
- ``\Theta_h`` is the ``n \times n`` impulse response matrix at horizon ``h``
- ``y_{t+h}`` is the ``n \times 1`` vector of endogenous variables at time ``t+h``
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks at time ``t``

For a VAR(p) model, the IRF at horizon ``h`` is computed recursively from the reduced-form MA coefficients:

```math
\Theta_h = \Phi_h \cdot B_0, \qquad \Phi_h = \sum_{i=1}^{\min(h,p)} A_i \, \Phi_{h-i}, \qquad \Phi_0 = I_n
```

where:
- ``A_i`` are the ``n \times n`` VAR coefficient matrices for lag ``i``
- ``\Phi_h`` are the reduced-form MA coefficients at horizon ``h``
- ``B_0 = L \cdot Q`` is the ``n \times n`` structural impact matrix
- ``L`` is the lower-triangular Cholesky factor of ``\Sigma``
- ``Q`` is the ``n \times n`` orthogonal rotation matrix (``Q = I_n`` for Cholesky)

Equivalently, the companion form representation computes the IRF as:

```math
\Theta_h = J \, F^h \, J' \, B_0
```

where:
- ``J = [I_n, 0, \ldots, 0]`` is the ``n \times np`` selection matrix
- ``F`` is the ``np \times np`` companion matrix

!!! note "Technical Note"
    The `ci_lower` and `ci_upper` arrays are populated only when `ci_type=:bootstrap` or `ci_type=:theoretical`. With `ci_type=:none` (the default), these arrays contain zeros. Always check `result.ci_type` before interpreting confidence bands.

The following example estimates a three-variable monetary policy VAR from FRED-MD data and computes Cholesky-identified IRFs with bootstrap confidence intervals (Kilian 1998).

```@example ia_irf
# Cholesky IRF with bootstrap 95% confidence intervals
H = 20
result = irf(model, H; ci_type=:bootstrap, reps=50, conf_level=0.95)
report(result)
```

```julia
plot_result(result)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The recursive ordering INDPRO, CPIAUCSL, FEDFUNDS implies that monetary policy shocks (federal funds rate innovations) do not contemporaneously affect industrial production or prices. The bootstrap confidence bands quantify estimation uncertainty: at impact (``h = 0``), the federal funds rate shock has zero effect on INDPRO by construction. By ``h = 8`` quarters, the contractionary transmission is visible as the INDPRO response becomes negative, consistent with standard monetary transmission mechanisms.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:narrative`, `:long_run`, `:fastica`, etc.) |
| `ci_type` | `Symbol` | `:none` | CI method: `:none`, `:bootstrap`, or `:theoretical` |
| `reps` | `Int` | `200` | Number of bootstrap or simulation replications |
| `conf_level` | `Real` | `0.95` | Confidence level for interval construction |
| `stationary_only` | `Bool` | `false` | Reject explosive bootstrap draws (Kilian 1998) |
| `check_func` | `Function` | `nothing` | Sign restriction check function (required for `:sign`) |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check (required for `:narrative`) |

### `ImpulseResponse` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Array{T,3}` | ``H \times n \times n`` IRF array: `values[h, i, j]` = response of variable ``i`` to shock ``j`` at horizon ``h-1`` |
| `ci_lower` | `Array{T,3}` | Lower confidence bound (same shape as `values`) |
| `ci_upper` | `Array{T,3}` | Upper confidence bound |
| `horizon` | `Int` | Maximum IRF horizon ``H`` |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `ci_type` | `Symbol` | CI method used (`:bootstrap`, `:theoretical`, `:none`) |

---

## Cumulative IRF

For variables measured in growth rates (e.g., log-differenced GDP or industrial production), the cumulative IRF recovers the effect on the level. The cumulative response through horizon ``H`` is:

```math
\Theta^{\text{cum}}_H = \sum_{h=0}^{H} \Theta_h
```

where:
- ``\Theta^{\text{cum}}_H`` is the ``n \times n`` cumulated response matrix at horizon ``H``
- ``\Theta_h`` is the pointwise IRF at horizon ``h``

!!! note "Cumulative IRF Confidence Intervals"
    For bootstrap or Bayesian CIs, the cumulative sum is computed *per draw* before extracting quantiles. This produces correct coverage. The naive approach of cumulating the pointwise median underestimates uncertainty because quantiles are not additive: ``Q_\alpha(\sum_h \Theta_h) \neq \sum_h Q_\alpha(\Theta_h)``.

The `cumulative_irf` function accepts `ImpulseResponse`, `BayesianImpulseResponse`, or `LPImpulseResponse` objects. When raw bootstrap draws are available (from a prior `irf()` call with `ci_type=:bootstrap`), the function cumulates each draw before extracting quantiles.

```@example ia_irf
# Pointwise IRF with bootstrap draws
result_ci = irf(model, 20; ci_type=:bootstrap, reps=50)

# Cumulate: each bootstrap draw is summed before extracting quantiles
cum_result = cumulative_irf(result_ci)
report(cum_result)
```

The cumulative response of INDPRO to a federal funds rate shock shows the total level effect of a monetary tightening. Since INDPRO enters the VAR in growth rates (transformation code 5 in FRED-MD), the cumulative IRF translates the growth-rate response into the implied level response. A persistently negative cumulative response indicates that a one-time monetary contraction permanently reduces the level of industrial production.

---

## Bayesian IRF

Bayesian IRFs replace bootstrap confidence intervals with posterior credible bands derived from the BVAR posterior distribution. For each posterior draw of ``(B^{(d)}, \Sigma^{(d)})``, the algorithm computes the full IRF and then reports posterior quantiles across draws.

```math
\Theta_h^{(d)} = \Phi_h^{(d)} \cdot B_0^{(d)}, \qquad d = 1, \ldots, D
```

where:
- ``\Theta_h^{(d)}`` is the IRF at horizon ``h`` for posterior draw ``d``
- ``D`` is the number of posterior draws
- ``B_0^{(d)} = L^{(d)} Q`` with ``L^{(d)}`` the Cholesky factor of ``\Sigma^{(d)}``

The returned quantiles (default: 16th, 50th, 84th percentiles) form 68% pointwise credible bands. The point estimate is the posterior mean by default; pass `point_estimate=:median` for the posterior median.

```@example ia_irf
# Bayesian IRFs with default 68% credible bands
bayes_result = irf(post, 20)
report(bayes_result)
```

```julia
plot_result(bayes_result)
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The posterior credible bands reflect both estimation uncertainty and prior information. The Minnesota prior shrinks VAR coefficients toward a random walk, which stabilizes IRFs at longer horizons. The 68% bands (16th--84th percentiles) are the standard reporting convention in BVAR applications. Wider 90% bands are obtained by passing `quantiles=[0.05, 0.5, 0.95]`.

### Bayesian IRF Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency: `:mean` or `:median` |

### `BayesianImpulseResponse` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times q``: dimension 4 indexes quantile levels |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |

---

## LP-Based IRF

Local Projections (Jordà 2005) estimate the impulse response directly via horizon-specific regressions, without imposing the dynamic restrictions of a VAR. Structural LP (Plagborg-Møller & Wolf 2021) combines VAR-based identification with LP estimation: structural shocks are recovered from a VAR, then used as regressors in LP regressions at each horizon.

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, \hat{\varepsilon}_{j,t} + \Gamma_{i,h}' \, w_t + u_{i,t+h}, \qquad h = 0, 1, \ldots, H
```

where:
- ``y_{i,t+h}`` is variable ``i`` at horizon ``h``
- ``\hat{\varepsilon}_{j,t}`` is the identified structural shock ``j``
- ``\beta_{i,h}`` is the LP-estimated impulse response at horizon ``h``
- ``w_t`` contains lags of ``y_t`` as controls
- Standard errors use Newey-West HAC to account for serial correlation in ``u_{i,t+h}``

LP and VAR produce identical IRFs under correct specification (Plagborg-Møller & Wolf 2021). LP is more robust to dynamic misspecification but less efficient, producing wider confidence bands.

**Standard LP IRF**

```@example ia_irf
# LP-IRF: response of all variables to a FFR shock (variable 3)
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west)
lp_result = lp_irf(lp; conf_level=0.95)
report(lp_result)
```

```julia
plot_result(lp_result)
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Structural LP IRF**

```@example ia_irf
# Structural LP with Cholesky identification
slp2 = structural_lp(Y, 20; method=:cholesky, lags=4)
slp_result = irf(slp2)
report(slp_result)
```

```julia
plot_result(slp2)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The structural LP produces the full ``n \times n`` IRF matrix (responses of all variables to all shocks), whereas the standard `estimate_lp` traces responses to a single reduced-form shock. The LP-based confidence bands are wider than their VAR counterparts because each horizon is estimated independently without cross-horizon restrictions. This width reflects LP's agnosticism about dynamic propagation: the cost of robustness is reduced precision.

For full LP estimation details, including LP-IV, smooth LP, and state-dependent LP, see [Local Projections](@ref lp_page).

---

## Stationarity Filtering

The residual bootstrap of Kilian (1998) can produce explosive bootstrap draws when the companion matrix has eigenvalues near the unit circle. Setting `stationary_only=true` rejects any draw whose companion matrix has ``|\lambda_{\max}| \geq 1`` and redraws, ensuring all bootstrap IRFs come from stationary parameter configurations.

!!! note "Technical Note"
    The algorithm attempts up to ``10 \times \text{reps}`` draws to collect the requested number of stationary replications. If fewer than `reps` stationary draws are obtained, a warning is emitted. In practice, well-specified models with moderate sample sizes produce rejection rates below 10%.

```@example ia_irf
# Bootstrap with stationarity filtering
result_stat = irf(model, 20; ci_type=:bootstrap, reps=50,
                  conf_level=0.95, stationary_only=true)
report(result_stat)
```

Stationarity filtering removes the heavy tails in bootstrap IRF distributions caused by near-unit-root draws. The filtered confidence bands are typically narrower and more symmetric, reflecting the prior belief that the data-generating process is covariance-stationary. Kilian (1998) demonstrates that this improves finite-sample coverage of bootstrap confidence intervals in monetary VARs.

---

## Complete Example

This example combines frequentist, Bayesian, and LP-based IRFs for a three-variable monetary policy VAR using FRED-MD data.

```@example ia_irf
# ── Frequentist VAR IRF ──────────────────────────────────────────────
freq_irf = irf(model, 20; ci_type=:bootstrap, reps=50, conf_level=0.95)
report(freq_irf)

# Cumulative IRF for growth-rate variables
cum_irf = cumulative_irf(freq_irf)

# Selected horizons via table()
print_table(stdout, freq_irf, "INDPRO", "Monetary policy"; horizons=[1, 4, 8, 12, 20])
```

```@example ia_irf
# ── Bayesian BVAR IRF ────────────────────────────────────────────────
bayes_irf = irf(post, 20)
report(bayes_irf)
```

```@example ia_irf
# ── Structural LP IRF ────────────────────────────────────────────────
slp_full = structural_lp(Y, 20; method=:cholesky, lags=4)
lp_full = irf(slp_full)
report(lp_full)
```

The three approaches produce qualitatively similar IRFs under correct specification, but differ in their uncertainty quantification. The bootstrap CI on the frequentist IRF reflects sampling variability in the OLS estimates. The Bayesian credible bands incorporate prior information from the Minnesota prior, which stabilizes long-horizon responses. The structural LP bands are the widest because LP does not impose the VAR's cross-horizon coefficient restrictions. Comparing all three provides a robustness check: if the qualitative shape and sign of the response are consistent across methods, the structural conclusion is credible.

---

## Common Pitfalls

1. **Explosive bootstrap draws.** Near-unit-root VARs produce bootstrap draws with ``|\lambda_{\max}| > 1``, causing IRF confidence bands to diverge at long horizons. Use `stationary_only=true` to filter these draws. This is the default recommendation in Kilian & Lütkepohl (2017, Chapter 12).

2. **Cumulating pointwise quantiles.** Never cumulate the upper and lower CI bounds directly. The correct approach is to cumulate each bootstrap draw and then extract quantiles. The `cumulative_irf()` function handles this automatically when raw draws are available.

3. **Interpreting Cholesky ordering.** The recursive identification assigns economic meaning based on variable ordering. Placing the federal funds rate last assumes monetary policy does not contemporaneously affect output or prices. Reversing the ordering changes the identified shocks entirely.

4. **Confusing array indexing.** The IRF array uses `values[h, i, j]` where the first index corresponds to horizon ``h-1`` (i.e., `values[1, :, :]` is the impact response at ``h=0``). This 1-based indexing matches Julia's convention.

5. **Insufficient bootstrap replications.** The default `reps=200` is adequate for point estimates but not for tight confidence bands. Use `reps=1000` or higher for publication-quality inference. Sign-restricted IRFs require even more draws because accepted rotations are a subset of all draws.

---

## References

- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
  *American Economic Review*, 95(1), 161--182. [DOI](https://doi.org/10.1257/0002828053828518)

- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions."
  *Review of Economics and Statistics*, 80(2), 218--230. [DOI](https://doi.org/10.1162/003465398557465)

- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*.
  Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8.

- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses."
  *Econometrica*, 89(2), 955--980. [DOI](https://doi.org/10.3982/ECTA17813)
