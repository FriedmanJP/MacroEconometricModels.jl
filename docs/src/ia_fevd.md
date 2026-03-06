# [Variance Decomposition](@id ia_fevd_page)

Forecast Error Variance Decomposition (FEVD) quantifies the proportion of each variable's forecast error variance attributable to each structural shock at a given horizon. FEVD answers the question "which shocks matter most for which variables?" and is a cornerstone of structural VAR analysis alongside impulse response functions and historical decomposition.

- **Frequentist FEVD**: VMA-based decomposition with optional bootstrap confidence intervals
- **Bayesian FEVD**: Posterior distributions over variance shares with credible intervals
- **LP-Based FEVD**: R²-based estimator robust to VAR dynamic misspecification (Gorodnichenko & Lee 2019)

## Quick Start

**Recipe 1: Basic FEVD with Cholesky identification**

```julia
using MacroEconometricModels

# Load FRED-MD: industrial production, CPI inflation, federal funds rate
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# FEVD at horizon 20 with Cholesky identification
decomp = fevd(model, 20)
report(decomp)
```

**Recipe 2: FEVD with bootstrap confidence intervals**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Bootstrap CI via IRF bootstrap (Kilian 1998)
irfs_ci = irf(model, 20; ci_type=:bootstrap, reps=500)
report(irfs_ci)
```

**Recipe 3: Bayesian FEVD with credible intervals**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
bfevd = fevd(post, 20; method=:cholesky)
report(bfevd)
```

**Recipe 4: LP-based FEVD with bias correction**

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

slp = structural_lp(Y, 20; method=:cholesky, lags=4)
lp_decomp = fevd(slp, 20; bias_correct=true, n_boot=500)
report(lp_decomp)
```

**Recipe 5: FEVD table output at selected horizons**

```julia
using MacroEconometricModels

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

decomp = fevd(model, 20)
print_table(stdout, decomp, "INDPRO"; horizons=[1, 4, 8, 12, 20])
```

---

## Frequentist FEVD

The FEVD measures the proportion of the ``h``-step ahead forecast error variance of variable ``i`` attributable to structural shock ``j``. It derives from the Vector Moving Average (VMA) representation of the structural VAR (Lutkepohl 2005, Section 2.3.3).

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n} (\Theta_s)_{ik}^2}
```

where:
- ``\text{FEVD}_{ij}(h)`` is the share of variable ``i``'s ``h``-step forecast error variance due to shock ``j``
- ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the structural impulse response matrix at horizon ``s``
- ``\Theta_s = \Phi_s P`` are the structural MA coefficients, with ``\Phi_s`` the reduced-form MA coefficients and ``P`` the impact matrix
- The numerator sums the squared contributions of shock ``j`` through horizon ``h-1``
- The denominator sums contributions from all ``n`` shocks, ensuring ``\sum_j \text{FEVD}_{ij}(h) = 1``

### Properties

The FEVD satisfies three fundamental properties:

1. **Boundedness**: ``0 \leq \text{FEVD}_{ij}(h) \leq 1`` for all ``i, j, h``
2. **Row-sum unity**: ``\sum_{j=1}^{n} \text{FEVD}_{ij}(h) = 1`` for all ``i, h`` — the variance shares exhaust the total forecast error variance
3. **Convergence**: As ``h \to \infty``, the FEVD converges to the unconditional variance decomposition, revealing the dominant long-run drivers of each variable's fluctuations

At short horizons, own shocks typically dominate (large diagonal entries in the FEVD matrix). As the horizon increases, transmission mechanisms allow other shocks to explain a growing share of the forecast error variance.

### Code Example

```julia
using MacroEconometricModels

# Load FRED-MD: monetary policy VAR
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
model = estimate_var(Y, 2)

# Cholesky FEVD: ordering implies INDPRO → CPI → FFR
decomp = fevd(model, 20)
report(decomp)

# Inspect variance shares at horizon 1 and 20 for INDPRO
h1_shares = decomp.proportions[1, :, 1]   # horizon 1: shock contributions to var 1
h20_shares = decomp.proportions[1, :, 20]  # horizon 20

# Tabular output at selected horizons
print_table(stdout, decomp, 1; horizons=[1, 4, 8, 12, 20])
```

The Cholesky ordering INDPRO, CPIAUCSL, FEDFUNDS implies that monetary policy shocks (FFR) do not contemporaneously affect output or prices. At horizon 1, the INDPRO own shock dominates by construction. By horizon 20, the FEVD reveals the relative importance of supply, demand, and monetary shocks in driving industrial production forecast uncertainty.

@raw html
<iframe src="../assets/plots/fevd_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>

### FEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `decomposition` | `Array{T,3}` | ``n \times n \times H`` raw cumulative variance contributions |
| `proportions` | `Array{T,3}` | ``n \times n \times H`` proportion of FEV: `proportions[i, j, h]` = share of variable ``i``'s FEV due to shock ``j`` at horizon ``h`` |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:long_run`, `:narrative`, ICA/ML variants) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |

---

## Bayesian FEVD

Bayesian FEVD integrates over parameter uncertainty by computing variance shares for each posterior draw and reporting posterior quantiles (Kilian & Lutkepohl 2017, Chapter 12). This produces credible intervals that reflect both estimation uncertainty and identification uncertainty.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Bayesian VAR with Minnesota prior
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])

# Bayesian FEVD with 68% credible intervals
bfevd = fevd(post, 20; method=:cholesky, quantiles=[0.16, 0.5, 0.84])
report(bfevd)

# Access posterior median FEVD for variable 1 at horizon 8
median_share = bfevd.point_estimate[8, 1, :]
```

The Bayesian FEVD computes variance shares for each accepted posterior draw, discarding non-stationary draws. The `point_estimate` array contains the posterior mean (or median) FEVD, while the `quantiles` array stores the full posterior distribution at each horizon. Wide credible bands indicate that the data are not sufficiently informative to pin down the relative importance of specific shocks.

@raw html
<iframe src="../assets/plots/fevd_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>

### BayesianFEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times n_q``: dimension 4 indexes quantile levels (e.g., 16th, 50th, 84th percentile) |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate FEVD proportions |
| `horizon` | `Int` | Maximum FEVD horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `threaded` | `Bool` | `false` | Enable threaded quantile computation for large systems |

---

## LP-Based FEVD

The LP-based FEVD of Gorodnichenko & Lee (2019) estimates variance shares directly via R² regressions rather than inverting the VAR lag polynomial. This approach is robust to dynamic misspecification: even if the VAR lag order is wrong, the LP-FEVD consistently estimates the true variance shares.

### R² Estimator

At each horizon ``h``, the LP-FEVD regresses the LP forecast error ``\hat{f}_{t+h|t-1}`` on structural shock leads ``[z_{t+h}, z_{t+h-1}, \ldots, z_t]``:

```math
\hat{s}_{ij}(h) = R^2 \left( \hat{f}_{i,t+h|t-1} \sim z_{j,t+h}, z_{j,t+h-1}, \ldots, z_{j,t} \right)
```

where:
- ``\hat{f}_{i,t+h|t-1}`` is the LP forecast error for variable ``i`` at horizon ``h``
- ``z_{j,t}`` is the identified structural shock ``j`` at time ``t``
- ``R^2`` measures the fraction of forecast error variance explained by shock ``j``

!!! note "Technical Note"
    The package implements three LP-FEVD estimators: `:r2` (baseline R² regression), `:lp_a` (uses LP-IRF coefficients directly), and `:lp_b` (hybrid with residual variance in the denominator). All three are consistent; the R² estimator is the default and performs best in finite samples (Gorodnichenko & Lee 2019, Section 3).

### Bias Correction

The raw R² estimator has a finite-sample upward bias. The package applies the VAR-based bootstrap bias correction of Gorodnichenko & Lee (2019, Section 3.4):

1. Fit a bivariate VAR on (shock, response) with HQIC lag selection
2. Compute the "true" FEVD from the fitted VAR
3. Simulate ``B`` bootstrap samples from the VAR, compute LP-FEVD for each
4. Bias = mean(bootstrap LP-FEVD) - true VAR-FEVD
5. Bias-corrected LP-FEVD = raw LP-FEVD - bias

### Code Example

```julia
using MacroEconometricModels, Random
Random.seed!(42)

fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]

# Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# LP-FEVD with bias correction and bootstrap CIs
lp_decomp = fevd(slp, 20; method=:r2, bias_correct=true,
                 n_boot=500, conf_level=0.95)
report(lp_decomp)
```

The LP-FEVD produces variance shares that are numerically close to VAR-based FEVD under correct specification, but the LP estimates have wider confidence intervals because each horizon is estimated independently. The bias-corrected shares are bounded to ``[0, 1]`` and clamped to ensure non-negativity.

@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>

For full details on structural LP estimation, see [Local Projections](@ref lp_page).

### LPFEVD Return Values

| Field | Type | Description |
|-------|------|-------------|
| `proportions` | `Array{T,3}` | ``n \times n \times H`` raw R²-based variance shares |
| `bias_corrected` | `Array{T,3}` | ``n \times n \times H`` bias-corrected shares |
| `se` | `Array{T,3}` | ``n \times n \times H`` bootstrap standard errors |
| `ci_lower` | `Array{T,3}` | ``n \times n \times H`` lower CI bound |
| `ci_upper` | `Array{T,3}` | ``n \times n \times H`` upper CI bound |
| `method` | `Symbol` | Estimator used (`:r2`, `:lp_a`, `:lp_b`) |
| `horizon` | `Int` | Maximum FEVD horizon |
| `n_boot` | `Int` | Number of bootstrap replications |
| `conf_level` | `T` | Confidence level |
| `bias_correction` | `Bool` | Whether bias correction was applied |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:r2` | Estimator (`:r2`, `:lp_a`, `:lp_b`) |
| `bias_correct` | `Bool` | `true` | Apply VAR-based bootstrap bias correction |
| `n_boot` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |
| `var_lags` | `Union{Nothing,Int}` | `nothing` | VAR lag order for bias correction (default: HQIC-selected) |

---

## Complete Example

This example computes frequentist, Bayesian, and LP-based FEVD for a three-variable monetary policy VAR using FRED-MD data, then compares the variance shares across methods.

```julia
using MacroEconometricModels, Random
Random.seed!(42)

# Load FRED-MD: industrial production, CPI inflation, federal funds rate
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
H = 20

# --- Frequentist FEVD ---
model = estimate_var(Y, 2)
freq_fevd = fevd(model, H)
report(freq_fevd)

# --- Bayesian FEVD ---
post = estimate_bvar(Y, 2; n_draws=1000, prior=:minnesota,
                     varnames=["INDPRO", "CPI", "FFR"])
bayes_fevd = fevd(post, H; method=:cholesky)
report(bayes_fevd)

# --- LP-based FEVD ---
slp = structural_lp(Y, H; method=:cholesky, lags=4)
lp_fevd_result = fevd(slp, H; bias_correct=true, n_boot=500)
report(lp_fevd_result)

# --- Tabular comparison at selected horizons ---
print_table(stdout, freq_fevd, 1; horizons=[1, 4, 8, 20])

# --- Visualization ---
plot_result(freq_fevd)
plot_result(bayes_fevd)
plot_result(lp_fevd_result)
```

The frequentist FEVD provides point estimates at each horizon. The Bayesian FEVD adds credible intervals reflecting parameter uncertainty. The LP-FEVD offers robustness to VAR lag misspecification at the cost of wider confidence bands. All three methods agree on the qualitative pattern: own shocks dominate at short horizons, while cross-variable transmission mechanisms become visible at longer horizons.

---

## Common Pitfalls

1. **FEVD depends on identification ordering.** With Cholesky identification, the FEVD is sensitive to the variable ordering. Placing a variable first gives it maximum contemporaneous explanatory power. Always report the ordering and justify it on economic grounds.

2. **Row sums equal one only for correctly identified systems.** If the structural impact matrix ``P`` is not orthogonal (e.g., with partial identification), the FEVD rows may not sum to exactly one. The package enforces normalization, but economically the shares are only meaningful under valid identification.

3. **LP-FEVD shares can exceed one before bias correction.** The raw R²-based LP-FEVD estimates individual shock shares independently, so they are not constrained to sum to one across shocks. The bias correction mitigates this but does not impose the adding-up constraint. Compare with VAR-FEVD to assess magnitude.

4. **Short samples inflate Bayesian FEVD uncertainty.** With fewer than 100 observations, the Minnesota prior dominates the posterior and the FEVD credible intervals may be uninformatively wide. Increase `n_draws` and check prior sensitivity.

5. **Horizon must not exceed effective sample size.** For LP-FEVD, each additional horizon reduces the effective sample by one observation. With ``T = 200`` and ``H = 40``, only 160 observations remain at the longest horizon, reducing estimation precision.

---

## References

- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Kilian, Lutz, and Helmut Lutkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [DOI: 10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Gorodnichenko, Yuriy, and Byoungchan Lee. 2019. "Forecast Error Variance Decompositions with Local Projections." *NBER Working Paper* No. 25380. [DOI: 10.3386/w25380](https://doi.org/10.3386/w25380)
- Plagborg-Moller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica*, 89(2), 955--980. [DOI: 10.3982/ECTA17813](https://doi.org/10.3982/ECTA17813)
