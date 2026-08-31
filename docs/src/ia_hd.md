# [Historical Decomposition](@id ia_hd_page)

Historical Decomposition (HD) splits the observed path of each variable into the contributions of individual structural shocks. Where FEVD answers "which shocks matter on average?", HD answers "which shocks drove *this* episode?" — making it the standard tool for narrative interpretation of a structural VAR (Kilian & Lütkepohl 2017, Chapter 4).

- **Frequentist HD**: an exact additive decomposition of the data into shock contributions plus a deterministic and initial-condition component
- **Bayesian HD**: posterior distributions over shock contributions with credible intervals
- **Accessors**: `contribution()`, `total_shock_contribution()`, and `verify_decomposition()` for programmatic analysis

For an overview and method comparison, see [Innovation Accounting](@ref innovation_accounting_page). For variance decomposition, see [Variance Decomposition](@ref ia_fevd_page); for the impulse responses that generate these contributions, see [Impulse Responses](@ref ia_irf_page).

```@setup ia_hd
using MacroEconometricModels, Random, Statistics
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[339:588, :]        # 1987:03-2007:12, the Great Moderation
Y[:, 1:2] .*= 100        # log differences expressed in percent
model = estimate_var(Y, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post = estimate_bvar(Y, 4; n_draws=200, varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"], seed=20260802)
```

The examples use a three-variable monthly monetary VAR(4) estimated on FRED-MD over the Great Moderation (1987:03--2007:12, 250 observations, 246 after the lags), with industrial production in percent monthly growth, the consumer price index as the monthly change in log inflation, and the federal funds rate as its monthly change.

## Quick Start

**Recipe 1: Historical decomposition of a VAR**

```@example ia_hd
# Cholesky identification, full effective sample
hd = historical_decomposition(model)
report(hd)
```

**Recipe 2: Check the additive identity**

```@example ia_hd
# Contributions plus the initial-condition component reproduce the data
verify_decomposition(hd)
```

**Recipe 3: Extract one shock's contribution**

```@example ia_hd
# Contribution of the monetary shock (3) to industrial production (1)
monetary_to_output = contribution(hd, "INDPRO", "FEDFUNDS")
round.((minimum(monetary_to_output), maximum(monetary_to_output)), digits=4)
```

**Recipe 4: Bayesian historical decomposition**

```@example ia_hd
bhd = historical_decomposition(post; method=:cholesky)
report(bhd)
```

**Recipe 5: Decomposition from a structural LP**

```@example ia_hd
# LP-estimated IRFs as the structural MA coefficients
slp = structural_lp(Y, 20; method=:cholesky, lags=4,
                    varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
report(historical_decomposition(slp))
```

---

## Frequentist HD

Historical decomposition derives from the structural VMA representation of the VAR. The observed data at time ``t`` is the accumulation of all past structural shocks plus a component that captures the intercept and the influence of pre-sample values.

```math
y_t = \sum_{s=0}^{t-1} \Theta_s \varepsilon_{t-s} + \text{initial}_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of observed variables at time ``t``
- ``\Theta_s = \Phi_s P`` are the ``n \times n`` structural MA coefficients at lag ``s``
- ``\Phi_s`` are the reduced-form MA coefficients from the VMA representation
- ``P = L Q`` is the ``n \times n`` structural impact matrix (Cholesky factor ``L`` times rotation ``Q``)
- ``\varepsilon_t = Q' L^{-1} u_t`` are the ``n \times 1`` structural shocks
- ``\text{initial}_t`` carries the deterministic terms and the decaying effect of pre-sample values

The contribution of shock ``j`` to variable ``i`` at time ``t`` is the convolution of that shock's history with the corresponding MA coefficients:

```math
\text{HD}_{ij}(t) = \sum_{s=0}^{t-1} (\Theta_s)_{ij} \, \varepsilon_j(t-s)
```

where ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the structural MA coefficient at lag ``s`` and ``\varepsilon_j(t-s)`` is the realized structural shock ``j`` at time ``t-s``.

!!! note "How the identity is enforced"
    `initial_conditions` is computed as the residual ``y_{i,t} - \sum_j \text{HD}_{ij}(t)``, so the additive identity ``y_{i,t} = \sum_j \text{HD}_{ij}(t) + \text{initial}_i(t)`` holds by construction. `verify_decomposition()` therefore checks the arithmetic — it catches a numerical failure in the MA recursion, not a misspecified model — and returns `true` whenever the contributions are finite.

```@example ia_hd
# Historical decomposition with Cholesky identification
hd = historical_decomposition(model)
report(hd)
```

```julia
plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The contribution summary reports the mean absolute contribution of each shock to each variable, which is a compact way to read the transmission matrix off the historical record. Industrial production is driven overwhelmingly by its own shock (mean absolute contribution 0.388 percent per month against 0.057 for price shocks and 0.085 for monetary shocks), and its total shock-driven component tracks the data almost exactly — the correlation with the actual series is 0.999, and the standard deviations are 0.507 against 0.510. The funds rate is the variable with the most cross-shock influence: output shocks contribute 0.071 on average against its own 0.139, the historical counterpart of the variance decomposition's finding that real activity explains a sixth of policy-rate variation.

```@example ia_hd
# The monetary shock's contribution to output, and the last ten periods in full
monetary_to_output = contribution(hd, "INDPRO", "FEDFUNDS")
total = total_shock_contribution(hd, "INDPRO")

print_table(stdout, hd, 1; periods=(hd.T_eff-9):hd.T_eff)
```

Monetary shocks move industrial production by up to ``+0.271`` percent in a single month (period 11) and as much as ``-0.410`` percent (period 23), with a standard deviation of 0.109 — roughly a fifth of the total variation in monthly output growth. Over the final ten months of the sample the own shock dominates every period: at ``t=243`` an output shock of ``+0.442`` percent carries a realized growth rate of ``+0.532``, and at ``t=246`` an output shock of ``-0.438`` drags growth to ``-0.393``. The `Initial` column is flat at 0.220 across all ten periods, which is the deterministic mean growth implied by the intercept, not a pre-sample effect: whatever the initial conditions contributed has long since decayed.

### Helper Functions

| Function | Description |
|----------|-------------|
| `contribution(hd, var, shock)` | Time series of shock ``j``'s contribution to variable ``i``; accepts `Int` indices or `String` names |
| `contribution(bhd, var, shock; stat)` | Bayesian version; `stat=:mean` for the point estimate or an `Int` index into `quantile_levels` |
| `total_shock_contribution(hd, var)` | Sum of all shock contributions for variable ``i``, excluding the initial-condition component |
| `verify_decomposition(hd; tol)` | Check the additive identity to tolerance `tol` (default ``10^{-10}``; ``10^{-6}`` for the Bayesian type) |

### `HistoricalDecomposition` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `contributions` | `Array{T,3}` | ``T_{eff} \times n \times n``: `contributions[t, i, j]` = contribution of shock ``j`` to variable ``i`` at time ``t`` |
| `initial_conditions` | `Matrix{T}` | ``T_{eff} \times n`` deterministic and initial-condition component |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `shocks` | `Matrix{T}` | ``T_{eff} \times n`` structural shocks |
| `T_eff` | `Int` | Effective number of time periods (sample size minus lag order) |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `method` | `Symbol` | Identification method (`:cholesky`, `:sign`, `:long_run`, ...) |

### Arguments

`historical_decomposition(model, horizon=effective_nobs(model); kwargs...)` takes the MA truncation horizon as a **positional** argument, capped internally at ``T_{eff}``.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `horizon` | `Int` | `effective_nobs(model)` | Number of structural MA coefficients ``\Theta_s`` computed |
| `method` | `Symbol` | `:cholesky` | Identification method |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `max_draws` | `Int` | `1000` | Maximum rotation draws for sign/narrative identification |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source for set-identified methods |

---

## Bayesian HD

Bayesian HD computes the historical decomposition for each posterior draw of a Bayesian VAR, producing posterior distributions over shock contributions (Kilian & Lütkepohl 2017, Chapter 12). Non-stationary draws are skipped, and a warning fires when more than half of the draws are lost.

```@example ia_hd
# Bayesian HD with 68% credible intervals
bhd = historical_decomposition(post; method=:cholesky, quantiles=[0.16, 0.5, 0.84])
report(bhd)
```

```julia
plot_result(bhd)
```

```@raw html
<iframe src="../assets/plots/hd_bayesian.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example ia_hd
# Posterior mean contribution and its 68% band at the largest positive episode
bayes_contrib = contribution(bhd, 1, 3; stat=:mean)
t_peak = argmax(bayes_contrib)
(period = t_peak,
 mean = round(bayes_contrib[t_peak], digits=4),
 band = round.((bhd.quantiles[t_peak, 1, 3, 1], bhd.quantiles[t_peak, 1, 3, 3]), digits=4))
```

All 200 posterior draws are stationary on this sample, so nothing is discarded. The posterior mean contributions are almost identical to the frequentist ones — 0.388 against 0.388 for the own shock's mean absolute contribution to industrial production — because 246 observations leave little room for the prior to move the coefficients. What the Bayesian version adds is the band: at the peak monetary episode the posterior mean contribution to output growth is ``+0.274`` percent with a 68% credible interval of ``[0.172, 0.375]``. The episode is real, in the sense that the whole interval is on one side of zero, but its magnitude is uncertain by a factor of two — a caveat invisible in the frequentist decomposition, which reports the point estimate alone.

!!! note "Verification tolerance is looser for the Bayesian type"
    `verify_decomposition` uses ``10^{-6}`` for a `BayesianHistoricalDecomposition` against ``10^{-10}`` for the frequentist one. The identity is exact for every individual draw; the point estimate is an average across draws, and averaging the residual-defined initial conditions leaves rounding at the level of the summation order.

### `BayesianHistoricalDecomposition` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``T_{eff} \times n \times n \times n_q`` contribution quantiles |
| `point_estimate` | `Array{T,3}` | ``T_{eff} \times n \times n`` posterior point estimate contributions |
| `initial_quantiles` | `Array{T,3}` | ``T_{eff} \times n \times n_q`` initial-condition quantiles |
| `initial_point_estimate` | `Matrix{T}` | ``T_{eff} \times n`` posterior point estimate initial conditions |
| `shocks_point_estimate` | `Matrix{T}` | ``T_{eff} \times n`` posterior point estimate structural shocks |
| `actual` | `Matrix{T}` | ``T_{eff} \times n`` actual data values |
| `T_eff` | `Int` | Effective number of time periods |
| `variables` | `Vector{String}` | Variable names |
| `shock_names` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |
| `method` | `Symbol` | Identification method |
| `n_requested` | `Int` | Posterior draws supplied by the sampler |
| `n_effective` | `Int` | Stationary draws actually used |
| `n_failed` | `Int` | Draws skipped as non-stationary |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `data` | `AbstractMatrix` | `post.data` | Override the data matrix |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |

---

## Other Estimation Routes

`historical_decomposition` dispatches on the object it is given, so the same accessors work across estimators.

Passing a `StructuralLP` uses the LP-estimated IRFs as the structural MA coefficients ``\Theta_h`` and the structural shocks from the underlying VAR identification, giving an LP-robust decomposition of the same data:

```@example ia_hd
# LP-based MA coefficients instead of the VAR's
hd_lp = historical_decomposition(slp)
report(hd_lp)
```

The LP decomposition spreads the same data across shocks far more evenly than the VAR does — the own-shock contribution to industrial production falls from 0.388 to 0.185 while the price and monetary contributions rise to 0.103 and 0.154 — and it leaves much more unexplained: the `Initial` column more than doubles, from 0.222 to 0.485. That last number is the one to read first. The LP IRFs are estimated horizon by horizon and truncated at ``H = 20``, so they reproduce the data less completely than the VAR's geometric MA weights, and everything they miss lands in the residual column by construction. Treat a large gap between the two decompositions as a diagnostic that the two representations of the dynamics disagree, not as a better answer from either one.

Passing a `VARModel` together with an `SVARRestrictions` object runs the Arias, Rubio-Ramírez & Waggoner (2018) zero-and-sign algorithm and returns a `BayesianHistoricalDecomposition` whose bands are importance-weighted across accepted rotations; `historical_decomposition(vecm)` routes a `VECMModel` through `to_var`. See [Structural Identification](@ref structural_identification_page) and [Vector Error Correction Models](@ref vecm_page) respectively.

A [generalized dynamic factor model](@ref factor_page) decomposes by **dynamic principal component** (FHLR 2000), not by a VAR rotation. A [structural DFM](@ref factor_page) uses the identification stored at estimation (`Q` / `B0`) and maps factor contributions through ``\Lambda``, with an `"Idiosyncratic"` column so the identity holds on the panel:

```@example ia_hd
gdfm_hd = estimate_gdfm(Y, 1; standardize=true)
hd_g = historical_decomposition(gdfm_hd)
sdfm_hd = estimate_structural_dfm(Y, 1; identification=:cholesky, p=1, H=12,
    varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
hd_s = historical_decomposition(sdfm_hd)
(gdfm_ok=verify_decomposition(hd_g),
 sdfm_ok=verify_decomposition(hd_s),
 sdfm_shocks=hd_s.shock_names)
```

---

## Complete Example

This workflow moves through the three innovation-accounting tools in the order an applied paper uses them: impulse responses for the transmission mechanism, variance decomposition for average importance, historical decomposition for the specific episodes.

```@example ia_hd
# Step 1: how do variables respond to shocks?
irfs = irf(model, 20; method=:cholesky, ci_type=:bootstrap, reps=50, seed=20260802)
report(irfs)
```

```@example ia_hd
# Step 2: which shocks matter on average?
decomp = fevd(model, 20)
print_table(stdout, decomp, "INDPRO"; horizons=[1, 4, 8, 12, 20])
```

```@example ia_hd
# Step 3: which shocks drove specific episodes?
hd_full = historical_decomposition(model)
monetary = contribution(hd_full, "INDPRO", "FEDFUNDS")

(largest_positive_month = argmax(monetary),
 largest_negative_month = argmin(monetary),
 range = round.((minimum(monetary), maximum(monetary)), digits=4),
 share_of_output_sd = round(std(monetary) / std(hd_full.actual[:, 1]), digits=3))
```

```julia
plot_result(irfs)
plot_result(decomp)
plot_result(hd_full)
```

```@raw html
<iframe src="../assets/plots/hd_freq.html" width="100%" height="600" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The three tools agree and each adds something the others cannot. The IRF shows the transmission shape: a contractionary policy shock raises the funds rate 0.155 percentage points on impact and depresses output growth from ``h=8`` onward, though no output horizon is individually significant. The FEVD prices that channel at 5.2% of industrial production's 20-month forecast error variance — small, but not zero. The HD then names the months: monetary shocks moved output growth over a range of ``-0.410`` to ``+0.271`` percent, with a standard deviation 21% as large as output growth's own. A channel that is minor on average can still be decisive in a particular episode, and only the historical decomposition shows which one.

---

## Common Pitfalls

1. **`verify_decomposition` cannot detect a bad model.** `initial_conditions` is defined as the residual `actual - sum(contributions)`, so the identity holds by construction and the check returns `true` for any finite decomposition, correctly or badly identified. Use it as a numerical guard, not as evidence that the identification is right.

2. **The `Initial` column is not just initial conditions.** It carries the deterministic component implied by the intercept as well as the decaying pre-sample influence. On this sample it settles at 0.220 for industrial production — essentially the unconditional mean growth rate of 0.230 — within about five years and stays there. Shock contributions are therefore deviations from the deterministic path, not from zero.

3. **HD inherits the entire identification scheme.** With Cholesky identification, reordering the variables changes every contribution. With sign restrictions, the decomposition reflects one draw from the identified set — re-run with several seeds, or use the Bayesian route, before building a narrative on it.

4. **Truncating `horizon` moves mass into the initial-condition column.** The positional `horizon` argument caps how many ``\Theta_s`` are computed. Setting it below ``T_{eff}`` does not break the identity — the residual definition absorbs the difference — it silently reattributes the dropped shock effects to `initial_conditions`. How much moves depends on how fast the MA coefficients decay: on this VAR, truncating at ``H=20`` changes the mean contributions by less than 0.001, while the LP decomposition above loses twice as much. Leave it at the default unless the truncation is deliberate.

5. **Bayesian HD discards non-stationary draws.** With a diffuse prior or a short sample the effective number of draws can fall far below `n_draws`; a warning fires past half. Check `n_effective` on the result before reporting bands, and tighten the prior or lengthen the sample rather than accepting a decomposition built on a handful of draws.

6. **String indexing requires exact names.** `contribution(hd, "INDPRO", "FEDFUNDS")` matches on the strings stored in the model. Inspect `hd.variables` and `hd.shock_names` when a lookup throws — a VAR estimated without `varnames=` carries `"y1"`, `"y2"`, `"y3"`.

---

## References

- Arias, Jonas E., Juan F. Rubio-Ramírez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications."
  *Econometrica*, 86(2), 685--720. [DOI](https://doi.org/10.3982/ECTA14468)

- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
  *American Economic Review*, 95(1), 161--182. [DOI](https://doi.org/10.1257/0002828053828518)

- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*.
  Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses."
  *Econometrica*, 89(2), 955--980. [DOI](https://doi.org/10.3982/ECTA17813)

- Sims, Christopher A. 1980. "Macroeconomics and Reality."
  *Econometrica*, 48(1), 1--48. [DOI](https://doi.org/10.2307/1912017)
