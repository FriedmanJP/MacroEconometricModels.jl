# [Variance Decomposition](@id ia_fevd_page)

Forecast Error Variance Decomposition (FEVD) quantifies the proportion of each variable's forecast error variance attributable to each structural shock at a given horizon. It answers the question "which shocks matter most for which variables?" and is, with impulse responses and historical decomposition, one of the three standard innovation-accounting tools of structural VAR analysis (Sims 1980).

- **Frequentist FEVD**: VMA-based decomposition of the orthogonalized forecast error (point estimate)
- **Generalized FEVD**: order-invariant decomposition that never orthogonalizes (Pesaran & Shin 1998)
- **Bayesian FEVD**: posterior distributions over variance shares with credible intervals
- **LP-based FEVD**: ``R^2``-based estimator robust to VAR dynamic misspecification (Gorodnichenko & Lee 2019)

For an overview and method comparison, see [Innovation Accounting](@ref innovation_accounting_page). For impulse responses, see [Impulse Responses](@ref ia_irf_page); for episode-level attribution, see [Historical Decomposition](@ref ia_hd_page).

```@setup ia_fevd
using MacroEconometricModels, Random
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[339:588, :]        # 1987:03-2007:12, the Great Moderation
Y[:, 1:2] .*= 100        # log differences expressed in percent
model = estimate_var(Y, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post = estimate_bvar(Y, 4; n_draws=200, varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"], seed=20260802)
```

The examples use a three-variable monthly monetary VAR(4) estimated on FRED-MD over the Great Moderation (1987:03--2007:12, 250 observations), with industrial production in percent monthly growth, the consumer price index as the monthly change in log inflation, and the federal funds rate as its monthly change.

## Quick Start

**Recipe 1: FEVD with Cholesky identification**

```@example ia_fevd
# Recursive ordering INDPRO -> CPIAUCSL -> FEDFUNDS
decomp = fevd(model, 20)
report(decomp)
```

**Recipe 2: Order-invariant generalized FEVD**

```@example ia_fevd
# No orthogonalization, so no dependence on the variable ordering
gdecomp = generalized_fevd(model, 20)
report(gdecomp)
```

**Recipe 3: Bayesian FEVD with credible intervals**

```@example ia_fevd
bfevd = fevd(post, 20; method=:cholesky)
report(bfevd)
```

**Recipe 4: LP-based FEVD with bias correction**

```@example ia_fevd
slp = structural_lp(Y, 20; method=:cholesky, lags=4,
                    varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
lp_decomp = fevd(slp, 20; bias_correct=true, n_boot=50, rng=MersenneTwister(20260802))
report(lp_decomp)
```

**Recipe 5: Table output at selected horizons**

```@example ia_fevd
print_table(stdout, decomp, "INDPRO"; horizons=[1, 4, 8, 12, 20])
```

---

## Frequentist FEVD

The FEVD measures the proportion of the ``h``-step ahead forecast error variance of variable ``i`` attributable to structural shock ``j``. It derives from the Vector Moving Average (VMA) representation of the structural VAR (Lütkepohl 2005, Section 2.3.3).

```math
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{n} (\Theta_s)_{ik}^2}
```

where:
- ``\text{FEVD}_{ij}(h)`` is the share of variable ``i``'s ``h``-step forecast error variance due to shock ``j``
- ``(\Theta_s)_{ij}`` is the ``(i,j)`` element of the structural impulse response matrix at horizon ``s``
- ``\Theta_s = \Phi_s P`` are the structural MA coefficients, with ``\Phi_s`` the reduced-form MA coefficients and ``P`` the impact matrix
- the numerator accumulates the squared contributions of shock ``j`` through horizon ``h-1``
- the denominator accumulates contributions from all ``n`` shocks

The decomposition satisfies three properties. It is **bounded**: ``0 \leq \text{FEVD}_{ij}(h) \leq 1``. Its **rows sum to one**, ``\sum_{j} \text{FEVD}_{ij}(h) = 1``, by construction of the normalization. And it **converges**: as ``h \to \infty`` the shares approach the unconditional variance decomposition, revealing the long-run drivers of each variable's fluctuations. At short horizons own shocks typically dominate; as the horizon grows, transmission lets other shocks explain more.

!!! warning "The normalization is not a validity check"
    Because each row is divided by its own total, the rows sum to one for *any* impact matrix, valid or not. The accumulated total is the true forecast error variance only when ``P P' = \Sigma``. `fevd` verifies this in the ``\Sigma``-metric and warns once when it fails — some ICA and heteroskedasticity-based identifications return a rotation that is not exactly orthonormal. Use `generalized_fevd` for genuinely non-orthogonal identifications.

```@example ia_fevd
# The funds-rate row of the decomposition reported above, at selected horizons
print_table(stdout, decomp, "FEDFUNDS"; horizons=[1, 4, 8, 12, 20])
```

```julia
plot_result(decomp)
```

```@raw html
<iframe src="../assets/plots/fevd_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

At ``h=1`` the recursive ordering fixes the answer: industrial production owes 100% of its one-month forecast error to its own shock, prices 99.6% to theirs, and the funds rate 97.2% to the policy shock. The interesting movement is off the diagonal at longer horizons. Monetary shocks explain 5.2% of industrial production's forecast error variance at ``h=20`` and price shocks 2.4%, so the vast majority of output variation over a 20-month window is not attributable to identified policy or price disturbances. The funds rate is the variable most explained by others: industrial-production shocks account for 17.8% of its forecast error variance at ``h=20``, up from 2.3% at impact. Read together, this is a systematic policy rule responding to real activity, not a large exogenous policy component.

### `FEVD` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `decomposition` | `Array{T,3}` | ``n \times n \times H`` cumulative squared-IRF contributions (unnormalized) |
| `proportions` | `Array{T,3}` | ``n \times n \times H`` variance shares: `proportions[i, j, h]` = share of variable ``i``'s FEV due to shock ``j`` at horizon ``h`` |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:long_run`, `:narrative`, ICA/ML/heteroskedasticity variants) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source for set-identified methods |

---

## Generalized FEVD (Pesaran-Shin)

The structural FEVD above accumulates squared **orthogonalized** IRFs. That is a proper variance decomposition only when the impact matrix satisfies ``PP' = \Sigma``, and under a Cholesky identification the answer depends on the variable ordering — an ordering that is often arbitrary. The generalized decomposition of Pesaran & Shin (1998), building on the generalized impulse response of Koop, Pesaran & Potter (1996), avoids both problems by using the reduced-form ``\Sigma`` directly, with no orthogonalization at all:

```math
gFEVD_{ij}(H) = \frac{\sigma_{jj}^{-1}\sum_{h=0}^{H-1}\left(e_i' \Phi_h \Sigma e_j\right)^2}
                      {\sum_{h=0}^{H-1} e_i' \Phi_h \Sigma \Phi_h' e_i}
```

where:
- ``\Phi_h`` are the **reduced-form** moving-average coefficients (``\Phi_0 = I``)
- ``\Sigma`` is the reduced-form error covariance and ``\sigma_{jj}`` its ``j``-th diagonal element
- ``e_i`` are selection vectors, so the numerator is the part of variable ``i``'s forecast error variance attributable to a shock to **variable** ``j``

```@example ia_fevd
g = generalized_fevd(model, 20)
gn = generalized_fevd(model, 20; normalize=true)

(raw_row_sums = round.(vec(sum(g.proportions[:, :, 20]; dims=2)); digits=3),
 normalized_row_sums = round.(vec(sum(gn.proportions[:, :, 20]; dims=2)); digits=6))
```

The generalized decomposition is **invariant to the variable ordering**: permuting the variables permutes the result and changes nothing else, to machine precision. Re-estimating this VAR with the ordering reversed and undoing the permutation moves the generalized shares by ``6 \times 10^{-16}``, while the Cholesky shares move by 0.080 — eight percentage points on a quantity bounded by one. That invariance is the reason to reach for it.

The price is that the generalized shocks are **correlated**, so the shares of a given variable do not sum to one. The raw row sums above are 1.027, 1.012, and 1.077: correlated shocks each take credit for the common component. Compare the two decompositions for the funds rate at ``h=20`` — Cholesky gives 80.9% to the policy shock, the generalized version 89.0%, and the difference is precisely the contemporaneous covariance that the recursive ordering hands to whichever variable comes first.

!!! warning "`normalize=true` is a convention, not an identity"
    Rescaling each row to sum to one is standard applied practice — it is what the Diebold-Yilmaz connectedness literature does — but it does not turn the generalized shares into an exclusive decomposition of the variance. They genuinely overlap. Report the raw shares when the overlap is itself informative, and say which version you used.

!!! note "Two exact properties worth checking on your own data"
    At the impact horizon ``\Phi_0 = I``, so ``gFEVD_{ij}(1) = \sigma_{ij}^2/(\sigma_{ii}\sigma_{jj}) = \mathrm{corr}(u_i, u_j)^2`` and the own-variable share is **exactly one**. On this sample that identity holds to ``2 \times 10^{-16}``. And when ``\Sigma`` is diagonal there is nothing to orthogonalize, so the generalized and Cholesky decompositions coincide.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `normalize` | `Bool` | `false` | Rescale each row to sum to one (a convention — see above) |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels (BVAR method) |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (BVAR method) |
| `max_draws` | `Int` | `1000` | Cap on posterior draws used (BVAR method) |
| `threaded` | `Bool` | `false` | Force threaded quantile computation (BVAR method) |

`generalized_fevd` accepts a `VARModel` (returning an [`FEVD`](@ref)) or a `BVARPosterior` (returning a [`BayesianFEVD`](@ref) with posterior bands). Since nothing is orthogonalized, the BVAR method has no `method`/`check_func` machinery — every draw contributes and there is no rotation to accept or reject.

---

## Bayesian FEVD

Bayesian FEVD integrates over parameter uncertainty by computing variance shares for each posterior draw and reporting posterior quantiles (Kilian & Lütkepohl 2017, Chapter 12). Non-stationary draws are discarded, and the counts of requested, usable, and dropped draws travel with the result.

!!! warning "Axis order differs between `FEVD` and `BayesianFEVD`"
    `FEVD.proportions` is indexed ``(\text{variable}, \text{shock}, \text{horizon})`` while `BayesianFEVD.point_estimate` and `.quantiles` are indexed ``(\text{horizon}, \text{variable}, \text{shock})``. Indexing one array with the other's convention silently returns the wrong number instead of erroring.

```@example ia_fevd
# Bayesian FEVD with 68% credible intervals
bfevd = fevd(post, 20; method=:cholesky, quantiles=[0.16, 0.5, 0.84])
report(bfevd)
```

```julia
plot_result(bfevd)
```

```@raw html
<iframe src="../assets/plots/fevd_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example ia_fevd
# point_estimate is (horizon, variable, shock); proportions is (variable, shock, horizon)
(bayesian = round.(bfevd.point_estimate[20, 1, :], digits=4),
 frequentist = round.(decomp.proportions[1, :, 20], digits=4),
 credible_band_policy_shock = round.((bfevd.quantiles[20, 1, 3, 1],
                                      bfevd.quantiles[20, 1, 3, 3]), digits=4))
```

All 200 posterior draws are usable here, so the bands rest on the full posterior. The posterior mean assigns industrial production's 20-month forecast error variance as 88.4% own, 4.0% price, and 7.6% policy, against the frequentist 92.4/2.4/5.2. The Bayesian shares are systematically less concentrated on the own shock because averaging over parameter draws mixes in configurations with stronger cross-variable transmission, and a share bounded below by zero cannot average downward as far as it can average upward. The 68% credible interval for the policy share is ``[0.043, 0.108]``, which comfortably contains the frequentist 0.052: the point estimate is not in question, but a band spanning a factor of 2.5 is the honest summary of what 250 observations reveal about the importance of monetary shocks.

### `BayesianFEVD` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times n_q``: dimension 4 indexes quantile levels |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate of the variance shares |
| `horizon` | `Int` | Maximum FEVD horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |
| `n_requested` | `Int` | Posterior draws supplied by the sampler |
| `n_effective` | `Int` | Draws that were stationary and identified |
| `n_failed` | `Int` | Draws dropped before the quantiles were formed |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency (`:mean` or `:median`) |
| `max_draws` | `Int` | `1000` | Cap on rotation draws per posterior draw for set-identified methods |
| `threaded` | `Bool` | `false` | Force threaded quantile computation (automatic above ``10^5`` cells) |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `data` | `AbstractMatrix` | `post.data` | Override the data used for narrative checks |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |

---

## LP-Based FEVD

The estimators above invert the VAR lag polynomial, so they inherit any misspecification of it. Gorodnichenko & Lee (2019) estimate variance shares directly from ``R^2`` regressions instead: at each horizon the LP forecast error of variable ``i`` is regressed on leads of structural shock ``j``, and the ``R^2`` is the share.

```math
\hat{s}_{ij}(h) = R^2 \left( \hat{f}_{i,t+h|t-1} \sim z_{j,t+h}, z_{j,t+h-1}, \ldots, z_{j,t} \right)
```

where:
- ``\hat{f}_{i,t+h|t-1}`` is the LP forecast error for variable ``i`` at horizon ``h``
- ``z_{j,t}`` is the identified structural shock ``j`` at time ``t``

`fevd(slp, H)` dispatches to this estimator and returns an `LPFEVD`, not the VMA-based `FEVD` above. A raw ``R^2`` is bounded below by zero and therefore biased upward whenever the true share is near zero, so the package applies the VAR-based bootstrap bias correction of Gorodnichenko & Lee (2019, Section 3.4) by default and builds centred bootstrap intervals in the manner of Kilian (1998).

```@example ia_fevd
# Structural LP with Cholesky identification, then the R2-based decomposition
lp_fevd_result = fevd(slp, 20; method=:r2, bias_correct=true,
                      n_boot=50, conf_level=0.95, rng=MersenneTwister(20260802))
report(lp_fevd_result)
```

```julia
plot_result(lp_fevd_result)
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@example ia_fevd
# Bias correction removes most of the raw share at every horizon
(raw = round.(lp_fevd_result.proportions[1, 3, [1, 4, 8, 20]], digits=4),
 corrected = round.(lp_fevd_result.bias_corrected[1, 3, [1, 4, 8, 20]], digits=4))
```

The raw ``R^2`` attributes a rising 0.0%, 5.3%, 6.7%, and 12.3% of industrial production's forecast error to the policy shock over horizons 1, 4, 8, and 20; the bias-corrected series is 0.0%, 3.3%, 2.9%, and 2.9%. Two thirds of the raw long-horizon share is finite-sample bias, exactly what the correction is designed to expose, and the corrected values sit below the VAR-based 5.2% rather than around it. Because each cell is estimated by its own regression, the corrected shares for a variable need not sum to one — they total 0.964 for industrial production at ``h=20`` — and the gap from one is a rough diagnostic of how far the LP and VAR representations of this system have drifted apart.

For the LP-A and LP-B alternatives, the full bias-correction algorithm, the keyword table, and the `LPFEVD` return values, see [Local Projections](@ref lp_page).

---

## Complete Example

This example runs all four estimators on the same monetary VAR and compares what each says about the policy-shock share of industrial production at a 20-month horizon.

```@example ia_fevd
# Four decompositions of one system
chol_fevd  = fevd(model, 20)
gen_fevd   = generalized_fevd(model, 20)
bayes_fevd = fevd(post, 20; method=:cholesky)
lp_r2_fevd = fevd(slp, 20; bias_correct=true, n_boot=50, rng=MersenneTwister(20260802))

# Mind the axis order: (variable, shock, horizon) except for the Bayesian result
(cholesky    = round(chol_fevd.proportions[1, 3, 20], digits=4),
 generalized = round(gen_fevd.proportions[1, 3, 20], digits=4),
 bayesian    = round(bayes_fevd.point_estimate[20, 1, 3], digits=4),
 lp_r2       = round(lp_r2_fevd.bias_corrected[1, 3, 20], digits=4))
```

```@example ia_fevd
# The full Bayesian decomposition, with credible bands
report(bayes_fevd)
```

The four estimators put the monetary contribution to industrial production between 2.9% and 7.7% at a 20-month horizon. The Cholesky and Bayesian figures (5.2% and 7.6%) bracket the same object under different treatments of parameter uncertainty. The generalized share is the largest (7.7%) because it credits the policy variable with the full contemporaneous covariance rather than assigning it to whichever variable is ordered first. The bias-corrected LP estimate is the smallest (2.9%) and the most robust to lag misspecification, at the cost of the widest sampling uncertainty. Every one of them says the same economics: identified monetary shocks are a minor driver of output fluctuations during the Great Moderation, and disagreement across methods is smaller than the sampling uncertainty within any one of them.

---

## Common Pitfalls

1. **The rows always sum to one, so that is not a diagnostic.** Each row is normalized by its own accumulated total, which makes the shares add up whether or not ``PP' = \Sigma`` holds. `fevd` checks orthonormality separately and warns once when it fails; heed that warning rather than the row sums, and switch to `generalized_fevd` when the identification is genuinely non-orthogonal.

2. **FEVD inherits the identification ordering.** Under Cholesky identification, reversing the ordering moved the shares by 8 percentage points on this sample. Always report the ordering and justify it economically, or use the order-invariant generalized decomposition.

3. **Axis order differs between the frequentist and Bayesian types.** `FEVD.proportions[i, j, h]` is (variable, shock, horizon); `BayesianFEVD.point_estimate[h, i, j]` is (horizon, variable, shock). Whenever the horizon is at least as large as the number of variables, indexing one with the other's convention stays in bounds and returns a plausible wrong number instead of throwing.

4. **Generalized shares do not decompose the variance.** They overlap, sum to more than one, and `normalize=true` rescales rather than fixes that. Say which version you report.

5. **LP-FEVD shares are estimated cell by cell.** Each ``R^2`` is clamped to ``[0,1]``, but nothing constrains a variable's shares to sum to one across shocks. Treat a total far from one as evidence that the LP and VAR representations disagree, and compare against the VAR-based decomposition before drawing conclusions.

6. **Horizon must not exceed the effective sample.** For LP-FEVD each additional horizon costs one observation. With ``T = 250`` and ``H = 20`` the longest-horizon regression uses 226 observations; pushing ``H`` toward ``T/4`` degrades precision quickly.

---

## References

- Gorodnichenko, Yuriy, and Byoungchan Lee. 2019. "Forecast Error Variance Decompositions with Local Projections."
  *Journal of Business & Economic Statistics*, 38(4), 921--933. [DOI](https://doi.org/10.1080/07350015.2019.1610661)

- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions."
  *Review of Economics and Statistics*, 80(2), 218--230. [DOI](https://doi.org/10.1162/003465398557465)

- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*.
  Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Koop, Gary, M. Hashem Pesaran, and Simon M. Potter. 1996. "Impulse Response Analysis in Nonlinear Multivariate Models."
  *Journal of Econometrics*, 74(1), 119--147. [DOI](https://doi.org/10.1016/0304-4076(95)01753-4)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Pesaran, H. Hashem, and Yongcheol Shin. 1998. "Generalized Impulse Response Analysis in Linear Multivariate Models."
  *Economics Letters*, 58(1), 17--29. [DOI](https://doi.org/10.1016/S0165-1765(97)00214-0)

- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses."
  *Econometrica*, 89(2), 955--980. [DOI](https://doi.org/10.3982/ECTA17813)

- Sims, Christopher A. 1980. "Macroeconomics and Reality."
  *Econometrica*, 48(1), 1--48. [DOI](https://doi.org/10.2307/1912017)
