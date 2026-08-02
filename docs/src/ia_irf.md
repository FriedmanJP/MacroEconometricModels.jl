# [Impulse Responses](@id ia_irf_page)

The **impulse response function** (IRF) traces the dynamic effect of a one-standard-deviation structural shock on each endogenous variable over time. `irf` computes impulse responses from VAR, BVAR, VECM, FAVAR, DSGE, and Local Projection objects through a single interface, with residual-bootstrap and asymptotic confidence intervals, Bayesian credible bands, cumulation for growth-rate variables, and stationarity-filtered inference.

For an overview and method comparison, see [Innovation Accounting](@ref innovation_accounting_page). For variance decomposition, see [Variance Decomposition](@ref ia_fevd_page); for episode-level attribution, see [Historical Decomposition](@ref ia_hd_page).

```@setup ia_irf
using MacroEconometricModels, Random
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[339:588, :]        # 1987:03-2007:12, the Great Moderation
Y[:, 1:2] .*= 100        # log differences expressed in percent
Y_levels = to_matrix(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]])[339:588, :]
model = estimate_var(Y, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
post = estimate_bvar(Y, 4; n_draws=200, varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"], seed=20260802)
```

The examples on this page use a three-variable monthly monetary VAR estimated on FRED-MD over the Great Moderation (1987:03--2007:12, 250 observations). Industrial production enters as monthly log growth in percent (FRED-MD transformation code 5), the consumer price index as the monthly change in log inflation in percentage points (code 6), and the federal funds rate as its monthly change in percentage points (code 2).

## Quick Start

**Recipe 1: Cholesky impulse responses**

```@example ia_irf
# Recursive ordering INDPRO -> CPIAUCSL -> FEDFUNDS
result = irf(model, 20)
report(result)
```

**Recipe 2: Bootstrap confidence intervals**

```@example ia_irf
# Residual bootstrap with 95% pointwise intervals; `seed` fixes the draws
result = irf(model, 20; ci_type=:bootstrap, reps=50, conf_level=0.95, seed=20260802)
report(result)
```

**Recipe 3: Cumulative response of a growth-rate variable**

```@example ia_irf
# Cumulating monthly growth gives the implied response of the level
cum_result = cumulative_irf(result)
report(cum_result)
```

**Recipe 4: Bayesian credible bands**

```@example ia_irf
# Posterior quantiles across BVAR draws, 68% bands by default
bayes_result = irf(post, 20)
report(bayes_result)
```

**Recipe 5: Sign-restricted identification**

```@example ia_irf
# Demand shock: output and prices both rise on impact
check_demand = irf_array -> irf_array[1, 1, 1] > 0 && irf_array[1, 2, 1] > 0
sign_result = irf(model, 20; method=:sign, check_func=check_demand, seed=20260802)
report(sign_result)
```

**Recipe 6: Local projections with bootstrap bands**

```@example ia_irf
# Horizon-by-horizon regressions, bands from a fixed-design residual bootstrap
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west,
                 varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
report(lp_irf(lp; ci_type=:bootstrap, reps=50, seed=20260802))
```

---

## Frequentist IRF

The impulse response function ``\Theta_h`` measures the dynamic causal effect of a structural shock at time ``t`` on the endogenous variables at time ``t+h``. Under a recursive (Cholesky) identification the ordering of variables determines the contemporaneous causal structure: a variable placed later cannot affect earlier variables within the period.

```math
\Theta_h = \frac{\partial y_{t+h}}{\partial \varepsilon_t'}
```

where:
- ``\Theta_h`` is the ``n \times n`` impulse response matrix at horizon ``h``
- ``y_{t+h}`` is the ``n \times 1`` vector of endogenous variables at time ``t+h``
- ``\varepsilon_t`` is the ``n \times 1`` vector of structural shocks at time ``t``

For a VAR(p) model, the IRF at horizon ``h`` is computed recursively from the reduced-form moving-average coefficients:

```math
\Theta_h = \Phi_h \cdot B_0, \qquad \Phi_h = \sum_{i=1}^{\min(h,p)} A_i \, \Phi_{h-i}, \qquad \Phi_0 = I_n
```

where:
- ``A_i`` are the ``n \times n`` VAR coefficient matrices for lag ``i``
- ``\Phi_h`` are the reduced-form MA coefficients at horizon ``h``
- ``B_0 = L \cdot Q`` is the ``n \times n`` structural impact matrix
- ``L`` is the lower-triangular Cholesky factor of ``\Sigma``
- ``Q`` is the ``n \times n`` orthogonal rotation matrix (``Q = I_n`` for Cholesky)

Equivalently, the companion form representation computes the IRF as ``\Theta_h = J F^h J' B_0``, where ``J = [I_n, 0, \ldots, 0]`` is the ``n \times np`` selection matrix and ``F`` is the ``np \times np`` companion matrix.

!!! note "Technical Note"
    The `ci_lower` and `ci_upper` arrays are populated only when `ci_type=:bootstrap` or `ci_type=:theoretical`. With `ci_type=:none` (the default) these arrays contain zeros and `_conf_level` is zero. Always check `result.ci_type` before interpreting confidence bands.

```@example ia_irf
# Cholesky IRF with bootstrap 95% confidence intervals
H = 20
boot_irf = irf(model, H; ci_type=:bootstrap, reps=50, conf_level=0.95, seed=20260802)
report(boot_irf)
```

```julia
plot_result(boot_irf)
```

```@raw html
<iframe src="../assets/plots/irf_freq.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The impact matrix is lower triangular by construction: a funds-rate innovation moves industrial production and prices by exactly zero at ``h=1``, and the own impact effects are 0.466, 0.214, and 0.155 for the three variables. The policy shock is persistent — the funds rate itself is still 0.023 percentage points above baseline at ``h=8``, and within the policy-shock column it is the only response whose interval excludes zero beyond impact. Industrial production rises 0.050 percent at ``h=4`` before turning negative from ``h=8`` onward (``-0.015``), the standard delayed-contraction shape, but no output horizon has a 95% interval that excludes zero. The reverse channel is sharper: an industrial-production shock raises the funds rate by 0.024 on impact and 0.035 at ``h=4``, both starred, which is the systematic policy reaction rather than a policy shock.

The point-estimate engine is also exposed for advanced users who supply their own rotation matrix ``Q``: [`compute_irf`](@ref)`(model, Q, horizon)` returns the ``H \times n \times n`` IRF array for any identification scheme. `irf` accepts a `VECMModel` as well, routing through `to_var` so that cointegrated systems reuse the same machinery — see [Vector Error Correction Models](@ref vecm_page).

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (`:cholesky`, `:sign`, `:narrative`, `:long_run`, ICA/ML/heteroskedasticity variants) |
| `ci_type` | `Symbol` | `:none` | CI method: `:none`, `:bootstrap`, or `:theoretical` |
| `reps` | `Int` | `200` | Number of bootstrap or simulation replications |
| `conf_level` | `Real` | `0.95` | Confidence level for interval construction |
| `stationary_only` | `Bool` | `false` | Reject explosive draws and redraw (Kilian & Lütkepohl 2017) |
| `bootstrap` | `Symbol` | `:iid` | Residual resampling scheme: `:iid`, `:wild`, or `:block` |
| `block_length` | `Int` | `0` | Moving-block length; `0` selects ``\lceil T^{1/3} \rceil`` |
| `wild_dist` | `Symbol` | `:rademacher` | Wild-bootstrap multiplier: `:rademacher` or `:mammen` |
| `bias_correct` | `Bool` | `false` | Kilian (1998) bootstrap-after-bootstrap bias correction of the bands |
| `bias_reps` | `Int` | `0` | Replications for the inner bias bootstrap; `0` reuses `reps` |
| `check_func` | `Function` | `nothing` | Sign restriction check function (required for `:sign`) |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check (required for `:narrative`) |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |
| `seed` | `Integer` | `nothing` | Owns the RNG so bands reproduce bit-for-bit |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source when no `seed` is given |

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
| `manifest` | `ReproManifest` | Seed and settings recorded by the bootstrap path; `nothing` otherwise |

---

## Bootstrap Inference

`ci_type=:bootstrap` runs a recursive-design residual bootstrap: residuals are resampled, data are regenerated from ``\hat{B}``, and the VAR is re-estimated once per replication. Three resampling schemes are available. The i.i.d. scheme resamples residual rows with replacement. The wild scheme multiplies each residual **row** by a single scalar draw, preserving the contemporaneous cross-equation covariance while randomizing the conditional variance (Gonçalves & Kilian 2004). The block scheme concatenates contiguous blocks of residuals, retaining serial dependence (Brüggemann, Jentsch & Trenkler 2016).

```@example ia_irf
# Interval width at h=8 for the INDPRO response to a policy shock
widths = map((:iid, :wild, :block)) do scheme
    r = irf(model, 20; ci_type=:bootstrap, reps=50, bootstrap=scheme, seed=20260802)
    scheme => round(r.ci_upper[8, 1, 3] - r.ci_lower[8, 1, 3], digits=4)
end
```

The i.i.d. and block schemes give nearly identical widths here (0.0571 and 0.0587): four lags absorb the serial dependence, so preserving residual blocks changes nothing the i.i.d. scheme was already getting right. The wild bootstrap is the tightest at 0.0469, because multiplying whole rows by a Rademacher draw reproduces the residual covariance exactly and only randomizes signs, which disperses less than resampling rows when the residuals are close to homoskedastic. Choose `:wild` when conditional heteroskedasticity is the concern and `:block` when residual autocorrelation is.

!!! note "Bias correction acts on the bands, not the point estimate"
    `bias_correct=true` runs Kilian's (1998) bootstrap-after-bootstrap: an inner bootstrap estimates the small-sample bias ``\Psi = E[B^*] - \hat{B}``, the DGP is re-centred at ``\hat{B} - \delta\Psi`` with Kilian's stationarity shrinkage, and every outer draw is corrected by the same ``\Psi``. The reported `values` array is the uncorrected point IRF in both cases; only `ci_lower` and `ci_upper` move.

Passing `seed` makes the bands reproducible bit-for-bit. The result carries a `ReproManifest`, and [`reproduce`](@ref) re-runs the recorded bootstrap and compares it against the stored arrays:

```@example ia_irf
reproduce(boot_irf, model)
```

The report confirms that `values`, `ci_lower`, and `ci_upper` all match to ``\max|\Delta| = 0``. The source `model` is passed explicitly because it is deliberately not retained on the IRF object. Per-replication sub-seeding is thread-invariant, so the same seed reproduces the same bands regardless of `JULIA_NUM_THREADS`.

---

## Cumulative IRF

For variables measured in growth rates, the cumulative IRF recovers the effect on the level. The cumulative response through horizon ``H`` is:

```math
\Theta^{\text{cum}}_H = \sum_{h=0}^{H} \Theta_h
```

where:
- ``\Theta^{\text{cum}}_H`` is the ``n \times n`` cumulated response matrix at horizon ``H``
- ``\Theta_h`` is the pointwise IRF at horizon ``h``

!!! note "Cumulative IRF Confidence Intervals"
    For bootstrap or Bayesian bands, the cumulative sum is computed *per draw* before extracting quantiles. This produces correct coverage. Cumulating the pointwise quantiles instead overstates uncertainty, because quantiles are not additive: ``Q_\alpha(\sum_h \Theta_h) \neq \sum_h Q_\alpha(\Theta_h)``.

`cumulative_irf` accepts `ImpulseResponse`, `BayesianImpulseResponse`, and `LPImpulseResponse` objects. When raw draws are available — from an `irf()` call with `ci_type=:bootstrap`, or from any Bayesian IRF — the function cumulates each draw before extracting quantiles; otherwise it falls back to cumulating the stored bounds.

```@example ia_irf
# Cumulate: each bootstrap draw is summed before extracting quantiles
cum_irf = cumulative_irf(boot_irf)

# The correct band versus the naive one built from cumulated bounds
(correct = round.((cum_irf.ci_lower[20, 1, 3], cum_irf.ci_upper[20, 1, 3]), digits=4),
 naive   = round.((sum(boot_irf.ci_lower[1:20, 1, 3]), sum(boot_irf.ci_upper[1:20, 1, 3])), digits=4))
```

Because INDPRO enters in monthly growth rates, the cumulative response is the implied path of the *level* of industrial production: it peaks at ``+0.155`` percent around ``h=4`` and decays to ``+0.044`` percent by ``h=20``, so a contractionary policy shock leaves no permanent level effect in this sample. The uncertainty comparison is the point of the example: the correct band at ``h=20`` is ``[-0.230, 0.331]``, while naively summing the pointwise bounds gives ``[-0.380, 0.458]`` — 49% wider. Summing bounds assumes every horizon reaches its extreme in the same draw, which no draw does.

---

## Bayesian IRF

Bayesian IRFs replace bootstrap confidence intervals with posterior credible bands derived from the BVAR posterior. For each posterior draw of ``(B^{(d)}, \Sigma^{(d)})``, the algorithm computes the full IRF and then reports posterior quantiles across draws.

```math
\Theta_h^{(d)} = \Phi_h^{(d)} \cdot B_0^{(d)}, \qquad d = 1, \ldots, D
```

where:
- ``\Theta_h^{(d)}`` is the IRF at horizon ``h`` for posterior draw ``d``
- ``D`` is the number of usable posterior draws
- ``B_0^{(d)} = L^{(d)} Q`` with ``L^{(d)}`` the Cholesky factor of ``\Sigma^{(d)}``

Non-stationary draws are discarded before the quantiles are formed, and the counts of requested, usable, and dropped draws are carried on the result so the reported bands can be judged against the Monte Carlo effort behind them.

```@example ia_irf
# Bayesian IRFs with default 68% credible bands
bayes_irf = irf(post, 20)
report(bayes_irf)
```

```julia
plot_result(bayes_irf)
```

```@raw html
<iframe src="../assets/plots/irf_bayesian.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

All 200 posterior draws are stationary on this sample, so the bands use the full posterior. `report` prints the **middle quantile** — the posterior median under the default `quantiles` — and stars an entry when the outer quantiles exclude zero; the posterior mean lives in `point_estimate` and surfaces as the `Mean` column of `print_table`. The median response of industrial production to a policy shock is ``+0.052`` percent at ``h=4`` and ``-0.014`` at ``h=8``, within three thousandths of the frequentist point estimates, because 250 observations dominate a diffuse prior. The stars need care: the default bands are 16th-to-84th percentiles, so the star at ``h=4`` marks a 68% interval that excludes zero, which is a weaker claim than the 95% bootstrap interval, which did not. At ``h=8`` the credible band spans 0.043 against 0.057 for the 95% bootstrap interval. Pass `quantiles=[0.05, 0.5, 0.95]` for 90% bands, and `point_estimate=:median` to store the posterior median in `point_estimate` instead of the mean.

### Bayesian IRF Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method |
| `quantiles` | `Vector{<:Real}` | `[0.16, 0.5, 0.84]` | Posterior quantile levels |
| `point_estimate` | `Symbol` | `:mean` | Central tendency: `:mean` or `:median` |
| `max_draws` | `Int` | `1000` | Cap on rotation draws per posterior draw for set-identified methods |
| `threaded` | `Bool` | `false` | Force threaded quantile computation (automatic above ``10^5`` cells) |
| `data` | `AbstractMatrix` | `post.data` | Override the data used for narrative checks |
| `shock_names` | `Vector{String}` | variable names | Labels for the shock dimension |

### `BayesianImpulseResponse` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `quantiles` | `Array{T,4}` | ``H \times n \times n \times q``: dimension 4 indexes quantile levels |
| `point_estimate` | `Array{T,3}` | ``H \times n \times n`` posterior point estimate |
| `horizon` | `Int` | Maximum IRF horizon |
| `variables` | `Vector{String}` | Variable names |
| `shocks` | `Vector{String}` | Shock names |
| `quantile_levels` | `Vector{T}` | Quantile levels (e.g., `[0.16, 0.5, 0.84]`) |
| `n_requested` | `Int` | Posterior draws supplied by the sampler |
| `n_effective` | `Int` | Draws that were stationary and identified |
| `n_failed` | `Int` | Draws dropped before the quantiles were formed |

---

## LP-Based IRF

Local Projections (Jordà 2005) estimate the impulse response directly via horizon-specific regressions, without imposing the dynamic restrictions of a VAR. Structural LP (Plagborg-Møller & Wolf 2021) combines VAR-based identification with LP estimation: structural shocks are recovered from a VAR, then used as the impulse regressor in an LP regression at each horizon.

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, \hat{\varepsilon}_{j,t} + \Gamma_{i,h}' \, w_t + u_{i,t+h}, \qquad h = 0, 1, \ldots, H
```

where:
- ``y_{i,t+h}`` is variable ``i`` at horizon ``h``
- ``\hat{\varepsilon}_{j,t}`` is the identified structural shock ``j``
- ``\beta_{i,h}`` is the LP-estimated impulse response at horizon ``h``
- ``w_t`` contains lags of ``y_t`` as controls
- standard errors use Newey-West HAC to account for the MA(``h``) serial correlation that direct projection induces in ``u_{i,t+h}``

LP and VAR target the same population impulse responses (Plagborg-Møller & Wolf 2021). LP is more robust to dynamic misspecification but less efficient, producing wider confidence bands.

```@example ia_irf
# Reduced-form LP: responses to a funds-rate innovation, not to an identified shock
lp_result = lp_irf(lp; conf_level=0.95)
report(lp_result)
```

`ci_type=:analytical` (the default, shown here) reports HAC bands from the estimated `vcov`; `ci_type=:bootstrap`, used in Recipe 6, replaces them with percentile bands from a fixed-design residual bootstrap that holds ``X_h`` fixed and resamples only the errors. The point estimates and standard errors are the analytical ones either way, so switching `ci_type` never moves the reported response.

```julia
plot_result(lp_result)
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

`estimate_lp(Y, 3, 20)` projects each variable on the *observed* funds rate, so the shock is normalized to one at ``h=0`` and the industrial-production response of ``+0.451`` percent on impact is dominated by reverse causality: the Fed raises rates when output is strong. That is exactly the quantity a structural identification is meant to purge, and it motivates `structural_lp`:

```@example ia_irf
# Structural LP: Cholesky-identified shocks used as the LP regressor
slp = structural_lp(Y, 20; method=:cholesky, lags=4,
                    varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
slp_irf = irf(slp)
report(slp_irf)
```

```julia
plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Replacing the observed funds rate with the identified shock pulls the impact response of industrial production down from ``+0.451`` to ``+0.108`` and turns the medium-horizon path negative (``-0.028`` at ``h=8``, ``-0.072`` at ``h=20``), close to the VAR shape. Structural LP estimates every horizon by OLS with its own control set and does not impose the recursive zero at impact, so its ``h=1`` entry is an unrestricted estimate of a quantity the Cholesky VAR fixes at zero — a specification check rather than a contradiction. Its bands are also markedly wider: at ``h=8`` the structural-LP interval on the output response spans roughly twice the VAR bootstrap interval, because nothing links the LP coefficients across horizons. That width is the price of robustness.

For full LP estimation details, including LP-IV, smooth LP, state-dependent LP, and LP-FEVD, see [Local Projections](@ref lp_page).

---

## Stationarity Filtering

The residual bootstrap can produce explosive draws when the companion matrix has eigenvalues near the unit circle. Setting `stationary_only=true` rejects any draw whose companion matrix has ``|\lambda_{\max}| \geq 1`` and redraws, so every retained bootstrap IRF comes from a stationary parameter configuration.

!!! note "Technical Note"
    The algorithm evaluates ``10 \times \text{reps}`` candidates and keeps the first `reps` that pass, in index order, so the bands do not depend on thread scheduling. If fewer than `reps` candidates pass, a warning reports how many were obtained.

Whether the filter binds depends entirely on the persistence of the system. The VAR above has ``|\lambda_{\max}| = 0.80`` and not one of 500 bootstrap redraws is explosive, so filtering changes only which draws are used, not their character. Re-estimating the same three series **in levels** produces a near-unit-root system where the filter does real work:

```@example ia_irf
# Levels rather than growth rates: a near-unit-root system
var_levels = estimate_var(Y_levels, 4; varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"])
unfiltered = irf(var_levels, 20; ci_type=:bootstrap, reps=50, seed=20260802)
filtered = irf(var_levels, 20; ci_type=:bootstrap, reps=50, stationary_only=true, seed=20260802)

(unfiltered_width = round(unfiltered.ci_upper[20, 1, 3] - unfiltered.ci_lower[20, 1, 3], digits=3),
 filtered_width   = round(filtered.ci_upper[20, 1, 3] - filtered.ci_lower[20, 1, 3], digits=3))
```

`estimate_var` warns that the levels system has ``|\lambda_{\max}| = 1.0006``, and 401 of 500 residual-bootstrap redraws from it are explosive. Discarding them narrows the 95% band on the ``h=20`` industrial-production response from 0.511 to 0.367, a 28% reduction, with similar gains at shorter horizons (0.407 to 0.300 at ``h=12``). The explosive draws are what generate the heavy tails: their IRFs diverge with the horizon, so they dominate the extreme quantiles. Rejecting them imposes the prior belief that the data-generating process is covariance-stationary, which is the default recommendation of Kilian & Lütkepohl (2017, Chapter 12) — and a reminder that a system this persistent belongs in a VECM (see [Vector Error Correction Models](@ref vecm_page)) rather than a levels VAR.

---

## Complete Example

This workflow combines frequentist, Bayesian, and LP-based impulse responses for the same monetary VAR, then reads selected horizons out of a table.

```@example ia_irf
# Frequentist VAR IRF with bootstrap bands
freq_irf = irf(model, 20; ci_type=:bootstrap, reps=50, conf_level=0.95, seed=20260802)
report(freq_irf)
```

```@example ia_irf
# Selected horizons for one response-shock pair
print_table(stdout, freq_irf, "INDPRO", "FEDFUNDS"; horizons=[1, 4, 8, 12, 20])
```

```@example ia_irf
# Bayesian BVAR IRF
bayes_full = irf(post, 20)
report(bayes_full)
```

```@example ia_irf
# Structural LP IRF
lp_full = irf(structural_lp(Y, 20; method=:cholesky, lags=4,
                            varnames=["INDPRO", "CPIAUCSL", "FEDFUNDS"]))
report(lp_full)
```

The three approaches agree on the shape of the transmission mechanism and disagree about how much confidence to place in it. The bootstrap interval reflects sampling variability in the OLS estimates conditional on the VAR being correctly specified. The Bayesian bands add prior information and integrate over parameter uncertainty, and here they nearly coincide with the frequentist point estimates because 250 observations dominate a diffuse prior. The structural LP bands are the widest because LP imposes no cross-horizon restrictions. Agreement in sign and shape across all three — a positive short-run output response followed by a small negative one, with the funds rate itself the only sharply estimated response — is the robustness check worth reporting; the width of the LP bands is the honest statement of what a 250-month sample can identify.

---

## Common Pitfalls

1. **Explosive bootstrap draws.** Near-unit-root VARs produce bootstrap draws with ``|\lambda_{\max}| > 1``, whose IRFs diverge and inflate the outer quantiles at long horizons. Use `stationary_only=true` to filter them, as recommended in Kilian & Lütkepohl (2017, Chapter 12). On a comfortably stationary system the filter binds on no draw and costs runtime for nothing.

2. **Cumulating pointwise quantiles.** Never cumulate the upper and lower bounds directly — it produced a band 49% too wide in the example above. `cumulative_irf()` cumulates each raw draw and then extracts quantiles whenever the draws are available, which is the only correct route.

3. **Interpreting Cholesky ordering.** Recursive identification assigns economic meaning by variable ordering. Placing the funds rate last assumes monetary policy does not affect output or prices within the month. Reversing the ordering changes the identified shocks entirely, and nothing in the output flags the change.

4. **Confusing array indexing.** `values[h, i, j]` is the response of variable ``i`` to shock ``j``, where the first index runs from 1, so `values[1, :, :]` is the impact response conventionally written ``h=0``. `print_table` labels this row `h=1` and prints the convention as a footnote.

5. **Insufficient bootstrap replications.** The default `reps=200` is adequate for a first look but not for publication-quality bands; use 1000 or more. Sign-restricted IRFs need considerably more, since accepted rotations are a subset of all draws. The examples on this page use `reps=50` to keep the documentation build fast.

6. **Bias correction does not move the point estimate.** `bias_correct=true` corrects the bootstrap draws and therefore the bands; `values` still holds the uncorrected point IRF. Compare `ci_lower`/`ci_upper` across settings, not `values`.

---

## References

- Brüggemann, Ralf, Carsten Jentsch, and Carsten Trenkler. 2016. "Inference in VARs with Conditional Heteroskedasticity of Unknown Form."
  *Journal of Econometrics*, 191(1), 69--85. [DOI](https://doi.org/10.1016/j.jeconom.2015.10.004)

- Gonçalves, Sílvia, and Lutz Kilian. 2004. "Bootstrapping Autoregressions with Conditional Heteroskedasticity of Unknown Form."
  *Journal of Econometrics*, 123(1), 89--120. [DOI](https://doi.org/10.1016/j.jeconom.2003.10.030)

- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
  *American Economic Review*, 95(1), 161--182. [DOI](https://doi.org/10.1257/0002828053828518)

- Kilian, Lutz. 1998. "Small-Sample Confidence Intervals for Impulse Response Functions."
  *Review of Economics and Statistics*, 80(2), 218--230. [DOI](https://doi.org/10.1162/003465398557465)

- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*.
  Cambridge: Cambridge University Press. [DOI](https://doi.org/10.1017/9781108164818)

- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses."
  *Econometrica*, 89(2), 955--980. [DOI](https://doi.org/10.3982/ECTA17813)

- Sims, Christopher A. 1980. "Macroeconomics and Reality."
  *Econometrica*, 48(1), 1--48. [DOI](https://doi.org/10.2307/1912017)
