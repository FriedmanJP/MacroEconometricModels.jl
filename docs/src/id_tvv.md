# [Time-Varying Volatility Identification](@id id_tvv_page)

Time-varying volatility (TVV) identification recovers the structural impact matrix ``B_0`` from the lagged cross-moments of squared shocks, without assuming any law of motion for the variances (Lewis 2021). When at least ``n - 1`` shocks have persistent, linearly independent variance paths, their squared autocovariances pin down the rotation ``Q``. The estimator is GMM on model-free moment conditions, so it stays consistent under Markov switching, stochastic volatility, or breaks — any variance process with serially dependent squares.

- **Model-free moments**: cross-autocovariances of squared shocks at lags ``1:K``; no regime or SV specification
- **GMM estimation**: one-step, Hansen two-step, or continuously updated weighting via `estimate_gmm`
- **Weak-identification screen**: a portmanteau diagnostic on squared residuals flags homoskedastic data before estimation misleads
- **J-test**: overidentifying-restrictions test rejects variance paths that carry no rotation information (e.g. common proportional shifts)

For parametric volatility models with a stated variance law, see [Heteroskedasticity](@ref id_heteroskedastic_page). For higher-moment GMM on skewness and kurtosis, see [Non-Gaussian Methods](@ref id_nongaussian_page). For the EM estimator under AR(1) log-volatility, see [Stochastic Volatility SVAR](@ref id_sv_svar_page). Restriction-based schemes live on [Structural Identification](@ref structural_identification_page).

```@setup tvv
using MacroEconometricModels, Random, LinearAlgebra
rng = Xoshiro(827)
n, Tobs = 2, 2000
rhos, sigs = [0.97, 0.85], [0.25, 0.20]
h = zeros(Tobs, n)
for t in 2:Tobs, i in 1:n
    h[t, i] = rhos[i] * h[t - 1, i] + sigs[i] * randn(rng)
end
E = randn(rng, Tobs, n) .* exp.(h ./ 2)
B0 = [1.0 0.4; -0.2 1.0]
A = [0.5 * Matrix{Float64}(I, n, n)]
Y = similar(E)
Y[1, :] = B0 * E[1, :]
for t in 2:Tobs
    Y[t, :] = A[1] * Y[t - 1, :] + B0 * E[t, :]
end
model = estimate_var(Y, 1; varnames=["y1", "y2"])
```

## Quick Start

**Recipe 1: Identify from time-varying volatility**

```@example tvv
# Persistent shock-specific variances identify B0 = L Q (Lewis 2021)
tvv = identify_lewis_tvv(model; rng=Xoshiro(828))
report(tvv)
```

**Recipe 2: Read the identification checks**

```@example tvv
# weak_id screens strength; J_pvalue tests the moment restrictions
(weak_id = tvv.weak_id, strength = round(tvv.id_strength, digits=2),
 J = round(tvv.J, digits=2), J_pvalue = round(tvv.J_pvalue, digits=3),
 converged = tvv.converged)
```

**Recipe 3: Impulse responses through the standard pipeline**

```@example tvv
ir_tvv = irf(model, 12; method=:lewis_tvv, rng=Xoshiro(828))
report(ir_tvv)
```

```julia
plot_result(ir_tvv)
```

**Recipe 4: Compare weighting schemes**

```@example tvv
# All three weightings land on the same rotation (signed-permutation
# distances to the two-step estimate); one-step reports no J p-value
# because identity weighting has no chi-square limit
Qs = [identify_lewis_tvv(model; weighting=w, n_starts=3,
                          rng=Xoshiro(829)).Q
      for w in (:one_step, :two_step, :cue)]
round.([MacroEconometricModels.q_distance(Qs[1], Q) for Q in Qs], digits=3)
```

---

## Moment Conditions

The estimator whitens the reduced-form residuals with the Cholesky factor and forms standardized shocks ``Z = U L^{-1}``. For a candidate rotation ``Q(\theta)``, the implied shocks are ``E = Z Q(\theta)`` with centered squares ``S = E^2 - 1``. Identification comes from the requirement that squared shocks be mutually unpredictable:

```math
E[(e^2_{it} - 1)(e^2_{j,t-k} - 1)] = 0, \quad i \neq j, \; k \in \mathrm{lags}
```

where:
- ``e_{it}`` is structural shock ``i`` at time ``t``
- ``i \neq j`` runs over ordered shock pairs (``n(n-1)`` conditions per lag)
- ``\mathrm{lags}`` defaults to ``1:K`` with ``K = 5``

Persistent shock-specific volatility makes squared shocks serially correlated within each shock but not across shocks at the true rotation; any wrong rotation mixes the variance paths and violates the conditions. With ``q`` moments and ``n(n-1)/2`` Givens angles, ``q`` must exceed the angle count or estimation throws.

!!! note "Technical Note"
    Lag ``0`` (contemporaneous independence) is allowed in `lags` but defaults exclude it: the ``k \geq 1`` autocovariance moments carry the TVV identifying power, while ``k = 0`` adds level conditions shared with cokurtosis GMM.

---

## Estimation

`identify_lewis_tvv` minimizes the GMM criterion through the shared `estimate_gmm` kernel: `:one_step` uses identity weighting, `:two_step` is Hansen two-step, and `:cue` iterates the weighting to convergence. The Q-space objective is non-convex, so estimation multi-starts (`n_starts`, default 10, first start at ``\theta = 0``) and keeps the lowest J-statistic. The J-statistic is ``\chi^2`` with ``q`` minus angle-count degrees of freedom under correct specification.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `K` | `Int` | `5` | Number of lags when `lags` not given |
| `lags` | `AbstractVector` | `1:K` | Lags in the moment block |
| `weighting` | `Symbol` | `:two_step` | One-step, two-step, or CUE weighting |
| `hac` | `Bool` | `true` | HAC covariance for the weighting matrix |
| `bandwidth` | `Int` | `0` | HAC bandwidth (0 selects automatically) |
| `n_starts` | `Int` | `10` | Random Givens starts (first is zero) |
| `max_iter` | `Int` | `100` | Optimizer iterations per start |
| `tol` | `Real` | `1e-8` | Optimizer tolerance |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Random starts (pass `Xoshiro` for reproducibility) |

| Field | Type | Description |
|-------|------|-------------|
| `B0` | `Matrix{T}` | ``n \times n`` structural impact matrix |
| `Q` | `Matrix{T}` | ``n \times n`` estimated rotation |
| `theta` | `Vector{T}` | Estimated Givens angles |
| `vcov` | `Matrix{T}` | Sandwich covariance of `theta` |
| `se` | `Matrix{T}` | Standard errors of `B0` (delta method) |
| `J` | `T` | J-statistic at the estimate |
| `J_pvalue` | `T` | ``\chi^2`` p-value of `J` |
| `weak_id` | `Bool` | True when the strength screen fails |
| `id_strength` | `T` | Strength statistic relative to the 99.9% null value |
| `converged` | `Bool` | Optimizer convergence (not identification) |
| `shocks` | `Matrix{T}` | Estimated structural shocks |

---

## Weak Identification

Two variance paths identify nothing, and the estimator says so through two separate channels. The `weak_id` flag screens **strength**: a portmanteau statistic on the autocorrelations of centered squared whitened residuals, calibrated against its homoskedastic null. Homoskedastic Gaussian data scores far below 1 and returns `weak_id == true` with a message naming the failure. Always inspect this flag — `converged` reports optimizer convergence only.

The J-test screens **validity**. Common proportional variance shifts pass the strength screen (a persistent scalar volatility process autocorrelates every squared shock) but carry no rotation information: the moments cannot vanish at any rotation, so the J-test rejects loudly instead of returning a silent estimate. A small `J_pvalue` on strong data means the variance paths do not support a constant-``B_0`` rotation — do not interpret that ``B_0``.

```@example tvv
# Homoskedastic data fails loud: weak flag plus message, not a silent B0
Yh = randn(Xoshiro(830), 2000, 2)
rh = identify_lewis_tvv(estimate_var(Yh, 1); n_starts=3, rng=Xoshiro(831))
(rh.weak_id, rh.message)
```

---

## Complete Example

The full pipeline on the simulated SV data recovers the rotation to `q_distance` ``0.267`` against the population rotation of the setup ``B_0``:

```@example tvv
Q = MacroEconometricModels.compute_Q(model, :lewis_tvv; rng=Xoshiro(828))
(norm(Q' * Q - I), MacroEconometricModels.q_distance(Q, [0.9285 0.3714; -0.3714 0.9285]))
```

Orthogonality holds to machine precision. The distance sits inside the recovery-table threshold (``0.3`` at ``T = 20000``) modulo the smaller ``T = 2000`` setup sample; the test suite pins the large-``T`` value.

---

## Common Pitfalls

1. **Fewer than 100 usable observations.** Estimation throws below ``T_{eff} - \max(lags) \geq 100``. Shorten `lags` or collect more data.
2. **Reading `converged` as identification.** `converged` covers the optimizer; `weak_id` covers identification. Check `weak_id` first.
3. **Proportional variance shifts.** A single market-stress episode scaling all variances together passes the strength screen but identifies no rotation — the J-test rejection is the signal. Split the sample or add a second independent variance path.
4. **Too few moments.** With ``n = 2`` and `lags=[1]`, ``q = 2`` exceeds one angle, but barely: raise `K` for stability.
5. **Forgetting `rng`.** Random starts differ across sessions; pass an explicit `Xoshiro` for reproducible tables.

---

## References

- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086--3124. [DOI](https://doi.org/10.1093/restud/rdab009)
