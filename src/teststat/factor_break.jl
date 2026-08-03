# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural break tests for factor models.

Implements three tests for structural instability in factor models:
- Breitung & Eickmeier (2011): per-series loading-break LM statistics, pooled
- Chen, Dolado & Gonzalo (2014): big break, via instability of the regression of
  the first estimated factor on the remaining ones
- Han & Inoue (2015): Loading instability with unknown break (sup-Wald)

All three treat the break date as unknown, so all three are suprema over a trimmed
grid of candidate dates and none of them is χ²: the reference distribution is
Andrews (1993)/Hansen (1997) for the two regression-based statistics and a
simulated null pool for the pooled Breitung-Eickmeier statistic (issue #583).

References:
- Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic
  factor models. Journal of Econometrics, 163(1), 71-84.
- Chen, L., Dolado, J. J., & Gonzalo, J. (2014). Detecting big structural breaks
  in large factor models. Journal of Econometrics, 180(1), 30-48.
- Han, X., & Inoue, A. (2015). Tests for parameter instability in dynamic factor
  models. Econometric Theory, 31(5), 1117-1152.
"""

# =============================================================================
# Main API
# =============================================================================

"""
    factor_break_test(X, r; method=:breitung_eickmeier) -> FactorBreakResult
    factor_break_test(fm::FactorModel; method=:breitung_eickmeier) -> FactorBreakResult
    factor_break_test(X; method=:chen_dolado_gonzalo) -> FactorBreakResult

Test for structural breaks in factor models.

# Methods
- `:breitung_eickmeier` — Breitung & Eickmeier (2011) pooled loading-stability LM test
- `:chen_dolado_gonzalo` — Chen, Dolado & Gonzalo (2014) big-break regression sup-LM test
- `:han_inoue` — Han & Inoue (2015) sup-Wald loading instability test

# Arguments
- `X`: Data matrix (T × N), observations × variables
- `r`: Number of factors (required for :breitung_eickmeier and :han_inoue; for
  :chen_dolado_gonzalo it is selected by Bai-Ng IC2 when omitted)
- `fm`: Estimated `FactorModel` (alternative to providing X and r)

# Keyword arguments
- `method`: Test method (see above)
- `nsim`, `nboot`, `seed`: control the simulated null reference of the
  `:breitung_eickmeier` pooled statistic (`nsim = 0` picks `clamp(100N, 2000, 20000)`
  draws). Ignored by the other methods. The default seed makes the p-value
  reproducible across calls.

# Returns
`FactorBreakResult{T}` with test statistic, p-value, estimated break date, and method.

# Examples
```julia
X = randn(200, 50)
result = factor_break_test(X, 3; method=:breitung_eickmeier)
result.pvalue < 0.05 && println("Reject loading stability at 5%")

# Using FactorModel dispatch
fm = estimate_factors(X, 3)
result = factor_break_test(fm; method=:han_inoue)

# Chen-Dolado-Gonzalo does not require r
result = factor_break_test(X; method=:chen_dolado_gonzalo)
```
"""
function factor_break_test(X::AbstractMatrix{T}, r::Int;
                           method::Symbol=:breitung_eickmeier,
                           kwargs...) where {T<:AbstractFloat}
    method ∈ (:breitung_eickmeier, :chen_dolado_gonzalo, :han_inoue) ||
        throw(ArgumentError("method must be :breitung_eickmeier, :chen_dolado_gonzalo, or :han_inoue; got :$method"))

    T_obs, N = size(X)
    T_obs < 30 && throw(ArgumentError("Time series too short (T=$T_obs), need at least 30 observations"))

    if method == :breitung_eickmeier
        return _breitung_eickmeier_test(X, r; kwargs...)
    elseif method == :chen_dolado_gonzalo
        return _chen_dolado_gonzalo_test(X, r)
    else  # :han_inoue
        return _han_inoue_test(X, r)
    end
end

# FactorModel dispatch
function factor_break_test(fm::FactorModel{T};
                           method::Symbol=:breitung_eickmeier,
                           kwargs...) where {T<:AbstractFloat}
    factor_break_test(fm.X, fm.r; method=method, kwargs...)
end

# Matrix-only dispatch (default to chen_dolado_gonzalo which doesn't need r)
function factor_break_test(X::AbstractMatrix{T};
                           method::Symbol=:chen_dolado_gonzalo) where {T<:AbstractFloat}
    method ∈ (:breitung_eickmeier, :chen_dolado_gonzalo, :han_inoue) ||
        throw(ArgumentError("method must be :breitung_eickmeier, :chen_dolado_gonzalo, or :han_inoue; got :$method"))

    if method ∈ (:breitung_eickmeier, :han_inoue)
        throw(ArgumentError("Method :$method requires the number of factors r. Use factor_break_test(X, r; method=:$method)"))
    end

    T_obs, N = size(X)
    T_obs < 30 && throw(ArgumentError("Time series too short (T=$T_obs), need at least 30 observations"))

    return _chen_dolado_gonzalo_test(X)
end

# Float64 fallbacks
factor_break_test(X::AbstractMatrix, r::Int; kwargs...) =
    factor_break_test(Float64.(X), r; kwargs...)
factor_break_test(X::AbstractMatrix; kwargs...) =
    factor_break_test(Float64.(X); kwargs...)

# =============================================================================
# Breitung-Eickmeier (2011) — Pooled per-series loading-stability LM test
# =============================================================================

"""
Default seed for the null reference draws of the pooled Breitung-Eickmeier test.
Fixed so repeated calls on the same data return the same p-value.
"""
const _BE_NULL_SEED = 20110711

"""
    _breitung_eickmeier_test(X, r; trimming, nsim, nboot, seed) -> FactorBreakResult

Breitung-Eickmeier (2011) pooled test for instability of the factor loadings.

For every series `i` the auxiliary regression

    x_it = λ_i'F̂_t + δ_i'F̂_t·1(t > τ) + e_it

is tested for `H₀: δ_i = 0` by the LM statistic

    LM_i(τ) = S_i(τ)'[A₁(τ)⁻¹ + A₂(τ)⁻¹]S_i(τ) / σ̂_i²,

with `S_i(τ) = Σ_{t≤τ}F̂_t·ê_it` the partial sum of the full-sample loading scores,
`A₁(τ) = Σ_{t≤τ}F̂_tF̂_t'`, `A₂(τ) = F̂'F̂ − A₁(τ)`, and `σ̂_i²` the full-sample
idiosyncratic variance. `LM_i(τ)` is algebraically the Chow-Wald statistic for
equality of the pre- and post-break loadings of series `i` and is asymptotically
`χ²(r)` at a *fixed* `τ`.

The break date is unknown here, so the series-level statistic is the supremum
`M_i = sup_τ LM_i(τ)` over the trimmed grid — which is *not* `χ²(r)`, so neither
`χ²(r)` moments nor a `χ²(N·r)` bar is a valid reference (issue #583). Instead the
null distribution of `M_i` conditional on `F̂` is obtained by simulation: `nsim`
independent `N(0,1)` series of length `T` are projected off `F̂` and run through the
same path, giving a reference pool `{M_b}`. The panel statistic is the standardized
pooled sum

    Z = (Σ_i M_i − N·μ̂) / (σ̂·√N),

`μ̂`, `σ̂` the pool mean and standard deviation, and the p-value is the upper-tail
Monte Carlo p-value of `Σ_i M_i` obtained by resampling `N` pool draws `nboot`
times (this handles the right skewness of `M_i` that a normal approximation to the
sum would miss). The reported break date is the maximizer of the pooled path
`Σ_i LM_i(τ)`; per-series suprema and dates are returned by `_be_sup_lm_path`.

Simulated size at 5% nominal is 0.023 (`T = 200, N = 60, r = 3`), 0.040 (100, 20, 2),
0.030 (300, 120, 3), 0.017 (60, 30, 2) and 0.043 (35, 8, 2) — mildly conservative,
with power 1.000 against a break in half or a quarter of the loadings. Two caveats:
the reference pool is drawn iid, so strongly autocorrelated `e_it` would leave the
test over-sized; and with near-nonstationary factors (AR(0.9)) it turns very
conservative (0.007).

A break that flips *every* loading is not an alternative this or any of the three
tests can see, and none of them reject it (0.035 here): `X = F̃Λ'` with
`F̃ = F·sign(t − τ)` fits the same data with stable loadings, so nothing is
identified.
"""
function _breitung_eickmeier_test(X::AbstractMatrix{T}, r::Int;
                                  trimming::Real=0.15, nsim::Int=0,
                                  nboot::Int=2000,
                                  seed::Integer=_BE_NULL_SEED) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)
    nsim >= 0 || throw(ArgumentError("nsim must be non-negative; got $nsim"))
    nboot > 0 || throw(ArgumentError("nboot must be positive; got $nboot"))

    # Estimate factors from the full sample; F̂'F̂/T = Iᵣ by construction
    fm = estimate_factors(X, r; standardize=true)
    F_hat = fm.factors    # T × r
    X_std = _standardize(X)

    # Full-sample loadings and residuals: Λ̂ = X'F̂(F̂'F̂)⁻¹, Ê = X − F̂Λ̂'
    FtF_inv = robust_inv(F_hat' * F_hat)
    Lambda_T = X_std' * F_hat * FtF_inv          # N × r
    E_hat = X_std - F_hat * Lambda_T'            # T × N
    sigma2 = max.(vec(sum(abs2, E_hat; dims=1)) ./ T(T_obs), T(1e-10))

    # Trimmed range: [0.15T, 0.85T]
    trim = max(round(Int, T(trimming) * T_obs), r + 1)
    t_start = trim
    t_end = T_obs - trim

    if t_start >= t_end
        # Insufficient observations for meaningful test
        return FactorBreakResult{T}(zero(T), one(T), nothing, :breitung_eickmeier,
                                    r, T_obs, N)
    end

    M, _, pooled_path = _be_sup_lm_path(F_hat, E_hat, sigma2, t_start, t_end)
    break_date = t_start + argmax(pooled_path) - 1

    # Simulated null reference for sup_τ LM(τ), conditional on the estimated factors
    n_draw = nsim > 0 ? nsim : clamp(100 * N, 2000, 20_000)
    rng = MersenneTwister(seed)
    pool = _be_null_pool(F_hat, t_start, t_end, n_draw, rng)

    mu = mean(pool)
    sd = max(std(pool), T(1e-10))
    obs = sum(M)
    stat = (obs - T(N) * mu) / (sd * sqrt(T(N)))
    pval = _be_pooled_pvalue(obs, pool, N, rng, nboot)

    FactorBreakResult{T}(stat, pval, break_date, :breitung_eickmeier, r, T_obs, N)
end

"""
    _be_sup_lm_path(F, E, sigma2, t_start, t_end) -> (M, dates, pooled)

Series-by-series sup-LM statistics for a loading break over `t_start:t_end`.

Returns the per-series suprema `M` (length `N`), their maximizing dates, and the
pooled path `Σ_i LM_i(τ)`. `A₁(τ)` and the score partial sums `S_i(τ)` are both
accumulated recursively, so the whole path costs `O(T·N·r + T·r³)` rather than one
subsample regression per candidate date.
"""
function _be_sup_lm_path(F::AbstractMatrix{T}, E::AbstractMatrix{T},
                         sigma2::AbstractVector{T},
                         t_start::Int, t_end::Int) where {T<:AbstractFloat}
    T_obs, r = size(F)
    N = size(E, 2)
    A = Matrix{T}(F' * F)
    A1 = zeros(T, r, r)
    C = zeros(T, r, N)      # C[:, i] = Σ_{s≤t} F_s·e_si
    GC = similar(C)
    f = Vector{T}(undef, r)
    M = fill(T(-Inf), N)
    dates = zeros(Int, N)
    pooled = Vector{T}(undef, t_end - t_start + 1)

    @inbounds for t in 1:t_end
        for j in 1:r
            f[j] = F[t, j]
        end
        for l in 1:r, j in 1:r
            A1[j, l] += f[j] * f[l]
        end
        for i in 1:N
            e_ti = E[t, i]
            for j in 1:r
                C[j, i] += f[j] * e_ti
            end
        end
        t < t_start && continue

        # Var(λ̂₁ − λ̂₂) = σ²(A₁⁻¹ + A₂⁻¹); the score form below is identical
        G = Matrix{T}(robust_inv(A1) + robust_inv(A - A1))
        mul!(GC, G, C)
        total = zero(T)
        for i in 1:N
            q = zero(T)
            for j in 1:r
                q += C[j, i] * GC[j, i]
            end
            lm_i = max(q / sigma2[i], zero(T))
            total += lm_i
            if lm_i > M[i]
                M[i] = lm_i
                dates[i] = t
            end
        end
        pooled[t-t_start+1] = total
    end

    return M, dates, pooled
end

"""
    _be_null_pool(F, t_start, t_end, nsim, rng) -> Vector

Draw `nsim` values of `sup_τ LM(τ)` under H₀ conditional on the estimated factors
`F`: each draw is an `N(0,1)` series projected off `F` (mirroring the way the real
residuals are orthogonal to `F` by construction) and pushed through
[`_be_sup_lm_path`](@ref). Generated in column chunks to bound memory.
"""
function _be_null_pool(F::AbstractMatrix{T}, t_start::Int, t_end::Int,
                       nsim::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    T_obs = size(F, 1)
    FtF_inv = robust_inv(F' * F)
    pool = Vector{T}(undef, nsim)
    chunk = 2000
    done = 0
    while done < nsim
        b = min(chunk, nsim - done)
        U = randn(rng, T, T_obs, b)
        U .-= F * (FtF_inv * (F' * U))
        s2 = max.(vec(sum(abs2, U; dims=1)) ./ T(T_obs), T(1e-10))
        M_b, _, _ = _be_sup_lm_path(F, U, s2, t_start, t_end)
        copyto!(view(pool, (done+1):(done+b)), M_b)
        done += b
    end
    return pool
end

"""
    _be_pooled_pvalue(obs, pool, N, rng, nboot) -> p

Upper-tail Monte Carlo p-value for the pooled sum `Σ_i M_i`: resample `N` draws
(with replacement) from the null `pool` `nboot` times and compare their sums to
`obs`. Uses the `(1 + #{≥})/(nboot + 1)` convention, so `p` is never exactly zero.
"""
function _be_pooled_pvalue(obs::T, pool::Vector{T}, N::Int,
                           rng::AbstractRNG, nboot::Int) where {T<:AbstractFloat}
    B = length(pool)
    count = 0
    @inbounds for _ in 1:nboot
        s = zero(T)
        for _ in 1:N
            s += pool[rand(rng, 1:B)]
        end
        s >= obs && (count += 1)
    end
    return T((count + 1) / (nboot + 1))
end

# =============================================================================
# Chen-Dolado-Gonzalo (2014) — Big break in the loadings, regression-based
# =============================================================================

"""
    _chen_dolado_gonzalo_test(X, r; trimming=0.15) -> FactorBreakResult

Chen-Dolado-Gonzalo (2014) regression-based test for a big break in the loadings.

A big break in `Λ` inflates the number of principal components needed to span the
common space, so the extra estimated factor is a *mixture* of the pre- and
post-break factor spaces. CDG exploit this: regress the first estimated factor on
the remaining ones,

    F̂_1t = c + β'F̂_{2:r,t} + u_t,

whose coefficients are constant (and zero, PCA factors being orthogonal in the full
sample) under `H₀`, but shift at the break date under `H₁`. The test is therefore a
sup-Wald/LM test for parameter instability in that regression over the trimmed grid
`π ∈ [trimming, 1−trimming]`, with `p = r` parameters (intercept plus `r−1` slopes)
subject to break, and the break date is the maximizer. With `r = 1` there are no
other factors and the regression degenerates to a test for a mean shift in `F̂_1`.

The p-value comes from the same Hansen (1997) sup-Wald tables used by
[`andrews_test`](@ref) (`_andrews_pvalue`, `p = r`), the correct null reference for
a supremum over unknown break dates. The previous implementation compared a scaled
maximum of eigenvalue-ratio *differences* — not a quadratic form — against
`χ²(r_max)`, which rejected essentially always (issue #583).

Estimated factors are serially correlated, so the statistic needs a HAC long-run
variance of the moments `Z_t·u_t`; see [`_sup_lm_hac`](@ref) for the bandwidth
choice, which drives both size and power here.

Whether to supply `r` is a real trade-off, not a formality. Under a big break the
full-sample Bai-Ng criterion inflates `r̂` (≈ 6 on an `r = 3` panel whose loadings
break) — that inflation is the mechanism the test exploits, but each extra regressor
also raises the sup-Wald critical value. Neither call dominates; at 5% on `T = 200,
N = 60, r = 3` panels breaking at `T/2` (200 reps):

| share of series that break | `r = 3` supplied | no `r` (IC2) |
|---|---|---|
| half the panel | 0.690 | 0.405 |
| a quarter of the panel | 0.150 | 0.455 |

Both calls hold their size (0.043); IC2 recovers `r̂ = 3.01` on average under H₀.

Size is 0.04 with iid factors and 0.043 with AR(0.7) ones, but 0.113 with AR(0.9)
factors: near-nonstationary factors are the one regime where the HAC correction
does not keep up, and the test over-rejects there.
"""
function _chen_dolado_gonzalo_test(X::AbstractMatrix{T}, r::Union{Int,Nothing}=nothing;
                                   trimming::Real=0.15) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Number of factors: as supplied, else Bai-Ng IC2 over the usual r_max grid
    r_use = r === nothing ? _cdg_select_r(X) : r
    validate_factor_inputs(T_obs, N, r_use)

    fm = estimate_factors(X, r_use; standardize=true)
    F_hat = fm.factors    # T × r, F̂'F̂/T = Iᵣ

    y = F_hat[:, 1]
    Z = r_use >= 2 ? hcat(ones(T, T_obs), F_hat[:, 2:r_use]) :
                     reshape(ones(T, T_obs), T_obs, 1)
    k = size(Z, 2)

    stat, break_date = _sup_lm_hac(y, Z, T(trimming))

    if break_date === nothing
        # No valid candidate dates after trimming
        return FactorBreakResult{T}(zero(T), one(T), nothing, :chen_dolado_gonzalo,
                                    r_use, T_obs, N)
    end

    pval = _andrews_pvalue(stat, k, :sup, T)

    FactorBreakResult{T}(stat, pval, break_date, :chen_dolado_gonzalo, r_use, T_obs, N)
end

"""
    _cdg_select_r(X) -> Int

Number of factors for the CDG test when the caller does not supply one: Bai-Ng
(2002) IC2 over `r = 1, …, min(⌊√min(T,N)⌋, 10)`, the same `r_max` rule the previous
eigenvalue-ratio implementation used.
"""
function _cdg_select_r(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    T_obs, N = size(X)
    r_max = clamp(floor(Int, sqrt(min(T_obs, N))), 1, 10)
    r_max == 1 && return 1
    return ic_criteria(X, r_max).r_IC2
end

"""
    _sup_lm_hac(y, Z, trimming; bandwidth=:nw94) -> (stat, break_date)

Sup-LM statistic for a break in the coefficients of `y = Zβ + u` at an unknown date.

With `h_t = Z_t·û_t` built from the *null-restricted* (full-sample) residuals,
`S(τ) = T^{-1/2}Σ_{t≤τ}h_t` and `Ω̂` a Newey-West long-run variance of `h_t`,

    LM(π) = S(τ)'Ω̂⁻¹S(τ) / (π(1−π)),   π = τ/T,

maximized over the trimmed grid. This is the score form of the Wald statistic
comparing pre- and post-break OLS estimates (`β̂₁ − β̂₂ = −M⁻¹S(τ)/(√T·π(1−π))`,
`M = Z'Z/T`) and its limit is Andrews' (1993) sup-Wald distribution with
`p = size(Z, 2)`, so `_andrews_pvalue(·, p, :sup)` applies. `break_date` is
`nothing` when trimming leaves no candidate date.

A HAC variance is needed because estimated factors are serially correlated; with a
homoskedastic-iid variance the test over-rejects badly. The bandwidth rule matters
more than usual, because `Ω̂` sits in the denominator of a supremum. Measured on
`T = 200, N = 60, r = 3` panels (300/200 reps, 5% nominal, break = half the loadings
flipping sign at `T/2`):

| bandwidth | size | power |
|---|---|---|
| `:andrews` (Andrews 1991 plug-in) | 0.020 | 0.415 |
| `:nw94` (Newey-West 1994) | 0.043 | 0.690 |
| fixed 8 | 0.027 | 0.760 |
| fixed 4 | 0.150 | 0.905 |
| fixed 2 | 0.390 | 0.945 |

`:nw94` is the default: the Andrews plug-in over-smooths these product moments
(mean bandwidth ≈ 9.6) and leaves the test conservative, while short fixed
bandwidths buy power with badly inflated size. Estimating `Ω̂(τ)` under the
*alternative* at each candidate date was also tried and is a trap — the supremum
then selects dates where the denominator happens to be small, and size rose to 0.55.
"""
function _sup_lm_hac(y::AbstractVector{T}, Z::AbstractMatrix{T},
                     trimming::T; bandwidth=:nw94) where {T<:AbstractFloat}
    n, k = size(Z)
    t1 = max(k + 1, ceil(Int, trimming * n))
    t2 = min(n - k, floor(Int, (one(T) - trimming) * n))
    t1 <= t2 || return (zero(T), nothing)

    beta = robust_inv(Z' * Z) * (Z' * y)
    u = y - Z * beta
    H = Z .* u                                     # n × k moments h_t = Z_t·u_t
    Omega = Matrix{T}(lrvar(H; demean=false, bandwidth=bandwidth))
    Omega_inv = Matrix{T}(robust_inv(Omega))

    S = zeros(T, k)
    scale = one(T) / sqrt(T(n))
    stat = zero(T)
    break_date = t1
    @inbounds for t in 1:t2
        for j in 1:k
            S[j] += H[t, j] * scale
        end
        t < t1 && continue
        pi_t = T(t) / T(n)
        lm_t = dot(S, Omega_inv * S) / (pi_t * (one(T) - pi_t))
        if lm_t > stat
            stat = lm_t
            break_date = t
        end
    end
    return (max(stat, zero(T)), break_date)
end

# =============================================================================
# Han-Inoue (2015) — Loading instability sup-Wald
# =============================================================================

function _han_inoue_test(X::AbstractMatrix{T}, r::Int) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    # Estimate factors from full sample
    fm = estimate_factors(X, r; standardize=true)
    F_hat = fm.factors    # T × r
    X_std = _standardize(X)

    # Trimmed range: [0.15T, 0.85T]
    trim = max(round(Int, 0.15 * T_obs), r + 1)
    t_start = trim
    t_end = T_obs - trim

    if t_start >= t_end
        return FactorBreakResult{T}(zero(T), one(T), nothing, :han_inoue,
                                    r, T_obs, N)
    end

    # For each candidate break date, compute sum of individual Wald statistics
    wald_path = Vector{T}(undef, t_end - t_start + 1)

    # Precompute full-sample quantities for each unit
    # Loading regression: X_i = F_hat * lambda_i + e_i
    FtF = F_hat' * F_hat  # r × r
    FtF_inv = robust_inv(FtF)

    for (idx, t) in enumerate(t_start:t_end)
        F1 = F_hat[1:t, :]           # t × r
        F2 = F_hat[(t+1):T_obs, :]   # (T-t) × r

        F1tF1 = F1' * F1             # r × r
        F2tF2 = F2' * F2             # r × r
        F1tF1_inv = robust_inv(F1tF1)
        F2tF2_inv = robust_inv(F2tF2)

        W_t = zero(T)
        for i in 1:N
            x_i = X_std[:, i]   # T × 1

            # Subsample loadings
            lambda1 = F1tF1_inv * (F1' * x_i[1:t])        # r × 1
            lambda2 = F2tF2_inv * (F2' * x_i[(t+1):T_obs]) # r × 1

            # Full-sample residual variance for unit i
            lambda_full = FtF_inv * (F_hat' * x_i)
            resid_i = x_i - F_hat * lambda_full
            sigma2_i = max(sum(resid_i .^ 2) / T_obs, T(1e-10))

            # Wald statistic for H0: lambda1 = lambda2
            diff_lambda = lambda1 - lambda2
            # Variance of difference: sigma2_i * (inv(F1'F1) + inv(F2'F2))
            V_diff = sigma2_i * (F1tF1_inv + F2tF2_inv)
            V_diff_inv = robust_inv(V_diff)

            W_i = diff_lambda' * V_diff_inv * diff_lambda
            W_t += max(W_i, zero(T))
        end

        wald_path[idx] = W_t / N
    end

    # Sup-Wald statistic
    stat = maximum(wald_path)
    break_idx = argmax(wald_path)
    break_date = t_start + break_idx - 1

    # P-value: Andrews (1993) sup-Wald distribution with df = r
    pval = _han_inoue_pvalue(stat, r)

    FactorBreakResult{T}(stat, pval, break_date, :han_inoue, r, T_obs, N)
end

"""
Approximate p-value for Han-Inoue sup-Wald statistic using
Hansen (1997) / Andrews (1993) critical values.
"""
function _han_inoue_pvalue(stat::T, k::Int) where {T<:AbstractFloat}
    # Use HANSEN_ANDREWS_CV table for k = 1,...,10
    k_clamped = clamp(k, 1, 10)
    cv = HANSEN_ANDREWS_CV[k_clamped]

    # Interpolate p-value from critical values at 1%, 5%, 10%
    cv1  = T(cv[1])   # 1% critical value
    cv5  = T(cv[5])   # 5% critical value
    cv10 = T(cv[10])  # 10% critical value

    if stat >= cv1
        # Beyond 1% CV — use chi-squared tail for extrapolation
        pval = ccdf(Chisq(k), stat)
        return clamp(pval, zero(T), T(0.01))
    elseif stat >= cv5
        # Between 1% and 5%: linear interpolation
        frac = (stat - cv5) / max(cv1 - cv5, T(1e-10))
        return T(0.05) - frac * T(0.04)
    elseif stat >= cv10
        # Between 5% and 10%: linear interpolation
        frac = (stat - cv10) / max(cv5 - cv10, T(1e-10))
        return T(0.10) - frac * T(0.05)
    else
        # Below 10% CV: use chi-squared for approximate large p-value
        pval = ccdf(Chisq(k), stat)
        return clamp(pval, T(0.10), one(T))
    end
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::FactorBreakResult{T}) where {T}
    method_label = Dict(
        :breitung_eickmeier => "Breitung-Eickmeier (2011) Pooled LM",
        :chen_dolado_gonzalo => "Chen-Dolado-Gonzalo (2014) Regression Sup-LM",
        :han_inoue => "Han-Inoue (2015) Sup-Wald",
    )
    label = get(method_label, r.method, string(r.method))

    spec_data = Any[
        "H₀"            "Factor loadings are stable";
        "H₁"            "Structural break in factor loadings";
        "Method"         label;
        "Factors"        r.n_factors;
        "Variables (N)"  r.n_vars;
        "Time (T)"       r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Factor Model Structural Break Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars = _significance_stars(r.pvalue)
    if r.break_date !== nothing
        results_data = Any[
            "Test statistic" string(round(r.statistic, digits=4), " ", stars);
            "P-value" _format_pvalue(r.pvalue);
            "Break date (index)" r.break_date
        ]
    else
        results_data = Any[
            "Test statistic" string(round(r.statistic, digits=4), " ", stars);
            "P-value" _format_pvalue(r.pvalue)
        ]
    end
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    reject = r.pvalue < 0.05
    conclusion = if reject && r.break_date !== nothing
        "Reject H₀ at 5% level: evidence of loading instability at observation $(r.break_date)"
    elseif reject
        "Reject H₀ at 5% level: evidence of loading instability"
    else
        "Fail to reject H₀: factor loadings appear stable"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end
