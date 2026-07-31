# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — residual resampling schemes and Kilian bias correction ([T271])
#
# The VAR IRF bands resampled residuals i.i.d., which is valid only under conditional
# homoskedasticity and independence. Two standard alternatives are added here, plus the
# bias-correction step that the bootstrap docstring had been claiming ([T060] relabelled it as
# "not the Kilian (1998) bias-corrected bootstrap" and deferred the implementation to here).
#
# References:
#   Kilian, L. (1998). Small-sample confidence intervals for impulse response functions.
#     Review of Economics and Statistics 80(2), 218-230.
#   Gonçalves, S. & Kilian, L. (2004). Bootstrapping autoregressions with conditional
#     heteroskedasticity of unknown form. Journal of Econometrics 123(1), 89-120.
#   Brüggemann, R., Jentsch, C. & Trenkler, C. (2016). Inference in VARs with conditional
#     heteroskedasticity of unknown form. Journal of Econometrics 191(1), 69-85.
#   Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional linear models.
#     Annals of Statistics 21(1), 255-285.

"""
    _default_block_length(T_eff) → Int

Moving-block length when the caller does not pick one: `⌈T^{1/3}⌉`, the standard rate for
block-bootstrap consistency under weak dependence (Brüggemann et al. 2016), clamped to at
least 1 and at most `T_eff`.
"""
_default_block_length(T_eff::Int) = clamp(ceil(Int, cbrt(T_eff)), 1, max(T_eff, 1))

"""
    _wild_weights(rng, m, dist, ::Type{T}) → Vector{T}

`m` scalar multipliers for the wild bootstrap.

- `:rademacher` — `±1` with probability ½ each. Mean 0, variance 1, and symmetric, so it
  matches the first two moments of the residual and kills odd moments.
- `:mammen` — Mammen's (1993) two-point distribution, which additionally matches the THIRD
  moment (`E[w³] = 1`), at the cost of asymmetry. Preferred when residual skewness matters.
"""
function _wild_weights(rng::AbstractRNG, m::Int, dist::Symbol, ::Type{T}) where {T<:AbstractFloat}
    if dist === :rademacher
        return T[rand(rng, Bool) ? one(T) : -one(T) for _ in 1:m]
    elseif dist === :mammen
        sq5 = sqrt(T(5))
        lo = -(sq5 - 1) / 2          # ≈ -0.618
        hi = (sq5 + 1) / 2           # ≈  1.618
        p_lo = (sq5 + 1) / (2 * sq5) # probability of the LOW value
        return T[rand(rng, T) < p_lo ? lo : hi for _ in 1:m]
    end
    throw(ArgumentError("wild_dist must be :rademacher or :mammen; got :$dist"))
end

"""
    _resample_residuals(U, scheme, rng; block_length=0, wild_dist=:rademacher) → Matrix

Draw a bootstrap residual sample from the `T_eff × n` residual matrix `U`.

- `:iid` — resample rows with replacement. Valid under i.i.d. errors; this is the historical
  behaviour and remains the default.
- `:wild` — multiply each residual ROW by one scalar draw. Because the whole row shares the
  multiplier, the contemporaneous cross-equation covariance is preserved exactly while the
  conditional variance is randomised — that is what makes it robust to conditional
  heteroskedasticity of unknown form (Gonçalves & Kilian 2004).
- `:block` — moving-block: concatenate `⌈T_eff/ℓ⌉` contiguous blocks of length `ℓ` drawn
  uniformly at random, then truncate to `T_eff` rows. Contiguity retains serial dependence
  within a block, which i.i.d. resampling destroys.

Every scheme returns exactly `T_eff` rows, so the caller's downstream code is unchanged.
"""
function _resample_residuals(U::AbstractMatrix{T}, scheme::Symbol, rng::AbstractRNG;
                             block_length::Int=0,
                             wild_dist::Symbol=:rademacher) where {T<:AbstractFloat}
    T_eff = size(U, 1)
    T_eff == 0 && return copy(U)
    if scheme === :iid
        return U[rand(rng, 1:T_eff, T_eff), :]
    elseif scheme === :wild
        return U .* _wild_weights(rng, T_eff, wild_dist, T)
    elseif scheme === :block
        ell = block_length > 0 ? min(block_length, T_eff) : _default_block_length(T_eff)
        n_blocks = cld(T_eff, ell)
        last_start = T_eff - ell + 1
        out = Matrix{T}(undef, n_blocks * ell, size(U, 2))
        @inbounds for b in 1:n_blocks
            s = rand(rng, 1:last_start)
            out[((b-1)*ell+1):(b*ell), :] = U[s:(s+ell-1), :]
        end
        return out[1:T_eff, :]
    end
    throw(ArgumentError("bootstrap must be :iid, :wild, or :block; got :$scheme"))
end

"""
    _kilian_bias_correction(B, Psi, n, p) → (B_corrected, delta)

Apply the Kilian (1998) bias correction `B − δ·Ψ` with the **stationarity adjustment**.

Subtracting the raw bias estimate can push the companion matrix outside the unit circle, which
would make the bias-corrected model explosive and the bands meaningless. Kilian's rule:

1. If the uncorrected estimate is already non-stationary, apply NO correction (`δ = 0`) — the
   bias formula is derived for the stationary case and correcting here would compound the
   problem.
2. Otherwise start at `δ = 1` and shrink by `0.01` until the corrected companion is stable.

Returns the corrected coefficients and the `δ` actually used, so callers can report how much of
the correction survived.
"""
function _kilian_bias_correction(B::AbstractMatrix{T}, Psi::AbstractMatrix{T},
                                 n::Int, p::Int) where {T<:AbstractFloat}
    rho(M) = maximum(abs.(eigvals(companion_matrix(M, n, p))); init=zero(T))
    rho(B) >= one(T) && return (copy(B), zero(T))     # rule 1
    delta = one(T)
    while delta > zero(T)
        Bc = B - delta * Psi
        rho(Bc) < one(T) && return (Bc, delta)
        delta -= T(0.01)
    end
    return (copy(B), zero(T))
end

"""
    _estimate_var_bias(model, reps, scheme, rng; block_length, wild_dist) → Psi

Inner-bootstrap estimate of the small-sample bias of the VAR OLS coefficients:
`Ψ = E[B*] − B̂`, with `B*` re-estimated from data regenerated at `B̂`.

This is the first stage of Kilian's bootstrap-after-bootstrap. Draws are seeded from `rng`
up front and written to fixed slots, so the estimate does not depend on thread scheduling
([T144]).
"""
function _estimate_var_bias(model::VARModel{T}, reps::Int, scheme::Symbol, rng::AbstractRNG;
                            block_length::Int=0,
                            wild_dist::Symbol=:rademacher) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    U, T_eff = model.U, size(model.U, 1)
    Y_init = model.Y[1:p, :]
    acc = [zeros(T, size(model.B)) for _ in 1:reps]
    ok = fill(false, reps)
    seeds = rand(rng, UInt64, reps)
    Threads.@threads for r in 1:reps
        local_rng = Random.MersenneTwister(seeds[r])
        _suppress_warnings() do
            U_boot = _resample_residuals(U, scheme, local_rng;
                                         block_length=block_length, wild_dist=wild_dist)
            Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
            m = estimate_var(Y_boot, p; check_stability=false)
            if size(m.B) == size(model.B) && all(isfinite, m.B)
                acc[r] = m.B
                ok[r] = true
            end
        end
    end
    kept = findall(ok)
    isempty(kept) && return zeros(T, size(model.B))
    return sum(acc[kept]) / length(kept) - model.B
end

"""
    _apply_boot_bias_correction(m, Psi, n, p, active) → VARModel

Second stage of Kilian's bootstrap-after-bootstrap: correct a single bootstrap re-estimate by
the SAME bias `Ψ` that re-centred the DGP, with the stationarity shrinkage applied per draw.

Returns `m` untouched when `active` is false, so the default path allocates nothing extra.
"""
function _apply_boot_bias_correction(m::VARModel{T}, Psi::AbstractMatrix{T},
                                     n::Int, p::Int, active::Bool) where {T<:AbstractFloat}
    active || return m
    size(m.B) == size(Psi) || return m
    Bc, _ = _kilian_bias_correction(m.B, Psi, n, p)
    return VARModel(m.Y, p, Bc, m.U, m.Sigma, m.aic, m.bic, m.hqic, m.varnames)
end
