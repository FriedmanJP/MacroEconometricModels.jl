# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# CCR — Canonical Cointegrating Regression, Park (1992)
# =============================================================================

# Park's CCR transforms the data so that plain OLS on the transformed system yields the
# efficient, endogeneity/serial-correlation-free long-run coefficients. With the stacked
# process ξ_t = (u_t, Δx_t')', contemporaneous covariance Σ̂ = Γ̂₀, two-sided Ω̂ and one-sided
# Λ̂ = Σ_{j≥0} Γ_j, partition Ω̂ = [[ω₁₁, ω₁₂],[ω₂₁, Ω₂₂]] and take the v-column block
# Λ̂₂ = Λ̂[:, v]:
#     x*_t = x_t − (Σ̂⁻¹ Λ̂₂)' ξ_t
#     y*_t = y_t − (Σ̂⁻¹ Λ̂₂ β̂ + [0; Ω̂₂₂⁻¹ ω̂₂₁])' ξ_t
# then OLS of y* on [D, x*]. SEs use ω̂_{u·Δx} = ω₁₁ − ω₁₂ Ω̂₂₂⁻¹ ω̂₂₁ and (Z*'Z*)⁻¹.

function _estimate_ccr(y::Vector{T}, X::Matrix{T}, trend::Symbol,
                       kernel::Symbol, bandwidth) where {T<:AbstractFloat}
    n, k = size(X)
    D, dnames = _cointreg_deter(n, trend, T)
    d = size(D, 2)
    Z = hcat(D, X)

    # Stage 1: OLS on levels.
    theta_ols = robust_inv(Z' * Z) * (Z' * y)
    beta_ols = theta_ols[(d + 1):(d + k)]            # slope block (k-vector)
    u_ols = y .- Z * theta_ols

    # Stacked (u, Δx), aligned to t = 2:T.
    u_lag = u_ols[2:n]
    Xdelta = diff(X; dims=1)                          # (T-1)×k
    U = hcat(u_lag, Xdelta)                           # (T-1)×(1+k)

    Ω, Λ, Σ = _cointreg_lrv(U; kernel=kernel, bandwidth=bandwidth)
    bw = _resolved_bw(U, kernel, bandwidth)

    vidx = 2:(k + 1)
    Ω_uv = reshape(Ω[1, vidx], 1, k)
    Ω_vu = reshape(Ω[vidx, 1], k, 1)
    Ω_vv = Ω[vidx, vidx]
    Ωvv_inv = robust_inv(Ω_vv)
    ω_uv = Ω[1, 1] - (Ω_uv*Ωvv_inv*Ω_vu)[1, 1]        # conditional LR variance

    # Park transformation matrices.
    Λ2 = Λ[:, vidx]                                    # (1+k)×k
    A = robust_inv(Σ) * Λ2                             # (1+k)×k
    ξ = U                                              # (T-1)×(1+k), rows = ξ_t, t=2:T
    # x*_t = x_t − ξ_t A   (row-wise);  A' ξ_t is k-vector
    Xstar = X[2:n, :] .- ξ * A                         # (T-1)×k
    # y*_t = y_t − ξ_t (A β̂ + [0; Ω₂₂⁻¹ ω₂₁])
    bvec = A * beta_ols .+ vcat(zeros(T, 1), vec(Ωvv_inv * Ω_vu))   # (1+k)
    ystar = y[2:n] .- ξ * bvec                         # (T-1)

    Dstar = D[2:n, :]
    Zstar = hcat(Dstar, Xstar)                         # (T-1)×(d+k)
    ZstZ = Symmetric(Zstar' * Zstar)
    ZstZ_inv = robust_inv(ZstZ)
    theta_ccr = vec(Matrix{T}(ZstZ_inv) * (Zstar' * ystar))

    vcov = Matrix{T}(ω_uv .* ZstZ_inv)
    fitted = Z * theta_ccr
    resid = y .- fitted
    varnames = vcat(dnames, _cointreg_xnames(k))

    return CointRegModel{T}(y, X, :ccr, trend, kernel, bw, theta_ccr, vcov,
                            resid, fitted, varnames, n, 0, 0, Ω, Λ, Σ, ω_uv, d, k)
end
