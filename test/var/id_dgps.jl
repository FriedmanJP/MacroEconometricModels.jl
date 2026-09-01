# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
SVAR data-generating processes for identification recovery tests.
"""

using LinearAlgebra, Random, Distributions

"""Draw `nobs × n` structural shocks. `:gaussian`, `:t`, `:skewnormal`, `:mixture`."""
function _draw_structural_shocks(n::Int, nobs::Int, shocks::Symbol, rng)
    if shocks === :gaussian
        return randn(rng, nobs, n)
    elseif shocks === :t
        ν = 5.0
        return rand(rng, TDist(ν), nobs, n) .* sqrt((ν - 2) / ν)
    elseif shocks === :skewnormal
        α = 5.0
        δ = α / sqrt(1 + α^2)
        x = δ .* abs.(randn(rng, nobs, n)) .+ sqrt(1 - δ^2) .* randn(rng, nobs, n)
        x .-= mean(x, dims=1)
        return x ./ std(x, dims=1)
    elseif shocks === :mixture
        p_mix = 0.8
        σ_wide = 3.0
        Z = randn(rng, nobs, n)
        mask = rand(rng, nobs, n) .< p_mix
        return ifelse.(mask, Z, σ_wide .* Z)
    else
        throw(ArgumentError("unknown shock distribution: $shocks"))
    end
end

function simulate_svar(B0::AbstractMatrix{T}, A::AbstractVector{<:AbstractMatrix};
                       Tobs::Int=200, shocks=:gaussian, shock_ar=nothing,
                       rng=Random.default_rng()) where {T<:AbstractFloat}
    n = size(B0, 1)
    p = length(A)
    ntot = Tobs + p + 50
    ε = Matrix{T}(_draw_structural_shocks(n, ntot, shocks, rng))
    if shock_ar !== nothing
        ρ = Vector{T}(shock_ar)
        length(ρ) == n || throw(ArgumentError("shock_ar must have length n=$n"))
        η = copy(ε)
        ε[1, :] = η[1, :]
        for t in 2:ntot
            ε[t, :] = ρ .* ε[t - 1, :] .+ η[t, :]
        end
        ε ./= std(ε; dims=1)
    end
    u = ε * B0'
    Y = zeros(T, ntot, n)
    for t in (p + 1):ntot
        yt = u[t, :]
        for lag in 1:p
            yt = yt + A[lag] * Y[t - lag, :]
        end
        Y[t, :] = yt
    end
    Y[(end - Tobs + 1):end, :], ε[(end - Tobs + 1):end, :]
end

function simulate_two_regime(B0, A, Λ; Tobs=500, split=0.5, rng=Random.default_rng())
    T1 = round(Int, split * Tobs)
    Y1, _ = simulate_svar(B0, A; Tobs=T1, rng=rng)
    B2 = B0 * Diagonal(sqrt.(Λ))
    Y2, _ = simulate_svar(B2, A; Tobs=Tobs - T1, rng=rng)
    vcat(Y1, Y2), vcat(fill(1, T1), fill(2, Tobs - T1))
end

"""K-regime heteroskedastic DGP: `Σ_k = B0 Λ_k B0'` with `Λ_1 = I`.

`Lambdas` is a vector of relative-variance vectors (`Λ₂, …, Λ_K`). One
continuous VAR path with known regime labels (equal-length blocks).
"""
function simulate_k_regime(B0, A, Lambdas; Tobs=3000, rng=Random.default_rng())
    n = size(B0, 1)
    K = length(Lambdas) + 1
    p = length(A)
    Tk = fill(Tobs ÷ K, K)
    Tk[end] += Tobs - sum(Tk)
    Λs = [ones(eltype(B0), n), Lambdas...]
    regimes = Int[]
    for k in 1:K
        append!(regimes, fill(k, Tk[k]))
    end
    T = float(eltype(B0))
    burn = 50
    ntot = Tobs + p + burn
    ε = randn(rng, T, ntot, n)
    u = zeros(T, ntot, n)
    t0 = ntot - Tobs
    @inbounds for t in 1:ntot
        k = t <= t0 ? 1 : regimes[t - t0]
        u[t, :] = B0 * (sqrt.(Λs[k]) .* ε[t, :])
    end
    Y = zeros(T, ntot, n)
    for t in (p + 1):ntot
        yt = u[t, :]
        for lag in 1:p
            yt = yt + A[lag] * Y[t - lag, :]
        end
        Y[t, :] = yt
    end
    return Y[(end - Tobs + 1):end, :], regimes
end

"""Proxy SVAR DGP: `z_t = ρ ε_{1:k,t} + √(1-ρ²) v_t` with `Corr(z_j, ε_j) = ρ`."""
function simulate_proxy_svar(B0::AbstractMatrix{T}, A::AbstractVector{<:AbstractMatrix};
                             Tobs::Int=200, ρ::Real=0.6, k::Int=1,
                             rng=Random.default_rng()) where {T<:AbstractFloat}
    (0 < k <= size(B0, 1)) || throw(ArgumentError("k must be in 1:n"))
    abs(ρ) <= 1 || throw(ArgumentError("ρ must be in [-1, 1]"))
    Y, ε = simulate_svar(B0, A; Tobs=Tobs, rng=rng)
    nobs = size(ε, 1)
    σv = sqrt(max(zero(T), one(T) - T(ρ)^2))
    Z = Matrix{T}(undef, nobs, k)
    for j in 1:k
        Z[:, j] = T(ρ) .* ε[:, j] .+ σv .* randn(rng, T, nobs)
    end
    return Y, ε, k == 1 ? vec(Z) : Z
end

"""Common-trend SVEC DGP: `n=3`, cointegrating rank `r=2` (one permanent shock).

The VECM is `Δy_t = α β' y_{t-1} + u_t` with `u_t = B₀ ε_t`. Gonzalo–Ng
orthogonalisation of `(α, β, Σ)` yields a just-identified KPSW rotation:
`Ξ B₀` has two exact zero (transitory) columns.

Returns `(Y, ε, B0, Xi)`.
"""
function simulate_common_trend_svec(; Tobs::Int=1000, rng=Random.default_rng())
    T = Float64
    n = 3
    beta = T[1.0 1.0; -1.0 0.0; 0.0 -1.0]
    alpha = T[-0.30 -0.10; 0.20 0.00; 0.05 0.25]
    Sigma = T[1.0 0.2 0.1; 0.2 1.2 0.15; 0.1 0.15 0.8]
    Psi = Matrix{T}(I, n, n)
    aperp = nullspace(Matrix{T}(alpha'))
    bperp = nullspace(Matrix{T}(beta'))
    Xi = bperp * ((aperp' * Psi * bperp) \ aperp')
    G = vcat(aperp', beta')
    P = cholesky(Symmetric(G * Sigma * G')).L
    B0 = G \ Matrix{T}(P)
    lr = Xi * B0
    if lr[1, 1] < 0
        B0[:, 1] .*= -1
    end
    burn = 100
    ntot = Tobs + burn
    ε = randn(rng, T, ntot, n)
    u = ε * B0'
    Y = zeros(T, ntot, n)
    @inbounds for t in 2:ntot
        Y[t, :] = Y[t-1, :] .+ alpha * (beta' * Y[t-1, :]) .+ u[t, :]
    end
    return Y[(end - Tobs + 1):end, :], ε[(end - Tobs + 1):end, :], B0, Xi
end

"""GARCH(1,1) structural shocks with shock-specific (ω, α, β).

Default parameters give distinct persistence so Normandin–Phaneuf identification
has a unique rotation. Returns `(Y, ε)`.
"""
function simulate_garch_svar(B0::AbstractMatrix{T}, A::AbstractVector{<:AbstractMatrix};
                             Tobs::Int=1500,
                             omega::AbstractVector=T[0.05, 0.10],
                             alpha::AbstractVector=T[0.10, 0.20],
                             beta::AbstractVector=T[0.85, 0.70],
                             rng=Random.default_rng()) where {T<:AbstractFloat}
    n = size(B0, 1)
    length(omega) == n && length(alpha) == n && length(beta) == n ||
        throw(ArgumentError("GARCH parameter vectors must have length n=$n"))
    p = length(A)
    burn = 100
    ntot = Tobs + p + burn
    h = ones(T, ntot, n)
    z = randn(rng, T, ntot, n)
    ε = zeros(T, ntot, n)
    ω = Vector{T}(omega)
    α = Vector{T}(alpha)
    β = Vector{T}(beta)
    @inbounds for t in 2:ntot
        for j in 1:n
            h[t, j] = ω[j] + α[j] * ε[t - 1, j]^2 + β[j] * h[t - 1, j]
            ε[t, j] = sqrt(h[t, j]) * z[t, j]
        end
    end
    u = ε * B0'
    Y = zeros(T, ntot, n)
    for t in (p + 1):ntot
        yt = u[t, :]
        for lag in 1:p
            yt = yt + A[lag] * Y[t - lag, :]
        end
        Y[t, :] = yt
    end
    return Y[(end - Tobs + 1):end, :], ε[(end - Tobs + 1):end, :]
end

"""News-shock / max-share DGP: shock 1 is highly persistent.

`identify_max_share` on variable 1 over a long horizon recovers `q ≈ e₁`.
Returns `(Y, ε, B0, q_true)`.
"""
function simulate_news_maxshare(; Tobs::Int=2000, rng=Random.default_rng())
    n = 3
    B0 = Matrix{Float64}(I, n, n)
    A = [Diagonal([0.85, 0.30, 0.15])]
    Y, ε = simulate_svar(B0, A; Tobs=Tobs, rng=rng)
    return Y, ε, B0, [1.0, 0.0, 0.0]
end
