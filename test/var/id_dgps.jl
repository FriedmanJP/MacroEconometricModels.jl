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
        return δ .* abs.(randn(rng, nobs, n)) .+ sqrt(1 - δ^2) .* randn(rng, nobs, n)
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
                       Tobs::Int=200, shocks=:gaussian, rng=Random.default_rng()) where {T<:AbstractFloat}
    n = size(B0, 1)
    p = length(A)
    ε = Matrix{T}(_draw_structural_shocks(n, Tobs + p + 50, shocks, rng))
    u = ε * B0'
    Y = zeros(T, Tobs + p + 50, n)
    for t in (p + 1):(size(Y, 1))
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
