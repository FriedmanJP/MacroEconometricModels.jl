# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790): analytic truth helpers living next to the simulators —
# closed-form moments a recovery assertion compares against.

using LinearAlgebra, Distributions

"""
    arma_spectrum(phi, theta, sigma, freqs) -> Vector

ARMA spectral density `S(ω) = σ²/2π · |1+Σθe^{−iω}|² / |1−Σφe^{−iω}|²`.
"""
function arma_spectrum(phi::AbstractVector, theta::AbstractVector,
                       sigma::Real, freqs::AbstractVector)
    ph, th = Vector{Float64}(phi), Vector{Float64}(theta)
    # No @. here: sum() over a generator must not be dotted.
    out = Vector{Float64}(undef, length(freqs))
    for (i, w) in enumerate(freqs)
        ar = 1 - sum(ph[j] * exp(-im * j * w) for j in eachindex(ph); init=0.0im)
        ma = 1 + sum(th[j] * exp(-im * j * w) for j in eachindex(th); init=0.0im)
        out[i] = sigma^2 / (2pi) * abs2(ma) / abs2(ar)
    end
    return out
end

"""
    mm_aggregate(F, Lambda_Q; weights=[1,2,3,2,1]) -> Matrix

Mariano–Murasawa aggregation of a monthly factor path into the quarterly
signal `Λ_Q (F_t + 2F_{t-1} + 3F_{t-2} + 2F_{t-3} + F_{t-4})`.
"""
function mm_aggregate(F::AbstractMatrix, Lambda_Q::AbstractMatrix;
                      weights=[1.0, 2.0, 3.0, 2.0, 1.0])
    T = size(F, 1)
    out = zeros(T, size(Lambda_Q, 1))
    for t in 5:T
        Fp = sum(weights[l] * F[t - l + 1, :] for l in 1:5)
        out[t, :] = Lambda_Q * Fp
    end
    return out
end

"""
    logit_ame(X, beta) / probit_ame(X, beta) -> Vector

Closed-form average marginal effects at `(X, beta)`:
`mean(pdf ⋅ β)` with logistic / Normal pdf.
"""
function logit_ame(X::AbstractMatrix, beta::AbstractVector)
    be = Vector{Float64}(beta)
    p = 1 ./ (1 .+ exp.(-(X * be)))
    return mean(p .* (1 .- p)) .* be
end

function probit_ame(X::AbstractMatrix, beta::AbstractVector)
    be = Vector{Float64}(beta)
    return mean(pdf.(Normal(), X * be)) .* be
end
