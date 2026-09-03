# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-11 (#800): non-Gaussian and heteroskedastic SVAR DGPs.
# Gaussian white noise leaves every non-Gaussian / heteroskedasticity-based
# identification scheme unidentified in population; these simulators give each
# method data with the feature it needs.

using Random, LinearAlgebra, Distributions

# Independent non-Gaussian structural shocks, unit variance.
function _nongauss_shocks(rng::AbstractRNG, T::Int, n::Int, dist::Symbol, nu::Float64)
    if dist === :gauss
        return randn(rng, T, n)
    elseif dist === :t
        return rand(rng, TDist(nu), T, n) ./ sqrt(nu / (nu - 2))  # standardised t
    elseif dist === :laplace
        return rand(rng, Laplace(0.0, 1 / sqrt(2)), T, n)          # Var = 1
    elseif dist === :mixture
        s = rand(rng, T, n) .< 0.5
        return @. ifelse(s, -1.5, 1.5) + randn(rng, T, n)         # bimodal, Var = 3.25
    elseif dist === :skew
        return (rand(rng, Chisq(3), T, n) .- 3) ./ sqrt(6)         # centred χ²₃
    else
        throw(ArgumentError("unknown dist :$dist (gauss|t|laplace|mixture|skew)"))
    end
end

"""
    dgp_nongaussian_var(rng; A, B0, dist, nu, T, burn) -> NamedTuple

VAR with independent non-Gaussian structural shocks
(`dist ∈ :gauss|:t|:laplace|:mixture|:skew`, default `:t` with `nu = 5`).
Returns `(Y, eps, A, Sigma, B0, dist)` with `Sigma = B0*B0'`.
"""
function dgp_nongaussian_var(rng::AbstractRNG;
                             A=[0.5 0.1 0.0; 0.2 0.4 0.1; 0.0 0.1 0.3],
                             B0=[1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 0.2 1.0],
                             dist::Symbol=:t, nu::Float64=5.0,
                             T::Int=1000, burn::Int=200)
    A1 = Matrix{Float64}(A)
    B = Matrix{Float64}(B0)
    n = size(A1, 1)
    Eps = _nongauss_shocks(rng, T + burn, n, dist, nu)
    if dist === :mixture
        Eps ./= sqrt(3.25)
    end
    Y = zeros(T + burn, n)
    y = zeros(n)
    for t in 1:(T + burn)
        y = A1 * y + B * Eps[t, :]
        Y[t, :] .= y
    end
    keep = (burn + 1):(T + burn)
    return (Y=Y[keep, :], eps=Eps[keep, :], A=[A1], Sigma=B * B', B0=B, dist=dist)
end

"""
    dgp_heteroskedastic_var(rng; A, B0, kind, Lambda, T, burn, kwargs...)
        -> NamedTuple

VAR with time-varying shock variances, `u_t = B0 * Λ_t^{1/2} * ε_t`.
`kind ∈ :markov | :garch | :smooth | :external`. `Lambda` (default
`diagm([1, 4, 0.25])`) gives genuinely distinct eigenvalue ratios, so the
heteroskedasticity carries identifying information. Returns
`(Y, eps, scales, B0, Sigma_full, path)` where `scales` is the `T×n`
variance path and `path` the regime/transition descriptor.
"""
function dgp_heteroskedastic_var(rng::AbstractRNG;
                                 A=[0.5 0.1 0.0; 0.2 0.4 0.1; 0.0 0.1 0.3],
                                 B0=[1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 0.2 1.0],
                                 kind::Symbol=:markov,
                                 Lambda=diagm([1.0, 4.0, 0.25]),
                                 T::Int=1000, burn::Int=200,
                                 P=[0.95 0.05; 0.05 0.95],
                                 garch_a::Float64=0.1, garch_b::Float64=0.85,
                                 gamma::Float64=5.0, break_at::Float64=0.5)
    A1 = Matrix{Float64}(A)
    B = Matrix{Float64}(B0)
    L = Matrix{Float64}(Lambda)
    n = size(A1, 1)
    N = T + burn
    Eps = randn(rng, N, n)
    scales = ones(N, n)
    path = zeros(Int, N)
    if kind === :markov
        s = 1
        for t in 1:N
            s = rand(rng) < P[s, 1] ? 1 : 2
            path[t] = s
            scales[t, :] .= s == 1 ? ones(n) : diag(L)
        end
    elseif kind === :garch
        h = ones(n)
        for t in 1:N
            t > 1 && (h = (1 - garch_a - garch_b) .+ garch_a .* Eps[t - 1, :] .^ 2 .+
                      garch_b .* h)
            scales[t, :] .= h
        end
    elseif kind === :smooth
        z = zeros(N)
        for t in 2:N
            z[t] = 0.9 * z[t - 1] + randn(rng)
        end
        G = @. 1 / (1 + exp(-gamma * z))
        for j in 1:n
            scales[:, j] .= @. (1 - G) * 1.0 + G * L[j, j]
        end
        path = G
    elseif kind === :external
        br = round(Int, break_at * N)
        path .= [t <= br ? 1 : 2 for t in 1:N]
        for t in 1:N
            scales[t, :] .= t <= br ? ones(n) : diag(L)
        end
    else
        throw(ArgumentError("unknown kind :$kind (markov|garch|smooth|external)"))
    end
    Y = zeros(N, n)
    y = zeros(n)
    for t in 1:N
        y = A1 * y + B * (sqrt.(scales[t, :]) .* Eps[t, :])
        Y[t, :] .= y
    end
    keep = (burn + 1):N
    return (Y=Y[keep, :], eps=Eps[keep, :], scales=scales[keep, :], B0=B,
            Sigma_full=B * B', Lambda=L, path=path[keep], kind=kind)
end
