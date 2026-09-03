# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-10 (#799): one GARCH-family simulator plus SV,
# multivariate-GARCH and MIDAS simulators. Every simulator returns the
# conditional-variance path the old fixtures discarded.

using Random, LinearAlgebra, Distributions

"""
    dgp_garch_family(rng; kind, omega, alpha, beta, gamma, delta, d, theta,
                     mu, innov, nu, T, burn) -> NamedTuple

GARCH-family simulator, `kind ∈ :arch|:garch|:egarch|:gjr|:aparch|:igarch|`
`:cgarch|:figarch|:fiegarch`. `innov ∈ :gauss|:t|:laplace` (standardised).
Unconditional variance `ω/(1−α−β)` holds for the stationary kinds.
Returns `(y, h, eps)` — returns, conditional variances, standardised shocks.
"""
function dgp_garch_family(rng::AbstractRNG; kind::Symbol=:garch,
                          omega::Float64=0.02, alpha::Float64=0.08,
                          beta::Float64=0.88, gamma::Float64=0.06,
                          delta::Float64=1.5, d::Float64=0.4,
                          theta::Float64=-0.05, mu::Float64=0.0,
                          innov::Symbol=:gauss, nu::Float64=8.0,
                          T::Int=3000, burn::Int=500)
    N = T + burn
    z = if innov === :gauss
        randn(rng, N)
    elseif innov === :t
        rand(rng, TDist(nu), N) ./ sqrt(nu / (nu - 2))
    elseif innov === :laplace
        rand(rng, Laplace(0.0, 1 / sqrt(2)), N)
    else
        throw(ArgumentError("unknown innov :$innov (gauss|t|laplace)"))
    end
    y, h = zeros(N), zeros(N)
    h0 = omega / max(0.05, 1 - alpha - beta)
    # All recursions floor the variance at 1e-8 (standard truncation: the
    # component-GARCH transient can otherwise dip below zero after a spike).
    if kind === :arch || kind === :garch || kind === :igarch || kind === :gjr
        h[1] = h0
        y[1] = mu + sqrt(h[1]) * z[1]
        for t in 2:N
            v = omega + alpha * (y[t - 1] - mu)^2 + beta * h[t - 1]
            kind === :gjr && (v += gamma * (y[t - 1] < mu) * (y[t - 1] - mu)^2)
            h[t] = max(v, 1e-8)
            y[t] = mu + sqrt(h[t]) * z[t]
        end
    elseif kind === :egarch
        lnh = log(h0)
        for t in 1:N
            h[t] = exp(lnh)
            y[t] = mu + sqrt(h[t]) * z[t]
            lnh = omega + alpha * (abs(z[t]) - sqrt(2 / pi)) +
                  gamma * z[t] + beta * lnh
        end
    elseif kind === :aparch
        s = h0^(delta / 2)
        for t in 1:N
            h[t] = s^(2 / delta)
            y[t] = mu + sqrt(h[t]) * z[t]
            s = omega + alpha * (abs(y[t] - mu) - gamma * (y[t] - mu))^delta +
                beta * s
        end
    elseif kind === :cgarch
        rho_p, phi_t = 0.99, 0.7  # permanent / transitory persistence
        q = h0
        h[1] = h0
        y[1] = mu + sqrt(h[1]) * z[1]
        for t in 2:N
            q = omega + rho_p * (q - omega) + phi_t * ((y[t - 1] - mu)^2 - h[t - 1])
            h[t] = max(q + alpha * ((y[t - 1] - mu)^2 - q) + beta * (h[t - 1] - q),
                       1e-8)
            y[t] = mu + sqrt(h[t]) * z[t]
        end
    elseif kind === :figarch || kind === :fiegarch
        # Truncated long-memory ARCH(∞): hyperbolic weights w_k ∝ k^-(1+d)
        # (right tail index of the FIGARCH expansion; summably truncated).
        trunc = 1000
        w = @. 1 / (1:trunc)^(1 + d)
        w ./= sum(w) * 2
        h[1] = h0
        y[1] = mu + sqrt(h[1]) * z[1]
        for t in 2:N
            m = min(t - 1, trunc)
            arch_inf = sum(w[1:m] .* (y[t - m:t - 1] .- mu) .^ 2)
            hh = omega / (1 - beta) + (kind === :figarch ? arch_inf :
                                       arch_inf * exp(theta * sign(y[t - 1] - mu)))
            h[t] = max(hh, 1e-8)
            y[t] = mu + sqrt(h[t]) * z[t]
        end
    else
        throw(ArgumentError("unknown kind :$kind"))
    end
    keep = (burn + 1):N
    return (y=y[keep], h=h[keep], eps=z[keep])
end

"""
    dgp_sv(rng; mu, phi, sigma_eta, rho_lev, nu, T, burn) -> NamedTuple

Stochastic volatility: `y_t = exp(h_t/2) ε_t`,
`h_t = μ + φ(h_{t-1} − μ) + σ_η η_t`, leverage `corr(ε, η) = ρ_lev`
(`nu = Inf` for Gaussian shocks, else standardised t; leverage applies to
the Gaussian design — under t shocks `ε` is redrawn). Returns `(y, h)`.
"""
function dgp_sv(rng::AbstractRNG; mu::Float64=-0.5, phi::Float64=0.95,
                sigma_eta::Float64=0.2, rho_lev::Float64=0.0,
                nu::Float64=Inf, T::Int=1500, burn::Int=200)
    N = T + burn
    h = zeros(N)
    h[1] = mu
    y = zeros(N)
    for t in 1:N
        eta = randn(rng)
        ep = rho_lev * eta + sqrt(1 - rho_lev^2) * randn(rng)
        if !isinf(nu)
            ep = rand(rng, TDist(nu)) / sqrt(nu / (nu - 2))
            eta = randn(rng)
        end
        t > 1 && (h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * eta)
        y[t] = exp(h[t] / 2) * ep
    end
    keep = (burn + 1):N
    return (y=y[keep], h=h[keep])
end

"""
    dgp_mgarch(rng; kind, n, T, R, a, b, C, A, B, burn) -> NamedTuple

Multivariate GARCH: `:ccc` (constant `R`, GARCH(1,1) vols), `:dcc`
(Engle DCC with `(a, b)`), `:bekk` (diagonal `H_t = CC′ + Aεε′A′ + BHB′`).
Returns `(Y, H, R)` with the `T×n×n` true covariance path.
"""
function dgp_mgarch(rng::AbstractRNG; kind::Symbol=:ccc, n::Int=2,
                    T::Int=1000, R=[1.0 0.4; 0.4 1.0], a::Float64=0.05,
                    b::Float64=0.9, C=nothing, A=nothing, B=nothing,
                    burn::Int=200)
    Rm = Matrix{Float64}(R)
    N = T + burn
    Y = zeros(N, n)
    H = zeros(N, n, n)
    if kind === :ccc
        gs = [dgp_garch_family(rng; T=N, burn=0) for _ in 1:n]
        Lr = cholesky(Symmetric(Rm)).L
        for t in 1:N
            Dt = Diagonal([sqrt(gs[j].h[t]) for j in 1:n])
            H[t, :, :] = Dt * Rm * Dt
            Y[t, :] = Dt * Lr * randn(rng, n)
        end
    elseif kind === :dcc
        gs = [dgp_garch_family(rng; T=N, burn=0) for _ in 1:n]
        Qb = copy(Rm)
        for t in 1:N
            Dt = Diagonal([sqrt(gs[j].h[t]) for j in 1:n])
            z = [gs[j].eps[t] for j in 1:n]
            Qb = (1 - a - b) * Rm + a * (z * z') + b * Qb
            dq = Diagonal(1 ./ sqrt.(diag(Qb)))
            Rt = dq * Qb * dq
            H[t, :, :] = Dt * Rt * Dt
            Y[t, :] = cholesky(Symmetric(H[t, :, :])).L * randn(rng, n)
        end
    elseif kind === :bekk
        Cm = C === nothing ? Matrix{Float64}(0.3 * I, n, n) : Matrix{Float64}(C)
        Am = A === nothing ? Matrix{Float64}(0.25 * I, n, n) : Matrix{Float64}(A)
        Bm = B === nothing ? Matrix{Float64}(0.9 * I, n, n) : Matrix{Float64}(B)
        Ht = Cm * Cm'
        for t in 1:N
            H[t, :, :] = Ht
            e = cholesky(Symmetric(Ht)).L * randn(rng, n)
            Y[t, :] = e
            Ht = Cm * Cm' + Am * (e * e') * Am' + Bm * Ht * Bm'
        end
    else
        throw(ArgumentError("unknown kind :$kind (ccc|dcc|bekk)"))
    end
    keep = (burn + 1):N
    return (Y=Y[keep, :], H=H[keep, :, :], R=Rm)
end

"""
    dgp_midas(rng; m, K, T_lf, theta, kind, beta, rho, sigma) -> NamedTuple

MIDAS: high-frequency AR(1) `x`, low-frequency
`y_t = β Σ_k w_k x_{t−k} + e_t` with normalised weights `w_true`
(`:expalmon`, `:beta2`, `:almon`). Returns `(y, x_hf, w_true, beta)`.
"""
function dgp_midas(rng::AbstractRNG; m::Int=3, K::Int=12, T_lf::Int=200,
                   theta=[-0.5], kind::Symbol=:expalmon, beta::Float64=1.0,
                   rho::Float64=0.5, sigma::Float64=0.5)
    th = Vector{Float64}(theta)
    N = T_lf * m
    x = zeros(N)
    for t in 2:N
        x[t] = rho * x[t - 1] + randn(rng)
    end
    ks = 1:K
    w = if kind === :expalmon
        exp.(th[1] .* ks .+ (length(th) > 1 ? th[2] .* ks .^ 2 : 0.0))
    elseif kind === :beta2
        a, b = exp(th[1]), exp(length(th) > 1 ? th[2] : 0.0)
        (ks ./ K) .^ (a - 1) .* (1 .- ks ./ K) .^ (b - 1)
    elseif kind === :almon
        v = zeros(K)
        Pp = length(th)
        for (j, kj) in enumerate(ks)
            v[j] = sum(th[i] * kj^(i - 1) for i in 1:Pp)
        end
        exp.(v)
    else
        throw(ArgumentError("unknown kind :$kind (expalmon|beta2|almon)"))
    end
    w ./= sum(w)
    y = zeros(T_lf)
    for t in 1:T_lf
        hi = t * m
        y[t] = beta * sum(w[k] * x[hi - k + 1] for k in 1:K if hi - k + 1 >= 1) +
               sigma * randn(rng)
    end
    return (y=y, x_hf=x, w_true=w, beta=beta)
end
