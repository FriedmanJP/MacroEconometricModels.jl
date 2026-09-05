# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-05 (#794): LP simulators — the exemplary `_lpiv_sim`
# promoted to rng-first, plus state-dependent VAR and confounded propensity.


"""
    dgp_lp_iv(rng; T, pi1, theta) -> NamedTuple

LP-IV DGP (promoted from `test_lp_weak_iv.jl::_lpiv_sim`): instrument `z`,
endogenous shock `s = π₁z + v`, outcome loading on `s` with impact response
`theta`, endogeneity through the shared `v`. Returns `(Y, Z, pi1, theta)`
with `Y = [s y x2]`.
"""
function dgp_lp_iv(rng::AbstractRNG; T::Int=400, pi1::Float64=1.5,
                   theta::Float64=1.0)
    z = randn(rng, T)
    v = randn(rng, T)
    s = pi1 .* z .+ v
    y = zeros(T)
    x2 = zeros(T)
    for t in 2:T
        y[t] = 0.5 * y[t - 1] + theta * s[t] + 0.6 * v[t] + randn(rng)
        x2[t] = 0.3 * x2[t - 1] + 0.4 * s[t] + randn(rng)
    end
    return (Y=hcat(s, y, x2), Z=reshape(z, :, 1), pi1=pi1, theta=theta)
end

"""
    dgp_state_dependent_var(rng; A_exp, A_rec, B0, gamma, T, burn)
        -> NamedTuple

Two-regime VAR with logistic transition `G(z_t)` (`z` AR(1)):
`Y_t = (1−G)A_rec Y_{t-1} + G·A_exp Y_{t-1} + B0 ε_t`. Returns
`(Y, G, z, A_exp, A_rec, B0)` plus each regime's true IRF
(`irf_exp`, `irf_rec` at `H = 12`).
"""
function dgp_state_dependent_var(rng::AbstractRNG;
                                 A_exp=[0.8 0.1; 0.0 0.6],
                                 A_rec=[0.3 0.0; 0.1 0.2],
                                 B0=[1.0 0.0; 0.5 1.0], gamma::Float64=3.0,
                                 T::Int=2000, burn::Int=200, H::Int=12)
    Ae = Matrix{Float64}(A_exp)
    Ar = Matrix{Float64}(A_rec)
    B = Matrix{Float64}(B0)
    n = size(Ae, 1)
    N = T + burn
    z = zeros(N)
    for t in 2:N
        z[t] = 0.9 * z[t - 1] + randn(rng)
    end
    G = @. 1 / (1 + exp(-gamma * z))
    Eps = randn(rng, N, n)
    Y = zeros(N, n)
    y = zeros(n)
    for t in 1:N
        y = ((1 - G[t]) * Ar + G[t] * Ae) * y + B * Eps[t, :]
        Y[t, :] .= y
    end
    keep = (burn + 1):N
    return (Y=Y[keep, :], G=G[keep], z=z[keep], A_exp=Ae, A_rec=Ar, B0=B,
            irf_exp=var_irf(Ae, B, H), irf_rec=var_irf(Ar, B, H))
end

"""
    dgp_propensity(rng; beta_ps, tau, gamma_y, n) -> NamedTuple

Propensity-score DGP: `X` covariates, `ps = logistic(Xβ_ps)`,
`D ~ Bernoulli(ps)`, `Y₀ = Xγ_y + e` (`confounding = true`; else `Y₀ = e`),
`Y = Y₀ + τD`. The naive difference-in-means is biased by the known
confounding amount. Returns `(Y, D, X, tau, beta_ps, att, ps)`.
"""
function dgp_propensity(rng::AbstractRNG; beta_ps=[0.5, 0.3], tau::Float64=1.0,
                        gamma_y=[0.7, -0.4], confounding::Bool=true,
                        n::Int=2000)
    bp = Vector{Float64}(beta_ps)
    gy = Vector{Float64}(gamma_y)
    k = length(bp)
    X = randn(rng, n, k)
    ps = 1 ./ (1 .+ exp.(-(X * bp)))
    D = rand(rng, n) .< ps
    Y0 = (confounding ? X * gy : zeros(n)) + randn(rng, n)
    Y = Y0 + tau * D
    return (Y=Y, D=D, X=X, tau=tau, beta_ps=bp, att=tau, ps=ps)
end

"""
    dgp_hac(rng; rho, T, k, x_first) -> NamedTuple

HAC-covariance regression DGP (DGP-05 #794): AR(1) errors
`u_t = ρ·u_{t-1} + ε_t` with `u_1 = 0` (exactly like the legacy inline
loops it replaces) and iid regressors `X = [1 X̃]`, `X̃ ~ N(0, I_k)`.
Returns `(X, u, rho, lrv)` with the population long-run variance
`lrv = 1/(1−ρ)²` (unit innovations).

Draw order matches the legacy LP blocks byte-for-byte so probed
HAC/White constants hold without re-probing: `u` first by default;
`x_first = true` draws `X` first (blocks that built `X` first; with
`rho = 0` the white `u` is one vector draw, as before). `k = 0` skips
the `X̃` block (no draws).
"""
function dgp_hac(rng::AbstractRNG; rho::Float64=0.5, T::Int=2000, k::Int=1,
                 x_first::Bool=false)
    if x_first
        X = k > 0 ? hcat(ones(T), randn(rng, T, k)) : ones(T, 1)
        u = rho == 0.0 ? randn(rng, T) : _ar1_errors(rng, rho, T)
    else
        u = _ar1_errors(rng, rho, T)
        X = k > 0 ? hcat(ones(T), randn(rng, T, k)) : ones(T, 1)
    end
    return (X=X, u=u, rho=rho, lrv=1 / (1 - rho)^2)
end

# AR(1) errors with u_1 = 0 (legacy inline-loop convention, DGP-05 #794).
function _ar1_errors(rng::AbstractRNG, rho::Float64, T::Int)
    u = zeros(T)
    for t in 2:T
        u[t] = rho * u[t-1] + randn(rng)
    end
    return u
end
