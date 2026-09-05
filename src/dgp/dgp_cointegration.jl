# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-04 (#793): cointegration, ARDL and panel-VAR simulators.
# The old corner fixture (instant full adjustment, no Γ, one shared trend) is
# replaced by a parametrised VECM with distinct trends and moderate adjustment.


"""
    dgp_vecm(rng; alpha, beta, Gamma, mu, Sigma, T, burn) -> NamedTuple

VECM `ΔY_t = αβ′Y_{t-1} + Σ_i Γ_i ΔY_{t-i} + μ + ε_t`, `ε ~ N(0, Sigma)`.
Default: 3-variable rank-1 with distinct dynamics (`α = (−0.3, 0.1, 0)`,
`β = (1, −1, 0)`, non-zero `Γ`). Returns `(Y, alpha, beta, Gamma, mu, eps)`.
"""
function dgp_vecm(rng::AbstractRNG; alpha=[-0.3, 0.1, 0.0],
                  beta=[1.0, -1.0, 0.0],
                  Gamma=[0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2],
                  mu=nothing, Sigma=nothing, T::Int=400, burn::Int=200)
    al = alpha isa AbstractVector ? reshape(Vector{Float64}(alpha), :, 1) :
                                      Matrix{Float64}(alpha)
    be = beta isa AbstractVector ? reshape(Vector{Float64}(beta), :, 1) :
                                   Matrix{Float64}(beta)
    Gs = Gamma isa AbstractMatrix ? [Matrix{Float64}(Gamma)] :
                                     [Matrix{Float64}(g) for g in Gamma]
    n, r = size(be, 1), size(be, 2)
    k = length(Gs)
    Sg = Sigma === nothing ? Matrix{Float64}(I, n, n) : Matrix{Float64}(Sigma)
    L = cholesky(Symmetric(Sg)).L
    mm = mu === nothing ? zeros(n) : Vector{Float64}(mu)
    N = T + burn
    Eps = randn(rng, N, n)
    Y = zeros(N + 1, n)
    for t in 2:(N + 1)
        dy = al * (be' * Y[t - 1, :]) + mm + L * Eps[min(t - 1, N), :]
        for i in 1:min(k, t - 2)
            dy += Gs[i] * (Y[t - i, :] - Y[t - i - 1, :])
        end
        Y[t, :] = Y[t - 1, :] + dy
    end
    keep = (burn + 2):(N + 1)
    return (Y=Y[keep, :], alpha=al, beta=be, Gamma=Gs, mu=mm, Sigma=Sg,
            eps=Eps[(burn + 1):N, :])
end

"""
    dgp_cointreg(rng; beta, T, endog_rho, sigma_u, spurious) -> NamedTuple

Cointegrating regression `y = β′x + u` with RW regressors and endogenous
errors (`corr(u, Δx) = endog_rho`; the FMOLS/DOLS bias-reduction target).
`spurious = true` gives independent RWs (no cointegration). Returns
`(y, X, beta, u)`.
"""
function dgp_cointreg(rng::AbstractRNG; beta=[2.0, -1.0], T::Int=500,
                      endog_rho::Float64=0.7, sigma_u::Float64=1.0,
                      spurious::Bool=false)
    be = Vector{Float64}(beta)
    m = length(be)
    dx = randn(rng, T, m)
    v = randn(rng, T)
    u = @. sigma_u * (endog_rho * dx[:, 1] + sqrt(1 - endog_rho^2) * v)
    X = cumsum(dx, dims=1)
    y = spurious ? cumsum(randn(rng, T)) : X * be + u
    return (y=y, X=X, beta=be, u=u)
end

"""
    dgp_panel_var(rng; A1, N, T, m, mu_sd, Sigma, burn) -> NamedTuple

Panel VAR(1) `y_{it} = μ_i + A1 y_{i,t-1} + ε_{it}` with random effects
`μ_i ~ N(0, mu_sd²I)`, stationary start (burn-in per unit). Returns
`(Y, id, time, A1, mu)` with `Y` stacked `NT×m`.
"""
function dgp_panel_var(rng::AbstractRNG; A1=[0.8 0.15; 0.05 0.7], N::Int=30,
                       T::Int=25, mu_sd::Float64=1.0, Sigma=nothing,
                       burn::Int=50)
    A = Matrix{Float64}(A1)
    m = size(A, 1)
    Sg = Sigma === nothing ? Matrix{Float64}(I, m, m) : Matrix{Float64}(Sigma)
    L = cholesky(Symmetric(Sg)).L
    Y = zeros(N * T, m)
    id = repeat(1:N, inner=T)
    time = repeat(1:T, outer=N)
    mu = mu_sd .* randn(rng, N, m)
    for i in 1:N
        y = zeros(m)
        for _ in 1:burn
            y = A * y + L * randn(rng, m)
        end
        for t in 1:T
            y = mu[i, :] + A * (y - mu[i, :]) + L * randn(rng, m)
            Y[(i - 1) * T + t, :] .= y
        end
    end
    return (Y=Y, id=id, time=time, A1=A, mu=mu, Sigma=Sg)
end

"""
    dgp_ardl(rng; phi, beta0, beta1, rho_x, T, burn) -> NamedTuple

ARDL(1,1) `y_t = c + φy_{t-1} + β₀x_t + β₁x_{t-1} + e_t` with AR(1) `x`.
Long-run multiplier `θ = (β₀+β₁)/(1−φ)` is returned for bounds-test asserts.
"""
function dgp_ardl(rng::AbstractRNG; phi::Float64=0.6, beta0::Float64=0.8,
                  beta1::Float64=0.4, rho_x::Float64=0.7, c::Float64=0.5,
                  T::Int=300, burn::Int=100)
    N = T + burn
    x = zeros(N)
    for t in 2:N
        x[t] = rho_x * x[t - 1] + randn(rng)
    end
    y = zeros(N)
    for t in 2:N
        y[t] = c + phi * y[t - 1] + beta0 * x[t] + beta1 * x[t - 1] + randn(rng)
    end
    keep = (burn + 1):N
    theta = (beta0 + beta1) / (1 - phi)
    return (y=y[keep], x=x[keep], phi=phi, beta=[beta0, beta1], theta=theta)
end

"""
    dgp_nardl(rng; phi, beta_pos, beta_neg, rho_x, T, burn) -> NamedTuple

NARDL(1,1) with partial-sum decomposition `x⁺`/`x⁻` and asymmetric
long-run multipliers `θ⁺`, `θ⁻`.
"""
function dgp_nardl(rng::AbstractRNG; phi::Float64=0.6, beta_pos::Float64=0.9,
                   beta_neg::Float64=0.3, rho_x::Float64=0.7, T::Int=300,
                   burn::Int=100)
    N = T + burn
    x = zeros(N)
    for t in 2:N
        x[t] = rho_x * x[t - 1] + randn(rng)
    end
    dx = [0.0; diff(x)]
    xp = cumsum(max.(dx, 0.0))
    xn = cumsum(min.(dx, 0.0))
    y = zeros(N)
    for t in 2:N
        y[t] = phi * y[t - 1] + beta_pos * xp[t] + beta_neg * xn[t] + randn(rng)
    end
    keep = (burn + 1):N
    return (y=y[keep], x=x[keep], xp=xp[keep], xn=xn[keep], phi=phi,
            theta_pos=beta_pos / (1 - phi), theta_neg=beta_neg / (1 - phi))
end

"""
    dgp_pmg(rng; theta, N, T, homogeneous, burn) -> NamedTuple

Panel ARDL(1,1) with common long-run `θ` and heterogeneous short-run
dynamics; `homogeneous = false` makes the long-run heterogeneous too
(the Hausman-test power arm). Returns `(Y, X, id, theta, theta_i, phi_i)`.
"""
function dgp_pmg(rng::AbstractRNG; theta::Float64=1.5, N::Int=20, T::Int=50,
                 homogeneous::Bool=true, burn::Int=50)
    Y = zeros(N * T)
    X = zeros(N * T)
    phi_i = 0.4 .+ 0.3 .* rand(rng, N)
    th_i = homogeneous ? fill(theta, N) : theta .+ 0.8 .* randn(rng, N)
    for i in 1:N
        x = 0.0
        y = 0.0
        for _ in 1:burn
            x = 0.7 * x + randn(rng)
            y = phi_i[i] * y + (1 - phi_i[i]) * th_i[i] * x + randn(rng)
        end
        for t in 1:T
            x = 0.7 * x + randn(rng)
            y = phi_i[i] * y + (1 - phi_i[i]) * th_i[i] * x + randn(rng)
            Y[(i - 1) * T + t] = y
            X[(i - 1) * T + t] = x
        end
    end
    return (Y=Y, X=X, id=repeat(1:N, inner=T), theta=theta, theta_i=th_i,
            phi_i=phi_i)
end
