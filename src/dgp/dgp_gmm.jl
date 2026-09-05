# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-08 (#797) / DGP-17 (#806) / DGP-13 (#802): GMM, PCE
# band-draw and DSGE-observation simulators.


"""
    dgp_gmm(rng; kind, beta, n, hetero, overid_k, invalid_k, pi1) -> NamedTuple

GMM DGP: heteroskedastic errors (`σᵢ = 0.5 + |x₁|`, so two-step weighting
matters), `overid_k` instruments, `invalid_k` of them correlated with the
error (J-test power arm), first-stage strength `pi1`. `kind ∈ :ols|:iv`.
The invalid instruments are the LAST `invalid_k` (the overidentifying ones):
contaminating the relevant first instrument instead lets the slope absorb the
violation (θ̂ biased, J silent) — verified on seed 7 (J p ≈ 0.19) — so that
variant belongs in-test, not here. Returns `(y, X, Z, beta, pi1)`.
"""
function dgp_gmm(rng::AbstractRNG; kind::Symbol=:iv, beta=[1.0, 0.5],
                 n::Int=1000, hetero::Bool=true, overid_k::Int=2,
                 invalid_k::Int=0, pi1::Float64=1.0)
    be = Vector{Float64}(beta)
    k = length(be)
    X = hcat(ones(n), randn(rng, n, k - 1))
    sig = hetero ? (0.5 .+ abs.(X[:, 2])) : ones(n)
    if kind === :ols
        y = X * be + sig .* randn(rng, n)
        return (y=y, X=X, Z=X, beta=be, pi1=pi1)
    elseif kind === :iv
        m = 1 + overid_k
        Z = hcat(ones(n), randn(rng, n, m - 1))
        v = randn(rng, n)
        u = sig .* (0.6 .* v + 0.8 .* randn(rng, n))
        if invalid_k > 0
            Z[:, (m - invalid_k + 1):m] .+= 0.8 .* u
        end
        x_endog = Z[:, 2] * pi1 + v
        XX = k > 2 ? hcat(ones(n), x_endog, X[:, 3:end]) : hcat(ones(n), x_endog)
        y = XX * be[1:size(XX, 2)] + u
        return (y=y, X=XX, Z=Z, beta=be[1:size(XX, 2)], pi1=pi1)
    else
        throw(ArgumentError("unknown kind :$kind (ols|iv)"))
    end
end

"""
    dgp_pce_draws(rng, ce_point; sd, corr, n_draws) -> NamedTuple

Genuine sampling distribution around a point policy menu: independent
Gaussian perturbations with known `sd` (scalar or per-element) and AR(1)
`corr` along the horizon. Returns `(draws, point, sd)` for band
width-scaling / coverage asserts. (Collapses to a point when `sd = 0` —
the degenerate case the old factories hardcoded.)
"""
function dgp_pce_draws(rng::AbstractRNG, ce_point::AbstractArray;
                       sd=0.1, corr::Float64=0.0, n_draws::Int=500)
    p = Array{Float64}(ce_point)
    s = sd isa Real ? fill(Float64(sd), size(p)) : Array{Float64}(sd)
    draws = similar(p, (n_draws, size(p)...))
    for i in 1:n_draws
        e = randn(rng, size(p))
        if corr != 0.0  # AR(1) along the first axis (horizon)
            for t in 2:size(p, 1)
                e[t, :] = corr * e[t - 1, :] + sqrt(1 - corr^2) * e[t, :]
            end
        end
        draws[i, :] = vec(p + s .* e)
    end
    return (draws=draws, point=p, sd=s)
end

"""
    dgp_dsge_observed(rng, y_clean; H, trends) -> NamedTuple

DSGE observation equation: `y_obs = y_clean + trends + sqrt(H)⋅η`.
Estimating a measurement-error variance the DGP does not contain is a
misspecification — this helper keeps them matched. Returns `(y_obs, H)`.
"""
function dgp_dsge_observed(rng::AbstractRNG, y_clean::AbstractMatrix;
                           H=nothing, trends=nothing)
    Y = Matrix{Float64}(y_clean)
    T, n = size(Y)
    Hv = H === nothing ? zeros(n) : Vector{Float64}(H)
    tr = trends === nothing ? zeros(T, n) : Matrix{Float64}(trends)
    Yobs = Y + tr + sqrt.(reshape(Hv, 1, n)) .* randn(rng, T, n)
    return (y_obs=Yobs, H=Hv)
end
