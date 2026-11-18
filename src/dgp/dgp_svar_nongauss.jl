# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-11 (#800): non-Gaussian and heteroskedastic SVAR DGPs.
# Gaussian white noise leaves every non-Gaussian / heteroskedasticity-based
# identification scheme unidentified in population; these simulators give each
# method data with the feature it needs.


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
`(Y, eps, scales, B0, Sigma_full, Lambda, path, kind)` where `scales` is the `T×n`
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

# =============================================================================
# NGID-05 (#905): Non-Gaussian Structural VARMA(p, q) Simulator (GMR 2020)
# =============================================================================

function _dgp_svarma_eltype(x)
    x === nothing && return Bool
    if x isa AbstractVector{<:AbstractMatrix}
        return isempty(x) ? Bool : promote_type(map(eltype, x)...)
    end
    return eltype(x)
end

function _dgp_svarma_infer_n(mu, Phi, Theta, C)
    if C !== nothing
        nr, nc = size(C)
        nr == nc || throw(ArgumentError("C must be square, got ($(nr), $(nc))"))
        return nr
    elseif Phi !== nothing
        if Phi isa AbstractMatrix
            nr, nc = size(Phi)
            nr == nc || throw(ArgumentError("Phi matrix must be square, got ($(nr), $(nc))"))
            return nr
        elseif Phi isa AbstractArray{<:Real,3}
            nr, nc, _ = size(Phi)
            nr == nc || throw(ArgumentError("Phi slices must be square, got ($(nr), $(nc))"))
            return nr
        elseif Phi isa AbstractVector && !isempty(Phi)
            nr, nc = size(Phi[1])
            nr == nc || throw(ArgumentError("Phi[1] must be square, got ($(nr), $(nc))"))
            return nr
        end
    elseif Theta !== nothing
        if Theta isa AbstractMatrix
            nr, nc = size(Theta)
            nr == nc || throw(ArgumentError("Theta matrix must be square, got ($(nr), $(nc))"))
            return nr
        elseif Theta isa AbstractArray{<:Real,3}
            nr, nc, _ = size(Theta)
            nr == nc || throw(ArgumentError("Theta slices must be square, got ($(nr), $(nc))"))
            return nr
        elseif Theta isa AbstractVector && !isempty(Theta)
            nr, nc = size(Theta[1])
            nr == nc || throw(ArgumentError("Theta[1] must be square, got ($(nr), $(nc))"))
            return nr
        end
    elseif mu !== nothing
        return length(mu)
    end
    return 2
end

function _dgp_svarma_to_3d(::Nothing, n::Int, ::Type{T}) where {T<:AbstractFloat}
    zeros(T, n, n, 0)
end

function _dgp_svarma_to_3d(M::AbstractMatrix, n::Int, ::Type{T}) where {T<:AbstractFloat}
    size(M) == (n, n) || throw(ArgumentError("Expected $n × $n matrix, got $(size(M))"))
    out = Array{T,3}(undef, n, n, 1)
    @views out[:, :, 1] .= T.(M)
    out
end

function _dgp_svarma_to_3d(M::AbstractArray{<:Real,3}, n::Int, ::Type{T}) where {T<:AbstractFloat}
    size(M, 1) == n && size(M, 2) == n ||
        throw(ArgumentError("Expected $n × $n × k array, got $(size(M))"))
    Array{T,3}(M)
end

function _dgp_svarma_to_3d(M::AbstractVector, n::Int, ::Type{T}) where {T<:AbstractFloat}
    q = length(M)
    q == 0 && return zeros(T, n, n, 0)
    out = Array{T,3}(undef, n, n, q)
    for k in 1:q
        mk = M[k]
        mk isa AbstractMatrix || throw(ArgumentError("Lag entry $k must be a matrix"))
        size(mk) == (n, n) || throw(ArgumentError("Lag entry $k must be $n × $n, got $(size(mk))"))
        @views out[:, :, k] .= T.(mk)
    end
    out
end

function _dgp_svarma_shocks_1d(rng::AbstractRNG, N::Int, dist::Symbol, nu::Real,
                               skew::Real, ::Type{T}) where {T<:AbstractFloat}
    if dist === :t || dist === :student_t
        nu_f = Float64(nu)
        nu_f > 2.0 || throw(ArgumentError("Student-t degrees of freedom nu must be > 2, got $nu"))
        scale = T(sqrt(nu_f / (nu_f - 2.0)))
        return T.(rand(rng, TDist(nu_f), N)) ./ scale
    elseif dist === :laplace
        return T.(rand(rng, Laplace(0.0, 1.0 / sqrt(2.0)), N))
    elseif dist === :mixture || dist === :mixture_normal
        s = rand(rng, N) .< 0.5
        z = randn(rng, N)
        raw = @. ifelse(s, -1.5, 1.5) + z
        return T.(raw ./ sqrt(3.25))
    elseif dist === :mixture_asym || dist === :mixture_normal_asym
        p_mix = 0.1
        r_mu = 0.65
        mu1 = sqrt((1.0 - p_mix) / p_mix) * r_mu
        mu2 = -sqrt(p_mix / (1.0 - p_mix)) * r_mu
        B_rem = 1.0 - r_mu^2
        w = 0.25
        s1 = sqrt(w * B_rem / p_mix)
        s2 = sqrt((1.0 - w) * B_rem / (1.0 - p_mix))
        sel = rand(rng, N) .< p_mix
        z = randn(rng, N)
        raw = @. ifelse(sel, mu1 + s1 * z, mu2 + s2 * z)
        return T.(raw)
    elseif dist === :skew || dist === :chi2
        return (T.(rand(rng, Chisq(3), N)) .- T(3)) ./ T(sqrt(6.0))
    elseif dist === :skew_normal
        alpha = Float64(skew) == 0.0 ? 3.0 : Float64(skew)
        delta = alpha / sqrt(1.0 + alpha^2)
        mu_z = delta * sqrt(2.0 / pi)
        sd_z = sqrt(1.0 - mu_z^2)
        z0 = abs.(randn(rng, N))
        z1 = randn(rng, N)
        raw = @. (delta * z0 + sqrt(1.0 - delta^2) * z1 - mu_z) / sd_z
        return T.(raw)
    elseif dist === :ged || dist === :laplace_generalized
        kappa = Float64(nu) > 0.0 ? Float64(nu) : 1.2
        if kappa == 5.0
            kappa = 1.2
        end
        # Symmetric GED with unit variance via Gamma(1/kappa, 1)^(1/kappa)
        g = rand(rng, Gamma(1.0 / kappa, 1.0), N)
        sgn = ifelse.(rand(rng, N) .< 0.5, -1.0, 1.0)
        raw = sgn .* (g .^ (1.0 / kappa))
        # Empirical/analytical unit-variance normalization via gamma ratio
        var_raw = exp(SpecialFunctions.loggamma(3.0 / kappa) - SpecialFunctions.loggamma(1.0 / kappa))
        return T.(raw ./ sqrt(var_raw))
    elseif dist === :gauss || dist === :normal
        return randn(rng, T, N)
    else
        throw(ArgumentError(
            "unknown dist :$dist (supported: :student_t/:t, :laplace, :mixture, " *
            ":mixture_normal_asym/:mixture_asym, :skew/:chi2, :skew_normal, :ged, :gauss)"))
    end
end

function _dgp_svarma_eigs_regime(Theta_3d::Array{T,3}, tol::Real) where {T<:AbstractFloat}
    n, _, q = size(Theta_3d)
    if q == 0
        return Complex{T}[], :fundamental
    end
    nq = n * q
    F_ma = zeros(T, nq, nq)
    @inbounds for k in 1:q
        F_ma[1:n, ((k - 1) * n + 1):(k * n)] .= -Theta_3d[:, :, k]
    end
    if q > 1
        @inbounds for i in 1:((q - 1) * n)
            F_ma[n + i, i] = one(T)
        end
    end
    eigs = Complex{T}.(eigvals(F_ma))
    tol_T = max(T(tol), T(16) * eps(T))
    moduli = abs.(eigs)
    n_unit = count(m -> abs(m - one(T)) <= tol_T, moduli)
    n_nonfund = count(m -> m > one(T) + tol_T, moduli)
    n_fund = count(m -> m < one(T) - tol_T, moduli)
    regime = if n_unit > 0
        :unit_root
    elseif n_nonfund == 0
        :fundamental
    elseif n_fund == 0
        :purely_nonfundamental
    else
        :mixed_nonfundamental
    end
    return eigs, regime
end

"""
    dgp_ssvarma(rng=Random.default_rng(); T=400, mu=nothing, Phi=nothing,
                Theta=nothing, C=nothing, dist=:student_t, nu=5.0, skew=0.0,
                burn=200, tol=1e-8, seed=nothing) -> NamedTuple

Simulate data from a Strong Structural VARMA(p, q) model with mutually and
serially independent non-Gaussian shocks (Gouriéroux, Monfort & Renne 2020):

```math
Y_t = \\mu + \\sum_{k=1}^{p} \\Phi_k Y_{t-k} + \\varepsilon_t + \\sum_{j=1}^{q} \\Theta_j \\varepsilon_{t-j}, \\qquad \\varepsilon_t = C \\eta_t
```

where:
- `Y_t` is the ``n \\times 1`` vector of endogenous variables at time ``t``
- `\\mu` is the ``n \\times 1`` intercept vector
- `\\Phi_k` are the ``n \\times n`` autoregressive lag coefficient matrices (``k = 1, \\ldots, p``)
- `\\Theta_j` are the ``n \\times n`` moving-average lag coefficient matrices (``j = 1, \\ldots, q``)
- `\\varepsilon_t` is the ``n \\times 1`` vector of reduced-form MA innovations
- `C` is the ``n \\times n`` nonsingular structural impact matrix
- `\\eta_t` is the ``n \\times 1`` vector of mutually and serially independent zero-mean unit-variance structural shocks

When `Phi`, `Theta`, and `C` are all omitted (`nothing`), defaults to a mixed
non-fundamental bivariate SSVARMA(1, 1) benchmark inspired by GMR (2020, eq. 4.27)
with ``\\Theta_1 = [-0.5\\; 0;\\; 1\\; -2]`` (companion eigenvalues ``0.5`` and ``2.0``).

Returns a `NamedTuple` `(Y, eta, shocks, mu, Phi, Theta, C, p, q, ma_eigenvalues, regime)`.
"""
function dgp_ssvarma(rng::AbstractRNG=Random.default_rng();
                     T::Int=400,
                     mu=nothing,
                     Phi=nothing,
                     Theta=nothing,
                     C=nothing,
                     dist::Union{Symbol,AbstractVector{Symbol}}=:student_t,
                     nu::Real=5.0,
                     skew::Real=0.0,
                     burn::Int=200,
                     tol::Real=1e-8,
                     seed::Union{Integer,Nothing}=nothing)
    T >= 1 || throw(ArgumentError("T must be >= 1, got $T"))
    burn >= 0 || throw(ArgumentError("burn must be >= 0, got $burn"))
    rng_eff = _resolve_repro_rng(rng, seed)

    use_default_design = (Phi === nothing && Theta === nothing && C === nothing && mu === nothing)
    if use_default_design
        Phi_use = [0.4 0.1; -0.05 0.35]
        Theta_use = [-0.5 0.0; 1.0 -2.0]
        C_use = [1.0 0.3; -0.4 0.9]
        mu_use = [0.0, 0.0]
    else
        Phi_use = Phi
        Theta_use = Theta
        C_use = C
        mu_use = mu
    end

    T_el = float(promote_type(
        _dgp_svarma_eltype(mu_use),
        _dgp_svarma_eltype(Phi_use),
        _dgp_svarma_eltype(Theta_use),
        _dgp_svarma_eltype(C_use),
    ))

    n = _dgp_svarma_infer_n(mu_use, Phi_use, Theta_use, C_use)
    mu_vec = mu_use === nothing ? zeros(T_el, n) : Vector{T_el}(mu_use)
    length(mu_vec) == n || throw(ArgumentError("mu length ($(length(mu_vec))) must match n ($n)"))

    Phi_3d = _dgp_svarma_to_3d(Phi_use, n, T_el)
    Theta_3d = _dgp_svarma_to_3d(Theta_use, n, T_el)
    C_mat = C_use === nothing ? Matrix{T_el}(I, n, n) : Matrix{T_el}(C_use)
    size(C_mat) == (n, n) || throw(ArgumentError("C must be $n × $n, got $(size(C_mat))"))

    p = size(Phi_3d, 3)
    q = size(Theta_3d, 3)

    dists = dist isa Symbol ? fill(dist, n) : Vector{Symbol}(dist)
    length(dists) == n || throw(ArgumentError("dist vector length ($(length(dists))) must match n ($n)"))

    T_obs = T
    N_total = T_obs + burn
    eta_all = Matrix{T_el}(undef, N_total + q, n)
    for j in 1:n
        eta_all[:, j] .= _dgp_svarma_shocks_1d(rng_eff, N_total + q, dists[j], nu, skew, T_el)
    end
    shock_mat_all = eta_all * transpose(C_mat)

    Y_all = zeros(T_el, N_total, n)
    # Initialize pre-sample Y at stationary mean when Phi(1) is nonsingular
    y_init = zeros(T_el, n)
    if p > 0 && any(!iszero, mu_vec)
        Phi_sum = zeros(T_el, n, n)
        @inbounds for k in 1:p
            Phi_sum .+= @view(Phi_3d[:, :, k])
        end
        A1 = Matrix{T_el}(I, n, n) .- Phi_sum
        if abs(det(A1)) > T_el(1e-10)
            y_init .= robust_inv(A1; silent=true) * mu_vec
        end
    end

    y_curr = Vector{T_el}(undef, n)
    @inbounds for t in 1:N_total
        copyto!(y_curr, mu_vec)
        for k in 1:p
            y_lag = t > k ? @view(Y_all[t - k, :]) : y_init
            mul!(y_curr, @view(Phi_3d[:, :, k]), y_lag, one(T_el), one(T_el))
        end
        e_idx = q + t
        y_curr .+= @view(shock_mat_all[e_idx, :])
        for j in 1:q
            mul!(y_curr, @view(Theta_3d[:, :, j]), @view(shock_mat_all[e_idx - j, :]),
                 one(T_el), one(T_el))
        end
        Y_all[t, :] .= y_curr
    end

    keep_y = (burn + 1):N_total
    keep_e = (q + burn + 1):(q + N_total)
    Y_out = Y_all[keep_y, :]
    eta_out = eta_all[keep_e, :]

    eigs, regime = _dgp_svarma_eigs_regime(Theta_3d, tol)
    return (
        Y = Y_out,
        eta = eta_out,
        shocks = eta_out,
        mu = mu_vec,
        Phi = Phi_3d,
        Theta = Theta_3d,
        C = C_mat,
        p = p,
        q = q,
        ma_eigenvalues = eigs,
        regime = regime,
    )
end
