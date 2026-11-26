# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Core types, accessors, display, and shared internal helpers for Strong Structural
VARMA(p, q) models (Gouriéroux, Monfort & Renne 2020).
"""

using LinearAlgebra, Statistics, StatsAPI

# =============================================================================
# Internal Shape & Companion Helpers
# =============================================================================

function _svarma_eltype(x)
    x === nothing && return Bool
    if x isa AbstractVector{<:AbstractMatrix}
        return isempty(x) ? Bool : promote_type(map(eltype, x)...)
    elseif x isa AbstractVector
        isempty(x) && return Bool
        return promote_type(map(_svarma_eltype, x)...)
    end
    return eltype(x)
end

function _svarma_to_3d(::Nothing, n::Int, ::Type{T}) where {T}
    zeros(T, n, n, 0)
end

function _svarma_to_3d(M::AbstractArray{<:Number,3}, n::Int, ::Type{T}) where {T}
    size(M, 1) == n && size(M, 2) == n ||
        throw(ArgumentError("Expected $n × $n × k array, got $(size(M))"))
    Array{T,3}(M)
end

function _svarma_to_3d(M::AbstractMatrix{<:Number}, n::Int, ::Type{T}) where {T}
    size(M) == (n, n) || throw(ArgumentError("Expected $n × $n matrix, got $(size(M))"))
    out = Array{T,3}(undef, n, n, 1)
    @views out[:, :, 1] .= T.(M)
    out
end

function _svarma_to_3d(M::AbstractVector, n::Int, ::Type{T}) where {T}
    k = length(M)
    k == 0 && return zeros(T, n, n, 0)
    out = Array{T,3}(undef, n, n, k)
    for j in 1:k
        mj = M[j]
        mj isa AbstractMatrix || throw(ArgumentError("Lag entry $j must be a matrix"))
        size(mj) == (n, n) ||
            throw(ArgumentError("Lag entry $j must be $n × $n, got $(size(mj))"))
        @views out[:, :, j] .= T.(mj)
    end
    out
end

"""
    _svarma_companion(Theta_3d::Array{T,3}) -> Matrix{T}

Build the `nq × nq` companion matrix `F_ma` for ``\\Theta(L) = I + \\sum_{j=1}^q \\Theta_j L^j``
so that ``E_t = F_{\\text{ma}} E_{t-1} + U_t`` satisfies ``U_t = \\varepsilon_t + \\sum_{j=1}^q \\Theta_j \\varepsilon_{t-j}``
and eigenvalues of `F_ma` equal reciprocal roots ``1/z`` of ``\\det \\Theta(z) = 0``.
"""
function _svarma_companion(Theta_3d::Array{T,3}) where {T<:Number}
    n, nc, q = size(Theta_3d)
    n == nc || throw(ArgumentError("Theta slices must be square, got $n × $nc"))
    q == 0 && return zeros(T, 0, 0)
    nq = n * q
    F = zeros(T, nq, nq)
    @inbounds for j in 1:q
        F[1:n, ((j - 1) * n + 1):(j * n)] .= -@view(Theta_3d[:, :, j])
    end
    if q > 1
        @inbounds for i in 1:((q - 1) * n)
            F[n + i, i] = one(T)
        end
    end
    F
end

"""
    _svarma_ar_residuals(Y, mu, Phi_3d) -> Matrix

Compute AR-filtered reduced-form residuals ``u_t = Y_t - \\mu - \\sum_{k=1}^p \\Phi_k Y_{t-k}``
for ``t = p+1, \\ldots, T_{\\text{obs}}``.
"""
function _svarma_ar_residuals(Y::AbstractMatrix, mu::AbstractVector,
                              Phi_3d::Array{<:Number,3})
    T_obs, n = size(Y)
    p = size(Phi_3d, 3)
    T_obs > p || throw(ArgumentError("Need T_obs ($T_obs) > p ($p)"))
    length(mu) == n || throw(ArgumentError("mu length ($(length(mu))) must match n ($n)"))
    T_out = float(promote_type(eltype(Y), eltype(mu), eltype(Phi_3d)))
    T_eff = T_obs - p
    U = Matrix{T_out}(undef, T_eff, n)
    buf = Vector{T_out}(undef, n)
    @inbounds for t in 1:T_eff
        idx = p + t
        for i in 1:n
            buf[i] = T_out(Y[idx, i]) - T_out(mu[i])
        end
        for k in 1:p
            for i in 1:n
                s = zero(T_out)
                for j in 1:n
                    s += T_out(Phi_3d[i, j, k]) * T_out(Y[idx - k, j])
                end
                buf[i] -= s
            end
        end
        for i in 1:n
            U[t, i] = buf[i]
        end
    end
    U
end

function _svarma_gmr_region(Theta_3d::Array{T,3}, tol_T::T) where {T<:AbstractFloat}
    n, _, q = size(Theta_3d)
    (n == 2 && q == 1) || return :none
    A_ma = -@view(Theta_3d[:, :, 1])
    tau = tr(A_ma)
    delta = det(A_ma)
    s1 = one(T) - tau + delta
    s2 = one(T) + tau + delta
    on_boundary = abs(s1) <= tol_T || abs(s2) <= tol_T ||
                  (abs(delta - one(T)) <= tol_T && abs(tau) <= T(2) + tol_T)
    on_boundary && return :unit_root
    if s1 > zero(T) && s2 > zero(T)
        return delta < one(T) ? :I : :III
    elseif s1 < zero(T) && s2 > zero(T)
        return :II
    elseif s1 > zero(T) && s2 < zero(T)
        return :IV
    else
        return :V
    end
end

# =============================================================================
# SVARMAModel{T} Type Definition
# =============================================================================

"""
    SVARMAModel{T<:AbstractFloat} <: AbstractVARModel

Strong Structural VARMA(p, q) model with mutually and serially independent
non-Gaussian structural shocks (Gouriéroux, Monfort & Renne 2020):

```math
Y_t = \\mu + \\sum_{k=1}^{p} \\Phi_k Y_{t-k} + \\varepsilon_t + \\sum_{j=1}^{q} \\Theta_j \\varepsilon_{t-j}, \\qquad \\varepsilon_t = C \\eta_t
```

where:
- `Y_t` is the ``n \\times 1`` vector of endogenous variables at time ``t``
- `\\mu` is the ``n \\times 1`` intercept vector
- `\\Phi_k` are the ``n \\times n`` autoregressive lag coefficient matrices (``k = 1, \\ldots, p``)
- `\\Theta_j` are the ``n \\times n`` moving-average lag coefficient matrices (``j = 1, \\ldots, q``)
- `\\varepsilon_t` is the ``n \\times 1`` reduced-form moving-average innovation vector
- `C` is the ``n \\times n`` nonsingular instantaneous structural impact matrix
- `\\eta_t` is the ``n \\times 1`` vector of independent unit-variance structural shocks

Fields:
- `Y::Matrix{T}` — observed sample (`T_obs × n`)
- `p::Int` — autoregressive lag order
- `q::Int` — moving-average lag order
- `mu::Vector{T}` — intercept vector (`n`)
- `Phi::Array{T,3}` — AR lag matrices (`n × n × p`)
- `Theta::Array{T,3}` — MA lag matrices (`n × n × q`)
- `C::Matrix{T}` — structural impact matrix (`n × n`)
- `shocks::Matrix{T}` — filtered structural shocks (`T_eff × n`)
- `ma_eig_re::Vector{T}` — real parts of MA companion eigenvalues (`n * q`)
- `ma_eig_im::Vector{T}` — imaginary parts of MA companion eigenvalues (`n * q`)
- `regime::Symbol` — `:fundamental`, `:mixed_nonfundamental`, `:purely_nonfundamental`, or `:unit_root`
- `gmr_region::Symbol` — GMR (2020, Fig. 6) region (`:I..:V`, `:unit_root`, or `:none`)
- `distributions::Vector{Symbol}` — marginal shock distribution families (`n`)
- `dist_params::Vector{Vector{T}}` — marginal shock distribution parameters (`n`)
- `loglik::T` — total log-likelihood
- `aic::T` — Akaike information criterion
- `bic::T` — Bayesian information criterion
- `varnames::Vector{String}` — variable names (`n`)
- `shock_names::Vector{String}` — structural shock names (`n`)
- `converged::Bool` — convergence indicator
- `method::Symbol` — estimation or construction method tag
- `vcov::Matrix{T}` — asymptotic parameter covariance matrix (`k × k`, empty when unestimated)
- `se::Vector{T}` — parameter standard errors corresponding to `coef(m)` (empty when unestimated)

Virtual properties accessible via `getproperty`:
- `ma_eigenvalues` — `Complex.(m.ma_eig_re, m.ma_eig_im)`
- `nonfundamental_order` — count of MA companion eigenvalues strictly outside the unit disk
- `Sigma` — reduced-form innovation covariance `C * C'`
- `residuals` — alias for `shocks`
- `distribution` — common family symbol or `:heterogeneous`
- `Phi_list`, `Theta_list` — lag-matrix views as `Vector{Matrix{T}}`
"""
struct SVARMAModel{T<:AbstractFloat} <: AbstractVARModel
    Y::Matrix{T}
    p::Int
    q::Int
    mu::Vector{T}
    Phi::Array{T,3}
    Theta::Array{T,3}
    C::Matrix{T}
    shocks::Matrix{T}
    ma_eig_re::Vector{T}
    ma_eig_im::Vector{T}
    regime::Symbol
    gmr_region::Symbol
    distributions::Vector{Symbol}
    dist_params::Vector{Vector{T}}
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    shock_names::Vector{String}
    converged::Bool
    method::Symbol
    vcov::Matrix{T}
    se::Vector{T}

    function SVARMAModel{T}(
        Y::AbstractMatrix,
        p::Integer,
        q::Integer,
        mu::AbstractVector,
        Phi::AbstractArray{<:Real,3},
        Theta::AbstractArray{<:Real,3},
        C::AbstractMatrix,
        shocks::AbstractMatrix,
        ma_eig_re::AbstractVector,
        ma_eig_im::AbstractVector,
        regime::Symbol,
        gmr_region::Symbol,
        distributions::AbstractVector,
        dist_params::AbstractVector,
        loglik::Real,
        aic::Real,
        bic::Real,
        varnames::AbstractVector,
        shock_names::AbstractVector,
        converged::Bool,
        method::Symbol,
        vcov::AbstractMatrix,
        se::AbstractVector,
    ) where {T<:AbstractFloat}
        Y_mat = Matrix{T}(Y)
        T_obs, n = size(Y_mat)
        p_int = Int(p)
        q_int = Int(q)
        p_int >= 0 || throw(ArgumentError("p must be >= 0, got $p_int"))
        q_int >= 0 || throw(ArgumentError("q must be >= 0, got $q_int"))
        mu_vec = Vector{T}(mu)
        length(mu_vec) == n || throw(ArgumentError("mu length ($(length(mu_vec))) must match n ($n)"))
        Phi_arr = Array{T,3}(Phi)
        size(Phi_arr) == (n, n, p_int) ||
            throw(ArgumentError("Phi size $(size(Phi_arr)) must match ($n, $n, $p_int)"))
        Theta_arr = Array{T,3}(Theta)
        size(Theta_arr) == (n, n, q_int) ||
            throw(ArgumentError("Theta size $(size(Theta_arr)) must match ($n, $n, $q_int)"))
        C_mat = Matrix{T}(C)
        size(C_mat) == (n, n) || throw(ArgumentError("C size $(size(C_mat)) must match ($n, $n)"))
        shocks_mat = Matrix{T}(shocks)
        size(shocks_mat, 2) == n ||
            throw(ArgumentError("shocks columns ($(size(shocks_mat, 2))) must match n ($n)"))
        re_vec = Vector{T}(ma_eig_re)
        im_vec = Vector{T}(ma_eig_im)
        length(re_vec) == n * q_int && length(im_vec) == n * q_int ||
            throw(ArgumentError("ma_eig_re/im must have length n*q = $(n * q_int)"))
        dists_vec = Symbol[Symbol(d) for d in distributions]
        length(dists_vec) == n ||
            throw(ArgumentError("distributions length ($(length(dists_vec))) must match n ($n)"))
        dp_vec = Vector{T}[Vector{T}(dp) for dp in dist_params]
        length(dp_vec) == n ||
            throw(ArgumentError("dist_params length ($(length(dp_vec))) must match n ($n)"))
        vn_vec = String[string(v) for v in varnames]
        length(vn_vec) == n ||
            throw(ArgumentError("varnames length ($(length(vn_vec))) must match n ($n)"))
        sn_vec = String[string(s) for s in shock_names]
        length(sn_vec) == n ||
            throw(ArgumentError("shock_names length ($(length(sn_vec))) must match n ($n)"))
        vcov_mat = Matrix{T}(vcov)
        isempty(vcov_mat) || size(vcov_mat, 1) == size(vcov_mat, 2) ||
            throw(ArgumentError("vcov must be square or empty, got $(size(vcov_mat))"))
        se_vec = Vector{T}(se)
        new{T}(
            Y_mat, p_int, q_int, mu_vec, Phi_arr, Theta_arr, C_mat, shocks_mat,
            re_vec, im_vec, regime, gmr_region, dists_vec, dp_vec,
            T(loglik), T(aic), T(bic), vn_vec, sn_vec, converged, method,
            vcov_mat, se_vec,
        )
    end
end

function SVARMAModel{T}(
    Y::AbstractMatrix,
    p::Integer,
    q::Integer,
    mu::AbstractVector,
    Phi::AbstractArray{<:Real,3},
    Theta::AbstractArray{<:Real,3},
    C::AbstractMatrix,
    shocks::AbstractMatrix,
    ma_eig_re::AbstractVector,
    ma_eig_im::AbstractVector,
    regime::Symbol,
    gmr_region::Symbol,
    distributions::AbstractVector,
    dist_params::AbstractVector,
    loglik::Real,
    aic::Real,
    bic::Real,
    varnames::AbstractVector,
    shock_names::AbstractVector,
    converged::Bool,
    method::Symbol,
) where {T<:AbstractFloat}
    SVARMAModel{T}(
        Y, p, q, mu, Phi, Theta, C, shocks, ma_eig_re, ma_eig_im,
        regime, gmr_region, distributions, dist_params, loglik, aic, bic,
        varnames, shock_names, converged, method, zeros(T, 0, 0), T[],
    )
end

function SVARMAModel(
    Y::AbstractMatrix,
    p::Integer,
    q::Integer,
    mu::AbstractVector,
    Phi::AbstractArray{<:Real,3},
    Theta::AbstractArray{<:Real,3},
    C::AbstractMatrix,
    shocks::AbstractMatrix,
    ma_eig_re::AbstractVector,
    ma_eig_im::AbstractVector,
    regime::Symbol,
    gmr_region::Symbol,
    distributions::AbstractVector,
    dist_params::AbstractVector,
    loglik::Real,
    aic::Real,
    bic::Real,
    varnames::AbstractVector,
    shock_names::AbstractVector,
    converged::Bool,
    method::Symbol,
)
    T = float(promote_type(eltype(Y), eltype(mu), eltype(Phi), eltype(Theta), eltype(C)))
    SVARMAModel{T}(
        Y, p, q, mu, Phi, Theta, C, shocks, ma_eig_re, ma_eig_im,
        regime, gmr_region, distributions, dist_params, loglik, aic, bic,
        varnames, shock_names, converged, method,
    )
end

function SVARMAModel(
    Y::AbstractMatrix,
    p::Integer,
    q::Integer,
    mu::AbstractVector,
    Phi::AbstractArray{<:Real,3},
    Theta::AbstractArray{<:Real,3},
    C::AbstractMatrix,
    shocks::AbstractMatrix,
    ma_eig_re::AbstractVector,
    ma_eig_im::AbstractVector,
    regime::Symbol,
    gmr_region::Symbol,
    distributions::AbstractVector,
    dist_params::AbstractVector,
    loglik::Real,
    aic::Real,
    bic::Real,
    varnames::AbstractVector,
    shock_names::AbstractVector,
    converged::Bool,
    method::Symbol,
    vcov::AbstractMatrix,
    se::AbstractVector,
)
    T = float(promote_type(eltype(Y), eltype(mu), eltype(Phi), eltype(Theta), eltype(C)))
    SVARMAModel{T}(
        Y, p, q, mu, Phi, Theta, C, shocks, ma_eig_re, ma_eig_im,
        regime, gmr_region, distributions, dist_params, loglik, aic, bic,
        varnames, shock_names, converged, method, vcov, se,
    )
end

# =============================================================================
# Properties & Accessors
# =============================================================================

function Base.getproperty(m::SVARMAModel{T}, s::Symbol) where {T}
    if s === :ma_eigenvalues
        return Complex.(getfield(m, :ma_eig_re), getfield(m, :ma_eig_im))
    elseif s === :nonfundamental_order
        re = getfield(m, :ma_eig_re)
        im_v = getfield(m, :ma_eig_im)
        tol_T = max(T(1e-8), T(16) * eps(T))
        return count(i -> hypot(re[i], im_v[i]) > one(T) + tol_T, eachindex(re))
    elseif s === :residuals
        return getfield(m, :shocks)
    elseif s === :Sigma
        C_mat = getfield(m, :C)
        return C_mat * transpose(C_mat)
    elseif s === :distribution
        ds = getfield(m, :distributions)
        isempty(ds) && return :none
        return length(unique(ds)) == 1 ? ds[1] : :heterogeneous
    elseif s === :Phi_list
        Ph = getfield(m, :Phi)
        return Matrix{T}[Ph[:, :, k] for k in 1:getfield(m, :p)]
    elseif s === :Theta_list
        Th = getfield(m, :Theta)
        return Matrix{T}[Th[:, :, k] for k in 1:getfield(m, :q)]
    end
    return getfield(m, s)
end

function Base.propertynames(::SVARMAModel, private::Bool=false)
    return (
        fieldnames(SVARMAModel)...,
        :ma_eigenvalues, :nonfundamental_order, :residuals, :Sigma,
        :distribution, :Phi_list, :Theta_list,
    )
end

nvars(m::SVARMAModel) = size(m.Y, 2)
nlags(m::SVARMAModel) = m.p
effective_nobs(m::SVARMAModel) = size(m.Y, 1) - m.p
varnames(m::SVARMAModel) = m.varnames

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.nobs(m::SVARMAModel) = size(m.Y, 1)
StatsAPI.coef(m::SVARMAModel{T}) where {T} = vcat(m.mu, vec(m.Phi), vec(m.Theta), vec(m.C))
StatsAPI.vcov(m::SVARMAModel) = m.vcov
StatsAPI.stderror(m::SVARMAModel) = m.se
StatsAPI.residuals(m::SVARMAModel) = m.shocks
StatsAPI.loglikelihood(m::SVARMAModel) = m.loglik
StatsAPI.aic(m::SVARMAModel) = m.aic
StatsAPI.bic(m::SVARMAModel) = m.bic
function StatsAPI.dof(m::SVARMAModel)
    n = nvars(m)
    n + m.p * n^2 + m.q * n^2 + n^2 + sum(length, m.dist_params; init=0)
end
StatsAPI.dof_residual(m::SVARMAModel) = size(m.shocks, 1) * nvars(m) - StatsAPI.dof(m)
StatsAPI.islinear(::SVARMAModel) = false

# =============================================================================
# Display & Reporting
# =============================================================================

function report(io::IO, m::SVARMAModel{T}) where {T}
    n = nvars(m)
    T_obs = size(m.Y, 1)
    T_eff = size(m.shocks, 1)
    eigs = m.ma_eigenvalues
    dist_str = m.distribution === :heterogeneous ?
        join(string.(m.distributions), ", ") : string(m.distribution)
    max_eig_str = isempty(eigs) ? "—" : _fmt(maximum(abs, eigs); digits=4)
    min_eig_str = isempty(eigs) ? "—" : _fmt(minimum(abs, eigs); digits=4)

    spec = Any[
        "Model"              "SVARMA($(m.p), $(m.q))";
        "Method"             string(m.method);
        "Variables"          n;
        "Observations"       "$T_obs (eff: $T_eff)";
        "Regime"             string(m.regime);
        "Nonfundamental (m)" m.nonfundamental_order;
        "GMR Fig. 6 region"  string(m.gmr_region);
        "Shock family"       dist_str;
        "MA |λ| min / max"   isempty(eigs) ? "—" : "$min_eig_str / $max_eig_str";
        "Log-likelihood"     isfinite(m.loglik) ? _fmt(m.loglik; digits=4) : "—";
        "AIC / BIC"          (isfinite(m.aic) && isfinite(m.bic)) ?
                             "$(_fmt(m.aic; digits=2)) / $(_fmt(m.bic; digits=2))" : "—";
        "Converged"          _yesno(m.converged)
    ]
    _pretty_table(io, spec;
        title = "Strong Structural VARMA($(m.p), $(m.q)) Model",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    _matrix_table(io, m.C, "Structural Impact Matrix (C)";
        row_labels = m.varnames, col_labels = m.shock_names)

    for k in 1:m.p
        _matrix_table(io, @view(m.Phi[:, :, k]), "AR Coefficient Matrix Φ_$k";
            row_labels = m.varnames, col_labels = m.varnames)
    end
    for j in 1:m.q
        _matrix_table(io, @view(m.Theta[:, :, j]), "MA Coefficient Matrix Θ_$j";
            row_labels = m.varnames, col_labels = m.varnames)
    end

    k_coef = n + m.p * n^2 + m.q * n^2 + n^2
    if length(m.se) == k_coef
        c_off = n + m.p * n^2 + m.q * n^2
        c_names = ["C[$i,$j]" for j in 1:n for i in 1:n]
        _coef_table(io, "Structural Impact Matrix (C) Inference",
                    c_names, vec(m.C), Vector{T}(m.se[(c_off + 1):k_coef]);
                    dist=:z, level=DEFAULT_CI_MACRO)
    end
    return nothing
end

report(m::SVARMAModel) = report(stdout, m)
Base.show(io::IO, m::SVARMAModel) = report(io, m)
