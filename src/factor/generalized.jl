# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Generalized Dynamic Factor Model (GDFM) via Spectral Methods.

Implements Forni, Hallin, Lippi & Reichlin (2000, 2005) GDFM:
X_t = χ_t + ξ_t (common + idiosyncratic components)

The common component has a factor structure with frequency-dependent loadings,
estimated via spectral density analysis.

References:
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The generalized dynamic-factor
  model: Identification and estimation. Review of Economics and Statistics.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). The generalized dynamic factor
  model: One-sided estimation and forecasting. Journal of the American Statistical Association.
"""

using LinearAlgebra, Statistics, StatsAPI
using FFTW

# =============================================================================
# Generalized Dynamic Factor Model Type
# =============================================================================

"""
    GeneralizedDynamicFactorModel{T} <: AbstractFactorModel

GDFM with frequency-dependent loadings: Xₜ = χₜ + ξₜ.

Fields: X, factors, common_component, idiosyncratic, loadings_spectral,
spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
q (dynamic factors), r (static factors), bandwidth, kernel, standardized, variance_explained,
varnames, spectral, Z (one-sided generalized-PC weights), factors_onesided,
common_component_onesided.
"""
struct GeneralizedDynamicFactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    common_component::Matrix{T}
    idiosyncratic::Matrix{T}
    loadings_spectral::Array{Complex{T},3}
    spectral_density_X::Array{Complex{T},3}
    spectral_density_chi::Array{Complex{T},3}
    eigenvalues_spectral::Matrix{T}
    frequencies::Vector{T}
    q::Int
    r::Int
    bandwidth::Int
    kernel::Symbol
    standardized::Bool
    variance_explained::Vector{T}
    varnames::Vector{String}
    spectral::Symbol
    Z::Matrix{T}
    factors_onesided::Matrix{T}
    common_component_onesided::Matrix{T}
end

# Backward-compatible constructor (pre-SDFM-12, no one-sided fields)
GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
    spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
    q, r, bandwidth, kernel, standardized, variance_explained, varnames, spectral) where {T} =
    GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
        spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
        q, r, bandwidth, kernel, standardized, variance_explained, varnames, spectral,
        zeros(T, size(X, 2), r), zeros(T, size(X, 1), r), zeros(T, size(X)))

# Backward-compatible constructor (pre-SDFM-11, no spectral)
GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
    spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
    q, r, bandwidth, kernel, standardized, variance_explained, varnames) where {T} =
    GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
        spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
        q, r, bandwidth, kernel, standardized, variance_explained, varnames,
        :smoothed_periodogram)

# Backward-compatible constructor (pre-#722, no varnames)
GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
    spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
    q, r, bandwidth, kernel, standardized, variance_explained) where {T} =
    GeneralizedDynamicFactorModel{T}(X, factors, common_component, idiosyncratic, loadings_spectral,
        spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
        q, r, bandwidth, kernel, standardized, variance_explained,
        ["Var $i" for i in 1:size(X, 2)], :smoothed_periodogram)

# =============================================================================
# GDFM Estimation
# =============================================================================

"""
    estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0,
                  varnames=nothing, spectral=:lag_window,
                  allow_missing=false, em_max_iter=50, em_tol=1e-4,
                  n_lags_proj=0) -> GeneralizedDynamicFactorModel

Estimate a Generalized Dynamic Factor Model using spectral principal components
(Forni, Hallin, Lippi & Reichlin, 2000, 2005), with optional EM-spectral imputation
for missing or ragged-edge observations and multi-lag stacked one-sided projection:

```math
X_t = \\chi_t + \\xi_t, \\qquad F_t^{(s)} = Z^{\\top} X_t^{(s)}, \\qquad X_t^{(s)} = [X_t^{\\top}, X_{t-1}^{\\top}, \\dots, X_{t-s}^{\\top}]^{\\top}
```

where:
- ``X_t`` is the ``N \\times 1`` observed panel vector at time ``t``
- ``\\chi_t`` is the ``N \\times 1`` common component driven by ``q`` dynamic shocks
- ``\\xi_t`` is the ``N \\times 1`` weakly correlated idiosyncratic component
- ``s`` is the stacked projection lag depth (`n_lags_proj`)
- ``X_t^{(s)}`` is the ``N(s + 1) \\times 1`` stacked panel vector
- ``Z`` is the ``N(s + 1) \\times r`` one-sided generalized eigenvector weight matrix
- ``F_t^{(s)}`` is the ``r \\times 1`` one-sided static factor vector

# Arguments
- `X`: Data matrix (`T_obs × N`)
- `q`: Number of dynamic factors (`1 ≤ q ≤ min(T_obs, N)`)

# Keyword Arguments
- `standardize::Bool=true`: Standardize data prior to spectral estimation
- `bandwidth::Int=0`: Lag truncation `M` under `:lag_window` (default `max(3, round(0.5√T))`),
  or frequency-ordinate bandwidth under `:smoothed_periodogram` (default `max(3, round(T^{1/3}))`)
- `kernel::Symbol=:bartlett`: Spectral kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `r::Int=0`: Number of static factors (`0` defaults to `q`)
- `varnames::Union{Nothing,Vector{String}}=nothing`: Names for the `N` panel variables
  (defaults to `"Var 1"`, …). Forwarded by the `TimeSeriesData` dispatch
- `spectral::Symbol=:lag_window`: `:lag_window` (FHLR 2000/2005) or `:smoothed_periodogram` (legacy)
- `allow_missing::Bool=false`: Allow `NaN` entries (interior or ragged-edge) and impute via
  cross-sectional EM-PCA warm-start followed by EM-spectral iteration (`Inf`/`-Inf` and
  all-`NaN` rows or columns are rejected)
- `em_max_iter::Int=50`: Maximum EM iterations when `allow_missing=true` and `X` contains `NaN`
- `em_tol::Real=1e-4`: Relative convergence tolerance on missing-cell updates
  ``\\max |\\Delta \\hat{\\chi}_{\\text{missing}}|``
- `n_lags_proj::Int=0`: Number of past lags `s` included in the stacked FHLR one-sided
  projection (`0` uses contemporaneous weights only)

# Returns
[`GeneralizedDynamicFactorModel`](@ref) with two-sided common/idiosyncratic components
(in-sample FHLR 2000) and one-sided weights `Z`, `factors_onesided`,
`common_component_onesided` (FHLR 2005). When `allow_missing=true`, `model.X` stores
the converged finite EM-imputed panel so downstream `forecast`, `bandpass_gdfm`, and
`historical_decomposition` remain finite. Use `forecast(gdfm, h; method=:one_sided)`
for the projection forecast.

# Example
```julia
gdfm = estimate_gdfm(X, 3)
common_variance_share(gdfm)  # Fraction of variance explained by common component
```
"""
function estimate_gdfm(X::AbstractMatrix{T}, q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett, r::Int=0,
    varnames::Union{Nothing,Vector{String}}=nothing,
    spectral::Symbol=:lag_window,
    allow_missing::Bool=false,
    em_max_iter::Int=50,
    em_tol::Real=1e-4,
    n_lags_proj::Int=0,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    if allow_missing
        _gdfm_validate_missing_data(X, "X")
    else
        _validate_data(X, "X")
    end
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))
    validate_option(spectral, "spectral", (:lag_window, :smoothed_periodogram))
    em_max_iter >= 1 || throw(ArgumentError("em_max_iter must be >= 1, got $em_max_iter"))
    em_tol > 0 || throw(ArgumentError("em_tol must be > 0, got $em_tol"))
    n_lags_proj >= 0 || throw(ArgumentError("n_lags_proj must be >= 0, got $n_lags_proj"))
    n_lags_proj < T_obs || throw(ArgumentError("n_lags_proj must be < T_obs ($T_obs), got $n_lags_proj"))

    vn = something(varnames, ["Var $i" for i in 1:N])
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $N columns"))

    r_static = r == 0 ? q : r
    r_static < q && throw(ArgumentError("r must be >= q"))
    max_r_proj = N * (n_lags_proj + 1)
    r_static <= max_r_proj || throw(ArgumentError("r must be <= $max_r_proj, got $r_static"))
    if bandwidth <= 0
        bandwidth = spectral === :lag_window ? _select_lag_window(T_obs) : _select_bandwidth(T_obs)
    end

    has_missing = allow_missing && any(isnan, X)
    X_cur = if has_missing
        _gdfm_em_spectral_impute(
            X, q, r_static, standardize, bandwidth, kernel, spectral,
            em_max_iter, T(em_tol), n_lags_proj)
    else
        Matrix{T}(X)
    end

    X_original = copy(X_cur)
    X_proc = standardize ? _standardize(X_original) : X_original

    # Spectral analysis
    frequencies, spectral_X = _estimate_spectrum(X_proc, bandwidth, kernel, spectral)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    spectral_chi = _compute_common_spectral_density(loadings, eigenvalues[1:q, :])
    common = _reconstruct_time_domain(loadings, X_proc, frequencies;
        spectral=spectral, bandwidth=bandwidth)
    factors = _extract_time_domain_factors(X_proc, loadings, frequencies;
        spectral=spectral, bandwidth=bandwidth)
    var_explained = _compute_variance_explained(eigenvalues, q)

    Z, F_os, chi_os = _gdfm_onesided_stacked(
        X_proc, spectral_X, spectral_chi, frequencies, spectral, r_static, n_lags_proj)

    # Unstandardize common component if needed
    if standardize
        μ, σ = mean(X_original, dims=1), max.(std(X_original, dims=1), T(1e-10))
        common = common .* σ .+ μ
        chi_os = chi_os .* σ .+ μ
    end
    idiosyncratic = X_original - common

    GeneralizedDynamicFactorModel{T}(X_original, factors, common, idiosyncratic, loadings,
        spectral_X, spectral_chi, eigenvalues, frequencies, q, r_static, bandwidth,
        kernel, standardize, var_explained, vn, spectral, Z, F_os, chi_os)
end

@float_fallback estimate_gdfm X

function Base.show(io::IO, m::GeneralizedDynamicFactorModel{T}) where {T}
    Tobs, N = size(m.X)
    spec = Any[
        "Dynamic factors"  m.q;
        "Static factors"   m.r;
        "Variables"        N;
        "Observations"     Tobs;
        "Kernel"           string(m.kernel);
        "Bandwidth"        m.bandwidth;
        "Spectral"         string(m.spectral);
        "Standardized"     m.standardized ? "Yes" : "No";
        "One-sided r"      size(m.Z, 2)
    ]
    _pretty_table(io, spec;
        title = "Generalized Dynamic Factor Model (q=$(m.q), r=$(m.r))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    n_show = min(m.r, 5)
    cum_var = cumsum(m.variance_explained)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(m.variance_explained[i])
        var_data[i, 3] = _fmt_pct(cum_var[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )
end

# =============================================================================
# Bandwidth Selection
# =============================================================================

"""Automatic frequency-ordinate bandwidth: T^(1/3) (smoothed periodogram)."""
_select_bandwidth(T_obs::Int) = max(3, round(Int, T_obs^(1/3)))

"""Automatic lag-window truncation M: ½√T (FHLR 2005 / Hallin–Liška 2007)."""
_select_lag_window(T_obs::Int) = max(3, round(Int, 0.5 * sqrt(T_obs)))

# =============================================================================
# Spectral Density Estimation
# =============================================================================

"""Dispatch spectral estimator (`:lag_window` or `:smoothed_periodogram`)."""
function _estimate_spectrum(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol, spectral::Symbol) where {T<:AbstractFloat}
    if spectral === :lag_window
        return _estimate_spectral_density_lagwindow(X, bandwidth, kernel)
    end
    T_obs, N = size(X)
    if 2 * bandwidth + 1 < N
        @warn "smoothed periodogram has rank ≤ $(2 * bandwidth + 1) < N=$N; prefer spectral=:lag_window"
    end
    _estimate_spectral_density(X, bandwidth, kernel)
end

"""FHLR lag-window spectral density: (1/2π) Σ_{k=-M}^M w(k/M) Γ̂_k e^{-ikθ}."""
function _estimate_spectral_density_lagwindow(X::AbstractMatrix{T}, M::Int, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    M = max(M, 0)
    n_freq = M + 1
    frequencies = [T(π * h / max(M, 1)) for h in 0:M]
    M == 0 && (frequencies = [zero(T)])

    Xc = X .- mean(X; dims=1)
    Gamma = Vector{Matrix{T}}(undef, M + 1)
    @inbounds for k in 0:M
        if k == 0
            Gamma[1] = (Xc' * Xc) / T_obs
        else
            Gamma[k + 1] = (Xc[1:(T_obs - k), :]' * Xc[(k + 1):T_obs, :]) / T_obs
        end
    end

    spectral = Array{Complex{T},3}(undef, N, N, n_freq)
    twoπ = T(2π)
    @inbounds for h in 1:n_freq
        θ = frequencies[h]
        S = zeros(Complex{T}, N, N)
        S .+= Gamma[1]
        for k in 1:M
            u = k / max(M, 1)
            w = kernel == :bartlett ? (1 - u) :
                kernel == :parzen ? (u <= 0.5 ? 1 - 6u^2 + 6u^3 : 2(1 - u)^3) :
                0.5 * (1 + cos(π * u))
            e = cis(-k * θ)
            Gk = Gamma[k + 1]
            S .+= w * (Gk * e + Gk' * conj(e))
        end
        S ./= twoπ
        spectral[:, :, h] = (S + S') / 2
    end
    frequencies, spectral
end

"""Align eigenvector signs/phases across adjacent frequencies before interpolation."""
function _align_eigenvector_phases(loadings::Array{Complex{T},3}) where {T<:AbstractFloat}
    L = copy(loadings)
    _, q, n_freq = size(L)
    @inbounds for j in 2:n_freq
        for k in 1:q
            overlap = dot(L[:, k, j - 1], L[:, k, j])
            a = abs(overlap)
            if a > eps(T)
                L[:, k, j] .*= conj(overlap) / a
            end
        end
    end
    L
end

"""Estimate spectral density matrix with kernel smoothing (legacy smoothed periodogram)."""
function _estimate_spectral_density(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    n_freq = div(T_obs, 2) + 1
    frequencies = [T(2π * (j-1) / T_obs) for j in 1:n_freq]

    # Periodogram
    X_fft = FFTW.fft(X, 1)
    periodogram = [X_fft[j, :] * X_fft[j, :]' / T_obs for j in 1:n_freq]

    # Kernel smoothing
    weights = _compute_kernel_weights(bandwidth, kernel)

    spectral = Array{Complex{T},3}(undef, N, N, n_freq)
    @inbounds for j in 1:n_freq
        S = zeros(Complex{T}, N, N)
        for k in -bandwidth:bandwidth
            idx = clamp(j + k < 1 ? 2 - (j + k) : (j + k > n_freq ? 2*n_freq - (j + k) : j + k), 1, n_freq)
            S .+= weights[abs(k) + 1] * periodogram[idx]
        end
        spectral[:, :, j] = (S + S') / 2
    end
    frequencies, spectral
end

"""Compute kernel weights for spectral smoothing."""
function _compute_kernel_weights(bandwidth::Int, kernel::Symbol)
    weights = zeros(bandwidth + 1)
    for k in 0:bandwidth
        u = k / (bandwidth + 1)
        weights[k + 1] = kernel == :bartlett ? 1 - u :
                         kernel == :parzen ? (u <= 0.5 ? 1 - 6u^2 + 6u^3 : 2(1-u)^3) :
                         0.5 * (1 + cos(π * u))  # tukey
    end
    total = weights[1] + 2sum(weights[2:end])
    weights ./ total
end

# =============================================================================
# Spectral Eigendecomposition
# =============================================================================

"""Eigendecomposition of spectral density at each frequency."""
function _spectral_eigendecomposition(spectral::Array{Complex{T},3}) where {T<:AbstractFloat}
    N, _, n_freq = size(spectral)
    eigenvalues, eigenvectors = Matrix{T}(undef, N, n_freq), Array{Complex{T},3}(undef, N, N, n_freq)

    @inbounds for j in 1:n_freq
        E = eigen(Hermitian(spectral[:, :, j]))
        idx = sortperm(real.(E.values), rev=true)
        eigenvalues[:, j] = real.(E.values[idx])
        eigenvectors[:, :, j] = E.vectors[:, idx]
    end
    eigenvalues, eigenvectors
end

# =============================================================================
# Common Component Reconstruction
# =============================================================================

"""Compute spectral density of common component from loadings and eigenvalues."""
function _compute_common_spectral_density(loadings::Array{Complex{T},3}, eigenvalues::AbstractMatrix) where {T}
    N, q, n_freq = size(loadings)
    spectral_chi = Array{Complex{T},3}(undef, N, N, n_freq)
    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        spectral_chi[:, :, j] = L * Diagonal(eigenvalues[:, j]) * L'
    end
    spectral_chi
end

"""Reflect out-of-range sample index `s` symmetrically into `1:T_obs`."""
function _reflect_index(s::Int, T_obs::Int)
    T_obs <= 1 && return 1
    while s < 1 || s > T_obs
        s = s < 1 ? 2 - s : 2 * T_obs - s
    end
    s
end

"""Symmetric reflection padding of `X` (`T_obs × N`) by `M` lags/leads (`(T_obs + 2M) × N`)."""
function _reflect_pad(X::AbstractMatrix{T}, M::Int) where {T<:AbstractFloat}
    T_obs, N = size(X)
    M <= 0 && return Matrix{T}(X)
    X_pad = Matrix{T}(undef, T_obs + 2 * M, N)
    @inbounds for s in (1 - M):(T_obs + M)
        idx = _reflect_index(s, T_obs)
        X_pad[s + M, :] .= @view(X[idx, :])
    end
    X_pad
end

"""Finite half-width `M` for symmetric two-sided time-domain GDFM filtering."""
function _gdfm_filter_halfwidth(T_obs::Int, n_freq::Int, spectral::Symbol, bandwidth::Int)
    T_obs <= 1 && return 0
    if spectral === :lag_window
        return min(max(n_freq - 1, 0), T_obs - 1)
    end
    M = bandwidth > 0 ? bandwidth : _select_lag_window(T_obs)
    min(max(M, 0), div(T_obs - 1, 2))
end

"""Invert `P(θ) = L(θ) L(θ)ᴴ` to real lag-`k` filter matrices `K_0, ..., K_M` (`K_{-k} = K_kᵀ`)."""
function _projector_filter_kernels(loadings::AbstractArray{Complex{T},3}, T_obs::Int,
    spectral::Symbol, M::Int) where {T<:AbstractFloat}
    N, _, n_freq = size(loadings)
    R_list = Vector{Matrix{T}}(undef, n_freq)
    I_list = Vector{Matrix{T}}(undef, n_freq)
    @inbounds for j in 1:n_freq
        Lj = @view loadings[:, :, j]
        Pj = Lj * Lj'
        R_list[j] = real.(Pj)
        I_list[j] = imag.(Pj)
    end
    K_list = Vector{Matrix{T}}(undef, M + 1)
    if n_freq == 1
        K_list[1] = R_list[1]
        @inbounds for k in 1:M
            K_list[k + 1] = zeros(T, N, N)
        end
        return K_list
    end
    if spectral === :lag_window
        M_grid = n_freq - 1
        inv_n = one(T) / T(2 * M_grid)
        @inbounds for k in 0:M
            K = copy(R_list[1])
            isodd(k) ? (K .-= R_list[n_freq]) : (K .+= R_list[n_freq])
            for j in 2:(n_freq - 1)
                θ = T(π * (j - 1) / M_grid)
                c, s = cos(k * θ), sin(k * θ)
                K .+= (2 * c) .* R_list[j] .+ (2 * s) .* I_list[j]
            end
            w_k = k == M_grid ? T(0.5) : one(T)
            K .*= w_k * inv_n
            K_list[k + 1] = K
        end
    else
        inv_T = one(T) / T(T_obs)
        has_nyquist = iseven(T_obs)
        j_hi = has_nyquist ? n_freq - 1 : n_freq
        @inbounds for k in 0:M
            K = copy(R_list[1])
            if has_nyquist
                isodd(k) ? (K .-= R_list[n_freq]) : (K .+= R_list[n_freq])
            end
            for j in 2:j_hi
                ω = T(2π * (j - 1) / T_obs)
                c, s = cos(k * ω), sin(k * ω)
                K .+= (2 * c) .* R_list[j] .- (2 * s) .* I_list[j]
            end
            K .*= inv_T
            K_list[k + 1] = K
        end
    end
    K_list
end

"""Apply finite-lag symmetric two-sided filter `χ_t = K_0 X_t + Σ_{k=1}^M (K_k X_{t-k} + K_kᵀ X_{t+k})`."""
function _apply_symmetric_filter(X::AbstractMatrix{T}, X_pad::AbstractMatrix{T},
    K_list::Vector{Matrix{T}}) where {T<:AbstractFloat}
    T_obs, _ = size(X)
    M = length(K_list) - 1
    chi = X * transpose(K_list[1])
    @inbounds for k in 1:M
        Kk = K_list[k + 1]
        X_lag = @view X_pad[(M + 1 - k):(M + T_obs - k), :]
        X_lead = @view X_pad[(M + 1 + k):(M + T_obs + k), :]
        mul!(chi, X_lag, transpose(Kk), one(T), one(T))
        mul!(chi, X_lead, Kk, one(T), one(T))
    end
    chi
end

"""Invert `B(θ) = (L(θ)ᴴ L(θ) + 1e-10 I)⁻¹ L(θ)ᴴ` to real lag filters `B_0`, `B_pos`, `B_neg`."""
function _factor_filter_kernels(loadings::Array{Complex{T},3}, T_obs::Int,
    spectral::Symbol, M::Int) where {T<:AbstractFloat}
    N, q, n_freq = size(loadings)
    L_al = _align_eigenvector_phases(loadings)
    BR_list = Vector{Matrix{T}}(undef, n_freq)
    BI_list = Vector{Matrix{T}}(undef, n_freq)
    reg = T(1e-10) * I
    @inbounds for j in 1:n_freq
        Lj = @view L_al[:, :, j]
        Bj = (Lj' * Lj + reg) \ Lj'
        BR_list[j] = real.(Bj)
        BI_list[j] = imag.(Bj)
    end
    B0 = zeros(T, q, N)
    B_pos = Vector{Matrix{T}}(undef, M)
    B_neg = Vector{Matrix{T}}(undef, M)
    if n_freq == 1
        B0 .= BR_list[1]
        @inbounds for k in 1:M
            B_pos[k] = zeros(T, q, N)
            B_neg[k] = zeros(T, q, N)
        end
        return B0, B_pos, B_neg
    end
    if spectral === :lag_window
        M_grid = n_freq - 1
        inv_n = one(T) / T(2 * M_grid)
        @inbounds for k in 0:M
            Ck = copy(BR_list[1])
            isodd(k) ? (Ck .-= BR_list[n_freq]) : (Ck .+= BR_list[n_freq])
            Sk = zeros(T, q, N)
            for j in 2:(n_freq - 1)
                θ = T(π * (j - 1) / M_grid)
                c, s = cos(k * θ), sin(k * θ)
                Ck .+= (2 * c) .* BR_list[j]
                Sk .+= (2 * s) .* BI_list[j]
            end
            w_k = k == M_grid ? T(0.5) : one(T)
            scale = w_k * inv_n
            if k == 0
                B0 .= scale .* Ck
            else
                B_pos[k] = scale .* (Ck .+ Sk)
                B_neg[k] = scale .* (Ck .- Sk)
            end
        end
    else
        inv_T = one(T) / T(T_obs)
        has_nyquist = iseven(T_obs)
        j_hi = has_nyquist ? n_freq - 1 : n_freq
        @inbounds for k in 0:M
            Ck = copy(BR_list[1])
            if has_nyquist
                isodd(k) ? (Ck .-= BR_list[n_freq]) : (Ck .+= BR_list[n_freq])
            end
            Sk = zeros(T, q, N)
            for j in 2:j_hi
                ω = T(2π * (j - 1) / T_obs)
                c, s = cos(k * ω), sin(k * ω)
                Ck .+= (2 * c) .* BR_list[j]
                Sk .+= (2 * s) .* BI_list[j]
            end
            if k == 0
                B0 .= inv_T .* Ck
            else
                B_pos[k] = inv_T .* (Ck .- Sk)
                B_neg[k] = inv_T .* (Ck .+ Sk)
            end
        end
    end
    B0, B_pos, B_neg
end

"""Reconstruct common component in time domain via finite-lag symmetric filtering."""
function _reconstruct_time_domain(loadings::Array{Complex{T},3}, X::AbstractMatrix{T},
    frequencies::Vector{T}=T[]; spectral::Symbol=:auto, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, _ = size(X)
    n_freq = size(loadings, 3)
    spec = if spectral === :auto
        (n_freq == div(T_obs, 2) + 1 && !isempty(frequencies) && n_freq > 1 &&
         abs(frequencies[2] - T(2π / T_obs)) < T(1e-6)) ? :smoothed_periodogram : :lag_window
    else
        spectral
    end
    M = _gdfm_filter_halfwidth(T_obs, n_freq, spec, bandwidth)
    X_pad = _reflect_pad(X, M)
    K_list = _projector_filter_kernels(loadings, T_obs, spec, M)
    _apply_symmetric_filter(X, X_pad, K_list)
end

"""Extract time-domain factors via finite-lag symmetric projection."""
function _extract_time_domain_factors(X::AbstractMatrix{T}, loadings::Array{Complex{T},3},
    frequencies::Vector{T}; spectral::Symbol=:auto, bandwidth::Int=0) where {T<:AbstractFloat}
    T_obs, _ = size(X)
    _, q, n_freq = size(loadings)
    spec = if spectral === :auto
        (n_freq == div(T_obs, 2) + 1 && !isempty(frequencies) && n_freq > 1 &&
         abs(frequencies[2] - T(2π / T_obs)) < T(1e-6)) ? :smoothed_periodogram : :lag_window
    else
        spectral
    end
    M = _gdfm_filter_halfwidth(T_obs, n_freq, spec, bandwidth)
    X_pad = _reflect_pad(X, M)
    B0, B_pos, B_neg = _factor_filter_kernels(loadings, T_obs, spec, M)
    factors = X * transpose(B0)
    @inbounds for k in 1:M
        X_lag = @view X_pad[(M + 1 - k):(M + T_obs - k), :]
        X_lead = @view X_pad[(M + 1 + k):(M + T_obs + k), :]
        mul!(factors, X_lag, transpose(B_pos[k]), one(T), one(T))
        mul!(factors, X_lead, transpose(B_neg[k]), one(T), one(T))
    end

    # Normalize factors to unit variance
    for i in 1:q
        σ = std(factors[:, i])
        σ > T(1e-10) && (factors[:, i] ./= σ)
    end
    factors
end

"""
    historical_decomposition(gdfm::GeneralizedDynamicFactorModel; by=:dynamic_pc) -> HistoricalDecomposition

Decompose each series into the `q` dynamic principal-component paths plus the
idiosyncratic remainder via finite-lag symmetric time-domain filtering
``\\hat{\\chi}_t^{(j)} = \\sum_{k=-M}^{M} K_k^{(j)} X_{t-k}`` with reflection boundary
padding. The `q` paths reconstruct `gdfm.common_component` (including the series
mean under `standardize=true`, matching `estimate_gdfm`).
"""
function historical_decomposition(gdfm::GeneralizedDynamicFactorModel{T};
    by::Symbol=:dynamic_pc) where {T<:AbstractFloat}
    by === :dynamic_pc || throw(ArgumentError("by must be :dynamic_pc, got :$by"))
    X = gdfm.X
    T_obs, N = size(X)
    q = gdfm.q
    X_proc = gdfm.standardized ? _standardize(X) : X
    L = gdfm.loadings_spectral
    n_freq = size(L, 3)
    M = _gdfm_filter_halfwidth(T_obs, n_freq, gdfm.spectral, gdfm.bandwidth)
    X_pad = _reflect_pad(X_proc, M)
    σ = gdfm.standardized ? max.(vec(std(X; dims=1)), T(1e-10)) : ones(T, N)
    contrib = zeros(T, T_obs, N, q + 1)
    @inbounds for k in 1:q
        K_list = _projector_filter_kernels(@view(L[:, k:k, :]), T_obs, gdfm.spectral, M)
        chi = _apply_symmetric_filter(X_proc, X_pad, K_list)
        if gdfm.standardized
            chi .*= σ'
        end
        contrib[:, :, k] = chi
    end
    if gdfm.standardized
        μ = vec(mean(X; dims=1))
        contrib[:, :, 1] .+= μ'
    end
    contrib[:, :, q + 1] = gdfm.idiosyncratic
    initial = zeros(T, T_obs, N)
    names = [["Dynamic factor $j" for j in 1:q]; "Idiosyncratic"]
    HistoricalDecomposition{T}(contrib, initial, copy(X), gdfm.factors, T_obs,
        copy(gdfm.varnames), names, :dynamic_pc)
end

"""Compute variance explained by first q factors (averaged across frequencies)."""
function _compute_variance_explained(eigenvalues::Matrix{T}, q::Int) where {T}
    total = mean(sum(eigenvalues, dims=1))
    [mean(eigenvalues[i, :]) / total for i in 1:q]
end

# =============================================================================
# FHLR (2005) one-sided factors from the (Γχ(0), Γξ(0)) pencil
# =============================================================================

"""Invert a [0, π] spectral density to the lag-`k` autocovariance ``Γ(k)=E[X_t X_{t-k}']``.

The lag-window estimator stores ``Σ(θ)`` on ``[0,π]`` as Hermitian. A real process
has ``Σ(-θ)=Σ(θ)^⊤`` (transpose, not adjoint). Folding with ``Σ(θ)^H`` is a no-op
on a Hermitian slice and recovers only ``(Γ(k)+Γ(-k))/2``. The inverse Fourier
convention matches FHLR: ``Γ(k)=∫ Σ(θ) exp(-i k θ) dθ``.
"""
function _gamma_from_spectrum(S::Array{Complex{T},3}, frequencies::Vector{T}, k::Int) where {T<:AbstractFloat}
    N, _, n_freq = size(S)
    n_freq >= 1 || throw(ArgumentError("spectral density is empty"))
    M = max(n_freq - 1, 1)
    n = 2M
    G = zeros(Complex{T}, N, N)
    twoπ = T(2π)
    @inbounds for j in 0:(n - 1)
        if j <= M
            Sj = S[:, :, min(j + 1, n_freq)]
        else
            j_pos = n - j
            # Real process: Σ(-θ) = transpose(Σ(θ)), not the adjoint.
            Sj = transpose(S[:, :, min(j_pos + 1, n_freq)])
        end
        G .+= Sj .* cis(-k * twoπ * j / n)
    end
    real.(G .* (twoπ / n))
end

"""Generalized eigenvectors of (Γχ(0), Γξ(0)); columns of `Z` satisfy `Z' Γ_X(0) Z ≈ I_r`."""
function _onesided_Z(Γχ0::AbstractMatrix{T}, Γξ0::AbstractMatrix{T}, r::Int) where {T<:AbstractFloat}
    N = size(Γχ0, 1)
    1 <= r <= N || throw(ArgumentError("r must be in [1, $N]"))
    A = Symmetric(T.(real.((Γχ0 + Γχ0') / 2)))
    ξscale = tr(real.(Γξ0)) / N
    jitter = max(T(1e-10) * (ξscale + one(T)), T(1e-12))
    B0 = T.(real.((Γξ0 + Γξ0') / 2))
    E = nothing
    for scale in (1, 10, 100, 1_000, 10_000, 100_000)
        B = Symmetric(B0 + T(scale) * jitter * I)
        try
            E = eigen(A, B)
            break
        catch
            continue
        end
    end
    E === nothing && error("generalized eigenproblem (Γχ(0), Γξ(0)) failed even with jitter")
    idx = sortperm(real.(E.values); rev=true)
    vecs = real.(E.vectors[:, idx[1:r]])
    ΓX0 = T.(real.((Γχ0 + Γξ0 + (Γχ0 + Γξ0)') / 2))
    G = vecs' * ΓX0 * vecs
    L = safe_cholesky(Symmetric((G + G') / 2); silent=true)
    Z = vecs / L'
    @inbounds for j in 1:r
        k = argmax(abs.(Z[:, j]))
        Z[k, j] < 0 && (Z[:, j] .*= -1)
    end
    Z
end

"""Contemporaneous FHLR (2005) factors `F_t = Z' X_t` and common component."""
function _gdfm_onesided(X::AbstractMatrix{T}, Sx::Array{Complex{T},3},
    Schi::Array{Complex{T},3}, frequencies::Vector{T}, r::Int) where {T<:AbstractFloat}
    Γχ0 = _gamma_from_spectrum(Schi, frequencies, 0)
    ΓX0 = _gamma_from_spectrum(Sx, frequencies, 0)
    Γξ0 = ΓX0 - Γχ0
    Z = _onesided_Z(Γχ0, Γξ0, r)
    Gz = Z' * ΓX0 * Z
    W = Γχ0 * Z / Symmetric((Gz + Gz') / 2)
    F = X * Z
    chi = F * W'
    Z, F, chi
end

"""Projection weights `Γχ(h) Z (Z' Γ_X(0) Z)⁻¹` for the FHLR (2005) forecast."""
function _fhlr_projection_weights(gdfm::GeneralizedDynamicFactorModel{T}, h::Int) where {T<:AbstractFloat}
    Z = gdfm.Z
    size(Z, 1) == size(gdfm.X, 2) || throw(ArgumentError("one-sided weights Z are empty; re-estimate the GDFM"))
    Γχh = _gamma_from_spectrum(gdfm.spectral_density_chi, gdfm.frequencies, h)
    ΓX0 = _gamma_from_spectrum(gdfm.spectral_density_X, gdfm.frequencies, 0)
    Gz = Z' * ΓX0 * Z
    Γχh * Z / Symmetric((Gz + Gz') / 2)
end

# =============================================================================
# Missing-Data Validation, EM-PCA Warm-Start, Wiener–Kolmogorov & Stacked FHLR
# =============================================================================

"""Validate missing-data matrix when `allow_missing=true`: allow NaN, reject Inf and all-NaN rows/columns."""
function _gdfm_validate_missing_data(X::AbstractMatrix, name::String="X")
    any(isinf, X) && throw(ArgumentError(
        "$name contains Inf/-Inf values; only finite observations and NaN are permitted when allow_missing=true"))
    T_obs, N = size(X)
    @inbounds for t in 1:T_obs
        all(isnan, @view(X[t, :])) && throw(ArgumentError(
            "$name has an all-NaN row at row $t; cannot impute an entirely unobserved period"))
    end
    @inbounds for j in 1:N
        all(isnan, @view(X[:, j])) && throw(ArgumentError(
            "$name has an all-NaN column at column $j; cannot impute an entirely unobserved series"))
    end
    nothing
end

"""Column-median initialization followed by cross-sectional EM-PCA warm-start (Stock & Watson 2002)."""
function _gdfm_init_em_pca(
    X::AbstractMatrix{T},
    r_init::Int,
    standardize::Bool;
    max_iter::Int=20,
    tol::T=T(1e-4),
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    obs_mask = BitMatrix(.!isnan.(X))
    miss_mask = .!obs_mask
    X_cur = Matrix{T}(X)

    @inbounds for j in 1:N
        vals = T[]
        for t in 1:T_obs
            obs_mask[t, j] && push!(vals, X_cur[t, j])
        end
        med_j = T(median(vals))
        for t in 1:T_obs
            obs_mask[t, j] || (X_cur[t, j] = med_j)
        end
    end

    obs_vals = X_cur[obs_mask]
    scale_obs = max(T(std(obs_vals)), T(1e-10))

    r_warm = min(r_init, max(1, N - 1), max(1, T_obs - 1))
    if any(miss_mask) && r_warm >= 1 && N > 1 && T_obs > 1
        for _ in 1:max_iter
            prev_miss = X_cur[miss_mask]
            μ = mean(X_cur; dims=1)
            σ = standardize ? max.(std(X_cur; dims=1), T(1e-10)) : ones(T, 1, N)
            Z = (X_cur .- μ) ./ σ
            sv = svd(Z)
            k_use = min(r_warm, length(sv.S))
            Z_hat = sv.U[:, 1:k_use] * Diagonal(sv.S[1:k_use]) * sv.Vt[1:k_use, :]
            X_hat = Z_hat .* σ .+ μ
            X_cur[miss_mask] .= X_hat[miss_mask]
            new_miss = X_cur[miss_mask]
            denom = max(maximum(abs, new_miss), scale_obs, T(1e-10))
            maximum(abs.(new_miss .- prev_miss)) / denom < tol && break
        end
    end

    X_cur, obs_mask, miss_mask, scale_obs
end

"""Linearly interpolate a [0, π] Hermitian spectral density tensor onto `M_fine + 1` uniform ordinates."""
function _interp_spectrum_grid(S::Array{Complex{T},3}, src_freq::Vector{T}, M_fine::Int) where {T<:AbstractFloat}
    N, _, n_src = size(S)
    n_out = M_fine + 1
    out_freq = [T(π * j / M_fine) for j in 0:M_fine]
    S_out = Array{Complex{T},3}(undef, N, N, n_out)
    @inbounds for j in 1:n_out
        ω = out_freq[j]
        if n_src == 1 || ω <= src_freq[1]
            Sj = @view S[:, :, 1]
            S_out[:, :, j] = (Sj + Sj') / T(2)
        elseif ω >= src_freq[end]
            Sj = @view S[:, :, n_src]
            S_out[:, :, j] = (Sj + Sj') / T(2)
        else
            i2 = searchsortedfirst(src_freq, ω)
            i1 = max(i2 - 1, 1)
            i2 = min(i2, n_src)
            Δ = src_freq[i2] - src_freq[i1]
            t = Δ > zero(T) ? (ω - src_freq[i1]) / Δ : zero(T)
            Sj = (one(T) - t) .* @view(S[:, :, i1]) .+ t .* @view(S[:, :, i2])
            S_out[:, :, j] = (Sj + Sj') / T(2)
        end
    end
    out_freq, S_out
end

"""Sample-scale spectral autocovariances `Γ_X(k), Γ_χ(k)` for `k = 0..k_max`, avoiding DFT lag aliasing."""
function _gdfm_spectral_autocovariances(
    Sx::Array{Complex{T},3},
    Schi::Array{Complex{T},3},
    frequencies::Vector{T},
    spectral::Symbol,
    T_orig::Int,
    k_max::Int,
) where {T<:AbstractFloat}
    N, _, n_freq = size(Sx)
    k_use = max(k_max, 0)
    Γ_X = Vector{Matrix{T}}(undef, k_use + 1)
    Γ_chi = Vector{Matrix{T}}(undef, k_use + 1)

    if spectral === :lag_window
        M = max(n_freq - 1, 1)
        if n_freq == 1 || k_use < M
            freq_use, Sx_use, Schi_use = frequencies, Sx, Schi
        else
            M_fine = max(M, k_use + 2)
            freq_use, Sx_use = _interp_spectrum_grid(Sx, frequencies, M_fine)
            _, Schi_use = _interp_spectrum_grid(Schi, frequencies, M_fine)
        end
        @inbounds for k in 0:k_use
            Γ_X[k + 1] = _gamma_from_spectrum(Sx_use, freq_use, k)
            Γ_chi[k + 1] = _gamma_from_spectrum(Schi_use, freq_use, k)
        end
    else
        # Smoothed periodogram omits 1/(2π); exact T_orig-point IDFT when k_use < T_orig ÷ 2
        if k_use < div(T_orig, 2) && n_freq == div(T_orig, 2) + 1
            twoπ = T(2π)
            inv_T = one(T) / T(T_orig)
            @inbounds for k in 0:k_use
                GX = zeros(Complex{T}, N, N)
                Gchi = zeros(Complex{T}, N, N)
                for j in 0:(T_orig - 1)
                    phase = cis(-k * twoπ * j / T_orig)
                    if j < n_freq
                        GX .+= @view(Sx[:, :, j + 1]) .* phase
                        Gchi .+= @view(Schi[:, :, j + 1]) .* phase
                    else
                        j_pos = T_orig - j
                        GX .+= transpose(@view(Sx[:, :, j_pos + 1])) .* phase
                        Gchi .+= transpose(@view(Schi[:, :, j_pos + 1])) .* phase
                    end
                end
                Γ_X[k + 1] = real.(GX .* inv_T)
                Γ_chi[k + 1] = real.(Gchi .* inv_T)
            end
        else
            M_fine = max(n_freq - 1, k_use + 2)
            freq_use, Sx_use = _interp_spectrum_grid(Sx, frequencies, M_fine)
            _, Schi_use = _interp_spectrum_grid(Schi, frequencies, M_fine)
            inv_2π = one(T) / T(2π)
            @inbounds for k in 0:k_use
                Γ_X[k + 1] = _gamma_from_spectrum(Sx_use, freq_use, k) .* inv_2π
                Γ_chi[k + 1] = _gamma_from_spectrum(Schi_use, freq_use, k) .* inv_2π
            end
        end
    end
    Γ_X[1] = Matrix{T}((Γ_X[1] + Γ_X[1]') / T(2))
    Γ_chi[1] = Matrix{T}((Γ_chi[1] + Γ_chi[1]') / T(2))
    Γ_X, Γ_chi
end

"""Exact linear MMSE Wiener–Kolmogorov projection `E[χ_{t_target} | X_{obs in t_lo:t_hi}]` and `diag(C Σ⁻¹ C')`."""
function _gdfm_wk_project(
    X_obs_mat::AbstractMatrix{T},
    obs_mask::AbstractMatrix{Bool},
    Γ_X_pos::Vector{Matrix{T}},
    Γ_chi_pos::Vector{Matrix{T}},
    t_lo::Int,
    t_hi::Int,
    t_target::Int,
) where {T<:AbstractFloat}
    N = size(X_obs_mat, 2)
    obs_t = Int[]
    obs_i = Int[]
    y_obs = T[]
    @inbounds for t in t_lo:t_hi, i in 1:N
        if obs_mask[t, i]
            push!(obs_t, t)
            push!(obs_i, i)
            push!(y_obs, X_obs_mat[t, i])
        end
    end
    m_obs = length(obs_t)
    m_obs == 0 && return zeros(T, N), zeros(T, N)

    Σ_obs = Matrix{T}(undef, m_obs, m_obs)
    C_chi_obs = Matrix{T}(undef, N, m_obs)
    for b in 1:m_obs
        tb = obs_t[b]
        ib = obs_i[b]
        dt_t = t_target - tb
        if dt_t >= 0
            G_ct = Γ_chi_pos[dt_t + 1]
            @inbounds for row in 1:N
                C_chi_obs[row, b] = G_ct[row, ib]
            end
        else
            G_ct = Γ_chi_pos[-dt_t + 1]
            @inbounds for row in 1:N
                C_chi_obs[row, b] = G_ct[ib, row]
            end
        end
        for a in 1:b
            ta = obs_t[a]
            ia = obs_i[a]
            dt = ta - tb
            val = dt >= 0 ? Γ_X_pos[dt + 1][ia, ib] : Γ_X_pos[-dt + 1][ib, ia]
            @inbounds begin
                Σ_obs[a, b] = val
                Σ_obs[b, a] = val
            end
        end
    end

    Σ_sym = Matrix{T}((Σ_obs + Σ_obs') / T(2))
    L_cov = try
        safe_cholesky(Σ_sym; silent=true)
    catch
        nothing
    end
    if L_cov !== nothing
        v = L_cov' \ (L_cov \ y_obs)
        chi_hat = Vector{T}(C_chi_obs * v)
        M_half = L_cov \ Matrix{T}(C_chi_obs')
        proj_var = Vector{T}(vec(sum(abs2, M_half; dims=1)))
    else
        Σ_inv = robust_inv(Σ_sym; silent=true)
        chi_hat = Vector{T}(C_chi_obs * (Σ_inv * y_obs))
        proj_var = Vector{T}(vec(sum(C_chi_obs .* (C_chi_obs * Σ_inv); dims=2)))
    end
    chi_hat, proj_var
end

"""EM-spectral missing-data imputation for `estimate_gdfm(..., allow_missing=true)`."""
function _gdfm_em_spectral_impute(
    X::AbstractMatrix{T},
    q::Int,
    r_static::Int,
    standardize::Bool,
    bandwidth::Int,
    kernel::Symbol,
    spectral::Symbol,
    em_max_iter::Int,
    em_tol::T,
    n_lags_proj::Int,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    X_cur, obs_mask, miss_mask, scale_obs = _gdfm_init_em_pca(
        X, max(q, r_static), standardize; tol=min(em_tol, T(1e-4)))
    X_raw = Matrix{T}(X)
    L_em = min(max(2, n_lags_proj), max(T_obs - 1, 0))
    t_bnd_start = max(1, T_obs - 2)
    tail_missing_rows = [t for t in t_bnd_start:T_obs if any(!, @view(obs_mask[t, :]))]

    for _ in 1:em_max_iter
        prev_miss = X_cur[miss_mask]
        μ = vec(mean(X_cur; dims=1))
        σ = standardize ? max.(vec(std(X_cur; dims=1)), T(1e-10)) : ones(T, N)
        X_proc = standardize ? (X_cur .- μ') ./ σ' : copy(X_cur)

        frequencies, spectral_X = _estimate_spectrum(X_proc, bandwidth, kernel, spectral)
        eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
        loadings = eigenvectors[:, 1:q, :]
        common_proc = _reconstruct_time_domain(loadings, X_proc, frequencies;
            spectral=spectral, bandwidth=bandwidth)

        if !isempty(tail_missing_rows)
            evals_q = max.(eigenvalues[1:q, :], zero(T))
            spectral_chi = _compute_common_spectral_density(loadings, evals_q)
            k_max_wk = min(max(T_obs - 1, 0), 2 * L_em)
            Γ_X_pos, Γ_chi_pos = _gdfm_spectral_autocovariances(
                spectral_X, spectral_chi, frequencies, spectral, T_obs, k_max_wk)
            X_obs_proc = (X_raw .- μ') ./ σ'
            for t in tail_missing_rows
                t_lo = max(1, t - L_em)
                t_hi = min(T_obs, t + L_em)
                chi_bnd, _ = _gdfm_wk_project(
                    X_obs_proc, obs_mask, Γ_X_pos, Γ_chi_pos, t_lo, t_hi, t)
                if standardize
                    common_proc[t, :] .= chi_bnd
                else
                    common_proc[t, :] .= chi_bnd .+ μ
                end
            end
        end

        common_unstd = standardize ? (common_proc .* σ' .+ μ') : common_proc
        X_cur[miss_mask] .= common_unstd[miss_mask]

        new_miss = X_cur[miss_mask]
        denom = max(maximum(abs, new_miss), scale_obs, T(1e-10))
        maximum(abs.(new_miss .- prev_miss)) / denom < em_tol && break
    end
    X_cur
end

"""Construct causal stacked lag matrix `[X_t, X_{t-1}, ..., X_{t-s}]` (`T_obs × N*(s+1)`)."""
function _gdfm_stacked_matrix(X::AbstractMatrix{T}, s::Int) where {T<:AbstractFloat}
    s == 0 && return Matrix{T}(X)
    T_obs, N = size(X)
    X_s = zeros(T, T_obs, N * (s + 1))
    @inbounds for lag in 0:s
        cols = (lag * N + 1):((lag + 1) * N)
        if lag == 0
            X_s[:, cols] .= X
        else
            X_s[(lag + 1):T_obs, cols] .= @view X[1:(T_obs - lag), :]
        end
    end
    X_s
end

"""Construct symmetric block-Toeplitz covariance `Γ^{(s)}` (`N*(s+1) × N*(s+1)`) from `Γ_pos[1:(s+1)]`."""
function _gdfm_block_toeplitz(Γ_pos::Vector{Matrix{T}}, s::Int) where {T<:AbstractFloat}
    N = size(Γ_pos[1], 1)
    dim_s = N * (s + 1)
    G = Matrix{T}(undef, dim_s, dim_s)
    @inbounds for j in 1:(s + 1), i in 1:(s + 1)
        ri = ((i - 1) * N + 1):(i * N)
        cj = ((j - 1) * N + 1):(j * N)
        k = j - i
        if k >= 0
            G[ri, cj] .= Γ_pos[k + 1]
        else
            G[ri, cj] .= transpose(Γ_pos[-k + 1])
        end
    end
    Matrix{T}((G + G') / T(2))
end

"""Stacked multi-lag FHLR (2005) factors `F_t = Z' X_t^{(s)}` and common component."""
function _gdfm_onesided_stacked(
    X::AbstractMatrix{T},
    Sx::Array{Complex{T},3},
    Schi::Array{Complex{T},3},
    frequencies::Vector{T},
    spectral::Symbol,
    r::Int,
    s::Int,
) where {T<:AbstractFloat}
    s == 0 && return _gdfm_onesided(X, Sx, Schi, frequencies, r)
    T_obs, _ = size(X)
    Γ_X_pos, Γ_chi_pos = _gdfm_spectral_autocovariances(Sx, Schi, frequencies, spectral, T_obs, s)
    ΓX_s = _gdfm_block_toeplitz(Γ_X_pos, s)
    Γχ_s = _gdfm_block_toeplitz(Γ_chi_pos, s)
    Γξ_s = Matrix{T}((ΓX_s - Γχ_s + (ΓX_s - Γχ_s)') / T(2))
    Z = _onesided_Z(Γχ_s, Γξ_s, r)
    Gz = Matrix{T}((Z' * ΓX_s * Z + (Z' * ΓX_s * Z)') / T(2))
    C_chi0 = hcat([Γ_chi_pos[lag + 1] for lag in 0:s]...)
    W0 = C_chi0 * Z * robust_inv(Symmetric(Gz); silent=true)
    X_s = _gdfm_stacked_matrix(X, s)
    F = X_s * Z
    chi = F * transpose(W0)
    Z, F, chi
end

# =============================================================================
# Ragged-Edge Wiener–Kolmogorov Nowcasting (`nowcast_gdfm`)
# =============================================================================

"""
    GDFMNowcastResult{T<:AbstractFloat}

Result of ragged-edge Wiener–Kolmogorov nowcasting and missing-data imputation from an
estimated [`GeneralizedDynamicFactorModel`](@ref):

```math
\\hat{\\chi}_{T_r + h \\mid T_r} = C_{\\chi, \\text{obs}} \\Sigma_{\\text{obs}, \\text{obs}}^{-1} X_{\\text{obs}}, \\qquad
\\text{MSE}_i = \\left(\\hat{\\Gamma}_{\\chi}(0) - C_{\\chi, \\text{obs}} \\Sigma_{\\text{obs}, \\text{obs}}^{-1} C_{\\text{obs}, \\chi}\\right)_{ii} + \\max\\left(\\hat{\\Gamma}_{\\xi}(0)_{ii},\\, 0\\right)
```

where:
- ``\\hat{\\chi}_{T_r + h \\mid T_r}`` is the conditional expectation of the ``N \\times 1`` common component at horizon ``T_r + h``
- ``X_{\\text{obs}}`` stacks all finite entries in the ragged-edge window ``X_{\\text{ragged}}[\\max(1, T_r - \\text{n\\_lags}):T_r, :]``
- ``\\Sigma_{\\text{obs}, \\text{obs}}`` is the spectral autocovariance matrix of ``X_{\\text{obs}}``
- ``C_{\\chi, \\text{obs}} = C_{\\text{obs}, \\chi}^{\\top}`` is the cross-covariance between ``\\chi_{T_r + h}`` and ``X_{\\text{obs}}``
- ``\\hat{\\Gamma}_{\\chi}(0)`` and ``\\hat{\\Gamma}_{\\xi}(0)`` are the lag-0 common and idiosyncratic covariance matrices
- ``\\text{MSE}_i`` is the theoretical predictive mean squared error for variable ``i`` prior to unstandardization

# Fields
- `nowcast::Vector{T}`: Unstandardized target common-component nowcast/forecast at ``T_r + h`` (length `N`)
- `se::Vector{T}`: Unstandardized conditional predictive standard errors (length `N`)
- `panel_imputed::Matrix{T}`: Unstandardized panel (`T_r × N`) with missing entries imputed by row-wise conditional expectation and observed entries preserved
- `horizon::Int`: Nowcast/forecast horizon `h` relative to `T_r`
- `n_lags::Int`: Lag window depth `n_lags` used in conditioning
- `observed_mask::BitMatrix`: Observation mask (`T_r × N`, `true` where `X_ragged` is finite)
- `varnames::Vector{String}`: Variable names inherited from the GDFM (length `N`)
"""
struct GDFMNowcastResult{T<:AbstractFloat}
    nowcast::Vector{T}
    se::Vector{T}
    panel_imputed::Matrix{T}
    horizon::Int
    n_lags::Int
    observed_mask::BitMatrix
    varnames::Vector{String}
end

function Base.show(io::IO, res::GDFMNowcastResult{T}) where {T}
    T_r, N = size(res.panel_imputed)
    n_obs = count(res.observed_mask)
    n_miss = length(res.observed_mask) - n_obs
    n_tail_miss = count(!, @view(res.observed_mask[end, :]))
    spec = Any[
        "Variables (N)"       N;
        "Sample rows (T_r)"   T_r;
        "Horizon (h)"         res.horizon;
        "Lag window (n_lags)" res.n_lags;
        "Missing cells"       "$n_miss / $(length(res.observed_mask))";
        "Tail missing (T_r)"  "$n_tail_miss / $N"
    ]
    _pretty_table(io, spec;
        title = "GDFM Ragged-Edge Nowcast (h=$(res.horizon), n_lags=$(res.n_lags))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    n_show = min(N, 10)
    vdata = Matrix{Any}(undef, n_show, 4)
    for i in 1:n_show
        vdata[i, 1] = res.varnames[i]
        vdata[i, 2] = res.observed_mask[end, i] ? "Observed" : "Missing"
        vdata[i, 3] = _fmt(res.nowcast[i]; digits=4)
        vdata[i, 4] = _fmt(res.se[i]; digits=4)
    end
    _pretty_table(io, vdata;
        title = "Nowcast Estimates" * (N > n_show ? " (first $n_show of $N)" : ""),
        column_labels = ["Variable", "T_r Status", "Nowcast", "SE"],
        alignment = [:l, :l, :r, :r],
    )
end

report(io::IO, res::GDFMNowcastResult) = (show(io, res); nothing)
report(res::GDFMNowcastResult) = report(stdout, res)

"""
    nowcast_gdfm(gdfm::GeneralizedDynamicFactorModel{T}, X_ragged::AbstractMatrix{<:Real},
                 h::Int=0; n_lags::Int=2) where {T<:AbstractFloat} -> GDFMNowcastResult{T}

Compute exact minimum-MSE linear Wiener–Kolmogorov nowcasts (`h = 0`), backcasts (`h < 0`),
or short-horizon forecasts (`h > 0`) on a ragged-edge panel `X_ragged` (`T_r × N`) using
spectral autocovariances from a fitted [`GeneralizedDynamicFactorModel`](@ref):

```math
\\hat{\\chi}_{T_r + h \\mid T_r} = C_{\\chi, \\text{obs}} \\Sigma_{\\text{obs}, \\text{obs}}^{-1} X_{\\text{obs}}, \\qquad
\\text{SE}_i = \\sigma_i \\sqrt{\\max\\!\\left(\\left(\\hat{\\Gamma}_{\\chi}(0) - C_{\\chi, \\text{obs}} \\Sigma_{\\text{obs}, \\text{obs}}^{-1} C_{\\text{obs}, \\chi}\\right)_{ii},\\, 0\\right) + \\max\\!\\left(\\hat{\\Gamma}_{\\xi}(0)_{ii},\\, 0\\right)}
```

where:
- ``\\hat{\\chi}_{T_r + h \\mid T_r}`` is the conditional expectation of the common component at ``T_r + h`` in standardized space
- ``X_{\\text{obs}}`` collects all finite entries in ``X_{\\text{ragged}}[\\max(1, T_r - \\text{n\\_lags}):T_r, :]`` after applying `gdfm` standardization
- ``\\Sigma_{\\text{obs}, \\text{obs}}`` is the spectral autocovariance matrix of ``X_{\\text{obs}}`` built from ``\\hat{\\Gamma}_X(k)``
- ``C_{\\chi, \\text{obs}} = C_{\\text{obs}, \\chi}^{\\top}`` is the cross-covariance between ``\\chi_{T_r + h}`` and ``X_{\\text{obs}}`` built from ``\\hat{\\Gamma}_{\\chi}(k)``
- ``\\hat{\\Gamma}_{\\chi}(0)`` and ``\\hat{\\Gamma}_{\\xi}(0) = \\hat{\\Gamma}_X(0) - \\hat{\\Gamma}_{\\chi}(0)`` are the lag-0 common and idiosyncratic covariance matrices
- ``\\sigma_i`` is the series scaling factor (`std(gdfm.X[:, i])` if `gdfm.standardized`, else `1`)

# Arguments
- `gdfm::GeneralizedDynamicFactorModel{T}`: Pre-estimated GDFM supplying spectral densities and scaling
- `X_ragged::AbstractMatrix{<:Real}`: Ragged-edge panel (`T_r × N`) where `NaN` marks missing observations (`Inf`/`-Inf` rejected)
- `h::Int=0`: Horizon relative to `T_r` (`0` = nowcast, `h > 0` = forecast, `h < 0` = backcast)

# Keyword Arguments
- `n_lags::Int=2`: Number of past lags included in the conditioning window `[max(1, T_r - n_lags), T_r]` (`≥ 0`)

# Returns
[`GDFMNowcastResult`](@ref) storing unstandardized `nowcast`, `se`, conditional-expectation
`panel_imputed`, `horizon`, `n_lags`, `observed_mask`, and `varnames`.
"""
function nowcast_gdfm(
    gdfm::GeneralizedDynamicFactorModel{T},
    X_ragged::AbstractMatrix{<:Real},
    h::Int=0;
    n_lags::Int=2,
) where {T<:AbstractFloat}
    n_lags >= 0 || throw(ArgumentError("n_lags must be >= 0, got $n_lags"))
    T_r, N_r = size(X_ragged)
    T_r >= 1 || throw(ArgumentError("X_ragged must have at least 1 row, got $T_r"))
    N = length(gdfm.varnames)
    N_r == N || throw(ArgumentError(
        "X_ragged has $N_r columns, expected $N matching gdfm.varnames"))
    any(isinf, X_ragged) && throw(ArgumentError(
        "X_ragged contains Inf/-Inf values; only finite values and NaN are permitted"))

    X_mat = Matrix{T}(X_ragged)
    obs_mask = BitMatrix(.!isnan.(X_mat))

    if gdfm.standardized
        μ = vec(mean(gdfm.X; dims=1))
        σ = max.(vec(std(gdfm.X; dims=1)), T(1e-10))
        X_proc = (X_mat .- μ') ./ σ'
    else
        μ = zeros(T, N)
        σ = ones(T, N)
        X_proc = copy(X_mat)
    end

    k_max = max(2 * n_lags, abs(h), abs(h + n_lags))
    Γ_X_pos, Γ_chi_pos = _gdfm_spectral_autocovariances(
        gdfm.spectral_density_X, gdfm.spectral_density_chi,
        gdfm.frequencies, gdfm.spectral, size(gdfm.X, 1), k_max)

    t_lo_now = max(1, T_r - n_lags)
    chi_proc, proj_var = _gdfm_wk_project(
        X_proc, obs_mask, Γ_X_pos, Γ_chi_pos, t_lo_now, T_r, T_r + h)

    Γ_xi0 = Γ_X_pos[1] - Γ_chi_pos[1]
    common_err_var = max.(diag(Γ_chi_pos[1]) .- proj_var, zero(T))
    idio_var_pos = max.(diag(Γ_xi0), zero(T))
    se_proc = sqrt.(common_err_var .+ idio_var_pos)

    nowcast_vec = Vector{T}(chi_proc .* σ .+ μ)
    se_vec = Vector{T}(se_proc .* σ)

    panel_imputed = copy(X_mat)
    @inbounds for t in 1:T_r
        any(!, @view(obs_mask[t, :])) || continue
        t_lo_row = max(1, t - n_lags)
        t_hi_row = min(T_r, t + n_lags)
        chi_row_proc, _ = _gdfm_wk_project(
            X_proc, obs_mask, Γ_X_pos, Γ_chi_pos, t_lo_row, t_hi_row, t)
        for i in 1:N
            obs_mask[t, i] || (panel_imputed[t, i] = chi_row_proc[i] * σ[i] + μ[i])
        end
    end

    GDFMNowcastResult{T}(
        nowcast_vec, se_vec, panel_imputed, h, n_lags, obs_mask, copy(gdfm.varnames))
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Predicted values (common component)."""
StatsAPI.predict(m::GeneralizedDynamicFactorModel) = m.common_component

"""Residuals (idiosyncratic component)."""
StatsAPI.residuals(m::GeneralizedDynamicFactorModel) = m.idiosyncratic

"""Number of observations."""
StatsAPI.nobs(m::GeneralizedDynamicFactorModel) = size(m.X, 1)

"""Degrees of freedom."""
StatsAPI.dof(m::GeneralizedDynamicFactorModel) = m.q * size(m.X, 2) * length(m.frequencies) + size(m.X, 1) * m.q

"""R² for each variable."""
function StatsAPI.r2(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [one(T) - var(m.idiosyncratic[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

# =============================================================================
# Information Criteria for GDFM
# =============================================================================

"""
    HallinLiskaResult{T}

Hallin–Liška (2007) information criterion for the number of dynamic factors.

# Fields
- `q::Int`: Selected ``q`` (second ``S_c = 0`` plateau)
- `q_by_c::Vector{Int}`: ``\\hat q(c)`` on the full panel
- `c_grid::Vector{T}`: Tuning-constant grid
- `S_c::Vector{T}`: Cross-subpanel standard deviation of ``\\hat q_j(c)``
- `stability_interval::Tuple{T,T}`: ``c`` interval of the chosen plateau
- `variance_of_q_by_c::T`: Variance of ``\\hat q(c)`` on that plateau
- `q_subpanel::Matrix{Int}`: ``J \\times n_c`` matrix of ``\\hat q_j(c)``
- `penalty::Symbol`: `:p1`, `:p2`, or `:p3`
"""
struct HallinLiskaResult{T<:AbstractFloat}
    q::Int
    q_by_c::Vector{Int}
    c_grid::Vector{T}
    S_c::Vector{T}
    stability_interval::Tuple{T,T}
    variance_of_q_by_c::T
    q_subpanel::Matrix{Int}
    penalty::Symbol
end

function Base.show(io::IO, r::HallinLiskaResult)
    data = Any[
        "q̂"                  r.q;
        "Penalty"             string(r.penalty);
        "Stability interval"  "[$(round(r.stability_interval[1]; digits=3)), $(round(r.stability_interval[2]; digits=3))]";
        "Var(q̂(c))"           _fmt(r.variance_of_q_by_c);
        "c grid"              length(r.c_grid);
        "Subpanels"           size(r.q_subpanel, 1)
    ]
    _pretty_table(io, data;
        title = "Hallin–Liška (2007)  q̂ = $(r.q)",
        column_labels = ["", ""],
        alignment = [:l, :r])
end
report(io::IO, r::HallinLiskaResult) = (show(io, r); nothing)
report(r::HallinLiskaResult) = report(stdout, r)

"""
    BaiNgQResult{T}

Bai–Ng (2007) residual-covariance rank statistics for the number of primitive shocks.

# Fields
- `q_D1::Int`, `q_D2::Int`: Selected ``q`` from ``D_{1,k}`` and ``D_{2,k}``
- `D1::Vector{T}`, `D2::Vector{T}`: Statistics at ``k = 0,\\ldots,r-1``
- `threshold::T`: ``m / \\min(N^{1/2-\\delta}, T^{1/2-\\delta})``
- `r::Int`: Number of static factors in the VAR
"""
struct BaiNgQResult{T<:AbstractFloat}
    q_D1::Int
    q_D2::Int
    D1::Vector{T}
    D2::Vector{T}
    threshold::T
    r::Int
end

function Base.show(io::IO, r::BaiNgQResult)
    data = Any[
        "q̂ (D1)"    r.q_D1;
        "q̂ (D2)"    r.q_D2;
        "Threshold" _fmt(r.threshold);
        "r"         r.r
    ]
    _pretty_table(io, data;
        title = "Bai–Ng (2007)  q̂_D1 = $(r.q_D1)",
        column_labels = ["", ""],
        alignment = [:l, :r])
end
report(io::IO, r::BaiNgQResult) = (show(io, r); nothing)
report(r::BaiNgQResult) = report(stdout, r)

"""
    AmengualWatsonResult

Amengual–Watson (2007): Bai–Ng (2002) IC applied to residuals of ``X_t``
projected on lagged static factors.

# Fields
- `q::Int`: Selected ``q`` (IC2)
- `r_IC1::Int`, `r_IC2::Int`, `r_IC3::Int`: Bai–Ng (2002) choices on the residuals
- `p::Int`: Lag order used in the projection
"""
struct AmengualWatsonResult
    q::Int
    r_IC1::Int
    r_IC2::Int
    r_IC3::Int
    p::Int
end

function Base.show(io::IO, r::AmengualWatsonResult)
    data = Any[
        "q̂ (IC2)"  r.q;
        "r_IC1"    r.r_IC1;
        "r_IC2"    r.r_IC2;
        "r_IC3"    r.r_IC3;
        "p"        r.p
    ]
    _pretty_table(io, data;
        title = "Amengual–Watson (2007)  q̂ = $(r.q)",
        column_labels = ["", ""],
        alignment = [:l, :r])
end
report(io::IO, r::AmengualWatsonResult) = (show(io, r); nothing)
report(r::AmengualWatsonResult) = report(stdout, r)

"""Hallin–Liška penalty ``p(n,T,M)``."""
function _hl_penalty(penalty::Symbol, n::Int, T_obs::Int, M::Int)
    M = max(M, 1)
    n = max(n, 2)
    minNTM = max(min(n, T_obs, M), 2)
    penalty === :p1 && return sqrt(1 / M + 1 / n)
    penalty === :p2 && return sqrt(1 / M + 1 / n) * log(minNTM)
    penalty === :p3 && return sqrt(log(minNTM) / minNTM)
    throw(ArgumentError("penalty must be :p1, :p2, or :p3, got :$penalty"))
end

"""Frequency-averaged dynamic eigenvalues of a (sub)panel."""
function _hl_avg_eigenvalues(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol,
    spectral::Symbol) where {T<:AbstractFloat}
    Xc = X .- mean(X; dims=1)
    _, spec = _estimate_spectrum(Xc, bandwidth, kernel, spectral)
    eigenvalues, _ = _spectral_eigendecomposition(spec)
    vec(mean(eigenvalues; dims=2))
end

"""``IC(k; c) = \\log((1/N) Σ_{j>k} λ̄_j) + k · c · p``; ``k = 0,…,q_{max}``."""
function _hl_ic_path(avg_eig::AbstractVector{T}, q_max::Int, c::T, p::T) where {T<:AbstractFloat}
    N = length(avg_eig)
    q_hi = min(q_max, N - 1)
    best_k, best_ic = 0, T(Inf)
    @inbounds for k in 0:q_hi
        V = sum(@view avg_eig[(k + 1):N]) / N
        V = max(V, T(1e-15))
        ic = log(V) + k * c * p
        if ic < best_ic
            best_ic = ic
            best_k = k
        end
    end
    best_k
end

"""Constant-``q`` runs with ``S_c = 0``. Discard the first ``q_{max}`` (near ``c = 0``);
the estimate is the **longest** remaining interval (short intermediate flats are not
Hallin–Liška's stability plateau)."""
function _hl_second_plateau(c_grid::AbstractVector{T}, q_full::AbstractVector{Int},
    S::AbstractVector{T}, q_max::Int) where {T<:AbstractFloat}
    n_c = length(c_grid)
    n_c >= 1 || return 0, (zero(T), zero(T)), T(NaN)
    stable = S .<= sqrt(eps(T))
    runs = UnitRange{Int}[]
    i = 1
    while i <= n_c
        if stable[i]
            j = i + 1
            q0 = q_full[i]
            while j <= n_c && stable[j] && q_full[j] == q0
                j += 1
            end
            push!(runs, i:(j - 1))
            i = j
        else
            i += 1
        end
    end
    # Drop the overfit plateau(s) at q_max / c ≈ 0
    rest = UnitRange{Int}[]
    for (idx, run) in enumerate(runs)
        q_run = q_full[first(run)]
        if (idx == 1 && first(run) == 1) || q_run == q_max
            continue
        end
        length(run) >= 2 && push!(rest, run)
    end
    if isempty(rest)
        cand = findall(i -> q_full[i] < q_max, 1:n_c)
        if isempty(cand)
            return q_full[end], (c_grid[end], c_grid[end]), zero(T)
        end
        i_star = cand[argmin(S[cand])]
        return q_full[i_star], (c_grid[i_star], c_grid[i_star]), zero(T)
    end
    run = rest[argmax(length.(rest))]
    q = q_full[first(run)]
    q, (c_grid[first(run)], c_grid[last(run)]), zero(T)
end

"""
    hallin_liska(X, q_max; c_grid, subpanels=4, bandwidth=0, kernel=:bartlett,
                 penalty=:p1, spectral=:lag_window, standardize=true) -> HallinLiskaResult

Hallin–Liška (2007) information criterion for the number of dynamic factors.

``IC(q; c) = \\log((1/N)\\sum_{j>q}\\bar\\lambda_j) + q · c · p(N,T)`` is minimised
on a grid of ``c``. Nested sub-panels ``(N_j, T_j)`` give ``S_c = \\mathrm{sd}_j(\\hat q_j(c))``.
The **second** interval with ``S_c = 0`` is the estimate; the first plateau near
``c = 0`` is ``q_{max}`` and is discarded.

# Keyword Arguments
- `c_grid`: Tuning constants (default `range(0, 3, length=100)`)
- `subpanels::Int=4`: Nested ``(N_j, T_j)`` panels, largest first
- `penalty::Symbol=:p1`: `:p1`, `:p2`, or `:p3` (Hallin–Liška §4)
- `spectral::Symbol=:lag_window`: Must be the FHLR lag-window for the theorem
"""
function hallin_liska(X::AbstractMatrix{T}, q_max::Int;
    c_grid=range(zero(T), T(3); length=100),
    subpanels::Int=4,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    penalty::Symbol=:p1,
    spectral::Symbol=:lag_window,
    standardize::Bool=true,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (q_max < 1 || q_max > N - 1) && throw(ArgumentError("q_max must be in [1, $(N-1)]"))
    subpanels < 2 && throw(ArgumentError("subpanels must be ≥ 2"))
    validate_option(penalty, "penalty", (:p1, :p2, :p3))
    validate_option(spectral, "spectral", (:lag_window, :smoothed_periodogram))
    if bandwidth <= 0
        bandwidth = spectral === :lag_window ? _select_lag_window(T_obs) : _select_bandwidth(T_obs)
    end
    X_proc = standardize ? _standardize(X) : X
    c_vec = collect(T.(c_grid))
    n_c = length(c_vec)
    n_c >= 1 || throw(ArgumentError("c_grid must be non-empty"))

    q_sub = Matrix{Int}(undef, subpanels, n_c)
    @inbounds for j in 1:subpanels
        frac = 1 - (j - 1) / (2 * subpanels)
        n_j = max(q_max + 2, round(Int, N * frac))
        t_j = max(2 * bandwidth + 20, round(Int, T_obs * frac))
        n_j = min(n_j, N)
        t_j = min(t_j, T_obs)
        Xj = X_proc[1:t_j, 1:n_j]
        M_j = spectral === :lag_window ? min(bandwidth, _select_lag_window(t_j)) : bandwidth
        avg = _hl_avg_eigenvalues(Xj, M_j, kernel, spectral)
        p_j = T(_hl_penalty(penalty, n_j, t_j, M_j))
        for ic in 1:n_c
            q_sub[j, ic] = _hl_ic_path(avg, min(q_max, n_j - 1), c_vec[ic], p_j)
        end
    end
    q_full = q_sub[1, :]
    S = Vector{T}(undef, n_c)
    @inbounds for ic in 1:n_c
        S[ic] = T(std(Float64.(q_sub[:, ic])))
    end
    q_hat, interval, var_q = _hl_second_plateau(c_vec, q_full, S, q_max)
    HallinLiskaResult{T}(q_hat, q_full, c_vec, S, interval, var_q, q_sub, penalty)
end

@float_fallback hallin_liska X

"""
    bai_ng_q(X, r; p=1, δ=0.1, m=1, standardize=true) -> BaiNgQResult

Bai–Ng (2007) rank statistics on the residual covariance of a VAR(`p`) in `r`
static PCA factors. ``D_{1,k} = (\\lambda_{k+1}^2 / \\sum_j \\lambda_j^2)^{1/2}`` and
``D_{2,k} = (\\sum_{j>k} \\lambda_j^2 / \\sum_j \\lambda_j^2)^{1/2}``; ``\\hat q`` is the
smallest ``k`` with the statistic below ``m / \\min(N^{1/2-\\delta}, T^{1/2-\\delta})``.
"""
function bai_ng_q(X::AbstractMatrix{T}, r::Int;
    p::Int=1, δ::Real=0.1, m::Real=1, standardize::Bool=true,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (r < 1 || r > min(T_obs, N)) && throw(ArgumentError("r must be in [1, min(T,N)]"))
    p < 1 && throw(ArgumentError("p must be ≥ 1"))
    fm = estimate_factors(X, r; standardize=standardize)
    bai_ng_q(fm.factors, N, T_obs; p=p, δ=δ, m=m)
end

"""Bai–Ng (2007) from a `T × r` factor matrix and the original panel dimensions."""
function bai_ng_q(F::AbstractMatrix{T}, N::Int, T_obs::Int;
    p::Int=1, δ::Real=0.1, m::Real=1,
) where {T<:AbstractFloat}
    Tf, r = size(F)
    Tf < p + 2 && throw(ArgumentError("factor sample is too short for VAR($p)"))
    var_m = estimate_var(F, p)
    Σ = var_m.Sigma
    evals = reverse(real.(eigvals(Symmetric((Σ + Σ') / 2))))
    evals = max.(evals, zero(T))
    tot2 = sum(abs2, evals)
    tot2 = max(tot2, T(1e-15))
    D1 = Vector{T}(undef, r)
    D2 = Vector{T}(undef, r)
    @inbounds for k in 0:(r - 1)
        D1[k + 1] = sqrt(evals[k + 1]^2 / tot2)
        D2[k + 1] = sqrt(sum(abs2, @view evals[(k + 1):r]) / tot2)
    end
    # D_{1,k} / D_{2,k} use λ_{k+1} / remaining mass from k+1; q̂ = min {k : D_k < thresh}
    expo = T(0.5) - T(δ)
    thresh = T(m) / min(N^expo, T_obs^expo)
    q_D1 = clamp(something(findfirst(<(thresh), D1), r + 1) - 1, 0, r)
    q_D2 = clamp(something(findfirst(<(thresh), D2), r + 1) - 1, 0, r)
    BaiNgQResult{T}(q_D1, q_D2, D1, D2, thresh, r)
end

@float_fallback bai_ng_q X

"""
    amengual_watson_q(X, r, p=1; standardize=true) -> AmengualWatsonResult

Amengual–Watson (2007): project ``X_t`` on lags ``1,…,p`` of ``r`` static PCA
factors and apply Bai–Ng (2002) IC to the residuals. ``\\hat q`` is IC2.
"""
function amengual_watson_q(X::AbstractMatrix{T}, r::Int, p::Int=1;
    standardize::Bool=true,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (r < 1 || r > min(T_obs, N)) && throw(ArgumentError("r must be in [1, min(T,N)]"))
    p < 1 && throw(ArgumentError("p must be ≥ 1"))
    T_obs > p + r + 2 || throw(ArgumentError("sample too short for Amengual–Watson with r=$r, p=$p"))
    fm = estimate_factors(X, r; standardize=standardize)
    F = fm.factors
    X_use = standardize ? _standardize(X) : X
    Y = X_use[(p + 1):end, :]
    Zlag = hcat(ntuple(j -> F[(p - j + 1):(end - j), :], p)...)
    B = Zlag \ Y
    E = Y - Zlag * B
    r_max = min(r, size(E, 1) - 1, size(E, 2))
    r_max < 1 && return AmengualWatsonResult(1, 1, 1, 1, p)
    ic = ic_criteria(E, r_max; standardize=false)
    AmengualWatsonResult(ic.r_IC2, ic.r_IC1, ic.r_IC2, ic.r_IC3, p)
end

@float_fallback amengual_watson_q X

"""
    ic_criteria_gdfm(X, max_q; standardize=true, bandwidth=0, kernel=:bartlett)

Eigenvalue-ratio and 90%-variance heuristics for the number of dynamic factors.

These are **not** consistent estimators of ``q``. Prefer [`hallin_liska`](@ref),
[`bai_ng_q`](@ref), or [`amengual_watson_q`](@ref). When the 90% threshold is
never reached, `q_variance == max_q` and `boundary=true`; a warning is emitted.

# Returns
Named tuple with `eigenvalue_ratios`, `cumulative_variance`, `avg_eigenvalues`,
`q_ratio`, `q_variance`, and `boundary`.
"""
function ic_criteria_gdfm(X::AbstractMatrix{T}, max_q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett,
    spectral::Symbol=:lag_window,
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (max_q < 1 || max_q > N) && throw(ArgumentError("max_q must be in [1, $N]"))
    validate_option(spectral, "spectral", (:lag_window, :smoothed_periodogram))
    if bandwidth <= 0
        bandwidth = spectral === :lag_window ? _select_lag_window(T_obs) : _select_bandwidth(T_obs)
    end

    X_proc = standardize ? _standardize(X) : X
    _, spec = _estimate_spectrum(X_proc, bandwidth, kernel, spectral)
    eigenvalues, _ = _spectral_eigendecomposition(spec)
    avg_eig = vec(mean(eigenvalues, dims=2))

    # Eigenvalue ratio criterion
    ratios = [avg_eig[i] / avg_eig[i+1] for i in 1:min(max_q, N-1)]
    cum_var = cumsum(avg_eig[1:max_q]) / sum(avg_eig)
    q_ratio = argmax(ratios[1:min(max_q, length(ratios))])
    q_variance = something(findfirst(>=(T(0.9)), cum_var), max_q)
    boundary = q_variance == max_q && (max_q == 0 || cum_var[end] < T(0.9) - T(1e-14))
    if boundary
        @warn "ic_criteria_gdfm: 90% variance threshold was not reached; q_variance=$q_variance equals max_q. Raise max_q, or use hallin_liska / bai_ng_q / amengual_watson_q."
    end

    (eigenvalue_ratios=ratios, cumulative_variance=cum_var, avg_eigenvalues=avg_eig[1:max_q],
     q_ratio=q_ratio, q_variance=q_variance, boundary=boundary)
end

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::GeneralizedDynamicFactorModel, h; method=:ar, ci_method=:theoretical,
             conf_level=0.95, n_boot=1000, rng=Random.default_rng(), seed=nothing,
             n_lags_proj=nothing)

Forecast `h` steps ahead from an estimated [`GeneralizedDynamicFactorModel`](@ref).

Under `method=:one_sided` or `method=:spectral`, forecasts follow the FHLR (2005)
projection onto the stacked one-sided generalized static factors:

```math
\\hat{\\chi}_{T+1 \\mid T} = C_{\\chi, 1}^{(s)} Z \\left(Z^{\\top} \\hat{\\Gamma}_X^{(s)} Z\\right)^{-1} Z^{\\top} X_T^{(s)}, \\qquad
\\hat{\\chi}_{T+h \\mid T} = W_1 \\Phi_F^{h-1} F_T^{(s)}
```

where:
- ``\\hat{\\chi}_{T+h \\mid T}`` is the ``h``-step-ahead projected common component
- ``s`` is the stacked projection lag depth (`n_lags_proj`, defaulting to the lag depth in `model.Z`)
- ``X_T^{(s)} = [X_T^{\\top}, X_{T-1}^{\\top}, \\dots, X_{T-s}^{\\top}]^{\\top}`` is the stacked panel at the forecast origin
- ``Z`` is the ``N(s + 1) \\times r`` one-sided generalized eigenvector weight matrix
- ``F_T^{(s)} = Z^{\\top} X_T^{(s)}`` is the ``r \\times 1`` contemporaneous stacked static factor vector
- ``\\hat{\\Gamma}_X^{(s)}`` is the ``N(s + 1) \\times N(s + 1)`` block-Toeplitz panel autocovariance
- ``C_{\\chi, 1}^{(s)} = [\\hat{\\Gamma}_{\\chi}(1), \\dots, \\hat{\\Gamma}_{\\chi}(s + 1)]`` is the one-step common cross-covariance
- ``W_1 = C_{\\chi, 1}^{(s)} Z (Z^{\\top} \\hat{\\Gamma}_X^{(s)} Z)^{-1}`` is the one-step projection loading matrix
- ``\\Phi_F`` is the stationary VAR(1) transition matrix estimated on centered one-sided factors

# Arguments
- `model`: Estimated [`GeneralizedDynamicFactorModel`](@ref)
- `h`: Forecast horizon (`≥ 1`)

# Keyword Arguments
- `method::Symbol=:ar`: `:ar` fits an AR(1) on each **two-sided** factor; `:one_sided`
  and `:spectral` compute the FHLR (2005) projection (`:spectral` is **not** an alias for `:ar`)
- `ci_method::Symbol=:theoretical`: Confidence interval method — `:none`, `:theoretical`, or `:bootstrap`.
  Under `:one_sided`/`:spectral`, `:theoretical` combines common-component projection
  error ``\\hat{\\Gamma}_\\chi(0) - W_h G_z W_h^{\\top}`` with idiosyncratic variance (and
  ``h``-step factor projection error for `factors_se`), and `:bootstrap` simulates
  factor VAR(1) innovations alongside resampled idiosyncratic residuals
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Bootstrap replications (for `:bootstrap`)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for bootstrap draws
- `seed::Union{Integer,Nothing}=nothing`: Optional integer seed overriding `rng` via `Xoshiro(seed)`
- `n_lags_proj::Union{Nothing,Int}=nothing`: Lag depth `s` for stacked FHLR `:one_sided`/`:spectral`
  projection (`nothing` uses the lag depth stored in `model.Z`, which is `0` by default)

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).
"""
function forecast(model::GeneralizedDynamicFactorModel{T}, h::Int; method::Symbol=:ar,
    ci_method::Symbol=:theoretical, conf_level::Real=0.95, n_boot::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    n_lags_proj::Union{Nothing,Int}=nothing) where {T}

    h < 1 && throw(ArgumentError("h must be positive"))
    method ∈ (:ar, :one_sided, :spectral) || throw(ArgumentError(
        "method must be :ar, :one_sided, or :spectral, got :$method"))
    ci_method ∈ (:none, :theoretical, :bootstrap) || throw(ArgumentError("ci_method must be :none, :theoretical, or :bootstrap"))
    n_lags_proj === nothing || n_lags_proj >= 0 || throw(ArgumentError(
        "n_lags_proj must be >= 0, got $n_lags_proj"))
    rng = _resolve_repro_rng(rng, seed)
    if method === :one_sided || method === :spectral
        return _forecast_gdfm_fhlr(model, h; ci_method=ci_method, conf_level=conf_level,
            n_boot=n_boot, rng=rng, n_lags_proj=n_lags_proj)
    end

    q = model.q
    factors = model.factors
    L_avg = real.(model.loadings_spectral[:, :, 1])
    N = size(model.X, 2)
    T_obs = size(factors, 1)

    # Fit AR(1) per factor and compute forecasts
    phi = Vector{T}(undef, q)
    sigma2 = Vector{T}(undef, q)
    F_fc = Matrix{T}(undef, h, q)

    for i in 1:q
        F_i = factors[:, i]
        phi[i] = dot(F_i[1:end-1], F_i[2:end]) / dot(F_i[1:end-1], F_i[1:end-1])
        resid_i = F_i[2:end] .- phi[i] .* F_i[1:end-1]
        sigma2[i] = var(resid_i)
        f = F_i[end]
        for t in 1:h
            f = phi[i] * f
            F_fc[t, i] = f
        end
    end

    X_fc = F_fc * L_avg'
    conf_T = T(conf_level)

    # Idiosyncratic variance (diagonal)
    idio_var = vec(var(model.idiosyncratic, dims=1))

    if ci_method == :none
        model.standardized && _unstandardize_point!(X_fc, model.X)
        return _build_factor_forecast(F_fc, X_fc,
            zeros(T, h, q), zeros(T, h, q), zeros(T, h, N), zeros(T, h, N),
            zeros(T, h, q), zeros(T, h, N), h, conf_T, :none)
    end

    if ci_method == :theoretical
        z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))

        # Closed-form AR(1) forecast variance: σ² Σ_{j=0}^{h-1} φ^{2j}
        F_se = Matrix{T}(undef, h, q)
        for step in 1:h
            for i in 1:q
                fvar = sigma2[i] * sum(phi[i]^(2j) for j in 0:(step-1))
                F_se[step, i] = sqrt(max(fvar, zero(T)))
            end
        end
        F_lo = F_fc .- z_val .* F_se
        F_hi = F_fc .+ z_val .* F_se

        # Observable SE: L_avg * diag(factor_var) * L_avg' + diag(idio_var)
        X_se = Matrix{T}(undef, h, N)
        for step in 1:h
            fvar_diag = [sigma2[i] * sum(phi[i]^(2j) for j in 0:(step-1)) for i in 1:q]
            obs_var = L_avg * Diagonal(fvar_diag) * L_avg'
            X_se[step, :] = sqrt.(max.(diag(obs_var) .+ idio_var, zero(T)))
        end
        X_lo = X_fc .- z_val .* X_se
        X_hi = X_fc .+ z_val .* X_se

        if model.standardized
            _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, model.X)
        end
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    # Bootstrap: resample AR(1) residuals per factor
    F_boot = zeros(T, n_boot, h, q)
    X_boot = zeros(T, n_boot, h, N)

    # Pre-compute residuals per factor
    resids_per_factor = [factors[2:end, i] .- phi[i] .* factors[1:end-1, i] for i in 1:q]
    idio_std = sqrt.(max.(idio_var, zero(T)))

    for b in 1:n_boot
        for i in 1:q
            f = factors[end, i]
            for t in 1:h
                boot_idx = rand(rng, 1:(T_obs-1))
                f = phi[i] * f + resids_per_factor[i][boot_idx]
                F_boot[b, t, i] = f
            end
        end
        for t in 1:h
            X_boot[b, t, :] = L_avg * F_boot[b, t, :] .+ idio_std .* randn(rng, T, N)
        end
    end

    if model.standardized
        μ = vec(mean(model.X, dims=1))
        σ = max.(vec(std(model.X, dims=1)), T(1e-10))
        X_fc .= X_fc .* σ' .+ μ'
        for b in 1:n_boot
            X_boot[b, :, :] = X_boot[b, :, :] .* σ' .+ μ'
        end
    end

    α_lo = (1 - conf_level) / 2
    α_hi = 1 - α_lo
    f_lo = T[quantile(F_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:q]
    f_hi = T[quantile(F_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:q]
    o_lo = T[quantile(X_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:N]
    o_hi = T[quantile(X_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:N]
    f_se = T[std(F_boot[:, hh, j]) for hh in 1:h, j in 1:q]
    o_se = T[std(X_boot[:, hh, j]) for hh in 1:h, j in 1:N]

    _build_factor_forecast(F_fc, X_fc, f_lo, f_hi, o_lo, o_hi, f_se, o_se, h, conf_T, :bootstrap)
end

"""FHLR (2005) projection forecast from one-sided weights `Z` (supporting stacked `n_lags_proj`)."""
function _forecast_gdfm_fhlr(model::GeneralizedDynamicFactorModel{T}, h::Int;
    ci_method::Symbol, conf_level::Real, n_boot::Int,
    rng::AbstractRNG,
    n_lags_proj::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}

    X = model.X
    T_obs, N = size(X)
    r = size(model.Z, 2)
    r < 1 && throw(ArgumentError("one-sided weights Z are empty; re-estimate the GDFM"))
    s_model = div(size(model.Z, 1), N) - 1
    (s_model >= 0 && size(model.Z, 1) == N * (s_model + 1)) || throw(ArgumentError(
        "one-sided weights Z have incompatible row dimension $(size(model.Z, 1)) for N=$N; re-estimate the GDFM"))
    s_use = n_lags_proj === nothing ? s_model : Int(n_lags_proj)
    s_use >= 0 || throw(ArgumentError("n_lags_proj must be >= 0, got $s_use"))
    s_use < T_obs || throw(ArgumentError("n_lags_proj must be < T_obs ($T_obs), got $s_use"))

    X_proc = model.standardized ? _standardize(X) : X

    if s_use == 0 && s_model == 0
        Z = model.Z
        xT = vec(X_proc[end, :])
        F_now = Z' * xT
        F_fc = repeat(F_now', h, 1)

        # Put Γχ(0) and ΓX(0) on the sample-variance scale (smoothed periodogram omits 1/(2π))
        scale_spec = model.spectral === :smoothed_periodogram ? T(2π) : one(T)
        Γχ0 = _gamma_from_spectrum(model.spectral_density_chi, model.frequencies, 0) ./ scale_spec
        ΓX0 = _gamma_from_spectrum(model.spectral_density_X, model.frequencies, 0) ./ scale_spec
        Gz_raw = Z' * ΓX0 * Z
        Gz = Symmetric((Gz_raw + Gz_raw') / 2)

        # Estimate stationary VAR(1) transition Φ_F on centered one-sided factors F_os = X_proc * Z
        F_os = X_proc * Z
        Fc = F_os .- mean(F_os; dims=1)
        Φ_F = _fhlr_factor_var1(Fc, T_obs, r)

        W_list = Vector{Matrix{T}}(undef, h)
        W_list[1] = _fhlr_projection_weights(model, 1)
        @inbounds for hh in 2:h
            W_list[hh] = W_list[hh - 1] * Φ_F
        end

        W0 = (Γχ0 * Z) * robust_inv(Gz; silent=true)
        chi_os_proc = if model.standardized
            μ = mean(X; dims=1)
            σ = max.(std(X; dims=1), T(1e-10))
            (model.common_component_onesided .- μ) ./ σ
        else
            model.common_component_onesided
        end
    else
        Γ_X_pos, Γ_chi_pos = _gdfm_spectral_autocovariances(
            model.spectral_density_X, model.spectral_density_chi,
            model.frequencies, model.spectral, T_obs, s_use + 1)
        ΓX_s = _gdfm_block_toeplitz(Γ_X_pos[1:(s_use + 1)], s_use)
        if s_use == s_model
            Z = model.Z
        else
            Γχ_s = _gdfm_block_toeplitz(Γ_chi_pos[1:(s_use + 1)], s_use)
            Γξ_s = Matrix{T}((ΓX_s - Γχ_s + (ΓX_s - Γχ_s)') / T(2))
            Z = _onesided_Z(Γχ_s, Γξ_s, r)
        end
        X_s = _gdfm_stacked_matrix(X_proc, s_use)
        xT_s = vec(X_s[end, :])
        F_now = Z' * xT_s
        F_fc = repeat(F_now', h, 1)

        Gz_raw = Z' * ΓX_s * Z
        Gz = Symmetric((Gz_raw + Gz_raw') / T(2))
        Gz_inv = robust_inv(Gz; silent=true)

        F_os = X_s * Z
        Fc = F_os .- mean(F_os; dims=1)
        Φ_F = _fhlr_factor_var1(Fc, T_obs, r)

        C_chi0 = hcat([Γ_chi_pos[lag + 1] for lag in 0:s_use]...)
        C_chi1 = hcat([Γ_chi_pos[lag + 2] for lag in 0:s_use]...)
        W0 = C_chi0 * Z * Gz_inv
        W_list = Vector{Matrix{T}}(undef, h)
        W_list[1] = C_chi1 * Z * Gz_inv
        @inbounds for hh in 2:h
            W_list[hh] = W_list[hh - 1] * Φ_F
        end
        Γχ0 = Γ_chi_pos[1]
        chi_os_proc = if s_use == s_model
            if model.standardized
                μ = mean(X; dims=1)
                σ = max.(std(X; dims=1), T(1e-10))
                (model.common_component_onesided .- μ) ./ σ
            else
                model.common_component_onesided
            end
        else
            F_os * transpose(W0)
        end
    end

    X_fc = Matrix{T}(undef, h, N)
    @inbounds for hh in 1:h
        X_fc[hh, :] = W_list[hh] * F_now
    end
    conf_T = T(conf_level)
    idio_proc = X_proc - chi_os_proc
    idio_var = vec(var(idio_proc; dims=1))

    if ci_method === :none
        model.standardized && _unstandardize_point!(X_fc, X)
        return _build_factor_forecast(F_fc, X_fc,
            zeros(T, h, r), zeros(T, h, r), zeros(T, h, N), zeros(T, h, N),
            zeros(T, h, r), zeros(T, h, N), h, conf_T, :none)
    end

    if ci_method === :theoretical
        z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
        X_se = Matrix{T}(undef, h, N)
        F_se = Matrix{T}(undef, h, r)
        C_F = Matrix{T}(I, r, r)
        idio_var_pos = max.(idio_var, zero(T))
        @inbounds for hh in 1:h
            C_F = C_F * Φ_F
            Wh = W_list[hh]
            common_err_var = max.(diag(Γχ0 - Wh * Gz * Wh'), zero(T))
            X_se[hh, :] = sqrt.(common_err_var .+ idio_var_pos)
            F_se[hh, :] = sqrt.(max.(diag(Gz - C_F * Gz * C_F'), zero(T)))
        end
        F_lo, F_hi = F_fc .- z_val .* F_se, F_fc .+ z_val .* F_se
        X_lo, X_hi = X_fc .- z_val .* X_se, X_fc .+ z_val .* X_se
        if model.standardized
            _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, X)
        end
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    # Bootstrap: propagate VAR(1) factor innovations alongside resampled idiosyncratic rows
    n_idio = size(idio_proc, 1)
    idio_c = idio_proc .- mean(idio_proc; dims=1)
    U_F = T_obs >= 3 ? (Fc[2:end, :] - Fc[1:end-1, :] * Φ_F') : zeros(T, 1, r)
    U_F .-= mean(U_F; dims=1)
    n_u = size(U_F, 1)
    X_boot = zeros(T, n_boot, h, N)
    F_boot = zeros(T, n_boot, h, r)
    @inbounds for b in 1:n_boot
        δF = zeros(T, r)
        for hh in 1:h
            δF = Φ_F * δF .+ U_F[rand(rng, 1:n_u), :]
            F_boot[b, hh, :] = F_fc[hh, :] .+ δF
            row = idio_c[rand(rng, 1:n_idio), :]
            X_boot[b, hh, :] = X_fc[hh, :] .+ W0 * δF .+ row
        end
    end
    if model.standardized
        μ = vec(mean(X; dims=1))
        σ = max.(vec(std(X; dims=1)), T(1e-10))
        X_fc .= X_fc .* σ' .+ μ'
        for b in 1:n_boot
            X_boot[b, :, :] = X_boot[b, :, :] .* σ' .+ μ'
        end
    end
    α_lo = (1 - conf_level) / 2
    α_hi = 1 - α_lo
    f_lo = T[quantile(F_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:r]
    f_hi = T[quantile(F_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:r]
    o_lo = T[quantile(X_boot[:, hh, j], α_lo) for hh in 1:h, j in 1:N]
    o_hi = T[quantile(X_boot[:, hh, j], α_hi) for hh in 1:h, j in 1:N]
    f_se = T[std(F_boot[:, hh, j]) for hh in 1:h, j in 1:r]
    o_se = T[std(X_boot[:, hh, j]) for hh in 1:h, j in 1:N]
    _build_factor_forecast(F_fc, X_fc, f_lo, f_hi, o_lo, o_hi, f_se, o_se, h, conf_T, :bootstrap)
end

"""Estimate stationary VAR(1) transition matrix on centered one-sided factors."""
function _fhlr_factor_var1(Fc::AbstractMatrix{T}, T_obs::Int, r::Int) where {T<:AbstractFloat}
    Φ_F = zeros(T, r, r)
    if T_obs >= 3
        Fc_lag = @view Fc[1:end-1, :]
        Fc_lead = @view Fc[2:end, :]
        S_lag = Fc_lag' * Fc_lag
        Φ_F = (Fc_lead' * Fc_lag) * robust_inv(Symmetric((S_lag + S_lag') / 2); silent=true)
        if any(!isfinite, Φ_F)
            Φ_F .= zero(T)
        else
            ρ = maximum(abs, eigvals(Φ_F))
            if !isfinite(ρ)
                Φ_F .= zero(T)
            elseif ρ >= T(0.995)
                Φ_F .*= T(0.995) / ρ
            end
        end
    end
    Φ_F
end

"""AR(1) forecast for each factor series."""
function _forecast_factors_ar(factors::Matrix{T}, h::Int) where {T<:AbstractFloat}
    T_obs, q = size(factors)
    fc = Matrix{T}(undef, h, q)

    for i in 1:q
        F = factors[:, i]
        # Estimate AR(1) coefficient
        phi = dot(F[1:end-1], F[2:end]) / dot(F[1:end-1], F[1:end-1])
        f = F[end]
        for t in 1:h
            f = phi * f
            fc[t, i] = f
        end
    end
    fc
end

# =============================================================================
# GDFM Utilities
# =============================================================================

"""
    common_variance_share(model::GeneralizedDynamicFactorModel) -> Vector

Fraction of each variable's variance explained by the common component.
"""
function common_variance_share(m::GeneralizedDynamicFactorModel{T}) where {T}
    N = size(m.X, 2)
    [var(m.common_component[:, i]) / max(var(m.X[:, i]), T(1e-10)) for i in 1:N]
end

"""
    spectral_eigenvalue_plot_data(model::GeneralizedDynamicFactorModel)

Return data for plotting eigenvalues across frequencies.
"""
spectral_eigenvalue_plot_data(m::GeneralizedDynamicFactorModel) =
    (frequencies=m.frequencies, eigenvalues=m.eigenvalues_spectral)
