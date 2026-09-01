# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0, spectral=:lag_window) -> GeneralizedDynamicFactorModel

Estimate Generalized Dynamic Factor Model using spectral methods.

# Arguments
- `X`: Data matrix (T × N)
- `q`: Number of dynamic factors

# Keyword Arguments
- `standardize::Bool=true`: Standardize data
- `bandwidth::Int=0`: Lag truncation `M` under `:lag_window` (default `max(3, round(0.5√T))`),
  or frequency-ordinate bandwidth under `:smoothed_periodogram` (default `max(3, round(T^{1/3}))`).
- `kernel::Symbol=:bartlett`: Kernel (`:bartlett`, `:parzen`, `:tukey`)
- `r::Int=0`: Number of static factors (0 = same as q)
- `spectral::Symbol=:lag_window`: `:lag_window` (FHLR 2000/2005) or `:smoothed_periodogram` (legacy)
- `varnames::Union{Nothing,Vector{String}}=nothing`: Names for the N panel variables
  (defaults to `"Var 1"`, …). Forwarded by the `TimeSeriesData` dispatch.

# Returns
`GeneralizedDynamicFactorModel` with two-sided common/idiosyncratic components
(in-sample FHLR 2000) and one-sided weights `Z`, `factors_onesided`,
`common_component_onesided` (FHLR 2005). Use `forecast(gdfm, h; method=:one_sided)`
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
) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))
    validate_option(spectral, "spectral", (:lag_window, :smoothed_periodogram))
    vn = something(varnames, ["Var $i" for i in 1:N])
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $N columns"))

    r_static = r == 0 ? q : r
    r_static < q && throw(ArgumentError("r must be >= q"))
    if bandwidth <= 0
        bandwidth = spectral === :lag_window ? _select_lag_window(T_obs) : _select_bandwidth(T_obs)
    end

    X_original = copy(X)
    X_proc = standardize ? _standardize(X) : X

    # Spectral analysis
    frequencies, spectral_X = _estimate_spectrum(X_proc, bandwidth, kernel, spectral)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    spectral_chi = _compute_common_spectral_density(loadings, eigenvalues[1:q, :])
    loadings_td = spectral === :lag_window ?
        _interp_loadings_to_fft_grid(loadings, frequencies, T_obs) : loadings
    common = _reconstruct_time_domain(loadings_td, X_proc)
    factors = _extract_time_domain_factors(X_proc, loadings_td, frequencies)
    var_explained = _compute_variance_explained(eigenvalues, q)

    Z, F_os, chi_os = _gdfm_onesided(X_proc, spectral_X, spectral_chi, frequencies, r_static)

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

"""Linear interpolation of spectral loadings onto the T/2+1 FFT ordinates."""
function _interp_loadings_to_fft_grid(loadings::Array{Complex{T},3}, src_freq::Vector{T}, T_obs::Int) where {T<:AbstractFloat}
    N, q, n_src = size(loadings)
    n_fft = div(T_obs, 2) + 1
    dest_freq = [T(2π * (j - 1) / T_obs) for j in 1:n_fft]
    L = _align_eigenvector_phases(loadings)
    L_out = Array{Complex{T},3}(undef, N, q, n_fft)
    src = copy(src_freq)
    # FFT grid lives in [0, π]; lag-window grid is already [0, π]
    @inbounds for j in 1:n_fft
        ω = dest_freq[j]
        if ω <= src[1]
            L_out[:, :, j] = L[:, :, 1]
        elseif ω >= src[end]
            L_out[:, :, j] = L[:, :, n_src]
        else
            i2 = searchsortedfirst(src, ω)
            i1 = max(i2 - 1, 1)
            i2 = min(i2, n_src)
            Δ = src[i2] - src[i1]
            t = Δ > 0 ? (ω - src[i1]) / Δ : zero(T)
            L_out[:, :, j] = (1 - t) .* L[:, :, i1] .+ t .* L[:, :, i2]
        end
    end
    L_out
end

"""Align eigenvector signs/phases across adjacent frequencies before interpolation."""
function _align_eigenvector_phases(loadings::Array{Complex{T},3}) where {T<:AbstractFloat}
    L = copy(loadings)
    _, q, n_freq = size(L)
    @inbounds for j in 2:n_freq
        for k in 1:q
            overlap = dot(L[:, k, j - 1], L[:, k, j])
            if overlap != 0
                L[:, k, j] .*= conj(overlap) / abs(overlap)
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

"""Reconstruct common component in time domain via inverse FFT."""
function _reconstruct_time_domain(loadings::Array{Complex{T},3}, X::AbstractMatrix{T}) where {T}
    T_obs, N = size(X)
    n_freq = size(loadings, 3)
    X_fft = FFTW.fft(X, 1)
    chi_fft = zeros(Complex{T}, T_obs, N)

    @inbounds for j in 1:n_freq
        # Forni idempotent dynamic-principal-component projector on the leading-q eigenvectors of
        # the SMOOTHED spectrum: χ(ω_j) = (Σ_k v_k v_k^H) X_fft(ω_j) = L Lᴴ X_fft(ω_j). The old
        # code inverted the raw rank-1 periodogram X_fft·X_fftᴴ/T — a degenerate/noisy filter.
        L = @view loadings[:, :, j]        # N × q, columns orthonormal (LᴴL ≈ I_q)
        chi_fft[j, :] = L * (L' * @view(X_fft[j, :]))
        j > 1 && j < n_freq && (chi_fft[T_obs - j + 2, :] = conj(chi_fft[j, :]))
    end
    real(FFTW.ifft(chi_fft, 1))
end

"""Extract time-domain factors via frequency-domain projection."""
function _extract_time_domain_factors(X::AbstractMatrix{T}, loadings::Array{Complex{T},3}, frequencies::Vector{T}) where {T}
    T_obs, N = size(X)
    _, q, n_freq = size(loadings)
    X_fft, F_fft = FFTW.fft(X, 1), zeros(Complex{T}, T_obs, q)

    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        F_fft[j, :] = (L' * L + T(1e-10) * I) \ (L' * X_fft[j, :])
        j > 1 && j < n_freq && (F_fft[T_obs - j + 2, :] = conj(F_fft[j, :]))
    end

    factors = real(FFTW.ifft(F_fft, 1))
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
idiosyncratic remainder. `χ^{(j)}(ω) = L_{:j}(ω) L_{:j}(ω)ᴴ X(ω)`, inverted
by IFFT. The `q` paths reconstruct `gdfm.common_component` (including the
series mean under `standardize=true`, matching `estimate_gdfm`).
"""
function historical_decomposition(gdfm::GeneralizedDynamicFactorModel{T};
    by::Symbol=:dynamic_pc) where {T<:AbstractFloat}
    by === :dynamic_pc || throw(ArgumentError("by must be :dynamic_pc, got :$by"))
    X = gdfm.X
    T_obs, N = size(X)
    q = gdfm.q
    X_proc = gdfm.standardized ? _standardize(X) : X
    L = gdfm.loadings_spectral
    n_fft = div(T_obs, 2) + 1
    L_td = size(L, 3) == n_fft ? L : _interp_loadings_to_fft_grid(L, gdfm.frequencies, T_obs)
    X_fft = FFTW.fft(X_proc, 1)
    contrib = zeros(T, T_obs, N, q + 1)
    @inbounds for k in 1:q
        chi_fft = zeros(Complex{T}, T_obs, N)
        for j in 1:n_fft
            lk = @view L_td[:, k, j]
            chi_fft[j, :] = lk .* (lk' * @view(X_fft[j, :]))
            j > 1 && j < n_fft && (chi_fft[T_obs - j + 2, :] = conj(chi_fft[j, :]))
        end
        chi = real(FFTW.ifft(chi_fft, 1))
        if gdfm.standardized
            σ = max.(vec(std(X; dims=1)), T(1e-10))
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
    forecast(model::GeneralizedDynamicFactorModel, h; method=:ar, ci_method=:theoretical, conf_level=0.95, n_boot=1000)

Forecast `h` steps ahead from a GDFM.

# Arguments
- `model`: Estimated GDFM
- `h`: Forecast horizon

# Keyword Arguments
- `method::Symbol=:ar`: `:ar` fits an AR(1) on each **two-sided** factor; `:one_sided`
  and `:spectral` are the FHLR (2005) projection
  ``\\hat\\chi_{T+h|T} = \\hat\\Gamma_\\chi(h) Z (Z'\\hat\\Gamma_X(0) Z)^{-1} Z' X_T``.
  `:spectral` is **not** an alias for `:ar`.
- `ci_method::Symbol=:theoretical`: CI method — `:none`, `:theoretical`, or `:bootstrap`
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Bootstrap replications (for `:bootstrap`)

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).
"""
function forecast(model::GeneralizedDynamicFactorModel{T}, h::Int; method::Symbol=:ar,
    ci_method::Symbol=:theoretical, conf_level::Real=0.95, n_boot::Int=1000,
    rng::AbstractRNG=Random.default_rng()) where {T}

    h < 1 && throw(ArgumentError("h must be positive"))
    method ∈ (:ar, :one_sided, :spectral) || throw(ArgumentError(
        "method must be :ar, :one_sided, or :spectral, got :$method"))
    ci_method ∈ (:none, :theoretical, :bootstrap) || throw(ArgumentError("ci_method must be :none, :theoretical, or :bootstrap"))
    if method === :one_sided || method === :spectral
        return _forecast_gdfm_fhlr(model, h; ci_method=ci_method, conf_level=conf_level,
            n_boot=n_boot, rng=rng)
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

"""FHLR (2005) projection forecast from one-sided weights `Z`."""
function _forecast_gdfm_fhlr(model::GeneralizedDynamicFactorModel{T}, h::Int;
    ci_method::Symbol, conf_level::Real, n_boot::Int,
    rng::AbstractRNG) where {T<:AbstractFloat}

    X = model.X
    T_obs, N = size(X)
    r = size(model.Z, 2)
    r < 1 && throw(ArgumentError("one-sided weights Z are empty; re-estimate the GDFM"))
    X_proc = model.standardized ? _standardize(X) : X
    xT = vec(X_proc[end, :])
    F_now = model.Z' * xT
    F_fc = repeat(F_now', h, 1)
    X_fc = Matrix{T}(undef, h, N)
    @inbounds for hh in 1:h
        W = _fhlr_projection_weights(model, hh)
        X_fc[hh, :] = W * F_now
    end
    conf_T = T(conf_level)
    chi_os_proc = if model.standardized
        μ = mean(X; dims=1)
        σ = max.(std(X; dims=1), T(1e-10))
        (model.common_component_onesided .- μ) ./ σ
    else
        model.common_component_onesided
    end
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
        X_se = repeat(sqrt.(max.(idio_var, zero(T)))', h, 1)
        F_se = zeros(T, h, r)
        F_lo, F_hi = F_fc .- z_val .* F_se, F_fc .+ z_val .* F_se
        X_lo, X_hi = X_fc .- z_val .* X_se, X_fc .+ z_val .* X_se
        if model.standardized
            _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, X)
        end
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    # Bootstrap: resample idiosyncratic rows and add to the projection
    n_idio = size(idio_proc, 1)
    X_boot = zeros(T, n_boot, h, N)
    F_boot = zeros(T, n_boot, h, r)
    @inbounds for b in 1:n_boot
        for hh in 1:h
            row = idio_proc[rand(rng, 1:n_idio), :]
            X_boot[b, hh, :] = X_fc[hh, :] .+ row
            F_boot[b, hh, :] = F_now
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
