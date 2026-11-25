# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Frequency-domain extensions for the Generalized Dynamic Factor Model (GDFM):
band-pass common component extraction, real-time Eurocoin growth indicator
(Altissimo et al., 2010), and spectral forecast error variance decomposition.

References:
- Altissimo, F., Cristadoro, R., Forni, M., Lippi, M., & Veronese, G. (2010).
  New Eurocoin: Tracking Economic Growth in Real Time. *Review of Economics and
  Statistics*, 92(4), 1024–1034.
- Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000, 2005). The Generalized
  Dynamic-Factor Model: Identification, Estimation, and One-Sided Prediction.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Result Types
# =============================================================================

"""
    BandPassGDFMResult{T<:AbstractFloat}

Frequency-band decomposition of a [`GeneralizedDynamicFactorModel`](@ref) common
component over angular frequencies ``|\\theta| \\in [\\omega_{\\text{low}}, \\omega_{\\text{high}}]``.

```math
\\chi_t^{[\\omega_1, \\omega_2]} = \\frac{1}{\\pi} \\sum_{k=-M}^{M} K_k^{[\\omega_1, \\omega_2]} X_{t-k}
```

where:
- ``\\chi_t^{[\\omega_1, \\omega_2]}`` is the two-sided band-limited common component at time ``t``
- ``[\\omega_1, \\omega_2] \\subseteq [0, \\pi]`` is the angular frequency band
- ``K_k^{[\\omega_1, \\omega_2]}`` is the band-restricted dynamic PCA projector kernel at lag ``k``
- ``M`` is the finite symmetric filter half-width
- ``X_{t-k}`` is the panel observation vector at time ``t - k``

# Fields
- `common_twosided::Matrix{T}`: Two-sided band-limited common component (`T_obs × N`)
- `common_onesided::Matrix{T}`: Real-time one-sided generalized-PC band projection (`T_obs × N`)
- `variance_share::Vector{T}`: Share of each variable's common spectral variance in `band` (length `N`)
- `band::Tuple{T,T}`: Angular frequency band `(omega_low, omega_high) ⊆ [0, π]`
- `r::Int`: Static factor rank used for the one-sided projection
- `varnames::Vector{String}`: Panel variable names (length `N`)
"""
struct BandPassGDFMResult{T<:AbstractFloat}
    common_twosided::Matrix{T}
    common_onesided::Matrix{T}
    variance_share::Vector{T}
    band::Tuple{T,T}
    r::Int
    varnames::Vector{String}
end

"""
    EurocoinResult{T<:AbstractFloat}

Real-time low-frequency growth indicator constructed by projecting the medium-to-long-run
component of `target` (``|\\theta| \\le \\text{cutoff}``) onto the one-sided FHLR
generalized static factor space ``Z' X_t`` (Altissimo et al., 2010):

```math
c_t = \\bar{y} + w^{\\top} (X_t - \\bar{X}), \\qquad w_{\\text{fac}} = (Z^{\\top} \\hat{\\Gamma}_X(0) Z)^{-1} \\int_{-\\omega_c}^{\\omega_c} \\hat{\\Sigma}_{F,y}(\\theta)\\, d\\theta
```

where:
- ``c_t`` is the real-time Eurocoin indicator at time ``t``
- ``\\bar{y}`` and ``\\bar{X}`` are sample means of `target` and `X`
- ``w`` and ``w_{\\text{fac}}`` are the observable and factor-space projection weight vectors
- ``X_t`` is the observed ``N \\times 1`` panel vector at time ``t``
- ``Z`` is the ``N \\times r`` FHLR generalized principal component weight matrix
- ``F_t = Z^{\\top} X_t`` are the contemporaneous static factors
- ``\\hat{\\Gamma}_X(0)`` is the lag-0 panel covariance reconstructed from the spectral density
- ``\\omega_c`` is the low-pass angular frequency cutoff (`cutoff`)
- ``\\hat{\\Sigma}_{F,y}(\\theta)`` is the estimated cross-spectral density between ``F_t`` and centered `target`

# Fields
- `indicator::Vector{T}`: Strictly one-sided real-time growth indicator (`T_obs`)
- `target_lowpass::Vector{T}`: Two-sided ideal low-pass filtered target benchmark (`T_obs`)
- `weights::Vector{T}`: Linear projection weights mapping demeaned raw `X` to `indicator` (length `N`)
- `r2::T`: In-sample ``R^2`` of `indicator` against `target_lowpass`
- `cutoff::T`: Angular frequency cutoff ``\\omega_c \\in (0, \\pi)``
- `q::Int`: Number of dynamic common shocks
- `r::Int`: Static factor dimension `size(Z, 2)`
"""
struct EurocoinResult{T<:AbstractFloat}
    indicator::Vector{T}
    target_lowpass::Vector{T}
    weights::Vector{T}
    r2::T
    cutoff::T
    q::Int
    r::Int
end

Base.length(r::EurocoinResult) = length(r.indicator)
Base.firstindex(::EurocoinResult) = 1
Base.lastindex(r::EurocoinResult) = length(r.indicator)
Base.getindex(r::EurocoinResult, i) = r.indicator[i]

"""
    SpectralFEVDResult{T<:AbstractFloat} <: AbstractFEVD

Frequency-band forecast error variance decomposition for a [`StructuralDFM`](@ref) over
``|\\omega| \\in [\\omega_{\\text{low}}, \\omega_{\\text{high}}] \\subseteq [0, \\pi]``:

```math
\\text{FEVD}_{i,j}([\\omega_1, \\omega_2]) = \\frac{\\int_{\\omega_1}^{\\omega_2} |\\Theta_{i,j}(\\omega)|^2\\, d\\omega}{\\sum_{s} \\int_{\\omega_1}^{\\omega_2} S_{i,s}(\\omega)\\, d\\omega}
```

where:
- ``\\Theta_{i,j}(\\omega)`` is the structural panel transfer function from shock ``j`` to variable ``i``
- ``S_{i,s}(\\omega)`` is the band spectral power contributed by component ``s`` (shocks and optionally idiosyncratic)
- ``\\omega_1, \\omega_2`` are the lower and upper angular frequency bounds

# Fields
- `shares::Matrix{T}`: Variance shares (`N × q` or `N × (q + 1)`), rows sum to `1`
- `power_band::Matrix{T}`: Band-integrated spectral variance contributions (`N × n_shocks`)
- `total_power_band::Vector{T}`: Total band spectral power per panel variable (`N`)
- `band::Tuple{T,T}`: Angular frequency band `(omega_low, omega_high)`
- `include_idiosyncratic::Bool`: Whether an `"Idiosyncratic"` column is included
- `varnames::Vector{String}`: Panel variable names (length `N`)
- `shock_names::Vector{String}`: Shock (and optional idiosyncratic) labels
"""
struct SpectralFEVDResult{T<:AbstractFloat} <: AbstractFEVD
    shares::Matrix{T}
    power_band::Matrix{T}
    total_power_band::Vector{T}
    band::Tuple{T,T}
    include_idiosyncratic::Bool
    varnames::Vector{String}
    shock_names::Vector{String}
end

point_estimate(r::SpectralFEVDResult) = r.shares
has_uncertainty(::SpectralFEVDResult) = false
uncertainty_bounds(::SpectralFEVDResult) = nothing

# =============================================================================
# Internal Quadrature & Frequency-Band Helpers (_gdfmbp_*)
# =============================================================================

"""Validate frequency band `0 <= omega_low < omega_high <= π` and return `(w_lo::T, w_hi::T)`."""
function _gdfmbp_validate_band(band::Tuple{Real,Real}, ::Type{T}) where {T<:AbstractFloat}
    length(band) == 2 || throw(ArgumentError("band must be a 2-tuple (omega_low, omega_high)"))
    w_lo = T(band[1])
    w_hi = T(band[2])
    isfinite(w_lo) && isfinite(w_hi) || throw(ArgumentError("band endpoints must be finite"))
    w_lo >= zero(T) || throw(ArgumentError("band lower bound must be >= 0, got $w_lo"))
    w_hi <= T(π) + T(1e-12) || throw(ArgumentError("band upper bound must be <= π, got $w_hi"))
    w_hi = min(w_hi, T(π))
    w_lo < w_hi || throw(ArgumentError("band must satisfy omega_low < omega_high, got ($w_lo, $w_hi)"))
    w_lo, w_hi
end

"""
Exact additive cell-overlap integration weights over `[w_lo, w_hi] ⊆ [0, π]` for a
sorted grid `frequencies ⊆ [0, π]`. Summing across any partition `(0, ω₀)` and `(ω₀, π)`
recovers the composite trapezoidal weights on `[0, π]` identically.
"""
function _gdfmbp_band_weights(
    frequencies::AbstractVector{T},
    w_lo::T,
    w_hi::T,
) where {T<:AbstractFloat}
    n_freq = length(frequencies)
    n_freq >= 1 || throw(ArgumentError("frequencies must be non-empty"))
    if n_freq == 1
        return T[w_hi - w_lo]
    end
    wt = zeros(T, n_freq)
    @inbounds for m in 1:(n_freq - 1)
        θ1 = frequencies[m]
        θ2 = frequencies[m + 1]
        Δ = θ2 - θ1
        Δ > zero(T) || continue
        a = max(θ1, w_lo)
        b = min(θ2, w_hi)
        if b > a
            u_a = (a - θ1) / Δ
            u_b = (b - θ1) / Δ
            du = u_b - u_a
            u_mid = T(0.5) * (u_a + u_b)
            wt[m] += Δ * du * (one(T) - u_mid)
            wt[m + 1] += Δ * du * u_mid
        end
    end
    # Trailing half-cell [frequencies[end], π] for odd-T smoothed periodogram grids
    θ_last = frequencies[n_freq]
    if θ_last < T(π) - T(1e-9)
        a_tail = max(θ_last, w_lo)
        b_tail = min(T(π), w_hi)
        b_tail > a_tail && (wt[n_freq] += b_tail - a_tail)
    end
    wt
end

"""
    _gdfmbp_gamma_band(S, frequencies, k, omega_low, omega_high) -> Matrix{T}

Integrate a one-sided Hermitian spectral density ``\\hat{\\Sigma}(\\theta)`` over
``|\\theta| \\in [\\omega_{\\text{low}}, \\omega_{\\text{high}}] \\subseteq [0, \\pi]``:

```math
\\Gamma_k^{[\\omega_1, \\omega_2]} = 2\\,\\operatorname{Re}\\!\\int_{\\omega_1}^{\\omega_2} \\hat{\\Sigma}(\\theta)\\,e^{-ik\\theta}\\, d\\theta
```

where:
- ``\\hat{\\Sigma}(\\theta)`` is the Hermitian spectral matrix at positive frequencies ``\\theta \\in [0, \\pi]``
- ``\\omega_1, \\omega_2`` are the band endpoints `omega_low` and `omega_high`
- ``k`` is the autocovariance lag order

Partitioning ``[0, \\pi]`` into ``[0, \\omega_0]`` and ``[\\omega_0, \\pi]`` yields exact
additivity matching `_gamma_from_spectrum(S, frequencies, k)`.
"""
function _gdfmbp_gamma_band(
    S::Array{Complex{T},3},
    frequencies::AbstractVector{<:Real},
    k::Int,
    omega_low::Real,
    omega_high::Real,
) where {T<:AbstractFloat}
    N, N2, n_freq = size(S)
    N == N2 || throw(ArgumentError("spectral slices must be square, got $(N)×$(N2)"))
    n_freq >= 1 || throw(ArgumentError("spectral density is empty"))
    length(frequencies) == n_freq || throw(ArgumentError(
        "frequencies length ($(length(frequencies))) must match spectral slices ($n_freq)"))

    w_lo, w_hi = _gdfmbp_validate_band((omega_low, omega_high), T)
    freq_T = Vector{T}(frequencies)
    wt = _gdfmbp_band_weights(freq_T, w_lo, w_hi)

    G = zeros(T, N, N)
    @inbounds for j in 1:n_freq
        wj = wt[j]
        wj > zero(T) || continue
        phase = cis(-k * freq_T[j])
        scale = T(2) * wj
        Sj = @view S[:, :, j]
        for col in 1:N, row in 1:N
            G[row, col] += scale * real(Sj[row, col] * phase)
        end
    end
    G
end

"""Band-limited symmetric time-domain projector kernels matching `_projector_filter_kernels`."""
function _gdfmbp_band_kernels(
    loadings::AbstractArray{Complex{T},3},
    frequencies::Vector{T},
    T_obs::Int,
    spectral::Symbol,
    M::Int,
    w_lo::T,
    w_hi::T,
) where {T<:AbstractFloat}
    N, _, n_freq = size(loadings)
    wt = _gdfmbp_band_weights(frequencies, w_lo, w_hi)

    R_list = Vector{Matrix{T}}(undef, n_freq)
    I_list = Vector{Matrix{T}}(undef, n_freq)
    @inbounds for j in 1:n_freq
        Lj = @view loadings[:, :, j]
        Pj = Lj * Lj'
        R_list[j] = real.(Pj)
        I_list[j] = imag.(Pj)
    end

    K_list = Vector{Matrix{T}}(undef, M + 1)
    if n_freq <= 1
        K_list[1] = R_list[1] .* ((w_hi - w_lo) / T(π))
        @inbounds for k in 1:M
            K_list[k + 1] = zeros(T, N, N)
        end
        return K_list
    end

    if spectral === :lag_window
        M_grid = n_freq - 1
        inv_n = one(T) / T(2 * M_grid)
        slice_wt = (T(2 * M_grid) / T(π)) .* wt
        @inbounds for k in 0:M
            K = slice_wt[1] .* R_list[1]
            K .+= (isodd(k) ? -slice_wt[n_freq] : slice_wt[n_freq]) .* R_list[n_freq]
            for j in 2:(n_freq - 1)
                swj = slice_wt[j]
                swj > zero(T) || continue
                θ = frequencies[j]
                c, s = cos(k * θ), sin(k * θ)
                K .+= (swj * c) .* R_list[j] .+ (swj * s) .* I_list[j]
            end
            w_k = k == M_grid ? T(0.5) : one(T)
            K .*= w_k * inv_n
            K_list[k + 1] = K
        end
    else
        inv_T = one(T) / T(T_obs)
        slice_wt = (T(T_obs) / T(π)) .* wt
        has_nyquist = iseven(T_obs)
        j_hi = has_nyquist ? n_freq - 1 : n_freq
        @inbounds for k in 0:M
            K = slice_wt[1] .* R_list[1]
            if has_nyquist
                K .+= (isodd(k) ? -slice_wt[n_freq] : slice_wt[n_freq]) .* R_list[n_freq]
            end
            for j in 2:j_hi
                swj = slice_wt[j]
                swj > zero(T) || continue
                ω = frequencies[j]
                c, s = cos(k * ω), sin(k * ω)
                K .+= (swj * c) .* R_list[j] .- (swj * s) .* I_list[j]
            end
            K .*= inv_T
            K_list[k + 1] = K
        end
    end
    K_list
end

"""Two-sided reflection-padded ideal low-pass filter on a 1D centered vector for `|θ| <= cutoff`."""
function _gdfmbp_lowpass_1d(y_c::Vector{T}, cutoff::T) where {T<:AbstractFloat}
    T_obs = length(y_c)
    K_lp = min(T_obs - 1, max(32, div(T_obs, 3)))
    K_lp <= 0 && return copy(y_c)

    h_pos = Vector{T}(undef, K_lp)
    @inbounds for k in 1:K_lp
        ideal = sin(T(k) * cutoff) / (T(k) * T(π))
        taper = Base.sinc(T(k) / T(K_lp + 1))
        h_pos[k] = ideal * taper
    end
    h0 = cutoff / T(π)
    denom = h0 + T(2) * sum(h_pos)
    scale = abs(denom) > T(1e-12) ? (one(T) / denom) : one(T)
    h0 *= scale
    h_pos .*= scale

    y_pad = vec(_reflect_pad(reshape(y_c, T_obs, 1), K_lp))
    out = h0 .* y_c
    @inbounds for t in 1:T_obs
        acc = zero(T)
        c_idx = K_lp + t
        for k in 1:K_lp
            acc += h_pos[k] * (y_pad[c_idx - k] + y_pad[c_idx + k])
        end
        out[t] += acc
    end
    out
end

"""Unit-matched panel residual matrix (`T_obs × N`) for `spectral_fevd(..., include_idiosyncratic=true)`."""
function _gdfmbp_idio_residuals(sdfm::StructuralDFM{T}) where {T<:AbstractFloat}
    xi = Matrix{T}(residuals(sdfm))
    if sdfm.method in (:fglr, :fhlz) && sdfm.gdfm.standardized && sdfm.units === :standardized
        σ = max.(vec(std(sdfm.gdfm.X; dims=1)), T(1e-10))
        xi ./= σ'
    end
    xi
end

"""Lag-window Bartlett spectral power of each residual series integrated over `|θ| ∈ [w_lo, w_hi]`."""
function _gdfmbp_idio_band_power(
    xi::Matrix{T},
    w_lo::T,
    w_hi::T,
) where {T<:AbstractFloat}
    T_obs, N = size(xi)
    xi_c = xi .- mean(xi; dims=1)
    M = _select_lag_window(T_obs)
    power = zeros(T, N)
    inv_π = one(T) / T(π)
    bw = w_hi - w_lo

    @inbounds for i in 1:N
        col = @view xi_c[:, i]
        γ0 = dot(col, col) / T(T_obs)
        p_i = (bw * inv_π) * γ0
        for k in 1:M
            k >= T_obs && break
            w_k = one(T) - T(k) / T(M + 1)
            γk = dot(@view(col[1:(T_obs - k)]), @view(col[(1 + k):T_obs])) / T(T_obs)
            band_k = (sin(T(k) * w_hi) - sin(T(k) * w_lo)) / (T(k) * T(π))
            p_i += T(2) * w_k * γk * band_k
        end
        power[i] = max(p_i, zero(T))
    end
    power
end

"""Panel transfer matrix `Θ(ω)` (`N × q` complex) for `StructuralDFM` under `:fglr` or `:gdfm_var`."""
function _gdfmbp_panel_transfer_var(
    Lambda::Matrix{T},
    A_mats::Vector{Matrix{T}},
    B0::Matrix{T},
    scale_vec::Union{Nothing,Vector{T}},
    ω::T,
) where {T<:AbstractFloat}
    n_fac = size(B0, 1)
    p = length(A_mats)
    Az = Matrix{Complex{T}}(I, n_fac, n_fac)
    @inbounds for lag in 1:p
        Az .-= Complex{T}.(A_mats[lag]) .* cis(-lag * ω)
    end
    F_fac = lu(Az; check=false)
    B0_c = Complex{T}.(B0)
    Psi_f = issuccess(F_fac) ? (F_fac \ B0_c) : (pinv(Az) * B0_c)
    Theta = Complex{T}.(Lambda) * Psi_f
    scale_vec !== nothing && (Theta .*= scale_vec)
    Theta
end

"""Panel transfer matrix `Θ(ω)` (`N × q` complex) for `StructuralDFM` under `:fhlz`."""
function _gdfmbp_panel_transfer_ma(C_irf::Array{T,3}, ω::T) where {T<:AbstractFloat}
    H_st, N, q = size(C_irf)
    Theta = zeros(Complex{T}, N, q)
    @inbounds for h in 0:(H_st - 1)
        e_h = cis(-h * ω)
        Theta .+= @view(C_irf[h + 1, :, :]) .* e_h
    end
    Theta
end

# =============================================================================
# Public API: bandpass_gdfm
# =============================================================================

"""
    bandpass_gdfm(gdfm::GeneralizedDynamicFactorModel, band; r=gdfm.r) -> BandPassGDFMResult

Extract the frequency-band common component of a [`GeneralizedDynamicFactorModel`](@ref)
over ``|\\theta| \\in [\\omega_{\\text{low}}, \\omega_{\\text{high}}] \\subseteq [0, \\pi]``.

Computes both the two-sided dynamic-PCA band-filtered component (`common_twosided`) and the
strictly one-sided FHLR generalized principal component projection (`common_onesided`).
Partitioning ``[0, \\pi]`` at any ``\\omega_0 \\in (0, \\pi)`` into `(0, ω₀)` and `(ω₀, π)`
recovers `gdfm.common_component` additively (`common_twosided((0, ω₀)) + common_twosided((ω₀, π)) ≈ gdfm.common_component`).

# Arguments
- `gdfm::GeneralizedDynamicFactorModel`: Estimated Generalized Dynamic Factor Model
- `band::Tuple{Real,Real}`: Angular frequency band `(omega_low, omega_high)` with `0 ≤ omega_low < omega_high ≤ π`

# Keyword Arguments
- `r::Int=gdfm.r`: Static factor rank for one-sided projection (`gdfm.q ≤ r ≤ N`)

# Returns
[`BandPassGDFMResult`](@ref) containing two-sided and one-sided band common components,
band variance shares, and metadata.
"""
function bandpass_gdfm(
    gdfm::GeneralizedDynamicFactorModel{T},
    band::Tuple{Real,Real};
    r::Int=gdfm.r,
) where {T<:AbstractFloat}
    T_obs, N = size(gdfm.X)
    w_lo, w_hi = _gdfmbp_validate_band(band, T)
    (gdfm.q <= r <= N) || throw(ArgumentError(
        "r must be in $(gdfm.q):$N, got $r"))

    X_proc = gdfm.standardized ? _standardize(gdfm.X) : copy(gdfm.X)
    n_freq = length(gdfm.frequencies)
    M = _gdfm_filter_halfwidth(T_obs, n_freq, gdfm.spectral, gdfm.bandwidth)
    X_pad = _reflect_pad(X_proc, M)

    K_band = _gdfmbp_band_kernels(
        gdfm.loadings_spectral, gdfm.frequencies, T_obs, gdfm.spectral, M, w_lo, w_hi)
    common_2s = _apply_symmetric_filter(X_proc, X_pad, K_band)

    # One-sided projection of the band common component onto contemporaneous Z' X_t
    Γχ_band = _gdfmbp_gamma_band(gdfm.spectral_density_chi, gdfm.frequencies, 0, w_lo, w_hi)
    Γχ_full = _gdfmbp_gamma_band(gdfm.spectral_density_chi, gdfm.frequencies, 0, zero(T), T(π))
    Γχ0 = _gamma_from_spectrum(gdfm.spectral_density_chi, gdfm.frequencies, 0)
    ΓX0 = _gamma_from_spectrum(gdfm.spectral_density_X, gdfm.frequencies, 0)
    Γξ0 = ΓX0 - Γχ0

    Z_use = (r == gdfm.r && size(gdfm.Z, 2) == r) ? gdfm.Z : _onesided_Z(Γχ0, Γξ0, r)
    Gz = Z_use' * ΓX0 * Z_use
    Gz_sym = Matrix{T}((Gz + Gz') / T(2))
    W_band = Γχ_band * Z_use * robust_inv(Gz_sym; silent=true)
    common_1s = (X_proc * Z_use) * transpose(W_band)

    if gdfm.standardized
        σ = max.(vec(std(gdfm.X; dims=1)), T(1e-10))
        common_2s .*= σ'
        common_1s .*= σ'
        if w_lo == zero(T)
            μ = vec(mean(gdfm.X; dims=1))
            common_2s .+= μ'
            common_1s .+= μ'
        end
    end

    variance_share = Vector{T}(undef, N)
    @inbounds for i in 1:N
        tot_i = Γχ_full[i, i]
        variance_share[i] = tot_i > T(1e-12) ?
            clamp(Γχ_band[i, i] / tot_i, zero(T), one(T)) : zero(T)
    end

    BandPassGDFMResult{T}(
        common_2s, common_1s, variance_share, (w_lo, w_hi), r, copy(gdfm.varnames))
end

# =============================================================================
# Public API: eurocoin_indicator
# =============================================================================

"""
    eurocoin_indicator(X, target, q; cutoff=π/6, r=0, bandwidth=0, kernel=:bartlett, standardize=true) -> EurocoinResult

Construct the Altissimo et al. (2010) real-time **Eurocoin** low-frequency growth
indicator by projecting the medium-to-long-run component of `target`
(``|\\theta| \\le \\text{cutoff}``) onto the strictly one-sided FHLR generalized static
factor space ``Z^{\\top} X_t``.

Because estimation of the factor projection weights uses only contemporaneous ``F_t = Z^{\\top} X_t``
weighted via the band-integrated cross-spectrum ``\\int_{-\\text{cutoff}}^{\\text{cutoff}}
\\hat{\\Sigma}_{F,y}(\\theta)\\, d\\theta``, the resulting indicator suffers no end-of-sample
phase shift or two-sided filter distortion at ``t = T_{\\text{obs}}``.

# Arguments
- `X::AbstractMatrix`: Panel of monthly/high-frequency indicators (`T_obs × N`)
- `target::AbstractVector`: Target growth series (length `T_obs`)
- `q::Int`: Number of dynamic common shocks (`1 ≤ q ≤ min(T_obs, N)`)

# Keyword Arguments
- `cutoff::Real=π/6`: Angular frequency low-pass cutoff (default `π/6`, periods `≥ 12` months)
- `r::Int=0`: Static factor dimension for one-sided weights (`0` defaults to `q`)
- `bandwidth::Int=0`: Spectral lag-window truncation `M` (`0` selects automatically)
- `kernel::Symbol=:bartlett`: Spectral lag-window kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `standardize::Bool=true`: Standardize columns of `X` before spectral and factor extraction

# Returns
[`EurocoinResult`](@ref) containing the real-time `indicator`, benchmark `target_lowpass`,
projection `weights`, `r2`, and model metadata.

# References
- Altissimo, F., Cristadoro, R., Forni, M., Lippi, M., & Veronese, G. (2010).
  New Eurocoin: Tracking Economic Growth in Real Time. *Review of Economics and
  Statistics*, 92(4), 1024–1034.
"""
function eurocoin_indicator(
    X::AbstractMatrix{T},
    target::AbstractVector,
    q::Int;
    cutoff::Real=π / 6,
    r::Int=0,
    bandwidth::Int=0,
    kernel::Symbol=:bartlett,
    standardize::Bool=true,
) where {T<:AbstractFloat}
    _validate_data(X, "X")
    _validate_data(target, "target")
    T_obs, N = size(X)
    length(target) == T_obs || throw(ArgumentError(
        "target length ($(length(target))) must equal number of rows in X ($T_obs)"))

    r_use = r == 0 ? q : r
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    r_use >= q || throw(ArgumentError("r must be >= q, got r=$r_use, q=$q"))
    validate_factor_inputs(T_obs, N, r_use; context="static factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))
    bandwidth >= 0 || throw(ArgumentError("bandwidth must be >= 0, got $bandwidth"))

    cutoff_T = T(cutoff)
    (isfinite(cutoff_T) && zero(T) < cutoff_T < T(π)) || throw(ArgumentError(
        "cutoff must satisfy 0 < cutoff < π, got $cutoff_T"))

    X_mat = Matrix{T}(X)
    μ_x = vec(mean(X_mat; dims=1))
    σ_x = standardize ? max.(vec(std(X_mat; dims=1)), T(1e-10)) : ones(T, N)
    X_proc = (X_mat .- μ_x') ./ σ_x'

    y_vec = T.(collect(target))
    μ_y = T(mean(y_vec))
    y_c = y_vec .- μ_y

    # Step 1: FHLR one-sided static factor space Z' X_t
    M_gdfm = bandwidth > 0 ? bandwidth : _select_lag_window(T_obs)
    freq_x, Sx = _estimate_spectral_density_lagwindow(X_proc, M_gdfm, kernel)
    evals_x, evecs_x = _spectral_eigendecomposition(Sx)
    Schi = _compute_common_spectral_density(evecs_x[:, 1:q, :], evals_x[1:q, :])
    Γχ0 = _gamma_from_spectrum(Schi, freq_x, 0)
    ΓX0 = _gamma_from_spectrum(Sx, freq_x, 0)
    Γξ0 = ΓX0 - Γχ0
    Z = _onesided_Z(Γχ0, Γξ0, r_use)
    F = X_proc * Z                                     # T_obs × r_use

    # Step 2: Two-sided ideal low-pass benchmark on centered target
    target_lowpass = _gdfmbp_lowpass_1d(y_c, cutoff_T) .+ μ_y

    # Step 3: Altissimo et al. (2010) frequency-band cross-spectral projection onto Z' X_t
    # Resolution bandwidth M_yf ensures sharp passband/stopband separation around cutoff
    M_yf = bandwidth > 0 ? bandwidth : min(max(T_obs - 2, 1), max(20, round(Int, T(2) * sqrt(T(T_obs)))))
    YF = hcat(y_c, F)
    freq_yf, S_yf = _estimate_spectral_density_lagwindow(YF, M_yf, kernel)
    Γ_yf_band = _gdfmbp_gamma_band(S_yf, freq_yf, 0, zero(T), cutoff_T)
    cov_F_ylp = Vector{T}(Γ_yf_band[2:(r_use + 1), 1])

    Σ_F = Matrix{T}((F' * F) / T(T_obs))
    Σ_F = Matrix{T}((Σ_F + Σ_F') / T(2))
    v = robust_inv(Σ_F; silent=true) * cov_F_ylp

    weights_proc = Vector{T}(Z * v)
    indicator = (X_proc * weights_proc) .+ μ_y
    weights = standardize ? (weights_proc ./ σ_x) : weights_proc

    y_lp_c = target_lowpass .- mean(target_lowpass)
    ss_tot = dot(y_lp_c, y_lp_c)
    ss_res = sum(abs2, target_lowpass .- indicator)
    r2_val = ss_tot > T(1e-12) ? max(one(T) - ss_res / ss_tot, zero(T)) : zero(T)

    EurocoinResult{T}(indicator, target_lowpass, weights, r2_val, cutoff_T, q, r_use)
end

function eurocoin_indicator(
    X::AbstractMatrix,
    target::AbstractVector,
    q::Int;
    kwargs...,
)
    T = promote_type(Float64, float(eltype(X)), float(eltype(target)))
    eurocoin_indicator(Matrix{T}(X), Vector{T}(target), q; kwargs...)
end

# =============================================================================
# Public API: spectral_fevd
# =============================================================================

"""
    spectral_fevd(sdfm::StructuralDFM, band=(2π/32, 2π/6); include_idiosyncratic=false) -> SpectralFEVDResult

Compute the frequency-band forecast error variance decomposition for each variable in a
[`StructuralDFM`](@ref) over angular frequencies
``|\\omega| \\in [\\omega_{\\text{low}}, \\omega_{\\text{high}}] \\subseteq [0, \\pi]``
(default `(2π/32, 2π/6)`, corresponding to monthly 6–32 period business-cycle frequencies).

Supports all estimation paths of `StructuralDFM` (`:fglr`, `:gdfm_var`, and `:fhlz`).
When `include_idiosyncratic=false`, each row of `shares` (`N × q`) sums to `1.0`; when
`include_idiosyncratic=true`, an `"Idiosyncratic"` column based on the residual band
spectral density is appended (`N × (q + 1)`).

# Arguments
- `sdfm::StructuralDFM`: Estimated Structural Dynamic Factor Model
- `band::Tuple{Real,Real}=(2π/32, 2π/6)`: Angular frequency band `(omega_low, omega_high)`

# Keyword Arguments
- `include_idiosyncratic::Bool=false`: Append idiosyncratic band variance share as final column

# Returns
[`SpectralFEVDResult`](@ref) with `shares`, `power_band`, `total_power_band`, and labels.
"""
function spectral_fevd(
    sdfm::StructuralDFM{T},
    band::Tuple{Real,Real}=(T(2π / 32), T(2π / 6));
    include_idiosyncratic::Bool=false,
) where {T<:AbstractFloat}
    w_lo, w_hi = _gdfmbp_validate_band(band, T)
    N = length(sdfm.varnames)
    q = sdfm.gdfm.q
    n_quad = 64
    nodes, qweights = _gauss_legendre_interval(n_quad, w_lo, w_hi)
    inv_π = one(T) / T(π)

    power_c = zeros(T, N, q)
    if sdfm.method === :fhlz
        @inbounds for i in eachindex(nodes)
            Theta = _gdfmbp_panel_transfer_ma(sdfm.structural_irf, nodes[i])
            wt_i = qweights[i] * inv_π
            for j in 1:q, v in 1:N
                power_c[v, j] += wt_i * abs2(Theta[v, j])
            end
        end
    else
        Lambda = sdfm.method === :fglr ? sdfm.loadings_static : sdfm.loadings_td
        n_fac = size(sdfm.B0, 1)
        A_mats = extract_ar_coefficients(sdfm.factor_var.B, n_fac, sdfm.factor_var.p)
        scale_vec = if sdfm.method === :fglr && sdfm.units === :raw && sdfm.gdfm.standardized
            max.(vec(std(sdfm.gdfm.X; dims=1)), T(1e-10))
        else
            nothing
        end
        @inbounds for i in eachindex(nodes)
            Theta = _gdfmbp_panel_transfer_var(Lambda, A_mats, sdfm.B0, scale_vec, nodes[i])
            wt_i = qweights[i] * inv_π
            for j in 1:q, v in 1:N
                power_c[v, j] += wt_i * abs2(Theta[v, j])
            end
        end
    end

    if include_idiosyncratic
        xi = _gdfmbp_idio_residuals(sdfm)
        p_idio = _gdfmbp_idio_band_power(xi, w_lo, w_hi)
        power_band = hcat(power_c, p_idio)
        snames = vcat(copy(sdfm.shock_names), ["Idiosyncratic"])
    else
        power_band = power_c
        snames = copy(sdfm.shock_names)
    end

    n_s = size(power_band, 2)
    total_power = Vector{T}(undef, N)
    shares = zeros(T, N, n_s)
    @inbounds for i in 1:N
        tot = zero(T)
        for j in 1:n_s
            tot += power_band[i, j]
        end
        total_power[i] = tot
        if tot > T(1e-15)
            for j in 1:n_s
                shares[i, j] = power_band[i, j] / tot
            end
        else
            fill!(@view(shares[i, :]), one(T) / T(n_s))
        end
    end

    SpectralFEVDResult{T}(
        shares, power_band, total_power, (w_lo, w_hi),
        include_idiosyncratic, copy(sdfm.varnames), snames)
end

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, res::BandPassGDFMResult{T}) where {T}
    T_obs, N = size(res.common_twosided)
    w_lo = round(Float64(res.band[1]); digits=4)
    w_hi = round(Float64(res.band[2]); digits=4)
    spec = Any[
        "Band [ω_low, ω_high]"  "[$w_lo, $w_hi]";
        "Observations (T)"      T_obs;
        "Variables (N)"         N;
        "One-sided rank (r)"    res.r;
        "Mean variance share"   _fmt_pct(mean(res.variance_share))
    ]
    _pretty_table(io, spec;
        title = "Band-Pass GDFM Decomposition",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    n_show = min(N, 10)
    vdata = Matrix{Any}(undef, n_show, 2)
    for i in 1:n_show
        vdata[i, 1] = res.varnames[i]
        vdata[i, 2] = _fmt_pct(res.variance_share[i])
    end
    _pretty_table(io, vdata;
        title = "Band Share of Common Variance (first $n_show)",
        column_labels = ["Variable", "Share"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, res::EurocoinResult{T}) where {T}
    T_obs = length(res.indicator)
    w_c = round(Float64(res.cutoff); digits=4)
    per = round(Float64(2π / res.cutoff); digits=1)
    spec = Any[
        "Observations (T)"   T_obs;
        "Panel width (N)"    length(res.weights);
        "Dynamic shocks (q)" res.q;
        "Static rank (r)"    res.r;
        "Cutoff frequency"   "$w_c rad (period ≥ $per)";
        "In-sample R²"       _fmt(res.r2; digits=4);
        "Latest indicator"   _fmt(res.indicator[end]; digits=4);
        "Latest target LP"   _fmt(res.target_lowpass[end]; digits=4)
    ]
    _pretty_table(io, spec;
        title = "Eurocoin Real-Time Low-Frequency Indicator",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, res::SpectralFEVDResult{T}) where {T}
    N, n_s = size(res.shares)
    w_lo = round(Float64(res.band[1]); digits=4)
    w_hi = round(Float64(res.band[2]); digits=4)
    n_show = min(N, 10)
    data = Matrix{Any}(undef, n_show, n_s + 1)
    for i in 1:n_show
        data[i, 1] = res.varnames[i]
        for j in 1:n_s
            data[i, j + 1] = _fmt_pct(res.shares[i, j])
        end
    end
    _pretty_table(io, data;
        title = "Spectral FEVD over [$w_lo, $w_hi] rad",
        column_labels = vcat(["Variable"], res.shock_names),
        alignment = vcat([:l], fill(:r, n_s)),
    )
end

report(io::IO, res::BandPassGDFMResult) = show(io, res)
report(res::BandPassGDFMResult) = report(stdout, res)
report(io::IO, res::EurocoinResult) = show(io, res)
report(res::EurocoinResult) = report(stdout, res)
report(io::IO, res::SpectralFEVDResult) = show(io, res)
report(res::SpectralFEVDResult) = report(stdout, res)
