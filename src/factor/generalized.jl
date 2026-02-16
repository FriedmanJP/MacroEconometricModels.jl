# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

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

using LinearAlgebra, Statistics, FFTW, StatsAPI

# =============================================================================
# Generalized Dynamic Factor Model Type
# =============================================================================

"""
    GeneralizedDynamicFactorModel{T} <: AbstractFactorModel

GDFM with frequency-dependent loadings: Xₜ = χₜ + ξₜ.

Fields: X, factors, common_component, idiosyncratic, loadings_spectral,
spectral_density_X, spectral_density_chi, eigenvalues_spectral, frequencies,
q (dynamic factors), r (static factors), bandwidth, kernel, standardized, variance_explained.
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
end

# =============================================================================
# GDFM Estimation
# =============================================================================

"""
    estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0) -> GeneralizedDynamicFactorModel

Estimate Generalized Dynamic Factor Model using spectral methods.

# Arguments
- `X`: Data matrix (T × N)
- `q`: Number of dynamic factors

# Keyword Arguments
- `standardize::Bool=true`: Standardize data
- `bandwidth::Int=0`: Kernel bandwidth (0 = automatic selection)
- `kernel::Symbol=:bartlett`: Kernel for spectral smoothing (:bartlett, :parzen, :tukey)
- `r::Int=0`: Number of static factors (0 = same as q)

# Returns
`GeneralizedDynamicFactorModel` with common/idiosyncratic components and spectral loadings.

# Example
```julia
gdfm = estimate_gdfm(X, 3)
common_variance_share(gdfm)  # Fraction of variance explained by common component
```
"""
function estimate_gdfm(X::AbstractMatrix{T}, q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett, r::Int=0
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey))

    r_static = r == 0 ? q : r
    r_static < q && throw(ArgumentError("r must be >= q"))
    bandwidth = bandwidth <= 0 ? _select_bandwidth(T_obs) : bandwidth

    X_original = copy(X)
    X_proc = standardize ? _standardize(X) : X

    # Spectral analysis
    frequencies, spectral_X = _estimate_spectral_density(X_proc, bandwidth, kernel)
    eigenvalues, eigenvectors = _spectral_eigendecomposition(spectral_X)
    loadings = eigenvectors[:, 1:q, :]
    spectral_chi = _compute_common_spectral_density(loadings, eigenvalues[1:q, :])
    common = _reconstruct_time_domain(spectral_chi, X_proc)
    factors = _extract_time_domain_factors(X_proc, loadings, frequencies)
    var_explained = _compute_variance_explained(eigenvalues, q)

    # Unstandardize common component if needed
    if standardize
        μ, σ = mean(X_original, dims=1), max.(std(X_original, dims=1), T(1e-10))
        common = common .* σ .+ μ
    end
    idiosyncratic = X_original - common

    GeneralizedDynamicFactorModel{T}(X_original, factors, common, idiosyncratic, loadings,
        spectral_X, spectral_chi, eigenvalues, frequencies, q, r_static, bandwidth,
        kernel, standardize, var_explained)
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
        "Standardized"     m.standardized ? "Yes" : "No"
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

"""Automatic bandwidth selection: T^(1/3)."""
_select_bandwidth(T_obs::Int) = max(3, round(Int, T_obs^(1/3)))

# =============================================================================
# Spectral Density Estimation
# =============================================================================

"""Estimate spectral density matrix with kernel smoothing."""
function _estimate_spectral_density(X::AbstractMatrix{T}, bandwidth::Int, kernel::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(X)
    n_freq = div(T_obs, 2) + 1
    frequencies = [T(2π * (j-1) / T_obs) for j in 1:n_freq]

    # Periodogram
    X_fft = fft(X, 1)
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
function _reconstruct_time_domain(spectral_chi::Array{Complex{T},3}, X::AbstractMatrix{T}) where {T}
    T_obs, N = size(X)
    n_freq = size(spectral_chi, 3)
    X_fft = fft(X, 1)
    chi_fft = zeros(Complex{T}, T_obs, N)

    @inbounds for j in 1:n_freq
        S_chi, S_X = spectral_chi[:, :, j], X_fft[j, :] * X_fft[j, :]' / T_obs
        P = S_chi * inv(Hermitian(S_X + T(1e-10) * I))
        chi_fft[j, :] = P * X_fft[j, :]
        j > 1 && j < n_freq && (chi_fft[T_obs - j + 2, :] = conj(chi_fft[j, :]))
    end
    real(ifft(chi_fft, 1))
end

"""Extract time-domain factors via frequency-domain projection."""
function _extract_time_domain_factors(X::AbstractMatrix{T}, loadings::Array{Complex{T},3}, frequencies::Vector{T}) where {T}
    T_obs, N = size(X)
    _, q, n_freq = size(loadings)
    X_fft, F_fft = fft(X, 1), zeros(Complex{T}, T_obs, q)

    @inbounds for j in 1:n_freq
        L = loadings[:, :, j]
        F_fft[j, :] = (L' * L + T(1e-10) * I) \ (L' * X_fft[j, :])
        j > 1 && j < n_freq && (F_fft[T_obs - j + 2, :] = conj(F_fft[j, :]))
    end

    factors = real(ifft(F_fft, 1))
    # Normalize factors to unit variance
    for i in 1:q
        σ = std(factors[:, i])
        σ > T(1e-10) && (factors[:, i] ./= σ)
    end
    factors
end

"""Compute variance explained by first q factors (averaged across frequencies)."""
function _compute_variance_explained(eigenvalues::Matrix{T}, q::Int) where {T}
    total = mean(sum(eigenvalues, dims=1))
    [mean(eigenvalues[i, :]) / total for i in 1:q]
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
    ic_criteria_gdfm(X, max_q; standardize=true, bandwidth=0, kernel=:bartlett)

Information criteria for selecting number of dynamic factors.

Uses eigenvalue ratio test and cumulative variance threshold.

# Returns
Named tuple with:
- `eigenvalue_ratios`: Ratios of consecutive eigenvalues
- `cumulative_variance`: Cumulative variance explained
- `q_ratio`: Optimal q from eigenvalue ratio
- `q_variance`: Optimal q from 90% variance threshold
"""
function ic_criteria_gdfm(X::AbstractMatrix{T}, max_q::Int;
    standardize::Bool=true, bandwidth::Int=0, kernel::Symbol=:bartlett
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    (max_q < 1 || max_q > N) && throw(ArgumentError("max_q must be in [1, $N]"))
    bandwidth = bandwidth <= 0 ? _select_bandwidth(T_obs) : bandwidth

    X_proc = standardize ? _standardize(X) : X
    _, spectral = _estimate_spectral_density(X_proc, bandwidth, kernel)
    eigenvalues, _ = _spectral_eigendecomposition(spectral)
    avg_eig = vec(mean(eigenvalues, dims=2))

    # Eigenvalue ratio criterion
    ratios = [avg_eig[i] / avg_eig[i+1] for i in 1:min(max_q, N-1)]
    cum_var = cumsum(avg_eig[1:max_q]) / sum(avg_eig)
    q_ratio = argmax(ratios[1:min(max_q, length(ratios))])
    q_variance = something(findfirst(>=(T(0.9)), cum_var), max_q)

    (eigenvalue_ratios=ratios, cumulative_variance=cum_var, avg_eigenvalues=avg_eig[1:max_q],
     q_ratio=q_ratio, q_variance=q_variance)
end

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::GeneralizedDynamicFactorModel, h; method=:ar, ci_method=:none, conf_level=0.95, n_boot=1000)

Forecast h steps ahead using AR extrapolation of factors.

# Arguments
- `model`: Estimated GDFM
- `h`: Forecast horizon

# Keyword Arguments
- `method::Symbol=:ar`: Forecasting method (currently only `:ar` supported)
- `ci_method::Symbol=:none`: CI method — `:none`, `:theoretical`, or `:bootstrap`
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Bootstrap replications (for `:bootstrap`)

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).
"""
function forecast(model::GeneralizedDynamicFactorModel{T}, h::Int; method::Symbol=:ar,
    ci_method::Symbol=:none, conf_level::Real=0.95, n_boot::Int=1000) where {T}

    h < 1 && throw(ArgumentError("h must be positive"))
    method ∉ (:ar, :spectral) && throw(ArgumentError("method must be :ar or :spectral"))
    ci_method ∈ (:none, :theoretical, :bootstrap) || throw(ArgumentError("ci_method must be :none, :theoretical, or :bootstrap"))

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
        z = zeros(T, h, q)
        zx = zeros(T, h, N)
        if model.standardized
            _unstandardize_factor_forecast!(X_fc, zx, zx, zx, model.X)
        end
        return _build_factor_forecast(F_fc, X_fc, z, z, zx, copy(zx), z, copy(zx), h, conf_T, :none)
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
                boot_idx = rand(1:(T_obs-1))
                f = phi[i] * f + resids_per_factor[i][boot_idx]
                F_boot[b, t, i] = f
            end
        end
        for t in 1:h
            X_boot[b, t, :] = L_avg * F_boot[b, t, :] .+ idio_std .* randn(T, N)
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
