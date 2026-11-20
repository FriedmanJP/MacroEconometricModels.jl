# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Time-Varying Locally Stationary Generalized Dynamic Factor Model (TV-GDFM).

Implements locally stationary time-varying spectral dynamic factor estimation following
Eichler, Motta & von Sachs (2011) and Barigozzi, Hallin, Soccorsi & von Sachs (2021):
``X_t = \\chi_t + \\xi_t``, where ``\\chi_t = B(t/T, L) u_t`` has smoothly varying
transfer functions in rescaled time ``u = t/T \\in (0, 1)``.

References:
- Barigozzi, M., Hallin, M., Soccorsi, S., & von Sachs, R. (2021). Time-varying
  generalized dynamic factor models. *Econometric Theory*, 37(6), 1152–1205.
- Eichler, M., Motta, G., & von Sachs, R. (2011). Fitting dynamic factor models to
  non-stationary time series. *Journal of Econometrics*, 163(1), 51–70.
"""

using LinearAlgebra, Statistics, StatsAPI

# =============================================================================
# TimeVaryingGDFM Type
# =============================================================================

"""
    TimeVaryingGDFM{T<:AbstractFloat} <: AbstractFactorModel

Locally stationary time-varying Generalized Dynamic Factor Model with smoothly varying
time-frequency spectral representation in rescaled time ``u = t / T``:

```math
X_{t} = \\chi_{t} + \\xi_{t}, \\qquad \\Sigma_X(u, \\theta) = L(u, \\theta) \\Lambda_q(u, \\theta) L(u, \\theta)^{H} + \\Sigma_\\xi(u, \\theta)
```

where:
- ``X_{t}`` is the ``N \\times 1`` observed panel vector at time ``t``
- ``\\chi_{t}`` is the time-varying ``q``-factor common component at time ``t``
- ``\\xi_{t}`` is the weakly cross-correlated idiosyncratic component at time ``t``
- ``\\Sigma_X(u, \\theta)`` is the local evolutionary spectral density matrix at rescaled time ``u \\in (0, 1)`` and angular frequency ``\\theta \\in [0, \\pi]``
- ``L(u, \\theta)`` is the ``N \\times q`` matrix of top-``q`` dynamic eigenvectors of ``\\Sigma_X(u, \\theta)``
- ``\\Lambda_q(u, \\theta)`` is the ``q \\times q`` diagonal matrix of top-``q`` dynamic eigenvalues
- ``\\Sigma_\\xi(u, \\theta)`` is the residual idiosyncratic local spectral density matrix

# Fields
- `X::Matrix{T}`: Original panel data (`T_obs × N`)
- `common_component::Matrix{T}`: Reconstructed time-varying common component (`T_obs × N`)
- `idiosyncratic::Matrix{T}`: Idiosyncratic remainder `X - common_component` (`T_obs × N`)
- `eigenvalues_path::Array{T,3}`: Descending local spectral eigenvalues (`n_grid × N × n_freq`)
- `frequencies::Vector{T}`: Angular frequency grid in `[0, π]` (length `n_freq`)
- `u_grid::Vector{T}`: Rescaled time grid points in `(0, 1)` (length `n_grid`)
- `variance_share_path::Matrix{T}`: Time-varying normalized top-`q` variance shares (`T_obs × q`)
- `q::Int`: Number of dynamic common factors
- `n_grid::Int`: Number of rescaled time grid points
- `time_bandwidth::T`: Local time-kernel bandwidth `b_T`
- `lag_bandwidth::Int`: Lag-window truncation `M_T`
- `time_kernel::Symbol`: Time-smoothing kernel (`:epanechnikov`, `:gaussian`, or `:uniform`)
- `lag_kernel::Symbol`: Spectral lag-window kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `standardized::Bool`: Whether estimation standardized `X` prior to filtering
- `varnames::Vector{String}`: Panel variable names (length `N`)
"""
struct TimeVaryingGDFM{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    common_component::Matrix{T}
    idiosyncratic::Matrix{T}
    eigenvalues_path::Array{T,3}
    frequencies::Vector{T}
    u_grid::Vector{T}
    variance_share_path::Matrix{T}
    q::Int
    n_grid::Int
    time_bandwidth::T
    lag_bandwidth::Int
    time_kernel::Symbol
    lag_kernel::Symbol
    standardized::Bool
    varnames::Vector{String}
end

# =============================================================================
# Internal Helpers (_tvgdfm_*)
# =============================================================================

"""Normalized kernel weights `w_t(u_r)` summing to 1 for rescaled observation times `u_obs`."""
function _tvgdfm_time_weights(
    u_obs::Vector{T},
    u_r::T,
    b_T::T,
    time_kernel::Symbol,
) where {T<:AbstractFloat}
    T_obs = length(u_obs)
    w = Vector{T}(undef, T_obs)
    @inbounds for t in 1:T_obs
        z = (u_obs[t] - u_r) / b_T
        if time_kernel === :epanechnikov
            w[t] = abs(z) <= one(T) ? (one(T) - z * z) : zero(T)
        elseif time_kernel === :gaussian
            w[t] = exp(-T(0.5) * z * z)
        else
            w[t] = abs(z) <= one(T) ? one(T) : zero(T)
        end
    end
    total = sum(w)
    if total > zero(T)
        w ./= total
    else
        # Fallback if a very narrow compact kernel misses all grid midpoints
        idx_near = argmin(abs.(u_obs .- u_r))
        w[idx_near] = one(T)
    end
    w
end

"""Lag-window weight `κ(k / M_T)` for `:bartlett`, `:parzen`, or `:tukey`."""
@inline function _tvgdfm_lag_weight(
    k::Int,
    M_T::Int,
    lag_kernel::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    u = T(k) / T(max(M_T, 1))
    if lag_kernel === :bartlett
        return one(T) - u
    elseif lag_kernel === :parzen
        return u <= T(0.5) ? (one(T) - T(6) * u^2 + T(6) * u^3) : T(2) * (one(T) - u)^3
    else
        return T(0.5) * (one(T) + cos(T(π) * u))
    end
end

"""
Time-localized Hermitian spectral density `Σ̂_X(u_r, θ_h)` using symmetric kernel tapering
`Γ̂(u_r, k) = ∑_t √(w_t w_{t-k}) X_t X_{t-k}'` to preserve PSD lag-window spectra.
"""
function _tvgdfm_local_spectrum(
    Xc::Matrix{T},
    w::Vector{T},
    frequencies::Vector{T},
    M_T::Int,
    lag_kernel::Symbol,
) where {T<:AbstractFloat}
    T_obs, N = size(Xc)
    n_freq = length(frequencies)
    K_eff = min(M_T, T_obs - 1)

    sqrt_w = sqrt.(w)
    Xw = Xc .* sqrt_w

    Gamma = Vector{Matrix{T}}(undef, K_eff + 1)
    Gamma[1] = Xw' * Xw
    @inbounds for k in 1:K_eff
        Gamma[k + 1] = (@view(Xw[(k + 1):T_obs, :]))' * (@view(Xw[1:(T_obs - k), :]))
    end

    spectral = Array{Complex{T},3}(undef, N, N, n_freq)
    inv_twoπ = one(T) / T(2π)
    S = zeros(Complex{T}, N, N)
    @inbounds for h in 1:n_freq
        θ = frequencies[h]
        fill!(S, zero(Complex{T}))
        S .+= Gamma[1]
        for k in 1:K_eff
            wk = _tvgdfm_lag_weight(k, M_T, lag_kernel, T)
            wk == zero(T) && continue
            e = cis(-k * θ)
            Gk = Gamma[k + 1]
            S .+= wk .* (Gk .* e .+ Gk' .* conj(e))
        end
        S .*= inv_twoπ
        spectral[:, :, h] = (S + S') .* T(0.5)
    end
    spectral
end

"""Align eigenvector columns across adjacent time grid points `u_{r-1} -> u_r` and frequencies."""
function _tvgdfm_align_eigenvectors(
    L_curr::Array{Complex{T},3},
    L_prev::Union{Nothing,Array{Complex{T},3}},
) where {T<:AbstractFloat}
    L = copy(L_curr)
    _, q, _ = size(L)
    if L_prev !== nothing
        @inbounds for k in 1:q
            overlap = dot(@view(L_prev[:, k, 1]), @view(L[:, k, 1]))
            a = abs(overlap)
            if a > eps(T)
                L[:, k, 1] .*= conj(overlap) / a
            end
        end
    end
    _align_eigenvector_phases(L)
end

"""Invert `P(u_r, θ) = L(u_r, θ) L(u_r, θ)ᴴ` on `[0, π]` to real symmetric filter kernels `K_0..K_{M_T}`."""
function _tvgdfm_projector_kernels(
    loadings::Array{Complex{T},3},
    frequencies::Vector{T},
    M_T::Int,
) where {T<:AbstractFloat}
    N, _, n_freq = size(loadings)
    R_list = Vector{Matrix{T}}(undef, n_freq)
    I_list = Vector{Matrix{T}}(undef, n_freq)
    @inbounds for h in 1:n_freq
        Lh = @view loadings[:, :, h]
        Ph = Lh * Lh'
        R_list[h] = real.(Ph)
        I_list[h] = imag.(Ph)
    end

    K_list = Vector{Matrix{T}}(undef, M_T + 1)
    if n_freq == 1
        K_list[1] = (R_list[1] + R_list[1]') .* T(0.5)
        @inbounds for k in 1:M_T
            K_list[k + 1] = zeros(T, N, N)
        end
        return K_list
    end

    M_grid = n_freq - 1
    inv_n = one(T) / T(2 * M_grid)
    @inbounds for k in 0:M_T
        K = copy(R_list[1])
        isodd(k) ? (K .-= R_list[n_freq]) : (K .+= R_list[n_freq])
        for h in 2:(n_freq - 1)
            θ = frequencies[h]
            c, s = cos(k * θ), sin(k * θ)
            K .+= (T(2) * c) .* R_list[h] .- (T(2) * s) .* I_list[h]
        end
        K .*= inv_n
        if k == 0
            K .= (K .+ K') .* T(0.5)
        end
        K_list[k + 1] = K
    end
    K_list
end

"""Normalized trapezoidal integration weights over `[0, π]` summing to 1."""
function _tvgdfm_freq_weights(frequencies::Vector{T}) where {T<:AbstractFloat}
    n_freq = length(frequencies)
    n_freq == 1 && return T[one(T)]
    wt = zeros(T, n_freq)
    @inbounds for h in 1:(n_freq - 1)
        Δ = frequencies[h + 1] - frequencies[h]
        wt[h] += T(0.5) * Δ
        wt[h + 1] += T(0.5) * Δ
    end
    total = sum(wt)
    total > zero(T) ? (wt ./ total) : fill(one(T) / T(n_freq), n_freq)
end

"""Precompute clamped linear interpolation indices and weights from `u_grid` to `u_obs`."""
function _tvgdfm_interp_weights(u_obs::Vector{T}, u_grid::Vector{T}) where {T<:AbstractFloat}
    T_obs = length(u_obs)
    n_grid = length(u_grid)
    idx1 = Vector{Int}(undef, T_obs)
    idx2 = Vector{Int}(undef, T_obs)
    alpha = Vector{T}(undef, T_obs)
    if n_grid == 1
        fill!(idx1, 1)
        fill!(idx2, 1)
        fill!(alpha, zero(T))
        return idx1, idx2, alpha
    end
    @inbounds for t in 1:T_obs
        ut = u_obs[t]
        if ut <= u_grid[1]
            idx1[t] = 1
            idx2[t] = 1
            alpha[t] = zero(T)
        elseif ut >= u_grid[n_grid]
            idx1[t] = n_grid
            idx2[t] = n_grid
            alpha[t] = zero(T)
        else
            i2 = searchsortedfirst(u_grid, ut)
            i1 = max(i2 - 1, 1)
            Δ = u_grid[i2] - u_grid[i1]
            idx1[t] = i1
            idx2[t] = i2
            alpha[t] = Δ > zero(T) ? (ut - u_grid[i1]) / Δ : zero(T)
        end
    end
    idx1, idx2, alpha
end

# =============================================================================
# Public Estimator
# =============================================================================

"""
    estimate_tv_gdfm(X, q; kwargs...) -> TimeVaryingGDFM

Estimate a locally stationary time-varying Generalized Dynamic Factor Model via localized
kernel-tapered lag-window spectral estimation and dynamic principal component projection
(Eichler, Motta & von Sachs 2011; Barigozzi, Hallin, Soccorsi & von Sachs 2021).

At each rescaled time point ``u_r \\in (0, 1)``, the estimator computes kernel-weighted
lag autocovariances using symmetric square-root tapering:

```math
\\hat{\\Gamma}(u_r, k) = \\sum_{t=k+1}^{T} \\sqrt{w_t(u_r)\\, w_{t-k}(u_r)}\\, X_t X_{t-k}^{\\top}
```

where:
- ``u_r`` is the ``r``-th rescaled time grid point in ``(0, 1)``
- ``w_t(u_r)`` are normalized time-kernel weights summing to ``1``
- ``X_t`` is the (standardized and centered) ``N \\times 1`` panel vector at time ``t``
- ``k`` is the lag index in ``0, \\dots, M_T``
- ``T`` is the sample size `size(X, 1)`

The resulting Hermitian local spectral density ``\\hat{\\Sigma}_X(u_r, \\theta)`` is
decomposed into its top-``q`` dynamic eigenvectors, inverted to local two-sided projector
kernels ``K_k(u_r)``, linearly interpolated to every observation ``t = 1, \\dots, T``, and
applied with symmetric reflection padding to recover `common_component`.

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T_obs × N`)
- `q::Int`: Number of dynamic common factors (`1 ≤ q ≤ min(T_obs, N)`)

# Keyword Arguments
- `time_bandwidth::Real=0.0`: Time-kernel bandwidth `b_T` (`0` selects `max(0.12, T_obs^(-0.2))`)
- `lag_bandwidth::Int=0`: Spectral lag-window truncation `M_T` (`0` selects `max(3, round(0.5 * sqrt(b_T * T_obs)))`)
- `time_kernel::Symbol=:epanechnikov`: Local time kernel (`:epanechnikov`, `:gaussian`, or `:uniform`)
- `lag_kernel::Symbol=:bartlett`: Lag-window kernel (`:bartlett`, `:parzen`, or `:tukey`)
- `n_grid::Int=25`: Number of rescaled time grid points in `(0, 1)` (`≥ 1`)
- `standardize::Bool=true`: Standardize panel columns prior to filtering, then unstandardize outputs
- `varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing`: Panel variable names (length `N`)

# Returns
[`TimeVaryingGDFM`](@ref) containing the reconstructed `common_component`, `idiosyncratic`
remainder, local `eigenvalues_path`, and time-varying `variance_share_path`.

# Example
```julia
X = dgp_dynamic_factors(Xoshiro(42); N=20, T=200).X
tv = estimate_tv_gdfm(X, 2)
common_variance_share(tv)
```

# References
- Barigozzi, M., Hallin, M., Soccorsi, S., & von Sachs, R. (2021). Time-varying
  generalized dynamic factor models. *Econometric Theory*, 37(6), 1152–1205.
- Eichler, M., Motta, G., & von Sachs, R. (2011). Fitting dynamic factor models to
  non-stationary time series. *Journal of Econometrics*, 163(1), 51–70.
"""
function estimate_tv_gdfm(
    X::AbstractMatrix{T},
    q::Int;
    time_bandwidth::Real=0.0,
    lag_bandwidth::Int=0,
    time_kernel::Symbol=:epanechnikov,
    lag_kernel::Symbol=:bartlett,
    n_grid::Int=25,
    standardize::Bool=true,
    varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    T_obs >= 4 || throw(ArgumentError("Need at least 4 observations, got $T_obs"))
    validate_factor_inputs(T_obs, N, q; context="dynamic factors")
    validate_option(time_kernel, "time_kernel", (:epanechnikov, :gaussian, :uniform))
    validate_option(lag_kernel, "lag_kernel", (:bartlett, :parzen, :tukey))
    n_grid >= 1 || throw(ArgumentError("n_grid must be >= 1, got $n_grid"))

    bw_in = T(time_bandwidth)
    (isfinite(bw_in) && bw_in >= zero(T)) || throw(ArgumentError(
        "time_bandwidth must be finite and >= 0, got $time_bandwidth"))
    lag_bandwidth >= 0 || throw(ArgumentError(
        "lag_bandwidth must be >= 0, got $lag_bandwidth"))

    vn = varnames === nothing ? ["Var $i" for i in 1:N] : String.(collect(varnames))
    length(vn) == N || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $N columns"))

    b_T = bw_in > zero(T) ? bw_in : max(T(0.12), T(T_obs)^(-T(0.2)))
    M_T = lag_bandwidth > 0 ? lag_bandwidth : min(T_obs - 1, max(3, round(Int, T(0.5) * sqrt(b_T * T(T_obs)))))

    X_orig = Matrix{T}(X)
    μ = mean(X_orig; dims=1)
    σ = max.(std(X_orig; dims=1), T(1e-10))
    X_proc = standardize ? (X_orig .- μ) ./ σ : copy(X_orig)
    Xc = standardize ? X_proc : (X_proc .- mean(X_proc; dims=1))

    # Frequency grid in [0, π] strictly finer than M_T so lag-k projector inversion has no Nyquist aliasing
    n_freq = max(2 * M_T + 1, 33)
    frequencies = T[T(π) * T(h - 1) / T(n_freq - 1) for h in 1:n_freq]
    freq_wt = _tvgdfm_freq_weights(frequencies)

    half_step = one(T) / T(2 * n_grid)
    u_grid = collect(range(half_step, one(T) - half_step; length=n_grid))
    u_obs = T[(T(t) - T(0.5)) / T(T_obs) for t in 1:T_obs]

    eigenvalues_path = Array{T,3}(undef, n_grid, N, n_freq)
    share_grid = Matrix{T}(undef, n_grid, q)

    X_pad = _reflect_pad(X_proc, M_T)
    chi_grid = Vector{Matrix{T}}(undef, n_grid)
    L_prev = nothing

    @inbounds for r in 1:n_grid
        w_r = _tvgdfm_time_weights(u_obs, u_grid[r], b_T, time_kernel)
        S_r = _tvgdfm_local_spectrum(Xc, w_r, frequencies, M_T, lag_kernel)
        evals_r, evecs_r = _spectral_eigendecomposition(S_r)
        eigenvalues_path[r, :, :] .= evals_r

        L_r = _tvgdfm_align_eigenvectors(evecs_r[:, 1:q, :], L_prev)
        L_prev = L_r

        K_list_r = _tvgdfm_projector_kernels(L_r, frequencies, M_T)
        chi_grid[r] = _apply_symmetric_filter(X_proc, X_pad, K_list_r)

        # Normalized integrated local spectral eigenvalues for time-varying variance shares
        int_evals = Vector{T}(undef, N)
        for j in 1:N
            acc = zero(T)
            for h in 1:n_freq
                acc += freq_wt[h] * max(evals_r[j, h], zero(T))
            end
            int_evals[j] = acc
        end
        total_eig = max(sum(int_evals), eps(T))
        for j in 1:q
            share_grid[r, j] = int_evals[j] / total_eig
        end
    end

    # Linearly interpolate local projector-filtered components and variance shares to t = 1..T_obs
    idx1, idx2, alpha = _tvgdfm_interp_weights(u_obs, u_grid)
    common = Matrix{T}(undef, T_obs, N)
    variance_share_path = Matrix{T}(undef, T_obs, q)
    @inbounds for t in 1:T_obs
        i1 = idx1[t]
        i2 = idx2[t]
        a = alpha[t]
        om_a = one(T) - a
        chi1 = chi_grid[i1]
        chi2 = chi_grid[i2]
        for col in 1:N
            common[t, col] = om_a * chi1[t, col] + a * chi2[t, col]
        end
        for j in 1:q
            variance_share_path[t, j] = om_a * share_grid[i1, j] + a * share_grid[i2, j]
        end
    end

    if standardize
        common .= common .* σ .+ μ
    end
    idiosyncratic = X_orig .- common

    TimeVaryingGDFM{T}(
        X_orig,
        common,
        idiosyncratic,
        eigenvalues_path,
        frequencies,
        u_grid,
        variance_share_path,
        q,
        n_grid,
        b_T,
        M_T,
        time_kernel,
        lag_kernel,
        standardize,
        vn,
    )
end

@float_fallback estimate_tv_gdfm X

# =============================================================================
# StatsAPI & Accessor Interface
# =============================================================================

"""Predicted common component (`T_obs × N`)."""
StatsAPI.predict(m::TimeVaryingGDFM) = m.common_component

"""Fitted common component (`T_obs × N`)."""
StatsAPI.fitted(m::TimeVaryingGDFM) = m.common_component

"""Idiosyncratic component residuals (`T_obs × N`)."""
StatsAPI.residuals(m::TimeVaryingGDFM) = m.idiosyncratic

"""Number of time-series observations (`T_obs`)."""
StatsAPI.nobs(m::TimeVaryingGDFM) = size(m.X, 1)

"""Number of panel variables (`N`)."""
nvars(m::TimeVaryingGDFM) = size(m.X, 2)

"""Panel variable names."""
varnames(m::TimeVaryingGDFM) = m.varnames

"""Approximate effective degrees of freedom."""
StatsAPI.dof(m::TimeVaryingGDFM) = max(1, m.n_grid * m.q * size(m.X, 2) + size(m.X, 1) * m.q)

"""In-sample R² for each panel variable."""
function StatsAPI.r2(m::TimeVaryingGDFM{T}) where {T}
    N = size(m.X, 2)
    [one(T) - var(@view(m.idiosyncratic[:, i])) / max(var(@view(m.X[:, i])), T(1e-10)) for i in 1:N]
end

"""
    common_variance_share(m::TimeVaryingGDFM) -> Matrix

Return the `T_obs × q` time-varying common-variance share path (`m.variance_share_path`).
"""
common_variance_share(m::TimeVaryingGDFM) = m.variance_share_path

# =============================================================================
# Display & Report
# =============================================================================

function Base.show(io::IO, m::TimeVaryingGDFM{T}) where {T}
    T_obs, N = size(m.X)
    spec = Any[
        "Dynamic factors (q)"  m.q;
        "Variables (N)"        N;
        "Observations (T)"     T_obs;
        "Time grid (n_grid)"   m.n_grid;
        "Time kernel"          string(m.time_kernel);
        "Time bandwidth"       _fmt(m.time_bandwidth; digits=4);
        "Lag kernel"           string(m.lag_kernel);
        "Lag bandwidth"        m.lag_bandwidth;
        "Standardized"         m.standardized ? "Yes" : "No"
    ]
    _pretty_table(io, spec;
        title = "Time-Varying Generalized Dynamic Factor Model (q=$(m.q), n_grid=$(m.n_grid))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    n_show = min(m.q, 5)
    t_mid = max(1, div(T_obs + 1, 2))
    vdata = Matrix{Any}(undef, n_show + 1, 5)
    @inbounds for j in 1:n_show
        col = @view m.variance_share_path[:, j]
        vdata[j, 1] = "Factor $j"
        vdata[j, 2] = _fmt_pct(col[1])
        vdata[j, 3] = _fmt_pct(col[t_mid])
        vdata[j, 4] = _fmt_pct(col[T_obs])
        vdata[j, 5] = _fmt_pct(mean(col))
    end
    tot_path = vec(sum(m.variance_share_path; dims=2))
    vdata[n_show + 1, 1] = "Total (q=$(m.q))"
    vdata[n_show + 1, 2] = _fmt_pct(tot_path[1])
    vdata[n_show + 1, 3] = _fmt_pct(tot_path[t_mid])
    vdata[n_show + 1, 4] = _fmt_pct(tot_path[T_obs])
    vdata[n_show + 1, 5] = _fmt_pct(mean(tot_path))

    _pretty_table(io, vdata;
        title = "Time-Varying Common Variance Share Path",
        column_labels = ["", "Start (u≈0)", "Mid (u≈0.5)", "End (u≈1)", "Mean"],
        alignment = [:l, :r, :r, :r, :r],
    )
end

report(io::IO, m::TimeVaryingGDFM) = (show(io, m); nothing)
report(m::TimeVaryingGDFM) = report(stdout, m)
