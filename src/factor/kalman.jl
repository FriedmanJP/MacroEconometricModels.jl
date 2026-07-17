# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Kalman filter/smoother utilities for dynamic factor models.

These utilities are used by the dynamic factor model estimation.
"""

using LinearAlgebra

# =============================================================================
# Factor Model Forecast Result
# =============================================================================

"""
    FactorForecast{T<:AbstractFloat}

Result of factor model forecasting with optional confidence intervals.

Fields: factors, observables, factors_lower, factors_upper, observables_lower, observables_upper,
factors_se, observables_se, horizon, conf_level, ci_method.

When `ci_method == :none`, CI and SE fields are zero matrices.
"""
struct FactorForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    factors::Matrix{T}            # h × r factor forecasts
    observables::Matrix{T}        # h × N observable forecasts
    factors_lower::Matrix{T}      # h × r lower CI for factors
    factors_upper::Matrix{T}      # h × r upper CI for factors
    observables_lower::Matrix{T}  # h × N lower CI for observables
    observables_upper::Matrix{T}  # h × N upper CI for observables
    factors_se::Matrix{T}         # h × r standard errors for factors
    observables_se::Matrix{T}     # h × N standard errors for observables
    horizon::Int
    conf_level::T
    ci_method::Symbol             # :none, :theoretical, :bootstrap, :simulation
end

# FactorForecast stores observables (not `.forecast`) and non-standard CI field names
point_forecast(f::FactorForecast) = f.observables
lower_bound(f::FactorForecast) = f.observables_lower
upper_bound(f::FactorForecast) = f.observables_upper

function Base.show(io::IO, fc::FactorForecast{T}) where {T}
    h, r = size(fc.factors)
    N = size(fc.observables, 2)
    ci_str = fc.ci_method == :none ? "none" : "$(fc.ci_method) ($(round(100*fc.conf_level, digits=1))%)"
    data = Any[
        "Horizon"     h;
        "Factors"     r;
        "Observables" N;
        "CI method"   ci_str
    ]
    _pretty_table(io, data;
        title = "Factor Forecast",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # Append the observable forecast values (was spec-table only). (S4/T168)
    hs = _select_horizons(h)
    nshow = min(N, 8)
    fdata = Matrix{Any}(undef, nshow, length(hs) + 1)
    for i in 1:nshow
        fdata[i, 1] = "obs$i"
        for (c, hh) in enumerate(hs)
            fdata[i, c+1] = _fmt(fc.observables[hh, i])
        end
    end
    _pretty_table(io, fdata;
        title = "Observable Forecasts" * (N > nshow ? " (first $nshow of $N)" : ""),
        column_labels = vcat([""], ["h=$hh" for hh in hs]),
        alignment = vcat([:l], fill(:r, length(hs))))
end

# =============================================================================
# Shared Utilities
# =============================================================================

"""Standardize matrix: subtract mean, divide by std."""
function _standardize(X::AbstractMatrix{T}) where {T}
    μ, σ = mean(X, dims=1), max.(std(X, dims=1), T(1e-10))
    (X .- μ) ./ σ
end

# =============================================================================
# Kalman Filter/Smoother for Dynamic Factor Model
# =============================================================================

"""
    _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p) -> (a_smooth, P_smooth, Pt_smooth, loglik)

Kalman filter and smoother for state-space form of dynamic factor model.

State-space representation:
- Observation: Y_t = Z * α_t + ε_t, ε_t ~ N(0, H)
- State: α_t = T * α_{t-1} + η_t, η_t ~ N(0, Q)

Where:
- α_t = [F_t', F_{t-1}', ..., F_{t-p+1}']' (stacked factors)
- Z = [Λ, 0, ..., 0] (observation matrix)
- T = companion matrix for factor VAR
- Q = [Sigma_eta, 0; 0, 0] (state innovation covariance)
- H = Sigma_e (observation noise covariance)

Returns smoothed state estimates, covariances, and log-likelihood.
"""
function _kalman_smoother_dfm(Y::AbstractMatrix{T}, Λ::AbstractMatrix{T}, A::Vector{Matrix{T}},
    Sigma_eta::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, r::Int, p::Int
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    # Build state-space matrices
    Z = zeros(T, N, state_dim); Z[:, 1:r] = Λ
    T_mat = zeros(T, state_dim, state_dim)
    for lag in 1:p
        T_mat[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (T_mat[(r+1):state_dim, 1:(state_dim-r)] = I(state_dim - r))

    Q = zeros(T, state_dim, state_dim); Q[1:r, 1:r] = Sigma_eta
    H = Sigma_e

    # Initialize from unconditional distribution
    a0, P0 = zeros(T, state_dim), _compute_unconditional_covariance(T_mat, Q, state_dim)

    # Forward filter (kernel, multivariate) + RTS smoother with lag-1 cross-covariance
    # (T147/#246). The DFM predicts-at-top with a0 = 0 (a_{0|0}); the existing unconditional
    # init is kept, so smoothed states / loglik are byte-stable vs the pre-#246 filter (Joseph
    # replaces the (I-KZ)P shorthand; safe_cholesky + triangular solves replace robust_inv;
    # always-add replaces the det_F>0 gate — all agree on the well-conditioned stationary path).
    # Kernel outputs are time-last [:,:,t]; transpose them back to the EM's time-first [t,:,:]
    # layout at the wrapper boundary so `_em_mstep` is unchanged.
    store = KalmanFilterStore{T}(state_dim, T_obs)
    loglik = _kalman_filter!(store, permutedims(Y), Z, T_mat, Q, Matrix{T}(H);
                             a0=a0, P0=P0, scalar=false)
    a_sm, P_sm, Plag = _rts_smoother(store, T_mat; nlag=1)

    a_smooth = permutedims(a_sm)                    # state_dim×T_obs → T_obs×state_dim
    P_smooth = permutedims(P_sm, (3, 1, 2))         # sd×sd×T_obs → T_obs×sd×sd
    Pt_smooth = zeros(T, T_obs - 1, state_dim, state_dim)
    @inbounds for t in 1:(T_obs-1)
        # OLD Pt_smooth[t] = Cov(α_{t+1}, α_t | Y) = kernel lag-1 Plag[1][:,:,t+1]
        Pt_smooth[t, :, :] = Plag[1][:, :, t+1]
    end

    a_smooth, P_smooth, Pt_smooth, loglik
end

# =============================================================================
# Unconditional Covariance Computation
# =============================================================================

"""
    _compute_unconditional_covariance(T_mat, Q, state_dim; max_iter=1000, tol=1e-10)

Compute unconditional covariance of state vector by solving the discrete Lyapunov equation:
P = T * P * T' + Q

For stationary systems, iterates until convergence. For non-stationary systems,
returns a large diagonal matrix as fallback.
"""
function _compute_unconditional_covariance(T_mat::AbstractMatrix{T}, Q::AbstractMatrix{T},
    state_dim::Int; max_iter::Int=1000, tol::Float64=1e-10
) where {T<:AbstractFloat}
    # Check stationarity — fall back to diffuse initialization for non-stationary systems
    maximum(abs.(eigvals(T_mat))) >= 1.0 && return Matrix{T}(10.0 * I(state_dim))

    # Solve discrete Lyapunov equation via shared utility
    P = _solve_discrete_lyapunov(T_mat, Q; max_iter=max_iter, tol=tol)
    return Matrix{T}(P)
end

# =============================================================================
# Factor Forecast Helpers (shared across Static FM, DFM, GDFM)
# =============================================================================

"""
    _factor_forecast_var_theoretical(A, Sigma_eta, r, p, h) -> Vector{Matrix{T}}

Compute h-step forecast error covariance for factor VAR(p) via VMA(∞) representation.

Returns vector of h covariance matrices (r × r each): MSE_h = Σ_{j=0}^{h-1} Ψ_j Σ_η Ψ_j'.
"""
function _factor_forecast_var_theoretical(A::Vector{<:AbstractMatrix{T}}, Sigma_eta::AbstractMatrix{T},
    r::Int, p::Int, h::Int) where {T<:AbstractFloat}

    state_dim = r * p
    # Build companion matrix
    C = zeros(T, state_dim, state_dim)
    for lag in 1:p
        C[1:r, ((lag-1)*r+1):(lag*r)] = A[lag]
    end
    p > 1 && (C[(r+1):end, 1:(r*(p-1))] = I(r * (p - 1)))

    # Build state-level Q
    Q = zeros(T, state_dim, state_dim)
    Q[1:r, 1:r] = Sigma_eta

    # Selector: first r rows of companion state
    J = zeros(T, r, state_dim)
    J[1:r, 1:r] = I(r)

    # Accumulate MSE via VMA representation
    mse = Vector{Matrix{T}}(undef, h)
    C_power = Matrix{T}(I, state_dim, state_dim)
    cumul = zeros(T, r, r)
    for step in 1:h
        Psi = J * C_power
        cumul += Psi * Q * Psi'
        mse[step] = copy(cumul)
        C_power = C_power * C
    end
    mse
end

"""
    _factor_forecast_obs_se(factor_mse, Lambda, Sigma_e, h) -> Matrix{T}

Compute observable forecast standard errors.

Var(X_{T+h} error) = Λ * MSE_factor_h * Λ' + Σ_e. Returns h × N matrix of SEs.
"""
function _factor_forecast_obs_se(factor_mse::Vector{Matrix{T}}, Lambda::AbstractMatrix{T},
    Sigma_e::AbstractMatrix{T}, h::Int) where {T<:AbstractFloat}

    N = size(Lambda, 1)
    obs_se = Matrix{T}(undef, h, N)
    for step in 1:h
        obs_var = Lambda * factor_mse[step] * Lambda' + Sigma_e
        obs_se[step, :] = sqrt.(max.(diag(obs_var), zero(T)))
    end
    obs_se
end

"""
    _factor_forecast_bootstrap(F_last, A, resids, Sigma_e, Lambda, h, r, p, n_boot, conf_level) -> tuple

Residual bootstrap for factor forecast CIs. Resamples factor VAR residuals,
simulates factor paths, projects to observables, computes percentile CIs.

Returns (f_lo, f_hi, o_lo, o_hi, f_se, o_se).
"""
function _factor_forecast_bootstrap(F_last::Vector{Vector{T}}, A::Vector{<:AbstractMatrix{T}},
    resids::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, Lambda::AbstractMatrix{T},
    h::Int, r::Int, p::Int, n_boot::Int, conf_level::T,
    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}

    N = size(Lambda, 1)
    T_resid = size(resids, 1)
    L_e = safe_cholesky(Sigma_e)

    F_boot = zeros(T, n_boot, h, r)
    X_boot = zeros(T, n_boot, h, N)

    for b in 1:n_boot
        for step in 1:h
            # VAR forecast with resampled innovation
            F_h = sum(A[lag] * (step - lag >= 1 ? F_boot[b, step - lag, :] : F_last[lag - step + 1]) for lag in 1:p)
            boot_idx = rand(rng, 1:T_resid)
            F_boot[b, step, :] = F_h + resids[boot_idx, :]
            X_boot[b, step, :] = Lambda * F_boot[b, step, :] + L_e * randn(rng, T, N)
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

    (f_lo, f_hi, o_lo, o_hi, f_se, o_se)
end

"""
    _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, X_original)

In-place unstandardization of observable forecasts using mean/std of original data.
"""
function _unstandardize_factor_forecast!(X_fc::Matrix{T}, X_lo::Matrix{T}, X_hi::Matrix{T},
    X_se::Matrix{T}, X_original::AbstractMatrix{T}) where {T<:AbstractFloat}

    μ = vec(mean(X_original, dims=1))
    σ = max.(vec(std(X_original, dims=1)), T(1e-10))
    X_fc .= X_fc .* σ' .+ μ'
    X_lo .= X_lo .* σ' .+ μ'
    X_hi .= X_hi .* σ' .+ μ'
    X_se .= X_se .* σ'
    nothing
end

"""Unstandardize only the point forecast in place (μ, σ from `X_original`). Used by the
`ci_method=:none` path so the zero bound/se buffers are never fed through the affine unstan-
dardizer — passing one shared zero array as lower/upper/se corrupted it into nonzero garbage."""
function _unstandardize_point!(X_fc::Matrix{T}, X_original::AbstractMatrix{T}) where {T<:AbstractFloat}
    μ = vec(mean(X_original, dims=1))
    σ = max.(vec(std(X_original, dims=1)), T(1e-10))
    X_fc .= X_fc .* σ' .+ μ'
    nothing
end

"""
    _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_level, ci_method) -> FactorForecast{T}

Construct a FactorForecast from components.
"""
function _build_factor_forecast(F_fc::Matrix{T}, X_fc::Matrix{T},
    F_lo::Matrix{T}, F_hi::Matrix{T}, X_lo::Matrix{T}, X_hi::Matrix{T},
    F_se::Matrix{T}, X_se::Matrix{T}, h::Int, conf_level::T, ci_method::Symbol) where {T<:AbstractFloat}

    FactorForecast{T}(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_level, ci_method)
end
