# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
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
Dynamic Factor Model with State-Space Representation.

Implements Stock & Watson (2002) dynamic factor model:
- Observation: X_t = Λ F_t + e_t
- State: F_t = Σ_i A_i F_{t-i} + η_t

Estimation methods:
- Two-step: PCA followed by VAR on factors
- EM algorithm: Maximum likelihood via Expectation-Maximization

References:
- Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a
  large number of predictors. Journal of the American Statistical Association.
"""

using LinearAlgebra, Statistics, StatsAPI
using Distributions: Normal, quantile

# =============================================================================
# Dynamic Factor Model Type
# =============================================================================

"""
    DynamicFactorModel{T} <: AbstractFactorModel

Dynamic factor model: Xₜ = Λ Fₜ + eₜ, Fₜ = Σᵢ Aᵢ Fₜ₋ᵢ + ηₜ.

Fields: X, factors, loadings, A (VAR coefficients), factor_residuals, Sigma_eta, Sigma_e,
eigenvalues, explained_variance, cumulative_variance, r, p, method, standardized,
converged, iterations, loglik.
"""
struct DynamicFactorModel{T<:AbstractFloat} <: AbstractFactorModel
    X::Matrix{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    A::Vector{Matrix{T}}
    factor_residuals::Matrix{T}
    Sigma_eta::Matrix{T}
    Sigma_e::Matrix{T}
    eigenvalues::Vector{T}
    explained_variance::Vector{T}
    cumulative_variance::Vector{T}
    r::Int
    p::Int
    method::Symbol
    standardized::Bool
    converged::Bool
    iterations::Int
    loglik::T
end

# =============================================================================
# Dynamic Factor Model Estimation
# =============================================================================

"""
    estimate_dynamic_factors(X, r, p; method=:twostep, standardize=true, max_iter=100, tol=1e-6)

Estimate dynamic factor model with VAR(p) factor dynamics.

# Arguments
- `X`: Data matrix (T × N)
- `r`: Number of factors
- `p`: Number of lags in factor VAR

# Keyword Arguments
- `method::Symbol=:twostep`: Estimation method (:twostep or :em)
- `standardize::Bool=true`: Standardize data
- `max_iter::Int=100`: Maximum EM iterations (if method=:em)
- `tol::Float64=1e-6`: Convergence tolerance (if method=:em)
- `diagonal_idio::Bool=true`: Assume diagonal idiosyncratic covariance

# Returns
`DynamicFactorModel` with factors, loadings, VAR coefficients, and diagnostics.

# Example
```julia
dfm = estimate_dynamic_factors(X, 3, 2)  # 3 factors, VAR(2)
forecast(dfm, 12)  # 12-step ahead forecast
```
"""
function estimate_dynamic_factors(X::AbstractMatrix{T}, r::Int, p::Int;
    method::Symbol=:twostep, standardize::Bool=true, max_iter::Int=100,
    tol::Float64=1e-6, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    validate_dynamic_factor_inputs(T_obs, N, r, p)
    validate_option(method, "method", (:twostep, :em))

    method == :twostep ?
        _estimate_dfm_twostep(X, r, p; standardize, diagonal_idio) :
        _estimate_dfm_em(X, r, p; standardize, max_iter, tol, diagonal_idio)
end

@float_fallback estimate_dynamic_factors X

function Base.show(io::IO, m::DynamicFactorModel{T}) where {T}
    Tobs, N = size(m.X)
    spec = Any[
        "Factors"        m.r;
        "Lags"           m.p;
        "Variables"      N;
        "Observations"   Tobs;
        "Method"         string(m.method);
        "Log-likelihood" _fmt(m.loglik; digits=4);
        "Converged"      m.converged ? "Yes" : "No";
        "Iterations"     m.iterations
    ]
    _pretty_table(io, spec;
        title = "Dynamic Factor Model (r=$(m.r), p=$(m.p))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    # Variance explained
    n_show = min(m.r, 5)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(m.explained_variance[i])
        var_data[i, 3] = _fmt_pct(m.cumulative_variance[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )
end

# =============================================================================
# Two-Step Estimation
# =============================================================================

"""Two-step estimation: PCA for factors, then VAR on extracted factors."""
function _estimate_dfm_twostep(X::AbstractMatrix{T}, r::Int, p::Int;
    standardize::Bool=true, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    static = estimate_factors(X, r; standardize)
    F, Λ = static.factors, static.loadings

    # Estimate VAR on factors
    var_model = estimate_var(F, p)
    A = [var_model.B[(2+(lag-1)*r):(1+lag*r), :]' for lag in 1:p]
    factor_residuals, Sigma_eta = var_model.U, var_model.Sigma

    # Idiosyncratic covariance
    X_proc = standardize ? _standardize(X) : X
    e = X_proc - F * Λ'
    Sigma_e = diagonal_idio ? diagm(vec(var(e, dims=1))) : (e'e) / T_obs

    loglik = _compute_dfm_loglikelihood(X, F, Λ, Sigma_e, standardize)

    DynamicFactorModel{T}(copy(X), F, Λ, A, factor_residuals, Sigma_eta, Sigma_e,
        static.eigenvalues, static.explained_variance, static.cumulative_variance,
        r, p, :twostep, standardize, true, 1, loglik)
end

# =============================================================================
# EM Algorithm Estimation
# =============================================================================

"""EM algorithm for maximum likelihood estimation."""
function _estimate_dfm_em(X::AbstractMatrix{T}, r::Int, p::Int;
    standardize::Bool=true, max_iter::Int=100, tol::Float64=1e-6, diagonal_idio::Bool=true
) where {T<:AbstractFloat}

    T_obs, N = size(X)
    Y = standardize ? _standardize(X) : copy(X)

    # Initialize from two-step
    init = _estimate_dfm_twostep(X, r, p; standardize, diagonal_idio)
    Λ, A, Sigma_eta, Sigma_e = copy(init.loadings), deepcopy(init.A), copy(init.Sigma_eta), copy(init.Sigma_e)

    prev_loglik, converged, iter = -Inf, false, 0

    for iteration in 1:max_iter
        iter = iteration
        # E-step: Kalman smoother
        F_smooth, P_smooth, Pt_smooth, loglik = _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p)

        # Check convergence
        abs(loglik - prev_loglik) < tol * abs(prev_loglik) && iteration > 1 && (converged = true; break)
        prev_loglik = loglik

        # M-step: Update parameters
        Λ, A, Sigma_eta, Sigma_e = _em_mstep(Y, F_smooth, P_smooth, Pt_smooth, r, p, diagonal_idio)
    end

    # Final smoother pass
    F_smooth, P_smooth, _, loglik = _kalman_smoother_dfm(Y, Λ, A, Sigma_eta, Sigma_e, r, p)
    F = F_smooth[:, 1:r]

    # Compute factor residuals
    T_eff = T_obs - p
    factor_residuals = zeros(T, T_eff, r)
    for t in (p+1):T_obs
        pred = sum(A[lag] * F[t-lag, :] for lag in 1:p)
        factor_residuals[t-p, :] = F[t, :] - pred
    end

    # Eigenvalues from loadings
    λ = sort(eigvals(Symmetric(Λ'Λ)), rev=true)
    full_λ = zeros(T, N); full_λ[1:r] = λ
    expl = λ / sum(λ)
    full_expl = zeros(T, N); full_expl[1:r] = expl
    cumul = cumsum(expl)
    full_cumul = zeros(T, N); full_cumul[1:r] = cumul; full_cumul[(r+1):end] .= 1.0

    DynamicFactorModel{T}(copy(X), F, Λ, A, factor_residuals, Sigma_eta, Sigma_e,
        full_λ, full_expl, full_cumul, r, p, :em, standardize, converged, iter, loglik)
end

# =============================================================================
# EM M-Step
# =============================================================================

"""M-step: update loadings, VAR coefficients, and covariances."""
function _em_mstep(Y::AbstractMatrix{T}, F_smooth::AbstractMatrix{T}, P_smooth::AbstractArray{T,3},
    Pt_smooth::AbstractArray{T,3}, r::Int, p::Int, diagonal_idio::Bool
) where {T<:AbstractFloat}

    T_obs, N = size(Y)
    state_dim = r * p

    # Update loadings: Λ = (Σ Y_t F_t') * (Σ F_t F_t' + P_t)^{-1}
    sum_yF = sum(@view(Y[t, :]) * @view(F_smooth[t, 1:r])' for t in 1:T_obs)
    sum_FF = sum(@view(F_smooth[t, 1:r]) * @view(F_smooth[t, 1:r])' + @view(P_smooth[t, 1:r, 1:r]) for t in 1:T_obs)
    Λ = sum_yF * robust_inv(sum_FF)

    # Update VAR coefficients
    sum_F_Fminus = zeros(T, r, state_dim)
    sum_Fminus_Fminus = zeros(T, state_dim, state_dim)
    for t in (p+1):T_obs
        sum_F_Fminus .+= @view(F_smooth[t, 1:r]) * @view(F_smooth[t-1, :])'
        t > p + 1 && (sum_F_Fminus[1:r, 1:state_dim] .+= @view(Pt_smooth[t-1, 1:r, :]))
        sum_Fminus_Fminus .+= @view(F_smooth[t-1, :]) * @view(F_smooth[t-1, :])' + @view(P_smooth[t-1, :, :])
    end
    A_stacked = sum_F_Fminus * robust_inv(sum_Fminus_Fminus)
    A = [A_stacked[:, ((lag-1)*r+1):(lag*r)] for lag in 1:p]

    # Update state innovation covariance
    T_eff = T_obs - p
    sum_eta = zeros(T, r, r)
    for t in (p+1):T_obs
        eta = F_smooth[t, 1:r] - sum(A[lag] * @view(F_smooth[t-lag, 1:r]) for lag in 1:p)
        sum_eta .+= eta * eta' + @view(P_smooth[t, 1:r, 1:r])
        for lag in 1:p
            cross = A[lag] * @view(Pt_smooth[min(t-1, T_obs-1), 1:r, 1:r])'
            sum_eta .-= cross .+ cross'
        end
    end
    Sigma_eta = (sum_eta + sum_eta') / (2 * T_eff)
    min_eig = minimum(eigvals(Symmetric(Sigma_eta)))
    min_eig < 1e-8 && (Sigma_eta += (1e-8 - min_eig) * I(r))

    # Update idiosyncratic covariance
    sum_ee = zeros(T, N, N)
    for t in 1:T_obs
        e = @view(Y[t, :]) - Λ * @view(F_smooth[t, 1:r])
        sum_ee .+= e * e' + Λ * @view(P_smooth[t, 1:r, 1:r]) * Λ'
    end
    Sigma_e = diagonal_idio ? diagm(diag(sum_ee / T_obs)) : (sum_ee + sum_ee') / (2 * T_obs)
    min_eig = minimum(eigvals(Symmetric(Sigma_e)))
    min_eig < 1e-8 && (Sigma_e += (1e-8 - min_eig) * I(N))

    Λ, A, Sigma_eta, Sigma_e
end

# =============================================================================
# Log-Likelihood
# =============================================================================

"""Compute Gaussian log-likelihood given factors and parameters."""
function _compute_dfm_loglikelihood(X::AbstractMatrix{T}, F::AbstractMatrix{T},
    Λ::AbstractMatrix{T}, Sigma_e::AbstractMatrix{T}, standardize::Bool
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    Y = standardize ? _standardize(X) : X
    e = Y - F * Λ'

    Sigma_sym = Symmetric(Sigma_e)
    Sigma_inv = try inv(Sigma_sym) catch; pinv(Sigma_sym) end

    ll = -0.5 * T_obs * N * log(2π) - 0.5 * T_obs * logdet(Sigma_sym)
    ll - 0.5 * sum(e[t, :]' * Sigma_inv * e[t, :] for t in 1:T_obs)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

"""Predicted values: F * Λ'."""
StatsAPI.predict(m::DynamicFactorModel) = m.factors * m.loadings'

"""Residuals: X - predicted."""
function StatsAPI.residuals(m::DynamicFactorModel{T}) where {T}
    fitted = predict(m)
    m.standardized ? _standardize(m.X) - fitted : m.X - fitted
end

"""R² for each variable."""
function StatsAPI.r2(m::DynamicFactorModel{T}) where {T}
    resid = residuals(m)
    X_ref = m.standardized ? _standardize(m.X) : m.X
    [max(zero(T), 1 - var(@view(resid[:, i])) / max(var(@view(X_ref[:, i])), T(1e-10)))
     for i in 1:size(m.X, 2)]
end

"""Number of observations."""
StatsAPI.nobs(m::DynamicFactorModel) = size(m.X, 1)

"""Log-likelihood of the fitted model."""
StatsAPI.loglikelihood(m::DynamicFactorModel) = m.loglik

"""Degrees of freedom: N*r + r²*p + r(r+1)/2 + N."""
function StatsAPI.dof(m::DynamicFactorModel)
    _, N = size(m.X)
    N * m.r + m.r^2 * m.p + div(m.r * (m.r + 1), 2) + N
end

"""Akaike Information Criterion."""
StatsAPI.aic(m::DynamicFactorModel) = -2m.loglik + 2dof(m)

"""Bayesian Information Criterion."""
StatsAPI.bic(m::DynamicFactorModel) = -2m.loglik + dof(m) * log(nobs(m))

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::DynamicFactorModel, h; ci_method=:none, conf_level=0.95, n_boot=1000, ci=false, ci_level=0.95)

Forecast factors and observables h steps ahead.

# Arguments
- `model`: Estimated dynamic factor model
- `h`: Forecast horizon

# Keyword Arguments
- `ci_method::Symbol=:none`: CI method — `:none`, `:theoretical`, `:bootstrap`, or `:simulation`
- `conf_level::Real=0.95`: Confidence level for intervals
- `n_boot::Int=1000`: Bootstrap replications (for `:bootstrap` and `:simulation`)
- `ci::Bool=false`: Legacy keyword — `ci=true` maps to `ci_method=:simulation`
- `ci_level::Real=0.95`: Legacy keyword — maps to `conf_level`

# Returns
`FactorForecast` with factor and observable forecasts (and CIs if requested).

# Example
```julia
fc = forecast(dfm, 12; ci_method=:theoretical)
fc.observables       # h×N matrix of forecasts
fc.observables_lower # h×N lower CI bounds
```
"""
function forecast(m::DynamicFactorModel{T}, h::Int; ci_method::Symbol=:none,
    conf_level::Real=0.95, n_boot::Int=1000,
    ci::Bool=false) where {T}

    h < 1 && throw(ArgumentError("h must be ≥ 1"))

    # Legacy compat: ci=true maps to :simulation
    if ci && ci_method == :none
        ci_method = :simulation
    end
    ci_method ∈ (:none, :theoretical, :bootstrap, :simulation) || throw(ArgumentError("ci_method must be :none, :theoretical, :bootstrap, or :simulation"))

    r, p, T_obs, N = m.r, m.p, size(m.X, 1), size(m.X, 2)
    F_last = [m.factors[T_obs-lag+1, :] for lag in 1:p]

    # Point forecasts
    F_fc, X_fc = zeros(T, h, r), zeros(T, h, N)
    for step in 1:h
        F_h = sum(m.A[lag] * (step - lag >= 1 ? F_fc[step - lag, :] : F_last[lag - step + 1]) for lag in 1:p)
        F_fc[step, :] = F_h
        X_fc[step, :] = m.loadings * F_h
    end

    conf_T = T(conf_level)

    if ci_method == :none
        z = zeros(T, h, r)
        zx = zeros(T, h, N)
        if m.standardized
            _unstandardize_factor_forecast!(X_fc, zx, zx, zx, m.X)
        end
        return _build_factor_forecast(F_fc, X_fc, z, z, zx, copy(zx), z, copy(zx), h, conf_T, :none)
    end

    if ci_method == :theoretical
        factor_mse = _factor_forecast_var_theoretical(m.A, m.Sigma_eta, r, p, h)
        z_val = T(quantile(Normal(), 1 - (1 - conf_level) / 2))

        F_se = Matrix{T}(undef, h, r)
        for step in 1:h
            F_se[step, :] = sqrt.(max.(diag(factor_mse[step]), zero(T)))
        end
        F_lo = F_fc .- z_val .* F_se
        F_hi = F_fc .+ z_val .* F_se

        X_se = _factor_forecast_obs_se(factor_mse, m.loadings, m.Sigma_e, h)
        X_lo = X_fc .- z_val .* X_se
        X_hi = X_fc .+ z_val .* X_se

        if m.standardized
            _unstandardize_factor_forecast!(X_fc, X_lo, X_hi, X_se, m.X)
        end
        return _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :theoretical)
    end

    if ci_method == :bootstrap
        factor_resids = m.factor_residuals
        Sigma_e = m.Sigma_e
        f_lo, f_hi, o_lo, o_hi, f_se, o_se = _factor_forecast_bootstrap(
            F_last, m.A, factor_resids, Sigma_e, m.loadings, h, r, p, n_boot, conf_T)

        if m.standardized
            _unstandardize_factor_forecast!(X_fc, o_lo, o_hi, o_se, m.X)
        end
        return _build_factor_forecast(F_fc, X_fc, f_lo, f_hi, o_lo, o_hi, f_se, o_se, h, conf_T, :bootstrap)
    end

    # :simulation — original Monte Carlo method
    n_sim = n_boot
    L_eta = safe_cholesky(m.Sigma_eta)
    L_e = safe_cholesky(m.Sigma_e)

    F_sims, X_sims = zeros(T, n_sim, h, r), zeros(T, n_sim, h, N)
    for sim in 1:n_sim
        for step in 1:h
            F_h = sum(m.A[lag] * (step - lag >= 1 ? F_sims[sim, step - lag, :] : F_last[lag - step + 1]) for lag in 1:p)
            F_sims[sim, step, :] = F_h + L_eta * randn(T, r)
            X_sims[sim, step, :] = m.loadings * F_sims[sim, step, :] + L_e * randn(T, N)
        end
    end

    if m.standardized
        μ, σ = vec(mean(m.X, dims=1)), max.(vec(std(m.X, dims=1)), T(1e-10))
        X_fc .= X_fc .* σ' .+ μ'
        for sim in 1:n_sim
            X_sims[sim, :, :] = X_sims[sim, :, :] .* σ' .+ μ'
        end
    end

    α_lo = (1 - conf_level) / 2
    α_hi = 1 - α_lo
    F_lo = T[quantile(F_sims[:, hh, j], α_lo) for hh in 1:h, j in 1:r]
    F_hi = T[quantile(F_sims[:, hh, j], α_hi) for hh in 1:h, j in 1:r]
    X_lo = T[quantile(X_sims[:, hh, j], α_lo) for hh in 1:h, j in 1:N]
    X_hi = T[quantile(X_sims[:, hh, j], α_hi) for hh in 1:h, j in 1:N]
    F_se = T[std(F_sims[:, hh, j]) for hh in 1:h, j in 1:r]
    X_se = T[std(X_sims[:, hh, j]) for hh in 1:h, j in 1:N]

    _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se, h, conf_T, :simulation)
end

# =============================================================================
# Model Selection
# =============================================================================

"""
    ic_criteria_dynamic(X, max_r, max_p; standardize=true, method=:twostep)

Select (r, p) via AIC/BIC grid search over factor and lag combinations.

# Returns
Named tuple with AIC/BIC matrices and optimal (r, p) combinations.
"""
function ic_criteria_dynamic(X::AbstractMatrix{T}, max_r::Int, max_p::Int;
    standardize::Bool=true, method::Symbol=:twostep
) where {T<:AbstractFloat}
    T_obs, N = size(X)
    1 <= max_r <= min(T_obs, N) || throw(ArgumentError("Invalid max_r"))
    1 <= max_p < T_obs - max_r || throw(ArgumentError("Invalid max_p"))

    AIC_mat, BIC_mat = fill(T(Inf), max_r, max_p), fill(T(Inf), max_r, max_p)
    for r in 1:max_r, p in 1:max_p
        p >= T_obs - r - 10 && continue
        try
            m = estimate_dynamic_factors(X, r, p; method, standardize)
            AIC_mat[r, p], BIC_mat[r, p] = aic(m), bic(m)
        catch; continue; end
    end

    aic_idx, bic_idx = argmin(AIC_mat), argmin(BIC_mat)
    (AIC=AIC_mat, BIC=BIC_mat, r_AIC=aic_idx[1], p_AIC=aic_idx[2], r_BIC=bic_idx[1], p_BIC=bic_idx[2])
end

# =============================================================================
# Companion Matrix and Stationarity
# =============================================================================

"""
    companion_matrix_factors(model::DynamicFactorModel)

Construct companion matrix for factor VAR dynamics.

The companion form converts VAR(p) to VAR(1):
[F_t; F_{t-1}; ...] = C [F_{t-1}; F_{t-2}; ...] + [η_t; 0; ...]
"""
function companion_matrix_factors(m::DynamicFactorModel{T}) where {T}
    r, p = m.r, m.p
    C = zeros(T, r * p, r * p)
    for lag in 1:p
        C[1:r, ((lag-1)*r+1):(lag*r)] = m.A[lag]
    end
    p > 1 && (C[(r+1):end, 1:(r*(p-1))] = I(r * (p - 1)))
    C
end

"""
    is_stationary(model::DynamicFactorModel) -> Bool

Check if factor dynamics are stationary (max |eigenvalue| < 1).
"""
is_stationary(m::DynamicFactorModel) = maximum(abs.(eigvals(companion_matrix_factors(m)))) < 1.0
