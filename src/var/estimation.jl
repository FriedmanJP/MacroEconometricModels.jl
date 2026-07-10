# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
VAR estimation via OLS with StatsAPI interface.
"""

using LinearAlgebra, Statistics, DataFrames, Distributions

# =============================================================================
# Core Estimation
# =============================================================================

"""
    estimate_var(Y::AbstractMatrix{T}, p::Int; check_stability::Bool=true) -> VARModel{T}

Estimate VAR(p) via OLS: Yₜ = c + A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + uₜ.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags
- `check_stability`: If true (default), warns if estimated VAR is non-stationary

# Returns
`VARModel` with estimated coefficients, residuals, covariance matrix, and information criteria.
"""
function estimate_var(Y::AbstractMatrix{T}, p::Int; check_stability::Bool=true, varnames::Union{Vector{String},Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(Y, "Y")
    T_obs, n = size(Y)
    validate_var_inputs(T_obs, n, p)

    Y_eff, X = construct_var_matrices(Y, p)
    T_eff, k = size(Y_eff, 1), size(X, 2)

    # OLS: B = (X'X)⁻¹X'Y
    B = robust_inv(X'X) * (X' * Y_eff)
    U = Y_eff - X * B

    # Residual covariance, ML estimate (denominator T_eff). This is the right choice for the
    # information criteria and the Gaussian log-likelihood. Coefficient standard errors instead
    # use the small-sample dof-adjusted covariance U'U/(T_eff−k) inside `vcov` (audit F-01:
    # the previous `max(T_eff-k, T_eff)` could never select T_eff−k, so SEs were too small).
    Sigma = (U'U) / T_eff

    # Information criteria (ML estimate)
    Sigma_ml = (U'U) / T_eff
    log_det = logdet_safe(Sigma_ml)
    aic = log_det + 2k / T_eff
    bic = log_det + k * log(T_eff) / T_eff
    hqic = log_det + 2k * log(log(T_eff)) / T_eff

    vn = something(varnames, ["y$i" for i in 1:n])
    model = VARModel(Y, p, B, U, Sigma, aic, bic, hqic, vn)

    # Check stationarity via companion matrix eigenvalues
    if check_stability
        F = companion_matrix(B, n, p)
        max_modulus = maximum(abs.(eigvals(F)))
        if max_modulus >= one(T)
            @warn "Estimated VAR is non-stationary (max eigenvalue modulus = $(round(max_modulus, digits=4))). " *
                  "Consider differencing the data or using a VECM specification."
        end
    end

    model
end

@float_fallback estimate_var Y

"""Estimate VAR from DataFrame. Use `vars` to select columns."""
function estimate_var(df::DataFrame, p::Int; vars::Vector{Symbol}=Symbol[], check_stability::Bool=true)
    data = isempty(vars) ? Matrix(df) : Matrix(df[:, vars])
    estimate_var(Float64.(data), p; check_stability=check_stability)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.fit(::Type{VARModel}, Y::AbstractMatrix, p::Int) = estimate_var(Y, p)
StatsAPI.fit(::Type{VARModel}, df::DataFrame, p::Int; vars::Vector{Symbol}=Symbol[]) =
    estimate_var(df, p; vars=vars)

StatsAPI.coef(model::VARModel) = model.B
StatsAPI.residuals(model::VARModel) = model.U
StatsAPI.dof(model::VARModel) = length(model.B)
StatsAPI.dof_residual(model::VARModel) = size(model.U, 1) - size(model.B, 1)
StatsAPI.nobs(model::VARModel) = size(model.Y, 1)
StatsAPI.aic(model::VARModel) = model.aic
StatsAPI.bic(model::VARModel) = model.bic
StatsAPI.islinear(::VARModel) = true

"""Covariance of vectorized coefficients: Σ̂_dof ⊗ (X'X)⁻¹ with the dof-adjusted Σ̂_dof = U'U/(T_eff−k)."""
function StatsAPI.vcov(model::VARModel{T}) where {T}
    _, X = construct_var_matrices(model.Y, model.p)
    T_eff, k = size(X, 1), size(X, 2)
    dof = T_eff > k ? T_eff - k : T_eff          # small-sample adjustment for coefficient SEs
    Sigma_dof = (model.U' * model.U) / T(dof)
    kron(Sigma_dof, robust_inv(X'X))
end

"""In-sample fitted values."""
StatsAPI.predict(model::VARModel) = @view(model.Y[(model.p+1):end, :]) - model.U

"""Out-of-sample forecasts for `steps` periods."""
function StatsAPI.predict(model::VARModel{T}, steps::Int) where {T}
    steps < 1 && throw(ArgumentError("steps must be positive"))

    n, p, B = nvars(model), model.p, model.B
    forecasts = Matrix{T}(undef, steps, n)
    intercept = @view B[1, :]
    A = extract_ar_coefficients(B, n, p)
    history = copy(model.Y[(end-p+1):end, :])

    @inbounds for h in 1:steps
        y_hat = copy(intercept)
        for lag in 1:p
            y_hat .+= A[lag] * @view(history[end-lag+1, :])
        end
        forecasts[h, :] = y_hat
        history = vcat(@view(history[2:end, :]), y_hat')
    end
    forecasts
end

"""R² for each equation."""
function StatsAPI.r2(model::VARModel{T}) where {T}
    Y_eff = @view model.Y[(model.p+1):end, :]
    [1 - var(@view(model.U[:, i])) / var(@view(Y_eff[:, i])) for i in 1:nvars(model)]
end

"""Gaussian log-likelihood."""
function StatsAPI.loglikelihood(model::VARModel{T}) where {T}
    n, T_eff = nvars(model), effective_nobs(model)
    -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * logdet_safe(model.Sigma) - T(T_eff * n / 2)
end

StatsAPI.stderror(model::VARModel) = sqrt.(diag(vcov(model)))

"""Confidence intervals at given level (default 95%)."""
function StatsAPI.confint(model::VARModel{T}; level::Real=0.95) where {T}
    B_vec, se = vec(model.B), stderror(model)
    crit = T(quantile(TDist(dof_residual(model)), 1 - (1 - level) / 2))
    hcat(B_vec .- crit .* se, B_vec .+ crit .* se)
end

# =============================================================================
# Model Selection
# =============================================================================

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::VARModel, h; ci_method=:bootstrap, reps=500, conf_level=0.95,
             stationary_only=false, rng=Random.default_rng()) -> VARForecast{T}

Forecast from VAR model for `h` steps ahead with optional confidence intervals.

The point forecast iterates the VAR recursion forward using the estimated
coefficients and the last `p` observations. Confidence bands are formed by one
of:

- `:bootstrap` (default) — **Kilian (1998) bootstrap-B**. Each replication draws
  a pseudo-sample `Y*` by resampling the residuals and iterating the VAR with the
  point estimate, **re-estimates** the coefficients on `Y*`, and then simulates
  the future path from the *true* last-`p` observations using the re-estimated
  coefficients plus fresh resampled future shocks. This propagates both future
  innovation uncertainty and coefficient-estimation uncertainty, so the bands are
  wider (and closer to nominal coverage in small samples) than a residual-only
  bootstrap. Cost: `reps` VAR re-estimations.
- `:analytic` — Lütkepohl (2005, §3.5) known-coefficient forecast MSE
  `Σ_y(h) = Σ_{i=0}^{h-1} Φ_i Σ_u Φ_i'` with symmetric Gaussian bands
  `point ± z·√diag`. This ignores coefficient-estimation uncertainty (the leading
  asymptotic term); use `:bootstrap` for finite-sample parameter uncertainty.
- `:none` — point forecast only (zero-width bands).

# Arguments
- `model`: Estimated VAR model
- `h`: Forecast horizon (number of steps ahead)
- `ci_method`: `:bootstrap` (default), `:analytic`, or `:none`
- `reps`: Number of bootstrap replications (default 500)
- `conf_level`: Confidence level for intervals (default 0.95)
- `stationary_only`: reject bootstrap draws whose re-estimated companion matrix is
  non-stationary (default `false`)
- `rng`: random number generator for the bootstrap (default `Random.default_rng()`)

# Returns
`VARForecast{T}` with point forecasts and CIs.

# Example
```julia
model = estimate_var(Y, 4)
fc = forecast(model, 12)  # 12-step ahead forecast with bootstrap-B CIs
```
"""
function forecast(model::VARModel{T}, h::Int;
                  ci_method::Symbol=:bootstrap,
                  reps::Int=500,
                  conf_level::Real=0.95,
                  stationary_only::Bool=false,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be positive"))
    ci_method ∈ (:none, :bootstrap, :analytic) ||
        throw(ArgumentError("ci_method must be :none, :bootstrap, or :analytic"))

    n = nvars(model)
    p = model.p
    point = predict(model, h)

    ci_lower = zeros(T, h, n)
    ci_upper = zeros(T, h, n)

    if ci_method == :bootstrap
        T_eff = effective_nobs(model)
        Y_init = model.Y[1:p, :]                     # first p obs seed the pseudo-sample
        last_p = model.Y[(end - p + 1):end, :]       # true origin for the forward path
        sim = Array{T,3}(undef, reps, h, n)

        rep = 0
        attempts = 0
        max_attempts = stationary_only ? 20 * reps : reps
        while rep < reps && attempts < max_attempts
            attempts += 1
            # Bootstrap-B: rebuild a pseudo-sample from the point estimate, re-estimate.
            U_boot = model.U[rand(rng, 1:T_eff, T_eff), :]
            Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
            m_star = estimate_var(Y_boot, p; check_stability=false)
            if stationary_only
                max_mod = maximum(abs.(eigvals(companion_matrix(m_star.B, n, p))))
                max_mod >= one(T) && continue        # discard explosive re-estimate
            end
            # Forward path from the TRUE last-p obs with the re-estimated coefficients.
            future = model.U[rand(rng, 1:T_eff, h), :]
            path = _simulate_var(last_p, m_star.B, future, p + h)
            rep += 1
            @inbounds sim[rep, :, :] = @view path[(p + 1):(p + h), :]
        end
        if rep < reps
            @warn "Only $rep/$reps $(stationary_only ? "stationary " : "")bootstrap forecast draws obtained after $attempts attempts"
            sim = sim[1:max(rep, 1), :, :]
        end

        alpha_half = (1 - T(conf_level)) / 2
        for hi in 1:h, j in 1:n
            d = @view sim[:, hi, j]
            ci_lower[hi, j] = quantile(d, alpha_half)
            ci_upper[hi, j] = quantile(d, 1 - alpha_half)
        end

    elseif ci_method == :analytic
        # Lütkepohl §3.5 known-coefficient forecast MSE via the MA(∞) coefficients
        # Φ_0 = I, Φ_i = Σ_{j=1}^{min(i,p)} Φ_{i-j} A_j (eq. 2.1.22).
        A = extract_ar_coefficients(model.B, n, p)
        Phi = Vector{Matrix{T}}(undef, h)            # Phi[i+1] = Φ_i, i = 0 … h-1
        Phi[1] = Matrix{T}(I, n, n)
        for i in 1:(h - 1)
            acc = zeros(T, n, n)
            for j in 1:min(i, p)
                acc .+= Phi[i - j + 1] * A[j]
            end
            Phi[i + 1] = acc
        end
        z = T(quantile(Normal(), 1 - (1 - T(conf_level)) / 2))
        mse = zeros(T, n, n)
        for hi in 1:h
            mse .+= Phi[hi] * model.Sigma * Phi[hi]'
            for j in 1:n
                se = sqrt(max(mse[j, j], zero(T)))
                ci_lower[hi, j] = point[hi, j] - z * se
                ci_upper[hi, j] = point[hi, j] + z * se
            end
        end
    end

    VARForecast{T}(point, ci_lower, ci_upper, h, ci_method, T(conf_level), model.varnames)
end

"""Select optimal lag order via information criterion (:aic, :bic, :hqic)."""
function select_lag_order(Y::AbstractMatrix{T}, max_p::Int; criterion::Symbol=:bic) where {T<:AbstractFloat}
    max_p < 1 && throw(ArgumentError("max_p must be positive"))
    size(Y, 1) <= max_p + 2 && throw(ArgumentError("Not enough observations"))

    ic = map(1:max_p) do p
        m = estimate_var(Y, p)
        criterion == :aic ? m.aic : criterion == :bic ? m.bic :
        criterion == :hqic ? m.hqic : throw(ArgumentError("Unknown criterion: $criterion"))
    end
    argmin(ic)
end
