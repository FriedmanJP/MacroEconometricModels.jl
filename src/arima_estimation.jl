"""
Estimation functions for AR, MA, ARMA, and ARIMA models.

Implements:
- OLS for AR models
- CSS (Conditional Sum of Squares) for MA/ARMA initialization
- MLE via Kalman filter for exact likelihood
- CSS+MLE: CSS initialization followed by MLE refinement
"""

using LinearAlgebra, Statistics, Distributions
import Optim

# =============================================================================
# Shared Helpers
# =============================================================================

"""
    _pack_arma_params(c, phi, theta; include_intercept=true, log_sigma2=nothing)

Pack ARMA parameters into a single vector for optimization.
"""
function _pack_arma_params(c::T, phi::Vector{T}, theta::Vector{T};
                            include_intercept::Bool=true,
                            log_sigma2::Union{Nothing,T}=nothing) where {T}
    params = T[]
    include_intercept && push!(params, c)
    append!(params, phi)
    append!(params, theta)
    log_sigma2 !== nothing && push!(params, log_sigma2)
    params
end

"""
    _compute_aic_bic(loglik, k, n)

Compute AIC and BIC from log-likelihood, number of parameters k, and sample size n.
"""
_compute_aic_bic(loglik::T, k::Int, n::Int) where {T} =
    (-2 * loglik + 2 * k, -2 * loglik + k * log(T(n)))

"""
    _white_noise_fit(y; include_intercept=true)

Fit a white-noise (p=0, q=0) model. Returns (c, sigma2, loglik, residuals, fitted).
"""
function _white_noise_fit(y::Vector{T}; include_intercept::Bool=true) where {T}
    n = length(y)
    c = include_intercept ? mean(y) : zero(T)
    residuals = y .- c
    sigma2 = var(residuals; corrected=false)
    fitted = fill(c, n)
    loglik = -T(n/2) * log(T(2π)) - T(n/2) * log(sigma2) - T(n/2)
    (c, sigma2, loglik, residuals, fitted)
end

# =============================================================================
# Input Validation
# =============================================================================

"""Validate ARIMA inputs."""
function _validate_arima_inputs(y::Vector{T}, p::Int, d::Int, q::Int) where {T}
    n = length(y)
    n < 10 && throw(ArgumentError("Time series too short (n=$n). Need at least 10 observations."))
    p < 0 && throw(ArgumentError("AR order p must be non-negative, got p=$p"))
    d < 0 && throw(ArgumentError("Integration order d must be non-negative, got d=$d"))
    q < 0 && throw(ArgumentError("MA order q must be non-negative, got q=$q"))
    n - d <= max(p, q) + 2 && throw(ArgumentError(
        "Not enough observations after differencing ($(n-d)) for p=$p, q=$q"))
end

# =============================================================================
# Differencing
# =============================================================================

"""Apply d-order differencing to a time series."""
function _difference(y::Vector{T}, d::Int) where {T<:AbstractFloat}
    d == 0 && return copy(y)
    y_diff = copy(y)
    for _ in 1:d
        y_diff = diff(y_diff)
    end
    y_diff
end

# =============================================================================
# Initial Parameter Estimation
# =============================================================================

"""
    _yule_walker(y, p) -> phi

Estimate AR coefficients via Yule-Walker equations.
"""
function _yule_walker(y::Vector{T}, p::Int) where {T<:AbstractFloat}
    p == 0 && return T[]

    n = length(y)
    y_centered = y .- mean(y)

    # Compute autocorrelations
    gamma = zeros(T, p + 1)
    for k in 0:p
        gamma[k+1] = sum(y_centered[1:n-k] .* y_centered[k+1:n]) / n
    end

    # Build Toeplitz matrix
    R = zeros(T, p, p)
    for i in 1:p, j in 1:p
        R[i, j] = gamma[abs(i - j) + 1]
    end

    # Solve for AR coefficients
    r = gamma[2:p+1]
    try
        phi = R \ r
        # Truncate to ensure stationarity if needed
        return _truncate_to_stationary(phi)
    catch
        # Fallback: return small coefficients
        return fill(T(0.1), p)
    end
end

"""Truncate AR coefficients to ensure stationarity."""
function _truncate_to_stationary(phi::Vector{T}; max_iter::Int=100) where {T<:AbstractFloat}
    _is_stationary(phi) && return phi

    # Scale down coefficients until stationary
    scale = T(0.99)
    phi_new = copy(phi)
    for _ in 1:max_iter
        phi_new .*= scale
        _is_stationary(phi_new) && return phi_new
    end

    # Last resort: return zeros
    return zeros(T, length(phi))
end

"""
    _innovations_algorithm(y, q) -> theta

Estimate MA coefficients via innovations algorithm.
"""
function _innovations_algorithm(y::Vector{T}, q::Int) where {T<:AbstractFloat}
    q == 0 && return T[]

    n = length(y)
    y_centered = y .- mean(y)

    # Compute autocovariances
    gamma = zeros(T, q + 1)
    for k in 0:q
        gamma[k+1] = sum(y_centered[1:n-k] .* y_centered[k+1:n]) / n
    end

    # Innovations algorithm
    theta = zeros(T, q, q)
    v = zeros(T, q + 1)
    v[1] = gamma[1]

    for n_iter in 1:q
        for k in 0:n_iter-1
            sum_term = zero(T)
            for j in 0:k-1
                sum_term += theta[k, k-j] * theta[n_iter, n_iter-j] * v[j+1]
            end
            theta[n_iter, n_iter-k] = (gamma[n_iter-k+1] - sum_term) / v[k+1]
        end
        sum_v = zero(T)
        for j in 0:n_iter-1
            sum_v += theta[n_iter, n_iter-j]^2 * v[j+1]
        end
        v[n_iter+1] = gamma[1] - sum_v
    end

    theta_final = theta[q, 1:q]
    # Truncate to ensure invertibility
    _truncate_to_invertible(theta_final)
end

"""Truncate MA coefficients to ensure invertibility."""
function _truncate_to_invertible(theta::Vector{T}; max_iter::Int=100) where {T<:AbstractFloat}
    _is_invertible(theta) && return theta

    scale = T(0.99)
    theta_new = copy(theta)
    for _ in 1:max_iter
        theta_new .*= scale
        _is_invertible(theta_new) && return theta_new
    end

    return zeros(T, length(theta))
end

# =============================================================================
# CSS Estimation
# =============================================================================

"""
    _css_objective(params, y, p, q; include_intercept=true) -> T

Conditional Sum of Squares objective function.
"""
function _css_objective(params::Vector{T}, y::Vector{T}, p::Int, q::Int; include_intercept::Bool=true) where {T<:AbstractFloat}
    c, phi, theta, _ = _unpack_arma_params(params, p, q; include_intercept=include_intercept)

    # Check stationarity/invertibility
    !_is_stationary(phi) && return T(1e10)
    !_is_invertible(theta) && return T(1e10)

    # Compute residuals
    residuals = _compute_arma_residuals(y, c, phi, theta)

    # Skip initial residuals for conditioning
    m = max(p, q)
    valid_residuals = residuals[m+1:end]

    sum(abs2, valid_residuals)
end

"""
    _estimate_css(y, p, q; include_intercept=true, max_iter=500, tol=1e-8) -> (params, converged, iterations)

Estimate ARMA parameters via Conditional Sum of Squares.
"""
function _estimate_css(y::Vector{T}, p::Int, q::Int; include_intercept::Bool=true, max_iter::Int=500, tol::T=T(1e-8)) where {T<:AbstractFloat}
    # Initial guess
    c_init = include_intercept ? mean(y) : zero(T)
    phi_init = _yule_walker(y, p)
    theta_init = _innovations_algorithm(y, q)

    params_init = _pack_arma_params(c_init, phi_init, theta_init; include_intercept=include_intercept)

    # Edge case: no parameters to optimize (white noise with given mean)
    if p == 0 && q == 0
        residuals = y .- c_init
        sigma2 = var(residuals; corrected=false)
        return params_init, sigma2, true, 0
    end

    # Optimize
    obj = params -> _css_objective(params, y, p, q; include_intercept=include_intercept)
    result = Optim.optimize(obj, params_init, Optim.NelderMead(),
        Optim.Options(iterations=max_iter, g_tol=tol, show_trace=false))

    params_opt = Optim.minimizer(result)
    converged = Optim.converged(result)
    iterations = Optim.iterations(result)

    c, phi, theta, _ = _unpack_arma_params(params_opt, p, q; include_intercept=include_intercept)

    residuals = _compute_arma_residuals(y, c, phi, theta)
    m = max(p, q)
    sigma2 = var(residuals[m+1:end]; corrected=false)

    params_opt, sigma2, converged, iterations
end

# =============================================================================
# MLE Estimation
# =============================================================================

"""
    _estimate_mle(y, p, q; include_intercept=true, init_params=nothing, max_iter=500, tol=1e-8)

Estimate ARMA parameters via Maximum Likelihood using Kalman filter.
"""
function _estimate_mle(y::Vector{T}, p::Int, q::Int; include_intercept::Bool=true,
                       init_params::Union{Nothing,Vector{T}}=nothing,
                       init_sigma2::Union{Nothing,T}=nothing,
                       max_iter::Int=500, tol::T=T(1e-8)) where {T<:AbstractFloat}
    n = length(y)

    # Get initial parameters
    if init_params === nothing
        c_init = include_intercept ? mean(y) : zero(T)
        phi_init = _yule_walker(y, p)
        theta_init = _innovations_algorithm(y, q)
        sigma2_init = var(y; corrected=false)
    else
        c_init, phi_init, theta_init, _ = _unpack_arma_params(init_params, p, q;
                                                               include_intercept=include_intercept)
        sigma2_init = init_sigma2 === nothing ? var(y; corrected=false) : init_sigma2
    end

    params_init = _pack_arma_params(c_init, phi_init, theta_init;
                                     include_intercept=include_intercept,
                                     log_sigma2=log(max(sigma2_init, T(1e-10))))

    # Edge case: white noise
    if p == 0 && q == 0
        c, sigma2, loglik, residuals, fitted = _white_noise_fit(y; include_intercept=include_intercept)
        return c, T[], T[], sigma2, loglik, residuals, fitted, true, 0
    end

    # Optimize using LBFGS with numerical gradients
    obj = params -> _arma_negloglik(params, y, p, q; include_intercept=include_intercept)
    result = Optim.optimize(obj, params_init, Optim.LBFGS(),
        Optim.Options(iterations=max_iter, g_tol=tol, show_trace=false))

    params_opt = Optim.minimizer(result)
    converged = Optim.converged(result)
    iterations = Optim.iterations(result)

    c, phi, theta, sigma2 = _unpack_arma_params(params_opt, p, q;
                                                 include_intercept=include_intercept,
                                                 has_log_sigma2=true)

    # Compute final log-likelihood and residuals
    loglik, residuals, fitted = _kalman_filter_arma(y, c, phi, theta, sigma2)

    c, phi, theta, sigma2, loglik, residuals, fitted, converged, iterations
end

# =============================================================================
# AR Estimation
# =============================================================================

"""
    estimate_ar(y, p; method=:ols, include_intercept=true) -> ARModel

Estimate AR(p) model: yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ

# Arguments
- `y`: Time series vector
- `p`: AR order (must be ≥ 1)
- `method`: Estimation method (:ols or :mle)
- `include_intercept`: Whether to include constant term

# Returns
`ARModel` with estimated coefficients and diagnostics.

# Example
```julia
y = randn(200)
model = estimate_ar(y, 2)
println(model.phi)  # AR coefficients
```
"""
function estimate_ar(y::AbstractVector{T}, p::Int; method::Symbol=:ols, include_intercept::Bool=true) where {T<:AbstractFloat}
    _validate_arima_inputs(y, p, 0, 0)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    if method == :ols
        return _estimate_ar_ols(y_vec, p; include_intercept=include_intercept)
    elseif method == :mle
        c, phi, _, sigma2, loglik, residuals, fitted, converged, iterations =
            _estimate_mle(y_vec, p, 0; include_intercept=include_intercept)

        n_eff = length(residuals)
        k = include_intercept ? p + 2 : p + 1  # intercept, phi, sigma2
        aic, bic = _compute_aic_bic(loglik, k, n_eff)

        return ARModel(y_vec, p, c, phi, sigma2, residuals, fitted, loglik, aic, bic, :mle, converged, iterations)
    else
        throw(ArgumentError("Unknown method: $method. Use :ols or :mle."))
    end
end

estimate_ar(y::AbstractVector, p::Int; kwargs...) = estimate_ar(Float64.(y), p; kwargs...)

"""Estimate AR model via OLS."""
function _estimate_ar_ols(y::Vector{T}, p::Int; include_intercept::Bool=true) where {T<:AbstractFloat}
    n = length(y)
    n_eff = n - p

    # Construct design matrix
    X = zeros(T, n_eff, include_intercept ? p + 1 : p)
    if include_intercept
        X[:, 1] .= one(T)
        for lag in 1:p
            X[:, lag+1] = y[p+1-lag:n-lag]
        end
    else
        for lag in 1:p
            X[:, lag] = y[p+1-lag:n-lag]
        end
    end

    # Response
    y_eff = y[p+1:n]

    # OLS estimation
    XtX = X' * X
    XtX_inv = robust_inv(XtX)
    beta = XtX_inv * (X' * y_eff)

    # Extract coefficients
    c = include_intercept ? beta[1] : zero(T)
    phi = include_intercept ? beta[2:end] : beta

    # Residuals and variance
    fitted = X * beta
    residuals = y_eff - fitted
    sigma2 = sum(abs2, residuals) / (n_eff - length(beta))

    # Log-likelihood (using MLE estimate of sigma2 for consistency)
    sigma2_ml = sum(abs2, residuals) / n_eff
    loglik = -T(n_eff/2) * log(T(2π)) - T(n_eff/2) * log(sigma2_ml) - T(n_eff/2)

    # Information criteria
    k = length(beta) + 1  # +1 for sigma2
    aic, bic = _compute_aic_bic(loglik, k, n_eff)

    ARModel(y, p, c, phi, sigma2, residuals, fitted, loglik, aic, bic, :ols, true, 0)
end

# =============================================================================
# MA Estimation
# =============================================================================

"""
    estimate_ma(y, q; method=:css_mle, include_intercept=true, max_iter=500) -> MAModel

Estimate MA(q) model: yₜ = c + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q

# Arguments
- `y`: Time series vector
- `q`: MA order (must be ≥ 1)
- `method`: Estimation method (:css, :mle, or :css_mle)
- `include_intercept`: Whether to include constant term
- `max_iter`: Maximum optimization iterations

# Returns
`MAModel` with estimated coefficients and diagnostics.

# Example
```julia
y = randn(200)
model = estimate_ma(y, 1)
println(model.theta)  # MA coefficient
```
"""
function estimate_ma(y::AbstractVector{T}, q::Int; method::Symbol=:css_mle, include_intercept::Bool=true, max_iter::Int=500) where {T<:AbstractFloat}
    _validate_arima_inputs(y, 0, 0, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    if method == :css
        params, sigma2, converged, iterations = _estimate_css(y_vec, 0, q; include_intercept=include_intercept, max_iter=max_iter)
        c = include_intercept ? params[1] : zero(T)
        theta = params[include_intercept ? 2 : 1:end]

        residuals = _compute_arma_residuals(y_vec, c, T[], theta)
        fitted = y_vec - residuals

        # Approximate log-likelihood
        m = q
        n_eff = n - m
        loglik = -T(n_eff/2) * log(T(2π)) - T(n_eff/2) * log(sigma2) -
                 sum(abs2, residuals[m+1:end]) / (2 * sigma2)

    elseif method == :mle || method == :css_mle
        # Use CSS for initialization if css_mle
        init_params = nothing
        init_sigma2 = nothing
        if method == :css_mle
            css_params, css_sigma2, _, _ = _estimate_css(y_vec, 0, q; include_intercept=include_intercept)
            init_params = css_params
            init_sigma2 = css_sigma2
        end

        c, _, theta, sigma2, loglik, residuals, fitted, converged, iterations =
            _estimate_mle(y_vec, 0, q; include_intercept=include_intercept,
                         init_params=init_params, init_sigma2=init_sigma2, max_iter=max_iter)
    else
        throw(ArgumentError("Unknown method: $method. Use :css, :mle, or :css_mle."))
    end

    n_eff = length(residuals)
    k = include_intercept ? q + 2 : q + 1
    aic, bic = _compute_aic_bic(loglik, k, n_eff)

    MAModel(y_vec, q, c, theta, sigma2, residuals, fitted, loglik, aic, bic, method, converged, iterations)
end

estimate_ma(y::AbstractVector, q::Int; kwargs...) = estimate_ma(Float64.(y), q; kwargs...)

# =============================================================================
# ARMA Estimation
# =============================================================================

"""
    estimate_arma(y, p, q; method=:css_mle, include_intercept=true, max_iter=500) -> ARMAModel

Estimate ARMA(p,q) model:
yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q

# Arguments
- `y`: Time series vector
- `p`: AR order
- `q`: MA order
- `method`: Estimation method (:css, :mle, or :css_mle)
- `include_intercept`: Whether to include constant term
- `max_iter`: Maximum optimization iterations

# Returns
`ARMAModel` with estimated coefficients and diagnostics.

# Example
```julia
y = randn(200)
model = estimate_arma(y, 1, 1)
println("AR: ", model.phi, " MA: ", model.theta)
```
"""
function estimate_arma(y::AbstractVector{T}, p::Int, q::Int; method::Symbol=:css_mle, include_intercept::Bool=true, max_iter::Int=500) where {T<:AbstractFloat}
    _validate_arima_inputs(y, p, 0, q)
    y_vec = Vector{T}(y)
    n = length(y_vec)

    # Handle special cases
    if p == 0 && q == 0
        c, sigma2, loglik, residuals, fitted = _white_noise_fit(y_vec; include_intercept=include_intercept)
        k = include_intercept ? 2 : 1
        aic, bic = _compute_aic_bic(loglik, k, n)
        return ARMAModel(y_vec, 0, 0, c, T[], T[], sigma2, residuals, fitted, loglik, aic, bic, method, true, 0)
    end

    if method == :css
        params, sigma2, converged, iterations = _estimate_css(y_vec, p, q; include_intercept=include_intercept, max_iter=max_iter)

        c, phi, theta, _ = _unpack_arma_params(params, p, q; include_intercept=include_intercept)

        residuals = _compute_arma_residuals(y_vec, c, phi, theta)
        fitted = y_vec - residuals

        m = max(p, q)
        n_eff = n - m
        loglik = -T(n_eff/2) * log(T(2π)) - T(n_eff/2) * log(sigma2) -
                 sum(abs2, residuals[m+1:end]) / (2 * sigma2)

    elseif method == :mle || method == :css_mle
        init_params = nothing
        init_sigma2 = nothing
        if method == :css_mle
            css_params, css_sigma2, _, _ = _estimate_css(y_vec, p, q; include_intercept=include_intercept)
            init_params = css_params
            init_sigma2 = css_sigma2
        end

        c, phi, theta, sigma2, loglik, residuals, fitted, converged, iterations =
            _estimate_mle(y_vec, p, q; include_intercept=include_intercept,
                         init_params=init_params, init_sigma2=init_sigma2, max_iter=max_iter)
    else
        throw(ArgumentError("Unknown method: $method. Use :css, :mle, or :css_mle."))
    end

    n_eff = length(residuals)
    k = (include_intercept ? 1 : 0) + p + q + 1
    aic, bic = _compute_aic_bic(loglik, k, n_eff)

    ARMAModel(y_vec, p, q, c, phi, theta, sigma2, residuals, fitted, loglik, aic, bic, method, converged, iterations)
end

estimate_arma(y::AbstractVector, p::Int, q::Int; kwargs...) = estimate_arma(Float64.(y), p, q; kwargs...)

# =============================================================================
# ARIMA Estimation
# =============================================================================

"""
    estimate_arima(y, p, d, q; method=:css_mle, include_intercept=true, max_iter=500) -> ARIMAModel

Estimate ARIMA(p,d,q) model by differencing d times and fitting ARMA(p,q).

# Arguments
- `y`: Time series vector
- `p`: AR order
- `d`: Integration order (number of differences)
- `q`: MA order
- `method`: Estimation method (:css, :mle, or :css_mle)
- `include_intercept`: Whether to include constant term (on differenced series)
- `max_iter`: Maximum optimization iterations

# Returns
`ARIMAModel` with estimated coefficients and diagnostics.

# Example
```julia
y = cumsum(randn(200))  # Random walk
model = estimate_arima(y, 1, 1, 0)  # ARIMA(1,1,0)
println(model.phi)
```
"""
function estimate_arima(y::AbstractVector{T}, p::Int, d::Int, q::Int;
                        method::Symbol=:css_mle, include_intercept::Bool=true,
                        max_iter::Int=500) where {T<:AbstractFloat}
    _validate_arima_inputs(y, p, d, q)
    y_vec = Vector{T}(y)

    # Difference the series
    y_diff = _difference(y_vec, d)

    # Fit ARMA to differenced series
    if p == 0 && q == 0
        c, sigma2, loglik, residuals, fitted = _white_noise_fit(y_diff; include_intercept=include_intercept)
        n_eff = length(residuals)
        k = include_intercept ? 2 : 1
        aic, bic = _compute_aic_bic(loglik, k, n_eff)

        return ARIMAModel(y_vec, y_diff, 0, d, 0, c, T[], T[], sigma2, residuals, fitted,
                         loglik, aic, bic, method, true, 0)
    end

    # Fit ARMA model
    arma = estimate_arma(y_diff, p, q; method=method, include_intercept=include_intercept, max_iter=max_iter)

    ARIMAModel(y_vec, y_diff, p, d, q, arma.c, arma.phi, arma.theta, arma.sigma2,
               arma.residuals, arma.fitted, arma.loglik, arma.aic, arma.bic,
               arma.method, arma.converged, arma.iterations)
end

estimate_arima(y::AbstractVector, p::Int, d::Int, q::Int; kwargs...) =
    estimate_arima(Float64.(y), p, d, q; kwargs...)

# =============================================================================
# StatsAPI fit Interface
# =============================================================================

StatsAPI.fit(::Type{ARModel}, y::AbstractVector, p::Int; kwargs...) = estimate_ar(y, p; kwargs...)
StatsAPI.fit(::Type{MAModel}, y::AbstractVector, q::Int; kwargs...) = estimate_ma(y, q; kwargs...)
StatsAPI.fit(::Type{ARMAModel}, y::AbstractVector, p::Int, q::Int; kwargs...) = estimate_arma(y, p, q; kwargs...)
StatsAPI.fit(::Type{ARIMAModel}, y::AbstractVector, p::Int, d::Int, q::Int; kwargs...) = estimate_arima(y, p, d, q; kwargs...)
