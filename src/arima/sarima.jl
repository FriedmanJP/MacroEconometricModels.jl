# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Multiplicative seasonal ARIMA — SARIMA(p,d,q)(P,D,Q)ₛ (T242 / #341).

```math
\\Phi_P(L^s)\\,\\phi_p(L)\\,(1-L)^d (1-L^s)^D y_t = \\Theta_Q(L^s)\\,\\theta_q(L)\\,\\varepsilon_t
```

The multiplicative structure is handled by **expanding** the seasonal and non-seasonal
polynomials into a single long ARMA, which the existing Kalman likelihood
(`src/arima/kalman.jl`) then evaluates unchanged. Only the parameterization is seasonal;
the filter is not re-implemented.

Provides:
- `SARIMAModel` — result container carrying both the seasonal and expanded coefficients
- `estimate_sarima(y, p, d, q, P, D, Q, s)` — CSS / MLE / CSS→MLE estimation
- `auto_sarima(y, s; ...)` — Hyndman–Khandakar-style seasonal order search
- `forecast(::SARIMAModel, h)` — forecasts un-differenced through both operators, with
  ψ-weight prediction intervals from the fully expanded non-differenced operator

References:
- Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (2008). *Time Series Analysis:
  Forecasting and Control*, 4th ed. Wiley. Ch. 9 (seasonal models, airline model).
- Hyndman, R. J. & Khandakar, Y. (2008). Automatic Time Series Forecasting: The
  forecast Package for R. *Journal of Statistical Software*, 27(3).
"""

using LinearAlgebra, Statistics

# =============================================================================
# Seasonal differencing and polynomial expansion
# =============================================================================

"""
    _seasonal_difference(y, D, s) -> Vector

Apply `(1 - Lˢ)^D` to `y`, dropping `D·s` leading observations.
"""
function _seasonal_difference(y::AbstractVector{T}, D::Int, s::Int) where {T<:AbstractFloat}
    D == 0 && return Vector{T}(y)
    s >= 1 || throw(ArgumentError("seasonal period s must be ≥ 1 when D > 0, got $s"))
    out = Vector{T}(y)
    for _ in 1:D
        length(out) > s || throw(ArgumentError(
            "series too short for seasonal differencing (D=$D, s=$s)"))
        out = out[(s+1):end] .- out[1:(end-s)]
    end
    return out
end

"""
    _sarima_difference(y, d, D, s) -> Vector

Apply the full differencing operator `(1-L)^d (1-Lˢ)^D`. Regular differencing runs first;
the two operators commute, so the order is immaterial for the result.
"""
_sarima_difference(y::AbstractVector{T}, d::Int, D::Int, s::Int) where {T<:AbstractFloat} =
    _seasonal_difference(_difference(Vector{T}(y), d), D, s)

"""
    _sarima_diff_poly(d, D, s, ::Type{T}) -> Vector{T}

Coefficients of `(1-L)^d (1-Lˢ)^D` as a full polynomial `[1, δ₁, …, δ_{d+Ds}]`.
"""
function _sarima_diff_poly(d::Int, D::Int, s::Int, ::Type{T}) where {T<:AbstractFloat}
    reg = T[T(binomial(d, k) * (-1)^k) for k in 0:d]
    seas = zeros(T, D * s + 1)
    for k in 0:D
        seas[k*s+1] = T(binomial(D, k) * (-1)^k)
    end
    return _arfima_polymul(reg, seas)
end

"""
    _expand_ar(phi, Phi, s) -> Vector

Expand the multiplicative AR operator `φ_p(L)·Φ_P(Lˢ)` into a single coefficient vector in
the package's recursion convention (`yₜ = Σ φᵢ yₜ₋ᵢ + …`, characteristic polynomial
`1 - Σ φᵢ Lⁱ`). Returns a vector of length `p + P·s`.
"""
function _expand_ar(phi::AbstractVector{T}, Phi::AbstractVector{T}, s::Int) where {T<:AbstractFloat}
    isempty(Phi) && return Vector{T}(phi)
    p, P = length(phi), length(Phi)
    a = zeros(T, p + 1); a[1] = one(T)
    for i in 1:p
        a[i+1] = -phi[i]
    end
    A = zeros(T, P * s + 1); A[1] = one(T)
    for k in 1:P
        A[k*s+1] = -Phi[k]
    end
    prod = _arfima_polymul(a, A)
    return T[-prod[j] for j in 2:length(prod)]
end

"""
    _expand_ma(theta, Theta, s) -> Vector

Expand the multiplicative MA operator `θ_q(L)·Θ_Q(Lˢ)` into a single coefficient vector in
the package's convention (`yₜ = … + εₜ + Σ θⱼ εₜ₋ⱼ`, polynomial `1 + Σ θⱼ Lʲ`).
Returns a vector of length `q + Q·s`.
"""
function _expand_ma(theta::AbstractVector{T}, Theta::AbstractVector{T}, s::Int) where {T<:AbstractFloat}
    isempty(Theta) && return Vector{T}(theta)
    q, Q = length(theta), length(Theta)
    b = zeros(T, q + 1); b[1] = one(T)
    for j in 1:q
        b[j+1] = theta[j]
    end
    B = zeros(T, Q * s + 1); B[1] = one(T)
    for k in 1:Q
        B[k*s+1] = Theta[k]
    end
    prod = _arfima_polymul(b, B)
    return prod[2:end]
end

# =============================================================================
# Parameter packing for the seasonal optimizer
# =============================================================================

"""
    _pack_sarima(c, phi, theta, Phi, Theta; include_intercept, log_sigma2=nothing)

Parameter layout `[c, φ, θ, Φ, Θ, (log σ²)]`. The intercept and `log σ²` slots are
present only when requested, matching `_pack_arma_params`.
"""
function _pack_sarima(c::T, phi::Vector{T}, theta::Vector{T}, Phi::Vector{T},
                      Theta::Vector{T}; include_intercept::Bool=true,
                      log_sigma2::Union{Nothing,T}=nothing) where {T<:AbstractFloat}
    params = T[]
    include_intercept && push!(params, c)
    append!(params, phi); append!(params, theta)
    append!(params, Phi); append!(params, Theta)
    log_sigma2 === nothing || push!(params, log_sigma2)
    return params
end

"""
    _unpack_sarima(params, p, q, P, Q; include_intercept, has_log_sigma2)
        -> (c, phi, theta, Phi, Theta, sigma2_or_nothing)
"""
function _unpack_sarima(params::Vector{T}, p::Int, q::Int, P::Int, Q::Int;
                        include_intercept::Bool=true,
                        has_log_sigma2::Bool=false) where {T}
    idx = 1
    c = zero(T)
    if include_intercept
        c = params[idx]; idx += 1
    end
    phi = p > 0 ? params[idx:idx+p-1] : T[]; idx += p
    theta = q > 0 ? params[idx:idx+q-1] : T[]; idx += q
    Phi = P > 0 ? params[idx:idx+P-1] : T[]; idx += P
    Theta = Q > 0 ? params[idx:idx+Q-1] : T[]; idx += Q
    sigma2 = has_log_sigma2 ? exp(params[idx]) : nothing
    return (c, phi, theta, Phi, Theta, sigma2)
end

"""
    _sarima_ok(phi, theta, Phi, Theta) -> Bool

Stationarity of both AR factors and invertibility of both MA factors. Checking the factors
is equivalent to checking the expanded polynomials (the roots of a product are the union of
the factors' roots) and is much better conditioned at high expanded order.
"""
_sarima_ok(phi, theta, Phi, Theta) =
    _is_stationary(phi) && _is_stationary(Phi) &&
    _is_invertible(theta) && _is_invertible(Theta)

"""
    _sarima_negloglik(params, y, p, q, P, Q, s; include_intercept=true)

Negative Kalman log-likelihood of the expanded ARMA at the seasonal parameterization.
Returns a large penalty outside the stationary/invertible region.
"""
function _sarima_negloglik(params::Vector{T}, y::Vector{T}, p::Int, q::Int,
                           P::Int, Q::Int, s::Int;
                           include_intercept::Bool=true) where {T<:AbstractFloat}
    c, phi, theta, Phi, Theta, sigma2 = _unpack_sarima(params, p, q, P, Q;
                                                       include_intercept=include_intercept,
                                                       has_log_sigma2=true)
    penalty = T(1e10)
    _sarima_ok(phi, theta, Phi, Theta) || return penalty
    loglik = _arma_loglik(y, c, _expand_ar(phi, Phi, s), _expand_ma(theta, Theta, s), sigma2)
    (isnan(loglik) || isinf(loglik)) && return penalty
    return -loglik
end

"""
    _sarima_css_objective(params, y, p, q, P, Q, s; include_intercept=true)

Conditional sum of squares on the expanded ARMA, skipping the first `max(p+Ps, q+Qs)`
residuals (which depend on unavailable pre-sample values).
"""
function _sarima_css_objective(params::Vector{T}, y::Vector{T}, p::Int, q::Int,
                               P::Int, Q::Int, s::Int;
                               include_intercept::Bool=true) where {T<:AbstractFloat}
    c, phi, theta, Phi, Theta, _ = _unpack_sarima(params, p, q, P, Q;
                                                  include_intercept=include_intercept)
    penalty = T(1e10)
    _sarima_ok(phi, theta, Phi, Theta) || return penalty
    phi_e = _expand_ar(phi, Phi, s)
    theta_e = _expand_ma(theta, Theta, s)
    resid = _compute_arma_residuals(y, c, phi_e, theta_e)
    m = max(length(phi_e), length(theta_e))
    m >= length(resid) && return penalty
    ss = sum(abs2, view(resid, m+1:length(resid)))
    return (isnan(ss) || isinf(ss)) ? penalty : ss
end

# =============================================================================
# Estimation
# =============================================================================

"""
    _estimate_sarima_internal(y, p, q, P, Q, s; method, include_intercept, max_iter)

Estimate the seasonal ARMA parameters on an already-differenced series. Returns
`(c, phi, theta, Phi, Theta, sigma2, loglik, residuals, fitted, converged, iterations)`.
"""
function _estimate_sarima_internal(y::Vector{T}, p::Int, q::Int, P::Int, Q::Int, s::Int;
                                   method::Symbol=:css_mle,
                                   include_intercept::Bool=true,
                                   max_iter::Int=500, tol::T=T(1e-8)) where {T<:AbstractFloat}
    method in (:css, :mle, :css_mle) ||
        throw(ArgumentError("Unknown method: $method. Use :css, :mle, or :css_mle."))

    n = length(y)
    c_init = include_intercept ? mean(y) : zero(T)
    # Start seasonal terms at zero and non-seasonal terms at the usual moment estimates;
    # a zero seasonal start is inside the stationary/invertible region by construction.
    phi_init = _yule_walker(y, p)
    theta_init = _innovations_algorithm(y, q)
    Phi_init = zeros(T, P); Theta_init = zeros(T, Q)

    if p == 0 && q == 0 && P == 0 && Q == 0
        c, sigma2, loglik, residuals, fitted = _white_noise_fit(y; include_intercept=include_intercept)
        return c, T[], T[], T[], T[], sigma2, loglik, residuals, fitted, true, 0
    end

    converged = true
    iterations = 0

    # ── CSS stage ────────────────────────────────────────────────────────
    if method in (:css, :css_mle)
        params0 = _pack_sarima(c_init, phi_init, theta_init, Phi_init, Theta_init;
                               include_intercept=include_intercept)
        obj = params -> _sarima_css_objective(params, y, p, q, P, Q, s;
                                              include_intercept=include_intercept)
        res = Optim.optimize(obj, params0, Optim.NelderMead(),
                             Optim.Options(iterations=max_iter, g_tol=tol, show_trace=false))
        popt = Optim.minimizer(res)
        converged = Optim.converged(res)
        iterations = Optim.iterations(res)
        c_init, phi_init, theta_init, Phi_init, Theta_init, _ =
            _unpack_sarima(popt, p, q, P, Q; include_intercept=include_intercept)
    end

    if method === :css
        phi_e = _expand_ar(phi_init, Phi_init, s)
        theta_e = _expand_ma(theta_init, Theta_init, s)
        residuals = _compute_arma_residuals(y, c_init, phi_e, theta_e)
        fitted = y .- residuals
        m = max(length(phi_e), length(theta_e))
        sigma2 = var(view(residuals, m+1:n); corrected=false)
        n_eff = n - m
        loglik = -T(n_eff / 2) * log(T(2π)) - T(n_eff / 2) * log(sigma2) -
                 sum(abs2, view(residuals, m+1:n)) / (2 * sigma2)
        return c_init, phi_init, theta_init, Phi_init, Theta_init, sigma2, loglik,
               residuals, fitted, converged, iterations
    end

    # ── MLE stage ────────────────────────────────────────────────────────
    sigma2_init = if method === :css_mle
        phi_e = _expand_ar(phi_init, Phi_init, s)
        theta_e = _expand_ma(theta_init, Theta_init, s)
        r0 = _compute_arma_residuals(y, c_init, phi_e, theta_e)
        m0 = max(length(phi_e), length(theta_e))
        m0 < n ? var(view(r0, m0+1:n); corrected=false) : var(y; corrected=false)
    else
        var(y; corrected=false)
    end

    params0 = _pack_sarima(c_init, phi_init, theta_init, Phi_init, Theta_init;
                           include_intercept=include_intercept,
                           log_sigma2=log(max(sigma2_init, T(1e-10))))
    obj = params -> _sarima_negloglik(params, y, p, q, P, Q, s;
                                      include_intercept=include_intercept)
    res = Optim.optimize(obj, params0, Optim.LBFGS(),
                         Optim.Options(iterations=max_iter, g_tol=tol, show_trace=false))
    popt = Optim.minimizer(res)
    converged = Optim.converged(res)
    iterations = Optim.iterations(res)

    c, phi, theta, Phi, Theta, sigma2 = _unpack_sarima(popt, p, q, P, Q;
                                                       include_intercept=include_intercept,
                                                       has_log_sigma2=true)
    phi_e = _expand_ar(phi, Phi, s)
    theta_e = _expand_ma(theta, Theta, s)
    loglik, residuals, fitted = _kalman_filter_arma(y, c, phi_e, theta_e, sigma2)

    # The state-space MLE parameterizes c as the process MEAN μ; the stored model and the
    # forecast recursion use the AR INTERCEPT. Convert with the EXPANDED AR sum (#121/T022).
    c = c * (one(T) - sum(phi_e))

    return c, phi, theta, Phi, Theta, sigma2, loglik, residuals, fitted, converged, iterations
end

"""
    estimate_sarima(y, p, d, q, P, D, Q, s; method=:css_mle, include_intercept=true,
                    max_iter=500) -> SARIMAModel

Estimate a multiplicative seasonal ARIMA model

```math
\\Phi_P(L^s)\\,\\phi_p(L)\\,(1-L)^d (1-L^s)^D y_t = \\Theta_Q(L^s)\\,\\theta_q(L)\\,\\varepsilon_t.
```

The series is differenced `d` times regularly and `D` times seasonally, then the seasonal
and non-seasonal polynomials are expanded into one long ARMA whose likelihood is evaluated
by the existing Kalman filter. Estimation therefore has the same three methods as
`estimate_arima`: `:css` (conditional sum of squares), `:mle` (exact Gaussian likelihood),
and `:css_mle` (CSS start, MLE refinement — the default).

With `P = D = Q = 0` this reduces exactly to `estimate_arima(y, p, d, q)`.

# Arguments
- `y` — time series vector
- `p`, `d`, `q` — non-seasonal AR order, differencing order, MA order
- `P`, `D`, `Q` — seasonal AR order, seasonal differencing order, seasonal MA order
- `s` — seasonal period (12 monthly, 4 quarterly)

# Keywords
- `method::Symbol=:css_mle` — `:css`, `:mle`, or `:css_mle`
- `include_intercept::Bool=true` — include a constant on the differenced series
- `max_iter::Int=500` — optimizer iteration cap

# Returns
[`SARIMAModel`](@ref).

# Example
```julia
# Airline model (0,1,1)(0,1,1)₁₂
m = estimate_sarima(y, 0, 1, 1, 0, 1, 1, 12)
```

# References
- Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (2008). *Time Series Analysis:
  Forecasting and Control*, 4th ed. Wiley.
"""
function estimate_sarima(y::AbstractVector{T}, p::Int, d::Int, q::Int,
                         P::Int, D::Int, Q::Int, s::Int;
                         method::Symbol=:css_mle, include_intercept::Bool=true,
                         max_iter::Int=500) where {T<:AbstractFloat}
    _validate_data(y, "y")
    all(>=(0), (p, d, q, P, D, Q)) ||
        throw(ArgumentError("orders p,d,q,P,D,Q must be non-negative"))
    if P > 0 || D > 0 || Q > 0
        s >= 2 || throw(ArgumentError(
            "seasonal period s must be ≥ 2 when any seasonal order is positive, got $s"))
    end
    s >= 0 || throw(ArgumentError("seasonal period s must be non-negative, got $s"))

    y_vec = Vector{T}(y)
    y_diff = _sarima_difference(y_vec, d, D, s)
    n_diff = length(y_diff)
    n_par = p + q + P + Q + (include_intercept ? 1 : 0) + 1
    n_diff > n_par || throw(ArgumentError(
        "differenced series has $n_diff observations but the model has $n_par parameters; " *
        "supply a longer series or reduce the orders"))

    c, phi, theta, Phi, Theta, sigma2, loglik, residuals, fitted, converged, iterations =
        _estimate_sarima_internal(y_diff, p, q, P, Q, s; method=method,
                                  include_intercept=include_intercept, max_iter=max_iter)

    phi_e = _expand_ar(phi, Phi, s)
    theta_e = _expand_ma(theta, Theta, s)
    k = p + q + P + Q + (include_intercept ? 1 : 0) + 1     # +1 for σ²
    N_ic = method === :css ? length(residuals) - max(length(phi_e), length(theta_e)) :
                             length(residuals)
    aic, bic = _compute_aic_bic(loglik, k, N_ic)

    SARIMAModel(y_vec, y_diff, p, d, q, P, D, Q, s, c, phi, theta, Phi, Theta,
                phi_e, theta_e, sigma2, residuals, fitted, loglik, aic, bic,
                method, converged, iterations)
end

estimate_sarima(y::AbstractVector, p::Int, d::Int, q::Int, P::Int, D::Int, Q::Int, s::Int;
                kwargs...) =
    estimate_sarima(Float64.(y), p, d, q, P, D, Q, s; kwargs...)

StatsAPI.fit(::Type{SARIMAModel}, y::AbstractVector, p::Int, d::Int, q::Int,
             P::Int, D::Int, Q::Int, s::Int; kwargs...) =
    estimate_sarima(y, p, d, q, P, D, Q, s; kwargs...)

# =============================================================================
# Standard errors
# =============================================================================

"""
    _sarima_mle_stderror(m::SARIMAModel) -> Vector

MLE standard errors from the numerical Hessian of the seasonal negative log-likelihood,
matching `coef(m) = [c, φ, θ, Φ, Θ]`.
"""
function _sarima_mle_stderror(m::SARIMAModel)
    T_f = eltype(m.y)
    n_coef = 1 + m.p + m.q + m.P + m.Q
    params = _pack_sarima(m.c, m.phi, m.theta, m.Phi, m.Theta;
                          include_intercept=true,
                          log_sigma2=log(max(m.sigma2, T_f(1e-10))))
    obj = par -> _sarima_negloglik(par, m.y_diff, m.p, m.q, m.P, m.Q, m.s;
                                   include_intercept=true)
    H = _numerical_hessian(obj, params)
    try
        C = robust_inv(H[1:n_coef, 1:n_coef])
        return sqrt.(max.(diag(C), zero(T_f)))
    catch
        return fill(T_f(NaN), n_coef)
    end
end

StatsAPI.stderror(m::SARIMAModel) = _sarima_mle_stderror(m)

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(model::SARIMAModel, h; conf_level=0.95) -> ARIMAForecast

Forecast a SARIMA model `h` steps ahead.

The expanded ARMA is projected on the differenced scale, then un-differenced through the
full operator `(1-L)^d (1-Lˢ)^D` by the recursion `yₜ = wₜ − Σᵢ δᵢ yₜ₋ᵢ`, where the `δᵢ`
are that operator's coefficients. Prediction intervals use the ψ-weights of the **fully
expanded non-differenced** AR operator `φ(L)Φ(Lˢ)(1-L)^d(1-Lˢ)^D`, so the bands widen with
the horizon at the rate the integrated seasonal process implies rather than at the
stationary ARMA rate.
"""
function forecast(model::SARIMAModel{T}, h::Int; conf_level::Real=0.95) where {T<:AbstractFloat}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive"))

    fc_diff = _forecast_arma(model.y_diff, model.residuals, model.c,
                             model.phi_expanded, model.theta_expanded,
                             model.sigma2, h, conf_level)
    (model.d == 0 && model.D == 0) && return fc_diff

    delta = _sarima_diff_poly(model.d, model.D, model.s, T)   # [1, δ₁, …]
    forecasts = _undifference(model.y, fc_diff.forecast, delta)

    phi_star = _expand_ar_with_diff_poly(model.phi_expanded, delta)
    psi = _compute_psi_weights(phi_star, model.theta_expanded, h)
    se = sqrt.(_forecast_variance(model.sigma2, psi, h))
    ci_lower, ci_upper = _confidence_band(forecasts, se, conf_level)

    ARIMAForecast(forecasts, ci_lower, ci_upper, se, h, conf_level)
end

"""
    _undifference(y, fc_diff, delta) -> Vector

Invert a differencing operator with full coefficient vector `delta = [1, δ₁, …, δ_m]`:
`yₜ = wₜ − Σᵢ δᵢ yₜ₋ᵢ`, seeded by the observed tail of `y`. Handles regular and seasonal
differencing (and their product) in one pass.
"""
function _undifference(y::AbstractVector{T}, fc_diff::AbstractVector{T},
                       delta::AbstractVector{T}) where {T<:AbstractFloat}
    m = length(delta) - 1
    h = length(fc_diff)
    n = length(y)
    m <= n || throw(ArgumentError(
        "cannot undifference: the operator needs $m past values but only $n are available"))
    ext = vcat(Vector{T}(y), zeros(T, h))
    @inbounds for j in 1:h
        acc = fc_diff[j]
        for i in 1:m
            acc -= delta[i+1] * ext[n+j-i]
        end
        ext[n+j] = acc
    end
    return ext[(n+1):(n+h)]
end

"""
    _expand_ar_with_diff_poly(phi, delta) -> Vector

Multiply the (recursion-convention) AR coefficients by a differencing polynomial given as
its full coefficient vector `delta`, returning recursion-convention coefficients for the
non-differenced operator. This is the seasonal generalization of
`_expand_ar_with_differencing`.
"""
function _expand_ar_with_diff_poly(phi::AbstractVector{T},
                                   delta::AbstractVector{T}) where {T<:AbstractFloat}
    p = length(phi)
    a = zeros(T, p + 1); a[1] = one(T)
    for i in 1:p
        a[i+1] = -phi[i]
    end
    prod = _arfima_polymul(a, Vector{T}(delta))
    return T[-prod[j] for j in 2:length(prod)]
end

# =============================================================================
# Automatic seasonal order selection
# =============================================================================

"""
    auto_sarima(y, s; d=nothing, D=nothing, max_p=2, max_q=2, max_P=1, max_Q=1,
                criterion=:aic, method=:css_mle, include_intercept=true) -> SARIMAModel

Search seasonal and non-seasonal orders for a SARIMA model at period `s`, returning the
best-fitting [`SARIMAModel`](@ref) by `criterion` (`:aic` or `:bic`).

Differencing orders are chosen first and held fixed while the ARMA orders are searched —
the Hyndman & Khandakar (2008) convention, since information criteria are not comparable
across different differencing orders.

`D` is selected by the [`hegy_test`](@ref) seasonal unit-root test (`D = 1` when a
seasonal unit root cannot be rejected at 5%), which requires `s ∈ {4, 12}` and at least
`3s + 5` observations; outside that range `D` defaults to `0` and should be supplied
explicitly. `d` is then selected by [`kpss_test`](@ref) on the seasonally differenced
series (`d = 1` when level stationarity is rejected at 5%). Both are capped at 1.

The ARMA order search runs with `method=:css` for speed — the Hyndman–Khandakar
convention — and the winning orders are refit with `method` before being returned.

# Keywords
- `d`, `D` — fix the differencing orders instead of selecting them
- `max_p`, `max_q`, `max_P`, `max_Q` — search bounds on the ARMA orders
- `criterion::Symbol=:aic` — `:aic` or `:bic`
- `method`, `include_intercept` — used for the final refit and passed to
  [`estimate_sarima`](@ref)

# References
- Hyndman, R. J. & Khandakar, Y. (2008). Automatic Time Series Forecasting.
  *Journal of Statistical Software*, 27(3).
"""
function auto_sarima(y::AbstractVector{T}, s::Int;
                     d::Union{Nothing,Int}=nothing, D::Union{Nothing,Int}=nothing,
                     max_p::Int=2, max_q::Int=2, max_P::Int=1, max_Q::Int=1,
                     criterion::Symbol=:aic, method::Symbol=:css_mle,
                     include_intercept::Bool=true) where {T<:AbstractFloat}
    criterion in (:aic, :bic) ||
        throw(ArgumentError("criterion must be :aic or :bic, got :$criterion"))
    s >= 1 || throw(ArgumentError("seasonal period s must be ≥ 1, got $s"))
    y_vec = Vector{T}(y)

    D_use = D === nothing ? _auto_seasonal_diff(y_vec, s) : D
    d_use = d === nothing ? _auto_regular_diff(_seasonal_difference(y_vec, D_use, s)) : d

    best_orders = nothing
    best_ic = T(Inf)
    for pp in 0:max_p, qq in 0:max_q, PP in 0:max_P, QQ in 0:max_Q
        model = try
            estimate_sarima(y_vec, pp, d_use, qq, PP, D_use, QQ, s;
                            method=:css, include_intercept=include_intercept)
        catch err
            # Order combinations that are infeasible for this sample length are skipped;
            # anything else is a genuine failure and must surface.
            err isa ArgumentError && continue
            rethrow(err)
        end
        ic = criterion === :aic ? model.aic : model.bic
        if isfinite(ic) && ic < best_ic
            best_ic = ic
            best_orders = (pp, qq, PP, QQ)
        end
    end
    best_orders === nothing && throw(ErrorException(
        "auto_sarima: no candidate model could be estimated on this series"))

    pp, qq, PP, QQ = best_orders
    return estimate_sarima(y_vec, pp, d_use, qq, PP, D_use, QQ, s;
                           method=method, include_intercept=include_intercept)
end

auto_sarima(y::AbstractVector, s::Int; kwargs...) = auto_sarima(Float64.(y), s; kwargs...)

"""
    _auto_seasonal_diff(y, s) -> Int

Seasonal differencing order (0 or 1) from the HEGY seasonal unit-root test: `1` when a
seasonal unit root cannot be rejected at 5% — the Nyquist ``t`` fails to reject
(``t > cv``) or any harmonic-pair ``F`` fails to reject (``F < cv``). Returns `0` when the
test does not apply (`s ∉ {4, 12}` or fewer than `3s + 5` observations).
"""
function _auto_seasonal_diff(y::Vector{T}, s::Int) where {T<:AbstractFloat}
    (s in (4, 12) && length(y) >= 3 * s + 5) || return 0
    r = hegy_test(y; frequency=s)
    seasonal_root = r.t_nyquist > r.t_nyquist_cv[5] || any(r.pair_F .< r.pair_F_cv[5])
    return seasonal_root ? 1 : 0
end

"""
    _auto_regular_diff(y) -> Int

Regular differencing order (0 or 1) from the KPSS level-stationarity test: `1` when level
stationarity is rejected at 5% (statistic above 0.463). Returns `0` when the series is too
short for the test (fewer than 10 observations).
"""
function _auto_regular_diff(y::Vector{T}) where {T<:AbstractFloat}
    length(y) >= 10 || return 0
    return kpss_test(y; regression=:constant).statistic > T(0.463) ? 1 : 0
end
