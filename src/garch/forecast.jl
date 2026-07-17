# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Variance forecasting for GARCH, EGARCH, and GJR-GARCH models.
"""

# =============================================================================
# GARCH Forecasting
# =============================================================================

"""
    forecast(m::GARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from a GARCH(p,q) model.

Uses analytical iteration for point forecasts and simulation for CIs.
Point forecasts converge to unconditional variance as h → ∞.
"""
function forecast(m::GARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    q = m.q
    p = m.p
    n = length(m.y)

    # Initialize from the tail of the fitted model
    last_eps_sq = m.residuals[end-max(q,p)+1:end] .^ 2
    last_h = m.conditional_variance[end-max(q,p)+1:end]

    # Simulation
    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        eps_sq_buf = copy(last_eps_sq)
        h_buf = copy(last_h)
        for t in 1:h
            ht = m.omega
            for i in 1:q
                idx = length(eps_sq_buf) - i + 1
                ht += m.alpha[i] * (idx >= 1 ? eps_sq_buf[idx] : mean(last_eps_sq))
            end
            for j in 1:p
                idx = length(h_buf) - j + 1
                ht += m.beta[j] * (idx >= 1 ? h_buf[idx] : mean(last_h))
            end
            ht = max(ht, eps(T))
            z = randn(rng, T)
            new_eps_sq = ht * z^2
            push!(eps_sq_buf, new_eps_sq)
            push!(h_buf, ht)
            paths[s, t] = ht
        end
    end

    _build_volatility_forecast(paths, h, conf_level, :garch)
end

# =============================================================================
# EGARCH Forecasting
# =============================================================================

"""
    forecast(m::EGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from an EGARCH(p,q) model via simulation.
"""
function forecast(m::EGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    q = m.q
    p = m.p
    E_abs_z = sqrt(T(2) / T(π))

    last_z = m.standardized_residuals[end-max(q,p)+1:end]
    last_log_h = log.(m.conditional_variance[end-max(q,p)+1:end])

    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        z_buf = copy(last_z)
        lh_buf = copy(last_log_h)
        for t in 1:h
            lht = m.omega
            for i in 1:q
                idx = length(z_buf) - i + 1
                zt = idx >= 1 ? z_buf[idx] : zero(T)
                lht += m.alpha[i] * (abs(zt) - E_abs_z) + m.gamma[i] * zt
            end
            for j in 1:p
                idx = length(lh_buf) - j + 1
                lht += m.beta[j] * (idx >= 1 ? lh_buf[idx] : last_log_h[end])
            end
            lht = clamp(lht, T(-50), T(50))
            ht = exp(lht)
            z = randn(rng, T)
            push!(z_buf, z)
            push!(lh_buf, lht)
            paths[s, t] = ht
        end
    end

    _build_volatility_forecast(paths, h, conf_level, :egarch)
end

# =============================================================================
# GJR-GARCH Forecasting
# =============================================================================

"""
    forecast(m::GJRGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from a GJR-GARCH(p,q) model via simulation.
"""
function forecast(m::GJRGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    q = m.q
    p = m.p

    last_eps = m.residuals[end-max(q,p)+1:end]
    last_h = m.conditional_variance[end-max(q,p)+1:end]

    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        eps_buf = copy(last_eps)
        h_buf = copy(last_h)
        for t in 1:h
            ht = m.omega
            for i in 1:q
                idx = length(eps_buf) - i + 1
                if idx >= 1
                    e = eps_buf[idx]
                    indicator = e < zero(T) ? one(T) : zero(T)
                    ht += (m.alpha[i] + m.gamma[i] * indicator) * e^2
                else
                    ht += (m.alpha[i] + m.gamma[i] * T(0.5)) * mean(last_eps .^ 2)
                end
            end
            for j in 1:p
                idx = length(h_buf) - j + 1
                ht += m.beta[j] * (idx >= 1 ? h_buf[idx] : mean(last_h))
            end
            ht = max(ht, eps(T))
            z = randn(rng, T)
            new_eps = sqrt(ht) * z
            push!(eps_buf, new_eps)
            push!(h_buf, ht)
            paths[s, t] = ht
        end
    end

    _build_volatility_forecast(paths, h, conf_level, :gjr_garch)
end

# =============================================================================
# Shared Forecast Builder
# =============================================================================

function _build_volatility_forecast(paths::Matrix{T}, h::Int, conf_level::T, model_type::Symbol) where {T}
    alpha_half = (one(T) - conf_level) / 2
    fc = vec(mean(paths, dims=1))
    ci_lo = vec(mapslices(x -> quantile(x, alpha_half), paths, dims=1))
    ci_hi = vec(mapslices(x -> quantile(x, one(T) - alpha_half), paths, dims=1))
    se = vec(std(paths, dims=1))
    VolatilityForecast(fc, ci_lo, ci_hi, se, h, conf_level, model_type)
end

# StatsAPI predict wrappers
StatsAPI.predict(m::GARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
StatsAPI.predict(m::EGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
StatsAPI.predict(m::GJRGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast

# =============================================================================
# IGARCH / Component-GARCH / APARCH forecasting (EV-15, #423)
# =============================================================================

"""
    forecast(m::IGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from an IGARCH(p,q) model. The point forecast is
computed **analytically** from the unit-persistence recursion
`E[σ²_{t+k}] = ω + Σαᵢ E[σ²_{t+k-i}] + Σβⱼ E[σ²_{t+k-j}]` (using `E[ε²]=E[σ²]` for
future shocks), which grows without bound (strictly increasing when `ω > 0`);
confidence bands come from simulated paths.
"""
function forecast(m::IGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    q, p = m.q, m.p

    # Analytic mean via E[ε²]=E[σ²] recursion, seeded with the fitted tail.
    m_lag = max(q, p)
    eps_sq_hist = m.residuals[end-m_lag+1:end] .^ 2
    h_hist = m.conditional_variance[end-m_lag+1:end]
    fc = Vector{T}(undef, h)
    ef = copy(eps_sq_hist)   # expected ε² buffer (history then forecasts)
    hf = copy(h_hist)        # expected σ² buffer
    for t in 1:h
        ht = m.omega
        for i in 1:q
            ht += m.alpha[i] * ef[end-i+1]
        end
        for j in 1:p
            ht += m.beta[j] * hf[end-j+1]
        end
        ht = max(ht, eps(T))
        fc[t] = ht
        push!(ef, ht)   # E[ε²_{t+k}] = E[σ²_{t+k}]
        push!(hf, ht)
    end

    # Simulated bands
    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        eps_sq_buf = copy(eps_sq_hist)
        h_buf = copy(h_hist)
        for t in 1:h
            ht = m.omega
            for i in 1:q
                ht += m.alpha[i] * eps_sq_buf[end-i+1]
            end
            for j in 1:p
                ht += m.beta[j] * h_buf[end-j+1]
            end
            ht = max(ht, eps(T))
            z = randn(rng, T)
            push!(eps_sq_buf, ht * z^2)
            push!(h_buf, ht)
            paths[s, t] = ht
        end
    end
    alpha_half = (one(T) - conf_level) / 2
    ci_lo = vec(mapslices(x -> quantile(x, alpha_half), paths, dims=1))
    ci_hi = vec(mapslices(x -> quantile(x, one(T) - alpha_half), paths, dims=1))
    se = vec(std(paths, dims=1))
    VolatilityForecast(fc, ci_lo, ci_hi, se, h, conf_level, :igarch)
end

"""
    forecast(m::CGARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from a Component-GARCH(1,1) model via simulation of
the permanent/transitory recursion; point forecasts revert to the long-run level `ω`.
"""
function forecast(m::CGARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    last_eps_sq = m.residuals[end]^2
    last_h = m.conditional_variance[end]
    last_q = m.permanent[end]

    paths = Matrix{T}(undef, n_sim, h)
    for s in 1:n_sim
        e2 = last_eps_sq; hh = last_h; qq = last_q
        for t in 1:h
            qt = m.omega + m.rho * (qq - m.omega) + m.phi * (e2 - hh)
            ht = qt + m.alpha * (e2 - qq) + m.beta * (hh - qq)
            qt = max(qt, eps(T)); ht = max(ht, eps(T))
            z = randn(rng, T)
            e2 = ht * z^2
            hh = ht; qq = qt
            paths[s, t] = ht
        end
    end
    _build_volatility_forecast(paths, h, conf_level, :cgarch)
end

"""
    forecast(m::APARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from an APARCH(p,q) model via simulation of the
power-`δ` recursion.
"""
function forecast(m::APARCHModel{T}, h::Int; conf_level::Real=0.95, n_sim::Int=10000,
                  rng::AbstractRNG=Random.default_rng()) where {T}
    conf_level = T(conf_level)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    q, p = m.q, m.p
    d = m.delta
    m_lag = max(q, p)
    last_eps = m.residuals[end-m_lag+1:end]
    last_s = m.sigma_delta[end-m_lag+1:end]

    paths = Matrix{T}(undef, n_sim, h)
    for si in 1:n_sim
        eps_buf = copy(last_eps)
        s_buf = copy(last_s)
        for t in 1:h
            st = m.omega
            for i in 1:q
                e = eps_buf[end-i+1]
                st += m.alpha[i] * (abs(e) - m.gamma[i] * e)^d
            end
            for j in 1:p
                st += m.beta[j] * s_buf[end-j+1]
            end
            st = max(st, eps(T))
            ht = st^(2 / d)
            z = randn(rng, T)
            push!(eps_buf, sqrt(ht) * z)
            push!(s_buf, st)
            paths[si, t] = ht
        end
    end
    _build_volatility_forecast(paths, h, conf_level, :aparch)
end

StatsAPI.predict(m::IGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
StatsAPI.predict(m::CGARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
StatsAPI.predict(m::APARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
