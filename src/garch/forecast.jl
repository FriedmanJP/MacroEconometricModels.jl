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
function forecast(m::GARCHModel{T}, h::Int; conf_level::T=T(0.95), n_sim::Int=10000) where {T}
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
            z = randn(T)
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
function forecast(m::EGARCHModel{T}, h::Int; conf_level::T=T(0.95), n_sim::Int=10000) where {T}
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
            z = randn(T)
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
function forecast(m::GJRGARCHModel{T}, h::Int; conf_level::T=T(0.95), n_sim::Int=10000) where {T}
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
            z = randn(T)
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
