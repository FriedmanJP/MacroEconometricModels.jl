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
VECM-specific forecasting that iterates directly in levels to preserve cointegrating relationships.
"""

using LinearAlgebra, Statistics

"""
    forecast(vecm::VECMModel, h; ci_method=:none, reps=500, conf_level=0.95) -> VECMForecast

Forecast from a VECM by iterating the VECM equations in levels.

Unlike VAR forecasting, this preserves the cointegrating relationships in the
forecast path.

# Arguments
- `vecm`: Estimated VECM
- `h`: Forecast horizon
- `ci_method`: `:none` (default), `:bootstrap`, or `:simulation`
- `reps`: Number of bootstrap/simulation replications (default 500)
- `conf_level`: Confidence level (default 0.95)

# Returns
`VECMForecast` with level and difference forecasts, plus CIs if requested.
"""
function forecast(vecm::VECMModel{T}, h::Int;
                  ci_method::Symbol=:none,
                  reps::Int=500,
                  conf_level::Real=0.95) where {T}

    h < 1 && throw(ArgumentError("Forecast horizon must be positive"))
    ci_method ∈ (:none, :bootstrap, :simulation) ||
        throw(ArgumentError("ci_method must be :none, :bootstrap, or :simulation"))

    n = nvars(vecm)
    p = vecm.p

    # Point forecast
    levels = _vecm_point_forecast(vecm, h)
    differences = diff(vcat(vecm.Y[end:end, :], levels), dims=1)

    ci_lower = zeros(T, h, n)
    ci_upper = zeros(T, h, n)

    if ci_method != :none
        alpha_half = (1 - T(conf_level)) / 2
        sim_forecasts = Matrix{T}(undef, reps, h * n)

        if ci_method == :bootstrap
            _vecm_bootstrap_forecast!(sim_forecasts, vecm, h, reps)
        else  # :simulation
            _vecm_simulation_forecast!(sim_forecasts, vecm, h, reps)
        end

        for hi in 1:h, j in 1:n
            col = (hi - 1) * n + j
            d = @view sim_forecasts[:, col]
            ci_lower[hi, j] = quantile(d, alpha_half)
            ci_upper[hi, j] = quantile(d, 1 - alpha_half)
        end
    end

    VECMForecast{T}(levels, differences, ci_lower, ci_upper, h, ci_method)
end

# =============================================================================
# Internal Helpers
# =============================================================================

function _vecm_point_forecast(vecm::VECMModel{T}, h::Int) where {T}
    n = nvars(vecm)
    p = vecm.p

    # History: last p observations in levels
    history = copy(vecm.Y[max(1, end-p+1):end, :])

    levels = Matrix{T}(undef, h, n)

    for step in 1:h
        Y_prev = history[end, :]  # Y_{t-1}

        # Error correction term: β'Y_{t-1}
        if vecm.rank > 0
            ecm = vecm.beta' * Y_prev  # r × 1
            dY = vecm.alpha * ecm      # n × 1
        else
            dY = zeros(T, n)
        end

        # Short-run dynamics: Γᵢ ΔY_{t-i}
        for (i, Gi) in enumerate(vecm.Gamma)
            if size(history, 1) >= i + 1
                dY_lag = history[end-i+1, :] - history[end-i, :]
                dY .+= Gi * dY_lag
            end
        end

        # Intercept
        dY .+= vecm.mu

        Y_new = Y_prev + dY
        levels[step, :] = Y_new

        # Update history
        history = vcat(history, Y_new')
    end

    levels
end

function _vecm_bootstrap_forecast!(sim_forecasts::Matrix{T},
                                    vecm::VECMModel{T}, h::Int, reps::Int) where {T}
    n = nvars(vecm)
    T_eff = effective_nobs(vecm)

    for rep in 1:reps
        # Resample residuals with replacement
        idx = rand(1:T_eff, h)
        shocks = vecm.U[idx, :]

        forecast_path = _vecm_simulate_path(vecm, h, shocks)
        for hi in 1:h
            sim_forecasts[rep, ((hi-1)*n+1):(hi*n)] = forecast_path[hi, :]
        end
    end
end

function _vecm_simulation_forecast!(sim_forecasts::Matrix{T},
                                     vecm::VECMModel{T}, h::Int, reps::Int) where {T}
    n = nvars(vecm)
    L = safe_cholesky(vecm.Sigma)

    for rep in 1:reps
        shocks = Matrix{T}(undef, h, n)
        for step in 1:h
            shocks[step, :] = L * randn(T, n)
        end

        forecast_path = _vecm_simulate_path(vecm, h, shocks)
        for hi in 1:h
            sim_forecasts[rep, ((hi-1)*n+1):(hi*n)] = forecast_path[hi, :]
        end
    end
end

function _vecm_simulate_path(vecm::VECMModel{T}, h::Int, shocks::Matrix{T}) where {T}
    n = nvars(vecm)
    p = vecm.p
    history = copy(vecm.Y[max(1, end-p+1):end, :])
    levels = Matrix{T}(undef, h, n)

    for step in 1:h
        Y_prev = history[end, :]

        if vecm.rank > 0
            ecm = vecm.beta' * Y_prev
            dY = vecm.alpha * ecm
        else
            dY = zeros(T, n)
        end

        for (i, Gi) in enumerate(vecm.Gamma)
            if size(history, 1) >= i + 1
                dY_lag = history[end-i+1, :] - history[end-i, :]
                dY .+= Gi * dY_lag
            end
        end

        dY .+= vecm.mu
        dY .+= shocks[step, :]

        Y_new = Y_prev + dY
        levels[step, :] = Y_new
        history = vcat(history, Y_new')
    end

    levels
end
