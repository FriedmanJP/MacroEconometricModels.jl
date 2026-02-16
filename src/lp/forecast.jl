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
Direct multi-step LP forecasting.

LP forecasts use horizon-specific regression coefficients directly (no recursion):
    ŷ_{T+h} = α_h + β_h·shock_h + Γ_h·controls_T

CI methods:
- `:analytical` — HAC standard errors + normal quantiles
- `:bootstrap` — residual resampling with percentile CIs
- `:none` — point forecasts only

References:
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local Projections."
- Plagborg-Møller, M. & Wolf, C. K. (2021). "Local Projections and VARs Estimate the Same
  Impulse Responses." *Econometrica*, 89(2), 955–980.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# LP Forecast from LPModel
# =============================================================================

"""
    forecast(lp::LPModel{T}, shock_path::AbstractVector{<:Real};
             ci_method=:analytical, conf_level=0.95, n_boot=500) -> LPForecast{T}

Compute direct multi-step LP forecasts given a shock trajectory.

For each horizon h=1,...,H, the forecast uses the LP regression coefficients:
    ŷ_{T+h} = α_h + β_h·shock_h + Γ_h·controls_T

where controls_T are the last `lags` observations of Y.

# Arguments
- `lp`: Estimated LP model
- `shock_path`: Vector of length H with assumed future shock values

# Keyword Arguments
- `ci_method`: `:analytical` (default), `:bootstrap`, or `:none`
- `conf_level`: Confidence level (default: 0.95)
- `n_boot`: Number of bootstrap replications (default: 500)

# Returns
`LPForecast{T}` with point forecasts, CIs, and standard errors.
"""
function forecast(lp::LPModel{T}, shock_path::AbstractVector{<:Real};
                  ci_method::Symbol=:analytical, conf_level::Real=0.95,
                  n_boot::Int=500) where {T<:AbstractFloat}
    H = lp.horizon
    @assert length(shock_path) == H "shock_path must have length H=$H"
    @assert ci_method ∈ (:analytical, :bootstrap, :none) "ci_method must be :analytical, :bootstrap, or :none"

    n_response = length(lp.response_vars)
    shock_path_T = T.(shock_path)

    # Build the control vector from the last observations
    x_controls = _build_forecast_controls(lp)

    # Compute point forecasts for each horizon
    forecasts = Matrix{T}(undef, H, n_response)
    se_mat = zeros(T, H, n_response)

    for h in 1:H
        B_h = lp.B[h+1]  # h=0 is index 1, h=1 is index 2, etc.
        k = size(B_h, 1)

        # Build regressor: [1, shock_h, controls...]
        x_h = zeros(T, k)
        x_h[1] = one(T)
        x_h[2] = shock_path_T[h]
        n_ctrl = min(length(x_controls), k - 2)
        if n_ctrl > 0
            x_h[3:2+n_ctrl] = x_controls[1:n_ctrl]
        end

        # Point forecast
        for j in 1:n_response
            forecasts[h, j] = dot(x_h, @view(B_h[:, j]))
        end

        # Analytical SE
        if ci_method == :analytical
            V_h = lp.vcov[h+1]
            for j in 1:n_response
                idx_start = (j - 1) * k
                var_forecast = zero(T)
                for a in 1:k, b in 1:k
                    var_forecast += x_h[a] * V_h[idx_start + a, idx_start + b] * x_h[b]
                end
                se_mat[h, j] = sqrt(max(var_forecast, zero(T)))
            end
        end
    end

    # Compute CIs
    ci_lower, ci_upper = _forecast_ci(forecasts, se_mat, T(conf_level), ci_method)

    if ci_method == :bootstrap
        ci_lower, ci_upper, se_mat = _lp_forecast_bootstrap(lp, shock_path_T,
                                                              x_controls, n_boot, T(conf_level))
    end

    LPForecast(forecasts, ci_lower, ci_upper, se_mat, H,
               lp.response_vars, lp.shock_var, shock_path_T, T(conf_level), ci_method)
end

"""Build the control vector from LP model's last observations."""
function _build_forecast_controls(lp::LPModel{T}) where {T<:AbstractFloat}
    Y = lp.Y
    T_obs, n = size(Y)
    controls = T[]
    for lag in 1:lp.lags
        t = T_obs - lag + 1
        if t >= 1
            append!(controls, @view(Y[t, :]))
        else
            append!(controls, zeros(T, n))
        end
    end
    controls
end

"""Compute forecast CIs from SEs using normal quantiles."""
function _forecast_ci(forecasts::Matrix{T}, se::Matrix{T}, conf_level::T,
                      ci_method::Symbol) where {T<:AbstractFloat}
    if ci_method == :none
        return copy(forecasts), copy(forecasts)
    end
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = forecasts .- z .* se
    ci_upper = forecasts .+ z .* se
    ci_lower, ci_upper
end

"""Bootstrap CIs for LP forecasts via residual resampling."""
function _lp_forecast_bootstrap(lp::LPModel{T}, shock_path::Vector{T},
                                 x_controls::Vector{T}, n_boot::Int,
                                 conf_level::T) where {T<:AbstractFloat}
    H = lp.horizon
    n_response = length(lp.response_vars)
    boot_forecasts = zeros(T, n_boot, H, n_response)

    for b in 1:n_boot
        for h in 1:H
            B_h = lp.B[h+1]
            U_h = lp.residuals[h+1]
            k = size(B_h, 1)

            # Build regressor
            x_h = zeros(T, k)
            x_h[1] = one(T)
            x_h[2] = shock_path[h]
            n_ctrl = min(length(x_controls), k - 2)
            if n_ctrl > 0
                x_h[3:2+n_ctrl] = x_controls[1:n_ctrl]
            end

            # Add resampled residual
            resid_idx = rand(1:size(U_h, 1))
            for j in 1:n_response
                boot_forecasts[b, h, j] = dot(x_h, @view(B_h[:, j])) + U_h[resid_idx, j]
            end
        end
    end

    alpha = (1 - conf_level) / 2
    ci_lower = Matrix{T}(undef, H, n_response)
    ci_upper = Matrix{T}(undef, H, n_response)
    se_mat = Matrix{T}(undef, H, n_response)

    @inbounds for h in 1:H, j in 1:n_response
        d = @view boot_forecasts[:, h, j]
        ci_lower[h, j] = quantile(d, alpha)
        ci_upper[h, j] = quantile(d, 1 - alpha)
        se_mat[h, j] = std(d)
    end

    ci_lower, ci_upper, se_mat
end

# =============================================================================
# LP Forecast from StructuralLP
# =============================================================================

"""
    forecast(slp::StructuralLP{T}, shock_idx::Int, shock_path::AbstractVector{<:Real};
             ci_method=:analytical, conf_level=0.95, n_boot=500) -> LPForecast{T}

Compute direct multi-step forecast from a structural LP model using a specific
orthogonalized shock.

# Arguments
- `slp`: Structural LP model
- `shock_idx`: Index of the structural shock to use (1:n)
- `shock_path`: Vector of length H with assumed shock values

# Keyword Arguments
- `ci_method`: `:analytical` (default), `:bootstrap`, or `:none`
- `conf_level`: Confidence level (default: 0.95)
- `n_boot`: Number of bootstrap replications (default: 500)
"""
function forecast(slp::StructuralLP{T}, shock_idx::Int,
                  shock_path::AbstractVector{<:Real};
                  ci_method::Symbol=:analytical, conf_level::Real=0.95,
                  n_boot::Int=500) where {T<:AbstractFloat}
    n = nvars(slp)
    @assert 1 <= shock_idx <= n "shock_idx must be in 1:$n"

    lp_model = slp.lp_models[shock_idx]
    forecast(lp_model, shock_path; ci_method=ci_method, conf_level=conf_level, n_boot=n_boot)
end
