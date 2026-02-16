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
ARCH(q) variance forecasting.
"""

"""
    forecast(m::ARCHModel, h; conf_level=0.95, n_sim=10000) -> VolatilityForecast

Forecast conditional variance from an ARCH(q) model.

For h > q, the forecast converges to the unconditional variance.
Confidence intervals are computed via simulation.

# Arguments
- `m`: Fitted ARCHModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)
- `n_sim`: Number of simulation paths for CIs (default 10000)
"""
function forecast(m::ARCHModel{T}, h::Int; conf_level::T=T(0.95), n_sim::Int=10000) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    q = m.q
    n = length(m.y)
    alpha = m.alpha
    omega = m.omega

    # Last q squared residuals for initialization
    last_eps_sq = m.residuals[end-q+1:end] .^ 2

    # Simulation-based forecasting
    z_alpha = quantile(Normal{T}(zero(T), one(T)), one(T) - (one(T) - conf_level) / 2)
    paths = Matrix{T}(undef, n_sim, h)

    for s in 1:n_sim
        eps_sq_buf = copy(last_eps_sq)
        for t in 1:h
            ht = omega
            for i in 1:q
                idx = length(eps_sq_buf) - i + 1
                ht += alpha[i] * (idx >= 1 ? eps_sq_buf[idx] : mean(last_eps_sq))
            end
            ht = max(ht, eps(T))
            z = randn(T)
            new_eps_sq = ht * z^2
            push!(eps_sq_buf, new_eps_sq)
            paths[s, t] = ht
        end
    end

    fc = vec(mean(paths, dims=1))
    ci_lo = vec(mapslices(x -> quantile(x, (one(T) - conf_level) / 2), paths, dims=1))
    ci_hi = vec(mapslices(x -> quantile(x, one(T) - (one(T) - conf_level) / 2), paths, dims=1))
    se = vec(std(paths, dims=1))

    VolatilityForecast(fc, ci_lo, ci_hi, se, h, conf_level, :arch)
end

StatsAPI.predict(m::ARCHModel, h::Int; kwargs...) = forecast(m, h; kwargs...).forecast
