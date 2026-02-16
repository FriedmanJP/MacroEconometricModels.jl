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
Stochastic Volatility posterior predictive forecasting.
"""

"""
    forecast(m::SVModel, h; conf_level=0.95) -> VolatilityForecast

Posterior predictive forecast of volatility from an SV model.

For each posterior draw (μ, φ, σ_η), simulates the log-volatility path forward
h_{T+1}, ..., h_{T+h} and returns quantiles of exp(hₜ).

# Arguments
- `m`: Fitted SVModel
- `h`: Forecast horizon
- `conf_level`: Confidence level for intervals (default 0.95)
"""
function forecast(m::SVModel{T}, h::Int; conf_level::T=T(0.95)) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))

    n_draws = m.n_samples
    n_obs = length(m.y)
    paths = Matrix{T}(undef, n_draws, h)

    for s in 1:n_draws
        mu = m.mu_post[s]
        phi = m.phi_post[s]
        sigma_eta = m.sigma_eta_post[s]

        # Get last latent state from stored draws
        h_last = m.h_draws[s, n_obs]

        h_prev = h_last
        for t in 1:h
            h_t = mu + phi * (h_prev - mu) + sigma_eta * randn(T)
            paths[s, t] = exp(h_t)
            h_prev = h_t
        end
    end

    _build_volatility_forecast(paths, h, conf_level, :sv)
end
