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
News impact curve for GARCH-family models.
"""

"""
    news_impact_curve(m; range=(-3,3), n_points=200)

Compute the news impact curve: how a shock εₜ₋₁ maps to σ²ₜ, holding all else at unconditional values.

Returns named tuple `(shocks, variance)` where both are vectors of length `n_points`.

# Supported models
- `GARCHModel`: Symmetric parabola
- `GJRGARCHModel`: Asymmetric parabola (steeper for negative shocks)
- `EGARCHModel`: Asymmetric exponential curve
"""
function news_impact_curve(m::GARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))

    # For GARCH(1,1): h_t = ω + α₁ε²_{t-1} + β₁h̄
    # where h̄ = unconditional variance
    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)

    variance = map(shocks) do e
        ht = m.omega
        ht += m.alpha[1] * e^2
        for j in 1:m.p
            ht += m.beta[j] * h_bar
        end
        max(ht, eps(T))
    end

    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::GJRGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))

    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)

    variance = map(shocks) do e
        ht = m.omega
        indicator = e < zero(T) ? one(T) : zero(T)
        ht += (m.alpha[1] + m.gamma[1] * indicator) * e^2
        for j in 1:m.p
            ht += m.beta[j] * h_bar
        end
        max(ht, eps(T))
    end

    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::EGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    E_abs_z = sqrt(T(2) / T(π))

    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)
    log_h_bar = log(h_bar)

    variance = map(shocks) do e
        z = e / sigma
        log_ht = m.omega
        log_ht += m.alpha[1] * (abs(z) - E_abs_z) + m.gamma[1] * z
        for j in 1:m.p
            log_ht += m.beta[j] * log_h_bar
        end
        exp(clamp(log_ht, T(-50), T(50)))
    end

    (shocks=shocks, variance=variance)
end
