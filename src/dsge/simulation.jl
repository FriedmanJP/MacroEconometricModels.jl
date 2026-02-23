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
Stochastic simulation and IRF/FEVD bridge for DSGE solutions.
"""

using Random

"""
    simulate(sol::DSGESolution{T}, T_periods::Int;
             shock_draws=nothing, rng=Random.default_rng()) -> Matrix{T}

Simulate the solved DSGE model: `y_t = G1 * y_{t-1} + impact * e_t + C_sol`.

Returns `T_periods x n_endog` matrix of levels (steady state + deviations).

# Arguments
- `sol`: solved DSGE model
- `T_periods`: number of periods to simulate

# Keyword Arguments
- `shock_draws`: `T_periods x n_shocks` matrix of pre-drawn shocks (default: `nothing`, draws from N(0,1))
- `rng`: random number generator (default: `Random.default_rng()`)
"""
function simulate(sol::DSGESolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)
    y_ss = sol.spec.steady_state

    # Draw or use provided shocks
    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_e) "shock_draws must be ($T_periods, $n_e)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_e)
    end

    # Simulate deviations from steady state
    dev = zeros(T, T_periods, n)
    for t in 1:T_periods
        y_prev = t == 1 ? zeros(T, n) : dev[t-1, :]
        dev[t, :] = sol.G1 * y_prev + sol.impact * e[t, :] + sol.C_sol
    end

    # Return levels (steady state + deviations)
    levels = dev .+ y_ss'
    return levels
end

"""
    irf(sol::DSGESolution{T}, horizon::Int; kwargs...) -> ImpulseResponse{T}

Compute analytical impulse responses from a solved DSGE model.

At horizon h for shock j: `Phi_h[:,j] = G1^(h-1) * impact[:,j]`.

Returns an `ImpulseResponse{T}` compatible with `plot_result()`.

# Arguments
- `sol`: solved DSGE model
- `horizon`: number of IRF periods

# Keyword Arguments
- `ci_type::Symbol=:none`: confidence interval type (`:none` for analytical DSGE)
"""
function irf(sol::DSGESolution{T}, horizon::Int;
             ci_type::Symbol=:none, kwargs...) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)

    # Analytical IRFs: Phi_h = G1^(h-1) * impact
    point_irf = zeros(T, horizon, n, n_e)
    G1_power = Matrix{T}(I, n, n)

    for h in 1:horizon
        for j in 1:n_e
            point_irf[h, :, j] = G1_power * sol.impact[:, j]
        end
        G1_power = G1_power * sol.G1
    end

    var_names = sol.spec.varnames
    shock_names = [string(s) for s in sol.spec.exog]

    ci_lower = zeros(T, horizon, n, n_e)
    ci_upper = zeros(T, horizon, n, n_e)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type)
end

"""
    fevd(sol::DSGESolution{T}, horizon::Int) -> FEVD{T}

Compute forecast error variance decomposition from a solved DSGE model.

Uses the analytical IRFs from the solution to compute the proportion of
h-step forecast error variance attributable to each structural shock.

Returns a `FEVD{T}` compatible with `plot_result()`.
"""
function fevd(sol::DSGESolution{T}, horizon::Int) where {T<:AbstractFloat}
    irf_result = irf(sol, horizon)
    n = nvars(sol)
    n_e = nshocks(sol)

    # Compute FEVD directly — _compute_fevd assumes n_vars == n_shocks
    decomp = zeros(T, n, n_e, horizon)
    props  = zeros(T, n, n_e, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n
            total = zero(T)
            for j in 1:n_e
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irf_result.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end

    var_names = sol.spec.varnames
    shock_names = [string(s) for s in sol.spec.exog]

    FEVD{T}(decomp, props, var_names, shock_names)
end
