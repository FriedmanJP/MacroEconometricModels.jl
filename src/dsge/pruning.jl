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
Pruned simulation, IRFs, and FEVD for higher-order perturbation solutions.

Implements the Kim, Kim, Schaumburg & Sims (2008) pruning method to prevent
explosive sample paths when simulating second- (and higher-) order approximations.

References:
- Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). "Calculating and Using
  Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models."
  Journal of Economic Dynamics and Control, 32(11), 3397-3414.
"""

using Random

# =============================================================================
# simulate — pruned stochastic simulation
# =============================================================================

"""
    simulate(sol::PerturbationSolution{T}, T_periods::Int;
             shock_draws=nothing, rng=Random.default_rng(),
             antithetic::Bool=false) -> Matrix{T}

Simulate a higher-order perturbation solution using Kim et al. (2008) pruning.

For order 1, this is the standard linear simulation. For order 2, the pruned
simulation tracks first-order and second-order state components separately to
prevent the explosive sample paths that arise from naive simulation of
second-order decision rules.

# Arguments
- `sol`: perturbation solution
- `T_periods`: number of periods to simulate

# Keyword Arguments
- `shock_draws`: `T_periods x n_shocks` matrix of pre-drawn shocks (default: N(0,1))
- `rng`: random number generator
- `antithetic::Bool=false`: if true, use antithetic variates (negate second half of shocks)

# Returns
`T_periods x n_vars` matrix of levels (steady state + deviations).
"""
function simulate(sol::PerturbationSolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng(),
                  antithetic::Bool=false) where {T<:AbstractFloat}
    nx = nstates(sol)
    ny = ncontrols(sol)
    n  = nvars(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps

    # Draw or use provided shocks
    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_eps) "shock_draws must be ($T_periods, $n_eps)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_eps)
    end

    # Antithetic variates: negate second half of shocks for variance reduction
    if antithetic && shock_draws === nothing
        half = div(T_periods, 2)
        for t in (half+1):T_periods
            mirror_t = t - half
            if mirror_t >= 1
                e[t, :] = -e[mirror_t, :]
            end
        end
    end

    # Extract first-order blocks — ensure compatible dimensions even when nx=0 or ny=0
    hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)            # nx x nx
    eta_x = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)        # nx x n_eps
    gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)           # ny x nx
    eta_y = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)        # ny x n_eps

    # Output: deviations from steady state
    dev = zeros(T, T_periods, n)

    if sol.order == 1
        # Standard first-order simulation
        xf = zeros(T, nx)
        for t in 1:T_periods
            eps_t = e[t, :]
            # State transition
            xf_new = hx_state * xf + eta_x * eps_t
            # Control
            y_t = gx_state * xf_new + eta_y * eps_t
            # Store
            for (k, si) in enumerate(sol.state_indices)
                dev[t, si] = xf_new[k]
            end
            for (k, ci) in enumerate(sol.control_indices)
                dev[t, ci] = y_t[k]
            end
            xf = xf_new
        end
    elseif sol.order >= 2
        # Pruned second-order simulation (Kim et al. 2008)
        Hxx = sol.hxx   # nx x nv^2
        Gxx = sol.gxx   # ny x nv^2
        h_ss = sol.hσσ  # nx
        g_ss = sol.gσσ  # ny

        xf = zeros(T, nx)   # first-order state
        xs = zeros(T, nx)   # second-order correction state

        for t in 1:T_periods
            eps_t = e[t, :]

            # First-order state update
            xf_new = hx_state * xf + eta_x * eps_t

            # Innovations vector for Kronecker product
            vf = zeros(T, nv)
            if nx > 0
                vf[1:nx] = xf
            end
            vf[nx+1:nv] = eps_t
            kron_vf = kron(vf, vf)   # nv^2

            # Second-order state correction
            xs_new = hx_state * xs
            if Hxx !== nothing && !isempty(kron_vf)
                xs_new += T(0.5) * Hxx * kron_vf
            end
            if h_ss !== nothing
                xs_new += T(0.5) * h_ss
            end

            # Total state
            x_total = xf_new + xs_new

            # Control output
            y_t = gx_state * x_total + eta_y * eps_t
            if Gxx !== nothing && !isempty(kron_vf)
                y_t += T(0.5) * Gxx * kron_vf
            end
            if g_ss !== nothing
                y_t += T(0.5) * g_ss
            end

            # Store
            for (k, si) in enumerate(sol.state_indices)
                dev[t, si] = x_total[k]
            end
            for (k, ci) in enumerate(sol.control_indices)
                dev[t, ci] = y_t[k]
            end

            xf = xf_new
            xs = xs_new
        end
    end

    # Convert to levels
    levels = dev .+ sol.steady_state'

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        return levels[:, orig_idx]
    end

    return levels
end


# =============================================================================
# irf — impulse responses for PerturbationSolution
# =============================================================================

"""
    irf(sol::PerturbationSolution{T}, horizon::Int;
        irf_type::Symbol=:analytical, n_draws::Int=500,
        shock_size::Real=1.0, ci_type::Symbol=:none) -> ImpulseResponse{T}

Compute impulse responses from a perturbation solution.

For `irf_type=:analytical` (default), computes the standard first-order analytical
IRFs: `Phi_h[:,j] = hx_state^(h-1) * eta * e_j` (same as DSGESolution).

For `irf_type=:girf`, computes Generalized IRFs via Monte Carlo simulation,
which captures second-order effects.

# Keyword Arguments
- `irf_type::Symbol=:analytical`: `:analytical` for first-order, `:girf` for simulation-based
- `n_draws::Int=500`: number of Monte Carlo draws for GIRF
- `shock_size::Real=1.0`: size of the impulse (in standard deviations)
- `ci_type::Symbol=:none`: confidence interval type
"""
function irf(sol::PerturbationSolution{T}, horizon::Int;
             irf_type::Symbol=:analytical, n_draws::Int=500,
             shock_size::Real=1.0, ci_type::Symbol=:none,
             kwargs...) where {T<:AbstractFloat}
    irf_type in (:analytical, :girf) ||
        throw(ArgumentError("irf_type must be :analytical or :girf; got $irf_type"))

    if irf_type == :girf
        return _girf(sol, horizon; n_draws=n_draws, shock_size=T(shock_size))
    end

    # Analytical first-order IRFs
    nx = nstates(sol)
    ny = ncontrols(sol)
    n  = nvars(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps

    # Extract blocks — ensure compatible dimensions even when nx=0 or ny=0
    hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
    gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)
    eta_y = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, ny, n_eps)

    # Build full impact and transition in original variable ordering
    point_irf = zeros(T, horizon, n, n_eps)

    # Power of hx_state: hx_state^0 = I at h=1
    hx_power = Matrix{T}(I, nx, nx)

    for h in 1:horizon
        for j in 1:n_eps
            # Shock vector: e_j (unit vector)
            ej = zeros(T, n_eps)
            ej[j] = T(shock_size)

            if h == 1
                # x_1 = eta_x * e_j
                x_h = eta_x * ej
                # y_1 = gx_state * x_1 + eta_y * e_j
                y_h = gx_state * x_h + eta_y * ej
            else
                # x_h = hx_state^(h-1) * eta_x * e_j
                x_h = hx_power * eta_x * ej
                # y_h = gx_state * x_h  (no direct shock effect for h > 1)
                y_h = gx_state * x_h
            end

            # Store in original variable ordering
            for (k, si) in enumerate(sol.state_indices)
                point_irf[h, si, j] = x_h[k]
            end
            for (k, ci) in enumerate(sol.control_indices)
                point_irf[h, ci, j] = y_h[k]
            end
        end
        hx_power = hx_power * hx_state
    end

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        point_irf = point_irf[:, orig_idx, :]
        var_names = [string(s) for s in sol.spec.original_endog]
        n_out = length(orig_idx)
    else
        var_names = sol.spec.varnames
        n_out = n
    end
    shock_names = [string(s) for s in sol.spec.exog]

    ci_lower = zeros(T, horizon, n_out, n_eps)
    ci_upper = zeros(T, horizon, n_out, n_eps)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type)
end


# =============================================================================
# _girf — Generalized IRF via Monte Carlo simulation
# =============================================================================

"""
    _girf(sol::PerturbationSolution{T}, horizon::Int;
          n_draws::Int=500, shock_size::T=one(T)) -> ImpulseResponse{T}

Compute Generalized Impulse Response Functions via Monte Carlo simulation.

GIRF = E[y_{t+h} | eps_t = shock] - E[y_{t+h} | eps_t = 0], averaged over
`n_draws` random draws of future shocks.
"""
function _girf(sol::PerturbationSolution{T}, horizon::Int;
               n_draws::Int=500, shock_size::T=one(T)) where {T<:AbstractFloat}
    n = nvars(sol)
    n_eps = nshocks(sol)

    # Determine output variable count
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        var_names = [string(s) for s in sol.spec.original_endog]
        n_out = length(orig_idx)
    else
        orig_idx = collect(1:n)
        var_names = sol.spec.varnames
        n_out = n
    end
    shock_names = [string(s) for s in sol.spec.exog]

    point_irf = zeros(T, horizon, n_out, n_eps)

    for j in 1:n_eps
        # Accumulate IRF across Monte Carlo draws
        irf_accum = zeros(T, horizon, n_out)

        for d in 1:n_draws
            rng_draw = Random.MersenneTwister(d * 31 + j * 17)

            # Common future shocks for both shocked and baseline
            future_shocks = randn(rng_draw, T, horizon, n_eps)

            # Shocked path: first period has the impulse
            shocked_shocks = copy(future_shocks)
            shocked_shocks[1, j] += shock_size

            # Baseline path: no impulse
            baseline_shocks = copy(future_shocks)

            # Simulate both paths
            sim_shocked  = simulate(sol, horizon; shock_draws=shocked_shocks)
            sim_baseline = simulate(sol, horizon; shock_draws=baseline_shocks)

            # Difference
            diff = sim_shocked .- sim_baseline
            irf_accum .+= diff[:, orig_idx]
        end

        point_irf[:, :, j] = irf_accum ./ n_draws
    end

    ci_lower = zeros(T, horizon, n_out, n_eps)
    ci_upper = zeros(T, horizon, n_out, n_eps)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, :none)
end


# =============================================================================
# fevd — forecast error variance decomposition
# =============================================================================

"""
    fevd(sol::PerturbationSolution{T}, horizon::Int) -> FEVD{T}

Compute forecast error variance decomposition from a perturbation solution.

Uses the analytical first-order IRFs to compute the proportion of h-step
forecast error variance attributable to each structural shock. This follows
the same IRF-based computation as `fevd(::DSGESolution, ...)`.
"""
function fevd(sol::PerturbationSolution{T}, horizon::Int) where {T<:AbstractFloat}
    irf_result = irf(sol, horizon)
    n_vars = length(irf_result.variables)
    n_eps = nshocks(sol)

    decomp = zeros(T, n_vars, n_eps, horizon)
    props  = zeros(T, n_vars, n_eps, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n_vars
            total = zero(T)
            for j in 1:n_eps
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irf_result.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end

    var_names = irf_result.variables
    shock_names = irf_result.shocks

    FEVD{T}(decomp, props, var_names, shock_names)
end
