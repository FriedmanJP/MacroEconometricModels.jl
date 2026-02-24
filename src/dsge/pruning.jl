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


# =============================================================================
# _dlyap_doubling — iterative doubling Lyapunov solver
# =============================================================================

"""
    _dlyap_doubling(A::AbstractMatrix{T}, B::AbstractMatrix{T};
                    tol::Real=1e-12, maxiter::Int=500) -> Matrix{T}

Solve the discrete Lyapunov equation `Σ = A·Σ·A' + B` via the doubling algorithm.

More numerically stable than Kronecker vectorization for large systems (O(n³) per
iteration vs O(n⁶) for the direct solve), and typically converges in O(log(1/ρ(A)))
iterations where ρ(A) is the spectral radius.

Algorithm (Smith 1991):
```
Aₖ = A,  Bₖ = B
repeat:
    Bₖ₊₁ = Aₖ · Bₖ · Aₖ' + Bₖ
    Aₖ₊₁ = Aₖ · Aₖ
until converged
return (Bₖ₊₁ + Bₖ₊₁') / 2
```

Convergence is declared when `maximum(abs.(Bₖ₊₁ - Bₖ)) < tol` or `norm(Aₖ) < tol`.
"""
function _dlyap_doubling(A::AbstractMatrix{T}, B::AbstractMatrix{T};
                         tol::Real=1e-12, maxiter::Int=500) where {T<:AbstractFloat}
    n = size(A, 1)
    size(A) == (n, n) || throw(ArgumentError("A must be square, got $(size(A))"))
    size(B) == (n, n) || throw(ArgumentError("B must be n×n, got $(size(B))"))

    Ak = Matrix{T}(A)
    Bk = Matrix{T}(B)

    for iter in 1:maxiter
        Bk_new = Ak * Bk * Ak' + Bk
        Ak_new = Ak * Ak

        # Check convergence: either Bk stabilized or Ak → 0
        if maximum(abs.(Bk_new - Bk)) < tol || opnorm(Ak_new, 1) < tol
            # Enforce exact symmetry
            return (Bk_new + Bk_new') / 2
        end

        Ak = Ak_new
        Bk = Bk_new
    end

    @warn "Lyapunov doubling did not converge in $maxiter iterations"
    return (Bk + Bk') / 2
end


# =============================================================================
# _extract_xx_block — extract state×state block from v⊗v Kronecker matrix
# =============================================================================

"""
    _extract_xx_block(Mvv::Matrix{T}, nx::Int, nv::Int) → Matrix{T}

Extract the `(xf⊗xf)` sub-block from a matrix with `nv²` columns (Kronecker
ordering of `v = [x; ε]`).  Returns a matrix with `nx²` columns corresponding
to the state×state indices only.
"""
function _extract_xx_block(Mvv::Matrix{T}, nx::Int, nv::Int) where {T}
    nrows = size(Mvv, 1)
    Mxx = zeros(T, nrows, nx * nx)
    for a in 1:nx
        for b in 1:nx
            col_vv = (a - 1) * nv + b   # column in nv² ordering
            col_xx = (a - 1) * nx + b    # column in nx² ordering
            @inbounds Mxx[:, col_xx] = Mvv[:, col_vv]
        end
    end
    return Mxx
end


# =============================================================================
# _innovation_variance_2nd — compute Var(innovations) for augmented state
# =============================================================================

"""
    _innovation_variance_2nd(hx_state, eta_x, Var_xf, nx, n_eps;
                              vectorMom3=nothing, vectorMom4=nothing) → Matrix{T}

Compute the innovation covariance matrix for the 2nd-order augmented state
`z = [xf; xs; vec(xf⊗xf)]`.

Follows `UnconditionalMoments_2nd_Lyap.m` from the GMM_ThirdOrder_v2 MATLAB
reference code (Andreasen 2015).

Arguments:
- `hx_state`: nx × nx state transition
- `eta_x`: nx × n_eps shock loading
- `Var_xf`: nx × nx unconditional variance of xf (from first-order Lyapunov)
- `vectorMom3`: n_eps vector of 3rd moments (default: zeros for symmetric shocks)
- `vectorMom4`: n_eps vector of 4th moments (default: 3s for Gaussian shocks)
"""
function _innovation_variance_2nd(hx_state::Matrix{T}, eta_x::Matrix{T},
                                   Var_xf::Matrix{T},
                                   nx::Int, n_eps::Int;
                                   vectorMom3::Union{Nothing,Vector{T}}=nothing,
                                   vectorMom4::Union{Nothing,Vector{T}}=nothing) where {T}
    nz = 2 * nx + nx^2
    Var_inov = zeros(T, nz, nz)

    # Default shock moments for Gaussian distribution
    if vectorMom3 === nothing
        vectorMom3 = zeros(T, n_eps)
    end
    if vectorMom4 === nothing
        vectorMom4 = fill(T(3), n_eps)
    end

    sigeta = eta_x

    # Block (1,1): first-order shock variance
    Var_inov[1:nx, 1:nx] = sigeta * sigeta'

    # Block (1,3) and (3,1): third-moment cross term
    if any(!iszero, vectorMom3)
        E_eps_eps2 = zeros(T, n_eps, n_eps^2)
        for phi1 in 1:n_eps
            for phi2 in 1:n_eps
                for phi3 in 1:n_eps
                    idx = (phi2 - 1) * n_eps + phi3
                    if phi1 == phi2 && phi1 == phi3
                        E_eps_eps2[phi1, idx] = vectorMom3[phi1]
                    end
                end
            end
        end
        block_13 = sigeta * E_eps_eps2 * kron(sigeta', sigeta')
        Var_inov[1:nx, (2*nx+1):(2*nx+nx^2)] = block_13
        Var_inov[(2*nx+1):(2*nx+nx^2), 1:nx] = block_13'
    end

    # Block (3,3): quartic terms
    # E[(xf⊗ε)(ε⊗xf)']
    E_xfeps_epsxf = zeros(T, nx * n_eps, nx * n_eps)
    for gama1 in 1:nx
        for phi1 in 1:n_eps
            idx1 = (gama1 - 1) * n_eps + phi1
            for phi2 in 1:n_eps
                for gama2 in 1:nx
                    idx2 = (phi2 - 1) * nx + gama2
                    if phi1 == phi2
                        E_xfeps_epsxf[idx1, idx2] = Var_xf[gama1, gama2]
                    end
                end
            end
        end
    end

    # E[(ε⊗ε)(ε⊗ε)'] — fourth moment matrix
    ne2 = n_eps^2
    E_eps2_eps2 = zeros(T, ne2, ne2)
    for phi4 in 1:n_eps
        for phi1 in 1:n_eps
            idx1 = (phi4 - 1) * n_eps + phi1
            for phi3 in 1:n_eps
                for phi2 in 1:n_eps
                    idx2 = (phi3 - 1) * n_eps + phi2
                    if phi1 == phi2 && phi3 == phi4 && phi1 != phi4
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi3 && phi2 == phi4 && phi1 != phi2
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi4 && phi2 == phi3 && phi1 != phi2
                        E_eps2_eps2[idx1, idx2] = one(T)
                    elseif phi1 == phi2 && phi1 == phi3 && phi1 == phi4
                        E_eps2_eps2[idx1, idx2] = vectorMom4[phi1]
                    end
                end
            end
        end
    end

    # Assemble block (3,3)
    I_ne = Matrix{T}(I, n_eps, n_eps)
    vec_I_ne = vec(I_ne)
    r1 = 2 * nx + 1
    r2 = 2 * nx + nx^2

    Var_inov[r1:r2, r1:r2] =
        kron(hx_state, sigeta) * kron(Var_xf, I_ne) * kron(hx_state, sigeta)' +
        kron(hx_state, sigeta) * E_xfeps_epsxf * kron(sigeta, hx_state)' +
        kron(sigeta, hx_state) * E_xfeps_epsxf' * kron(hx_state, sigeta)' +
        kron(sigeta, hx_state) * kron(I_ne, Var_xf) * kron(sigeta, hx_state)' +
        kron(sigeta, sigeta) * (E_eps2_eps2 - vec_I_ne * vec_I_ne') * kron(sigeta, sigeta)'

    # Enforce symmetry
    Var_inov = (Var_inov + Var_inov') / 2

    return Var_inov
end


# =============================================================================
# _augmented_moments_2nd — closed-form 2nd-order moments
# =============================================================================

"""
    _augmented_moments_2nd(sol::PerturbationSolution{T};
                            lags::Vector{Int}=[1]) → Dict{Symbol, Any}

Compute closed-form unconditional moments for a 2nd-order perturbation solution
using the augmented-state Lyapunov approach (Andreasen et al. 2018).

The augmented state is `z = [xf; xs; vec(xf⊗xf)]` of dimension `2nx + nx²`.
The system is `z(t+1) = A·z(t) + c + u(t)` where u(t) captures stochastic
innovations from the pruned dynamics.

Returns moments for ALL n = nx + ny variables, in the original variable ordering
(matching `sol.state_indices` and `sol.control_indices`).

Returns a Dict with keys:
- `:E_y` — n-vector of unconditional means (deviations from SS)
- `:Var_y` — n×n unconditional variance-covariance
- `:Cov_y` — n×n×max_lag autocovariance tensor
- `:E_z`, `:Var_z` — augmented state moments (for diagnostics)
"""
function _augmented_moments_2nd(sol::PerturbationSolution{T};
                                 lags::Vector{Int}=[1]) where {T}
    nx = nstates(sol)
    ny = ncontrols(sol)
    n  = nvars(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps

    # Extract first-order blocks
    hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x    = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
    gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)
    eta_y    = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)

    # Extract state×state blocks from hxx, gxx (nv² → nx²)
    hxx_xx = sol.hxx !== nothing ? _extract_xx_block(sol.hxx, nx, nv) : zeros(T, nx, nx^2)
    gxx_xx = sol.gxx !== nothing ? _extract_xx_block(sol.gxx, nx, nv) : zeros(T, ny, nx^2)

    nz = 2 * nx + nx^2

    # Build transition matrix A (nz × nz)
    A = zeros(T, nz, nz)
    A[1:nx, 1:nx] = hx_state                                        # xf → xf
    A[nx+1:2*nx, nx+1:2*nx] = hx_state                              # xs → xs
    A[nx+1:2*nx, 2*nx+1:nz] = T(0.5) * hxx_xx                      # kron(xf,xf) → xs
    A[2*nx+1:nz, 2*nx+1:nz] = kron(hx_state, hx_state)             # kron → kron

    # Build constant vector c (nz)
    I_ne = Matrix{T}(I, n_eps, n_eps)
    c = zeros(T, nz)
    if sol.hσσ !== nothing
        c[nx+1:2*nx] = T(0.5) * sol.hσσ
    end
    c[2*nx+1:nz] = kron(eta_x, eta_x) * vec(I_ne)

    # Unconditional mean: E[z] = (I - A) \ c
    E_z = (Matrix{T}(I, nz, nz) - A) \ c

    # First-order state variance (for innovation variance computation)
    Var_xf = nx > 0 ? _dlyap_doubling(hx_state, eta_x * eta_x') : zeros(T, 0, 0)

    # Innovation variance
    Var_inov = _innovation_variance_2nd(hx_state, eta_x, Var_xf, nx, n_eps)

    # Solve augmented Lyapunov: Var_z = A·Var_z·A' + Var_inov
    Var_z = _dlyap_doubling(A, Var_inov)

    # ---------------------------------------------------------------
    # Observation mapping for ALL n = nx + ny variables
    #
    # We express stored observations in terms of z(t) and ε(t), where
    # z(t) is pre-shock and ε(t) is independent of z(t):
    #   state_stored(t) = [hx, hx, 0.5·hxx_xx]·z(t) + 0.5·hσσ + eta_x·ε(t)
    #   ctrl_stored(t)  = gx·(hx·z_xf(t) + eta_x·ε(t) + hx·z_xs(t) + ...)
    #                     + eta_y·ε(t) + 0.5·gxx_xx·kron(xf,xf) + 0.5·gσσ
    # Var(obs) = C·Var_z·C' + noise·noise'  since z(t) ⊥ ε(t)
    # ---------------------------------------------------------------

    # State observation: C_state = [hx, hx, 0.5·hxx_xx]
    C_state = zeros(T, nx, nz)
    C_state[:, 1:nx] = hx_state
    C_state[:, nx+1:2*nx] = hx_state
    if nx > 0
        C_state[:, 2*nx+1:nz] = T(0.5) * hxx_xx
    end
    noise_state = eta_x
    d_state = sol.hσσ !== nothing ? T(0.5) * sol.hσσ : zeros(T, nx)

    # Control observation: ctrl = gx·x_total + eta_y·ε + 0.5·gxx·kron + 0.5·gσσ
    # where x_total = C_state·z + d_state + eta_x·ε
    C_ctrl = zeros(T, ny, nz)
    if ny > 0 && nx > 0
        C_ctrl = gx_state * C_state
        C_ctrl[:, 2*nx+1:nz] += T(0.5) * gxx_xx
    end
    noise_ctrl = ny > 0 ? gx_state * eta_x + eta_y : zeros(T, 0, n_eps)
    d_ctrl = zeros(T, ny)
    if ny > 0 && nx > 0
        d_ctrl = gx_state * d_state
    end
    if sol.gσσ !== nothing && ny > 0
        d_ctrl += T(0.5) * sol.gσσ
    end

    # Assemble into full n-vector in original variable ordering
    C_full = zeros(T, n, nz)
    noise_full = zeros(T, n, n_eps)
    d_full = zeros(T, n)
    for (k, si) in enumerate(sol.state_indices)
        C_full[si, :] = C_state[k, :]
        noise_full[si, :] = noise_state[k, :]
        d_full[si] = d_state[k]
    end
    for (k, ci) in enumerate(sol.control_indices)
        C_full[ci, :] = C_ctrl[k, :]
        noise_full[ci, :] = noise_ctrl[k, :]
        d_full[ci] = d_ctrl[k]
    end

    # Output moments
    E_y = C_full * E_z + d_full
    Var_y = C_full * Var_z * C_full' + noise_full * noise_full'
    Var_y = (Var_y + Var_y') / 2  # enforce symmetry

    # Autocovariances: Cov(w(t), w(t-k)) = C · A^k · Var_z · C'
    max_lag = maximum(lags)
    Cov_y = zeros(T, n, n, max_lag)
    A_power = copy(A)
    for lag in 1:max_lag
        Cov_z_lag = A_power * Var_z
        Cov_y[:, :, lag] = C_full * Cov_z_lag * C_full'
        A_power = A_power * A
    end

    # Handle augmented models: filter to original variables
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        E_y = E_y[orig_idx]
        Var_y = Var_y[orig_idx, orig_idx]
        Cov_y = Cov_y[orig_idx, orig_idx, :]
    end

    Dict{Symbol, Any}(
        :E_y => E_y,
        :Var_y => Var_y,
        :Cov_y => Cov_y,
        :E_z => E_z,
        :Var_z => Var_z,
    )
end


# =============================================================================
# analytical_moments — closed-form moments for PerturbationSolution
# =============================================================================

"""
    analytical_moments(sol::PerturbationSolution{T}; lags::Int=1,
                       format::Symbol=:covariance) -> Vector{T}

Compute analytical moment vector from a perturbation solution.

# Keyword Arguments
- `lags::Int=1` — number of autocovariance lags
- `format::Symbol=:covariance` — moment format:
  - `:covariance` (default): upper-triangle of var-cov + diagonal autocov
    (backward compatible with DSGESolution format)
  - `:gmm`: means + upper-triangle product moments + diagonal autocov
    (for GMM estimation with higher-order perturbation)

For **order 1** with `:covariance` format, uses the doubling Lyapunov solver.
For **order ≥ 2** with `:covariance` format, uses simulation-based moments.
For `:gmm` format at any order, uses closed-form augmented Lyapunov (order ≥ 2)
or standard Lyapunov (order 1).
"""
function analytical_moments(sol::PerturbationSolution{T};
                              lags::Int=1,
                              format::Symbol=:covariance) where {T<:AbstractFloat}
    format in (:covariance, :gmm) ||
        throw(ArgumentError("format must be :covariance or :gmm; got $format"))

    if format == :gmm
        return _analytical_moments_gmm(sol; lags=lags)
    end

    # Default :covariance format — backward compatible
    # For order >= 2: simulation-based moments via pruned simulation
    if sol.order >= 2
        return _simulation_moments(sol; lags=lags)
    end

    # Order 1: closed-form Lyapunov approach
    nx = nstates(sol)
    ny = ncontrols(sol)
    n  = nvars(sol)
    n_eps = nshocks(sol)
    nv = nx + n_eps

    # Extract first-order blocks
    hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)          # nx × nx
    eta_x    = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)   # nx × n_eps
    gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)         # ny × nx
    eta_y    = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)   # ny × n_eps

    # State covariance via Lyapunov: Σ_x = hx_state · Σ_x · hx_state' + η_x · η_x'
    if nx > 0
        Sigma_x = _dlyap_doubling(hx_state, eta_x * eta_x')
    else
        Sigma_x = zeros(T, 0, 0)
    end

    # Build full n×n covariance in original variable ordering
    Sigma = zeros(T, n, n)
    if nx > 0
        Sigma[sol.state_indices, sol.state_indices] = Sigma_x
        if ny > 0
            Sigma_xy = Sigma_x * gx_state'
            Sigma[sol.state_indices, sol.control_indices] = Sigma_xy
            Sigma[sol.control_indices, sol.state_indices] = Sigma_xy'
            Sigma[sol.control_indices, sol.control_indices] = gx_state * Sigma_x * gx_state' + eta_y * eta_y'
        end
    elseif ny > 0
        # Pure forward-looking model: only contemporaneous shock variance
        Sigma[sol.control_indices, sol.control_indices] = eta_y * eta_y'
    end

    # Handle augmented models: filter to original variables
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        Sigma = Sigma[orig_idx, orig_idx]
        k = length(orig_idx)
    else
        k = n
    end

    # Build G1-equivalent transition matrix for autocovariances
    G1_equiv = zeros(T, n, n)
    if nx > 0
        G1_equiv[sol.state_indices, sol.state_indices] = hx_state
        if ny > 0
            G1_equiv[sol.control_indices, sol.state_indices] = gx_state * hx_state
        end
    end
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        G1_equiv = G1_equiv[orig_idx, orig_idx]
    end

    # Extract moments in same format as DSGESolution version
    moments = T[]

    # Upper triangle of variance-covariance matrix
    for i in 1:k
        for j in i:k
            push!(moments, Sigma[i, j])
        end
    end

    # Autocovariances at each lag: Gamma_h = G1^h * Sigma, extract diagonal
    G1_power = copy(G1_equiv)
    for lag in 1:lags
        Gamma_h = G1_power * Sigma
        for i in 1:k
            push!(moments, Gamma_h[i, i])
        end
        G1_power = G1_power * G1_equiv
    end

    return moments
end


"""
    _analytical_moments_gmm(sol::PerturbationSolution{T}; lags::Int=1) → Vector{T}

Compute GMM-format moment vector: means + product moments + diagonal autocovariances.

For order >= 2, uses closed-form augmented Lyapunov.
For order 1, uses standard Lyapunov (means are zero).
"""
function _analytical_moments_gmm(sol::PerturbationSolution{T}; lags::Int=1) where {T}
    lag_vec = collect(1:lags)

    if sol.order >= 2
        result = _augmented_moments_2nd(sol; lags=lag_vec)
        E_y = result[:E_y]
        Var_y = result[:Var_y]
        Cov_y = result[:Cov_y]
    else
        # Order 1: standard Lyapunov, means are zero
        nx = nstates(sol)
        ny = ncontrols(sol)
        n  = nvars(sol)
        n_eps = nshocks(sol)
        nv = nx + n_eps

        hx_state = nx > 0 ? sol.hx[:, 1:nx] : zeros(T, 0, 0)
        eta_x    = nx > 0 ? sol.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
        gx_state = ny > 0 ? sol.gx[:, 1:nx] : zeros(T, 0, nx)
        eta_y    = ny > 0 ? sol.gx[:, nx+1:nv] : zeros(T, 0, n_eps)

        Var_xf = nx > 0 ? _dlyap_doubling(hx_state, eta_x * eta_x') : zeros(T, 0, 0)

        E_y = zeros(T, n)
        Var_y = zeros(T, n, n)
        if nx > 0
            Var_y[sol.state_indices, sol.state_indices] = Var_xf
            if ny > 0
                Var_y[sol.state_indices, sol.control_indices] = Var_xf * gx_state'
                Var_y[sol.control_indices, sol.state_indices] = gx_state * Var_xf
                Var_y[sol.control_indices, sol.control_indices] = gx_state * Var_xf * gx_state' + eta_y * eta_y'
            end
        elseif ny > 0
            Var_y[sol.control_indices, sol.control_indices] = eta_y * eta_y'
        end

        # Autocovariances
        G1_equiv = zeros(T, n, n)
        if nx > 0
            G1_equiv[sol.state_indices, sol.state_indices] = hx_state
            if ny > 0
                G1_equiv[sol.control_indices, sol.state_indices] = gx_state * hx_state
            end
        end

        max_lag = lags
        Cov_y = zeros(T, n, n, max_lag)
        G1_power = copy(G1_equiv)
        for lag in 1:max_lag
            Cov_y[:, :, lag] = G1_power * Var_y
            G1_power = G1_power * G1_equiv
        end

        # Handle augmented models
        if sol.spec.augmented
            orig_idx = _original_var_indices(sol.spec)
            E_y = E_y[orig_idx]
            Var_y = Var_y[orig_idx, orig_idx]
            Cov_y = Cov_y[orig_idx, orig_idx, :]
        end
    end

    ny_out = length(E_y)

    # Collect moments: means, product moments, diagonal autocov
    moments = T[]

    # 1. Means: E[y_i]
    append!(moments, E_y)

    # 2. Product moments: E[y_i * y_j] = Var_y[i,j] + E_y[i]*E_y[j], upper triangle
    for i in 1:ny_out
        for j in i:ny_out
            push!(moments, Var_y[i, j] + E_y[i] * E_y[j])
        end
    end

    # 3. Diagonal autocovariances at each lag: E[y_i,t * y_i,t-k]
    for lag in 1:lags
        for i in 1:ny_out
            push!(moments, Cov_y[i, i, lag] + E_y[i]^2)
        end
    end

    return moments
end


"""
    _simulation_moments(sol::PerturbationSolution{T}; lags::Int=1) -> Vector{T}

Compute moments via pruned simulation for higher-order perturbation solutions.

Uses a fixed RNG seed (12345) for reproducibility and T=100,000 simulation periods.
"""
function _simulation_moments(sol::PerturbationSolution{T}; lags::Int=1) where {T<:AbstractFloat}
    T_sim = 100_000
    sim = simulate(sol, T_sim; rng=Random.MersenneTwister(12345))

    k = size(sim, 2)

    # Compute sample mean and center
    mu = vec(sum(sim; dims=1)) / T_sim
    centered = sim .- mu'

    # Sample covariance (unbiased)
    Sigma = (centered' * centered) / (T_sim - 1)

    moments = T[]

    # Upper triangle of variance-covariance matrix
    for i in 1:k
        for j in i:k
            push!(moments, Sigma[i, j])
        end
    end

    # Diagonal autocovariances at each lag
    for lag in 1:lags
        for i in 1:k
            autocov = zero(T)
            for t in 1:(T_sim - lag)
                autocov += centered[t, i] * centered[t + lag, i]
            end
            autocov /= (T_sim - lag)
            push!(moments, autocov)
        end
    end

    return moments
end
