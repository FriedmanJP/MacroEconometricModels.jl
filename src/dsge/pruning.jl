# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    n_eps = nshocks(sol)

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

    # One canonical pruned recursion, shared with the moments and the unconditional FEVD
    # ([T269]); in particular the control map reads the LAGGED components (S-01 / #119),
    # which orders 2 and 3 used to get wrong.
    pss = pruned_state_space(sol)
    dev = _pss_simulate_dev(pss, e)

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
                # x_1 = eta_x * e_j ; control at impact = direct shock only (lagged state x_0 = 0)
                x_h = eta_x * ej
                y_h = eta_y * ej
            else
                # x_h = hx_state^(h-1) * eta_x * e_j
                x_h = hx_power * eta_x * ej
                # Control uses the LAGGED state x_{h-1} (stored at horizon h-1); gx_state loads
                # the lagged state, and the direct shock enters only at impact (audit S-01 / #119).
                x_lag = T[point_irf[h-1, si, j] for si in sol.state_indices]
                y_h = gx_state * x_lag
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
    fevd(sol::PerturbationSolution{T}, horizon::Int;
         unconditional::Bool=false) -> FEVD{T}

Compute forecast error variance decomposition from a perturbation solution.

# Keyword Arguments
- `unconditional::Bool=false`: if `true` and `sol.order >= 2`, compute the
  unconditional (asymptotic) FEVD using the Andreasen et al. (2018) augmented
  Lyapunov approach. This properly accounts for second-order cross-terms (Hxx)
  when attributing variance to individual shocks. The returned FEVD has
  `horizon=1` with the asymptotic decomposition.
  If `false`, uses analytical first-order IRFs (same as `fevd(::DSGESolution)`).
"""
function fevd(sol::PerturbationSolution{T}, horizon::Int;
              unconditional::Bool=false) where {T<:AbstractFloat}
    if unconditional && sol.order >= 2
        return _fevd_unconditional(sol)
    end

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

"""
    _fevd_unconditional(sol::PerturbationSolution{T}) -> FEVD{T}

Compute unconditional (asymptotic) FEVD using the augmented Lyapunov approach.
For each shock j, zeros out all other shocks and re-solves the augmented system
to get the variance contribution. Properly handles second-order cross-terms.
"""
function _fevd_unconditional(sol::PerturbationSolution{T}) where {T}
    pss = pruned_state_space(sol)
    n, n_eps = pss.n, pss.n_eps

    # Shared augmented system + observation map ([T269]). This routine previously built its
    # own and loaded the control's shock channel with gx·η_x + η_y — the state channel counted
    # twice, the opposite of the error the moment routine made (S-01 / #119).
    C_full, noise_full, _ = _pss_obs_map_2nd(pss)

    decomp = zeros(T, n, n_eps, 1)
    props  = zeros(T, n, n_eps, 1)

    for j in 1:n_eps
        # Keep only shock j: zero every other column of the shock loadings.
        eta_x_j = zeros(T, pss.nx, n_eps)
        pss.nx > 0 && (eta_x_j[:, j] = pss.eta_x[:, j])
        noise_j = zeros(T, n, n_eps)
        noise_j[:, j] = noise_full[:, j]

        _, _, Var_z_j, _, _ = _pss_augmented_2nd(pss; eta_x_override=eta_x_j)
        Var_y_j = C_full * Var_z_j * C_full' + noise_j * noise_j'
        Var_y_j = (Var_y_j + Var_y_j') / 2

        for i in 1:n
            decomp[i, j, 1] = max(Var_y_j[i, i], zero(T))
        end
    end

    # Normalize: at order≥2 the per-shock contributions don't sum to the total
    # variance due to cross-shock quartic moment terms. We normalize by the
    # sum of per-shock contributions (not by total variance) to ensure rows
    # sum to 1, following Andreasen et al. (2018) §4.2.
    for i in 1:n
        row_sum = zero(T)
        for j in 1:n_eps
            row_sum += decomp[i, j, 1]
        end
        if row_sum > 0
            for j in 1:n_eps
                props[i, j, 1] = decomp[i, j, 1] / row_sum
            end
        end
    end

    # Handle augmented models
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        decomp = decomp[orig_idx, :, :]
        props = props[orig_idx, :, :]
        n = length(orig_idx)
    end

    var_names = [string(sol.spec.endog[i]) for i in 1:(sol.spec.augmented ?
                 length(_original_var_indices(sol.spec)) : n)]
    shock_names = [string(s) for s in sol.spec.exog]

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
# _extract_xxx_block — extract state×state×state block from v⊗v⊗v Kronecker
# =============================================================================

"""
    _extract_xxx_block(Mvvv::Matrix{T}, nx::Int, nv::Int) → Matrix{T}

Extract the `(xf⊗xf⊗xf)` sub-block from a matrix with `nv³` columns (Kronecker
ordering of `v = [x; ε]`).  Returns a matrix with `nx³` columns corresponding
to the state×state×state indices only.
"""
function _extract_xxx_block(Mvvv::Matrix{T}, nx::Int, nv::Int) where {T}
    nrows = size(Mvvv, 1)
    Mxxx = zeros(T, nrows, nx^3)
    for a in 1:nx
        for b in 1:nx
            for c in 1:nx
                col_vvv = ((a - 1) * nv + b - 1) * nv + c
                col_xxx = ((a - 1) * nx + b - 1) * nx + c
                @inbounds Mxxx[:, col_xxx] = Mvvv[:, col_vvv]
            end
        end
    end
    return Mxxx
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
    pss = pruned_state_space(sol)
    n = pss.n

    # One shared definition of the augmented system and of the observation map ([T269]) —
    # this routine and `_fevd_unconditional` used to build (and disagree about) their own.
    A, c, Var_z, E_z, M = _pss_augmented_2nd(pss)
    C_full, noise_full, d_full = _pss_obs_map_2nd(pss)

    E_y = C_full * E_z + d_full
    Var_y = C_full * Var_z * C_full' + noise_full * noise_full'
    Var_y = (Var_y + Var_y') / 2  # enforce symmetry

    # Autocovariances: y_t = C·z_{t-1} + noise·ε_t + d, and z_{t-1} carries ε_{t-1} through
    # the innovation, so the lag-k covariance needs the cross term C·A^{k-1}·M·noise'.
    max_lag = maximum(lags)
    Cov_y = zeros(T, n, n, max_lag)
    nz = size(A, 1)
    A_power = copy(A)
    A_power_prev = Matrix{T}(I, nz, nz)
    for lag in 1:max_lag
        Cov_y[:, :, lag] = C_full * A_power * Var_z * C_full' +
                           C_full * A_power_prev * M * noise_full'
        A_power_prev = A_power
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
# _augmented_moments_3rd — closed-form 3rd-order moments
# =============================================================================

"""
    _augmented_moments_3rd(sol::PerturbationSolution{T};
                             lags::Vector{Int}=[1]) → Dict{Symbol, Any}

Compute closed-form unconditional moments for a 3rd-order perturbation solution
using the augmented-state Lyapunov approach (Andreasen et al. 2018).

The augmented state is `z = [xf; xs; vec(xf⊗xf); xrd; vec(xf⊗xs); vec(xf⊗xf⊗xf)]`
of dimension `3nx + 2nx² + nx³`.  The system is `z(t+1) = A·z(t) + c + u(t)`.

Returns a Dict with keys:
- `:E_y` — n-vector of unconditional means (deviations from SS)
- `:Var_y` — n×n unconditional variance-covariance
- `:Cov_y` — n×n×max_lag autocovariance tensor
- `:E_z`, `:Var_z` — augmented state moments (for diagnostics)
"""
function _augmented_moments_3rd(sol::PerturbationSolution{T};
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

    # Extract xxx block from hxxx/gxxx (nv³ → nx³)
    hxxx_xxx = sol.hxxx !== nothing ? _extract_xxx_block(sol.hxxx, nx, nv) : zeros(T, nx, nx^3)
    gxxx_xxx = sol.gxxx !== nothing ? _extract_xxx_block(sol.gxxx, nx, nv) : zeros(T, ny, nx^3)

    # Extract xx block from hσσx/gσσx (nv → nx)
    hssx_xx = sol.hσσx !== nothing ? sol.hσσx[:, 1:nx] : zeros(T, nx, nx)
    gssx_xx = sol.gσσx !== nothing ? sol.gσσx[:, 1:nx] : zeros(T, ny, nx)

    nz = 3 * nx + 2 * nx^2 + nx^3

    # Handle nx == 0
    if nx == 0
        E_y = zeros(T, n)
        Var_y = zeros(T, n, n)
        max_lag = maximum(lags)
        Cov_y = zeros(T, n, n, max_lag)
        if ny > 0
            Var_y[sol.control_indices, sol.control_indices] = eta_y * eta_y'
        end
        return Dict{Symbol, Any}(:E_y => E_y, :Var_y => Var_y, :Cov_y => Cov_y,
                                  :E_z => zeros(T, 0), :Var_z => zeros(T, 0, 0))
    end

    # ---- Block ranges for the augmented state z ----
    r1 = 1:nx                                # xf
    r2 = nx+1:2*nx                           # xs
    r3 = 2*nx+1:2*nx+nx^2                    # kron(xf,xf)
    r4 = 2*nx+nx^2+1:3*nx+nx^2              # xrd
    r5 = 3*nx+nx^2+1:3*nx+2*nx^2            # kron(xf,xs)
    r6 = 3*nx+2*nx^2+1:nz                   # kron(xf,xf,xf)

    # ---- Build 6-block transition matrix A (nz × nz) ----
    A = zeros(T, nz, nz)
    A[r1, r1] = hx_state                                        # (1,1)
    A[r2, r2] = hx_state                                        # (2,2)
    A[r2, r3] = T(0.5) * hxx_xx                                 # (2,3)
    A[r3, r3] = kron(hx_state, hx_state)                        # (3,3)
    A[r4, r1] = T(0.5) * hssx_xx                                # (4,1)
    A[r4, r4] = hx_state                                        # (4,4)
    A[r4, r5] = hxx_xx                                           # (4,5)
    A[r4, r6] = (one(T) / T(6)) * hxxx_xxx                      # (4,6)
    # (5,1): kron(hx*xf, 0.5*hσσ) → matrix: kron(hx, reshape(0.5*hσσ, nx, 1))
    h_ss = sol.hσσ !== nothing ? sol.hσσ : zeros(T, nx)
    A[r5, r1] = kron(hx_state, reshape(T(0.5) * h_ss, nx, 1))  # (5,1)
    A[r5, r5] = kron(hx_state, hx_state)                        # (5,5)
    A[r5, r6] = kron(hx_state, T(0.5) * hxx_xx)                 # (5,6)
    A[r6, r6] = kron(hx_state, kron(hx_state, hx_state))        # (6,6)

    # ---- Build constant vector c (nz) ----
    I_ne = Matrix{T}(I, n_eps, n_eps)
    c = zeros(T, nz)
    c[r2] = T(0.5) * h_ss
    c[r3] = kron(eta_x, eta_x) * vec(I_ne)
    if sol.hσσσ !== nothing
        c[r4] = (one(T) / T(6)) * sol.hσσσ
    end

    # ---- Unconditional mean ----
    E_z = (Matrix{T}(I, nz, nz) - A) \ c

    # ---- Augmented variance: exact, simulation-free ([T269]) ----
    # `Var(xi)` and the `Cov(xi_{t+1}, z_t)` cross term are computed by exact Gauss-Hermite
    # integration over the shocks and solved as a fixed point with the Lyapunov equation. This
    # replaces a Monte-Carlo estimate of `Var(xi)`, and it supplies the cross term, which was
    # missing entirely — it is nonzero at third order (Andreasen et al.'s `BCov_xiLeadS_z`)
    # because the eps^2*xf terms in the xf(x)xf(x)xf block correlate the innovation with the
    # state it is added to.
    pss = pruned_state_space(sol)
    Var_z, BCov, Xbar, S1 = _pss_augmented_3rd(pss, A, c, E_z, hxx_xx)

    # ---- Observation mapping ----
    # Use transition-based convention (consistent with _augmented_moments_2nd):
    # state_obs(t) = C_state · z(t-1) + d_state + noise_state · ε(t)
    C_state = zeros(T, nx, nz)
    C_state[:, r1] = hx_state + T(0.5) * hssx_xx
    C_state[:, r2] = hx_state
    C_state[:, r3] = T(0.5) * hxx_xx
    C_state[:, r4] = hx_state
    C_state[:, r5] = hxx_xx
    C_state[:, r6] = (one(T) / T(6)) * hxxx_xxx
    noise_state = eta_x
    d_state = T(0.5) * h_ss
    if sol.hσσσ !== nothing
        d_state = d_state + (one(T) / T(6)) * sol.hσσσ
    end

    # Control observation. The control map has the SAME SHAPE as the state map with h → g,
    # because both are the policy function evaluated on the LAGGED components and the current
    # shock ([T269]). It was previously `gx_state·C_state`, i.e. gx applied to the *current*
    # state, which propagates the state channel twice and also drags the state's ½·hσσ
    # intercept into the control (S-01 / #119).
    C_ctrl = zeros(T, ny, nz)
    if ny > 0 && nx > 0
        C_ctrl[:, r1] = gx_state + T(0.5) * gssx_xx
        C_ctrl[:, r2] = gx_state
        C_ctrl[:, r3] = T(0.5) * gxx_xx
        C_ctrl[:, r4] = gx_state
        C_ctrl[:, r5] = gxx_xx
        C_ctrl[:, r6] = (one(T) / T(6)) * gxxx_xxx
    end
    noise_ctrl = ny > 0 ? eta_y : zeros(T, 0, n_eps)
    d_ctrl = zeros(T, ny)
    if sol.gσσ !== nothing && ny > 0
        d_ctrl += T(0.5) * sol.gσσ
    end
    if sol.gσσσ !== nothing && ny > 0
        d_ctrl += (one(T) / T(6)) * sol.gσσσ
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

    # Autocovariances. With y_t = C·z_{t-1} + noise·ε_t + d,
    #     Cov(y_t, y_{t-k}) = C·G_k·C' + C·S_k·noise',
    #     G_k = Cov(z_t, z_{t-k}) = A·G_{k-1} + Xbar·[0; G_{k-1}],   G_0 = Var_z
    #     S_k = Cov(z_{t-1}, ε_{t-k}) = A·S_{k-1} + Xbar·[0; S_{k-1}]
    # The `Xbar·[0; ·]` terms are Andreasen et al.'s `Get_BCov_xiLeadS_z` at lead k; they come
    # out of the same Ξ(ε) representation as the variance rather than a separate derivation,
    # and they vanish identically at order 2 (where Xbar = 0), recovering `A^k·Var_z`.
    lift(X) = vcat(zeros(T, 1, size(X, 2)), X)
    max_lag = maximum(lags)
    Cov_y = zeros(T, n, n, max_lag)
    G = Matrix{T}(Var_z)
    S = Matrix{T}(S1)
    for lag in 1:max_lag
        G = A * G + Xbar * lift(G)
        Cov_y[:, :, lag] = C_full * G * C_full' + C_full * S * noise_full'
        S = A * S + Xbar * lift(S)
    end

    # Handle augmented models
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

    # Default :covariance format — backward compatible. Orders 2 AND 3 use the closed-form
    # augmented-state Lyapunov recursion ([T269]); both are simulation-free and reproduce the
    # analytic moments of an exactly linear model to machine precision.
    if sol.order == 2
        res = _augmented_moments_2nd(sol; lags=collect(1:max(lags, 1)))
        return _moment_vector_from_dict(res, lags)
    elseif sol.order >= 3
        res = _augmented_moments_3rd(sol; lags=collect(1:max(lags, 1)))
        return _moment_vector_from_dict(res, lags)
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
        Sigma_x = _dlyap(hx_state, eta_x * eta_x')
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

    if sol.order >= 3
        result = _augmented_moments_3rd(sol; lags=lag_vec)
        E_y = result[:E_y]
        Var_y = result[:Var_y]
        Cov_y = result[:Cov_y]
    elseif sol.order >= 2
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

        Var_xf = nx > 0 ? _dlyap(hx_state, eta_x * eta_x') : zeros(T, 0, 0)

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
    _moment_vector_from_dict(res, lags) → Vector{T}

Flatten a closed-form moment `Dict` (`:Var_y`, `:Cov_y`) into the `:covariance` moment-vector
layout shared with `analytical_moments(::DSGESolution)`: the upper triangle of the
variance-covariance matrix, then the diagonal autocovariance at each lag.
"""
function _moment_vector_from_dict(res::Dict{Symbol,Any}, lags::Int)
    Var_y = res[:Var_y]
    Cov_y = res[:Cov_y]
    k = size(Var_y, 1)
    T = eltype(Var_y)
    out = T[]
    for i in 1:k, j in i:k
        push!(out, Var_y[i, j])
    end
    for lag in 1:lags
        for i in 1:k
            push!(out, Cov_y[i, i, lag])
        end
    end
    return out
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
