# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Discrete-time two-asset stationary equilibrium.

Closes the liquid and illiquid markets simultaneously, following the same
`(K, r_b)` damped iteration as [`ct_two_asset_ge`](@ref): illiquid wealth
clears against firm capital and liquid wealth against a fixed bond supply.
The inner household block is nested EGM or two-asset Bellman VFI.

# References
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary Policy According
  to HANK. *American Economic Review*, 108(3), 697–743.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# Illiquid law of motion and liquid Euler residual
# =============================================================================

"""
    _two_asset_a_prime(d_pol, grid, r_a) → Array{T,3}

Next-period illiquid holdings `a' = (1 + r_a) a + d` on the joint grid.
"""
function _two_asset_a_prime(d_pol::AbstractArray{T,3}, grid::HAGrid{T},
                            r_a::T) where {T<:AbstractFloat}
    a_grid = grid.grids[2]
    n_b, n_a, n_e = size(d_pol)
    a_next = similar(d_pol)
    @inbounds for je in 1:n_e, ia in 1:n_a, ib in 1:n_b
        a_next[ib, ia, je] = (one(T) + r_a) * a_grid[ia] + d_pol[ib, ia, je]
    end
    return a_next
end

"""
    _two_asset_euler_error_stats(c_pol, b_pol, a_pol, ip, grid, income, prices;
                                 points=:midpoints) → NamedTuple

Liquid Euler residual of a two-asset policy:

    err = |1 − β(1+r_b) E[u'(c(b', a', e'))] / u'(c)|

Continuation consumption is bilinear in `(b', a')`. Constrained
(`b' ≤ b_min`) and off-grid (`b' > b_max` or `a' > a_max`) cells are
counted, not scored — same convention as [`_euler_error_stats`](@ref).
"""
function _two_asset_euler_error_stats(c_pol::AbstractArray{T,3},
                                      b_pol::AbstractArray{T,3},
                                      a_pol::AbstractArray{T,3},
                                      ip::IndividualProblem{T},
                                      grid::HAGrid{T},
                                      income::IncomeProcess{T},
                                      prices::Dict{Symbol,T};
                                      points::Symbol=:midpoints) where {T<:AbstractFloat}
    points in (:nodes, :midpoints) ||
        throw(ArgumentError("points must be :nodes or :midpoints; got :$points"))
    b_grid = grid.grids[1]
    a_grid = grid.grids[2]
    n_b = length(b_grid)
    n_a = length(a_grid)
    n_e = length(income.states)
    b_min = ip.borrowing_constraint[1]
    b_max = b_grid[end]
    a_max = a_grid[end]
    beta = ip.beta
    u_prime = ip.utility_prime
    r_b = get(prices, :r_b, prices[:r])
    Pi = income.transition
    constraint_tol = b_min + T(1e-6)

    eval_b = points === :nodes ? b_grid :
             T[(b_grid[i] + b_grid[i + 1]) / 2 for i in 1:(n_b - 1)]
    eval_a = points === :nodes ? a_grid :
             T[(a_grid[i] + a_grid[i + 1]) / 2 for i in 1:(n_a - 1)]

    max_err = zero(T)
    sum_err = zero(T)
    n_checked = 0
    n_constrained = 0
    n_offgrid = 0

    @inbounds for je in 1:n_e
        for (ia, a_pt) in enumerate(eval_a), (ib, b_pt) in enumerate(eval_b)
            if points === :nodes
                c_here = c_pol[ib, ia, je]
                bp = b_pol[ib, ia, je]
                ap = a_pol[ib, ia, je]
            else
                c_here = _bilinear_interp(b_grid, a_grid, view(c_pol, :, :, je), b_pt, a_pt)
                bp = _bilinear_interp(b_grid, a_grid, view(b_pol, :, :, je), b_pt, a_pt)
                ap = _bilinear_interp(b_grid, a_grid, view(a_pol, :, :, je), b_pt, a_pt)
            end
            if bp <= constraint_tol
                n_constrained += 1
                continue
            end
            if bp > b_max || ap > a_max
                n_offgrid += 1
                continue
            end
            emu = zero(T)
            for jep in 1:n_e
                ct = _bilinear_interp(b_grid, a_grid, view(c_pol, :, :, jep), bp, ap)
                emu += Pi[je, jep] * u_prime(max(ct, T(1e-15)))
            end
            up_today = u_prime(max(c_here, T(1e-15)))
            if up_today > zero(T) && isfinite(emu)
                resid = abs(one(T) - beta * (one(T) + r_b) * emu / up_today)
                resid > max_err && (max_err = resid)
                sum_err += resid
                n_checked += 1
            end
        end
    end
    lg(x) = x > zero(T) ? log10(x) : T(-16)
    (points = points,
     max = n_checked == 0 ? T(NaN) : lg(max_err),
     mean = n_checked == 0 ? T(NaN) : lg(sum_err / T(n_checked)),
     n_evaluated = n_checked,
     n_constrained = n_constrained,
     n_offgrid = n_offgrid)
end

# =============================================================================
# Two-asset Young / EGM / VFI household helper
# =============================================================================

"""
    _two_asset_hh_solve(ip, grid, income, prices; hh_solver=:egm, kwargs...)
        → (c, b', a', d, V)

Solve the two-asset household problem at given prices and reconstruct `a'`.
"""
function _two_asset_hh_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                             income::IncomeProcess{T}, prices::Dict{Symbol,T};
                             hh_solver::Symbol=:egm,
                             max_iter::Int=200, tol::T=T(1e-6),
                             howard_steps::Int=10,
                             init_value=nothing) where {T<:AbstractFloat}
    r_a = get(prices, :r_a, prices[:r])
    pol = if hh_solver === :vfi
        _two_asset_vfi_solve(ip, grid, income, prices;
                             max_iter=max_iter, tol=tol, howard_steps=howard_steps,
                             init_value=init_value)
    elseif hh_solver === :egm
        _two_asset_egm_solve(ip, grid, income, prices;
                             max_iter=max_iter, tol=tol, howard_steps=howard_steps,
                             init_value=init_value)
    else
        throw(ArgumentError("_two_asset_hh_solve: hh_solver must be :egm or :vfi, got :$hh_solver"))
    end
    c = pol[:consumption]
    b = pol[:liquid_savings]
    d = pol[:deposit]
    V = pol[:value]
    a_next = _two_asset_a_prime(d, grid, r_a)
    return c, b, a_next, d, V
end

# =============================================================================
# _ha_two_asset_steady_state — damped (K, r_b) closer
# =============================================================================

"""
    _ha_two_asset_steady_state(ip, grid, income, params; kwargs...) → HASteadyState

Stationary equilibrium of a two-asset production economy.

Unknowns `(K, r_b)`. Firm FOCs give `r_a` and `w`; the government budget
sets `τ = r_b * B_supply`. Markets:

- illiquid: `A = ∫a dμ = K`
- liquid: `B = ∫b dμ = B_supply`

Damped updates match [`ct_two_asset_ge`](@ref).
"""
function _ha_two_asset_steady_state(ip::IndividualProblem{T}, grid::HAGrid{T},
                                    income::IncomeProcess{T},
                                    params::Dict{Symbol,T};
                                    K_init::T=T(10),
                                    rb_init::Union{Nothing,T}=nothing,
                                    max_iter::Int=60,
                                    tol::Real=T(1e-4),
                                    relax_K::Real=T(0.3),
                                    relax_rb::Real=T(0.02),
                                    hh_solver::Symbol=:egm,
                                    hh_max_iter::Int=80,
                                    hh_tol::T=T(1e-6),
                                    howard_steps::Int=10,
                                    grid_check::Symbol=:none,
                                    ceiling_mass_tol::Real=T(1e-6),
                                    residual_tol::Real=T(1e-6),
                                    euler_points::Symbol=:midpoints,
                                    verbose::Bool=false,
                                    distribution::Symbol=:young) where {T<:AbstractFloat}
    grid.n_dims == 2 || throw(ArgumentError(
        "_ha_two_asset_steady_state requires n_dims == 2"))
    ip.n_asset_dims == 2 || throw(ArgumentError(
        "_ha_two_asset_steady_state requires n_asset_dims == 2"))
    hh_solver in (:egm, :vfi) || throw(ArgumentError(
        "_ha_two_asset_steady_state: hh_solver must be :egm or :vfi, got :$hh_solver"))
    distribution === :young || throw(ArgumentError(
        "compute_steady_state: two-asset models support distribution=:young only " *
        "(got :$distribution). A joint Winberry family is not implemented."))

    alpha = get(params, :alpha, T(0.36))
    delta = get(params, :delta, T(0.025))
    Z = get(params, :Z, one(T))
    L = get(params, :L, one(T))
    B_supply = get(params, :B_supply, one(T))
    tol_T = T(tol)

    firm_ra(K) = alpha * Z * (K / L)^(alpha - one(T)) - delta
    firm_w(K)  = (one(T) - alpha) * Z * (K / L)^alpha
    firm_Y(K)  = Z * K^alpha * L^(one(T) - alpha)

    # Representative-agent capital is a safe lower start (precautionary saving
    # pushes K above it). Liquid guess: half the illiquid return, as in CT.
    r_cap = one(T) / ip.beta - one(T) - T(1e-4)
    ra0 = firm_ra(max(K_init, T(1e-4)))
    if K_init == T(10) && ra0 + delta > zero(T)
        # Default K_init=10 is the one-asset convention; replace with the
        # representative-agent stock implied by r = 1/β − 1.
        r_ra = max(r_cap, T(1e-4))
        K = max(((alpha * Z) / (r_ra + delta))^(one(T) / (one(T) - alpha)) * L, T(1e-4))
    else
        K = max(K_init, T(1e-4))
    end
    r_b = something(rb_init, clamp(firm_ra(K) / 2, -r_cap, r_cap))

    n_b = grid.n_points[1]
    n_a = grid.n_points[2]
    n_e = grid.n_income

    V_warm = nothing
    local c_pol, b_pol, a_pol, d_pol, V, dist, prices
    resid_a = T(Inf)
    resid_b = T(Inf)
    cleared = false
    final_iter = 0
    K_used = K
    ra_used = firm_ra(K)
    rb_used = r_b
    w_used = firm_w(K)
    tau_used = r_b * B_supply

    for it in 1:max_iter
        final_iter = it
        K = max(K, T(1e-6))
        r_a = firm_ra(K)
        w = firm_w(K)
        # Allow r_b ≷ r_a during the iteration: a hard r_b < r_a floor can pin
        # every household at the liquid borrowing constraint and make B_supply
        # unattainable. Equilibrium still typically has a positive premium.
        r_b = clamp(r_b, -r_cap, r_cap)
        tau = r_b * B_supply
        K_used = K; ra_used = r_a; rb_used = r_b; w_used = w; tau_used = tau
        prices = Dict{Symbol,T}(
            :r => r_a, :r_a => r_a, :r_b => r_b, :w => w,
            :tau => tau, :div => zero(T)
        )
        c_pol, b_pol, a_pol, d_pol, V = _two_asset_hh_solve(
            ip, grid, income, prices; hh_solver=hh_solver,
            max_iter=hh_max_iter, tol=hh_tol, howard_steps=howard_steps,
            init_value=V_warm)
        V_warm = V
        Lambda = _build_transition_matrix(b_pol, a_pol, grid, income)
        dist, _ = _stationary_dist_young(Lambda)
        B = _aggregate(dist, grid; var_index=1)
        A = _aggregate(dist, grid; var_index=2)
        resid_a = A - K
        resid_b = B - B_supply
        if verbose
            @info "two-asset GE $it: r_a=$(round(r_a; sigdigits=5)) r_b=$(round(r_b; sigdigits=5)) " *
                  "K=$(round(K; sigdigits=6)) A−K=$(round(resid_a; sigdigits=3)) " *
                  "B−B̄=$(round(resid_b; sigdigits=3))"
        end
        if abs(resid_a) < tol_T && abs(resid_b) < tol_T
            cleared = true
            break
        end
        K += T(relax_K) * resid_a
        r_b += T(relax_rb) * (B_supply - B)
    end

    B = _aggregate(dist, grid; var_index=1)
    A = _aggregate(dist, grid; var_index=2)
    Y_val = firm_Y(K_used)

    euler_mid = _two_asset_euler_error_stats(c_pol, b_pol, a_pol, ip, grid, income, prices;
                                             points=:midpoints)
    euler_nodes = _two_asset_euler_error_stats(c_pol, b_pol, a_pol, ip, grid, income, prices;
                                               points=:nodes)
    euler_stats = (midpoints=euler_mid, nodes=euler_nodes)
    euler_err = euler_points === :nodes ? euler_nodes.max : euler_mid.max

    gdiag = _ha_grid_diagnostics(b_pol, dist, grid;
                                 ceiling_mass_tol=ceiling_mass_tol,
                                 residual_tol=residual_tol)
    grid_check === :none ||
        _check_grid_adequacy(gdiag, grid_check; context="compute_steady_state(two-asset)")

    policies = Dict{Symbol,Array{T}}(
        :consumption => c_pol,
        :liquid_savings => b_pol,
        :illiquid_savings => a_pol,
        :deposit => d_pol,
        :savings => a_pol
    )
    dist_reshaped = reshape(dist, n_b, n_a, n_e)
    aggregates = Dict{Symbol,T}(
        :K => K_used,
        :A => A,
        :B => B,
        :B_supply => B_supply,
        :Y => Y_val,
        :L => L,
        :excess_demand => max(abs(resid_a), abs(resid_b)),
        :resid_illiquid => resid_a,
        :resid_liquid => resid_b,
        :A_policy => dot(vec(a_pol), dist),
        :B_policy => dot(vec(b_pol), dist),
        :A_residual => gdiag.clearing_residual
    )
    return HASteadyState{T}(
        policies,
        dist_reshaped,
        V,
        prices,
        aggregates,
        grid,
        income,
        cleared,
        final_iter,
        euler_err,
        aggregates[:excess_demand];
        euler=euler_stats
    )
end
