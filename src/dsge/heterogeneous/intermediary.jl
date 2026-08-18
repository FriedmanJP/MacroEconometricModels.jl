# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heterogeneous financial intermediaries — Jamilov–Monacelli Bewley Banks.

A bank with net worth `n` and idiosyncratic return draw `ξ` chooses lending `l`
and deposits `b` subject to the balance-sheet identity `l = n + b` and the
Gertler–Karadi incentive constraint `λ l ≤ V`. Convex operating costs
`ζ₁ l^{ζ₂}` (`ζ₂ > 1`) break the scale invariance of the representative
Gertler–Karadi (2011) bank, so the net-worth distribution is a state.

The first fixture is partial-equilibrium franchise-value VFI given `(R, rᵏ)`,
one permanent type `κ`, Young (2010) stationary credit-market clearing, and a
MIT TFP impulse of aggregate lending `L` (or output `Y`). Gertler–Karadi 2011
is the nested representative special case `ζ₁ = 0` with degenerate `ξ`, not a
separate type. Deposit market power, countercyclical `ξ`, too-big-to-fail, and
HANK + banks + firms in one `solve` are out of scope.

# References
- Jamilov, R., & Monacelli, T. (2026). Bewley Banks. *Review of Economic
  Studies*, 93(3), 1889–1925. https://doi.org/10.1093/restud/rdaf062
- Gertler, M., & Karadi, P. (2011). A model of unconventional monetary policy.
  *Journal of Monetary Economics*, 58(1), 17–34.
- Gertler, M., & Kiyotaki, N. (2010). Financial intermediation and credit
  policy in business cycle analysis. In *Handbook of Monetary Economics*
  (Vol. 3, pp. 547–599).
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# IntermediarySystem — own payload (not HouseholdSystem / IndividualProblem)
# =============================================================================

"""
    IntermediarySystem{T} <: AbstractAgentSystem{T}

Bewley-bank population (Jamilov–Monacelli, *REStud* 93(3), 2026).
The `ModelSpec.agents` NamedTuple key is the population name (`:banks`,
`:dealers`, …); this type is the problem kind.

State is `(n, ξ)` on an [`HAGrid`](@ref) (net worth) plus a Rouwenhorst
[`IncomeProcess`](@ref) for transitory return risk. Banks maximize franchise
value by VFI, not household `u(c)`.

# Fields
- `grid` — net-worth grid (`labels = [:assets]`)
- `xi` — idiosyncratic return Markov chain (levels, not logs)
- `kappa` — one permanent return type `κ`
- `beta`, `sigma` — discount factor and survival rate
- `lambda` — divertable fraction in the GK incentive constraint `λ l ≤ V`
- `zeta1`, `zeta2` — operating-cost scale and exponent (`ζ₂ > 1`)
- `R`, `rk` — PE deposit rate (gross) and net return on claims
- `Z`, `alpha` — TFP and capital share for credit-market closing
- `n_enter` — start-up net worth of replacement banks
- `model` — `:bewley_banks` (Gertler–Karadi is the `zeta1 = 0` nest)
- `distribution` — `:young`
"""
struct IntermediarySystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    grid::HAGrid{T}
    xi::IncomeProcess{T}
    kappa::T
    beta::T
    sigma::T
    lambda::T
    zeta1::T
    zeta2::T
    R::T
    rk::T
    Z::T
    alpha::T
    n_enter::T
    het_params::Dict{Symbol,T}
    aggregation::Vector{Pair{Symbol,Function}}
    model::Symbol
    distribution::Symbol
end

grid(s::IntermediarySystem) = s.grid
idiosyncratic(s::IntermediarySystem) = s.xi
aggregation(s::IntermediarySystem) = s.aggregation
distribution(s::IntermediarySystem) = s.distribution
het_params(s::IntermediarySystem) = s.het_params
ssj_inputs(::IntermediarySystem) = [:R, :rk]
ssj_outputs(s::IntermediarySystem) = first.(s.aggregation)

"""
    _xi_process(rho, sigma, n; mu=1) → IncomeProcess

Rouwenhorst chain for transitory bank return risk. States are **levels**
`μ · exp(y)` of an AR(1) in logs, so `ξ > 0` and `E[ξ] ≈ μ`. A single state
is the degenerate chain `ξ = μ` (Gertler–Karadi nest, `ξ` off).
"""
function _xi_process(rho::Real, sigma::Real, n::Int; mu::Real=1.0)
    T = Float64
    mu_T = T(mu)
    mu_T > zero(T) || throw(ArgumentError("IntermediarySystem: xi mean must be positive"))
    n >= 1 || throw(ArgumentError("IntermediarySystem: n_xi must be ≥ 1"))
    if n == 1
        return IncomeProcess{T}(ones(T, 1, 1), T[mu_T], T[one(T)], :xi)
    end
    sigma_T = T(sigma)
    sigma_T > zero(T) || throw(ArgumentError(
        "IntermediarySystem: sigma_xi must be positive when n_xi > 1 (got $sigma)"))
    proc = rouwenhorst(T(rho), sigma_T, n)
    levels = mu_T .* exp.(proc.states)
    return IncomeProcess{T}(proc.transition, levels, proc.stationary_dist, :xi)
end

"""
    IntermediarySystem(; kwargs...) → IntermediarySystem{Float64}

Keyword constructor for a Bewley-bank population.

# Keywords
- `n_min`, `n_max`, `n_n` — net-worth grid (default `0.05`, `8.0`, `25`)
- `n_xi`, `rho_xi`, `sigma_xi` — Rouwenhorst `ξ` (default `3`, `0.553`, `0.085`)
- `kappa` — permanent type (default `1`)
- `beta`, `sigma`, `lambda` — discount, survival, diversion (GK defaults)
- `zeta1`, `zeta2` — convex operating cost (default `0.02`, `2`)
- `R`, `rk` — PE prices (gross deposit rate, net claim return)
- `Z`, `alpha` — TFP and capital share for `Kᵈ(rᵏ) = (α Z / rᵏ)^{1/(1-α)}`
- `n_enter` — replacement-bank net worth (default `n_min`)
- `grid_type` — passed to [`HAGrid`](@ref) (default `:geometric`)
"""
function IntermediarySystem(;
        n_min::Real=0.05,
        n_max::Real=8.0,
        n_n::Int=25,
        n_xi::Int=3,
        rho_xi::Real=0.553,
        sigma_xi::Real=0.085,
        xi_mean::Real=1.0,
        kappa::Real=1.0,
        beta::Real=0.99,
        sigma::Real=0.95,
        lambda::Real=0.20,
        zeta1::Real=0.02,
        zeta2::Real=2.0,
        R::Real=1.01,
        rk::Real=0.05,
        Z::Real=0.25,
        alpha::Real=0.33,
        n_enter::Union{Nothing,Real}=nothing,
        grid_type::Symbol=:geometric,
        model::Symbol=:bewley_banks,
        distribution::Symbol=:young,
        aggregation::Union{Nothing,Vector{Pair{Symbol,Function}}}=nothing,
        het_params::Union{Nothing,Dict{Symbol,Float64}}=nothing)
    T = Float64
    zero(T) < T(beta) < one(T) || throw(ArgumentError(
        "IntermediarySystem: beta must lie in (0, 1), got $beta"))
    zero(T) < T(sigma) <= one(T) || throw(ArgumentError(
        "IntermediarySystem: sigma (survival) must lie in (0, 1], got $sigma"))
    T(lambda) > zero(T) || throw(ArgumentError(
        "IntermediarySystem: lambda must be positive, got $lambda"))
    T(zeta1) >= zero(T) || throw(ArgumentError(
        "IntermediarySystem: zeta1 must be non-negative, got $zeta1"))
    T(zeta2) > one(T) || throw(ArgumentError(
        "IntermediarySystem: zeta2 must exceed 1 (convex cost), got $zeta2"))
    T(kappa) > zero(T) || throw(ArgumentError(
        "IntermediarySystem: kappa must be positive, got $kappa"))
    T(R) > zero(T) || throw(ArgumentError(
        "IntermediarySystem: R must be positive, got $R"))
    T(Z) > zero(T) || throw(ArgumentError(
        "IntermediarySystem: Z must be positive, got $Z"))
    zero(T) < T(alpha) < one(T) || throw(ArgumentError(
        "IntermediarySystem: alpha must lie in (0, 1), got $alpha"))
    distribution === :young || throw(ArgumentError(
        "IntermediarySystem: distribution must be :young, got :$distribution"))
    T(n_min) > zero(T) || throw(ArgumentError(
        "IntermediarySystem: n_min must be positive (no negative equity)"))
    T(n_max) > T(n_min) || throw(ArgumentError(
        "IntermediarySystem: n_max must exceed n_min"))

    xi = _xi_process(rho_xi, sigma_xi, n_xi; mu=xi_mean)
    g = HAGrid(; assets=(T(n_min), T(n_max), n_n), income_states=n_xi,
               grid_type=grid_type)
    n0 = n_enter === nothing ? T(n_min) : T(n_enter)
    n0 >= T(n_min) || throw(ArgumentError(
        "IntermediarySystem: n_enter must be ≥ n_min"))

    hp = het_params === nothing ? Dict{Symbol,T}(
        :kappa => T(kappa), :lambda => T(lambda),
        :zeta1 => T(zeta1), :zeta2 => T(zeta2),
        :sigma => T(sigma), :beta => T(beta)) : het_params
    agg = aggregation === nothing ? Pair{Symbol,Function}[
        :L => (l, d) -> sum(l .* d),
        :N => (n, d) -> sum(n .* d),
    ] : aggregation

    return IntermediarySystem{T}(
        g, xi, T(kappa), T(beta), T(sigma), T(lambda), T(zeta1), T(zeta2),
        T(R), T(rk), T(Z), T(alpha), n0, hp, agg, model, distribution)
end

# =============================================================================
# to_spec
# =============================================================================

"""
    to_spec(sys::IntermediarySystem; agent_name=:banks) → ModelSpec

Wrap an [`IntermediarySystem`](@ref) as a [`ModelSpec`](@ref) with an empty
aggregate residual block (partial GE). `has_kind(spec, IntermediarySystem)`
is the dispatch test; the key is a free name. `solve` / `compute_steady_state`
route to [`intermediary_steady_state`](@ref).
"""
function to_spec(sys::IntermediarySystem{T}; agent_name::Symbol=:banks) where {T<:AbstractFloat}
    params = [:beta, :sigma, :lambda, :zeta1, :zeta2, :kappa, :R, :rk, :Z, :alpha]
    param_values = Dict{Symbol,T}(
        :beta => sys.beta, :sigma => sys.sigma, :lambda => sys.lambda,
        :zeta1 => sys.zeta1, :zeta2 => sys.zeta2, :kappa => sys.kappa,
        :R => sys.R, :rk => sys.rk, :Z => sys.Z, :alpha => sys.alpha,
    )
    return ModelSpec{T}(
        Symbol[], Symbol[], params, param_values,
        NamedEquation[], Function[],
        0, Int[], T[];
        agents=NamedTuple{(agent_name,)}((sys,)),
    )
end

# =============================================================================
# Franchise-value algebra
# =============================================================================

"""Net-worth law of motion `n' = Rᵀ l − R b − ζ₁ l^{ζ₂}` with `b = l − n`."""
@inline function _bank_nprime(n::T, l::T, xi_next::T, kappa::T, rk::T, R::T,
                              zeta1::T, zeta2::T) where {T<:AbstractFloat}
    RT = one(T) + kappa * xi_next * rk
    return RT * l - R * (l - n) - zeta1 * (l <= zero(T) ? zero(T) : l^zeta2)
end

"""Expected franchise payoff of lending `l` at `(n, ξ_j)` given `V`."""
function _bank_payoff(l::T, n::T, j::Int, V::AbstractMatrix{T},
                      n_grid::AbstractVector{T}, xi::IncomeProcess{T},
                      kappa::T, rk::T, R::T, zeta1::T, zeta2::T,
                      beta::T, sigma::T) where {T<:AbstractFloat}
    l < zero(T) && return T(-Inf)
    acc = zero(T)
    Pi = xi.transition
    n_e = length(xi.states)
    n_lo = n_grid[1]
    @inbounds for jp in 1:n_e
        p = Pi[j, jp]
        p < T(1e-20) && continue
        np = _bank_nprime(n, l, xi.states[jp], kappa, rk, R, zeta1, zeta2)
        if np <= zero(T)
            # limited liability: franchise is wiped out
            continue
        end
        np_c = clamp(np, n_lo, n_grid[end])
        Vp = _linear_interp(n_grid, view(V, :, jp), np_c)
        acc += p * ((one(T) - sigma) * np + sigma * Vp)
    end
    return beta * acc
end

"""Best lending on `[0, l_hi]` by a coarse scan plus golden-section refine."""
function _best_lending(n::T, j::Int, l_hi::T, V::AbstractMatrix{T},
                       n_grid::AbstractVector{T}, xi::IncomeProcess{T},
                       kappa::T, rk::T, R::T, zeta1::T, zeta2::T,
                       beta::T, sigma::T; n_try::Int=16) where {T<:AbstractFloat}
    l_hi = max(l_hi, zero(T))
    pay = l -> _bank_payoff(l, n, j, V, n_grid, xi, kappa, rk, R,
                            zeta1, zeta2, beta, sigma)
    best_l = zero(T)
    best_v = pay(zero(T))
    if l_hi > zero(T)
        @inbounds for k in 1:n_try
            l = l_hi * T(k) / T(n_try)
            v = pay(l)
            if v > best_v
                best_v = v
                best_l = l
            end
        end
        step = l_hi / T(n_try)
        lo = max(zero(T), best_l - step)
        hi = min(l_hi, best_l + step)
        if hi > lo + T(1e-14)
            l_r, v_r = _golden_argmax(pay, lo, hi)
            if isfinite(v_r) && v_r >= best_v
                return l_r, v_r
            end
        end
    end
    return best_l, best_v
end

# =============================================================================
# PE franchise-value VFI
# =============================================================================

"""
    IntermediaryPE{T}

Partial-equilibrium franchise value and policies given `(R, rᵏ)`.
"""
struct IntermediaryPE{T<:AbstractFloat}
    V::Matrix{T}
    l_policy::Matrix{T}
    b_policy::Matrix{T}
    prices::Dict{Symbol,T}
    converged::Bool
    iterations::Int
end

"""
    intermediary_pe(sys; R=sys.R, rk=sys.rk, max_iter=250, tol=1e-6,
                    howard_steps=8, init_value=nothing) → IntermediaryPE

Solve `V(n, ξ)` by franchise-value VFI (Howard-accelerated) at given prices.
The incentive constraint is `l ≤ V/λ` using the previous iterate of `V`.
Does **not** use household EGM / [`IndividualProblem`](@ref).
"""
function intermediary_pe(sys::IntermediarySystem{T};
                         R::Real=sys.R,
                         rk::Real=sys.rk,
                         max_iter::Int=250,
                         tol::Real=T(1e-6),
                         howard_steps::Int=8,
                         init_value::Union{Nothing,AbstractMatrix{T}}=nothing,
                         n_try::Int=16) where {T<:AbstractFloat}
    n_grid = sys.grid.grids[1]
    n_n = length(n_grid)
    n_e = length(sys.xi.states)
    R_T = T(R)
    rk_T = T(rk)
    R_T > zero(T) || throw(ArgumentError("intermediary_pe: R must be positive"))
    kappa, zeta1, zeta2 = sys.kappa, sys.zeta1, sys.zeta2
    beta, sigma, λ = sys.beta, sys.sigma, sys.lambda

    V = zeros(T, n_n, n_e)
    if init_value !== nothing && size(init_value) == (n_n, n_e)
        copyto!(V, init_value)
    else
        # scale-free GK guess V ≈ ν n; ν = 0.4 gives room under λ l ≤ V
        @inbounds for j in 1:n_e, i in 1:n_n
            V[i, j] = T(0.4) * n_grid[i]
        end
    end

    l_pol = zeros(T, n_n, n_e)
    V_new = similar(V)
    converged = false
    final_iter = 0
    l_cap = T(20) * n_grid[end]

    for iter in 1:max_iter
        final_iter = iter
        @inbounds for j in 1:n_e, i in 1:n_n
            n = n_grid[i]
            l_ic = V[i, j] / λ
            l_hi = min(max(l_ic, zero(T)), l_cap)
            l_star, v_star = _best_lending(n, j, l_hi, V, n_grid, sys.xi,
                                           kappa, rk_T, R_T, zeta1, zeta2,
                                           beta, sigma; n_try=n_try)
            if !isfinite(v_star)
                l_star = zero(T)
                v_star = _bank_payoff(zero(T), n, j, V, n_grid, sys.xi,
                                      kappa, rk_T, R_T, zeta1, zeta2, beta, sigma)
            end
            # enforce IC on the reported policy
            l_star = min(l_star, l_hi)
            l_pol[i, j] = l_star
            V_new[i, j] = _bank_payoff(l_star, n, j, V, n_grid, sys.xi,
                                       kappa, rk_T, R_T, zeta1, zeta2, beta, sigma)
        end

        for _h in 1:howard_steps
            @inbounds for j in 1:n_e, i in 1:n_n
                V_new[i, j] = _bank_payoff(l_pol[i, j], n_grid[i], j, V_new,
                                           n_grid, sys.xi, kappa, rk_T, R_T,
                                           zeta1, zeta2, beta, sigma)
            end
        end

        max_diff = zero(T)
        @inbounds for j in 1:n_e, i in 1:n_n
            δ = abs(V_new[i, j] - V[i, j])
            if !isfinite(δ)
                max_diff = T(Inf)
                break
            elseif δ > max_diff
                max_diff = δ
            end
        end
        copyto!(V, V_new)
        if isfinite(max_diff) && max_diff < T(tol)
            converged = true
            break
        end
    end
    if !converged
        @warn "intermediary VFI did not converge after $max_iter iterations (||ΔV||_∞ = $max_diff, tol = $tol)"
    end

    b_pol = l_pol .- n_grid
    return IntermediaryPE{T}(V, l_pol, b_pol,
                             Dict{Symbol,T}(:R => R_T, :rk => rk_T),
                             converged, final_iter)
end

# =============================================================================
# Young histogram of (n, ξ) with ξ-dependent n' and GK exit/entry
# =============================================================================

"""
    _intermediary_transition(l_pol, sys, R, rk) → SparseMatrixCSC

Young (2010) lottery on `n'(n, ξ, ξ')`, times the `ξ` chain, times survival
`σ`. Mass `1 − σ` is replaced by new banks at `n_enter` with `ξ ~ π*`.
"""
function _intermediary_transition(l_pol::AbstractMatrix{T},
                                  sys::IntermediarySystem{T},
                                  R::T, rk::T) where {T<:AbstractFloat}
    n_grid = sys.grid.grids[1]
    n_n = length(n_grid)
    n_e = length(sys.xi.states)
    N = n_n * n_e
    Pi = sys.xi.transition
    pi_stat = sys.xi.stationary_dist
    sigma = sys.sigma
    kappa, zeta1, zeta2 = sys.kappa, sys.zeta1, sys.zeta2
    n_enter = clamp(sys.n_enter, n_grid[1], n_grid[end])

    max_nnz = 2 * n_e * N + 2 * n_e * N
    rows = Vector{Int}(undef, max_nnz)
    cols = Vector{Int}(undef, max_nnz)
    vals = Vector{T}(undef, max_nnz)
    count = 0
    @inline function _push!(r, c, v)
        if v > T(1e-20)
            count += 1
            rows[count] = r
            cols[count] = c
            vals[count] = v
        end
    end

    ke, we_lo, we_hi = _young_bracket(n_grid, n_enter)
    @inbounds for j in 1:n_e, i in 1:n_n
        col = (j - 1) * n_n + i
        n = n_grid[i]
        l = l_pol[i, j]
        default_mass = zero(T)
        for jp in 1:n_e
            p = Pi[j, jp]
            p < T(1e-20) && continue
            np = _bank_nprime(n, l, sys.xi.states[jp], kappa, rk, R, zeta1, zeta2)
            if np <= zero(T)
                default_mass += p
                continue
            end
            np = clamp(np, n_grid[1], n_grid[end])
            k, w_lo, w_hi = _young_bracket(n_grid, np)
            wσ = sigma * p
            _push!((jp - 1) * n_n + k, col, wσ * w_lo)
            _push!((jp - 1) * n_n + k + 1, col, wσ * w_hi)
        end
        entry_w = (one(T) - sigma) + sigma * default_mass
        if entry_w > T(1e-20)
            for jp in 1:n_e
                ps = pi_stat[jp]
                ps < T(1e-20) && continue
                _push!((jp - 1) * n_n + ke, col, entry_w * ps * we_lo)
                _push!((jp - 1) * n_n + ke + 1, col, entry_w * ps * we_hi)
            end
        end
    end
    resize!(rows, count)
    resize!(cols, count)
    resize!(vals, count)
    return sparse(rows, cols, vals, N, N)
end

function _reshape_dist(d::AbstractVector{T}, n_n::Int, n_e::Int) where {T}
    return reshape(d, n_n, n_e)
end

function _agg_from_policy(l_pol::AbstractMatrix{T}, n_grid::AbstractVector{T},
                          d::AbstractMatrix{T}, Z::T, alpha::T) where {T<:AbstractFloat}
    L = zero(T)
    Nbar = zero(T)
    n_n, n_e = size(l_pol)
    @inbounds for j in 1:n_e, i in 1:n_n
        m = d[i, j]
        L += l_pol[i, j] * m
        Nbar += n_grid[i] * m
    end
    B = L - Nbar
    lev = Nbar > T(1e-14) ? L / Nbar : T(NaN)
    Y = Z * (L > zero(T) ? L^alpha : zero(T))
    return Dict{Symbol,T}(:L => L, :N => Nbar, :B => B, :leverage => lev, :Y => Y)
end

"""Capital demand `Kᵈ(rᵏ) = (α Z / rᵏ)^{1/(1-α)}` (H = 1, Q = 1)."""
function _capital_demand(alpha::T, Z::T, rk::T) where {T<:AbstractFloat}
    rk > zero(T) || return T(Inf)
    return (alpha * Z / rk)^(one(T) / (one(T) - alpha))
end

# =============================================================================
# Stationary credit-market equilibrium
# =============================================================================

"""
    IntermediarySteadyState{T}

Stationary Bewley-bank equilibrium: franchise value, lending policy, Young
histogram over `(n, ξ)`, prices, and aggregates (`L`, `N`, leverage, `Y`).
"""
struct IntermediarySteadyState{T<:AbstractFloat}
    system::IntermediarySystem{T}
    V::Matrix{T}
    l_policy::Matrix{T}
    b_policy::Matrix{T}
    distribution::Matrix{T}
    prices::Dict{Symbol,T}
    aggregates::Dict{Symbol,T}
    grid::HAGrid{T}
    xi::IncomeProcess{T}
    converged::Bool
    iterations::Int
    excess_demand::T
end

"""
    intermediary_steady_state(sys; r_bounds, tol, max_iter, pe_kwargs...)

Bisection on `rᵏ` so bank lending `L = ∫ l(n, ξ) dΓ` equals firm capital
demand `Kᵈ(rᵏ)`. `R` is held at `sys.R` (no deposit-market power).
"""
function intermediary_steady_state(sys::IntermediarySystem{T};
                                   r_bounds::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                                   tol::Real=T(1e-4),
                                   max_iter::Int=24,
                                   pe_max_iter::Int=200,
                                   pe_tol::Real=T(1e-6),
                                   howard_steps::Int=8) where {T<:AbstractFloat}
    R = sys.R
    lo = r_bounds === nothing ? max(R - one(T) + T(0.002), T(0.005)) : T(r_bounds[1])
    hi = r_bounds === nothing ? T(0.60) : T(r_bounds[2])
    hi > lo || throw(ArgumentError(
        "intermediary_steady_state: r_bounds must satisfy lo < hi"))

    V_warm = Ref{Union{Nothing,Matrix{T}}}(nothing)
    pe_last = Ref{Union{Nothing,IntermediaryPE{T}}}(nothing)
    d_last = Ref{Matrix{T}}(fill(one(T) / (length(sys.grid.grids[1]) *
                                           length(sys.xi.states)),
                                 length(sys.grid.grids[1]), length(sys.xi.states)))
    L_last = Ref(zero(T))

    function _eval(rk::T)
        pe = intermediary_pe(sys; R=R, rk=rk, max_iter=pe_max_iter, tol=pe_tol,
                             howard_steps=howard_steps, init_value=V_warm[])
        V_warm[] = pe.V
        pe_last[] = pe
        Λ = _intermediary_transition(pe.l_policy, sys, R, rk)
        dvec, _ = _stationary_dist_young(Λ)
        d = _reshape_dist(dvec, length(sys.grid.grids[1]), length(sys.xi.states))
        d_last[] = d
        L = _agg_from_policy(pe.l_policy, sys.grid.grids[1], d, sys.Z, sys.alpha)[:L]
        L_last[] = L
        Kd = _capital_demand(sys.alpha, sys.Z, rk)
        return L - Kd
    end

    flo = _eval(lo)
    fhi = _eval(hi)
    # Expand the bracket if both residuals have the same sign.
    expand = 0
    while flo * fhi > zero(T) && expand < 6
        if flo < zero(T)
            hi = min(hi * T(1.6), T(2.0))
            fhi = _eval(hi)
        else
            lo = max(lo * T(0.6), T(1e-4))
            flo = _eval(lo)
        end
        expand += 1
    end

    rk = hi
    excess = fhi
    it = 0
    ok = false
    if flo * fhi <= zero(T)
        a, b = lo, hi
        fa, fb = flo, fhi
        for k in 1:max_iter
            it = k
            rk = (a + b) / T(2)
            excess = _eval(rk)
            if abs(excess) < T(tol) || (b - a) < T(tol) * max(one(T), abs(rk))
                ok = true
                break
            end
            if fa * excess <= zero(T)
                b, fb = rk, excess
            else
                a, fa = rk, excess
            end
        end
        ok = ok || abs(excess) < T(10) * T(tol)
    else
        # No sign change: keep the endpoint with smaller |excess|.
        it = expand
        if abs(flo) < abs(fhi)
            rk = lo
            excess = flo
        else
            rk = hi
            excess = fhi
        end
        ok = false
        @warn "intermediary_steady_state: no sign change in the r^k bracket " *
              "[$(lo), $(hi)]; credit market residual at returned point = $(excess). " *
              "Widen r_bounds or check the calibration."
    end

    pe = pe_last[]
    pe === nothing && throw(ErrorException("intermediary_steady_state: no PE evaluate"))
    d = d_last[]
    aggs = _agg_from_policy(pe.l_policy, sys.grid.grids[1], d, sys.Z, sys.alpha)
    return IntermediarySteadyState{T}(
        sys, pe.V, pe.l_policy, pe.b_policy, d,
        Dict{Symbol,T}(:R => R, :rk => rk),
        aggs, sys.grid, sys.xi, ok && pe.converged, it, excess)
end

# =============================================================================
# MIT TFP impulse (predetermined capital, Young forward)
# =============================================================================

"""
    IntermediaryTransition{T}

MIT path of a Bewley-bank economy: TFP `{Z_t}`, predetermined capital, lending,
output, and the claim return.
"""
struct IntermediaryTransition{T<:AbstractFloat}
    Z::Vector{T}
    L::Vector{T}
    Y::Vector{T}
    K::Vector{T}
    rk::Vector{T}
    ss::IntermediarySteadyState{T}
    method::Symbol
    converged::Bool
end

"""
    intermediary_mit(ss, Z_path; pe_max_iter=80, pe_tol=1e-5) → IntermediaryTransition

One-pass MIT shock. Capital is predetermined (`K₁ = L_ss`, `K_{t+1} = L_t`);
`rᵏ_t = α Z_t K_t^{α-1}`; banks reoptimize franchise value at each `t` (warm
started) and the Young histogram is stepped with [`_forward_iterate`](@ref).
"""
function intermediary_mit(ss::IntermediarySteadyState{T}, Z_path::AbstractVector;
                          pe_max_iter::Int=80,
                          pe_tol::Real=T(1e-5),
                          howard_steps::Int=6) where {T<:AbstractFloat}
    Z = collect(T, Z_path)
    length(Z) >= 2 || throw(ArgumentError("intermediary_mit: Z_path needs at least 2 points"))
    all(>(zero(T)), Z) || throw(ArgumentError("intermediary_mit: every TFP value must be positive"))
    sys = ss.system
    H = length(Z)
    K = zeros(T, H)
    L = zeros(T, H)
    Y = zeros(T, H)
    rk = zeros(T, H)
    K[1] = max(ss.aggregates[:L], T(1e-8))
    d = copy(ss.distribution)
    V = copy(ss.V)
    R = ss.prices[:R]
    n_grid = sys.grid.grids[1]
    for t in 1:H
        Kt = max(K[t], T(1e-8))
        rk[t] = sys.alpha * Z[t] * Kt^(sys.alpha - one(T))
        pe = intermediary_pe(sys; R=R, rk=rk[t], max_iter=pe_max_iter,
                             tol=pe_tol, howard_steps=howard_steps, init_value=V)
        V = pe.V
        L[t] = _agg_from_policy(pe.l_policy, n_grid, d, Z[t], sys.alpha)[:L]
        Y[t] = Z[t] * Kt^sys.alpha
        if t < H
            Λ = _intermediary_transition(pe.l_policy, sys, R, rk[t])
            dvec = _forward_iterate(Λ, vec(d))
            d = _reshape_dist(dvec, size(d, 1), size(d, 2))
            K[t + 1] = max(L[t], T(1e-8))
        end
    end
    ok = all(isfinite, L) && all(isfinite, Y) && all(isfinite, rk)
    return IntermediaryTransition{T}(Z, L, Y, K, rk, ss, :mit, ok)
end

"""
    irf(ss::IntermediarySteadyState, horizon; shock_size=0.01, persist=0.5)

MIT impulse of `L`, `Y`, and `Z` in deviations from the stationary point.
`irf(spec, H)` on a spec that `has_kind(..., IntermediarySystem)` goes through
`solve` then this method.
"""
function irf(ss::IntermediarySteadyState{T}, horizon::Int;
             shock_size::Real=T(0.01), persist::Real=T(0.5),
             kwargs...) where {T<:AbstractFloat}
    Zpath = _ct_z_path(ss.system.Z, horizon, shock_size, persist)
    tr = intermediary_mit(ss, Zpath; kwargs...)
    Lbar = ss.aggregates[:L]
    Ybar = ss.aggregates[:Y]
    vals = hcat(tr.L .- Lbar, tr.Y .- Ybar, tr.Z .- ss.system.Z)
    return _path_to_irf(vals, ("L", "Y", "Z"), "Z")
end

# =============================================================================
# solve / compute_steady_state dispatch (gensys.jl is mutex — extend here)
# =============================================================================

# gensys.jl is mutex and owns `_solve_by_agent_kind(::ModelSpec)`. A method
# whose `A` is exactly one `IntermediarySystem` is more specific and does not
# overwrite the generic router (precompile forbids overwrite).
function _solve_by_agent_kind(spec::ModelSpec{T, NamedTuple{K, Tuple{IntermediarySystem{T}}}};
                              kwargs...) where {T<:AbstractFloat, K}
    return intermediary_steady_state(only(values(spec.agents)); kwargs...)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, sys::IntermediarySystem{T}) where {T}
    n_n = sys.grid.n_points[1]
    n_e = length(sys.xi.states)
    print(io, "IntermediarySystem{$T}: Bewley Banks, n×ξ = ", n_n, "×", n_e,
          ", κ = ", sys.kappa, ", λ = ", sys.lambda,
          ", ζ₁ = ", sys.zeta1)
end

function Base.show(io::IO, pe::IntermediaryPE{T}) where {T}
    print(io, "IntermediaryPE{$T}: VFI ", pe.converged ? "converged" : "not converged",
          " in ", pe.iterations, " iters, rᵏ = ",
          round(pe.prices[:rk]; digits=4))
end

function Base.show(io::IO, ss::IntermediarySteadyState{T}) where {T}
    print(io, "IntermediarySteadyState{$T}: L = ",
          round(ss.aggregates[:L]; digits=4),
          ", leverage = ", round(ss.aggregates[:leverage]; digits=3),
          ", rᵏ = ", round(ss.prices[:rk]; digits=4),
          ", converged = ", ss.converged)
end

function report(io::IO, ss::IntermediarySteadyState{T}) where {T}
    println(io, "IntermediarySteadyState{$T}")
    println(io, "  Converged            ", ss.converged ? "Yes" : "No")
    println(io, "  Iterations           ", ss.iterations)
    println(io, "  r^k                  ", ss.prices[:rk])
    println(io, "  L                    ", ss.aggregates[:L])
    println(io, "  Excess demand        ", ss.excess_demand)
end
report(ss::IntermediarySteadyState) = report(stdout, ss)

function Base.show(io::IO, tr::IntermediaryTransition{T}) where {T}
    print(io, "IntermediaryTransition{$T}: ", length(tr.Z), " periods, method=:",
          tr.method, ", L: ", round(tr.L[1]; digits=4), " → ",
          round(tr.L[end]; digits=4), ", converged=", tr.converged)
end
