# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Blanchard (1985) perpetual-youth overlapping-generations model (discrete time).

A closed-economy neoclassical growth model in which agents survive each period with
probability `γ ∈ (0,1]` and newborns enter with zero financial wealth. Fair annuities pay
survivors the gross return `(1+r)/γ`. With log utility the marginal propensity to consume
out of total wealth is `1 − βγ`, and aggregating the individual consumption rule across
cohorts (eliminating human wealth) yields the aggregate system

    Euler:  C_{t+1} = (1+r_{t+1}) · [ β C_t − λ (k_{t+1} + b) ],   λ = (1−βγ)(1−γ)/γ
    Budget: k_{t+1} = (1+r_t) k_t + w_t − C_t
    Prices: r_t = α Z k_t^{α−1} − δ,   w_t = (1−α) Z k_t^α

The `(1−γ)` term is the Blanchard correction from generational turnover. When `γ = 1` it
vanishes and the model collapses to the representative-agent Ramsey economy with
`r* = 1/β − 1`. For `γ < 1` the steady state requires `β(1+r) > 1`, i.e. `r* > 1/β − 1`:
finite horizons push the interest rate above the pure rate of time preference. Government
debt `b` is net wealth, so it crowds out capital (`∂k*/∂b < 0`, `∂r*/∂b > 0`) — the
failure of Ricardian equivalence.

# References
- Blanchard, O. J. (1985). Debt, Deficits, and Finite Horizons. *Journal of Political
  Economy*, 93(2), 223–247.
- Yaari, M. E. (1965). Uncertain Lifetime, Life Insurance, and the Theory of the Consumer.
  *Review of Economic Studies*, 32(2), 137–150.
- Fujiwara, I., & Teranishi, Y. (2008). A dynamic new Keynesian life-cycle model.
  *Journal of Economic Dynamics and Control*, 32(7), 2398–2427.
"""

using LinearAlgebra

# =============================================================================
# Types
# =============================================================================

"""
    BlanchardOLG{T}

Discrete-time Blanchard (1985) perpetual-youth model parameters.

# Fields
- `alpha::T` — capital share
- `beta::T` — discount factor (pure time preference)
- `delta::T` — depreciation rate
- `gamma::T` — per-period survival probability (`1` = representative-agent Ramsey)
- `Z::T` — total factor productivity
- `b::T` — per-capita government debt (net wealth; `0` = no debt)
"""
struct BlanchardOLG{T<:AbstractFloat}
    alpha::T
    beta::T
    delta::T
    gamma::T
    Z::T
    b::T
end

function BlanchardOLG(; alpha::Real=0.36, beta::Real=0.96, delta::Real=0.08,
                        gamma::Real=0.98, Z::Real=1.0, b::Real=0.0)
    @assert 0 < gamma <= 1 "survival probability gamma must be in (0, 1]"
    @assert 0 < beta < 1 "beta must be in (0, 1)"
    @assert 0 < alpha < 1 "alpha must be in (0, 1)"
    T = promote_type(typeof(alpha), typeof(beta), typeof(delta),
                     typeof(gamma), typeof(Z), typeof(b), Float64)
    return BlanchardOLG{T}(T(alpha), T(beta), T(delta), T(gamma), T(Z), T(b))
end

"""
    BlanchardOLGSteadyState{T}

Steady state of a Blanchard OLG model: capital `k`, consumption `C`, interest rate `r`,
wage `w`, aggregate human wealth `H`, the log-utility MPC `mpc = 1 − βγ`, debt `b`.
"""
struct BlanchardOLGSteadyState{T<:AbstractFloat}
    k::T
    C::T
    r::T
    w::T
    H::T
    mpc::T
    b::T
    converged::Bool
end

"""
    BlanchardOLGSolution{T}

Local (saddle-path) solution of the Blanchard OLG model. `M` is the 2×2 linearized
transition of `(k − k*, C − C*)`; `stable_eig` is the saddle-path (modulus < 1) eigenvalue
governing convergence; `policy_slope` is `dC/dk` along the saddle path; `determinate` is
true when exactly one eigenvalue lies inside the unit circle.
"""
struct BlanchardOLGSolution{T<:AbstractFloat}
    ss::BlanchardOLGSteadyState{T}
    M::Matrix{T}
    eigenvalues::Vector{ComplexF64}
    stable_eig::T
    policy_slope::T
    determinate::Bool
end

# =============================================================================
# Prices and steady state
# =============================================================================

# Competitive factor prices at capital k.
function _blanchard_prices(m::BlanchardOLG{T}, k::T) where {T<:AbstractFloat}
    r = m.alpha * m.Z * k^(m.alpha - one(T)) - m.delta
    w = (one(T) - m.alpha) * m.Z * k^m.alpha
    return r, w
end

# Ramsey capital stock implied by r = 1/β − 1 (the γ = 1 benchmark / upper bound on k*).
function _ramsey_k(m::BlanchardOLG{T}) where {T<:AbstractFloat}
    r_ram = one(T) / m.beta - one(T)
    return (m.alpha * m.Z / (r_ram + m.delta))^(one(T) / (one(T) - m.alpha))
end

"""
    blanchard_steady_state(m::BlanchardOLG; tol=1e-10, max_iter=200) → BlanchardOLGSteadyState

Compute the stationary equilibrium by bisecting on capital `k` until the budget-implied
consumption `C = r k + w` (`= f(k) − δk`; debt service `r·b` cancels in aggregate)
equals the Euler-implied consumption
`C = (1+r) λ (k + b) / (β(1+r) − 1)`, with `λ = (1−βγ)(1−γ)/γ`. For `γ = 1` (Ramsey) the
interest rate is pinned directly at `1/β − 1`.
"""
function blanchard_steady_state(m::BlanchardOLG{T}; tol::Real=1e-10,
                                 max_iter::Int=200) where {T<:AbstractFloat}
    k_ram = _ramsey_k(m)

    # Ramsey limit: γ = 1 ⟹ r = 1/β − 1 pins k directly (the wedge vanishes).
    if m.gamma >= one(T) - eps(T)
        k = k_ram
        r, w = _blanchard_prices(m, k)
        # Aggregate consumption C = r·k + w = f(k) − δk (#237). Debt is net wealth
        # and taxes T = r·b service it, so in aggregate the debt-service terms
        # cancel (deriving the capital law from a = k+b minus the government budget
        # gives k_{t+1} = (1+r)k + w − C); a spurious −r·b here double-counted it
        # for b ≠ 0. Human wealth H below keeps −r·b (after-tax labour income).
        C = r * k + w
        H = m.b == zero(T) ? w * (one(T) + r) / (one(T) + r - m.gamma) :
            (w - r * m.b) * (one(T) + r) / (one(T) + r - m.gamma)
        return BlanchardOLGSteadyState{T}(k, C, r, w, H, one(T) - m.beta * m.gamma, m.b, true)
    end

    lambda = (one(T) - m.beta * m.gamma) * (one(T) - m.gamma) / m.gamma

    # resid(k) = C_budget(k) − C_euler(k); equilibrium k* lies below the Ramsey k
    # (where β(1+r) > 1). Bracket on (ε·k_ram, (1−ε)·k_ram).
    function resid(k::T)
        r, w = _blanchard_prices(m, k)
        denom = m.beta * (one(T) + r) - one(T)
        denom <= zero(T) && return T(Inf)            # r ≤ 1/β−1: not an OLG equilibrium
        C_budget = r * k + w                          # = f(k) − δk; debt service cancels (#237)
        C_euler = (one(T) + r) * lambda * (k + m.b) / denom
        return C_budget - C_euler
    end

    # Scan (0, k_ram) for a sign change of resid to bracket the root robustly. With
    # government debt the budget consumption can turn negative at low k, so a fixed
    # bracket is not reliable; a coarse scan locates the sign change first.
    # Scan from high k downward and take the FIRST sign change — the high-capital root
    # continuously connected to the Ramsey (γ→1) / no-debt equilibrium. (The OLG
    # consumption function can admit a second, degenerate low-k root with an absurd r.)
    n_scan = 400
    k_grid = collect(range(T(1e-4) * k_ram, (one(T) - T(1e-6)) * k_ram; length=n_scan))
    k_lo = k_grid[1]; k_hi = k_grid[end]
    found = false
    f_next = resid(k_grid[end])
    for i in (n_scan - 1):-1:1
        f_cur = resid(k_grid[i])
        if isfinite(f_next) && isfinite(f_cur) && f_cur * f_next < zero(T)
            k_lo = k_grid[i]; k_hi = k_grid[i+1]; found = true
            break
        end
        f_next = f_cur
    end

    converged = false
    k = T(0.5) * (k_lo + k_hi)
    if found
        f_lo = resid(k_lo)
        for _ in 1:max_iter
            k = T(0.5) * (k_lo + k_hi)
            fk = resid(k)
            if abs(fk) < tol
                converged = true
                break
            end
            if f_lo * fk < zero(T)
                k_hi = k
            else
                k_lo = k; f_lo = fk
            end
        end
        converged = converged || abs(resid(k)) < T(1e-6)
    end

    r, w = _blanchard_prices(m, k)
    C = r * k + w                        # = f(k) − δk; debt service cancels (#237)
    H = (w - r * m.b) * (one(T) + r) / (one(T) + r - m.gamma)
    return BlanchardOLGSteadyState{T}(k, C, r, w, H, one(T) - m.beta * m.gamma, m.b, converged)
end

# =============================================================================
# Local dynamics (saddle path)
# =============================================================================

"""
    blanchard_solve(m::BlanchardOLG, ss=blanchard_steady_state(m)) → BlanchardOLGSolution

Linearize the aggregate `(k, C)` system around the steady state and solve the saddle path.
Returns the 2×2 transition `M`, its eigenvalues, the stable (saddle-path) eigenvalue, the
consumption policy slope `dC/dk`, and a determinacy flag (exactly one stable root).
"""
function blanchard_solve(m::BlanchardOLG{T},
                          ss::BlanchardOLGSteadyState{T}=blanchard_steady_state(m)) where {T<:AbstractFloat}
    r = ss.r; k = ss.k; C = ss.C
    lambda = (one(T) - m.beta * m.gamma) * (one(T) - m.gamma) / m.gamma

    # d/dk[(1+r)k + w] = 1 + r (the rental and wage terms net out), so
    #   dk_{t+1} = (1+r) dk_t − dC_t.
    # r'(k) = (α−1)(r+δ)/k.  Linearizing the Euler around the SS (where
    #   βC − λ(k+b) = C/(1+r)) gives the C row below.
    rprime = (m.alpha - one(T)) * (r + m.delta) / k
    B = rprime * C / (one(T) + r) - (one(T) + r) * lambda

    M = T[ (one(T) + r)        (-one(T)) ;
           B * (one(T) + r)    (one(T) + r) * m.beta - B ]

    ev = eigvals(M)
    ev_c = ComplexF64.(ev)
    moduli = abs.(ev_c)
    n_stable = count(<(one(Float64) - 1e-9), moduli)
    determinate = (n_stable == 1)

    # Saddle path: the stable eigenvalue and its eigenvector give dC = slope · dk.
    stable_idx = argmin(moduli)
    stable_eig = T(real(ev_c[stable_idx]))
    F = eigen(M)
    vec_s = F.vectors[:, stable_idx]
    # policy slope dC/dk = v_C / v_k along the stable eigenvector
    policy_slope = abs(real(vec_s[1])) > eps(T) ? T(real(vec_s[2] / vec_s[1])) : zero(T)

    return BlanchardOLGSolution{T}(ss, M, ev_c, stable_eig, policy_slope, determinate)
end

"""
    blanchard_transition(m::BlanchardOLG, sol::BlanchardOLGSolution, k0; H=50) → NamedTuple

Simulate the transitional dynamics from an initial capital `k0` along the saddle path.
Capital and consumption follow the linear saddle path (`dk_{t+1} = stable_eig · dk_t`,
`dC = policy_slope · dk`); the interest rate and wage are evaluated at each period's
capital. Returns paths of length `H+1` for `k`, `C`, `r`, `w`, converging to the steady
state at the stable rate.
"""
function blanchard_transition(m::BlanchardOLG{T}, sol::BlanchardOLGSolution{T}, k0::Real;
                               H::Int=50) where {T<:AbstractFloat}
    ss = sol.ss
    kpath = zeros(T, H + 1)
    Cpath = zeros(T, H + 1)
    rpath = zeros(T, H + 1)
    wpath = zeros(T, H + 1)
    dk = T(k0) - ss.k
    for t in 0:H
        kt = ss.k + dk
        kpath[t+1] = kt
        Cpath[t+1] = ss.C + sol.policy_slope * dk
        r_t, w_t = _blanchard_prices(m, max(kt, T(1e-8)))
        rpath[t+1] = r_t
        wpath[t+1] = w_t
        dk = sol.stable_eig * dk
    end
    return (k=kpath, C=Cpath, r=rpath, w=wpath)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::BlanchardOLG{T}) where {T}
    print(io, "BlanchardOLG{$T}(α=$(m.alpha), β=$(m.beta), δ=$(m.delta), ",
              "γ=$(m.gamma), Z=$(m.Z), b=$(m.b))")
end

function Base.show(io::IO, ss::BlanchardOLGSteadyState{T}) where {T}
    print(io, "BlanchardOLGSteadyState{$T}: k=$(round(ss.k; digits=4)), ",
              "C=$(round(ss.C; digits=4)), r=$(round(ss.r; digits=5))")
end

function Base.show(io::IO, sol::BlanchardOLGSolution{T}) where {T}
    print(io, "BlanchardOLGSolution{$T}: stable_eig=$(round(sol.stable_eig; digits=4)), ",
              "determinate=$(sol.determinate)")
end

"""
    report(ss::BlanchardOLGSteadyState)

Print the Blanchard OLG steady state: capital, consumption, prices, human wealth, the
log-utility MPC, debt, and convergence.
"""
function report(ss::BlanchardOLGSteadyState{T}) where {T}
    println("Blanchard (1985) Perpetual-Youth OLG — Steady State")
    println("  Capital        k = ", round(ss.k; digits=6))
    println("  Consumption    C = ", round(ss.C; digits=6))
    println("  Interest rate  r = ", round(ss.r; digits=6))
    println("  Wage           w = ", round(ss.w; digits=6))
    println("  Human wealth   H = ", round(ss.H; digits=6))
    println("  MPC (1 − βγ)     = ", round(ss.mpc; digits=6))
    println("  Govt debt      b = ", round(ss.b; digits=6))
    println("  Converged        = ", ss.converged)
    return nothing
end

