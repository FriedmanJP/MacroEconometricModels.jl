# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Continuous-time heterogeneous-agent model (one asset), solved with the finite-difference
methods of Achdou, Han, Lasry, Lions & Moll (2022): an implicit upwind scheme for the
Hamilton-Jacobi-Bellman (HJB) equation and the Kolmogorov-Forward (Fokker-Planck) equation
for the stationary distribution. The same sparse infinitesimal generator `A` drives both
the HJB (implicitly) and the KFE (via its transpose) — the key elegance of the method.

The household solves, with CRRA utility `u(c)=c^{1-σ}/(1-σ)` (or `log c`), idiosyncratic
labor productivity `z` following a two-state Poisson process, and wealth `a ≥ a_min`:

    ρ v(a,z) = max_c u(c) + ∂_a v(a,z)·(w·z + r·a − c) + Σ_{z'} λ_{z→z'}[v(a,z') − v(a,z)]

with the state-constraint boundary `∂_a v(a_min,z) ≥ u'(w·z + r·a_min)` (saving ≥ 0 at the
borrowing constraint). The stationary density `g` solves `Aᵀ g = 0`, `∫ g = 1`.

# References
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022). Income and Wealth
  Distribution in Macroeconomics: A Continuous-Time Approach. *Review of Economic Studies*,
  89(1), 45–86.
- Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving. *QJE*.
"""

using SparseArrays
using LinearAlgebra

# =============================================================================
# Types
# =============================================================================

"""
    CTPoissonIncome{T}

Two-state Poisson labor-productivity process: states `z = [z1, z2]` with switching
intensities `λ = [λ12, λ21]` (`λ12` = rate of leaving state 1 for state 2). Stationary
probabilities are `π1 = λ21/(λ12+λ21)`, `π2 = λ12/(λ12+λ21)`.
"""
struct CTPoissonIncome{T<:AbstractFloat}
    z::Vector{T}        # [z1, z2]
    lambda::Vector{T}   # [λ12, λ21]
end

"""
    CTAiyagari{T}

Continuous-time Aiyagari model parameters: capital share `alpha`, discount rate `rho`,
CRRA `sigma`, depreciation `delta`, TFP `Z`, the income process, and the wealth grid
`[a_min, a_max]` with `I` points.
"""
struct CTAiyagari{T<:AbstractFloat}
    alpha::T
    rho::T
    sigma::T
    delta::T
    Z::T
    income::CTPoissonIncome{T}
    a_min::T
    a_max::T
    I::Int
end

function CTAiyagari(; alpha::Real=0.36, rho::Real=0.05, sigma::Real=2.0, delta::Real=0.05,
                      Z::Real=1.0, z::AbstractVector=[0.1, 0.2],
                      lambda::AbstractVector=[0.5, 0.5],
                      a_min::Real=0.0, a_max::Real=30.0, I::Int=500)
    T = promote_type(typeof(alpha), typeof(rho), typeof(sigma), typeof(delta),
                     typeof(Z), eltype(z), eltype(lambda), typeof(a_min), typeof(a_max), Float64)
    inc = CTPoissonIncome{T}(collect(T, z), collect(T, lambda))
    return CTAiyagari{T}(T(alpha), T(rho), T(sigma), T(delta), T(Z), inc,
                          T(a_min), T(a_max), I)
end

"""
    CTSteadyState{T}

Stationary equilibrium of a continuous-time Aiyagari model.

# Fields
- `r`, `w` — equilibrium interest rate and wage
- `K`, `L` — aggregate capital (= ∫a g) and effective labor (= ∫z g)
- `a` — wealth grid (`I`-vector)
- `g` — stationary density over `(a, z)` (`I×2`, integrates to 1 with the grid spacing)
- `v` — value function (`I×2`)
- `c` — consumption policy (`I×2`)
- `s` — saving drift `w·z + r·a − c` (`I×2`)
- `A` — sparse infinitesimal generator (`2I × 2I`)
- `converged` — HJB and equilibrium convergence flag
"""
struct CTSteadyState{T<:AbstractFloat}
    r::T
    w::T
    K::T
    L::T
    a::Vector{T}
    g::Matrix{T}
    v::Matrix{T}
    c::Matrix{T}
    s::Matrix{T}
    A::SparseMatrixCSC{T,Int}
    converged::Bool
end

# =============================================================================
# Utility helpers
# =============================================================================

# CRRA utility, marginal utility, and the inverse of marginal utility (log when σ = 1).
_ct_u(c, σ) = σ ≈ 1 ? log(c) : c^(1 - σ) / (1 - σ)
_ct_uprime(c, σ) = c^(-σ)
_ct_uprime_inv(dv, σ) = dv^(-1 / σ)     # c = (v_a)^{-1/σ}

# Constant income-switching block of the generator (Poisson, 2 states), 2I×2I.
# Ordering is column-major over (a, z): index k = i + (j-1)*I.
function _ct_aswitch(m::CTAiyagari{T}) where {T<:AbstractFloat}
    I = m.I; la = m.income.lambda
    As = spzeros(T, 2I, 2I)
    for i in 1:I
        k1 = i; k2 = i + I
        As[k1, k1] -= la[1]; As[k1, k2] += la[1]
        As[k2, k2] -= la[2]; As[k2, k1] += la[2]
    end
    return As
end

"""
    _ct_policy_and_generator(m, v, r, w, a, da, Aswitch) → (c, s, A)

Given a value function `v` and prices `(r, w)`, compute the consumption policy `c`, the
saving drift `s`, and the sparse infinitesimal generator `A` via the upwind scheme:
forward differences where the drift is positive, backward where negative, with the
state-constraint boundaries (zero saving at `a_min` and `a_max`). `A` includes the income
switching block `Aswitch`.
"""
function _ct_policy_and_generator(m::CTAiyagari{T}, v::Matrix{T}, r::T, w::T,
                                   a::Vector{T}, da::T,
                                   Aswitch::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    I = m.I; z = m.income.z; σ = m.sigma
    c = zeros(T, I, 2); s = zeros(T, I, 2)
    rows = Int[]; cols = Int[]; vals = T[]
    for j in 1:2
        off = (j - 1) * I
        for i in 1:I
            dVf = i < I ? (v[i+1, j] - v[i, j]) / da : _ct_uprime(w * z[j] + r * a[I], σ)
            dVb = i > 1 ? (v[i, j] - v[i-1, j]) / da : _ct_uprime(w * z[j] + r * a[1], σ)
            cf = _ct_uprime_inv(max(dVf, T(1e-12)), σ)
            cb = _ct_uprime_inv(max(dVb, T(1e-12)), σ)
            c0 = w * z[j] + r * a[i]
            sf = c0 - cf; sb = c0 - cb
            If = sf > zero(T); Ib = sb < zero(T)
            If && Ib && (Ib = false)
            I0 = !(If || Ib)
            c[i, j] = max(cf * If + cb * Ib + c0 * I0, T(1e-10))
            s[i, j] = c0 - c[i, j]
            X = -min(sb, zero(T)) / da
            Z_ = max(sf, zero(T)) / da
            k = i + off
            push!(rows, k); push!(cols, k); push!(vals, -X - Z_)
            i > 1 && (push!(rows, k); push!(cols, k - 1); push!(vals, X))
            i < I && (push!(rows, k); push!(cols, k + 1); push!(vals, Z_))
        end
    end
    A = sparse(rows, cols, vals, 2I, 2I) + Aswitch
    return c, s, A
end

# =============================================================================
# HJB — implicit upwind scheme
# =============================================================================

"""
    ct_hjb(m::CTAiyagari, r, w; max_iter=100, tol=1e-6, Delta=1000.0) → (v, c, s, A)

Solve the HJB equation at prices `(r, w)` by the implicit upwind finite-difference method
(Achdou et al. 2022). Returns the value function `v`, consumption `c`, saving drift `s`
(all `I×2`), and the sparse generator `A` (`2I×2I`) combining the upwinded wealth drift and
the Poisson income switching. `Delta` is the implicit step size (large = fast convergence).
"""
function ct_hjb(m::CTAiyagari{T}, r::T, w::T; max_iter::Int=100, tol::Real=1e-6,
                Delta::Real=1000.0) where {T<:AbstractFloat}
    I = m.I
    z = m.income.z
    la = m.income.lambda
    σ = m.sigma
    ρ = m.rho

    a = collect(range(m.a_min, m.a_max; length=I))
    da = a[2] - a[1]
    Δ = T(Delta)
    Aswitch = _ct_aswitch(m)

    # Initial guess: value of staying put (consume income, zero saving) forever.
    v = zeros(T, I, 2)
    for j in 1:2, i in 1:I
        v[i, j] = _ct_u(max(w * z[j] + r * a[i], T(1e-10)), σ) / ρ
    end

    c = zeros(T, I, 2); s = zeros(T, I, 2)
    A = spzeros(T, 2I, 2I)
    converged = false

    for _ in 1:max_iter
        c, s, A = _ct_policy_and_generator(m, v, r, w, a, da, Aswitch)
        # Implicit update: (1/Δ + ρ) v^{n+1} − A v^{n+1} = u(c) + v^n/Δ.
        u_vec = vec([_ct_u(c[i, j], σ) for i in 1:I, j in 1:2])
        B = (one(T) / Δ + ρ) * LinearAlgebra.I - A   # UniformScaling − sparse → sparse
        v_new = reshape(B \ (u_vec + vec(v) / Δ), I, 2)
        dist = maximum(abs.(v_new - v))
        v = v_new
        if dist < tol
            converged = true
            break
        end
    end
    # Recompute policy/generator at the converged value for exact consistency.
    c, s, A = _ct_policy_and_generator(m, v, r, w, a, da, Aswitch)

    return v, c, s, A, a, converged
end

# =============================================================================
# KFE — stationary distribution
# =============================================================================

"""
    ct_kfe(A, I, da) → g::Matrix

Solve the Kolmogorov-Forward equation `Aᵀ g = 0` with the normalization `∫ g da = 1` for
the stationary density over `(a, z)`. Returns an `I×2` matrix. `A` is the generator from
[`ct_hjb`](@ref).
"""
function ct_kfe(A::SparseMatrixCSC{T,Int}, I::Int, da::T) where {T<:AbstractFloat}
    AT = copy(transpose(A))
    n = 2I
    # Replace one equation with the normalization Σ g_k da = 1 (fix g_1).
    b = zeros(T, n)
    AT[1, :] .= zero(T)
    AT[1, 1] = one(T)
    b[1] = one(T)
    # Sparse LU on the SparseMatrixCSC directly — never materialise the dense
    # 2I×2I matrix (#242).
    g_vec = lu(AT) \ b
    g_vec = max.(g_vec, zero(T))
    mass = sum(g_vec) * da
    g_vec ./= mass
    return reshape(g_vec, I, 2)
end

# =============================================================================
# Steady-state equilibrium
# =============================================================================

"""
    ct_steady_state(m::CTAiyagari; r_bounds=(0.001, m.rho-1e-4), max_iter=100, tol=1e-6,
                    hjb_kwargs...) → CTSteadyState

Compute the stationary general equilibrium by bisecting on the interest rate `r` until the
capital supplied by households (`∫ a g`) equals the capital demanded by firms (from the
Cobb-Douglas FOC `r = α Z (K/L)^{α-1} − δ`). At each `r` the wage is `w = (1-α)Z(K/L)^α`,
the HJB is solved by [`ct_hjb`](@ref), and the distribution by [`ct_kfe`](@ref).
"""
function ct_steady_state(m::CTAiyagari{T};
                          r_bounds::Tuple{Real,Real}=(0.0001, m.rho - 1e-4),
                          max_iter::Int=100, tol::Real=1e-6,
                          hjb_max_iter::Int=100, hjb_tol::Real=1e-6,
                          Delta::Real=1000.0) where {T<:AbstractFloat}
    z = m.income.z; la = m.income.lambda
    # Stationary income distribution and aggregate effective labor L.
    π1 = la[2] / (la[1] + la[2]); π2 = one(T) - π1
    L = z[1] * π1 + z[2] * π2

    # Firm FOC: given r, the capital/labor ratio and wage.
    function firm(r::T)
        kl = (m.alpha * m.Z / (r + m.delta))^(one(T) / (one(T) - m.alpha))  # K/L
        w = (one(T) - m.alpha) * m.Z * kl^m.alpha
        return kl * L, w   # K_demand, w
    end

    r_lo = T(r_bounds[1]); r_hi = T(r_bounds[2])
    local best_ss
    converged = false
    r_mid = T(0.5) * (r_lo + r_hi)

    for _ in 1:max_iter
        r_mid = T(0.5) * (r_lo + r_hi)
        K_d, w = firm(r_mid)
        v, c, s, A, a, hjb_ok = ct_hjb(m, r_mid, w;
                                        max_iter=hjb_max_iter, tol=hjb_tol, Delta=Delta)
        da = a[2] - a[1]
        g = ct_kfe(A, m.I, da)
        K_s = sum(a .* g[:, 1] .+ a .* g[:, 2]) * da    # ∫ a g
        excess = K_s - K_d
        best_ss = CTSteadyState{T}(r_mid, w, K_s, L, a, g, v, c, s, A, hjb_ok)
        if abs(excess) < tol
            converged = true
            break
        end
        # K_s increasing in r (toward ρ savings explode), K_d decreasing in r.
        if excess > zero(T)
            r_hi = r_mid     # supply too high → lower r
        else
            r_lo = r_mid
        end
    end

    return CTSteadyState{T}(best_ss.r, best_ss.w, best_ss.K, best_ss.L, best_ss.a,
                            best_ss.g, best_ss.v, best_ss.c, best_ss.s, best_ss.A,
                            converged && best_ss.converged)
end

# =============================================================================
# MIT-shock transition (deterministic perfect-foresight path)
# =============================================================================

"""
    CTTransition{T}

Deterministic transition path of a continuous-time Aiyagari economy after an unanticipated
aggregate shock (an "MIT shock"). Fields are length-`(N+1)` time series on the grid `t`:
TFP `Z`, aggregate capital `K`, interest rate `r`, wage `w`, aggregate consumption `C`.
"""
struct CTTransition{T<:AbstractFloat}
    t::Vector{T}
    Z::Vector{T}
    K::Vector{T}
    r::Vector{T}
    w::Vector{T}
    C::Vector{T}
    converged::Bool
    iterations::Int
end

"""
    ct_mit_shock(m::CTAiyagari, ss0::CTSteadyState, Z_path; dt=0.25, max_iter=300,
                 tol=1e-6, relax=0.3) → CTTransition

Compute the perfect-foresight transition after an unanticipated aggregate TFP shock. The
economy begins at the initial steady state `ss0` (distribution `g_0 = ss0.g`) and, given
the deterministic TFP path `Z_path` (which must return to `m.Z` so the terminal steady
state is `ss0`), converges back to `ss0`. The algorithm shoots on the capital path `K_t`:

1. Given `{K_t, Z_t}`, set prices `r_t = α Z_t (K_t/L)^{α-1} − δ`, `w_t = (1-α) Z_t (K_t/L)^α`.
2. Solve the HJB **backward** from the terminal value `v(·,T) = ss0.v` (implicit step).
3. Solve the KFE **forward** from `g(·,0) = ss0.g` using the time-`t` generators (implicit).
4. Update `K_t = ∫ a g_t` by relaxation until the path converges.

`K_0` is pinned by the initial distribution. `dt` is the time step, `relax ∈ (0,1]` the
damping on the capital-path update.
"""
function ct_mit_shock(m::CTAiyagari{T}, ss0::CTSteadyState{T}, Z_path::AbstractVector;
                       dt::Real=0.25, max_iter::Int=300, tol::Real=1e-6,
                       relax::Real=0.3) where {T<:AbstractFloat}
    Np1 = length(Z_path)          # number of time points (N+1)
    N = Np1 - 1
    Z = collect(T, Z_path)
    dt_T = T(dt); relax_T = T(relax)
    a = ss0.a; da = a[2] - a[1]; I = m.I; L = ss0.L
    Aswitch = _ct_aswitch(m)
    avec = vcat(a, a)             # asset value per state index (column-major (a,z))

    firm_r(Zt, Kt) = m.alpha * Zt * (Kt / L)^(m.alpha - one(T)) - m.delta
    firm_w(Zt, Kt) = (one(T) - m.alpha) * Zt * (Kt / L)^m.alpha

    K = fill(ss0.K, Np1)          # initial guess: flat at the steady state
    g0 = vec(ss0.g)
    vT = vec(ss0.v)

    Avec = Vector{SparseMatrixCSC{T,Int}}(undef, Np1)
    cmat = [zeros(T, I, 2) for _ in 1:Np1]
    converged = false
    iters = 0

    for outer in 1:max_iter
        iters = outer
        rpath = [firm_r(Z[n], K[n]) for n in 1:Np1]
        wpath = [firm_w(Z[n], K[n]) for n in 1:Np1]

        # ── Backward HJB: v(·,T) = ss0.v ──
        V = copy(vT)
        for n in N:-1:1
            Vmat = reshape(V, I, 2)
            c, _, A = _ct_policy_and_generator(m, Vmat, rpath[n], wpath[n], a, da, Aswitch)
            Avec[n] = A
            cmat[n] = c
            u_vec = vec([_ct_u(c[i, j], m.sigma) for i in 1:I, j in 1:2])
            B = (one(T) / dt_T + m.rho) * LinearAlgebra.I - A
            V = B \ (u_vec + V / dt_T)
        end

        # ── Forward KFE: g(·,0) = ss0.g ──
        gcur = copy(g0)
        K_new = similar(K)
        K_new[1] = sum(avec .* gcur) * da
        for n in 1:N
            B = LinearAlgebra.I - dt_T * copy(transpose(Avec[n]))
            gnext = B \ gcur
            gnext = max.(gnext, zero(T))
            gnext ./= (sum(gnext) * da)        # renormalize (guards numerical drift)
            gcur = gnext
            K_new[n+1] = sum(avec .* gcur) * da
        end

        diff = maximum(abs.(K_new .- K))
        # K_0 is pinned by the initial distribution; relax the rest.
        for n in 2:Np1
            K[n] = relax_T * K_new[n] + (one(T) - relax_T) * K[n]
        end
        if diff < tol
            converged = true
            break
        end
    end

    # Final pass: prices and aggregate consumption along the converged path.
    rpath = [firm_r(Z[n], K[n]) for n in 1:Np1]
    wpath = [firm_w(Z[n], K[n]) for n in 1:Np1]
    Cpath = zeros(T, Np1)
    gcur = copy(g0)
    for n in 1:Np1
        gmat = reshape(gcur, I, 2)
        cn = n <= N ? cmat[n] : cmat[max(N, 1)]
        Cpath[n] = sum(cn .* gmat) * da
        if n <= N
            B = LinearAlgebra.I - dt_T * copy(transpose(Avec[n]))
            gnext = B \ gcur; gnext = max.(gnext, zero(T)); gnext ./= (sum(gnext) * da)
            gcur = gnext
        end
    end

    tgrid = collect(T, range(zero(T); step=dt_T, length=Np1))
    return CTTransition{T}(tgrid, Z, copy(K), rpath, wpath, Cpath, converged, iters)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::CTAiyagari{T}) where {T}
    print(io, "CTAiyagari{$T}(α=$(m.alpha), ρ=$(m.rho), σ=$(m.sigma), δ=$(m.delta), I=$(m.I))")
end

function Base.show(io::IO, ss::CTSteadyState{T}) where {T}
    print(io, "CTSteadyState{$T}: r=$(round(ss.r; digits=5)), K=$(round(ss.K; digits=4)), ",
              "converged=$(ss.converged)")
end

function Base.show(io::IO, tr::CTTransition{T}) where {T}
    print(io, "CTTransition{$T}: ", length(tr.t), " periods, ",
              "K: $(round(tr.K[1]; digits=4)) → $(round(maximum(tr.K); digits=4)) → ",
              "$(round(tr.K[end]; digits=4)), converged=$(tr.converged)")
end

"""
    report(ss::CTSteadyState)

Print the continuous-time Aiyagari steady state: equilibrium prices, aggregates, the
fraction of households at the borrowing constraint, and convergence.
"""
function report(io::IO, ss::CTSteadyState{T}) where {T}
    da = ss.a[2] - ss.a[1]
    constrained = (ss.g[1, 1] + ss.g[1, 2]) * da
    mean_a = ss.K
    println(io, "Continuous-Time Aiyagari (Achdou et al. 2022) — Steady State")
    println(io, "  Interest rate   r = ", round(ss.r; digits=6))
    println(io, "  Wage            w = ", round(ss.w; digits=6))
    println(io, "  Capital         K = ", round(ss.K; digits=6))
    println(io, "  Effective labor L = ", round(ss.L; digits=6))
    println(io, "  Mean wealth       = ", round(mean_a; digits=6))
    println(io, "  At constraint     = ", round(constrained; digits=6))
    println(io, "  Converged         = ", _yesno(ss.converged))
    return nothing
end
report(ss::CTSteadyState) = report(stdout, ss)   # G-17 (#254): io-routed report
