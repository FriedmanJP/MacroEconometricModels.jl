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

# CRRA marginal utility and its inverse (log when σ = 1).
_ct_u(c, σ) = σ ≈ 1 ? log(c) : c^(1 - σ) / (1 - σ)
_ct_uprime_inv(dv, σ) = dv^(-1 / σ)     # c = (v_a)^{-1/σ}

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

    # Income switching block (constant): 2I×2I with -la on block diagonal, +la off.
    # Ordering: column-major over (a, z): index k = i + (j-1)*I.
    Aswitch = spzeros(T, 2I, 2I)
    for i in 1:I
        k1 = i              # state (i, 1)
        k2 = i + I          # state (i, 2)
        Aswitch[k1, k1] -= la[1]; Aswitch[k1, k2] += la[1]
        Aswitch[k2, k2] -= la[2]; Aswitch[k2, k1] += la[2]
    end

    # Initial guess: value of staying put (consume income, zero saving) forever.
    v = zeros(T, I, 2)
    for j in 1:2, i in 1:I
        c0 = w * z[j] + r * a[i]
        v[i, j] = _ct_u(max(c0, T(1e-10)), σ) / ρ
    end

    c = zeros(T, I, 2)
    s = zeros(T, I, 2)
    A = spzeros(T, 2I, 2I)
    converged = false

    for _ in 1:max_iter
        dVf = zeros(T, I, 2)
        dVb = zeros(T, I, 2)
        for j in 1:2
            # forward difference; upper boundary: zero saving at a_max
            for i in 1:I-1
                dVf[i, j] = (v[i+1, j] - v[i, j]) / da
            end
            dVf[I, j] = _ct_uprime(w * z[j] + r * a[I], σ)
            # backward difference; lower boundary: zero saving at a_min (state constraint)
            for i in 2:I
                dVb[i, j] = (v[i, j] - v[i-1, j]) / da
            end
            dVb[1, j] = _ct_uprime(w * z[j] + r * a[1], σ)
        end

        # Upwind: consumption and drift from forward/backward, else stay-put.
        ssf = zeros(T, I, 2)   # forward drift
        ssb = zeros(T, I, 2)   # backward drift
        for j in 1:2, i in 1:I
            cf = _ct_uprime_inv(max(dVf[i, j], T(1e-12)), σ)
            cb = _ct_uprime_inv(max(dVb[i, j], T(1e-12)), σ)
            c0 = w * z[j] + r * a[i]
            ssf[i, j] = w * z[j] + r * a[i] - cf
            ssb[i, j] = w * z[j] + r * a[i] - cb
            If = ssf[i, j] > zero(T)
            Ib = ssb[i, j] < zero(T)
            I0 = !(If || Ib)
            # at most one of If, Ib should hold; if both, prefer the consistent one
            if If && Ib
                Ib = false
            end
            cc = cf * If + cb * Ib + c0 * I0
            c[i, j] = max(cc, T(1e-10))
            s[i, j] = w * z[j] + r * a[i] - c[i, j]
        end

        # Build sparse generator A from upwind drifts.
        rows = Int[]; cols = Int[]; vals = T[]
        for j in 1:2
            off = (j - 1) * I
            for i in 1:I
                k = i + off
                X = -min(ssb[i, j], zero(T)) / da          # to i-1
                Z_ = max(ssf[i, j], zero(T)) / da          # to i+1
                Y = -X - Z_                                # diagonal (drift part)
                push!(rows, k); push!(cols, k); push!(vals, Y)
                if i > 1
                    push!(rows, k); push!(cols, k - 1); push!(vals, X)
                end
                if i < I
                    push!(rows, k); push!(cols, k + 1); push!(vals, Z_)
                end
            end
        end
        A = sparse(rows, cols, vals, 2I, 2I) + Aswitch

        # Implicit update: (1/Δ + ρ) v^{n+1} − A v^{n+1} = u(c) + v^n/Δ.
        u_vec = vec([_ct_u(c[i, j], σ) for i in 1:I, j in 1:2])
        v_vec = vec(v)
        B = (one(T) / Δ + ρ) * LinearAlgebra.I - A   # UniformScaling − sparse → sparse
        b = u_vec + v_vec / Δ
        v_new = B \ b
        v_mat = reshape(v_new, I, 2)

        dist = maximum(abs.(v_mat - v))
        v = v_mat
        if dist < tol
            converged = true
            break
        end
    end

    return v, c, s, A, a, converged
end

_ct_uprime(c, σ) = c^(-σ)

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
    g_vec = Matrix(AT) \ b
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
# Display
# =============================================================================

function Base.show(io::IO, m::CTAiyagari{T}) where {T}
    print(io, "CTAiyagari{$T}(α=$(m.alpha), ρ=$(m.rho), σ=$(m.sigma), δ=$(m.delta), I=$(m.I))")
end

function Base.show(io::IO, ss::CTSteadyState{T}) where {T}
    print(io, "CTSteadyState{$T}: r=$(round(ss.r; digits=5)), K=$(round(ss.K; digits=4)), ",
              "converged=$(ss.converged)")
end

"""
    report(ss::CTSteadyState)

Print the continuous-time Aiyagari steady state: equilibrium prices, aggregates, the
fraction of households at the borrowing constraint, and convergence.
"""
function report(ss::CTSteadyState{T}) where {T}
    da = ss.a[2] - ss.a[1]
    constrained = (ss.g[1, 1] + ss.g[1, 2]) * da
    mean_a = ss.K
    println("Continuous-Time Aiyagari (Achdou et al. 2022) — Steady State")
    println("  Interest rate   r = ", round(ss.r; digits=6))
    println("  Wage            w = ", round(ss.w; digits=6))
    println("  Capital         K = ", round(ss.K; digits=6))
    println("  Effective labor L = ", round(ss.L; digits=6))
    println("  Mean wealth       = ", round(mean_a; digits=6))
    println("  At constraint     = ", round(constrained; digits=6))
    println("  Converged         = ", ss.converged)
    return nothing
end
