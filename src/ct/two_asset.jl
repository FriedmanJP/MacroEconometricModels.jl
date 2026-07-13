# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Continuous-time two-asset heterogeneous-agent model in the spirit of Kaplan, Moll &
Violante (2018), solved with the finite-difference methods of Achdou et al. (2022).

Households hold a **liquid** asset `b` (return `r_b`) and an **illiquid** asset `a`
(return `r_a > r_b`). Moving funds between them — the **deposit** `d` — is subject to a
smooth convex adjustment cost `χ(d) = (χ/2) d²`, so households tolerate a low liquid return
to hold high-return illiquid wealth, generating a large stock of illiquid wealth and a
hand-to-mouth liquid margin (the central KMV mechanism). The household solves

    ρ V(b,a,z) = max_{c,d} u(c)
                 + V_b·(w z + r_b b − d − χ(d) − c)
                 + V_a·(r_a a + d)
                 + Σ_{z'} λ_{z→z'}[V(b,a,z') − V(b,a,z)]

with first-order conditions `c = (V_b)^{-1/σ}` and `d = (V_a/V_b − 1)/χ`. The stationary
joint density of `(b, a, z)` solves the Kolmogorov-Forward equation `Aᵀ g = 0`, sharing the
generator `A` with the HJB.

This is a tractable continuous-time two-asset solver: it uses a smooth quadratic adjustment
cost (rather than the kinked linear-plus-quadratic cost of KMV), a single liquid return
(no borrowing wedge), and is solved at given returns `(r_a, r_b, w)` — the household block
and stationary distribution that underlie a two-asset HANK general equilibrium.

# References
- Kaplan, G., Moll, B., & Violante, G. L. (2018). Monetary Policy According to HANK.
  *American Economic Review*, 108(3), 697–743.
- Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022). Income and Wealth
  Distribution in Macroeconomics: A Continuous-Time Approach. *RES*, 89(1), 45–86.
"""

using SparseArrays
using LinearAlgebra

# =============================================================================
# Types
# =============================================================================

"""
    CTTwoAsset{T}

Continuous-time two-asset model parameters: CRRA `sigma`, discount rate `rho`, illiquid and
liquid returns `r_a > r_b`, quadratic adjustment-cost coefficient `chi`, wage `w`, the
two-state Poisson income process, and the liquid/illiquid grids `[0, b_max]` (`Ib` points)
and `[0, a_max]` (`Ia` points).
"""
struct CTTwoAsset{T<:AbstractFloat}
    sigma::T
    rho::T
    r_a::T
    r_b::T
    chi::T
    w::T
    income::CTPoissonIncome{T}
    b_max::T
    a_max::T
    Ib::Int
    Ia::Int
end

function CTTwoAsset(; sigma::Real=2.0, rho::Real=0.06, r_a::Real=0.05, r_b::Real=0.02,
                      chi::Real=2.0, w::Real=1.0, z::AbstractVector=[0.8, 1.2],
                      lambda::AbstractVector=[0.5, 0.5],
                      b_max::Real=20.0, a_max::Real=20.0, Ib::Int=40, Ia::Int=40)
    @assert r_a > r_b "illiquid return r_a must exceed liquid return r_b"
    T = promote_type(typeof(sigma), typeof(rho), typeof(r_a), typeof(r_b), typeof(chi),
                     typeof(w), eltype(z), eltype(lambda), typeof(b_max), typeof(a_max), Float64)
    inc = CTPoissonIncome{T}(collect(T, z), collect(T, lambda))
    return CTTwoAsset{T}(T(sigma), T(rho), T(r_a), T(r_b), T(chi), T(w), inc,
                          T(b_max), T(a_max), Ib, Ia)
end

"""
    CTTwoAssetSolution{T}

Solution of a continuous-time two-asset model: value `V`, consumption `c`, deposit `d`,
saving drifts `sb` (liquid) and `sa` (illiquid) — all `Ib×Ia×2` — the stationary joint
density `g`, aggregate liquid `B` and illiquid `A` holdings, the sparse generator `gen`, and
convergence flags.
"""
struct CTTwoAssetSolution{T<:AbstractFloat}
    b::Vector{T}
    a::Vector{T}
    V::Array{T,3}
    c::Array{T,3}
    d::Array{T,3}
    sb::Array{T,3}
    sa::Array{T,3}
    g::Array{T,3}
    B::T
    A::T
    gen::SparseMatrixCSC{T,Int}
    hjb_converged::Bool
end

# =============================================================================
# Solver
# =============================================================================

# Linear index for (i_b, j_a, k_z), column-major: b fastest, then a, then z.
@inline _idx2(i, j, k, Ib, Ia) = i + (j - 1) * Ib + (k - 1) * Ib * Ia

# Income switching block for the two-asset state space (2·Ib·Ia square).
function _ct2_aswitch(m::CTTwoAsset{T}) where {T<:AbstractFloat}
    Ib = m.Ib; Ia = m.Ia; la = m.income.lambda
    n = 2 * Ib * Ia
    rows = Int[]; cols = Int[]; vals = T[]
    for j in 1:Ia, i in 1:Ib
        k1 = _idx2(i, j, 1, Ib, Ia); k2 = _idx2(i, j, 2, Ib, Ia)
        push!(rows, k1); push!(cols, k1); push!(vals, -la[1])
        push!(rows, k1); push!(cols, k2); push!(vals, la[1])
        push!(rows, k2); push!(cols, k2); push!(vals, -la[2])
        push!(rows, k2); push!(cols, k1); push!(vals, la[2])
    end
    return sparse(rows, cols, vals, n, n)
end

"""
    _ct2_policy_and_generator(m, V, b, a, db, da, Aswitch) → (c, d, sb, sa, A)

Compute the consumption `c`, deposit `d`, liquid/illiquid drifts, and the sparse generator
from the value function `V` via an upwind scheme. Consumption is upwinded in the liquid
dimension; the deposit's illiquid direction is chosen by its sign (deposit ⇒ forward in `a`,
withdrawal ⇒ backward), which enforces the `a ≥ 0` / `a ≤ a_max` boundaries. The generator
is upwinded by the resulting drift signs, so it conserves probability mass by construction.
"""
function _ct2_policy_and_generator(m::CTTwoAsset{T}, V::Array{T,3}, b::Vector{T},
                                    a::Vector{T}, db::T, da::T,
                                    Aswitch::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    Ib = m.Ib; Ia = m.Ia; z = m.income.z; σ = m.sigma; χ = m.chi
    n = 2 * Ib * Ia
    c = zeros(T, Ib, Ia, 2); d = zeros(T, Ib, Ia, 2)
    sb = zeros(T, Ib, Ia, 2); sa = zeros(T, Ib, Ia, 2)
    rows = Int[]; cols = Int[]; vals = T[]

    @inbounds for k in 1:2
        for j in 1:Ia
            for i in 1:Ib
                inc = m.w * z[k]
                # Liquid marginal value (upwind in b): forward and backward.
                VbF = i < Ib ? (V[i+1, j, k] - V[i, j, k]) / db :
                               _ct_uprime(inc + m.r_b * b[Ib], σ)
                VbB = i > 1 ? (V[i, j, k] - V[i-1, j, k]) / db :
                              _ct_uprime(inc + m.r_b * b[1], σ)
                VbF = max(VbF, T(1e-12)); VbB = max(VbB, T(1e-12))
                # Illiquid marginal value (available directions enforce a-boundaries).
                VaF = j < Ia ? max((V[i, j+1, k] - V[i, j, k]) / da, T(1e-12)) : T(NaN)
                VaB = j > 1 ? max((V[i, j, k] - V[i, j-1, k]) / da, T(1e-12)) : T(NaN)

                # Deposit given a liquid marginal value Vb: pick the illiquid direction
                # by the deposit sign (deposit>0 ⇒ forward a; withdraw<0 ⇒ backward a).
                function dep(Vb)
                    if isfinite(VaF)
                        dF = (VaF / Vb - one(T)) / χ
                        dF > zero(T) && return dF
                    end
                    if isfinite(VaB)
                        dB = (VaB / Vb - one(T)) / χ
                        dB < zero(T) && return dB
                    end
                    return zero(T)
                end

                # Forward-b candidate.
                cF = VbF^(-one(T) / σ)
                dFc = dep(VbF)
                sbF = inc + m.r_b * b[i] - cF - dFc - (χ / 2) * dFc^2
                # Backward-b candidate.
                cB = VbB^(-one(T) / σ)
                dBc = dep(VbB)
                sbB = inc + m.r_b * b[i] - cB - dBc - (χ / 2) * dBc^2

                # Upwind selection in b.
                if sbF > zero(T)
                    cc = cF; dd = dFc; s_b = sbF
                elseif sbB < zero(T)
                    cc = cB; dd = dBc; s_b = sbB
                else
                    # Stay-put: zero liquid drift; deposit from the stay-put value.
                    Vb0 = _ct_uprime(max(inc + m.r_b * b[i], T(1e-10)), σ)
                    dd = dep(Vb0)
                    cc = max(inc + m.r_b * b[i] - dd - (χ / 2) * dd^2, T(1e-10))
                    s_b = zero(T)
                end
                cc = max(cc, T(1e-10))
                s_a = m.r_a * a[j] + dd

                c[i, j, k] = cc; d[i, j, k] = dd; sb[i, j, k] = s_b; sa[i, j, k] = s_a

                kk = _idx2(i, j, k, Ib, Ia)
                # Upwind transitions with reflecting boundaries: any flow that would
                # leave the grid is zeroed (both the off-diagonal AND its diagonal term),
                # so every row sums to zero — a valid, mass-conserving generator.
                Xb = i > 1 ? -min(s_b, zero(T)) / db : zero(T)    # to (i-1)
                Zb = i < Ib ? max(s_b, zero(T)) / db : zero(T)    # to (i+1)
                Xa = j > 1 ? -min(s_a, zero(T)) / da : zero(T)    # to (j-1)
                Za = j < Ia ? max(s_a, zero(T)) / da : zero(T)    # to (j+1)
                push!(rows, kk); push!(cols, kk); push!(vals, -(Xb + Zb + Xa + Za))
                Xb != zero(T) && (push!(rows, kk); push!(cols, _idx2(i-1, j, k, Ib, Ia)); push!(vals, Xb))
                Zb != zero(T) && (push!(rows, kk); push!(cols, _idx2(i+1, j, k, Ib, Ia)); push!(vals, Zb))
                Xa != zero(T) && (push!(rows, kk); push!(cols, _idx2(i, j-1, k, Ib, Ia)); push!(vals, Xa))
                Za != zero(T) && (push!(rows, kk); push!(cols, _idx2(i, j+1, k, Ib, Ia)); push!(vals, Za))
            end
        end
    end
    A = sparse(rows, cols, vals, n, n) + Aswitch
    return c, d, sb, sa, A
end

"""
    ct_two_asset_solve(m::CTTwoAsset; max_iter=200, tol=1e-6, Delta=1000.0) → CTTwoAssetSolution

Solve the two-asset household problem at the given returns/wage and compute the stationary
joint distribution. The HJB is iterated by the implicit upwind scheme; the stationary
density solves `Aᵀ g = 0` with `∫ g = 1`. Returns the value, policies, distribution,
aggregate liquid (`B`) and illiquid (`A`) holdings, and the generator.
"""
function ct_two_asset_solve(m::CTTwoAsset{T}; max_iter::Int=200, tol::Real=1e-6,
                             Delta::Real=1000.0) where {T<:AbstractFloat}
    Ib = m.Ib; Ia = m.Ia; z = m.income.z; σ = m.sigma; ρ = m.rho
    n = 2 * Ib * Ia
    b = collect(range(zero(T), m.b_max; length=Ib))
    a = collect(range(zero(T), m.a_max; length=Ia))
    db = b[2] - b[1]; da = a[2] - a[1]
    Δ = T(Delta)
    Aswitch = _ct2_aswitch(m)

    # Initial guess: consume liquid income plus both asset returns, no deposit.
    V = zeros(T, Ib, Ia, 2)
    for k in 1:2, j in 1:Ia, i in 1:Ib
        flow = m.w * z[k] + m.r_b * b[i] + m.r_a * a[j]
        V[i, j, k] = _ct_u(max(flow, T(1e-10)), σ) / ρ
    end

    c = zeros(T, Ib, Ia, 2); d = similar(c); sb = similar(c); sa = similar(c)
    A = spzeros(T, n, n)
    converged = false
    for _ in 1:max_iter
        c, d, sb, sa, A = _ct2_policy_and_generator(m, V, b, a, db, da, Aswitch)
        u_vec = vec([_ct_u(c[i, j, k], σ) for i in 1:Ib, j in 1:Ia, k in 1:2])
        B = (one(T) / Δ + ρ) * LinearAlgebra.I - A
        V_new = reshape(B \ (u_vec + vec(V) / Δ), Ib, Ia, 2)
        dist = maximum(abs.(V_new - V))
        V = V_new
        if dist < tol
            converged = true
            break
        end
    end
    c, d, sb, sa, A = _ct2_policy_and_generator(m, V, b, a, db, da, Aswitch)

    # Stationary distribution via the implicit KFE iterated to stationarity. This avoids a
    # near-singular direct solve of Aᵀ g = 0: (I − dt·Aᵀ) is strictly diagonally dominant
    # (hence nonsingular), and iterating drives g to the dominant (stationary) eigenvector.
    AT = copy(transpose(A))
    dt_kfe = T(1e4)                                # large step ⟹ fast projection
    Bk = LinearAlgebra.I - dt_kfe * AT             # nonsingular for any dt > 0
    g_vec = fill(one(T) / (n * db * da), n)        # uniform start, ∫ = 1
    for _ in 1:5000
        g_new = Bk \ g_vec
        g_new = max.(g_new, zero(T))
        g_new ./= (sum(g_new) * db * da)
        delta = maximum(abs.(g_new - g_vec))
        g_vec = g_new
        delta < T(1e-12) && break
    end
    kfe_ok = maximum(abs.(AT * g_vec)) < T(1e-6)   # true stationarity residual ‖Aᵀg‖
    converged = converged && kfe_ok
    g = reshape(g_vec, Ib, Ia, 2)

    # Aggregates.
    Bagg = zero(T); Aagg = zero(T)
    for k in 1:2, j in 1:Ia, i in 1:Ib
        Bagg += b[i] * g[i, j, k] * db * da
        Aagg += a[j] * g[i, j, k] * db * da
    end

    return CTTwoAssetSolution{T}(b, a, V, c, d, sb, sa, g, Bagg, Aagg, A, converged)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::CTTwoAsset{T}) where {T}
    print(io, "CTTwoAsset{$T}(σ=$(m.sigma), ρ=$(m.rho), r_a=$(m.r_a), r_b=$(m.r_b), ",
              "χ=$(m.chi), Ib=$(m.Ib), Ia=$(m.Ia))")
end

function Base.show(io::IO, s::CTTwoAssetSolution{T}) where {T}
    print(io, "CTTwoAssetSolution{$T}: liquid B=$(round(s.B; digits=4)), ",
              "illiquid A=$(round(s.A; digits=4)), converged=$(s.hjb_converged)")
end

"""
    report(s::CTTwoAssetSolution)

Print aggregate liquid and illiquid holdings, the illiquid share of wealth, and the
fraction of (poor) hand-to-mouth households (low liquid wealth).
"""
function report(io::IO, s::CTTwoAssetSolution{T}) where {T}
    total = s.B + s.A
    illiq_share = total > zero(T) ? s.A / total : zero(T)
    _show_spec_table(io, "Continuous-Time Two-Asset HANK (KMV-style) — Stationary Solution",
        ["Aggregate liquid B" => _fmt(s.B; digits=6), "Aggregate illiquid A" => _fmt(s.A; digits=6),
         "Illiquid wealth share" => _fmt(illiq_share; digits=4), "HJB converged" => _yesno(s.hjb_converged)])
    return nothing
end
report(s::CTTwoAssetSolution) = report(stdout, s)   # G-17 (#254): io-routed report
