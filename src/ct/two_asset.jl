# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Continuous-time two-asset heterogeneous-agent model in the spirit of Kaplan, Moll &
Violante (2018), solved with the finite-difference methods of Achdou et al. (2022).

Households hold a **liquid** asset `b` (return `r_b`) and an **illiquid** asset `a`
(return `r_a > r_b`). Moving funds between them — the **deposit** `d` — is costly, so
households tolerate a low liquid return to hold high-return illiquid wealth, generating a
large stock of illiquid wealth and a hand-to-mouth liquid margin (the central KMV
mechanism). The household solves

    ρ V(b,a,z) = max_{c,d} u(c)
                 + V_b·(w z − τ + r_b b − d − χ(d,a) − c)
                 + V_a·(r_a a + d)
                 + Σ_{z'} λ_{z→z'}[V(b,a,z') − V(b,a,z)]

with first-order conditions `c = (V_b)^{-1/σ}` and `∂χ/∂d = V_a/V_b − 1`. The stationary
joint density of `(b, a, z)` solves the Kolmogorov-Forward equation `Aᵀ g = 0`, sharing the
generator `A` with the HJB.

Two adjustment-cost specifications are available (`cost=`):

- `:quadratic` (default) — the smooth `χ(d) = (χ/2) d²`, giving `d = (V_a/V_b − 1)/χ`.
  Smooth ⇒ **no inaction region**: every household with `V_a ≠ V_b` adjusts, so the model
  cannot generate wealthy hand-to-mouth households.
- `:kinked` — the KMV linear-plus-quadratic `χ(d,a) = χ₀|d| + (χ₁/2)(d/ā)² ā`, with
  `ā = a + a_kink`. The `|d|` term makes the cost non-differentiable at `d = 0`, so the
  first-order condition holds only outside a band and the deposit policy is

      d = ā(V_a/V_b − 1 − χ₀)/χ₁   if V_a/V_b − 1 >  χ₀   (deposit)
      d = ā(V_a/V_b − 1 + χ₀)/χ₁   if V_a/V_b − 1 < −χ₀   (withdrawal)
      d = 0                        otherwise               (INACTION)

  The kink is resolved, not smoothed: the sign of the candidate deposit selects the upwind
  direction in `a`, and when neither sign condition holds the household is inactive. The
  resulting mass of households with illiquid wealth but no liquid buffer is the KMV
  **wealthy hand-to-mouth** group.

General equilibrium ([`ct_two_asset_ge`](@ref)) closes the model with a Cobb-Douglas firm
renting the illiquid asset as capital (`r_a = αZ(K/L)^{α−1} − δ`, `w = (1−α)Z(K/L)^α`) and
government bonds in fixed net supply `B_supply` financed by a lump-sum tax `τ = r_b·B_supply`.
[`ct_two_asset_mit`](@ref) computes the deterministic transition after an aggregate shock.

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

Continuous-time two-asset model parameters.

Household block: CRRA `sigma`, discount rate `rho`, illiquid and liquid returns `r_a > r_b`,
wage `w`, lump-sum tax `tau`, the two-state Poisson income process, and the liquid/illiquid
grids `[0, b_max]` (`Ib` points) and `[0, a_max]` (`Ia` points).

Adjustment cost: `cost = :quadratic` uses `chi`; `cost = :kinked` uses the KMV
`chi0·|d| + (chi1/2)(d/ā)²ā` with `ā = a + a_kink`.

General-equilibrium block (used only by [`ct_two_asset_ge`](@ref) and
[`ct_two_asset_mit`](@ref)): capital share `alpha`, depreciation `delta`, TFP `Z`, and the
net supply of liquid government bonds `B_supply`.
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
    cost::Symbol
    chi0::T
    chi1::T
    chi2::T
    a_kink::T
    dmax::T
    a_power::T
    b_power::T
    tau::T
    alpha::T
    delta::T
    Z::T
    B_supply::T
end

function CTTwoAsset(; sigma::Real=2.0, rho::Real=0.06, r_a::Real=0.05, r_b::Real=0.02,
                      chi::Real=2.0, w::Real=1.0, z::AbstractVector=[0.8, 1.2],
                      lambda::AbstractVector=[0.5, 0.5],
                      b_max::Real=20.0, a_max::Real=20.0, Ib::Int=40, Ia::Int=40,
                      cost::Symbol=:quadratic, chi0::Real=0.04383, chi1::Real=0.48236,
                      chi2::Real=0.40176, a_kink::Real=0.0219, dmax::Real=0.0,
                      a_power::Real=1.0, b_power::Real=1.0, tau::Real=0.0,
                      alpha::Real=0.36, delta::Real=0.05, Z::Real=1.0, B_supply::Real=1.0)
    @assert r_a > r_b "illiquid return r_a must exceed liquid return r_b"
    cost in (:quadratic, :kinked) || throw(ArgumentError(
        "CTTwoAsset: cost must be :quadratic or :kinked, got :$cost"))
    cost === :kinked && chi1 <= 0 && throw(ArgumentError(
        "CTTwoAsset: the kinked cost needs chi1 > 0 (the convex-term scale), got $chi1"))
    (0 < a_power <= 1) || throw(ArgumentError(
        "CTTwoAsset: a_power must lie in (0, 1] — 1 is uniform, smaller is more L-shaped, got $a_power"))
    (0 < b_power <= 1) || throw(ArgumentError(
        "CTTwoAsset: b_power must lie in (0, 1] — 1 is uniform, smaller is more L-shaped, got $b_power"))
    cost === :kinked && chi2 <= 0 && throw(ArgumentError(
        "CTTwoAsset: the kinked cost needs chi2 > 0 (the convex-term exponent), got $chi2"))
    cost === :kinked && chi0 < 0 && throw(ArgumentError(
        "CTTwoAsset: chi0 must be >= 0, got $chi0"))
    cost === :kinked && a_kink <= 0 && throw(ArgumentError(
        "CTTwoAsset: a_kink must be > 0 — it regularizes (d/a)^2 at a = 0, got $a_kink"))
    T = promote_type(typeof(sigma), typeof(rho), typeof(r_a), typeof(r_b), typeof(chi),
                     typeof(w), eltype(z), eltype(lambda), typeof(b_max), typeof(a_max),
                     typeof(chi0), typeof(chi1), typeof(chi2), typeof(a_kink), typeof(dmax),
                     typeof(a_power), typeof(b_power), typeof(tau),
                     typeof(alpha), typeof(delta), typeof(Z), typeof(B_supply), Float64)
    inc = CTPoissonIncome{T}(collect(T, z), collect(T, lambda))
    return CTTwoAsset{T}(T(sigma), T(rho), T(r_a), T(r_b), T(chi), T(w), inc,
                          T(b_max), T(a_max), Ib, Ia, cost, T(chi0), T(chi1), T(chi2),
                          T(a_kink), T(dmax > 0 ? dmax : a_max + b_max),
                          T(a_power), T(b_power), T(tau),
                          T(alpha), T(delta), T(Z), T(B_supply))
end

"""
    _ct2_reprice(m::CTTwoAsset, r_a, r_b, w, tau) → CTTwoAsset

Copy `m` with new prices. Used by the GE loop and the transition path, which re-solve the
same household block at many price vectors.
"""
function _ct2_reprice(m::CTTwoAsset{T}, r_a::T, r_b::T, w::T, tau::T) where {T<:AbstractFloat}
    return CTTwoAsset{T}(m.sigma, m.rho, r_a, r_b, m.chi, w, m.income, m.b_max, m.a_max,
                         m.Ib, m.Ia, m.cost, m.chi0, m.chi1, m.chi2, m.a_kink, m.dmax,
                         m.a_power, m.b_power, tau,
                         m.alpha, m.delta, m.Z, m.B_supply)
end

"""
    _ct2_deposit(m, R, a_eff) → d

Deposit policy from the adjustment-cost FOC `∂χ/∂d = R − 1`, where `R = V_a/V_b`.

For the smooth quadratic cost this is `d = (R − 1)/χ`, which is nonzero whenever `R ≠ 1`.
For the KMV kinked cost `χ = χ₀|d| + (χ₁/2)(d/ā)²ā` the marginal cost jumps from `−χ₀` to
`+χ₀` at `d = 0`, so the FOC has **no solution** for `|R − 1| ≤ χ₀` and the household is
inactive — the band that generates wealthy hand-to-mouth households.
"""
@inline function _ct2_deposit(m::CTTwoAsset{T}, R::T, a_eff::T) where {T<:AbstractFloat}
    x = R - one(T)
    if m.cost === :kinked
        # KMV `adjcostfn1inv`: the inverse marginal adjustment cost. The band
        # -chi0 <= x <= chi0 has NO solution to the FOC, so the deposit is exactly zero —
        # this is the inaction region, resolved rather than smoothed.
        # KMV cap the deposit at `dmax` ("maximum deposit rate, for numerical stability
        # while converging"). It matters: the inverse exponent is 1/chi2 = 2.49 at their
        # calibration, so a locally flat V_b (hence a large V_a/V_b) sends the raw FOC
        # deposit to ~1e9 and destroys the next HJB iterate.
        if x > m.chi0
            return min(m.chi1 * (x - m.chi0)^(one(T) / m.chi2) * a_eff, m.dmax)
        elseif x < -m.chi0
            return max(-m.chi1 * (-x - m.chi0)^(one(T) / m.chi2) * a_eff, -m.dmax)
        else
            return zero(T)
        end
    else
        return x / m.chi
    end
end

"""
    _ct2_adj_cost(m, d, a_eff) → χ(d, a)

Adjustment cost paid on a deposit `d` at illiquid holdings `a`.
"""
@inline function _ct2_adj_cost(m::CTTwoAsset{T}, d::T, a_eff::T) where {T<:AbstractFloat}
    d == zero(T) && return zero(T)
    if m.cost === :kinked
        # KMV `adjcostfn`: chi0*|x| + |x|^(1+chi2) * chi1^(-chi2) / (1+chi2), scaled by a_eff,
        # where x = d / a_eff is the deposit RATE.
        x = abs(d) / a_eff
        return (m.chi0 * x + x^(one(T) + m.chi2) * m.chi1^(-m.chi2) / (one(T) + m.chi2)) * a_eff
    else
        return (m.chi / 2) * d^2
    end
end

"""
    CTTwoAssetSolution{T}

Solution of a continuous-time two-asset model: value `V`, consumption `c`, deposit `d`,
saving drifts `sb` (liquid) and `sa` (illiquid) — all `Ib×Ia×2` — the stationary joint
density `g`, aggregate liquid `B` and illiquid `A` holdings, the sparse generator `gen`, and
convergence flags.

`hjb_converged` is `true` only when **both** the HJB and the stationary KFE converged;
`kfe_residual` reports the true stationarity residual `‖Aᵀg‖_∞` and `hjb_iterations` the
number of implicit HJB steps taken, so a non-converged solve is diagnosable rather than
silent.
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
    kfe_residual::T
    hjb_iterations::Int
    bdelta::Vector{T}
    adelta::Vector{T}

    # Trailing diagnostics are keywords with defaults, so the existing 12-positional
    # construction contract is unchanged.
    function CTTwoAssetSolution{T}(b, a, V, c, d, sb, sa, g, B, A, gen, hjb_converged;
                                   kfe_residual::Real=T(NaN),
                                   hjb_iterations::Int=0,
                                   bdelta::Vector{T}=_ct2_deltas(collect(T, b))[2],
                                   adelta::Vector{T}=_ct2_deltas(collect(T, a))[2]
                                   ) where {T<:AbstractFloat}
        new{T}(b, a, V, c, d, sb, sa, g, B, A, gen, hjb_converged,
               T(kfe_residual), hjb_iterations, bdelta, adelta)
    end
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
    _ct2_power_grid(T, lo, hi, n, k) → Vector{T}

Power-spaced grid, KMV's `PowerSpacedGrid`: `x = range(0,1,n)`, `z = x^(1/k)`,
`y = lo + (hi-lo)·z`. `k = 1` is uniform and `k → 0` is L-shaped, i.e. increasingly
concentrated near `lo`. KMV run `k = 0.15` on the illiquid grid and `k = 0.35` on the liquid
one, which is what lets 40 nodes span an illiquid ceiling of 2000× quarterly output.

Resolution matters here for a reason specific to the deposit FOC: it divides by `V_b`, and a
UNIFORM grid over a wide `[0, b_max]` leaves `V` almost flat in `b` at the top, so `V_b → 0`,
`V_a/V_b` explodes, and the FOC deposit runs into `dmax`.
"""
function _ct2_power_grid(::Type{T}, lo::Real, hi::Real, n::Int, k::Real) where {T<:AbstractFloat}
    n >= 2 || throw(ArgumentError("_ct2_power_grid: need n >= 2, got $n"))
    n == 2 && return T[lo, hi]
    x = range(zero(T), one(T); length=n)
    z = x .^ (one(T) / T(k))
    g = T(lo) .+ (T(hi) - T(lo)) .* z
    g[1] = T(lo); g[end] = T(hi)          # exact endpoints against roundoff
    return collect(T, g)
end

"""
    _ct2_deltas(g::Vector{T}) → (dg, delta)

Node spacings `dg[i] = g[i+1] - g[i]` (length `n-1`) and the trapezoidal integration weights
`delta` (length `n`) attached to each node, matching KMV's `dagrid`/`adelta`. On a uniform
grid `delta` is constant at the step, so every mass integral reduces to the old `db*da` form.
"""
function _ct2_deltas(g::Vector{T}) where {T<:AbstractFloat}
    n = length(g)
    dg = diff(g)
    delta = zeros(T, n)
    delta[1] = dg[1] / 2
    @inbounds for i in 2:(n - 1)
        delta[i] = (dg[i - 1] + dg[i]) / 2
    end
    delta[n] = dg[n - 1] / 2
    return dg, delta
end

"""
    _ct2_adj_scale(m, a) → ā

Illiquid scale in the KMV adjustment cost, `ā = max(κ₃, a)` (their `max(kappa3, la)`), which
keeps the deposit *rate* `d/ā` finite at the borrowing constraint.
"""
@inline _ct2_adj_scale(m::CTTwoAsset{T}, a::T) where {T<:AbstractFloat} = max(a, m.a_kink)

"""
    _ct2_policy_and_generator(m, V, b, a, dbg, dag, Aswitch) → (c, d, sb, sa, A)

Consumption, deposit, liquid/illiquid drifts and the sparse generator for the two-asset HJB,
following Kaplan, Moll & Violante (2018), `HJBUpdate.f90` in their replication code.

The scheme makes **two separate** upwind decisions, each with its own Hamiltonian:

1. **Consumption/saving** — three candidates (forward `b`, backward `b`, and the stationary
   point where the budget constraint replaces the FOC), compared by
   `H_c = u(c) + V_b·s` with `s = income + r_b·b − c`. The deposit is *excluded* from `s`
   here; it is added to the liquid drift afterwards.
2. **The deposit** — three candidates from the derivative pairs `(V_aF,V_bB)`, `(V_aB,V_bF)`
   and `(V_aB,V_bB)`, compared by the **deposit-only** Hamiltonian

       H_d = V_a·d − V_b·(d + χ(d,a))

   which is zero at `d = 0`. A candidate is admissible only if `H_d > 0` — that is the
   inaction test, and it is why the kinked cost produces a genuine inaction region rather
   than a smoothed one. The `(a,b)` direction pair must also be consistent with the flows the
   candidate implies, and the relevant test for the liquid direction is the sign of
   `d + χ(d,a)` (the net liquid outflow), not the sign of `d`:

   | candidate | admissible when |
   |---|---|
   | `(V_aF, V_bB)` deposit | `d > 0` and `H_d > 0` |
   | `(V_aB, V_bF)` withdraw, liquid rises | `d ≤ −χ(d,a)` and `H_d > 0` |
   | `(V_aB, V_bB)` withdraw, liquid falls | `−χ(d,a) < d ≤ 0` and `H_d > 0` |

   At most one is admissible; if none is, `d = 0`.

Comparing *full* Hamiltonians for the deposit (mixing in `u(c)` and the saving term) is what
made earlier attempts oscillate: it compares objectives built from different derivative
approximations, and the winner then flips with tiny changes in `V`.

Finally `s_b = s − d − χ(d,a)` and `s_a = r_a·a + d`. The generator is upwinded on those
drift signs with reflecting boundaries, so every row sums to zero and mass is conserved.
"""
function _ct2_policy_and_generator(m::CTTwoAsset{T}, V::Array{T,3}, b::Vector{T},
                                    a::Vector{T}, dbg::Vector{T}, dag::Vector{T},
                                    Aswitch::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    Ib = m.Ib; Ia = m.Ia; z = m.income.z; σ = m.sigma
    n = 2 * Ib * Ia
    c = zeros(T, Ib, Ia, 2); d = zeros(T, Ib, Ia, 2)
    sb = zeros(T, Ib, Ia, 2); sa = zeros(T, Ib, Ia, 2)
    rows = Int[]; cols = Int[]; vals = T[]

    c_min = T(1e-10)
    NEG = T(-1e12)

    @inbounds for k in 1:2
        for j in 1:Ia
            a_eff = _ct2_adj_scale(m, a[j])
            for i in 1:Ib
                inc = m.w * z[k] - m.tau
                res = inc + m.r_b * b[i]          # KMV `gbdrift + h*gnetwage`

                # Non-uniform spacings: the forward difference at `i` divides by `dbg[i]`,
                # the backward one by `dbg[i-1]`.
                VbF = i < Ib ? (V[i+1, j, k] - V[i, j, k]) / dbg[i] : T(NaN)
                VbB = i > 1 ? (V[i, j, k] - V[i-1, j, k]) / dbg[i-1] : T(NaN)
                VaF = j < Ia ? (V[i, j+1, k] - V[i, j, k]) / dag[j] : T(NaN)
                VaB = j > 1 ? (V[i, j, k] - V[i, j-1, k]) / dag[j-1] : T(NaN)

                # ── 1. consumption / saving (deposit excluded) ──
                # `OptimalConsumption`: from the FOC when V_b is usable, else from the budget
                # constraint at the stationary point (s ≡ 0).
                @inline function optcons(Vb)
                    if isfinite(Vb) && Vb > zero(T)
                        cc_ = Vb^(-one(T) / σ)
                        ss_ = res - cc_
                        return cc_, ss_, (cc_ > zero(T) ? _ct_u(cc_, σ) + Vb * ss_ : NEG)
                    else
                        cc_ = max(res, c_min)
                        return cc_, zero(T), _ct_u(cc_, σ)
                    end
                end

                cF, sF, HcF = i < Ib ? optcons(VbF) : (max(res, c_min), zero(T), NEG)
                i < Ib || (sF = zero(T))
                cB, sB, HcB = i > 1 ? optcons(VbB) : optcons(T(NaN))
                c0, s0, Hc0 = optcons(T(NaN))
                validF = (i < Ib) && sF > zero(T)
                validB = sB < zero(T)

                local cc::T, s_cons::T
                if validF && (!validB || HcF >= HcB) && HcF >= Hc0
                    cc = cF; s_cons = sF
                elseif validB && (!validF || HcB >= HcF) && HcB >= Hc0
                    cc = cB; s_cons = sB
                else
                    cc = c0; s_cons = s0
                end

                # ── 2. deposit: deposit-only Hamiltonian, H_d > 0 is the inaction test ──
                @inline function Hdep(dc, Va, Vb)
                    return Va * dc - Vb * (dc + _ct2_adj_cost(m, dc, a_eff))
                end

                dFB = zero(T); HdFB = NEG; okFB = false
                if j < Ia && i > 1 && isfinite(VaF) && isfinite(VbB) && VbB > zero(T)
                    dFB = _ct2_deposit(m, VaF / VbB, a_eff)
                    HdFB = Hdep(dFB, VaF, VbB)
                    okFB = dFB > zero(T) && HdFB > zero(T)
                end

                dBF = zero(T); HdBF = NEG; okBF = false
                if j > 1 && i < Ib && isfinite(VaB) && isfinite(VbF) && VbF > zero(T)
                    dBF = _ct2_deposit(m, VaB / VbF, a_eff)
                    HdBF = Hdep(dBF, VaB, VbF)
                    okBF = dBF <= -_ct2_adj_cost(m, dBF, a_eff) && HdBF > zero(T)
                end

                dBB = zero(T); HdBB = NEG; okBB = false
                if j > 1 && isfinite(VaB)
                    # At the liquid floor there is no backward difference; KMV substitute the
                    # marginal utility of the stationary-point consumption.
                    VbB_use = i == 1 ? _ct_uprime(max(cB, c_min), σ) : VbB
                    if isfinite(VbB_use) && VbB_use > zero(T)
                        dBB = _ct2_deposit(m, VaB / VbB_use, a_eff)
                        HdBB = Hdep(dBB, VaB, VbB_use)
                        okBB = dBB > -_ct2_adj_cost(m, dBB, a_eff) && dBB <= zero(T) &&
                               HdBB > zero(T)
                    end
                end

                dd = zero(T)
                if okFB && (!okBF || HdFB >= HdBF) && (!okBB || HdFB >= HdBB)
                    dd = dFB
                elseif okBF && (!okFB || HdBF >= HdFB) && (!okBB || HdBF >= HdBB)
                    dd = dBF
                elseif okBB && (!okFB || HdBB >= HdFB) && (!okBF || HdBB >= HdBF)
                    dd = dBB
                end

                # ── drifts ──
                s_b = s_cons - dd - _ct2_adj_cost(m, dd, a_eff)
                s_a = m.r_a * a[j] + dd
                cc = max(cc, c_min)

                c[i, j, k] = cc; d[i, j, k] = dd; sb[i, j, k] = s_b; sa[i, j, k] = s_a

                kk = _idx2(i, j, k, Ib, Ia)
                # Upwind transitions with reflecting boundaries: any flow that would leave
                # the grid is zeroed (both the off-diagonal AND its diagonal term), so every
                # row sums to zero — a valid, mass-conserving generator.
                Xb = i > 1 ? -min(s_b, zero(T)) / dbg[i-1] : zero(T)    # to (i-1)
                Zb = i < Ib ? max(s_b, zero(T)) / dbg[i] : zero(T)      # to (i+1)
                Xa = j > 1 ? -min(s_a, zero(T)) / dag[j-1] : zero(T)    # to (j-1)
                Za = j < Ia ? max(s_a, zero(T)) / dag[j] : zero(T)      # to (j+1)
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
    ct_two_asset_solve(m::CTTwoAsset; max_iter=200, tol=1e-6, Delta=1000.0,
                       V_init=nothing) → CTTwoAssetSolution

Solve the two-asset household problem at the given returns/wage and compute the stationary
joint distribution. The HJB is iterated by the implicit upwind scheme; the stationary
density solves `Aᵀ g = 0` with `∫ g = 1`. Returns the value, policies, distribution,
aggregate liquid (`B`) and illiquid (`A`) holdings, and the generator.

Pass `V_init` (an `Ib×Ia×2` array, e.g. the value function from a nearby price vector) to
warm-start the HJB. The general-equilibrium loop uses this to cut the cost of re-solving the
household block at each candidate price.
"""
function ct_two_asset_solve(m::CTTwoAsset{T}; max_iter::Int=200, tol::Real=1e-6,
                             Delta::Real=1000.0,
                             V_init::Union{Nothing,AbstractArray}=nothing) where {T<:AbstractFloat}
    Ib = m.Ib; Ia = m.Ia; z = m.income.z; σ = m.sigma; ρ = m.rho
    n = 2 * Ib * Ia
    b = _ct2_power_grid(T, zero(T), m.b_max, Ib, m.b_power)
    a = _ct2_power_grid(T, zero(T), m.a_max, Ia, m.a_power)
    dbg, bdelta = _ct2_deltas(b)
    dag, adelta = _ct2_deltas(a)
    # Mass weight per linear index, matching `_idx2` (b fastest, then a, then z).
    wvec = zeros(T, n)
    for kk in 1:2, jj in 1:Ia, ii in 1:Ib
        wvec[_idx2(ii, jj, kk, Ib, Ia)] = bdelta[ii] * adelta[jj]
    end
    Δ = T(Delta)
    Aswitch = _ct2_aswitch(m)

    # Initial guess: warm start, or consume liquid income plus both asset returns.
    V = if V_init !== nothing
        size(V_init) == (Ib, Ia, 2) || throw(ArgumentError(
            "ct_two_asset_solve: V_init must be $((Ib, Ia, 2)), got $(size(V_init))"))
        Array{T,3}(V_init)
    else
        V0 = zeros(T, Ib, Ia, 2)
        for k in 1:2, j in 1:Ia, i in 1:Ib
            flow = m.w * z[k] - m.tau + m.r_b * b[i] + m.r_a * a[j]
            V0[i, j, k] = _ct_u(max(flow, T(1e-10)), σ) / ρ
        end
        V0
    end

    c = zeros(T, Ib, Ia, 2); d = similar(c); sb = similar(c); sa = similar(c)
    A = spzeros(T, n, n)
    converged = false
    hjb_iters = 0
    for it in 1:max_iter
        hjb_iters = it
        c, d, sb, sa, A = _ct2_policy_and_generator(m, V, b, a, dbg, dag, Aswitch)
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
    c, d, sb, sa, A = _ct2_policy_and_generator(m, V, b, a, dbg, dag, Aswitch)

    # Stationary distribution via the implicit KFE iterated to stationarity. This avoids a
    # near-singular direct solve of Aᵀ g = 0: (I − dt·Aᵀ) is strictly diagonally dominant
    # (hence nonsingular), and iterating drives g to the dominant (stationary) eigenvector.
    AT = copy(transpose(A))
    dt_kfe = T(1e4)                                # large step ⟹ fast projection
    Bk = LinearAlgebra.I - dt_kfe * AT             # nonsingular for any dt > 0
    g_vec = fill(one(T) / sum(wvec), n)            # uniform start, ∫ = 1
    for _ in 1:5000
        g_new = Bk \ g_vec
        g_new = max.(g_new, zero(T))
        g_new ./= sum(g_new .* wvec)
        delta = maximum(abs.(g_new - g_vec))
        g_vec = g_new
        delta < T(1e-12) && break
    end
    kfe_res = maximum(abs.(AT * g_vec))            # true stationarity residual ‖Aᵀg‖
    converged = converged && (kfe_res < T(1e-6))
    g = reshape(g_vec, Ib, Ia, 2)

    # Aggregates.
    Bagg = zero(T); Aagg = zero(T)
    for k in 1:2, j in 1:Ia, i in 1:Ib
        w = bdelta[i] * adelta[j]
        Bagg += b[i] * g[i, j, k] * w
        Aagg += a[j] * g[i, j, k] * w
    end

    return CTTwoAssetSolution{T}(b, a, V, c, d, sb, sa, g, Bagg, Aagg, A, converged;
                                 kfe_residual=kfe_res, hjb_iterations=hjb_iters,
                                 bdelta=bdelta, adelta=adelta)
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

Print aggregate liquid and illiquid holdings, the illiquid share of wealth, the poor and
wealthy hand-to-mouth shares (see [`hand_to_mouth`](@ref)), the mass sitting on the illiquid
ceiling, and the convergence flags.
"""
function report(io::IO, s::CTTwoAssetSolution{T}) where {T}
    total = s.B + s.A
    illiq_share = total > zero(T) ? s.A / total : zero(T)
    htm = hand_to_mouth(s)
    cm = ceiling_mass(s)
    _show_spec_table(io, "Continuous-Time Two-Asset HANK (KMV-style) — Stationary Solution",
        ["Aggregate liquid B" => _fmt(s.B; digits=6), "Aggregate illiquid A" => _fmt(s.A; digits=6),
         "Illiquid wealth share" => _fmt(illiq_share; digits=4),
         "Poor hand-to-mouth" => _fmt(htm.poor; digits=4),
         "Wealthy hand-to-mouth" => _fmt(htm.wealthy; digits=4),
         "Mass on illiquid ceiling" => _fmt(cm.illiquid; digits=4),
         "HJB converged" => _yesno(s.hjb_converged),
         "HJB iterations" => string(s.hjb_iterations)])
    return nothing
end
report(s::CTTwoAssetSolution) = report(stdout, s)   # G-17 (#254): io-routed report

# =============================================================================
# Hand-to-mouth statistics and grid diagnostics
# =============================================================================

"""
    hand_to_mouth(s::CTTwoAssetSolution; b_threshold=nothing, a_threshold=nothing)
        → (poor, wealthy, total, b_threshold, a_threshold)

Share of households holding (almost) no liquid wealth, split by illiquid holdings.

A household is **hand-to-mouth** when `b ≤ b_threshold`. It is **poor** hand-to-mouth when it
also has `a ≤ a_threshold`, and **wealthy** hand-to-mouth when `a > a_threshold`: illiquid
wealth, no liquid buffer. The wealthy group is the target of Kaplan, Moll & Violante (2018)
and exists only under an adjustment cost with an **inaction region** — see `cost=:kinked`.

Defaults put `b_threshold` at one liquid grid step and `a_threshold` at one illiquid grid
step, i.e. "on the first grid cell". Supply explicit thresholds (e.g. a fraction of average
income `w·E[z]`) for a calibration-based definition.

# Examples
```julia
htm = hand_to_mouth(sol)
htm.wealthy      # share with illiquid wealth and no liquid buffer
```
"""
function hand_to_mouth(s::CTTwoAssetSolution{T};
                       b_threshold::Union{Nothing,Real}=nothing,
                       a_threshold::Union{Nothing,Real}=nothing) where {T<:AbstractFloat}
    b_thr = b_threshold === nothing ? s.bdelta[1] * 2 : T(b_threshold)
    a_thr = a_threshold === nothing ? s.adelta[1] * 2 : T(a_threshold)

    poor = zero(T); wealthy = zero(T)
    @inbounds for k in axes(s.g, 3), j in axes(s.g, 2), i in axes(s.g, 1)
        s.b[i] <= b_thr || continue
        mass = s.g[i, j, k] * s.bdelta[i] * s.adelta[j]
        if s.a[j] <= a_thr
            poor += mass
        else
            wealthy += mass
        end
    end
    return (poor=poor, wealthy=wealthy, total=poor + wealthy,
            b_threshold=b_thr, a_threshold=a_thr)
end

"""
    ceiling_mass(s::CTTwoAssetSolution) → (liquid, illiquid)

Probability mass sitting on the top node of each asset grid. Both should be negligible; a
non-negligible illiquid figure usually means the calibration is on the wrong side of the
stationarity condition (see [`ct_two_asset_stationarity`](@ref)) rather than that `a_max` is
merely too small.
"""
function ceiling_mass(s::CTTwoAssetSolution{T}) where {T<:AbstractFloat}
    liq = sum(s.g[end, j, k] * s.bdelta[end] * s.adelta[j]
              for j in axes(s.g, 2), k in axes(s.g, 3))
    ill = sum(s.g[i, end, k] * s.bdelta[i] * s.adelta[end]
              for i in axes(s.g, 1), k in axes(s.g, 3))
    return (liquid=liq, illiquid=ill)
end

"""
    ct_two_asset_stationarity(m::CTTwoAsset) → (ok, bound, message)

Check whether the calibration can support a **bounded** illiquid-wealth distribution.

In the no-deposit region illiquid wealth drifts at `s_a = r_a·a`, so halting it requires a
withdrawal that grows with `a`. Whether the adjustment cost can deliver one depends on its
specification:

- `cost = :quadratic` — `d = (V_a/V_b − 1)/χ` with `V_a/V_b ≥ 0`, so the largest withdrawal
  is the **constant** `1/χ`. Illiquid wealth diverges above `a* = 1/(χ·r_a)`; the model is
  usable only on a grid with `a_max < a*`.
- `cost = :kinked` — `d = ā(V_a/V_b − 1 + χ₀)/χ₁`, whose magnitude **scales with `a`**.
  Stationary for any `a_max` iff `χ₁ < (1 − χ₀)/r_a`.

Returns the check, the relevant bound, and a message. See issue #509.
"""
function ct_two_asset_stationarity(m::CTTwoAsset{T}) where {T<:AbstractFloat}
    if m.cost === :kinked
        bound = (one(T) - m.chi0) / m.r_a
        ok = m.chi1 < bound
        msg = ok ?
            "kinked cost: chi1 = $(m.chi1) < (1 - chi0)/r_a = $(round(bound; sigdigits=4)); " *
            "illiquid wealth is bounded." :
            "kinked cost: chi1 = $(m.chi1) >= (1 - chi0)/r_a = $(round(bound; sigdigits=4)). " *
            "The largest withdrawal a household can make is smaller than the return r_a*a " *
            "it accrues, so illiquid wealth diverges and the stationary distribution is a " *
            "grid artifact. Lower chi1 or lower r_a."
        return (ok=ok, bound=bound, message=msg)
    else
        bound = one(T) / (m.chi * m.r_a)
        ok = m.a_max < bound
        msg = ok ?
            "quadratic cost: a_max = $(m.a_max) < 1/(chi*r_a) = $(round(bound; sigdigits=4)); " *
            "illiquid wealth is bounded on this grid." :
            "quadratic cost: a_max = $(m.a_max) >= 1/(chi*r_a) = $(round(bound; sigdigits=4)). " *
            "The level-quadratic cost caps withdrawals at the constant 1/chi, which cannot " *
            "offset the return r_a*a, so illiquid wealth diverges above a* and the mass " *
            "piles onto the ceiling. Use cost=:kinked, or shrink a_max below a*. See #509."
        return (ok=ok, bound=bound, message=msg)
    end
end

# =============================================================================
# General equilibrium
# =============================================================================

"""
    CTTwoAssetGE{T}

Stationary general equilibrium of the continuous-time two-asset economy.

Fields: equilibrium prices `r_a`, `r_b`, `w` and the lump-sum tax `tau`; the aggregate
capital stock `K` (= illiquid wealth), liquid bond holdings `B`, effective labor `L` and
output `Y`; the underlying household `solution`; the two market-clearing residuals
`resid_illiquid = A − K` and `resid_liquid = B − B_supply`; and
`markets_cleared` / `converged` / `iterations`.

The two flags are reported **separately** on purpose. `markets_cleared` is `true` when both
residuals are within `tol`; `converged` additionally requires that the household block itself
converged. Market clearing can succeed while the inner HJB does not, and collapsing the two
would hide which of them failed.
"""
struct CTTwoAssetGE{T<:AbstractFloat}
    r_a::T
    r_b::T
    w::T
    tau::T
    K::T
    B::T
    L::T
    Y::T
    solution::CTTwoAssetSolution{T}
    resid_illiquid::T
    resid_liquid::T
    markets_cleared::Bool
    converged::Bool
    iterations::Int
end

# Effective aggregate labor supply: the stationary mean of the two-state Poisson process.
function _ct2_labor(m::CTTwoAsset{T}) where {T<:AbstractFloat}
    la = m.income.lambda; z = m.income.z
    π1 = la[2] / (la[1] + la[2])
    return z[1] * π1 + z[2] * (one(T) - π1)
end

"""
    ct_two_asset_ge(m::CTTwoAsset; kwargs...) → CTTwoAssetGE

Close the two-asset model in general equilibrium.

A representative Cobb-Douglas firm rents the illiquid asset as capital, so given `K` the
illiquid return and the wage are

```math
r_a = \\alpha Z (K/L)^{\\alpha-1} - \\delta, \\qquad w = (1-\\alpha) Z (K/L)^{\\alpha},
```

and liquid government bonds are in fixed net supply `B_supply`, financed by the lump-sum tax
`τ = r_b · B_supply` so the government budget balances. Equilibrium requires

- **illiquid market**: household illiquid wealth `A` equals the capital stock `K`
- **liquid market**: household liquid wealth `B` equals `B_supply`
- **labor market**: `L` is the stationary mean of the income process (labor is supplied
  inelastically), which the wage equation clears by construction

The solver iterates both conditions simultaneously with damped updates — `K ← K + relax_K(A − K)`
and `r_b ← r_b + relax_rb(B_supply − B)` — re-solving the household block at each step and
**warm-starting** it from the previous value function, which is what makes the loop
affordable. At fixed prices the household block reduces exactly to
[`ct_two_asset_solve`](@ref).

# Keyword Arguments
- `K_init`, `rb_init` — starting guesses (defaults: the representative-agent capital stock
  implied by `rho`, and `r_b = r_a/2`)
- `max_iter::Int=60`, `tol::Real=1e-4` — market-clearing iterations and tolerance
- `relax_K::Real=0.3`, `relax_rb::Real=0.02` — damping on the two updates
- `hjb_max_iter`, `hjb_tol`, `Delta` — passed to [`ct_two_asset_solve`](@ref)
- `verbose::Bool=false` — log each iteration's residuals through `@info`

# Returns
A [`CTTwoAssetGE`](@ref). Check `converged` and the two residuals before using the result.
"""
function ct_two_asset_ge(m::CTTwoAsset{T};
                         K_init::Union{Nothing,Real}=nothing,
                         rb_init::Union{Nothing,Real}=nothing,
                         max_iter::Int=60, tol::Real=1e-4,
                         relax_K::Real=0.3, relax_rb::Real=0.02,
                         hjb_max_iter::Int=200, hjb_tol::Real=1e-6,
                         Delta::Real=1000.0,
                         verbose::Bool=false) where {T<:AbstractFloat}
    L = _ct2_labor(m)
    α = m.alpha; δ = m.delta; Z = m.Z

    firm_ra(K) = α * Z * (K / L)^(α - one(T)) - δ
    firm_w(K)  = (one(T) - α) * Z * (K / L)^α
    firm_Y(K)  = Z * K^α * L^(one(T) - α)

    # Representative-agent capital: r_a = rho ⟹ K/L = (αZ/(ρ+δ))^{1/(1-α)}. Households are
    # risk-averse and face uninsurable risk, so equilibrium K exceeds this; it is a safe
    # lower starting point.
    K = T(something(K_init, (α * Z / (m.rho + δ))^(one(T) / (one(T) - α)) * L))
    r_b = T(something(rb_init, max(firm_ra(K) / 2, T(1e-4))))

    local sol
    V_warm = nothing
    cleared = false
    iters = 0
    resid_a = T(Inf); resid_b = T(Inf)
    # The prices that produced the RETURNED solution. They must be tracked separately from
    # the running iterates, which are advanced after the solve: returning the post-update
    # values breaks the contract that `ct_two_asset_solve` at the reported prices reproduces
    # the reported allocation, and makes the reported `r_a`, `w` inconsistent with `K`.
    K_used = K; ra_used = firm_ra(K); rb_used = r_b
    w_used = firm_w(K); tau_used = r_b * m.B_supply

    for it in 1:max_iter
        iters = it
        K = max(K, T(1e-6))
        r_a = firm_ra(K)
        w = firm_w(K)
        # Keep the model well posed: the illiquidity premium must be positive and the liquid
        # return below the discount rate (else liquid wealth itself diverges).
        # The liquid return must stay below the illiquid one (a positive illiquidity premium
        # is what makes the two-asset problem meaningful) and below the discount rate (else
        # liquid wealth itself diverges). The lower bound has to be generous: household
        # liquid demand can exceed any plausible bond supply, and a tight floor silently
        # pins `r_b` at the clamp and leaves the bond market uncleared.
        r_b = clamp(r_b, -m.rho, min(r_a - T(1e-4), m.rho - T(1e-4)))
        tau = r_b * m.B_supply
        K_used = K; ra_used = r_a; rb_used = r_b; w_used = w; tau_used = tau

        mk = _ct2_reprice(m, r_a, r_b, w, tau)
        sol = ct_two_asset_solve(mk; max_iter=hjb_max_iter, tol=hjb_tol, Delta=Delta,
                                 V_init=V_warm)
        V_warm = sol.V

        resid_a = sol.A - K
        resid_b = sol.B - m.B_supply

        if verbose
            @info "GE iteration $it: r_a=$(round(r_a; sigdigits=5)) r_b=$(round(r_b; sigdigits=5)) " *
                  "K=$(round(K; sigdigits=6)) A−K=$(round(resid_a; sigdigits=3)) " *
                  "B−B̄=$(round(resid_b; sigdigits=3))"
        end

        if abs(resid_a) < tol && abs(resid_b) < tol
            cleared = true
            break
        end

        K += T(relax_K) * resid_a          # excess illiquid demand raises the capital stock
        r_b += T(relax_rb) * (-resid_b)    # too little liquid held ⇒ the bond return must rise
    end

    converged = cleared && sol.hjb_converged
    if !converged
        @warn "ct_two_asset_ge did not fully converge in $iters iterations: markets cleared = " *
              "$cleared (illiquid residual A−K = $resid_a, liquid residual B−B_supply = " *
              "$resid_b), household block converged = $(sol.hjb_converged)." maxlog = 1
    end

    return CTTwoAssetGE{T}(ra_used, rb_used, w_used, tau_used, K_used, sol.B, L,
                           firm_Y(K_used), sol, resid_a, resid_b, cleared, converged, iters)
end

# =============================================================================
# MIT-shock transition
# =============================================================================

"""
    CTTwoAssetTransition{T}

Deterministic transition of the two-asset economy after an unanticipated aggregate shock.
All fields are length-`(N+1)` series on the time grid `t`: TFP `Z`, capital `K`, illiquid
and liquid returns `r_a`, `r_b`, wage `w`, aggregate liquid holdings `B`, and aggregate
consumption `C`.
"""
struct CTTwoAssetTransition{T<:AbstractFloat}
    t::Vector{T}
    Z::Vector{T}
    K::Vector{T}
    r_a::Vector{T}
    r_b::Vector{T}
    w::Vector{T}
    B::Vector{T}
    C::Vector{T}
    converged::Bool
    iterations::Int
end

"""
    ct_two_asset_mit(m::CTTwoAsset, ge0::CTTwoAssetGE, Z_path; kwargs...)
        → CTTwoAssetTransition

Perfect-foresight transition after an unanticipated aggregate TFP shock (an "MIT shock").

The economy starts at the stationary equilibrium `ge0` — so the joint distribution at `t = 0`
is `ge0.solution.g` — and, given a deterministic TFP path `Z_path` that returns to `m.Z`,
converges back to `ge0`. The algorithm shoots on **both** aggregate paths, mirroring the
one-asset [`ct_mit_shock`](@ref):

1. Given `{K_t, r_{b,t}, Z_t}`, set `r_{a,t} = αZ_t(K_t/L)^{α−1} − δ` and `w_t = (1−α)Z_t(K_t/L)^α`.
2. Solve the HJB **backward** from the terminal value `V(·,T) = ge0.solution.V`.
3. Solve the KFE **forward** from `g(·,0) = ge0.solution.g` with the time-`t` generators.
4. Update `K_t` toward household illiquid wealth and `r_{b,t}` toward liquid market clearing,
   by relaxation, until both paths converge.

`K_0` and `B_0` are pinned by the predetermined distribution and cannot move on impact.

# Keyword Arguments
- `dt::Real=0.25` — time step
- `max_iter::Int=200`, `tol::Real=1e-5` — shooting iterations and path tolerance
- `relax_K::Real=0.3`, `relax_rb::Real=0.02` — damping on the two path updates
- `verbose::Bool=false`
"""
function ct_two_asset_mit(m::CTTwoAsset{T}, ge0::CTTwoAssetGE{T}, Z_path::AbstractVector;
                          dt::Real=0.25, max_iter::Int=200, tol::Real=1e-5,
                          relax_K::Real=0.3, relax_rb::Real=0.02,
                          verbose::Bool=false) where {T<:AbstractFloat}
    Np1 = length(Z_path)
    Np1 >= 2 || throw(ArgumentError("ct_two_asset_mit: Z_path needs at least 2 points"))
    N = Np1 - 1
    Z = collect(T, Z_path)
    dt_T = T(dt)

    Ib = m.Ib; Ia = m.Ia; σ = m.sigma; ρ = m.rho
    L = ge0.L; α = m.alpha; δ = m.delta
    b = ge0.solution.b; a = ge0.solution.a
    dbg, bdelta = _ct2_deltas(b)
    dag, adelta = _ct2_deltas(a)
    Aswitch = _ct2_aswitch(m)

    firm_ra(Zt, Kt) = α * Zt * (Kt / L)^(α - one(T)) - δ
    firm_w(Zt, Kt)  = (one(T) - α) * Zt * (Kt / L)^α

    K = fill(ge0.K, Np1)
    r_b = fill(ge0.r_b, Np1)
    g0 = vec(ge0.solution.g)
    VT = vec(ge0.solution.V)

    # Asset value per linear index, matching `_idx2` (b fastest, then a, then z).
    bvec = zeros(T, 2 * Ib * Ia); avec = zeros(T, 2 * Ib * Ia); wvec = zeros(T, 2 * Ib * Ia)
    for k in 1:2, j in 1:Ia, i in 1:Ib
        idx = _idx2(i, j, k, Ib, Ia)
        bvec[idx] = b[i]; avec[idx] = a[j]; wvec[idx] = bdelta[i] * adelta[j]
    end

    Avec = Vector{SparseMatrixCSC{T,Int}}(undef, Np1)
    cvec = [zeros(T, Ib, Ia, 2) for _ in 1:Np1]
    converged = false
    iters = 0

    for outer_it in 1:max_iter
        iters = outer_it
        ra_path = [firm_ra(Z[n], K[n]) for n in 1:Np1]
        w_path  = [firm_w(Z[n], K[n]) for n in 1:Np1]
        rb_path = [clamp(r_b[n], T(-0.5) * abs(ra_path[n]),
                         min(ra_path[n] - T(1e-4), ρ - T(1e-4))) for n in 1:Np1]

        # ── Backward HJB from the terminal steady-state value ──
        V = copy(VT)
        for n in N:-1:1
            mk = _ct2_reprice(m, ra_path[n], rb_path[n], w_path[n], rb_path[n] * m.B_supply)
            Varr = reshape(V, Ib, Ia, 2)
            c, _, _, _, A = _ct2_policy_and_generator(mk, Varr, b, a, dbg, dag, Aswitch)
            Avec[n] = A
            cvec[n] = c
            u_vec = vec([_ct_u(c[i, j, k], σ) for i in 1:Ib, j in 1:Ia, k in 1:2])
            V = ((one(T) / dt_T + ρ) * LinearAlgebra.I - A) \ (u_vec + V / dt_T)
        end
        Avec[Np1] = Avec[max(N, 1)]
        cvec[Np1] = cvec[max(N, 1)]

        # ── Forward KFE from the initial steady-state distribution ──
        gcur = copy(g0)
        K_new = similar(K); B_new = similar(K)
        K_new[1] = sum(avec .* gcur .* wvec)
        B_new[1] = sum(bvec .* gcur .* wvec)
        for n in 1:N
            Bk = LinearAlgebra.I - dt_T * copy(transpose(Avec[n]))
            gnext = Bk \ gcur
            gnext = max.(gnext, zero(T))
            gnext ./= sum(gnext .* wvec)
            gcur = gnext
            K_new[n+1] = sum(avec .* gcur .* wvec)
            B_new[n+1] = sum(bvec .* gcur .* wvec)
        end

        diff_K = maximum(abs.(K_new .- K))
        diff_B = maximum(abs.(B_new .- m.B_supply))
        if verbose
            @info "MIT iteration $outer_it: ‖ΔK‖∞ = $(round(diff_K; sigdigits=3)), " *
                  "‖B − B̄‖∞ = $(round(diff_B; sigdigits=3))"
        end

        # K_0 and B_0 are pinned by the predetermined distribution; relax the rest.
        for n in 2:Np1
            K[n] += T(relax_K) * (K_new[n] - K[n])
            r_b[n] += T(relax_rb) * (m.B_supply - B_new[n])
        end

        if diff_K < tol && diff_B < tol
            converged = true
            break
        end
    end

    # Final pass: prices, liquid holdings and aggregate consumption along the converged path.
    ra_path = [firm_ra(Z[n], K[n]) for n in 1:Np1]
    w_path  = [firm_w(Z[n], K[n]) for n in 1:Np1]
    rb_path = [clamp(r_b[n], T(-0.5) * abs(ra_path[n]),
                     min(ra_path[n] - T(1e-4), ρ - T(1e-4))) for n in 1:Np1]
    Cpath = zeros(T, Np1); Bpath = zeros(T, Np1)
    gcur = copy(g0)
    for n in 1:Np1
        Bpath[n] = sum(bvec .* gcur .* wvec)
        Cpath[n] = sum(vec(cvec[n]) .* gcur .* wvec)
        if n <= N
            Bk = LinearAlgebra.I - dt_T * copy(transpose(Avec[n]))
            gnext = Bk \ gcur
            gnext = max.(gnext, zero(T)); gnext ./= sum(gnext .* wvec)
            gcur = gnext
        end
    end

    if !converged
        @warn "ct_two_asset_mit did not converge in $iters shooting iterations." maxlog = 1
    end

    tgrid = collect(T, range(zero(T); step=dt_T, length=Np1))
    return CTTwoAssetTransition{T}(tgrid, Z, copy(K), ra_path, rb_path, w_path,
                                   Bpath, Cpath, converged, iters)
end

# =============================================================================
# Display — GE and transition
# =============================================================================

function Base.show(io::IO, ge::CTTwoAssetGE{T}) where {T}
    print(io, "CTTwoAssetGE{$T}: r_a=$(round(ge.r_a; digits=5)), r_b=$(round(ge.r_b; digits=5)), ",
              "K=$(round(ge.K; digits=4)), B=$(round(ge.B; digits=4)), converged=$(ge.converged)")
end

function Base.show(io::IO, tr::CTTwoAssetTransition{T}) where {T}
    print(io, "CTTwoAssetTransition{$T}: ", length(tr.t), " periods, ",
              "converged=$(tr.converged), iterations=$(tr.iterations)")
end

"""
    report(ge::CTTwoAssetGE)

Print the equilibrium prices, aggregates, market-clearing residuals and hand-to-mouth shares.
"""
function report(io::IO, ge::CTTwoAssetGE{T}) where {T}
    htm = hand_to_mouth(ge.solution)
    cm = ceiling_mass(ge.solution)
    _show_spec_table(io, "Continuous-Time Two-Asset HANK — Stationary General Equilibrium",
        ["Illiquid return r_a" => _fmt(ge.r_a; digits=6),
         "Liquid return r_b" => _fmt(ge.r_b; digits=6),
         "Wage w" => _fmt(ge.w; digits=6),
         "Lump-sum tax tau" => _fmt(ge.tau; digits=6),
         "Capital K" => _fmt(ge.K; digits=6),
         "Liquid bonds B" => _fmt(ge.B; digits=6),
         "Output Y" => _fmt(ge.Y; digits=6),
         "Illiquid residual A-K" => _fmt(ge.resid_illiquid; digits=3),
         "Liquid residual B-Bbar" => _fmt(ge.resid_liquid; digits=3),
         "Markets cleared" => _yesno(ge.markets_cleared),
         "Poor hand-to-mouth" => _fmt(htm.poor; digits=4),
         "Wealthy hand-to-mouth" => _fmt(htm.wealthy; digits=4),
         "Mass on illiquid ceiling" => _fmt(cm.illiquid; digits=4),
         "Converged" => _yesno(ge.converged),
         "Iterations" => string(ge.iterations)])
    return nothing
end
report(ge::CTTwoAssetGE) = report(stdout, ge)

