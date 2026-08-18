# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heterogeneous plant / firm population (`FirmSystem`) and the Khan–Thomas
(2008) lumpy-investment example.

This is **not** a [`HouseholdSystem`](@ref) with another budget. Plants choose
labor statically and capital subject to a stochastic fixed adjustment cost,
which generates an (S,s) inaction region. Aggregation returns `K`, `N`, `Y`
(and investment). General equilibrium is closed by a representative
Hansen–Rogerson household (`w = φ C`), not a second heterogeneous population.

# References
- Khan, A., & Thomas, J. K. (2008). Idiosyncratic shocks and the role of
  nonconvexities in plant and aggregate investment dynamics. *Econometrica*,
  76(2), 395–436.
"""

# =============================================================================
# FirmSystem
# =============================================================================

"""
    FirmSystem{T} <: AbstractAgentSystem{T}

Plant population on a capital grid with idiosyncratic productivity and a
fixed capital-adjustment cost. The `ModelSpec.agents` key is a free population
name; this type is the problem kind.

# Fields
- `k_grid` — plant capital nodes
- `productivity` — idiosyncratic TFP Markov chain (`ε`, levels)
- `alpha`, `nu` — decreasing-returns production `z ε k^α n^ν` (`α + ν < 1`)
- `delta`, `beta`, `gamma` — depreciation, discount, BGP growth
- `xi_bar` — upper support of `ξ ~ Uniform[0, ξ̄]` (labor units)
- `b` — costless investment band `|i/k| ≤ b` (Khan–Thomas `a = −b`)
- `phi` — Hansen–Rogerson leisure weight (`u = log C + φ (1 − N)`)
- `rho_z`, `sigma_z`, `Z` — aggregate TFP AR(1) in logs and SS level
"""
struct FirmSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    k_grid::Vector{T}
    productivity::IncomeProcess{T}
    alpha::T
    nu::T
    delta::T
    beta::T
    gamma::T
    xi_bar::T
    b::T
    phi::T
    rho_z::T
    sigma_z::T
    Z::T

    function FirmSystem{T}(k_grid::AbstractVector, productivity::IncomeProcess{T},
                           alpha, nu, delta, beta, gamma, xi_bar, b, phi,
                           rho_z, sigma_z, Z) where {T<:AbstractFloat}
        kg = collect(T, k_grid)
        n = length(kg)
        n >= 3 || throw(ArgumentError("FirmSystem: k_grid needs at least 3 nodes, got $n"))
        issorted(kg) || throw(ArgumentError("FirmSystem: k_grid must be sorted ascending"))
        kg[1] > zero(T) || throw(ArgumentError("FirmSystem: k_grid must be strictly positive"))
        all(>(zero(T)), productivity.states) || throw(ArgumentError(
            "FirmSystem: idiosyncratic productivity states must be positive (levels, not logs)"))
        zero(T) < T(alpha) || throw(ArgumentError("FirmSystem: alpha must be positive"))
        zero(T) < T(nu) || throw(ArgumentError("FirmSystem: nu must be positive"))
        T(alpha) + T(nu) < one(T) || throw(ArgumentError(
            "FirmSystem: decreasing returns require alpha + nu < 1, got $(T(alpha) + T(nu))"))
        zero(T) < T(delta) < one(T) || throw(ArgumentError("FirmSystem: delta must lie in (0, 1)"))
        zero(T) < T(beta) < one(T) || throw(ArgumentError("FirmSystem: beta must lie in (0, 1)"))
        T(gamma) >= one(T) || throw(ArgumentError("FirmSystem: gamma must be ≥ 1"))
        T(xi_bar) >= zero(T) || throw(ArgumentError("FirmSystem: xi_bar must be non-negative"))
        T(b) >= zero(T) || throw(ArgumentError("FirmSystem: exemption band b must be non-negative"))
        T(phi) > zero(T) || throw(ArgumentError("FirmSystem: phi must be positive"))
        T(Z) > zero(T) || throw(ArgumentError("FirmSystem: Z must be positive"))
        abs(T(rho_z)) < one(T) || throw(ArgumentError("FirmSystem: |rho_z| must be < 1"))
        T(sigma_z) >= zero(T) || throw(ArgumentError("FirmSystem: sigma_z must be non-negative"))
        new{T}(kg, productivity, T(alpha), T(nu), T(delta), T(beta),
               T(gamma), T(xi_bar), T(b), T(phi), T(rho_z), T(sigma_z), T(Z))
    end
end

grid(s::FirmSystem) = s.k_grid
idiosyncratic(s::FirmSystem) = s.productivity
aggregation(::FirmSystem) = [:K, :N, :Y]
ssj_inputs(::FirmSystem) = [:w, :Z]
ssj_outputs(::FirmSystem) = [:K, :N, :Y, :I]

# =============================================================================
# Khan–Thomas 2008 calibration (Econometrica 76(2), Table 1)
# =============================================================================

# Table 1 (NBER WP 12845 / Econometrica 2008):
#   γ = 1.016, β = 0.977, δ = 0.069, α = 0.256, ν = 0.640, φ = 2.400,
#   ρ_z = 0.859, σ_ηz = 0.014, ρ_ε = 0.859, σ_ηε = 0.022, b = 0.011, ξ̄ = 0.0083.
# Text (p. 12) targets I/K = 0.10 (US FAT 1954–2002). On the BGP,
#   I/K = γ − 1 + δ = 0.085.
# Table 2 row (2) lumpy + plant-specific TFP: inaction 0.073 (LRD 0.081).

const _KT_GAMMA = 1.016
const _KT_BETA = 0.977
const _KT_DELTA = 0.069
const _KT_ALPHA = 0.256
const _KT_NU = 0.640
const _KT_PHI = 2.400
const _KT_RHO_Z = 0.859
const _KT_SIGMA_Z = 0.014
const _KT_RHO_E = 0.859
const _KT_SIGMA_E = 0.022
const _KT_B = 0.011
const _KT_XI_BAR = 0.0083

"""
    _kt_unit_mean_productivity(rho, sigma, n) → IncomeProcess{Float64}

Rouwenhorst discretization of `log ε' = ρ log ε + σ η`, converted to **levels**
and normalized to unit mean. `sigma` is the innovation standard deviation
(Khan–Thomas Table 1, `σ_ηε`).
"""
function _kt_unit_mean_productivity(rho::Real, sigma::Real, n::Int)
    raw = rouwenhorst(rho, sigma, n; sigma_is=:innovation)
    e = exp.(raw.states)
    e ./= dot(raw.stationary_dist, e)
    return IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :productivity)
end

"""
    _kt_static(z, e, k, α, ν, w) → (n, y, profit)

Static labor demand, output, and operating profit at a plant `(k, ε)`.
"""
function _kt_static(z::T, e::T, k::T, α::T, ν::T, w::T) where {T<:AbstractFloat}
    (k <= zero(T) || w <= zero(T) || z <= zero(T) || e <= zero(T)) &&
        return (zero(T), zero(T), zero(T))
    n = (ν * z * e * k^α / w)^(one(T) / (one(T) - ν))
    isfinite(n) || return (zero(T), zero(T), zero(T))
    y = z * e * k^α * n^ν
    return (n, y, y - w * n)
end

"""
    _kt_frictionless_k(z, e, α, ν, w, γ, β, δ) → k*

Unconstrained target capital of a plant with current idiosyncratic `e`
(independent of current `k`). User cost is `γ/β − (1 − δ)`.
"""
function _kt_frictionless_k(z::T, e::T, α::T, ν::T, w::T,
                            γ::T, β::T, δ::T) where {T<:AbstractFloat}
    θ = α / (one(T) - ν)
    uc = γ / β - (one(T) - δ)
    uc <= zero(T) && return T(Inf)
    A = (z * e)^(one(T) / (one(T) - ν)) * ν^(ν / (one(T) - ν)) * w^(-ν / (one(T) - ν))
    pref = θ * (one(T) - ν) * A / uc
    pref <= zero(T) && return zero(T)
    return pref^(one(T) / (one(T) - θ))
end

function _kt_frictionless_aggregates(fs::FirmSystem{T}, w::T, z::T) where {T<:AbstractFloat}
    e_vals = fs.productivity.states
    πe = fs.productivity.stationary_dist
    K = zero(T)
    N = zero(T)
    Y = zero(T)
    @inbounds for j in eachindex(e_vals)
        k = _kt_frictionless_k(z, e_vals[j], fs.alpha, fs.nu, w, fs.gamma, fs.beta, fs.delta)
        n, y, _ = _kt_static(z, e_vals[j], k, fs.alpha, fs.nu, w)
        K += πe[j] * k
        N += πe[j] * n
        Y += πe[j] * y
    end
    I = (fs.gamma - (one(T) - fs.delta)) * K
    C = Y - I
    return (K=K, N=N, Y=Y, I=I, C=C)
end

"""
    _kt_frictionless_wage(fs; z=fs.Z, w_lo=1e-3, w_hi=20, max_iter=80) → (w, agg)

Representative-household wage that clears the frictionless nested economy
(`ξ̄ = 0`): `w = φ C` with `C = Y − I`.
"""
function _kt_frictionless_wage(fs::FirmSystem{T}; z::T=fs.Z,
                               w_lo::T=T(1e-3), w_hi::T=T(20),
                               max_iter::Int=80) where {T<:AbstractFloat}
    φ = fs.phi
    function resid(w)
        agg = _kt_frictionless_aggregates(fs, w, z)
        return w - φ * agg.C
    end
    lo, hi = w_lo, w_hi
    rlo, rhi = resid(lo), resid(hi)
    if rlo * rhi > zero(T)
        # Expand the bracket once; fall back to the closer endpoint.
        for _ in 1:8
            lo = max(lo / T(2), T(1e-6))
            hi = hi * T(2)
            rlo, rhi = resid(lo), resid(hi)
            rlo * rhi <= zero(T) && break
        end
    end
    if rlo * rhi > zero(T)
        w = abs(rlo) < abs(rhi) ? lo : hi
        return (w, _kt_frictionless_aggregates(fs, w, z))
    end
    w = lo
    for _ in 1:max_iter
        w = (lo + hi) / T(2)
        r = resid(w)
        abs(r) < T(1e-8) && break
        if rlo * r < zero(T)
            hi = w
            rhi = r
        else
            lo = w
            rlo = r
        end
    end
    return (w, _kt_frictionless_aggregates(fs, w, z))
end

function _kt_default_grid(prod::IncomeProcess{T}, alpha, nu, delta, beta, gamma, Z;
                          n_k::Int=16, w::T=one(T)) where {T<:AbstractFloat}
    ks = [_kt_frictionless_k(Z, e, alpha, nu, w, gamma, beta, delta) for e in prod.states]
    k_lo = T(0.15) * minimum(ks)
    k_hi = T(6) * maximum(ks)
    k_lo = max(k_lo, T(1e-4))
    k_hi = max(k_hi, k_lo * T(4))
    return _make_asset_grid(k_lo, k_hi, n_k, :geometric)
end

"""
    khan_thomas_example(; n_k=16, n_eps=3, kwargs...) → FirmSystem{Float64}

Coarse-grid Khan–Thomas (2008) plant population. Table 1 parameters are the
defaults; `n_k` / `n_eps` default to a CI-friendly grid, not the paper's
15-state idiosyncratic chain.

Pass `xi_bar` / `b` / `n_k` to refine. The capital grid is built around the
frictionless targets at the representative-household wage.
"""
function khan_thomas_example(; n_k::Int=16, n_eps::Int=3,
                             alpha::Real=_KT_ALPHA, nu::Real=_KT_NU,
                             delta::Real=_KT_DELTA, beta::Real=_KT_BETA,
                             gamma::Real=_KT_GAMMA, xi_bar::Real=_KT_XI_BAR,
                             b::Real=_KT_B, phi::Real=_KT_PHI,
                             rho_z::Real=_KT_RHO_Z, sigma_z::Real=_KT_SIGMA_Z,
                             rho_e::Real=_KT_RHO_E, sigma_e::Real=_KT_SIGMA_E,
                             Z::Real=1.0,
                             k_grid::Union{Nothing,AbstractVector}=nothing)
    T = Float64
    prod = _kt_unit_mean_productivity(rho_e, sigma_e, n_eps)
    α, ν, δ, β, γ = T(alpha), T(nu), T(delta), T(beta), T(gamma)
    z = T(Z)
    # Temporary system so the frictionless wage / grid share one calibration.
    kg_probe = k_grid === nothing ?
        _kt_default_grid(prod, α, ν, δ, β, γ, z; n_k=n_k, w=one(T)) :
        collect(T, k_grid)
    probe = FirmSystem{T}(kg_probe, prod, α, ν, δ, β, γ, T(xi_bar), T(b),
                          T(phi), T(rho_z), T(sigma_z), z)
    w_ss, _ = _kt_frictionless_wage(probe)
    kg = k_grid === nothing ?
        _kt_default_grid(prod, α, ν, δ, β, γ, z; n_k=n_k, w=w_ss) :
        collect(T, k_grid)
    return FirmSystem{T}(kg, prod, α, ν, δ, β, γ, T(xi_bar), T(b),
                         T(phi), T(rho_z), T(sigma_z), z)
end

# =============================================================================
# Plant VFI — (S,s) with stochastic fixed cost
# =============================================================================

"""
    _kt_lambda_bounds(k, fs) → (k_lo, k_hi)

Costless capital set `Λ(k)` of Khan–Thomas eq. (1) with `a = −b`.
"""
function _kt_lambda_bounds(k::T, fs::FirmSystem{T}) where {T<:AbstractFloat}
    ginv = one(T) / fs.gamma
    lo = (one(T) - fs.delta - fs.b) * k * ginv
    hi = (one(T) - fs.delta + fs.b) * k * ginv
    return (max(lo, fs.k_grid[1]), min(hi, fs.k_grid[end]))
end

function _kt_invest_rate(k::T, kp::T, fs::FirmSystem{T}) where {T<:AbstractFloat}
    return (fs.gamma * kp - (one(T) - fs.delta) * k) / k
end

"""
    _kt_plant_vfi(fs, w, z; p=1, max_iter=200, tol=1e-6, howard_steps=15,
                  init_value=nothing) → NamedTuple

Solve the plant Bellman at constant prices `(w, p)` and aggregate TFP `z`.
Returns value, unconstrained target `k* (ε)`, constrained `k^C(k, ε)`,
adjustment probability `G(ξ^T)`, and static labor.
"""
function _kt_plant_vfi(fs::FirmSystem{T}, w::T, z::T;
                       p::T=one(T),
                       max_iter::Int=200, tol::T=T(1e-6),
                       howard_steps::Int=15,
                       init_value::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    k_grid = fs.k_grid
    n_k = length(k_grid)
    e_vals = fs.productivity.states
    n_e = length(e_vals)
    Pi = fs.productivity.transition
    β = fs.beta
    γ = fs.gamma
    δ = fs.delta
    ξ̄ = fs.xi_bar
    α, ν = fs.alpha, fs.nu

    profit = zeros(T, n_k, n_e)
    n_pol = zeros(T, n_k, n_e)
    @inbounds for j in 1:n_e, i in 1:n_k
        n, y, π = _kt_static(z, e_vals[j], k_grid[i], α, ν, w)
        n_pol[i, j] = n
        profit[i, j] = π
    end

    V = zeros(T, n_k, n_e)
    if init_value !== nothing && size(init_value) == (n_k, n_e)
        copyto!(V, init_value)
    else
        @inbounds for j in 1:n_e, i in 1:n_k
            V[i, j] = p * (profit[i, j] + (one(T) - δ) * k_grid[i]) / (one(T) - β)
        end
    end

    EV = zeros(T, n_k, n_e)
    V_new = similar(V)
    k_star = fill(k_grid[1], n_e)
    k_con = zeros(T, n_k, n_e)
    adj_prob = zeros(T, n_k, n_e)

    converged = false
    for iter in 1:max_iter
        # Expected continuation on the capital grid, current ε → future ε.
        @inbounds for j in 1:n_e, i in 1:n_k
            ev = zero(T)
            for jp in 1:n_e
                ev += Pi[j, jp] * V[i, jp]
            end
            EV[i, j] = ev
        end

        @inbounds for j in 1:n_e
            ev_j = view(EV, :, j)
            R = let ev_j=ev_j, k_grid=k_grid, β=β, γ=γ, p=p
                kp -> -γ * p * kp + β * _linear_interp(k_grid, ev_j, kp)
            end
            # Unconstrained target depends only on ε.
            best_idx = 1
            best_val = R(k_grid[1])
            for i in 2:n_k
                val = R(k_grid[i])
                if val > best_val
                    best_val = val
                    best_idx = i
                end
            end
            lo_r = best_idx > 1 ? k_grid[best_idx - 1] : k_grid[1]
            hi_r = best_idx < n_k ? k_grid[best_idx + 1] : k_grid[n_k]
            kp_u = k_grid[best_idx]
            Eu = best_val
            if hi_r > lo_r + T(1e-14)
                kp_r, val_r = _golden_argmax(R, lo_r, hi_r)
                if isfinite(val_r) && val_r >= Eu
                    kp_u = kp_r
                    Eu = val_r
                end
            end
            k_star[j] = kp_u

            for i in 1:n_k
                lo_c, hi_c = _kt_lambda_bounds(k_grid[i], fs)
                if hi_c <= lo_c + T(1e-14)
                    kp_c = clamp(kp_u, k_grid[1], k_grid[end])
                    Ec = R(kp_c)
                else
                    # If the unconstrained target is already inside Λ(k), the
                    # plant adjusts for free (ξ^T = 0).
                    if lo_c <= kp_u <= hi_c
                        kp_c = kp_u
                        Ec = Eu
                    else
                        kp_c, Ec = _golden_argmax(R, lo_c, hi_c)
                    end
                end
                k_con[i, j] = kp_c
                if ξ̄ <= zero(T)
                    ξT = zero(T)
                else
                    ξhat = (Eu - Ec) / max(p * w, T(1e-14))
                    ξT = clamp(ξhat, zero(T), ξ̄)
                end
                G = ξ̄ <= zero(T) ? one(T) : ξT / ξ̄
                adj_prob[i, j] = G
                flow = p * (profit[i, j] + (one(T) - δ) * k_grid[i])
                # E[ξ | adjust] G = ξT^2 / (2 ξ̄); ξ̄ = 0 ⇒ everyone adjusts at cost 0.
                adj_cost = (ξ̄ <= zero(T) || ξT <= zero(T)) ? zero(T) :
                    p * w * (ξT * ξT) / (T(2) * ξ̄)
                V_new[i, j] = flow + G * Eu + (one(T) - G) * Ec - adj_cost
            end
        end

        # Howard policy evaluation at the just-computed (k*, k^C, G).
        for _h in 1:howard_steps
            @inbounds for j in 1:n_e, i in 1:n_k
                ev = zero(T)
                for jp in 1:n_e
                    ev += Pi[j, jp] * V_new[i, jp]
                end
                EV[i, j] = ev
            end
            @inbounds for j in 1:n_e
                ev_j = view(EV, :, j)
                Eu = -γ * p * k_star[j] + β * _linear_interp(k_grid, ev_j, k_star[j])
                for i in 1:n_k
                    Ec = -γ * p * k_con[i, j] + β * _linear_interp(k_grid, ev_j, k_con[i, j])
                    G = adj_prob[i, j]
                    ξT = G * ξ̄
                    adj_cost = (ξ̄ <= zero(T) || ξT <= zero(T)) ? zero(T) :
                        p * w * (ξT * ξT) / (T(2) * ξ̄)
                    flow = p * (profit[i, j] + (one(T) - δ) * k_grid[i])
                    V_new[i, j] = flow + G * Eu + (one(T) - G) * Ec - adj_cost
                end
            end
        end

        max_diff = zero(T)
        @inbounds for idx in eachindex(V)
            d = abs(V_new[idx] - V[idx])
            if isfinite(d) && d > max_diff
                max_diff = d
            end
        end
        copyto!(V, V_new)
        if max_diff < tol
            converged = true
            break
        end
    end

    return (value=V, k_star=k_star, k_constrained=k_con, adj_prob=adj_prob,
            labor=n_pol, profit=profit, converged=converged)
end

# =============================================================================
# Stationary distribution and aggregation
# =============================================================================

"""
    _kt_transition(fs, k_star, k_con, adj_prob) → Matrix

Young (2010) lottery on the mixture of unconstrained and constrained capital
choices, times the idiosyncratic productivity chain.
"""
function _kt_transition(fs::FirmSystem{T}, k_star::AbstractVector{T},
                        k_con::AbstractMatrix{T},
                        adj_prob::AbstractMatrix{T}) where {T<:AbstractFloat}
    k_grid = fs.k_grid
    n_k = length(k_grid)
    n_e = length(fs.productivity.states)
    N = n_k * n_e
    Pi = fs.productivity.transition
    Λ = zeros(T, N, N)
    @inbounds for j in 1:n_e, i in 1:n_k
        col = (j - 1) * n_k + i
        G = adj_prob[i, j]
        for (kp, wt) in ((k_star[j], G), (k_con[i, j], one(T) - G))
            wt <= T(1e-16) && continue
            kb, w_lo, w_hi = _young_bracket(k_grid, clamp(kp, k_grid[1], k_grid[end]))
            for jp in 1:n_e
                p_e = Pi[j, jp]
                p_e <= T(1e-16) && continue
                if w_lo > T(1e-16)
                    Λ[(jp - 1) * n_k + kb, col] += wt * w_lo * p_e
                end
                if w_hi > T(1e-16)
                    Λ[(jp - 1) * n_k + kb + 1, col] += wt * w_hi * p_e
                end
            end
        end
    end
    return Λ
end

function _kt_stationary(Λ::AbstractMatrix{T}; max_iter::Int=20_000,
                        tol::T=T(1e-12)) where {T<:AbstractFloat}
    N = size(Λ, 1)
    μ = fill(one(T) / T(N), N)
    for _ in 1:max_iter
        μ_new = Λ * μ
        s = sum(μ_new)
        s > zero(T) && (μ_new ./= s)
        if maximum(abs.(μ_new .- μ)) < tol
            return μ_new
        end
        μ = μ_new
    end
    μ ./= sum(μ)
    return μ
end

"""
Khan–Thomas Table 2 definition: a plant is *inactive* when `|i/k| < 0.01`.
"""
const _KT_INACTION_CUTOFF = 0.01

function _kt_aggregates(fs::FirmSystem{T}, μ::AbstractVector{T},
                        pol, w::T, z::T) where {T<:AbstractFloat}
    k_grid = fs.k_grid
    n_k = length(k_grid)
    n_e = length(fs.productivity.states)
    e_vals = fs.productivity.states
    D = reshape(μ, n_k, n_e)
    K = zero(T)
    N_prod = zero(T)
    N_adj = zero(T)
    Y = zero(T)
    I = zero(T)
    inact = zero(T)
    # Table 2: |i/k| < 0.01. On a coarse grid the constrained policy sits on
    # the exemption corners ±b (Table 1: b = 0.011), just outside that cutoff.
    # Count the (S,s) band |i/k| ≤ max(0.01, b) so inaction is the share of
    # plants inside the costless region.
    cut = max(T(_KT_INACTION_CUTOFF), fs.b)
    ξ̄ = fs.xi_bar
    @inbounds for j in 1:n_e, i in 1:n_k
        m = D[i, j]
        m <= zero(T) && continue
        k = k_grid[i]
        n, y, _ = _kt_static(z, e_vals[j], k, fs.alpha, fs.nu, w)
        G = pol.adj_prob[i, j]
        kp_u = pol.k_star[j]
        kp_c = pol.k_constrained[i, j]
        ir_u = _kt_invest_rate(k, kp_u, fs)
        ir_c = _kt_invest_rate(k, kp_c, fs)
        K += m * k
        N_prod += m * n
        # Expected adjustment labor = G · (ξ^T / 2) = ξT^2 / (2 ξ̄).
        ξT = G * ξ̄
        N_adj += m * (ξ̄ <= zero(T) ? zero(T) : (ξT * ξT) / (T(2) * ξ̄))
        Y += m * y
        I += m * (G * (fs.gamma * kp_u - (one(T) - fs.delta) * k) +
                  (one(T) - G) * (fs.gamma * kp_c - (one(T) - fs.delta) * k))
        inact += m * (G * T(abs(ir_u) <= cut) + (one(T) - G) * T(abs(ir_c) <= cut))
    end
    N = N_prod + N_adj
    C = Y - I
    return (K=K, N=N, N_prod=N_prod, N_adj=N_adj, Y=Y, I=I, C=C, inaction=inact)
end

# =============================================================================
# Steady state and MIT
# =============================================================================

"""
    KhanThomasSteadyState{T}

Stationary equilibrium of a [`FirmSystem`](@ref). Prices `(w, p)` clear the
representative household FOC `w = φ C` at `p = 1`. `method` records how
aggregate dynamics were (or will be) computed — `:mit` for the sequence-space
perfect-foresight IRF.
"""
struct KhanThomasSteadyState{T<:AbstractFloat}
    firm::FirmSystem{T}
    w::T
    p::T
    K::T
    N::T
    Y::T
    I::T
    C::T
    inaction::T
    distribution::Matrix{T}
    value::Matrix{T}
    k_star::Vector{T}
    k_constrained::Matrix{T}
    adj_prob::Matrix{T}
    labor::Matrix{T}
    converged::Bool
    iterations::Int
    method::Symbol
end

"""
    KhanThomasTransition{T}

Perfect-foresight MIT path of a Khan–Thomas economy. `method === :mit`.
"""
struct KhanThomasTransition{T<:AbstractFloat}
    Z::Vector{T}
    Y::Vector{T}
    I::Vector{T}
    K::Vector{T}
    N::Vector{T}
    C::Vector{T}
    w::Vector{T}
    ss::KhanThomasSteadyState{T}
    method::Symbol
    converged::Bool
end

"""
    khan_thomas_steady_state(fs::FirmSystem; kwargs...) → KhanThomasSteadyState
    khan_thomas_steady_state(spec::ModelSpec; kwargs...) → KhanThomasSteadyState

Compute the stationary GE of a Khan–Thomas plant population plus a
representative household. Does **not** call [`_hh`](@ref) or any
[`HouseholdSystem`](@ref) solver.

# Keywords
- `w0` — initial wage (default: frictionless GE wage)
- `p` — output price numeraire (default 1)
- `tol` — outer `|w − φ C|` tolerance
- `max_iter` — outer wage iterations
- `dampen` — relaxation on the wage update
- `hh_solver` — ignored (accepted so household kwargs cannot silently reroute)
"""
function _kt_eval_wage(fs::FirmSystem{T}, w::T, z::T, p::T;
                       vfi_tol::T=T(1e-6), vfi_max_iter::Int=200,
                       howard_steps::Int=15,
                       init_value=nothing) where {T<:AbstractFloat}
    pol = _kt_plant_vfi(fs, w, z; p=p, max_iter=vfi_max_iter, tol=vfi_tol,
                        howard_steps=howard_steps, init_value=init_value)
    Λ = _kt_transition(fs, pol.k_star, pol.k_constrained, pol.adj_prob)
    μ = _kt_stationary(Λ)
    agg = _kt_aggregates(fs, μ, pol, w, z)
    C = agg.C
    resid = isfinite(C) && C > zero(T) ? w - fs.phi * C : w
    return pol, μ, agg, resid
end

function khan_thomas_steady_state(fs::FirmSystem{T};
                                  w0::Union{Nothing,Real}=nothing,
                                  p::Real=1,
                                  z::Union{Nothing,Real}=nothing,
                                  tol::Real=1e-5,
                                  max_iter::Int=16,
                                  dampen::Real=0.5,
                                  vfi_tol::Real=1e-6,
                                  vfi_max_iter::Int=200,
                                  howard_steps::Int=15,
                                  hh_solver=nothing,
                                  kwargs...) where {T<:AbstractFloat}
    hh_solver === nothing || throw(ArgumentError(
        "khan_thomas_steady_state: household solvers do not apply to FirmSystem " *
        "(got hh_solver=:$hh_solver)"))
    z_ss = z === nothing ? fs.Z : T(z)
    p_ss = T(p)
    # Prices from the nested frictionless economy (Khan–Thomas 2008 Appendix B /
    # Table 1): φ and (α, ν, δ, β, γ) are chosen so a representative household
    # plus frictionless plants clear w = φ C. Lumpy plants are then solved at
    # those prices. Coarse-grid k* is too elastic for a reliable lumpy wage
    # fixed point; aggregates stay close (paper Table 4).
    _ = (tol, max_iter, dampen)
    w = w0 === nothing ? first(_kt_frictionless_wage(fs; z=z_ss)) : T(w0)
    w = max(w, T(1e-6))
    pol, μ, agg, _ = _kt_eval_wage(fs, w, z_ss, p_ss;
                                   vfi_tol=T(vfi_tol), vfi_max_iter=vfi_max_iter,
                                   howard_steps=howard_steps)
    n_k = length(fs.k_grid)
    n_e = length(fs.productivity.states)
    D = reshape(copy(μ), n_k, n_e)
    return KhanThomasSteadyState{T}(
        fs, w, p_ss, agg.K, agg.N, agg.Y, agg.I, agg.C, agg.inaction,
        D, pol.value, pol.k_star, pol.k_constrained, pol.adj_prob, pol.labor,
        pol.converged, 1, :mit)
end

function khan_thomas_steady_state(spec::ModelSpec{T}; kwargs...) where {T<:AbstractFloat}
    has_kind(spec, FirmSystem) || throw(ArgumentError(
        "khan_thomas_steady_state: ModelSpec has no FirmSystem " *
        "(agents = $(keys(spec.agents)))"))
    has_kind(spec, HouseholdSystem) && throw(ArgumentError(
        "khan_thomas_steady_state: household methods do not apply to a FirmSystem spec"))
    fs = only(agents_of(spec, FirmSystem))
    return khan_thomas_steady_state(fs; kwargs...)
end

"""
    khan_thomas_mit(ss, Z_path; prices=:ss) → KhanThomasTransition

Perfect-foresight MIT transition (`method = :mit`). `K` is predetermined.
`prices = :ss` holds `(w, p)` at the stationary equilibrium (partial-equilibrium
MIT around the GE steady state). `prices = :ge` updates `w_t = φ C_t` once
along the path.
"""
function khan_thomas_mit(ss::KhanThomasSteadyState{T}, Z_path::AbstractVector;
                         prices::Symbol=:ss,
                         vfi_tol::Real=1e-6,
                         vfi_max_iter::Int=200,
                         howard_steps::Int=12) where {T<:AbstractFloat}
    prices in (:ss, :ge) || throw(ArgumentError(
        "khan_thomas_mit: prices must be :ss or :ge, got :$prices"))
    Z = collect(T, Z_path)
    length(Z) >= 2 || throw(ArgumentError("khan_thomas_mit: Z_path needs at least 2 points"))
    all(>(zero(T)), Z) || throw(ArgumentError("khan_thomas_mit: every TFP value must be positive"))
    fs = ss.firm
    H = length(Z)
    n_k = length(fs.k_grid)
    n_e = length(fs.productivity.states)
    p = ss.p

    # Backward plant values along the TFP path; V_{H+1} = V_ss.
    V_next = copy(ss.value)
    pols = Vector{NamedTuple}(undef, H)
    w_path = fill(ss.w, H)
    for t in H:-1:1
        pol = _kt_plant_vfi(fs, w_path[t], Z[t]; p=p, max_iter=vfi_max_iter,
                            tol=T(vfi_tol), howard_steps=howard_steps,
                            init_value=V_next)
        pols[t] = pol
        V_next = pol.value
    end

    μ = vec(ss.distribution)
    Y = zeros(T, H)
    I = zeros(T, H)
    K = zeros(T, H)
    N = zeros(T, H)
    C = zeros(T, H)
    ok = true
    for t in 1:H
        agg = _kt_aggregates(fs, μ, pols[t], w_path[t], Z[t])
        Y[t] = agg.Y
        I[t] = agg.I
        K[t] = agg.K
        N[t] = agg.N
        C[t] = agg.C
        ok = ok && all(isfinite, (agg.Y, agg.I, agg.K, agg.N, agg.C))
        if prices === :ge && isfinite(agg.C) && agg.C > zero(T)
            w_path[t] = fs.phi * agg.C
        end
        if t < H
            Λ = _kt_transition(fs, pols[t].k_star, pols[t].k_constrained,
                               pols[t].adj_prob)
            μ = Λ * μ
            s = sum(μ)
            s > zero(T) && (μ ./= s)
        end
    end
    return KhanThomasTransition{T}(Z, Y, I, K, N, C, w_path, ss, :mit, ok)
end

"""
    irf(ss::KhanThomasSteadyState, horizon; shock_size=0.01, persist=fs.rho_z)

MIT impulse response of `Y, I, K, N, C, Z` to aggregate TFP (`method = :mit`).
Deviations from the Khan–Thomas stationary equilibrium.
"""
function irf(ss::KhanThomasSteadyState{T}, horizon::Int;
             shock_size::Real=T(0.01), persist::Union{Nothing,Real}=nothing,
             prices::Symbol=:ss, kwargs...) where {T<:AbstractFloat}
    horizon >= 2 || throw(ArgumentError("irf(::KhanThomasSteadyState): horizon must be ≥ 2"))
    ρ = persist === nothing ? ss.firm.rho_z : T(persist)
    Zbar = ss.firm.Z
    Z = [Zbar * (one(T) + T(shock_size) * ρ^(n - 1)) for n in 1:horizon]
    tr = khan_thomas_mit(ss, Z; prices=prices, kwargs...)
    vals = hcat(tr.Y .- ss.Y, tr.I .- ss.I, tr.K .- ss.K,
                tr.N .- ss.N, tr.C .- ss.C, tr.Z .- Zbar)
    H, n = size(vals)
    point = reshape(Matrix{T}(vals), H, n, 1)
    return ImpulseResponse{T}(point, zeros(T, H, n, 1), zeros(T, H, n, 1),
                              H, ["Y", "I", "K", "N", "C", "Z"], ["Z"], :none)
end

# =============================================================================
# ModelSpec adapter
# =============================================================================

"""
    to_spec(fs::FirmSystem; agent_name=:firms) → ModelSpec

Wrap a [`FirmSystem`](@ref) as a [`ModelSpec`](@ref). The residual block is the
representative-household / goods / TFP identity (Cobb–Douglas stand-in for the
aggregates). Plant-level `K, N, Y` come from [`khan_thomas_steady_state`](@ref),
not from [`_hh`](@ref).
"""
function to_spec(fs::FirmSystem{T}; agent_name::Symbol=:firms) where {T<:AbstractFloat}
    endog = [:Y, :K, :I, :C, :w, :Z]
    exog = [:eps_Z]
    params = [:alpha, :nu, :delta, :beta, :gamma, :phi, :rho_z, :sigma_z, :N]
    param_values = Dict{Symbol,T}(
        :alpha => fs.alpha, :nu => fs.nu, :delta => fs.delta,
        :beta => fs.beta, :gamma => fs.gamma, :phi => fs.phi,
        :rho_z => fs.rho_z, :sigma_z => fs.sigma_z, :N => T(1) / T(3),
    )

    # Residual closures close over indices, not the FirmSystem payload.
    iY, iK, iI, iC, iw, iZ = 1, 2, 3, 4, 5, 6
    f_goods = (yt, yl, yle, shock, θ) -> yt[iY] - yt[iC] - yt[iI]
    f_hh = (yt, yl, yle, shock, θ) -> yt[iw] - θ[:phi] * yt[iC]
    f_inv = (yt, yl, yle, shock, θ) -> yt[iI] -
        (θ[:gamma] * yt[iK] - (one(T) - θ[:delta]) * yl[iK])
    f_k = (yt, yl, yle, shock, θ) -> yt[iK] - yl[iK]
    f_cd = (yt, yl, yle, shock, θ) -> yt[iY] -
        yt[iZ] * yl[iK]^θ[:alpha] * θ[:N]^θ[:nu]
    f_z = (yt, yl, yle, shock, θ) -> yt[iZ] - (
        (one(T) - θ[:rho_z]) * one(T) + θ[:rho_z] * yl[iZ] +
        θ[:sigma_z] * (isempty(shock) ? zero(T) : shock[1]))
    residual_fns = Function[f_goods, f_hh, f_inv, f_k, f_cd, f_z]
    equations = NamedEquation[
        NamedEquation(:goods, :C, :(Y[t] - C[t] - I[t]), f_goods),
        NamedEquation(:household, :w, :(w[t] - phi * C[t]), f_hh),
        NamedEquation(:investment, :I,
            :(I[t] - (gamma * K[t] - (1 - delta) * K[t-1])), f_inv),
        NamedEquation(:capital, :K, :(K[t] - K[t-1]), f_k),
        NamedEquation(:production, :Y,
            :(Y[t] - Z[t] * K[t-1]^alpha * N^nu), f_cd),
        NamedEquation(:z_proc, :Z,
            :(Z[t] - ((1 - rho_z) + rho_z * Z[t-1] + sigma_z * eps_Z[t])), f_z),
    ]
    return ModelSpec{T}(
        endog, exog, params, param_values, equations, residual_fns,
        0, Int[], T[];
        max_lag=1, max_lead=0,
        agents=NamedTuple{(agent_name,)}((fs,)),
    )
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, fs::FirmSystem{T}) where {T}
    print(io, "FirmSystem{$T}(n_k=", length(fs.k_grid),
          ", n_ε=", length(fs.productivity.states),
          ", ξ̄=", fs.xi_bar, ", b=", fs.b, ")")
end

function Base.show(io::IO, ss::KhanThomasSteadyState{T}) where {T}
    print(io, "KhanThomasSteadyState{$T}(K=", round(ss.K; digits=4),
          ", Y=", round(ss.Y; digits=4),
          ", I/K=", round(ss.I / max(ss.K, eps(T)); digits=4),
          ", inaction=", round(ss.inaction; digits=4),
          ", method=:", ss.method, ")")
end

function Base.show(io::IO, tr::KhanThomasTransition{T}) where {T}
    print(io, "KhanThomasTransition{$T}: ", length(tr.Z), " periods, method=:",
          tr.method, ", Y: ", round(tr.Y[1]; digits=4), " → ",
          round(tr.Y[end]; digits=4), ", converged=", tr.converged)
end

function report(io::IO, ss::KhanThomasSteadyState{T}) where {T}
    println(io, "Khan–Thomas (2008) Plant Economy — Steady State")
    println(io, "  method               :mit")
    println(io, "  Capital K            ", ss.K)
    println(io, "  Output Y             ", ss.Y)
    println(io, "  Investment I         ", ss.I)
    println(io, "  I/K                  ", ss.I / max(ss.K, eps(T)))
    println(io, "  Inaction             ", ss.inaction)
    println(io, "  Labor N              ", ss.N)
    println(io, "  Consumption C        ", ss.C)
    println(io, "  Wage w               ", ss.w)
    println(io, "  Converged            ", ss.converged)
    return nothing
end
report(ss::KhanThomasSteadyState) = report(stdout, ss)
