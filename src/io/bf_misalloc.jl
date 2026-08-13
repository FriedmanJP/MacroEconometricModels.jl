# bf_misalloc.jl — Baqaee–Farhi (2020) Proposition 5 Harberger triangle / H_μ
#
# Orientation: ROW (B&F). Hessian of log Y in log μ (same sign as BFLocal.second_order).
# Distance L = log(Y*/Y) ≈ −½ Δlogμ' H_μ Δlogμ at the efficient point.
# K_μ = Σ_j λ_j θ_j (diag(ω^j) − ω^j ω^j'); H_μ = −Ψ_P' K_μ Ψ_P − Ψ_P' K_μ Ψ_F D.

"""
    BFMisallocation{T<:AbstractFloat}

Baqaee–Farhi (2020) Proposition 5 Harberger-style misallocation distance and
the Hessian of log real output in log markups (`H_μ`).

# Fields
- `distance` — exact `L = log(Y*/Y)` from `bf_equilibrium`
- `first_order` — Theorem 1 piece `−λ̃'Δlogμ − Λ̃'ΔlogΛ` at `point` (0 at `:efficient`)
- `second_order` — `−½ Δlogμ' H Δlogμ`
- `H_mu` — `n×n` Hessian of log Y in log μ; `0×0` if `hessian=:none`
- `delta_logmu` — log μ of real-sector outers (0 if μ≡1)
- `point` — `:efficient` | `:observed`
- `lambda` — evaluation-point Domar on outers (length `n`)
- `mu` — sectoral markups (length `n`)
- `sectors` — real-sector labels

# Orientation
Row (B&F).
"""
struct BFMisallocation{T<:AbstractFloat}
    distance::T                 # exact L = log(Y*/Y) from bf_equilibrium
    first_order::T              # Theorem 1 piece −λ̃'Δlogμ − Λ̃'ΔlogΛ at `point` (0 at :efficient)
    second_order::T             # −½ Δlogμ' H Δlogμ
    H_mu::Matrix{T}             # n×n Hessian of log Y in log μ; 0×0 if hessian=:none
    delta_logmu::Vector{T}      # log μ of real-sector outers (0 if μ≡1)
    point::Symbol               # :efficient | :observed
    lambda::Vector{T}           # evaluation-point Domar on outers (length n)
    mu::Vector{T}               # sectoral markups (length n)
    sectors::Vector{String}
end

# ═══════════════════════════════════════════════════════════════════════════
# K_μ / efficient-point H_μ (Prop 5; multi-factor via Prop 3)
# ═══════════════════════════════════════════════════════════════════════════

"""
Build ``K_μ = Σ_{j=1}^{M+1} λ_j θ_j (diag(ω^j) − ω^j ω^j')``.

`λ_nodes` is length `N = 1+M+F`; only rows `1:M+1` are used. Not the 2019
productivity ``K``, which weights rows by ``(θ_j − 1) λ̃_j``.
"""
function _bf_assemble_K_mu(θ::AbstractVector{T}, λ_nodes::AbstractVector{T},
                           Ω::AbstractMatrix{T}, M::Int) where {T}
    # K_μ = Σ_{j=1}^{M+1} λ_j θ_j (diag(ω^j) − ω^j ω^j')
    # λ_nodes is length N = 1+M+F, but only rows 1:M+1 are used.
    N = size(Ω, 1)
    K = zeros(T, N, N)
    @inbounds for j in 1:(M + 1)
        coef = λ_nodes[j] * θ[j]
        abs(coef) <= eps(T) && continue
        ω = T[Ω[j, k] for k in 1:N]
        for a in 1:N
            wa = ω[a]
            wa == 0 && continue
            K[a, a] += coef * wa
            ca = coef * wa
            for b in 1:N
                K[a, b] -= ca * ω[b]
            end
        end
    end
    return K
end

function _bf_assemble_K_mu(net::ProductionNetwork{T},
                           λ::AbstractVector{T},
                           Ω::AbstractMatrix{T},
                           θ::AbstractVector{T}=net.theta) where {T<:AbstractFloat}
    _bf_assemble_K_mu(θ, λ, Ω, net.M)
end

"""
    _bf_dlogLambda_dlogmu(net, point=:efficient) -> Matrix{T}

Proposition 3 factor-share Jacobian ``D = ∂logΛ / ∂logμ`` (``F×n``) at
`point`. At `:efficient`, ``μ=1``, ``λ=λ̃``, ``Λ=Λ̃``, ``Ψ=Ψ̃`` and the
Cov-block matches `_bf_local_system` (2019 ``K`` with weights
``(θ_j-1)λ̃_j``), plus the extra ``-λ_k Ψ̃_{kf}/Λ_f`` impulse:

```
[diag(Λ̃) + Γ] D = −X − [λ_k Ψ̃_{kf}]_{f,k}
```

Single-factor: ``Λ≡1`` so ``D = 0``. `:observed` is reserved.
"""
function _bf_dlogLambda_dlogmu(net::ProductionNetwork{T},
                               point::Symbol=:efficient) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))
    point === :observed && error("_bf_dlogLambda_dlogmu: point=:observed not implemented")

    n, M, F = net.n, net.M, net.F
    N = 1 + M + F
    if F == 0
        return zeros(T, 0, n)
    end
    if F == 1
        return zeros(T, 1, n)
    end

    outer = net.outer_nodes
    fac_idx = collect(M + 2:N)
    Λ = Vector{T}(net.lambda[fac_idx])
    λ_out = T[net.lambda[g] for g in outer]

    # Prop 3 Cov-block: same K / Γ / X as 2019 local system
    K = _bf_assemble_K(net)
    Ψ_P = _bf_psi_columns(net, outer)
    Ψ_F = _bf_psi_columns(net, fac_idx)
    Γ = transpose(Ψ_F) * (K * Ψ_F)
    X = transpose(Ψ_F) * (K * Ψ_P)

    # Impulse[f,k] = λ_k Ψ̃_{k f}; Ψ_F[outer_k, f] = Ψ̃_{k f}
    impulse = transpose(Ψ_F[outer, :]) * Diagonal(λ_out)
    # [diag(Λ)+Γ] D = −X − impulse  (pattern of `_bf_solve_dlogw`, different RHS)
    return _bf_solve_dlogw(Γ, -X .- impulse, Λ)
end

"""
Efficient-point ``H_μ`` (Hessian of ``log Y`` in outer-node ``log μ``).
Leading term ``H_μ = −Ψ_P' K_μ Ψ_P``; multi-factor second line of
Proposition 5 / eq. (17) adds ``−Ψ_P' K_μ Ψ_F D`` (then symmetrize) so
``L ≈ −½ v' H_μ v``. Single-factor skips the ``D`` block.
"""
function _bf_H_mu_efficient(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    K = _bf_assemble_K_mu(net.theta, net.lambda, net.Omega, net.M)
    Ψ_P = _bf_psi_columns(net, net.outer_nodes)
    # log-Y Hessian = −Ψ'KΨ (K_μ is the Var operator; L ≈ ½ v'(Ψ'KΨ)v)
    H_raw = -(transpose(Ψ_P) * (K * Ψ_P))
    if net.F > 1
        D = _bf_dlogLambda_dlogmu(net, :efficient)
        fac_idx = collect(net.M + 2:1 + net.M + net.F)
        Ψ_F = _bf_psi_columns(net, fac_idx)
        # eq. (17) second line: L += ½ (Ψ_F D v)' K_μ (Ψ_P v)
        # L ≈ −½ v' H_μ v ⇒ H_μ += −Ψ_P' K_μ Ψ_F D
        H_raw = H_raw - (transpose(Ψ_P) * (K * Ψ_F) * D)
    end
    asym = maximum(abs.(H_raw .- transpose(H_raw)); init=zero(T))
    if asym > T(1e-8)
        @warn "bf H_μ asymmetry exceeds 1e-8; possible transcription bug" asymmetry = Float64(asym)
    end
    return T(0.5) .* (H_raw .+ transpose(H_raw))
end

"""Efficient twin: same ``Ω̃`` / elasticities, ``μ ≡ 1``."""
function _bf_efficient_twin(net::ProductionNetwork{T}) where {T}
    any(m -> m > one(T) + T(1e-14), net.mu) || return net
    μ1 = ones(T, net.n)
    return _bf_finish_network(net.io, net.Omega, net.theta, net.node_names,
                              net.parent, net.n, net.M, net.F, net.nests,
                              net.outer_nodes, μ1)
end

"""Sectoral (outer-node) markups, length `n`."""
function _bf_outer_mu(net::ProductionNetwork{T}) where {T}
    return T[net.mu[g - 1] for g in net.outer_nodes]
end

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

"""
    bf_misallocation(net; point=:efficient, hessian=:auto) -> BFMisallocation

Proposition 5 Harberger distance and markup Hessian for a
[`ProductionNetwork`](@ref).

At `point=:efficient` the first-order term vanishes (envelope) and
``L ≈ -½ Δlogμ' H_μ Δlogμ``. `H_μ` is the Hessian of `log Y` in `log μ`
(same sign as [`BFLocal`](@ref).`second_order`). Exact `distance` is
``log(Y*/Y)`` from [`bf_equilibrium`](@ref) on the efficient twin.

# Keyword arguments
- `point` — `:efficient` (default). `:observed` is reserved.
- `hessian` — `:full` form the `n×n` matrix; `:none` skip it; `:auto`
  (default) uses `:full` when `n ≤ 500` and `:none` otherwise.

# Orientation
Row (B&F).
"""
function bf_misallocation(net::ProductionNetwork{T}; point::Symbol=:efficient,
                          hessian::Symbol=:auto) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))
    hessian in (:full, :none, :auto) || throw(ArgumentError(
        "hessian must be :full, :none, or :auto; got $hessian"))
    point === :observed && error("bf_misallocation: point=:observed not implemented")

    n = net.n
    do_H = if hessian === :auto
        n <= 500
    else
        hessian === :full
    end
    if hessian === :full && n > 500
        @warn "bf_misallocation: forming full H_μ for n=$n > 500; consider hessian=:none"
    end

    μ_sec = _bf_outer_mu(net)
    v = log.(μ_sec)
    λ_out = T[net.lambda[g] for g in net.outer_nodes]
    sectors = String.(net.io.sectors)
    v_zero = maximum(abs, v; init=zero(T)) <= T(1e-14)

    H_mu = if do_H
        _bf_H_mu_efficient(net)
    else
        zeros(T, 0, 0)
    end

    second_order = if v_zero || !do_H
        zero(T)
    else
        -T(0.5) * dot(v, H_mu * v)
    end

    distance = if v_zero
        zero(T)
    else
        eq = bf_equilibrium(_bf_efficient_twin(net); dlogmu=v)
        -eq.dlogY
    end

    BFMisallocation{T}(distance, zero(T), second_order, H_mu, v, :efficient,
                       λ_out, μ_sec, sectors)
end
