# bf_misalloc.jl — Baqaee–Farhi (2020) Proposition 5 Harberger triangle / H_μ
#
# Orientation: ROW (B&F). At the efficient point, H_μ is the Hessian of log Y
# in log μ (same sign as BFLocal.second_order). Distance L = log(Y*/Y) ≈
# −½ Δlogμ' H_μ Δlogμ only at :efficient. :observed H_μ is local curvature.
# K_μ = Σ_j λ_j θ_j (diag(ω^j) − ω^j ω^j'); H_μ = −Ψ_P' K_μ Ψ_P − Ψ_P' K_μ Ψ_F D.

"""
    BFMisallocation{T<:AbstractFloat}

Baqaee–Farhi (2020) Proposition 5 Harberger-style misallocation distance and
the Hessian of log real output in log markups (`H_μ`).

# Fields
- `distance` — exact `L = log(Y*/Y)` from `bf_equilibrium` (independent of `point`)
- `first_order` — 0 at `:efficient`; at `:observed`, eq. 14 linear term
  `g'Δlogμ` (gradient of `log Y` in `log μ` at the observed point)
- `second_order` — `−½ Δlogμ' H Δlogμ` (Harberger of `L` only at `:efficient`)
- `H_mu` — `n×n` Hessian of log Y in log μ at `point`; `0×0` if `hessian=:none`
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
    first_order::T              # 0 at :efficient; eq. 14 g'Δlogμ at :observed
    second_order::T             # −½ Δlogμ' H Δlogμ (Harberger of L only at :efficient)
    H_mu::Matrix{T}             # n×n Hessian of log Y in log μ at `point`; 0×0 if hessian=:none
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
Evaluation-point Domar vector and IO matrix for ``K_μ``.

`:efficient` uses cost ``λ̃, Ω̃``. `:observed` uses revenue ``λ`` and
``Ω = diag(1/μ) Ω̃`` on producer rows.
"""
function _bf_eval_lambda_omega(net::ProductionNetwork{T},
                               point::Symbol) where {T<:AbstractFloat}
    if point === :efficient
        return net.lambda, net.Omega
    else
        return net.lambda_rev, _bf_revenue_omega(net.Omega, net.mu, net.M, net.F)
    end
end

"""Selected columns of ``(I−Ω)^{-1}`` (any evaluation-point IO matrix)."""
function _bf_leontief_columns(Ω::SparseMatrixCSC{T,Int},
                              indices::AbstractVector{Int}) where {T}
    N = size(Ω, 1)
    IΩ = sparse(T(1) * I, N, N) - Ω
    Fct = factorize(IΩ)
    nidx = length(indices)
    Ψ = Matrix{T}(undef, N, nidx)
    e = zeros(T, N)
    @inbounds for (k, j) in enumerate(indices)
        fill!(e, zero(T))
        e[j] = one(T)
        Ψ[:, k] = Fct \ e
    end
    return Ψ
end

"""
Prop 3 Cov-weight matrix at the distorted point:
``K = Σ_j (λ_j^{rev}/μ_j)(θ_j-1)(diag(ω̃^j) − ω̃^j ω̃^{j'})`` on cost ``Ω̃``.
Household row has ``μ=1``.
"""
function _bf_assemble_K_prop3_observed(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    M = net.M
    N = 1 + M + net.F
    Ω = net.Omega
    θ = net.theta
    λr = net.lambda_rev
    μ = net.mu
    K = zeros(T, N, N)
    @inbounds for j in 1:(M + 1)
        μj = j == 1 ? one(T) : μ[j - 1]
        coef = (λr[j] / μj) * (θ[j] - one(T))
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

"""
    _bf_dlogLambda_dlogmu(net, point=:efficient) -> Matrix{T}

Proposition 3 factor-share Jacobian ``D = ∂logΛ / ∂logμ`` (``F×n``) at
`point`. At `:efficient`, ``μ=1``, ``λ=λ̃``, ``Λ=Λ̃``, ``Ψ=Ψ̃`` and the
Cov-block matches `_bf_local_system` (2019 ``K`` with weights
``(θ_j-1)λ̃_j``), plus the extra ``-λ_k Ψ̃_{kf}/Λ_f`` impulse:

```
[diag(Λ̃) + Γ] D = −X − [λ_k Ψ̃_{kf}]_{f,k}
```

`:observed` uses revenue ``λ, Λ`` and weights ``(λ_j^{rev}/μ_j)(θ_j-1)``
on cost ``Ω̃``. Cov first argument is cost ``Ψ̃``; second argument is
revenue ``Ψ`` (eq. 12). Single-factor at `:efficient`: ``Λ≡1`` so
``D = 0``. At `:observed` the ``F=1`` row is
``D = (−X − impulse)/Λ``.
"""
function _bf_dlogLambda_dlogmu(net::ProductionNetwork{T},
                               point::Symbol=:efficient) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))

    n, M, F = net.n, net.M, net.F
    N = 1 + M + F
    if F == 0
        return zeros(T, 0, n)
    end
    if F == 1 && point === :efficient
        return zeros(T, 1, n)
    end

    outer = net.outer_nodes
    fac_idx = collect(M + 2:N)
    Ψ̃_P = _bf_psi_columns(net, outer)
    Ψ̃_F = _bf_psi_columns(net, fac_idx)
    if point === :efficient
        Λ = Vector{T}(net.lambda[fac_idx])
        λ_out = T[net.lambda[g] for g in outer]
        K = _bf_assemble_K(net)
        Ψ_F = Ψ̃_F
        impulse_Ψ = Ψ̃_F
    else
        Λ = Vector{T}(net.lambda_rev[fac_idx])
        λ_out = T[net.lambda_rev[g] for g in outer]
        K = _bf_assemble_K_prop3_observed(net)
        Ω_rev = _bf_revenue_omega(net.Omega, net.mu, M, F)
        Ψ_F = _bf_leontief_columns(Ω_rev, fac_idx)
        impulse_Ψ = Ψ̃_F                         # impulse uses cost Ψ̃_{kf}
    end
    Γ = transpose(Ψ_F) * (K * Ψ̃_F)
    X = transpose(Ψ_F) * (K * Ψ̃_P)

    # Impulse[f,k] = λ_k Ψ̃_{k f}
    impulse = transpose(impulse_Ψ[outer, :]) * Diagonal(λ_out)
    rhs = -X .- impulse
    # F=1 :observed: no relative factor prices; divide by revenue Λ
    if F == 1
        return rhs ./ Λ[1]
    end
    # [diag(Λ)+Γ] D = −X − impulse  (pattern of `_bf_solve_dlogw`, different RHS)
    return _bf_solve_dlogw(Γ, rhs, Λ)
end

"""
``H_μ`` (Hessian of ``log Y`` in outer-node ``log μ``) at `point`.
Leading term ``H_μ = −Ψ_P' K_μ Ψ_P`` with evaluation-point ``Ψ``;
multi-factor second line of Proposition 5 / eq. (17) adds
``−Ψ_P' K_μ Ψ_F D`` (then symmetrize). Single-factor skips the ``D``
block.

At `:efficient`, ``K_μ`` uses cost ``λ̃, Ω̃`` and ``L ≈ −½ v' H_μ v``.
At `:observed`, ``K_μ`` and ``Ψ`` use revenue ``λ`` and
``Ω=diag(1/μ)Ω̃``; this is local curvature, not the distance-to-frontier
identity.
"""
function _bf_H_mu(net::ProductionNetwork{T},
                  point::Symbol=:efficient) where {T<:AbstractFloat}
    λ_nodes, Ω_eval = _bf_eval_lambda_omega(net, point)
    K = _bf_assemble_K_mu(net.theta, λ_nodes, Ω_eval, net.M)
    Ψ_P = _bf_leontief_columns(Ω_eval, net.outer_nodes)
    # log-Y Hessian = −Ψ'KΨ (K_μ is the Var operator)
    H_raw = -(transpose(Ψ_P) * (K * Ψ_P))
    if net.F > 1
        D = _bf_dlogLambda_dlogmu(net, point)
        fac_idx = collect(net.M + 2:1 + net.M + net.F)
        Ψ_F = _bf_leontief_columns(Ω_eval, fac_idx)
        # eq. (17) second line: L += ½ (Ψ_F D v)' K_μ (Ψ_P v)
        # H_μ += −Ψ_P' K_μ Ψ_F D
        H_raw = H_raw - (transpose(Ψ_P) * (K * Ψ_F) * D)
    end
    asym = maximum(abs.(H_raw .- transpose(H_raw)); init=zero(T))
    # :observed mixes Prop 3 D with revenue K_μ; asymmetry is expected
    if point === :efficient && asym > T(1e-8)
        @warn "bf H_μ asymmetry exceeds 1e-8; possible transcription bug" asymmetry = Float64(asym)
    end
    return T(0.5) .* (H_raw .+ transpose(H_raw))
end

"""
Eq. 14 gradient of ``log Y`` in ``log μ`` at the observed point,
contracted with `v`. Zero at `:efficient`.

```
g = (Ψ_F^{rev} (Λ̃./Λ))' K_{14} (Ψ̃_P + Ψ̃_F D)
```

with ``K_{14}`` weighted by ``(λ_j^{rev}/μ_j)θ_j`` on cost ``Ω̃``.
"""
function _bf_first_order(net::ProductionNetwork{T}, v::AbstractVector{T},
                         point::Symbol) where {T<:AbstractFloat}
    point === :efficient && return zero(T)
    n, M, F = net.n, net.M, net.F
    N = 1 + M + F
    outer = net.outer_nodes
    fac_idx = collect(M + 2:N)
    D = _bf_dlogLambda_dlogmu(net, :observed)
    Ψ̃_P = _bf_psi_columns(net, outer)
    Ψ̃_F = F > 0 ? _bf_psi_columns(net, fac_idx) : zeros(T, N, 0)
    Ω_rev = _bf_revenue_omega(net.Omega, net.mu, M, F)
    Ψ_F = F > 0 ? _bf_leontief_columns(Ω_rev, fac_idx) : zeros(T, N, 0)
    # K_14 = Σ_j (λ_j^{rev}/μ_j) θ_j (diag(ω̃^j) − ω̃^j ω̃^{j'})
    K14 = zeros(T, N, N)
    Ω̃ = net.Omega
    θ = net.theta
    λr = net.lambda_rev
    μ = net.mu
    @inbounds for j in 1:(M + 1)
        μj = j == 1 ? one(T) : μ[j - 1]
        coef = (λr[j] / μj) * θ[j]
        abs(coef) <= eps(T) && continue
        ω = T[Ω̃[j, k] for k in 1:N]
        for a in 1:N
            wa = ω[a]
            wa == 0 && continue
            K14[a, a] += coef * wa
            ca = coef * wa
            for b in 1:N
                K14[a, b] -= ca * ω[b]
            end
        end
    end
    z = Ψ̃_P + (F > 0 ? Ψ̃_F * D : zeros(T, N, n))
    if F == 0
        return zero(T)
    end
    Λ̃ = net.lambda[fac_idx]
    Λ = net.lambda_rev[fac_idx]
    scale = Λ̃ ./ Λ
    g = vec(transpose(Ψ_F * scale) * (K14 * z))
    return dot(g, v)
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
    bf_wedge_quadratic(net, v; point=:efficient) -> Real

Quadratic form ``v' H_μ v`` of the markup Hessian without forming the
dense `n×n` matrix. One sparse solve for ``Ψ_P v`` (same trick as
[`bf_quadratic`](@ref)).

At `point=:efficient` this is the Harberger curvature
(``L ≈ −½ v' H_μ v``). At `point=:observed` it is local curvature of
``log Y`` at current ``μ``, not the distance-to-frontier identity.

`v` is length `n` (real-sector ``log μ`` directions). Orientation: row (B&F).
"""
function bf_wedge_quadratic(net::ProductionNetwork{T}, v::AbstractVector;
                            point::Symbol=:efficient) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))
    n = net.n
    vv = _bf_as_vector(T, v, n, "v")
    M, F = net.M, net.F
    N = 1 + M + F
    outer = net.outer_nodes

    λ_nodes, Ω_eval = _bf_eval_lambda_omega(net, point)
    K = _bf_assemble_K_mu(net.theta, λ_nodes, Ω_eval, M)

    # Ψ_P v via one solve at the evaluation-point Ω
    IΩ = sparse(T(1) * I, N, N) - Ω_eval
    rhs = zeros(T, N)
    @inbounds for (k, g) in enumerate(outer)
        rhs[g] += vv[k]
    end
    z = IΩ \ rhs                                # z = Ψ_P v
    q = -dot(z, K * z)
    if F > 1
        D = _bf_dlogLambda_dlogmu(net, point)
        fac_idx = collect(M + 2:N)
        Ψ_F = _bf_leontief_columns(Ω_eval, fac_idx)
        q -= dot(z, K * (Ψ_F * (D * vv)))
    end
    return q
end

"""
    bf_misallocation(net; point=:efficient, hessian=:auto) -> BFMisallocation

Proposition 5 Harberger distance and markup Hessian for a
[`ProductionNetwork`](@ref).

At `point=:efficient` the first-order term vanishes (envelope) and
``L ≈ -½ Δlogμ' H_μ Δlogμ``. `H_μ` is the Hessian of `log Y` in `log μ`
(same sign as [`BFLocal`](@ref).`second_order`). Exact `distance` is
``log(Y*/Y)`` from [`bf_equilibrium`](@ref) on the efficient twin and
does **not** depend on `point`.

At `point=:observed` the objects in ``K_μ`` are revenue Domars and
revenue ``Ω``, and Proposition 3 is evaluated at the distorted point.
This is local curvature at current ``μ``. The Taylor of ``L`` for a
*further* shock ``δ`` is ``L ≈ L_0 - g'δ - \\tfrac12 δ'H δ``; do not
read ``L ≈ -½ v'H v`` at `:observed`. `first_order` is the eq. 14
linear term ``g'Δlogμ``.

# Keyword arguments
- `point` — `:efficient` (default) or `:observed`.
- `hessian` — `:full` form the `n×n` matrix; `:none` skip it (quadratic
  form still via [`bf_wedge_quadratic`](@ref)); `:auto` (default) uses
  `:full` when `n ≤ 500` and `:none` otherwise.

# Orientation
Row (B&F).
"""
function bf_misallocation(net::ProductionNetwork{T}; point::Symbol=:efficient,
                          hessian::Symbol=:auto) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))
    hessian in (:full, :none, :auto) || throw(ArgumentError(
        "hessian must be :full, :none, or :auto; got $hessian"))

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
    λ_nodes, _ = _bf_eval_lambda_omega(net, point)
    λ_out = T[λ_nodes[g] for g in net.outer_nodes]
    sectors = String.(net.io.sectors)
    v_zero = maximum(abs, v; init=zero(T)) <= T(1e-14)

    H_mu = if do_H
        _bf_H_mu(net, point)
    else
        zeros(T, 0, 0)
    end

    second_order = if v_zero
        zero(T)
    elseif do_H
        -T(0.5) * dot(v, H_mu * v)
    else
        -T(0.5) * bf_wedge_quadratic(net, v; point=point)
    end

    first_order = v_zero ? zero(T) : _bf_first_order(net, v, point)

    distance = if v_zero
        zero(T)
    else
        eq = bf_equilibrium(_bf_efficient_twin(net); dlogmu=v)
        -eq.dlogY
    end

    BFMisallocation{T}(distance, first_order, second_order, H_mu, v, point,
                       λ_out, μ_sec, sectors)
end
