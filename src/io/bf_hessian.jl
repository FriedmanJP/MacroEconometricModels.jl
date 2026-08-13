# bf_hessian.jl — generalized B&F second-order Hessian + incidence + shock curve
#
# Orientation: ROW (B&F). All objects evaluated at the base ProductionNetwork.
# Ψ columns via sparse solves of (I − Ω̃); never form dense Ψ for the full N.
# See Baqaee & Farhi (2019) §4 and docs/plans IO2-B2.

# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

"""
    BFElasticities{T<:AbstractFloat}

First-order distributional elasticities of a [`ProductionNetwork`](@ref) at the
base point (Baqaee–Farhi 2019 §4).

# Fields (all **row orientation**)
- `dlogw_dlogA` — `F×n` factor-price incidence ``∂ log w_f / ∂ log A_j``
- `dlogp_dlogA` — `n×n` real-sector (outer-node) price incidence
- `dlambda_dlogA` — `n×n` sales-share (Domar) reallocation; equals the Hessian of
  ``log Y`` in ``log A`` when that Hessian is available
- `factor_names` — length-`F` labels
- `sectors` — length-`n` real-sector labels
"""
struct BFElasticities{T<:AbstractFloat}
    dlogw_dlogA::Matrix{T}
    dlogp_dlogA::Matrix{T}
    dlambda_dlogA::Matrix{T}
    factor_names::Vector{String}
    sectors::Vector{String}
end

"""
    BFLocal{T<:AbstractFloat}

Generalized local (Hulten + second-order) approximation for a
[`ProductionNetwork`](@ref). Prefer this over the legacy scalar-θ
[`BaqaeeFarhiResult`](@ref) when elasticities are heterogeneous, multi-factor,
or nested.

# Fields
- `first_order` — length-`n` Domar weights on real-sector outer nodes (Hulten)
- `second_order` — `n×n` Hessian ``H_{jk} = ∂² log Y / (∂ log A_j ∂ log A_k)``;
  empty `0×0` when `hessian=:none`
- `lambda` — full base Domar vector length `1+M+F`
- `Lambda` — base factor income shares length `F`
- `elasticities` — [`BFElasticities`](@ref) or `nothing` when not requested
- `sectors` — real-sector labels
- `nests` — nesting scheme of the source network
"""
struct BFLocal{T<:AbstractFloat}
    first_order::Vector{T}
    second_order::Matrix{T}
    lambda::Vector{T}
    Lambda::Vector{T}
    elasticities::Union{BFElasticities{T},Nothing}
    sectors::Vector{String}
    nests::Symbol
end

"""
    BFShockCurve{T<:AbstractFloat}

One-sector productivity shock sweep comparing the exact nonlinear equilibrium
against Hulten's first-order line and the local second-order Taylor curve.

# Fields
- `shocks` — grid of ``Δ log A`` for the shocked sector
- `exact` — ``Δ log Y`` from [`bf_equilibrium`](@ref)
- `hulten` — first-order ``λ_i · Δ log A``
- `second_order` — Hulten + ``½ H_{ii} (Δ log A)²``
- `sector` — shocked sector label
- `sector_index` — 1-based real-sector index
"""
struct BFShockCurve{T<:AbstractFloat}
    shocks::Vector{T}
    exact::Vector{T}
    hulten::Vector{T}
    second_order::Vector{T}
    sector::String
    sector_index::Int
end

# ═══════════════════════════════════════════════════════════════════════════
# Analytic Hessian assembly (B&F 2019 §4)
# ═══════════════════════════════════════════════════════════════════════════

"""
Build the covariance-weight matrix
``K = Σ_{i=1}^{M+1} (θ_i − 1) λ̃_i (diag(ω^{(i)}) − ω^{(i)}(ω^{(i)})')``
over household + producer nodes. Dense `N×N` (N = 1+M+F); used only for n ≤ 500
full-Hessian paths and for `bf_quadratic`.
"""
function _bf_assemble_K(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    M, F = net.M, net.F
    N = 1 + M + F
    Ω = net.Omega
    θ = net.theta
    λ = net.lambda
    K = zeros(T, N, N)
    @inbounds for i in 1:(M + 1)
        coef = (θ[i] - one(T)) * λ[i]
        abs(coef) <= eps(T) && continue
        # Row i of Ω̃ (cost shares of node i); factors have zero rows and are skipped.
        ω = Vector{T}(undef, N)
        for j in 1:N
            ω[j] = Ω[i, j]
        end
        # K += coef * (diag(ω) − ω ω')
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
Sparse-solve selected columns of ``Ψ̃ = (I − Ω̃)^{-1}``.
`indices` are 1-based global node indices; returns `N × length(indices)`.
"""
function _bf_psi_columns(net::ProductionNetwork{T}, indices::AbstractVector{Int}) where {T}
    N = 1 + net.M + net.F
    IΩ = sparse(T(1) * I, N, N) - net.Omega
    # UMFPACK / SparseArrays direct solve; factorize once.
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
Solve ``[diag(Λ̃) + Γ] dlogw = X dlogA`` for the `F×n` incidence matrix.

Single factor ⇒ `dlogw = 0` (E = 1 pins `w = 1/L` at fixed L). Multi-factor:
ridge-stabilized least squares with soft numéraire ``Λ̃' dlogw = 0``.
"""
function _bf_solve_dlogw(Γ::Matrix{T}, X::Matrix{T}, Λ::Vector{T}) where {T}
    F, n = size(X)
    F == size(Γ, 1) || throw(ArgumentError("Γ/X factor dimension mismatch"))
    length(Λ) == F || throw(ArgumentError("Λ length must equal F"))
    if F == 0
        return zeros(T, 0, n)
    end
    if F == 1
        return zeros(T, 1, n)
    end
    Sys = Diagonal(Λ) + Γ
    # Soft numéraire + ridge (Walras rank F−1)
    ε = T(1e-12)
    A = Sys' * Sys + ε * I
    # Also penalize Λ' dw
    A = A + ε * (Λ * Λ')
    B = Sys' * X
    return Matrix{T}(robust_inv(Hermitian(A))) * B
end

"""
    _bf_local_system(net) -> NamedTuple

Core second-order objects at the base network (row orientation):

- `H` — `n×n` Hessian of `log Y` in real-sector `log A` (outer nodes)
- `dlogw_dlogA` — `F×n`
- `dlogp_dlogA` — `n×n` outer-node prices
- `Ψ_P`, `Ψ_F` — selected columns of `Ψ̃`
- `base`, `Γ`, `X` — Cov-blocks
- `asym` — max|H − H'| before symmetrization
"""
function _bf_local_system(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    n, M, F = net.n, net.M, net.F
    N = 1 + M + F
    outer = net.outer_nodes
    fac_idx = collect(M + 2:N)
    Λ = net.lambda[fac_idx]

    K = _bf_assemble_K(net)
    Ψ_P = _bf_psi_columns(net, outer)          # N × n
    Ψ_F = F > 0 ? _bf_psi_columns(net, fac_idx) : zeros(T, N, 0)

    KP = K * Ψ_P
    base = transpose(Ψ_P) * KP                 # n × n
    if F > 0
        KF = K * Ψ_F
        Γ = transpose(Ψ_F) * KF                # F × F
        X = transpose(Ψ_F) * KP                # F × n
    else
        Γ = zeros(T, 0, 0)
        X = zeros(T, 0, n)
    end

    dlogw = _bf_solve_dlogw(Γ, X, Vector{T}(Λ))
    H_raw = base - transpose(X) * dlogw

    asym = maximum(abs.(H_raw .- transpose(H_raw)); init=zero(T))
    if asym > T(1e-8)
        @warn "bf Hessian asymmetry exceeds 1e-8; possible transcription bug" asymmetry = Float64(asym)
    end
    H = T(0.5) .* (H_raw .+ transpose(H_raw))

    # Price incidence: dlog p = −Ψ dlogA + Ψ_F dlog w  (rows = outer nodes)
    # Ψ_P[outer, :] is the producer-on-producer block of Ψ at outer indices.
    Ψ_PP = Ψ_P[outer, :]                       # n × n
    if F > 0
        Ψ_PF = Ψ_F[outer, :]                   # n × F
        dlogp = -Ψ_PP + Ψ_PF * dlogw
    else
        dlogp = -Ψ_PP
    end

    return (H=H, dlogw_dlogA=dlogw, dlogp_dlogA=dlogp,
            dlambda_dlogA=H, Ψ_P=Ψ_P, Ψ_F=Ψ_F,
            base=base, Γ=Γ, X=X, K=K, Lambda=Vector{T}(Λ), asym=asym)
end

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

"""
    baqaee_farhi(net::ProductionNetwork; hessian=:auto, elasticities=true) -> BFLocal

Generalized Baqaee–Farhi (2019) local approximation on a standard-form
[`ProductionNetwork`](@ref) (row orientation).

First order is Hulten's theorem on real-sector outer nodes. Second order is the
full multi-factor Hessian with heterogeneous elasticities and endogenous factor
prices (B&F §4). Factors sit inside the CES aggregator, so this Hessian
**differs** from the legacy intermediate-only `baqaee_farhi(io)` formula;
first-order Domar weights match. Multi-factor systems correct via
``[diag(Λ̃)+Γ] dlog w = X dlog A``. The formula is the efficient-economy
(2019) expansion: if `net.mu` is not identically 1 a warning is emitted.

# Keyword arguments
- `hessian` — `:full` form the `n×n` matrix; `:none` skip it; `:auto` (default)
  uses `:full` when `n ≤ 500` and `:none` otherwise.
- `elasticities` — if `true` (default), attach a [`BFElasticities`](@ref) block
  (`dlog w/dlogA`, `dlog p/dlogA`, `dλ/dlogA`).

For large shocks use [`bf_equilibrium`](@ref). For large `n` without forming `H`
use [`bf_quadratic`](@ref).

See also [`BaqaeeFarhiResult`](@ref) for the legacy scalar-θ `IOData` API.
"""
function baqaee_farhi(net::ProductionNetwork{T};
                      hessian::Symbol=:auto,
                      elasticities::Bool=true) where {T<:AbstractFloat}
    hessian in (:full, :none, :auto) || throw(ArgumentError(
        "hessian must be :full, :none, or :auto; got $hessian"))

    n = net.n
    do_H = if hessian === :auto
        n <= 500
    else
        hessian === :full
    end
    if hessian === :full && n > 500
        @warn "baqaee_farhi: forming full Hessian for n=$n > 500; consider hessian=:none and bf_quadratic"
    end
    if any(m -> m > one(T) + T(1e-14), net.mu)
        @warn "baqaee_farhi: network has μ ≠ 1; the local Hessian is the efficient-economy (2019) formula on cost shares. Use bf_equilibrium / bf_wedge_decomp for wedges."
    end

    λ_out = T[net.lambda[g] for g in net.outer_nodes]
    Λ = net.lambda[net.M + 2:net.M + 1 + net.F]
    sectors = String.(net.io.sectors)
    factor_names = String.(net.node_names[net.M + 2:net.M + 1 + net.F])

    if !do_H && !elasticities
        return BFLocal{T}(λ_out, zeros(T, 0, 0), copy(net.lambda), Vector{T}(Λ),
                          nothing, sectors, net.nests)
    end

    sys = _bf_local_system(net)
    H = do_H ? sys.H : zeros(T, 0, 0)

    elast = if elasticities
        BFElasticities{T}(sys.dlogw_dlogA, sys.dlogp_dlogA, sys.dlambda_dlogA,
                          factor_names, sectors)
    else
        nothing
    end

    BFLocal{T}(λ_out, H, copy(net.lambda), Vector{T}(Λ), elast, sectors, net.nests)
end

"""
    bf_quadratic(net::ProductionNetwork, v::AbstractVector) -> Real

Quadratic form ``v' H v`` of the B&F Hessian without forming the dense `n×n`
matrix. Uses the Cov-block identity

```
v' H v = (Ψ_P v)' K (Ψ_P v) − (X v)' (dlog w)
```

with ``[diag(Λ̃)+Γ] dlog w = X v``. Orientation: row (B&F). `v` is length `n`
(real-sector productivity directions).
"""
function bf_quadratic(net::ProductionNetwork{T}, v::AbstractVector) where {T<:AbstractFloat}
    n = net.n
    vv = _bf_as_vector(T, v, n, "v")
    M, F = net.M, net.F
    N = 1 + M + F
    outer = net.outer_nodes
    fac_idx = collect(M + 2:N)
    Λ = Vector{T}(net.lambda[fac_idx])

    K = _bf_assemble_K(net)
    # Ψ_P v via one solve: (I−Ω) z = e_outer · v  (linear combo of columns)
    IΩ = sparse(T(1) * I, N, N) - net.Omega
    rhs = zeros(T, N)
    @inbounds for (k, g) in enumerate(outer)
        rhs[g] += vv[k]
    end
    z = IΩ \ rhs                                # z = Ψ_P v
    Kz = K * z
    base_q = dot(z, Kz)

    if F == 0 || F == 1
        return base_q
    end
    Ψ_F = _bf_psi_columns(net, fac_idx)
    # X v = Ψ_F' K Ψ_P v = Ψ_F' K z
    Xv = transpose(Ψ_F) * Kz
    # Γ = Ψ_F' K Ψ_F
    KF = K * Ψ_F
    Γ = transpose(Ψ_F) * KF
    dw = vec(_bf_solve_dlogw(Γ, reshape(Xv, F, 1), Λ))
    return base_q - dot(Xv, dw)
end

"""
    bf_elasticities(net::ProductionNetwork) -> BFElasticities

Factor-price, goods-price, and Domar-share incidence matrices at the base
point of `net` (row orientation). Equivalent to
`baqaee_farhi(net; hessian=:none, elasticities=true).elasticities` but always
computes the local system.
"""
function bf_elasticities(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    sys = _bf_local_system(net)
    factor_names = String.(net.node_names[net.M + 2:net.M + 1 + net.F])
    sectors = String.(net.io.sectors)
    BFElasticities{T}(sys.dlogw_dlogA, sys.dlogp_dlogA, sys.dlambda_dlogA,
                      factor_names, sectors)
end

"""
    bf_shock_curve(net, sector; range=(-0.5, 0.5), points=41, kwargs...) -> BFShockCurve

Sweep a single real-sector productivity shock over a grid and compare

1. exact ``Δ log Y`` from [`bf_equilibrium`](@ref),
2. Hulten first-order ``λ_i · Δ log A``,
3. second-order Taylor ``λ_i · Δ log A + ½ H_{ii} (Δ log A)²``.

`sector` is a 1-based index or a sector name matching `net.io.sectors`.
`range` is a `(lo, hi)` tuple for ``Δ log A`` (keyword name matches the plan API;
does not shadow `Base.range` inside the body).
Additional `kwargs` are forwarded to `bf_equilibrium` (e.g. `tol`, `method`).
"""
function bf_shock_curve(net::ProductionNetwork{T}, sector;
                        range::Tuple{<:Real,<:Real}=(-0.5, 0.5),
                        points::Int=41,
                        kwargs...) where {T<:AbstractFloat}
    points >= 2 || throw(ArgumentError("points must be ≥ 2"))
    lo, hi = T(range[1]), T(range[2])
    lo < hi || throw(ArgumentError("range must be increasing"))

    idx = _bf_sector_index(net, sector)
    label = String(net.io.sectors[idx])
    λ_i = net.lambda[net.outer_nodes[idx]]

    # Diagonal Hessian entry (only need H_ii)
    sys = _bf_local_system(net)
    H_ii = sys.H[idx, idx]

    shocks = collect(Base.range(lo, hi; length=points))
    exact = Vector{T}(undef, points)
    hult = Vector{T}(undef, points)
    so = Vector{T}(undef, points)
    dA = zeros(T, net.n)
    for (t, s) in enumerate(shocks)
        dA[idx] = s
        eq = bf_equilibrium(net; dlogA=dA, kwargs...)
        exact[t] = eq.dlogY
        hult[t] = λ_i * s
        so[t] = λ_i * s + T(0.5) * H_ii * s * s
        dA[idx] = zero(T)
    end
    BFShockCurve{T}(shocks, exact, hult, so, label, idx)
end

function _bf_sector_index(net::ProductionNetwork, sector::Integer)
    i = Int(sector)
    1 <= i <= net.n || throw(ArgumentError(
        "sector index $i out of range 1:$(net.n)"))
    return i
end

function _bf_sector_index(net::ProductionNetwork, sector::AbstractString)
    secs = net.io.sectors
    for (i, s) in enumerate(secs)
        String(s) == String(sector) && return i
    end
    throw(ArgumentError("sector $(repr(sector)) not found in network sectors"))
end

function _bf_sector_index(net::ProductionNetwork, sector::Symbol)
    _bf_sector_index(net, String(sector))
end
