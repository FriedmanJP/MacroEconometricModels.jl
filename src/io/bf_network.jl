# bf_network.jl — Baqaee–Farhi standard-form ProductionNetwork calibration
#
# Orientation: ROW (B&F convention). Ω[i,j] = expenditure share of buyer i on
# input j. Converted ONCE from column-oriented IOData (A[i,j] = Z[i,j]/x[j]).

"""
    ProductionNetwork{T<:AbstractFloat}

Baqaee–Farhi (2019) standard-form nested-CES production network in **row
orientation**.

# Fields
- `Omega` — sparse base cost-share matrix `Ω̃`, size `(1+M+F)²`. Row `i` is the
  expenditure shares of node `i` (rows sum to 1 for household + producers; 0 for
  factors).
- `theta` — length `1+M` elasticities (household first as `σ`; factors have none).
- `lambda` — base Domar weights `λ̃ = e₁'(I−Ω̃)⁻¹`, length `1+M+F`.
- `factor_supplies` — `L_f` normalized so base `w_f = 1` ⇒ `L_f = Λ̃_f`.
- `node_names` — labels for all `1+M+F` nodes.
- `parent` — map node → real-sector index (`0` for household / factors).
- `n`, `M`, `F` — real sectors, producer nodes (incl. fictitious nests), factors.
- `nests` — nesting scheme (`:single` or `:two`).
- `outer_nodes` — global indices of the `n` real-sector outer producer nodes.
- `io` — provenance [`IOData`](@ref).

Node layout (1-based): index 1 = household; `2:M+1` = producers; `M+2:M+F+1` =
primary factors.
"""
struct ProductionNetwork{T<:AbstractFloat}
    Omega::SparseMatrixCSC{T,Int}
    theta::Vector{T}
    lambda::Vector{T}
    factor_supplies::Vector{T}
    node_names::Vector{String}
    parent::Vector{Int}
    n::Int
    M::Int
    F::Int
    nests::Symbol
    outer_nodes::Vector{Int}
    io::IOData{T}
end

"""
    production_network(io::IOData; theta=1.0, sigma=1.0, epsilon=1.0, eta=1.0,
                       nests=:single, factors=:single, check=true) -> ProductionNetwork

Calibrate a Baqaee–Farhi standard-form network from a column-oriented
[`IOData`](@ref) table. Shares are built **once** in row orientation:

- producer `i` share on producer `j`: `Z[j,i] / x[i]`
- producer `i` factor shares: `va[f,i] / x[i]` (after the `factors` mapping)
- household shares: `βⱼ = yⱼ / Σ y` with `y = rowsum(Y)`

# Keyword arguments
- `theta` — scalar or length-`n` vector: elasticity across intermediate inputs
  (`:single` → all inputs; `:two` → within the intermediate bundle).
- `sigma` — household elasticity across goods.
- `epsilon` — scalar or length-`n`: outer elasticity VA-bundle vs intermediate
  bundle (`:two` only).
- `eta` — scalar or length-`n`: across-factor elasticity inside the VA bundle
  (`:two` only).
- `nests` — `:single` (M = n) or `:two` (M = 3n; outer + intermediate + VA
  fictitious nodes per sector). `:custom` is reserved for a future constructor.
- `factors` — `:single` (sum VA rows), `:va_cats` (one factor per VA row), or an
  `F×n` matrix of factor payments.
- `check` — if `true` (default), error when clipped negative mass exceeds 1% of
  any row.

Negative table entries (net taxes, inventory drawdowns) are clipped to zero and
the row is renormalized, with a single `@warn`. CES cost shares must be ≥ 0.
"""
function production_network(io::IOData{T};
                            theta=1.0,
                            sigma=1.0,
                            epsilon=1.0,
                            eta=1.0,
                            nests::Symbol=:single,
                            factors=:single,
                            check::Bool=true) where {T<:AbstractFloat}
    nests in (:single, :two) || throw(ArgumentError(
        "nests must be :single or :two (got $nests); :custom is not yet implemented"))

    n = length(io.x)
    x = io.x
    Z = io.Z
    Y = io.Y

    # ── factors ──────────────────────────────────────────────────────────────
    V, factor_names = _bf_factor_matrix(io, factors)
    F = size(V, 1)
    F >= 1 || throw(ArgumentError("production_network requires at least one factor"))

    # ── elasticities ─────────────────────────────────────────────────────────
    θ_sec = _bf_as_vector(T, theta, n, "theta")
    ε_sec = _bf_as_vector(T, epsilon, n, "epsilon")
    η_sec = _bf_as_vector(T, eta, n, "eta")
    σ = T(sigma)

    # ── intermediate + factor payments (column of IOData) ────────────────────
    # inter_share[j,i] = Z[j,i] / x[i]   (buyer i on supplier j)
    # fac_share[f,i]   = V[f,i] / x[i]
    inter = Matrix{T}(undef, n, n)
    fac = Matrix{T}(undef, F, n)
    for i in 1:n
        xi = x[i]
        xi > 0 || throw(ArgumentError("gross output x[$i] must be positive"))
        for j in 1:n
            inter[j, i] = Z[j, i] / xi
        end
        for f in 1:F
            fac[f, i] = V[f, i] / xi
        end
    end

    # household final-demand shares
    y = vec(sum(Y; dims=2))
    ysum = sum(y)
    ysum > 0 || throw(ArgumentError("total final demand must be positive"))
    β = y ./ ysum

    # clip negatives once across all share material
    clip_msgs = String[]
    for i in 1:n
        col_inter = @view inter[:, i]
        col_fac = @view fac[:, i]
        row = vcat(Vector{T}(col_inter), Vector{T}(col_fac))
        _bf_clip_renorm!(row, check, "producer $i"; msgs=clip_msgs)
        inter[:, i] .= row[1:n]
        fac[:, i] .= row[n+1:end]
    end
    βc = copy(β)
    _bf_clip_renorm!(βc, check, "household"; msgs=clip_msgs)
    β = βc
    if !isempty(clip_msgs)
        @warn "production_network: clipped negative cost shares and renormalized" details = clip_msgs
    end

    if nests === :single
        return _bf_build_single(io, inter, fac, β, θ_sec, σ, factor_names)
    else
        return _bf_build_two(io, inter, fac, β, θ_sec, ε_sec, η_sec, σ, factor_names)
    end
end

# ── helpers ──────────────────────────────────────────────────────────────────

function _bf_as_vector(::Type{T}, x, n::Int, name::String) where {T}
    if x isa AbstractVector
        length(x) == n || throw(ArgumentError(
            "$name must be a scalar or length-$n vector; got length $(length(x))"))
        return T[T(v) for v in x]
    else
        return fill(T(x), n)
    end
end

function _bf_factor_matrix(io::IOData{T}, factors) where {T}
    if factors === :single
        V = reshape(vec(sum(io.va; dims=1)), 1, length(io.x))
        return Matrix{T}(V), ["factor"]
    elseif factors === :va_cats
        return Matrix{T}(io.va), String.(io.va_cats)
    elseif factors isa AbstractMatrix
        Vm = Matrix{T}(factors)
        size(Vm, 2) == length(io.x) || throw(ArgumentError(
            "factors matrix must have n=$(length(io.x)) columns; got $(size(Vm, 2))"))
        size(Vm, 1) >= 1 || throw(ArgumentError("factors matrix must have ≥1 row"))
        names = ["factor$f" for f in 1:size(Vm, 1)]
        return Vm, names
    else
        throw(ArgumentError(
            "factors must be :single, :va_cats, or an F×n matrix; got $factors"))
    end
end

"""
Clip negative entries of `row` to 0 and renormalize to sum 1.
`check=true` errors if clipped mass > 1% of the pre-clip positive mass (or of 1
if the row is empty of positives). Appends a message to `msgs` when clipping.
"""
function _bf_clip_renorm!(row::AbstractVector{T}, check::Bool, label::String;
                          msgs::Vector{String}) where {T<:AbstractFloat}
    neg_mass = zero(T)
    for i in eachindex(row)
        if row[i] < 0
            neg_mass += -row[i]
            row[i] = zero(T)
        end
    end
    s = sum(row)
    if neg_mass > 0
        push!(msgs, "$label: clipped mass $(Float64(neg_mass))")
        # Plan: error if clipped mass > 1% of the (pre-clip absolute) row mass.
        orig = s + neg_mass
        if check && orig > 0 && neg_mass > T(0.01) * orig
            throw(ArgumentError(
                "production_network: clipped negative mass in $label " *
                "($(Float64(neg_mass))) exceeds 1% of row total " *
                "($(Float64(orig))); set check=false to allow"))
        end
    end
    if s > 0
        row ./= s
    end
    return row
end

function _bf_domar(Ω::SparseMatrixCSC{T,Int}) where {T}
    N = size(Ω, 1)
    IΩ = sparse(T(1) * I, N, N) - Ω
    e1 = zeros(T, N)
    e1[1] = one(T)
    # λ = (I−Ω)^{-T} e₁  ⇔  (I−Ω)' λ = e₁  (dense RHS — UMFPACK needs it)
    λ = IΩ' \ e1
    return λ
end

function _bf_build_single(io::IOData{T}, inter::Matrix{T}, fac::Matrix{T},
                          β::Vector{T}, θ_sec::Vector{T}, σ::T,
                          factor_names::Vector{String}) where {T}
    n = length(io.x)
    F = size(fac, 1)
    M = n
    N = 1 + M + F

    rows = Int[]; cols = Int[]; vals = T[]
    # household row → real producers (nodes 2:n+1)
    for j in 1:n
        β[j] == 0 && continue
        push!(rows, 1); push!(cols, j + 1); push!(vals, β[j])
    end
    # producer rows
    for i in 1:n
        g = i + 1
        for j in 1:n
            inter[j, i] == 0 && continue
            push!(rows, g); push!(cols, j + 1); push!(vals, inter[j, i])
        end
        for f in 1:F
            fac[f, i] == 0 && continue
            push!(rows, g); push!(cols, 1 + M + f); push!(vals, fac[f, i])
        end
    end
    Ω = sparse(rows, cols, vals, N, N)

    theta = vcat(σ, θ_sec)                          # length 1+M
    parent = zeros(Int, N)
    for i in 1:n
        parent[i + 1] = i
    end
    outer_nodes = collect(2:n+1)
    node_names = String["household"]
    append!(node_names, String.(io.sectors))
    append!(node_names, factor_names)

    λ = _bf_domar(Ω)
    Λ = λ[M+2:N]
    # numerical guard: factor Domars should sum to 1
    factor_supplies = copy(Λ)

    ProductionNetwork{T}(Ω, theta, λ, factor_supplies, node_names, parent,
                         n, M, F, :single, outer_nodes, io)
end

function _bf_build_two(io::IOData{T}, inter::Matrix{T}, fac::Matrix{T},
                       β::Vector{T}, θ_sec::Vector{T}, ε_sec::Vector{T},
                       η_sec::Vector{T}, σ::T,
                       factor_names::Vector{String}) where {T}
    n = length(io.x)
    F = size(fac, 1)
    M = 3n
    N = 1 + M + F

    # node indices
    # outer O_i     : 1 + i
    # inter bundle B_i : 1 + n + i
    # VA bundle V_i    : 1 + 2n + i
    # factor f         : 1 + 3n + f
    O(i) = 1 + i
    B(i) = 1 + n + i
    V(i) = 1 + 2n + i
    Ff(f) = 1 + 3n + f

    rows = Int[]; cols = Int[]; vals = T[]

    # household → outer nodes
    for j in 1:n
        β[j] == 0 && continue
        push!(rows, 1); push!(cols, O(j)); push!(vals, β[j])
    end

    for i in 1:n
        inter_tot = sum(@view inter[:, i])
        fac_tot = sum(@view fac[:, i])
        # outer node: buys B_i and V_i with shares inter_tot, fac_tot (already sum≈1)
        if inter_tot > 0
            push!(rows, O(i)); push!(cols, B(i)); push!(vals, inter_tot)
        end
        if fac_tot > 0
            push!(rows, O(i)); push!(cols, V(i)); push!(vals, fac_tot)
        end
        # if one side is zero, the other should already be 1 after clip_renorm

        # intermediate bundle: renormalized shares on outer nodes (goods)
        if inter_tot > 0
            for j in 1:n
                s = inter[j, i] / inter_tot
                s == 0 && continue
                push!(rows, B(i)); push!(cols, O(j)); push!(vals, s)
            end
        end

        # VA bundle: renormalized shares on factors
        if fac_tot > 0
            for f in 1:F
                s = fac[f, i] / fac_tot
                s == 0 && continue
                push!(rows, V(i)); push!(cols, Ff(f)); push!(vals, s)
            end
        end
    end
    Ω = sparse(rows, cols, vals, N, N)

    # theta: [σ, ε_1..ε_n, θ_1..θ_n, η_1..η_n]
    theta = vcat(σ, ε_sec, θ_sec, η_sec)

    parent = zeros(Int, N)
    for i in 1:n
        parent[O(i)] = i
        parent[B(i)] = i
        parent[V(i)] = i
    end
    outer_nodes = [O(i) for i in 1:n]

    node_names = String["household"]
    for i in 1:n
        push!(node_names, String(io.sectors[i]))
    end
    for i in 1:n
        push!(node_names, "inter_bundle_" * String(io.sectors[i]))
    end
    for i in 1:n
        push!(node_names, "va_bundle_" * String(io.sectors[i]))
    end
    append!(node_names, factor_names)

    λ = _bf_domar(Ω)
    factor_supplies = copy(λ[M+2:N])

    ProductionNetwork{T}(Ω, theta, λ, factor_supplies, node_names, parent,
                         n, M, F, :two, outer_nodes, io)
end
