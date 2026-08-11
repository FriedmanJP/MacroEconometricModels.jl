# ras.jl — RAS / GRAS matrix balancing and IOData repair
#
# RAS  — biproportional update (Bacharach / Stone) for non-negative matrices.
# GRAS — sign-preserving generalized RAS (Junius & Oosterhaven 2003; target
#         function corrected by Lenzen, Wood & Gallego 2007 / Temurshoev et al.).
#
# Orientation: operates on an arbitrary flow matrix; when used via `balance(io)`
# the intermediate block Z is treated in the package's column orientation.

"""
    RASResult{T}

Result of a biproportional matrix update ([`ras`](@ref) or [`gras`](@ref)).

# Fields
- `X` — balanced matrix (same size as the prior)
- `r`, `s` — row and column multipliers relative to the prior
- `iterations` — number of outer iterations performed
- `converged` — `true` if max margin residual ≤ `tol`
- `method` — `:ras` or `:gras`
- `residual_u`, `residual_v` — final row- and column-sum residuals
"""
struct RASResult{T<:AbstractFloat}
    X::Matrix{T}
    r::Vector{T}
    s::Vector{T}
    iterations::Int
    converged::Bool
    method::Symbol
    residual_u::Vector{T}
    residual_v::Vector{T}
end

# ---------------------------------------------------------------------------
# RAS — classical biproportional scaling (non-negative)
# ---------------------------------------------------------------------------

"""
    ras(A0, u, v; tol=1e-10, maxiter=1000) -> RASResult

Biproportional (RAS) update of a non-negative prior matrix `A0` to target row
sums `u` and column sums `v` (Bacharach 1970; Stone 1961).

Iterates row scaling then column scaling:

```math
X \\leftarrow \\hat{r}\\,X, \\qquad r_i = u_i / \\textstyle\\sum_j X_{ij}
```
```math
X \\leftarrow X\\,\\hat{s}, \\qquad s_j = v_j / \\textstyle\\sum_i X_{ij}
```

until margin residuals fall below `tol`. Zero cells in `A0` stay zero
(structure-preserving). Requires ``\\sum u = \\sum v`` (up to `tol`) and
non-negative `A0`, `u`, `v`.

# Orientation
Generic flow-matrix operator; no IO orientation is implied. For intermediate
flows of an [`IOData`](@ref), see [`balance`](@ref).
"""
function ras(A0::AbstractMatrix, u::AbstractVector, v::AbstractVector;
             tol::Real=1e-10, maxiter::Integer=1000)
    T = promote_type(eltype(A0), eltype(u), eltype(v), Float64)
    A = Matrix{T}(A0)
    uv = Vector{T}(u)
    vv = Vector{T}(v)
    m, n = size(A)
    length(uv) == m || throw(ArgumentError("length(u)=$(length(uv)) ≠ nrows=$m"))
    length(vv) == n || throw(ArgumentError("length(v)=$(length(vv)) ≠ ncols=$n"))
    any(<(zero(T)), A) && throw(ArgumentError(
        "ras requires a non-negative prior; use gras for matrices with negatives"))
    any(<(zero(T)), uv) && throw(ArgumentError("u must be non-negative for ras"))
    any(<(zero(T)), vv) && throw(ArgumentError("v must be non-negative for ras"))
    abs(sum(uv) - sum(vv)) ≤ max(T(tol), eps(T) * max(one(T), abs(sum(uv)))) ||
        throw(ArgumentError("sum(u)=$(sum(uv)) ≠ sum(v)=$(sum(vv)); " *
                            "row and column margins must agree"))

    X = copy(A)
    r_tot = ones(T, m)
    s_tot = ones(T, n)
    converged = false
    iter = 0
    for it in 1:Int(maxiter)
        iter = it
        # Row scaling
        rs = vec(sum(X, dims=2))
        r = _ras_ratio(uv, rs)
        X .*= r
        r_tot .*= r
        # Column scaling
        cs = vec(sum(X, dims=1))
        s = _ras_ratio(vv, cs)
        X .*= reshape(s, 1, n)
        s_tot .*= s
        # Check residuals
        ru = uv .- vec(sum(X, dims=2))
        rv = vv .- vec(sum(X, dims=1))
        if maximum(abs, ru; init=zero(T)) ≤ T(tol) &&
           maximum(abs, rv; init=zero(T)) ≤ T(tol)
            converged = true
            break
        end
    end
    ru = uv .- vec(sum(X, dims=2))
    rv = vv .- vec(sum(X, dims=1))
    return RASResult{T}(X, r_tot, s_tot, iter, converged, :ras, ru, rv)
end

"Guarded target/current ratio: 0/0 → 1, target/0 with target≠0 → error."
function _ras_ratio(target::AbstractVector{T}, current::AbstractVector{T}) where {T}
    r = similar(target)
    @inbounds for i in eachindex(target)
        ti, ci = target[i], current[i]
        if ci == zero(T)
            ti == zero(T) || throw(ArgumentError(
                "RAS infeasible: target[$i]=$ti but current row/col sum is 0 " *
                "(no positive mass to scale)"))
            r[i] = one(T)
        else
            r[i] = ti / ci
        end
    end
    return r
end

# ---------------------------------------------------------------------------
# GRAS — sign-preserving generalized RAS
# ---------------------------------------------------------------------------

"""
    gras(A0, u, v; tol=1e-10, maxiter=1000) -> RASResult

Sign-preserving biproportional update of a prior matrix that may contain
**negative** entries (Junius & Oosterhaven 2003; algorithm with the Lenzen,
Wood & Gallego 2007 corrected target function as implemented by Temurshoev
et al. 2013).

Decompose ``A_0 = P - N`` with ``P, N ≥ 0`` elementwise disjoint, and iterate
row/column multipliers ``r, s > 0``:

```math
X = \\hat{r}\\,P\\,\\hat{s} - \\hat{r}^{-1}\\,N\\,\\hat{s}^{-1}
```

with closed-form updates (for rows with positive mass ``p_i = (P s)_i``)

```math
r_i = \\frac{u_i + \\sqrt{u_i^2 + 4 p_i n_i}}{2 p_i}, \\qquad
n_i = (N s^{-1})_i
```

and the analogous column update. Zeros in `A0` remain zero. Requires
``\\sum u = \\sum v``.

When `A0 ≥ 0`, GRAS reduces to (a reparameterization of) classical RAS.

# Orientation
Generic flow-matrix operator. For intermediate flows of an [`IOData`](@ref),
see [`balance`](@ref).
"""
function gras(A0::AbstractMatrix, u::AbstractVector, v::AbstractVector;
              tol::Real=1e-10, maxiter::Integer=1000)
    T = promote_type(eltype(A0), eltype(u), eltype(v), Float64)
    A = Matrix{T}(A0)
    uv = Vector{T}(u)
    vv = Vector{T}(v)
    m, n = size(A)
    length(uv) == m || throw(ArgumentError("length(u)=$(length(uv)) ≠ nrows=$m"))
    length(vv) == n || throw(ArgumentError("length(v)=$(length(vv)) ≠ ncols=$n"))
    abs(sum(uv) - sum(vv)) ≤ max(T(tol), eps(T) * max(one(T), abs(sum(uv)))) ||
        throw(ArgumentError("sum(u)=$(sum(uv)) ≠ sum(v)=$(sum(vv)); " *
                            "row and column margins must agree"))

    # P = positive part, N = absolute negative part
    P = max.(A, zero(T))
    N = max.(-A, zero(T))

    r = ones(T, m)
    s = ones(T, n)
    inv_safe(x) = ifelse(x == zero(T), one(T), one(T) / x)

    converged = false
    iter = 0
    for it in 1:Int(maxiter)
        iter = it
        r_old = copy(r)
        s_old = copy(s)

        # --- update r given s ---
        pr = P * s                         # p_i(s)
        nr = N * (inv_safe.(s))            # n_i(s)
        for i in 1:m
            r[i] = _gras_mult(uv[i], pr[i], nr[i])
        end

        # --- update s given r ---
        ps = P' * r
        ns = N' * (inv_safe.(r))
        for j in 1:n
            s[j] = _gras_mult(vv[j], ps[j], ns[j])
        end

        # Convergence on multipliers (and, periodically, on margins)
        if maximum(abs, r .- r_old; init=zero(T)) ≤ T(tol) &&
           maximum(abs, s .- s_old; init=zero(T)) ≤ T(tol)
            converged = true
            break
        end
    end

    X = (r .* P .* s') .- (inv_safe.(r) .* N .* inv_safe.(s)')
    # Preserve exact zeros of the prior
    @inbounds for i in eachindex(A)
        if A[i] == zero(T)
            X[i] = zero(T)
        end
    end
    ru = uv .- vec(sum(X, dims=2))
    rv = vv .- vec(sum(X, dims=1))
    if !converged
        # Also accept if margins are already within tol
        if maximum(abs, ru; init=zero(T)) ≤ T(tol) &&
           maximum(abs, rv; init=zero(T)) ≤ T(tol)
            converged = true
        end
    end
    return RASResult{T}(X, r, s, iter, converged, :gras, ru, rv)
end

"""
Closed-form GRAS multiplier for one margin.
`p` = positive mass, `n` = absolute negative mass, `t` = target sum.
"""
function _gras_mult(t::T, p::T, n::T) where {T<:AbstractFloat}
    if p > zero(T)
        # Positive root of p r − n/r = t  ⇒  p r² − t r − n = 0
        return (t + sqrt(t * t + 4 * p * n)) / (2 * p)
    elseif n > zero(T)
        # Only negative mass: −n/r = t  ⇒  r = −n/t (requires t < 0)
        t == zero(T) && throw(ArgumentError(
            "GRAS infeasible: zero target with pure-negative row/column mass"))
        t > zero(T) && throw(ArgumentError(
            "GRAS infeasible: positive target with pure-negative row/column mass"))
        return -n / t
    else
        # No mass at all
        t == zero(T) || throw(ArgumentError(
            "GRAS infeasible: nonzero target with zero row/column mass"))
        return one(T)
    end
end

# ---------------------------------------------------------------------------
# balance — repair IOData intermediate flows
# ---------------------------------------------------------------------------

"""
    balance(io; method=:ras, tol=1e-10, maxiter=1000) -> IOData

Repair an [`IOData`](@ref) whose intermediate block fails the accounting
identities, by biproportionally adjusting ``Z`` to the intermediate margins
implied by gross output and the (held fixed) final-demand / value-added blocks:

```math
u = x - Y\\mathbf{1}, \\qquad v = x - \\mathbf{1}'V
```

so that after balancing, ``\\mathrm{rowsum}(Z) + \\mathrm{rowsum}(Y) = x`` and
``\\mathrm{colsum}(Z) + \\mathrm{colsum}(V) = x``.

# Arguments
- `method` — `:ras` (default; non-negative ``Z``) or `:gras` (sign-preserving)
- `tol`, `maxiter` — passed to [`ras`](@ref) / [`gras`](@ref)

Labels, extensions, and metadata are copied through. The returned table is
constructed with `check=true`. If the prior is already balanced, it is a fixed
point of the procedure (returned essentially unchanged).

# Orientation
Column orientation: ``Z_{ij}`` is the flow of ``i`` into ``j``; ``u`` are
intermediate *sales* by row, ``v`` intermediate *purchases* by column.
"""
function balance(io::IOData{T};
                 method::Symbol=:ras,
                 tol::Real=1e-10,
                 maxiter::Integer=1000) where {T<:AbstractFloat}
    method in (:ras, :gras) ||
        throw(ArgumentError("method must be :ras or :gras; got :$method"))

    # Intermediate margins implied by the published gross-output vector and the
    # (held fixed) final-demand / value-added blocks.
    u = io.x .- vec(sum(io.Y, dims=2))
    v = io.x .- vec(sum(io.va, dims=1))

    su, sv = sum(u), sum(v)
    gap = abs(su - sv)
    tolT = max(T(tol), eps(T) * max(one(T), abs(su), abs(sv)))
    if gap > tolT
        # Small float/parse noise: equalize totals by a common mid-scale so the
        # biproportional step remains feasible. Large gaps are user error.
        gap ≤ 1e-4 * max(one(T), abs(su), abs(sv)) || throw(ArgumentError(
            "balance: intermediate row total ($(su)) and column total ($(sv)) " *
            "differ by $gap — check Y/va/x consistency before balancing"))
        mid = (su + sv) / T(2)
        su != zero(T) && (u = u .* (mid / su))
        sv != zero(T) && (v = v .* (mid / sv))
    end

    # Vacuous case: no intermediate flows to adjust
    if all(iszero, u) && all(iszero, v)
        return io
    end

    res = method === :ras ?
        ras(io.Z, u, v; tol=tol, maxiter=maxiter) :
        gras(io.Z, u, v; tol=tol, maxiter=maxiter)

    res.converged || @warn "balance: $method did not converge in $(res.iterations) iterations" maxlog=1

    # Rebuild: Z meets intermediate margins of the original x, so both
    # accounting identities hold with the original Y, va, and x.
    out = IOData(res.X, io.Y, io.x;
                 va=copy(io.va),
                 sectors=copy(io.sectors),
                 regions=copy(io.regions),
                 fd_cats=copy(io.fd_cats),
                 va_cats=copy(io.va_cats),
                 unit=io.unit, year=io.year, source=io.source,
                 meta=io.meta, check=true)
    for (name, ext) in io.extensions
        add_extension!(out, name, ext.F; stressors=ext.stressors,
                       unit=ext.unit, F_Y=ext.F_Y)
    end
    return out
end
