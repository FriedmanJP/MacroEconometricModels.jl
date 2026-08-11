# impact.jl — final-demand impact / scenario API (open, closed, mixed)

"""
    ImpactResult

Output of [`impact`](@ref): total and per-sector impacts of a final-demand
scenario through the Leontief inverse (optionally Type II or mixed).

# Fields
- `total` — economy-wide impact (sum of `by_sector` for `:output`; weighted sum otherwise)
- `by_sector` — ``n \\times 1`` impact by producing sector
- `dy` — ``n \\times 1`` final-demand change used
- `kind` — quantity impacted (`:output`, `:va`, `:income`, `:employment`, or an extension name)
- `type` — `:I` (open) or `:II` (household-closed); ignored when `fixed` is nonempty
- `sectors` — sector labels
- `fixed` — indices of output-exogenous sectors (mixed model); empty for the pure demand model
"""
struct ImpactResult
    total::Float64
    by_sector::Vector{Float64}
    dy::Vector{Float64}
    kind::Symbol
    type::Symbol
    sectors::Vector{String}
    fixed::Vector{Int}
end

"""
    impact(io, dy; kind=:output, type=:I, fix=Dict()) -> ImpactResult

Scenario wrapper for classical impact analysis (column orientation).

Given a final-demand change ``\\Delta y`` (length-``n`` vector or
`Dict(sector => Δ)`), returns the production programme

```math
\\Delta x = L\\,\\Delta y
```

and the associated impact of `kind` through ``L`` (or the Type II closed inverse
when `type=:II`).

# Arguments
- `dy` — length-``n`` final-demand change, or a `Dict` of sector index/name → change
- `kind` — `:output` (default), `:va` / `:income` (value-added weighted),
  `:employment` (needs an `employment` extension), or a `Symbol`/`String` naming
  any satellite extension (uses the first stressor row as the intensity)
- `type` — `:I` open model or `:II` household-closed (Type II). Ignored when
  `fix` is nonempty.
- `fix` — `Dict(sector => x̄)` of **exogenous outputs** for the mixed model
  (Miller & Blair ch. 13). Endogenous sectors still respond to demand; exogenous
  sectors' outputs are held at `x̄` and their residual final demand is implied.

# Orientation
Column orientation: ``A = Z\\hat{x}^{-1}``, ``x = L y``, ``A_{ij}`` = input of
``i`` per unit output of ``j``.
"""
function impact(io::IOData, dy;
                kind=:output, type::Symbol=:I, fix=Dict())
    n = length(io.x)
    dyv = _impact_dy(io, dy, n)
    kind_sym = kind isa Symbol ? kind : Symbol(kind)

    fixed_idx = Int[]
    if !(fix isa AbstractDict) || !isempty(fix)
        fix isa AbstractDict || throw(ArgumentError("fix must be a Dict(sector => x̄)"))
        for (k, _) in fix
            append!(fixed_idx, _sector_indices(io, k))
        end
        unique!(sort!(fixed_idx))
    end

    if !isempty(fixed_idx)
        x_new = _mixed_model_output(io, dyv, fix)
        dx = x_new .- io.x
        by_sec, total = _impact_from_dx(io, dx, kind_sym)
        return ImpactResult(total, by_sec, dyv, kind_sym, :mixed,
                            copy(io.sectors), fixed_idx)
    end

    if type === :I
        L = leontief_inverse(io)
        dx = L * dyv
    elseif type === :II
        L2 = _closed_leontief(io)
        # Type II: augment dy with a zero household-exogenous entry and take
        # the production block of the response.
        dy2 = vcat(dyv, 0.0)
        dx2 = L2 * dy2
        dx = dx2[1:n]
    else
        throw(ArgumentError("type must be :I or :II; got :$type"))
    end

    by_sec, total = _impact_from_dx(io, dx, kind_sym)
    ImpactResult(total, by_sec, dyv, kind_sym, type, copy(io.sectors), Int[])
end

function _impact_dy(io::IOData, dy, n::Int)
    if dy isa AbstractDict
        v = zeros(Float64, n)
        for (k, val) in dy
            idx = _sector_indices(io, k)
            length(idx) == 1 || throw(ArgumentError(
                "dy key must identify a single sector; got $k → $idx"))
            v[idx[1]] = Float64(val)
        end
        return v
    end
    v = Float64.(vec(collect(dy)))
    length(v) == n || throw(ArgumentError(
        "dy length $(length(v)) must equal n=$n"))
    return v
end

# Map a production change Δx into the requested impact metric.
function _impact_from_dx(io::IOData, dx::AbstractVector, kind::Symbol)
    h = _impact_coeffs(io, kind)
    by_sec = kind === :output ? Float64.(dx) : Float64.(h .* dx)
    total = kind === :output ? sum(dx) : sum(h .* dx)
    return by_sec, Float64(total)
end

function _impact_coeffs(io::IOData, kind::Symbol)
    invx = _invdiag(io.x)
    if kind === :output
        return ones(Float64, length(io.x))
    elseif kind === :va || kind === :income
        return Float64.(vec(sum(io.va, dims=1)) .* invx)
    elseif kind === :employment
        haskey(io.extensions, "employment") ||
            throw(ArgumentError("no 'employment' extension; add one with add_extension!"))
        F = io.extensions["employment"].F
        return Float64.(vec(sum(F, dims=1)) .* invx)
    else
        # Treat as satellite-extension name (first stressor row).
        name = String(kind)
        haskey(io.extensions, name) || throw(ArgumentError(
            "kind must be :output, :va, :income, :employment, or an extension name; " *
            "got :$kind (available extensions: $(collect(keys(io.extensions))))"))
        S = io.extensions[name].S
        return Float64.(vec(S[1, :]))
    end
end

# Miller & Blair ch. 13 mixed endogenous/exogenous model.
# Sectors in `fix` have exogenous gross output; remaining sectors are demand-driven.
# Returns the full output vector x under the scenario.
function _mixed_model_output(io::IOData, dy::AbstractVector, fix::AbstractDict)
    n = length(io.x)
    A = technical_coefficients(io)
    y0 = vec(sum(io.Y, dims=2))
    y = y0 .+ dy

    fixed = Int[]
    x_fix = zeros(Float64, n)
    for (k, val) in fix
        idx = _sector_indices(io, k)
        length(idx) == 1 || throw(ArgumentError(
            "fix key must identify a single sector; got $k → $idx"))
        i = idx[1]
        push!(fixed, i)
        x_fix[i] = Float64(val)
    end
    unique!(sort!(fixed))
    endo = setdiff(collect(1:n), fixed)
    isempty(endo) && return Float64.(x_fix)

    # Partition: x_U = (I − A_UU)⁻¹ (y_U + A_UK x_K)
    A_UU = A[endo, endo]
    A_UK = isempty(fixed) ? zeros(Float64, length(endo), 0) : A[endo, fixed]
    x_K = x_fix[fixed]
    rhs = y[endo] .+ A_UK * x_K
    x_U = Matrix{Float64}(robust_inv(Matrix(I - A_UU))) * rhs

    x = zeros(Float64, n)
    x[endo] = x_U
    x[fixed] = x_K
    return x
end
