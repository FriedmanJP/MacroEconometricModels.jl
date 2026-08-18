# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — determinacy-region mapping over parameter grids ([T268])
#
# "Over what region of the parameter space is the equilibrium determinate?" is one of the
# standard questions asked of a linear DSGE model — the Taylor principle being the canonical
# instance. Answering it means re-solving the model at every point of a parameter grid and
# recording the existence/uniqueness verdict, which users otherwise hand-roll around `solve`.
#
# The verdict itself is the Sims (2002) rank test ([T267]); this file only sweeps and packages.
#
# References:
#   Sims, C. A. (2002). Solving linear rational expectations models.
#     Computational Economics 20(1-2), 1-20.
#   Lubik, T. A. & Schorfheide, F. (2003). Computing sunspot equilibria in linear rational
#     expectations models. Journal of Economic Dynamics and Control 28(2), 273-285.
#   Lubik, T. A. & Schorfheide, F. (2004). Testing for indeterminacy: an application to
#     U.S. monetary policy. American Economic Review 94(1), 190-217.

"""
Verdict codes stored in [`DeterminacyMap`](@ref)`.verdict`. Ordered, so a sequential colour
ramp over them is meaningful: worse outcomes are more negative.

| Code | Meaning | `eu` |
|------|---------|------|
| `1` | determinate — a unique stable solution | `[1, 1]` |
| `0` | indeterminate — a continuum of stable solutions (sunspots) | `[1, 0]` |
| `-1` | no stable solution | `[0, ·]` |
| `-2` | the model could not be solved at this grid point | — |
"""
const DETERMINACY_CODES = (determinate=1, indeterminate=0, no_solution=-1, failed=-2)

"""
    determinacy_label(code::Integer) → String

Human-readable name for a [`DETERMINACY_CODES`](@ref) verdict code.
"""
function determinacy_label(code::Integer)
    code == 1 && return "determinate"
    code == 0 && return "indeterminate"
    code == -1 && return "no stable solution"
    code == -2 && return "solve failed"
    throw(ArgumentError("unknown determinacy code $code"))
end

"""
    DeterminacyMap{T}

Result of [`determinacy_region`](@ref): the determinacy verdict of a DSGE model over a grid of
one or two parameters.

# Fields
- `params::Vector{Symbol}` — the swept parameter name(s), 1 or 2 of them
- `axes::Vector{Vector{T}}` — grid values, one vector per swept parameter
- `verdict::Matrix{Int}` — `length(axes[1]) × length(axes[2])` verdict codes
  ([`DETERMINACY_CODES`](@ref)); the second dimension is `1` for a one-parameter sweep
- `eu::Array{Int,3}` — the raw Sims `[existence, uniqueness]` pair at each cell,
  `size(verdict)..., 2`; `[-1, -1]` where the solve failed
- `failures::Dict{Tuple{Int,Int},String}` — error message at each failed cell, keyed by
  grid index
- `base_values::Dict{Symbol,T}` — the parameter vector the sweep varies around
- `div::Float64` — the stable/unstable eigenvalue boundary used ([T123])
- `method::Symbol` — the linear solver used at each grid point
"""
struct DeterminacyMap{T<:AbstractFloat}
    params::Vector{Symbol}
    axes::Vector{Vector{T}}
    verdict::Matrix{Int}
    eu::Array{Int,3}
    failures::Dict{Tuple{Int,Int},String}
    base_values::Dict{Symbol,T}
    div::Float64
    method::Symbol
end

"""
    is_1d(m::DeterminacyMap) → Bool

`true` when only one parameter was swept.
"""
_dm_is_1d(m::DeterminacyMap) = length(m.params) == 1

"""
    determinacy_region(spec, θ_base=spec.param_values; params, grids, …) → DeterminacyMap

Sweep one or two parameters of a DSGE model and record the determinacy verdict at each grid
point.

# Arguments
- `spec::ModelSpec` — the model. It is re-specified at each grid point through the shared
  `_respec` helper, so `linear=`, augmentation metadata and the steady-state function are
  preserved; the steady state is recomputed per point by `solve`.
- `θ_base` — parameter values to hold fixed away from the swept dimensions. Defaults to the
  model's own values.

# Keywords
- `params` — a `Symbol` or a 1-2 element collection of `Symbol`s to sweep
- `grids` — matching grid values: one vector for a single parameter, or a 2-element
  collection of vectors
- `div::Real=1.0+1e-8` — stable/unstable eigenvalue boundary, forwarded to the solver so
  knife-edge classification is controllable ([T123])
- `rank_rtol::Real=1e-8` — relative tolerance of the Sims rank tests ([T267])
- `method::Symbol=:gensys` — linear solver (`:gensys`, `:klein`, `:blanchard_kahn`)
- `threaded::Bool=false` — evaluate grid points on all available threads. Each point builds
  its own spec and solution, so there is no shared mutable state; results are written to
  disjoint indices and are therefore identical to the serial sweep.
- `quiet::Bool=true` — suppress per-point solver warnings. A 40×40 sweep crosses the
  determinacy boundary hundreds of times and would otherwise emit a warning at each; the
  information is not lost, since every non-determinate cell is recorded in `verdict` and
  every failure in `failures`.

# Failures
A grid point that cannot be solved at all — a steady state that does not converge, a singular
system, a parameter value outside the model's domain — is recorded as code `-2` with its error
message in `failures`, and the sweep continues. Only `InterruptException` propagates.

# Example
```julia
spec = compute_steady_state(nk_spec)
m = determinacy_region(spec; params=:phi_pi, grids=range(0.0, 3.0; length=61))
report(m)                       # boundary at the Taylor principle
```
"""
function determinacy_region(spec::ModelSpec{T},
                            theta_base::AbstractDict{Symbol,<:Real}=spec.param_values;
                            params,
                            grids,
                            div::Real=1.0 + 1e-8,
                            rank_rtol::Real=1e-8,
                            method::Symbol=:gensys,
                            threaded::Bool=false,
                            quiet::Bool=true) where {T<:AbstractFloat}
    pnames = params isa Symbol ? [params] : collect(Symbol.(params))
    (1 <= length(pnames) <= 2) || throw(ArgumentError(
        "determinacy_region sweeps 1 or 2 parameters, got $(length(pnames))"))
    for p in pnames
        p in spec.params || throw(ArgumentError(
            "parameter :$p is not a parameter of this model (have $(spec.params))"))
    end
    length(unique(pnames)) == length(pnames) || throw(ArgumentError(
        "the swept parameters must be distinct, got $pnames"))
    method in (:gensys, :klein, :blanchard_kahn) || throw(ArgumentError(
        "method must be :gensys, :klein, or :blanchard_kahn; got :$method"))

    # A bare vector is the natural spelling for one parameter; a collection of vectors for two.
    gaxes = if length(pnames) == 1 && !(grids isa Tuple) &&
               !(grids isa AbstractVector{<:AbstractVector})
        [collect(T, grids)]
    else
        [collect(T, g) for g in grids]
    end
    length(gaxes) == length(pnames) || throw(ArgumentError(
        "got $(length(pnames)) parameter(s) but $(length(gaxes)) grid(s)"))
    all(!isempty, gaxes) || throw(ArgumentError("grids must be non-empty"))

    base = Dict{Symbol,T}(k => T(v) for (k, v) in theta_base)
    n1 = length(gaxes[1])
    n2 = length(pnames) == 2 ? length(gaxes[2]) : 1

    verdict = Matrix{Int}(undef, n1, n2)
    eu = Array{Int,3}(undef, n1, n2, 2)
    msgs = Vector{Union{Nothing,String}}(nothing, n1 * n2)   # flat ⇒ no lock when threaded

    function evaluate!(idx::Int)
        i = ((idx - 1) % n1) + 1
        j = ((idx - 1) ÷ n1) + 1
        pv = copy(base)
        pv[pnames[1]] = gaxes[1][i]
        length(pnames) == 2 && (pv[pnames[2]] = gaxes[2][j])
        try
            sol = solve(_respec(spec, pv); method=method, div=div, rank_rtol=rank_rtol)
            e = sol.eu
            eu[i, j, 1] = e[1]
            eu[i, j, 2] = e[2]
            verdict[i, j] = e[1] == 0 ? DETERMINACY_CODES.no_solution :
                            (e[2] == 1 ? DETERMINACY_CODES.determinate :
                                         DETERMINACY_CODES.indeterminate)
        catch err
            err isa InterruptException && rethrow()
            verdict[i, j] = DETERMINACY_CODES.failed
            eu[i, j, 1] = -1
            eu[i, j, 2] = -1
            msgs[idx] = sprint(showerror, err)
        end
        return nothing
    end

    run_sweep() = if threaded
        Threads.@threads for idx in 1:(n1 * n2)
            evaluate!(idx)
        end
    else
        for idx in 1:(n1 * n2)
            evaluate!(idx)
        end
    end
    quiet ? _suppress_warnings(run_sweep) : run_sweep()

    failures = Dict{Tuple{Int,Int},String}()
    for idx in 1:(n1 * n2)
        m = msgs[idx]
        m === nothing && continue
        failures[(((idx - 1) % n1) + 1, ((idx - 1) ÷ n1) + 1)] = m
    end

    return DeterminacyMap{T}(pnames, gaxes, verdict, eu, failures, base,
                             Float64(div), method)
end

"""
    determinacy_boundary(m::DeterminacyMap) → Vector

For a ONE-parameter sweep, the grid values at which the verdict changes, reported as the
midpoint of each bracketing pair. Empty when the verdict is constant across the grid.

Pairs involving a **failed** grid point are skipped: a solve failure is missing information, not
a determinacy region, so the step from "failed" to "determinate" is not a boundary crossing and
reporting it as one would invent a frontier out of a numerical gap.

The resolution of a boundary located this way is the grid spacing — refine the grid to sharpen
it. Errors for a two-parameter sweep, where the boundary is a curve rather than a set of points.
"""
function determinacy_boundary(m::DeterminacyMap{T}) where {T}
    _dm_is_1d(m) || throw(ArgumentError(
        "determinacy_boundary is defined for a one-parameter sweep; this map sweeps " *
        "$(length(m.params)) parameters — read `verdict` directly."))
    g = m.axes[1]
    v = vec(m.verdict)
    fail = DETERMINACY_CODES.failed
    return T[(g[i] + g[i+1]) / 2 for i in 1:(length(g)-1)
             if v[i] != v[i+1] && v[i] != fail && v[i+1] != fail]
end

"""
    _dm_counts(m::DeterminacyMap) → Vector{Pair{String,Int}}

Cell counts per verdict category, in code order, omitting empty categories.
"""
function _dm_counts(m::DeterminacyMap)
    out = Pair{String,Int}[]
    for code in (1, 0, -1, -2)
        c = count(==(code), m.verdict)
        c > 0 && push!(out, determinacy_label(code) => c)
    end
    return out
end

# Single characters for the terminal map; `report` prints the legend alongside.
_dm_char(code::Integer) = code == 1 ? 'D' : code == 0 ? 'I' : code == -1 ? '.' : 'x'

function Base.show(io::IO, m::DeterminacyMap{T}) where {T}
    dims = join(string.(length.(m.axes)), "×")
    print(io, "DeterminacyMap{$T}: ", join(string.(m.params), " × "), " ($dims grid), ",
          count(==(1), m.verdict), "/", length(m.verdict), " determinate")
end

"""
    report(m::DeterminacyMap)

Print the determinacy sweep: setup, verdict counts, an ASCII map of the grid, and — for a
one-parameter sweep — the located boundary.

The map is drawn with the first parameter on the vertical axis (ascending downward) and the
second on the horizontal, so the printed layout matches `m.verdict` element for element.
"""
function report(io::IO, m::DeterminacyMap{T}) where {T}
    n1, n2 = size(m.verdict)

    setup = Any[
        "Swept parameters"  join(string.(m.params), ", ");
        "Grid"              join(string.(length.(m.axes)), " × ");
        "Range"             join([string(_fmt(first(a); digits=4), " … ",
                                         _fmt(last(a); digits=4)) for a in m.axes], ",  ");
        "Solver"            ":$(m.method)";
        "div"               _fmt(m.div; digits=10)
    ]
    _pretty_table(io, setup; title="Determinacy Region", column_labels=["", ""],
                  alignment=[:l, :r])

    total = length(m.verdict)
    counts = _dm_counts(m)
    cdata = Matrix{Any}(undef, length(counts), 3)
    for (r, p) in enumerate(counts)
        cdata[r, 1] = first(p)
        cdata[r, 2] = last(p)
        cdata[r, 3] = _fmt(100 * last(p) / total; digits=1) * "%"
    end
    _pretty_table(io, cdata; title="Verdicts", column_labels=["Region", "Cells", "Share"],
                  alignment=[:l, :r, :r])

    # ASCII map. Wide grids are printed in full — truncating a determinacy map would hide
    # exactly the boundary the user is looking for.
    println(io)
    println(io, "  Map  (D determinate, I indeterminate, . no stable solution, x solve failed)")
    if _dm_is_1d(m)
        println(io, "  ", String([_dm_char(v) for v in vec(m.verdict)]))
        println(io, "  ", string(m.params[1]), ": ", _fmt(first(m.axes[1]); digits=4),
                " → ", _fmt(last(m.axes[1]); digits=4))
        b = determinacy_boundary(m)
        if isempty(b)
            println(io, "  No determinacy boundary located on this grid.")
        else
            println(io, "  Boundary at ", join([_fmt(x; digits=4) for x in b], ", "),
                    "  (± half a grid step)")
        end
    else
        for i in 1:n1
            println(io, "  ", String([_dm_char(m.verdict[i, j]) for j in 1:n2]))
        end
        println(io, "  rows: ", string(m.params[1]), " ", _fmt(first(m.axes[1]); digits=4),
                " → ", _fmt(last(m.axes[1]); digits=4), " (top→bottom)")
        println(io, "  cols: ", string(m.params[2]), " ", _fmt(first(m.axes[2]); digits=4),
                " → ", _fmt(last(m.axes[2]); digits=4), " (left→right)")
    end

    if !isempty(m.failures)
        println(io)
        println(io, "  $(length(m.failures)) grid point(s) failed to solve; first message:")
        k = first(sort(collect(keys(m.failures))))
        println(io, "    at ", k, ": ", first(split(m.failures[k], '\n')))
    end
    return nothing
end

report(m::DeterminacyMap) = report(stdout, m)
