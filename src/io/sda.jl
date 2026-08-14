# sda.jl — Structural Decomposition Analysis (Dietzenbacher & Los 1998)
# General n-determinant two-polar average (additive) + 2-factor multiplicative.

"""
    SDAResult

Result of a structural decomposition: per-determinant `effects`, the `total`
change (or ratio), the additive/multiplicative `residual`, the `method`, the
indicator `on` (`:output` or an extension name), and the ordered `factors`.

# Orientation
Column orientation: ``A = Z\\hat{x}^{-1}``, ``x = L y``. Emission SDA uses the
same Leontief inverse with intensities ``S = F\\hat{x}^{-1}``.
"""
struct SDAResult
    effects::Dict{Symbol,Vector{Float64}}
    total::Vector{Float64}
    residual::Vector{Float64}
    method::Symbol
    on::Any                 # :output or String extension name
    factors::Vector{Symbol}
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    sda(io0, io1; method=:additive, factors=nothing, on=:output) -> SDAResult

Decompose the change in an indicator between two IO tables into contributions
from each listed determinant using the Dietzenbacher & Los (1998) **two-polar
average**.

# Indicators (`on`)
- `:output` (default) — gross output ``x = L y``
- `"<extension>"` or `:<extension>` — total stressor account
  ``e = S L y`` (emission / satellite SDA). `S` is the intensity matrix of the
  named extension on each table.

# Factors
Pass an ordered `Vector{Symbol}` of determinants. Defaults:
- `on=:output` → `[:technology, :final_demand]` (keys stored as `:L`, `:Y` for
  backward compatibility when `factors` is omitted)
- extension → `[:intensity, :technology, :final_demand]`

Recognized determinants:

| Symbol | Meaning |
|--------|---------|
| `:technology` | Leontief inverse ``L`` |
| `:final_demand` / `:fd` | Total final-demand vector ``y = Y\\mathbf{1}`` |
| `:fd_level` | Scalar FD level ``g = \\mathbf{1}'y`` |
| `:fd_mix` | Product composition ``m = y/g`` (or within-category mix when
  `:fd_destination` is also present) |
| `:fd_destination` | FD-category shares ``d`` (requires multi-column ``Y``) |
| `:intensity` | Extension intensities ``S`` (only with an extension `on`) |

Final-demand splits are mutually exclusive with bare `:final_demand`:
- `[:technology, :final_demand]` — classical two-factor ``L``/``y``
- `[:technology, :fd_level, :fd_mix]` — level × mix
- `[:technology, :fd_level, :fd_mix, :fd_destination]` — level × within-category
  mix × category shares

# Methods
- `method=:additive` — two-polar arithmetic average; residual ≈ 0 by construction
- `method=:multiplicative` — two-polar geometric mean (supported for the
  classical two-factor output path only)

# Orientation
Column orientation throughout: ``A = Z\\hat{x}^{-1}``, ``x = L y``,
``A_{ij}`` = input of ``i`` per unit output of ``j``.
"""
function sda(io0::IOData, io1::IOData;
             method::Symbol=:additive,
             factors=nothing,
             on=:output)
    length(io0.x) == length(io1.x) ||
        throw(ArgumentError("io0 and io1 must have the same number of sectors; " *
                            "got $(length(io0.x)) and $(length(io1.x))"))

    on_key, is_ext = _sda_parse_on(on)
    default_facs = is_ext ?
        [:intensity, :technology, :final_demand] :
        [:technology, :final_demand]

    # Backward-compat path: no factors kw → classical :L / :Y keys
    use_legacy_keys = factors === nothing && !is_ext &&
                      method in (:additive, :multiplicative)
    facs = factors === nothing ? default_facs : collect(Symbol.(factors))
    isempty(facs) && throw(ArgumentError("factors must be non-empty"))
    _sda_validate_factors(facs, is_ext)

    if method == :multiplicative
        (is_ext || facs != [:technology, :final_demand]) && throw(ArgumentError(
            "method=:multiplicative is only implemented for the two-factor " *
            "output path factors=[:technology, :final_demand]"))
        return _sda_multiplicative_output(io0, io1; legacy_keys=use_legacy_keys)
    elseif method != :additive
        throw(ArgumentError("method must be :additive or :multiplicative"))
    end

    return _sda_additive(io0, io1, facs, on_key, is_ext;
                         legacy_keys=use_legacy_keys)
end

# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

function _sda_parse_on(on)
    if on === :output || on === "output"
        return :output, false
    elseif on isa Symbol
        return String(on), true
    elseif on isa AbstractString
        return String(on), true
    else
        throw(ArgumentError("on must be :output or an extension name; got $on"))
    end
end

const _SDA_TECH = (:technology,)
const _SDA_FD_TOTAL = (:final_demand, :fd)
const _SDA_FD_SPLIT = (:fd_level, :fd_mix, :fd_destination)
const _SDA_INTENSITY = (:intensity, :emission_intensity)

function _sda_validate_factors(facs::Vector{Symbol}, is_ext::Bool)
    allowed = Set{Symbol}([_SDA_TECH..., _SDA_FD_TOTAL..., _SDA_FD_SPLIT...,
                           _SDA_INTENSITY...])
    for f in facs
        f in allowed || throw(ArgumentError(
            "unknown SDA factor :$f; allowed: $(sort(collect(allowed)))"))
    end
    has_int = any(f -> f in _SDA_INTENSITY, facs)
    has_int && !is_ext && throw(ArgumentError(
        ":intensity requires on=<extension name>"))
    is_ext && !has_int && throw(ArgumentError(
        "emission SDA requires :intensity among factors"))
    !is_ext && has_int && throw(ArgumentError(
        ":intensity is only valid with an extension on=..."))

    has_total = any(f -> f in _SDA_FD_TOTAL, facs)
    has_split = any(f -> f in _SDA_FD_SPLIT, facs)
    has_total && has_split && throw(ArgumentError(
        "cannot combine :final_demand with :fd_level/:fd_mix/:fd_destination"))

    # Require technology for output/emission SDA
    any(f -> f in _SDA_TECH, facs) || throw(ArgumentError(
        "factors must include :technology"))

    # Destination requires mix (and level) for a coherent product
    if :fd_destination in facs
        (:fd_mix in facs && :fd_level in facs) || throw(ArgumentError(
            ":fd_destination requires :fd_level and :fd_mix"))
    end
    if :fd_mix in facs && !(:fd_level in facs)
        throw(ArgumentError(":fd_mix requires :fd_level"))
    end
    if :fd_level in facs && !(:fd_mix in facs)
        throw(ArgumentError(":fd_level requires :fd_mix (or use :final_demand)"))
    end

    length(unique(facs)) == length(facs) ||
        throw(ArgumentError("duplicate factors: $facs"))
    return nothing
end

# ---------------------------------------------------------------------------
# Factor extraction
# ---------------------------------------------------------------------------

"Total final-demand vector ``y = Y\\mathbf{1}``."
_sda_y(io::IOData) = vec(sum(io.Y, dims=2))

"""
Build the ordered factor state for one table.
Returns a Vector of Any (matrices / vectors / scalars) aligned with `facs`.
"""
function _sda_states(io::IOData{T}, facs::Vector{Symbol},
                     on_key, is_ext::Bool) where {T<:AbstractFloat}
    L = leontief_inverse(io)
    y = _sda_y(io)
    S = is_ext ? Matrix{T}(intensities(io, on_key)) : nothing

    # Precompute FD splits when needed
    g = sum(y)
    has_dest = :fd_destination in facs
    n_fd = size(io.Y, 2)

    states = Vector{Any}(undef, length(facs))
    for (k, f) in enumerate(facs)
        if f === :technology
            states[k] = L
        elseif f in _SDA_INTENSITY
            states[k] = S
        elseif f in _SDA_FD_TOTAL
            states[k] = y
        elseif f === :fd_level
            states[k] = g
        elseif f === :fd_mix && !has_dest
            # product composition (length-n shares)
            states[k] = g == zero(T) ? fill(zero(T), length(y)) : y ./ g
        elseif f === :fd_mix && has_dest
            # within-category product mix: columns of Y normalized to sum 1
            B = Matrix{T}(undef, size(io.Y)...)
            for j in 1:n_fd
                col = @view io.Y[:, j]
                s = sum(col)
                B[:, j] = s == zero(T) ? fill(zero(T), length(y)) : col ./ s
            end
            states[k] = B
        elseif f === :fd_destination
            # category shares of total FD
            colsums = vec(sum(io.Y, dims=1))
            states[k] = g == zero(T) ? fill(zero(T), n_fd) : colsums ./ g
        else
            throw(ArgumentError("unhandled factor :$f"))
        end
    end
    return states
end

# ---------------------------------------------------------------------------
# Evaluation: states → indicator vector
# ---------------------------------------------------------------------------

"""
Evaluate the indicator given an ordered factor state vector aligned with `facs`.
Returns a `Vector{Float64}` (length-n for output; length-k stressors for extension).
"""
function _sda_eval(facs::Vector{Symbol}, states::Vector{Any}, is_ext::Bool)
    # Collect named pieces
    L = nothing
    S = nothing
    y = nothing
    g = nothing
    m = nothing
    B = nothing
    d = nothing
    for (f, st) in zip(facs, states)
        if f === :technology
            L = st
        elseif f in _SDA_INTENSITY
            S = st
        elseif f in _SDA_FD_TOTAL
            y = st
        elseif f === :fd_level
            g = st
        elseif f === :fd_mix
            if st isa AbstractMatrix
                B = st
            else
                m = st
            end
        elseif f === :fd_destination
            d = st
        end
    end

    # Build final-demand vector
    if y === nothing
        if d !== nothing && B !== nothing && g !== nothing
            y = g .* (B * d)
        elseif m !== nothing && g !== nothing
            y = g .* m
        else
            throw(ArgumentError("cannot form final demand from factors $facs"))
        end
    end

    x = L * y                                    # n-vector gross output
    if !is_ext
        return Vector{Float64}(x)
    else
        # total stressor: e = S * x  (k-vector)
        return Vector{Float64}(vec(S * x))
    end
end

# ---------------------------------------------------------------------------
# Two-polar additive average (Dietzenbacher & Los)
# ---------------------------------------------------------------------------

"""
Dietzenbacher–Los two-polar additive decomposition of
``Δv = v(s¹) − v(s⁰)`` into per-factor effects.

Polar 1 (forward): start at period 0, switch factors 1…n to period 1 in order.
Polar 2 (reverse): start at period 1, switch factors n…1 to period 0 in reverse
(equivalently: start at 0, switch factors n…1 to period 1 in reverse order).
Average the two polar contributions.
"""
function _two_polar_additive(facs::Vector{Symbol},
                             states0::Vector{Any},
                             states1::Vector{Any},
                             is_ext::Bool)
    n = length(facs)
    v0 = _sda_eval(facs, states0, is_ext)
    v1 = _sda_eval(facs, states1, is_ext)
    total = v1 .- v0
    dim = length(total)

    # Polar 1: forward order 1 → n
    e1 = [zeros(Float64, dim) for _ in 1:n]
    cur = copy(states0)
    prev_val = v0
    for k in 1:n
        cur[k] = states1[k]
        new_val = _sda_eval(facs, cur, is_ext)
        e1[k] = new_val .- prev_val
        prev_val = new_val
    end

    # Polar 2: reverse order n → 1
    e2 = [zeros(Float64, dim) for _ in 1:n]
    cur = copy(states0)
    prev_val = v0
    for k in n:-1:1
        cur[k] = states1[k]
        new_val = _sda_eval(facs, cur, is_ext)
        e2[k] = new_val .- prev_val
        prev_val = new_val
    end

    effects = Dict{Symbol,Vector{Float64}}()
    for k in 1:n
        effects[facs[k]] = 0.5 .* (e1[k] .+ e2[k])
    end
    summed = reduce(.+, values(effects); init=zeros(Float64, dim))
    resid = total .- summed
    return effects, total, resid
end

function _sda_additive(io0::IOData, io1::IOData,
                       facs::Vector{Symbol}, on_key, is_ext::Bool;
                       legacy_keys::Bool=false)
    s0 = _sda_states(io0, facs, on_key, is_ext)
    s1 = _sda_states(io1, facs, on_key, is_ext)
    effects, total, resid = _two_polar_additive(facs, s0, s1, is_ext)

    if legacy_keys
        # Map :technology → :L, :final_demand → :Y for frozen API
        effects = Dict{Symbol,Vector{Float64}}(
            :L => effects[:technology],
            :Y => effects[:final_demand])
        out_facs = [:L, :Y]
    else
        out_facs = copy(facs)
    end
    return SDAResult(effects, total, resid, :additive, on_key, out_facs)
end

# ---------------------------------------------------------------------------
# Multiplicative two-factor (output only) — frozen path
# ---------------------------------------------------------------------------

function _sda_multiplicative_output(io0::IOData, io1::IOData; legacy_keys::Bool=true)
    L0 = leontief_inverse(io0); L1 = leontief_inverse(io1)
    y0 = _sda_y(io0); y1 = _sda_y(io1)
    x0 = L0 * y0; x1 = L1 * y1
    T = eltype(x0)
    epsT = eps(T)
    ratio = x1 ./ max.(x0, epsT)
    L_p1 = (L1 * y0) ./ max.(x0, epsT)
    Y_p1 = (L0 * y1) ./ max.(x0, epsT)
    L_p2 = (L1 * y1) ./ max.(L0 * y1, epsT)
    Y_p2 = (L1 * y1) ./ max.(L1 * y0, epsT)
    L_eff = sqrt.(max.(L_p1 .* L_p2, zero(T)))
    Y_eff = sqrt.(max.(Y_p1 .* Y_p2, zero(T)))
    if legacy_keys
        eff = Dict{Symbol,Vector{Float64}}(:L => Vector{Float64}(L_eff),
                                           :Y => Vector{Float64}(Y_eff))
        out_facs = [:L, :Y]
    else
        eff = Dict{Symbol,Vector{Float64}}(:technology => Vector{Float64}(L_eff),
                                           :final_demand => Vector{Float64}(Y_eff))
        out_facs = [:technology, :final_demand]
    end
    return SDAResult(eff, Vector{Float64}(ratio),
                     Vector{Float64}(ratio .- (L_eff .* Y_eff)),
                     :multiplicative, :output, out_facs)
end
