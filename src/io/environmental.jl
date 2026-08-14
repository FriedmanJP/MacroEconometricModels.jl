# environmental.jl — satellite accounts, intensities, multipliers, footprints

"Consumption-based footprint of a stressor account: total (stressor × fd_cat)
and the per-sector contribution (stressor × sector)."
struct FootprintResult
    total::Matrix{Float64}        # stressor × fd_cat (consumption-based)
    by_sector::Matrix{Float64}    # stressor × sector
    stressors::Vector{String}
    name::String
end

"""
    add_extension!(io, name, F; stressors, unit, F_Y=nothing)

Attach a satellite account `name` with stressor flows `F` (`n_stressor × n`) to
`io`, computing intensities `S = F x̂⁻¹`. `F_Y` (`n_stressor × n_fd`) gives
direct stressor flows in final demand (defaults to zeros).

`unit` accepts a single string (applied to every stressor row) or a vector with
one entry per row; `stressors` accepts a vector of names, or a single string
when `F` has one row (#520).
"""
function add_extension!(io::IOData{T}, name::AbstractString, F::AbstractMatrix;
                        stressors, unit, F_Y=nothing) where {T}
    Fm = Matrix{T}(F)
    n_s = size(Fm, 1)
    size(Fm, 2) == length(io.x) ||
        throw(ArgumentError("F cols ($(size(Fm, 2))) must equal n=$(length(io.x))"))
    S = Fm * Diagonal(_invdiag(io.x))
    FYm = F_Y === nothing ? zeros(T, n_s, size(io.Y, 2)) : Matrix{T}(F_Y)
    # Accept a scalar unit string (broadcast to all stressors) or a per-row vector.
    # A bare `String` used to be iterated character-by-character → Char MethodError (#520).
    unit_vec = if unit isa AbstractString
        fill(String(unit), n_s)
    else
        u = collect(String.(unit))
        length(u) == n_s || throw(ArgumentError(
            "unit must be a string or a vector of $n_s strings (one per extension row); got length $(length(u))"))
        u
    end
    # Same scalar-vs-vector trap as `unit` (#520): a bare String broadcasts
    # character-by-character, so handle it explicitly.
    stress_vec = if stressors isa AbstractString
        n_s == 1 || throw(ArgumentError(
            "stressors must be a vector of $n_s names (one per extension row); " *
            "got the single string $(repr(stressors))"))
        [String(stressors)]
    else
        s = collect(String.(stressors))
        length(s) == n_s || throw(ArgumentError(
            "stressors must have length $n_s (one per extension row); got $(length(s))"))
        s
    end
    io.extensions[String(name)] =
        IOExtension{T}(Fm, FYm, S, stress_vec, unit_vec)
    io
end

_get_ext(io, name) = haskey(io.extensions, name) ? io.extensions[name] :
    throw(ArgumentError("no extension '$name'; available: $(collect(keys(io.extensions)))"))

"Per-unit-output intensities `S = F x̂⁻¹` of extension `name`."
intensities(io::IOData, name::AbstractString) = _get_ext(io, name).S

"Consumption-based emission multipliers `M = S L` of extension `name`."
emission_multipliers(io::IOData, name::AbstractString) =
    _get_ext(io, name).S * leontief_inverse(io)

"""
    footprint(io, name; by=:sector) -> FootprintResult
    footprint(io, name; by=:region) -> RegionalFootprintResult

Consumption-based account of extension `name`.

- `by=:sector` (default) — `total = M·Y + F_Y` (stressor × final-demand category)
  and `by_sector = M ⊙ y'` (stressor × sector), where `M = S·L` and `y` is total
  final demand. Returns [`FootprintResult`](@ref).
- `by=:region` — production-based vs consumption-based totals per region. Returns
  [`RegionalFootprintResult`](@ref) (defined in the MRIO trade-accounting layer).
"""
function footprint(io::IOData, name::AbstractString; by::Symbol=:sector)
    if by === :region
        return _footprint_by_region(io, name)
    elseif by !== :sector
        throw(ArgumentError("by must be :sector or :region; got :$by"))
    end
    ext = _get_ext(io, name)
    L = leontief_inverse(io)
    M = ext.S * L                                  # stressor × sector
    total = M * io.Y .+ ext.F_Y                    # stressor × fd_cat
    y = vec(sum(io.Y, dims=2))
    by_sector = M .* reshape(y, 1, :)              # stressor × sector contribution
    FootprintResult(total, by_sector, ext.stressors, String(name))
end
