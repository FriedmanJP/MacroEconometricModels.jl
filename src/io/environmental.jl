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
"""
function add_extension!(io::IOData{T}, name::AbstractString, F::AbstractMatrix;
                        stressors, unit, F_Y=nothing) where {T}
    Fm = Matrix{T}(F)
    size(Fm, 2) == length(io.x) ||
        throw(ArgumentError("F cols ($(size(Fm, 2))) must equal n=$(length(io.x))"))
    S = Fm * Diagonal(_invdiag(io.x))
    FYm = F_Y === nothing ? zeros(T, size(Fm, 1), size(io.Y, 2)) : Matrix{T}(F_Y)
    io.extensions[String(name)] =
        IOExtension{T}(Fm, FYm, S, collect(String.(stressors)), collect(String.(unit)))
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
    footprint(io, name) -> FootprintResult

Consumption-based account of extension `name`: `total = M·Y + F_Y` (stressor ×
final-demand category) and `by_sector = M ⊙ y'` (stressor × sector), where
`M = S·L` and `y` is total final demand.
"""
function footprint(io::IOData, name::AbstractString)
    ext = _get_ext(io, name)
    L = leontief_inverse(io)
    M = ext.S * L                                  # stressor × sector
    total = M * io.Y .+ ext.F_Y                    # stressor × fd_cat
    y = vec(sum(io.Y, dims=2))
    by_sector = M .* reshape(y, 1, :)              # stressor × sector contribution
    FootprintResult(total, by_sector, ext.stressors, String(name))
end
