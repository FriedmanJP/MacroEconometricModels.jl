# types.jl — Input-Output data container & metadata

"""
    IOExtension{T}

Satellite / extension account attached to an [`IOData`](@ref) table, e.g.
emissions or employment. `F` holds stressor flows (`n_stressor × n`), `F_Y`
the stressor content of final demand (`n_stressor × n_fd`, may be empty), and
`S = F x̂⁻¹` the per-unit-output intensities.
"""
struct IOExtension{T}
    F::Matrix{T}            # stressor flows           (n_stressor × n)
    F_Y::Matrix{T}          # stressor in final demand (n_stressor × n_fd), may be empty
    S::Matrix{T}            # intensities S = F x̂⁻¹    (n_stressor × n)
    stressors::Vector{String}
    unit::Vector{String}
end

"""
    IOMetaData

Provenance / download log for an IO table (analogue of pymrio's `MRIOMetaData`).
`files` maps each source `url => local filename`; `history` is a list of
timestamped log lines.
"""
mutable struct IOMetaData
    source::String
    version::String
    history::Vector{String}
    files::Vector{Pair{String,String}}   # url => local filename
end
IOMetaData(; source="", version="") =
    IOMetaData(source, version, String[], Pair{String,String}[])

"""
    IOData{T}

Input-Output table container.

- `Z`  — `n×n` intermediate-flow matrix (`n = regions·sectors`)
- `Y`  — `n × n_fd` final-demand matrix
- `va` — `n_va × n` value-added matrix
- `x`  — length-`n` gross output

It is MRIO-aware: `regions` has length 1 for a single-region table. Satellite
accounts live in `extensions`.

# Constructors
    IOData(Z, Y, va; sectors, regions, fd_cats, va_cats, kwargs...)
    IOData(Z, Y, x::AbstractVector; va=nothing, kwargs...)

The first form computes `x` from the row balance `x = rowsum(Z) + rowsum(Y)`;
the second takes `x` directly (and derives a single value-added row when `va`
is not supplied). When `check=true` (default) the accounting identities are
validated with a relaxed tolerance.
"""
struct IOData{T}
    Z::Matrix{T}
    Y::Matrix{T}
    va::Matrix{T}
    x::Vector{T}
    sectors::Vector{String}
    regions::Vector{String}
    fd_cats::Vector{String}
    va_cats::Vector{String}
    extensions::Dict{String,IOExtension{T}}
    unit::String
    year::Union{Int,Nothing}
    source::String
    meta::IOMetaData
end

"Guarded reciprocal of a vector (`0 ↦ 0`), used to form `x̂⁻¹`."
_invdiag(x::AbstractVector{T}) where {T} = T[xi == zero(T) ? zero(T) : one(T) / xi for xi in x]

function IOData(Z::AbstractMatrix, Y::AbstractMatrix, va::AbstractMatrix;
                sectors=String[], regions=["total"],
                fd_cats=String[], va_cats=String[],
                unit="", year=nothing, source="", meta=IOMetaData(),
                check::Bool=true)
    T = promote_type(eltype(Z), eltype(Y), eltype(va), Float64)
    Zm = Matrix{T}(Z); Ym = Matrix{T}(Y); Vm = Matrix{T}(va)
    x = vec(sum(Zm, dims=2)) .+ vec(sum(Ym, dims=2))
    _build_iodata(Zm, Ym, Vm, x, sectors, regions, fd_cats, va_cats,
                  unit, year, source, meta, check)
end

function IOData(Z::AbstractMatrix, Y::AbstractMatrix, x::AbstractVector;
                va=nothing, sectors=String[], regions=["total"],
                fd_cats=String[], va_cats=String[],
                unit="", year=nothing, source="", meta=IOMetaData(),
                check::Bool=true)
    T = promote_type(eltype(Z), eltype(Y), eltype(x), Float64)
    Zm = Matrix{T}(Z); Ym = Matrix{T}(Y); xv = Vector{T}(x)
    Vm = va === nothing ? reshape(xv .- vec(sum(Zm, dims=1)), 1, length(xv)) : Matrix{T}(va)
    _build_iodata(Zm, Ym, Vm, xv, sectors, regions, fd_cats, va_cats,
                  unit, year, source, meta, check)
end

function _build_iodata(Zm::Matrix{T}, Ym, Vm, x, sectors, regions, fd_cats, va_cats,
                       unit, year, source, meta, check) where {T}
    n = size(Zm, 1)
    size(Zm, 2) == n || throw(ArgumentError("Z must be square; got $(size(Zm))"))
    size(Ym, 1) == n || throw(ArgumentError("Y rows ($(size(Ym, 1))) must equal n=$n"))
    size(Vm, 2) == n || throw(ArgumentError("va cols ($(size(Vm, 2))) must equal n=$n"))
    secs = isempty(sectors) ? ["sector$(i)" for i in 1:n] : collect(String.(sectors))
    fds  = isempty(fd_cats) ? ["fd$(j)" for j in 1:size(Ym, 2)] : collect(String.(fd_cats))
    vas  = isempty(va_cats) ? ["va$(j)" for j in 1:size(Vm, 1)] : collect(String.(va_cats))
    if check
        rowbal = vec(sum(Zm, dims=2)) .+ vec(sum(Ym, dims=2))
        colbal = vec(sum(Zm, dims=1)) .+ vec(sum(Vm, dims=1))
        all(abs.(rowbal .- x) .<= 1e-6 .* max.(1.0, abs.(x))) ||
            throw(ArgumentError("row balance violated: x ≠ rowsum(Z)+rowsum(Y)"))
        all(abs.(colbal .- x) .<= 1e-6 .* max.(1.0, abs.(x))) ||
            throw(ArgumentError("column balance violated: x ≠ colsum(Z)+colsum(va)"))
    end
    IOData{T}(Zm, Ym, Vm, x, secs, collect(String.(regions)), fds, vas,
              Dict{String,IOExtension{T}}(), unit, year, source, meta)
end

"Number of sectors (rows divided by the number of regions)."
nsectors(io::IOData) = length(io.sectors) ÷ max(1, length(io.regions))
"Number of regions in the table (1 for single-region)."
nregions(io::IOData) = length(io.regions)
"Sector labels of the table."
labels(io::IOData) = io.sectors
Base.size(io::IOData) = (length(io.x), size(io.Y, 2))
