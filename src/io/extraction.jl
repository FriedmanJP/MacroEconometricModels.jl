# extraction.jl — hypothetical extraction method

"Output loss from hypothetically removing one or more sectors."
struct ExtractionResult
    total_loss::Float64
    sector_loss::Vector{Float64}
    extracted::Vector{Int}
end

_sector_indices(io::IOData, s::Integer) = [Int(s)]
_sector_indices(io::IOData, s::AbstractVector{<:Integer}) = collect(Int, s)
function _sector_indices(io::IOData, s::AbstractString)
    idx = findfirst(==(s), io.sectors)
    idx === nothing && throw(ArgumentError("sector '$s' not found"))
    [idx]
end
_sector_indices(io::IOData, s::AbstractVector{<:AbstractString}) =
    reduce(vcat, _sector_indices.(Ref(io), s))

"""
    hypothetical_extraction(io, sectors) -> ExtractionResult

Total-output loss from removing `sectors` (an index, vector of indices, or
sector name(s)): zero the extracted rows and columns of `A` and the extracted
final demand, re-solve `x = (I−A)⁻¹ y`, and compare to the baseline.
"""
function hypothetical_extraction(io::IOData, sectors)
    idx = _sector_indices(io, sectors)
    A = technical_coefficients(io)
    y = vec(sum(io.Y, dims=2))
    x_base = (I - A) \ y
    Ae = copy(A); Ae[idx, :] .= 0.0; Ae[:, idx] .= 0.0
    ye = copy(y); ye[idx] .= 0.0
    x_red = (I - Ae) \ ye
    loss = x_base .- x_red
    ExtractionResult(sum(loss), loss, idx)
end
