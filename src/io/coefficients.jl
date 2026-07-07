# coefficients.jl — technical/allocation coefficients & Leontief/Ghosh inverses

"Demand-driven (Leontief) representation of an IO table: `A`, `L = (I-A)⁻¹`, `x`."
struct LeontiefModel{T}
    A::Matrix{T}
    L::Matrix{T}
    x::Vector{T}
    io::IOData{T}
end

"Supply-driven (Ghosh) representation of an IO table: `B`, `G = (I-B)⁻¹`, `x`."
struct GhoshModel{T}
    B::Matrix{T}
    G::Matrix{T}
    x::Vector{T}
    io::IOData{T}
end

"""
    technical_coefficients(io) -> Matrix

Technical-coefficients matrix `A = Z x̂⁻¹` (input of `i` per unit output of `j`).
"""
technical_coefficients(io::IOData) = io.Z * Diagonal(_invdiag(io.x))

"""
    leontief_inverse(io) -> Matrix

Leontief inverse `L = (I − A)⁻¹` (total requirements matrix).
"""
function leontief_inverse(io::IOData{T}) where {T}
    A = technical_coefficients(io)
    Matrix{T}(inv(I - A))
end

"""
    allocation_coefficients(io) -> Matrix

Allocation-coefficients matrix `B = x̂⁻¹ Z` (Ghosh / supply-side).
"""
allocation_coefficients(io::IOData) = Diagonal(_invdiag(io.x)) * io.Z

"""
    ghosh_inverse(io) -> Matrix

Ghosh inverse `G = (I − B)⁻¹` (output-allocation requirements matrix).
"""
function ghosh_inverse(io::IOData{T}) where {T}
    B = allocation_coefficients(io)
    Matrix{T}(inv(I - B))
end

"""
    leontief(io) -> LeontiefModel

Build the demand-driven Leontief representation of `io`.
"""
function leontief(io::IOData{T}) where {T}
    A = technical_coefficients(io)
    L = Matrix{T}(inv(I - A))
    LeontiefModel{T}(A, L, copy(io.x), io)
end

"""
    ghosh(io) -> GhoshModel

Build the supply-driven Ghosh representation of `io`.
"""
function ghosh(io::IOData{T}) where {T}
    B = allocation_coefficients(io)
    G = Matrix{T}(inv(I - B))
    GhoshModel{T}(B, G, copy(io.x), io)
end
