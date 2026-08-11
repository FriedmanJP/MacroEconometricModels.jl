# price.jl — Leontief cost-push price model (and Ghosh dual)

"""
    PriceModelResult

Output of [`price_model`](@ref): sectoral price responses to primary-cost shocks.

# Fields
- `dp` — ``n \\times 1`` price changes ``\\Delta p``
- `p` — ``n \\times 1`` new prices (base prices are ones in a value table)
- `dv` — ``n \\times 1`` effective primary-cost shock ``\\Delta v`` (value-added + tax)
- `mode` — `:leontief` (cost-push dual) or `:ghosh` (supply-side dual)
- `sectors` — sector labels
"""
struct PriceModelResult
    dp::Vector{Float64}
    p::Vector{Float64}
    dv::Vector{Float64}
    mode::Symbol
    sectors::Vector{String}
end

"""
    price_model(io; dva=nothing, dtax=nothing, mode=:leontief) -> PriceModelResult

Leontief **cost-push price model** (column orientation): pass-through of
primary-cost shocks into sectoral prices

```math
\\Delta p = (I - A')^{-1} \\Delta v
```

where ``A = Z\\hat{x}^{-1}`` is the technical-coefficients matrix and ``\\Delta v``
is the change in primary-input cost per unit of output. Base prices in a value
table satisfy ``p = (I - A')^{-1} v = \\mathbf{1}`` with
``v_j = (\\sum_k V_{kj}) / x_j``.

# Arguments
- `dva` — length-``n`` change in value-added coefficients, or a `Dict` mapping
  sector index/name → shock. Defaults to zeros.
- `dtax` — same shape; added to `dva` (e.g. a production-tax wedge).
- `mode` — `:leontief` (default) uses ``(I - A')^{-1}``; `:ghosh` uses the
  Ghosh dual ``(I - B)^{-1}`` (see note below).

# Orientation
Column orientation: ``A_{ij}`` = input of ``i`` per unit output of ``j``.
The dual multiplies on the left by ``(I - A')^{-1} = L'``.

!!! warning "Ghosh dual is descriptive"
    With `mode=:ghosh`, prices are obtained from the allocation coefficients
    ``B`` via ``\\Delta p = (I - B)^{-1}\\Delta v``. Dietzenbacher (1997) shows
    that the Ghosh system is a valid *price* model equivalent (under fixed
    quantities) to the Leontief dual; the *quantity* interpretation of Ghosh
    is the one Oosterhaven (1988) finds implausible. Prefer `:leontief` for
    cost-push analysis.
"""
function price_model(io::IOData;
                     dva=nothing, dtax=nothing, mode::Symbol=:leontief)
    n = length(io.x)
    dv = _price_shock_vector(io, dva, n) .+ _price_shock_vector(io, dtax, n)
    if mode === :leontief
        A = technical_coefficients(io)
        # Δp = (I − A')⁻¹ Δv  = L' Δv  (column dual)
        M = Matrix{Float64}(robust_inv(Matrix(I - A')))
        dp = M * dv
    elseif mode === :ghosh
        B = allocation_coefficients(io)
        # Descriptive Ghosh dual: Δp = (I − B)⁻¹ Δv
        M = Matrix{Float64}(robust_inv(Matrix(I - B)))
        dp = M * dv
    else
        throw(ArgumentError("mode must be :leontief or :ghosh; got :$mode"))
    end
    p = ones(Float64, n) .+ dp
    PriceModelResult(dp, p, dv, mode, copy(io.sectors))
end

# Resolve a shock specification into a length-n Float64 vector.
function _price_shock_vector(io::IOData, shock, n::Int)
    shock === nothing && return zeros(Float64, n)
    if shock isa AbstractDict
        v = zeros(Float64, n)
        for (k, val) in shock
            idx = _sector_indices(io, k)
            length(idx) == 1 || throw(ArgumentError(
                "price shock key must identify a single sector; got $k → $idx"))
            v[idx[1]] = Float64(val)
        end
        return v
    end
    v = Float64.(vec(collect(shock)))
    length(v) == n || throw(ArgumentError(
        "shock vector length $(length(v)) must equal n=$n"))
    return v
end
