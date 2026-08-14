# extraction.jl — hypothetical extraction method (Dietzenbacher–Lahr taxonomy)

"""
    ExtractionResult

Output of [`hypothetical_extraction`](@ref): total and per-sector gross-output
losses from removing (or partially removing) one or more sectors.

# Fields
- `total_loss` — economy-wide loss of gross output
- `sector_loss` — ``n \\times 1`` loss by sector
- `extracted` — indices of the extracted sectors
- `mode` — `:complete`, `:backward`, `:forward`, or `:partial`
- `share` — extraction intensity in ``(0, 1]`` (1 = full extraction)
- `loss_pct_go` — `total_loss` as a fraction of baseline gross output
- `loss_pct_gdp` — `total_loss` as a fraction of baseline GDP (total value added)
"""
struct ExtractionResult
    total_loss::Float64
    sector_loss::Vector{Float64}
    extracted::Vector{Int}
    mode::Symbol
    share::Float64
    loss_pct_go::Float64
    loss_pct_gdp::Float64
end

# ── sector / region index resolution ─────────────────────────────────────────

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
Indices of all industries belonging to `region` (MRIO block layout:
region ``r`` occupies rows/cols ``(r-1)n_s+1:r n_s``).
"""
function _region_indices(io::IOData, region::AbstractString)
    ridx = findfirst(==(region), io.regions)
    ridx === nothing && throw(ArgumentError(
        "region '$region' not found; available: $(io.regions)"))
    ns = nsectors(io)                              # sectors per region
    ns >= 1 || throw(ArgumentError("cannot resolve region blocks: nsectors=$ns"))
    collect((ridx - 1) * ns + 1 : ridx * ns)
end

"""
Resolve the extraction target: sector index/name(s), or a whole region via
the `region` keyword.
"""
function _extraction_target(io::IOData, sectors; region=nothing)
    if region !== nothing
        region isa AbstractString || throw(ArgumentError(
            "region must be a region name string; got $(typeof(region))"))
        return _region_indices(io, region)
    end
    # Allow a bare region name when it is not also a sector label.
    if sectors isa AbstractString && sectors in io.regions && !(sectors in io.sectors)
        return _region_indices(io, sectors)
    end
    return _sector_indices(io, sectors)
end

# ── main API ─────────────────────────────────────────────────────────────────

"""
    hypothetical_extraction(io, sectors; mode=:complete, share=1.0, region=nothing)
        -> ExtractionResult

Total-output loss from hypothetically removing (or partially removing)
`sectors` — an index, vector of indices, sector name(s), or a whole MRIO
`region`.

Follows the Dietzenbacher & Lahr (2013) taxonomy:

| `mode` | Action on the technical-coefficients matrix ``A`` |
|--------|-----------------------------------------------------|
| `:complete` (default) | Zero extracted rows **and** columns of ``A``; zero extracted final demand |
| `:backward` | Zero extracted **columns** of ``A`` only (sever purchases / backward linkages) |
| `:forward` | Zero extracted **rows** of ``A`` only (sever sales / forward linkages) |
| `:partial` | Scale the links that `:complete` would zero by ``(1 - \\mathrm{share})`` |

With `share < 1`, every mode scales the affected entries toward zero by
`share` rather than nullifying them fully (`share=1` recovers full extraction).

Reports losses in levels, as a share of baseline gross output, and as a share
of baseline GDP (total value added).

# Orientation
Column orientation: ``A = Z\\hat{x}^{-1}``, ``x = (I - A)^{-1} y``.

# Backward compatibility
`mode=:complete, share=1.0` (the defaults) reproduces the pre-IO2 complete
extraction exactly.
"""
function hypothetical_extraction(io::IOData, sectors;
                                 mode::Symbol=:complete,
                                 share::Real=1.0,
                                 region=nothing)
    (0.0 < share <= 1.0 + 1e-15) || throw(ArgumentError(
        "share must be in (0, 1]; got $share"))
    share_f = Float64(min(share, 1.0))
    idx = _extraction_target(io, sectors; region=region)
    isempty(idx) && throw(ArgumentError("no sectors to extract"))
    all(1 .<= idx .<= length(io.x)) || throw(ArgumentError(
        "extracted indices out of bounds: $idx"))

    A = technical_coefficients(io)
    y = vec(sum(io.Y, dims=2))
    x_base = (I - A) \ y

    Ae, ye = _extract_coefficients(A, y, idx, mode, share_f)
    x_red = (I - Ae) \ ye
    loss = x_base .- x_red
    total = sum(loss)
    go = sum(x_base)
    gdp = sum(io.va)
    ExtractionResult(Float64(total), Float64.(loss), idx, mode, share_f,
                     Float64(total / max(go, eps(Float64))),
                     Float64(total / max(gdp, eps(Float64))))
end

# Apply the Dietzenbacher–Lahr extraction to A and y.
function _extract_coefficients(A::AbstractMatrix, y::AbstractVector,
                               idx::AbstractVector{<:Integer},
                               mode::Symbol, share::Float64)
    Ae = float.(copy(A))
    ye = float.(copy(y))
    keep = 1.0 - share                             # residual weight on extracted links
    if mode === :complete || mode === :partial
        # Scale extracted rows AND columns of A, and extracted final demand.
        # Own-block A[idx,idx] must be scaled once (not keep²): stash, restore.
        own = copy(Ae[idx, idx])
        Ae[idx, :] .*= keep
        Ae[:, idx] .*= keep
        Ae[idx, idx] = keep .* own
        ye[idx] .*= keep
    elseif mode === :backward
        # Sever intermediate *purchases* of the extracted sectors (columns).
        Ae[:, idx] .*= keep
        # Final demand of extracted sectors is left intact (they still sell to FD).
    elseif mode === :forward
        # Sever intermediate *sales* of the extracted sectors (rows).
        Ae[idx, :] .*= keep
        # Also remove their final-demand deliveries when extracted.
        ye[idx] .*= keep
    else
        throw(ArgumentError(
            "mode must be :complete, :backward, :forward, or :partial; got :$mode"))
    end
    return Ae, ye
end
