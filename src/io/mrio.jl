# mrio.jl — multi-regional IO trade accounting
#
# Orientation (package convention, column model):
#   A = Z x̂⁻¹,  x = L y,  A[i,j] = input of i per unit output of j.
# Block layout: region r occupies rows/cols (r-1)·n_s+1 : r·n_s.
# Final demand: when size(Y,2) is divisible by nregions and nregions > 1,
# Y columns are blocked by destination region in the same order as `regions`
# (each block has n_fd_r = size(Y,2) ÷ G final-demand categories).

# ══════════════════════════════════════════════════════════════════════════════
# Index helpers
# ══════════════════════════════════════════════════════════════════════════════

"""
    _ns_per_region(io) -> Int

Number of sectors per region (`nsectors`). Requires a balanced block layout
(`length(sectors) == nregions · nsectors`).
"""
function _ns_per_region(io::IOData)
    G = nregions(io)
    n = length(io.x)
    ns = nsectors(io)
    G * ns == n || throw(ArgumentError(
        "unbalanced MRIO layout: length(x)=$n, nregions=$G, nsectors=$ns " *
        "(expected n = G·nsectors)"))
    ns
end

"Industry indices of region `r` (1-based region index)."
function _region_range(io::IOData, r::Integer)
    ns = _ns_per_region(io)
    (r - 1) * ns + 1 : r * ns
end

"""
    region_indices(io, region) -> Vector{Int}

Industry indices belonging to `region` (name or 1-based index).
"""
function region_indices(io::IOData, region::AbstractString)
    _region_indices(io, region)
end
function region_indices(io::IOData, region::Integer)
    G = nregions(io)
    (1 <= region <= G) || throw(ArgumentError(
        "region index $region out of 1:$G"))
    collect(_region_range(io, Int(region)))
end

"""
    _resolve_region(io, region) -> Int

1-based region index from a name or integer.
"""
function _resolve_region(io::IOData, region::AbstractString)
    ridx = findfirst(==(region), io.regions)
    ridx === nothing && throw(ArgumentError(
        "region '$region' not found; available: $(io.regions)"))
    ridx
end
function _resolve_region(io::IOData, region::Integer)
    G = nregions(io)
    (1 <= region <= G) || throw(ArgumentError(
        "region index $region out of 1:$G"))
    Int(region)
end

"""
    _fd_layout(io) -> (n_fd_r, blocked)

Final-demand layout. When `nregions > 1` and `size(Y,2)` is divisible by
`nregions`, columns are treated as destination-region blocks of width
`n_fd_r = size(Y,2) ÷ G`. Otherwise `blocked=false` and all FD is treated as
having unknown destination (exports of final goods cannot be identified).
"""
function _fd_layout(io::IOData)
    G = nregions(io)
    n_fd = size(io.Y, 2)
    if G > 1 && n_fd > 0 && n_fd % G == 0
        return (n_fd ÷ G, true)
    end
    return (n_fd, false)
end

"Column range of final-demand destination region `r` (1-based), or `1:0` if unblocked."
function _fd_dest_range(io::IOData, r::Integer)
    n_fd_r, blocked = _fd_layout(io)
    blocked || return 1:0
    (r - 1) * n_fd_r + 1 : r * n_fd_r
end

"""
    _Y_block(io, s, r) -> Matrix

Final-demand block from producing region `s` to destination region `r`
(`n_s × n_fd_r`). When FD is unblocked, returns the full `Y[Is, :]` only for
`s == r` and zeros otherwise (all FD treated as domestic).
"""
function _Y_block(io::IOData{T}, s::Integer, r::Integer) where {T}
    Is = _region_range(io, s)
    n_fd_r, blocked = _fd_layout(io)
    if blocked
        Jr = _fd_dest_range(io, r)
        return Matrix{T}(io.Y[Is, Jr])
    else
        # No destination info: treat all FD of producers in s as domestic (s→s).
        return s == r ? Matrix{T}(io.Y[Is, :]) : zeros(T, length(Is), size(io.Y, 2))
    end
end

"""
    _Y_vec(io, s, r) -> Vector

Row-sum of `_Y_block` — final goods produced in `s` absorbed in `r` (N×1).
"""
function _Y_vec(io::IOData{T}, s::Integer, r::Integer) where {T}
    Yb = _Y_block(io, s, r)
    vec(sum(Yb; dims=2))
end

"""
    _va_coeffs(io) -> Vector

Direct value-added coefficients `v_j = (Σ_f va[f,j]) / x_j` (length n; 0 when x=0).
"""
function _va_coeffs(io::IOData{T}) where {T}
    va_tot = vec(sum(io.va; dims=1))
    n = length(io.x)
    v = Vector{T}(undef, n)
    @inbounds for j in 1:n
        xj = io.x[j]
        v[j] = xj == zero(T) ? zero(T) : va_tot[j] / xj
    end
    v
end

"""
    _sector_types(io) -> Vector{String}

Sector-type labels from the first region block (length `nsectors`).
"""
function _sector_types(io::IOData)
    ns = _ns_per_region(io)
    String[io.sectors[i] for i in 1:ns]
end

# ══════════════════════════════════════════════════════════════════════════════
# Region-block accessors & bilateral trade
# ══════════════════════════════════════════════════════════════════════════════

"""
    region_block(io, r, s) -> Matrix

Intermediate-flow block ``Z^{rs}`` from supplying region `r` to using region `s`
(`n_s × n_s`). Arguments accept region names or 1-based indices.

# Orientation
Column orientation: ``Z^{rs}[i,j]`` is the flow of sector ``i`` in region ``r``
used as intermediate input by sector ``j`` in region ``s``.
"""
function region_block(io::IOData{T}, r, s) where {T}
    ri = _resolve_region(io, r)
    si = _resolve_region(io, s)
    Ir = _region_range(io, ri)
    Is = _region_range(io, si)
    Matrix{T}(io.Z[Ir, Is])
end

"""
    bilateral_trade(io, exporter, importer; kind=:total) -> NamedTuple

Bilateral trade flows from `exporter` to `importer`.

| `kind` | Contents |
|--------|----------|
| `:intermediate` | Intermediate exports ``Z^{ex,im}`` only |
| `:final` | Final-demand exports ``Y^{ex→im}`` (requires region-blocked `Y`) |
| `:total` (default) | Intermediate + final |

Returns `(intermediate, final, total, by_sector)` where `by_sector` is the
N-vector of sectoral gross exports from exporter to importer, and the three
scalars are economy-wide totals.

# Orientation
Column orientation (package convention).
"""
function bilateral_trade(io::IOData{T}, exporter, importer;
                         kind::Symbol=:total) where {T}
    kind in (:total, :intermediate, :final) || throw(ArgumentError(
        "kind must be :total, :intermediate, or :final; got :$kind"))
    ri = _resolve_region(io, exporter)
    si = _resolve_region(io, importer)
    Zrs = region_block(io, ri, si)
    inter_by_sec = vec(sum(Zrs; dims=2))          # N×1 intermediate exports
    final_by_sec = _Y_vec(io, ri, si)             # N×1 final exports
    if kind === :intermediate
        by_sec = inter_by_sec
        fin = zero(T)
        inter = sum(inter_by_sec)
        tot = inter
    elseif kind === :final
        by_sec = final_by_sec
        inter = zero(T)
        fin = sum(final_by_sec)
        tot = fin
    else
        by_sec = inter_by_sec .+ final_by_sec
        inter = sum(inter_by_sec)
        fin = sum(final_by_sec)
        tot = inter + fin
    end
    (intermediate=Float64(inter), final=Float64(fin), total=Float64(tot),
     by_sector=Float64.(by_sec))
end

"""
    gross_exports(io, region) -> Vector

N-vector of sectoral gross exports of `region` to all other regions
(intermediate + final). For a single-region table returns zeros.
"""
function gross_exports(io::IOData{T}, region) where {T}
    ri = _resolve_region(io, region)
    G = nregions(io)
    ns = _ns_per_region(io)
    E = zeros(T, ns)
    for s in 1:G
        s == ri && continue
        bt = bilateral_trade(io, ri, s; kind=:total)
        E .+= T.(bt.by_sector)
    end
    E
end

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate sectors / regions
# ══════════════════════════════════════════════════════════════════════════════

"""
    aggregate(io; region_map=nothing, sector_map=nothing) -> IOData

Aggregate an [`IOData`](@ref) table over regions and/or sectors.

- `region_map` — `Dict` mapping each old region name → new region name.
  Regions that share a new name are summed. Unmapped regions keep their name.
- `sector_map` — `Dict` mapping each old *sector-type* name (labels of the first
  region block) → new sector-type name. Applied uniformly across all regions.

Extension matrices (`F`, `F_Y`) are aggregated along the industry axis (and
along destination-region FD blocks of `F_Y` when FD is region-blocked).
Accounting identities are re-checked after aggregation.

# Orientation
Column orientation. Aggregation sums flows; coefficients are *not* averaged.

# Examples
```julia
# Collapse two sectors into one across all regions
aggregate(io; sector_map=Dict("Agriculture" => "Primary",
                              "Mining" => "Primary"))
# Collapse EU members into one "EU" region
aggregate(io; region_map=Dict("DEU" => "EU", "FRA" => "EU", "ITA" => "EU"))
```
"""
function aggregate(io::IOData{T};
                   region_map=nothing,
                   sector_map=nothing) where {T}
    G = nregions(io)
    ns = _ns_per_region(io)
    n = length(io.x)
    sec_types = _sector_types(io)

    # ── region mapping ────────────────────────────────────────────────────
    rmap = Dict{String,String}()
    if region_map !== nothing
        for (k, v) in region_map
            rmap[String(k)] = String(v)
        end
    end
    old_to_new_r = Vector{String}(undef, G)
    for r in 1:G
        name = io.regions[r]
        old_to_new_r[r] = get(rmap, name, name)
    end
    new_regions = unique(old_to_new_r)
    G2 = length(new_regions)
    r_new_idx = Dict(new_regions[i] => i for i in 1:G2)
    r_of = [r_new_idx[old_to_new_r[r]] for r in 1:G]   # old r → new r index

    # ── sector-type mapping ───────────────────────────────────────────────
    smap = Dict{String,String}()
    if sector_map !== nothing
        for (k, v) in sector_map
            smap[String(k)] = String(v)
        end
    end
    old_to_new_s = Vector{String}(undef, ns)
    for j in 1:ns
        old_to_new_s[j] = get(smap, sec_types[j], sec_types[j])
    end
    new_sec_types = unique(old_to_new_s)
    ns2 = length(new_sec_types)
    s_new_idx = Dict(new_sec_types[i] => i for i in 1:ns2)
    s_of = [s_new_idx[old_to_new_s[j]] for j in 1:ns]  # old sector-type → new

    # No-op fast path
    if G2 == G && ns2 == ns && all(old_to_new_r[r] == io.regions[r] for r in 1:G) &&
       all(old_to_new_s[j] == sec_types[j] for j in 1:ns)
        return io
    end

    n2 = G2 * ns2

    # Map old industry index → new industry index
    function new_ind(old_i::Int)
        r_old = (old_i - 1) ÷ ns + 1
        s_old = (old_i - 1) % ns + 1
        r2 = r_of[r_old]
        s2 = s_of[s_old]
        (r2 - 1) * ns2 + s2
    end

    # ── aggregate Z, x ────────────────────────────────────────────────────
    Z2 = zeros(T, n2, n2)
    x2 = zeros(T, n2)
    @inbounds for i in 1:n, j in 1:n
        Z2[new_ind(i), new_ind(j)] += io.Z[i, j]
    end
    @inbounds for i in 1:n
        x2[new_ind(i)] += io.x[i]
    end

    # ── aggregate Y ───────────────────────────────────────────────────────
    n_fd = size(io.Y, 2)
    n_fd_r, blocked = _fd_layout(io)
    if blocked && G2 != G
        # Re-block FD by new destination regions: sum columns within each
        # old-dest-region that map to the same new region, preserving fd cats.
        n_fd2 = G2 * n_fd_r
        Y2 = zeros(T, n2, n_fd2)
        @inbounds for j in 1:n_fd
            r_old = (j - 1) ÷ n_fd_r + 1
            cat = (j - 1) % n_fd_r + 1
            r2 = r_of[r_old]
            j2 = (r2 - 1) * n_fd_r + cat
            for i in 1:n
                Y2[new_ind(i), j2] += io.Y[i, j]
            end
        end
        # FD category labels: keep per-region cat names from first block
        fd_cats2 = if length(io.fd_cats) == n_fd
            # Take first region's cat labels and tile
            base = [io.fd_cats[c] for c in 1:n_fd_r]
            # Strip a leading "Region_" prefix if present; else keep as-is
            vcat([base for _ in 1:G2]...)
        else
            ["fd$(j)" for j in 1:n_fd2]
        end
    else
        Y2 = zeros(T, n2, n_fd)
        @inbounds for i in 1:n, j in 1:n_fd
            Y2[new_ind(i), j] += io.Y[i, j]
        end
        fd_cats2 = copy(io.fd_cats)
    end

    # ── aggregate va ──────────────────────────────────────────────────────
    n_va = size(io.va, 1)
    va2 = zeros(T, n_va, n2)
    @inbounds for f in 1:n_va, j in 1:n
        va2[f, new_ind(j)] += io.va[f, j]
    end

    # ── sector labels for new table ───────────────────────────────────────
    # Tile new sector types across new regions; prefix with region when G2>1
    # only if the original labels were region-prefixed uniqueness-style.
    # Keep simple: for G2==1 use type names; for G2>1 use "region_type".
    secs2 = String[]
    for r in 1:G2
        for s in 1:ns2
            if G2 == 1
                push!(secs2, new_sec_types[s])
            else
                # Prefer original style: if first-region labels already unique
                # across the full table use "reg_sec"; else just type name
                # repeated (as in many MRIOs).
                push!(secs2, string(new_regions[r], "_", new_sec_types[s]))
            end
        end
    end

    out = IOData(Z2, Y2, va2;
                 sectors=secs2, regions=new_regions,
                 fd_cats=fd_cats2, va_cats=copy(io.va_cats),
                 unit=io.unit, year=io.year,
                 source=isempty(io.source) ? "aggregate" : io.source * " [aggregated]",
                 meta=io.meta, check=true)

    # Override x with the aggregated x (row-balance construction is fine, but
    # keep the summed x for exact mass conservation when check is tight).
    # IOData constructor recomputes x from Z+Y; that is the right mass.

    # ── aggregate extensions ──────────────────────────────────────────────
    for (name, ext) in io.extensions
        n_s = size(ext.F, 1)
        F2 = zeros(T, n_s, n2)
        @inbounds for k in 1:n_s, j in 1:n
            F2[k, new_ind(j)] += ext.F[k, j]
        end
        # F_Y
        n_fd_y = size(ext.F_Y, 2)
        if n_fd_y == n_fd && blocked && G2 != G
            FY2 = zeros(T, n_s, G2 * n_fd_r)
            @inbounds for j in 1:n_fd
                r_old = (j - 1) ÷ n_fd_r + 1
                cat = (j - 1) % n_fd_r + 1
                r2 = r_of[r_old]
                j2 = (r2 - 1) * n_fd_r + cat
                for k in 1:n_s
                    FY2[k, j2] += ext.F_Y[k, j]
                end
            end
        elseif n_fd_y == n_fd
            FY2 = zeros(T, n_s, n_fd)
            @inbounds for k in 1:n_s, j in 1:n_fd
                FY2[k, j] += ext.F_Y[k, j]
            end
        else
            FY2 = zeros(T, n_s, size(out.Y, 2))
        end
        S2 = F2 * Diagonal(_invdiag(out.x))
        out.extensions[name] = IOExtension{T}(F2, FY2, S2,
                                              copy(ext.stressors), copy(ext.unit))
    end
    out
end

# ══════════════════════════════════════════════════════════════════════════════
# Vertical specialization (Hummels–Ishii–Yi)
# ══════════════════════════════════════════════════════════════════════════════

"""
    VerticalSpecialization

Hummels–Ishii–Yi (2001) import content of exports, with the KWW (2014)
multi-country generalisation.

# Fields
- `vs` — foreign content in exports (scalar; HIY when G=1-import block, KWW eq. 38 when multi-country)
- `vs_share` — `vs / gross_exports`
- `vs1` — indirect VS: home intermediates embodied in foreign exports (KWW eq. 42)
- `domestic_content` — domestic content in exports (`gross_exports − vs`)
- `dc_share` — domestic content share
- `gross_exports` — total gross exports of the region
- `region` — region name
- `by_sector` — N-vector of sectoral foreign content in that sector's exports
"""
struct VerticalSpecialization
    vs::Float64
    vs_share::Float64
    vs1::Float64
    domestic_content::Float64
    dc_share::Float64
    gross_exports::Float64
    region::String
    by_sector::Vector{Float64}
end

"""
    vertical_specialization(io, region=nothing) -> VerticalSpecialization

Import content of exports (vertical specialization) for `region`.

For a multi-region table this is the KWW (2014) generalisation of Hummels,
Ishii & Yi (2001): foreign value-added *plus* pure foreign double counting in
the region's gross exports (KWW eq. 38). For a single-region table with an
import row absent from `Z`, returns zeros (no import block to measure).

When `region` is omitted and the table has one region, that region is used;
with multiple regions `region` is required.

# Orientation
Column orientation: ``A = Z\\hat{x}^{-1}``, Leontief ``B = (I-A)^{-1}``.
"""
function vertical_specialization(io::IOData{T}, region=nothing) where {T}
    G = nregions(io)
    if region === nothing
        G == 1 || throw(ArgumentError(
            "region is required when nregions=$(G) > 1"))
        region = io.regions[1]
    end
    ri = _resolve_region(io, region)
    rname = io.regions[ri]
    ns = _ns_per_region(io)
    E = gross_exports(io, ri)
    ge = sum(E)
    if ge == zero(T) || G == 1
        # Single-region: no foreign block in Z → VS = 0
        return VerticalSpecialization(0.0, 0.0, 0.0, Float64(ge),
                                      ge == 0 ? 0.0 : 1.0, Float64(ge),
                                      rname, zeros(Float64, ns))
    end

    A = technical_coefficients(io)
    B = leontief_inverse(io)                      # full (I−A)⁻¹
    v = _va_coeffs(io)

    # Foreign content in exports of s: Σ_{t≠s} v_t · B_{t s} · E_s  (KWW eq. 38)
    Is = _region_range(io, ri)
    vs_by_sec = zeros(T, ns)
    vs_total = zero(T)
    for t in 1:G
        t == ri && continue
        It = _region_range(io, t)
        # B_ts is B[It, Is]; v_t is v[It]
        # contribution: (v_t' * B_ts) * E   elementwise for by-sector of E
        vt = v[It]                                # N×1
        Bts = B[It, Is]                           # N×N
        # foreign VA content per unit of s-sector output destined for export:
        fva_per_out = vec(vt' * Bts)              # 1×N → N
        vs_by_sec .+= fva_per_out .* E
        vs_total += dot(fva_per_out, E)
    end

    # VS1: home intermediates used in foreign exports (KWW eq. 42 leading term)
    # VS1_s = v_s · Σ_{r≠s} B_sr · E_r
    vs1 = zero(T)
    vs_s = v[Is]
    for r in 1:G
        r == ri && continue
        Ir = _region_range(io, r)
        Er = gross_exports(io, r)
        Bsr = B[Is, Ir]
        vs1 += dot(vs_s, Bsr * Er)
    end

    dc = ge - vs_total
    VerticalSpecialization(Float64(vs_total),
                           ge == 0 ? 0.0 : Float64(vs_total / ge),
                           Float64(vs1),
                           Float64(dc),
                           ge == 0 ? 0.0 : Float64(dc / ge),
                           Float64(ge),
                           rname,
                           Float64.(vs_by_sec))
end

# ══════════════════════════════════════════════════════════════════════════════
# KWW (2014) gross-export decomposition
# ══════════════════════════════════════════════════════════════════════════════

"""
    ExportDecomposition

Koopman–Wang–Wei (2014) decomposition of a region's gross exports into four
aggregates that sum to gross exports:

| Aggregate | Meaning | KWW (36) terms |
|-----------|---------|----------------|
| `dva` | Domestic value-added absorbed abroad (value-added exports) | 1–3 |
| `rdv` | Domestic value-added that returns and is consumed at home | 4–5 |
| `fva` | Foreign value-added embodied in exports | 7–8 |
| `pdc` | Pure double counting (two-way intermediate trade) | 6, 9 |

# Fields
- `dva`, `rdv`, `fva`, `pdc` — scalars
- `gross_exports` — `dva + rdv + fva + pdc`
- `vax_ratio` — `dva / gross_exports` (Johnson–Noguera VAX ratio)
- `region` — region name
- `terms` — 9-vector of the individual KWW (36) terms (zeros when G < 3 for
  third-country routes that cannot arise)
- `by_sector` — `N × 4` matrix of (DVA, RDV, FVA, PDC) by sector of export
"""
struct ExportDecomposition
    dva::Float64
    rdv::Float64
    fva::Float64
    pdc::Float64
    gross_exports::Float64
    vax_ratio::Float64
    region::String
    terms::Vector{Float64}
    by_sector::Matrix{Float64}   # N × 4
    sectors::Vector{String}
end

"""
    export_decomposition(io, region=nothing) -> ExportDecomposition

Koopman–Wang–Wei (2014) nine-term decomposition of `region`'s gross exports,
aggregated to DVA / RDV / FVA / PDC.

Implements equation (36) of KWW (2014) for G countries and N sectors. The
four aggregates satisfy the adding-up identity

```math
\\mathrm{DVA} + \\mathrm{RDV} + \\mathrm{FVA} + \\mathrm{PDC}
= E^{s*}
```

When `region` is omitted and the table has one region, that region is used
(all components zero — a closed economy has no exports). With multiple
regions `region` is required.

# Orientation
Column orientation. ``A = Z\\hat{x}^{-1}``, ``B = (I-A)^{-1}``, ``V_s`` =
direct value-added coefficients of region ``s``. Final-demand destination
blocks of ``Y`` must be region-blocked (``n_{fd}`` divisible by ``G``) for
final-goods export routes to be identified; otherwise all final demand is
treated as domestic and final-export DVA/FVA terms are zero.
"""
function export_decomposition(io::IOData{T}, region=nothing) where {T}
    G = nregions(io)
    if region === nothing
        G == 1 || throw(ArgumentError(
            "region is required when nregions=$(G) > 1"))
        region = io.regions[1]
    end
    s = _resolve_region(io, region)
    rname = io.regions[s]
    ns = _ns_per_region(io)
    sec_types = _sector_types(io)

    if G == 1
        return ExportDecomposition(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rname,
                                   zeros(9), zeros(ns, 4), sec_types)
    end

    A = technical_coefficients(io)
    B = leontief_inverse(io)
    v = _va_coeffs(io)
    Is = _region_range(io, s)
    vs = v[Is]                                    # N-vector
    Es = gross_exports(io, s)                     # N-vector
    ge = sum(Es)

    # Local Leontief L_ss = (I − A_ss)⁻¹
    Ass = A[Is, Is]
    Lss = Matrix{T}(inv(I - Ass))

    # Precompute blocks
    # Y_sr[r] = final goods produced in s absorbed in r (N-vector)
    Ys = [_Y_vec(io, s, r) for r in 1:G]
    # For each country r: Y_rr, E_r, A_sr, A_rs, B_sr, B_ts, L_rr
    Yr = [_Y_vec(io, r, r) for r in 1:G]          # domestic final of r
    Er = [r == s ? Es : gross_exports(io, r) for r in 1:G]
    Lrr = Vector{Matrix{T}}(undef, G)
    for r in 1:G
        Ir = _region_range(io, r)
        Lrr[r] = Matrix{T}(inv(I - A[Ir, Ir]))
    end

    # Nine terms (scalars) and sectoral DVA/RDV/FVA/PDC attribution
    t = zeros(T, 9)
    # Sectoral attribution: charge each term to the exporting sector of s
    # via the last "touch" of Es / Ysr where possible.
    dva_sec = zeros(T, ns)
    rdv_sec = zeros(T, ns)
    fva_sec = zeros(T, ns)
    pdc_sec = zeros(T, ns)

    # ── Terms 1–3: DVA (value-added exports) ─────────────────────────────
    # (1) Vs Bss Ysr  (r≠s)
    Bss = B[Is, Is]
    for r in 1:G
        r == s && continue
        # contrib vector: diag(vs) * Bss * Ysr  — or elementwise
        # scalar = vs' * Bss * Ysr
        ysr = Ys[r]
        contrib = Bss * ysr                       # N-vector of s-output
        # VA content: vs ⊙ contrib, sum = term
        va_c = vs .* contrib
        t[1] += sum(va_c)
        dva_sec .+= va_c
    end

    # (2) Vs Bsr Yrr  (r≠s)
    for r in 1:G
        r == s && continue
        Ir = _region_range(io, r)
        Bsr = B[Is, Ir]
        yrr = Yr[r]
        contrib = Bsr * yrr
        va_c = vs .* contrib
        t[2] += sum(va_c)
        dva_sec .+= va_c
    end

    # (3) Vs Bsr Yrt  (r≠s, t≠s,r) — third-country absorption
    for r in 1:G
        r == s && continue
        Ir = _region_range(io, r)
        Bsr = B[Is, Ir]
        for tt in 1:G
            (tt == s || tt == r) && continue
            yrt = _Y_vec(io, r, tt)
            contrib = Bsr * yrt
            va_c = vs .* contrib
            t[3] += sum(va_c)
            dva_sec .+= va_c
        end
    end

    # ── Terms 4–5: RDV ───────────────────────────────────────────────────
    # (4) Vs Bsr Yrs  (r≠s) — returned via final imports
    for r in 1:G
        r == s && continue
        Ir = _region_range(io, r)
        Bsr = B[Is, Ir]
        yrs = _Y_vec(io, r, s)
        contrib = Bsr * yrs
        va_c = vs .* contrib
        t[4] += sum(va_c)
        rdv_sec .+= va_c
    end

    # (5) Vs Bsr Ars Lss Yss  (r≠s)
    yss = Ys[s]
    Lss_yss = Lss * yss
    for r in 1:G
        r == s && continue
        Ir = _region_range(io, r)
        Bsr = B[Is, Ir]
        Ars = A[Ir, Is]
        contrib = Bsr * (Ars * Lss_yss)
        va_c = vs .* contrib
        t[5] += sum(va_c)
        rdv_sec .+= va_c
    end

    # ── Term 6: pure double counted domestic ─────────────────────────────
    # (6) Vs Bsr Ars Lss Es  (r≠s)
    Lss_Es = Lss * Es
    for r in 1:G
        r == s && continue
        Ir = _region_range(io, r)
        Bsr = B[Is, Ir]
        Ars = A[Ir, Is]
        contrib = Bsr * (Ars * Lss_Es)
        va_c = vs .* contrib
        t[6] += sum(va_c)
        pdc_sec .+= va_c
    end

    # ── Terms 7–8: FVA ───────────────────────────────────────────────────
    # (7) Vt Bts Ysr  (t≠s, r≠s)
    for tt in 1:G
        tt == s && continue
        It = _region_range(io, tt)
        vt = v[It]
        Bts = B[It, Is]
        for r in 1:G
            r == s && continue
            ysr = Ys[r]
            # scalar = vt' * Bts * ysr
            # sectoral: attribute to s-export sectors via (Bts' * vt) ⊙ ysr
            fva_per = vec(vt' * Bts)              # N
            va_c = fva_per .* ysr
            t[7] += sum(va_c)
            fva_sec .+= va_c
        end
    end

    # (8) Vt Bts Asr Lrr Yrr  (t≠s, r≠s)
    for tt in 1:G
        tt == s && continue
        It = _region_range(io, tt)
        vt = v[It]
        Bts = B[It, Is]
        fva_per = vec(vt' * Bts)
        for r in 1:G
            r == s && continue
            Ir = _region_range(io, r)
            Asr = A[Is, Ir]
            mid = Asr * (Lrr[r] * Yr[r])          # N-vector of s-output
            va_c = fva_per .* mid
            t[8] += sum(va_c)
            fva_sec .+= va_c
        end
    end

    # ── Term 9: pure double counted foreign ──────────────────────────────
    # (9) Vt Bts Asr Lrr Er  (t≠s, r≠s)
    for tt in 1:G
        tt == s && continue
        It = _region_range(io, tt)
        vt = v[It]
        Bts = B[It, Is]
        fva_per = vec(vt' * Bts)
        for r in 1:G
            r == s && continue
            Ir = _region_range(io, r)
            Asr = A[Is, Ir]
            mid = Asr * (Lrr[r] * Er[r])
            va_c = fva_per .* mid
            t[9] += sum(va_c)
            pdc_sec .+= va_c
        end
    end

    dva = t[1] + t[2] + t[3]
    rdv = t[4] + t[5]
    fva = t[7] + t[8]
    pdc = t[6] + t[9]

    by_sec = hcat(Float64.(dva_sec), Float64.(rdv_sec),
                  Float64.(fva_sec), Float64.(pdc_sec))

    ExportDecomposition(Float64(dva), Float64(rdv), Float64(fva), Float64(pdc),
                        Float64(ge),
                        ge == 0 ? 0.0 : Float64(dva / ge),
                        rname, Float64.(t), by_sec, sec_types)
end

# ══════════════════════════════════════════════════════════════════════════════
# Regional footprints
# ══════════════════════════════════════════════════════════════════════════════

"""
    RegionalFootprintResult

Production-based vs consumption-based stressor accounts by region for one
satellite extension.

# Fields
- `production` — stressor × region territorial (direct) emissions
- `consumption` — stressor × region consumption-based footprint
  (``M · Y^{\\cdot r} + F_Y^{\\cdot r}``)
- `stressors`, `regions`, `name`
"""
struct RegionalFootprintResult
    production::Matrix{Float64}
    consumption::Matrix{Float64}
    stressors::Vector{String}
    regions::Vector{String}
    name::String
end

# `footprint(io, name; by=:region)` lives in environmental.jl and dispatches here.
function _footprint_by_region(io::IOData{T}, name::AbstractString) where {T}
    ext = _get_ext(io, name)
    G = nregions(io)
    n_s = size(ext.F, 1)
    L = leontief_inverse(io)
    M = ext.S * L                                 # stressor × industry

    production = zeros(Float64, n_s, G)
    consumption = zeros(Float64, n_s, G)

    for r in 1:G
        Ir = _region_range(io, r)
        # Production-based: direct F on industries of region r
        production[:, r] .= Float64.(vec(sum(ext.F[:, Ir]; dims=2)))

        # Consumption-based: M · (final demand absorbed in r) + F_Y for dest r
        y_r = zeros(T, length(io.x))
        n_fd_r, blocked = _fd_layout(io)
        if blocked
            Jr = _fd_dest_range(io, r)
            y_r .= vec(sum(io.Y[:, Jr]; dims=2))
            FY_r = vec(sum(ext.F_Y[:, Jr]; dims=2))
        else
            # Unblocked: assign all FD of producers in r as r's consumption
            # (no true destination split). Global sum still matches.
            y_r[Ir] .= vec(sum(io.Y[Ir, :]; dims=2))
            if size(ext.F_Y, 2) == size(io.Y, 2)
                # Split F_Y proportional to region's share of total FD mass
                # — for single-region this is exact; for multi without block
                # we put all F_Y on region 1 only if G>1? Better: split by
                # producer region's FD row mass.
                FY_r = zeros(T, n_s)
                # Approximate: attribute F_Y columns proportionally to each
                # region's share of total final demand.
                tot_fd = sum(io.Y)
                reg_fd = sum(io.Y[Ir, :])
                share = tot_fd == 0 ? zero(T) : reg_fd / tot_fd
                FY_r .= vec(sum(ext.F_Y; dims=2)) .* share
            else
                FY_r = zeros(T, n_s)
            end
        end
        consumption[:, r] .= Float64.(vec(M * y_r) .+ FY_r)
    end

    RegionalFootprintResult(production, consumption,
                            copy(ext.stressors), copy(io.regions),
                            String(name))
end
