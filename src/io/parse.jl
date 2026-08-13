# parse.jl — turn downloaded files into IOData (zip/xlsx via package extensions)
#
# Generic block parser (`parse_io`) plus source-specific recipes (`parse_icio`,
# `parse_wiod`) that recover labeled multi-region tables from OECD ICIO CSV/ZIP
# and WIOD 2013 WIOT xlsx layouts (pymrio-compatible).

"""
    parse_io(path; source, year=nothing, kwargs...) -> IOData

Parse a downloaded IO file into an [`IOData`](@ref). Dispatches on the file
extension: `.csv`/`.tsv`/`.txt` are parsed in-core (via `DelimitedFiles`), while
`.zip` and `.xlsx` require the optional `ZipFile` / `XLSX` packages (loaded as
package extensions) and raise an actionable error if those are not available.

For labeled OECD ICIO / WIOD recipes that recover regions, sectors, and final-
demand categories automatically, see [`parse_icio`](@ref) and
[`parse_wiod`](@ref).
"""
function parse_io(path::AbstractString; source::Symbol, year=nothing, kwargs...)
    ext = lowercase(splitext(path)[2])
    # Always forward provenance so the returned IOData carries source/year (#519).
    src = String(source)
    if ext in (".csv", ".tsv", ".txt")
        return _parse_csv_io(path; source=src, year=year, kwargs...)
    elseif ext == ".zip"
        return _parse_zip_io(path; source=src, year=year, kwargs...)
    elseif ext in (".xlsx", ".xls")
        return _parse_xlsx_io(path; source=src, year=year, kwargs...)
    else
        throw(ArgumentError("unsupported file type '$ext' for parse_io"))
    end
end

"""
    _parse_csv_io(path; n_sectors, n_fd=1, sectors=String[], delim=',',
                  source="", year=nothing) -> IOData

Parse a delimited IO block: the first `n_sectors` columns are the intermediate
matrix `Z`, the next `n_fd` columns are final demand. Tables are built with
`check=false` and a single derived value-added row from the column balance.
"""
function _parse_csv_io(path::AbstractString; n_sectors::Int, n_fd::Int=1,
                       sectors=String[], delim::AbstractChar=',',
                       source::AbstractString="", year=nothing)
    raw = readdlm(path, delim, Float64)
    Z = raw[1:n_sectors, 1:n_sectors]
    Y = raw[1:n_sectors, n_sectors+1:n_sectors+n_fd]
    IOData(Z, Y, vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2));
           sectors=sectors, source=String(source), year=year, check=false)
end

# Extension entry points — real methods live in ext/ and override these.
_parse_zip_io(path; kwargs...) =
    error("Parsing zipped IO archives requires the ZipFile package. " *
          "Run `]add ZipFile` and `using ZipFile` to enable it.")
_parse_xlsx_io(path; kwargs...) =
    error("Parsing Excel IO tables requires the XLSX package. " *
          "Run `]add XLSX` and `using XLSX` to enable it.")

# Read a zip member as text (ICIO labeled CSVs). Overridden by ZipFile extension.
_zip_member_text(path; member::AbstractString="", max_uncompressed::Integer=1_000_000_000) =
    error("Reading zip members requires the ZipFile package. " *
          "Run `]add ZipFile` and `using ZipFile` to enable it.")

# List member names in a zip archive. Overridden by ZipFile extension.
_zip_member_names(path) =
    error("Listing zip members requires the ZipFile package. " *
          "Run `]add ZipFile` and `using ZipFile` to enable it.")

# Raw used-range matrix from an Excel sheet (mixed Any cells). Overridden by XLSX.
_xlsx_sheet_matrix(path; sheet=1) =
    error("Reading Excel sheets requires the XLSX package. " *
          "Run `]add XLSX` and `using XLSX` to enable it.")


# ══════════════════════════════════════════════════════════════════════════════
# OECD ICIO recipe
# ══════════════════════════════════════════════════════════════════════════════

# Final-demand column tokens used by OECD ICIO releases (pymrio parity).
const _ICIO_FD_RE = r"HFCE|NPISH|NPS|GGFC|GFCF|INVNT|INV|DIRP|DPABR|FD|P33|DISC"
# Value-added / tax rows.
const _ICIO_VA_EXACT = Set(["TLS", "VA", "VALU", "TAXES", "TAXSUB"])
const _ICIO_VA_RE = r"VALU|TAX"
const _ICIO_TOTAL_COL = Set(["OUT", "TOTAL"])
const _ICIO_TOTAL_ROW = Set(["OUT", "OUTPUT"])

"""
    parse_icio(path; year=nothing, member="", aggregate_cn_mx=true,
               check=false, unit="Million USD") -> IOData

Parse an OECD Inter-Country Input-Output (ICIO) table into a labeled multi-region
[`IOData`](@ref).

Accepts a path to a `.csv` (or `.CSV`) file, or a `.zip` archive containing one
or more ICIO CSVs (requires `using ZipFile`). When `path` is a folder, `year`
must be given and the matching `ICIO*_YYYY.*` file is selected.

# Layout (pymrio / OECD convention)

- Row/column labels are `REGION_SECTOR` for intermediate flows (e.g. `AUS_D01T03`).
- Final-demand columns match `HFCE|NPISH|GGFC|GFCF|INVNT|…` (optionally prefixed
  or suffixed by the destination region).
- Value-added rows are `VA` / `TLS` / labels matching `VALU|TAX`.
- Totals `OUT` / `TOTAL` / `OUTPUT` are dropped.
- Optional CN1… / MX1… sub-national blocks are aggregated into `CHN` / `MEX`
  when `aggregate_cn_mx=true` (default; matches pymrio). Destination
  final-demand columns (`CN1_HFCE` + `CN2_HFCE` → `CHN_HFCE`, and likewise
  for MX) are collapsed in the same pass so `Y` stays destination-blocked.

Region order follows first appearance in the intermediate block; sector labels
on the returned table are the full `REGION_SECTOR` product (length
`nregions · nsectors_per_region`) so that [`region_block`](@ref) /
[`export_decomposition`](@ref) work out of the box. Final-demand columns are
kept in source order and labeled `REGION_CAT` (or `ALL_CAT` for residual
discrepancy columns).

# Arguments

- `path` — file or directory
- `year` — table year (metadata; also used to pick a member / file in a folder)
- `member` — zip member name (default: auto-select by `year` or first `.csv`)
- `aggregate_cn_mx` — collapse CN*/MX* subregions into CHN/MEX
- `check` — pass through to [`IOData`](@ref) balance check (default `false`;
  published ICIO tables rarely balance to machine precision)
- `unit` — stored on the result (default `"Million USD"`)

# Orientation

Column model (package convention): `Z[i,j]` is intermediate sales from industry
`i` to industry `j`; `Y` columns are final-demand destinations.
"""
function parse_icio(path::AbstractString; year=nothing, member::AbstractString="",
                    aggregate_cn_mx::Bool=true, check::Bool=false,
                    unit::AbstractString="Million USD")
    resolved, content, detected_year = _icio_resolve_source(path; year, member)
    yr = year === nothing ? detected_year : Int(year)
    io = _parse_icio_csv_text(content; year=yr, unit=unit, check=check,
                              aggregate_cn_mx=aggregate_cn_mx,
                              source_path=resolved)
    io
end

function _icio_resolve_source(path::AbstractString; year=nothing,
                              member::AbstractString="")
    path = abspath(path)
    if isdir(path)
        year === nothing && throw(ArgumentError(
            "parse_icio: directory given but year= is missing; " *
            "pass year or a specific file path"))
        cands = filter(f -> occursin(r"(?i)^icio" * string(year), f) ||
                            occursin(string(year), f),
                       readdir(path))
        cands = filter(f -> endswith(lowercase(f), ".csv") ||
                            endswith(lowercase(f), ".zip"), cands)
        isempty(cands) && throw(ArgumentError(
            "parse_icio: no ICIO file for year=$year under $path"))
        length(cands) > 1 && sort!(cands)
        return _icio_resolve_source(joinpath(path, first(cands)); year, member)
    end
    isfile(path) || throw(ArgumentError("parse_icio: file not found: $path"))
    ext = lowercase(splitext(path)[2])
    detected = _icio_year_from_name(basename(path))
    if ext in (".csv", ".txt")
        return path, read(path, String), something(year === nothing ? detected : Int(year), detected)
    elseif ext == ".zip"
        names = _zip_member_names(path)
        csv_members = filter(n -> endswith(lowercase(n), ".csv"), names)
        isempty(csv_members) && throw(ArgumentError(
            "parse_icio: no .csv member in zip $path; members=$(names)"))
        m = if !isempty(member)
            member in names || throw(ArgumentError(
                "parse_icio: member '$member' not found in $path; have $names"))
            member
        elseif year !== nothing
            ystr = string(year)
            hits = filter(n -> occursin(ystr, n), csv_members)
            isempty(hits) && throw(ArgumentError(
                "parse_icio: no zip member matching year=$year in $path; " *
                "csv members=$(csv_members)"))
            first(sort(hits))
        else
            first(sort(csv_members))
        end
        text = _zip_member_text(path; member=m)
        dy = something(year === nothing ? _icio_year_from_name(m) : Int(year),
                       detected)
        return path, text, dy
    else
        throw(ArgumentError(
            "parse_icio: unsupported extension '$ext' (want .csv or .zip)"))
    end
end

function _icio_year_from_name(name::AbstractString)
    # Prefer the *second* 4-digit token (ICIO2023_2016.csv → 2016); fall back to first.
    years = collect(eachmatch(r"(?<!\d)(\d{4})(?!\d)", name))
    isempty(years) && return nothing
    length(years) >= 2 && return parse(Int, years[2].captures[1])
    parse(Int, years[1].captures[1])
end

"""
    _parse_icio_csv_text(text; year, unit, check, aggregate_cn_mx, source_path)

Core OECD ICIO CSV parser (labels + numeric block). Public tests may call this
on synthetic fixtures without going through the filesystem.
"""
function _parse_icio_csv_text(text::AbstractString; year=nothing,
                              unit::AbstractString="Million USD",
                              check::Bool=false,
                              aggregate_cn_mx::Bool=true,
                              source_path::AbstractString="")
    rows = _icio_split_csv(text)
    length(rows) >= 2 || throw(ArgumentError(
        "parse_icio: need a header row and at least one data row"))
    header = rows[1]
    # First header cell is the blank corner / index name; remaining are column labels.
    col_labels = String[strip(string(h)) for h in header[2:end]]
    row_labels = String[]
    data_rows = Vector{Vector{Float64}}()
    for r in rows[2:end]
        isempty(r) && continue
        lab = strip(string(r[1]))
        isempty(lab) && continue
        push!(row_labels, lab)
        vals = Float64[]
        for j in 1:length(col_labels)
            v = j + 1 <= length(r) ? r[j + 1] : ""
            push!(vals, _icio_parse_float(v))
        end
        # Pad / trim to header width.
        while length(vals) < length(col_labels)
            push!(vals, 0.0)
        end
        length(vals) > length(col_labels) && resize!(vals, length(col_labels))
        push!(data_rows, vals)
    end
    isempty(data_rows) && throw(ArgumentError("parse_icio: no data rows"))
    M = reduce(vcat, (reshape(row, 1, :) for row in data_rows))

    # Drop total columns / rows.
    keep_c = [j for (j, c) in enumerate(col_labels) if !(c in _ICIO_TOTAL_COL)]
    keep_r = [i for (i, r) in enumerate(row_labels) if !(r in _ICIO_TOTAL_ROW)]
    col_labels = col_labels[keep_c]
    row_labels = row_labels[keep_r]
    M = M[keep_r, keep_c]

    is_va_row = [_icio_is_va_label(r) for r in row_labels]
    is_fd_col = [_icio_is_fd_label(c) for c in col_labels]
    ind_rows = findall(!, is_va_row)
    va_rows  = findall(identity, is_va_row)
    ind_cols = findall(!, is_fd_col)
    fd_cols  = findall(identity, is_fd_col)

    isempty(ind_rows) && throw(ArgumentError("parse_icio: no industry rows found"))
    isempty(ind_cols) && throw(ArgumentError("parse_icio: no industry columns found"))
    # Industry block must be square in the OECD layout.
    length(ind_rows) == length(ind_cols) || throw(ArgumentError(
        "parse_icio: industry block is not square " *
        "($(length(ind_rows)) rows × $(length(ind_cols)) cols)"))

    Z = Matrix{Float64}(M[ind_rows, ind_cols])
    Y = isempty(fd_cols) ? zeros(Float64, length(ind_rows), 0) :
        Matrix{Float64}(M[ind_rows, fd_cols])
    va = if isempty(va_rows)
        # Derive a single VA row from the column balance when the publisher
        # omitted factor-input rows.
        x_tmp = vec(sum(Z; dims=2)) .+ vec(sum(Y; dims=2))
        reshape(x_tmp .- vec(sum(Z; dims=1)), 1, length(x_tmp))
    else
        Matrix{Float64}(M[va_rows, ind_cols])
    end

    ind_labs = row_labels[ind_rows]
    # Prefer row labels; column industry labels should match.
    regions_raw = String[_icio_split_region_sector(lab)[1] for lab in ind_labs]
    regions = unique(regions_raw)
    # Sector product labels keep REGION_SECTOR for MRIO indexing.
    sector_labels = ind_labs

    fd_labs = isempty(fd_cols) ? String[] : col_labels[fd_cols]
    fd_cats = [_icio_fd_cat_label(c, regions) for c in fd_labs]
    va_cats = isempty(va_rows) ? ["VA"] : row_labels[va_rows]

    if aggregate_cn_mx
        Z, Y, va, sector_labels, regions, fd_cats =
            _icio_aggregate_cn_mx(Z, Y, va, sector_labels, regions, fd_cats)
    end

    yr = year === nothing ? nothing : Int(year)
    meta = IOMetaData(; source="OECD ICIO", version=yr === nothing ? "" : string(yr))
    if !isempty(source_path)
        push!(meta.history, string(Dates.now()) * ": parsed " * source_path)
    end
    IOData(Z, Y, va;
           sectors=sector_labels, regions=regions, fd_cats=fd_cats,
           va_cats=va_cats, unit=String(unit), year=yr, source="OECD ICIO",
           meta=meta, check=check)
end

function _icio_split_csv(text::AbstractString)
    rows = Vector{Vector{String}}()
    for line in split(text, r"\r\n|\n|\r")
        s = strip(line)
        isempty(s) && continue
        # OECD CSVs are comma-separated; fields are not quoted in practice.
        # Fall back to a light quote-aware split if a quote is present.
        if occursin('"', s)
            push!(rows, _icio_split_quoted(s))
        else
            push!(rows, String[strip(p) for p in split(s, ',')])
        end
    end
    rows
end

function _icio_split_quoted(s::AbstractString)
    out = String[]
    buf = IOBuffer()
    in_q = false
    for c in s
        if c == '"'
            in_q = !in_q
        elseif c == ',' && !in_q
            push!(out, strip(String(take!(buf))))
        else
            write(buf, c)
        end
    end
    push!(out, strip(String(take!(buf))))
    out
end

function _icio_parse_float(v)
    v isa Number && return Float64(v)
    s = strip(string(v))
    (isempty(s) || s == "NA" || s == "na" || s == "NaN") && return 0.0
    Float64(parse(Float64, s))
end

_icio_is_va_label(lab::AbstractString) =
    lab in _ICIO_VA_EXACT || occursin(_ICIO_VA_RE, lab)

_icio_is_fd_label(lab::AbstractString) = occursin(_ICIO_FD_RE, lab)

function _icio_split_region_sector(lab::AbstractString)
    # REGION_SECTOR — split on first underscore (sectors may contain underscores).
    i = findfirst('_', lab)
    i === nothing && return lab, lab
    lab[1:i-1], lab[i+1:end]
end

function _icio_fd_cat_label(lab::AbstractString, regions::AbstractVector{<:AbstractString})
    parts = split(lab, '_'; limit=2)
    if length(parts) == 1
        return "ALL_" * parts[1]
    end
    a, b = String(parts[1]), String(parts[2])
    # 2016 ICIO sometimes reverses to CAT_REGION; detect by membership.
    if a in regions
        return a * "_" * b
    elseif b in regions
        return b * "_" * a
    else
        return a * "_" * b
    end
end

"""
Aggregate CN1…CNk into CHN and MX1…MXk into MEX (pymrio parity). Operates on
the industry product labels `REGION_SECTOR`.
"""
function _icio_aggregate_cn_mx(Z::AbstractMatrix, Y::AbstractMatrix,
                               va::AbstractMatrix,
                               sector_labels::Vector{String},
                               regions::Vector{String},
                               fd_cats::Vector{String})
    function _parent(r)
        if occursin(r"^CN\d", r)
            return "CHN"
        elseif occursin(r"^MX\d", r)
            return "MEX"
        else
            return r
        end
    end
    parents = [_parent(r) for r in regions]
    parents == regions && return Z, Y, va, sector_labels, regions, fd_cats

    # Map each industry index → (parent region, sector type).
    n = length(sector_labels)
    reg_of = Vector{String}(undef, n)
    sec_of = Vector{String}(undef, n)
    for (i, lab) in enumerate(sector_labels)
        r, s = _icio_split_region_sector(lab)
        reg_of[i] = _parent(r)
        sec_of[i] = s
    end
    # New region order: unique parents in first-seen order.
    new_regions = unique(reg_of)
    # Sector types: unique in first-seen order across the table.
    sec_types = unique(sec_of)
    new_labels = String[r * "_" * s for r in new_regions for s in sec_types]
    n_new = length(new_labels)
    key_to_new = Dict{Tuple{String,String},Int}()
    for j in 1:n_new
        key_to_new[(_icio_split_region_sector(new_labels[j]))] = j
    end
    old_to_new = [key_to_new[(reg_of[i], sec_of[i])] for i in 1:n]

    T = promote_type(eltype(Z), Float64)
    Z2 = zeros(T, n_new, n_new)
    for i in 1:n, j in 1:n
        Z2[old_to_new[i], old_to_new[j]] += Z[i, j]
    end
    n_fd = size(Y, 2)
    n_va = size(va, 1)
    va2 = zeros(T, n_va, n_new)
    for k in 1:n_va, j in 1:n
        va2[k, old_to_new[j]] += va[k, j]
    end

    # Destination axis: rewrite CN*/MX* labels, then merge columns that share
    # a (parent region, FD category) key and emit them destination-blocked
    # (region order = `new_regions`, cats in first-seen order). Unprefixed
    # leftovers (DISC, …) are appended after the blocked block.
    Y2, fd2 = _icio_collapse_fd(Y, fd_cats, old_to_new, n_new, new_regions)
    return Z2, Y2, va2, new_labels, new_regions, fd2
end

"""Collapse FD columns after CN/MX industry aggregation so `Y` stays blocked."""
function _icio_collapse_fd(Y::AbstractMatrix, fd_cats::Vector{String},
                           old_to_new::Vector{Int}, n_new::Int,
                           new_regions::Vector{String})
    T = promote_type(eltype(Y), Float64)
    n = size(Y, 1)
    n_fd = size(Y, 2)
    n_fd == length(fd_cats) || throw(ArgumentError(
        "_icio_collapse_fd: fd_cats length $(length(fd_cats)) ≠ n_fd=$n_fd"))

    dests = Vector{String}(undef, n_fd)
    cats  = Vector{String}(undef, n_fd)
    for j in 1:n_fd
        dests[j], cats[j] = _icio_split_fd_label(
            _icio_rewrite_fd_region(fd_cats[j]), new_regions)
    end

    cat_order = String[]
    other_order = String[]
    for j in 1:n_fd
        if dests[j] == ""
            cats[j] in other_order || push!(other_order, cats[j])
        else
            cats[j] in cat_order || push!(cat_order, cats[j])
        end
    end

    fd2 = String[r * "_" * c for r in new_regions for c in cat_order]
    append!(fd2, other_order)
    n_fd2 = length(fd2)
    key_to_new = Dict{Tuple{String,String},Int}()
    for (j, lab) in enumerate(fd2)
        if j <= length(new_regions) * length(cat_order)
            ridx = (j - 1) ÷ max(length(cat_order), 1) + 1
            cidx = (j - 1) % max(length(cat_order), 1) + 1
            key_to_new[(new_regions[ridx], cat_order[cidx])] = j
        else
            key_to_new[("", lab)] = j
        end
    end

    Y2 = zeros(T, n_new, n_fd2)
    for i in 1:n, j in 1:n_fd
        key = dests[j] == "" ? ("", cats[j]) : (dests[j], cats[j])
        Y2[old_to_new[i], key_to_new[key]] += Y[i, j]
    end
    return Y2, fd2
end

"Split a rewritten FD label into `(destination, category)`; dest is `\"\"` if unknown."
function _icio_split_fd_label(lab::AbstractString, regions::Vector{String})
    parts = split(String(lab), '_'; limit=2)
    length(parts) < 2 && return "", String(lab)
    a, b = parts[1], parts[2]
    a in regions && return a, b
    b in regions && return b, a
    return "", String(lab)
end

function _icio_rewrite_fd_region(lab::AbstractString)
    parts = split(lab, '_'; limit=2)
    length(parts) < 2 && return lab
    r, rest = String(parts[1]), String(parts[2])
    if occursin(r"^CN\d", r)
        return "CHN_" * rest
    elseif occursin(r"^MX\d", r)
        return "MEX_" * rest
    else
        return lab
    end
end


# ══════════════════════════════════════════════════════════════════════════════
# WIOD 2013 WIOT recipe
# ══════════════════════════════════════════════════════════════════════════════

"""
    parse_wiod(path; year=nothing, sheet=1, last_interind_code="c35",
               check=false, names=:isic) -> IOData

Parse a WIOD 2013 release World Input-Output Table (WIOT) `.xlsx` into a labeled
multi-region [`IOData`](@ref). Requires `using XLSX`.

# Layout (WIOD November 2013 / pymrio convention)

The first sheet carries overlapping metadata and a four-row / four-column header:

| Header row (after dropping two blank top rows) | Content        |
|------------------------------------------------|----------------|
| 0                                              | ISIC / row code |
| 1                                              | sector name    |
| 2                                              | region         |
| 3                                              | c-code (`c1`…`c35` industries; `c37`+ final demand) |

Interindustry ends at the last column/row whose c-code equals
`last_interind_code` (default `"c35"` for the official 35-sector tables; use
`"c2"` etc. for reduced fixtures). Factor-input rows below the interindustry
block become `va` categories (totals `r60`/`r69` are dropped). The rightmost
total column is discarded.

When `path` is a directory, `year` selects `wiotYY*.xlsx` (two-digit year).

# Arguments

- `path` — `.xlsx` file or directory of WIOT files
- `year` — table year (metadata / file selection)
- `sheet` — worksheet index (default 1)
- `last_interind_code` — c-code marking the last intermediate sector
- `check` — [`IOData`](@ref) balance check (default `false`)
- `names` — `:isic` (default; row codes), `:full` (sector names), or `:c_codes`

# Orientation

Column model (package convention): `Z[i,j]` sales from `i` to `j`.
"""
function parse_wiod(path::AbstractString; year=nothing, sheet::Integer=1,
                    last_interind_code::AbstractString="c35",
                    check::Bool=false, names::Symbol=:isic)
    resolved, yr = _wiod_resolve_path(path; year)
    raw = _xlsx_sheet_matrix(resolved; sheet=Int(sheet))
    _parse_wiod_matrix(raw; year=yr, last_interind_code=String(last_interind_code),
                       check=check, names=names, source_path=resolved)
end

function _wiod_resolve_path(path::AbstractString; year=nothing)
    path = abspath(path)
    if isdir(path)
        year === nothing && throw(ArgumentError(
            "parse_wiod: directory given but year= is missing"))
        yy = lpad(string(Int(year) % 100), 2, '0')
        cands = filter(f -> startswith(lowercase(f), "wiot" * yy) &&
                            endswith(lowercase(f), ".xlsx"),
                       readdir(path))
        isempty(cands) && throw(ArgumentError(
            "parse_wiod: no wiot$(yy)*.xlsx for year=$year under $path"))
        return joinpath(path, first(sort(cands))), Int(year)
    end
    isfile(path) || throw(ArgumentError("parse_wiod: file not found: $path"))
    ext = lowercase(splitext(path)[2])
    ext in (".xlsx", ".xls") || throw(ArgumentError(
        "parse_wiod: unsupported extension '$ext' (want .xlsx)"))
    detected = _wiod_year_from_name(basename(path))
    yr = year === nothing ? detected : Int(year)
    return path, yr
end

function _wiod_year_from_name(name::AbstractString)
    # wiot09_row_apr12.xlsx → 2009; also accept four-digit years.
    m = match(r"wiot(\d{2})", lowercase(name))
    if m !== nothing
        yy = parse(Int, m.captures[1])
        return yy >= 50 ? 1900 + yy : 2000 + yy
    end
    m4 = match(r"(?<!\d)(19|20)\d{2}(?!\d)", name)
    m4 === nothing ? nothing : parse(Int, m4.match)
end

"""
    _parse_wiod_matrix(raw; year, last_interind_code, check, names, source_path)

Core WIOT sheet parser. `raw` is the used-range matrix (mixed `Any` cells) as
returned by `_xlsx_sheet_matrix`. Public tests may call this on synthetic
matrices without XLSX.
"""
function _parse_wiod_matrix(raw::AbstractMatrix; year=nothing,
                            last_interind_code::AbstractString="c35",
                            check::Bool=false, names::Symbol=:isic,
                            source_path::AbstractString="")
    A = _wiod_as_any_matrix(raw)
    nrows, ncols = size(A)
    nrows >= 8 && ncols >= 8 || throw(ArgumentError(
        "parse_wiod: sheet too small ($(nrows)×$(ncols)); expected WIOT layout"))

    # Meta from the original top-left before row drops (pymrio wiot_meta).
    year_cell = _wiod_cell_str(A[1, 1])
    unit_cell = nrows >= 4 ? _wiod_cell_str(A[4, 1]) : ""
    yr = year
    if yr === nothing
        m = match(r"(\d{4})", year_cell)
        yr = m === nothing ? nothing : parse(Int, m.captures[1])
    else
        yr = Int(yr)
    end
    unit = _wiod_strip_parens(unit_cell)
    isempty(unit) && (unit = "millions of US\$")

    # Blank out meta in column 1 for the first 5 rows, drop top two empty rows.
    for r in 1:min(5, nrows)
        A[r, 1] = missing
    end
    # Drop rows 1 and 2 (1-based) ≡ pymrio wiot_empty_top_rows [0,1].
    data = A[3:end, :]
    # Drop total column (last column).
    data = data[:, 1:end-1]
    nr, nc = size(data)

    # Header rows after the drop: code=1, sector_names=2, region=3, c_code=4 (1-based).
    h_code, h_name, h_region, h_ccode = 1, 2, 3, 4
    # Replace ROM → ROU (early WIOD years).
    for j in 1:nc
        if _wiod_cell_str(data[h_region, j]) == "ROM"
            data[h_region, j] = "ROU"
        end
    end
    for i in 1:nr
        if _wiod_cell_str(data[i, h_region]) == "ROM"
            data[i, h_region] = "ROU"
        end
    end

    # Last interindustry column / row: last index whose c_code == mark.
    mark = String(last_interind_code)
    last_z_col = 0
    for j in 1:nc
        if lowercase(_wiod_cell_str(data[h_ccode, j])) == lowercase(mark)
            last_z_col = j
        end
    end
    last_z_row = 0
    for i in 1:nr
        if lowercase(_wiod_cell_str(data[i, h_ccode])) == lowercase(mark)
            last_z_row = i
        end
    end
    last_z_col > 0 && last_z_row > 0 || throw(ArgumentError(
        "parse_wiod: last interindustry c-code '$mark' not found in header; " *
        "pass last_interind_code= for reduced tables"))
    last_z_col == last_z_row || throw(ArgumentError(
        "parse_wiod: interindustry block not square " *
        "(last col=$last_z_col, last row=$last_z_row)"))

    # Data origin after the 4 header rows / 4 label columns.
    d0 = 5
    d0 <= last_z_row || throw(ArgumentError("parse_wiod: no industry data rows"))
    z_idx = d0:last_z_row
    y_cols = (last_z_col + 1):nc
    # Factor-input rows below Z (skip totals r60/r69).
    fac_rows = Int[]
    for i in (last_z_row + 1):nr
        cc = lowercase(_wiod_cell_str(data[i, h_ccode]))
        cc in ("r60", "r69") && continue
        # Skip fully empty trailing rows.
        any(!_wiod_is_empty(data[i, j]) for j in d0:last_z_col) || continue
        push!(fac_rows, i)
    end

    n = length(z_idx)
    Z = Matrix{Float64}(undef, n, n)
    for (ii, i) in enumerate(z_idx), (jj, j) in enumerate(z_idx)
        Z[ii, jj] = _wiod_float(data[i, j])
    end
    n_fd = length(y_cols)
    Y = Matrix{Float64}(undef, n, n_fd)
    for (ii, i) in enumerate(z_idx), (jj, j) in enumerate(y_cols)
        Y[ii, jj] = _wiod_float(data[i, j])
    end
    if isempty(fac_rows)
        x_tmp = vec(sum(Z; dims=2)) .+ vec(sum(Y; dims=2))
        va = reshape(x_tmp .- vec(sum(Z; dims=1)), 1, n)
        va_cats = ["VA"]
    else
        va = Matrix{Float64}(undef, length(fac_rows), n)
        for (kk, i) in enumerate(fac_rows), (jj, j) in enumerate(z_idx)
            va[kk, jj] = _wiod_float(data[i, j])
        end
        va_cats = String[_wiod_cell_str(data[i, h_name]) for i in fac_rows]
        for (k, c) in enumerate(va_cats)
            isempty(c) && (va_cats[k] = "va$k")
        end
    end

    # Labels from the industry block.
    codes   = String[_wiod_cell_str(data[i, h_code]) for i in z_idx]
    secnames = String[_wiod_cell_str(data[i, h_name]) for i in z_idx]
    reg_row = String[_wiod_cell_str(data[i, h_region]) for i in z_idx]
    ccodes  = String[_wiod_cell_str(data[i, h_ccode]) for i in z_idx]
    # ROM fix already applied.
    regions = unique(reg_row)

    sec_lab = if names === :full
        secnames
    elseif names === :c_codes
        ccodes
    else
        # :isic default — use the code column; fall back to c_code if blank.
        String[isempty(codes[i]) ? ccodes[i] : codes[i] for i in 1:n]
    end
    # Product labels REGION_SECTOR for MRIO layout (one entry per industry row).
    # Prefer the raw region×sector-type form so nsectors = n / nregions works:
    # use the sector token that is unique within a region (c_code is ideal).
    sec_type = if names === :full
        # Full names can repeat wording; keep them but prefix region for uniqueness.
        secnames
    elseif names === :c_codes
        ccodes
    else
        String[isempty(codes[i]) ? ccodes[i] : codes[i] for i in 1:n]
    end
    sector_labels = String[reg_row[i] * "_" * sec_type[i] for i in 1:n]

    fd_regs = String[_wiod_cell_str(data[h_region, j]) for j in y_cols]
    fd_cc   = String[_wiod_cell_str(data[h_ccode, j]) for j in y_cols]
    fd_nm   = String[_wiod_cell_str(data[h_name, j]) for j in y_cols]
    fd_cats = if names === :full
        String[(isempty(fd_regs[j]) ? "ALL" : fd_regs[j]) * "_" *
               (isempty(fd_nm[j]) ? fd_cc[j] : fd_nm[j]) for j in 1:n_fd]
    else
        String[(isempty(fd_regs[j]) ? "ALL" : fd_regs[j]) * "_" *
               (isempty(fd_cc[j]) ? "fd$j" : fd_cc[j]) for j in 1:n_fd]
    end

    meta = IOMetaData(; source="WIOD 2013",
                      version=yr === nothing ? "2013" : string(yr))
    if !isempty(source_path)
        push!(meta.history, string(Dates.now()) * ": parsed " * source_path)
    end
    IOData(Z, Y, va;
           sectors=sector_labels, regions=regions, fd_cats=fd_cats,
           va_cats=va_cats, unit=unit, year=yr, source="WIOD 2013",
           meta=meta, check=check)
end

function _wiod_as_any_matrix(raw::AbstractMatrix)
    # Ensure a dense Array{Any} we can mutate.
    A = Array{Any}(undef, size(raw)...)
    for i in eachindex(raw)
        A[i] = raw[i]
    end
    A
end

_wiod_is_empty(x) = x === nothing || x === missing ||
    (x isa AbstractString && isempty(strip(x))) ||
    (x isa Number && isnan(Float64(x)))

function _wiod_cell_str(x)
    _wiod_is_empty(x) && return ""
    x isa AbstractString && return strip(x)
    x isa Symbol && return String(x)
    # Avoid "1.0" for integer-like floats in codes.
    if x isa Integer
        return string(x)
    elseif x isa Real
        v = Float64(x)
        return abs(v - round(v)) < 1e-12 ? string(Int(round(v))) : string(v)
    else
        return strip(string(x))
    end
end

function _wiod_float(x)
    _wiod_is_empty(x) && return 0.0
    x isa Number && return Float64(x)
    s = strip(string(x))
    (isempty(s) || s == "NA") && return 0.0
    Float64(parse(Float64, s))
end

function _wiod_strip_parens(s::AbstractString)
    t = strip(s)
    if startswith(t, "(") && endswith(t, ")")
        return strip(t[2:end-1])
    end
    t
end
