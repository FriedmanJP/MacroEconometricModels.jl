# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Example dataset loader for MacroEconometricModels.jl.

Loads built-in datasets stored as TOML files in the `data/` directory.
"""

using TOML

# Path to the data directory (repo root / data)
const _DATA_DIR = joinpath(dirname(dirname(@__DIR__)), "data")

# Available datasets — maps name to (filename, type)
const _EXAMPLE_DATASETS = Dict{Symbol, Tuple{String, Symbol}}(
    :fred_md => ("fred_md.toml", :timeseries),
    :fred_qd => ("fred_qd.toml", :timeseries),
    :pwt     => ("pwt.toml",     :panel),
    :ddcg    => ("ddcg.toml",    :panel),
    :mpdta   => ("mpdta.toml",   :panel),
    :wiot    => ("wiot.toml",    :io),
    :nile    => ("nile.toml",    :timeseries),   # EV-13 (#421); shared with EV-37
    :mroz    => ("mroz.toml",    :crosssection), # EV-18 (#426): Mroz (1987) labor supply
)

# Parse frequency string to Frequency enum
function _parse_frequency(s::String)
    s == "Monthly"   && return Monthly
    s == "Quarterly" && return Quarterly
    s == "Yearly"    && return Yearly
    s == "Daily"     && return Daily
    return Other
end

"""
    load_example(name::Symbol) -> AbstractMacroData

Load a built-in example dataset.

# Available Datasets
- `:fred_md` — FRED-MD Monthly Database, January 2026 vintage (126 variables × 804 months) → `TimeSeriesData`
- `:fred_qd` — FRED-QD Quarterly Database, January 2026 vintage (245 variables × 268 quarters) → `TimeSeriesData`
- `:pwt` — Penn World Table 10.01, 38 OECD countries (42 variables × 74 years, 1950–2023) → `PanelData`
- `:ddcg` — Acemoglu et al. (2019) Democracy-GDP Panel (2 variables × 9,384 obs, 184 countries × 51 years) → `PanelData`
- `:mpdta` — Callaway & Sant'Anna (2021) Minimum Wage Panel (3 variables × 2,500 obs, 500 counties × 5 years) → `PanelData`
- `:nile` — Annual flow of the river Nile at Aswan, 1871–1970 (100 obs, 10^8 m^3) → `TimeSeriesData`
- `:mroz` — Mroz (1987) female labor-supply extract (753 obs × 22 vars; `lwage`/`wage` are `NaN`
  for the 325 non-participants) → `CrossSectionData`. Used for the Heckman selection model.

For time series datasets, the returned `TimeSeriesData` includes variable names,
transformation codes, frequency, per-variable descriptions (via `vardesc`),
dataset description (via `desc`), and bibliographic references (via `refs`).

For panel datasets, the returned `PanelData` includes country identifiers as
groups, year identifiers as time index, variable descriptions, and references.

# Examples
```julia
# Load FRED-MD
md = load_example(:fred_md)
nobs(md)       # 804
nvars(md)      # 126
desc(md)       # "FRED-MD Monthly Database, January 2026 Vintage (McCracken & Ng 2016)"
vardesc(md, "INDPRO")  # "IP Index"
refs(md)       # McCracken & Ng (2016)

# Apply recommended transformations
md_transformed = apply_tcode(md, md.tcode)

# Load FRED-QD
qd = load_example(:fred_qd)

# Load Penn World Table (panel data)
pwt = load_example(:pwt)
nobs(pwt)         # 2812 (38 countries × 74 years)
nvars(pwt)        # 42
ngroups(pwt)      # 38
groups(pwt)       # ["AUS", "AUT", ..., "USA"]
isbalanced(pwt)   # true
g = group_data(pwt, "USA")  # extract single country as TimeSeriesData
refs(pwt)         # Feenstra, Inklaar & Timmer (2015)
```
"""
function load_example(name::Symbol)
    haskey(_EXAMPLE_DATASETS, name) || throw(ArgumentError(
        "Unknown dataset :$name. Available: $(sort(collect(keys(_EXAMPLE_DATASETS))))"))

    filename, dtype = _EXAMPLE_DATASETS[name]
    toml_file = joinpath(_DATA_DIR, filename)
    isfile(toml_file) || throw(ErrorException(
        "Dataset file not found: $toml_file"))

    d = TOML.parsefile(toml_file)

    if dtype == :panel
        _load_panel_example(d)
    elseif dtype == :io
        _load_io_example(d)
    elseif dtype == :crosssection
        _load_crosssection_example(d)
    else
        _load_timeseries_example(d)
    end
end

# Build an (r×c) Float64 matrix from a TOML array-of-rows.
function _toml_matrix(rows, r::Int, c::Int)
    M = zeros(Float64, r, c)
    for i in 1:r, j in 1:c
        M[i, j] = Float64(rows[i][j])
    end
    M
end

# Load an Input-Output example (:wiot). Builds an `IOData` with value added,
# final demand, and any satellite extensions declared in the TOML.
function _load_io_example(d::Dict)
    meta = d["metadata"]; dims = d["dims"]; flows = d["flows"]
    secs = String.(dims["sectors"])
    n = length(secs)
    Z  = _toml_matrix(flows["Z"], n, n)
    Y  = _toml_matrix(flows["Y"], n, length(dims["fd_cats"]))
    va = _toml_matrix(flows["va"], length(dims["va_cats"]), n)
    io = IOData(Z, Y, va;
                sectors=secs, regions=String.(dims["regions"]),
                fd_cats=String.(dims["fd_cats"]), va_cats=String.(dims["va_cats"]),
                unit=get(meta, "unit", ""), year=get(meta, "year", nothing),
                source=get(meta, "source", ""))
    # Build IOExtension structs directly (avoids depending on add_extension!,
    # which is defined in a later-included file).
    T = eltype(io.x)
    for ext in get(d, "extensions", Any[])
        F = Matrix{T}(_toml_matrix(ext["F"], length(ext["stressors"]), n))
        S = F * Diagonal(_invdiag(io.x))
        io.extensions[String(ext["name"])] = IOExtension{T}(
            F, zeros(T, size(F, 1), size(io.Y, 2)), S,
            String.(ext["stressors"]), String.(ext["unit"]))
    end
    io
end

# Load a time series example (FRED-MD, FRED-QD)
function _load_timeseries_example(d::Dict)
    meta = d["metadata"]
    vars = d["variables"]
    descs = get(d, "descriptions", Dict{String,Any}())
    data_dict = d["data"]

    varnames = String.(vars["names"])
    tcodes = Int.(vars["tcodes"])
    n_vars = length(varnames)
    n_obs = meta["n_obs"]

    # Build data matrix column by column
    mat = Matrix{Float64}(undef, n_obs, n_vars)
    for (j, vn) in enumerate(varnames)
        col = data_dict[vn]
        for i in 1:n_obs
            mat[i, j] = Float64(col[i])
        end
    end

    freq = _parse_frequency(meta["frequency"])
    sr = Symbol.(get(meta, "source_refs", String[]))
    vardesc_dict = Dict{String,String}(String(k) => String(v) for (k, v) in descs)
    ds = get(meta, "desc", "")

    TimeSeriesData(mat;
                   varnames=varnames,
                   frequency=freq,
                   tcode=tcodes,
                   desc=ds,
                   vardesc=vardesc_dict,
                   source_refs=sr)
end

# Load a cross-sectional example (:mroz). Returns CrossSectionData; columns with
# missing values (e.g. lwage/wage for non-participants) carry NaN.
function _load_crosssection_example(d::Dict)
    meta = d["metadata"]
    vars = d["variables"]
    descs = get(d, "descriptions", Dict{String,Any}())
    data_dict = d["data"]

    varnames = String.(vars["names"])
    n_vars = length(varnames)
    n_obs = meta["n_obs"]

    mat = Matrix{Float64}(undef, n_obs, n_vars)
    for (j, vn) in enumerate(varnames)
        col = data_dict[vn]
        for i in 1:n_obs
            mat[i, j] = Float64(col[i])
        end
    end

    sr = Symbol.(get(meta, "source_refs", String[]))
    vardesc_dict = Dict{String,String}(String(k) => String(v) for (k, v) in descs)
    ds = get(meta, "desc", "")

    CrossSectionData(mat; varnames=varnames, desc=ds, vardesc=vardesc_dict,
                     source_refs=sr)
end

# Load a panel example (Penn World Table)
function _load_panel_example(d::Dict)
    meta = d["metadata"]
    vars = d["variables"]
    descs = get(d, "descriptions", Dict{String,Any}())
    countries = d["countries"]
    data_dict = d["data"]

    varnames = String.(vars["names"])
    n_vars = length(varnames)
    country_codes = String.(countries["codes"])
    country_names_raw = String.(countries["names"])
    n_countries = meta["n_countries"]
    n_years = meta["n_years"]
    years = Int.(meta["years"])
    n_total = n_countries * n_years

    # Build data matrix: rows are stacked (country_1 years, country_2 years, ...)
    mat = Matrix{Float64}(undef, n_total, n_vars)
    for (j, vn) in enumerate(varnames)
        col = data_dict[vn]
        for i in 1:n_total
            mat[i, j] = Float64(col[i])
        end
    end

    # Build group_id and time_id
    group_id = Vector{Int}(undef, n_total)
    time_id = Vector{Int}(undef, n_total)
    idx = 0
    for g in 1:n_countries
        for t in 1:n_years
            idx += 1
            group_id[idx] = g
            time_id[idx] = years[t]
        end
    end

    freq = _parse_frequency(meta["frequency"])
    sr = Symbol.(get(meta, "source_refs", String[]))
    vardesc_dict = Dict{String,String}(String(k) => String(v) for (k, v) in descs)
    ds = get(meta, "desc", "")

    PanelData{Float64}(mat, varnames, freq, ones(Int, n_vars),
                        group_id, time_id, nothing, country_codes,
                        n_countries, n_vars, n_total, true,
                        [ds], vardesc_dict, sr)
end
