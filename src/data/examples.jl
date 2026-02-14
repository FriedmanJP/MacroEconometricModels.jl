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
    else
        _load_timeseries_example(d)
    end
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
                        group_id, time_id, country_codes,
                        n_countries, n_vars, n_total, true,
                        [ds], vardesc_dict, sr)
end
