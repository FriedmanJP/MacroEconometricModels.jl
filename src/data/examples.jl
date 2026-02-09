"""
Example dataset loader for MacroEconometricModels.jl.

Loads built-in datasets stored as TOML files in the `data/` directory.
"""

using TOML

# Path to the data directory (repo root / data)
const _DATA_DIR = joinpath(dirname(dirname(@__DIR__)), "data")

# Available datasets
const _EXAMPLE_DATASETS = Dict{Symbol, String}(
    :fred_md => "fred_md.toml",
    :fred_qd => "fred_qd.toml",
)

"""
    load_example(name::Symbol) -> TimeSeriesData{Float64}

Load a built-in example dataset.

# Available Datasets
- `:fred_md` — FRED-MD Monthly Database, January 2026 vintage (126 variables × 804 months)
- `:fred_qd` — FRED-QD Quarterly Database, January 2026 vintage (245 variables × 268 quarters)

Returned `TimeSeriesData` includes variable names, transformation codes, frequency,
per-variable descriptions (via `vardesc`), dataset description (via `desc`), and
bibliographic references (via `refs`).

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
```
"""
function load_example(name::Symbol)
    haskey(_EXAMPLE_DATASETS, name) || throw(ArgumentError(
        "Unknown dataset :$name. Available: $(collect(keys(_EXAMPLE_DATASETS)))"))

    toml_file = joinpath(_DATA_DIR, _EXAMPLE_DATASETS[name])
    isfile(toml_file) || throw(ErrorException(
        "Dataset file not found: $toml_file"))

    d = TOML.parsefile(toml_file)

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

    # Frequency
    freq_str = meta["frequency"]
    freq = if freq_str == "Monthly"
        Monthly
    elseif freq_str == "Quarterly"
        Quarterly
    elseif freq_str == "Yearly"
        Yearly
    elseif freq_str == "Daily"
        Daily
    else
        Other
    end

    # Source refs
    sr = Symbol.(get(meta, "source_refs", String[]))

    # Variable descriptions
    vardesc_dict = Dict{String,String}(String(k) => String(v) for (k, v) in descs)

    # Description
    ds = get(meta, "desc", "")

    TimeSeriesData(mat;
                   varnames=varnames,
                   frequency=freq,
                   tcode=tcodes,
                   desc=ds,
                   vardesc=vardesc_dict,
                   source_refs=sr)
end
