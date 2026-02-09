"""
Panel data support for MacroEconometricModels.jl — Stata-style `xtset`,
balanced/unbalanced detection, group extraction.
"""

# =============================================================================
# xtset — construct PanelData from DataFrame
# =============================================================================

"""
    xtset(df::DataFrame, group_col::Symbol, time_col::Symbol;
          varnames=nothing, frequency=Other, tcode=nothing) -> PanelData{Float64}

Construct a `PanelData` container from a DataFrame, analogous to Stata's `xtset`.

Extracts all numeric columns (excluding `group_col` and `time_col`), sorts by
(group, time), validates no duplicate (group, time) pairs, and detects whether
the panel is balanced.

# Arguments
- `df::DataFrame` — input data
- `group_col::Symbol` — column name identifying groups (entities)
- `time_col::Symbol` — column name identifying time periods

# Keyword Arguments
- `varnames::Union{Vector{String},Nothing}` — override variable names (default: column names)
- `frequency::Frequency` — data frequency (default: `Other`)
- `tcode::Union{Vector{Int},Nothing}` — transformation codes per variable

# Examples
```julia
using DataFrames
df = DataFrame(id=repeat(1:3, inner=50), t=repeat(1:50, 3),
               x=randn(150), y=randn(150))
pd = xtset(df, :id, :t)
```
"""
function xtset(df::DataFrame, group_col::Symbol, time_col::Symbol;
               varnames::Union{Vector{String},Nothing}=nothing,
               frequency::Frequency=Other,
               tcode::Union{Vector{Int},Nothing}=nothing,
               desc::String="",
               vardesc::Union{Dict{String,String},Nothing}=nothing)
    hasproperty(df, group_col) || throw(ArgumentError("Column :$group_col not found in DataFrame"))
    hasproperty(df, time_col) || throw(ArgumentError("Column :$time_col not found in DataFrame"))

    # Sort by (group, time)
    sorted = sort(df, [group_col, time_col])

    # Extract numeric columns excluding group and time
    exclude = Set([string(group_col), string(time_col)])
    num_cols = [n for n in names(sorted)
                if n ∉ exclude && eltype(sorted[!, n]) <: Union{Missing, Number}]
    isempty(num_cols) && throw(ArgumentError("No numeric variable columns found"))

    n_vars = length(num_cols)
    T_total = nrow(sorted)

    # Build data matrix
    mat = Matrix{Float64}(undef, T_total, n_vars)
    for (j, col) in enumerate(num_cols)
        for i in 1:T_total
            v = sorted[i, col]
            mat[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end

    # Group and time IDs
    raw_groups = sorted[!, group_col]
    raw_times = sorted[!, time_col]

    # Map groups to integer IDs
    unique_groups = unique(raw_groups)
    group_map = Dict(g => i for (i, g) in enumerate(unique_groups))
    group_id = [group_map[g] for g in raw_groups]
    group_names = [string(g) for g in unique_groups]
    n_groups = length(unique_groups)

    # Map times to integer IDs
    if eltype(raw_times) <: Integer
        time_id = Int.(raw_times)
    else
        unique_times = sort(unique(raw_times))
        time_map = Dict(t => i for (i, t) in enumerate(unique_times))
        time_id = [time_map[t] for t in raw_times]
    end

    # Validate: no duplicate (group, time) pairs
    seen = Set{Tuple{Int,Int}}()
    for i in 1:T_total
        pair = (group_id[i], time_id[i])
        pair ∈ seen && throw(ArgumentError(
            "Duplicate (group, time) pair: group=$(group_names[group_id[i]]), time=$(time_id[i])"))
        push!(seen, pair)
    end

    # Detect balanced/unbalanced
    obs_per_group = [count(==(g), group_id) for g in 1:n_groups]
    balanced = all(==(obs_per_group[1]), obs_per_group)

    vn = something(varnames, num_cols)
    length(vn) != n_vars && throw(ArgumentError("varnames length must match n_vars"))
    tc = something(tcode, ones(Int, n_vars))
    length(tc) != n_vars && throw(ArgumentError("tcode length must match n_vars"))

    vd = something(vardesc, Dict{String,String}())
    PanelData{Float64}(mat, vn, frequency, tc, group_id, time_id,
                        group_names, n_groups, n_vars, T_total, balanced,
                        [desc], vd, Symbol[])
end

# =============================================================================
# Panel accessors
# =============================================================================

"""
    isbalanced(d::PanelData) -> Bool

Return `true` if all groups have the same number of observations.
"""
isbalanced(d::PanelData) = d.balanced

"""
    groups(d::PanelData) -> Vector{String}

Return the group names.
"""
groups(d::PanelData) = d.group_names

"""
    ngroups(d::PanelData) -> Int

Return the number of groups.
"""
ngroups(d::PanelData) = d.n_groups

"""
    group_data(d::PanelData, g) -> TimeSeriesData

Extract data for a single group as a `TimeSeriesData` container.

`g` can be an integer group index or a string group name.

# Examples
```julia
pd = xtset(df, :id, :t)
g1 = group_data(pd, 1)       # by index
g1 = group_data(pd, "1")     # by name
```
"""
function group_data(d::PanelData{T}, g::Int) where {T}
    1 <= g <= d.n_groups || throw(ArgumentError("Group index $g out of range 1:$(d.n_groups)"))
    mask = d.group_id .== g
    ti = d.time_id[mask]
    TimeSeriesData(d.data[mask, :];
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=copy(d.tcode),
                   time_index=ti,
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs))
end

function group_data(d::PanelData{T}, g::String) where {T}
    idx = findfirst(==(g), d.group_names)
    idx === nothing && throw(ArgumentError("Group '$g' not found. Available: $(d.group_names)"))
    group_data(d, idx)
end

# =============================================================================
# panel_summary
# =============================================================================

"""
    panel_summary(d::PanelData)

Display a summary table of the panel structure: number of groups, observations
per group (min, mean, max), and balance status.

# Examples
```julia
pd = xtset(df, :id, :t)
panel_summary(pd)
```
"""
function panel_summary(d::PanelData)
    panel_summary(stdout, d)
end

function panel_summary(io::IO, d::PanelData)
    obs_per_group = [count(==(g), d.group_id) for g in 1:d.n_groups]
    min_obs = minimum(obs_per_group)
    max_obs = maximum(obs_per_group)
    avg_obs = round(mean(obs_per_group), digits=1)

    println(io, "Panel Structure: $(d.n_groups) groups, $(d.T_obs) total observations")
    println(io, "  Balance: ", d.balanced ? "balanced" : "unbalanced")
    println(io, "  Obs/group: min=$min_obs, avg=$avg_obs, max=$max_obs")
    println(io, "  Variables: ", join(d.varnames, ", "))
    if d.frequency != Other
        println(io, "  Frequency: $(d.frequency)")
    end
end
