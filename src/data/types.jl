# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Data container types for MacroEconometricModels.jl — Frequency enum,
TimeSeriesData, PanelData, CrossSectionData with constructors and accessors.
"""

# =============================================================================
# Frequency Enum
# =============================================================================

"""
    Frequency

Data-frequency enumeration for [`TimeSeriesData`](@ref) and [`PanelData`](@ref) containers.

Used as informational metadata for dataset labeling, summary tables, and plot titles.
Estimation routines and filter bandwidths remain explicitly parameterized.

# Values
- [`Daily`](@ref): Daily observation frequency (business days or calendar days)
- [`Monthly`](@ref): Monthly observation frequency (e.g., FRED-MD)
- [`Quarterly`](@ref): Quarterly observation frequency (e.g., FRED-QD)
- [`Yearly`](@ref): Annual observation frequency (e.g., Penn World Table)
- [`Mixed`](@ref): Mixed-frequency data combining series with different sampling intervals
- [`Other`](@ref): Unspecified, irregular, or custom observation frequency (default)
"""
@enum Frequency begin
    Daily
    Monthly
    Quarterly
    Yearly
    Mixed
    Other
end

"""
    Daily

Daily observation frequency instance of [`Frequency`](@ref).

Used for daily time series and panel datasets (e.g., business-day financial series or
7-day calendar data).
"""
Daily

"""
    Monthly

Monthly observation frequency instance of [`Frequency`](@ref).

Used for monthly time series and panel datasets, such as the FRED-MD macroeconomic
database loaded via [`load_example`](@ref)`(:fred_md)`.
"""
Monthly

"""
    Quarterly

Quarterly observation frequency instance of [`Frequency`](@ref).

Used for quarterly time series and panel datasets, such as the FRED-QD macroeconomic
database loaded via [`load_example`](@ref)`(:fred_qd)`.
"""
Quarterly

"""
    Yearly

Annual (yearly) observation frequency instance of [`Frequency`](@ref).

Used for annual time series and panel datasets, such as the Penn World Table loaded
via [`load_example`](@ref)`(:pwt)` or the Nile river flow series loaded via
[`load_example`](@ref)`(:nile)`.
"""
Yearly

"""
    Mixed

Mixed observation frequency instance of [`Frequency`](@ref).

Used for datasets combining series observed at different sampling rates (e.g., monthly
indicators paired with quarterly aggregates in mixed-frequency VARs or MIDAS regressions).
"""
Mixed

"""
    Other

Unspecified, irregular, or custom observation frequency instance of [`Frequency`](@ref).

Default frequency in [`TimeSeriesData`](@ref) and [`PanelData`](@ref) constructors when
no explicit frequency is specified.
"""
Other


# =============================================================================
# TimeSeriesData
# =============================================================================

"""
    TimeSeriesData{T<:AbstractFloat} <: AbstractMacroData

Container for time series data with metadata.

# Fields
- `data::Matrix{T}` — T_obs × n_vars data matrix
- `varnames::Vector{String}` — variable names
- `frequency::Frequency` — data frequency
- `tcode::Vector{Int}` — FRED transformation codes per variable (default: all 1 = levels)
- `time_index::Vector{Int}` — integer time identifiers (default: 1:T)
- `T_obs::Int` — number of observations
- `n_vars::Int` — number of variables
- `desc::Vector{String}` — dataset description (length-1 vector for mutability)
- `vardesc::Dict{String,String}` — per-variable descriptions keyed by variable name
- `source_refs::Vector{Symbol}` — reference keys for bibliographic citations (see `refs()`)
- `dates::Vector{String}` — date labels (default: empty; use `set_dates!` to populate)

# Constructors
```julia
TimeSeriesData(data::Matrix; varnames, frequency=Other, tcode, time_index, desc, vardesc, source_refs, dates)
TimeSeriesData(data::Vector; varname="x1", frequency=Other, desc, vardesc, source_refs, dates)
TimeSeriesData(df::DataFrame; frequency=Other, varnames, desc, vardesc, source_refs, dates)
```
"""
struct TimeSeriesData{T<:AbstractFloat} <: AbstractMacroData
    data::Matrix{T}
    varnames::Vector{String}
    frequency::Frequency
    tcode::Vector{Int}
    time_index::Vector{Int}
    T_obs::Int
    n_vars::Int
    desc::Vector{String}
    vardesc::Dict{String,String}
    source_refs::Vector{Symbol}
    dates::Vector{String}
end

# Matrix constructor
function TimeSeriesData(data::AbstractMatrix{T};
                        varnames::Union{Vector{String},Nothing}=nothing,
                        frequency::Frequency=Other,
                        tcode::Union{Vector{Int},Nothing}=nothing,
                        time_index::Union{Vector{Int},Nothing}=nothing,
                        desc::String="",
                        vardesc::Union{Dict{String,String},Nothing}=nothing,
                        source_refs::Vector{Symbol}=Symbol[],
                        dates::Union{Vector{String},Nothing}=nothing) where {T<:AbstractFloat}
    T_obs, n_vars = size(data)
    T_obs < 1 && throw(ArgumentError("Data must have at least 1 observation, got $T_obs"))
    n_vars < 1 && throw(ArgumentError("Data must have at least 1 variable, got $n_vars"))
    vn = something(varnames, ["x$i" for i in 1:n_vars])
    length(vn) != n_vars && throw(ArgumentError("varnames length ($(length(vn))) must match n_vars ($n_vars)"))
    tc = something(tcode, ones(Int, n_vars))
    length(tc) != n_vars && throw(ArgumentError("tcode length ($(length(tc))) must match n_vars ($n_vars)"))
    all(t -> 1 <= t <= 7, tc) || throw(ArgumentError("tcode values must be in 1:7"))
    ti = something(time_index, collect(1:T_obs))
    length(ti) != T_obs && throw(ArgumentError("time_index length ($(length(ti))) must match T_obs ($T_obs)"))
    dt = something(dates, String[])
    !isempty(dt) && length(dt) != T_obs && throw(ArgumentError("dates length ($(length(dt))) must match T_obs ($T_obs)"))
    vd = something(vardesc, Dict{String,String}())
    TimeSeriesData{T}(Matrix{T}(data), vn, frequency, tc, ti, T_obs, n_vars, [desc], vd, source_refs, dt)
end

# Non-float matrix fallback
function TimeSeriesData(data::AbstractMatrix; kwargs...)
    TimeSeriesData(Float64.(data); kwargs...)
end

# Vector constructor (univariate)
function TimeSeriesData(data::AbstractVector{T};
                        varname::String="x1",
                        frequency::Frequency=Other,
                        tcode::Int=1,
                        time_index::Union{Vector{Int},Nothing}=nothing,
                        desc::String="",
                        vardesc::Union{Dict{String,String},Nothing}=nothing,
                        source_refs::Vector{Symbol}=Symbol[],
                        dates::Union{Vector{String},Nothing}=nothing) where {T<:AbstractFloat}
    TimeSeriesData(reshape(data, :, 1); varnames=[varname], frequency=frequency,
                   tcode=[tcode], time_index=time_index, desc=desc, vardesc=vardesc,
                   source_refs=source_refs, dates=dates)
end

# Non-float vector fallback
function TimeSeriesData(data::AbstractVector; kwargs...)
    TimeSeriesData(Float64.(data); kwargs...)
end

# DataFrame constructor
function TimeSeriesData(df::DataFrame;
                        frequency::Frequency=Other,
                        varnames::Union{Vector{String},Nothing}=nothing,
                        tcode::Union{Vector{Int},Nothing}=nothing,
                        time_index::Union{Vector{Int},Nothing}=nothing,
                        desc::String="",
                        vardesc::Union{Dict{String,String},Nothing}=nothing,
                        source_refs::Vector{Symbol}=Symbol[],
                        dates::Union{Vector{String},Nothing}=nothing)
    # Select numeric columns
    num_cols = [n for n in names(df) if eltype(df[!, n]) <: Union{Missing, Number}]
    isempty(num_cols) && throw(ArgumentError("DataFrame has no numeric columns"))

    # Convert to matrix, replacing missing with NaN
    mat = Matrix{Float64}(undef, nrow(df), length(num_cols))
    for (j, col) in enumerate(num_cols)
        for i in 1:nrow(df)
            v = df[i, col]
            mat[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end

    vn = something(varnames, num_cols)
    TimeSeriesData(mat; varnames=vn, frequency=frequency, tcode=tcode,
                   time_index=time_index, desc=desc, vardesc=vardesc,
                   source_refs=source_refs, dates=dates)
end

# =============================================================================
# PanelData
# =============================================================================

"""
    PanelData{T<:AbstractFloat} <: AbstractMacroData

Container for panel (longitudinal) data with group and time identifiers.

# Fields
- `data::Matrix{T}` — stacked data matrix (sum of obs across groups × n_vars)
- `varnames::Vector{String}` — variable names
- `frequency::Frequency` — data frequency
- `tcode::Vector{Int}` — FRED transformation codes per variable
- `group_id::Vector{Int}` — group identifier per row
- `time_id::Vector{Int}` — time identifier per row
- `cohort_id::Union{Vector{Int}, Nothing}` — treatment cohort per row (`nothing` if unused)
- `group_names::Vector{String}` — unique group labels
- `n_groups::Int` — number of groups
- `n_vars::Int` — number of variables
- `T_obs::Int` — total number of rows
- `balanced::Bool` — true if all groups have same number of observations
- `desc::Vector{String}` — dataset description (length-1 vector for mutability)
- `vardesc::Dict{String,String}` — per-variable descriptions keyed by variable name
- `source_refs::Vector{Symbol}` — reference keys for bibliographic citations (see `refs()`)
"""
struct PanelData{T<:AbstractFloat} <: AbstractMacroData
    data::Matrix{T}
    varnames::Vector{String}
    frequency::Frequency
    tcode::Vector{Int}
    group_id::Vector{Int}
    time_id::Vector{Int}
    cohort_id::Union{Vector{Int}, Nothing}
    group_names::Vector{String}
    n_groups::Int
    n_vars::Int
    T_obs::Int
    balanced::Bool
    desc::Vector{String}
    vardesc::Dict{String,String}
    source_refs::Vector{Symbol}

    function PanelData{T}(data, varnames, frequency, tcode, group_id, time_id,
                          cohort_id, group_names, n_groups, n_vars, T_obs, balanced,
                          desc, vardesc, source_refs) where {T<:AbstractFloat}
        all(t -> 1 <= t <= 7, tcode) || throw(ArgumentError("tcode values must be in 1:7"))
        new{T}(data, varnames, frequency, tcode, group_id, time_id,
               cohort_id, group_names, n_groups, n_vars, T_obs, balanced,
               desc, vardesc, source_refs)
    end
end

# =============================================================================
# CrossSectionData
# =============================================================================

"""
    CrossSectionData{T<:AbstractFloat} <: AbstractMacroData

Container for cross-sectional data (single time point, multiple observations).

# Fields
- `data::Matrix{T}` — N_obs × n_vars data matrix
- `varnames::Vector{String}` — variable names
- `obs_id::Vector{Int}` — observation identifiers
- `N_obs::Int` — number of observations
- `n_vars::Int` — number of variables
- `desc::Vector{String}` — dataset description (length-1 vector for mutability)
- `vardesc::Dict{String,String}` — per-variable descriptions keyed by variable name
- `source_refs::Vector{Symbol}` — reference keys for bibliographic citations (see `refs()`)
"""
struct CrossSectionData{T<:AbstractFloat} <: AbstractMacroData
    data::Matrix{T}
    varnames::Vector{String}
    obs_id::Vector{Int}
    N_obs::Int
    n_vars::Int
    desc::Vector{String}
    vardesc::Dict{String,String}
    source_refs::Vector{Symbol}
end

function CrossSectionData(data::AbstractMatrix{T};
                          varnames::Union{Vector{String},Nothing}=nothing,
                          obs_id::Union{Vector{Int},Nothing}=nothing,
                          desc::String="",
                          vardesc::Union{Dict{String,String},Nothing}=nothing,
                          source_refs::Vector{Symbol}=Symbol[]) where {T<:AbstractFloat}
    N_obs, n_vars = size(data)
    N_obs < 1 && throw(ArgumentError("Data must have at least 1 observation"))
    n_vars < 1 && throw(ArgumentError("Data must have at least 1 variable"))
    vn = something(varnames, ["x$i" for i in 1:n_vars])
    length(vn) != n_vars && throw(ArgumentError("varnames length must match n_vars"))
    oid = something(obs_id, collect(1:N_obs))
    length(oid) != N_obs && throw(ArgumentError("obs_id length must match N_obs"))
    vd = something(vardesc, Dict{String,String}())
    CrossSectionData{T}(Matrix{T}(data), vn, oid, N_obs, n_vars, [desc], vd, source_refs)
end

function CrossSectionData(data::AbstractMatrix; kwargs...)
    CrossSectionData(Float64.(data); kwargs...)
end

function CrossSectionData(df::DataFrame;
                          varnames::Union{Vector{String},Nothing}=nothing,
                          obs_id::Union{Vector{Int},Nothing}=nothing,
                          desc::String="",
                          vardesc::Union{Dict{String,String},Nothing}=nothing,
                          source_refs::Vector{Symbol}=Symbol[])
    num_cols = [n for n in names(df) if eltype(df[!, n]) <: Union{Missing, Number}]
    isempty(num_cols) && throw(ArgumentError("DataFrame has no numeric columns"))
    mat = Matrix{Float64}(undef, nrow(df), length(num_cols))
    for (j, col) in enumerate(num_cols)
        for i in 1:nrow(df)
            v = df[i, col]
            mat[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end
    vn = something(varnames, num_cols)
    CrossSectionData(mat; varnames=vn, obs_id=obs_id, desc=desc, vardesc=vardesc,
                     source_refs=source_refs)
end

# =============================================================================
# Accessors
# =============================================================================

"""
    StatsAPI.nobs(d::AbstractMacroData)

Return the number of observations.
"""
StatsAPI.nobs(d::TimeSeriesData) = d.T_obs
StatsAPI.nobs(d::PanelData) = d.T_obs
StatsAPI.nobs(d::CrossSectionData) = d.N_obs

"""
    nvars(d::AbstractMacroData)

Return the number of variables.
"""
nvars(d::TimeSeriesData) = d.n_vars
nvars(d::PanelData) = d.n_vars
nvars(d::CrossSectionData) = d.n_vars

"""
    varnames(d::AbstractMacroData)

Return variable names.
"""
varnames(d::AbstractMacroData) = d.varnames

"""
    frequency(d::TimeSeriesData)
    frequency(d::PanelData)

Return the data frequency.
"""
frequency(d::TimeSeriesData) = d.frequency
frequency(d::PanelData) = d.frequency

"""
    time_index(d::TimeSeriesData)

Return the integer time index vector.
"""
time_index(d::TimeSeriesData) = d.time_index

"""
    dates(d::TimeSeriesData) -> Vector{String}

Return the date labels vector. Returns empty vector if no dates are set.
"""
dates(d::TimeSeriesData) = d.dates

"""
    obs_id(d::CrossSectionData)

Return the observation identifier vector.
"""
obs_id(d::CrossSectionData) = d.obs_id

"""
    desc(d::AbstractMacroData) -> String

Return the dataset description. Returns `""` if no description has been set.

# Examples
```julia
d = TimeSeriesData(randn(100, 3); desc="US macro quarterly data")
desc(d)  # "US macro quarterly data"
```
"""
desc(d::AbstractMacroData) = d.desc[1]

"""
    vardesc(d::AbstractMacroData) -> Dict{String,String}

Return the dictionary of per-variable descriptions.

# Examples
```julia
d = TimeSeriesData(randn(100, 2); varnames=["GDP","CPI"],
    vardesc=Dict("GDP" => "Real GDP growth", "CPI" => "Consumer prices"))
vardesc(d)  # Dict("GDP" => "Real GDP growth", "CPI" => "Consumer prices")
```
"""
vardesc(d::AbstractMacroData) = d.vardesc

"""
    vardesc(d::AbstractMacroData, name::String) -> String

Return the description for variable `name`. Throws `ArgumentError` if no
description exists for that variable.

# Examples
```julia
vardesc(d, "GDP")  # "Real GDP growth"
```
"""
function vardesc(d::AbstractMacroData, name::String)
    haskey(d.vardesc, name) || throw(ArgumentError(
        "No description for variable '$name'. Available: $(collect(keys(d.vardesc)))"))
    d.vardesc[name]
end

# =============================================================================
# Matrix/Vector Conversion
# =============================================================================

Base.Matrix(d::TimeSeriesData) = d.data
Base.Matrix(d::PanelData) = d.data
Base.Matrix(d::CrossSectionData) = d.data

function Base.Vector(d::TimeSeriesData)
    d.n_vars == 1 || throw(ArgumentError("Vector conversion requires exactly 1 variable, got $(d.n_vars)"))
    d.data[:, 1]
end

# =============================================================================
# Indexing
# =============================================================================

# Array interface methods
Base.axes(d::TimeSeriesData) = (Base.OneTo(d.T_obs), Base.OneTo(d.n_vars))

function Base.axes(d::TimeSeriesData, dim::Integer)
    dim >= 1 || throw(BoundsError((Base.OneTo(d.T_obs), Base.OneTo(d.n_vars)), dim))
    if dim == 1
        return Base.OneTo(d.T_obs)
    elseif dim == 2
        return Base.OneTo(d.n_vars)
    else
        return Base.OneTo(1)
    end
end

Base.firstindex(d::TimeSeriesData) = 1

function Base.firstindex(d::TimeSeriesData, dim::Integer)
    dim >= 1 || throw(BoundsError((Base.OneTo(d.T_obs), Base.OneTo(d.n_vars)), dim))
    return 1
end

Base.lastindex(d::TimeSeriesData) = d.T_obs * d.n_vars

function Base.lastindex(d::TimeSeriesData, dim::Integer)
    dim >= 1 || throw(BoundsError((Base.OneTo(d.T_obs), Base.OneTo(d.n_vars)), dim))
    if dim == 1
        return d.T_obs
    elseif dim == 2
        return d.n_vars
    else
        return 1
    end
end

function Base.size(d::TimeSeriesData, dim::Integer)
    if dim == 1
        return d.T_obs
    elseif dim == 2
        return d.n_vars
    else
        return 1
    end
end

# Internal index category aliases
const _TSRowScalar = Union{Integer, AbstractString}
const _TSColScalar = Union{Integer, Symbol, AbstractString}
const _TSRowColl = Union{
    Colon,
    AbstractVector{<:Integer},
    AbstractVector{Bool},
    AbstractVector{<:AbstractString},
}
const _TSColColl = Union{
    Colon,
    AbstractVector{<:Integer},
    AbstractVector{Bool},
    AbstractVector{Symbol},
    AbstractVector{<:AbstractString},
}

# --- Column Resolvers ---

function _resolve_ts_col_index(d::TimeSeriesData, col::Bool)
    throw(ArgumentError("invalid index: $col of type Bool"))
end

function _resolve_ts_col_index(d::TimeSeriesData, col::Integer)
    (1 <= col <= d.n_vars) || throw(BoundsError(d, (:, col)))
    return Int(col)
end

function _resolve_ts_col_index(d::TimeSeriesData, col::Symbol)
    col_str = String(col)
    idx = findfirst(==(col_str), d.varnames)
    idx === nothing && throw(ArgumentError(
        "Variable '$col' not found. Available: $(d.varnames)",
    ))
    return idx
end

function _resolve_ts_col_index(d::TimeSeriesData, col::AbstractString)
    idx = findfirst(==(col), d.varnames)
    idx === nothing && throw(ArgumentError(
        "Variable '$col' not found. Available: $(d.varnames)",
    ))
    return idx
end

function _resolve_ts_col_index(d::TimeSeriesData, col)
    throw(ArgumentError("Unsupported column index type: $(typeof(col))"))
end

_resolve_ts_col_indices(d::TimeSeriesData, ::Colon) = 1:d.n_vars

function _resolve_ts_col_indices(d::TimeSeriesData, cols::AbstractVector{Bool})
    length(cols) == d.n_vars || throw(ArgumentError(
        "boolean column mask length ($(length(cols))) must match n_vars ($(d.n_vars))",
    ))
    idxs = findall(cols)
    isempty(idxs) && throw(ArgumentError("No columns selected"))
    return idxs
end

function _resolve_ts_col_indices(d::TimeSeriesData, cols::AbstractVector{<:Integer})
    isempty(cols) && throw(ArgumentError("No columns selected"))
    idxs = Vector{Int}(undef, length(cols))
    @inbounds for (i, j) in enumerate(cols)
        (1 <= j <= d.n_vars) || throw(BoundsError(d, (:, cols)))
        idxs[i] = Int(j)
    end
    return idxs
end

function _resolve_ts_col_indices(d::TimeSeriesData, cols::AbstractVector{Symbol})
    isempty(cols) && throw(ArgumentError("No columns selected"))
    idxs = Vector{Int}(undef, length(cols))
    @inbounds for (i, s) in enumerate(cols)
        s_str = String(s)
        idx = findfirst(==(s_str), d.varnames)
        idx === nothing && throw(ArgumentError(
            "Variable '$s' not found. Available: $(d.varnames)",
        ))
        idxs[i] = idx
    end
    return idxs
end

function _resolve_ts_col_indices(d::TimeSeriesData, cols::AbstractVector{<:AbstractString})
    isempty(cols) && throw(ArgumentError("No columns selected"))
    idxs = Vector{Int}(undef, length(cols))
    @inbounds for (i, s) in enumerate(cols)
        idx = findfirst(==(s), d.varnames)
        idx === nothing && throw(ArgumentError(
            "Variable '$s' not found. Available: $(d.varnames)",
        ))
        idxs[i] = idx
    end
    return idxs
end

function _resolve_ts_col_indices(d::TimeSeriesData, cols)
    throw(ArgumentError("Unsupported column index type: $(typeof(cols))"))
end

# --- Row Resolvers ---

function _resolve_ts_row_index(d::TimeSeriesData, row::Bool)
    throw(ArgumentError("invalid index: $row of type Bool"))
end

function _resolve_ts_row_index(d::TimeSeriesData, row::Integer)
    (1 <= row <= d.T_obs) || throw(BoundsError(d, (row, :)))
    return Int(row)
end

function _resolve_ts_row_index(d::TimeSeriesData, row::AbstractString)
    isempty(d.dates) && throw(ArgumentError("No dates set. Use `set_dates!` first."))
    idx = findfirst(==(row), d.dates)
    idx === nothing && throw(ArgumentError(
        "Date '$row' not found. Available: $(d.dates[1]):$(d.dates[end])",
    ))
    return idx
end

function _resolve_ts_row_index(d::TimeSeriesData, row)
    throw(ArgumentError("Unsupported row index type: $(typeof(row))"))
end

_resolve_ts_row_indices(d::TimeSeriesData, ::Colon) = 1:d.T_obs

function _resolve_ts_row_indices(d::TimeSeriesData, rows::AbstractVector{Bool})
    length(rows) == d.T_obs || throw(ArgumentError(
        "boolean row mask length ($(length(rows))) must match T_obs ($(d.T_obs))",
    ))
    idxs = findall(rows)
    isempty(idxs) && throw(ArgumentError("No rows selected"))
    return idxs
end

function _resolve_ts_row_indices(d::TimeSeriesData, rows::AbstractVector{<:Integer})
    isempty(rows) && throw(ArgumentError("No rows selected"))
    idxs = Vector{Int}(undef, length(rows))
    @inbounds for (i, r) in enumerate(rows)
        (1 <= r <= d.T_obs) || throw(BoundsError(d, (rows, :)))
        idxs[i] = Int(r)
    end
    return idxs
end

function _resolve_ts_row_indices(d::TimeSeriesData, rows::AbstractVector{<:AbstractString})
    isempty(d.dates) && throw(ArgumentError("No dates set. Use `set_dates!` first."))
    isempty(rows) && throw(ArgumentError("No rows selected"))
    idxs = Vector{Int}(undef, length(rows))
    @inbounds for (i, dt) in enumerate(rows)
        idx = findfirst(==(dt), d.dates)
        idx === nothing && throw(ArgumentError("Date '$dt' not found."))
        idxs[i] = idx
    end
    return idxs
end

function _resolve_ts_row_indices(d::TimeSeriesData, rows)
    throw(ArgumentError("Unsupported row index type: $(typeof(rows))"))
end

# --- 2D getindex Methods ---

"""
    Base.getindex(d::TimeSeriesData, rows, cols)

2D indexing on `TimeSeriesData` supporting 4 combinations of row and column selectors:
1. `(collection, collection)` -> returns a sub-`TimeSeriesData` preserving metadata
2. `(collection, single)`     -> returns `Vector{T}` across rows for selected column
3. `(single, collection)`     -> returns `Vector{T}` across columns for selected row
4. `(single, single)`         -> returns scalar `T` element
"""
function Base.getindex(
    d::TimeSeriesData{T},
    rows::_TSRowColl,
    cols::_TSColColl,
) where {T}
    row_idxs = _resolve_ts_row_indices(d, rows)
    col_idxs = _resolve_ts_col_indices(d, cols)
    isempty(row_idxs) && throw(ArgumentError("No rows selected"))
    isempty(col_idxs) && throw(ArgumentError("No columns selected"))
    sub_varnames = d.varnames[col_idxs]
    sub_vd = Dict{String,String}(
        k => d.vardesc[k] for k in sub_varnames if haskey(d.vardesc, k)
    )
    sub_dates = isempty(d.dates) ? String[] : d.dates[row_idxs]
    return TimeSeriesData{T}(
        d.data[row_idxs, col_idxs],
        sub_varnames,
        d.frequency,
        d.tcode[col_idxs],
        d.time_index[row_idxs],
        length(row_idxs),
        length(col_idxs),
        copy(d.desc),
        sub_vd,
        copy(d.source_refs),
        sub_dates,
    )
end

function Base.getindex(
    d::TimeSeriesData{T},
    rows::_TSRowColl,
    col::_TSColScalar,
) where {T}
    row_idxs = _resolve_ts_row_indices(d, rows)
    col_idx = _resolve_ts_col_index(d, col)
    return d.data[row_idxs, col_idx]
end

function Base.getindex(
    d::TimeSeriesData{T},
    row::_TSRowScalar,
    cols::_TSColColl,
) where {T}
    row_idx = _resolve_ts_row_index(d, row)
    col_idxs = _resolve_ts_col_indices(d, cols)
    return d.data[row_idx, col_idxs]
end

function Base.getindex(
    d::TimeSeriesData{T},
    row::_TSRowScalar,
    col::_TSColScalar,
) where {T}
    row_idx = _resolve_ts_row_index(d, row)
    col_idx = _resolve_ts_col_index(d, col)
    return d.data[row_idx, col_idx]
end

function Base.getindex(d::CrossSectionData{T}, ::Colon, col::String) where {T}
    idx = findfirst(==(col), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$col' not found. Available: $(d.varnames)"))
    d.data[:, idx]
end

# =============================================================================
# Rename
# =============================================================================

"""
    rename_vars!(d::AbstractMacroData, old => new)
    rename_vars!(d::AbstractMacroData, names::Vector{String})

Rename variables in a data container. With a `Pair`, renames a single variable.
With a `Vector{String}`, replaces all variable names. Also updates `vardesc` keys.
"""
function rename_vars!(d::TimeSeriesData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    if haskey(d.vardesc, pair.first)
        d.vardesc[pair.second] = pop!(d.vardesc, pair.first)
    end
    d
end

function rename_vars!(d::TimeSeriesData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
    for (old, new) in zip(d.varnames, names)
        if old != new && haskey(d.vardesc, old)
            d.vardesc[new] = pop!(d.vardesc, old)
        end
    end
    copy!(d.varnames, names)
    d
end

function rename_vars!(d::PanelData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    if haskey(d.vardesc, pair.first)
        d.vardesc[pair.second] = pop!(d.vardesc, pair.first)
    end
    d
end

function rename_vars!(d::PanelData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
    for (old, new) in zip(d.varnames, names)
        if old != new && haskey(d.vardesc, old)
            d.vardesc[new] = pop!(d.vardesc, old)
        end
    end
    copy!(d.varnames, names)
    d
end

function rename_vars!(d::CrossSectionData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    if haskey(d.vardesc, pair.first)
        d.vardesc[pair.second] = pop!(d.vardesc, pair.first)
    end
    d
end

function rename_vars!(d::CrossSectionData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
    for (old, new) in zip(d.varnames, names)
        if old != new && haskey(d.vardesc, old)
            d.vardesc[new] = pop!(d.vardesc, old)
        end
    end
    copy!(d.varnames, names)
    d
end

# =============================================================================
# Setters
# =============================================================================

"""
    set_time_index!(d::TimeSeriesData, idx::Vector{Int})

Set the time index for a TimeSeriesData container.
"""
function set_time_index!(d::TimeSeriesData, idx::Vector{Int})
    length(idx) != d.T_obs && throw(ArgumentError("time_index length must match T_obs"))
    copy!(d.time_index, idx)
    d
end

"""
    set_obs_id!(d::CrossSectionData, ids::Vector{Int})

Set observation identifiers for a CrossSectionData container.
"""
function set_obs_id!(d::CrossSectionData, ids::Vector{Int})
    length(ids) != d.N_obs && throw(ArgumentError("obs_id length must match N_obs"))
    copy!(d.obs_id, ids)
    d
end

"""
    set_dates!(d::TimeSeriesData, dates::Vector{String})

Set the date labels for a TimeSeriesData container. Length must match T_obs,
or pass an empty vector to clear dates.

# Examples
```julia
d = TimeSeriesData(randn(4, 2))
set_dates!(d, ["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
dates(d)  # ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
```
"""
function set_dates!(d::TimeSeriesData, dt::Vector{String})
    if !isempty(dt)
        length(dt) != d.T_obs && throw(ArgumentError("dates length ($(length(dt))) must match T_obs ($(d.T_obs))"))
    end
    if isempty(d.dates)
        append!(d.dates, dt)
    else
        resize!(d.dates, length(dt))
        copy!(d.dates, dt)
    end
    d
end

"""
    set_desc!(d::AbstractMacroData, text::String)

Set the dataset description.

# Examples
```julia
d = TimeSeriesData(randn(100, 3))
set_desc!(d, "US macroeconomic quarterly data 1959-2024")
desc(d)  # "US macroeconomic quarterly data 1959-2024"
```
"""
function set_desc!(d::AbstractMacroData, text::String)
    d.desc[1] = text
    d
end

"""
    set_vardesc!(d::AbstractMacroData, name::String, text::String)

Set the description for a single variable. The variable must exist in the data.

# Examples
```julia
d = TimeSeriesData(randn(100, 2); varnames=["GDP", "CPI"])
set_vardesc!(d, "GDP", "Real Gross Domestic Product, seasonally adjusted")
set_vardesc!(d, "CPI", "Consumer Price Index for All Urban Consumers")
vardesc(d, "GDP")  # "Real Gross Domestic Product, seasonally adjusted"
```
"""
function set_vardesc!(d::AbstractMacroData, name::String, text::String)
    name ∈ d.varnames || throw(ArgumentError(
        "Variable '$name' not found. Available: $(d.varnames)"))
    d.vardesc[name] = text
    d
end

"""
    set_vardesc!(d::AbstractMacroData, descriptions::Dict{String,String})

Set descriptions for multiple variables at once. All keys must be valid variable names.

# Examples
```julia
set_vardesc!(d, Dict("GDP" => "Real GDP", "CPI" => "Consumer prices"))
```
"""
function set_vardesc!(d::AbstractMacroData, descriptions::Dict{String,String})
    for name in keys(descriptions)
        name ∈ d.varnames || throw(ArgumentError(
            "Variable '$name' not found. Available: $(d.varnames)"))
    end
    merge!(d.vardesc, descriptions)
    d
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, d::TimeSeriesData{T}) where {T}
    print(io, "TimeSeriesData{$T}: $(d.T_obs) obs × $(d.n_vars) vars")
    if d.frequency != Other
        print(io, " ($(d.frequency))")
    end
    if d.n_vars <= 10
        print(io, " [", join(d.varnames, ", "), "]")
    end
    if !isempty(d.desc[1])
        print(io, "\n  ", d.desc[1])
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::TimeSeriesData{T}) where {T}
    show(io, d)
end

function Base.show(io::IO, d::PanelData{T}) where {T}
    print(io, "PanelData{$T}: $(d.T_obs) obs × $(d.n_vars) vars, $(d.n_groups) groups")
    if d.frequency != Other
        print(io, " ($(d.frequency))")
    end
    print(io, d.balanced ? " [balanced]" : " [unbalanced]")
    if d.cohort_id !== nothing
        n_cohorts = length(unique(d.cohort_id))
        print(io, ", $n_cohorts cohorts")
    end
    if !isempty(d.desc[1])
        print(io, "\n  ", d.desc[1])
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::PanelData{T}) where {T}
    show(io, d)
end

function Base.show(io::IO, d::CrossSectionData{T}) where {T}
    print(io, "CrossSectionData{$T}: $(d.N_obs) obs × $(d.n_vars) vars")
    if d.n_vars <= 10
        print(io, " [", join(d.varnames, ", "), "]")
    end
    if !isempty(d.desc[1])
        print(io, "\n  ", d.desc[1])
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::CrossSectionData{T}) where {T}
    show(io, d)
end

# =============================================================================
# Size and iteration helpers
# =============================================================================

Base.size(d::TimeSeriesData) = (d.T_obs, d.n_vars)
Base.size(d::PanelData) = (d.T_obs, d.n_vars)
Base.size(d::CrossSectionData) = (d.N_obs, d.n_vars)
Base.length(d::TimeSeriesData) = d.T_obs * d.n_vars
Base.length(d::PanelData) = d.T_obs * d.n_vars
Base.length(d::CrossSectionData) = d.N_obs * d.n_vars
