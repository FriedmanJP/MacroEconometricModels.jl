"""
Data container types for MacroEconometricModels.jl — Frequency enum,
TimeSeriesData, PanelData, CrossSectionData with constructors and accessors.
"""

# =============================================================================
# Frequency Enum
# =============================================================================

"""
    Frequency

Enumeration of time series data frequencies.

Values: `Daily`, `Monthly`, `Quarterly`, `Yearly`, `Mixed`, `Other`.
"""
@enum Frequency Daily Monthly Quarterly Yearly Mixed Other

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

# Constructors
```julia
TimeSeriesData(data::Matrix; varnames, frequency=Other, tcode, time_index)
TimeSeriesData(data::Vector; varname="x1", frequency=Other)
TimeSeriesData(df::DataFrame; frequency=Other, varnames)
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
end

# Matrix constructor
function TimeSeriesData(data::AbstractMatrix{T};
                        varnames::Union{Vector{String},Nothing}=nothing,
                        frequency::Frequency=Other,
                        tcode::Union{Vector{Int},Nothing}=nothing,
                        time_index::Union{Vector{Int},Nothing}=nothing) where {T<:AbstractFloat}
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
    TimeSeriesData{T}(Matrix{T}(data), vn, frequency, tc, ti, T_obs, n_vars)
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
                        time_index::Union{Vector{Int},Nothing}=nothing) where {T<:AbstractFloat}
    TimeSeriesData(reshape(data, :, 1); varnames=[varname], frequency=frequency,
                   tcode=[tcode], time_index=time_index)
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
                        time_index::Union{Vector{Int},Nothing}=nothing)
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
    TimeSeriesData(mat; varnames=vn, frequency=frequency, tcode=tcode, time_index=time_index)
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
- `group_names::Vector{String}` — unique group labels
- `n_groups::Int` — number of groups
- `n_vars::Int` — number of variables
- `T_obs::Int` — total number of rows
- `balanced::Bool` — true if all groups have same number of observations
"""
struct PanelData{T<:AbstractFloat} <: AbstractMacroData
    data::Matrix{T}
    varnames::Vector{String}
    frequency::Frequency
    tcode::Vector{Int}
    group_id::Vector{Int}
    time_id::Vector{Int}
    group_names::Vector{String}
    n_groups::Int
    n_vars::Int
    T_obs::Int
    balanced::Bool
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
"""
struct CrossSectionData{T<:AbstractFloat} <: AbstractMacroData
    data::Matrix{T}
    varnames::Vector{String}
    obs_id::Vector{Int}
    N_obs::Int
    n_vars::Int
end

function CrossSectionData(data::AbstractMatrix{T};
                          varnames::Union{Vector{String},Nothing}=nothing,
                          obs_id::Union{Vector{Int},Nothing}=nothing) where {T<:AbstractFloat}
    N_obs, n_vars = size(data)
    N_obs < 1 && throw(ArgumentError("Data must have at least 1 observation"))
    n_vars < 1 && throw(ArgumentError("Data must have at least 1 variable"))
    vn = something(varnames, ["x$i" for i in 1:n_vars])
    length(vn) != n_vars && throw(ArgumentError("varnames length must match n_vars"))
    oid = something(obs_id, collect(1:N_obs))
    length(oid) != N_obs && throw(ArgumentError("obs_id length must match N_obs"))
    CrossSectionData{T}(Matrix{T}(data), vn, oid, N_obs, n_vars)
end

function CrossSectionData(data::AbstractMatrix; kwargs...)
    CrossSectionData(Float64.(data); kwargs...)
end

function CrossSectionData(df::DataFrame;
                          varnames::Union{Vector{String},Nothing}=nothing,
                          obs_id::Union{Vector{Int},Nothing}=nothing)
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
    CrossSectionData(mat; varnames=vn, obs_id=obs_id)
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
    obs_id(d::CrossSectionData)

Return the observation identifier vector.
"""
obs_id(d::CrossSectionData) = d.obs_id

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

function Base.getindex(d::TimeSeriesData{T}, ::Colon, col::String) where {T}
    idx = findfirst(==(col), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$col' not found. Available: $(d.varnames)"))
    d.data[:, idx]
end

function Base.getindex(d::TimeSeriesData{T}, ::Colon, cols::Vector{String}) where {T}
    idxs = Int[]
    for col in cols
        idx = findfirst(==(col), d.varnames)
        idx === nothing && throw(ArgumentError("Variable '$col' not found. Available: $(d.varnames)"))
        push!(idxs, idx)
    end
    TimeSeriesData{T}(d.data[:, idxs], d.varnames[idxs], d.frequency,
                      d.tcode[idxs], d.time_index, d.T_obs, length(idxs))
end

function Base.getindex(d::TimeSeriesData{T}, ::Colon, col::Int) where {T}
    1 <= col <= d.n_vars || throw(BoundsError(d, (Colon(), col)))
    d.data[:, col]
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
With a `Vector{String}`, replaces all variable names.
"""
function rename_vars!(d::TimeSeriesData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    d
end

function rename_vars!(d::TimeSeriesData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
    copy!(d.varnames, names)
    d
end

function rename_vars!(d::PanelData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    d
end

function rename_vars!(d::PanelData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
    copy!(d.varnames, names)
    d
end

function rename_vars!(d::CrossSectionData, pair::Pair{String,String})
    idx = findfirst(==(pair.first), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$(pair.first)' not found"))
    d.varnames[idx] = pair.second
    d
end

function rename_vars!(d::CrossSectionData, names::Vector{String})
    length(names) != d.n_vars && throw(ArgumentError("names length must match n_vars"))
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
end

function Base.show(io::IO, ::MIME"text/plain", d::PanelData{T}) where {T}
    show(io, d)
end

function Base.show(io::IO, d::CrossSectionData{T}) where {T}
    print(io, "CrossSectionData{$T}: $(d.N_obs) obs × $(d.n_vars) vars")
    if d.n_vars <= 10
        print(io, " [", join(d.varnames, ", "), "]")
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
