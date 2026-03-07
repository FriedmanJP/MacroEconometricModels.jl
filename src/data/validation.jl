# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Data validation and diagnostics for MacroEconometricModels.jl —
diagnose data issues, fix them, and validate compatibility with model types.
"""

# =============================================================================
# DataDiagnostic
# =============================================================================

"""
    DataDiagnostic

Result of `diagnose(d)` — per-variable issue counts and overall cleanliness.

# Fields
- `n_nan::Vector{Int}` — NaN count per variable
- `n_inf::Vector{Int}` — Inf count per variable
- `is_constant::Vector{Bool}` — true if variable has zero variance
- `is_short::Bool` — true if series has fewer than 10 observations
- `varnames::Vector{String}` — variable names
- `is_clean::Bool` — true if no issues detected
"""
struct DataDiagnostic
    n_nan::Vector{Int}
    n_inf::Vector{Int}
    is_constant::Vector{Bool}
    is_short::Bool
    varnames::Vector{String}
    is_clean::Bool
end

function Base.show(io::IO, d::DataDiagnostic)
    if d.is_clean
        println(io, "DataDiagnostic: clean (no issues detected)")
        return
    end
    println(io, "DataDiagnostic: issues detected")
    n = length(d.varnames)
    # Build table data
    rows = Any[]
    for i in 1:n
        has_issue = d.n_nan[i] > 0 || d.n_inf[i] > 0 || d.is_constant[i]
        if has_issue
            push!(rows, Any[d.varnames[i], d.n_nan[i], d.n_inf[i],
                           d.is_constant[i] ? "yes" : "no"])
        end
    end
    if !isempty(rows)
        data = Matrix{Any}(undef, length(rows), 4)
        for (k, r) in enumerate(rows)
            for j in 1:4
                data[k, j] = r[j]
            end
        end
        _pretty_table(io, data;
            column_labels = ["Variable", "NaN", "Inf", "Constant"],
            alignment = [:l, :r, :r, :c])
    end
    if d.is_short
        println(io, "  Warning: series has fewer than 10 observations")
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::DataDiagnostic)
    show(io, d)
end

# =============================================================================
# diagnose
# =============================================================================

"""
    diagnose(d::AbstractMacroData) -> DataDiagnostic

Scan data for NaN, Inf, constant columns, and very short series.

# Examples
```julia
d = TimeSeriesData(randn(100, 3))
diag = diagnose(d)
diag.is_clean  # true if no issues
```
"""
function diagnose(d::AbstractMacroData)
    mat = Matrix(d)
    T_obs, n = size(mat)
    vn = varnames(d)

    n_nan = [count(isnan, @view(mat[:, j])) for j in 1:n]
    n_inf = [count(isinf, @view(mat[:, j])) for j in 1:n]

    is_const = Vector{Bool}(undef, n)
    for j in 1:n
        col = @view(mat[:, j])
        finite_vals = filter(x -> isfinite(x), col)
        if length(finite_vals) <= 1
            is_const[j] = true
        else
            is_const[j] = all(x -> x == finite_vals[1], finite_vals)
        end
    end

    short = T_obs < 10

    clean = all(==(0), n_nan) && all(==(0), n_inf) && !any(is_const) && !short

    DataDiagnostic(n_nan, n_inf, is_const, short, copy(vn), clean)
end

# =============================================================================
# fix
# =============================================================================

"""
    fix(d::TimeSeriesData; method=:listwise) -> TimeSeriesData

Fix data issues and return a clean copy.

# Methods
- `:listwise` — drop rows with any NaN or Inf (default)
- `:interpolate` — linear interpolation for interior NaN, forward-fill edges
- `:mean` — replace NaN with column mean of finite values

Inf values are always replaced with NaN first (then handled by the chosen method).
Constant columns are dropped with a warning.

# Examples
```julia
d = TimeSeriesData([1.0 NaN; 2.0 3.0; 3.0 4.0])
d_clean = fix(d; method=:listwise)  # drops row 1
```
"""
function fix(d::TimeSeriesData{T}; method::Symbol=:listwise) where {T}
    method ∈ (:listwise, :interpolate, :mean) ||
        throw(ArgumentError("method must be :listwise, :interpolate, or :mean, got :$method"))

    mat = copy(d.data)
    T_obs, n = size(mat)

    # Replace Inf with NaN
    for j in 1:n, i in 1:T_obs
        if isinf(mat[i, j])
            mat[i, j] = T(NaN)
        end
    end

    if method == :listwise
        good_rows = [!any(isnan, @view(mat[i, :])) for i in 1:T_obs]
        mat = mat[good_rows, :]
        ti = d.time_index[good_rows]
        new_dates = isempty(d.dates) ? nothing : d.dates[good_rows]
    elseif method == :interpolate
        for j in 1:n
            _interpolate_column!(view(mat, :, j))
        end
        ti = copy(d.time_index)
        new_dates = isempty(d.dates) ? nothing : copy(d.dates)
    elseif method == :mean
        for j in 1:n
            col = @view(mat[:, j])
            finite_vals = filter(isfinite, col)
            if !isempty(finite_vals)
                m = mean(finite_vals)
                for i in 1:T_obs
                    if isnan(col[i])
                        col[i] = T(m)
                    end
                end
            end
        end
        ti = copy(d.time_index)
        new_dates = isempty(d.dates) ? nothing : copy(d.dates)
    end

    # Drop constant columns
    keep_cols = Int[]
    for j in 1:size(mat, 2)
        col = @view(mat[:, j])
        finite_vals = filter(isfinite, col)
        if length(finite_vals) > 1 && !all(x -> x == finite_vals[1], finite_vals)
            push!(keep_cols, j)
        else
            @warn "Dropping constant column '$(d.varnames[j])'"
        end
    end

    if isempty(keep_cols)
        throw(ArgumentError("All columns are constant after fixing — no data remaining"))
    end

    kept_names = d.varnames[keep_cols]
    sub_vd = Dict(k => v for (k, v) in d.vardesc if k in kept_names)
    TimeSeriesData(mat[:, keep_cols];
                   varnames=kept_names,
                   frequency=d.frequency,
                   tcode=d.tcode[keep_cols],
                   time_index=ti,
                   desc=desc(d),
                   vardesc=sub_vd,
                   source_refs=copy(d.source_refs),
                   dates=new_dates)
end

"""Linear interpolation for interior NaN, forward-fill for edges."""
function _interpolate_column!(col::AbstractVector{T}) where {T}
    n = length(col)
    n == 0 && return

    # Find first and last finite values
    first_finite = findfirst(isfinite, col)
    last_finite = findlast(isfinite, col)
    first_finite === nothing && return  # all NaN — nothing to do

    # Forward-fill leading NaN
    for i in 1:(first_finite - 1)
        col[i] = col[first_finite]
    end
    # Backward-fill trailing NaN
    for i in (last_finite + 1):n
        col[i] = col[last_finite]
    end

    # Linear interpolation for interior NaN
    i = first_finite + 1
    while i <= last_finite
        if isnan(col[i])
            # Find next finite value
            j = i + 1
            while j <= last_finite && isnan(col[j])
                j += 1
            end
            # Interpolate between col[i-1] and col[j]
            span = j - (i - 1)
            for k in i:(j - 1)
                frac = T(k - (i - 1)) / T(span)
                col[k] = col[i - 1] + frac * (col[j] - col[i - 1])
            end
            i = j + 1
        else
            i += 1
        end
    end
end

"""
    fix(d::PanelData; method=:listwise) -> PanelData

Fix data issues and return a clean copy.

# Methods
- `:listwise` — drop rows with any NaN or Inf (default)
- `:interpolate` — linear interpolation within each group
- `:mean` — replace NaN with within-group column mean

Inf values are replaced with NaN first. Constant columns are dropped with a warning.
"""
function fix(d::PanelData{T}; method::Symbol=:listwise) where {T}
    method ∈ (:listwise, :interpolate, :mean) ||
        throw(ArgumentError("method must be :listwise, :interpolate, or :mean, got :$method"))

    mat = copy(d.data)
    T_obs = d.T_obs

    # Replace Inf with NaN
    for j in 1:d.n_vars, i in 1:T_obs
        isinf(mat[i, j]) && (mat[i, j] = T(NaN))
    end

    if method == :listwise
        good = [all(x -> !isnan(x), @view(mat[i, :])) for i in 1:T_obs]
        idx = findall(good)
        isempty(idx) && throw(ArgumentError("No rows remaining after listwise deletion"))
        new_mat = mat[idx, :]
        new_gid = d.group_id[idx]
        new_tid = d.time_id[idx]
        new_cid = d.cohort_id === nothing ? nothing : d.cohort_id[idx]
    elseif method == :interpolate
        for g in 1:d.n_groups
            gidx = findall(d.group_id .== g)
            for j in 1:d.n_vars
                _interpolate_column!(view(mat, gidx, j))
            end
        end
        new_mat = mat
        new_gid = copy(d.group_id)
        new_tid = copy(d.time_id)
        new_cid = d.cohort_id === nothing ? nothing : copy(d.cohort_id)
    elseif method == :mean
        for g in 1:d.n_groups
            gidx = findall(d.group_id .== g)
            for j in 1:d.n_vars
                col = @view(mat[gidx, j])
                finite_vals = filter(isfinite, col)
                if !isempty(finite_vals)
                    m = mean(finite_vals)
                    for i in eachindex(col)
                        isnan(col[i]) && (col[i] = T(m))
                    end
                end
            end
        end
        new_mat = mat
        new_gid = copy(d.group_id)
        new_tid = copy(d.time_id)
        new_cid = d.cohort_id === nothing ? nothing : copy(d.cohort_id)
    end

    # Drop constant columns
    keep_cols = Int[]
    for j in 1:size(new_mat, 2)
        col = @view(new_mat[:, j])
        finite_vals = filter(isfinite, col)
        if length(finite_vals) > 1 && !all(x -> x == finite_vals[1], finite_vals)
            push!(keep_cols, j)
        else
            @warn "Dropping constant column '$(d.varnames[j])'"
        end
    end
    isempty(keep_cols) && throw(ArgumentError("All columns constant after fixing — no data remaining"))

    kept_names = d.varnames[keep_cols]
    sub_vd = Dict(k => v for (k, v) in d.vardesc if k in kept_names)
    n_new = size(new_mat, 1)
    obs_per_group = [count(==(g), new_gid) for g in 1:d.n_groups]
    balanced = length(obs_per_group) > 0 && all(==(obs_per_group[1]), obs_per_group)

    PanelData{T}(new_mat[:, keep_cols], kept_names, d.frequency, d.tcode[keep_cols],
                  new_gid, new_tid, new_cid, copy(d.group_names),
                  d.n_groups, length(kept_names), n_new, balanced,
                  copy(d.desc), sub_vd, copy(d.source_refs))
end

"""
    fix(d::CrossSectionData; method=:listwise) -> CrossSectionData

Fix data issues and return a clean copy.

# Methods
- `:listwise` — drop rows with any NaN or Inf (default)
- `:mean` — replace NaN with column mean of finite values
- `:interpolate` — same as `:mean` (no time ordering in cross-section data)

Inf values are replaced with NaN first. Constant columns are dropped with a warning.
"""
function fix(d::CrossSectionData{T}; method::Symbol=:listwise) where {T}
    method ∈ (:listwise, :interpolate, :mean) ||
        throw(ArgumentError("method must be :listwise, :interpolate, or :mean, got :$method"))

    effective = method == :interpolate ? :mean : method

    mat = copy(d.data)
    N_obs = d.N_obs

    # Replace Inf with NaN
    for j in 1:d.n_vars, i in 1:N_obs
        isinf(mat[i, j]) && (mat[i, j] = T(NaN))
    end

    if effective == :listwise
        good = [all(x -> !isnan(x), @view(mat[i, :])) for i in 1:N_obs]
        idx = findall(good)
        isempty(idx) && throw(ArgumentError("No rows remaining after listwise deletion"))
        new_mat = mat[idx, :]
        new_oid = d.obs_id[idx]
    elseif effective == :mean
        for j in 1:d.n_vars
            col = @view(mat[:, j])
            finite_vals = filter(isfinite, col)
            if !isempty(finite_vals)
                m = mean(finite_vals)
                for i in 1:N_obs
                    isnan(col[i]) && (col[i] = T(m))
                end
            end
        end
        new_mat = mat
        new_oid = copy(d.obs_id)
    end

    # Drop constant columns
    keep_cols = Int[]
    for j in 1:size(new_mat, 2)
        col = @view(new_mat[:, j])
        finite_vals = filter(isfinite, col)
        if length(finite_vals) > 1 && !all(x -> x == finite_vals[1], finite_vals)
            push!(keep_cols, j)
        else
            @warn "Dropping constant column '$(d.varnames[j])'"
        end
    end
    isempty(keep_cols) && throw(ArgumentError("All columns constant after fixing — no data remaining"))

    kept_names = d.varnames[keep_cols]
    sub_vd = Dict(k => v for (k, v) in d.vardesc if k in kept_names)
    CrossSectionData(new_mat[:, keep_cols];
                     varnames=kept_names,
                     obs_id=new_oid,
                     desc=desc(d),
                     vardesc=sub_vd,
                     source_refs=copy(d.source_refs))
end

# =============================================================================
# dropna
# =============================================================================

"""
    dropna(d::TimeSeriesData; vars=nothing) -> TimeSeriesData
    dropna(d::PanelData; vars=nothing) -> PanelData
    dropna(d::CrossSectionData; vars=nothing) -> CrossSectionData

Drop rows containing NaN or Inf values and return a new data container.

# Keyword Arguments
- `vars::Union{Vector{String}, Nothing}` — check only these variables (default: all)

# Examples
```julia
d_clean = dropna(d)                 # drop rows with any NaN/Inf
d_clean = dropna(d; vars=["GDP"])   # drop only if GDP is NaN/Inf
```
"""
function dropna(d::TimeSeriesData{T}; vars::Union{Vector{String},Nothing}=nothing) where {T}
    mat = d.data
    if vars === nothing
        good = [all(isfinite, @view(mat[i, :])) for i in 1:d.T_obs]
    else
        col_idx = [findfirst(==(v), d.varnames) for v in vars]
        any(isnothing, col_idx) && throw(ArgumentError(
            "Variable(s) not found: $(vars[findall(isnothing, col_idx)])"))
        cidx = Int[c for c in col_idx]
        good = [all(isfinite, @view(mat[i, cidx])) for i in 1:d.T_obs]
    end
    idx = findall(good)
    isempty(idx) && throw(ArgumentError("All rows contain NaN or Inf — no data remaining"))
    new_dates = isempty(d.dates) ? nothing : d.dates[idx]
    TimeSeriesData(mat[idx, :];
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=copy(d.tcode),
                   time_index=d.time_index[idx],
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs),
                   dates=new_dates)
end

function dropna(d::PanelData{T}; vars::Union{Vector{String},Nothing}=nothing) where {T}
    mat = d.data
    if vars === nothing
        good = [all(isfinite, @view(mat[i, :])) for i in 1:d.T_obs]
    else
        col_idx = [findfirst(==(v), d.varnames) for v in vars]
        any(isnothing, col_idx) && throw(ArgumentError(
            "Variable(s) not found: $(vars[findall(isnothing, col_idx)])"))
        cidx = Int[c for c in col_idx]
        good = [all(isfinite, @view(mat[i, cidx])) for i in 1:d.T_obs]
    end
    idx = findall(good)
    isempty(idx) && throw(ArgumentError("All rows contain NaN or Inf — no data remaining"))

    new_gid = d.group_id[idx]
    new_tid = d.time_id[idx]
    new_cid = d.cohort_id === nothing ? nothing : d.cohort_id[idx]
    obs_per_group = [count(==(g), new_gid) for g in 1:d.n_groups]
    balanced = length(obs_per_group) > 0 && all(==(obs_per_group[1]), obs_per_group)

    PanelData{T}(mat[idx, :], copy(d.varnames), d.frequency, copy(d.tcode),
                  new_gid, new_tid, new_cid, copy(d.group_names),
                  d.n_groups, d.n_vars, length(idx), balanced,
                  copy(d.desc), copy(d.vardesc), copy(d.source_refs))
end

function dropna(d::CrossSectionData{T}; vars::Union{Vector{String},Nothing}=nothing) where {T}
    mat = d.data
    if vars === nothing
        good = [all(isfinite, @view(mat[i, :])) for i in 1:d.N_obs]
    else
        col_idx = [findfirst(==(v), d.varnames) for v in vars]
        any(isnothing, col_idx) && throw(ArgumentError(
            "Variable(s) not found: $(vars[findall(isnothing, col_idx)])"))
        cidx = Int[c for c in col_idx]
        good = [all(isfinite, @view(mat[i, cidx])) for i in 1:d.N_obs]
    end
    idx = findall(good)
    isempty(idx) && throw(ArgumentError("All rows contain NaN or Inf — no data remaining"))
    CrossSectionData(mat[idx, :];
                     varnames=copy(d.varnames),
                     obs_id=d.obs_id[idx],
                     desc=desc(d),
                     vardesc=copy(d.vardesc),
                     source_refs=copy(d.source_refs))
end

# =============================================================================
# keeprows
# =============================================================================

"""
    keeprows(d, mask::BitVector)
    keeprows(d, indices::Vector{Int})

Keep only the specified rows and return a new data container.

# Examples
```julia
d2 = keeprows(d, d[:, "GDP"] .> 0)         # BitVector mask
d2 = keeprows(d, [1, 3, 5, 7])             # integer indices
```
"""
function keeprows(d::TimeSeriesData{T}, mask::BitVector) where {T}
    length(mask) == d.T_obs || throw(ArgumentError(
        "mask length ($(length(mask))) must match T_obs ($(d.T_obs))"))
    keeprows(d, findall(mask))
end

function keeprows(d::TimeSeriesData{T}, idx::Vector{Int}) where {T}
    isempty(idx) && throw(ArgumentError("No rows selected — empty result"))
    all(i -> 1 <= i <= d.T_obs, idx) || throw(BoundsError(d, idx))
    new_dates = isempty(d.dates) ? nothing : d.dates[idx]
    TimeSeriesData(d.data[idx, :];
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=copy(d.tcode),
                   time_index=d.time_index[idx],
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs),
                   dates=new_dates)
end

function keeprows(d::PanelData{T}, mask::BitVector) where {T}
    length(mask) == d.T_obs || throw(ArgumentError(
        "mask length ($(length(mask))) must match T_obs ($(d.T_obs))"))
    keeprows(d, findall(mask))
end

function keeprows(d::PanelData{T}, idx::Vector{Int}) where {T}
    isempty(idx) && throw(ArgumentError("No rows selected — empty result"))
    all(i -> 1 <= i <= d.T_obs, idx) || throw(BoundsError(d, idx))
    new_gid = d.group_id[idx]
    new_tid = d.time_id[idx]
    new_cid = d.cohort_id === nothing ? nothing : d.cohort_id[idx]
    obs_per_group = [count(==(g), new_gid) for g in 1:d.n_groups]
    balanced = length(obs_per_group) > 0 && all(==(obs_per_group[1]), obs_per_group)
    PanelData{T}(d.data[idx, :], copy(d.varnames), d.frequency, copy(d.tcode),
                  new_gid, new_tid, new_cid, copy(d.group_names),
                  d.n_groups, d.n_vars, length(idx), balanced,
                  copy(d.desc), copy(d.vardesc), copy(d.source_refs))
end

function keeprows(d::CrossSectionData{T}, mask::BitVector) where {T}
    length(mask) == d.N_obs || throw(ArgumentError(
        "mask length ($(length(mask))) must match N_obs ($(d.N_obs))"))
    keeprows(d, findall(mask))
end

function keeprows(d::CrossSectionData{T}, idx::Vector{Int}) where {T}
    isempty(idx) && throw(ArgumentError("No rows selected — empty result"))
    all(i -> 1 <= i <= d.N_obs, idx) || throw(BoundsError(d, idx))
    CrossSectionData(d.data[idx, :];
                     varnames=copy(d.varnames),
                     obs_id=d.obs_id[idx],
                     desc=desc(d),
                     vardesc=copy(d.vardesc),
                     source_refs=copy(d.source_refs))
end

# =============================================================================
# validate_for_model
# =============================================================================

"""
    validate_for_model(d::AbstractMacroData, model_type::Symbol)

Check that data is compatible with the specified model type. Throws `ArgumentError` on mismatch.

# Model types requiring multivariate data (n_vars ≥ 2)
`:var`, `:vecm`, `:bvar`, `:factors`, `:dynamic_factors`, `:gdfm`

# Model types requiring univariate data (n_vars == 1)
`:arima`, `:ar`, `:ma`, `:arma`, `:arch`, `:garch`, `:egarch`, `:gjr_garch`, `:sv`,
`:hp_filter`, `:hamilton_filter`, `:beveridge_nelson`, `:baxter_king`, `:boosted_hp`,
`:adf`, `:kpss`, `:pp`, `:za`, `:ngperron`

# Model types accepting any dimensionality
`:lp`, `:lp_iv`, `:smooth_lp`, `:state_lp`, `:propensity_lp`, `:gmm`

# Examples
```julia
d = TimeSeriesData(randn(100, 3))
validate_for_model(d, :var)    # OK
validate_for_model(d, :arima)  # throws ArgumentError
```
"""
function validate_for_model(d::AbstractMacroData, model_type::Symbol)
    n = nvars(d)

    multivariate = (:var, :vecm, :bvar, :factors, :dynamic_factors, :gdfm)
    univariate = (:arima, :ar, :ma, :arma, :arch, :garch, :egarch, :gjr_garch, :sv,
                  :hp_filter, :hamilton_filter, :beveridge_nelson, :baxter_king, :boosted_hp,
                  :adf, :kpss, :pp, :za, :ngperron)
    flexible = (:lp, :lp_iv, :smooth_lp, :state_lp, :propensity_lp, :gmm)

    if model_type ∈ multivariate
        n < 2 && throw(ArgumentError(
            "Model :$model_type requires multivariate data (n_vars ≥ 2), got n_vars=$n"))
    elseif model_type ∈ univariate
        n != 1 && throw(ArgumentError(
            "Model :$model_type requires univariate data (n_vars == 1), got n_vars=$n"))
    elseif model_type ∉ flexible
        throw(ArgumentError("Unknown model type :$model_type"))
    end

    nothing
end
