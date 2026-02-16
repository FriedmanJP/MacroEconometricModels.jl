# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Apply time series filters (HP, Hamilton, BN, BK, Boosted HP) to variables in
`TimeSeriesData` and `PanelData` containers.

Provides `apply_filter` which applies filters per-variable, extracts trend or
cycle components, aligns output to a common valid range, and preserves metadata.
"""

# =============================================================================
# Filter symbol → function mapping
# =============================================================================

const _FILTER_MAP = Dict{Symbol, Function}(
    :hp         => hp_filter,
    :hamilton   => hamilton_filter,
    :bn         => beveridge_nelson,
    :bk         => baxter_king,
    :boosted_hp => boosted_hp,
)

# =============================================================================
# Valid range helpers — determine which indices into the original series are valid
# =============================================================================

_filter_valid_range(r::HPFilterResult)         = 1:r.T_obs
_filter_valid_range(r::HamiltonFilterResult)    = r.valid_range
_filter_valid_range(r::BeveridgeNelsonResult)   = 1:r.T_obs
_filter_valid_range(r::BaxterKingResult)        = r.valid_range
_filter_valid_range(r::BoostedHPResult)         = 1:r.T_obs

# =============================================================================
# Extract component from a filter result
# =============================================================================

function _extract_component(r::AbstractFilterResult, component::Symbol)
    if component == :cycle
        return cycle(r)
    elseif component == :trend
        return trend(r)
    else
        throw(ArgumentError("component must be :cycle or :trend, got :$component"))
    end
end

# =============================================================================
# Parse a single filter specification
# =============================================================================

"""
    _parse_filter_spec(spec, component::Symbol) -> (filter_or_result_or_nothing, component_sym)

Parse a single element of the filter specification vector.

- `spec::Symbol` → look up in `_FILTER_MAP`, use `component`
- `spec::AbstractFilterResult` → use directly with `component`
- `spec::Tuple{<:Any, Symbol}` → parse first element, use second as component
- `spec::Nothing` → pass-through (no filtering)
"""
function _parse_filter_spec(spec::Symbol, component::Symbol)
    haskey(_FILTER_MAP, spec) || throw(ArgumentError(
        "Unknown filter symbol :$spec. Available: $(collect(keys(_FILTER_MAP)))"))
    return (_FILTER_MAP[spec], component)
end

function _parse_filter_spec(spec::AbstractFilterResult, component::Symbol)
    return (spec, component)
end

function _parse_filter_spec(spec::Tuple, component::Symbol)
    length(spec) == 2 || throw(ArgumentError(
        "Tuple filter spec must have 2 elements (filter, component), got $(length(spec))"))
    filter_part, comp_part = spec
    comp_part isa Symbol || throw(ArgumentError(
        "Second element of tuple must be a Symbol (:cycle or :trend), got $(typeof(comp_part))"))
    parsed_filter, _ = _parse_filter_spec(filter_part, comp_part)
    return (parsed_filter, comp_part)
end

function _parse_filter_spec(spec::Nothing, component::Symbol)
    return (nothing, :none)
end

# =============================================================================
# Core: apply_filter for TimeSeriesData with per-variable specs vector
# =============================================================================

"""
    apply_filter(d::TimeSeriesData, specs::AbstractVector; component=:cycle, kwargs...)

Apply per-variable filter specifications to a `TimeSeriesData` container.

Each element of `specs` can be:
- `Symbol` — filter name (`:hp`, `:hamilton`, `:bn`, `:bk`, `:boosted_hp`)
- `AbstractFilterResult` — a pre-computed filter result
- `Tuple{filter, Symbol}` — filter with per-variable component override (e.g., `(:hp, :trend)`)
- `nothing` — pass-through (no filtering for this variable)

The output is trimmed to the intersection of valid ranges across all variables.

# Arguments
- `d::TimeSeriesData` — input data
- `specs::AbstractVector` — per-variable filter specifications (length must equal `n_vars`)

# Keyword Arguments
- `component::Symbol=:cycle` — default component to extract (`:cycle` or `:trend`)
- Additional kwargs are forwarded to filter functions

# Returns
A new `TimeSeriesData` with filtered data, trimmed to the common valid range.

# Examples
```julia
d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])

# Per-variable: HP cycle for GDP, Hamilton cycle for CPI, pass-through FFR
d2 = apply_filter(d, [:hp, :hamilton, nothing])

# Per-variable component overrides via tuples
d3 = apply_filter(d, [(:hp, :trend), (:hamilton, :cycle), nothing])
```
"""
function apply_filter(d::TimeSeriesData{T}, specs::AbstractVector;
                      component::Symbol=:cycle, kwargs...) where {T}
    n = d.n_vars
    length(specs) != n && throw(ArgumentError(
        "specs length ($(length(specs))) must match n_vars ($n)"))

    # Parse all specs
    parsed = [_parse_filter_spec(specs[j], component) for j in 1:n]

    # Run filters and collect results + valid ranges
    results = Vector{Union{Nothing, AbstractFilterResult}}(undef, n)
    components = Vector{Union{Nothing, Vector}}(undef, n)
    valid_ranges = Vector{UnitRange{Int}}(undef, n)

    for j in 1:n
        filter_fn_or_result, comp = parsed[j]

        if filter_fn_or_result === nothing
            # Pass-through: raw column, full range
            components[j] = nothing
            valid_ranges[j] = 1:d.T_obs
        elseif filter_fn_or_result isa AbstractFilterResult
            # Pre-computed result
            result = filter_fn_or_result
            results[j] = result
            components[j] = _extract_component(result, comp)
            valid_ranges[j] = _filter_valid_range(result)
        else
            # filter_fn_or_result is a Function — run it
            col = d.data[:, j]
            result = filter_fn_or_result(col; kwargs...)
            results[j] = result
            components[j] = _extract_component(result, comp)
            valid_ranges[j] = _filter_valid_range(result)
        end
    end

    # Compute common valid range intersection
    common_start = maximum(r.start for r in valid_ranges)
    common_end = minimum(r.stop for r in valid_ranges)
    common_start > common_end && throw(ArgumentError(
        "No common valid range after filtering. Ranges: $valid_ranges"))
    new_T = common_end - common_start + 1

    # Build aligned data matrix
    new_data = Matrix{Float64}(undef, new_T, n)
    for j in 1:n
        if components[j] === nothing
            # Pass-through: use raw data in common range
            new_data[:, j] = d.data[common_start:common_end, j]
        else
            # Filtered: offset into the component vector
            vr = valid_ranges[j]
            offset_start = common_start - vr.start + 1
            offset_end = offset_start + new_T - 1
            new_data[:, j] = components[j][offset_start:offset_end]
        end
    end

    # Trim time index
    new_ti = d.time_index[common_start:common_end]

    TimeSeriesData(new_data;
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=copy(d.tcode),
                   time_index=new_ti,
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs))
end

# =============================================================================
# Convenience: single filter symbol/result for all (or selected) variables
# =============================================================================

"""
    apply_filter(d::TimeSeriesData, spec::Union{Symbol, AbstractFilterResult};
                 component=:cycle, vars=nothing, kwargs...)

Apply a single filter to all variables (or a subset specified by `vars`).

Variables not in `vars` are passed through unchanged.

# Arguments
- `d::TimeSeriesData` — input data
- `spec` — filter symbol (`:hp`, `:hamilton`, `:bn`, `:bk`, `:boosted_hp`) or pre-computed result

# Keyword Arguments
- `component::Symbol=:cycle` — component to extract (`:cycle` or `:trend`)
- `vars::Union{Nothing, Vector{String}, Vector{Int}}` — variables to filter (default: all)
- Additional kwargs are forwarded to filter functions

# Examples
```julia
d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])

# HP cycle for all variables
d_hp = apply_filter(d, :hp)

# HP trend for selected variables only
d_sel = apply_filter(d, :hp; vars=["GDP", "CPI"], component=:trend)
```
"""
function apply_filter(d::TimeSeriesData, spec::Union{Symbol, AbstractFilterResult};
                      component::Symbol=:cycle,
                      vars::Union{Nothing, Vector{String}, Vector{Int}}=nothing,
                      kwargs...)
    if vars === nothing
        # Apply to all variables
        specs = fill(spec, d.n_vars)
    else
        # Build specs vector with nothing for non-selected vars
        specs = Vector{Any}(nothing, d.n_vars)
        if vars isa Vector{String}
            for v in vars
                idx = findfirst(==(v), d.varnames)
                idx === nothing && throw(ArgumentError(
                    "Variable '$v' not found. Available: $(d.varnames)"))
                specs[idx] = spec
            end
        else  # Vector{Int}
            for idx in vars
                1 <= idx <= d.n_vars || throw(BoundsError(d, idx))
                specs[idx] = spec
            end
        end
    end
    apply_filter(d, specs; component=component, kwargs...)
end

# =============================================================================
# PanelData: apply filter group-by-group
# =============================================================================

"""
    apply_filter(d::PanelData, spec; component=:cycle, vars=nothing, kwargs...)

Apply filters to a `PanelData` container group-by-group.

Each group is extracted via `group_data`, filtered, and the results are
reassembled into a new `PanelData`. Each group is trimmed independently
to its own common valid range (groups may have different resulting lengths
if unbalanced).

# Arguments
- `d::PanelData` — input panel data
- `spec` — filter specification: a single `Symbol`, `AbstractFilterResult`,
  or `AbstractVector` of per-variable specs (same formats as `TimeSeriesData`)

# Keyword Arguments
- `component::Symbol=:cycle` — component to extract
- `vars` — variables to filter (others pass-through)
- Additional kwargs forwarded to filter functions

# Examples
```julia
pd = xtset(df, :id, :t)
pd_hp = apply_filter(pd, :hp; component=:cycle)
pd_sel = apply_filter(pd, :hp; vars=["GDP"], component=:trend)
```
"""
function apply_filter(d::PanelData{T}, spec;
                      component::Symbol=:cycle,
                      vars::Union{Nothing, Vector{String}, Vector{Int}}=nothing,
                      kwargs...) where {T}
    # Collect filtered group data
    group_datas = Vector{TimeSeriesData}(undef, d.n_groups)
    for g in 1:d.n_groups
        gd = group_data(d, g)
        if spec isa AbstractVector
            group_datas[g] = apply_filter(gd, spec; component=component, kwargs...)
        else
            group_datas[g] = apply_filter(gd, spec; component=component, vars=vars, kwargs...)
        end
    end

    # Reassemble into PanelData
    total_rows = sum(nobs(gd) for gd in group_datas)
    n_v = d.n_vars
    new_data = Matrix{Float64}(undef, total_rows, n_v)
    new_group_id = Vector{Int}(undef, total_rows)
    new_time_id = Vector{Int}(undef, total_rows)

    row = 1
    for g in 1:d.n_groups
        gd = group_datas[g]
        nr = nobs(gd)
        new_data[row:row+nr-1, :] = gd.data
        new_group_id[row:row+nr-1] .= g
        new_time_id[row:row+nr-1] = gd.time_index
        row += nr
    end

    # Detect balanced
    obs_per_group = [nobs(gd) for gd in group_datas]
    balanced = all(==(obs_per_group[1]), obs_per_group)

    PanelData{Float64}(new_data, copy(d.varnames), d.frequency, copy(d.tcode),
                        new_group_id, new_time_id, copy(d.group_names),
                        d.n_groups, n_v, total_rows, balanced,
                        copy(d.desc), copy(d.vardesc), copy(d.source_refs))
end
