# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
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
FRED transformation codes (tcodes 1–7) for MacroEconometricModels.jl.

| tcode | Name              | Formula                 |
|-------|-------------------|-------------------------|
| 1     | Level             | x_t                     |
| 2     | First difference  | Δx_t                    |
| 3     | Second difference | Δ²x_t                   |
| 4     | Log               | log(x_t)                |
| 5     | Diff of log       | Δlog(x_t)               |
| 6     | Second diff of log| Δ²log(x_t)              |
| 7     | Delta pct change  | Δ(x_t/x_{t-1} − 1)     |
"""

# =============================================================================
# Univariate apply_tcode
# =============================================================================

"""
    apply_tcode(y::AbstractVector{<:Real}, tcode::Int) -> Vector{Float64}

Apply FRED transformation code to a univariate series.

Codes 4–7 require strictly positive data.
The output vector is shorter than the input for difference-based codes:
- tcode 1: same length
- tcode 2, 4, 5: length T-1
- tcode 3, 6, 7: length T-2

# Examples
```julia
y = [100.0, 102.0, 105.0, 103.0, 108.0]
apply_tcode(y, 5)  # log first differences (approx growth rates)
```
"""
function apply_tcode(y::AbstractVector{<:Real}, tcode::Int)
    1 <= tcode <= 7 || throw(ArgumentError("tcode must be in 1:7, got $tcode"))
    x = Float64.(y)
    T = length(x)

    if tcode >= 4
        any(v -> v <= 0, x) && throw(ArgumentError(
            "tcode $tcode requires strictly positive data"))
    end

    if tcode == 1
        return x
    elseif tcode == 2
        return diff(x)
    elseif tcode == 3
        return diff(diff(x))
    elseif tcode == 4
        return log.(x)
    elseif tcode == 5
        return diff(log.(x))
    elseif tcode == 6
        return diff(diff(log.(x)))
    else  # tcode == 7
        pct = x[2:end] ./ x[1:end-1] .- 1.0
        return diff(pct)
    end
end

# =============================================================================
# Data container apply_tcode
# =============================================================================

"""
    apply_tcode(d::TimeSeriesData, tcodes::Vector{Int}) -> TimeSeriesData

Apply per-variable FRED transformation codes. Rows are trimmed consistently
(to the shortest transformed series).

# Examples
```julia
d = TimeSeriesData(rand(200, 3) .+ 1; varnames=["GDP","CPI","FFR"])
d2 = apply_tcode(d, [5, 5, 1])  # log-diff GDP and CPI, leave FFR in levels
```
"""
function apply_tcode(d::TimeSeriesData{T}, tcodes::Vector{Int}) where {T}
    n = d.n_vars
    length(tcodes) != n && throw(ArgumentError(
        "tcodes length ($(length(tcodes))) must match n_vars ($n)"))

    # Compute the number of rows lost per tcode
    function _rows_lost(tc::Int)
        tc == 1 && return 0
        tc ∈ (2, 4, 5) && return 1
        tc ∈ (3, 6, 7) && return 2
        return 0
    end

    max_lost = maximum(_rows_lost(tc) for tc in tcodes)
    new_T = d.T_obs - max_lost
    new_T < 1 && throw(ArgumentError("Not enough observations after transformation"))

    # Transform each column and align to common length
    new_data = Matrix{Float64}(undef, new_T, n)
    for j in 1:n
        col_transformed = apply_tcode(d.data[:, j], tcodes[j])
        # Take the last new_T elements (align to end)
        offset = length(col_transformed) - new_T
        new_data[:, j] = col_transformed[(offset + 1):end]
    end

    # Trim time_index to match
    new_ti = d.time_index[(max_lost + 1):end]

    new_dates = isempty(d.dates) ? nothing : d.dates[(max_lost + 1):end]
    TimeSeriesData(new_data;
                   varnames=copy(d.varnames),
                   frequency=d.frequency,
                   tcode=tcodes,
                   time_index=new_ti,
                   desc=desc(d),
                   vardesc=copy(d.vardesc),
                   source_refs=copy(d.source_refs),
                   dates=new_dates)
end

"""
    apply_tcode(d::TimeSeriesData, tcode::Int) -> TimeSeriesData

Apply the same FRED transformation code to all variables.
"""
function apply_tcode(d::TimeSeriesData, tcode::Int)
    apply_tcode(d, fill(tcode, d.n_vars))
end

"""
    apply_tcode(d::TimeSeriesData) -> TimeSeriesData

Convenience method: apply the transformation codes stored in `d.tcode`.
Equivalent to `apply_tcode(d, d.tcode)`.

# Examples
```julia
d = load_example(:fred_md)  # tcode already set
d_transformed = apply_tcode(d)
```
"""
apply_tcode(d::TimeSeriesData) = apply_tcode(d, d.tcode)

# =============================================================================
# PanelData apply_tcode (group-by-group)
# =============================================================================

"""
    apply_tcode(d::PanelData, tcodes::Vector{Int}) -> PanelData

Apply per-variable FRED transformation codes to a `PanelData` container,
processing each group independently. Each group is trimmed to its own
common length after transformation, then reassembled.

# Examples
```julia
pd = xtset(df, :id, :t)
pd2 = apply_tcode(pd, [5, 5, 1])  # log-diff first two vars, leave third in levels
```
"""
function apply_tcode(d::PanelData{T}, tcodes::Vector{Int}) where {T}
    n = d.n_vars
    length(tcodes) != n && throw(ArgumentError(
        "tcodes length ($(length(tcodes))) must match n_vars ($n)"))

    # Transform each group independently
    group_datas = Vector{TimeSeriesData}(undef, d.n_groups)
    for g in 1:d.n_groups
        gd = group_data(d, g)
        group_datas[g] = apply_tcode(gd, tcodes)
    end

    # Reassemble into PanelData
    total_rows = sum(nobs(gd) for gd in group_datas)
    new_data = Matrix{Float64}(undef, total_rows, n)
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

    obs_per_group = [nobs(gd) for gd in group_datas]
    balanced = all(==(obs_per_group[1]), obs_per_group)

    PanelData{Float64}(new_data, copy(d.varnames), d.frequency, tcodes,
                        new_group_id, new_time_id, copy(d.group_names),
                        d.n_groups, n, total_rows, balanced,
                        copy(d.desc), copy(d.vardesc), copy(d.source_refs))
end

"""
    apply_tcode(d::PanelData, tcode::Int) -> PanelData

Apply the same FRED transformation code to all variables in a PanelData container.
"""
function apply_tcode(d::PanelData, tcode::Int)
    apply_tcode(d, fill(tcode, d.n_vars))
end

# =============================================================================
# Inverse transformation
# =============================================================================

"""
    inverse_tcode(y::AbstractVector{<:Real}, tcode::Int;
                  x_prev::Union{AbstractVector{<:Real},Nothing}=nothing) -> Vector{Float64}

Undo a FRED transformation to recover the original series.

For difference-based codes (2, 3, 5, 6, 7), `x_prev` must supply the initial values
needed for reconstruction:
- tcode 2: `x_prev = [x₀]` (1 value)
- tcode 3: `x_prev = [x₀, x₁]` (2 values, original levels)
- tcode 5: `x_prev = [x₀]` (1 value, original level)
- tcode 6: `x_prev = [x₀, x₁]` (2 values, original levels)
- tcode 7: `x_prev = [x₀, x₁]` (2 values, original levels)

# Examples
```julia
y = [100.0, 102.0, 105.0]
yd = apply_tcode(y, 2)                        # [2.0, 3.0]
inverse_tcode(yd, 2; x_prev=[100.0])          # [102.0, 105.0]
```
"""
function inverse_tcode(y::AbstractVector{<:Real}, tcode::Int;
                       x_prev::Union{AbstractVector{<:Real},Nothing}=nothing)
    1 <= tcode <= 7 || throw(ArgumentError("tcode must be in 1:7, got $tcode"))
    z = Float64.(y)
    T = length(z)

    if tcode == 1
        return z
    elseif tcode == 4
        return exp.(z)
    elseif tcode == 2
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 2"))
        length(x_prev) < 1 && throw(ArgumentError("x_prev must have at least 1 value"))
        out = Vector{Float64}(undef, T)
        prev = Float64(x_prev[end])
        for i in 1:T
            prev = prev + z[i]
            out[i] = prev
        end
        return out
    elseif tcode == 3
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 3"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        x0 = Float64(x_prev[end-1])
        x1 = Float64(x_prev[end])
        d1_prev = x1 - x0
        d1 = Vector{Float64}(undef, T)
        prev_d = d1_prev
        for i in 1:T
            prev_d = prev_d + z[i]
            d1[i] = prev_d
        end
        out = Vector{Float64}(undef, T)
        prev_x = x1
        for i in 1:T
            prev_x = prev_x + d1[i]
            out[i] = prev_x
        end
        return out
    elseif tcode == 5
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 5"))
        length(x_prev) < 1 && throw(ArgumentError("x_prev must have at least 1 value"))
        log_prev = log(Float64(x_prev[end]))
        out = Vector{Float64}(undef, T)
        for i in 1:T
            log_prev = log_prev + z[i]
            out[i] = exp(log_prev)
        end
        return out
    elseif tcode == 6
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 6"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        lx0 = log(Float64(x_prev[end-1]))
        lx1 = log(Float64(x_prev[end]))
        dlog_prev = lx1 - lx0
        out = Vector{Float64}(undef, T)
        log_prev = lx1
        for i in 1:T
            dlog_prev = dlog_prev + z[i]
            log_prev = log_prev + dlog_prev
            out[i] = exp(log_prev)
        end
        return out
    else  # tcode == 7
        x_prev === nothing && throw(ArgumentError("x_prev required for inverse tcode 7"))
        length(x_prev) < 2 && throw(ArgumentError("x_prev must have at least 2 values"))
        xm2 = Float64(x_prev[end-1])
        xm1 = Float64(x_prev[end])
        pct_prev = xm1 / xm2 - 1.0
        out = Vector{Float64}(undef, T)
        prev_x = xm1
        for i in 1:T
            pct_prev = pct_prev + z[i]
            prev_x = prev_x * (1.0 + pct_prev)
            out[i] = prev_x
        end
        return out
    end
end
