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
Summary statistics for MacroEconometricModels.jl data containers —
`describe_data()` with PrettyTables display.
"""

# =============================================================================
# DataSummary
# =============================================================================

"""
    DataSummary

Result of `describe_data(d)` — per-variable summary statistics.

# Fields
- `varnames::Vector{String}` — variable names
- `n::Vector{Int}` — non-NaN observation count per variable
- `mean::Vector{Float64}` — mean of finite values
- `std::Vector{Float64}` — standard deviation
- `min::Vector{Float64}` — minimum
- `p25::Vector{Float64}` — 25th percentile
- `median::Vector{Float64}` — median (50th percentile)
- `p75::Vector{Float64}` — 75th percentile
- `max::Vector{Float64}` — maximum
- `skewness::Vector{Float64}` — skewness
- `kurtosis::Vector{Float64}` — excess kurtosis
- `T_obs::Int` — total observations
- `n_vars::Int` — number of variables
- `frequency::Frequency` — data frequency
"""
struct DataSummary
    varnames::Vector{String}
    n::Vector{Int}
    mean::Vector{Float64}
    std::Vector{Float64}
    min::Vector{Float64}
    p25::Vector{Float64}
    median::Vector{Float64}
    p75::Vector{Float64}
    max::Vector{Float64}
    skewness::Vector{Float64}
    kurtosis::Vector{Float64}
    T_obs::Int
    n_vars::Int
    frequency::Frequency
end

function Base.show(io::IO, s::DataSummary)
    freq_str = s.frequency != Other ? " $(s.frequency)" : ""
    println(io, "Summary Statistics —$freq_str ($(s.T_obs) × $(s.n_vars))")

    n = s.n_vars
    data = Matrix{Any}(undef, n, 11)
    for i in 1:n
        data[i, 1] = s.varnames[i]
        data[i, 2] = s.n[i]
        data[i, 3] = round(s.mean[i], digits=4)
        data[i, 4] = round(s.std[i], digits=4)
        data[i, 5] = round(s.min[i], digits=4)
        data[i, 6] = round(s.p25[i], digits=4)
        data[i, 7] = round(s.median[i], digits=4)
        data[i, 8] = round(s.p75[i], digits=4)
        data[i, 9] = round(s.max[i], digits=4)
        data[i, 10] = round(s.skewness[i], digits=4)
        data[i, 11] = round(s.kurtosis[i], digits=4)
    end

    _pretty_table(io, data;
        column_labels = ["Variable", "N", "Mean", "Std", "Min", "P25",
                        "Median", "P75", "Max", "Skew", "Kurt"],
        alignment = vcat([:l], fill(:r, 10)))
end

function Base.show(io::IO, ::MIME"text/plain", s::DataSummary)
    show(io, s)
end

# =============================================================================
# describe_data
# =============================================================================

"""
    describe_data(d::AbstractMacroData) -> DataSummary

Compute per-variable summary statistics (N, Mean, Std, Min, P25, Median, P75,
Max, Skewness, Kurtosis). NaN values are excluded from all computations.

For `PanelData`, also prints panel dimensions via `panel_summary`.

# Examples
```julia
d = TimeSeriesData(randn(200, 3); varnames=["GDP","CPI","FFR"], frequency=Quarterly)
describe_data(d)
```
"""
function describe_data(d::TimeSeriesData)
    _compute_summary(d.data, d.varnames, d.T_obs, d.n_vars, d.frequency)
end

function describe_data(d::CrossSectionData)
    _compute_summary(d.data, d.varnames, d.N_obs, d.n_vars, Other)
end

function describe_data(d::PanelData)
    s = _compute_summary(d.data, d.varnames, d.T_obs, d.n_vars, d.frequency)
    # Show panel summary (catch IOError for threaded test environments)
    try
        panel_summary(stdout, d)
    catch e
        e isa Base.IOError || rethrow()
    end
    s
end

function _compute_summary(mat::Matrix{<:AbstractFloat}, vn::Vector{String},
                          T_obs::Int, n_vars::Int, freq::Frequency)
    ns = Vector{Int}(undef, n_vars)
    means = Vector{Float64}(undef, n_vars)
    stds = Vector{Float64}(undef, n_vars)
    mins = Vector{Float64}(undef, n_vars)
    p25s = Vector{Float64}(undef, n_vars)
    meds = Vector{Float64}(undef, n_vars)
    p75s = Vector{Float64}(undef, n_vars)
    maxs = Vector{Float64}(undef, n_vars)
    skews = Vector{Float64}(undef, n_vars)
    kurts = Vector{Float64}(undef, n_vars)

    for j in 1:n_vars
        col = @view(mat[:, j])
        finite_vals = filter(isfinite, col)
        nf = length(finite_vals)
        ns[j] = nf

        if nf == 0
            means[j] = stds[j] = mins[j] = p25s[j] = meds[j] = p75s[j] = maxs[j] = NaN
            skews[j] = kurts[j] = NaN
        else
            sorted = sort(finite_vals)
            m = mean(sorted)
            means[j] = m
            stds[j] = nf > 1 ? std(sorted) : 0.0
            mins[j] = sorted[1]
            maxs[j] = sorted[end]
            p25s[j] = _quantile(sorted, 0.25)
            meds[j] = _quantile(sorted, 0.50)
            p75s[j] = _quantile(sorted, 0.75)

            if nf > 2
                s = stds[j]
                if s > 0
                    centered = sorted .- m
                    skews[j] = mean(centered .^ 3) / s^3
                    kurts[j] = mean(centered .^ 4) / s^4 - 3.0
                else
                    skews[j] = 0.0
                    kurts[j] = 0.0
                end
            else
                skews[j] = 0.0
                kurts[j] = 0.0
            end
        end
    end

    DataSummary(copy(vn), ns, means, stds, mins, p25s, meds, p75s, maxs,
                skews, kurts, T_obs, n_vars, freq)
end

"""Simple quantile computation for a sorted vector."""
function _quantile(sorted::Vector{<:Real}, p::Float64)
    n = length(sorted)
    n == 1 && return Float64(sorted[1])
    h = (n - 1) * p + 1.0
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo = clamp(lo, 1, n)
    hi = clamp(hi, 1, n)
    frac = h - lo
    return Float64(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
end
