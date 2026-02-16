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
plot_result methods for filter types: HPFilter, Hamilton, BN, BK, BoostedHP.
"""

# =============================================================================
# Shared Filter Plotting
# =============================================================================

"""
Generate 2-panel Figure for a filter result: (1) Original + Trend, (2) Cycle.

- `tr`: trend component
- `cyc`: cycle component
- `filter_name`: display name
- `original`: original series (optional, reconstructed from trend+cycle if nil)
- `offset`: index offset for shorter filters (Hamilton, BK)
- `T_obs`: original series length
"""
function _plot_filter_panels(tr::AbstractVector, cyc::AbstractVector,
                             filter_name::String;
                             original::Union{AbstractVector,Nothing}=nothing,
                             offset::Int=0, T_obs::Int=0)
    orig = original !== nothing ? original : tr .+ cyc
    data_json = _filter_data_json(tr, cyc; original=orig, offset=offset)

    # Panel 1: Original + Trend
    id1 = _next_plot_id("filt_tc")
    s1 = _series_json(["Original", "Trend"], [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                       keys=["orig", "trend"], dash=["", "6,3"])
    js1 = _render_line_js(id1, data_json, s1; xlabel="Period", ylabel="Value")
    p1 = _PanelSpec(id1, "Trend-Cycle Decomposition", js1)

    # Panel 2: Cycle + zero line
    id2 = _next_plot_id("filt_cy")
    s2 = _series_json(["Cycle"], [_PLOT_COLORS[3]]; keys=["cycle"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data_json, s2;
                          ref_lines_json=refs, xlabel="Period", ylabel="Cycle")
    p2 = _PanelSpec(id2, "Cyclical Component", js2)

    [p1, p2]
end

# =============================================================================
# HPFilterResult
# =============================================================================

"""
    plot_result(r::HPFilterResult; title="", save_path=nothing)

Plot HP filter: original+trend and cycle.
"""
function plot_result(r::HPFilterResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "HP Filter";
                                 original=r.trend .+ r.cycle)
    if isempty(title)
        title = "Hodrick-Prescott Filter (λ=$(round(r.lambda, sigdigits=4)))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# HamiltonFilterResult
# =============================================================================

"""
    plot_result(r::HamiltonFilterResult; original=nothing, title="", save_path=nothing)

Plot Hamilton filter: original+trend and cycle.

Pass the original series via `original` kwarg since HamiltonFilterResult
doesn't store it.
"""
function plot_result(r::HamiltonFilterResult{T};
                     original::Union{AbstractVector,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "Hamilton Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs)
    if isempty(title)
        title = "Hamilton (2018) Filter (h=$(r.h), p=$(r.p))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BeveridgeNelsonResult
# =============================================================================

"""
    plot_result(r::BeveridgeNelsonResult; title="", save_path=nothing)

Plot BN decomposition: permanent+original and transitory.
"""
function plot_result(r::BeveridgeNelsonResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.permanent, r.transitory, "BN Decomposition";
                                 original=r.permanent .+ r.transitory)
    if isempty(title)
        p, d, q = r.arima_order
        title = "Beveridge-Nelson Decomposition (ARIMA($p,$d,$q))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BaxterKingResult
# =============================================================================

"""
    plot_result(r::BaxterKingResult; original=nothing, title="", save_path=nothing)

Plot BK band-pass filter.
"""
function plot_result(r::BaxterKingResult{T};
                     original::Union{AbstractVector,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "BK Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs)
    if isempty(title)
        title = "Baxter-King Band-Pass Filter ([$(r.pl), $(r.pu)], K=$(r.K))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BoostedHPResult
# =============================================================================

"""
    plot_result(r::BoostedHPResult; title="", save_path=nothing)

Plot boosted HP filter.
"""
function plot_result(r::BoostedHPResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "Boosted HP";
                                 original=r.trend .+ r.cycle)
    if isempty(title)
        title = "Boosted HP Filter ($(r.stopping), $(r.iterations) iter)"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
