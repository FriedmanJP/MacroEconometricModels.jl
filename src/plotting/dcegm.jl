# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# DCEGM plots (G-22, #656)
#
#   • DCEGMSolution  — discrete-continuous EGM (Iskhakov et al. 2017):
#                      :policy     consumption vs cash-on-hand, one series per
#                                  discrete option (period 1 / stationary)
#                      :threshold  upper-envelope kink counts vs period
#                                  (or vs option when only one period is stored)
#
# Lane-local converters below (A5); all rendering goes through the frozen render.jl.
# =============================================================================

# -----------------------------------------------------------------------------
# Lane-local converters (A5 — documented `_dcegm_*` helpers; NOT in helpers.jl).
# -----------------------------------------------------------------------------

"""
    _dcegm_option_names(sol) -> Vector{String}

Display names of the discrete options, taken from `sol.prob.options`.
"""
_dcegm_option_names(sol::DCEGMSolution) = String[string(opt) for opt in sol.prob.options]

"""
    _dcegm_union_M(sol, t, j) -> Vector

Sorted unique cash-on-hand knots of every option's endogenous grid in period `t`,
income state `j`. Envelope kinks are stored as `(M*, nextfloat(M*))` pairs and
survive `unique!` because the two values are not equal.
"""
function _dcegm_union_M(sol::DCEGMSolution{T}, t::Int, j::Int) where {T<:AbstractFloat}
    n_d = size(sol.M, 2)
    xs = T[]
    for d in 1:n_d
        Mv = sol.M[t, d, j]
        isempty(Mv) || append!(xs, Mv)
    end
    sort!(xs)
    unique!(xs)
    return xs
end

"""
    _dcegm_c_at(Mv, cv, m) -> T

Linear interpolation of consumption on one option's endogenous grid. Returns
`NaN` outside the stored support so the renderer can leave a gap (plotrule
Robustness: `NaN` → `null` → `.defined()`).
"""
function _dcegm_c_at(Mv::AbstractVector{T}, cv::AbstractVector{T}, m::T) where {T<:AbstractFloat}
    (isempty(Mv) || isempty(cv) || m < Mv[1] || m > Mv[end]) && return T(NaN)
    return _seg_interp(Mv, cv, m)
end

"""
    _dcegm_subsample_idx(xs, max_pts) -> Vector{Int}

Keep at most `max_pts` knots, always including endpoints and any adjacent
`nextfloat` pair (an envelope kink). Extra kink knots may push the returned
length slightly above `max_pts` so the consumption jump is never dropped
(plotrule C7 / no silent truncation of structure).
"""
function _dcegm_subsample_idx(xs::AbstractVector, max_pts::Int)
    n = length(xs)
    n == 0 && return Int[]
    max_pts = max(2, max_pts)
    keep = falses(n)
    keep[1] = keep[n] = true
    if n > max_pts
        step = max(1, div(n - 1, max_pts - 1))
        @inbounds for i in 1:step:n
            keep[i] = true
        end
    else
        fill!(keep, true)
    end
    @inbounds for i in 1:(n - 1)
        if xs[i + 1] == nextfloat(xs[i])
            keep[i] = keep[i + 1] = true
        end
    end
    return findall(keep)
end

"""
    _dcegm_kink_counts(sol) -> Matrix{Int}

Kink counts per `(period, option)`, summed over income states. Rows are stored
periods (`sol.n_periods`); columns are discrete options.
"""
function _dcegm_kink_counts(sol::DCEGMSolution)
    nT, n_d, _ = size(sol.n_kinks)
    counts = zeros(Int, nT, n_d)
    for t in 1:nT, d in 1:n_d
        counts[t, d] = sum(@view sol.n_kinks[t, d, :])
    end
    return counts
end

# =============================================================================
# DCEGMSolution — policy and threshold views
# =============================================================================

"""
    plot_result(sol::DCEGMSolution; view=:policy, period=1, income=1, max_pts=200,
                title="", save_path=nothing)

Plot a discrete-continuous EGM solution (Iskhakov, Jørgensen, Rust & Schjerning
2017).

# Views
- `:policy` (default) — consumption vs cash-on-hand, one series per discrete
  option, for stored period `period` (default 1, which is also the stationary
  period of an infinite-horizon solution) and income state `income`. Each option
  is drawn on the union of the endogenous grids `sol.M[t, d, j]`; envelope kinks
  (consumption jumps) are kept even when the grid is subsampled to `max_pts`.
- `:threshold` — upper-envelope kink counts `sol.n_kinks` (summed over income).
  Several stored periods → grouped bar vs period, one series per option.
  A single stored period (infinite horizon, or `n_periods = 1`) → horizontal
  bar vs option.

Unknown `view` throws an `ArgumentError` listing the valid views (plotrule C5).

# Keyword arguments
- `period::Int=1` — stored period for `:policy` (`1:sol.n_periods`).
- `income::Int=1` — income-state index for `:policy`.
- `max_pts::Int=200` — cash-on-hand knot cap for `:policy` (C7-surfaced).
"""
function plot_result(sol::DCEGMSolution{T};
                     view::Symbol=:policy,
                     period::Int=1,
                     income::Int=1,
                     max_pts::Int=200,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :policy
        p = _plot_dcegm_policy(sol; period=period, income=income,
                               max_pts=max_pts, title=title)
    elseif view === :threshold
        p = _plot_dcegm_threshold(sol; title=title)
    else
        throw(ArgumentError("Unknown view: :$view. Use :policy or :threshold."))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""Consumption vs cash-on-hand, one series per discrete option."""
function _plot_dcegm_policy(sol::DCEGMSolution{T}; period::Int, income::Int,
                            max_pts::Int, title::String) where {T}
    nT = sol.n_periods
    n_d = size(sol.M, 2)
    n_e = size(sol.M, 3)
    (1 <= period <= nT) ||
        throw(ArgumentError("period $period out of range 1:$nT"))
    (1 <= income <= n_e) ||
        throw(ArgumentError("income $income out of range 1:$n_e"))

    names = _dcegm_option_names(sol)
    colors = _colors_for(names)
    dashes = String[isodd(d) ? "" : "6,3" for d in 1:n_d]

    xs = _dcegm_union_M(sol, period, income)
    n_all = length(xs)
    idxs = _dcegm_subsample_idx(xs, max_pts)
    if isempty(idxs)
        # Degenerate envelope: one padded row so the renderer has a domain.
        push!(xs, zero(T))
        idxs = [1]
        n_all = 1
    end

    rows = Vector{Pair{String,String}}[]
    for i in idxs
        m = xs[i]
        row = Pair{String,String}["x" => _json(m)]
        for d in 1:n_d
            push!(row, "d$d" => _json(_dcegm_c_at(sol.M[period, d, income],
                                                  sol.c[period, d, income], m)))
        end
        push!(rows, row)
    end

    id = _next_plot_id("dcegm_pol")
    s_json = _series_json(names, colors; keys=["d$d" for d in 1:n_d], dash=dashes)
    js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                         xlabel="Cash-on-hand", ylabel="Consumption")
    ptitle = _cap_title("Consumption by option", length(idxs), n_all)
    p1 = _PanelSpec(id, ptitle, js)

    if isempty(title)
        when = sol.prob.n_periods == 0 ? "stationary" : "period $period"
        inc = n_e > 1 ? ", income $income" : ""
        title = "DCEGM — Consumption Policy ($when$inc)"
    end
    _make_plot([p1]; title=title, ncols=1)
end

"""Upper-envelope kink counts vs period, or vs option when only one period is stored."""
function _plot_dcegm_threshold(sol::DCEGMSolution{T}; title::String) where {T}
    counts = _dcegm_kink_counts(sol)
    nT, n_d = size(counts)
    names = _dcegm_option_names(sol)
    colors = _colors_for(names)

    if nT == 1
        rows = Vector{Pair{String,String}}[]
        for d in 1:n_d
            push!(rows, ["x" => _json(names[d]), "k" => _json(counts[1, d])])
        end
        id = _next_plot_id("dcegm_kink")
        s_json = _series_json(["Kinks"], [_PLOT_SERIES[1]]; keys=["k"])
        js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                            mode="grouped", orientation="h",
                            xlabel="Kinks", ylabel="Option")
        p1 = _PanelSpec(id, "Upper-envelope kinks by option", js)
        isempty(title) && (title = "DCEGM — Switching Thresholds")
        return _make_plot([p1]; title=title, ncols=1)
    end

    rows = Vector{Pair{String,String}}[]
    for t in 1:nT
        row = Pair{String,String}["x" => _json(string(t))]
        for d in 1:n_d
            push!(row, "d$d" => _json(counts[t, d]))
        end
        push!(rows, row)
    end
    id = _next_plot_id("dcegm_kink")
    s_json = _series_json(names, colors; keys=["d$d" for d in 1:n_d])
    js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                        mode="grouped", xlabel="Period", ylabel="Kinks")
    p1 = _PanelSpec(id, "Upper-envelope kinks by period", js)
    isempty(title) && (title = "DCEGM — Switching Thresholds")
    _make_plot([p1]; title=title, ncols=1)
end
