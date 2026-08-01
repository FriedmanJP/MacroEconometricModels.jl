# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# OLG & continuous-time plots (PLT-35, #497)
#
#   • CTSteadyState        — continuous-time Aiyagari stationary equilibrium:
#                            :distribution / :policy / :lorenz (mirrors the
#                            HASteadyState `_plot_ha_*` helpers)
#   • CTTransition         — MIT-shock transition paths of the aggregates
#   • CTTwoAssetSolution   — 2-D (liquid × illiquid) policy / density heatmaps
#   • BlanchardOLGSolution — perpetual-youth saddle path in (k, C) space
#
# The bare `BlanchardOLGSteadyState` is a set of scalars → its `report` table, NOT a
# chart (plotrule form rule: a single headline number is not a chart).
# Lane-local converters below (A5); all rendering goes through the frozen render.jl.
# =============================================================================

# -----------------------------------------------------------------------------
# Lane-local converters (A5 — documented `_ct_*` helpers; NOT in helpers.jl).
# -----------------------------------------------------------------------------

"""
    _ct_marginal_mass(g, a) -> Vector{Float64}

Per-node probability **mass** of the continuous-time stationary density `g` (`I×2`,
`∫g da = 1` on a uniform grid `a`), marginalized over the two income states: mass at
node `i` is `(g[i,1]+g[i,2])·da`, normalized to sum to 1. Converting the density to
bin masses is what lets the displayed marginal conserve mass under bin-aggregation
(plotrule distributions-over-a-grid rule).
"""
function _ct_marginal_mass(g::AbstractMatrix, a::AbstractVector)
    I = size(g, 1)
    da = length(a) > 1 ? Float64(a[2] - a[1]) : 1.0
    mass = Float64[sum(@view g[i, :]) * da for i in 1:I]
    s = sum(mass)
    s > 0 && (mass ./= s)
    mass
end

"""
    _ct_bin_mass(mass, a, max_bars) -> (labels::Vector{String}, binned::Vector{Float64})

Aggregate per-node `mass` over asset nodes into at most `max_bars` contiguous bins by
**summing** (no node dropped ⇒ bin masses sum to the total mass; plotrule
Anti-Pattern #7). Bins use the same edge scheme as `_plot_ha_distribution`; each is
labelled by the asset value at its right edge.
"""
function _ct_bin_mass(mass::AbstractVector, a::AbstractVector, max_bars::Int)
    n = length(mass)
    nbins = min(n, max(1, max_bars))
    edges = round.(Int, range(0, n; length=nbins + 1))
    labels = String[]
    binned = Float64[]
    for b in 1:nbins
        lo = edges[b] + 1
        hi = edges[b + 1]
        hi < lo && continue
        push!(binned, sum(@view mass[lo:hi]))
        push!(labels, _fmt_grid_label(a[hi]))
    end
    labels, binned
end

"""
    _ct_lorenz(mass, a) -> (cum_pop, cum_wealth, gini)

Lorenz curve (cumulative population share vs cumulative wealth share) and the implied
Gini coefficient from the per-node asset `mass` and grid `a` (trapezoidal Gini
`1 − Σ pᵢ (Lᵢ₋₁ + Lᵢ)`). Both cumulative vectors have length `length(a)+1`.
"""
function _ct_lorenz(mass::AbstractVector, a::AbstractVector)
    perm = sortperm(collect(a))
    a_s = Float64.(collect(a)[perm])
    p = Float64.(collect(mass)[perm])
    s = sum(p)
    s > 0 && (p ./= s)
    n = length(a_s)
    cum_pop = zeros(Float64, n + 1)
    cum_wealth = zeros(Float64, n + 1)
    for i in 1:n
        cum_pop[i + 1] = cum_pop[i] + p[i]
        cum_wealth[i + 1] = cum_wealth[i] + p[i] * a_s[i]
    end
    tw = cum_wealth[end]
    tw > 0 && (cum_wealth ./= tw)
    gini = 1.0 - sum(p[i] * (cum_wealth[i] + cum_wealth[i + 1]) for i in 1:n; init=0.0)
    cum_pop, cum_wealth, gini
end

# =============================================================================
# CTSteadyState — continuous-time Aiyagari stationary equilibrium
# =============================================================================

"""
    plot_result(ss::CTSteadyState; view=:distribution, max_bars=60, max_pts=80,
                title="", save_path=nothing)

Plot a continuous-time Aiyagari stationary equilibrium, mirroring the `HASteadyState`
views (plotrule form rules).

# Views
- `:distribution` (default) — marginal wealth distribution (`g` summed over income),
  bin-aggregated so the displayed marginal conserves mass.
- `:policy` — consumption `c` and saving-drift `s` vs assets, one line per income
  state distinguished by **both** dash and color (plotrule Color).
- `:lorenz` — Lorenz curve of the marginal wealth distribution with the 45° equality
  reference.

Unknown `view` throws an `ArgumentError` listing the valid views (plotrule C5).
"""
function plot_result(ss::CTSteadyState{T};
                     view::Symbol=:distribution, max_bars::Int=60, max_pts::Int=80,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :distribution
        p = _plot_ct_distribution(ss; max_bars=max_bars, title=title)
    elseif view === :policy
        p = _plot_ct_policy(ss; max_pts=max_pts, title=title)
    elseif view === :lorenz
        p = _plot_ct_lorenz(ss; title=title)
    else
        throw(ArgumentError("Unknown view: :$view. Use :distribution, :policy, or :lorenz."))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""Marginal wealth distribution bar chart (mass-conserving bin-aggregation)."""
function _plot_ct_distribution(ss::CTSteadyState{T}; max_bars::Int, title::String) where {T}
    mass = _ct_marginal_mass(ss.g, ss.a)
    labels, binned = _ct_bin_mass(mass, ss.a, max_bars)
    n_a = length(mass)

    rows = Vector{Pair{String,String}}[]
    for b in eachindex(labels)
        push!(rows, ["x" => _json(labels[b]), "mass" => _json(binned[b])])
    end
    id = _next_plot_id("ct_dist")
    s_json = _series_json(["Probability mass"], [_PLOT_SERIES[1]]; keys=["mass"])
    js = _render_bar_js(id, _json_array_of_objects(rows), s_json;
                        mode="grouped", xlabel="Assets", ylabel="Probability mass")
    ptitle = _cap_title("Marginal Wealth Distribution", length(labels), n_a)
    p1 = _PanelSpec(id, ptitle, js)

    if isempty(title)
        _, _, gini = _ct_lorenz(mass, ss.a)
        title = "CT Aiyagari — Wealth Distribution (Gini = $(_fmt(gini; digits=3)), " *
                "r = $(_fmt(float(ss.r); digits=4)))"
    end
    _make_plot([p1]; title=title, ncols=1)
end

"""Consumption and saving-drift policies vs assets, one line per income state."""
function _plot_ct_policy(ss::CTSteadyState{T}; max_pts::Int, title::String) where {T}
    a = ss.a
    n_a = length(a)
    n_e = size(ss.c, 2)
    step = max(1, div(n_a, max(1, max_pts)))
    idxs = collect(1:step:n_a)
    last(idxs) != n_a && push!(idxs, n_a)

    enames = ["z$j" for j in 1:n_e]
    ecolors = _colors_for(enames)
    edash = String[isodd(j) ? "" : "6,3" for j in 1:n_e]       # dash + color per state

    panels = _PanelSpec[]
    for (fld, plabel, ylab) in ((ss.c, "Consumption Policy", "c(a, z)"),
                                (ss.s, "Saving Drift", "s(a, z)"))
        id = _next_plot_id("ct_pol")
        rows = Vector{Pair{String,String}}[]
        for i in idxs
            row = Pair{String,String}["x" => _json(a[i])]
            for j in 1:n_e
                push!(row, "e$j" => _json(fld[i, j]))
            end
            push!(rows, row)
        end
        s_json = _series_json(enames, ecolors; keys=["e$j" for j in 1:n_e], dash=edash)
        js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                             xlabel="Assets", ylabel=ylab)
        push!(panels, _PanelSpec(id, plabel, js))
    end

    isempty(title) && (title = "CT Aiyagari — Policy Functions (r = $(_fmt(float(ss.r); digits=4)))")
    _make_plot(panels; title=title, ncols=min(length(panels), 2))
end

"""Lorenz curve of the marginal wealth distribution + 45° equality line."""
function _plot_ct_lorenz(ss::CTSteadyState{T}; title::String) where {T}
    mass = _ct_marginal_mass(ss.g, ss.a)
    cum_pop, cum_wealth, gini = _ct_lorenz(mass, ss.a)

    # Subsample the Lorenz polyline for a manageable payload (endpoints kept).
    n = length(cum_pop)
    step = max(1, div(n, 200))
    idxs = collect(1:step:n)
    last(idxs) != n && push!(idxs, n)

    rows = Vector{Pair{String,String}}[]
    for i in idxs
        push!(rows, ["x" => _json(cum_pop[i]), "lorenz" => _json(cum_wealth[i]),
                     "equality" => _json(cum_pop[i])])
    end
    id = _next_plot_id("ct_lorenz")
    s_json = "[{\"name\":\"Lorenz Curve\",\"color\":$(_json(_PLOT_SERIES[1])),\"key\":\"lorenz\",\"dash\":\"\"}," *
             "{\"name\":\"Perfect Equality\",\"color\":\"#999999\",\"key\":\"equality\",\"dash\":\"6,3\"}]"
    js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                         xlabel="Cumulative Population Share",
                         ylabel="Cumulative Wealth Share")
    p1 = _PanelSpec(id, "Lorenz Curve", js)

    isempty(title) && (title = "CT Aiyagari — Lorenz Curve (Gini = $(_fmt(gini; digits=3)))")
    _make_plot([p1]; title=title, ncols=1)
end

# =============================================================================
# CTTransition — MIT-shock deterministic transition path
# =============================================================================

"""
    plot_result(tr::CTTransition; var=nothing, title="", save_path=nothing)

Multi-panel transition of a continuous-time Aiyagari economy after an unanticipated
aggregate (MIT) shock: TFP `Z`, capital `K`, rate `r`, wage `w` and aggregate
consumption `C` against time `t`, each panel carrying a horizontal reference line at
the terminal (returns-to-steady-state) value. `var` selects a single series by `Int`
or `String` (plotrule C3); `nothing` plots all five.
"""
function plot_result(tr::CTTransition{T}; var::Union{Int,String,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    names = ["Z", "K", "r", "w", "C"]
    series = Any[tr.Z, tr.K, tr.r, tr.w, tr.C]
    sel = var === nothing ? collect(1:length(names)) : [_resolve_var(var, names)]

    panels = _PanelSpec[]
    for k in sel
        vals = series[k]
        id = _next_plot_id("ct_trans")
        rows = Vector{Pair{String,String}}[]
        for i in eachindex(tr.t)
            push!(rows, ["x" => _json(tr.t[i]), "y" => _json(vals[i])])
        end
        s_json = _series_json([names[k]], [_PLOT_SERIES[mod1(k, length(_PLOT_SERIES))]]; keys=["y"])
        term = isempty(vals) ? 0.0 : Float64(vals[end])
        refs = "[{\"value\":$(_json(term)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                             ref_lines_json=refs, xlabel="Time", ylabel=names[k])
        push!(panels, _PanelSpec(id, "$(names[k]) transition", js))
    end

    isempty(title) && (title = "CT Aiyagari — MIT-Shock Transition")
    p = _make_plot(panels; title=title, ncols=min(length(panels), 3))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# CTTwoAssetSolution — 2-D (liquid × illiquid) policy / density heatmap
# =============================================================================

"""
    plot_result(sol::CTTwoAssetSolution; view=:consumption, income=1, title="",
                save_path=nothing)

Heatmap of a continuous-time two-asset policy/density surface over the liquid grid
`b` (rows) × illiquid grid `a` (columns) for one income state, with a mandatory
color-scale legend (plotrule Heatmaps).

# Views
- `:consumption` (default) → `c`  (sequential scale)
- `:density`               → `g`  (sequential scale)
- `:deposit`               → `d`  (diverging — deposits into illiquid are signed)
- `:liquid_drift`          → `sb` (diverging)
- `:illiquid_drift`        → `sa` (diverging)

`income` selects the income state (`1:size(field,3)`); an out-of-range `income` or an
unknown `view` throws an `ArgumentError` (plotrule C5).
"""
function plot_result(sol::CTTwoAssetSolution{T}; view::Symbol=:consumption,
                     income::Int=1, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    fld, scale, vlabel = if view === :consumption
        sol.c, :sequential, "Consumption c(b, a)"
    elseif view === :density
        sol.g, :sequential, "Density g(b, a)"
    elseif view === :deposit
        sol.d, :diverging, "Deposit d(b, a)"
    elseif view === :liquid_drift
        sol.sb, :diverging, "Liquid drift sb(b, a)"
    elseif view === :illiquid_drift
        sol.sa, :diverging, "Illiquid drift sa(b, a)"
    else
        throw(ArgumentError("Unknown view: :$view. Use :consumption, :density, " *
            ":deposit, :liquid_drift, or :illiquid_drift."))
    end
    n_inc = size(fld, 3)
    (1 <= income <= n_inc) ||
        throw(ArgumentError("income $income out of range 1:$n_inc"))

    M = @view fld[:, :, income]                 # Ib × Ia
    Ib, Ia = size(M)
    b_labels = String[_fmt_grid_label(sol.b[i]) for i in 1:Ib]
    a_labels = String[_fmt_grid_label(sol.a[j]) for j in 1:Ia]

    rows = Vector{Pair{String,String}}[]
    for i in 1:Ib, j in 1:Ia
        push!(rows, ["x" => _json(a_labels[j]), "y" => _json(b_labels[i]),
                     "v" => _json(M[i, j])])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ct2a")
    js = _render_heatmap_js(id, data_json, _json(reverse(b_labels)), _json(a_labels);
                            xlabel="Illiquid a", ylabel="Liquid b", tip_label="a",
                            scale=scale, midpoint=0)
    p1 = _PanelSpec(id, "$(vlabel) — income state $(income)", js)

    isempty(title) && (title = "CT Two-Asset Model — $(vlabel)")
    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BlanchardOLGSolution — perpetual-youth saddle path
# =============================================================================

"""
    plot_result(sol::BlanchardOLGSolution; k_span=0.4, title="", save_path=nothing)

Saddle-path phase plot of a Blanchard (1985) perpetual-youth OLG model in
`(k, C)` space: the stable manifold `C = C* + policy_slope·(k − k*)` drawn through the
steady state over `k* ± k_span·k*`, the steady state marked with a scatter point and a
reference circle, and `stable_eig` / `determinate` annotated in the panel title
(rounded, plotrule C9). The bare `BlanchardOLGSteadyState` (a set of scalars) has no
chart — read it through `report` (plotrule form rule).
"""
function plot_result(sol::BlanchardOLGSolution{T}; k_span::Real=0.4,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    kstar = Float64(sol.ss.k)
    cstar = Float64(sol.ss.C)
    slope = Float64(sol.policy_slope)
    span = abs(k_span) * (kstar == 0 ? 1.0 : abs(kstar))
    k0 = kstar - span
    k1 = kstar + span
    c0 = cstar + slope * (k0 - kstar)
    c1 = cstar + slope * (k1 - kstar)

    id = _next_plot_id("blanchard")
    data_json = _json_array_of_objects([
        ["x" => _json(kstar), "y" => _json(cstar), "group" => _json("Steady state")]])
    groups_json = "[{\"name\":\"Steady state\",\"color\":$(_json(_PLOT_ALERT))}]"
    overlay = "[{\"x1\":$(_json(k0)),\"y1\":$(_json(c0))," *
              "\"x2\":$(_json(k1)),\"y2\":$(_json(c1))," *
              "\"color\":$(_json(_PLOT_SERIES[1])),\"dash\":\"\"}]"
    rad = 0.04 * (span == 0 ? 1.0 : span)
    shapes = "[{\"type\":\"circle\",\"cx\":$(_json(kstar)),\"cy\":$(_json(cstar))," *
             "\"r\":$(_json(rad)),\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"3,3\"}]"
    js = _render_scatter_js(id, data_json, groups_json;
                            line_overlays_json=overlay, ref_shapes_json=shapes,
                            xlabel="Capital k", ylabel="Consumption C")
    det = sol.determinate ? "determinate" : "indeterminate"
    ptitle = "Saddle path (stable eig = $(_fmt(float(sol.stable_eig); digits=3)), $(det))"
    p1 = _PanelSpec(id, ptitle, js)

    isempty(title) && (title = "Blanchard OLG — Saddle Path")
    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# LifeCycleSteadyState — age profiles, wealth by age, policies ([T255]/#354)
# =============================================================================

"""
    plot_result(ss::LifeCycleSteadyState; view=:profiles, title="", save_path=nothing)

Visualize a life-cycle OLG stationary equilibrium.

| `view` | Panels |
|---|---|
| `:profiles` (default) | Mean assets, consumption, and income by age, plus the cohort mass |
| `:distribution` | Mean and interquartile wealth by age — the spread the mean profile hides |
| `:policy` | Consumption policy by asset level at three representative ages |

The retirement age is marked with a dashed reference line on the age-indexed
panels, since every profile turns there.
"""
function plot_result(ss::LifeCycleSteadyState{T}; view::Symbol=:profiles,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view in (:profiles, :distribution, :policy) || throw(ArgumentError(
        "plot_result(::LifeCycleSteadyState): view must be :profiles, :distribution " *
        "or :policy (got :$view)"))
    p = if view === :profiles
        _plot_lc_profiles(ss; title=title)
    elseif view === :distribution
        _plot_lc_distribution(ss; title=title)
    else
        _plot_lc_policy(ss; title=title)
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"Reference line at the retirement age, shared by the age-indexed panels."
_lc_retire_ref(ss::LifeCycleSteadyState) =
    ss.spec.J_retire <= ss.spec.J ?
    "[{\"axis\":\"x\",\"value\":$(_json(float(ss.spec.J_retire)))," *
    "\"color\":\"#999999\",\"dash\":\"5,4\",\"label\":\"retirement\"}]" : "[]"

function _plot_lc_profiles(ss::LifeCycleSteadyState{T}; title::String) where {T}
    J = ss.spec.J
    ages = collect(1:J)
    panels = _PanelSpec[]
    refs = _lc_retire_ref(ss)

    id = _next_plot_id("lc_prof")
    rows = [Pair{String,String}["x" => _json(float(j)),
                                "assets" => _json(float(ss.asset_profile[j])),
                                "cons" => _json(float(ss.consumption_profile[j])),
                                "inc" => _json(float(ss.income_profile[j]))] for j in ages]
    s_json = _series_json(["Assets", "Consumption", "Income"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2], _PLOT_SERIES[3]];
                          keys=["assets", "cons", "inc"])
    push!(panels, _PanelSpec(id, "Age Profiles",
        _render_line_js(id, _json_array_of_objects(rows), s_json;
                        ref_lines_json=refs, integer_x=true,
                        xlabel="Age", ylabel="Level")))

    id2 = _next_plot_id("lc_mass")
    rows2 = [Pair{String,String}["x" => _json(float(j)),
                                 "mass" => _json(float(ss.cohort_mass[j]))] for j in ages]
    push!(panels, _PanelSpec(id2, "Population by Age",
        _render_line_js(id2, _json_array_of_objects(rows2),
                        _series_json(["Cohort mass"], [_PLOT_SERIES[4]]; keys=["mass"]);
                        ref_lines_json=refs, integer_x=true,
                        xlabel="Age", ylabel="Population share")))

    isempty(title) && (title = "Life-Cycle OLG — Age Profiles (r = $(_fmt(float(ss.r); digits=4)))")
    _make_plot(panels; title=title, ncols=2)
end

function _plot_lc_distribution(ss::LifeCycleSteadyState{T}; title::String) where {T}
    J = ss.spec.J
    a_grid = ss.spec.grid.grids[1]
    lo = zeros(Float64, J); hi = zeros(Float64, J); med = zeros(Float64, J)
    for j in 1:J
        marg = vec(sum(@view(ss.dist[:, :, j]); dims=2))
        m = sum(marg)
        if m <= 0
            lo[j] = hi[j] = med[j] = 0.0
            continue
        end
        cum = cumsum(marg) ./ m
        q(p) = a_grid[clamp(searchsortedfirst(cum, p), 1, length(a_grid))]
        lo[j] = q(0.25); med[j] = q(0.50); hi[j] = q(0.75)
    end

    id = _next_plot_id("lc_dist")
    rows = [Pair{String,String}["x" => _json(float(j)),
                                "mean" => _json(float(ss.asset_profile[j])),
                                "med" => _json(med[j])] for j in 1:J]
    bands = "[{\"key\":\"mean\",\"lower\":[" *
            join((_json(lo[j]) for j in 1:J), ",") * "],\"upper\":[" *
            join((_json(hi[j]) for j in 1:J), ",") * "]}]"
    s_json = _series_json(["Mean wealth", "Median wealth"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2]]; keys=["mean", "med"],
                          dash=["", "5,3"])
    js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                         bands_json=bands, ref_lines_json=_lc_retire_ref(ss),
                         integer_x=true, xlabel="Age", ylabel="Assets")
    p1 = _PanelSpec(id, "Wealth by Age (band = interquartile range)", js)

    isempty(title) && (title = "Life-Cycle OLG — Wealth Distribution by Age")
    _make_plot([p1]; title=title, ncols=1)
end

function _plot_lc_policy(ss::LifeCycleSteadyState{T}; title::String) where {T}
    J = ss.spec.J
    a_grid = ss.spec.grid.grids[1]
    n_a = length(a_grid)
    step = max(1, div(n_a, 150))
    idxs = collect(1:step:n_a)
    last(idxs) != n_a && push!(idxs, n_a)

    # Young, just-before-retirement, and old — where the policy differs most.
    sel = unique(clamp.([max(1, div(J, 6)), max(1, ss.spec.J_retire - 1),
                         max(1, J - 1)], 1, J))
    e_mid = cld(length(ss.spec.income.states), 2)
    labels = ["Age $(j)" for j in sel]
    rows = Vector{Pair{String,String}}[]
    for i in idxs
        row = Pair{String,String}["x" => _json(float(a_grid[i]))]
        for (k, j) in enumerate(sel)
            push!(row, "a$k" => _json(float(ss.c_policy[i, e_mid, j])))
        end
        push!(rows, row)
    end
    id = _next_plot_id("lc_pol")
    s_json = _series_json(labels, [_PLOT_SERIES[k] for k in 1:length(sel)];
                          keys=["a$k" for k in 1:length(sel)])
    js = _render_line_js(id, _json_array_of_objects(rows), s_json;
                         xlabel="Assets", ylabel="Consumption")
    p1 = _PanelSpec(id, "Consumption Policy by Age (median productivity)", js)

    isempty(title) && (title = "Life-Cycle OLG — Consumption Policy")
    _make_plot([p1]; title=title, ncols=1)
end
