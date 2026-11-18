# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for filter types: HPFilter, Hamilton, BN, BK, BoostedHP.
"""

# =============================================================================
# Shared Filter Plotting
# =============================================================================

"""
Generate a 2-panel figure for a filter result: (1) Original + Trend, (2) Cycle.

- `tr`: trend component
- `cyc`: cycle component
- `filter_name`: display name
- `original`: original series (optional). When absent and `offset > 0` (Hamilton/BK
  drop leading observations) the pre-trim original **cannot** be recovered from the
  trimmed trend/cycle, so the Original line is **omitted** rather than fabricated
  (plotrule Anti-Pattern #5); a subtitle notes it. When absent and `offset == 0` it
  is reconstructed as `trend + cycle` and labelled "Original (reconstructed)".
- `offset`: index offset for shorter filters (Hamilton, BK)
- `T_obs`: original series length
- `dates`: optional calendar labels (length ≥ `offset + n`) aligned to the *full*
  series so each drawn point shows its calendar position (PLT-08).
"""
function _plot_filter_panels(tr::AbstractVector, cyc::AbstractVector,
                             filter_name::String;
                             original::Union{AbstractVector,Nothing}=nothing,
                             offset::Int=0, T_obs::Int=0,
                             dates::Union{AbstractVector,Nothing}=nothing)
    n = length(tr)
    reconstructed = false
    if original !== nothing
        orig = original
        has_original = true
    elseif offset == 0
        orig = tr .+ cyc
        has_original = true
        reconstructed = true
    else
        orig = nothing
        has_original = false
    end

    data_json = _filter_data_json(tr, cyc; original=orig, offset=offset)

    # Date axis: label calendar positions offset+1 .. offset+n with dates.
    xlabel = "Period"
    x_ticks_json = "null"
    if dates !== nothing
        if length(dates) < offset + n
            throw(ArgumentError("dates has length $(length(dates)); expected at least " *
                "$(offset + n) to cover the filter's calendar range (offset=$offset, n=$n)"))
        end
        xvals = collect((1:n) .+ offset)
        x_ticks_json = _x_ticks_json(xvals, dates[(1:n) .+ offset])
        xlabel = "Date"
    end

    # Panel 1: Original (or reconstructed) + Trend
    id1 = _next_plot_id("filt_tc")
    if has_original
        orig_label = reconstructed ? "Original (reconstructed)" : "Original"
        s1 = _series_json([orig_label, "Trend"], [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["orig", "trend"], dash=["", "6,3"])
    else
        s1 = _series_json(["Trend"], [_PLOT_COLORS[2]]; keys=["trend"], dash=["6,3"])
    end
    js1 = _render_line_js(id1, data_json, s1; xlabel=xlabel, ylabel="Value",
                          x_ticks_json=x_ticks_json)
    subtitle1 = if !has_original
        "Trend-Cycle Decomposition (original not supplied; pass original= to overlay it)"
    elseif reconstructed
        "Trend-Cycle Decomposition (original = trend + cycle)"
    else
        "Trend-Cycle Decomposition"
    end
    p1 = _PanelSpec(id1, subtitle1, js1)

    # Panel 2: Cycle + zero line
    id2 = _next_plot_id("filt_cy")
    s2 = _series_json(["Cycle"], [_PLOT_COLORS[3]]; keys=["cycle"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data_json, s2;
                          ref_lines_json=refs, xlabel=xlabel, ylabel="Cycle",
                          x_ticks_json=x_ticks_json)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "HP Filter";
                                 original=r.trend .+ r.cycle, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "Hamilton Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.permanent, r.transitory, "BN Decomposition";
                                 original=r.permanent .+ r.transitory, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    offset = r.valid_range.start - 1
    panels = _plot_filter_panels(r.trend, r.cycle, "BK Filter";
                                 original=original, offset=offset,
                                 T_obs=r.T_obs, dates=dates)
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
                     dates::Union{Vector{String},Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    panels = _plot_filter_panels(r.trend, r.cycle, "Boosted HP";
                                 original=r.trend .+ r.cycle, dates=dates)
    if isempty(title)
        title = "Boosted HP Filter ($(r.stopping), $(r.iterations) iter)"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# X13FilterResult
# =============================================================================

"""
    plot_result(r::X13FilterResult; title="", save_path=nothing)

Plot X-13 seasonal adjustment: (1) Original + Trend + SA, (2) Seasonal, (3) Irregular.
"""
function plot_result(r::X13FilterResult{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    n = r.T_obs

    # Panel 1: Original + Trend + SA
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows1, ["x" => _json(i), "orig" => _json(r.original[i]),
                       "trend" => _json(r.trend[i]), "sa" => _json(r.adjusted[i])])
    end
    data1 = _json_array_of_objects(rows1)
    id1 = _next_plot_id("x13_tc")
    s1 = _series_json(["Original", "Trend", "Seasonally Adjusted"],
                       [_PLOT_COLORS[1], _PLOT_COLORS[2], _PLOT_COLORS[4]];
                       keys=["orig", "trend", "sa"], dash=["", "6,3", "3,3"])
    js1 = _render_line_js(id1, data1, s1; xlabel="Period", ylabel="Value")
    p1 = _PanelSpec(id1, "Trend-Cycle Decomposition", js1)

    # Panel 2: Seasonal component
    rows2 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows2, ["x" => _json(i), "seasonal" => _json(r.seasonal[i])])
    end
    data2 = _json_array_of_objects(rows2)
    id2 = _next_plot_id("x13_seas")
    s2 = _series_json(["Seasonal"], [_PLOT_COLORS[3]]; keys=["seasonal"])
    ref_val = r.transform == :log ? "1" : "0"
    refs2 = "[{\"value\":$ref_val,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js2 = _render_line_js(id2, data2, s2; ref_lines_json=refs2,
                          xlabel="Period", ylabel="Seasonal")
    p2 = _PanelSpec(id2, "Seasonal Component", js2)

    # Panel 3: Irregular component
    rows3 = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows3, ["x" => _json(i), "irregular" => _json(r.irregular[i])])
    end
    data3 = _json_array_of_objects(rows3)
    id3 = _next_plot_id("x13_irr")
    s3 = _series_json(["Irregular"], [_PLOT_COLORS[5]]; keys=["irregular"])
    js3 = _render_line_js(id3, data3, s3; ref_lines_json=refs2,
                          xlabel="Period", ylabel="Irregular")
    p3 = _PanelSpec(id3, "Irregular Component", js3)

    if isempty(title)
        p_o, d_o, q_o, P_o, D_o, Q_o = r.arima_order
        title = "X-13ARIMA-SEATS ($(uppercase(string(r.method))), ($p_o,$d_o,$q_o)($P_o,$D_o,$Q_o)_$(r.frequency))"
    end
    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Rational Inattention & Steady-State Kalman Plots (#923)
# =============================================================================

# Validate conf_level in (0, 1) and return (z_critical, ci_percent).
function _ri_conf_crit(conf_level::Real)
    c = Float64(conf_level)
    (isfinite(c) && 0 < c < 1) || throw(ArgumentError(
        "conf_level must be in (0, 1); got $conf_level"))
    z = quantile(Normal(), 1 - (1 - c) / 2)
    return z, round(Int, 100 * c)
end

# Build [{name, effect, ci_lo, ci_hi}] JSON rows for _render_coef_plot_js.
function _ri_coef_data_json(names::AbstractVector{<:AbstractString},
                            ests::AbstractVector{<:Real},
                            ses::AbstractVector{<:Real},
                            z::Real)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(names)
        est = ests[i]
        se = ses[i]
        push!(rows, [
            "name"   => _json(String(names[i])),
            "effect" => _json(est),
            "ci_lo"  => _json(est - z * se),
            "ci_hi"  => _json(est + z * se),
        ])
    end
    _json_array_of_objects(rows)
end

# Build scatter rows for complex closed-loop eigenvalues relative to the unit circle.
function _ri_eigs_scatter_json(eigs::AbstractVector)
    rows = Vector{Pair{String,String}}[]
    used = Set{String}()
    all_inside = true
    for lambda in eigs
        m = abs(lambda)
        inside = isfinite(m) && m < 1
        inside || (all_inside = false)
        grp = inside ? "Inside unit circle" : "On/outside unit circle"
        push!(used, grp)
        push!(rows, ["x" => _json(real(lambda)),
                     "y" => _json(imag(lambda)),
                     "group" => _json(grp)])
    end
    return _json_array_of_objects(rows), used, all_inside
end

# Exact full-information (κ = ∞) action IRF array of size H × d × m.
function _ri_fullinfo_action_irf(prob::RationalInattentionProblem{T}, H::Int) where {T}
    n = size(prob.A, 1)
    m = size(prob.B, 2)
    d = size(prob.H, 2)
    A = Matrix{Float64}(prob.A)
    B = Matrix{Float64}(prob.B)
    F = Matrix{Float64}(prob.H')
    Bu = prob.B_u === nothing ? zeros(Float64, n, d) : Matrix{Float64}(prob.B_u)
    Acl = A + Bu * F
    out = Array{Float64,3}(undef, H, d, m)
    for si in 1:m
        x = copy(@view B[:, si])
        out[1, :, si] .= F * x
        for h in 2:H
            x = Acl * x
            out[h, :, si] .= F * x
        end
    end
    out
end

# Assemble capacity-indexed action IRF panels for SimsKalmanSolution.
function _plot_ri_irf_by_capacity(sol::SimsKalmanSolution{T};
                                  capacities::AbstractVector{<:Real},
                                  horizon::Int,
                                  var::Union{Int,String,Nothing},
                                  shock::Union{Int,String,Nothing},
                                  ncols::Int,
                                  title::String) where {T}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
    for c in capacities
        isnan(c) && throw(ArgumentError("capacities cannot contain NaN"))
        c >= 0 || throw(ArgumentError("capacities entries must be >= 0 (or Inf), got $c"))
    end

    prob = sol.problem
    d = size(prob.H, 2)
    m = size(prob.B, 2)
    vars_to_plot = var === nothing ? collect(1:d) : [_resolve_var(var, prob.action_names)]
    shocks_to_plot = shock === nothing ? collect(1:m) : [_resolve_var(shock, prob.shock_names)]

    cur_cap = Float64(sol.capacity_bits)
    cur_irf = Float64.(irf(sol, horizon; observables=:actions).values[:, :, 1:m])

    # Entry tuple: (sort_key, label, values_Hdm, dash)
    entries = NamedTuple{(:cap, :label, :vals, :dash), Tuple{Float64, String, Array{Float64,3}, String}}[]
    seen_caps = Float64[]
    matched_current = false

    for c_raw in capacities
        c = Float64(c_raw)
        any(x -> (isinf(x) && isinf(c)) || (isfinite(x) && isfinite(c) && isapprox(x, c; atol=1e-4)), seen_caps) && continue
        push!(seen_caps, c)

        if isinf(c)
            push!(entries, (cap=Inf, label="Full info (κ = ∞)", vals=_ri_fullinfo_action_irf(prob, horizon), dash="5,3"))
        elseif isfinite(cur_cap) && isapprox(c, cur_cap; atol=1e-4)
            matched_current = true
            push!(entries, (cap=cur_cap, label="κ = $(_fmt(cur_cap; digits=3)) bits (current)", vals=cur_irf, dash=""))
        elseif c == 0.0
            push!(entries, (cap=0.0, label="κ = 0 bits", vals=zeros(Float64, horizon, d, m), dash="2,2"))
        else
            try
                prob_c = RationalInattentionProblem(
                    Matrix{Float64}(prob.A), Matrix{Float64}(prob.B), Matrix{Float64}(prob.H);
                    M=Matrix{Float64}(prob.M),
                    B_u=prob.B_u === nothing ? nothing : Matrix{Float64}(prob.B_u),
                    beta=Float64(prob.beta), capacity=c, units=:bits,
                    state_names=prob.state_names, action_names=prob.action_names, shock_names=prob.shock_names,
                )
                sol_c = solve_ri_kalman(prob_c)
                vals_c = Float64.(irf(sol_c, horizon; observables=:actions).values[:, :, 1:m])
                push!(entries, (cap=c, label="κ = $(_fmt(c; digits=3)) bits", vals=vals_c, dash=""))
            catch e
                e isa ArgumentError || rethrow()
            end
        end
    end

    if !matched_current
        push!(entries, (cap=cur_cap, label="κ = $(_fmt(cur_cap; digits=3)) bits (current)", vals=cur_irf, dash=""))
    end
    sort!(entries; by=e -> e.cap)

    labels = String[e.label for e in entries]
    colors = _colors_for(labels)
    dashes = String[e.dash for e in entries]
    keys = String["k$(ci)" for ci in eachindex(entries)]
    s_json = _series_json(labels, colors; keys=keys, dash=dashes)
    xs = collect(0:(horizon - 1))
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    panels = _PanelSpec[]
    for ai in vars_to_plot
        for si in shocks_to_plot
            id = _next_plot_id("ri_cap_irf")
            cols = Pair{String,Vector{Float64}}[keys[ci] => entries[ci].vals[:, ai, si] for ci in eachindex(entries)]
            data_json = _keyed_rows_json(xs, cols)
            js = _render_line_js(id, data_json, s_json;
                                 ref_lines_json=refs, integer_x=true,
                                 xlabel="Horizon", ylabel="Action Response")
            push!(panels, _PanelSpec(id, "$(prob.action_names[ai]) ← $(prob.shock_names[si])", js))
        end
    end

    ncols_eff = ncols <= 0 ? (length(shocks_to_plot) == 1 ? min(length(vars_to_plot), 3) : length(shocks_to_plot)) : ncols
    ftitle = isempty(title) ?
        "RI Action IRFs by Capacity (current κ = $(_fmt(cur_cap; digits=3)) bits)" : title
    _make_plot(panels; title=ftitle, ncols=ncols_eff)
end

# Compute whitened precision-benefit eigenvalues safely (returns NaN vector on non-finite input).
function _ri_wf_spectrum(M::Matrix{Float64}, Oeff::Matrix{Float64})
    n = size(M, 1)
    (all(isfinite, M) && all(isfinite, Oeff)) || return fill(NaN, n)
    _, _, d, _ = _ri_whitened_modes(M, Oeff)
    return d
end

# Assemble prior vs posterior precision eigenvalues against threshold ω/2.
function _plot_ri_water_filling(sol::SimsKalmanSolution{T}; title::String) where {T}
    n = size(sol.Sigma_prior, 1)
    Sp = _ri_symmetrize(Matrix{Float64}(sol.Sigma_prior))
    Spost = _ri_symmetrize(Matrix{Float64}(sol.Sigma_post))
    Oeff = _ri_symmetrize(Matrix{Float64}(sol.Omega_benefit))
    d_prior = _ri_wf_spectrum(Sp, Oeff)
    d_post = _ri_wf_spectrum(Spost, Oeff)

    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, ["x" => _json(i), "y" => _json(d_prior[i]), "group" => _json("Prior eigenvalue")])
        push!(rows, ["x" => _json(i), "y" => _json(d_post[i]), "group" => _json("Posterior eigenvalue")])
    end
    data_json = _json_array_of_objects(rows)
    groups_json = "[{\"name\":\"Prior eigenvalue\",\"color\":$(_json(_PLOT_SERIES[1]))}," *
                  "{\"name\":\"Posterior eigenvalue\",\"color\":$(_json(_PLOT_SERIES[2]))}]"

    curves_json = if n >= 2
        pts_prior = _json_array_of_objects([["x" => _json(i), "y" => _json(d_prior[i])] for i in 1:n])
        pts_post = _json_array_of_objects([["x" => _json(i), "y" => _json(d_post[i])] for i in 1:n])
        "[{\"points\":$(pts_prior),\"color\":$(_json(_PLOT_SERIES[1])),\"dash\":\"\"}," *
        "{\"points\":$(pts_post),\"color\":$(_json(_PLOT_SERIES[2])),\"dash\":\"5,3\"}]"
    else
        "[]"
    end

    thresh = Float64(sol.omega) / 2
    ref_lines = isfinite(thresh) ?
        "[{\"value\":$(_json(thresh)),\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"6,3\"}]" :
        "[]"

    id = _next_plot_id("ri_wf")
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=ref_lines,
                            curve_overlays_json=curves_json,
                            integer_x=true,
                            xlabel="Mode", ylabel="Marginal Value of Precision")
    ptitle = "Water-Filling Spectrum (ω/2 = $(_fmt(thresh; digits=4)), active rank = $(sol.signal_rank)/$(n))"
    ftitle = isempty(title) ? "RI Water-Filling Precision Eigenvalues" : title
    _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=1)
end

"""
    plot_result(prob::RationalInattentionProblem; title="", save_path=nothing)

Plot a 2-panel diverging heatmap of the state transition matrix `A` (`n × n`,
`state_names × state_names`) and the target action loading matrix `H` (`n × d`,
`state_names × action_names`), each with a zero-centered diverging color legend.
"""
function plot_result(prob::RationalInattentionProblem{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    snames = prob.state_names
    anames = prob.action_names

    id1 = _next_plot_id("ri_A")
    data1 = _matrix_heat_json(Matrix{Float64}(prob.A), snames, snames)
    js1 = _render_heatmap_js(id1, data1, _json(snames), _json(snames);
                             scale=:diverging, midpoint=0,
                             xlabel="From State", ylabel="To State", tip_label="A")
    p1 = _PanelSpec(id1, "State Transition A", js1)

    id2 = _next_plot_id("ri_H")
    data2 = _matrix_heat_json(Matrix{Float64}(prob.H), snames, anames)
    js2 = _render_heatmap_js(id2, data2, _json(snames), _json(anames);
                             scale=:diverging, midpoint=0,
                             xlabel="Action", ylabel="State", tip_label="H")
    p2 = _PanelSpec(id2, "Target Action Loading H", js2)

    if isempty(title)
        spec_str = prob.capacity !== nothing ?
            "κ = $(_fmt(prob.capacity; digits=3)) $(prob.units)" :
            "ω = $(_fmt(prob.omega; digits=3)) / nat"
        title = "Rational Inattention Problem ($(spec_str))"
    end
    p = _make_plot([p1, p2]; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(ssk::SteadyStateKalman; title="", save_path=nothing)

Plot the closed-loop Kalman transition eigenvalues `ssk.closed_loop_eigs` in
the complex plane against a dashed unit-circle reference (`_render_scatter_js`),
flagging roots with `|λ| ≥ 1` in the alert color.
"""
function plot_result(ssk::SteadyStateKalman{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    data_json, used, all_inside = _ri_eigs_scatter_json(ssk.closed_loop_eigs)
    order = ["Inside unit circle" => _PLOT_SERIES[1],
             "On/outside unit circle" => _PLOT_ALERT]
    gparts = String[]
    for (nm, col) in order
        nm in used || continue
        push!(gparts, "{\"name\":$(_json(nm)),\"color\":$(_json(col))}")
    end
    groups_json = "[" * join(gparts, ",") * "]"
    shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    refs = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#ccc\",\"dash\":\"2,2\"}," *
           "{\"value\":0,\"axis\":\"y\",\"color\":\"#ccc\",\"dash\":\"2,2\"}]"
    id = _next_plot_id("ssk_eigs")
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=refs, ref_shapes_json=shapes,
                            xlabel="Real", ylabel="Imaginary")

    finite_mods = Float64[abs(z) for z in ssk.closed_loop_eigs if isfinite(abs(z))]
    rho_str = isempty(finite_mods) ? "NaN" : _fmt(maximum(finite_mods); digits=4)
    status = all_inside ? "STABLE" : "UNSTABLE"
    ptitle = "Closed-Loop Eigenvalues ($(status) — max |λ| = $(rho_str))"
    isempty(title) && (title = "Steady-State Kalman Filter")

    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(sol::SimsKalmanSolution; view=:irf_by_capacity,
                capacities=[0.25, 1.0, Inf], horizon=20,
                var=nothing, shock=nothing, ncols=0,
                title="", save_path=nothing)

Visualize a steady-state LQG Rational Inattention Kalman solution (`SimsKalmanSolution`).
`view` selects:

- `:irf_by_capacity` (default) — action impulse responses to fundamental shocks across
  `capacities` (in bits, including full information `Inf`) plus `sol.capacity_bits`.
  Accepts `horizon`, `var` (action index or name), `shock` (fundamental shock index or name),
  and `ncols`.
- `:fevd` — stacked-area forecast error variance decomposition from `fevd(sol, horizon)`
  partitioning action variance into fundamental shocks vs. endogenous RI processing noise.
  Accepts `horizon`, `var`, and `ncols` (`shock` errors).
- `:water_filling` — prior vs. posterior whitened precision-benefit eigenvalues against
  the dashed shadow-cost threshold `sol.omega / 2` (`var` and `shock` error).

Unknown `view` throws `ArgumentError`.
"""
function plot_result(sol::SimsKalmanSolution{T};
                     view::Symbol=:irf_by_capacity,
                     capacities::AbstractVector{<:Real}=[0.25, 1.0, Inf],
                     horizon::Int=20,
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    p = if view === :irf_by_capacity
        _plot_ri_irf_by_capacity(sol; capacities=capacities, horizon=horizon,
                                 var=var, shock=shock, ncols=ncols, title=title)
    elseif view === :fevd
        shock === nothing || throw(ArgumentError(
            "shock does not apply to view=:fevd (all shocks are stacked)"))
        horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
        f = fevd(sol, horizon; observables=:actions)
        ftitle = isempty(title) ?
            "RI Action FEVD — Fundamental vs. Processing Noise (κ = $(_fmt(sol.capacity_bits; digits=3)) bits)" : title
        plot_result(f; var=var, ncols=ncols, title=ftitle)
    elseif view === :water_filling
        (var === nothing && shock === nothing) || throw(ArgumentError(
            "var and shock do not apply to view=:water_filling"))
        _plot_ri_water_filling(sol; title=title)
    else
        throw(ArgumentError(
            "Unknown view :$view. Valid views: :irf_by_capacity, :fevd, :water_filling"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(sol::DynamicRISolution; view=:attention_path, title="", save_path=nothing)

Visualize an Afrouzi–Yang (2021) dynamic Rational Inattention transition path (`DynamicRISolution`):

- `:attention_path` (default) — 2-panel figure of per-period information flow `capacity_bits_path`
  and uncertainty traces `tr(Σ_{t|t-1})` (dashed) vs. `tr(Σ_{t|t})` (solid).
- `:omega` — 2-panel figure of shadow information cost `omega_path` and active endogenous signal
  dimension `signal_rank_path`.

Unknown `view` throws `ArgumentError`.
"""
function plot_result(sol::DynamicRISolution{T};
                     view::Symbol=:attention_path,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    xs = collect(1:sol.T_path)
    refs_zero = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    panels = if view === :attention_path
        id1 = _next_plot_id("dynri_cap")
        data1 = _keyed_rows_json(xs, ["cap" => sol.capacity_bits_path])
        s1 = _series_json(["Capacity (bits)"], [_PLOT_SERIES[1]]; keys=["cap"])
        js1 = _render_line_js(id1, data1, s1; ref_lines_json=refs_zero, integer_x=true,
                              xlabel="Period", ylabel="Capacity (bits)")
        p1 = _PanelSpec(id1, "Endogenous Information Flow", js1)

        tr_prior = Float64[tr(@view sol.Sigma_prior_path[:, :, t]) for t in xs]
        tr_post = Float64[tr(@view sol.Sigma_post_path[:, :, t]) for t in xs]
        id2 = _next_plot_id("dynri_tr")
        data2 = _keyed_rows_json(xs, ["prior" => tr_prior, "post" => tr_post])
        s2 = _series_json(["Prior tr(Σ_{t|t-1})", "Posterior tr(Σ_{t|t})"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                          keys=["prior", "post"], dash=["6,3", ""])
        js2 = _render_line_js(id2, data2, s2; ref_lines_json=refs_zero, integer_x=true,
                              xlabel="Period", ylabel="Trace")
        p2 = _PanelSpec(id2, "State Uncertainty Trace", js2)
        [p1, p2]
    elseif view === :omega
        id1 = _next_plot_id("dynri_om")
        data1 = _keyed_rows_json(xs, ["om" => sol.omega_path])
        s1 = _series_json(["Shadow cost ω_t"], [_PLOT_SERIES[1]]; keys=["om"])
        js1 = _render_line_js(id1, data1, s1; ref_lines_json=refs_zero, integer_x=true,
                              xlabel="Period", ylabel="Shadow Cost ω_t (/nat)")
        p1 = _PanelSpec(id1, "Marginal Attention Shadow Cost", js1)

        id2 = _next_plot_id("dynri_rk")
        data2 = _keyed_rows_json(xs, ["rk" => Float64.(sol.signal_rank_path)])
        s2 = _series_json(["Signal rank r_s"], [_PLOT_SERIES[3]]; keys=["rk"])
        js2 = _render_line_js(id2, data2, s2; ref_lines_json=refs_zero, integer_x=true,
                              xlabel="Period", ylabel="Active Rank")
        p2 = _PanelSpec(id2, "Endogenous Signal Rank", js2)
        [p1, p2]
    else
        throw(ArgumentError(
            "Unknown view :$view. Valid views: :attention_path, :omega"))
    end

    if isempty(title)
        title = view === :attention_path ?
            "Dynamic RI Attention & Uncertainty Path (T = $(sol.T_path))" :
            "Dynamic RI Shadow Cost & Signal Rank (T = $(sol.T_path))"
    end
    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::MWAttentionResult; title="", save_path=nothing)

Plot a 2-panel Maćkowiak–Wiederholt (2009, 2015) equilibrium impulse-response figure:
(1) aggregate price response `agg_price_irf` (solid) vs. idiosyncratic price response
`idio_price_irf` (dashed) and (2) real output gap response `real_output_irf`, each on
an integer horizon axis from 0 with a horizontal zero line.
"""
function plot_result(r::MWAttentionResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    H = length(r.agg_price_irf)
    hs = collect(0:(H - 1))
    refs_zero = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"

    id1 = _next_plot_id("mw_price")
    data1 = _keyed_rows_json(hs, ["agg" => r.agg_price_irf, "idio" => r.idio_price_irf])
    s1 = _series_json(["Aggregate price p_h", "Idiosyncratic price p_h^idio"],
                      [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                      keys=["agg", "idio"], dash=["", "6,3"])
    js1 = _render_line_js(id1, data1, s1; ref_lines_json=refs_zero, integer_x=true,
                          xlabel="Horizon", ylabel="Price Response")
    p1 = _PanelSpec(id1, "Price Impulse Responses (1-s.d. Shocks)", js1)

    id2 = _next_plot_id("mw_output")
    data2 = _keyed_rows_json(hs, ["out" => r.real_output_irf])
    s2 = _series_json(["Real output y_h"], [_PLOT_SERIES[3]]; keys=["out"])
    js2 = _render_line_js(id2, data2, s2; ref_lines_json=refs_zero, integer_x=true,
                          xlabel="Horizon", ylabel="Output Response")
    p2 = _PanelSpec(id2, "Real Output Response (y_h = q_h − p_h)", js2)

    if isempty(title)
        title = "Maćkowiak–Wiederholt Equilibrium (κ_agg = $(_fmt(r.kappa_agg; digits=3)), κ_idio = $(_fmt(r.kappa_idio; digits=3)) $(r.units), r = $(_fmt(r.strategic_complementarity; digits=2)))"
    end
    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::RIEstimationResult; conf_level=0.95, title="", save_path=nothing)

Horizontal dot-and-whisker coefficient plot (`_render_coef_plot_js`) of the estimated
Rational Inattention parameters `r.params ± z * r.stderr` at confidence level
`conf_level` (default `0.95`), with a zero reference line.
"""
function plot_result(r::RIEstimationResult{T};
                     conf_level::Real=0.95,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    z, pct = _ri_conf_crit(conf_level)
    id = _next_plot_id("ri_est")
    data_json = _ri_coef_data_json(r.param_names, r.params, r.stderr, z)
    js = _render_coef_plot_js(id, data_json; ref_value=0, xlabel="Estimate", ylabel="")
    meth = r.method === :mle ? "MLE" : "Bayesian"
    ptitle = "Parameter Estimates ($(meth), $(pct)% CI; κ = $(_fmt(r.capacity; digits=3)), LR p = $(_format_pvalue(r.lr_pvalue)))"
    ftitle = isempty(title) ? "Rational Inattention Estimation" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::CGTestResult; conf_level=0.95, title="", save_path=nothing)

Horizontal dot-and-whisker coefficient plot (`_render_coef_plot_js`) of the
Coibion–Gorodnichenko regression estimates `[alpha, beta_cg]` with HAC intervals
`± z * se` at confidence level `conf_level` (default `0.95`), plus a zero reference line.
"""
function plot_result(r::CGTestResult{T};
                     conf_level::Real=0.95,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    z, pct = _ri_conf_crit(conf_level)
    id = _next_plot_id("ri_cg")
    data_json = _ri_coef_data_json(
        ["α (Intercept)", "β_CG (Revision)"],
        [r.alpha, r.beta_cg],
        [r.se_alpha, r.se_cg],
        z,
    )
    js = _render_coef_plot_js(id, data_json; ref_value=0, xlabel="Coefficient", ylabel="")
    gain_str = _fmt(r.implied_gain; digits=3)
    cap_str = isfinite(r.implied_capacity) ? "$(_fmt(r.implied_capacity; digits=3)) bits" : "NaN"
    ptitle = "CG Regression Estimates ($(pct)% HAC CI; Ĝ = $(gain_str), κ̂ = $(cap_str), p = $(_format_pvalue(r.pvalue)))"
    ftitle = isempty(title) ? "Coibion–Gorodnichenko Information Rigidity Test" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ftitle, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
