# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for model types: ARCH/GARCH/EGARCH/GJR/SV,
FactorModel, DynamicFactorModel, TimeSeriesData, PanelData.
"""

# =============================================================================
# Volatility Model Diagnostics (shared)
# =============================================================================

"""
Generate 3-panel volatility diagnostic Figure:
1. Returns
2. Conditional volatility (σₜ)
3. Standardized residuals (with ±2σ reference lines)
"""
function _plot_volatility_diagnostics(y::AbstractVector, cond_var::AbstractVector,
                                      model_name::String; title::String="",
                                      save_path::Union{String,Nothing}=nothing)
    data_json = _volatility_data_json(y, cond_var)

    # Panel 1: Returns
    id1 = _next_plot_id("vol_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Conditional volatility
    id2 = _next_plot_id("vol_sig")
    s2 = _series_json(["Cond. Volatility (σ)"], [_PLOT_COLORS[2]]; keys=["vol"])
    js2 = _render_line_js(id2, data_json, s2; xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Conditional Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("vol_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period",
                          ylabel="zₜ = εₜ/σₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "$model_name — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# ARCHModel
# =============================================================================

"""
    plot_result(m::ARCHModel; title="", save_path=nothing)

Plot ARCH model diagnostics: returns, conditional volatility, standardized residuals.
"""
function plot_result(m::ARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "ARCH($(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GARCHModel
# =============================================================================

"""
    plot_result(m::GARCHModel; title="", save_path=nothing)

Plot GARCH model diagnostics.
"""
function plot_result(m::GARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# EGARCHModel
# =============================================================================

"""
    plot_result(m::EGARCHModel; title="", save_path=nothing)

Plot EGARCH model diagnostics.
"""
function plot_result(m::EGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "EGARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# GJRGARCHModel
# =============================================================================

"""
    plot_result(m::GJRGARCHModel; title="", save_path=nothing)

Plot GJR-GARCH model diagnostics.
"""
function plot_result(m::GJRGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "GJR-GARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# FIGARCHModel / FIEGARCHModel (EV-14, #422)
# =============================================================================

"""
    plot_result(m::FIGARCHModel; title="", save_path=nothing)

Plot FIGARCH conditional-volatility diagnostics (returns + fitted σ²).
"""
function plot_result(m::FIGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "FIGARCH($(m.p),d,$(m.q))";
                                  title=title, save_path=save_path)
end

"""
    plot_result(m::FIEGARCHModel; title="", save_path=nothing)

Plot FIEGARCH conditional-volatility diagnostics (returns + fitted σ²).
"""
function plot_result(m::FIEGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "FIEGARCH($(m.p),d,$(m.q))";
                                  title=title, save_path=save_path)
end

# =============================================================================
# IGARCHModel / APARCHModel / CGARCHModel (EV-15, #423)
# =============================================================================

"""
    plot_result(m::IGARCHModel; title="", save_path=nothing)

Plot IGARCH conditional-volatility diagnostics (returns + fitted σ²).
"""
function plot_result(m::IGARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "IGARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

"""
    plot_result(m::APARCHModel; title="", save_path=nothing)

Plot APARCH conditional-volatility diagnostics (returns + fitted σ²).
"""
function plot_result(m::APARCHModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    _plot_volatility_diagnostics(m.y, m.conditional_variance, "APARCH($(m.p),$(m.q))";
                                  title=title, save_path=save_path)
end

"""
    plot_result(m::CGARCHModel; view=:components, title="", save_path=nothing)

Plot a Component-GARCH(1,1) fit.

- `view=:components` — stacked-area decomposition of the permanent (`√q`) and
  transitory (`√max(σ²−q,0)`) volatility contributions over the sample.
- `view=:default` — the standard 3-panel volatility diagnostic figure.
"""
function plot_result(m::CGARCHModel{T}; view::Symbol=:components,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :components
        p = _cgarch_components_plot(m, title)
    elseif view === :default
        return _plot_volatility_diagnostics(m.y, m.conditional_variance, "Component-GARCH(1,1)";
                                             title=title, save_path=save_path)
    else
        throw(ArgumentError("unknown view $view — use :components or :default"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _cgarch_components_plot(m::CGARCHModel{T}, title::String) where {T}
    id = _next_plot_id("cg_comp")
    n = length(m.conditional_variance)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        perm = sqrt(max(m.permanent[i], zero(T)))
        trans = sqrt(max(m.transitory[i], zero(T)))
        push!(rows, ["x" => _json(i), "permanent" => _json(perm), "transitory" => _json(trans)])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Permanent √q", "Transitory √(σ²−q)"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]]; keys=["permanent", "transitory"])
    js = _render_area_js(id, data_json, s_json;
                         xlabel="Observation", ylabel="Volatility contribution")
    isempty(title) && (title = "Component-GARCH Volatility Decomposition")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

# =============================================================================
# GarchMidasModel (EV-02, #410)
# =============================================================================

"""
    plot_result(m::GarchMidasModel; view=:components, title="", save_path=nothing)

Plot a GARCH-MIDAS fit.

- `view=:components` — overlay total conditional volatility `√σ²` and the
  long-run component `√τ` over the retained sample.
- `view=:weights` — the fitted Beta MIDAS weight curve `φ_k` versus lag `k`.
"""
function plot_result(m::GarchMidasModel{T}; view::Symbol=:components,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    if view === :components
        p = _garch_midas_components_plot(m, title)
    elseif view === :weights
        p = _garch_midas_weight_plot(m, title)
    else
        throw(ArgumentError("unknown view $view — use :components or :weights"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _garch_midas_components_plot(m::GarchMidasModel{T}, title::String) where {T}
    id = _next_plot_id("gm_comp")
    n = length(m.conditional_variance)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, [
            "x" => _json(i),
            "total" => _json(sqrt(max(m.conditional_variance[i], zero(T)))),
            "longrun" => _json(sqrt(max(m.tau[i], zero(T)))),
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Total √σ²", "Long-run √τ"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]];
                          keys=["total", "longrun"], dash=["", "6,3"])
    js = _render_line_js(id, data_json, s_json;
                         xlabel="Retained HF observation", ylabel="Volatility")
    isempty(title) && (title = "GARCH-MIDAS Volatility Components")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

function _garch_midas_weight_plot(m::GarchMidasModel{T}, title::String) where {T}
    id = _next_plot_id("gm_w")
    rows = Vector{Pair{String,String}}[]
    for k in 1:m.K
        push!(rows, ["x" => _json(k), "w" => _json(m.weights[k]), "zero" => "0"])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Weight φₖ"], [_PLOT_COLORS[1]]; keys=["w"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data_json, s_json;
                         ref_lines_json=refs, xlabel="MIDAS lag k", ylabel="Weight")
    isempty(title) && (title = "GARCH-MIDAS Beta Weights (K=$(m.K))")
    _make_plot([_PanelSpec(id, title, js)]; title=title)
end

# =============================================================================
# SVModel
# =============================================================================

"""
    plot_result(m::SVModel; title="", save_path=nothing)

Plot SV model diagnostics with posterior quantile bands on volatility.
"""
function plot_result(m::SVModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    data_json = _sv_data_json(m.y, m.volatility_mean, m.volatility_quantiles,
                               m.quantile_levels)
    nq = length(m.quantile_levels)

    # Panel 1: Returns
    id1 = _next_plot_id("sv_ret")
    s1 = _series_json(["Returns"], [_PLOT_COLORS[1]]; keys=["ret"])
    refs1 = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js1 = _render_line_js(id1, data_json, s1;
                          ref_lines_json=refs1, xlabel="", ylabel="Returns")
    p1 = _PanelSpec(id1, "Returns", js1)

    # Panel 2: Posterior volatility with CI band
    id2 = _next_plot_id("sv_vol")
    s2 = _series_json(["Posterior mean σ"], [_PLOT_COLORS[2]]; keys=["vol_mean"])
    bands2 = nq >= 2 ?
        "[{\"lo_key\":\"q1\",\"hi_key\":\"q$(nq)\",\"color\":\"$(_PLOT_COLORS[2])\",\"alpha\":$(_PLOT_CI_ALPHA)}]" : "[]"
    js2 = _render_line_js(id2, data_json, s2;
                          bands_json=bands2, xlabel="", ylabel="σₜ")
    p2 = _PanelSpec(id2, "Stochastic Volatility", js2)

    # Panel 3: Standardized residuals
    id3 = _next_plot_id("sv_zr")
    s3 = _series_json(["Std. Residuals"], [_PLOT_COLORS[3]]; keys=["std_resid"])
    refs3 = "[{\"value\":2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":-2,\"color\":\"#d62728\",\"dash\":\"4,3\"},{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js3 = _render_line_js(id3, data_json, s3;
                          ref_lines_json=refs3, xlabel="Period", ylabel="zₜ")
    p3 = _PanelSpec(id3, "Standardized Residuals", js3)

    if isempty(title)
        title = "Stochastic Volatility Model — Diagnostic Plots"
    end

    p = _make_plot([p1, p2, p3]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# FactorModel
# =============================================================================

"""
    plot_result(fm::FactorModel; title="", save_path=nothing)

Plot factor model: scree plot (eigenvalues) + extracted factor series.
"""
function plot_result(fm::FactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree plot (eigenvalues as bar chart)
    id1 = _next_plot_id("fm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("fm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _PLOT_COLORS[1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors", js2)

    if isempty(title)
        title = "Static Factor Model (r=$(fm.r))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# DynamicFactorModel
# =============================================================================

"""
    plot_result(fm::DynamicFactorModel; title="", save_path=nothing)

Plot dynamic factor model: scree + factor series.
"""
function plot_result(fm::DynamicFactorModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    # Panel 1: Scree
    id1 = _next_plot_id("dfm_scree")
    n_eig = min(length(fm.eigenvalues), 10)
    rows1 = Vector{Pair{String,String}}[]
    for i in 1:n_eig
        push!(rows1, ["x" => _json("PC $i"), "eig" => _json(fm.eigenvalues[i])])
    end
    data1 = _json_array_of_objects(rows1)
    s1 = _series_json(["Eigenvalue"], [_PLOT_COLORS[1]]; keys=["eig"])
    js1 = _render_bar_js(id1, data1, s1; mode="grouped", ylabel="Eigenvalue")
    p1 = _PanelSpec(id1, "Scree Plot", js1)

    # Panel 2: Factor series
    id2 = _next_plot_id("dfm_fac")
    T_obs, r = size(fm.factors)
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    data2 = _timeseries_data_json(fm.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, _PLOT_COLORS[1:n_plot]; keys=["v$i" for i in 1:n_plot])
    js2 = _render_line_js(id2, data2, s2; xlabel="Period", ylabel="Factor Value")
    p2 = _PanelSpec(id2, "Extracted Factors (VAR($( fm.p)))", js2)

    if isempty(title)
        title = "Dynamic Factor Model (r=$(fm.r), p=$(fm.p))"
    end

    p = _make_plot([p1, p2]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# NOTE: plot_result(::TimeSeriesData) relocated to plotting/timeseries.jl and
# plot_result(::PanelData) relocated to plotting/panel.jl (PLT plotting overhaul,
# Wave-2 data-view lanes). See those files.

# =============================================================================
# OccBinIRF
# =============================================================================

"""
    plot_result(oirf::OccBinIRF; title="", save_path=nothing)

Plot OccBin IRF comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(oirf::OccBinIRF{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    H, n_vars = size(oirf.piecewise)
    panels = _PanelSpec[]

    for j in 1:n_vars
        id = _next_plot_id("oirf")
        ptitle = oirf.varnames[j]

        # Build data JSON: {h, lin, pw, bind}
        binding = vec(any(oirf.regime_history .> 0; dims=2))
        rows = Vector{Pair{String,String}}[]
        for h in 1:H
            push!(rows, [
                "h" => _json(h),
                "lin" => _json(oirf.linear[h, j]),
                "pw" => _json(oirf.piecewise[h, j]),
                "bind" => _json(binding[h] ? 1 : 0)
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json, "Horizon", oirf.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin IRF — Shock: $(oirf.shock_name)"
    end

    p = _make_plot(panels; title=title, ncols=min(n_vars, 3))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# OccBinSolution
# =============================================================================

"""
    plot_result(sol::OccBinSolution; title="", save_path=nothing)

Plot OccBin solution comparison: linear (dashed) vs piecewise-linear (solid) with
shaded binding-period rectangles for each variable.
"""
function plot_result(sol::OccBinSolution{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    nperiods, n_vars = size(sol.piecewise_path)
    panels = _PanelSpec[]

    for j in 1:n_vars
        id = _next_plot_id("osol")
        ptitle = sol.varnames[j]

        # Build data JSON: {h, lin, pw, bind}
        binding = vec(any(sol.regime_history .> 0; dims=2))
        rows = Vector{Pair{String,String}}[]
        for h in 1:nperiods
            push!(rows, [
                "h" => _json(h),
                "lin" => _json(sol.linear_path[h, j]),
                "pw" => _json(sol.piecewise_path[h, j]),
                "bind" => _json(binding[h] ? 1 : 0)
            ])
        end
        data_json = _json_array_of_objects(rows)

        js = _render_occbin_panel_js(id, data_json, "Period", sol.varnames[j])
        push!(panels, _PanelSpec(id, ptitle, js))
    end

    if isempty(title)
        title = "OccBin Piecewise-Linear Solution"
    end

    p = _make_plot(panels; title=title, ncols=min(n_vars, 3))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Shared OccBin Panel Renderer
# =============================================================================

"""
Render a single OccBin panel with linear (dashed) vs piecewise (solid) lines,
shaded binding-period rectangles, and zero reference line.

Data format: [{h, lin, pw, bind}, ...] where bind is 0 or 1.
"""
function _render_occbin_panel_js(id::String, data_json::String,
                                  xlabel::String, ylabel::String)
    lin_color = _PLOT_COLORS[3]   # green for linear
    pw_color = _PLOT_COLORS[1]    # blue for piecewise
    bind_color = "#d62728"        # red for binding region
    """
(function() {
    const data_$(id) = $(data_json);
    const container_$(id) = d3.select('#$(id)');
    const W_$(id) = Math.max(container_$(id).node().clientWidth - 24, 280);
    const margin_$(id) = {top:10, right:15, bottom:35, left:55};
    const w_$(id) = W_$(id) - margin_$(id).left - margin_$(id).right;
    const h_$(id) = Math.min(w_$(id) * 0.6, 250);

    const svg_$(id) = container_$(id).append('svg')
        .attr('width', W_$(id))
        .attr('height', h_$(id) + margin_$(id).top + margin_$(id).bottom);
    const g_$(id) = svg_$(id).append('g')
        .attr('transform', 'translate('+margin_$(id).left+','+margin_$(id).top+')');

    // Compute domains
    const xVals_$(id) = data_$(id).map(d => d.h);
    const allY_$(id) = [];
    data_$(id).forEach(d => {
        if(d.lin!==null) allY_$(id).push(d.lin);
        if(d.pw!==null) allY_$(id).push(d.pw);
    });
    allY_$(id).push(0);

    const x_$(id) = d3.scaleLinear().domain(d3.extent(xVals_$(id))).range([0, w_$(id)]);
    const yExt_$(id) = d3.extent(allY_$(id));
    const yPad_$(id) = (yExt_$(id)[1] - yExt_$(id)[0]) * 0.08 || 0.1;
    const y_$(id) = d3.scaleLinear()
        .domain([yExt_$(id)[0] - yPad_$(id), yExt_$(id)[1] + yPad_$(id)])
        .range([h_$(id), 0]);

    // Grid
    g_$(id).append('g').attr('class','grid')
        .call(d3.axisLeft(y_$(id)).tickSize(-w_$(id)).tickFormat(''));

    // Binding-period shaded rectangles
    const xStep_$(id) = w_$(id) / Math.max(xVals_$(id).length - 1, 1);
    data_$(id).forEach((d, i) => {
        if(d.bind === 1) {
            const x0_$(id) = x_$(id)(d.h) - xStep_$(id) * 0.5;
            const x1_$(id) = x_$(id)(d.h) + xStep_$(id) * 0.5;
            g_$(id).append('rect')
                .attr('x', Math.max(0, x0_$(id)))
                .attr('y', 0)
                .attr('width', Math.min(x1_$(id), w_$(id)) - Math.max(0, x0_$(id)))
                .attr('height', h_$(id))
                .attr('fill', '$(bind_color)')
                .attr('opacity', 0.08);
        }
    });

    // Zero reference line
    g_$(id).append('line')
        .attr('x1', 0).attr('x2', w_$(id))
        .attr('y1', y_$(id)(0)).attr('y2', y_$(id)(0))
        .attr('stroke', '#999').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3');

    // Linear line (dashed)
    const linLine_$(id) = d3.line().x(d => x_$(id)(d.h)).y(d => y_$(id)(d.lin))
        .defined(d => d.lin !== null);
    g_$(id).append('path').datum(data_$(id)).attr('d', linLine_$(id))
        .attr('fill', 'none').attr('stroke', '$(lin_color)')
        .attr('stroke-width', 1.8).attr('stroke-dasharray', '6,3');

    // Piecewise line (solid)
    const pwLine_$(id) = d3.line().x(d => x_$(id)(d.h)).y(d => y_$(id)(d.pw))
        .defined(d => d.pw !== null);
    g_$(id).append('path').datum(data_$(id)).attr('d', pwLine_$(id))
        .attr('fill', 'none').attr('stroke', '$(pw_color)')
        .attr('stroke-width', 1.8);

    // Axes
    g_$(id).append('g').attr('class','axis')
        .attr('transform', 'translate(0,'+h_$(id)+')')
        .call(d3.axisBottom(x_$(id)).ticks(Math.min(xVals_$(id).length, 8)));
    g_$(id).append('g').attr('class','axis')
        .call(d3.axisLeft(y_$(id)).ticks(6));

$(_axis_labels_js(xlabel, ylabel; g="g_$(id)", w="w_$(id)", h="h_$(id)"))

    // Legend
    const leg_$(id) = g_$(id).append('g').attr('class','legend')
        .attr('transform','translate(5,-5)');
    // Linear (dashed)
    const g1_$(id) = leg_$(id).append('g').attr('transform','translate(0,0)');
    g1_$(id).append('line').attr('x1',0).attr('x2',16).attr('y1',0).attr('y2',0)
        .attr('stroke','$(lin_color)').attr('stroke-width',2)
        .attr('stroke-dasharray','6,3');
    g1_$(id).append('text').attr('x',20).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Linear');
    // Piecewise (solid)
    const g2_$(id) = leg_$(id).append('g').attr('transform','translate(80,0)');
    g2_$(id).append('line').attr('x1',0).attr('x2',16).attr('y1',0).attr('y2',0)
        .attr('stroke','$(pw_color)').attr('stroke-width',2);
    g2_$(id).append('text').attr('x',20).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Piecewise');
    // Binding
    const g3_$(id) = leg_$(id).append('g').attr('transform','translate(175,0)');
    g3_$(id).append('rect').attr('width',12).attr('height',10).attr('y',-5)
        .attr('fill','$(bind_color)').attr('opacity',0.15);
    g3_$(id).append('text').attr('x',16).attr('y',4)
        .attr('font-size','10px').attr('fill','#555').text('Binding');

    // Tooltip
    svg_$(id).append('rect')
        .attr('width', W_$(id))
        .attr('height', h_$(id) + margin_$(id).top + margin_$(id).bottom)
        .attr('fill','none').attr('pointer-events','all')
        .on('mousemove', function(evt) {
            const [mx] = d3.pointer(evt, g_$(id).node());
            const x0 = x_$(id).invert(mx);
            const idx = d3.minIndex(data_$(id), d => Math.abs(d.h - x0));
            const d = data_$(id)[idx];
            let html = '<b>h='+d.h+'</b>';
            if(d.lin!==null) html += '<br>Linear: '+fmt(d.lin);
            if(d.pw!==null) html += '<br>Piecewise: '+fmt(d.pw);
            if(d.bind===1) html += '<br><span style="color:$(bind_color)">Binding</span>';
            showTip(evt, html);
        })
        .on('mouseout', hideTip);
})();
"""
end

# NOTE: plot_result(::BayesianDSGE) relocated to plotting/mcmc.jl (PLT plotting
# overhaul, Wave-2 MCMC lane; PLT-05 posterior-mean vline applied there).

# =============================================================================
# FAVARModel
# =============================================================================

"""
    plot_result(m::FAVARModel; title="", save_path=nothing)

Plot FAVAR model: extracted factor series.
"""
function plot_result(m::FAVARModel{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    r = m.n_factors
    n_plot = min(r, 5)
    fac_names = ["Factor $i" for i in 1:n_plot]
    fac_colors = _PLOT_COLORS[1:n_plot]
    data2 = _timeseries_data_json(m.factors[:, 1:n_plot], fac_names)
    s2 = _series_json(fac_names, fac_colors; keys=["v$i" for i in 1:n_plot])

    id = _next_plot_id("favar_fac")
    js = _render_line_js(id, data2, s2; xlabel="Period", ylabel="Factor Value")
    p1 = _PanelSpec(id, "Extracted Factors", js)

    if isempty(title)
        title = "FAVAR Model (r=$(m.n_factors), p=$(m.p))"
    end

    p = _make_plot([p1]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BayesianFAVAR
# =============================================================================

"""
    plot_result(m::BayesianFAVAR; title="", save_path=nothing)

Plot Bayesian FAVAR: posterior mean factor series with 68% credible intervals.
"""
function plot_result(m::BayesianFAVAR{T};
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    r = m.n_factors
    n_plot = min(r, 5)
    F_mean = dropdims(mean(m.factor_draws, dims=1), dims=1)  # T_obs x r
    F_lo = dropdims(mapslices(x -> quantile(x, 0.16), m.factor_draws; dims=1), dims=1)
    F_hi = dropdims(mapslices(x -> quantile(x, 0.84), m.factor_draws; dims=1), dims=1)

    panels = _PanelSpec[]
    for j in 1:n_plot
        id = _next_plot_id("bfavar_f")
        fname = "Factor $j"
        T_obs = size(F_mean, 1)

        # Build data with mean + lo + hi
        rows = Vector{Pair{String,String}}[]
        for t in 1:T_obs
            push!(rows, [
                "x" => _json(t),
                "mean" => _json(F_mean[t, j]),
                "lo" => _json(F_lo[t, j]),
                "hi" => _json(F_hi[t, j])
            ])
        end
        data_json = _json_array_of_objects(rows)

        s_json = _series_json([fname], [_PLOT_COLORS[j]]; keys=["mean"])
        bands_json = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[j])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"
        js = _render_line_js(id, data_json, s_json;
                             bands_json=bands_json, xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id, "$fname (68% CI)", js))
    end

    if isempty(title)
        title = "Bayesian FAVAR (r=$(m.n_factors), p=$(m.p))"
    end

    p = _make_plot(panels; title=title, ncols=min(n_plot, 2))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# StructuralDFM
# =============================================================================

"""
    plot_result(m::StructuralDFM; title="", save_path=nothing, var=nothing, ncols=0)

Plot Structural DFM: structural IRFs for panel variables.
"""
function plot_result(m::StructuralDFM{T};
                     title::String="", save_path::Union{String,Nothing}=nothing,
                     var::Union{Nothing,Int,String}=nothing, ncols::Int=0) where {T}
    r = irf(m, size(m.structural_irf, 1))
    plot_result(r; title=isempty(title) ? "Structural DFM IRFs" : title,
                save_path=save_path, var=var, ncols=ncols)
end

# =============================================================================
# HASteadyState — Heterogeneous Agent wealth distribution
# =============================================================================

"""
    plot_result(ss::HASteadyState; view=:default, title="", save_path=nothing)

Plot heterogeneous agent steady state results.

# Views
- `:default` / `:distribution` — Wealth distribution histogram (marginal over income)
- `:lorenz` — Lorenz curve with 45-degree equality reference
- `:policy` — Policy functions (consumption and savings by income state)

# Example
```julia
p = plot_result(ss)                        # wealth distribution
p = plot_result(ss; view=:lorenz)          # Lorenz curve
p = plot_result(ss; view=:policy)          # policy functions
```
"""
function plot_result(ss::HASteadyState{T};
                     view::Symbol=:default,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view == :default || view == :distribution
        p = _plot_ha_distribution(ss; title=title)
    elseif view == :lorenz
        p = _plot_ha_lorenz(ss; title=title)
    elseif view == :policy
        p = _plot_ha_policy(ss; title=title)
    else
        throw(ArgumentError("Unknown view: $view. Use :distribution, :lorenz, or :policy"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
Wealth distribution histogram: marginal asset distribution (summed over income states)
plotted as a bar chart with asset grid on x-axis and density on y-axis.
"""
function _plot_ha_distribution(ss::HASteadyState{T}; title::String="") where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income

    # Compute marginal asset distribution
    d_mat = reshape(vec(ss.distribution), n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))

    # Normalize
    total_mass = sum(d_asset)
    if total_mass > zero(T)
        d_asset ./= total_mass
    end

    # Aggregate the marginal mass into (at most 60) contiguous bins so NO grid
    # node's mass is dropped — bar heights are probability mass per bin and sum to
    # the total distribution mass (≈1). Point-sampling one node per bin would
    # discard the mass on skipped nodes (plotrule Anti-Pattern #7 / mass must be
    # conserved). When n_a ≤ 60 each bin holds one node, so heights equal the
    # per-node masses (values unchanged from the old path, now provably conserving).
    max_bars = min(n_a, 60)
    nbins = max_bars
    bin_edges = round.(Int, range(0, n_a; length=nbins + 1))

    rows = Vector{Pair{String,String}}[]
    for b in 1:nbins
        lo = bin_edges[b] + 1
        hi = bin_edges[b + 1]
        hi < lo && continue
        bin_mass = sum(@view d_asset[lo:hi])
        # Label the bin by its right-edge asset value.
        label = _fmt_grid_label(a_grid[hi])
        push!(rows, [
            "x" => _json(label),
            "mass" => _json(bin_mass)
        ])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ha_dist")
    s_json = _series_json(["Probability mass"], [_PLOT_COLORS[1]]; keys=["mass"])
    js = _render_bar_js(id, data_json, s_json;
                        mode="grouped", xlabel="Assets", ylabel="Probability mass")
    binned = nbins < n_a
    ptitle = binned ?
        "Marginal Wealth Distribution ($(n_a) grid nodes in $(nbins) bins)" :
        "Marginal Wealth Distribution"
    p1 = _PanelSpec(id, ptitle, js)

    if isempty(title)
        gini = _gini_coefficient(vec(ss.distribution), ss.grid)
        title = "Wealth Distribution (Gini = $(_fmt(gini; digits=3)))"
    end

    _make_plot([p1]; title=title, ncols=1)
end

"""Format asset grid value as a compact label string."""
function _fmt_grid_label(x::Real)
    ax = abs(x)
    if ax >= 1000
        return string(round(Int, x))
    elseif ax >= 1
        return string(round(x; digits=1))
    else
        return string(round(x; digits=2))
    end
end

"""
Lorenz curve: cumulative population share vs cumulative wealth share,
with the 45-degree equality line for reference.
"""
function _plot_ha_lorenz(ss::HASteadyState{T}; title::String="") where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income

    # Marginal asset distribution
    d_mat = reshape(vec(ss.distribution), n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))

    # Sort by asset level
    perm = sortperm(a_grid)
    a_sorted = a_grid[perm]
    d_sorted = d_asset[perm]

    # Normalize
    total_mass = sum(d_sorted)
    if total_mass > zero(T)
        d_sorted ./= total_mass
    end

    mean_wealth = dot(d_sorted, a_sorted)

    # Compute Lorenz curve points
    n_pts = length(a_sorted)
    cum_pop = zeros(T, n_pts + 1)
    cum_wealth = zeros(T, n_pts + 1)

    for i in 1:n_pts
        cum_pop[i + 1] = cum_pop[i] + d_sorted[i]
        cum_wealth[i + 1] = cum_wealth[i] + d_sorted[i] * a_sorted[i]
    end

    # Normalize wealth share by total wealth
    total_wealth = cum_wealth[end]
    if total_wealth > zero(T)
        cum_wealth ./= total_wealth
    end

    # Subsample for manageable plot size
    max_pts = 100
    step = max(1, div(n_pts + 1, max_pts))
    idxs = 1:step:(n_pts + 1)
    if last(idxs) != n_pts + 1
        idxs = vcat(collect(idxs), n_pts + 1)
    end

    rows = Vector{Pair{String,String}}[]
    for idx in idxs
        push!(rows, [
            "x" => _json(cum_pop[idx]),
            "lorenz" => _json(cum_wealth[idx]),
            "equality" => _json(cum_pop[idx])
        ])
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("ha_lorenz")
    s_json = "[{\"name\":\"Lorenz Curve\",\"color\":\"$(_PLOT_COLORS[1])\",\"key\":\"lorenz\",\"dash\":\"\"}," *
             "{\"name\":\"Perfect Equality\",\"color\":\"#999999\",\"key\":\"equality\",\"dash\":\"6,3\"}]"

    js = _render_line_js(id, data_json, s_json;
                         xlabel="Cumulative Population Share",
                         ylabel="Cumulative Wealth Share")
    p1 = _PanelSpec(id, "Lorenz Curve", js)

    if isempty(title)
        gini = _gini_coefficient(vec(ss.distribution), ss.grid)
        title = "Lorenz Curve (Gini = $(_fmt(gini; digits=3)))"
    end

    _make_plot([p1]; title=title, ncols=1)
end

"""
Policy function plot: consumption and savings as functions of assets,
one line per income state. Two panels side by side.
"""
function _plot_ha_policy(ss::HASteadyState{T}; title::String="") where {T}
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income
    panels = _PanelSpec[]

    # Subsample grid points for cleaner plots
    max_pts = 80
    step = max(1, div(n_a, max_pts))
    idxs = 1:step:n_a
    if last(idxs) != n_a
        idxs = vcat(collect(idxs), n_a)
    end

    income_names = ["e$j" for j in 1:n_e]
    n_plot_e = min(n_e, length(_PLOT_COLORS))

    for (pol_key, pol_label, pol_ylabel) in [
        (:consumption, "Consumption Policy", "c(a, e)"),
        (:savings, "Savings Policy", "a'(a, e)")
    ]
        haskey(ss.policies, pol_key) || continue
        pol = ss.policies[pol_key]

        id = _next_plot_id("ha_pol")
        rows = Vector{Pair{String,String}}[]
        for idx in idxs
            row = Pair{String,String}["x" => _json(a_grid[idx])]
            for j in 1:n_plot_e
                push!(row, "e$j" => _json(pol[idx, j]))
            end
            push!(rows, row)
        end
        data_json = _json_array_of_objects(rows)

        s_json = _series_json(income_names[1:n_plot_e],
                              _PLOT_COLORS[1:n_plot_e];
                              keys=["e$j" for j in 1:n_plot_e])
        js = _render_line_js(id, data_json, s_json;
                             xlabel="Assets", ylabel=pol_ylabel)
        push!(panels, _PanelSpec(id, pol_label, js))
    end

    if isempty(panels)
        throw(ArgumentError("No :consumption or :savings policy found in HASteadyState"))
    end

    if isempty(title)
        r_val = get(ss.prices, :r, NaN)
        title = "Policy Functions (r = $(_fmt(r_val; digits=4)))"
    end

    _make_plot(panels; title=title, ncols=min(length(panels), 2))
end

# =============================================================================
# StateSpaceModel (EV-37, #445)
# =============================================================================

"""
    plot_result(ss::StateSpaceModel; obs=1, state=nothing, title="", save_path=nothing)

Overlay the observed series (`obs`-th observable) with the filtered and smoothed level
of a fitted state-space model, plus a ±1.96·√`smoothed_cov` band around the smoothed
path. `state` selects which state component to overlay (defaults to the state that the
`obs`-th observation loads on, or state 1).
"""
function plot_result(ss::StateSpaceModel{T}; obs::Int=1, state::Union{Int,Nothing}=nothing,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    isfitted(ss) || throw(ArgumentError("plot_result requires a fitted StateSpaceModel"))
    (1 <= obs <= ss.n_obs) || throw(ArgumentError("obs must be in 1:$(ss.n_obs)"))
    # Default state: the first state the obs-th observation loads on.
    st = state === nothing ? something(findfirst(!iszero, ss.Z[obs, :]), 1) : state
    (1 <= st <= ss.n_state) || throw(ArgumentError("state must be in 1:$(ss.n_state)"))

    z = ss.Z[obs, st]                                    # loading of obs on state st
    z = z == 0 ? one(T) : z
    rows = Vector{Pair{String,String}}[]
    for t in 1:ss.T_obs
        yv = ss.y[t, obs]
        filt = z * ss.filtered_state[t, st]
        smth = z * ss.smoothed_state[t, st]
        sd = z * sqrt(max(ss.smoothed_cov[st, st, t], zero(T)))
        push!(rows, Pair{String,String}[
            "x"    => _json(t),
            "obs"  => isnan(yv) ? "null" : _json(yv),
            "filt" => _json(filt),
            "smth" => _json(smth),
            "lo"   => _json(smth - T(1.96) * sd),
            "hi"   => _json(smth + T(1.96) * sd),
        ])
    end
    data_json = _json_array_of_objects(rows)
    id = _next_plot_id("ss_level")
    series = _series_json(["Observed", "Filtered", "Smoothed"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[3], _PLOT_COLORS[2]];
                          keys=["obs", "filt", "smth"], dash=["", "4,3", ""])
    bands = "[{\"lo_key\":\"lo\",\"hi_key\":\"hi\",\"color\":\"$(_PLOT_COLORS[2])\"}]"
    js = _render_line_js(id, data_json, series; bands_json=bands,
                         xlabel="Period", ylabel=ss.n_obs == 1 ? "Value" : "Series $obs")
    panel = _PanelSpec(id, "Filtered vs smoothed level (95% band)", js)
    isempty(title) && (title = "State-Space Model — level estimates")
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
