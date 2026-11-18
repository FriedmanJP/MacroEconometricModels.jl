# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for spectral analysis types: ACFResult, SpectralDensityResult,
CrossSpectrumResult, TransferFunctionResult.
"""

# NOTE: `_render_vbar_js` was relocated to render.jl (plotrule A1 — renderers live
# only in render.jl) in the PLT plotting overhaul (PLT-19). The call sites in this
# file (ACF/PACF/CCF panels) use it unchanged.

# =============================================================================
# ACF Data Helpers
# =============================================================================

"""Convert ACF/PACF lag values to JSON [{x, y}, ...]."""
function _acf_bar_data_json(lags::AbstractVector{Int}, vals::AbstractVector{T}) where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(lags)
        push!(rows, ["x" => _json(lags[i]), "y" => _json(vals[i])])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# ACFResult
# =============================================================================

"""
    plot_result(r::ACFResult; title="", save_path=nothing)

Plot ACF and PACF as side-by-side bar charts with ±CI dashed lines.
For CCF results (when `r.ccf` is populated), a single cross-correlation bar chart.
"""
function plot_result(r::ACFResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    ci_refs = "[{\"value\":$(_json(r.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"},{\"value\":$(_json(-r.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"}]"

    if r.ccf !== nothing
        # CCF plot — single panel
        id = _next_plot_id("ccf")
        data_json = _acf_bar_data_json(r.lags, r.ccf)
        js = _render_vbar_js(id, data_json;
                             bar_color=_PLOT_COLORS[1],
                             ref_lines_json=ci_refs, integer_x=true,
                                     xlabel="Lag", ylabel="CCF")
        panels = [_PanelSpec(id, "Cross-Correlation Function", js)]
        if isempty(title)
            title = "Cross-Correlation Function (n=$(r.nobs))"
        end
        p = _make_plot(panels; title=title, ncols=1)
    else
        # ACF + PACF side-by-side
        panels = _PanelSpec[]

        # ACF panel
        has_acf = any(!iszero, r.acf)
        if has_acf
            id_acf = _next_plot_id("acf")
            data_acf = _acf_bar_data_json(r.lags, r.acf)
            js_acf = _render_vbar_js(id_acf, data_acf;
                                     bar_color=_PLOT_COLORS[1],
                                     ref_lines_json=ci_refs, integer_x=true,
                                     xlabel="Lag", ylabel="ACF")
            push!(panels, _PanelSpec(id_acf, "Autocorrelation", js_acf))
        end

        # PACF panel
        has_pacf = any(!iszero, r.pacf)
        if has_pacf
            id_pacf = _next_plot_id("pacf")
            data_pacf = _acf_bar_data_json(r.lags, r.pacf)
            js_pacf = _render_vbar_js(id_pacf, data_pacf;
                                      bar_color=_PLOT_COLORS[2],
                                      ref_lines_json=ci_refs, integer_x=true,
                                     xlabel="Lag", ylabel="PACF")
            push!(panels, _PanelSpec(id_pacf, "Partial Autocorrelation", js_pacf))
        end

        # Fallback if neither populated (shouldn't happen)
        if isempty(panels)
            id_acf = _next_plot_id("acf")
            data_acf = _acf_bar_data_json(r.lags, r.acf)
            js_acf = _render_vbar_js(id_acf, data_acf;
                                     bar_color=_PLOT_COLORS[1],
                                     ref_lines_json=ci_refs, integer_x=true,
                                     xlabel="Lag", ylabel="ACF")
            push!(panels, _PanelSpec(id_acf, "Autocorrelation", js_acf))
        end

        if isempty(title)
            title = "ACF / PACF (n=$(r.nobs))"
        end
        ncols = length(panels) == 2 ? 2 : 1
        p = _make_plot(panels; title=title, ncols=ncols)
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Spectral Density Data Helpers
# =============================================================================

"""Convert spectral density to JSON [{x, density, ci_lo, ci_hi}, ...]."""
function _spectral_data_json(freq::AbstractVector{T}, density::AbstractVector{T},
                             ci_lo::AbstractVector{T}, ci_hi::AbstractVector{T};
                             log_scale::Bool=false) where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        d_val = log_scale ? log10(max(density[i], T(1e-30))) : density[i]
        lo_val = log_scale ? log10(max(ci_lo[i], T(1e-30))) : ci_lo[i]
        hi_val = log_scale ? log10(max(ci_hi[i], T(1e-30))) : ci_hi[i]
        push!(rows, [
            "x" => _json(freq[i]),
            "density" => _json(d_val),
            "ci_lo" => _json(lo_val),
            "ci_hi" => _json(hi_val)
        ])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# SpectralDensityResult
# =============================================================================

"""
    plot_result(r::SpectralDensityResult; title="", save_path=nothing, log_scale=false)

Plot spectral density line with CI shading band (linear scale by default; `log_scale=true` for log10).
"""
function plot_result(r::SpectralDensityResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     log_scale::Bool=false) where {T}
    id = _next_plot_id("spec")
    data_json = _spectral_data_json(r.freq, r.density, r.ci_lower, r.ci_upper; log_scale=log_scale)

    series_label = log_scale ? "Log Spectral Density" : "Spectral Density"
    ylabel = log_scale ? "log₁₀ Spectral Density" : "Spectral Density"
    s_json = _series_json([series_label], [_PLOT_COLORS[1]]; keys=["density"])
    bands = "[{\"lo_key\":\"ci_lo\",\"hi_key\":\"ci_hi\",\"color\":\"$(_PLOT_COLORS[1])\",\"alpha\":$(_PLOT_CI_ALPHA)}]"

    js = _render_line_js(id, data_json, s_json;
                         bands_json=bands,
                         xlabel="Frequency (radians)", ylabel=ylabel)
    panels = [_PanelSpec(id, "Spectral Density ($(r.method))", js)]

    if isempty(title)
        title = "Spectral Density Estimate ($(r.method), n=$(r.nobs))"
    end

    p = _make_plot(panels; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Cross-Spectrum Data Helpers
# =============================================================================

"""Convert cross-spectrum fields to JSON for coherence + phase panels."""
function _cross_spectrum_data_json(freq::AbstractVector{T},
                                   vals::AbstractVector{T};
                                   key::String="y") where {T}
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(freq)
        push!(rows, ["x" => _json(freq[i]), key => _json(vals[i])])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# CrossSpectrumResult
# =============================================================================

"""
    plot_result(r::CrossSpectrumResult; title="", save_path=nothing)

Plot cross-spectrum: 2-panel with coherence line and phase line.
"""
function plot_result(r::CrossSpectrumResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]

    # Panel 1: Coherence
    id_coh = _next_plot_id("coh")
    data_coh = _cross_spectrum_data_json(r.freq, r.coherence; key="coh")
    s_coh = _series_json(["Squared Coherence"], [_PLOT_COLORS[1]]; keys=["coh"])
    js_coh = _render_line_js(id_coh, data_coh, s_coh;
                             xlabel="Frequency (radians)", ylabel="Coherence")
    push!(panels, _PanelSpec(id_coh, "Squared Coherence", js_coh))

    # Panel 2: Phase
    id_ph = _next_plot_id("phase")
    data_ph = _cross_spectrum_data_json(r.freq, r.phase; key="phase")
    s_ph = _series_json(["Phase"], [_PLOT_COLORS[2]]; keys=["phase"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js_ph = _render_line_js(id_ph, data_ph, s_ph;
                            ref_lines_json=refs,
                            xlabel="Frequency (radians)", ylabel="Phase (radians)")
    push!(panels, _PanelSpec(id_ph, "Phase Spectrum", js_ph))

    if isempty(title)
        title = "Cross-Spectral Analysis (n=$(r.nobs))"
    end

    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TransferFunctionResult
# =============================================================================

"""
    plot_result(r::TransferFunctionResult; title="", save_path=nothing)

Plot transfer function: 2-panel with gain line and phase line.
"""
function plot_result(r::TransferFunctionResult{T};
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    panels = _PanelSpec[]

    # Panel 1: Gain
    id_gain = _next_plot_id("tfgain")
    data_gain = _cross_spectrum_data_json(r.freq, r.gain; key="gain")
    s_gain = _series_json(["Gain"], [_PLOT_COLORS[1]]; keys=["gain"])
    js_gain = _render_line_js(id_gain, data_gain, s_gain;
                              xlabel="Frequency (radians)", ylabel="Gain")
    push!(panels, _PanelSpec(id_gain, "Gain (Amplitude)", js_gain))

    # Panel 2: Phase
    id_ph = _next_plot_id("tfphase")
    data_ph = _cross_spectrum_data_json(r.freq, r.phase; key="phase")
    s_ph = _series_json(["Phase"], [_PLOT_COLORS[2]]; keys=["phase"])
    refs = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js_ph = _render_line_js(id_ph, data_ph, s_ph;
                            ref_lines_json=refs,
                            xlabel="Frequency (radians)", ylabel="Phase (radians)")
    push!(panels, _PanelSpec(id_ph, "Phase Shift", js_ph))

    if isempty(title)
        title = "Transfer Function — $(r.filter) Filter"
    end

    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# Shared Internal Panel Builders for GDFM Types (GDFM-08 #900)
# =============================================================================

# Multi-panel line overlay (two series: e.g. observed vs common or two-sided vs
# one-sided) or single-series path across selected variable indices.
function _gdfm_comparison_panels(
    prefix::String,
    xs::AbstractVector,
    Y1::AbstractMatrix,
    Y2::Union{AbstractMatrix,Nothing},
    var_idxs::Vector{Int},
    varnames::Vector{String},
    s1_label_fn::Function,
    s2_label::String;
    ptitle_fn::Function=(vi -> varnames[vi]),
    xlabel::String="Period",
    ylabel_fn::Function=(vi -> varnames[vi]),
    ref_lines_json::String="[]",
)
    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id(prefix)
        cols = Pair{String,AbstractVector}["s1" => @view(Y1[:, vi])]
        names = String[s1_label_fn(vi)]
        colors = String[_PLOT_SERIES[1]]
        keys = String["s1"]
        dash = String[""]
        if Y2 !== nothing
            push!(cols, "s2" => @view(Y2[:, vi]))
            push!(names, s2_label)
            push!(colors, _PLOT_SERIES[2])
            push!(keys, "s2")
            push!(dash, "6,3")
        end
        data_json = _keyed_rows_json(xs, cols)
        s_json = _series_json(names, colors; keys=keys, dash=dash)
        js = _render_line_js(id, data_json, s_json;
                             ref_lines_json=ref_lines_json, integer_x=true,
                             xlabel=xlabel, ylabel=ylabel_fn(vi))
        push!(panels, _PanelSpec(id, ptitle_fn(vi), js))
    end
    panels
end

# Multi-panel IRF curves across selected variable indices for one shock column.
function _gdfm_irf_panels(
    prefix::String,
    irf_slice::AbstractMatrix,
    var_idxs::Vector{Int},
    varnames::Vector{String},
    shock_label::String;
    ptitle_fn::Function=(vi -> "$(varnames[vi]) ← $(shock_label)"),
)
    H = size(irf_slice, 1)
    xs = collect(0:(H - 1))
    zero_ref = "[{\"value\":0,\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    panels = _PanelSpec[]
    for vi in var_idxs
        id = _next_plot_id(prefix)
        data_json = _keyed_rows_json(xs, Pair{String,AbstractVector}["irf" => @view(irf_slice[:, vi])])
        s_json = _series_json(["IRF"], [_PLOT_SERIES[1]]; keys=["irf"])
        js = _render_line_js(id, data_json, s_json;
                             ref_lines_json=zero_ref, integer_x=true,
                             xlabel="Horizon", ylabel="Response")
        push!(panels, _PanelSpec(id, ptitle_fn(vi), js))
    end
    panels
end

# =============================================================================
# FHLZModel
# =============================================================================

"""
    plot_result(m::FHLZModel; view=:common, vars=1:min(4, size(m.X, 2)),
                horizon=min(20, size(m.impulse_filters, 1)), shock=1,
                ncols=0, title="", save_path=nothing) -> PlotOutput

Plot an estimated [`FHLZModel`](@ref) (Forni–Hallin–Lippi–Zaffaroni 2015, 2017).

# Views
- `:common` (default) — overlay observed `X` and one-sided `common_component` across `vars`
- `:innovations` / `:factors` — time paths of estimated orthonormal innovations `m.innovations`
  (up to 6 factors) and a bar chart of spectral `variance_explained`
- `:irf` — one-sided dynamic impulse filter curves from `m.impulse_filters` for `shock` over `0:(horizon-1)`

# Keyword Arguments
- `view::Symbol=:common`: View to render (`:common`, `:innovations`, `:factors`, or `:irf`)
- `vars=1:min(4, size(m.X, 2))`: Variable indices or names for `:common` and `:irf`
- `horizon::Int=min(20, size(m.impulse_filters, 1))`: Number of IRF lags (`1 ≤ horizon ≤ size(m.impulse_filters, 1)`)
- `shock::Int=1`: Dynamic shock index for `:irf` (`1 ≤ shock ≤ m.q`)
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(m::FHLZModel{T};
                     view::Symbol=:common,
                     vars=1:min(4, size(m.X, 2)),
                     horizon::Int=min(20, size(m.impulse_filters, 1)),
                     shock::Int=1,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    T_obs, N = size(m.X)
    T_eff = size(m.common_component, 1)
    offset = T_obs - T_eff

    if view === :common
        var_idxs = _gdfm_resolve_vars(vars, m.varnames)
        xs = collect((offset + 1):T_obs)
        X_eff = @view m.X[(offset + 1):T_obs, :]
        panels = _gdfm_comparison_panels(
            "fhlz_com", xs, X_eff, m.common_component, var_idxs, m.varnames,
            vi -> m.varnames[vi], "Common (one-sided)",
        )
        ft = isempty(title) ? "FHLZ One-Sided GDFM — Common Component (q=$(m.q))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    elseif view === :innovations || view === :factors
        xs = collect((offset + 1):T_obs)
        n_show = min(m.q, 6)
        zero_ref = "[{\"value\":0,\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        panels = _PanelSpec[]
        for k in 1:n_show
            id = _next_plot_id("fhlz_inn")
            data_json = _keyed_rows_json(xs, Pair{String,AbstractVector}["u" => @view(m.innovations[:, k])])
            s_json = _series_json(["Innovation $k"], [_PLOT_SERIES[mod1(k, length(_PLOT_SERIES))]]; keys=["u"])
            js = _render_line_js(id, data_json, s_json;
                                 ref_lines_json=zero_ref, integer_x=true,
                                 xlabel="Period", ylabel="Innovation")
            pct = _fmt(100 * m.variance_explained[k]; digits=1)
            push!(panels, _PanelSpec(id, "Innovation $k (spectral share $(pct)%)", js))
        end
        id_bar = _next_plot_id("fhlz_var")
        f_labels = String["Factor $k" for k in 1:m.q]
        data_bar = _keyed_rows_json(f_labels, Pair{String,AbstractVector}["v" => m.variance_explained])
        s_bar = _series_json(["Variance Explained"], [_PLOT_SERIES[1]]; keys=["v"])
        js_bar = _render_bar_js(id_bar, data_bar, s_bar;
                                mode="grouped", orientation="v",
                                xlabel="Dynamic Factor", ylabel="Spectral Share")
        push!(panels, _PanelSpec(id_bar, "Spectral Variance Explained", js_bar))
        ft = isempty(title) ? "FHLZ One-Sided GDFM — Dynamic Innovations (q=$(m.q))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("innovations", n_show, m.q, ""))
    elseif view === :irf
        H_max = size(m.impulse_filters, 1)
        (1 <= shock <= m.q) || throw(ArgumentError("shock $shock out of range 1:$(m.q)"))
        (1 <= horizon <= H_max) || throw(ArgumentError("horizon $horizon out of range 1:$(H_max)"))
        var_idxs = _gdfm_resolve_vars(vars, m.varnames)
        panels = _gdfm_irf_panels(
            "fhlz_irf", @view(m.impulse_filters[1:horizon, :, shock]),
            var_idxs, m.varnames, "Shock $shock",
        )
        ft = isempty(title) ? "FHLZ Impulse Filters — Shock $shock (h=0:$(horizon - 1))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    else
        throw(ArgumentError("Unknown view: :$view. Expected :common, :innovations, :factors, or :irf"))
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# CointegratedGDFM
# =============================================================================

"""
    plot_result(m::CointegratedGDFM; view=:common, vars=1:min(4, size(m.X, 2)),
                shock=1, ncols=0, title="", save_path=nothing) -> PlotOutput

Plot a cointegrated ``I(1)`` [`CointegratedGDFM`](@ref) (Barigozzi–Lippi–Luciani 2021).

# Views
- `:common` (default) — overlay level panel `X` and `common_levels` across `vars`
- `:irf` — level impulse responses `m.irf_matrices` for `shock` over `0:m.irf_horizon`
- `:idiosyncratic` — stationary idiosyncratic component `m.idiosyncratic` across `vars`

# Keyword Arguments
- `view::Symbol=:common`: View to render (`:common`, `:irf`, or `:idiosyncratic`)
- `vars=1:min(4, size(m.X, 2))`: Variable indices or names
- `shock::Union{Int,String}=1`: Structural shock index or name for `:irf`
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(m::CointegratedGDFM{T};
                     view::Symbol=:common,
                     vars=1:min(4, size(m.X, 2)),
                     shock::Union{Int,String}=1,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    T_obs, N = size(m.X)

    if view === :common
        var_idxs = _gdfm_resolve_vars(vars, m.varnames)
        xs = collect(1:T_obs)
        panels = _gdfm_comparison_panels(
            "gdfmi1_com", xs, m.X, m.common_levels, var_idxs, m.varnames,
            vi -> m.varnames[vi], "Common level",
        )
        ft = isempty(title) ?
            "Cointegrated I(1) GDFM — Common Levels (q=$(m.q), perm=$(m.n_permanent))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    elseif view === :irf
        si = _resolve_var(shock isa AbstractString ? String(shock) : Int(shock), m.shock_names)
        var_idxs = _gdfm_resolve_vars(vars, m.varnames)
        panels = _gdfm_irf_panels(
            "gdfmi1_irf", @view(m.irf_matrices[:, :, si]),
            var_idxs, m.varnames, m.shock_names[si];
            ptitle_fn=vi -> "$(m.varnames[vi]) ← $(m.shock_names[si]) (C(1)=$(_fmt(m.long_run_impact[vi, si]; digits=3)))",
        )
        ft = isempty(title) ?
            "Cointegrated GDFM Level IRFs — $(m.shock_names[si]) ($(m.identification))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    elseif view === :idiosyncratic
        var_idxs = _gdfm_resolve_vars(vars, m.varnames)
        xs = collect(1:T_obs)
        zero_ref = "[{\"value\":0,\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
        panels = _gdfm_comparison_panels(
            "gdfmi1_id", xs, m.idiosyncratic, nothing, var_idxs, m.varnames,
            vi -> "$(m.varnames[vi]) idiosyncratic", "";
            ylabel_fn=_ -> "Idiosyncratic",
            ref_lines_json=zero_ref,
        )
        ft = isempty(title) ? "Cointegrated I(1) GDFM — Idiosyncratic Component" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    else
        throw(ArgumentError("Unknown view: :$view. Expected :common, :irf, or :idiosyncratic"))
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# BandPassGDFMResult
# =============================================================================

"""
    plot_result(r::BandPassGDFMResult; view=:comparison, vars=nothing, ncols=0,
                title="", save_path=nothing) -> PlotOutput

Plot frequency-band common-component extraction from [`BandPassGDFMResult`](@ref).

# Views
- `:comparison` (default) — overlay two-sided (`common_twosided`) and one-sided
  (`common_onesided`) band-limited common components across selected `vars`
- `:variance_share` — horizontal bar chart of per-variable `variance_share` in the band

# Keyword Arguments
- `view::Symbol=:comparison`: View to render (`:comparison` or `:variance_share`)
- `vars=nothing`: Variable indices or names (`nothing` defaults to first 4 in `:comparison`, first 20 in `:variance_share`)
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(r::BandPassGDFMResult{T};
                     view::Symbol=:comparison,
                     vars=nothing,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    T_obs, N = size(r.common_twosided)
    band_str = "[$(_fmt(r.band[1]; digits=3)), $(_fmt(r.band[2]; digits=3))] rad"

    if view === :comparison
        sel = vars === nothing ? (1:min(4, N)) : vars
        var_idxs = _gdfm_resolve_vars(sel, r.varnames)
        xs = collect(1:T_obs)
        zero_ref = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
        panels = _gdfm_comparison_panels(
            "bpgdfm_cmp", xs, r.common_twosided, r.common_onesided, var_idxs, r.varnames,
            _ -> "Two-sided", "One-sided";
            ptitle_fn=vi -> "$(r.varnames[vi]) (share=$(_fmt(100 * r.variance_share[vi]; digits=1))%)",
            ylabel_fn=_ -> "Band-pass common",
            ref_lines_json=zero_ref,
        )
        ft = isempty(title) ? "Band-Pass GDFM $(band_str) (r=$(r.r))" : title
        p = _make_plot(panels; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    elseif view === :variance_share
        sel = vars === nothing ? (1:min(20, N)) : vars
        var_idxs = _gdfm_resolve_vars(sel, r.varnames)
        id = _next_plot_id("bpgdfm_sh")
        labels = String[r.varnames[vi] for vi in var_idxs]
        vals = T[r.variance_share[vi] for vi in var_idxs]
        data_json = _keyed_rows_json(labels, Pair{String,AbstractVector}["share" => vals])
        s_json = _series_json(["Band Variance Share"], [_PLOT_SERIES[1]]; keys=["share"])
        js = _render_bar_js(id, data_json, s_json;
                            mode="grouped", orientation="h",
                            xlabel="Share of Common Variance", ylabel="")
        ptitle = _cap_title("Common Variance Share in $(band_str)", length(var_idxs), N)
        ft = isempty(title) ? "Band-Pass GDFM Variance Shares $(band_str)" : title
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols,
                       note=_cap_note("variables", length(var_idxs), N, "vars"))
    else
        throw(ArgumentError("Unknown view: :$view. Expected :comparison or :variance_share"))
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# EurocoinResult
# =============================================================================

"""
    plot_result(r::EurocoinResult; view=:indicator, ncols=0, title="",
                save_path=nothing) -> PlotOutput

Plot the Altissimo et al. (2010) real-time low-frequency [`EurocoinResult`](@ref)
indicator overlaid against the two-sided ideal low-pass benchmark `r.target_lowpass`.

# Views
- `:indicator` (default) — time series comparison of `r.indicator` and `r.target_lowpass`

# Keyword Arguments
- `view::Symbol=:indicator`: View to render (`:indicator`)
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(r::EurocoinResult{T};
                     view::Symbol=:indicator,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    view === :indicator || throw(ArgumentError("Unknown view: :$view. Expected :indicator"))
    T_obs = length(r.indicator)
    xs = collect(1:T_obs)
    id = _next_plot_id("eurocoin")
    data_json = _keyed_rows_json(xs, Pair{String,AbstractVector}[
        "ind" => r.indicator,
        "lp" => r.target_lowpass,
    ])
    s_json = _series_json(["Eurocoin indicator", "Target low-pass"],
                          [_PLOT_SERIES[1], _PLOT_SERIES[2]];
                          keys=["ind", "lp"], dash=["", "6,3"])
    zero_ref = "[{\"value\":0,\"color\":\"#999\",\"dash\":\"4,3\"}]"
    js = _render_line_js(id, data_json, s_json;
                         ref_lines_json=zero_ref, integer_x=true,
                         xlabel="Period", ylabel="Growth")
    ptitle = "Real-Time Indicator vs Ideal Low-Pass (R²=$(_fmt(r.r2; digits=3)), ω_c=$(_fmt(r.cutoff; digits=3)))"
    ft = isempty(title) ? "Eurocoin Real-Time Low-Frequency Indicator (q=$(r.q), r=$(r.r))" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# SpectralFEVDResult
# =============================================================================

"""
    plot_result(r::SpectralFEVDResult; horizon=0, vars=nothing, max_vars=20,
                mode=:stacked, ncols=0, title="", save_path=nothing) -> PlotOutput

Plot frequency-band forecast error variance decomposition [`SpectralFEVDResult`](@ref)
as horizontal stacked or grouped bars of `r.shares` across panel variables and shocks
(including `"Idiosyncratic"` when present).

# Keyword Arguments
- `horizon::Int=0`: Must remain `0` (`SpectralFEVDResult` is frequency-band integrated)
- `vars=nothing`: Optional variable indices or names (`nothing` selects up to `max_vars`)
- `max_vars::Int=20`: Maximum number of variables shown when `vars === nothing`
- `mode::Symbol=:stacked`: Bar layout (`:stacked` or `:grouped`)
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(r::SpectralFEVDResult{T};
                     horizon::Int=0,
                     vars=nothing,
                     max_vars::Int=20,
                     mode::Symbol=:stacked,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    horizon == 0 || throw(ArgumentError(
        "SpectralFEVDResult is frequency-band integrated and has no horizon dimension (leave horizon=0, got $horizon)"))
    mode in (:stacked, :grouped) || throw(ArgumentError(
        "mode must be :stacked or :grouped, got :$mode"))

    N, n_s = size(r.shares)
    sel = vars === nothing ? (1:min(max_vars, N)) : vars
    var_idxs = _gdfm_resolve_vars(sel, r.varnames)

    id = _next_plot_id("spec_fevd")
    labels = String[r.varnames[vi] for vi in var_idxs]
    cols = Pair{String,AbstractVector}["s$j" => r.shares[var_idxs, j] for j in 1:n_s]
    data_json = _keyed_rows_json(labels, cols)
    s_json = _series_json(r.shock_names, _colors_for(r.shock_names);
                          keys=["s$j" for j in 1:n_s])
    js = _render_bar_js(id, data_json, s_json;
                        mode=String(mode), orientation="h",
                        xlabel="Variance Share", ylabel="")
    band_str = "[$(_fmt(r.band[1]; digits=3)), $(_fmt(r.band[2]; digits=3))] rad"
    ptitle = _cap_title("Spectral FEVD Shares over $(band_str)", length(var_idxs), N)
    ft = isempty(title) ? "Spectral FEVD over $(band_str)" : title
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols,
                   note=_cap_note("variables", length(var_idxs), N, "max_vars"))
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# TimeVaryingGDFM
# =============================================================================

"""
    plot_result(m::TimeVaryingGDFM; view=:heatmap, freq_idx=1, ncols=0,
                title="", save_path=nothing) -> PlotOutput

Plot a locally stationary [`TimeVaryingGDFM`](@ref) (Eichler–Motta–von Sachs 2011;
Barigozzi–Hallin–Soccorsi–von Sachs 2021).

# Views
- `:heatmap` (default) — two panels: sequential heatmap of the leading local dynamic
  eigenvalue `λ_1(u, θ)` across rescaled time `m.u_grid` and angular frequencies
  `m.frequencies`, plus a line panel of top local eigenvalues at `m.frequencies[freq_idx]`
- `:variance` / `:connectedness` — time-varying systemic common-variance share path
  `m.variance_share_path` over `1:T_obs`

# Keyword Arguments
- `view::Symbol=:heatmap`: View to render (`:heatmap`, `:variance`, or `:connectedness`)
- `freq_idx::Int=1`: Frequency index in `1:length(m.frequencies)` for the `:heatmap` slice panel
- `ncols::Int=0`: Grid column count (`0` = auto)
- `title::String=""`: Custom figure title (`""` = auto-generated)
- `save_path::Union{Nothing,String}=nothing`: Optional HTML output file path
"""
function plot_result(m::TimeVaryingGDFM{T};
                     view::Symbol=:heatmap,
                     freq_idx::Int=1,
                     ncols::Int=0,
                     title::String="",
                     save_path::Union{Nothing,String}=nothing) where {T}
    T_obs, N = size(m.X)
    n_freq = length(m.frequencies)
    (1 <= freq_idx <= n_freq) || throw(ArgumentError(
        "freq_idx $freq_idx out of range 1:$(n_freq)"))

    if view === :heatmap
        # eigenvalues_path has dimensions (n_grid, N, n_freq); transpose slice j=1 to (n_freq × n_grid)
        lam1 = permutedims(m.eigenvalues_path[:, 1, :])
        row_labels = String[_fmt(θ; digits=3) for θ in m.frequencies]
        col_labels = String[_fmt(u; digits=3) for u in m.u_grid]
        id_hm = _next_plot_id("tvgdfm_hm")
        data_hm = _matrix_heat_json(lam1, row_labels, col_labels)
        js_hm = _render_heatmap_js(id_hm, data_hm, _json(row_labels), _json(col_labels);
                                   xlabel="Rescaled time u", ylabel="Frequency θ (rad)",
                                   tip_label="u =", scale=:sequential)
        p_hm = _PanelSpec(id_hm, "Leading Dynamic Eigenvalue λ₁(u, θ)", js_hm)

        # Second panel: top dynamic eigenvalues at m.frequencies[freq_idx] along u_grid
        n_eig_show = min(m.q + 1, N, 5)
        id_sl = _next_plot_id("tvgdfm_sl")
        cols_sl = Pair{String,AbstractVector}[
            "l$j" => vec(@view(m.eigenvalues_path[:, j, freq_idx])) for j in 1:n_eig_show
        ]
        data_sl = _keyed_rows_json(m.u_grid, cols_sl)
        s_sl = _series_json(["λ$j" for j in 1:n_eig_show],
                            String[_PLOT_SERIES[mod1(j, length(_PLOT_SERIES))] for j in 1:n_eig_show];
                            keys=["l$j" for j in 1:n_eig_show])
        js_sl = _render_line_js(id_sl, data_sl, s_sl;
                                xlabel="Rescaled time u", ylabel="Eigenvalue")
        θ_str = _fmt(m.frequencies[freq_idx]; digits=3)
        p_sl = _PanelSpec(id_sl, "Local Eigenvalues at θ = $(θ_str) rad", js_sl)

        ft = isempty(title) ? "Time-Varying GDFM — Local Dynamic Spectrum (q=$(m.q))" : title
        p = _make_plot([p_hm, p_sl]; title=ft, ncols=ncols)
    elseif view === :variance || view === :connectedness
        xs = collect(1:T_obs)
        tot_share = vec(sum(m.variance_share_path; dims=2))
        id = _next_plot_id("tvgdfm_var")
        cols = Pair{String,AbstractVector}["tot" => tot_share]
        names = String["Total Common Share"]
        colors = String[_PLOT_SERIES[1]]
        keys = String["tot"]
        dash = String[""]
        n_fac_show = m.q > 1 ? min(m.q, 5) : 0
        for j in 1:n_fac_show
            push!(cols, "f$j" => @view(m.variance_share_path[:, j]))
            push!(names, "Factor $j")
            push!(colors, _PLOT_SERIES[mod1(j + 1, length(_PLOT_SERIES))])
            push!(keys, "f$j")
            push!(dash, "6,3")
        end
        data_json = _keyed_rows_json(xs, cols)
        s_json = _series_json(names, colors; keys=keys, dash=dash)
        js = _render_line_js(id, data_json, s_json; integer_x=true,
                             xlabel="Period", ylabel="Variance Share")
        ptitle = m.q > 1 ? _cap_title("Systemic Connectedness Path", n_fac_show, m.q) :
                           "Systemic Connectedness Path"
        ft = isempty(title) ?
            "Time-Varying GDFM — Common Variance Share Path (q=$(m.q))" : title
        note = m.q > 5 ? _cap_note("factors", n_fac_show, m.q, "") : ""
        p = _make_plot([_PanelSpec(id, ptitle, js)]; title=ft, ncols=ncols, note=note)
    else
        throw(ArgumentError("Unknown view: :$view. Expected :heatmap, :variance, or :connectedness"))
    end

    save_path !== nothing && save_plot(p, save_path)
    p
end
