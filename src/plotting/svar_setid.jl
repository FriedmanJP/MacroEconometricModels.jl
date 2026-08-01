# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for set-identified SVARs (PLT-31):

- `SignIdentifiedSet` — pointwise identified-set envelope (nested quantile-fan bands)
  with the pointwise median line, per (response ← shock).
- `AriasSVARResult` — same layout built from the existing **weighted** helpers
  `irf_percentiles` / `irf_mean` (A5 — no quantile logic re-derived here).
- `UhligSVARResult` — a single penalty-optimal rotation: a line only, no band.

All rendering reuses the frozen quantile-fan / line renderers (A1).
"""

# =============================================================================
# Lane-local converters (A5) — helpers.jl is frozen for Wave 2
# =============================================================================

# Pointwise quantile with NaN/Inf propagation: a non-finite draw makes the whole
# cell non-finite (→ a JSON null / visible gap), rather than being silently dropped.
function _pw_quantile(v::AbstractVector, q::Float64)
    any(x -> !isfinite(x), v) && return NaN
    _quantile(sort(Float64[Float64(x) for x in v]), q)
end

"""
    _setid_quantile_json(draws_ij, quantiles, central; xstart=0) -> String

Fan `data_json` for one (response ← shock) slice of a sign-identified draw stack.
`draws_ij` is `n_draws × horizon`; pointwise quantiles are computed **across draws**
(dim 1) at each horizon for every level in `quantiles`, and the length-`horizon`
`central` line is passed through. Delegates row assembly to `_fan_data_json` so the
quantile columns are re-keyed to match `_fan_bands_json`.
"""
function _setid_quantile_json(draws_ij::AbstractMatrix, quantiles::Vector{Float64},
                              central::AbstractVector; xstart::Int=0)
    _, horizon = size(draws_ij)
    Q = Array{Float64}(undef, horizon, length(quantiles))
    for h in 1:horizon
        col = @view draws_ij[:, h]
        for (qi, q) in enumerate(quantiles)
            Q[h, qi] = _pw_quantile(col, q)
        end
    end
    _fan_data_json(Q, quantiles, central; xvals=collect(xstart:(xstart + horizon - 1)))
end

# Synthetic "Var i" / "Shock j" names for result types that carry no names.
_synth_names(prefix::AbstractString, n::Int) = String["$(prefix) $i" for i in 1:n]

# Band percentage from the outer quantile pair, for the C7 draw-count title.
function _setid_band_pct(quantiles::AbstractVector)
    qs = sort(Float64[Float64(q) for q in quantiles])
    round(Int, (qs[end] - qs[1]) * 100)
end

# Shared fan panel: zero reference line + nested bands + central line.
function _setid_fan_panel(id::String, data_json::String, quantiles::Vector{Float64},
                          central_label::String, ptitle::String)
    fan_json = _fan_bands_json(quantiles; color=_PLOT_SERIES[1])
    refs = "[{\"value\":0,\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    js = _render_fan_js(id, data_json, fan_json; median_key="med",
                        central_label=central_label, ref_lines_json=refs,
                        xlabel="Horizon", ylabel="Response")
    _PanelSpec(id, ptitle, js)
end

# =============================================================================
# SignIdentifiedSet — pointwise identified-set fan
# =============================================================================

"""
    plot_result(s::SignIdentifiedSet; var=nothing, shock=nothing,
                quantiles=[0.16, 0.5, 0.84], ncols=0, title="", save_path=nothing)

Draw the sign-/zero-restriction identified set as a quantile fan per (response ←
shock): the pointwise median IRF line plus nested pointwise-quantile envelope bands,
with a zero reference line and integer horizon ticks (x from 0). `var`/`shock` select
one response/shock by `Int` or name (`s.variables`/`s.shocks`); the accepted-draw
count is stated in the figure title (plotrule C7).
"""
function plot_result(s::SignIdentifiedSet{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     quantiles::Vector{Float64}=[0.16, 0.5, 0.84],
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n_draws, horizon, n_vars, n_shocks = size(s.irf_draws)
    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, s.variables)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, s.shocks)]

    panels = _PanelSpec[]
    for sj in shocks_to_plot
        for vi in vars_to_plot
            id = _next_plot_id("signset")
            draws_ij = s.irf_draws[:, :, vi, sj]                      # n_draws × horizon
            central = Float64[_pw_quantile(view(draws_ij, :, h), 0.5) for h in 1:horizon]
            data_json = _setid_quantile_json(draws_ij, quantiles, central; xstart=0)
            ptitle = "$(s.variables[vi]) ← $(s.shocks[sj])"
            push!(panels, _setid_fan_panel(id, data_json, quantiles, "Median", ptitle))
        end
    end

    if isempty(title)
        title = "Sign-Identified IRF set ($(_setid_band_pct(quantiles))% band, $(s.n_accepted) draws)"
    end
    ncols <= 0 && (ncols = length(shocks_to_plot))
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# AriasSVARResult — weighted identified-set fan (reuses weighted helpers)
# =============================================================================

"""
    plot_result(r::AriasSVARResult; var=nothing, shock=nothing,
                quantiles=[0.16, 0.5, 0.84], variables=nothing, shocks=nothing,
                ncols=0, title="", save_path=nothing)

Draw the Arias–Rubio-Ramírez–Waggoner (2018) identified set as a quantile fan per
(response ← shock), built from the existing **weighted** helpers `irf_percentiles`
(bands) and `irf_mean` (central line) — no quantile logic is re-implemented here (A5).
The result carries no names, so `Var i`/`Shock j` are synthesized from
`r.restrictions`; pass `variables=`/`shocks=` to override. `var`/`shock` select one
response/shock by `Int` or (resolved) name; the draw count is in the title (C7).
"""
function plot_result(r::AriasSVARResult{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     quantiles::Vector{Float64}=[0.16, 0.5, 0.84],
                     variables::Union{Vector{String},Nothing}=nothing,
                     shocks::Union{Vector{String},Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    n_draws, horizon, n_vars, n_shocks = size(r.irf_draws)
    varnames = variables === nothing ? _synth_names("Var", n_vars) : variables
    shocknames = shocks === nothing ? _synth_names("Shock", n_shocks) : shocks
    length(varnames) == n_vars || throw(ArgumentError(
        "variables length ($(length(varnames))) must match n_vars ($n_vars)"))
    length(shocknames) == n_shocks || throw(ArgumentError(
        "shocks length ($(length(shocknames))) must match n_shocks ($n_shocks)"))

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, varnames)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, shocknames)]

    pct = irf_percentiles(r; quantiles=Float64.(quantiles))   # horizon × n × n × nq (weighted)
    mean_irf = irf_mean(r)                                     # horizon × n × n     (weighted)

    panels = _PanelSpec[]
    for sj in shocks_to_plot
        for vi in vars_to_plot
            id = _next_plot_id("ariasset")
            Q = pct[:, vi, sj, :]                             # horizon × nq
            central = mean_irf[:, vi, sj]
            data_json = _fan_data_json(Q, quantiles, central; xvals=collect(0:(horizon - 1)))
            ptitle = "$(varnames[vi]) ← $(shocknames[sj])"
            push!(panels, _setid_fan_panel(id, data_json, quantiles, "Mean", ptitle))
        end
    end

    if isempty(title)
        title = "Arias Set-Identified IRF ($(_setid_band_pct(quantiles))% band, $(size(r.irf_draws, 1)) draws)"
    end
    ncols <= 0 && (ncols = length(shocks_to_plot))
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# UhligSVARResult — single penalty-optimal rotation (line only, no band)
# =============================================================================

"""
    plot_result(r::UhligSVARResult; var=nothing, shock=nothing,
                variables=nothing, shocks=nothing, ncols=0, title="", save_path=nothing)

Draw the Mountford–Uhlig penalty-optimal identification as a single IRF **line** per
(response ← shock) — this is one rotation, not a set, so **no band** is drawn (scope
per the result type). Zero reference line, integer horizon ticks (x from 0). The
result carries no names, so `Var i`/`Shock j` are synthesized from `r.restrictions`
(override via `variables=`/`shocks=`); convergence is noted in the figure title.
"""
function plot_result(r::UhligSVARResult{T};
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     variables::Union{Vector{String},Nothing}=nothing,
                     shocks::Union{Vector{String},Nothing}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    horizon, n_vars, n_shocks = size(r.irf)
    varnames = variables === nothing ? _synth_names("Var", n_vars) : variables
    shocknames = shocks === nothing ? _synth_names("Shock", n_shocks) : shocks
    length(varnames) == n_vars || throw(ArgumentError(
        "variables length ($(length(varnames))) must match n_vars ($n_vars)"))
    length(shocknames) == n_shocks || throw(ArgumentError(
        "shocks length ($(length(shocknames))) must match n_shocks ($n_shocks)"))

    vars_to_plot = var === nothing ? (1:n_vars) : [_resolve_var(var, varnames)]
    shocks_to_plot = shock === nothing ? (1:n_shocks) : [_resolve_var(shock, shocknames)]

    refs = "[{\"value\":0,\"axis\":\"y\",\"color\":$(_json(_PLOT_ALERT)),\"dash\":\"4,3\"}]"
    panels = _PanelSpec[]
    for sj in shocks_to_plot
        for vi in vars_to_plot
            id = _next_plot_id("uhlig")
            rows = Vector{Pair{String,String}}[]
            for h in 1:horizon
                push!(rows, ["x" => _json(h - 1), "val" => _json(r.irf[h, vi, sj])])
            end
            data_json = _json_array_of_objects(rows)
            s_json = _series_json(["IRF"], [_PLOT_SERIES[1]]; keys=["val"])
            js = _render_line_js(id, data_json, s_json; ref_lines_json=refs, integer_x=true,
                                 xlabel="Horizon", ylabel="Response")
            ptitle = "$(varnames[vi]) ← $(shocknames[sj])"
            push!(panels, _PanelSpec(id, ptitle, js))
        end
    end

    if isempty(title)
        conv = r.converged ? "converged" : "NOT converged"
        title = "Mountford–Uhlig Penalty-Optimal IRF ($(conv))"
    end
    ncols <= 0 && (ncols = length(shocks_to_plot))
    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
