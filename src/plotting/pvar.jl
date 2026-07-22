# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result methods for Panel VAR (PLT-29): OIRF / GIRF / FEVD via the canonical
IRF / FEVD dispatches, plus a companion-eigenvalue unit-circle stability scatter.
"""

# =============================================================================
# Lane-local converter (A5): companion eigenvalues → scatter rows
# =============================================================================

"""
    _pvar_stability_scatter_json(s::PVARStability) -> String

Scatter rows `[{x, y, group}]` for the companion-eigenvalue unit-circle plot:
`x = Re λ`, `y = Im λ`, `group` = "Inside unit circle" (|λ| < 1) or
"On/outside unit circle" (|λ| ≥ 1, the alert color). Non-finite eigenvalues
serialize to `null` (a visible gap; plotrule Robustness).
"""
function _pvar_stability_scatter_json(s::PVARStability)
    rows = Vector{Pair{String,String}}[]
    for (i, lambda) in enumerate(s.eigenvalues)
        m = i <= length(s.moduli) ? s.moduli[i] : abs(lambda)
        grp = (isfinite(m) && m < 1) ? "Inside unit circle" : "On/outside unit circle"
        push!(rows, ["x" => _json(real(lambda)),
                     "y" => _json(imag(lambda)),
                     "group" => _json(grp)])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# PVARModel — view dispatch
# =============================================================================

"""
    plot_result(m::PVARModel; view=:oirf, var=nothing, shock=nothing, H=20,
                ci=nothing, ncols=0, title="", save_path=nothing)

Plot Panel-VAR structural analysis. `view` selects the exhibit:

- `:oirf` (default) — orthogonalized (Cholesky) impulse responses: line + CI band +
  zero line, horizon from 0.
- `:girf` — generalized (Pesaran–Shin) impulse responses, same form.
- `:fevd` — forecast-error variance decomposition: stacked area 0–100 %, horizon
  from 1.
- `:stability` — companion-eigenvalue unit-circle scatter.

Kwargs:
- `var` / `shock` — select one response / shock by `Int` **or** name (bounds-checked
  via `_resolve_var` on `m.varnames`). `shock` applies to `:oirf`/`:girf` only.
- `H` — maximum horizon (`:oirf`/`:girf`/`:fevd`).
- `ci` — the `pvar_bootstrap_irf` NamedTuple to draw bootstrap bands on
  `:oirf`/`:girf`; `nothing` ⇒ no bands.

Unknown `view` throws an `ArgumentError` naming the four valid views.
"""
function plot_result(m::PVARModel{T}; view::Symbol=:oirf,
                     var::Union{Int,String,Nothing}=nothing,
                     shock::Union{Int,String,Nothing}=nothing,
                     H::Int=20, ci::Union{Nothing,NamedTuple}=nothing,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view === :oirf || view === :girf
        r = pvar_irf(m, H; irf_type=view, ci=ci)
        if isempty(title)
            flavor = view === :oirf ? "Orthogonalized" : "Generalized"
            band = ci === nothing ? "" : " (bootstrap CI)"
            title = "Panel VAR $(flavor) Impulse Responses$(band)"
        end
        return plot_result(r; var=var, shock=shock, ncols=ncols,
                           title=title, save_path=save_path)
    elseif view === :fevd
        shock === nothing ||
            throw(ArgumentError("shock does not apply to view=:fevd (all shocks are stacked)"))
        f = pvar_fevd_result(m, H)
        vsel = var === nothing ? nothing : _resolve_var(var, m.varnames)
        isempty(title) && (title = "Panel VAR Forecast Error Variance Decomposition")
        return plot_result(f; var=vsel, ncols=ncols, title=title, save_path=save_path)
    elseif view === :stability
        (var === nothing && shock === nothing) ||
            throw(ArgumentError("var/shock do not apply to view=:stability"))
        return plot_result(pvar_stability(m); title=title, save_path=save_path)
    else
        throw(ArgumentError("Unknown view :$(view). Valid views: :oirf, :girf, :fevd, :stability"))
    end
end

# =============================================================================
# PVARStability — companion-eigenvalue unit-circle scatter
# =============================================================================

"""
    plot_result(s::PVARStability; title="", save_path=nothing)

Companion-eigenvalue stability scatter: each eigenvalue `λ` is drawn at
`(Re λ, Im λ)` inside a unit-circle reference on `[-1.1, 1.1]²`, with the real and
imaginary zero axes marked. Roots inside the circle (stable) and on/outside it
(unstable, alert color) are color-coded with a legend. The panel subtitle states
stability and the largest modulus, e.g. `"Companion eigenvalues (STABLE — max |λ|
= 0.870)"`.
"""
function plot_result(s::PVARStability{T}; title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    id = _next_plot_id("pvarstab")
    data_json = _pvar_stability_scatter_json(s)
    groups_json = "[{\"name\":\"Inside unit circle\",\"color\":$(_json(_PLOT_SERIES[1]))}," *
                  "{\"name\":\"On/outside unit circle\",\"color\":$(_json(_PLOT_ALERT))}]"
    ref_shapes = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1," *
                 "\"color\":\"#999\",\"dash\":\"4,3\"}]"
    ref_lines = "[{\"value\":0,\"axis\":\"x\",\"color\":\"#bbb\",\"dash\":\"2,2\"}," *
                "{\"value\":0,\"axis\":\"y\",\"color\":\"#bbb\",\"dash\":\"2,2\"}]"
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=ref_lines, ref_shapes_json=ref_shapes,
                            xlabel="Real", ylabel="Imaginary")

    mx = isempty(s.moduli) ? zero(T) : maximum(s.moduli)
    status = s.is_stable ? "STABLE" : "UNSTABLE"
    ptitle = "Companion eigenvalues ($(status) — max |λ| = $(_fmt(mx; digits=3)))"

    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title)
    save_path !== nothing && save_plot(p, save_path)
    p
end
