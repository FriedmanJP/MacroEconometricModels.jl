# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
PLT-24 — generic residual-diagnostics view. NEW `plot_result(m; view=:diagnostics)`
dispatches for the multi-equation / panel families that previously had no residual
figure: `VARModel`/`VECMModel` (with an `eq=` equation selector) and
`PanelRegModel`/`PanelIVModel`. Every dispatch routes through the single frozen
`_residual_diagnostics_panels` converter (A6), so the resid-vs-fitted scatter, the
histogram + fitted-normal overlay, the Normal Q-Q (with its A4 45° `line_overlays_json`
line, not a scale-clone), and the residual-ACF panel all share one implementation with
`RegModel` and the ARIMA/GARCH families.
"""

# =============================================================================
# Lane-local converter (A5): per-equation (residual, fitted) for a VAR/VECM. `U` is the
# T_eff×n residual matrix; the equation-`eq` fitted reconstructs actual−residual from the
# tail of the level data `Y` aligned to the residual length (exact for the VAR level
# equations; a level proxy for the differenced VECM — its distribution/Q-Q/ACF panels,
# which use residuals only, are unaffected).
# =============================================================================
function _var_resid_fitted(Y::AbstractMatrix, U::AbstractMatrix, eq::Int)
    u = Float64[Float64(v) for v in @view U[:, eq]]
    nrows = length(u)
    yt = Float64[Float64(v) for v in @view Y[end-nrows+1:end, eq]]
    (u, yt .- u)
end

# =============================================================================
# VARModel
# =============================================================================

"""
    plot_result(m::VARModel; view=:diagnostics, eq=nothing, acf_lags=0, title="", save_path=nothing)

Four-panel residual diagnostics for one equation of a VAR (PLT-24): residual-vs-fitted,
residual histogram + fitted normal, Normal Q-Q, and residual ACF. `eq` selects the
equation by index or name (`nothing` ⇒ the first, with the count surfaced in a note, C7).
"""
function plot_result(m::VARModel{T}; view::Symbol=:diagnostics,
                     eq::Union{Int,String,Nothing}=nothing, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view === :diagnostics ||
        throw(ArgumentError("Unknown view :$view for VARModel; use :diagnostics."))
    n = length(m.varnames)
    eqi = _resolve_var(eq, m.varnames, 1)
    resid, fitted = _var_resid_fitted(m.Y, m.U, eqi)
    panels = _residual_diagnostics_panels(resid, fitted; varname=m.varnames[eqi], acf_lags=acf_lags)
    note = eq === nothing ? _cap_note("equations", 1, n, "eq") : ""
    isempty(title) && (title = "VAR Residual Diagnostics — $(m.varnames[eqi])")
    p = _make_plot(panels; title=title, ncols=2, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# VECMModel
# =============================================================================

"""
    plot_result(m::VECMModel; view=:diagnostics, eq=nothing, acf_lags=0, title="", save_path=nothing)

Four-panel residual diagnostics for one equation of a VECM (PLT-24). `eq` selects the
equation by index or name (`nothing` ⇒ the first, with the count surfaced in a note, C7).
"""
function plot_result(m::VECMModel{T}; view::Symbol=:diagnostics,
                     eq::Union{Int,String,Nothing}=nothing, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view === :diagnostics ||
        throw(ArgumentError("Unknown view :$view for VECMModel; use :diagnostics."))
    n = length(m.varnames)
    eqi = _resolve_var(eq, m.varnames, 1)
    resid, fitted = _var_resid_fitted(m.Y, m.U, eqi)
    panels = _residual_diagnostics_panels(resid, fitted; varname=m.varnames[eqi], acf_lags=acf_lags)
    note = eq === nothing ? _cap_note("equations", 1, n, "eq") : ""
    isempty(title) && (title = "VECM Residual Diagnostics — $(m.varnames[eqi])")
    p = _make_plot(panels; title=title, ncols=2, note=note)
    save_path !== nothing && save_plot(p, save_path)
    p
end

# =============================================================================
# PanelRegModel / PanelIVModel
# =============================================================================

"""
    plot_result(m::PanelRegModel; view=:diagnostics, acf_lags=0, title="", save_path=nothing)

Four-panel residual diagnostics for a panel regression (PLT-24), using the model's
stored `residuals`/`fitted`.
"""
function plot_result(m::PanelRegModel{T}; view::Symbol=:diagnostics, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view === :diagnostics ||
        throw(ArgumentError("Unknown view :$view for PanelRegModel; use :diagnostics."))
    resid = Float64[Float64(v) for v in m.residuals]
    fitted = Float64[Float64(v) for v in m.fitted]
    panels = _residual_diagnostics_panels(resid, fitted; acf_lags=acf_lags)
    isempty(title) && (title = "Panel Regression Residual Diagnostics ($(m.method))")
    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::PanelIVModel; view=:diagnostics, acf_lags=0, title="", save_path=nothing)

Four-panel residual diagnostics for a panel IV/2SLS regression (PLT-24).
"""
function plot_result(m::PanelIVModel{T}; view::Symbol=:diagnostics, acf_lags::Int=0,
                     title::String="", save_path::Union{String,Nothing}=nothing) where {T}
    view === :diagnostics ||
        throw(ArgumentError("Unknown view :$view for PanelIVModel; use :diagnostics."))
    resid = Float64[Float64(v) for v in m.residuals]
    fitted = Float64[Float64(v) for v in m.fitted]
    panels = _residual_diagnostics_panels(resid, fitted; acf_lags=acf_lags)
    isempty(title) && (title = "Panel IV Residual Diagnostics ($(m.method))")
    p = _make_plot(panels; title=title, ncols=2)
    save_path !== nothing && save_plot(p, save_path)
    p
end
