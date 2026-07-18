# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Binscatter (PLT-23): quantile-binned means of `y` against `x`, optionally after
partialling out controls (Frisch–Waugh–Lovell) and/or within-group demeaning. Added
as `view=:binscatter` to the TimeSeriesData / PanelData / CrossSectionData
`plot_result` methods, all routing through the shared `_plot_binscatter` assembler
here (plotrule A5 — a documented `_plot_*` assembly helper in the owning file). The
statistical work lives in `_binscatter_compute`; the frozen `_fwl_residualize` /
`_ols_fit_line` converters do the OLS. No renderer is defined here (plotrule A1).
"""

# =============================================================================
# Binscatter
# =============================================================================

"""
    _binscatter_compute(xvec, yvec; n_bins=20, controls=nothing, group_id=nothing,
                        demean=:none)
        -> (; xbar, ybar, slope, x_plot, y_plot, n_used)

Bin-and-fit core. Drops rows non-finite in `x`, `y` or any control column; optionally
within-group demeans (`demean=:within`, needs `group_id`); residualizes `x` and `y` on
`[1 controls]` (FWL — when `controls` is a matrix) or just centers them (no controls);
re-adds the original means so the axes stay on the data's scale; sorts by residualized
`x` and cuts into `n_bins` **equal-count** quantile bins, returning per-bin means
`(xbar, ybar)`. `slope` is the degree-1 OLS slope of the residualized `y` on the
residualized `x` — which, by the FWL theorem, equals the multivariate OLS coefficient
on `x` in `y ~ 1 + x + controls`. `x_plot`/`y_plot` are the (re-centered) per-obs
residuals feeding the fit overlay.
"""
function _binscatter_compute(xvec::AbstractVector, yvec::AbstractVector;
                             n_bins::Int=20,
                             controls::Union{AbstractMatrix,Nothing}=nothing,
                             group_id::Union{AbstractVector,Nothing}=nothing,
                             demean::Symbol=:none)
    n_bins >= 1 || throw(ArgumentError("n_bins must be ≥ 1, got $n_bins"))
    x0 = Float64[Float64(v) for v in xvec]
    y0 = Float64[Float64(v) for v in yvec]
    n = length(x0)
    C = controls === nothing ? nothing : Float64.(Matrix(controls))

    # Finite mask across x, y and all control columns.
    mask = trues(n)
    for i in 1:n
        (isfinite(x0[i]) && isfinite(y0[i])) || (mask[i] = false; continue)
        if C !== nothing
            @inbounds for k in 1:size(C, 2)
                isfinite(C[i, k]) || (mask[i] = false; break)
            end
        end
    end
    xs = x0[mask]; ys = y0[mask]
    Cs = C === nothing ? nothing : C[mask, :]
    gs = group_id === nothing ? nothing : Int[Int(g) for g in group_id[mask]]
    m = length(xs)
    m >= 2 || throw(ArgumentError("binscatter needs ≥ 2 complete observations, got $m"))

    # Within-group demeaning of x, y and controls (panel :within).
    if demean === :within
        gs === nothing && throw(ArgumentError("demean=:within needs group_id"))
        xs = _bs_within_demean(xs, gs); ys = _bs_within_demean(ys, gs)
        if Cs !== nothing
            for k in 1:size(Cs, 2)
                Cs[:, k] = _bs_within_demean(Cs[:, k], gs)
            end
        end
    end

    xm = sum(xs) / m; ym = sum(ys) / m
    if Cs === nothing
        xr = xs .- xm; yr = ys .- ym
    else
        xr = _fwl_residualize(xs, Cs); yr = _fwl_residualize(ys, Cs)
    end
    # Re-center residuals to the original means so the axes read naturally.
    x_plot = xr .+ xm; y_plot = yr .+ ym

    sxx = sum(abs2, xr)
    slope = sxx > 0 ? sum(xr .* yr) / sxx : NaN

    # Equal-count quantile bins by rank of the residualized x.
    perm = sortperm(x_plot)
    k = min(n_bins, m)
    xbar = Float64[]; ybar = Float64[]
    for b in 1:k
        lo = floor(Int, (b - 1) * m / k) + 1
        hi = floor(Int, b * m / k)
        hi < lo && continue
        idxb = @view perm[lo:hi]
        push!(xbar, sum(@view x_plot[idxb]) / length(idxb))
        push!(ybar, sum(@view y_plot[idxb]) / length(idxb))
    end
    (; xbar, ybar, slope, x_plot, y_plot, n_used=m)
end

# Subtract per-group means from `v` (parallel to `g`).
function _bs_within_demean(v::AbstractVector, g::Vector{Int})
    acc = Dict{Int,Tuple{Float64,Int}}()
    for i in eachindex(v)
        s, c = get(acc, g[i], (0.0, 0))
        acc[g[i]] = (s + v[i], c + 1)
    end
    means = Dict(k => s / c for (k, (s, c)) in acc)
    Float64[v[i] - means[g[i]] for i in eachindex(v)]
end

"""
    _plot_binscatter(data, varnames, x, y; n_bins=0, controls=nothing, fit=:auto,
                     demean=:none, group_id=nothing, title="", ncols=0, save_path=nothing)

Assemble the binscatter panel from a `data` matrix + `varnames`. `x`/`y` are required
variable selectors (String or Int, plotrule C3); `controls` a vector of selectors to
partial out (FWL); `fit` ∈ `:linear/:quadratic/:none` (`:auto`/`:ols` ⇒ `:linear`).
The binned means are drawn through `_render_scatter_js`; the fit line is a
data-coordinate `line_overlays_json` overlay on the residualized data (plotrule A4).
The subtitle reports the bin count, any partialling/within transform, and the rounded
slope (`_fmt`, plotrule C9).
"""
function _plot_binscatter(data::AbstractMatrix, varnames::Vector{String}, x, y;
                          n_bins::Int=0,
                          controls::Union{Vector,Nothing}=nothing,
                          fit::Symbol=:auto, demean::Symbol=:none,
                          group_id::Union{AbstractVector,Nothing}=nothing,
                          title::String="", ncols::Int=0,
                          save_path::Union{String,Nothing}=nothing)
    (x === nothing || y === nothing) &&
        throw(ArgumentError("binscatter requires both x and y variable selectors"))
    fitn = fit === :auto || fit === :ols ? :linear : fit
    fitn in (:linear, :quadratic, :none) || throw(ArgumentError(
        "fit must be :linear, :quadratic or :none, got :$fit"))
    demean in (:none, :within) || throw(ArgumentError(
        "demean must be :none or :within for binscatter, got :$demean"))

    ix = _resolve_var(x, varnames); iy = _resolve_var(y, varnames)
    cidx = controls === nothing ? Int[] : Int[_resolve_var(c, varnames) for c in controls]
    Cm = isempty(cidx) ? nothing : data[:, cidx]

    nb = n_bins > 0 ? n_bins : 20
    res = _binscatter_compute(data[:, ix], data[:, iy];
                              n_bins=nb, controls=Cm, group_id=group_id, demean=demean)

    id = _next_plot_id("bins")
    data_json = _xy_scatter_json(res.xbar, res.ybar; group=varnames[iy])
    groups_json = _series_json([varnames[iy]], [_palette(1)])
    overlays = if fitn === :none
        "[]"
    else
        _ols_fit_line(res.x_plot, res.y_plot;
                      degree=(fitn === :quadratic ? 2 : 1), color=_PLOT_ALERT)
    end
    js = _render_scatter_js(id, data_json, groups_json;
                            line_overlays_json=overlays,
                            xlabel=varnames[ix], ylabel=varnames[iy])

    parts = String["$(length(res.xbar)) bins"]
    isempty(cidx) || push!(parts, "controls: " * join([varnames[c] for c in cidx], ", "))
    demean === :within && push!(parts, "within-group")
    fitn === :none || push!(parts, "slope=$(_fmt(res.slope))")
    ptitle = "$(varnames[iy]) vs $(varnames[ix]) — " * join(parts, "; ")

    isempty(title) && (title = "Binscatter")
    p = _make_plot([_PanelSpec(id, ptitle, js)]; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
