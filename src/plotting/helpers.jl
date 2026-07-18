# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Internal helper functions for converting result struct fields to JSON data arrays.
"""

# =============================================================================
# Variable Resolution
# =============================================================================

"""
    _cap_title(base, shown, total) -> String

Append a `" (shown of total)"` suffix to a panel title when `shown < total`, so a
silent truncation becomes visible in the output (plotrule C7 "No silent truncation").
Returns `base` unchanged when everything is shown.
"""
_cap_title(base::AbstractString, shown::Integer, total::Integer) =
    shown < total ? "$(base) ($(shown) of $(total))" : String(base)

"""
    _cap_note(kind, shown, total, kw) -> String

Figure-level note for a truncation (plotrule C7): `"Showing <shown> of <total>
<kind> (raise with <kw>=)."` — used when the cap applies across every panel of a
multi-panel figure rather than to one panel's title. Empty when nothing is capped.
"""
_cap_note(kind::AbstractString, shown::Integer, total::Integer, kw::AbstractString) =
    shown < total ? "Showing $(shown) of $(total) $(kind) (raise with $(kw)=)." : ""

"""Resolve a variable name or index to an integer index."""
function _resolve_var(var::Union{Int,String,Nothing}, names::Vector{String}, default::Int=1)
    var === nothing && return default
    if var isa Int
        (1 <= var <= length(names)) ||
            throw(ArgumentError("Index $var out of range 1:$(length(names))"))
        return var
    end
    idx = findfirst(==(var), names)
    idx === nothing && throw(ArgumentError("Variable '$var' not found. Available: $names"))
    idx
end

# =============================================================================
# Series Config JSON
# =============================================================================

"""Build series config JSON array for chart renderers."""
function _series_json(names::Vector{String}, colors::Vector{String};
                      keys::Union{Vector{String},Nothing}=nothing,
                      dash::Union{Vector{String},Nothing}=nothing)
    n = length(names)
    ks = something(keys, ["s$i" for i in 1:n])
    ds = something(dash, fill("", n))
    parts = String[]
    for i in 1:n
        c = colors[mod1(i, length(colors))]
        push!(parts, "{\"name\":$(_json(names[i])),\"color\":$(_json(c)),\"key\":$(_json(ks[i])),\"dash\":$(_json(ds[i]))}")
    end
    "[" * join(parts, ",") * "]"
end

# =============================================================================
# IRF Data
# =============================================================================

"""Convert IRF arrays to JSON data array with x, irf, ci_lower, ci_upper, zero."""
function _irf_data_json(values::AbstractVector, ci_lo::AbstractVector,
                        ci_up::AbstractVector, H::Int)
    rows = Vector{Pair{String,String}}[]
    for h in 1:H
        push!(rows, [
            "x" => _json(h - 1),
            "irf" => _json(values[h]),
            "ci_lo" => _json(ci_lo[h]),
            "ci_hi" => _json(ci_up[h]),
            "zero" => "0"
        ])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# FEVD Data
# =============================================================================

"""Convert FEVD proportions to JSON data array with x + shock columns."""
function _fevd_data_json(proportions::AbstractMatrix, shock_names::Vector{String}, H::Int)
    # proportions: H × n_shocks (for a single variable)
    rows = Vector{Pair{String,String}}[]
    for h in 1:H
        row = Pair{String,String}["x" => _json(h)]
        for (j, name) in enumerate(shock_names)
            push!(row, "s$j" => _json(proportions[h, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# HD Data
# =============================================================================

"""Convert HD contributions to JSON data array with x + shock columns."""
function _hd_data_json(contributions::AbstractMatrix, shock_names::Vector{String}, T_eff::Int)
    # contributions: T_eff × n_shocks (for a single variable)
    rows = Vector{Pair{String,String}}[]
    for t in 1:T_eff
        row = Pair{String,String}["x" => _json(t)]
        for (j, name) in enumerate(shock_names)
            push!(row, "s$j" => _json(contributions[t, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Filter Data
# =============================================================================

"""
Convert filter trend/cycle to JSON data array.

The x value of drawn point `i` is its **calendar position** `i + offset` (filters
like Hamilton/BK drop the first `offset` observations). When `original` is supplied
it must cover that calendar range — `length(original) >= n + offset` — otherwise an
`ArgumentError` is thrown naming the expected length; the auxiliary series is never
silently shifted or truncated (plotrule Robustness "Misaligned auxiliary series",
Anti-Pattern #5). The reading `original[i + offset]` then aligns each drawn point to
its true calendar observation.
"""
function _filter_data_json(trend::AbstractVector, cyc::AbstractVector;
                           original::Union{AbstractVector,Nothing}=nothing,
                           offset::Int=0)
    n = length(trend)
    if original !== nothing && length(original) < n + offset
        throw(ArgumentError("original has length $(length(original)); expected at least " *
            "$(n + offset) to cover the filter's calendar range (offset=$offset, n=$n)"))
    end
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        row = Pair{String,String}["x" => _json(i + offset)]
        push!(row, "trend" => _json(trend[i]))
        push!(row, "cycle" => _json(cyc[i]))
        if original !== nothing
            push!(row, "orig" => _json(original[i + offset]))
        end
        push!(row, "zero" => "0")
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Forecast Data
# =============================================================================

"""Convert forecast + optional history to JSON data array."""
function _forecast_data_json(fc::AbstractVector, ci_lo::AbstractVector,
                             ci_up::AbstractVector;
                             history::Union{AbstractVector,Nothing}=nothing,
                             n_history::Int=50)
    rows = Vector{Pair{String,String}}[]
    start_idx = 0

    if history !== nothing
        nh = min(n_history, length(history))
        start_idx = -nh
        for i in 1:nh
            t = i - nh  # negative time index for history
            push!(rows, [
                "x" => _json(t),
                "hist" => _json(history[end - nh + i]),
                "fc" => "null",
                "ci_lo" => "null",
                "ci_hi" => "null"
            ])
        end
    end

    for i in 1:length(fc)
        push!(rows, [
            "x" => _json(i),
            "hist" => "null",
            "fc" => _json(fc[i]),
            "ci_lo" => _json(ci_lo[i]),
            "ci_hi" => _json(ci_up[i])
        ])
    end

    # Bridge point: connect history to forecast. The last history row already carries
    # a `"fc" => "null"` pair — OVERWRITE it (never push a second `"fc"`, which would
    # emit a duplicate JSON key that only parses by browser accident; plotrule A7).
    if history !== nothing && !isempty(rows)
        bridge_row = rows[end - length(fc)]
        bi = findfirst(p -> first(p) == "fc", bridge_row)
        if bi === nothing
            push!(bridge_row, "fc" => _json(history[end]))
        else
            bridge_row[bi] = "fc" => _json(history[end])
        end
    end

    _json_array_of_objects(rows)
end

# =============================================================================
# Event-study (dot-and-whisker) Data
# =============================================================================

"""
    _whisker_data_json(event_times, coefs, ci_lo, ci_hi, reference_period) -> String

Build the `[{x, y, lo, hi, ref}]` payload for `_render_whisker_js` (event-study
coefplot; PLT-17). Each estimated event time carries its point + CI; the reference /
omitted period is emitted with `ref:1`, `null` CI (so the renderer draws a hollow
marker and no whisker). When `reference_period` is not among `event_times`, a
synthetic `(reference_period, 0)` row is added so the omitted category is always
visible. Rows are sorted by event time.
"""
function _whisker_data_json(event_times::AbstractVector{<:Integer},
                            coefs::AbstractVector, ci_lo::AbstractVector,
                            ci_hi::AbstractVector, reference_period::Integer)
    xs = Int[]; ys = Float64[]
    los = Union{Float64,Nothing}[]; his = Union{Float64,Nothing}[]; refs = Int[]
    ref_present = false
    for i in eachindex(event_times)
        et = Int(event_times[i]); isref = et == Int(reference_period)
        isref && (ref_present = true)
        push!(xs, et); push!(ys, Float64(coefs[i]))
        push!(los, isref ? nothing : Float64(ci_lo[i]))
        push!(his, isref ? nothing : Float64(ci_hi[i]))
        push!(refs, isref ? 1 : 0)
    end
    if !ref_present
        push!(xs, Int(reference_period)); push!(ys, 0.0)
        push!(los, nothing); push!(his, nothing); push!(refs, 1)
    end
    perm = sortperm(xs)
    rows = Vector{Pair{String,String}}[]
    for k in perm
        push!(rows, ["x" => _json(xs[k]), "y" => _json(ys[k]),
                     "lo" => _json(los[k]), "hi" => _json(his[k]), "ref" => _json(refs[k])])
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Volatility Data
# =============================================================================

"""Convert returns + conditional variance to JSON data array."""
function _volatility_data_json(y::AbstractVector, cond_var::AbstractVector)
    n = length(y)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        push!(rows, [
            "x" => _json(i),
            "ret" => _json(y[i]),
            "vol" => _json(sqrt(max(cond_var[min(i, length(cond_var))], 0.0))),
            "std_resid" => _json(y[i] / sqrt(max(cond_var[min(i, length(cond_var))], 1e-10)))
        ])
    end
    _json_array_of_objects(rows)
end

"""Convert SV model to JSON with posterior quantile bands."""
function _sv_data_json(y::AbstractVector, vol_mean::AbstractVector,
                       vol_q::AbstractMatrix, q_levels::AbstractVector)
    n = length(y)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        row = Pair{String,String}[
            "x" => _json(i),
            "ret" => _json(y[i]),
            "vol_mean" => _json(sqrt(max(vol_mean[i], 0.0))),
            "std_resid" => _json(y[i] / sqrt(max(vol_mean[i], 1e-10)))
        ]
        for (qi, ql) in enumerate(q_levels)
            push!(row, "q$(qi)" => _json(sqrt(max(vol_q[i, qi], 0.0))))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Time Series Data
# =============================================================================

"""Convert a time series matrix to JSON data array with x + variable columns."""
function _timeseries_data_json(data::AbstractMatrix, varnames::Vector{String};
                               time_index::Union{AbstractVector,Nothing}=nothing)
    T_obs, n_vars = size(data)
    rows = Vector{Pair{String,String}}[]
    for t in 1:T_obs
        row = Pair{String,String}["x" => _json(time_index === nothing ? t : time_index[t])]
        for j in 1:n_vars
            push!(row, "v$j" => _json(data[t, j]))
        end
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

# =============================================================================
# Date / calendar axis helpers (PLT-08)
# =============================================================================

"""
    _x_ticks_json(xvals, labels) -> String

Build the renderer `x_ticks_json` payload: `"null"` when `labels === nothing`
(renderer falls back to integer/auto ticks), otherwise a JSON array of
`{"v":<number>,"label":<string>}` with `labels` parallel to `xvals`. `v` is the
numeric data x-position each label attaches to; the renderer draws ticks only at
these positions (no fractional/date-less ticks).
"""
function _x_ticks_json(xvals::AbstractVector, labels::Union{AbstractVector,Nothing})
    labels === nothing && return "null"
    length(labels) == length(xvals) || throw(ArgumentError(
        "x_ticks labels length ($(length(labels))) must match xvals length ($(length(xvals)))"))
    rows = Vector{Pair{String,String}}[]
    for (v, lab) in zip(xvals, labels)
        push!(rows, ["v" => _json(v), "label" => _json(string(lab))])
    end
    _json_array_of_objects(rows)
end

"""
    _fmt_period(freq::Frequency, k::Integer) -> String

Format a bare period index `k` (1-based) under a known frequency as a compact
label when no calendar dates are stored. Without a calendar anchor this yields a
frequency-flavoured position label (e.g. quarterly `"Q3"`, monthly `"M07"`), used
only as a fallback; when real `dates` exist they take precedence.
"""
function _fmt_period(freq::Frequency, k::Integer)
    if freq == Quarterly
        return "Q$(mod1(k, 4))"
    elseif freq == Monthly
        return "M$(lpad(mod1(k, 12), 2, '0'))"
    elseif freq == Yearly
        return string(k)
    else
        return string(k)
    end
end

"""
    _date_axis_labels(d::TimeSeriesData) -> Union{Vector{String},Nothing}

Return one calendar label per observation for the x-axis, or `nothing` (integer
fallback). Uses `d.dates` when populated (via `set_dates!`); otherwise `nothing`
— without a calendar anchor the integer `time_index` cannot be turned into real
dates, so the axis stays numeric rather than fabricating one.
"""
function _date_axis_labels(d::TimeSeriesData)
    dts = dates(d)
    isempty(dts) ? nothing : dts
end

# =============================================================================
# Statistical converters for the PLT-19 primitives (histogram / KDE / box / fan)
# and the data-analysis + diagnostic Wave-2 lanes. Statistics live here (A5/A6),
# never inline in a renderer or a plot method. helpers.jl is FROZEN for Wave 2 —
# every cross-lane converter ships here now.
# =============================================================================

"""
    _histogram_bins(x; n_bins=0, density=false) -> String

Build the `bins_json` payload `[{x0, x1, y}, …]` of contiguous linear-x bins for
`_render_histogram_js`. Non-finite (`NaN`/`Inf`) values are dropped. `n_bins == 0`
picks the bin count by the Freedman–Diaconis rule (`h = 2·IQR·n^{-1/3}`), falling
back to the `√n` rule when the IQR is zero. `density=true` scales bar heights to a
probability density (`count / (n · width)`, integrates to 1); otherwise heights are
raw counts. A constant/single-obs series yields one padded unit-width bin.
"""
function _histogram_bins(x::AbstractVector; n_bins::Int=0, density::Bool=false)
    v = Float64[Float64(xi) for xi in x if isfinite(xi)]
    n = length(v)
    n == 0 && return "[]"
    mn, mx = extrema(v)
    if mx == mn
        half = mn == 0 ? 0.5 : max(abs(mn) * 0.05, 0.5)
        yval = density ? 1.0 / (2 * half) : Float64(n)
        return "[{\"x0\":$(_json(mn - half)),\"x1\":$(_json(mn + half)),\"y\":$(_json(yval))}]"
    end
    k = if n_bins > 0
        n_bins
    else
        sorted = sort(v)
        iqr = _quantile(sorted, 0.75) - _quantile(sorted, 0.25)
        h = iqr > 0 ? 2 * iqr * n^(-1 / 3) : 0.0
        (h > 0 && isfinite(h)) ? clamp(ceil(Int, (mx - mn) / h), 1, 200) : max(1, ceil(Int, sqrt(n)))
    end
    k = max(1, k)
    width = (mx - mn) / k
    counts = zeros(Int, k)
    for xi in v
        bi = clamp(floor(Int, (xi - mn) / width) + 1, 1, k)
        counts[bi] += 1
    end
    rows = Vector{Pair{String,String}}[]
    for b in 1:k
        yval = density ? counts[b] / (n * width) : Float64(counts[b])
        push!(rows, ["x0" => _json(mn + (b - 1) * width),
                     "x1" => _json(mn + b * width),
                     "y"  => _json(yval)])
    end
    _json_array_of_objects(rows)
end

"""
    _kde_line(x; n_grid=200, bw=0) -> (xs::Vector{Float64}, dens::Vector{Float64})

Gaussian kernel density estimate of `x` on an `n_grid`-point grid spanning
`[min-3h, max+3h]`. `bw == 0` uses the Silverman bandwidth `1.06·σ·n^{-1/5}`; a
positive `bw` overrides it. Non-finite values are dropped. Returns the grid and
density so callers (e.g. the BayesianDSGE prior/posterior overlay) can evaluate a
second curve on the SAME grid; use `_kde_line_json` for the histogram overlay.
"""
function _kde_line(x::AbstractVector; n_grid::Int=200, bw::Real=0)
    v = Float64[Float64(xi) for xi in x if isfinite(xi)]
    n = length(v)
    n == 0 && return (Float64[], Float64[])
    h = bw > 0 ? Float64(bw) : 1.06 * std(v) * n^(-0.2)
    h = max(h, 1e-10)
    lo = minimum(v) - 3h
    hi = maximum(v) + 3h
    hi <= lo && (hi = lo + 1.0)
    xs = collect(range(lo, hi; length=n_grid))
    dens = zeros(Float64, n_grid)
    c = 1.0 / (n * h * sqrt(2π))
    for (gi, xg) in enumerate(xs)
        acc = 0.0
        for dv in v
            z = (xg - dv) / h
            acc += exp(-0.5 * z * z)
        end
        dens[gi] = c * acc
    end
    (xs, dens)
end

"""
    _kde_line_json(x; n_grid=200, bw=0) -> String

KDE overlay payload `[{x, d}, …]` for the `density_json` argument of
`_render_histogram_js`. Thin wrapper over `_kde_line`.
"""
function _kde_line_json(x::AbstractVector; n_grid::Int=200, bw::Real=0)
    xs, dens = _kde_line(x; n_grid=n_grid, bw=bw)
    rows = Vector{Pair{String,String}}[]
    for i in eachindex(xs)
        push!(rows, ["x" => _json(xs[i]), "d" => _json(dens[i])])
    end
    _json_array_of_objects(rows)
end

"""
    _boxplot_stats(x) -> NamedTuple

Tukey box-plot statistics of `x` (non-finite dropped): `q1`/`med`/`q3` via
`_quantile`, whiskers = the furthest observation within `1.5·IQR` of the box, and
`outliers` beyond the fences. A constant series collapses (`whislo == q1 == med ==
q3 == whishi`, no outliers).
"""
function _boxplot_stats(x::AbstractVector)
    v = sort(Float64[Float64(xi) for xi in x if isfinite(xi)])
    n = length(v)
    n == 0 && return (whislo=0.0, q1=0.0, med=0.0, q3=0.0, whishi=0.0, mean=0.0, outliers=Float64[])
    q1 = _quantile(v, 0.25); med = _quantile(v, 0.5); q3 = _quantile(v, 0.75)
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr; hi_fence = q3 + 1.5 * iqr
    inb = Float64[xi for xi in v if lo_fence <= xi <= hi_fence]
    whislo = isempty(inb) ? q1 : minimum(inb)
    whishi = isempty(inb) ? q3 : maximum(inb)
    outliers = Float64[xi for xi in v if xi < lo_fence || xi > hi_fence]
    (whislo=whislo, q1=q1, med=med, q3=q3, whishi=whishi, mean=sum(v) / n, outliers=outliers)
end

"""
    _boxes_json(groups, cols) -> String

Build the `boxes_json` payload for `_render_box_js` from named `groups` and their
value vectors `cols` (each summarized by `_boxplot_stats`).
"""
function _boxes_json(groups::AbstractVector{<:AbstractString}, cols::AbstractVector)
    parts = String[]
    for (grp, c) in zip(groups, cols)
        s = _boxplot_stats(c)
        push!(parts, "{\"group\":$(_json(String(grp)))," *
            "\"whislo\":$(_json(s.whislo)),\"q1\":$(_json(s.q1)),\"med\":$(_json(s.med))," *
            "\"q3\":$(_json(s.q3)),\"whishi\":$(_json(s.whishi)),\"mean\":$(_json(s.mean))," *
            "\"outliers\":$(_json(s.outliers))}")
    end
    "[" * join(parts, ",") * "]"
end

"""
    _fan_data_json(quantiles, levels, central; xvals=nothing) -> String

Build the `data_json` payload `[{x, q1, …, qk, med}, …]` for `_render_fan_js`.
`quantiles` is `T×nq`, `levels` the `nq` quantile levels, `central` the length-`T`
central line (posterior median or mean). Quantile columns are re-keyed `q1..qnq`
in ascending-level order so `_fan_bands_json(levels)` pairs the right keys.
"""
function _fan_data_json(quantiles::AbstractMatrix, levels::AbstractVector,
                        central::AbstractVector;
                        xvals::Union{AbstractVector,Nothing}=nothing)
    Tn, nq = size(quantiles)
    length(levels) == nq || throw(ArgumentError(
        "levels length ($(length(levels))) must match quantile columns ($nq)"))
    length(central) == Tn || throw(ArgumentError(
        "central length ($(length(central))) must match quantile rows ($Tn)"))
    perm = sortperm(Float64[Float64(l) for l in levels])
    rows = Vector{Pair{String,String}}[]
    for t in 1:Tn
        row = Pair{String,String}["x" => _json(xvals === nothing ? t : xvals[t])]
        for (ci, j) in enumerate(perm)
            push!(row, "q$ci" => _json(quantiles[t, j]))
        end
        push!(row, "med" => _json(central[t]))
        push!(rows, row)
    end
    _json_array_of_objects(rows)
end

"""
    _fan_bands_json(levels; color=_PLOT_COLORS[1], alpha_min=0.12, alpha_max=0.35) -> String

Build the `fan_json` nested-band specs for `_render_fan_js` from `levels`: each
symmetric pair `(levels[i], levels[k+1-i])` becomes a band keyed `q{i}`/`q{k+1-i}`
(matching `_fan_data_json`'s sorted re-keying), labelled by its percent pair
(e.g. "5–95%"), in one `color` with alpha ramped from `alpha_min` (outer, widest)
to `alpha_max` (inner). The median level is not a band.
"""
function _fan_bands_json(levels::AbstractVector; color::String=_PLOT_COLORS[1],
                         alpha_min::Real=0.12, alpha_max::Real=0.35)
    ls = sort(Float64[Float64(l) for l in levels])
    k = length(ls)
    npairs = fld(k, 2)
    parts = String[]
    for i in 1:npairs
        lo = ls[i]; hi = ls[k + 1 - i]
        lo ≈ hi && continue
        frac = npairs == 1 ? 1.0 : (i - 1) / (npairs - 1)
        alpha = alpha_min + frac * (alpha_max - alpha_min)
        label = "$(round(Int, lo * 100))–$(round(Int, hi * 100))%"
        push!(parts, "{\"lo_key\":$(_json("q$i")),\"hi_key\":$(_json("q$(k + 1 - i)"))," *
            "\"label\":$(_json(label)),\"alpha\":$(_json(alpha)),\"color\":$(_json(color))}")
    end
    "[" * join(parts, ",") * "]"
end

"""
    _ols_fit_line(x, y; degree=1, color="#d62728", dash="", n_seg=60) -> String

Fit a degree-`degree` polynomial of `y` on `x` by OLS (non-finite pairs dropped)
and emit `line_overlays_json` rows `[{x1, y1, x2, y2, color, dash}, …]` for the
scatter renderer's data-coordinate overlay (A4). `degree == 1` returns one segment;
higher degrees return an `n_seg`-segment polyline across the x-range.
"""
function _ols_fit_line(x::AbstractVector, y::AbstractVector; degree::Int=1,
                       color::String="#d62728", dash::String="", n_seg::Int=60)
    idx = [i for i in eachindex(x) if isfinite(x[i]) && isfinite(y[i])]
    length(idx) < degree + 1 && return "[]"
    xv = Float64[Float64(x[i]) for i in idx]
    yv = Float64[Float64(y[i]) for i in idx]
    X = ones(Float64, length(xv), degree + 1)
    for d in 1:degree
        @views X[:, d + 1] .= xv .^ d
    end
    β = X \ yv
    xmn, xmx = extrema(xv)
    xmn == xmx && return "[]"
    fitval(xx) = sum(β[d + 1] * xx^d for d in 0:degree)
    seg(a, b) = "{\"x1\":$(_json(a)),\"y1\":$(_json(fitval(a)))," *
                "\"x2\":$(_json(b)),\"y2\":$(_json(fitval(b)))," *
                "\"color\":$(_json(color)),\"dash\":$(_json(dash))}"
    parts = String[]
    if degree == 1
        push!(parts, seg(xmn, xmx))
    else
        xs = range(xmn, xmx; length=n_seg + 1)
        for s in 1:n_seg
            push!(parts, seg(xs[s], xs[s + 1]))
        end
    end
    "[" * join(parts, ",") * "]"
end

"""
    _fwl_residualize(y, X) -> Vector{Float64}

Frisch–Waugh–Lovell residualization: partial `X` (with an intercept added) out of
`y` by OLS and return the residuals. `X` may be a vector (single control) or a
matrix. Used by binscatter's control adjustment.
"""
function _fwl_residualize(y::AbstractVector, X::AbstractVecOrMat)
    yf = Float64[Float64(v) for v in y]
    Xm = X isa AbstractVector ? reshape(Float64[Float64(v) for v in X], :, 1) :
                                Float64.(Matrix(X))
    n = length(yf)
    size(Xm, 1) == n || throw(ArgumentError(
        "X rows ($(size(Xm, 1))) must match y length ($n)"))
    Xf = hcat(ones(n), Xm)
    yf .- Xf * (Xf \ yf)
end

"""
    _corr_matrix_json(M; labels) -> String

Pairwise-complete Pearson correlation matrix over the columns of `M` (each labelled
by `labels`), emitted as heatmap cells `[{x, y, v}, …]` with `x`/`y` the column
labels and `v` the correlation (`null` for pairs with < 2 complete observations).
Feed the returned string plus `_json(labels)` (row + col) to `_render_heatmap_js`.
"""
function _corr_matrix_json(M::AbstractMatrix; labels::Vector{String})
    n, k = size(M)
    length(labels) == k || throw(ArgumentError(
        "labels length ($(length(labels))) must match columns ($k)"))
    C = fill(NaN, k, k)
    for a in 1:k, b in 1:k
        rows = [i for i in 1:n if isfinite(M[i, a]) && isfinite(M[i, b])]
        if length(rows) >= 2
            va = Float64[Float64(M[i, a]) for i in rows]
            vb = Float64[Float64(M[i, b]) for i in rows]
            ma = sum(va) / length(va); mb = sum(vb) / length(vb)
            da = va .- ma; db = vb .- mb
            denom = sqrt(sum(abs2, da) * sum(abs2, db))
            C[a, b] = denom > 0 ? sum(da .* db) / denom : NaN
        end
    end
    rows = Vector{Pair{String,String}}[]
    for a in 1:k, b in 1:k
        push!(rows, ["x" => _json(labels[b]), "y" => _json(labels[a]), "v" => _json(C[a, b])])
    end
    _json_array_of_objects(rows)
end

# Standard-normal inverse CDF for Q-Q theoretical quantiles (Distributions is a dep;
# `quantile` dispatches on the distribution, so no Statistics.quantile clash).
_norm_quantile(p::Real) = quantile(Normal(), clamp(Float64(p), 1e-6, 1 - 1e-6))

"""
    _residual_diagnostics_panels(resid, fitted; varname="", acf_lags=0, standardized=false)
        -> Vector{_PanelSpec}

The shared four-panel residual-diagnostics figure (A6 — one implementation reused by
`RegModel` and every `residuals`-bearing family in PLT-24/25):

1. residual-vs-fitted scatter with a zero reference line;
2. residual histogram (density) with a fitted-normal overlay curve;
3. Normal Q-Q via the scatter renderer with a 45° `line_overlays_json` reference
   (replaces the old `reg.jl` scale-clone, A4);
4. residual ACF via `_render_vbar_js` + `acf`, with ±CI reference lines (lags ≥ 1).

`varname` labels the residual axes; `standardized` flags standardized residuals;
`acf_lags == 0` auto-picks `min(24, n/4)`.
"""
function _residual_diagnostics_panels(resid::AbstractVector, fitted::AbstractVector;
                                      varname::String="", acf_lags::Int=0,
                                      standardized::Bool=false)
    rv = Float64[Float64(v) for v in resid]
    fv = Float64[Float64(v) for v in fitted]
    finite_r = Float64[v for v in rv if isfinite(v)]
    rlabel = (isempty(varname) ? "" : varname * " ") *
             (standardized ? "Std. Residual" : "Residual")
    panels = _PanelSpec[]

    # Panel 1 — residual vs fitted (+ horizontal zero line)
    id1 = _next_plot_id("resid_fit")
    rows1 = Vector{Pair{String,String}}[]
    for i in eachindex(rv)
        push!(rows1, ["x" => _json(fv[i]), "y" => _json(rv[i]), "group" => _json("resid")])
    end
    groups1 = "[{\"name\":\"resid\",\"color\":$(_json(_PLOT_COLORS[1]))}]"
    refs1 = "[{\"value\":0,\"color\":\"#d62728\",\"dash\":\"4,3\"}]"
    js1 = _render_scatter_js(id1, _json_array_of_objects(rows1), groups1;
                             ref_lines_json=refs1, xlabel="Fitted", ylabel=rlabel)
    push!(panels, _PanelSpec(id1, "Residual vs Fitted", js1))

    # Panel 2 — residual histogram (density) + fitted-normal overlay
    id2 = _next_plot_id("resid_hist")
    bins2 = _histogram_bins(rv; density=true)
    dens2 = "[]"
    if length(finite_r) >= 2
        m = sum(finite_r) / length(finite_r)
        sd = sqrt(sum(abs2, finite_r .- m) / (length(finite_r) - 1))
        if sd > 0
            xs = range(minimum(finite_r), maximum(finite_r); length=100)
            rows2 = Vector{Pair{String,String}}[]
            for xg in xs
                dv = exp(-0.5 * ((xg - m) / sd)^2) / (sd * sqrt(2π))
                push!(rows2, ["x" => _json(xg), "d" => _json(dv)])
            end
            dens2 = _json_array_of_objects(rows2)
        end
    end
    series2 = "[{\"name\":\"Residuals\",\"color\":$(_json(_PLOT_COLORS[1]))}," *
              "{\"name\":\"Normal\",\"color\":\"#d62728\"}]"
    js2 = _render_histogram_js(id2, bins2, series2; density_json=dens2,
                               xlabel=rlabel, ylabel="Density")
    push!(panels, _PanelSpec(id2, "Residual Distribution", js2))

    # Panel 3 — Normal Q-Q (scatter + 45° overlay line)
    id3 = _next_plot_id("resid_qq")
    sr = sort(finite_r); nq = length(sr)
    rows3 = Vector{Pair{String,String}}[]
    if nq >= 1
        m = sum(sr) / nq
        sd = nq >= 2 ? sqrt(sum(abs2, sr .- m) / (nq - 1)) : 1.0
        for i in 1:nq
            theo = _norm_quantile((i - 0.5) / nq)
            samp = sd > 0 ? (sr[i] - m) / sd : 0.0
            push!(rows3, ["x" => _json(theo), "y" => _json(samp), "group" => _json("qq")])
        end
    end
    groups3 = "[{\"name\":\"qq\",\"color\":$(_json(_PLOT_COLORS[1]))}]"
    a = 3.5
    overlay3 = "[{\"x1\":$(_json(-a)),\"y1\":$(_json(-a)),\"x2\":$(_json(a)),\"y2\":$(_json(a))," *
               "\"color\":\"#d62728\",\"dash\":\"4,3\"}]"
    js3 = _render_scatter_js(id3, _json_array_of_objects(rows3), groups3;
                             line_overlays_json=overlay3,
                             xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")
    push!(panels, _PanelSpec(id3, "Normal Q-Q", js3))

    # Panel 4 — residual ACF (vbar) + ±CI reference lines (needs ≥3 obs for `acf`)
    id4 = _next_plot_id("resid_acf")
    data4 = "[]"; ci4 = "[]"
    if length(finite_r) >= 3
        nlags = acf_lags > 0 ? acf_lags : max(1, min(24, fld(length(finite_r), 4)))
        ar = acf(finite_r; lags=nlags)
        rows4 = Vector{Pair{String,String}}[]
        for i in eachindex(ar.lags)
            push!(rows4, ["x" => _json(ar.lags[i]), "y" => _json(ar.acf[i])])
        end
        data4 = _json_array_of_objects(rows4)
        ci4 = "[{\"value\":$(_json(ar.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"}," *
              "{\"value\":$(_json(-ar.ci)),\"color\":\"#d62728\",\"dash\":\"5,4\"}]"
    end
    js4 = _render_vbar_js(id4, data4;
                          bar_color=_PLOT_COLORS[1], ref_lines_json=ci4,
                          xlabel="Lag", ylabel="Residual ACF")
    push!(panels, _PanelSpec(id4, "Residual ACF", js4))

    panels
end
