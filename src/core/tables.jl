# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Tables.jl integration (T247 / #346)
# =============================================================================
# Make result objects programmatically tabular so `DataFrame(result)`, CSV export,
# and R/Python hand-off work uniformly.
#
#   1. Coefficient-bearing models (RegModel, Logit/Probit, Ordered/Multinomial,
#      the PanelReg family, VARModel, MarginalEffects, DIDResult, count models,
#      QuantileRegModel, RDDResult, SUR/3SLS, DiD event-study results,
#      MultinomialMarginalEffects) implement the Tables.jl *source* interface —
#      the natural coefficient table (term, estimate, std_error, stat, p_value,
#      ci_lower, ci_upper, plus key columns) is the exposed table, so
#      `DataFrame(m)` returns the same numbers `report(m)` prints.
#   2. Array-valued results (IRF, FEVD, forecasts) expose a tidy/long table via
#      `long_table(result)` — one row per cell, with explicit horizon/variable/shock
#      keys — since a single rectangular shape is ambiguous.
#   3. `write_csv(result, path)` writes any of the above to CSV using the stdlib
#      `DelimitedFiles` (no CSV.jl dependency).
#
# This is purely additive: `report()` output is unchanged. The Tables source
# interface is implemented; DataFrames is reached only through Tables, so no direct
# DataFrames dependency is required for `DataFrame(result)` to work.

# ─────────────────────────────────────────────────────────────────────────────
# Numeric coefficient columns (raw Float64, not the display strings of _coef_table)
# ─────────────────────────────────────────────────────────────────────────────

# Build the tidy NamedTuple of coefficient columns from the same (names, coef, se)
# inputs a type's `report()` feeds `_coef_table`, computing stat/p/CI numerically.
function _coef_nt_simple(names, coefs, se, dist::Symbol, dof_r::Int; level::Real=0.95)
    est = Float64.(collect(coefs))
    s   = Float64.(collect(se))
    stat = est ./ s
    a = 1 - level
    use_t = dist === :t && dof_r >= 1          # fall back to z when dof is degenerate
    if use_t
        zc = quantile(TDist(dof_r), 1 - a / 2)
        pval = Float64[2 * (1 - cdf(TDist(dof_r), abs(z))) for z in stat]
    else
        zc = quantile(Normal(), 1 - a / 2)
        pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    end
    return (term = string.(collect(names)),
            estimate = est,
            std_error = s,
            stat = stat,
            p_value = pval,
            ci_lower = est .- zc .* s,
            ci_upper = est .+ zc .* s)
end

# Single-block StatsAPI models — reuse coef/stderror/dof_residual + varnames.
_coef_nt(m::RegModel)        = _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
_coef_nt(m::PanelRegModel)   = _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
_coef_nt(m::PanelIVModel)    = _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
_coef_nt(m::LogitModel)      = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::ProbitModel)     = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::PanelLogitModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::PanelProbitModel)= _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)

# MarginalEffects — every column is precomputed; drop the non-finite (intercept) rows
# exactly as `report()` does.
function _coef_nt(me::MarginalEffects)
    keep = findall(isfinite, me.effects)
    return (term = me.varnames[keep],
            estimate = Float64.(me.effects[keep]),
            std_error = Float64.(me.se[keep]),
            stat = Float64.(me.z_stat[keep]),
            p_value = Float64.(me.p_values[keep]),
            ci_lower = Float64.(me.ci_lower[keep]),
            ci_upper = Float64.(me.ci_upper[keep]))
end

# DIDResult — event-study coefficients keyed by event time; CI bounds are stored.
function _coef_nt(r::DIDResult)
    est = Float64.(collect(r.att))
    s   = Float64.(collect(r.se))
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (event_time = collect(r.event_times),
            term = ["e=$(e)" for e in r.event_times],
            estimate = est,
            std_error = s,
            stat = stat,
            p_value = pval,
            ci_lower = Float64.(collect(r.ci_lower)),
            ci_upper = Float64.(collect(r.ci_upper)))
end

# Ordered logit/probit — two blocks (slopes + cutpoints) tagged by a `block` column.
_coef_nt(m::OrderedLogitModel)  = _coef_nt_ordered(m)
_coef_nt(m::OrderedProbitModel) = _coef_nt_ordered(m)
function _coef_nt_ordered(m)
    K = length(m.beta)
    J = length(m.cutpoints)
    names = vcat(String.(m.varnames), ["cut$j" for j in 1:J])
    coefs = vcat(collect(m.beta), collect(m.cutpoints))
    se    = stderror(m)                       # joint SE vector, length K + J
    block = vcat(fill("coef", K), fill("cutpoint", J))
    base = _coef_nt_simple(names, coefs, se, :z, 0)
    return merge((block = block,), base)
end

# Multinomial logit — one block of K rows per non-base alternative, tagged by `alternative`.
function _coef_nt(m::MultinomialLogitModel)
    K = length(m.varnames)
    se_all = stderror(m)                      # length K*(J-1), column-major over alternatives
    J_1 = size(m.beta, 2)
    alts = string.(m.categories)              # base category = alts[1]; categories may be Int/Symbol/String
    terms = String[]; alt = String[]; est = Float64[]; s = Float64[]
    for j in 1:J_1
        off = (j - 1) * K
        append!(terms, String.(m.varnames))
        append!(alt, fill("$(alts[j+1]) vs $(alts[1])", K))
        append!(est, Float64.(m.beta[:, j]))
        append!(s, Float64.(se_all[off+1:off+K]))
    end
    base = _coef_nt_simple(terms, est, s, :z, 0)
    return merge((alternative = alt,), base)
end

# VARModel — one row per (equation, term); SEs from the equation-by-equation OLS vcov,
# mirroring `report(::VARModel)`.
function _coef_nt(model::VARModel)
    n = nvars(model)
    p = model.p
    coef_names = vcat(_INTERCEPT_LABEL, ["$(model.varnames[v]).L$l" for l in 1:p for v in 1:n])
    _, X = construct_var_matrices(model.Y, p)
    XtX_inv = robust_inv(Matrix(X' * X))
    dof_r = effective_nobs(model) - ncoefs(model)
    eqs = String[]; terms = String[]; est = Float64[]; s = Float64[]
    for j in 1:n
        se_j = sqrt.(max.(diag(XtX_inv) .* model.Sigma[j, j], 0))
        append!(eqs, fill(model.varnames[j], length(coef_names)))
        append!(terms, coef_names)
        append!(est, Float64.(model.B[:, j]))
        append!(s, Float64.(se_j))
    end
    base = _coef_nt_simple(terms, est, s, :t, dof_r)
    return merge((equation = eqs,), base)
end

# ── TIDY batch (v1.0.2, #853 series) ──────────────────────────────────────────

# Count models — z-based, mirroring `report()`. NegBin appends the dispersion
# row tagged by `block`, like the ordered two-block shape. (#854)
_coef_nt(m::PoissonModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
function _coef_nt(m::NegBinModel)
    k = length(m.beta)
    base = _coef_nt_simple(vcat(String.(m.varnames), ["alpha"]),
        vcat(Float64.(m.beta), [Float64(m.alpha)]),
        vcat(Float64.(stderror(m)), [Float64(m.alpha_se)]), :z, 0)
    block = vcat(fill("coef", k), ["dispersion"])
    return merge((block = block,), base)
end

# Quantile regression — one block per tau tagged by a `tau` column; same
# t-distributed inference (dof n-k) as each per-quantile `report()` table. (#855)
function _coef_nt(m::QuantileRegModel)
    dof_r = size(m.X, 1) - size(m.X, 2)
    taus = Float64[]; terms = String[]; est = Float64[]; s = Float64[]
    for (j, t) in enumerate(m.taus)
        append!(taus, fill(Float64(t), length(m.varnames)))
        append!(terms, String.(m.varnames))
        append!(est, Float64.(m.beta[:, j]))
        append!(s, Float64.(m.stderr[:, j]))
    end
    base = _coef_nt_simple(terms, est, s, :t, dof_r)
    return merge((tau = taus,), base)
end

# RDD — conventional + robust treatment-effect rows at the stored level, with
# the main/pilot bandwidths attached. (#855)
function _coef_nt(r::RDDResult)
    base = _coef_nt_simple(["Conventional", "Robust (bias-corrected)"],
        [r.tau_conventional, r.tau_bias_corrected],
        [r.se_conventional, r.se_robust], :z, 0; level=r.level)
    return merge(base, (h = fill(Float64(r.h), 2), b = fill(Float64(r.b), 2)))
end

# SUR / 3SLS — one row per (equation, term); dof varies per equation so each
# block is built separately, mirroring the per-equation `report()` tables. (#856)
function _coef_nt_eqstack(eqnames, varnames, betas, ses, nobs)
    eqs = String[]; terms = String[]; est = Float64[]; s = Float64[]
    stat = Float64[]; pval = Float64[]; lo = Float64[]; hi = Float64[]
    for j in eachindex(eqnames)
        blk = _coef_nt_simple(varnames[j], betas[j], ses[j], :t,
            max(nobs - length(betas[j]), 1))
        append!(eqs, fill(eqnames[j], length(blk.term)))
        append!(terms, blk.term); append!(est, blk.estimate); append!(s, blk.std_error)
        append!(stat, blk.stat); append!(pval, blk.p_value)
        append!(lo, blk.ci_lower); append!(hi, blk.ci_upper)
    end
    return (equation=eqs, term=terms, estimate=est, std_error=s, stat=stat,
        p_value=pval, ci_lower=lo, ci_upper=hi)
end

function _coef_nt(m::SURModel)
    base = _coef_nt_eqstack(m.eqnames, m.varnames, m.betas, m.ses, m.nobs)
    n = length(base.term)
    return merge(base, (nobs=fill(m.nobs, n), mcelroy_r2=fill(Float64(m.mcelroy_r2), n),
        det_sigma=fill(Float64(m.det_sigma), n), loglik=fill(Float64(m.loglik), n)))
end

function _coef_nt(m::ThreeSLSModel)
    base = _coef_nt_eqstack(m.eqnames, m.varnames, m.betas, m.ses, m.nobs)
    n = length(base.term)
    ni = Int[]
    for j in eachindex(m.eqnames)
        append!(ni, fill(m.n_instruments[j], length(m.betas[j])))
    end
    return merge(base, (nobs=fill(m.nobs, n), mcelroy_r2=fill(Float64(m.mcelroy_r2), n),
        det_sigma=fill(Float64(m.det_sigma), n), n_instruments=ni))
end

# EventStudyLP — event-time coefficients keyed by event time; CIs are stored,
# like `DIDResult`. (#866)
function _coef_nt(r::EventStudyLP)
    est = Float64.(collect(r.coefficients))
    s = Float64.(collect(r.se))
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (event_time = collect(r.event_times),
            term = ["h=$(e)" for e in r.event_times],
            estimate = est,
            std_error = s,
            stat = stat,
            p_value = pval,
            ci_lower = Float64.(collect(r.ci_lower)),
            ci_upper = Float64.(collect(r.ci_upper)))
end

# LPDiDResult — dynamic rows plus pooled rows tagged by `block`; pooled rows
# carry `missing` event times. (#866)
function _coef_nt(r::LPDiDResult)
    et = Union{Missing,Int}[e for e in r.event_times]
    term = ["h=$(e)" for e in r.event_times]
    est = Float64.(collect(r.coefficients))
    s = Float64.(collect(r.se))
    lo = Float64.(collect(r.ci_lower))
    hi = Float64.(collect(r.ci_upper))
    block = fill("dynamic", length(term))
    for (nm, pool) in (("Pre-pooled", r.pooled_pre), ("Post-pooled", r.pooled_post))
        pool === nothing && continue
        push!(et, missing); push!(term, nm)
        push!(est, Float64(pool.coef)); push!(s, Float64(pool.se))
        push!(lo, Float64(pool.ci_lower)); push!(hi, Float64(pool.ci_upper))
        push!(block, "pooled")
    end
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (block=block, event_time=et, term=term, estimate=est, std_error=s,
        stat=stat, p_value=pval, ci_lower=lo, ci_upper=hi)
end

# BaconDecomposition — one row per 2x2 comparison. (#866)
function _coef_nt(r::BaconDecomposition)
    return (type=string.(r.comparison_type),
            cohort_i=collect(r.cohort_i), cohort_j=collect(r.cohort_j),
            estimate=Float64.(collect(r.estimates)),
            weight=Float64.(collect(r.weights)))
end

# Per-category marginal effects — long (variable, category) rows with z-based
# inference. `se === nothing` (model covariance unavailable) yields NaN
# inference columns, matching the NaN fallback of the ordered AME. (#863)
function _marginal_effects_long(effects::AbstractMatrix, se, varnames, categories;
        skip_first::Bool)
    keep = findall(v -> _display_intercept(v) != _INTERCEPT_LABEL, varnames)
    isempty(keep) && (keep = collect(1:size(effects, 1)))
    jrange = skip_first ? (2:size(effects, 2)) : (1:size(effects, 2))
    variable = String[]; category = String[]; est = Float64[]; s = Float64[]
    for j in jrange, i in keep
        push!(variable, string(varnames[i])); push!(category, string(categories[j]))
        push!(est, Float64(effects[i, j]))
        push!(s, se === nothing ? NaN : Float64(se[i, j]))
    end
    base = _coef_nt_simple(variable, est, s, :z, 0)
    return (variable=base.term, category=category, estimate=base.estimate,
        std_error=base.std_error, stat=base.stat, p_value=base.p_value,
        ci_lower=base.ci_lower, ci_upper=base.ci_upper)
end

# Multinomial AME skips the base category (column 1), mirroring `show()`.
_coef_nt(me::MultinomialMarginalEffects) =
    _marginal_effects_long(me.effects, me.se, me.varnames, me.categories; skip_first=true)

# ─────────────────────────────────────────────────────────────────────────────
# Tables.jl source interface for coefficient-bearing types
# ─────────────────────────────────────────────────────────────────────────────

const _COEF_TABLE_TYPES = (RegModel, LogitModel, ProbitModel, PanelRegModel, PanelIVModel,
    PanelLogitModel, PanelProbitModel, MarginalEffects, OrderedLogitModel, OrderedProbitModel,
    MultinomialLogitModel, VARModel, DIDResult, PoissonModel, NegBinModel,
    QuantileRegModel, RDDResult, SURModel, ThreeSLSModel, EventStudyLP, LPDiDResult,
    BaconDecomposition, MultinomialMarginalEffects)

for MT in _COEF_TABLE_TYPES
    @eval Tables.istable(::Type{<:$MT}) = true
    @eval Tables.columnaccess(::Type{<:$MT}) = true
    @eval Tables.columns(m::$MT) = _coef_nt(m)
    @eval Tables.schema(m::$MT) = Tables.schema(_coef_nt(m))
end

# ─────────────────────────────────────────────────────────────────────────────
# long_table — tidy/long views of array-valued results
# ─────────────────────────────────────────────────────────────────────────────

"""
    long_table(result) -> DataFrame

Return a tidy (long) table with one row per cell of an array-valued result — the
complement to the wide, per-(variable, shock) `table()` view. Every returned table
carries explicit index columns so downstream scripts are uniform across result types.

| Result type | Columns |
|---|---|
| `ImpulseResponse` / `BayesianImpulseResponse` | `horizon, variable, shock, value, lower, upper` |
| `FEVD` | `horizon, variable, shock, value` |
| `BayesianFEVD` | `horizon, variable, shock, value, lower, upper` |
| `LPImpulseResponse` | `horizon, variable, shock, value, se, lower, upper` |
| `LPFEVD` | `horizon, variable, shock, value, se, lower, upper` |
| `HistoricalDecomposition` | `time, variable, shock, value` |
| `BayesianHistoricalDecomposition` | `time, variable, shock, value, lower, upper` |
| `AbstractForecastResult` (VAR/BVAR/VECM/LP) | `horizon, variable, value, lower, upper` |
| `MidasForecast` | `horizon, variable, value, se, lower, upper` (horizon label is the direct `h`) |
| ordered `marginal_effects` NamedTuple | `variable, category, estimate, std_error, stat, p_value, ci_lower, ci_upper` |

Horizons are 1-based (matching [`table`](@ref)). `lower`/`upper` are `missing` when the
result carries no uncertainty bands (`ci_type == :none` / `ci_method == :none`). The
result is a `DataFrame`, so it round-trips directly to CSV via [`write_csv`](@ref) or to
any Tables.jl sink.

```julia
model = estimate_var(Y, 2)
irf = compute_irf(model, compute_Q(model, :cholesky; horizon=20), 20)
df = long_table(irf)     # (horizon, variable, shock, value, lower, upper)
```
"""
function long_table end

function long_table(irf::ImpulseResponse)
    H = size(irf.values, 1)
    nv = length(irf.variables)
    ns = length(irf.shocks)
    has_ci = irf.ci_type != :none
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, irf.variables[v]); push!(shock, irf.shocks[s])
        push!(value, irf.values[h, v, s])
        push!(lower, has_ci ? irf.ci_lower[h, v, s] : missing)
        push!(upper, has_ci ? irf.ci_upper[h, v, s] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

function long_table(irf::BayesianImpulseResponse)
    H = size(irf.point_estimate, 1)
    nv = length(irf.variables)
    ns = length(irf.shocks)
    nq = size(irf.quantiles, 4)
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, irf.variables[v]); push!(shock, irf.shocks[s])
        push!(value, irf.point_estimate[h, v, s])
        push!(lower, nq > 0 ? irf.quantiles[h, v, s, 1] : missing)
        push!(upper, nq > 0 ? irf.quantiles[h, v, s, nq] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

function long_table(f::FEVD)
    nv, ns, H = size(f.proportions)     # (variable, shock, horizon)
    horizon = Int[]; variable = String[]; shock = String[]; value = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, f.variables[v]); push!(shock, f.shocks[s])
        push!(value, f.proportions[v, s, h])
    end
    return DataFrame(; horizon, variable, shock, value)
end

function long_table(irf::LPImpulseResponse)
    H1, nresp = size(irf.values)         # (horizon 0..H stored in rows, response)
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; se = Float64[]; lower = Float64[]; upper = Float64[]
    for h in 1:H1, v in 1:nresp
        push!(horizon, h - 1); push!(variable, irf.response_vars[v]); push!(shock, irf.shock_var)
        push!(value, irf.values[h, v]); push!(se, irf.se[h, v])
        push!(lower, irf.ci_lower[h, v]); push!(upper, irf.ci_upper[h, v])
    end
    return DataFrame(; horizon, variable, shock, value, se, lower, upper)
end

function long_table(f::AbstractForecastResult)
    # Univariate forecasts (ARIMA/Volatility) store an h-vector; multivariate ones a (h, n)
    # matrix. Reshape to a common (h, n) so a single implementation covers both.
    _as_mat(x) = x isa AbstractVector ? reshape(x, :, 1) : x
    pf = _as_mat(point_forecast(f))
    H, nv = size(pf)
    names = (hasproperty(f, :varnames) && length(f.varnames) == nv) ?
        f.varnames : ["y$i" for i in 1:nv]
    has_ci = !((hasproperty(f, :ci_method) && getproperty(f, :ci_method) === :none) ||
               (hasproperty(f, :ci_type) && getproperty(f, :ci_type) === :none))
    lo = has_ci ? _as_mat(lower_bound(f)) : nothing
    up = has_ci ? _as_mat(upper_bound(f)) : nothing
    has_ci = has_ci && lo !== nothing && size(lo) == size(pf)
    horizon = Int[]; variable = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv
        push!(horizon, h); push!(variable, names[v]); push!(value, pf[h, v])
        push!(lower, has_ci ? lo[h, v] : missing)
        push!(upper, has_ci ? up[h, v] : missing)
    end
    return DataFrame(; horizon, variable, value, lower, upper)
end

# MidasForecast — direct h-step forecast: single-element vectors plus the TRUE direct
# horizon. The generic method would size H from the array (length 1) and mislabel the
# row horizon=1; label f.horizon instead and carry the prediction se. (#867)
function long_table(f::MidasForecast)
    n = length(f.forecast)
    horizon = fill(Int(f.horizon), n)
    variable = fill("y1", n)
    value = Float64.(collect(f.forecast))
    se = Float64.(collect(f.se))
    lower = Float64.(collect(f.ci_lower))
    upper = Float64.(collect(f.ci_upper))
    return DataFrame(; horizon, variable, value, se, lower, upper)
end

# HistoricalDecomposition — shock contributions in the same (time, variable, shock,
# value) long shape as FEVD. (#862)
function long_table(hd::HistoricalDecomposition)
    Teff, nv, ns = size(hd.contributions)     # (time, variable, shock)
    time = Int[]; variable = String[]; shock = String[]; value = Float64[]
    for t in 1:Teff, v in 1:nv, s in 1:ns
        push!(time, t); push!(variable, hd.variables[v]); push!(shock, hd.shock_names[s])
        push!(value, hd.contributions[t, v, s])
    end
    return DataFrame(; time, variable, shock, value)
end

function long_table(hd::BayesianHistoricalDecomposition)
    Teff, nv, ns = size(hd.point_estimate)    # (time, variable, shock)
    nq = size(hd.quantiles, 4)
    time = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for t in 1:Teff, v in 1:nv, s in 1:ns
        push!(time, t); push!(variable, hd.variables[v]); push!(shock, hd.shock_names[s])
        push!(value, hd.point_estimate[t, v, s])
        push!(lower, nq > 0 ? hd.quantiles[t, v, s, 1] : missing)
        push!(upper, nq > 0 ? hd.quantiles[t, v, s, nq] : missing)
    end
    return DataFrame(; time, variable, shock, value, lower, upper)
end

# BayesianFEVD — same (horizon, variable, shock) keys as FEVD plus the outer
# posterior-quantile interval. (#864)
function long_table(f::BayesianFEVD)
    nv, ns, H = size(f.point_estimate)        # (variable, shock, horizon)
    nq = size(f.quantiles, 4)
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, f.variables[v]); push!(shock, f.shocks[s])
        push!(value, f.point_estimate[v, s, h])
        push!(lower, nq > 0 ? f.quantiles[v, s, h, 1] : missing)
        push!(upper, nq > 0 ? f.quantiles[v, s, h, nq] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

# LPFEVD — same (horizon, variable, shock) keys as FEVD; value is the
# bias-corrected headline estimate with bootstrap se/CI. (#865)
function long_table(f::LPFEVD)
    nv, ns, H = size(f.bias_corrected)        # (variable, shock, horizon)
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; se = Float64[]; lower = Float64[]; upper = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, f.variables[v]); push!(shock, f.shocks[s])
        push!(value, f.bias_corrected[v, s, h]); push!(se, f.se[v, s, h])
        push!(lower, f.ci_lower[v, s, h]); push!(upper, f.ci_upper[v, s, h])
    end
    return DataFrame(; horizon, variable, shock, value, se, lower, upper)
end

# Ordered-model `marginal_effects` returns a bare NamedTuple (no display struct),
# so it cannot be a Tables.jl source (that would pirate Base.NamedTuple) — expose
# the same long (variable, category) shape via `long_table` instead. All J
# categories are kept: ordered AMEs have no base normalization. (#863)
function long_table(nt::NamedTuple{(:effects, :se, :varnames, :categories)})
    cols = _marginal_effects_long(nt.effects, nt.se, nt.varnames, nt.categories;
        skip_first=false)
    return DataFrame(; cols...)
end

# ─────────────────────────────────────────────────────────────────────────────
# write_csv — export any Tables-compatible result or long_table to CSV
# ─────────────────────────────────────────────────────────────────────────────

# Coerce a result to a Tables.jl-compatible object: coefficient models and DataFrames
# are already tables; array-valued results route through long_table.
_tabular(x) = x
_tabular(x::ImpulseResponse)         = long_table(x)
_tabular(x::BayesianImpulseResponse) = long_table(x)
_tabular(x::FEVD)                    = long_table(x)
_tabular(x::BayesianFEVD)            = long_table(x)
_tabular(x::LPImpulseResponse)       = long_table(x)
_tabular(x::LPFEVD)                  = long_table(x)
_tabular(x::HistoricalDecomposition) = long_table(x)
_tabular(x::BayesianHistoricalDecomposition) = long_table(x)
_tabular(x::AbstractForecastResult)  = long_table(x)

_csv_cell(::Missing) = ""
_csv_cell(x::Real) = string(x)
function _csv_cell(x)
    s = string(x)
    (occursin(',', s) || occursin('"', s) || occursin('\n', s)) ?
        '"' * replace(s, '"' => "\"\"") * '"' : s
end

"""
    write_csv(result, path) -> path

Write a result to a comma-separated file at `path`. Coefficient-bearing models
(`RegModel`, `LogitModel`, the PanelReg family, `MarginalEffects`, `DIDResult`, …) are
written as their coefficient table; array-valued results (`ImpulseResponse`, `FEVD`,
`LPImpulseResponse`, forecasts) are written as their [`long_table`](@ref). Any other
Tables.jl-compatible object (a `DataFrame`, a `NamedTuple` of vectors) is written as-is.

The header row is the column names; string cells containing a comma, quote, or newline
are quoted. Uses the stdlib `DelimitedFiles`/`Base` I/O — no CSV.jl dependency.

```julia
m = estimate_reg(y, X)
write_csv(m, "coefficients.csv")           # term,estimate,std_error,stat,p_value,ci_lower,ci_upper
write_csv(long_table(irf), "irf.csv")      # or pass the result directly: write_csv(irf, "irf.csv")
```
"""
function write_csv(result, path::AbstractString)
    tbl = Tables.columns(_tabular(result))
    colnames = collect(Tables.columnnames(tbl))
    vecs = [Tables.getcolumn(tbl, nm) for nm in colnames]
    nrow = isempty(vecs) ? 0 : length(vecs[1])
    open(path, "w") do io
        println(io, join(colnames, ","))
        for i in 1:nrow
            println(io, join((_csv_cell(v[i]) for v in vecs), ","))
        end
    end
    return path
end
