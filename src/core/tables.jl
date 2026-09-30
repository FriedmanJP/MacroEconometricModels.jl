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
#   1. Single-shape results implement the Tables.jl *source* interface: coefficient
#      tables for models (RegModel, Logit/Probit, Ordered/Multinomial, the
#      PanelReg family, VARModel, MarginalEffects, DIDResult, count models,
#      QuantileRegModel, RDDResult, SUR/3SLS, DiD event-study results,
#      MultinomialMarginalEffects), metric/weight tables for forecast evaluation
#      and combination, path/moment tables for policy counterfactuals, and
#      sector tables for input-output results — so `DataFrame(m)` returns the
#      same numbers `report(m)` prints.
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

# ── Forecast evaluation & combination (#857) ──────────────────────────────────

# One row per model: accuracy metrics plus the Theil decomposition shares.
function _coef_nt(ev::ForecastEvaluation)
    m = length(ev.models)
    cols = Any[collect(ev.models)]
    names = Symbol[:model]
    for (k, mt) in enumerate(ev.metrics)
        push!(names, Symbol(mt))
        push!(cols, Float64.(ev.values[:, k]))
    end
    append!(names, [:theil_bias, :theil_variance, :theil_covariance, :n])
    push!(cols, Float64.(ev.decomp[:, 1]))
    push!(cols, Float64.(ev.decomp[:, 2]))
    push!(cols, Float64.(ev.decomp[:, 3]))
    push!(cols, fill(ev.n, m))
    return NamedTuple{tuple(names...)}(tuple(cols...))
end

# One row per combined model: weight and individual MSE, method repeated.
function _coef_nt(c::ForecastCombination)
    n = length(c.models)
    return (model=collect(c.models), weight=Float64.(collect(c.weights)),
        mse=Float64.(collect(c.mse)), method=fill(string(c.method), n))
end

# ── Policy counterfactuals (#858) ─────────────────────────────────────────────

# Long (period, variable) path rows for outcome/instrument paths with baselines,
# counterfactuals, and first/last draw bands.
function _cf_push_paths!(period, variable, role, baseline, counterfactual, lower, upper,
        names, rolename, base, cf, bands)
    for (i, nm) in enumerate(names)
        b = base[i]
        cc = cf[i]
        Bd = bands === nothing ? nothing : bands[i]
        nq = Bd === nothing ? 0 : size(Bd, 2)
        for h in eachindex(b)
            push!(period, h); push!(variable, string(nm)); push!(role, rolename)
            push!(baseline, Float64(b[h])); push!(counterfactual, Float64(cc[h]))
            push!(lower, nq > 0 ? Float64(Bd[h, 1]) : missing)
            push!(upper, nq > 0 ? Float64(Bd[h, nq]) : missing)
        end
    end
end

function _coef_nt(pc::PolicyCounterfactual)
    period = Int[]; variable = String[]; role = String[]
    baseline = Float64[]; counterfactual = Float64[]
    lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    _cf_push_paths!(period, variable, role, baseline, counterfactual, lower, upper,
        pc.outcomes, "outcome", pc.x_base, pc.x_cf, pc.x_bands)
    _cf_push_paths!(period, variable, role, baseline, counterfactual, lower, upper,
        pc.instruments, "instrument", pc.z_base, pc.z_cf, pc.z_bands)
    return (period=period, variable=variable, role=role, baseline=baseline,
        counterfactual=counterfactual, lower=lower, upper=upper)
end

# Second moments in pair grain: covariances and correlations, baseline vs cf.
function _coef_nt(cm::CounterfactualMoments)
    n = length(cm.varnames)
    vi = String[]; vj = String[]
    cb = Float64[]; cc = Float64[]; rb = Float64[]; rc = Float64[]
    for i in 1:n, j in 1:n
        push!(vi, string(cm.varnames[i])); push!(vj, string(cm.varnames[j]))
        push!(cb, Float64(cm.Sigma_base[i, j])); push!(cc, Float64(cm.Sigma_cf[i, j]))
        push!(rb, Float64(cm.corr_base[i, j])); push!(rc, Float64(cm.corr_cf[i, j]))
    end
    return (variable_i=vi, variable_j=vj, cov_base=cb, cov_cf=cc,
        corr_base=rb, corr_cf=rc)
end

# Realized vs counterfactual panel in (date, variable) long shape; the per-date
# implementation residual repeats across variables.
function _coef_nt(ch::CounterfactualHistory)
    nd, nv = size(ch.realized)
    date = String[]; variable = String[]
    realized = Float64[]; counterfactual = Float64[]
    lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    relres = Float64[]
    nq = ch.cf_bands === nothing ? 0 : size(ch.cf_bands, 3)
    for d in 1:nd, v in 1:nv
        push!(date, ch.dates[d]); push!(variable, string(ch.varnames[v]))
        push!(realized, Float64(ch.realized[d, v]))
        push!(counterfactual, Float64(ch.cf[d, v]))
        push!(lower, nq > 0 ? Float64(ch.cf_bands[d, v, 1]) : missing)
        push!(upper, nq > 0 ? Float64(ch.cf_bands[d, v, nq]) : missing)
        push!(relres, Float64(ch.rel_residual[d]))
    end
    return (date=date, variable=variable, realized=realized,
        counterfactual=counterfactual, cf_lower=lower, cf_upper=upper,
        rel_residual=relres)
end

function _coef_nt(bp::BaselinePath)
    period = Int[]; variable = String[]; role = String[]; value = Float64[]
    for (i, nm) in enumerate(bp.outcomes), h in eachindex(bp.x[i])
        push!(period, h); push!(variable, string(nm)); push!(role, "outcome")
        push!(value, Float64(bp.x[i][h]))
    end
    for (k, nm) in enumerate(bp.instruments), h in eachindex(bp.z[k])
        push!(period, h); push!(variable, string(nm)); push!(role, "instrument")
        push!(value, Float64(bp.z[k][h]))
    end
    return (period=period, variable=variable, role=role, value=value)
end

function _coef_nt(pf::PolicyForecast)
    period = Int[]; variable = String[]; value = Float64[]
    for (i, nm) in enumerate(pf.outcomes), h in eachindex(pf.values[i])
        push!(period, h); push!(variable, string(nm))
        push!(value, Float64(pf.values[i][h]))
    end
    return (period=period, variable=variable, value=value)
end

# OPP at every decision date with the news/preference/aging revision split.
function _coef_nt(sq::OPPSequence)
    ns, nd = size(sq.delta)
    date = String[]; shock = String[]
    delta = Float64[]; delta_tc = Float64[]
    news = Float64[]; pref = Float64[]; aging = Float64[]
    for t in 1:nd, s in 1:ns
        push!(date, sq.dates[t]); push!(shock, sq.shock_labels[s])
        push!(delta, Float64(sq.delta[s, t])); push!(delta_tc, Float64(sq.delta_tc[s, t]))
        push!(news, Float64(sq.news_part[s, t])); push!(pref, Float64(sq.pref_part[s, t]))
        push!(aging, Float64(sq.aging_part[s, t]))
    end
    return (date=date, shock=shock, delta=delta, delta_tc=delta_tc,
        news=news, pref=pref, aging=aging)
end

function _coef_nt(fs::ForecastSufficiency)
    H, no = size(fs.fev_ratio)
    horizon = Int[]; observable = String[]
    fev_ratio = Float64[]; one_step_ratio = Float64[]
    for h in 1:H, o in 1:no
        push!(horizon, h); push!(observable, string(fs.observables[o]))
        push!(fev_ratio, Float64(fs.fev_ratio[h, o]))
        push!(one_step_ratio, Float64(fs.one_step_ratio[o]))
    end
    return (horizon=horizon, observable=observable, fev_ratio=fev_ratio,
        one_step_ratio=one_step_ratio)
end

# ── Input-output (#859) ───────────────────────────────────────────────────────

function _coef_nt(m::LeontiefModel)
    n = length(m.x)
    si = String[]; sj = String[]; av = Float64[]; lv = Float64[]
    for i in 1:n, j in 1:n
        push!(si, string(m.io.sectors[i])); push!(sj, string(m.io.sectors[j]))
        push!(av, Float64(m.A[i, j])); push!(lv, Float64(m.L[i, j]))
    end
    return (sector_i=si, sector_j=sj, A=av, L=lv)
end

function _coef_nt(m::GhoshModel)
    n = length(m.x)
    si = String[]; sj = String[]; bv = Float64[]; gv = Float64[]
    for i in 1:n, j in 1:n
        push!(si, string(m.io.sectors[i])); push!(sj, string(m.io.sectors[j]))
        push!(bv, Float64(m.B[i, j])); push!(gv, Float64(m.G[i, j]))
    end
    return (sector_i=si, sector_j=sj, B=bv, G=gv)
end

function _coef_nt(r::LinkageResult)
    return (sector=collect(r.sectors), backward=Float64.(collect(r.backward)),
        forward=Float64.(collect(r.forward)), Ui=Float64.(collect(r.Ui)),
        Uj=Float64.(collect(r.Uj)), classification=string.(r.classification))
end

function _coef_nt(m::IOMultipliers)
    n = length(m.sectors)
    return (sector=collect(m.sectors), value=Float64.(collect(m.values)),
        kind=fill(string(m.kind), n), type=fill(string(m.type), n))
end

# Per-stressor, per-sector contributions; `total` repeats the show() headline
# (consumption total summed over final-demand categories).
function _coef_nt(fp::FootprintResult)
    ns, nsec = size(fp.by_sector)
    totals = [sum(@view fp.total[i, :]) for i in 1:ns]
    stressor = String[]; sector = Int[]; value = Float64[]; total = Float64[]
    for i in 1:ns, j in 1:nsec
        push!(stressor, fp.stressors[i]); push!(sector, j)
        push!(value, Float64(fp.by_sector[i, j])); push!(total, Float64(totals[i]))
    end
    return (stressor=stressor, sector=sector, value=value, total=total)
end

# One row per (factor, index); factor order mirrors `show()` (r.factors first,
# then extra keys sorted). `total`/`residual` repeat across factors.
function _coef_nt(r::SDAResult)
    fkeys = [k for k in r.factors if haskey(r.effects, k)]
    append!(fkeys, sort!(setdiff!(collect(keys(r.effects)), r.factors)))
    factor = String[]; index = Int[]
    effect = Float64[]; total = Float64[]; residual = Float64[]
    for k in fkeys
        v = r.effects[k]
        for i in eachindex(v)
            push!(factor, string(k)); push!(index, i)
            push!(effect, Float64(v[i]))
            push!(total, Float64(r.total[i])); push!(residual, Float64(r.residual[i]))
        end
    end
    m = length(factor)
    return (factor=factor, index=index, effect=effect, total=total,
        residual=residual, method=fill(string(r.method), m), on=fill(string(r.on), m))
end

function _coef_nt(fp::RegionalFootprintResult)
    ns, ng = size(fp.production)
    stressor = String[]; region = String[]
    production = Float64[]; consumption = Float64[]
    for i in 1:ns, r in 1:ng
        push!(stressor, fp.stressors[i]); push!(region, fp.regions[r])
        push!(production, Float64(fp.production[i, r]))
        push!(consumption, Float64(fp.consumption[i, r]))
    end
    return (stressor=stressor, region=region, production=production,
        consumption=consumption)
end

# ── Test battery (#860) ───────────────────────────────────────────────────────
# Uniform one-row-per-test (or per-hypothesis) tables: test label, statistic,
# p-value, 5% decision mirroring show() (critical-value comparison with the
# show() tail when the type carries CVs, else p < 0.05), and 1/5/10% CVs.
# Test-specific findings (breaks, ranks, estimates) ride in extra columns;
# spec metadata (lags, df, nobs, kernels) stays in fields.

_new_test_cols() = (test=String[], statistic=Float64[],
    p_value=Union{Missing,Float64}[], decision=Union{Missing,String}[],
    cv_1pct=Union{Missing,Float64}[], cv_5pct=Union{Missing,Float64}[],
    cv_10pct=Union{Missing,Float64}[])

_test_p(p) = (p === nothing || p === missing || !isfinite(Float64(p))) ? missing : Float64(p)

# Push one hypothesis row. `tail` is :left / :right for CV-based decisions
# (mirroring show()) or :p to decide by p-value even when CVs are printed.
function _test_push!(c, label, stat, p, cv, tail::Symbol)
    push!(c.test, label)
    push!(c.statistic, Float64(stat))
    pv = _test_p(p)
    push!(c.p_value, pv)
    hascv = cv !== nothing && !isempty(cv)
    if hascv
        push!(c.cv_1pct, Float64(cv[1]))
        push!(c.cv_5pct, Float64(cv[5]))
        push!(c.cv_10pct, Float64(cv[10]))
    else
        push!(c.cv_1pct, missing); push!(c.cv_5pct, missing); push!(c.cv_10pct, missing)
    end
    dec = if hascv && tail !== :p
        hit = tail === :left ? Float64(stat) < Float64(cv[5]) :
                               Float64(stat) > Float64(cv[5])
        hit ? "reject" : "fail to reject"
    elseif pv !== missing
        pv < 0.05 ? "reject" : "fail to reject"
    else
        missing
    end
    push!(c.decision, dec)
    return c
end

# --- Single-hypothesis tests: unit roots, breaks, panel, serial, others ---

_coef_nt(r::ADFResult) = _test_push!(_new_test_cols(), "ADF", r.statistic, r.pvalue, r.critical_values, :left)
_coef_nt(r::KPSSResult) = _test_push!(_new_test_cols(), "KPSS", r.statistic, r.pvalue, r.critical_values, :right)
_coef_nt(r::PPResult) = _test_push!(_new_test_cols(), "Phillips-Perron", r.statistic, r.pvalue, r.critical_values, :left)
_coef_nt(r::ERSResult) = _test_push!(_new_test_cols(), "ERS point-optimal", r.P_T, r.pvalue, r.critical_values, :left)

function _coef_nt(r::ZAResult)
    c = _test_push!(_new_test_cols(), "Zivot-Andrews", r.statistic, r.pvalue, r.critical_values, :left)
    return merge(c, (break_index=[r.break_index], break_fraction=[Float64(r.break_fraction)]))
end

function _coef_nt(r::AndrewsResult)
    c = _test_push!(_new_test_cols(), "Andrews $(r.test_type)", r.statistic, r.pvalue, r.critical_values, :p)
    return merge(c, (break_index=[r.break_index], break_fraction=[Float64(r.break_fraction)]))
end

function _coef_nt(r::ADF2BreakResult)
    c = _test_push!(_new_test_cols(), "ADF two-break", r.statistic, r.pvalue, r.critical_values, :left)
    return merge(c, (break_1=[r.break1], break_2=[r.break2],
        breakfrac_1=[Float64(r.break1_fraction)], breakfrac_2=[Float64(r.break2_fraction)]))
end

function _coef_nt(r::LMUnitRootResult)
    c = _test_push!(_new_test_cols(), "LM unit root", r.statistic, r.pvalue, r.critical_values, :left)
    gd(i) = length(r.break_dates) >= i ? r.break_dates[i] : missing
    gf(i) = length(r.break_fractions) >= i ? Float64(r.break_fractions[i]) : missing
    return merge(c, (break_1=[gd(1)], break_2=[gd(2)], breakfrac_1=[gf(1)], breakfrac_2=[gf(2)]))
end

function _coef_nt(r::FactorBreakResult)
    c = _test_push!(_new_test_cols(), "Factor break $(r.method)", r.statistic, r.pvalue, nothing, :p)
    return merge(c, (break_index=[r.break_date === nothing ? missing : r.break_date],))
end

_coef_nt(r::LLCResult) = _test_push!(_new_test_cols(), "Levin-Lin-Chu", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left)
_coef_nt(r::IPSResult) = _test_push!(_new_test_cols(), "Im-Pesaran-Shin", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left)
_coef_nt(r::BreitungPanelResult) = _test_push!(_new_test_cols(), "Breitung", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left)
_coef_nt(r::HadriResult) = _test_push!(_new_test_cols(), "Hadri", r.statistic, r.pvalue, _NORMAL_RIGHT_CV, :right)
_coef_nt(r::PesaranCIPSResult) = _test_push!(_new_test_cols(), "Pesaran CIPS", r.cips_statistic, r.pvalue, r.critical_values, :left)

_coef_nt(r::FisherTestResult) = merge(
    _test_push!(_new_test_cols(), "Fisher periodicity", r.statistic, r.pvalue, nothing, :p),
    (peak_freq=[Float64(r.peak_freq)],))
_coef_nt(r::BartlettWhiteNoiseResult) = _test_push!(_new_test_cols(), "Bartlett white noise", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::LjungBoxResult) = _test_push!(_new_test_cols(), "Ljung-Box", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::BoxPierceResult) = _test_push!(_new_test_cols(), "Box-Pierce", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::DurbinWatsonResult) = _test_push!(_new_test_cols(), "Durbin-Watson", r.statistic, r.pvalue, nothing, :p)

function _coef_nt(r::GrangerCausalityResult)
    c = _test_push!(_new_test_cols(), "Granger causality", r.statistic, r.pvalue, nothing, :p)
    return merge(c, (cause=[join(string.(r.cause), ",")], effect=[r.effect]))
end

_coef_nt(r::LRTestResult) = _test_push!(_new_test_cols(), "Likelihood ratio", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::LMTestResult) = _test_push!(_new_test_cols(), "Lagrange multiplier", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::EngleGrangerResult) = _test_push!(_new_test_cols(), "Engle-Granger", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::HansenInstabilityResult) = _test_push!(_new_test_cols(), "Hansen instability", r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::ParkAddedResult) = _test_push!(_new_test_cols(), "Park added variables", r.statistic, r.pvalue, nothing, :p)

_coef_nt(r::BubbleResult) = _test_push!(_new_test_cols(), r.kind == :sadf ? "SADF" : "GSADF",
    r.statistic, r.pvalue, r.critical_values, :right)

function _coef_nt(r::EDFTestResult)
    label = get(_EDF_TEST_LABEL, r.test, string(r.test)) * " (" *
            get(_EDF_DIST_LABEL, r.dist, string(r.dist)) * ")"
    c = _test_push!(_new_test_cols(), label, r.statistic, r.pvalue, r.critical_values, :p)
    return merge(c, (raw_statistic=[Float64(r.raw_statistic)],))
end

_coef_nt(r::EqualityTestResult) = _test_push!(_new_test_cols(),
    get(_EQ_TEST_LABELS, r.test_name, string(r.test_name)), r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::PanelTestResult) = _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::PVARTestResult) = _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
_coef_nt(r::NormalityTestResult) = _test_push!(_new_test_cols(), _normality_test_label(r), r.statistic, r.pvalue, nothing, :p)

function _coef_nt(r::CorTestResult)
    method = r.method === :pearson ? "Pearson" : r.method === :spearman ? "Spearman" : "Kendall"
    c = _test_push!(_new_test_cols(), "Correlation ($method)", r.statistic, r.pvalue, nothing, :p)
    hasci = r.method === :pearson && isfinite(r.ci_lower) && isfinite(r.ci_upper)
    return merge(c, (estimate=[Float64(r.estimate)],
        ci_lower=[hasci ? Float64(r.ci_lower) : missing],
        ci_upper=[hasci ? Float64(r.ci_upper) : missing]))
end

function _coef_nt(r::RegDiagnosticResult)
    c = _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
    return merge(c, (f_stat=[r.f_stat === nothing ? missing : Float64(r.f_stat)],
        f_pvalue=[r.f_pvalue === nothing ? missing : Float64(r.f_pvalue)],
        aux_r2=[Float64(r.aux_r2)]))
end

function _coef_nt(r::DMTestResult)
    c = _test_push!(_new_test_cols(), "Diebold-Mariano", r.statistic, r.pvalue, nothing, :p)
    return merge(c, (dbar=[Float64(r.dbar)], lrvar=[Float64(r.lrvar)]))
end

function _coef_nt(r::ClarkWestResult)
    c = _test_push!(_new_test_cols(), "Clark-West", r.statistic, r.pvalue, nothing, :p)
    return merge(c, (fbar=[Float64(r.fbar)], lrvar=[Float64(r.lrvar)]))
end

function _coef_nt(r::ForecastEncompassingResult)
    c = _test_push!(_new_test_cols(), "Forecast encompassing", r.tstat, r.pvalue, nothing, :p)
    return merge(c, (b1=[Float64(r.b1)], b2=[Float64(r.b2)]))
end

# --- Multi-hypothesis tests: named statistics, trace/max pairs, paths ---

function _test_named_rows!(c, prefix, names, stats, pvals)
    for s in eachindex(names, stats, pvals)
        _test_push!(c, "$prefix $(names[s])", stats[s], pvals[s], nothing, :p)
    end
    return c
end

function _coef_nt(r::KaoResult)
    return _test_named_rows!(_new_test_cols(), "Kao", r.names, r.statistics, r.pvalues)
end

function _coef_nt(r::PedroniResult)
    c = _test_named_rows!(_new_test_cols(), "Pedroni", r.names, r.statistics, r.pvalues)
    return merge(c, (raw=Float64.(collect(r.raw)),))
end

function _coef_nt(r::WesterlundResult)
    c = _test_named_rows!(_new_test_cols(), "Westerlund", r.names, r.statistics, r.pvalues)
    hasb = r.bootstrap > 0 && length(r.bootstrap_pvalues) == length(r.names)
    bp = Union{Missing,Float64}[hasb ? Float64(r.bootstrap_pvalues[s]) : missing
        for s in eachindex(r.names)]
    return merge(c, (bootstrap_p=bp,))
end

function _coef_nt(r::FisherPanelResult)
    c = _new_test_cols()
    _test_push!(c, "P (Maddala-Wu)", r.P, r.P_pvalue, nothing, :p)
    _test_push!(c, "Z (Choi)", r.Z, r.Z_pvalue, nothing, :p)
    _test_push!(c, "L* (Choi)", r.Lstar, r.Lstar_pvalue, nothing, :p)
    _test_push!(c, "Pm (Choi)", r.Pm, r.Pm_pvalue, nothing, :p)
    return c
end

function _coef_nt(r::MoonPerronResult)
    c = _new_test_cols()
    _test_push!(c, "Moon-Perron t*_a", r.t_a_statistic, r.pvalue_a, nothing, :p)
    _test_push!(c, "Moon-Perron t*_b", r.t_b_statistic, r.pvalue_b, nothing, :p)
    return c
end

function _coef_nt(r::PhillipsOuliarisResult)
    c = _new_test_cols()
    _test_push!(c, "Phillips-Ouliaris Zt", r.statistic, r.pvalue, nothing, :p)
    _test_push!(c, "Phillips-Ouliaris Za", r.z_alpha, r.z_alpha_pvalue, nothing, :p)
    return c
end

function _coef_nt(r::DumitrescuHurlinResult)
    c = _new_test_cols()
    _test_push!(c, "Dumitrescu-Hurlin Zbar", r.Zbar, r.Zbar_pvalue, nothing, :p)
    _test_push!(c, "Dumitrescu-Hurlin Ztilde", r.Ztilde, r.Ztilde_pvalue, nothing, :p)
    bp = (r.bootstrap > 0 && isfinite(r.bootstrap_pvalue)) ? Float64(r.bootstrap_pvalue) : missing
    return merge(c, (Wbar=[Float64(r.Wbar), Float64(r.Wbar)], boot_p=[bp, missing],
        cause=[string(r.cause), string(r.cause)], effect=[string(r.effect), string(r.effect)]))
end

function _coef_nt(r::MincerZarnowitzResult)
    c = _new_test_cols()
    _test_push!(c, "Mincer-Zarnowitz Wald", r.wald, r.pvalue_wald, nothing, :p)
    _test_push!(c, "Mincer-Zarnowitz F", r.fstat, r.pvalue_f, nothing, :p)
    return merge(c, (a=[Float64(r.a), Float64(r.a)], b=[Float64(r.b), Float64(r.b)],
        se_a=[Float64(r.se[1]), Float64(r.se[1])], se_b=[Float64(r.se[2]), Float64(r.se[2])]))
end

function _coef_nt(r::FourierADFResult)
    c = _new_test_cols()
    _test_push!(c, "Fourier ADF", r.statistic, r.pvalue, r.critical_values, :left)
    _test_push!(c, "Fourier ADF F", r.f_statistic, r.f_pvalue, r.f_critical_values, :right)
    return merge(c, (frequency=[r.frequency, r.frequency],))
end

function _coef_nt(r::FourierKPSSResult)
    c = _new_test_cols()
    _test_push!(c, "Fourier KPSS", r.statistic, r.pvalue, r.critical_values, :right)
    _test_push!(c, "Fourier KPSS F", r.f_statistic, r.f_pvalue, r.f_critical_values, :right)
    return merge(c, (frequency=[r.frequency, r.frequency],))
end

function _coef_nt(r::GregoryHansenResult)
    c = _new_test_cols()
    brk = Int[]
    _test_push!(c, "Gregory-Hansen ADF*", r.adf_statistic, r.adf_pvalue, r.adf_critical_values, :left)
    push!(brk, r.adf_break)
    _test_push!(c, "Gregory-Hansen Zt*", r.zt_statistic, r.zt_pvalue, r.adf_critical_values, :left)
    push!(brk, r.zt_break)
    _test_push!(c, "Gregory-Hansen Za*", r.za_statistic, r.za_pvalue, r.za_critical_values, :left)
    push!(brk, r.za_break)
    return merge(c, (break_index=brk,))
end

function _coef_nt(r::NgPerronResult)
    c = _new_test_cols()
    for (nm, st) in (("MZa", r.MZa), ("MZt", r.MZt), ("MSB", r.MSB), ("MPT", r.MPT))
        _test_push!(c, "Ng-Perron $nm", st, missing, r.critical_values[Symbol(nm)], :left)
    end
    return c
end

function _coef_nt(r::DFGLSResult)
    c = _new_test_cols()
    _test_push!(c, "DF-GLS τ", r.statistic, r.pvalue, r.critical_values, :left)
    _test_push!(c, "ERS Pt", r.pt_statistic, r.pt_pvalue, r.pt_critical_values, :left)
    for (nm, st) in (("MZa", r.MZa), ("MZt", r.MZt), ("MSB", r.MSB), ("MPT", r.MPT))
        _test_push!(c, "DF-GLS $nm", st, missing, r.mgls_critical_values[Symbol(nm)], :left)
    end
    return c
end

function _coef_nt(r::JohansenResult)
    c = _new_test_cols()
    rk = Int[]; kd = String[]
    for i in eachindex(r.trace_stats)
        rank = i - 1
        cvt = Dict(1 => r.critical_values_trace[i, 3], 5 => r.critical_values_trace[i, 2],
            10 => r.critical_values_trace[i, 1])
        _test_push!(c, "Johansen trace (rank ≤ $rank)", r.trace_stats[i], r.trace_pvalues[i], cvt, :right)
        push!(rk, rank); push!(kd, "trace")
        cvm = Dict(1 => r.critical_values_max[i, 3], 5 => r.critical_values_max[i, 2],
            10 => r.critical_values_max[i, 1])
        _test_push!(c, "Johansen max (rank = $rank)", r.max_eigen_stats[i], r.max_eigen_pvalues[i], cvm, :right)
        push!(rk, rank); push!(kd, "max")
    end
    return merge(c, (rank=rk, kind=kd))
end

function _coef_nt(r::FisherJohansenResult)
    c = _new_test_cols()
    rk = Int[]; kd = String[]
    for j in eachindex(r.ranks)
        _test_push!(c, "Fisher-Johansen trace (rank ≤ $(r.ranks[j]))",
            r.trace_statistics[j], r.trace_pvalues[j], nothing, :p)
        push!(rk, r.ranks[j]); push!(kd, "trace")
        _test_push!(c, "Fisher-Johansen max (rank = $(r.ranks[j]))",
            r.max_statistics[j], r.max_pvalues[j], nothing, :p)
        push!(rk, r.ranks[j]); push!(kd, "max")
    end
    return merge(c, (rank=rk, kind=kd))
end

function _coef_nt(r::PANICResult)
    c = _new_test_cols()
    for j in 1:r.n_factors
        _test_push!(c, "PANIC factor $j", r.factor_adf_stats[j], r.factor_adf_pvalues[j], nothing, :p)
    end
    _test_push!(c, "PANIC pooled", r.pooled_statistic, r.pooled_pvalue, nothing, :p)
    return c
end

function _coef_nt(r::HEGYResult)
    c = _new_test_cols()
    _test_push!(c, "HEGY t(0)", r.t_zero, missing, r.t_zero_cv, :left)
    _test_push!(c, "HEGY t(π)", r.t_nyquist, missing, r.t_nyquist_cv, :left)
    for (i, F) in enumerate(r.pair_F)
        _test_push!(c, "HEGY F(ω=$(round(r.pair_freqs[i], digits=3)))", F, missing, r.pair_F_cv, :right)
    end
    _test_push!(c, "HEGY F seasonal", r.F_seasonal, missing, nothing, :p)
    _test_push!(c, "HEGY F all", r.F_all, missing, nothing, :p)
    return c
end

function _coef_nt(r::VarianceRatioResult)
    c = _new_test_cols()
    qv = Union{Missing,Int}[]; vrv = Union{Missing,Float64}[]; boot = Union{Missing,Float64}[]
    nq = length(r.q)
    for i in 1:nq
        _test_push!(c, "VR Z(q=$(r.q[i]))", r.z[i], r.z_pvalue[i], nothing, :p)
        push!(qv, r.q[i]); push!(vrv, Float64(r.vr[i])); push!(boot, missing)
        zb = (r.bootstrap > 0 && length(r.z_star_boot_pvalue) >= i &&
              isfinite(r.z_star_boot_pvalue[i])) ? Float64(r.z_star_boot_pvalue[i]) : missing
        _test_push!(c, "VR Z*(q=$(r.q[i]))", r.z_star[i], r.z_star_pvalue[i], nothing, :p)
        push!(qv, r.q[i]); push!(vrv, Float64(r.vr[i])); push!(boot, zb)
    end
    if r.wright
        for i in 1:nq
            for (nm, st, pv) in (("R1", r.R1, r.R1_pvalue), ("R2", r.R2, r.R2_pvalue),
                    ("S1", r.S1, r.S1_pvalue))
                _test_push!(c, "Wright $nm(q=$(r.q[i]))", st[i], pv[i], nothing, :p)
                push!(qv, r.q[i]); push!(vrv, missing); push!(boot, missing)
            end
        end
    end
    cdb = isfinite(r.cd_boot_pvalue) ? Float64(r.cd_boot_pvalue) : missing
    _test_push!(c, "Chow-Denning CD", r.cd_stat, r.cd_pvalue, nothing, :p)
    push!(qv, missing); push!(vrv, missing); push!(boot, cdb)
    _test_push!(c, "Chow-Denning CD*", r.cd_star_stat, r.cd_star_pvalue, nothing, :p)
    push!(qv, missing); push!(vrv, missing); push!(boot, cdb)
    return merge(c, (q=qv, vr=vrv, boot_p=boot))
end

function _coef_nt(r::BDSResult)
    c = _new_test_cols()
    mv = Int[]; ev = Float64[]; cvv = Float64[]; boot = Union{Missing,Float64}[]
    for (im, m) in enumerate(r.m), (je, eps) in enumerate(r.eps)
        _test_push!(c, "BDS", r.statistic[im, je], r.pvalue[im, je], nothing, :p)
        push!(mv, m); push!(ev, Float64(eps)); push!(cvv, Float64(r.C[im, je]))
        bp = r.boot_pvalue[im, je]
        push!(boot, isfinite(bp) ? Float64(bp) : missing)
    end
    return merge(c, (m=mv, eps=ev, C_m=cvv, boot_p=boot))
end

function _coef_nt(r::BaiPerronResult)
    c = _new_test_cols()
    for l in eachindex(r.supf_stats)
        _test_push!(c, "Bai-Perron sup-F($l)", r.supf_stats[l], r.supf_pvalues[l], nothing, :p)
    end
    for i in eachindex(r.sequential_stats)
        _test_push!(c, "Bai-Perron seq($(i+1)|$i)", r.sequential_stats[i], r.sequential_pvalues[i], nothing, :p)
    end
    return c
end

# --- Battery suites (stacked member rows) + stability paths ---

function _stack_test_rows!(c, nt)
    append!(c.test, nt.test); append!(c.statistic, nt.statistic)
    append!(c.p_value, nt.p_value); append!(c.decision, nt.decision)
    append!(c.cv_1pct, nt.cv_1pct); append!(c.cv_5pct, nt.cv_5pct)
    append!(c.cv_10pct, nt.cv_10pct)
    return c
end

function _coef_nt(s::NormalityTestSuite)
    c = _new_test_cols()
    for r in s.results
        _stack_test_rows!(c, _coef_nt(r))
    end
    return c
end

function _coef_nt(s::PanelUnitRootSummary)
    c = _new_test_cols()
    for m in (s.panic, s.cips, s.moon_perron, s.llc, s.ips, s.breitung, s.fisher, s.hadri)
        m === nothing && continue
        _stack_test_rows!(c, _coef_nt(m))
    end
    return c
end

# CUSUM path in (period) long shape with per-point band breaches.
function _coef_nt(r::StabilityResult)
    n = length(r.tindex)
    stat = Float64.(collect(r.stat_path))
    upper = Float64.(collect(r.upper))
    lower = Float64.(collect(r.lower))
    breached = Bool[!isnan(s) && (s < lo || s > up) for (s, lo, up) in zip(stat, lower, upper)]
    return (period=collect(r.tindex), kind=fill(string(r.kind), n), stat=stat,
        upper=upper, lower=lower, breached=breached)
end

# Companion eigenvalues in (eigen) long shape.
function _coef_nt(r::VARStationarityResult)
    n = length(r.eigenvalues)
    return (test=fill("VAR stationarity", n), eigen_index=collect(1:n),
        re=Float64[real(e) for e in r.eigenvalues],
        im=Float64[imag(e) for e in r.eigenvalues],
        modulus=Float64[abs(e) for e in r.eigenvalues],
        is_stationary=fill(r.is_stationary, n))
end

# ─────────────────────────────────────────────────────────────────────────────
# Tables.jl source interface for coefficient-bearing types
# ─────────────────────────────────────────────────────────────────────────────

const _COEF_TABLE_TYPES = (RegModel, LogitModel, ProbitModel, PanelRegModel, PanelIVModel,
    PanelLogitModel, PanelProbitModel, MarginalEffects, OrderedLogitModel, OrderedProbitModel,
    MultinomialLogitModel, VARModel, DIDResult, PoissonModel, NegBinModel,
    QuantileRegModel, RDDResult, SURModel, ThreeSLSModel, EventStudyLP, LPDiDResult,
    BaconDecomposition, MultinomialMarginalEffects, ForecastEvaluation, ForecastCombination,
    PolicyCounterfactual, CounterfactualMoments, CounterfactualHistory, BaselinePath,
    PolicyForecast, OPPSequence, ForecastSufficiency, LeontiefModel, GhoshModel,
    LinkageResult, IOMultipliers, FootprintResult, SDAResult, RegionalFootprintResult)

for MT in _COEF_TABLE_TYPES
    @eval Tables.istable(::Type{<:$MT}) = true
    @eval Tables.columnaccess(::Type{<:$MT}) = true
    @eval Tables.columns(m::$MT) = _coef_nt(m)
    @eval Tables.schema(m::$MT) = Tables.schema(_coef_nt(m))
end

# Test-battery result types share the same Tables.jl wiring; their `_coef_nt`
# builders emit the uniform one-row-per-hypothesis shape (#860).
const _TEST_TABLE_TYPES = (ADFResult, KPSSResult, PPResult, ERSResult,
    ZAResult, AndrewsResult, ADF2BreakResult, LMUnitRootResult, FactorBreakResult,
    LLCResult, IPSResult, BreitungPanelResult, HadriResult, PesaranCIPSResult,
    FisherTestResult, BartlettWhiteNoiseResult, LjungBoxResult, BoxPierceResult,
    DurbinWatsonResult, GrangerCausalityResult, LRTestResult, LMTestResult,
    EngleGrangerResult, HansenInstabilityResult, ParkAddedResult, BubbleResult,
    EDFTestResult, EqualityTestResult, PanelTestResult, PVARTestResult,
    NormalityTestResult, CorTestResult, RegDiagnosticResult, DMTestResult,
    ClarkWestResult, ForecastEncompassingResult, KaoResult, PedroniResult,
    WesterlundResult, FisherPanelResult, MoonPerronResult, PhillipsOuliarisResult,
    DumitrescuHurlinResult, MincerZarnowitzResult, FourierADFResult,
    FourierKPSSResult, GregoryHansenResult, NgPerronResult, DFGLSResult,
    JohansenResult, FisherJohansenResult, PANICResult, HEGYResult,
    VarianceRatioResult, BDSResult, BaiPerronResult, NormalityTestSuite,
    PanelUnitRootSummary, StabilityResult, VARStationarityResult)

for MT in _TEST_TABLE_TYPES
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
