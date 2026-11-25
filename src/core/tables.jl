# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

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
    s = Float64.(collect(se))
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
    return (
        term=string.(collect(names)),
        estimate=est,
        std_error=s,
        stat=stat,
        p_value=pval,
        ci_lower=est .- zc .* s,
        ci_upper=est .+ zc .* s,
    )
end

# Single-block StatsAPI models — reuse coef/stderror/dof_residual + varnames.
function _coef_nt(m::RegModel)
    return _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
end
function _coef_nt(m::PanelRegModel)
    return _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
end
function _coef_nt(m::PanelIVModel)
    return _coef_nt_simple(m.varnames, coef(m), stderror(m), :t, dof_residual(m))
end
_coef_nt(m::LogitModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::ProbitModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::PanelLogitModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
_coef_nt(m::PanelProbitModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)

# MarginalEffects — every column is precomputed; drop the non-finite (intercept) rows
# exactly as `report()` does.
function _coef_nt(me::MarginalEffects)
    keep = findall(isfinite, me.effects)
    return (
        term=me.varnames[keep],
        estimate=Float64.(me.effects[keep]),
        std_error=Float64.(me.se[keep]),
        stat=Float64.(me.z_stat[keep]),
        p_value=Float64.(me.p_values[keep]),
        ci_lower=Float64.(me.ci_lower[keep]),
        ci_upper=Float64.(me.ci_upper[keep]),
    )
end

# DIDResult — event-study coefficients keyed by event time; CI bounds are stored.
function _coef_nt(r::DIDResult)
    est = Float64.(collect(r.att))
    s = Float64.(collect(r.se))
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (
        event_time=collect(r.event_times),
        term=["e=$(e)" for e in r.event_times],
        estimate=est,
        std_error=s,
        stat=stat,
        p_value=pval,
        ci_lower=Float64.(collect(r.ci_lower)),
        ci_upper=Float64.(collect(r.ci_upper)),
    )
end

# Ordered logit/probit — two blocks (slopes + cutpoints) tagged by a `block` column.
_coef_nt(m::OrderedLogitModel) = _coef_nt_ordered(m)
_coef_nt(m::OrderedProbitModel) = _coef_nt_ordered(m)
function _coef_nt_ordered(m)
    K = length(m.beta)
    J = length(m.cutpoints)
    names = vcat(String.(m.varnames), ["cut$j" for j in 1:J])
    coefs = vcat(collect(m.beta), collect(m.cutpoints))
    se = stderror(m)                       # joint SE vector, length K + J
    block = vcat(fill("coef", K), fill("cutpoint", J))
    base = _coef_nt_simple(names, coefs, se, :z, 0)
    return merge((block=block,), base)
end

# Multinomial logit — one block of K rows per non-base alternative, tagged by `alternative`.
function _coef_nt(m::MultinomialLogitModel)
    K = length(m.varnames)
    # length K*(J-1), column-major over alternatives
    se_all = stderror(m)
    J_1 = size(m.beta, 2)
    # base category = alts[1]; categories may be Int/Symbol/String
    alts = string.(m.categories)
    terms = String[]
    alt = String[]
    est = Float64[]
    s = Float64[]
    for j in 1:J_1
        off = (j - 1) * K
        append!(terms, String.(m.varnames))
        append!(alt, fill("$(alts[j + 1]) vs $(alts[1])", K))
        append!(est, Float64.(m.beta[:, j]))
        append!(s, Float64.(se_all[(off + 1):(off + K)]))
    end
    base = _coef_nt_simple(terms, est, s, :z, 0)
    return merge((alternative=alt,), base)
end

# VARModel — one row per (equation, term); SEs from the equation-by-equation OLS vcov,
# mirroring `report(::VARModel)`.
function _coef_nt(model::VARModel)
    n = nvars(model)
    p = model.p
    coef_names = vcat(
        _INTERCEPT_LABEL,
        ["$(model.varnames[v]).L$l" for l in 1:p for v in 1:n],
    )
    _, X = construct_var_matrices(model.Y, p)
    XtX_inv = robust_inv(Matrix(X' * X))
    dof_r = effective_nobs(model) - ncoefs(model)
    eqs = String[]
    terms = String[]
    est = Float64[]
    s = Float64[]
    for j in 1:n
        se_j = sqrt.(max.(diag(XtX_inv) .* model.Sigma[j, j], 0))
        append!(eqs, fill(model.varnames[j], length(coef_names)))
        append!(terms, coef_names)
        append!(est, Float64.(model.B[:, j]))
        append!(s, Float64.(se_j))
    end
    base = _coef_nt_simple(terms, est, s, :t, dof_r)
    return merge((equation=eqs,), base)
end

# ── TIDY batch (v1.0.2, #853 series) ──────────────────────────────────────────

# Count models — z-based, mirroring `report()`. NegBin appends the dispersion
# row tagged by `block`, like the ordered two-block shape. (#854)
_coef_nt(m::PoissonModel) = _coef_nt_simple(m.varnames, coef(m), stderror(m), :z, 0)
function _coef_nt(m::NegBinModel)
    k = length(m.beta)
    base = _coef_nt_simple(
        vcat(String.(m.varnames), ["alpha"]),
        vcat(Float64.(m.beta), [Float64(m.alpha)]),
        vcat(Float64.(stderror(m)), [Float64(m.alpha_se)]),
        :z,
        0,
    )
    block = vcat(fill("coef", k), ["dispersion"])
    return merge((block=block,), base)
end

# Quantile regression — one block per tau tagged by a `tau` column; same
# t-distributed inference (dof n-k) as each per-quantile `report()` table. (#855)
function _coef_nt(m::QuantileRegModel)
    dof_r = size(m.X, 1) - size(m.X, 2)
    taus = Float64[]
    terms = String[]
    est = Float64[]
    s = Float64[]
    for (j, t) in enumerate(m.taus)
        append!(taus, fill(Float64(t), length(m.varnames)))
        append!(terms, String.(m.varnames))
        append!(est, Float64.(m.beta[:, j]))
        append!(s, Float64.(m.stderr[:, j]))
    end
    base = _coef_nt_simple(terms, est, s, :t, dof_r)
    return merge((tau=taus,), base)
end

# RDD — conventional + robust treatment-effect rows at the stored level, with
# the main/pilot bandwidths attached. (#855)
function _coef_nt(r::RDDResult)
    base = _coef_nt_simple(
        ["Conventional", "Robust (bias-corrected)"],
        [r.tau_conventional, r.tau_bias_corrected],
        [r.se_conventional, r.se_robust],
        :z,
        0;
        level=r.level,
    )
    return merge(base, (h=fill(Float64(r.h), 2), b=fill(Float64(r.b), 2)))
end

# SUR / 3SLS — one row per (equation, term); dof varies per equation so each
# block is built separately, mirroring the per-equation `report()` tables. (#856)
function _coef_nt_eqstack(eqnames, varnames, betas, ses, nobs)
    eqs = String[]
    terms = String[]
    est = Float64[]
    s = Float64[]
    stat = Float64[]
    pval = Float64[]
    lo = Float64[]
    hi = Float64[]
    for j in eachindex(eqnames)
        blk = _coef_nt_simple(
            varnames[j], betas[j], ses[j], :t, max(nobs - length(betas[j]), 1),
        )
        append!(eqs, fill(eqnames[j], length(blk.term)))
        append!(terms, blk.term)
        append!(est, blk.estimate)
        append!(s, blk.std_error)
        append!(stat, blk.stat)
        append!(pval, blk.p_value)
        append!(lo, blk.ci_lower)
        append!(hi, blk.ci_upper)
    end
    return (
        equation=eqs, term=terms, estimate=est, std_error=s, stat=stat,
        p_value=pval, ci_lower=lo, ci_upper=hi,
    )
end

function _coef_nt(m::SURModel)
    base = _coef_nt_eqstack(m.eqnames, m.varnames, m.betas, m.ses, m.nobs)
    n = length(base.term)
    return merge(
        base,
        (
            nobs=fill(m.nobs, n),
            mcelroy_r2=fill(Float64(m.mcelroy_r2), n),
            det_sigma=fill(Float64(m.det_sigma), n),
            loglik=fill(Float64(m.loglik), n),
        ),
    )
end

function _coef_nt(m::ThreeSLSModel)
    base = _coef_nt_eqstack(m.eqnames, m.varnames, m.betas, m.ses, m.nobs)
    n = length(base.term)
    ni = Int[]
    for j in eachindex(m.eqnames)
        append!(ni, fill(m.n_instruments[j], length(m.betas[j])))
    end
    return merge(
        base,
        (
            nobs=fill(m.nobs, n),
            mcelroy_r2=fill(Float64(m.mcelroy_r2), n),
            det_sigma=fill(Float64(m.det_sigma), n),
            n_instruments=ni,
        ),
    )
end

# EventStudyLP — event-time coefficients keyed by event time; CIs are stored,
# like `DIDResult`. (#866)
function _coef_nt(r::EventStudyLP)
    est = Float64.(collect(r.coefficients))
    s = Float64.(collect(r.se))
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (
        event_time=collect(r.event_times),
        term=["h=$(e)" for e in r.event_times],
        estimate=est,
        std_error=s,
        stat=stat,
        p_value=pval,
        ci_lower=Float64.(collect(r.ci_lower)),
        ci_upper=Float64.(collect(r.ci_upper)),
    )
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
        push!(et, missing)
        push!(term, nm)
        push!(est, Float64(pool.coef))
        push!(s, Float64(pool.se))
        push!(lo, Float64(pool.ci_lower))
        push!(hi, Float64(pool.ci_upper))
        push!(block, "pooled")
    end
    stat = est ./ s
    pval = Float64[2 * (1 - cdf(Normal(), abs(z))) for z in stat]
    return (
        block=block, event_time=et, term=term, estimate=est, std_error=s,
        stat=stat, p_value=pval, ci_lower=lo, ci_upper=hi,
    )
end

# BaconDecomposition — one row per 2x2 comparison. (#866)
function _coef_nt(r::BaconDecomposition)
    return (
        type=string.(r.comparison_type),
        cohort_i=collect(r.cohort_i),
        cohort_j=collect(r.cohort_j),
        estimate=Float64.(collect(r.estimates)),
        weight=Float64.(collect(r.weights)),
    )
end

# Per-category marginal effects — long (variable, category) rows with z-based
# inference. `se === nothing` (model covariance unavailable) yields NaN
# inference columns, matching the NaN fallback of the ordered AME. (#863)
function _marginal_effects_long(
    effects::AbstractMatrix, se, varnames, categories; skip_first::Bool,
)
    keep = findall(v -> _display_intercept(v) != _INTERCEPT_LABEL, varnames)
    isempty(keep) && (keep = collect(1:size(effects, 1)))
    jrange = skip_first ? (2:size(effects, 2)) : (1:size(effects, 2))
    variable = String[]
    category = String[]
    est = Float64[]
    s = Float64[]
    for j in jrange, i in keep
        push!(variable, string(varnames[i]))
        push!(category, string(categories[j]))
        push!(est, Float64(effects[i, j]))
        push!(s, se === nothing ? NaN : Float64(se[i, j]))
    end
    base = _coef_nt_simple(variable, est, s, :z, 0)
    return (
        variable=base.term, category=category, estimate=base.estimate,
        std_error=base.std_error, stat=base.stat, p_value=base.p_value,
        ci_lower=base.ci_lower, ci_upper=base.ci_upper,
    )
end

# Multinomial AME skips the base category (column 1), mirroring `show()`.
function _coef_nt(me::MultinomialMarginalEffects)
    return _marginal_effects_long(
        me.effects, me.se, me.varnames, me.categories; skip_first=true,
    )
end

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
    return (
        model=collect(c.models), weight=Float64.(collect(c.weights)),
        mse=Float64.(collect(c.mse)), method=fill(string(c.method), n),
    )
end

# ── Policy counterfactuals (#858) ─────────────────────────────────────────────

# Long (period, variable) path rows for outcome/instrument paths with baselines,
# counterfactuals, and first/last draw bands.
function _cf_push_paths!(
    period, variable, role, baseline, counterfactual, lower, upper,
    names, rolename, base, cf, bands,
)
    for (i, nm) in enumerate(names)
        b = base[i]
        cc = cf[i]
        Bd = bands === nothing ? nothing : bands[i]
        nq = Bd === nothing ? 0 : size(Bd, 2)
        for h in eachindex(b)
            push!(period, h)
            push!(variable, string(nm))
            push!(role, rolename)
            push!(baseline, Float64(b[h]))
            push!(counterfactual, Float64(cc[h]))
            push!(lower, nq > 0 ? Float64(Bd[h, 1]) : missing)
            push!(upper, nq > 0 ? Float64(Bd[h, nq]) : missing)
        end
    end
    return nothing
end

function _coef_nt(pc::PolicyCounterfactual)
    period = Int[]
    variable = String[]
    role = String[]
    baseline = Float64[]
    counterfactual = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    _cf_push_paths!(
        period, variable, role, baseline, counterfactual, lower, upper,
        pc.outcomes, "outcome", pc.x_base, pc.x_cf, pc.x_bands,
    )
    _cf_push_paths!(
        period, variable, role, baseline, counterfactual, lower, upper,
        pc.instruments, "instrument", pc.z_base, pc.z_cf, pc.z_bands,
    )
    return (
        period=period, variable=variable, role=role, baseline=baseline,
        counterfactual=counterfactual, lower=lower, upper=upper,
    )
end

# Second moments in pair grain: covariances and correlations, baseline vs cf.
function _coef_nt(cm::CounterfactualMoments)
    n = length(cm.varnames)
    vi = String[]
    vj = String[]
    cb = Float64[]
    cc = Float64[]
    rb = Float64[]
    rc = Float64[]
    for i in 1:n, j in 1:n
        push!(vi, string(cm.varnames[i]))
        push!(vj, string(cm.varnames[j]))
        push!(cb, Float64(cm.Sigma_base[i, j]))
        push!(cc, Float64(cm.Sigma_cf[i, j]))
        push!(rb, Float64(cm.corr_base[i, j]))
        push!(rc, Float64(cm.corr_cf[i, j]))
    end
    return (
        variable_i=vi, variable_j=vj, cov_base=cb, cov_cf=cc,
        corr_base=rb, corr_cf=rc,
    )
end

# Realized vs counterfactual panel in (date, variable) long shape; the per-date
# implementation residual repeats across variables.
function _coef_nt(ch::CounterfactualHistory)
    nd, nv = size(ch.realized)
    date = String[]
    variable = String[]
    realized = Float64[]
    counterfactual = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    relres = Float64[]
    nq = ch.cf_bands === nothing ? 0 : size(ch.cf_bands, 3)
    for d in 1:nd, v in 1:nv
        push!(date, ch.dates[d])
        push!(variable, string(ch.varnames[v]))
        push!(realized, Float64(ch.realized[d, v]))
        push!(counterfactual, Float64(ch.cf[d, v]))
        push!(lower, nq > 0 ? Float64(ch.cf_bands[d, v, 1]) : missing)
        push!(upper, nq > 0 ? Float64(ch.cf_bands[d, v, nq]) : missing)
        push!(relres, Float64(ch.rel_residual[d]))
    end
    return (
        date=date, variable=variable, realized=realized,
        counterfactual=counterfactual, cf_lower=lower, cf_upper=upper,
        rel_residual=relres,
    )
end

function _coef_nt(bp::BaselinePath)
    period = Int[]
    variable = String[]
    role = String[]
    value = Float64[]
    for (i, nm) in enumerate(bp.outcomes), h in eachindex(bp.x[i])
        push!(period, h)
        push!(variable, string(nm))
        push!(role, "outcome")
        push!(value, Float64(bp.x[i][h]))
    end
    for (k, nm) in enumerate(bp.instruments), h in eachindex(bp.z[k])
        push!(period, h)
        push!(variable, string(nm))
        push!(role, "instrument")
        push!(value, Float64(bp.z[k][h]))
    end
    return (period=period, variable=variable, role=role, value=value)
end

function _coef_nt(pf::PolicyForecast)
    period = Int[]
    variable = String[]
    value = Float64[]
    for (i, nm) in enumerate(pf.outcomes), h in eachindex(pf.values[i])
        push!(period, h)
        push!(variable, string(nm))
        push!(value, Float64(pf.values[i][h]))
    end
    return (period=period, variable=variable, value=value)
end

# OPP at every decision date with the news/preference/aging revision split.
function _coef_nt(sq::OPPSequence)
    ns, nd = size(sq.delta)
    date = String[]
    shock = String[]
    delta = Float64[]
    delta_tc = Float64[]
    news = Float64[]
    pref = Float64[]
    aging = Float64[]
    for t in 1:nd, s in 1:ns
        push!(date, sq.dates[t])
        push!(shock, sq.shock_labels[s])
        push!(delta, Float64(sq.delta[s, t]))
        push!(delta_tc, Float64(sq.delta_tc[s, t]))
        push!(news, Float64(sq.news_part[s, t]))
        push!(pref, Float64(sq.pref_part[s, t]))
        push!(aging, Float64(sq.aging_part[s, t]))
    end
    return (
        date=date, shock=shock, delta=delta, delta_tc=delta_tc,
        news=news, pref=pref, aging=aging,
    )
end

function _coef_nt(fs::ForecastSufficiency)
    H, no = size(fs.fev_ratio)
    horizon = Int[]
    observable = String[]
    fev_ratio = Float64[]
    one_step_ratio = Float64[]
    for h in 1:H, o in 1:no
        push!(horizon, h)
        push!(observable, string(fs.observables[o]))
        push!(fev_ratio, Float64(fs.fev_ratio[h, o]))
        push!(one_step_ratio, Float64(fs.one_step_ratio[o]))
    end
    return (
        horizon=horizon, observable=observable, fev_ratio=fev_ratio,
        one_step_ratio=one_step_ratio,
    )
end

# ── Input-output (#859) ───────────────────────────────────────────────────────

function _coef_nt(m::LeontiefModel)
    n = length(m.x)
    si = String[]
    sj = String[]
    av = Float64[]
    lv = Float64[]
    for i in 1:n, j in 1:n
        push!(si, string(m.io.sectors[i]))
        push!(sj, string(m.io.sectors[j]))
        push!(av, Float64(m.A[i, j]))
        push!(lv, Float64(m.L[i, j]))
    end
    return (sector_i=si, sector_j=sj, A=av, L=lv)
end

function _coef_nt(m::GhoshModel)
    n = length(m.x)
    si = String[]
    sj = String[]
    bv = Float64[]
    gv = Float64[]
    for i in 1:n, j in 1:n
        push!(si, string(m.io.sectors[i]))
        push!(sj, string(m.io.sectors[j]))
        push!(bv, Float64(m.B[i, j]))
        push!(gv, Float64(m.G[i, j]))
    end
    return (sector_i=si, sector_j=sj, B=bv, G=gv)
end

function _coef_nt(r::LinkageResult)
    return (
        sector=collect(r.sectors), backward=Float64.(collect(r.backward)),
        forward=Float64.(collect(r.forward)), Ui=Float64.(collect(r.Ui)),
        Uj=Float64.(collect(r.Uj)), classification=string.(r.classification),
    )
end

function _coef_nt(m::IOMultipliers)
    n = length(m.sectors)
    return (
        sector=collect(m.sectors), value=Float64.(collect(m.values)),
        kind=fill(string(m.kind), n), type=fill(string(m.type), n),
    )
end

# Per-stressor, per-sector contributions; `total` repeats the show() headline
# (consumption total summed over final-demand categories).
function _coef_nt(fp::FootprintResult)
    ns, nsec = size(fp.by_sector)
    totals = [sum(@view fp.total[i, :]) for i in 1:ns]
    stressor = String[]
    sector = Int[]
    value = Float64[]
    total = Float64[]
    for i in 1:ns, j in 1:nsec
        push!(stressor, fp.stressors[i])
        push!(sector, j)
        push!(value, Float64(fp.by_sector[i, j]))
        push!(total, Float64(totals[i]))
    end
    return (stressor=stressor, sector=sector, value=value, total=total)
end

# One row per (factor, index); factor order mirrors `show()` (r.factors first,
# then extra keys sorted). `total`/`residual` repeat across factors.
function _coef_nt(r::SDAResult)
    fkeys = [k for k in r.factors if haskey(r.effects, k)]
    append!(fkeys, sort!(setdiff!(collect(keys(r.effects)), r.factors)))
    factor = String[]
    index = Int[]
    effect = Float64[]
    total = Float64[]
    residual = Float64[]
    for k in fkeys
        v = r.effects[k]
        for i in eachindex(v)
            push!(factor, string(k))
            push!(index, i)
            push!(effect, Float64(v[i]))
            push!(total, Float64(r.total[i]))
            push!(residual, Float64(r.residual[i]))
        end
    end
    m = length(factor)
    return (
        factor=factor, index=index, effect=effect, total=total,
        residual=residual, method=fill(string(r.method), m), on=fill(string(r.on), m),
    )
end

function _coef_nt(fp::RegionalFootprintResult)
    ns, ng = size(fp.production)
    stressor = String[]
    region = String[]
    production = Float64[]
    consumption = Float64[]
    for i in 1:ns, r in 1:ng
        push!(stressor, fp.stressors[i])
        push!(region, fp.regions[r])
        push!(production, Float64(fp.production[i, r]))
        push!(consumption, Float64(fp.consumption[i, r]))
    end
    return (
        stressor=stressor, region=region, production=production,
        consumption=consumption,
    )
end

# ── Test battery (#860) ───────────────────────────────────────────────────────
# Uniform one-row-per-test (or per-hypothesis) tables: test label, statistic,
# p-value, 5% decision mirroring show() (critical-value comparison with the
# show() tail when the type carries CVs, else p < 0.05), and 1/5/10% CVs.
# Test-specific findings (breaks, ranks, estimates) ride in extra columns;
# spec metadata (lags, df, nobs, kernels) stays in fields.

function _new_test_cols()
    return (
        test=String[],
        statistic=Float64[],
        p_value=Union{Missing,Float64}[],
        decision=Union{Missing,String}[],
        cv_1pct=Union{Missing,Float64}[],
        cv_5pct=Union{Missing,Float64}[],
        cv_10pct=Union{Missing,Float64}[],
    )
end

function _test_p(p)
    return (p === nothing || p === missing || !isfinite(Float64(p))) ? missing : Float64(p)
end

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
        push!(c.cv_1pct, missing)
        push!(c.cv_5pct, missing)
        push!(c.cv_10pct, missing)
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

function _coef_nt(r::ADFResult)
    return _test_push!(
        _new_test_cols(), "ADF", r.statistic, r.pvalue, r.critical_values, :left,
    )
end
function _coef_nt(r::KPSSResult)
    return _test_push!(
        _new_test_cols(), "KPSS", r.statistic, r.pvalue, r.critical_values, :right,
    )
end
function _coef_nt(r::PPResult)
    return _test_push!(
        _new_test_cols(), "Phillips-Perron", r.statistic, r.pvalue, r.critical_values,
        :left,
    )
end
function _coef_nt(r::ERSResult)
    return _test_push!(
        _new_test_cols(), "ERS point-optimal", r.P_T, r.pvalue, r.critical_values, :left,
    )
end

function _coef_nt(r::ZAResult)
    c = _test_push!(
        _new_test_cols(), "Zivot-Andrews", r.statistic, r.pvalue, r.critical_values, :left,
    )
    return merge(
        c, (break_index=[r.break_index], break_fraction=[Float64(r.break_fraction)]),
    )
end

function _coef_nt(r::AndrewsResult)
    c = _test_push!(
        _new_test_cols(), "Andrews $(r.test_type)", r.statistic, r.pvalue,
        r.critical_values, :p,
    )
    return merge(
        c, (break_index=[r.break_index], break_fraction=[Float64(r.break_fraction)]),
    )
end

function _coef_nt(r::ADF2BreakResult)
    c = _test_push!(
        _new_test_cols(), "ADF two-break", r.statistic, r.pvalue, r.critical_values, :left,
    )
    return merge(
        c,
        (
            break_1=[r.break1],
            break_2=[r.break2],
            breakfrac_1=[Float64(r.break1_fraction)],
            breakfrac_2=[Float64(r.break2_fraction)],
        ),
    )
end

function _coef_nt(r::LMUnitRootResult)
    c = _test_push!(
        _new_test_cols(), "LM unit root", r.statistic, r.pvalue, r.critical_values, :left,
    )
    gd(i) = length(r.break_dates) >= i ? r.break_dates[i] : missing
    gf(i) = length(r.break_fractions) >= i ? Float64(r.break_fractions[i]) : missing
    return merge(
        c,
        (break_1=[gd(1)], break_2=[gd(2)], breakfrac_1=[gf(1)], breakfrac_2=[gf(2)]),
    )
end

function _coef_nt(r::FactorBreakResult)
    c = _test_push!(
        _new_test_cols(), "Factor break $(r.method)", r.statistic, r.pvalue, nothing, :p,
    )
    return merge(c, (break_index=[r.break_date === nothing ? missing : r.break_date],))
end

function _coef_nt(r::LLCResult)
    return _test_push!(
        _new_test_cols(), "Levin-Lin-Chu", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left,
    )
end
function _coef_nt(r::IPSResult)
    return _test_push!(
        _new_test_cols(), "Im-Pesaran-Shin", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left,
    )
end
function _coef_nt(r::BreitungPanelResult)
    return _test_push!(
        _new_test_cols(), "Breitung", r.statistic, r.pvalue, _NORMAL_LEFT_CV, :left,
    )
end
function _coef_nt(r::HadriResult)
    return _test_push!(
        _new_test_cols(), "Hadri", r.statistic, r.pvalue, _NORMAL_RIGHT_CV, :right,
    )
end
function _coef_nt(r::PesaranCIPSResult)
    return _test_push!(
        _new_test_cols(), "Pesaran CIPS", r.cips_statistic, r.pvalue, r.critical_values,
        :left,
    )
end

function _coef_nt(r::FisherTestResult)
    return merge(
        _test_push!(
            _new_test_cols(), "Fisher periodicity", r.statistic, r.pvalue, nothing, :p,
        ),
        (peak_freq=[Float64(r.peak_freq)],),
    )
end
function _coef_nt(r::BartlettWhiteNoiseResult)
    return _test_push!(
        _new_test_cols(), "Bartlett white noise", r.statistic, r.pvalue, nothing, :p,
    )
end
function _coef_nt(r::LjungBoxResult)
    return _test_push!(_new_test_cols(), "Ljung-Box", r.statistic, r.pvalue, nothing, :p)
end
function _coef_nt(r::BoxPierceResult)
    return _test_push!(_new_test_cols(), "Box-Pierce", r.statistic, r.pvalue, nothing, :p)
end
function _coef_nt(r::DurbinWatsonResult)
    return _test_push!(
        _new_test_cols(), "Durbin-Watson", r.statistic, r.pvalue, nothing, :p,
    )
end

function _coef_nt(r::GrangerCausalityResult)
    c = _test_push!(
        _new_test_cols(), "Granger causality", r.statistic, r.pvalue, nothing, :p,
    )
    return merge(c, (cause=[join(string.(r.cause), ",")], effect=[r.effect]))
end

function _coef_nt(r::LRTestResult)
    return _test_push!(
        _new_test_cols(), "Likelihood ratio", r.statistic, r.pvalue, nothing, :p,
    )
end
function _coef_nt(r::LMTestResult)
    return _test_push!(
        _new_test_cols(), "Lagrange multiplier", r.statistic, r.pvalue, nothing, :p,
    )
end
function _coef_nt(r::EngleGrangerResult)
    return _test_push!(
        _new_test_cols(), "Engle-Granger", r.statistic, r.pvalue, nothing, :p,
    )
end
function _coef_nt(r::HansenInstabilityResult)
    return _test_push!(
        _new_test_cols(), "Hansen instability", r.statistic, r.pvalue, nothing, :p,
    )
end
function _coef_nt(r::ParkAddedResult)
    return _test_push!(
        _new_test_cols(), "Park added variables", r.statistic, r.pvalue, nothing, :p,
    )
end

function _coef_nt(r::BubbleResult)
    return _test_push!(
        _new_test_cols(), r.kind == :sadf ? "SADF" : "GSADF",
        r.statistic, r.pvalue, r.critical_values, :right,
    )
end

function _coef_nt(r::EDFTestResult)
    label = get(_EDF_TEST_LABEL, r.test, string(r.test)) * " (" *
            get(_EDF_DIST_LABEL, r.dist, string(r.dist)) * ")"
    c = _test_push!(_new_test_cols(), label, r.statistic, r.pvalue, r.critical_values, :p)
    return merge(c, (raw_statistic=[Float64(r.raw_statistic)],))
end

function _coef_nt(r::EqualityTestResult)
    label = get(_EQ_TEST_LABELS, r.test_name, string(r.test_name))
    return _test_push!(_new_test_cols(), label, r.statistic, r.pvalue, nothing, :p)
end
function _coef_nt(r::PanelTestResult)
    return _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
end
function _coef_nt(r::PVARTestResult)
    return _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
end
function _coef_nt(r::NormalityTestResult)
    return _test_push!(
        _new_test_cols(), _normality_test_label(r), r.statistic, r.pvalue, nothing, :p,
    )
end

function _coef_nt(r::CorTestResult)
    method = if r.method === :pearson
        "Pearson"
    elseif r.method === :spearman
        "Spearman"
    else
        "Kendall"
    end
    c = _test_push!(
        _new_test_cols(), "Correlation ($method)", r.statistic, r.pvalue, nothing, :p,
    )
    hasci = r.method === :pearson && isfinite(r.ci_lower) && isfinite(r.ci_upper)
    return merge(
        c,
        (
            estimate=[Float64(r.estimate)],
            ci_lower=[hasci ? Float64(r.ci_lower) : missing],
            ci_upper=[hasci ? Float64(r.ci_upper) : missing],
        ),
    )
end

function _coef_nt(r::RegDiagnosticResult)
    c = _test_push!(_new_test_cols(), r.test_name, r.statistic, r.pvalue, nothing, :p)
    return merge(
        c,
        (
            f_stat=[r.f_stat === nothing ? missing : Float64(r.f_stat)],
            f_pvalue=[r.f_pvalue === nothing ? missing : Float64(r.f_pvalue)],
            aux_r2=[Float64(r.aux_r2)],
        ),
    )
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
    c = _test_push!(
        _new_test_cols(), "Forecast encompassing", r.tstat, r.pvalue, nothing, :p,
    )
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
    bp = Union{Missing,Float64}[
        hasb ? Float64(r.bootstrap_pvalues[s]) : missing for s in eachindex(r.names)
    ]
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
    bp = if r.bootstrap > 0 && isfinite(r.bootstrap_pvalue)
        Float64(r.bootstrap_pvalue)
    else
        missing
    end
    return merge(
        c,
        (
            Wbar=[Float64(r.Wbar), Float64(r.Wbar)],
            boot_p=[bp, missing],
            cause=[string(r.cause), string(r.cause)],
            effect=[string(r.effect), string(r.effect)],
        ),
    )
end

function _coef_nt(r::MincerZarnowitzResult)
    c = _new_test_cols()
    _test_push!(c, "Mincer-Zarnowitz Wald", r.wald, r.pvalue_wald, nothing, :p)
    _test_push!(c, "Mincer-Zarnowitz F", r.fstat, r.pvalue_f, nothing, :p)
    return merge(
        c,
        (
            a=[Float64(r.a), Float64(r.a)],
            b=[Float64(r.b), Float64(r.b)],
            se_a=[Float64(r.se[1]), Float64(r.se[1])],
            se_b=[Float64(r.se[2]), Float64(r.se[2])],
        ),
    )
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
    _test_push!(
        c, "Gregory-Hansen ADF*", r.adf_statistic, r.adf_pvalue, r.adf_critical_values,
        :left,
    )
    push!(brk, r.adf_break)
    _test_push!(
        c, "Gregory-Hansen Zt*", r.zt_statistic, r.zt_pvalue, r.adf_critical_values, :left,
    )
    push!(brk, r.zt_break)
    _test_push!(
        c, "Gregory-Hansen Za*", r.za_statistic, r.za_pvalue, r.za_critical_values, :left,
    )
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
    rk = Int[]
    kd = String[]
    for i in eachindex(r.trace_stats)
        rank = i - 1
        cvt = Dict(
            1 => r.critical_values_trace[i, 3],
            5 => r.critical_values_trace[i, 2],
            10 => r.critical_values_trace[i, 1],
        )
        _test_push!(
            c, "Johansen trace (rank ≤ $rank)",
            r.trace_stats[i], r.trace_pvalues[i], cvt, :right,
        )
        push!(rk, rank)
        push!(kd, "trace")
        cvm = Dict(
            1 => r.critical_values_max[i, 3],
            5 => r.critical_values_max[i, 2],
            10 => r.critical_values_max[i, 1],
        )
        _test_push!(
            c, "Johansen max (rank = $rank)",
            r.max_eigen_stats[i], r.max_eigen_pvalues[i], cvm, :right,
        )
        push!(rk, rank)
        push!(kd, "max")
    end
    return merge(c, (rank=rk, kind=kd))
end

function _coef_nt(r::FisherJohansenResult)
    c = _new_test_cols()
    rk = Int[]
    kd = String[]
    for j in eachindex(r.ranks)
        _test_push!(
            c, "Fisher-Johansen trace (rank ≤ $(r.ranks[j]))",
            r.trace_statistics[j], r.trace_pvalues[j], nothing, :p,
        )
        push!(rk, r.ranks[j])
        push!(kd, "trace")
        _test_push!(
            c, "Fisher-Johansen max (rank = $(r.ranks[j]))",
            r.max_statistics[j], r.max_pvalues[j], nothing, :p,
        )
        push!(rk, r.ranks[j])
        push!(kd, "max")
    end
    return merge(c, (rank=rk, kind=kd))
end

function _coef_nt(r::PANICResult)
    c = _new_test_cols()
    for j in 1:r.n_factors
        _test_push!(
            c, "PANIC factor $j",
            r.factor_adf_stats[j], r.factor_adf_pvalues[j], nothing, :p,
        )
    end
    _test_push!(c, "PANIC pooled", r.pooled_statistic, r.pooled_pvalue, nothing, :p)
    return c
end

function _coef_nt(r::HEGYResult)
    c = _new_test_cols()
    _test_push!(c, "HEGY t(0)", r.t_zero, missing, r.t_zero_cv, :left)
    _test_push!(c, "HEGY t(π)", r.t_nyquist, missing, r.t_nyquist_cv, :left)
    for (i, F) in enumerate(r.pair_F)
        freq_s = round(r.pair_freqs[i]; digits=3)
        _test_push!(c, "HEGY F(ω=$freq_s)", F, missing, r.pair_F_cv, :right)
    end
    _test_push!(c, "HEGY F seasonal", r.F_seasonal, missing, nothing, :p)
    _test_push!(c, "HEGY F all", r.F_all, missing, nothing, :p)
    return c
end

function _coef_nt(r::VarianceRatioResult)
    c = _new_test_cols()
    qv = Union{Missing,Int}[]
    vrv = Union{Missing,Float64}[]
    boot = Union{Missing,Float64}[]
    nq = length(r.q)
    for i in 1:nq
        _test_push!(c, "VR Z(q=$(r.q[i]))", r.z[i], r.z_pvalue[i], nothing, :p)
        push!(qv, r.q[i])
        push!(vrv, Float64(r.vr[i]))
        push!(boot, missing)
        has_zb = r.bootstrap > 0 && length(r.z_star_boot_pvalue) >= i &&
                 isfinite(r.z_star_boot_pvalue[i])
        zb = has_zb ? Float64(r.z_star_boot_pvalue[i]) : missing
        _test_push!(c, "VR Z*(q=$(r.q[i]))", r.z_star[i], r.z_star_pvalue[i], nothing, :p)
        push!(qv, r.q[i])
        push!(vrv, Float64(r.vr[i]))
        push!(boot, zb)
    end
    if r.wright
        for i in 1:nq
            for (nm, st, pv) in (
                ("R1", r.R1, r.R1_pvalue),
                ("R2", r.R2, r.R2_pvalue),
                ("S1", r.S1, r.S1_pvalue),
            )
                _test_push!(c, "Wright $nm(q=$(r.q[i]))", st[i], pv[i], nothing, :p)
                push!(qv, r.q[i])
                push!(vrv, missing)
                push!(boot, missing)
            end
        end
    end
    cdb = isfinite(r.cd_boot_pvalue) ? Float64(r.cd_boot_pvalue) : missing
    _test_push!(c, "Chow-Denning CD", r.cd_stat, r.cd_pvalue, nothing, :p)
    push!(qv, missing)
    push!(vrv, missing)
    push!(boot, cdb)
    _test_push!(c, "Chow-Denning CD*", r.cd_star_stat, r.cd_star_pvalue, nothing, :p)
    push!(qv, missing)
    push!(vrv, missing)
    push!(boot, cdb)
    return merge(c, (q=qv, vr=vrv, boot_p=boot))
end

function _coef_nt(r::BDSResult)
    c = _new_test_cols()
    mv = Int[]
    ev = Float64[]
    cvv = Float64[]
    boot = Union{Missing,Float64}[]
    for (im, m) in enumerate(r.m), (je, eps_val) in enumerate(r.eps)
        _test_push!(c, "BDS", r.statistic[im, je], r.pvalue[im, je], nothing, :p)
        push!(mv, m)
        push!(ev, Float64(eps_val))
        push!(cvv, Float64(r.C[im, je]))
        bp = r.boot_pvalue[im, je]
        push!(boot, isfinite(bp) ? Float64(bp) : missing)
    end
    return merge(c, (m=mv, eps=ev, C_m=cvv, boot_p=boot))
end

function _coef_nt(r::BaiPerronResult)
    c = _new_test_cols()
    for l in eachindex(r.supf_stats)
        _test_push!(
            c, "Bai-Perron sup-F($l)", r.supf_stats[l], r.supf_pvalues[l], nothing, :p,
        )
    end
    for i in eachindex(r.sequential_stats)
        _test_push!(
            c, "Bai-Perron seq($(i + 1)|$i)",
            r.sequential_stats[i], r.sequential_pvalues[i], nothing, :p,
        )
    end
    return c
end

# --- Battery suites (stacked member rows) + stability paths ---

function _stack_test_rows!(c, nt)
    append!(c.test, nt.test)
    append!(c.statistic, nt.statistic)
    append!(c.p_value, nt.p_value)
    append!(c.decision, nt.decision)
    append!(c.cv_1pct, nt.cv_1pct)
    append!(c.cv_5pct, nt.cv_5pct)
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

# --- DSGE / HA-DSGE solutions and simulations (#861) ---

# First-order policy functions in long (block, equation, variable) shape;
# v = [states; shocks] columns use state names then `spec.exog` shock names.
# Higher-order tensors and the steady state keep their native shapes (fields).
function _coef_nt(sol::PerturbationSolution)
    states = sol.spec.varnames[sol.state_indices]
    controls = sol.spec.varnames[sol.control_indices]
    vnames = vcat(states, string.(sol.spec.exog))
    block = String[]
    equation = String[]
    variable = String[]
    coef = Float64[]
    for i in axes(sol.gx, 1), j in axes(sol.gx, 2)
        push!(block, "g")
        push!(equation, controls[i])
        push!(variable, vnames[j])
        push!(coef, Float64(sol.gx[i, j]))
    end
    for i in axes(sol.hx, 1), j in axes(sol.hx, 2)
        push!(block, "h")
        push!(equation, states[i])
        push!(variable, vnames[j])
        push!(coef, Float64(sol.hx[i, j]))
    end
    return (block=block, equation=equation, variable=variable, coefficient=coef)
end

# Linear law of motion y = C + G1·y₋ + impact·ε in the same long shape; the
# constant block completes the law (variable == "constant").
function _coef_nt(sol::DSGESolution)
    vn = sol.spec.varnames
    sn = string.(sol.spec.exog)
    block = String[]
    equation = String[]
    variable = String[]
    coef = Float64[]
    for i in axes(sol.G1, 1), j in axes(sol.G1, 2)
        push!(block, "G1")
        push!(equation, vn[i])
        push!(variable, vn[j])
        push!(coef, Float64(sol.G1[i, j]))
    end
    for i in axes(sol.impact, 1), j in axes(sol.impact, 2)
        push!(block, "impact")
        push!(equation, vn[i])
        push!(variable, sn[j])
        push!(coef, Float64(sol.impact[i, j]))
    end
    for i in eachindex(sol.C_sol)
        push!(block, "C")
        push!(equation, vn[i])
        push!(variable, "constant")
        push!(coef, Float64(sol.C_sol[i]))
    end
    return (block=block, equation=equation, variable=variable, coefficient=coef)
end

# Equilibrium scalars: prices and aggregates, name-sorted for determinism.
function _coef_nt(ss::HASteadyState)
    block = String[]
    name = String[]
    value = Float64[]
    for (k, v) in sort!(collect(ss.prices); by=first)
        push!(block, "price")
        push!(name, string(k))
        push!(value, Float64(v))
    end
    for (k, v) in sort!(collect(ss.aggregates); by=first)
        push!(block, "aggregate")
        push!(name, string(k))
        push!(value, Float64(v))
    end
    return (block=block, name=name, value=value)
end

# Posterior simulation fan in (period, variable) long shape with outer bands.
function _coef_nt(s::BayesianDSGESimulation)
    Tp, nv = size(s.point_estimate)
    nq = size(s.quantiles, 3)
    period = Int[]
    variable = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for t in 1:Tp, v in 1:nv
        push!(period, t)
        push!(variable, s.variables[v])
        push!(value, Float64(s.point_estimate[t, v]))
        push!(lower, nq > 0 ? Float64(s.quantiles[t, v, 1]) : missing)
        push!(upper, nq > 0 ? Float64(s.quantiles[t, v, nq]) : missing)
    end
    return (period=period, variable=variable, value=value, lower=lower, upper=upper)
end

function _coef_nt(pf::PerfectForesightPath)
    Tp, nv = size(pf.path)
    period = Int[]
    variable = String[]
    level = Float64[]
    deviation = Float64[]
    for t in 1:Tp, v in 1:nv
        push!(period, t)
        push!(variable, pf.spec.varnames[v])
        push!(level, Float64(pf.path[t, v]))
        push!(deviation, Float64(pf.deviations[t, v]))
    end
    return (period=period, variable=variable, level=level, deviation=deviation)
end

function _coef_nt(sol::OccBinSolution)
    Tp, nv = size(sol.piecewise_path)
    period = Int[]
    variable = String[]
    linear = Float64[]
    piecewise = Float64[]
    for t in 1:Tp, v in 1:nv
        push!(period, t)
        push!(variable, sol.varnames[v])
        push!(linear, Float64(sol.linear_path[t, v]))
        push!(piecewise, Float64(sol.piecewise_path[t, v]))
    end
    return (period=period, variable=variable, linear=linear, piecewise=piecewise)
end

function _coef_nt(oirf::OccBinIRF)
    H, nv = size(oirf.piecewise)
    horizon = Int[]
    variable = String[]
    shock = String[]
    linear = Float64[]
    piecewise = Float64[]
    for h in 1:H, v in 1:nv
        push!(horizon, h)
        push!(variable, oirf.varnames[v])
        push!(shock, oirf.shock_name)
        push!(linear, Float64(oirf.linear[h, v]))
        push!(piecewise, Float64(oirf.piecewise[h, v]))
    end
    return (
        horizon=horizon, variable=variable, shock=shock,
        linear=linear, piecewise=piecewise,
    )
end

# CUSUM path in (period) long shape with per-point band breaches.
function _coef_nt(r::StabilityResult)
    n = length(r.tindex)
    stat = Float64.(collect(r.stat_path))
    upper = Float64.(collect(r.upper))
    lower = Float64.(collect(r.lower))
    breached = Bool[
        !isnan(s) && (s < lo || s > up) for (s, lo, up) in zip(stat, lower, upper)
    ]
    return (
        period=collect(r.tindex), kind=fill(string(r.kind), n), stat=stat,
        upper=upper, lower=lower, breached=breached,
    )
end

# Companion eigenvalues in (eigen) long shape.
function _coef_nt(r::VARStationarityResult)
    n = length(r.eigenvalues)
    return (
        test=fill("VAR stationarity", n),
        eigen_index=collect(1:n),
        re=Float64[real(e) for e in r.eigenvalues],
        im=Float64[imag(e) for e in r.eigenvalues],
        modulus=Float64[abs(e) for e in r.eigenvalues],
        is_stationary=fill(r.is_stationary, n),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Tables.jl source interface for coefficient-bearing types
# ─────────────────────────────────────────────────────────────────────────────

const _COEF_TABLE_TYPES = (
    RegModel, LogitModel, ProbitModel, PanelRegModel, PanelIVModel,
    PanelLogitModel, PanelProbitModel, MarginalEffects, OrderedLogitModel,
    OrderedProbitModel, MultinomialLogitModel, VARModel, DIDResult, PoissonModel,
    NegBinModel, QuantileRegModel, RDDResult, SURModel, ThreeSLSModel, EventStudyLP,
    LPDiDResult, BaconDecomposition, MultinomialMarginalEffects, ForecastEvaluation,
    ForecastCombination, PolicyCounterfactual, CounterfactualMoments,
    CounterfactualHistory, BaselinePath, PolicyForecast, OPPSequence,
    ForecastSufficiency, LeontiefModel, GhoshModel, LinkageResult, IOMultipliers,
    FootprintResult, SDAResult, RegionalFootprintResult,
)

for MT in _COEF_TABLE_TYPES
    @eval Tables.istable(::Type{<:$MT}) = true
    @eval Tables.columnaccess(::Type{<:$MT}) = true
    @eval Tables.columns(m::$MT) = _coef_nt(m)
    @eval Tables.schema(m::$MT) = Tables.schema(_coef_nt(m))
end

# Test-battery result types share the same Tables.jl wiring; their `_coef_nt`
# builders emit the uniform one-row-per-hypothesis shape (#860).
const _TEST_TABLE_TYPES = (
    ADFResult, KPSSResult, PPResult, ERSResult,
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
    PanelUnitRootSummary, StabilityResult, VARStationarityResult,
)

for MT in _TEST_TABLE_TYPES
    @eval Tables.istable(::Type{<:$MT}) = true
    @eval Tables.columnaccess(::Type{<:$MT}) = true
    @eval Tables.columns(m::$MT) = _coef_nt(m)
    @eval Tables.schema(m::$MT) = Tables.schema(_coef_nt(m))
end

# DSGE / HA-DSGE solutions and simulations (#861).
const _DSGE_TABLE_TYPES = (
    PerturbationSolution, DSGESolution, HASteadyState,
    BayesianDSGESimulation, PerfectForesightPath, OccBinSolution, OccBinIRF,
)

for MT in _DSGE_TABLE_TYPES
    @eval Tables.istable(::Type{<:$MT}) = true
    @eval Tables.columnaccess(::Type{<:$MT}) = true
    @eval Tables.columns(m::$MT) = _coef_nt(m)
    @eval Tables.schema(m::$MT) = Tables.schema(_coef_nt(m))
end

# ─────────────────────────────────────────────────────────────────────────────
# v1.1.0 Series Models, Tests, State-Space, DSGE-VAR, and Rational Inattention
# ─────────────────────────────────────────────────────────────────────────────

_safe_labels(names::AbstractVector{<:AbstractString}, n::Int, prefix::String) =
    length(names) == n ? String[string(s) for s in names] : String["$(prefix)_$i" for i in 1:n]

# -- QFM & QFAVAR series --

function _coef_nt(m::QuantileFactorModel)
    N, r = size(m.loadings)
    vn = _safe_labels(m.varnames, N, "var")
    r == 0 && return (
        variable=vn, factor=fill(0, N), tau=fill(Float64(m.tau), N),
        loading=fill(NaN, N), std_error=Vector{Union{Missing,Float64}}(fill(missing, N)),
        pseudo_r2=Float64.(collect(m.pseudo_r2)),
    )
    variable = String[]; factor = Int[]; tau = Float64[]
    loading = Float64[]; std_error = Union{Missing,Float64}[]; pseudo_r2 = Float64[]
    has_se = m.se_loadings !== nothing && size(m.se_loadings) == (N, r)
    pr2 = length(m.pseudo_r2) == N ? m.pseudo_r2 : fill(NaN, N)
    for k in 1:r, j in 1:N
        push!(variable, vn[j]); push!(factor, k); push!(tau, Float64(m.tau))
        push!(loading, Float64(m.loadings[j, k]))
        push!(std_error, has_se ? Float64(m.se_loadings[j, k]) : missing)
        push!(pseudo_r2, Float64(pr2[j]))
    end
    return (; variable, factor, tau, loading, std_error, pseudo_r2)
end

function _coef_nt(ic::QFMICResult)
    rmax = ic.r_max
    return (
        r=collect(1:rmax),
        tau=fill(Float64(ic.tau), rmax),
        sigma_sq=Float64.(collect(ic.sigma_sq[1:rmax])),
        loss=Float64.(collect(ic.loss[1:rmax])),
        IC1=Float64.(collect(ic.IC1[1:rmax])),
        IC2=Float64.(collect(ic.IC2[1:rmax])),
        IC3=Float64.(collect(ic.IC3[1:rmax])),
        r_IC1=fill(ic.r_IC1, rmax),
        r_IC2=fill(ic.r_IC2, rmax),
        r_IC3=fill(ic.r_IC3, rmax),
        r_rank=fill(ic.r_rank, rmax),
    )
end

function _coef_nt(g::QuantileFactorGrid)
    nt = length(g.taus)
    return (
        tau=Float64.(collect(g.taus)),
        r=Int.(collect(g.r_by_tau)),
        mean_pseudo_r2=Float64[mean(m.pseudo_r2) for m in g.models],
        converged=Bool[m.converged for m in g.models],
    )
end

function _coef_nt(s::QFMSpanningResult)
    tau = Float64[]; ols_r2 = Float64[]; quantile_r1 = Float64[]
    cancor = Float64[]; unspanned = Bool[]
    for (ti, t) in enumerate(s.taus)
        m = length(s.ols_r2[ti])
        for k in 1:m
            push!(tau, Float64(t))
            push!(ols_r2, Float64(s.ols_r2[ti][k]))
            push!(quantile_r1, Float64(s.quantile_r1[ti][min(k, end)]))
            push!(cancor, Float64(s.cancor[ti][min(k, end)]))
            push!(unspanned, Bool(s.unspanned[ti][min(k, end)]))
        end
    end
    return (tau=tau, factor=reduce(vcat, (collect(1:length(s.ols_r2[ti])) for ti in eachindex(s.taus)); init=Int[]),
            ols_r2=ols_r2, quantile_r1=quantile_r1, cancor=cancor, unspanned=unspanned)
end

function _coef_nt(t::QFMLoadingConstancyTest)
    return (
        test=["QFM loading constancy"],
        tau1=[Float64(t.tau1)],
        tau2=[Float64(t.tau2)],
        statistic=[Float64(t.statistic)],
        p_value=[Float64(t.pvalue)],
        n_boot=[t.n_boot],
        dof=[t.dof],
    )
end

function _coef_nt(g::GrowthAtRiskResult)
    T_eff = length(g.gar)
    act = length(g.actual_target) == T_eff ? Float64.(collect(g.actual_target)) : fill(NaN, T_eff)
    return (
        period=collect(1:T_eff),
        target=fill(g.target_name, T_eff),
        gar=Float64.(collect(g.gar)),
        expected_shortfall=Float64.(collect(g.expected_shortfall)),
        upside_entropy=Float64.(collect(g.upside_entropy)),
        downside_entropy=Float64.(collect(g.downside_entropy)),
        actual_target=act,
    )
end

function _coef_nt(m::QuantileVARModel)
    n = size(m.Y, 2)
    vn = _safe_labels(m.varnames, n, "y")
    k_lag = isempty(m.B_list) ? 0 : size(m.B_list[1], 1)
    tnames = vcat(["const"], ["L$(lag)_$v" for lag in 1:m.p for v in vn])
    length(tnames) == k_lag || (tnames = ["coef_$i" for i in 1:k_lag])
    tau = Float64[]; equation = String[]; term = String[]
    estimate = Float64[]; std_error = Float64[]
    for (ti, t) in enumerate(m.taus)
        B = m.B_list[ti]; SE = m.se_list[ti]
        for eq in 1:n, c in 1:size(B, 1)
            push!(tau, Float64(t)); push!(equation, vn[eq]); push!(term, tnames[c])
            push!(estimate, Float64(B[c, eq])); push!(std_error, Float64(SE[c, eq]))
        end
    end
    return (; tau, equation, term, estimate, std_error)
end

function _coef_nt(q::QuantileImpulseResponse)
    df = long_table(q)
    return NamedTuple{Tuple(Symbol.(names(df)))}(Tuple(df[!, col] for col in names(df)))
end

function _coef_nt(m::QuantileFAVAR)
    df = long_table(m)
    return NamedTuple{Tuple(Symbol.(names(df)))}(Tuple(df[!, col] for col in names(df)))
end

function _coef_nt(m::VBQuantileFAVAR)
    df = long_table(m)
    return NamedTuple{Tuple(Symbol.(names(df)))}(Tuple(df[!, col] for col in names(df)))
end

# -- GDFM & Factor Test series --

function _coef_nt(m::FHLZModel)
    nv, nq = size(m.impulse_filters, 2), size(m.impulse_filters, 3)
    vn = _safe_labels(m.varnames, nv, "var")
    ve = length(m.variance_explained)
    horizon = Int[]; variable = String[]; shock = String[]; value = Float64[]; var_explained = Float64[]
    H = size(m.impulse_filters, 1)
    for s in 1:nq, v in 1:nv, h in 1:H
        push!(horizon, h - 1); push!(variable, vn[v]); push!(shock, "u_$s")
        push!(value, Float64(m.impulse_filters[h, v, s]))
        push!(var_explained, s <= ve ? Float64(m.variance_explained[s]) : NaN)
    end
    return (; horizon, variable, shock, value, var_explained)
end

function _coef_nt(r::OnatskiQTestResult)
    return (
        test=["Onatski (2009) dynamic rank q-test"],
        k0=[r.k0], k1=[r.k1],
        statistic=[Float64(r.statistic)],
        p_value=[Float64(r.pvalue)],
        crit_05=[isempty(r.crit_vals) ? NaN : Float64(r.crit_vals[min(2, end)])],
        rejected_05=[r.rejected_05],
    )
end

function _coef_nt(r::AhnHorensteinResult)
    kmax = min(r.k_max, length(r.eigenvalues), length(r.er), length(r.gr))
    return (
        k=collect(1:kmax),
        eigenvalue=Float64.(collect(r.eigenvalues[1:kmax])),
        er=Float64.(collect(r.er[1:kmax])),
        gr=Float64.(collect(r.gr[1:kmax])),
        r_er=fill(r.r_er, kmax),
        r_gr=fill(r.r_gr, kmax),
    )
end

function _coef_nt(r::OnatskiEDResult)
    n = length(r.eigenvalues)
    return (
        k=collect(1:n),
        eigenvalue=Float64.(collect(r.eigenvalues)),
        delta=fill(Float64(r.delta), n),
        r_ed=fill(r.r_ed, n),
        converged=fill(r.converged, n),
    )
end

function _coef_nt(m::CointegratedGDFM)
    H, nv, nq = size(m.irf_matrices)
    vn = _safe_labels(m.varnames, nv, "var")
    sn = _safe_labels(m.shock_names, nq, "shock")
    horizon = Int[]; variable = String[]; shock = String[]; value = Float64[]
    for h in 1:H, v in 1:nv, s in 1:nq
        push!(horizon, h - 1); push!(variable, vn[v]); push!(shock, sn[s])
        push!(value, Float64(m.irf_matrices[h, v, s]))
    end
    return (; horizon, variable, shock, value)
end

function _coef_nt(r::BandPassGDFMResult)
    N = length(r.variance_share)
    return (
        variable=_safe_labels(r.varnames, N, "var"),
        variance_share=Float64.(collect(r.variance_share)),
        r=fill(r.r, N),
        band_low=fill(Float64(r.band[1]), N),
        band_high=fill(Float64(r.band[2]), N),
    )
end

function _coef_nt(r::EurocoinResult)
    T_eff = length(r.indicator)
    tlp = length(r.target_lowpass) == T_eff ? Float64.(collect(r.target_lowpass)) : fill(NaN, T_eff)
    return (
        period=collect(1:T_eff),
        indicator=Float64.(collect(r.indicator)),
        target_lowpass=tlp,
        r2=fill(Float64(r.r2), T_eff),
        cutoff=fill(Float64(r.cutoff), T_eff),
    )
end

function _coef_nt(r::SpectralFEVDResult)
    nv, ns = size(r.shares)
    vn = _safe_labels(r.varnames, nv, "var")
    sn = _safe_labels(r.shock_names, ns, "shock")
    variable = String[]; shock = String[]; share = Float64[]; power_band = Float64[]
    for v in 1:nv, s in 1:ns
        push!(variable, vn[v]); push!(shock, sn[s])
        push!(share, Float64(r.shares[v, s]))
        push!(power_band, Float64(r.power_band[v, s]))
    end
    return (; variable, shock, share, power_band)
end

function _coef_nt(m::TimeVaryingGDFM)
    T_obs, q = size(m.variance_share_path)
    time = Int[]; factor = Int[]; variance_share = Float64[]
    for k in 1:q, t in 1:T_obs
        push!(time, t); push!(factor, k); push!(variance_share, Float64(m.variance_share_path[t, k]))
    end
    return (; time, factor, variance_share)
end

function _coef_nt(r::GDFMNowcastResult)
    n = length(r.nowcast)
    return (
        variable=_safe_labels(r.varnames, n, "var"),
        nowcast=Float64.(collect(r.nowcast)),
        std_error=Float64.(collect(r.se)),
        horizon=fill(r.horizon, n),
        observed=Bool.(collect(r.observed_mask[1:min(n, end)])),
    )
end

# -- NGID & SSVARMA series --

function _coef_nt(m::LMSSVARResult)
    n = size(m.B0, 1)
    sn = _safe_labels(m.shock_names, n, "shock")
    param = String[]; estimate = Float64[]; std_error = Float64[]
    for j in 1:n, i in 1:n
        push!(param, "B0[$i,$(sn[j])]")
        push!(estimate, Float64(m.B0[i, j]))
        push!(std_error, Float64(m.se_B0[i, j]))
    end
    for j in 1:length(m.sigma)
        push!(param, "sigma[$(sn[j])]")
        push!(estimate, Float64(m.sigma[j]))
        push!(std_error, j <= length(m.se_sigma) ? Float64(m.se_sigma[j]) : NaN)
    end
    return (parameter=param, estimate=estimate, std_error=std_error)
end

function _coef_nt(t::SVARHypothesisTest)
    return (
        test=[string(t.hypothesis_kind, " (", t.test_type, ")")],
        statistic=[Float64(t.statistic)],
        p_value=[Float64(t.pvalue)],
        bootstrap_pvalue=[t.bootstrap_pvalue === nothing ? missing : Float64(t.bootstrap_pvalue)],
        df=[t.df],
        nobs=[t.nobs],
        converged=[t.converged],
    )
end

function _coef_nt(d::NonGaussianDiagnostics)
    n = length(d.jb_stats)
    return (
        shock=_safe_labels(d.shock_names, n, "shock"),
        jb_stat=Float64.(collect(d.jb_stats)),
        jb_pval=Float64.(collect(d.jb_pvals)),
        jb_pval_holm=Float64.(collect(d.jb_pvals_holm)),
        skewness=Float64.(collect(d.skewness)),
        excess_kurtosis=Float64.(collect(d.excess_kurtosis)),
        lb_levels_pval=Float64.(collect(d.lb_levels_pvals)),
        lb_squares_pval=Float64.(collect(d.lb_squares_pvals)),
        bds_pval=Float64.(collect(d.bds_pvals)),
    )
end

function _coef_nt(m::LanneLuotoGMMResult)
    n = size(m.B0, 1)
    sn = _safe_labels(m.shock_names, n, "shock")
    param = String[]; estimate = Float64[]; std_error = Float64[]
    for j in 1:n, i in 1:n
        push!(param, "B0[$i,$(sn[j])]")
        push!(estimate, Float64(m.B0[i, j]))
        push!(std_error, Float64(m.se_B0[i, j]))
    end
    return (parameter=param, estimate=estimate, std_error=std_error,
            J_stat=fill(Float64(m.J_stat), length(param)),
            J_pvalue=fill(Float64(m.J_pvalue), length(param)))
end

function _coef_nt(r::GMMMomentSelectionResult)
    isempty(r.candidates) && return (
        n_asym=[length(r.msc_asym_pairs)], n_sym=[length(r.msc_sym_pairs)],
        n_moments=[r.n_moments], df=[r.df],
        J_stat=[Float64(r.J_stat)], J_pvalue=[Float64(r.J_pvalue)],
        msc=[Float64(r.msc_score)], rmsc=[Float64(r.rmsc_score)],
    )
    return (
        n_asym=[length(c.asym_pairs) for c in r.candidates],
        n_sym=[length(c.sym_pairs) for c in r.candidates],
        n_moments=[c.n_moments for c in r.candidates],
        df=[c.df for c in r.candidates],
        J_stat=[Float64(c.J_stat) for c in r.candidates],
        J_pvalue=[Float64(c.J_pvalue) for c in r.candidates],
        msc=[Float64(c.msc) for c in r.candidates],
        rmsc=[Float64(c.rmsc) for c in r.candidates],
        rank_ok=[c.rank_ok for c in r.candidates],
    )
end

function _coef_nt(m::SVARMAModel)
    n = length(m.mu)
    vn = _safe_labels(m.varnames, n, "y")
    sn = _safe_labels(m.shock_names, n, "shock")
    param = String[]; estimate = Float64[]
    for i in 1:n
        push!(param, "mu[$(vn[i])]")
        push!(estimate, Float64(m.mu[i]))
    end
    for l in 1:size(m.Phi, 3), j in 1:n, i in 1:n
        push!(param, "Phi$(l)[$(vn[i]),$(vn[j])]")
        push!(estimate, Float64(m.Phi[i, j, l]))
    end
    for l in 1:size(m.Theta, 3), j in 1:n, i in 1:n
        push!(param, "Theta$(l)[$(vn[i]),$(sn[j])]")
        push!(estimate, Float64(m.Theta[i, j, l]))
    end
    for j in 1:n, i in 1:n
        push!(param, "C[$(vn[i]),$(sn[j])]")
        push!(estimate, Float64(m.C[i, j]))
    end
    return (parameter=param, estimate=estimate,
            regime=fill(string(m.regime), length(param)),
            loglik=fill(Float64(m.loglik), length(param)))
end

function _coef_nt(r::FundamentalnessTestResult)
    k = length(r.lag_statistics)
    k == 0 && return (
        lag=[0], statistic=[Float64(r.statistic)], p_value=[Float64(r.pvalue)],
        overall_statistic=[Float64(r.statistic)], overall_pvalue=[Float64(r.pvalue)],
    )
    return (
        lag=collect(1:k),
        statistic=Float64.(collect(r.lag_statistics)),
        p_value=Float64.(collect(r.lag_pvalues)),
        overall_statistic=fill(Float64(r.statistic), k),
        overall_pvalue=fill(Float64(r.pvalue), k),
    )
end

function _coef_nt(r::SufficientInformationTestResult)
    k = length(r.r_grid)
    k == 0 && return (
        r=[0], f_statistic=[Float64(r.statistic)], f_pvalue=[Float64(r.pvalue)],
        wald_statistic=[Float64(r.wald_statistic)], wald_pvalue=[Float64(r.wald_pvalue)],
    )
    return (
        r=Int.(collect(r.r_grid)),
        f_statistic=Float64.(collect(r.f_statistics)),
        f_pvalue=Float64.(collect(r.f_pvalues)),
        wald_statistic=Float64.(collect(r.wald_statistics)),
        wald_pvalue=Float64.(collect(r.wald_pvalues)),
    )
end

# -- ABCD & DSGE-VAR series --

function _coef_nt(sys::ABCDSystem)
    eigs = size(sys.A, 1) > 0 ? Complex{Float64}.(eigvals(sys.A)) : Complex{Float64}[]
    n = length(eigs)
    n == 0 && return (eigen_index=[0], re=[NaN], im=[NaN], modulus=[NaN],
                      n_states=[size(sys.A, 1)], n_obs=[size(sys.C, 1)], n_shocks=[size(sys.B, 2)])
    return (
        eigen_index=collect(1:n),
        re=real.(eigs),
        im=imag.(eigs),
        modulus=abs.(eigs),
        n_states=fill(size(sys.A, 1), n),
        n_obs=fill(size(sys.C, 1), n),
        n_shocks=fill(size(sys.B, 2), n),
    )
end

function _coef_nt(r::InvertibilityResult)
    n = length(r.zero_re)
    n == 0 && return (
        zero_index=[0], re=[NaN], im=[NaN], modulus=[NaN],
        invertible=[r.invertible], status=[string(r.status)], max_modulus=[Float64(r.max_modulus)],
    )
    re = Float64.(collect(r.zero_re))
    im_v = Float64.(collect(r.zero_im))
    return (
        zero_index=collect(1:n),
        re=re,
        im=im_v,
        modulus=hypot.(re, im_v),
        invertible=fill(r.invertible, n),
        status=fill(string(r.status), n),
        max_modulus=fill(Float64(r.max_modulus), n),
    )
end

function _coef_nt(innov::InnovationsRepresentation)
    n_s, k = size(innov.K)
    sn = _safe_labels(innov.state_names, n_s, "state")
    on = _safe_labels(innov.obs_names, k, "obs")
    n_s == 0 && return (state=["none"], obs=["none"], K_gain=[0.0],
                        invertible=[innov.invertible], dare_residual=[Float64(innov.dare_residual)])
    state = String[]; obs = String[]; K_gain = Float64[]
    for j in 1:k, i in 1:n_s
        push!(state, sn[i]); push!(obs, on[j]); push!(K_gain, Float64(innov.K[i, j]))
    end
    m = length(K_gain)
    return (state=state, obs=obs, K_gain=K_gain,
            invertible=fill(innov.invertible, m),
            dare_residual=fill(Float64(innov.dare_residual), m))
end

function _coef_nt(r::ShockRecoverabilityResult)
    df = long_table(r)
    return NamedTuple{Tuple(Symbol.(names(df)))}(Tuple(df[!, col] for col in names(df)))
end

function _coef_nt(a::DSGEVARIRFAudit)
    H = a.horizon
    return (
        horizon=collect(0:(H - 1)),
        total_rmse=Float64.(collect(a.rmse_by_horizon)),
        noninv_wedge=Float64.(collect(a.noninv_by_horizon)),
        trunc_wedge=Float64.(collect(a.trunc_by_horizon)),
        id_wedge=Float64.(collect(a.id_by_horizon)),
        is_invertible=fill(a.is_invertible, H),
    )
end

function _coef_nt(p::DSGEVARPrior)
    k, n = size(p.Phi_star)
    on = _safe_labels(p.obs_names, n, "obs")
    eq = String[]; row = Int[]; phi_star = Float64[]
    for j in 1:n, i in 1:k
        push!(eq, on[j]); push!(row, i); push!(phi_star, Float64(p.Phi_star[i, j]))
    end
    m = length(phi_star)
    return (equation=eq, regressor=row, phi_star=phi_star,
            lambda=fill(Float64(p.lambda), m), lambda_min=fill(Float64(p.lambda_min), m))
end

function _coef_nt(pr::DSGEVARLambdaProfile)
    m = length(pr.lambdas)
    return (
        lambda=Float64.(collect(pr.lambdas)),
        log_marginal_likelihood=Float64.(collect(pr.log_marginal_likelihoods)),
        lambda_hat=fill(Float64(pr.lambda_hat), m),
        logml_hat=fill(Float64(pr.logml_hat), m),
        logml_inf=fill(Float64(pr.logml_inf), m),
    )
end

function _coef_nt(post::DSGEVARPosterior)
    if post.theta_draws !== nothing && !isempty(post.param_names) && size(post.theta_draws, 2) == length(post.param_names)
        pnames = String[string(s) for s in post.param_names]
        vals = [Float64.(vec(post.theta_draws[:, j])) for j in 1:length(pnames)]
        return (
            parameter=pnames,
            mean=Float64[mean(v) for v in vals],
            std=Float64[std(v) for v in vals],
            q05=Float64[quantile(v, 0.05) for v in vals],
            median=Float64[quantile(v, 0.5) for v in vals],
            q95=Float64[quantile(v, 0.95) for v in vals],
        )
    end
    k, n, nd = size(post.B_draws)
    on = _safe_labels(post.varnames, n, "y")
    param = String[]; mn = Float64[]; sd = Float64[]; q05 = Float64[]; med = Float64[]; q95 = Float64[]
    for j in 1:n, i in 1:k
        v = Float64.(vec(post.B_draws[i, j, :]))
        push!(param, "B[$i,$(on[j])]")
        push!(mn, mean(v)); push!(sd, std(v))
        push!(q05, quantile(v, 0.05)); push!(med, quantile(v, 0.5)); push!(q95, quantile(v, 0.95))
    end
    return (parameter=param, mean=mn, std=sd, q05=q05, median=med, q95=q95)
end

# -- Rational Inattention & SteadyStateKalman series --

function _coef_nt(ssk::SteadyStateKalman)
    n = length(ssk.closed_loop_eigs)
    eigs = Complex{Float64}.(ssk.closed_loop_eigs)
    return (
        eigen_index=collect(1:n),
        re=real.(eigs),
        im=imag.(eigs),
        modulus=abs.(eigs),
        mean_K_diag=[i <= size(ssk.K_gain, 1) && i <= size(ssk.K_gain, 2) ? Float64(ssk.K_gain[i, i]) : NaN for i in 1:n],
    )
end

function _coef_nt(prob::RationalInattentionProblem)
    m = size(prob.A, 1)
    sn = _safe_labels(prob.state_names, m, "state")
    return (
        state=sn,
        A_diag=Float64[prob.A[i, i] for i in 1:m],
        beta=fill(Float64(prob.beta), m),
        capacity=Vector{Union{Missing,Float64}}(fill(prob.capacity === nothing ? missing : Float64(prob.capacity), m)),
        omega=Vector{Union{Missing,Float64}}(fill(prob.omega === nothing ? missing : Float64(prob.omega), m)),
        units=fill(string(prob.units), m),
    )
end

function _coef_nt(sol::SimsKalmanSolution)
    m, k = size(sol.K_gain)
    sn = _safe_labels(sol.problem.state_names, m, "state")
    state = String[]; signal = Int[]; K_gain = Float64[]
    for j in 1:k, i in 1:m
        push!(state, sn[i]); push!(signal, j); push!(K_gain, Float64(sol.K_gain[i, j]))
    end
    if isempty(K_gain)
        return (state=["none"], signal=[0], K_gain=[0.0], omega=[Float64(sol.omega)],
                capacity_bits=[Float64(sol.capacity_bits)], signal_rank=[sol.signal_rank],
                expected_loss=[Float64(sol.expected_loss)])
    end
    len = length(K_gain)
    return (state=state, signal=signal, K_gain=K_gain,
            omega=fill(Float64(sol.omega), len),
            capacity_bits=fill(Float64(sol.capacity_bits), len),
            signal_rank=fill(sol.signal_rank, len),
            expected_loss=fill(Float64(sol.expected_loss), len))
end

function _coef_nt(sol::DynamicRISolution)
    return (
        period=collect(1:sol.T_path),
        omega=Float64.(collect(sol.omega_path)),
        capacity_bits=Float64.(collect(sol.capacity_bits_path)),
        capacity_nats=Float64.(collect(sol.capacity_nats_path)),
        signal_rank=Int.(collect(sol.signal_rank_path)),
        converged=fill(sol.converged, sol.T_path),
    )
end

function _coef_nt(r::RIEstimationResult)
    isempty(r.params) && return (
        term=["omega"], estimate=[Float64(r.omega)], std_error=[NaN],
        stat=[NaN], p_value=[NaN], ci_lower=[NaN], ci_upper=[NaN],
    )
    return _coef_nt_simple(r.param_names, r.params, r.stderr, :z, 0)
end

function _coef_nt(r::CGTestResult)
    return (
        parameter=["alpha", "beta_cg"],
        estimate=[Float64(r.alpha), Float64(r.beta_cg)],
        std_error=[Float64(r.se_alpha), Float64(r.se_cg)],
        t_stat=[Float64(r.alpha / r.se_alpha), Float64(r.t_stat)],
        p_value=[2 * (1 - cdf(Normal(), abs(Float64(r.alpha / r.se_alpha)))), Float64(r.pvalue)],
        implied_gain=[Float64(r.implied_gain), Float64(r.implied_gain)],
        implied_capacity=[Float64(r.implied_capacity), Float64(r.implied_capacity)],
    )
end

function _coef_nt(r::MWAttentionResult)
    H = length(r.agg_price_irf)
    return (
        horizon=collect(0:(H - 1)),
        agg_price_irf=Float64.(collect(r.agg_price_irf)),
        idio_price_irf=Float64.(collect(r.idio_price_irf)),
        real_output_irf=Float64.(collect(r.real_output_irf)),
        kappa_agg=fill(Float64(r.kappa_agg), H),
        kappa_idio=fill(Float64(r.kappa_idio), H),
        omega=fill(Float64(r.omega), H),
    )
end

function _coef_nt(sol::RIDiscreteChoiceSolution)
    S, J = size(sol.cond_probs)
    state = Int[]; action = Int[]; payoff = Float64[]; cond_prob = Float64[]; uncond_prob = Float64[]
    for j in 1:J, s in 1:S
        push!(state, s); push!(action, j)
        push!(payoff, Float64(sol.payoffs[s, j]))
        push!(cond_prob, Float64(sol.cond_probs[s, j]))
        push!(uncond_prob, Float64(sol.uncond_probs[j]))
    end
    return (; state, action, payoff, cond_prob, uncond_prob)
end

function _coef_nt(d::RIIIADiagnostic)
    return (
        diagnostic=["IIA Violation"],
        max_iia_violation=[Float64(d.max_iia_violation)],
        cond_odds_ratio_max_diff=[Float64(d.cond_odds_ratio_max_diff)],
        base_info_bits=[Float64(d.base_solution.mutual_info_bits)],
        expanded_info_bits=[Float64(d.expanded_solution.mutual_info_bits)],
    )
end

function _coef_nt(m::RILogitModel)
    K = length(m.beta)
    if K == 0
        return (
            term=["omega"], estimate=[Float64(m.omega)], std_error=[NaN],
            stat=[NaN], p_value=[NaN], ci_lower=[NaN], ci_upper=[NaN],
        )
    end
    se = size(m.vcov_mat) == (K, K) ? Float64[sqrt(max(Float64(m.vcov_mat[i, i]), 0.0)) for i in 1:K] : fill(NaN, K)
    vn = _safe_labels(m.varnames, K, "x")
    return _coef_nt_simple(vn, m.beta, se, :z, 0)
end

const _V110_TABLE_TYPES = (
    QuantileFactorModel, QFMICResult, QuantileFactorGrid, QFMSpanningResult,
    QFMLoadingConstancyTest, GrowthAtRiskResult, QuantileVARModel,
    QuantileImpulseResponse, QuantileFAVAR, VBQuantileFAVAR,
    FHLZModel, OnatskiQTestResult, AhnHorensteinResult, OnatskiEDResult,
    CointegratedGDFM, BandPassGDFMResult, EurocoinResult, SpectralFEVDResult,
    TimeVaryingGDFM, GDFMNowcastResult,
    LMSSVARResult, SVARHypothesisTest, NonGaussianDiagnostics,
    LanneLuotoGMMResult, GMMMomentSelectionResult, SVARMAModel,
    FundamentalnessTestResult, SufficientInformationTestResult,
    ABCDSystem, InvertibilityResult, InnovationsRepresentation,
    ShockRecoverabilityResult, DSGEVARIRFAudit, DSGEVARPrior,
    DSGEVARLambdaProfile, DSGEVARPosterior,
    SteadyStateKalman, RationalInattentionProblem, SimsKalmanSolution,
    DynamicRISolution, RIEstimationResult, CGTestResult, MWAttentionResult,
    RIDiscreteChoiceSolution, RIIIADiagnostic, RILogitModel,
)

for MT in _V110_TABLE_TYPES
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
| `ImpulseResponse` | `horizon, variable, shock, value, lower, upper` |
| `BayesianImpulseResponse` | `horizon, variable, shock, value, lower, upper` |
| `FEVD` | `horizon, variable, shock, value` |
| `BayesianFEVD` | `horizon, variable, shock, value, lower, upper` |
| `LPImpulseResponse` | `horizon, variable, shock, value, se, lower, upper` |
| `LPFEVD` | `horizon, variable, shock, value, se, lower, upper` |
| `HistoricalDecomposition` | `time, variable, shock, value` |
| `BayesianHistoricalDecomposition` | `time, variable, shock, value, lower, upper` |
| `AbstractForecastResult` (VAR/BVAR/VECM/LP) | `horizon, variable, value, lower, upper` |
| `MidasForecast` | `horizon, variable, value, se, lower, upper` (direct `h`) |
| ordered `marginal_effects` NamedTuple | `variable, category, estimate, std_error, ...` |
| `QuantileImpulseResponse` | `horizon, variable, shock, tau, value, lower, upper` |
| `QuantileFactorModel` / `QuantileFactorGrid` | `(tau), variable, factor, loading, std_error, pseudo_r2` |
| `GrowthAtRiskResult` | `period, tau, fitted_quantile, gar, expected_shortfall, actual` |
| `QuantileFAVAR` / `VBQuantileFAVAR` | `tau, variable, component, estimate` |
| `FHLZModel` / `CointegratedGDFM` | `horizon, variable, shock, value` |
| `BandPassGDFMResult` / `TimeVaryingGDFM` | `time, variable/factor, ...` |
| `SpectralFEVDResult` | `variable, shock, share, power_band` |
| `DSGEVARIRFAudit` / `ShockRecoverabilityResult` | `horizon/lead, variable/shock, ...` |
| `DynamicRISolution` / `RIDiscreteChoiceSolution` | `period/state, ..., value` |

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
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h)
        push!(variable, irf.variables[v])
        push!(shock, irf.shocks[s])
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
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h)
        push!(variable, irf.variables[v])
        push!(shock, irf.shocks[s])
        push!(value, irf.point_estimate[h, v, s])
        push!(lower, nq > 0 ? irf.quantiles[h, v, s, 1] : missing)
        push!(upper, nq > 0 ? irf.quantiles[h, v, s, nq] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

function long_table(f::FEVD)
    nv, ns, H = size(f.proportions)     # (variable, shock, horizon)
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h)
        push!(variable, f.variables[v])
        push!(shock, f.shocks[s])
        push!(value, f.proportions[v, s, h])
    end
    return DataFrame(; horizon, variable, shock, value)
end

function long_table(irf::LPImpulseResponse)
    H1, nresp = size(irf.values)         # (horizon 0..H stored in rows, response)
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    se = Float64[]
    lower = Float64[]
    upper = Float64[]
    for h in 1:H1, v in 1:nresp
        push!(horizon, h - 1)
        push!(variable, irf.response_vars[v])
        push!(shock, irf.shock_var)
        push!(value, irf.values[h, v])
        push!(se, irf.se[h, v])
        push!(lower, irf.ci_lower[h, v])
        push!(upper, irf.ci_upper[h, v])
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
    horizon = Int[]
    variable = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv
        push!(horizon, h)
        push!(variable, names[v])
        push!(value, pf[h, v])
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
    time = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    for t in 1:Teff, v in 1:nv, s in 1:ns
        push!(time, t)
        push!(variable, hd.variables[v])
        push!(shock, hd.shock_names[s])
        push!(value, hd.contributions[t, v, s])
    end
    return DataFrame(; time, variable, shock, value)
end

function long_table(hd::BayesianHistoricalDecomposition)
    Teff, nv, ns = size(hd.point_estimate)    # (time, variable, shock)
    nq = size(hd.quantiles, 4)
    time = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for t in 1:Teff, v in 1:nv, s in 1:ns
        push!(time, t)
        push!(variable, hd.variables[v])
        push!(shock, hd.shock_names[s])
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
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    lower = Union{Missing,Float64}[]
    upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h)
        push!(variable, f.variables[v])
        push!(shock, f.shocks[s])
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
    horizon = Int[]
    variable = String[]
    shock = String[]
    value = Float64[]
    se = Float64[]
    lower = Float64[]
    upper = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h)
        push!(variable, f.variables[v])
        push!(shock, f.shocks[s])
        push!(value, f.bias_corrected[v, s, h])
        push!(se, f.se[v, s, h])
        push!(lower, f.ci_lower[v, s, h])
        push!(upper, f.ci_upper[v, s, h])
    end
    return DataFrame(; horizon, variable, shock, value, se, lower, upper)
end

# Ordered-model `marginal_effects` returns a bare NamedTuple (no display struct),
# so it cannot be a Tables.jl source (that would pirate Base.NamedTuple) — expose
# the same long (variable, category) shape via `long_table` instead. All J
# categories are kept: ordered AMEs have no base normalization. (#863)
function long_table(nt::NamedTuple{(:effects, :se, :varnames, :categories)})
    cols = _marginal_effects_long(
        nt.effects, nt.se, nt.varnames, nt.categories; skip_first=false,
    )
    return DataFrame(; cols...)
end

function long_table(irf::QuantileImpulseResponse)
    H, nv, ns = size(irf.values)
    vn = _safe_labels(irf.variables, nv, "var")
    sn = _safe_labels(irf.shocks, ns, "shock")
    has_ci = irf.ci_type != :none
    horizon = Int[]; variable = String[]; shock = String[]; tau = Float64[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, vn[v]); push!(shock, sn[s])
        push!(tau, Float64(irf.tau_profile[min(h, end)]))
        push!(value, Float64(irf.values[h, v, s]))
        push!(lower, has_ci ? Float64(irf.ci_lower[h, v, s]) : missing)
        push!(upper, has_ci ? Float64(irf.ci_upper[h, v, s]) : missing)
    end
    return DataFrame(; horizon, variable, shock, tau, value, lower, upper)
end

long_table(m::QuantileFactorModel) = DataFrame(_coef_nt(m))

function long_table(g::QuantileFactorGrid)
    tau = Float64[]; variable = String[]; factor = Int[]; loading = Float64[]; r_at_tau = Int[]
    for (ti, m) in enumerate(g.models)
        N, r = size(m.loadings)
        vn = _safe_labels(m.varnames, N, "var")
        t = Float64(g.taus[ti]); rk = g.r_by_tau[ti]
        for k in 1:r, j in 1:N
            push!(tau, t); push!(variable, vn[j]); push!(factor, k)
            push!(loading, Float64(m.loadings[j, k])); push!(r_at_tau, rk)
        end
    end
    return DataFrame(; tau, variable, factor, loading, r_at_tau)
end

function long_table(g::GrowthAtRiskResult)
    Teff, nq = size(g.fitted_quantiles)
    act = length(g.actual_target) == Teff ? g.actual_target : fill(NaN, Teff)
    period = Int[]; tau = Float64[]; fitted_quantile = Float64[]
    gar = Float64[]; expected_shortfall = Float64[]; actual = Float64[]
    for q in 1:nq, t in 1:Teff
        push!(period, t); push!(tau, Float64(g.taus[min(q, end)]))
        push!(fitted_quantile, Float64(g.fitted_quantiles[t, q]))
        push!(gar, Float64(g.gar[t]))
        push!(expected_shortfall, Float64(g.expected_shortfall[t]))
        push!(actual, Float64(act[t]))
    end
    return DataFrame(; period, tau, fitted_quantile, gar, expected_shortfall, actual)
end

function long_table(m::QuantileFAVAR)
    tau = Float64[]; variable = String[]; component = String[]; estimate = Float64[]
    N = size(m.X_panel, 2)
    pvn = _safe_labels(m.panel_varnames, N, "panel")
    for (ti, t) in enumerate(m.taus)
        L = m.loadings_list[ti]; Ly = m.lambda_y_list[ti]
        for k in 1:size(L, 2), i in 1:size(L, 1)
            push!(tau, Float64(t)); push!(variable, pvn[min(i, end)])
            push!(component, "factor_$k"); push!(estimate, Float64(L[i, k]))
        end
        for k in 1:size(Ly, 2), i in 1:size(Ly, 1)
            push!(tau, Float64(t)); push!(variable, pvn[min(i, end)])
            push!(component, "key_$k"); push!(estimate, Float64(Ly[i, k]))
        end
    end
    return DataFrame(; tau, variable, component, estimate)
end

function long_table(m::VBQuantileFAVAR)
    tau = Float64[]; variable = String[]; component = String[]; estimate = Float64[]
    N = size(m.X_panel, 2)
    pvn = _safe_labels(m.panel_varnames, N, "panel")
    for (ti, t) in enumerate(m.taus)
        L = m.loadings_mean[ti]; Ly = m.lambda_y_mean[ti]
        for k in 1:size(L, 2), i in 1:size(L, 1)
            push!(tau, Float64(t)); push!(variable, pvn[min(i, end)])
            push!(component, "factor_$k"); push!(estimate, Float64(L[i, k]))
        end
        for k in 1:size(Ly, 2), i in 1:size(Ly, 1)
            push!(tau, Float64(t)); push!(variable, pvn[min(i, end)])
            push!(component, "key_$k"); push!(estimate, Float64(Ly[i, k]))
        end
    end
    return DataFrame(; tau, variable, component, estimate)
end

long_table(m::FHLZModel) = DataFrame(_coef_nt(m))
long_table(m::CointegratedGDFM) = DataFrame(_coef_nt(m))

function long_table(r::BandPassGDFMResult)
    Teff, N = size(r.common_twosided)
    vn = _safe_labels(r.varnames, N, "var")
    time = Int[]; variable = String[]; twosided = Float64[]; onesided = Float64[]
    for j in 1:N, t in 1:Teff
        push!(time, t); push!(variable, vn[j])
        push!(twosided, Float64(r.common_twosided[t, j]))
        push!(onesided, Float64(r.common_onesided[t, j]))
    end
    return DataFrame(; time, variable, twosided, onesided)
end

long_table(r::SpectralFEVDResult) = DataFrame(_coef_nt(r))
long_table(m::TimeVaryingGDFM) = DataFrame(_coef_nt(m))

function long_table(a::DSGEVARIRFAudit)
    H, nv, ns = size(a.irf_dsge)
    vn = _safe_labels(a.obs_names, nv, "obs")
    sn = _safe_labels(a.shock_names, ns, "shock")
    horizon = Int[]; variable = String[]; shock = String[]
    dsge_irf = Float64[]; wold_best_irf = Float64[]; wold_id_irf = Float64[]; svar_irf = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h - 1); push!(variable, vn[v]); push!(shock, sn[s])
        push!(dsge_irf, Float64(a.irf_dsge[h, v, s]))
        push!(wold_best_irf, Float64(a.irf_wold_best[h, v, s]))
        push!(wold_id_irf, Float64(a.irf_wold_id[h, v, s]))
        push!(svar_irf, Float64(a.irf_svar[h, v, s]))
    end
    return DataFrame(; horizon, variable, shock, dsge_irf, wold_best_irf, wold_id_irf, svar_irf)
end

function long_table(r::ShockRecoverabilityResult)
    L, m = size(r.r2_matrix)
    sn = _safe_labels(r.shock_names, m, "shock")
    lead = Int[]; shock = String[]; r2 = Float64[]
    for l in 1:L, s in 1:m
        push!(lead, l - 1); push!(shock, sn[s]); push!(r2, Float64(r.r2_matrix[l, s]))
    end
    return DataFrame(; lead, shock, r2)
end

long_table(sol::DynamicRISolution) = DataFrame(_coef_nt(sol))
long_table(sol::RIDiscreteChoiceSolution) = DataFrame(_coef_nt(sol))

# ─────────────────────────────────────────────────────────────────────────────
# write_csv — export any Tables-compatible result or long_table to CSV
# ─────────────────────────────────────────────────────────────────────────────

# Coerce a result to a Tables.jl-compatible object: coefficient models and DataFrames
# are already tables; array-valued results route through long_table.
_tabular(x) = x
_tabular(x::ImpulseResponse) = long_table(x)
_tabular(x::BayesianImpulseResponse) = long_table(x)
_tabular(x::FEVD) = long_table(x)
_tabular(x::BayesianFEVD) = long_table(x)
_tabular(x::LPImpulseResponse) = long_table(x)
_tabular(x::LPFEVD) = long_table(x)
_tabular(x::HistoricalDecomposition) = long_table(x)
_tabular(x::BayesianHistoricalDecomposition) = long_table(x)
_tabular(x::AbstractForecastResult) = long_table(x)
_tabular(x::QuantileImpulseResponse) = long_table(x)
_tabular(x::QuantileFactorGrid) = long_table(x)
_tabular(x::GrowthAtRiskResult) = long_table(x)
_tabular(x::FHLZModel) = long_table(x)
_tabular(x::CointegratedGDFM) = long_table(x)
_tabular(x::BandPassGDFMResult) = long_table(x)
_tabular(x::SpectralFEVDResult) = long_table(x)
_tabular(x::TimeVaryingGDFM) = long_table(x)
_tabular(x::DSGEVARIRFAudit) = long_table(x)
_tabular(x::ShockRecoverabilityResult) = long_table(x)
_tabular(x::DynamicRISolution) = long_table(x)
_tabular(x::RIDiscreteChoiceSolution) = long_table(x)

_csv_cell(::Missing) = ""
_csv_cell(x::Real) = string(x)
function _csv_cell(x)
    s = string(x)
    return (occursin(',', s) || occursin('"', s) || occursin('\n', s)) ?
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
# term,estimate,std_error,stat,p_value,ci_lower,ci_upper
write_csv(m, "coefficients.csv")
# or pass the result directly: write_csv(irf, "irf.csv")
write_csv(long_table(irf), "irf.csv")
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
