# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Per-type `reproduce` methods for randomized estimators (RSER-13 / #786).
# Loaded after every result type so the methods can dispatch.

function _reproduce_or_decline(x, howto::AbstractString)
    m = x.manifest
    m === nothing && return _no_manifest_report(string(nameof(typeof(x))))
    m.seed === nothing && return _no_seed_report(m, howto)
    return m
end

function reproduce(m::SVModel)
    man = _reproduce_or_decline(m, "estimate_sv(y; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    fresh = estimate_sv(m.y;
                        n_samples=m.n_samples,
                        burnin=Int(get(s, "burnin", 1000)),
                        dist=m.dist,
                        leverage=m.leverage,
                        quantile_levels=m.quantile_levels,
                        seed=man.seed)
    diffs = [_repro_field_diff("h_draws", m.h_draws, fresh.h_draws),
             _repro_field_diff("mu_post", m.mu_post, fresh.mu_post),
             _repro_field_diff("phi_post", m.phi_post, fresh.phi_post),
             _repro_field_diff("sigma_eta_post", m.sigma_eta_post, fresh.sigma_eta_post)]
    return _finalize_repro(diffs, man)
end

function reproduce(post::TVPVARPosterior)
    man = _reproduce_or_decline(post, "estimate_tvpvar(Y, p; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    fresh = estimate_tvpvar(post.Y, post.p;
                            tvp=post.tvp, sv=post.sv,
                            n_draws=Int(get(s, "n_draws", size(post.B_draws, 1))),
                            n_burn=Int(get(s, "n_burn", 0)),
                            thin=Int(get(s, "thin", 1)),
                            n_train=Int(get(s, "n_train", post.n_train)),
                            k_Q=get(s, "k_Q", 0.01),
                            k_S=get(s, "k_S", 0.1),
                            k_W=get(s, "k_W", 0.01),
                            varnames=post.varnames,
                            seed=man.seed)
    diffs = [_repro_field_diff("B_draws", post.B_draws, fresh.B_draws),
             _repro_field_diff("H_draws", post.H_draws, fresh.H_draws)]
    return _finalize_repro(diffs, man)
end

function reproduce(model::PVARModel)
    man = _reproduce_or_decline(model, "pvar_bootstrap_irf(model, H; seed=N)")
    man isa ReproReport && return man
    model.boot_draws === nothing && return _no_manifest_report("PVARModel (no bootstrap draws)")
    s = man.settings
    fresh = pvar_bootstrap_irf(model, Int(get(s, "H", size(model.boot_irf, 1) - 1));
                               irf_type=Symbol(get(s, "irf_type", "oirf")),
                               n_draws=Int(get(s, "n_draws", size(model.boot_draws, 1))),
                               ci=get(s, "ci", 0.95),
                               seed=man.seed)
    diffs = [_repro_field_diff("draws", model.boot_draws, fresh.draws),
             _repro_field_diff("irf", model.boot_irf, fresh.irf)]
    return _finalize_repro(diffs, man)
end

function reproduce(r::DIDResult)
    man = _reproduce_or_decline(r, "estimate_did(pd, y, d; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    haskey(s, "data") || return _needs_source_report("DIDResult",
        "estimate_did(pd, y, d; seed=N) (panel data was not stored on the manifest)")
    covs = get(s, "covariates", String[])
    covs = covs isa AbstractVector ? String[string(c) for c in covs] : String[]
    fresh = estimate_did(s["data"], Symbol(s["outcome"]), Symbol(s["treatment"]);
                         method=Symbol(get(s, "method", "did_multiplegt")),
                         leads=Int(get(s, "leads", 0)),
                         horizon=Int(get(s, "horizon", 5)),
                         covariates=covs,
                         control_group=Symbol(get(s, "control_group", "never_treated")),
                         cluster=Symbol(get(s, "cluster", "unit")),
                         conf_level=get(s, "conf_level", 0.95),
                         n_boot=Int(get(s, "n_boot", 200)),
                         base_period=Symbol(get(s, "base_period", "varying")),
                         seed=man.seed)
    diffs = [_repro_field_diff("att", r.att, fresh.att),
             _repro_field_diff("se", r.se, fresh.se)]
    return _finalize_repro(diffs, man)
end

function reproduce(fc::MSForecast)
    man = _reproduce_or_decline(fc, "forecast(ms, h; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    haskey(s, "model") || return _needs_source_report("MSForecast",
        "forecast(ms, h; seed=N) (source MSRegModel was not stored on the manifest)")
    msrc = s["model"]
    fresh = if haskey(s, "X_new")
        forecast(msrc, s["X_new"]; reps=Int(get(s, "reps", fc.reps)),
                 level=get(s, "level", fc.conf_level), seed=man.seed)
    else
        forecast(msrc, Int(get(s, "h", fc.horizon)); reps=Int(get(s, "reps", fc.reps)),
                 level=get(s, "level", fc.conf_level), seed=man.seed)
    end
    diffs = [_repro_field_diff("forecast", fc.forecast, fresh.forecast),
             _repro_field_diff("ci_lower", fc.ci_lower, fresh.ci_lower),
             _repro_field_diff("ci_upper", fc.ci_upper, fresh.ci_upper)]
    return _finalize_repro(diffs, man)
end

function reproduce(r::OPPResult)
    man = _reproduce_or_decline(r, "estimate_opp(fc, ce, loss; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    (haskey(s, "fc") && haskey(s, "ce") && haskey(s, "loss")) ||
        return _needs_source_report("OPPResult",
            "estimate_opp(fc, ce, loss; seed=N) (forecast/causal-effects/loss were not stored)")
    lv = get(s, "levels", [0.60, 0.75, 0.90])
    levels = lv isa AbstractVector ? Tuple(Float64.(lv)) : Tuple(lv)
    fresh = estimate_opp(s["fc"], s["ce"], s["loss"];
                         n_sim=Int(get(s, "n_sim", 2000)),
                         independent=Bool(get(s, "independent", true)),
                         levels=levels,
                         seed=man.seed)
    diffs = [_repro_field_diff("delta_draws", r.delta_draws, fresh.delta_draws),
             _repro_field_diff("delta", r.delta, fresh.delta)]
    return _finalize_repro(diffs, man)
end

function reproduce(post::MFVARPosterior)
    man = _reproduce_or_decline(post, "estimate_mfvar(data, p; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    fresh = estimate_mfvar(post.data, post.p;
                           low_freq=post.low_freq, freq_ratio=post.freq_ratio,
                           aggregation=length(post.aggregation) == 1 ? post.aggregation[1] : post.aggregation,
                           n_draws=Int(get(s, "n_draws", size(post.B_draws, 1))),
                           n_burn=Int(get(s, "n_burn", 0)),
                           prior=Symbol(get(s, "prior", "minnesota")),
                           varnames=post.varnames, seed=man.seed)
    diffs = [_repro_field_diff("B_draws", post.B_draws, fresh.B_draws),
             _repro_field_diff("Sigma_draws", post.Sigma_draws, fresh.Sigma_draws)]
    return _finalize_repro(diffs, man)
end

function reproduce(fc::STARForecast)
    man = _reproduce_or_decline(fc, "forecast(star, h; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    haskey(s, "model") || return _needs_source_report("STARForecast",
        "forecast(star, h; seed=N)")
    fresh = forecast(s["model"], Int(get(s, "h", fc.horizon));
                     reps=Int(get(s, "reps", fc.reps)),
                     level=get(s, "level", fc.conf_level), seed=man.seed)
    diffs = [_repro_field_diff("forecast", fc.forecast, fresh.forecast),
             _repro_field_diff("ci_lower", fc.ci_lower, fresh.ci_lower)]
    return _finalize_repro(diffs, man)
end

function reproduce(fc::ThresholdForecast)
    man = _reproduce_or_decline(fc, "forecast(setar, h; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    haskey(s, "model") || return _needs_source_report("ThresholdForecast",
        "forecast(setar, h; seed=N)")
    fresh = forecast(s["model"], Int(get(s, "h", fc.horizon));
                     reps=Int(get(s, "reps", fc.reps)),
                     level=get(s, "level", fc.conf_level), seed=man.seed)
    diffs = [_repro_field_diff("forecast", fc.forecast, fresh.forecast),
             _repro_field_diff("ci_lower", fc.ci_lower, fresh.ci_lower)]
    return _finalize_repro(diffs, man)
end

function reproduce(t::HansenLinearityTest)
    man = _reproduce_or_decline(t, "hansen_linearity_test(y, X, q; seed=N)")
    man isa ReproReport && return man
    s = man.settings
    (haskey(s, "y") && haskey(s, "X") && haskey(s, "q")) ||
        return _needs_source_report("HansenLinearityTest", "hansen_linearity_test(y, X, q; seed=N)")
    fresh = hansen_linearity_test(s["y"], s["X"], s["q"];
                                  trim=get(s, "trim", t.trim),
                                  reps=Int(get(s, "reps", t.reps)), seed=man.seed)
    diffs = [_repro_field_diff("sup_lm", t.sup_lm, fresh.sup_lm),
             _repro_field_diff("pvalue_lm", t.pvalue_lm, fresh.pvalue_lm)]
    return _finalize_repro(diffs, man)
end

function reproduce(b::WildClusterBootstrap)
    man = _reproduce_or_decline(b, "wild_cluster_bootstrap(model, coef; seed=N)")
    man isa ReproReport && return man
    return _needs_source_report("WildClusterBootstrap",
        "wild_cluster_bootstrap(model, coefficient; seed=N) (source regression is not retained)")
end

function reproduce(::BayesianFAVAR)
    _needs_source_report("BayesianFAVAR", "estimate_favar(X, Y_key, r, p; method=:bayesian, seed=N)")
end

function reproduce(::StructuralDFM)
    _needs_source_report("StructuralDFM", "estimate_structural_dfm(X, q; seed=N)")
end

function reproduce(::SMMModel)
    _needs_source_report("SMMModel", "estimate_smm(simulator, moments, theta0, data; seed=N)")
end

function reproduce(::QuantileRegModel)
    _needs_source_report("QuantileRegModel", "estimate_qreg(y, X; seed=N)")
end

function reproduce(::RobustRegModel)
    _needs_source_report("RobustRegModel", "estimate_robust(y, X; seed=N)")
end

function reproduce(::ThresholdModel)
    _needs_source_report("ThresholdModel", "estimate_threshold(y, X, q; seed=N)")
end

function reproduce(::STARModel)
    _needs_source_report("STARModel", "estimate_star(y, p; seed=N)")
end

function reproduce(::StructuralLP)
    _needs_source_report("StructuralLP", "structural_lp(Y, H; seed=N)")
end

function reproduce(::ConditionalForecast)
    _needs_source_report("ConditionalForecast", "conditional_forecast(model, conditions, h; seed=N)")
end

function reproduce(::BayesianHistoricalDecomposition)
    _needs_source_report("BayesianHistoricalDecomposition",
        "historical_decomposition(post; seed=N)")
end

function reproduce(::OPPSequence)
    _needs_source_report("OPPSequence", "opp_sequence(forecasts, ce, loss; seed=N)")
end

function reproduce(::PolicyCausalEffects)
    _needs_source_report("PolicyCausalEffects", "policy_causal_effects(...; seed=N)")
end

function reproduce(::PolicyForecast)
    _needs_source_report("PolicyForecast", "policy_forecast(...; seed=N)")
end

function reproduce(::SpanningDiagnostic)
    _needs_source_report("SpanningDiagnostic", "spanning_diagnostic(...; seed=N)")
end

function reproduce(::ModelBankMember)
    _needs_source_report("ModelBankMember", "irf_match(...; seed=N)")
end

function reproduce(::NARDLMultipliers)
    _needs_source_report("NARDLMultipliers", "dynamic_multipliers(m, H; seed=N)")
end
