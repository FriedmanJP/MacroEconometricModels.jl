# Counterfactual module — report() displays (CF-21, #401)
#
# Every public result type speaks report() (Stata/EViews-style tables via
# _pretty_table/_coef_table) with the honesty diagnostics — implementation
# error, spanning verdict, band polarity — carried into what users actually
# read. Base.show stays the cheap one-liner (types.jl); the rich content
# lives on report(io, x).

report(x::AbstractCounterfactual) = report(stdout, x)

_cf_hsel(H::Int) = sort(unique(filter(<=(H), [1, 4, 8, 16, H])))

const _OPP_POLARITY_NOTE = "60/75/90% credible bands; rejection at LOWER levels is the conservative choice for a policymaker averse to non-optimality"

function report(io::IO, ce::PolicyCausalEffects{T}) where {T}
    spec = Any["Source" string(ce.source);
               "Horizon H" ce.H;
               "Shock columns n_s" n_shocks(ce);
               "Menu shape" (is_square(ce) ? "square (full news menu)" : "thin (identified subset)");
               "Draws" n_draws(ce)]
    _pretty_table(io, spec; title="Policy Causal Effects",
                  column_labels=["", ""], alignment=[:l, :r])
    n_show = min(n_shocks(ce), 5)
    vars = vcat(ce.outcomes, ce.instruments)
    mats = vcat(ce.Theta_x, ce.Theta_z)
    data = Matrix{Any}(undef, length(vars), 1 + n_show)
    for (v, sym) in enumerate(vars)
        data[v, 1] = string(sym)
        for k in 1:n_show
            data[v, k+1] = _fmt(mats[v][1, k])
        end
    end
    note = n_show < n_shocks(ce) ? " (first $n_show of $(n_shocks(ce)))" : ""
    _pretty_table(io, data; title="Impact responses (h = 1)$note",
                  column_labels=vcat([""], [ce.shock_labels[k] for k in 1:n_show]),
                  alignment=vcat([:l], fill(:r, n_show)))
    return nothing
end

function report(io::IO, pc::PolicyCounterfactual{T}) where {T}
    spec = Any["Rule / loss" pc.rule_name;
               "Horizon H" pc.H;
               "Draws used (failed)" "$(pc.n_draws_used) ($(pc.n_draws_failed))"]
    if !isnan(pc.loss_cf)
        spec = vcat(spec, Any["Loss baseline → counterfactual" "$(_fmt(pc.loss_base)) → $(_fmt(pc.loss_cf))";
                              "‖FOC‖ at optimum" _fmt(pc.foc_norm)])
    end
    _pretty_table(io, spec; title="Policy Counterfactual (McKay–Wolf)",
                  column_labels=["", ""], alignment=[:l, :r])

    hs = _cf_hsel(pc.H)
    has_b = pc.x_bands !== nothing
    for (i, sym) in enumerate(pc.outcomes)
        ncol = has_b ? 5 : 3
        data = Matrix{Any}(undef, length(hs), ncol)
        for (r, h) in enumerate(hs)
            data[r, 1] = h
            data[r, 2] = _fmt(pc.x_base[i][h])
            data[r, 3] = _fmt(pc.x_cf[i][h])
            if has_b
                data[r, 4] = _fmt(pc.x_bands[i][h, 1])
                data[r, 5] = _fmt(pc.x_bands[i][h, end])
            end
        end
        _pretty_table(io, data; title="$(sym)",
                      column_labels=has_b ? ["h", "Baseline", "Counterfactual", "Lower", "Upper"] :
                                    ["h", "Baseline", "Counterfactual"],
                      alignment=fill(:r, ncol))
    end
    for (k, sym) in enumerate(pc.instruments)
        data = Matrix{Any}(undef, length(hs), 3)
        for (r, h) in enumerate(hs)
            data[r, 1] = h
            data[r, 2] = _fmt(pc.z_base[k][h])
            data[r, 3] = _fmt(pc.z_cf[k][h])
        end
        _pretty_table(io, data; title="$(sym) (instrument)",
                      column_labels=["h", "Baseline", "Counterfactual"],
                      alignment=[:r, :r, :r])
    end

    # diagnostics block — the MW honesty signal must be un-missable
    n_show = min(length(pc.nu), 8)
    nud = Matrix{Any}(undef, n_show, 2)
    for k in 1:n_show
        nud[k, 1] = pc.shock_labels[k]
        nud[k, 2] = _fmt(pc.nu[k])
    end
    _pretty_table(io, nud;
                  title="Enforcing shock vector ν*" *
                        (n_show < length(pc.nu) ? " (first $n_show of $(length(pc.nu)))" : ""),
                  column_labels=["shock", "ν"], alignment=[:l, :r])
    verdict = pc.spanned ? "yes" : "NO"
    println(io, "Rule enforceable within shock span: $verdict (rel. residual = $(_fmt(pc.rel_residual)))")
    pc.spanned || println(io, "⚠ the counterfactual is a best approximation — inspect error_path")
    return nothing
end

function report(io::IO, r::OPPResult{T}) where {T}
    origin = isempty(r.origin) ? "—" : r.origin
    spec = Any["Forecast origin" origin;
               "Horizon H" r.H;
               "Loss announced → perturbed" "$(_fmt(r.loss_base)) → $(_fmt(r.loss_opp))";
               "‖gradient‖ (FOC statistic)" _fmt(norm(r.gradient))]
    _pretty_table(io, spec; title="Optimal Policy Perturbation (Barnichon–Mesters)",
                  column_labels=["", ""], alignment=[:l, :r])
    n_s = length(r.delta)
    if r.bands === nothing
        data = Matrix{Any}(undef, n_s, 2)
        for k in 1:n_s
            data[k, 1] = r.shock_labels[k]
            data[k, 2] = _fmt(r.delta[k])
        end
        _pretty_table(io, data; title="OPP δ* (plug-in)",
                      column_labels=["shock direction", "δ"], alignment=[:l, :r])
    else
        lv = sort(collect(keys(r.bands)))
        data = Matrix{Any}(undef, n_s, 3 + 2 * length(lv) + 1)
        for k in 1:n_s
            data[k, 1] = r.shock_labels[k]
            data[k, 2] = _fmt(r.delta[k])
            data[k, 3] = _fmt(r.delta_plugin[k])
            for (j, l) in enumerate(lv)
                data[k, 2+2j] = _fmt(r.bands[l][k, 1])
                data[k, 3+2j] = _fmt(r.bands[l][k, 2])
            end
            data[k, end] = join([string(round(Int, 100l)) for l in lv if r.reject[l][k]], "/")
        end
        labels = vcat(["shock direction", "δ (median)", "δ (plug-in)"],
                      reduce(vcat, [["$(round(Int, 100l))% lo", "$(round(Int, 100l))% hi"] for l in lv]),
                      ["rejects at"])
        _pretty_table(io, data; title="OPP δ* with credible bands ($(size(r.delta_draws, 2)) sims, $(r.n_failed) failed)",
                      column_labels=labels, alignment=vcat([:l], fill(:r, size(data, 2) - 2), [:l]))
        println(io, "Note: ", _OPP_POLARITY_NOTE, ".")
    end
    return nothing
end

function report(io::IO, sq::OPPSequence{T}) where {T}
    n_s, n_d = size(sq.delta)
    ok = [t for t in 1:n_d if all(isfinite, @view(sq.delta[:, t]))]
    spec = Any["Loss" sq.loss_name;
               "Dates (valid)" "$n_d ($(length(ok)))";
               "Shock directions" n_s]
    _pretty_table(io, spec; title="OPP Sequence", column_labels=["", ""],
                  alignment=[:l, :r])
    if sq.reject !== nothing && !isempty(ok)
        lv = sort(collect(keys(sq.reject)))
        data = Matrix{Any}(undef, length(lv), 2)
        for (j, l) in enumerate(lv)
            share = count(t -> any(@view(sq.reject[l][:, t])), ok) / length(ok)
            data[j, 1] = "$(round(Int, 100l))%"
            data[j, 2] = _fmt(100 * share) * "%"
        end
        _pretty_table(io, data; title="Share of dates rejecting optimality",
                      column_labels=["band level", "share"], alignment=[:l, :r])
        println(io, "Note: ", _OPP_POLARITY_NOTE, ".")
    end
    if !isempty(ok)
        mags = [(maximum(abs, @view(sq.delta[:, t])), t) for t in ok]
        top = sort(mags; rev=true)[1:min(3, length(mags))]
        data = Matrix{Any}(undef, length(top), 2)
        for (j, (m, t)) in enumerate(top)
            data[j, 1] = sq.dates[t]
            data[j, 2] = _fmt(m)
        end
        _pretty_table(io, data; title="Largest |δ| episodes",
                      column_labels=["date", "max |δ|"], alignment=[:l, :r])
    end
    return nothing
end

function report(io::IO, cm::CounterfactualMoments{T}) where {T}
    n = length(cm.varnames)
    data = Matrix{Any}(undef, n, 4)
    for v in 1:n
        data[v, 1] = string(cm.varnames[v])
        data[v, 2] = _fmt(cm.sd_base[v])
        data[v, 3] = _fmt(cm.sd_cf[v])
        data[v, 4] = _fmt(cm.sd_base[v] > 0 ? cm.sd_cf[v] / cm.sd_base[v] : T(NaN))
    end
    band = cm.freq_band === nothing ? "" :
           " (band ω ∈ [$(_fmt(cm.freq_band[1])), $(_fmt(cm.freq_band[2]))])"
    _pretty_table(io, data; title="Second-Moment Counterfactual \"$(cm.policy_name)\"$band",
                  column_labels=["variable", "sd baseline", "sd counterfactual", "ratio"],
                  alignment=[:l, :r, :r, :r])
    if cm.sd_cf_bands !== nothing
        bd = Matrix{Any}(undef, n, 3)
        for v in 1:n
            bd[v, 1] = string(cm.varnames[v])
            bd[v, 2] = _fmt(cm.sd_cf_bands[v, 1])
            bd[v, 3] = _fmt(cm.sd_cf_bands[v, end])
        end
        _pretty_table(io, bd; title="sd bands",
                      column_labels=["variable", "lower", "upper"],
                      alignment=[:l, :r, :r])
    end
    tail = "VMA tail share (last 10 rows): $(_fmt(100 * cm.tail_share))%"
    println(io, cm.tail_share > T(0.01) ?
                "⚠ $tail — the VMA sum has NOT converged; increase H" : tail)
    return nothing
end

function report(io::IO, sd::SpanningDiagnostic{T}) where {T}
    verdict = sd.spanned ? "data suffice — the model choice does not matter for this counterfactual" :
              "NOT spanned — the counterfactual loads on directions outside the empirical span"
    spec = Any["Verdict" verdict;
               "Loading inside empirical span" _fmt(sd.loading_inside);
               "Thin-solve rel. residual" _fmt(sd.rel_residual_emp)]
    _pretty_table(io, spec; title="Spanning Diagnostic (\"do we need a structural model?\")",
                  column_labels=["", ""], alignment=[:l, :r])
    data = Matrix{Any}(undef, length(sd.outcomes), 3)
    for (i, sym) in enumerate(sd.outcomes)
        data[i, 1] = string(sym)
        data[i, 2] = _fmt(sd.gap[i])
        data[i, 3] = _fmt(sd.gap_rel[i])
    end
    _pretty_table(io, data; title="Empirical-vs-model counterfactual gap",
                  column_labels=["outcome", "max |gap|", "relative L2 gap"],
                  alignment=[:l, :r, :r])
    return nothing
end

function report(io::IO, fs::ForecastSufficiency{T}) where {T}
    spec = Any["Invertible (exact one-step)" (fs.invertible ? "yes" : "no");
               "Max FEV ratio over h ≤ $(fs.H)" _fmt(maximum(fs.fev_ratio))]
    _pretty_table(io, spec; title="Forecast Sufficiency (invertibility laboratory)",
                  column_labels=["", ""], alignment=[:l, :r])
    data = Matrix{Any}(undef, length(fs.observables), 2)
    for (j, sym) in enumerate(fs.observables)
        data[j, 1] = string(sym)
        data[j, 2] = _fmt(fs.one_step_ratio[j])
    end
    _pretty_table(io, data; title="One-step FEV ratios (Wold / full information)",
                  column_labels=["observable", "ratio"], alignment=[:l, :r])
    println(io, "Invertibility is sufficient, not necessary: ratios ≈ 1 are what counterfactuals require.")
    return nothing
end

function report(io::IO, mb::ModelBankMember{T}) where {T}
    spec = Any["Model" mb.name;
               "Kept draws" size(mb.theta_draws, 1);
               "Log marginal likelihood" _fmt(mb.log_marglik);
               "Acceptance rate" _fmt(mb.acceptance_rate);
               "News horizons matched (H_news)" mb.H_news]
    _pretty_table(io, spec; title="Model Bank Member (IRF matching)",
                  column_labels=["", ""], alignment=[:l, :r])
    np = length(mb.param_names)
    data = Matrix{Any}(undef, np, 4)
    for p in 1:np
        v = @view mb.theta_draws[:, p]
        data[p, 1] = string(mb.param_names[p])
        data[p, 2] = _fmt(median(v))
        data[p, 3] = _fmt(quantile(v, 0.16))
        data[p, 4] = _fmt(quantile(v, 0.84))
    end
    _pretty_table(io, data; title="Posterior (limited-information)",
                  column_labels=["parameter", "median", "16%", "84%"],
                  alignment=[:l, :r, :r, :r])
    return nothing
end

"""
    report([io,] members::Vector{ModelBankMember}; prior=uniform)

CMW-style posterior model-probability table over a bank of members.
"""
report(members::AbstractVector{<:ModelBankMember}; kwargs...) =
    report(stdout, members; kwargs...)
function report(io::IO, members::AbstractVector{<:ModelBankMember};
                prior::AbstractVector{<:Real}=fill(1 / length(members), length(members)))
    probs = posterior_model_probs(members; prior=prior)
    data = Matrix{Any}(undef, length(members), 3)
    for (j, m) in enumerate(members)
        data[j, 1] = m.name
        data[j, 2] = _fmt(m.log_marglik)
        data[j, 3] = _fmt(probs[j])
    end
    _pretty_table(io, data; title="Posterior Model Probabilities",
                  column_labels=["model", "log ML", "probability"],
                  alignment=[:l, :r, :r])
    return nothing
end

function report(io::IO, ch::CounterfactualHistory{T}) where {T}
    spec = Any["Policy" ch.policy_name;
               "Window" "$(first(ch.dates)) … $(last(ch.dates)) ($(length(ch.dates)) dates)";
               "Max per-date rel. residual" _fmt(maximum(ch.rel_residual; init=zero(T)));
               "Draws used (failed)" "$(ch.n_draws_used) ($(ch.n_draws_failed))"]
    _pretty_table(io, spec; title="Historical Counterfactual",
                  column_labels=["", ""], alignment=[:l, :r])
    n_v = length(ch.varnames)
    data = Matrix{Any}(undef, n_v, 4)
    for v in 1:n_v
        gapv = ch.cf[:, v] - ch.realized[:, v]
        data[v, 1] = string(ch.varnames[v])
        data[v, 2] = _fmt(sum(ch.realized[:, v]) / size(ch.realized, 1))
        data[v, 3] = _fmt(sum(ch.cf[:, v]) / size(ch.cf, 1))
        data[v, 4] = _fmt(maximum(abs, gapv))
    end
    _pretty_table(io, data; title="Realized vs counterfactual (window means)",
                  column_labels=["variable", "mean realized", "mean counterfactual", "max |gap|"],
                  alignment=[:l, :r, :r, :r])
    return nothing
end
