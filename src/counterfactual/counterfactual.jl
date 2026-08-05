# Counterfactual module — McKay-Wolf rule counterfactuals (CF-10, #390)
#
# Given baseline paths of (x, z) to a non-policy shock and causal effects of
# policy (news) shocks — all under the PREVAILING rule — construct the paths
# under an alternative rule, Lucas-robustly, via the date-0 policy-shock
# vector nu that enforces the new rule (McKay & Wolf 2023 Prop. 1; CMW A.5).
# All shocks are dated 0: one nu, applied once. Re-solving period-by-period
# with fresh shocks would be Sims-Zha, NOT Lucas-robust — the API deliberately
# has no such path.

# Align rule symbols to container/baseline entries and assemble (M, b).
function _mw_assemble(base::BaselinePath{T}, ce::PolicyCausalEffects{T},
                      rule::PolicyRule{T};
                      Theta_x=ce.Theta_x, Theta_z=ce.Theta_z,
                      x_base=base.x, z_base=base.z) where {T<:AbstractFloat}
    H = ce.H
    n_s = size(Theta_x[1], 2)
    M = zeros(T, H, n_s)
    b = -copy(rule.wedge)
    for (i, sym) in enumerate(rule.outcomes)
        ci = findfirst(==(sym), ce.outcomes)
        bi = findfirst(==(sym), base.outcomes)
        M .+= rule.A_x[i] * Theta_x[ci]
        b .+= rule.A_x[i] * x_base[bi]
    end
    for (k, sym) in enumerate(rule.instruments)
        ck = findfirst(==(sym), ce.instruments)
        bk = findfirst(==(sym), base.instruments)
        M .+= rule.A_z[k] * Theta_z[ck]
        b .+= rule.A_z[k] * z_base[bk]
    end
    return M, b
end

"""
    policy_counterfactual(base::BaselinePath, ce::PolicyCausalEffects,
                          rule::PolicyRule; method=:auto, draws=:auto,
                          baseline_draws=:fixed, quantiles=(0.16, 0.5, 0.84),
                          spanned_tol=0.05) -> PolicyCounterfactual

McKay–Wolf rule counterfactual: find the date-0 policy-shock vector

`ν* = argmin ‖M·ν + b‖`,  `M = Σᵢ A_x[i]·Θ_x[i] + Σₖ A_z[k]·Θ_z[k]`,
`b = Σᵢ A_x[i]·x_base[i] + Σₖ A_z[k]·z_base[k] − wedge`,

and report `x_cf = x_base + Θ_x·ν*`, `z_cf = z_base + Θ_z·ν*` — the paths
under the alternative `rule`, Lucas-robust because only the policy-shock
composition changes. With a **square** container (model news menu, CF-07) the
solve is exact and the rule holds exactly; with a **thin** empirical container
it is a least-squares projection and the implementation-error path is the
honesty signal.

- Rule symbols are matched to `ce`/`base` entries **by symbol**; every
  container variable gets a counterfactual path, but only rule-referenced ones
  enter `(M, b)`.
- `method` is forwarded to the projection kernel (`:auto` picks the exact
  solve for square containers).
- `draws = :auto` propagates uncertainty when `ce` carries draws (`:on`
  forces, `:off` disables). `baseline_draws = :fixed` holds the baseline at
  its point (the MW replication convention when baseline and policy IRFs come
  from different estimations); `:match` pairs baseline draw `d` with container
  draw `d` (correct when both come from the same posterior; requires equal
  draw counts).
- `spanned = rel_residual < spanned_tol` is a reporting convenience; the
  honest object is `rel_residual` itself. An un-enforceable rule triggers a
  warning, never a silent approximation.

Existence caveat: the construction assumes the counterfactual rule induces a
unique equilibrium (MW Assumption 2); a huge `rel_residual` in the *square*
case signals non-existence — it is surfaced, not crashed on.
"""
function policy_counterfactual(base::BaselinePath{T}, ce::PolicyCausalEffects{T},
                               rule::PolicyRule{T};
                               method::Symbol=:auto,
                               draws::Symbol=:auto,
                               baseline_draws::Symbol=:fixed,
                               quantiles::Union{Tuple,AbstractVector}=(0.16, 0.5, 0.84),
                               spanned_tol::Real=0.05) where {T<:AbstractFloat}
    H = ce.H
    base.H == H || throw(ArgumentError(
        "baseline H = $(base.H) does not match the container H = $H"))
    _rule_horizon(rule) == H || throw(ArgumentError(
        "rule H = $(_rule_horizon(rule)) does not match the container H = $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    baseline_draws in (:fixed, :match) || throw(ArgumentError(
        "baseline_draws: expected :fixed or :match, got :$baseline_draws"))
    for sym in rule.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "rule outcome :$sym not found in the container outcomes $(ce.outcomes)"))
        sym in base.outcomes || throw(ArgumentError(
            "rule outcome :$sym not found in the baseline outcomes $(base.outcomes)"))
    end
    for sym in rule.instruments
        sym in ce.instruments || throw(ArgumentError(
            "rule instrument :$sym not found in the container instruments $(ce.instruments)"))
        sym in base.instruments || throw(ArgumentError(
            "rule instrument :$sym not found in the baseline instruments $(base.instruments)"))
    end
    for sym in ce.outcomes
        sym in base.outcomes || throw(ArgumentError(
            "container outcome :$sym has no baseline path in $(base.outcomes)"))
    end
    for sym in ce.instruments
        sym in base.instruments || throw(ArgumentError(
            "container instrument :$sym has no baseline path in $(base.instruments)"))
    end

    # aligned baselines in container order
    xb = [Vector{T}(base.x[findfirst(==(s), base.outcomes)]) for s in ce.outcomes]
    zb = [Vector{T}(base.z[findfirst(==(s), base.instruments)]) for s in ce.instruments]

    # --- point solve ---
    M, b = _mw_assemble(base, ce, rule)
    res = _policy_projection(M, b; method=method)
    nu = res.nu
    x_cf = [xb[i] + ce.Theta_x[i] * nu for i in eachindex(ce.outcomes)]
    z_cf = [zb[k] + ce.Theta_z[k] * nu for k in eachindex(ce.instruments)]
    spanned = res.rel_residual < T(spanned_tol)
    spanned || @warn "the alternative rule \"$(rule.name)\" is not enforceable within the span of the supplied policy shocks (rel_residual = $(round(res.rel_residual, sigdigits=3))); the counterfactual is a best approximation — inspect error_path/rel_residual"

    # --- draw propagation ---
    nd = n_draws(ce)
    use_draws = draws == :on || (draws == :auto && nd > 0)
    use_draws && nd == 0 && throw(ArgumentError(
        "draws = :on requires a draws-bearing container"))
    qlev = collect(T, quantiles)
    x_bands = nothing
    z_bands = nothing
    rr_bands = nothing
    n_used = 0
    n_failed = 0
    if use_draws
        nb = n_draws(base)
        baseline_draws == :match && nb != nd && throw(ArgumentError(
            "baseline_draws = :match requires matching draw counts, got baseline $nb vs container $nd (use :fixed when the two come from different estimations)"))
        baseline_draws == :match && !isempty(ce.instruments) && base.z_draws === nothing &&
            throw(ArgumentError(
                "baseline_draws = :match requires instrument draws on the baseline (z_draws is missing)"))
        xd = [Matrix{T}(undef, H, nd) for _ in ce.outcomes]
        zd = [Matrix{T}(undef, H, nd) for _ in ce.instruments]
        rrd = Vector{T}(undef, nd)
        keep = falses(nd)
        _suppress_warnings() do
            for d in 1:nd
                Tx_d = [@view(ce.Theta_x_draws[i][:, :, d]) for i in eachindex(ce.outcomes)]
                Tz_d = [@view(ce.Theta_z_draws[k][:, :, d]) for k in eachindex(ce.instruments)]
                xb_d = baseline_draws == :fixed ? xb :
                       [Vector{T}(base.x_draws[findfirst(==(s), base.outcomes)][:, d]) for s in ce.outcomes]
                zb_d = baseline_draws == :fixed ? zb :
                       [Vector{T}(base.z_draws[findfirst(==(s), base.instruments)][:, d]) for s in ce.instruments]
                ok = true
                try
                    M_d, b_d = _mw_assemble(base, ce, rule;
                                            Theta_x=Tx_d, Theta_z=Tz_d,
                                            x_base=xb_d, z_base=zb_d)
                    r_d = _policy_projection(Matrix{T}(M_d), b_d; method=method)
                    ok = all(isfinite, r_d.nu)
                    if ok
                        for i in eachindex(ce.outcomes)
                            xd[i][:, d] = xb_d[i] + Tx_d[i] * r_d.nu
                        end
                        for k in eachindex(ce.instruments)
                            zd[k][:, d] = zb_d[k] + Tz_d[k] * r_d.nu
                        end
                        rrd[d] = r_d.rel_residual
                    end
                catch
                    ok = false
                end
                keep[d] = ok
            end
        end
        used = findall(keep)
        n_used = length(used)
        n_failed = nd - n_used
        n_failed > 0 && @warn "policy_counterfactual: $n_failed of $nd draws failed (non-finite or errored solves) and were dropped"
        if n_used > 0
            x_bands = [Matrix{T}([quantile(@view(xd[i][h, used]), q) for h in 1:H, q in qlev])
                       for i in eachindex(ce.outcomes)]
            z_bands = [Matrix{T}([quantile(@view(zd[k][h, used]), q) for h in 1:H, q in qlev])
                       for k in eachindex(ce.instruments)]
            rr_bands = [T(quantile(rrd[used], q)) for q in qlev]
        end
    end

    PolicyCounterfactual{T}(copy(ce.outcomes), copy(ce.instruments),
                            xb, zb, x_cf, z_cf, x_bands, z_bands,
                            nu, copy(ce.shock_labels),
                            res.error_path, res.rel_residual, rr_bands, spanned,
                            rule.name, H, qlev, n_used, n_failed)
end
