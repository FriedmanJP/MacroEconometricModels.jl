# [Policy Counterfactuals](@id counterfactual_page)

The `counterfactual` module answers the central question of applied monetary policy analysis — *what would the economy have looked like under a different policy?* — using the sufficient-statistics approach of McKay & Wolf (2023), Barnichon & Mesters (2023), and Caravello, McKay & Wolf (2025). The module implements the *methods*, not the papers' numbers: every example is illustrative, and correctness rests on theorem-level identities that hold exactly in linear laboratories (the `test/counterfactual/test_oracles.jl` suite), not on replicating published tables.

All three strands reduce to one weighted projection: choose the policy-shock vector ``\nu`` minimizing ``(M\nu + b)' W (M\nu + b)``, with the method determining what ``M`` (causal effects), ``b`` (baseline), and ``W`` (weights) are. The load-bearing input is a [`PolicyCausalEffects`](@ref) container: for each outcome and instrument, a matrix whose column ``k`` is the impulse response to identified policy shock ``k``.

## Quick Start

```@setup cfhub
using MacroEconometricModels, Random
Random.seed!(42)
spec = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    i[t] = φπ * π[t] + eps_i[t]
end
H = 20
```

**Recipe: a rate-peg counterfactual from a model news menu**

```@example cfhub
ce = policy_news_matrix(spec, :eps_i, [:infl => :π, :ygap => :y], [:rate => :i]; H=H)
base = baseline_path(irf(solve(spec), H), "eps_d",
                     [:infl => "π", :ygap => "y"], [:rate => "i"]; H=H)
pc = policy_counterfactual(base, ce, rate_peg_rule(H))
report(pc)
```

```julia
plot_result(pc)
```

```@raw html
<iframe src="../assets/plots/cf_counterfactual.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Choosing a Method

| You have | You want | Use | Page |
|---|---|---|---|
| Forecast of objective gaps + policy-shock IRFs | "Is announced policy optimal? By how much to adjust?" | [`opp`](@ref) / [`estimate_opp`](@ref) | [Optimal Policy Perturbations](@ref counterfactual_opp_page) |
| Baseline shock IRFs + policy-shock IRFs | Paths/moments under an alternative *rule* | [`policy_counterfactual`](@ref) / [`optimal_policy`](@ref) / [`counterfactual_moments`](@ref) | [Rule Counterfactuals](@ref counterfactual_rules_page) |
| Worry that identified shocks don't span the policy change | Model extrapolation, disciplined by data | [`irf_match`](@ref) / [`model_average`](@ref) / [`spanning_diagnostic`](@ref) | [Model Bank & Diagnostics](@ref counterfactual_bank_page) |

The inputs come from anywhere the package estimates causal effects: VAR/BVAR/LP/sign-set IRFs ([`policy_causal_effects`](@ref)), linear DSGE news menus ([`policy_news_matrix`](@ref)), or heterogeneous-agent sequence-space Jacobians ([`sequence_jacobian`](@ref)).

---

## The Assumptions Ladder

Each additional capability costs one additional assumption. Climb only as far as the application requires:

1. **Level counterfactuals** ([`policy_counterfactual`](@ref), [`opp`](@ref)) need only the causal effects of policy shocks and a baseline — no invertibility, no full identification of the non-policy block. The honesty diagnostic is the **implementation error** ``M\nu^* + b``: a large relative residual means the alternative rule is not enforceable with the available shocks, and the result is a best approximation, reported as such.
2. **Second-moment counterfactuals** ([`counterfactual_moments`](@ref)) additionally assume the Wold innovations span the structural shocks (**invertibility**, or more precisely forecast-sufficiency).
3. **Historical counterfactuals** ([`counterfactual_history`](@ref)) rest on forecast-sufficiency: the observables' forecasts must approximate full-information forecasts. [`forecast_sufficiency`](@ref) measures exactly this in a model laboratory — invertibility is sufficient but *not* necessary.

---

## Child Pages

- [Rule Counterfactuals (McKay–Wolf)](@ref counterfactual_rules_page) — rule templates, `policy_counterfactual`, `optimal_policy`, second moments
- [Optimal Policy Perturbations (Barnichon–Mesters)](@ref counterfactual_opp_page) — the OPP statistic, inference, ZLB constraints, decision-date sequences
- [Model Bank & Diagnostics (Caravello–McKay–Wolf)](@ref counterfactual_bank_page) — model menus, IRF matching, model averaging, historical counterfactuals, spanning and invertibility diagnostics

---

## Honest Limitations

- **Stabilization rules only.** The framework compares rules around a *fixed* steady state. Policies that change the steady state itself (a different inflation-target level, a new operating framework) are out of scope by construction.
- **Lucas-robustness boundary.** Rule counterfactuals are Lucas-robust because only the date-0 policy-shock composition changes; validity requires the rule coefficients not to feed the non-policy block (regime-switching credibility models break this). The *detection* of non-optimality by the OPP is robust more broadly than the *improvement* claim.
- **Approximation quality is measurable.** With thin empirical containers the projection is least squares; `rel_residual` and the [`spanning_diagnostic`](@ref) quantify how much of the desired policy change lives outside the identified span. The module surfaces these numbers everywhere rather than hiding them.
- **Not a replication.** Menus, forecasts, and posteriors are whatever the user estimates; the module contributes projection algebra plus honesty diagnostics, validated by exact linear-world identities.

---

## Saving Results

[`save_model`](@ref) persists the fitted result to a versioned JLD2 file; [`load_model`](@ref) reconstructs it. JLD2 is a package dependency --- no extra `using` is required. Every exported result type on this page is saveable; the living catalog is the [API Reference](@ref api_page) Persistence table. See [Data Management](@ref data_page) for bundles, `note=`, `model_info`, compression, and the reproducibility manifest.

```@example cfhub
path = joinpath(mktempdir(), "counterfactual.jld2")
save_model(pc, path)
pc2 = load_model(path)
typeof(pc2)
```

---

## References

- Barnichon, R., and G. Mesters (2023). "A Sufficient Statistics Approach for Macroeconomic Policy." *American Economic Review* 113(11), 2809–2845. doi:10.1257/aer.20220581
- Caravello, T. E., A. McKay, and C. K. Wolf (2025). "Evaluating Policy Counterfactuals: A VAR-Plus Approach." Working Paper, MIT.
- McKay, A., and C. K. Wolf (2023). "What Can Time-Series Regressions Tell Us About Policy Counterfactuals?" *Econometrica* 91(5), 1695–1725. doi:10.3982/ECTA21045
- Sims, C. A., and T. Zha (1998). "Bayesian Methods for Dynamic Multivariate Models." *International Economic Review* 39(4), 949–968. doi:10.2307/2527347
