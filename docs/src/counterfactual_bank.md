# [Model Bank & Diagnostics (Caravello–McKay–Wolf)](@id counterfactual_bank_page)

This page documents the Caravello, McKay & Wolf (2025) strand of the [Policy Counterfactuals](@ref counterfactual_page) module — the answer to "(when) do we need structural models?" Model-implied news menus extrapolate beyond the empirical span; limited-information IRF matching disciplines them with data; posterior model probabilities and pooling make model uncertainty part of the bands; and two diagnostics measure exactly when any of this matters.

- Model menus: [`policy_news_matrix`](@ref) (linear DSGE), [`policy_causal_effects`](@ref) on HA steady states, [`behavioral`](@ref) operators
- [`irf_match`](@ref) limited-information estimation, [`posterior_model_probs`](@ref), [`model_average`](@ref)
- [`counterfactual_history`](@ref)/[`counterfactual_forecast`](@ref) via forecast revisions
- [`spanning_diagnostic`](@ref) and [`forecast_sufficiency`](@ref)

## Quick Start

```@setup cfbank
using MacroEconometricModels, Random, Distributions, LinearAlgebra
Random.seed!(42)
spec = @dsge begin
    parameters: β = 0.99, κ = 0.1, σ = 1.0, φπ = 1.5
    endogenous: π, y, i
    exogenous: eps_i, eps_d
    π[t] = β * π[t+1] + κ * y[t]
    y[t] = y[t+1] - σ * (i[t] - π[t+1]) + eps_d[t]
    i[t] = φπ * π[t] + eps_i[t]
end
H = 12
```

**Recipe 1: a square news menu from a linear DSGE**

```@example cfbank
ce = policy_news_matrix(spec, :eps_i, [:infl => :π, :ygap => :y], [:rate => :i]; H=H)
report(ce)
```

**Recipe 2: a behavioral variant via cognitive discounting**

```@example cfbank
ce_b = behavioral(ce; m=0.85)
round.(ce_b.Theta_x[1][1:3, 3] ./ ce.Theta_x[1][1:3, 3], digits=3)
```

**Recipe 3: HA sequence-space policy effects**

```@example cfbank
ha = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=80)
ss = compute_steady_state(ha)
ce_ha = policy_causal_effects(ha, ss; outcomes=[:cons => :C], H=8, T_horizon=60)
report(ce_ha)
```

---

## Model Menus

**Linear DSGE**: [`policy_news_matrix`](@ref) assembles the full square menu ``\Theta_\nu`` — one column per announcement horizon — by augmenting the spec with a shared news pipeline and solving once. The state dimension grows by ``H - 1``; solver support is linear only (`:gensys`/`:klein`/`:blanchard_kahn`). The baseline closure rule must render the system determinate but is otherwise immaterial for the counterfactuals built from the menu.

**Heterogeneous agents**: [`sequence_jacobian`](@ref) exposes the fake-news Jacobians ``J[t,s] = \partial O_t/\partial I_s`` (anticipation included: entries with ``t < s`` are nonzero), and [`policy_causal_effects`](@ref) on a `ModelSpec` with a `HouseholdSystem` builds rate-wedge menus under two closures — `:administered` (the rate follows the wedge; any one-asset model) and `:market` (`:huggett` only, where zero-net-supply general equilibrium offsets the wedge exactly).

**Behavioral variants**: [`cognitive_discounting`](@ref) (Gabaix: news ``s`` periods ahead down-weighted by ``m^s``) and [`sticky_expectations`](@ref) (per-period update probability ``1-\theta``) act on sequence-space Jacobians; [`behavioral`](@ref) maps whole square containers. Apply them at the *block* level before general-equilibrium closure where possible — applying to a closed menu is an approximation, and thin empirical containers must never be behavioralized (they are data).

---

## Limited-Information IRF Matching

[`irf_match`](@ref) estimates one bank member with the CMW quasi-likelihood: the model-implied target is the menu times a **restricted GLS best-fit news vector** (only the first `H_news` news columns per empirical shock dimension), evaluated against the [`stacked_irf_target`](@ref) with a CTW-damped covariance from [`ctw_covariance`](@ref). Non-diagonal weighting is load-bearing — a diagonal ``V`` makes posterior model probabilities artificially decisive.

```@example cfbank
menu(psi) = policy_news_matrix(
    MacroEconometricModels._respec(spec, merge(spec.param_values, Dict(:κ => psi[1]))),
    :eps_i, [:infl => :π, :ygap => :y]; H=H)
ce0 = menu([0.1])
nu_true = [0.5, -0.2, 0.1]
theta_hat = vcat(ce0.Theta_x[1][:, 1:3] * nu_true, ce0.Theta_x[2][:, 1:3] * nu_true)
V0 = [5e-6 * 0.9^abs(a - b) for a in 1:2H, b in 1:2H]
V = ctw_covariance(V0, H; bandwidth=4).V
index = vcat([(var=:infl, shock=1, h=h) for h in 1:H],
             [(var=:ygap, shock=1, h=h) for h in 1:H])
target = (theta_hat=theta_hat, V_bar=V, index=index)
member = irf_match(menu, target, [truncated(Normal(0.1, 0.05), 0.01, 0.5)], [:kappa];
                   name="RE", H_news=3, n_adapt=200, n_burn=100, n_keep=400,
                   thin=4, proposal_scale=5.66, rng=MersenneTwister(1))
report(member)
```

!!! warning "Rule coefficients are unidentified"
    Rule-free IRF matching cannot identify policy-rule coefficients: the GLS news re-fit absorbs the closure rule exactly (this is rule immateriality at work). Estimate non-policy parameters — slopes, elasticities, behavioral frictions.

```@example cfbank
member_b = irf_match(psi -> behavioral(menu(psi); m=0.3), target,
                     [truncated(Normal(0.1, 0.05), 0.01, 0.5)], [:kappa];
                     name="behavioral", H_news=3, n_adapt=200, n_burn=100,
                     n_keep=400, thin=4, proposal_scale=5.66,
                     rng=MersenneTwister(2))
probs = posterior_model_probs([member, member_b])
report([member, member_b])
```

[`model_average`](@ref) then draws (model, parameter) pairs jointly and stacks the sampled menus into a pooled container (`source = :pooled`) whose draws carry model uncertainty into every downstream band.

---

## Historical and Conditional Counterfactuals

[`counterfactual_history`](@ref) reconstructs the path the economy would have followed under an alternative policy over a historical window — from **forecast revisions only**, never from identified structural shocks. Per date, the revision ``(E_t - E_{t-1})[y_{t+h}]`` is re-solved under the policy and rolled forward; the counterfactual level is realized data minus the accumulated raw revisions plus the accumulated counterfactual ones. Raw forecasts would double-count; only revisions enter the solve. [`counterfactual_forecast`](@ref) is the single-date version, adding the re-solved revision back onto the pre-existing trajectory. This is where **forecast-sufficiency** earns its keep — see the assumptions ladder on the [overview page](@ref counterfactual_page).

```@example cfbank
sol = solve(spec)
T_all = 14
E = zeros(T_all, 2); E[4:12, 2] = randn(MersenneTwister(3), 9)
Y = zeros(T_all, 3)
for t in 1:T_all
    prev = t == 1 ? zeros(3) : Y[t-1, :]
    Y[t, :] = sol.G1 * prev + sol.impact * E[t, :]
end
m = VARModel(Y, 1, vcat(zeros(1, 3), Matrix(sol.G1')), zeros(T_all - 1, 3),
             Matrix(sol.impact * sol.impact' + 1e-12I), 0.0, 0.0, 0.0,
             ["π", "y", "i"])
ce16 = policy_news_matrix(spec, :eps_i, [:infl => :π, :ygap => :y],
                          [:rate => :i]; H=16)
ch = counterfactual_history(m, Y, 4:12, ce16,
                            taylor_rule(16; rho=0.0, phi_pi=3.0, phi_y=0.0,
                                        outcomes=[:infl, :ygap], instruments=[:rate]);
                            outcomes=[:infl => "π", :ygap => "y"],
                            instruments=[:rate => "i"], H=16)
report(ch)
```

---

## Spanning and Forecast Sufficiency

[`spanning_diagnostic`](@ref) runs the same counterfactual with the thin empirical container and the square model menu and quantifies the disagreement — the per-outcome path gaps and the share of the required instrument perturbation inside the empirical span. It answers "does the model choice matter for THIS counterfactual", not "is the model true"; the determinant in practice is the persistence of the object being offset.

```@example cfbank
ce_thin = PolicyCausalEffects(outcomes=[:infl, :ygap], instruments=[:rate],
                              Theta_x=[ce.Theta_x[1][:, 1:2], ce.Theta_x[2][:, 1:2]],
                              Theta_z=[ce.Theta_z[1][:, 1:2]])
base = baseline_path(irf(sol, H), "eps_d", [:infl => "π", :ygap => "y"],
                     [:rate => "i"]; H=H)
sd = spanning_diagnostic(base, ce_thin, ce,
                         policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25]))
report(sd)
```

```@raw html
<iframe src="../assets/plots/cf_spanning.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

[`forecast_sufficiency`](@ref) is the population laboratory for the moments/history assumption: it compares the observables' own innovations-representation forecast-error variances against the full-information filter. Invertibility is sufficient, not necessary — badly non-invertible information sets pass when the observables forecast well.

```@example cfbank
fs = forecast_sufficiency(sol, [:π, :y]; H=24)
report(fs)
```

```@raw html
<iframe src="../assets/plots/cf_sufficiency.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Complete Example

The full CMW loop on the laboratory spec: estimate two bank members, pool them, and use the pooled menu for a rule counterfactual whose bands carry model uncertainty.

```@example cfbank
pooled = model_average([member, member_b], probs; n_pool=60,
                       rng=MersenneTwister(8))
base_p = baseline_path(irf(sol, H), "eps_d", [:infl => "π", :ygap => "y"]; H=H)
pc = policy_counterfactual(base_p, pooled,
                           inflation_target_rule(H; outcomes=[:infl, :ygap],
                                                 instruments=Symbol[]))
report(pc)
```

The pooled container is square, so the strict-targeting rule is enforced exactly draw by draw; the bands reflect both parameter and model uncertainty, weighted by the posterior model probabilities.

---

## Common Pitfalls

1. **Estimating rule coefficients.** Unidentified under rule-free matching — the flat quasi-likelihood is rule immateriality, not a bug.
2. **Diagonal target covariances.** Artificially decisive model probabilities; always pass the draw covariance through [`ctw_covariance`](@ref).
3. **Behavioral operators on thin containers.** Empirically identified causal effects are data; `behavioral` refuses them. Apply the operators to model blocks, ideally before GE closure.
4. **Raw forecasts in historical counterfactuals.** Only *revisions* enter the per-date solve; feeding forecast levels double-counts. The end-to-end linear oracle (`test_oracles.jl`, oracle 7) pins the correct construction.
5. **Two-asset HA menus.** Out of scope — the functions error rather than approximate; one-asset models only.
6. **Marginal-likelihood levels.** The quasi-likelihood includes the Gaussian constant ``-\tfrac{1}{2}\log\det V``; without it, model probabilities are silently wrong across members with different targets.

---

## References

- Caravello, T. E., A. McKay, and C. K. Wolf (2025). "Evaluating Policy Counterfactuals: A VAR-Plus Approach." Working Paper, MIT.
- Auclert, A., B. Bardóczy, M. Rognlie, and L. Straub (2021). "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89(5), 2375–2408. doi:10.3982/ECTA17434
- Auclert, A., M. Rognlie, and L. Straub (2020). "Micro Jumps, Macro Humps: Monetary Policy and Business Cycles in an Estimated HANK Model." NBER Working Paper 26647. doi:10.3386/w26647
- Christiano, L. J., M. Trabandt, and K. Walentin (2010). "DSGE Models for Monetary Policy Analysis." *Handbook of Monetary Economics* 3, 285–367. doi:10.1016/B978-0-444-53238-1.00007-7
- Gabaix, X. (2020). "A Behavioral New Keynesian Model." *American Economic Review* 110(8), 2271–2327. doi:10.1257/aer.20162005
- Geweke, J. (1999). "Using Simulation Methods for Bayesian Econometric Models: Inference, Development, and Communication." *Econometric Reviews* 18(1), 1–73. doi:10.1080/07474939908800428
