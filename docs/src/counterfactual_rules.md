# [Rule Counterfactuals (McKay–Wolf)](@id counterfactual_rules_page)

This page documents the McKay & Wolf (2023) strand of the [Policy Counterfactuals](@ref counterfactual_page) module: given the causal effects of identified policy shocks and a baseline response to a non-policy disturbance — both estimated under the *prevailing* rule — construct the paths the economy would follow under an **alternative rule**, Lucas-robustly, without re-estimating anything.

- Rule templates: peg, rate target, strict inflation/output-gap targeting, NGDP-level targeting, inertial Taylor rules, and a raw-matrix escape hatch
- [`policy_counterfactual`](@ref) for rule counterfactuals, [`optimal_policy`](@ref)/[`optimal_rule`](@ref) for loss minimization
- [`counterfactual_moments`](@ref) for "how volatile would the economy have been" questions

## Quick Start

```@setup cfmw
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
ce = policy_news_matrix(spec, :eps_i, [:infl => :π, :ygap => :y], [:rate => :i]; H=H)
base = baseline_path(irf(solve(spec), H), "eps_d",
                     [:infl => "π", :ygap => "y"], [:rate => "i"]; H=H)
```

**Recipe 1: NGDP-level targeting counterfactual**

```@example cfmw
rule = ngdp_rule(H)
pc = policy_counterfactual(base, ce, rule)
report(pc)
```

**Recipe 2: optimal policy under a dual-mandate loss**

```@example cfmw
loss = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25], beta=0.99)
po = optimal_policy(base, ce, loss)
round.(po.nu[1:4], digits=4)
```

**Recipe 3: the implied optimal targeting rule closes the circle**

```@example cfmw
rule_star = optimal_rule(ce, loss)
pc_star = policy_counterfactual(base, ce, rule_star)
maximum(abs.(pc_star.x_cf[1] - po.x_cf[1])) < 1e-8
```

---

## The Construction

Write the alternative rule as ``\sum_i A_x[i]\, x_i + \sum_k A_z[k]\, z_k = \text{wedge}`` over stacked ``H``-period paths. The enforcing date-0 policy-shock vector is

```math
\nu^* = \arg\min_\nu \, \lVert M\nu + b \rVert^2, \qquad
M = \sum_i A_x[i]\,\Theta_x[i] + \sum_k A_z[k]\,\Theta_z[k], \qquad
b = \sum_i A_x[i]\,x^{base}_i + \sum_k A_z[k]\,z^{base}_k - \text{wedge}
```

where:
- ``\Theta_x[i]`` is the ``H \times n_s`` causal-effect matrix of outcome ``i`` to the identified policy shocks
- ``x^{base}_i`` is the baseline path of outcome ``i`` (the response to the non-policy shock being offset)
- ``\nu^*`` is applied **once**, at date 0 — re-shocking period by period is the Sims–Zha exercise, which is *not* Lucas-robust; the oracle suite (`test_oracles.jl`, oracle 6) demonstrates the difference

With a **square** container (a model news menu, one column per announcement horizon) the solve is exact and the rule holds path-by-path (McKay–Wolf Proposition 1). With a **thin** empirical container it is a least-squares projection, and the implementation-error path ``M\nu^* + b`` is the honesty signal: `report(pc)` prints the enforceability verdict, and the plot appends an error panel whenever `rel_residual` exceeds the threshold.

!!! warning "Rule immateriality"
    The news menu ``\Theta_\nu`` itself depends on the baseline closure rule, but the counterfactuals built from it do not — any determinacy-inducing closure gives identical results (verified to ``10^{-8}`` in the oracle suite). What matters is that the *same* prevailing rule generated both the menu and the baseline.

---

## Rule Templates

All templates return a validated [`PolicyRule`](@ref); rows match [`PolicyCausalEffects`](@ref) entries **by symbol**, never by position.

| Template | Rule | Notes |
|---|---|---|
| [`rate_peg_rule`](@ref) | ``z = 0`` | counterfactual instrument peg |
| [`rate_target_rule`](@ref) | ``z = \text{path}`` | arbitrary instrument path |
| [`inflation_target_rule`](@ref) | ``\pi = 0`` | strict targeting |
| [`output_gap_rule`](@ref) | ``y = 0`` | strict targeting |
| [`ngdp_rule`](@ref) | ``\pi_t + y_t - y_{t-1} = 0`` | level targeting; first row truncates |
| [`taylor_rule`](@ref) | ``z_t = \rho z_{t-1} + (1-\rho)(\phi_\pi \pi_t + \phi_y y_t)`` | pre-sample ``z_0`` enters via the wedge |
| [`custom_rule`](@ref) | user matrices | validated escape hatch |

```@example cfmw
tr = taylor_rule(H; rho=0.85, phi_pi=2.0, phi_y=0.25)
tr
```

---

## Optimal Policy

[`optimal_policy`](@ref) minimizes a quadratic [`PolicyLoss`](@ref) over the reachable set ``\{x^{base} + \Theta_x \nu\}`` (McKay–Wolf Proposition 2). By certainty equivalence the projection on expected paths solves the stochastic LQ problem. Loss builders: [`policy_loss`](@ref) (discounted diagonal blocks), [`ait_loss`](@ref) (average-inflation targeting; the weight matrix is PSD-singular by construction), and [`smoothing_penalty`](@ref) (instrument smoothing with a `z_lag` initial condition that enters the solve as a linear term).

```@example cfmw
sp = smoothing_penalty(H; lambda=0.2, beta=0.99, z_lag=0.25)
loss_s = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25],
                     instruments=[:rate], W_z=[sp.W_z])
po_s = optimal_policy(base, ce, loss_s; z_wedge=[sp.wedge_term])
round(po_s.loss_cf, digits=6) <= round(po_s.loss_base, digits=6)
```

The result carries `loss_base`/`loss_cf` and `foc_norm` — the norm of the McKay–Wolf optimality condition ``\sum_i \Theta_x[i]' W_x[i]\, x^{cf}_i`` ("at the optimum, IRFs are orthogonal to counterfactual paths"), which is ``\approx 0`` at the solution.

---

## Second-Moment Counterfactuals

[`counterfactual_moments`](@ref) re-solves every orthonormalized Wold column under the alternative policy and sums the VMA: ``\Sigma^{cf} = \sum_h \tilde\Theta_h \tilde\Theta_h'``. This **additionally assumes invertibility** (Wold innovations span the structural shocks) — level counterfactuals do not. The orthogonalization is immaterial: any rotation cancels in the sum.

```@example cfmw
w = wold_representation(estimate_var(randn(Random.MersenneTwister(1), 200, 3), 1); H=40)
ce3 = PolicyCausalEffects(outcomes=[:x1, :x2], instruments=[:z],
                          Theta_x=[randn(40, 2), randn(40, 2)], Theta_z=[randn(40, 2)])
cm = counterfactual_moments(w, ce3, rate_peg_rule(40; outcomes=[:x1, :x2], instruments=[:z]);
                            outcomes=[:x1 => 1, :x2 => 2], instruments=[:z => 3],
                            warn_invertibility=false)
report(cm)
```

The `tail_share` diagnostic warns when the VMA sum has not converged (persistent counterfactual rules push mass into the tail — increase `H`). Band-limited variances are available via `frequencies=:business_cycle` (``\omega \in [2\pi/32, 2\pi/6]``).

```julia
plot_result(cm)
```

```@raw html
<iframe src="../assets/plots/cf_moments.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Complete Example

A rule counterfactual on US data: identify a monetary shock in a small VAR on `load_example(:mp_shocks)`, extract the causal effects with uncertainty, and ask how the demand-driven component of the early 1980s would have evolved under NGDP targeting.

```@setup cfmw2
using MacroEconometricModels, Random
Random.seed!(7)
```

```@example cfmw2
td = load_example(:mp_shocks)
names = td.varnames
rows = findall(t -> all(isfinite, td.data[t, 1:3]), 1:size(td.data, 1))
Y = td.data[rows, 1:3]                       # ygap, infl, ffr (1969Q1–2007Q4)
m = estimate_var(Y, 4)
ir = irf(m, 24; ci_type=:bootstrap, reps=50, rng=MersenneTwister(2))
Hc = 20
ce = policy_causal_effects(ir, [3], [:ygap => 1, :infl => 2], [:ffr => 3]; H=Hc)
base = baseline_path(ir, 1, [:ygap => 1, :infl => 2], [:ffr => 3]; H=Hc)
pc = policy_counterfactual(base, ce, ngdp_rule(Hc; pi_var=:infl, y_var=:ygap,
                                               outcomes=[:ygap, :infl],
                                               instruments=[:ffr]))
report(pc)
```

A single identified shock cannot enforce an ``H``-period rule exactly — the report's relative residual quantifies how far the projection falls short, which is exactly the McKay–Wolf spanning discussion: transitory rule changes are nearly enforceable, persistent ones are not. The [`spanning_diagnostic`](@ref) on the [Model Bank & Diagnostics](@ref counterfactual_bank_page) page turns this observation into a formal comparison against a model-extrapolated menu.

---

## Common Pitfalls

1. **Solve horizon vs. reporting horizon.** The container's `H` must exceed the horizon you plot: truncation-edge bias contaminates the last columns of news menus. McKay–Wolf use `H = 100` and report 40; the adapters error when `H` exceeds the stored IRF horizon.
2. **Thin-container columns are not horizon-ordered.** Columns of an empirical container are whatever shocks were identified — never assume column ``k`` is a horizon-``k`` news shock.
3. **`baseline_draws` semantics.** `:fixed` (default) holds the baseline at its point — correct when the baseline and policy IRFs come from different estimations. `:match` pairs draw ``d`` with draw ``d`` — correct only for a joint posterior, and errors on mismatched counts.
4. **One `ν`, applied once.** Period-by-period re-shocking (Sims–Zha) enforces a rule ex post but answers a different, non-Lucas-robust question; the API deliberately provides no such path.
5. **Invertibility is a moments-only assumption.** `policy_counterfactual` never needs it; `counterfactual_moments` does — do not carry the assumption where it is not required.

---

## References

- McKay, A., and C. K. Wolf (2023). "What Can Time-Series Regressions Tell Us About Policy Counterfactuals?" *Econometrica* 91(5), 1695–1725. doi:10.3982/ECTA21045
- Caravello, T. E., A. McKay, and C. K. Wolf (2025). "Evaluating Policy Counterfactuals: A VAR-Plus Approach." Working Paper, MIT.
- Plagborg-Møller, M., and C. K. Wolf (2021). "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica* 89(2), 955–980. doi:10.3982/ECTA17813
