# [Optimal Policy Perturbations (Barnichon–Mesters)](@id counterfactual_opp_page)

This page documents the Barnichon & Mesters (2023) strand of the [Policy Counterfactuals](@ref counterfactual_page) module: is announced policy optimal, and if not, by how much should it be adjusted? Two sufficient statistics answer both questions — the baseline forecast of objective **gaps** and the impulse responses of those objectives to identified policy shocks.

- [`PolicyForecast`](@ref) gap containers from package forecasts or external (SEP-style) point + dispersion inputs
- [`opp`](@ref) point statistic and optimality test, [`estimate_opp`](@ref) two-source inference with 60/75/90% bands
- [`constrained_opp`](@ref) for the ZLB and pre-commitment pledges, [`opp_sequence`](@ref) for decision-date histories

## Quick Start

```@setup cfopp
using MacroEconometricModels, Random
Random.seed!(42)
H = 20
rng = MersenneTwister(3)
Tx_u = randn(rng, H, 2)
Tx_pi = randn(rng, H, 2)
noises = 0.08 .* randn(rng, 60)
Dx_u = cat((Tx_u .* (1 + e) for e in noises)...; dims=3)
Dx_pi = cat((Tx_pi .* (1 + e) for e in noises)...; dims=3)
ce = PolicyCausalEffects(outcomes=[:ugap, :infl],
                         Theta_x=[Tx_u, Tx_pi],
                         Theta_x_draws=[Dx_u, Dx_pi],
                         shock_labels=["level", "slope"], source=:bvar)
```

**Recipe 1: the OPP from an external gap forecast**

```@example cfopp
fc = policy_forecast([:ugap, :infl],
                     [collect(range(1.2, 0.0; length=H)), fill(0.6, H)];
                     sd=[fill(0.4, H), fill(0.3, H)], rho=0.9,
                     n_draws=200, rng=MersenneTwister(4), origin="2021Q2")
loss = policy_loss([:ugap, :infl], H; lambda=[1.0, 1.0], beta=0.99)
r = opp(fc, ce, loss)
report(r)
```

**Recipe 2: two-source inference and the optimality test**

```@example cfopp
r2 = estimate_opp(fc, ce, loss; n_sim=400, rng=MersenneTwister(5))
report(r2)
```

**Recipe 3: a ZLB-constrained recommendation**

```@example cfopp
Tz = 0.5 .* randn(MersenneTwister(6), H, 2) .+ 1.0
ce_z = PolicyCausalEffects(outcomes=[:ugap, :infl], instruments=[:rate],
                           Theta_x=[Tx_u, Tx_pi], Theta_z=[Tz],
                           shock_labels=["level", "slope"], source=:bvar)
out = constrained_opp(fc, ce_z, loss, [zlb_constraint(floor=0.0)];
                      instrument_path=[:rate => fill(0.25, H)])
round.(out.result.delta, digits=4)
```

---

## The OPP Statistic

With loss ``L = \tfrac{1}{2}\, E[Y' W Y]`` over stacked objective gaps, the optimal policy perturbation is

```math
\delta^* = -\left(R_y' W R_y\right)^{-1} R_y' W \, E_t Y^0
```

where:
- ``E_t Y^0`` is the variable-major stack of the baseline **gap** forecasts (deviations from target)
- ``R_y`` stacks the objectives' impulse responses to the identified policy shocks (columns = shocks)
- ``\delta^*`` is literally *minus* the WLS regression coefficient of the forecast on the IRFs — dropping the sign worsens the loss

The announced policy is optimal **iff** the gradient ``R_y' W\, E_t Y^0 = 0`` **iff** ``\delta^* = 0``: at the optimum, forecasts are orthogonal to IRFs — a model-free targeting rule. The empirically feasible version is the **subset OPP** (a thin container, e.g. level and slope shock dimensions): a nonzero subset OPP still rejects optimality; it improves but need not reach the full optimum.

!!! warning "Gaps, not levels"
    Forecasts must enter as gaps from target — the [`PolicyForecast`](@ref) container is gap-typed by construction, and [`policy_forecast`](@ref) applies per-outcome targets. Feeding levels makes the OPP drive *levels* to zero. Forecasts must also be conditional on the *baseline* rule: when OPP recommendations are adopted repeatedly, each subsequent forecast is still constructed under the old rule.

The `instrument_path` keyword supplies the announced instrument paths, needed for the *recommendation* ``E_t P^{opt} = E_t P^0 + R_p\,\delta^*`` — the pure optimality *test* needs no instruments at all.

---

## Inference and Band Polarity

[`estimate_opp`](@ref) simulates the OPP over two independent uncertainty sources: IRF estimation uncertainty (container draws) and forecast uncertainty (forecast draws). Sources from different estimations resample independently (`independent=true`, the empirical norm); a joint posterior pairs draws (`independent=false`). The reported `delta` is the per-dimension draw median; `delta_plugin` keeps the plug-in point.

!!! note "Reversed test polarity"
    The bands are 60/75/90% — not 90/95/99. Rejection at a LOWER band level rejects more readily, which is the conservative choice for a policymaker averse to running non-optimal policy. `report()` carries this note; do not re-interpret the bands as conventional significance levels.

```julia
plot_result(r2)                # δ components with the widest band's whiskers
```

---

## Constrained OPP

With constraints — the ZLB on the recommended path, or state-contingent pre-commitment pledges — the OPP has no closed form. [`constrained_opp`](@ref) solves the quadratic program by SLSQP, warm-started at the unconstrained OPP:

- [`zlb_constraint`](@ref) builds the linear floor `P⁰ + R_p·δ ≥ floor`
- [`FunctionConstraint`](@ref) accepts arbitrary `g(δ, paths) ≥ 0` (pledges are products of paths — nonconvex; use `multistart`)
- `method=:projection` is the crude published fallback (zero the wrong-signed components, re-solve) — explicitly labeled and warned as *not* the constrained optimum

The return carries KKT diagnostics: the binding set, the projected-gradient residual, and whether the warm start was already feasible. Infeasible constraint sets error with the offending constraint named.

---

## Sequences and Time Consistency

[`opp_sequence`](@ref) runs the OPP at every decision date — a policy-evaluation history. Revisions decompose exactly into three parts:

```math
\delta_t - \delta_{t-1} = \underbrace{D_t\,(E_t Y - E_{t-1\to t} Y)}_{\text{news}}
 + \underbrace{(D_t - D_{t-1})\, E_{t-1\to t}Y}_{\text{preference}}
 + \underbrace{D_{t-1}\,(E_{t-1\to t}Y - E_{t-1}Y)}_{\text{aging}}
```

where ``E_{t-1\to t}Y`` is the date-``t-1`` forecast advanced one period onto date-``t``'s calendar and ``D = -(R'WR)^{-1}R'W``. The **time-consistent OPP** removes the preference part: ``\delta^{tc} = \delta - \text{pref}``. With constant causal effects and weights the preference part is identically zero. [`opp_sensitivity`](@ref) reruns the sequence over a grid of loss weights, and [`robust_weights`](@ref) extracts the weights that make past policy look most optimal.

```julia
plot_result(sq; view=:fan)     # BM Figure-1 style fans over decision dates
```

```@raw html
<iframe src="../assets/plots/cf_opp_sequence.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Complete Example

An OPP history over eight synthetic decision dates with time-consistency decomposition and preference-weight sensitivity:

```@example cfopp
fcs = [policy_forecast([:ugap, :infl],
                       [randn(MersenneTwister(10 + q), H) .* 0.5,
                        randn(MersenneTwister(30 + q), H) .* 0.4];
                       sd=[fill(0.3, H), fill(0.25, H)], n_draws=60,
                       rng=MersenneTwister(50 + q), origin="date $q")
       for q in 1:8]
sq = opp_sequence(fcs, ce, loss; dates=["$(1999+q)Q4" for q in 1:8],
                  n_sim=150, rng=MersenneTwister(9))
report(sq)
```

```@example cfopp
grid = [0.5, 1.0, 2.0]
seqs = opp_sensitivity(fcs, ce, H; lambda_grid=grid,
                       build_loss=l -> policy_loss([:ugap, :infl], H;
                                                   lambda=[l, 1.0], beta=0.99))
[round(s.delta[1, 1], digits=3) for s in seqs]
```

The sensitivity run shows how the recommended level-shock adjustment at the first date moves with the unemployment weight — the preference-robustness check Barnichon–Mesters run over ``\lambda \in [0.2, 2]``.

---

## Common Pitfalls

1. **Levels instead of gaps.** The single most consequential mistake: the OPP then targets levels of zero. Use [`policy_forecast`](@ref) with explicit `targets`.
2. **Band polarity.** 60/75/90% with rejection-at-lower-levels-is-conservative semantics; do not relabel as 90/95/99.
3. **Degenerate SEP dispersions.** Long-run forecasts pinned at target with zero dispersion make the simulated covariance singular; `min_sd` floors them with a warning.
4. **Square menus.** The full-menu OPP (``n_s = H``) is ill-posed as columns grow; BM design the method around thin shock subsets — select a few identified shock dimensions.
5. **Instrument penalties need announced paths.** A loss with `W_z` (or a smoothing wedge) requires `instrument_path`; the pure test does not.
6. **The OPP cannot separate a bad rule from exogenous mistakes** — it detects and corrects non-optimality without attributing its source.

---

## References

- Barnichon, R., and G. Mesters (2023). "A Sufficient Statistics Approach for Macroeconomic Policy." *American Economic Review* 113(11), 2809–2845. doi:10.1257/aer.20220581
- Gertler, M., and P. Karadi (2015). "Monetary Policy Surprises, Credit Costs, and Economic Activity." *American Economic Journal: Macroeconomics* 7(1), 44–76. doi:10.1257/mac.20130329
- Romer, C. D., and D. H. Romer (2004). "A New Measure of Monetary Shocks: Derivation and Implications." *American Economic Review* 94(4), 1055–1084. doi:10.1257/0002828042002651
