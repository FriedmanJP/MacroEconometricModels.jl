# [Rational Inattention](@id rational_inattention_page)

**Rational inattention (RI)** models agents who face finite Shannon mutual-information processing constraints or marginal information costs when observing economic fundamentals. Rather than assuming exogenous observation noise or fixed adjustment hazards, RI derives optimal signal structures, endogenous Kalman gains, sticky-expectations impulse responses, consideration sets, and choice probabilities from first principles across macroeconomic and discrete-choice environments. This module integrates steady-state and dynamic linear-quadratic-Gaussian (LQG) attention filters (Sims, 2003; Luo, 2008; Afrouzi & Yang, 2021), joint state-space embeddings connecting directly to [State-Space Models](@ref statespace_page), structural MLE/Bayesian estimation and Coibion–Gorodnichenko forecast-revision diagnostics (Coibion & Gorodnichenko, 2015), multi-block Maćkowiak–Wiederholt equilibrium pricing (Maćkowiak & Wiederholt, 2009), sequence-space RI Jacobians for [Policy Counterfactuals](@ref counterfactual_page), and Matějka–McKay generalized multinomial logit discrete choice alongside [Ordered & Multinomial Choice](@ref ordered_multinomial_page).

## Quick Start

Specify a linear state transition, shock loading, and action-tracking rule in `RationalInattentionProblem`, solve for the optimal Shannon channel with `solve_ri_kalman`, simulate noisy beliefs, compute impulse responses or variance decompositions, or solve discrete-choice attention problems with `solve_ri_discrete`.

```@setup ri
using MacroEconometricModels, LinearAlgebra, Random, Statistics

rng_ri = Random.Xoshiro(92301)

# Scalar Luo (2008) / Sims (2003) persistent tracking problem
A_qs = reshape([0.9], 1, 1)
B_qs = reshape([1.0], 1, 1)
H_qs = reshape([1.0], 1, 1)
prob_qs = RationalInattentionProblem(
    A_qs, B_qs, H_qs;
    capacity=0.8,
    units=:bits,
    state_names=["Permanent Income"],
    action_names=["Consumption"],
    shock_names=["Income Shock"],
)
```

Solve the steady-state LQG rational inattention problem and inspect the resulting endogenous capacity shadow cost, signal structure, and mutual information:

```@example ri
sol_qs = solve_ri_kalman(prob_qs)
sol_qs
```

Compute inertial action impulse responses and variance shares decomposing fundamental shocks from endogenous rational-inattention noise:

```@example ri
ir_qs = irf(sol_qs, 12; observables=:actions)
fv_qs = fevd(sol_qs, 12; observables=:actions)
(gain=(sol_qs.K_gain * sol_qs.Y_signal')[1, 1],
 irf_h1=ir_qs.values[1, 1, 1],
 noise_share_h1=fv_qs.proportions[1, 2, 1])
```

Solve a discrete-choice rational inattention problem (Matějka & McKay, 2015) with three alternatives across three payoff states:

```@example ri
U_qs = [2.0 0.2 1.0;
        0.3 2.1 0.8;
       -4.0 -4.0 -4.0]
mu_qs = [0.40, 0.35, 0.25]
dc_qs = solve_ri_discrete(U_qs, mu_qs; omega=0.45)
dc_qs
```

Visualize the optimal signal, impulse responses across capacities, variance decomposition, and discrete-choice probability curves using interactive D3 plots:

```julia
plot_result(prob_qs)
plot_result(sol_qs; view=:irf_by_capacity, capacities=[0.25, 0.8, Inf], horizon=20)
plot_result(sol_qs; view=:fevd, horizon=20)
plot_result(sol_qs; view=:water_filling)
plot_result(dc_qs; view=:choice_curves)
```

---

## Steady-State LQG Rational Inattention

### Problem Specification and Shannon Channel

In the discounted LQG tracking problem of Sims (2003, 2010), an $n_x$-dimensional state vector evolves under Gaussian fundamental innovations while the agent chooses $n_u$-dimensional actions $u_t = \mathbb{E}_t[H' x_t]$ subject to a quadratic tracking loss and an information flow limit:

```math
x_{t+1} = A x_t + B_u u_t + B w_{t+1}, \qquad w_{t+1} \sim \mathcal{N}(0, I_{n_w}),
```

where:
- $x_t \in \mathbb{R}^{n_x}$ is the unobserved economic state vector,
- $u_t \in \mathbb{R}^{n_u}$ is the vector of agent control actions,
- $A \in \mathbb{R}^{n_x \times n_x}$ is the state transition matrix,
- $B_u \in \mathbb{R}^{n_x \times n_u}$ maps control actions back into state dynamics (zero by default when `B_u=nothing`),
- $B \in \mathbb{R}^{n_x \times n_w}$ is the fundamental innovation impact matrix,
- $w_{t+1} \in \mathbb{R}^{n_w}$ is an i.i.d. standard Gaussian shock vector.

Under a symmetric positive-semidefinite loss matrix $M \succeq 0$ and tracking matrix $H \in \mathbb{R}^{n_x \times n_u}$, the discounted objective penalizes filtering error $\mathbb{E}[\sum_{t=0}^\infty \beta^t (x_t - \hat{x}_{t|t})' H M H' (x_t - \hat{x}_{t|t})]$ plus the information shadow cost $\omega \, I(x_t; s_t \mid s^{t-1})$ (or constrains capacity $I(x_t; s_t \mid s^{t-1}) \le \kappa$). Mutual information in nats satisfies:

```math
\mathcal{I}(x_t; s_t \mid s^{t-1}) = \frac{1}{2} \log \frac{|\Sigma_{\text{prior}}|}{|\Sigma_{\text{post}}|}, \qquad \kappa_{\text{bits}} = \frac{\mathcal{I}}{\ln 2},
```

where:
- $\Sigma_{\text{prior}} = \mathbb{E}[(x_t - \hat{x}_{t|t-1})(x_t - \hat{x}_{t|t-1})']$ is the one-step-ahead prior covariance before signal realization,
- $\Sigma_{\text{post}} = \mathbb{E}[(x_t - \hat{x}_{t|t})(x_t - \hat{x}_{t|t})']$ is the posterior error covariance after observing $s_t$,
- $\kappa_{\text{bits}}$ is channel capacity expressed in bits (`units=:bits`, default) when divided by $\ln 2$.

`RationalInattentionProblem(A, B, H; M=nothing, B_u=nothing, beta=1.0, capacity=nothing, omega=nothing, units=:bits, state_names=nothing, action_names=nothing, shock_names=nothing)` validates all dimensions and requires exactly one of `capacity` (bits or nats) or marginal shadow price `omega`.

### Water-Filling and Log-Determinant Barrier Solvers

At the steady state, `solve_ri_kalman(prob; method=:water_filling, tol=1e-11, max_iter=2000)` solves the coupled discrete algebraic Riccati prior update $\Sigma_{\text{prior}} = A \Sigma_{\text{post}} A' + B B'$ and Shannon water-filling optimality condition (Miao, Wu & Young, 2022). In whitened prior coordinates $\tilde{v}_i$ where $\Sigma_{\text{prior}}^{1/2} \Omega_{\text{benefit}} \Sigma_{\text{prior}}^{1/2} = Q \operatorname{diag}(d_1, \dots, d_{n_x}) Q'$, each mode $i$ receives positive attention if and only if twice its benefit eigenvalue exceeds the shadow price $\omega$:

```math
d_{\text{post}, i} = \min\!\left(1, \frac{\omega}{2 d_i}\right), \qquad \operatorname{rank}(Y) = \#\!\left\{i : 2 d_i > \omega\right\},
```

where:
- $d_i \ge 0$ is the $i$-th eigenvalue of the whitened marginal value of information matrix $\Sigma_{\text{prior}}^{1/2} \Omega_{\text{benefit}} \Sigma_{\text{prior}}^{1/2}$,
- $\omega > 0$ is the Lagrange multiplier (shadow cost per nat of mutual information),
- $d_{\text{post}, i} \in (0, 1]$ is the posterior-to-prior variance ratio along mode $i$,
- $\operatorname{rank}(Y)$ is the endogenous signal dimension (`signal_rank`) of the optimal channel $s_t = Y' x_t + \psi_t$.

Passing `method=:logdet_barrier` solves the same fixed point using a smooth log-determinant barrier continuation scheme, providing an independent verification check for ill-conditioned multivariate benefit matrices. For the scalar tracking case (`n_x = 1`), the closed-form Sims (2003) and Luo (2008) steady state yields exact gain $K_{\text{eff}} = 1 - 2^{-2\kappa}$ and posterior variance $\Sigma_{\text{post}} = \sigma_w^2 / (2^{2\kappa} - \rho^2)$:

```@example ri
sigma_w_luo = 1.4
rho_luo = 0.9
kappa_luo = 0.8
prob_luo = RationalInattentionProblem(
    reshape([rho_luo], 1, 1), reshape([sigma_w_luo], 1, 1), reshape([1.0], 1, 1);
    capacity=kappa_luo, units=:bits,
)
sol_wf = solve_ri_kalman(prob_luo; method=:water_filling)
sol_lb = solve_ri_kalman(prob_luo; method=:logdet_barrier)

K_exact = 1.0 - 2.0^(-2.0 * kappa_luo)
K_num = (sol_wf.K_gain * sol_wf.Y_signal')[1, 1]
sig_post_exact = sigma_w_luo^2 / (2.0^(2.0 * kappa_luo) - rho_luo^2)
(K_exact=K_exact, K_wf=K_num, sig_post_exact=sig_post_exact,
 sig_post_wf=sol_wf.Sigma_post[1, 1],
 solver_gap=abs(sol_wf.Sigma_post[1, 1] - sol_lb.Sigma_post[1, 1]))
```

Notice that `sol_wf` and `sol_lb` agree to machine precision and reproduce the exact Luo (2008) scalar filter gain `K_exact = 0.6701` (`sig_post_exact = 0.8823`) at `0.8` bits of capacity.

### Joint State-Space Embedding, IRFs, FEVD, and Simulation

Once solved, `to_statespace(sol; observables=:all)` and `to_abcd(sol; observables=:actions)` construct the exact augmented linear system over joint state $z_t = [x_t'; \hat{x}_{t|t}']'$ driven by both fundamental shocks $w_t$ and endogenous signal noise $\psi_t \sim \mathcal{N}(0, \Sigma_\psi)$:

```math
\hat{x}_{t|t} = (I - K Y') A \hat{x}_{t-1|t-1} + (I - K Y') B_u u_{t-1} + K Y' x_t + K \psi_t,
```

where:
- $\hat{x}_{t|t} \in \mathbb{R}^{n_x}$ is the agent's posterior expectation of the state at time $t$,
- $K \in \mathbb{R}^{n_x \times k}$ is the optimal Kalman gain matrix `sol.K_gain`,
- $Y \in \mathbb{R}^{n_x \times k}$ is the optimal signal loading matrix `sol.Y_signal`,
- $\psi_t \sim \mathcal{N}(0, \Sigma_\psi)$ is the orthogonal rational-inattention noise shock vector.

This joint representation immediately unlocks multivariate structural `irf`, `fevd`, `simulate`, and `SteadyStateKalman` diagnostics in [State-Space Models](@ref statespace_page):

```@example ri
A_2 = [0.85 0.10; 0.00 0.70]
B_2 = [1.0 0.0; 0.0 0.7]
H_2 = [1.0 0.25; 0.35 0.90]
prob_2 = RationalInattentionProblem(
    A_2, B_2, H_2;
    capacity=1.2,
    state_names=["Output", "Cost"],
    action_names=["Price", "Wage"],
    shock_names=["Demand", "Supply"],
)
sol_2 = solve_ri_kalman(prob_2)
ss_2 = to_statespace(sol_2; observables=:actions)
ssk_2 = steady_state_kalman(ss_2)
sim_2 = simulate(sol_2, 200; burn=50, rng=rng_ri)
(obs_dim=size(ss_2.Z), eigs=abs.(ssk_2.closed_loop_eigs),
 action_cov=cov(sim_2.actions), signal_rank=sol_2.signal_rank)
```

For plotting, inspect the problem matrices, closed-loop poles, capacity-indexed impulse responses, fundamental-versus-RI-noise FEVD shares, and whitened water-filling spectrum with static `plot_result` calls:

```julia
plot_result(prob_2)
plot_result(ssk_2)
plot_result(sol_2; view=:irf_by_capacity, capacities=[0.3, 1.2, Inf], horizon=20)
plot_result(sol_2; view=:fevd, horizon=20)
plot_result(sol_2; view=:water_filling)
```

---

## Dynamic Transition Paths

### Afrouzi–Yang Backward–Forward Transition Solver

When prior uncertainty starts away from its stationary fixed point—such as after a structural variance break, monetary regime shift, or discrete policy announcement—rational agents solve a finite-horizon time-varying attention problem (Afrouzi & Yang, 2021). Given an initial prior covariance $\Sigma_{0|-1}$ and optional time-varying paths `omega_path` or `capacity_path`, `solve_ri_transition(prob, Sigma_init; T_path=100, tol=1e-10, max_iter=500, damping=0.7)` iterates backward on the marginal benefit of information $\Xi_t$ and forward on the state covariance $\Sigma_{t|t-1}$:

```math
\Sigma_{t+1|t} = A \Sigma_{t|t} A' + B B', \qquad \Sigma_{t|t} = \Sigma_{t|t-1} - \Sigma_{t|t-1} Y_t (Y_t' \Sigma_{t|t-1} Y_t + \Sigma_{\psi,t})^{-1} Y_t' \Sigma_{t|t-1},
```

where:
- $\Sigma_{t|t-1}$ is the date-$t$ prior error covariance initialized at `Sigma_init`,
- $\Sigma_{t|t}$ is the date-$t$ posterior error covariance along the transition path,
- $Y_t$ and $\Sigma_{\psi,t}$ are the time-$t$ water-filling signal loading and noise covariance matrices.

From either low initial uncertainty (`0.2 * sol_ss.Sigma_prior`) or elevated post-crisis uncertainty (`4.0 * sol_ss.Sigma_prior`), the transition path converges smoothly to the `SimsKalmanSolution` steady state:

```@example ri
prob_dyn = RationalInattentionProblem(A_2, B_2, H_2; omega=0.45)
sol_dyn_ss = solve_ri_kalman(prob_dyn)
dyn = solve_ri_transition(prob_dyn, 4.0 * sol_dyn_ss.Sigma_prior; T_path=80, tol=1e-9)
(converged=dyn.converged,
 cap_t1=dyn.capacity_bits_path[1],
 cap_end=dyn.capacity_bits_path[end],
 cap_ss=sol_dyn_ss.capacity_bits,
 prior_gap=norm(dyn.Sigma_prior_path[:, :, end] - sol_dyn_ss.Sigma_prior))
```

Notice how elevated initial uncertainty drives attention up sharply at `t = 1` (`cap_t1 = 3.903` bits) before decaying toward the stationary allocation `cap_ss = 1.903` bits (`prior_gap < 1e-6`). Inspect both the capacity/uncertainty path (`:attention_path`) and the shadow-cost/signal-rank trajectory (`:omega`):

```julia
plot_result(dyn; view=:attention_path)
plot_result(dyn; view=:omega)
```

---

## Empirical Estimation and Forecast-Revision Tests

### Structural MLE and Bayesian Estimation

Given observed actions or macro series $y_{1:T}$, `estimate_ri(build_or_prob, data; method=:mle, estimate_param=:capacity, ...)` evaluates the exact log-likelihood of the augmented state-space representation `to_statespace(sol)` via steady-state Kalman innovation prediction errors, estimating either fixed `capacity` ($\kappa$) or shadow cost ($\omega$) alongside optional structural model parameters:

```math
\ell(\theta; y_{1:T}) = -\frac{1}{2} \sum_{t=1}^{T} \left[ n_y \ln(2\pi) + \ln |F_t(\theta)| + v_t(\theta)' F_t(\theta)^{-1} v_t(\theta) \right],
```

where:
- $\theta$ contains the attention parameter (`capacity` or `omega`) and any structural parameters specified via `extra_params_init`,
- $v_t(\theta) = y_t - \hat{y}_{t|t-1}(\theta)$ is the date-$t$ Kalman filter innovation vector,
- $F_t(\theta) = Z P_{t|t-1}(\theta) Z' + R$ is the innovation covariance matrix from `to_statespace(sol(theta))`.

`RIEstimationResult` reports parameter estimates, Hessian or sandwich standard errors, likelihood ratio tests against full information (`K \to I`), and information criteria (`aic`, `bic`). Passing `method=:bayes` executes random-walk Metropolis–Hastings with user priors and retains posterior parameter `draws`:

```@example ri
sim_est = simulate(sol_wf, 180; burn=50, rng=rng_ri)
res_mle = estimate_ri(prob_luo, sim_est.actions; method=:mle, estimate_param=:capacity)
res_mle
```

### Coibion–Gorodnichenko Forecast-Error Regression

Coibion & Gorodnichenko (2015) test noisy-information rigidities without specifying structural likelihoods by regressing *ex post* forecast errors on *ex ante* forecast revisions:

```math
x_{t+h} - F_t x_{t+h} = \alpha + \beta_{\text{CG}} \left(F_t x_{t+h} - F_{t-1} x_{t+h}\right) + \varepsilon_{t+h}, \qquad \beta_{\text{CG}} = \frac{1 - K}{K},
```

where:
- $F_t x_{t+h}$ is the mean or individual forecast made at date $t$ for horizon $h$,
- $\beta_{\text{CG}} > 0$ captures forecast under-reaction due to noisy updating ($K < 1$),
- $K = (1 + \beta_{\text{CG}})^{-1}$ is the implied scalar Kalman gain recovered by `coibion_gorodnichenko_test`,
- $\varepsilon_{t+h}$ is the rational-expectations error evaluated with Newey–West HAC standard errors (`hac_lags=:auto`).

The helper `implied_cg_slope(sol; horizon=1)` maps any solved multivariate `SimsKalmanSolution` directly to its model-implied theoretical $\beta_{\text{CG}}$ slope, allowing direct comparison between estimated survey regressions and theoretical LQG models:

```@example ri
beta_theory = implied_cg_slope(sol_wf; horizon=1)
T_cg = 160
noise_cg = randn(rng_ri, T_cg)
f_now_cg = sim_est.beliefs[1:T_cg, 1]
f_lag_cg = vcat(0.0, f_now_cg[1:end-1])
realized_cg = f_now_cg .+ beta_theory .* (f_now_cg .- f_lag_cg) .+ 0.3 .* noise_cg
cg_res = coibion_gorodnichenko_test(realized_cg, f_now_cg, f_lag_cg)
(beta_theory=beta_theory, beta_hat=cg_res.beta_cg,
 implied_gain=cg_res.implied_gain, implied_cap=cg_res.implied_capacity)
```

Plot both structural parameter estimates and Coibion–Gorodnichenko regression coefficients with confidence bands:

```julia
plot_result(res_mle; conf_level=0.95)
plot_result(cg_res; conf_level=0.95)
```

---

## Equilibrium Pricing and Sequence-Space Jacobians

### Maćkowiak–Wiederholt Multi-Block Attention

Maćkowiak & Wiederholt (2009) study price-setting firms that allocate a joint capacity budget $\kappa$ between tracking persistent aggregate nominal demand shocks (`agg_block`) and volatile idiosyncratic cost conditions (`idio_block`):

```math
\max_{\kappa_{\text{agg}} + \kappa_{\text{idio}} \le \kappa} \; -L_{\text{agg}}(\kappa_{\text{agg}}) - L_{\text{idio}}(\kappa_{\text{idio}}) \quad \text{subject to} \quad p_{i,t}^* = (1 - \xi) q_t + \xi p_t + z_{i,t},
```

where:
- $\kappa_{\text{agg}}, \kappa_{\text{idio}} \ge 0$ are the optimal attention allocations solved by equalizing marginal value $\omega$ across both blocks,
- $\xi \in [0, 1)$ is the `strategic_complementarity` parameter dampening aggregate price adjustment,
- $p_{i,t}^*$ is the firm's frictionless profit-maximizing price given aggregate nominal spending $q_t$, aggregate price level $p_t$, and idiosyncratic shock $z_{i,t}$,
- $L_{\text{agg}}, L_{\text{idio}}$ are the steady-state tracking losses returned by `solve_ri_kalman`.

Calling `solve_mw_attention(agg_block, idio_block; capacity=..., strategic_complementarity=0.85, horizon=40)` solves the equilibrium fixed point and generates aggregate price, idiosyncratic price, and hump-shaped real output impulse responses:

```@example ri
agg_blk = RationalInattentionProblem(reshape([0.92], 1, 1), reshape([1.0], 1, 1), reshape([1.0], 1, 1); omega=0.2)
idio_blk = RationalInattentionProblem(reshape([0.65], 1, 1), reshape([2.0], 1, 1), reshape([1.0], 1, 1); omega=0.2)
mw_res = solve_mw_attention(agg_blk, idio_blk; capacity=1.5, strategic_complementarity=0.85, horizon=30)
mw_res
```

Because idiosyncratic shocks have higher instantaneous variance (`B = 2.0`), firms devote most capacity (`kappa_idio`) to idiosyncratic cost tracking and allocate smaller attention (`kappa_agg`) to aggregate demand, generating sluggish aggregate price adjustment and persistent real output effects. Plot all three IRFs side by side with:

```julia
plot_result(mw_res)
```

### Sequence-Space RI Jacobian for HANK Counterfactuals

To embed rational inattention directly into heterogeneous-agent sequence-space Jacobians from [Policy Counterfactuals](@ref counterfactual_page), `rational_inattention_jacobian(J, val; mode=:gain, rho_news=0.8)` applies the sticky-expectations / Sims–Kalman cognitive discounting operator $W(K, \rho_{\text{news}})$ to any full-information behavioral Jacobian $J$:

```math
J_{\text{RI}} = W(K, \rho_{\text{news}}) \, J, \qquad K = \begin{cases} \text{val} & \text{if } \texttt{mode = :gain}, \\ 1 - 2^{-2\kappa} & \text{if } \texttt{mode = :capacity}. \end{cases}
```

where:
- $J \in \mathbb{R}^{H \times H}$ is the full-information sequence-space Jacobian with respect to future shocks or policy announcements,
- $W(K, \rho_{\text{news}})$ is the lower-triangular attention attenuation matrix combining contemporaneous Kalman gain $K$ and news-recognition persistence $\rho_{\text{news}}$,
- $J_{\text{RI}}$ is the rationally inattentive behavioral Jacobian ready for `general_equilibrium_counterfactual` or HANK policy experiments.

```@example ri
H_jac = 6
J_full = [0.6^abs(t - s) for t in 1:H_jac, s in 1:H_jac]
J_ri = rational_inattention_jacobian(J_full, 0.8; mode=:capacity, rho_news=0.8)
(full_impact=J_full[1, 1], ri_impact=J_ri[1, 1], full_anticipation=J_full[1, 4], ri_anticipation=J_ri[1, 4])
```

---

## Discrete-Choice Rational Inattention

### Matějka–McKay Optimal Choice Rule and IIA Diagnostics

In Matějka & McKay (2015), a decision maker chooses among $M$ discrete alternatives across $S$ payoff states with utility matrix $U \in \mathbb{R}^{M \times S}$ and prior $\mu \in \Delta^{S-1}$, maximizing expected utility net of mutual information cost $\omega \, \mathcal{I}(j; s)$ (or subject to capacity $\kappa$). By Theorem 1, optimal state-contingent choice probabilities follow an asymmetric generalized multinomial logit shifted by optimal unconditional choice shares $P(j)$:

```math
P(j \mid s) = \frac{P(j) \exp(U_{j,s} / \omega)}{\sum_{k=1}^M P(k) \exp(U_{k,s} / \omega)}, \qquad P(j) = \sum_{s=1}^S \mu_s P(j \mid s),
```

where:
- $U_{j,s}$ is the payoff to choosing alternative $j \in \{1, \dots, M\}$ in state $s \in \{1, \dots, S\}$,
- $\mu_s \ge 0$ is the prior probability of state $s$ with $\sum_{s=1}^S \mu_s = 1$,
- $\omega > 0$ is the marginal cost of information in natural units (nats),
- $P(j) \in [0, 1]$ is the endogenous unconditional choice probability determining the Caplin, Dean & Leahy (2019) consideration set $\{j : P(j) > 0\}$ via the KKT multiplier $\zeta_j = \sum_s \mu_s \frac{\exp(U_{j,s}/\omega)}{\sum_k P(k)\exp(U_{k,s}/\omega)} \le 1$.

When all alternatives are ex-ante symmetric under a uniform prior, $P(j) = 1/M$ and the rule reduces exactly to standard Multinomial Logit (Proposition 1). Conversely, adding a duplicate alternative ("Red Bus / Blue Bus") preserves the non-duplicate share $P(\text{Car})$ and splits the bus share equally—violating Independence of Irrelevant Alternatives (IIA), which `ri_iia_diagnostic` quantifies:

```@example ri
U_bb = [1.0 0.0; 0.0 1.0]
mu_bb = [0.5, 0.5]
diag_bb = ri_iia_diagnostic(U_bb, mu_bb, [0.0, 1.0]; omega=0.25, units=:nats)
(p_base=diag_bb.base_solution.uncond_probs,
 p_exp=diag_bb.expanded_solution.uncond_probs,
 odds_shift_car_red=diag_bb.uncond_odds_ratio_shift[1, 2],
 max_iia_violation=diag_bb.max_iia_violation)
```

### Rational Inattention Logit Estimation

For empirical discrete-choice data across environments $e \in \{1, \dots, E\}$ with observable menu attributes $X_{j,s,k,e}$ (or pre-computed base payoffs `base_payoff`), `estimate_ri_logit` maximizes the Matějka–McKay sample log-likelihood jointly over preference weights $\beta$ and information cost $\omega$, returning an `RILogitModel` with likelihood-ratio tests comparing against nested standard MNL (see [Ordered & Multinomial Choice](@ref ordered_multinomial_page)):

```@example ri
M_e, S_e, K_e, E_e = 3, 2, 2, 4
beta_e = [1.2, -0.8]
omega_e = 0.5
menu_e = randn(rng_ri, M_e, S_e, K_e, E_e)
priors_e = fill(0.5, S_e, E_e)
env_id_e = repeat(1:E_e, inner=60)
state_id_e = repeat(1:S_e, outer=120)
choices_e = Vector{Int}(undef, length(env_id_e))
for e in 1:E_e
    U_e = [sum(beta_e[k] * menu_e[j, s, k, e] for k in 1:K_e) for j in 1:M_e, s in 1:S_e]
    sol_e = solve_ri_discrete(U_e, priors_e[:, e]; omega=omega_e, units=:nats)
    for idx in findall(==(e), env_id_e)
        u_draw = rand(rng_ri)
        cdf_vec = cumsum(sol_e.cond_probs[:, state_id_e[idx]])
        hit = findfirst(>=(u_draw), cdf_vec)
        choices_e[idx] = hit === nothing ? M_e : hit
    end
end

ri_logit = estimate_ri_logit(
    choices_e, menu_e, env_id_e;
    state_id=state_id_e, priors=priors_e, normalize=:omega_one, varnames=["Price", "Quality"],
)
ri_logit
```

Visualize consideration-set KKT multipliers, Red-Bus/Blue-Bus share reallocation, and estimated preference parameters:

```julia
plot_result(dc_qs; view=:kkt)
plot_result(diag_bb)
plot_result(ri_logit; conf_level=0.95)
```

---

## Complete Example

The following end-to-end workflow compares two-state LQG rational inattention across steady-state, dynamic transition, structural MLE, Coibion–Gorodnichenko regression, and discrete-choice welfare diagnostics on a unified problem:

```@example ri
# 1. Steady-state two-state LQG filter
prob_full = RationalInattentionProblem(
    [0.88 0.12; 0.00 0.72], [1.0 0.0; 0.0 0.65], [1.0 0.3; 0.2 1.0];
    capacity=1.1,
    state_names=["Output", "Markup"],
    action_names=["Price", "Investment"],
    shock_names=["Demand", "Supply"],
)
sol_full = solve_ri_kalman(prob_full)

# 2. Verify Afrouzi-Yang transition converges to stationary Omega & Sigma_prior
prob_full_om = RationalInattentionProblem(
    prob_full.A, prob_full.B, prob_full.H; omega=sol_full.omega,
)
dyn_full = solve_ri_transition(prob_full_om, 0.25 * sol_full.Sigma_prior; T_path=90)

# 3. Simulate data, recover capacity by MLE, and evaluate implied CG slope
sim_full = simulate(sol_full, 200; burn=50, rng=rng_ri)
est_full = estimate_ri(prob_full, sim_full.actions; method=:mle, estimate_param=:capacity)
cg_slope_full = implied_cg_slope(sol_full; horizon=1)

# 4. Compare discrete-choice consideration set shrinkage under tighter capacity
dc_tight = solve_ri_discrete(U_qs, mu_qs; capacity=0.20)
dc_loose = solve_ri_discrete(U_qs, mu_qs; capacity=0.65)

(ss_omega=sol_full.omega,
 dyn_converged=dyn_full.converged,
 mle_kappa=est_full.capacity,
 implied_cg=cg_slope_full,
 tight_active=length(dc_tight.consideration_set),
 loose_welfare=dc_loose.net_welfare)
```

---

## Saving Results

Every `plot_result` method honors `save_path` to export standalone interactive HTML visualizations, and fitted model objects can be archived directly via `JLD2` or summarized with tabular formatting:

```julia
# Save standalone interactive HTML visualizations
plot_result(sol_full; view=:irf_by_capacity, save_path="ri_irf_by_capacity.html")
plot_result(dyn_full; view=:attention_path, save_path="ri_transition.html")
plot_result(mw_res; save_path="mw_pricing_irf.html")
plot_result(dc_loose; view=:choice_curves, save_path="ri_discrete_choice.html")

# Or save via save_plot helper
p_wf = plot_result(sol_full; view=:water_filling, title="Water-Filling Spectrum")
save_plot(p_wf, "ri_water_filling.html")
```

---

## Common Pitfalls

1. **Confusing `bits` and `nats` in `capacity` and `omega`**: By default, `RationalInattentionProblem`, `solve_ri_discrete`, and `solve_mw_attention` use `units=:bits`, converting capacity in bits to nats via $\ln 2$ while reporting both `capacity_bits` and `capacity_nats` (`mutual_info_bits` and `mutual_info_nats`). Verify `units` consistency when comparing `omega` against Matějka–McKay logit exponents $\exp(U_{j,s}/\omega)$, which always scale payoffs in nats.
2. **Violating the minimum-stabilizability capacity bound**: If the state transition matrix `A` has unstable eigenvalues ($|\lambda_i(A)| \ge 1$), stabilizing tracking error requires strictly exceeding Sims's rate-distortion lower bound $\kappa > \sum_{|\lambda_i(A)| > 1} \log_2 |\lambda_i(A)|$; supplying `capacity` below this bound triggers an `ArgumentError`.
3. **Specifying both or neither of `capacity` and `omega`**: `RationalInattentionProblem`, `solve_ri_discrete`, and `solve_mw_attention` require either a capacity budget `capacity` OR a marginal cost `omega`, never both simultaneously.
4. **Over-identifying scale in `estimate_ri_logit` with `base_payoff`**: When passing pre-scaled `base_payoff` ($K = 0$), $\omega$ is already identified relative to baseline utility units; combining `base_payoff` with `normalize=:omega_one` over-normalizes scale and throws an `ArgumentError`. Use `normalize=:none` with `base_payoff`, or `normalize=:omega_one` / `:first_coef` with attribute tensor `menu_attributes`.
5. **Assuming standard Multinomial Logit when priors or menus are asymmetric**: Matějka–McKay reduces to standard MNL *only* when alternatives are ex-ante symmetric under uniform priors; otherwise endogenous prior shares $P(j)$ bias choices toward familiar or broadly strong options and cause predictable IIA violations (`ri_iia_diagnostic`).

---

## References

- Afrouzi, H., & Yang, C. (2021). Dynamic Rational Inattention and the Phillips Curve. *CESifo Working Paper* 8840.
- Caplin, A., Dean, M., & Leahy, J. (2019). Rational Inattention, Optimal Consideration Sets, and Stochastic Choice. *Review of Economic Studies*, 86(3), 1061–1094. [https://doi.org/10.1093/restud/rdy037](https://doi.org/10.1093/restud/rdy037)
- Coibion, O., & Gorodnichenko, Y. (2015). Information Rigidity and the Expectations Formation Process: A Simple Framework and New Facts. *American Economic Review*, 105(8), 2644–2678. [https://doi.org/10.1257/aer.20110306](https://doi.org/10.1257/aer.20110306)
- Luo, Y. (2008). Consumption Dynamics under Information Processing Constraints. *Review of Economic Dynamics*, 11(2), 366–385. [https://doi.org/10.1016/j.red.2007.07.003](https://doi.org/10.1016/j.red.2007.07.003)
- Maćkowiak, B., & Wiederholt, M. (2009). Optimal Sticky Prices under Rational Inattention. *American Economic Review*, 99(3), 769–803. [https://doi.org/10.1257/aer.99.3.769](https://doi.org/10.1257/aer.99.3.769)
- Maćkowiak, B., & Wiederholt, M. (2015). Business Cycle Dynamics under Rational Inattention. *Review of Economic Studies*, 82(4), 1502–1532. [https://doi.org/10.1093/restud/rdv027](https://doi.org/10.1093/restud/rdv027)
- Matějka, F., & McKay, A. (2015). Rational Inattention to Discrete Choices: A New Foundation for the Multinomial Logit Model. *American Economic Review*, 105(1), 272–298. [https://doi.org/10.1257/aer.20130047](https://doi.org/10.1257/aer.20130047)
- Miao, J., Wu, J., & Young, E. R. (2022). Multivariate Rational Inattention. *Econometrica*, 90(2), 907–945. [https://doi.org/10.3982/ECTA17921](https://doi.org/10.3982/ECTA17921)
- Sims, C. A. (2003). Implications of Rational Inattention. *Journal of Monetary Economics*, 50(3), 665–690. [https://doi.org/10.1016/S0304-3932(03)00029-1](https://doi.org/10.1016/S0304-3932(03)00029-1)
- Sims, C. A. (2010). Rational Inattention and Monetary Economics. In B. M. Friedman & M. Woodford (Eds.), *Handbook of Monetary Economics* (Vol. 3, pp. 155–181). Elsevier. [https://doi.org/10.1016/B978-0-444-53238-1.00004-1](https://doi.org/10.1016/B978-0-444-53238-1.00004-1)
