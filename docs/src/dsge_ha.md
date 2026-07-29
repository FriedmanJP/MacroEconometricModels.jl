# [Heterogeneous Agent DSGE](@id dsge_ha)

Standard DSGE models assume a **representative agent** whose decisions aggregate to macroeconomic outcomes. In reality, households differ in wealth, income, and consumption --- heterogeneity that shapes aggregate responses to shocks, especially monetary and fiscal policy. MacroEconometricModels.jl provides a complete toolkit for **heterogeneous agent DSGE (HA-DSGE)** models: the **Endogenous Grid Method** (Carroll 2006) and **VFI** for solving individual problems, **Young (2010) histogram** tracking for the wealth distribution, and three aggregate solution methods --- **Sequence-Space Jacobian** (Auclert, Bardóczy, Rognlie & Straub 2021), **Reiter (2009) linearization**, and **Krusell-Smith (1998) simulation**. The module supports one-asset and two-asset HANK models with Bayesian estimation.

- Individual problem solvers: EGM (one-asset and nested two-asset) and VFI with Howard improvement
- Income discretization: Rouwenhorst (1995) and Tauchen (1986)
- Endogenous labor supply: GHH (no wealth effect) and additively separable
- Distribution: Young (2010) non-stochastic histogram with sparse transition matrices
- Steady state: bisection on the interest rate with EGM + distribution + market clearing
- Three aggregate solution methods: SSJ, Reiter, Krusell-Smith
- Distribution representation: Young (2010) histogram or Winberry (2018) parametric moments
- Built-in models: Krusell-Smith (1998), one-asset HANK, two-asset HANK, Huggett (1993), GHH endogenous labor
- Bayesian estimation via RWMH + Kalman filter on reduced system
- Visualization: wealth distribution, Lorenz curve, policy functions

```@setup dsge_ha
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: Krusell-Smith steady state**

```@example dsge_ha
spec = load_ha_example(:krusell_smith)
ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04),
                           max_iter=80, tol=1e-4)
report(ss)
```

**Recipe 2: Solve and compute IRFs via SSJ**

```@example dsge_ha
sol = solve(spec; method=:ssj, ss=ss, T_horizon=50, n_reduced=15)
report(sol)
```

**Recipe 3: Wealth distribution visualization**

```julia
plot_result(ss)                     # wealth histogram with Gini
plot_result(ss; view=:lorenz)       # Lorenz curve
plot_result(ss; view=:policy)       # consumption and savings by income
```

**Recipe 4: Simulate individual panel data**

```@example dsge_ha
panel = simulate_panel(ss; N_agents=500, T_periods=100)
size(panel)
```

The panel matrix contains simulated asset holdings for 500 agents over 100 periods, enabling cross-sectional and longitudinal analyses of wealth dynamics at the micro level.

**Recipe 5: Inequality dynamics at steady state**

```@example dsge_ha
ineq = inequality_irf(ss; T_periods=10)
(gini = round(ineq[:gini][1], digits=4),
 p50  = round(ineq[:p50][1], digits=2),
 p90  = round(ineq[:p90][1], digits=2))
```

**Recipe 6: Krusell-Smith simulation method**

```@example dsge_ha
ks_result = solve(spec; method=:krusell_smith, ss=ss,
                  T_sim=500, T_burn=100, max_outer=3,
                  rho_z=0.95, sigma_z=0.007)
report(ks_result)
```

---

## Individual Problem

The full model is assembled into an [`HADSGESpec`](@ref) (built by [`load_ha_example`](@ref)), which bundles the discretized [`IncomeProcess`](@ref) and the household [`IndividualProblem`](@ref) — the utility, marginal utility, budget, and borrowing-constraint fields the EGM/VFI inner loops consume.

Households solve a consumption-savings problem with idiosyncratic income risk and a borrowing constraint:

```math
V(a, e) = \max_{c, a'} \; u(c) + \beta \, \mathbb{E}\bigl[V(a', e') \mid e\bigr]
```

subject to:

```math
c + a' = (1 + r) \, a + w \, e, \qquad a' \geq \underline{a}
```

where:
- ``a`` is individual asset holdings
- ``e`` is idiosyncratic productivity (Markov chain)
- ``r, w`` are aggregate prices (interest rate, wage)
- ``\underline{a}`` is the borrowing constraint
- ``\beta`` is the discount factor

### Endogenous Grid Method

The **EGM** (Carroll 2006) avoids root-finding by inverting the Euler equation on an endogenous grid:

1. Fix end-of-period assets ``a'`` on the exogenous grid
2. Compute expected marginal utility: ``\text{EMU}_i = \beta (1+r) \sum_{j'} \pi(j, j') \, u'(c(a'_i, e_{j'}))``
3. Invert the Euler equation: ``c_i = (u')^{-1}(\text{EMU}_i)``
4. Recover beginning-of-period assets (endogenous): ``a_i = (c_i + a'_i - w e_j) / (1+r)``
5. Interpolate back to the exogenous grid
6. Apply the borrowing constraint: if ``a < a_{\text{endo},1}``, consume all cash-on-hand

The EGM converges in 200--400 iterations for typical calibrations. The `compute_steady_state` function calls EGM internally at each bisection step:

```@example dsge_ha
spec_ks = load_ha_example(:krusell_smith)
ss_egm = compute_steady_state(spec_ks; K_init=10.0, r_bounds=(-0.02, 0.04),
                               max_iter=80, tol=1e-4)
report(ss_egm)
```

The steady state report displays convergence diagnostics, equilibrium prices, aggregate quantities, and wealth distribution statistics. A negative Euler error (in ``\log_{10}`` units) indicates the accuracy of the consumption policy --- values below ``-3`` are standard in the literature.

### Choosing an Asset Grid

Two decisions matter, and they trade off against each other on one grid shape but not the other.

**The ceiling must not bind.** The Young (2010) transition clamps the savings policy into ``[a_{\min}, a_{\max}]``, so any mass that wants to save past ``a_{\max}`` is silently pushed back onto the top node --- see Common Pitfalls. Choose ``a_{\max}`` from the *right tail* of the wealth distribution, not from the mean: the built-in examples run to ``a_{\max} = 1000`` against an equilibrium ``K \approx 42``, a ratio of about ``24``.

**Raising the ceiling must not cost resolution at the constraint.** The `:double_exp` grid is a fixed curve rescaled by ``(a_{\max} - a_{\min})``, so its bottom spacing is *linear* in ``a_{\max}`` --- widening the grid five-fold coarsens the borrowing constraint five-fold. The `:geometric` grid is equidistant in ``\log(a - a_{\min} + \text{pivot})``, so its bottom spacing grows only logarithmically. Measured on the Krusell--Smith calibration at ``n_a = 200``:

| Grid | ``a_{\max}`` | First step ``\Delta a_1`` | Mass at floor | Mass at ceiling | Euler error |
|------|--------------|---------------------------|---------------|-----------------|-------------|
| `:double_exp` | 200 | 0.2208 | 6.41% | ``7.1 \times 10^{-3}`` | ``-4.54`` |
| `:double_exp` | 1000 | 1.1039 | 8.15% | ``0`` | ``-4.21`` |
| `:geometric` | 200 | 0.0085 | 6.12% | ``8.7 \times 10^{-3}`` | ``-6.09`` |
| `:geometric` | 1000 | 0.0106 | 6.31% | ``4.4 \times 10^{-13}`` | ``-6.04`` |

Widening `:double_exp` fixes the ceiling but degrades the Euler error from ``-4.54`` to ``-4.21``; `:geometric` clears the ceiling *and* holds accuracy near ``-6``. This is why the built-in one-asset examples use `grid_type=:geometric`. The default remains `:double_exp` for backward compatibility.

Grid *resolution* is a separate matter, and refining ``n_a`` does **not** fix a binding ceiling. On the truncating configuration, sweeping ``n_a`` from 100 to 1600 at fixed ``a_{\max} = 200`` moves the relative clearing residual only from ``1.6844\%`` to ``1.6848\%`` --- unchanged to four significant figures across a 16-fold refinement, while the solve time rises from 2.4 s to 47.6 s. Only ``a_{\max}`` fixes truncation.

### VFI with Howard Improvement

When EGM is not applicable (non-separable utility, complex constraints), **Value Function Iteration** with **Howard improvement steps** provides a robust alternative. Each VFI iteration consists of one policy maximization step followed by ``K`` policy-evaluation steps (default ``K = 20``), which are cheap linear operations that dramatically accelerate convergence.

!!! note "Two-asset steady states"
    `compute_steady_state` bisects on a single interest rate and therefore supports **one-asset models only**. Clearing a two-asset model requires a two-dimensional market-clearing solve, which is not implemented; calling `compute_steady_state(load_ha_example(:two_asset_hank))` raises an `ArgumentError` saying so. The two-asset individual problem itself (nested EGM, adjustment costs) is fully available --- it is the general-equilibrium close that is missing.

---

## [Endogenous Labor Supply](@id ha_labor)

By default households choose only consumption and savings. Attaching a [`LaborSupply`](@ref) to the [`IndividualProblem`](@ref) makes hours a choice too, which is what transmits shocks through the labor market and is essential for the fiscal and monetary experiments HANK models are built for.

Disutility of hours is isoelastic, ``v(n) = \psi n^{1 + 1/\varphi} / (1 + 1/\varphi)``, so ``v'(n) = \psi n^{1/\varphi}`` and ``\varphi`` is the Frisch elasticity. Two preference specifications are available, and they differ in exactly one respect: whether hours carry a **wealth effect**.

### GHH: no wealth effect

Greenwood, Hercowitz & Huffman (1988) preferences put the disutility inside the felicity function, ``U\bigl(c - \psi n^{1+1/\varphi}/(1+1/\varphi)\bigr)``. The intratemporal condition is then

```math
\psi n^{1/\varphi} = w e \qquad \Longrightarrow \qquad n(e) = \left(\frac{w e}{\psi}\right)^{\varphi}
```

Hours depend on the effective wage alone — not on consumption, not on assets. That makes GHH the tractable default: substituting hours out leaves an ordinary one-dimensional consumption-savings problem in the composite good ``x = c - v(n)``, with net labor income ``\tilde{y}(e) = w e n(e) - v(n(e))``. The EGM solves that problem unchanged and recovers ``c = x + v(n)`` at the end.

### Separable: with a wealth effect

Additively separable preferences ``u(c) - v(n)`` give

```math
\psi n^{1/\varphi} = w e \, u'(c)
```

which couples hours to consumption: a richer household consumes more, values a marginal dollar less, and works less. On the unconstrained EGM branch this costs nothing — the Euler inversion delivers ``c`` first, so hours follow in closed form. At the borrowing constraint the two conditions must hold jointly, and `_egm_solve` runs a bracketed scalar root-find (``g(c) = c + \underline{a} - Ra - w e n(c)`` is strictly increasing, so the root is unique).

### Solving

```@example dsge_ha
spec_lab = load_ha_example(:endogenous_labor)
ss_lab = compute_steady_state(spec_lab)
(r = round(ss_lab.prices[:r], digits=6),
 K = round(ss_lab.aggregates[:K], digits=4),
 L = round(ss_lab.aggregates[:L], digits=4),
 N = round(ss_lab.aggregates[:N], digits=4))
```

Two labor aggregates are reported and they are different objects: `aggregates[:L]` is efficiency units ``\int e\,n\,d\mu`` — what enters the production function — while `aggregates[:N]` is mean hours ``\int n\,d\mu``. They coincide only if every income state equals 1. The hours policy itself is `ss.policies[:labor]`.

The built-in example sets ``\psi = 3`` so that ``L \approx 1``, matching the ``L = 1`` normalization the exogenous-labor examples impose — which makes the two economies directly comparable. Swap in separable preferences by rebuilding the individual problem:

```@example dsge_ha
ls_sep = LaborSupply(; kind=:separable, psi=1.0, frisch=0.5)
(kind = ls_sep.kind, psi = ls_sep.psi, frisch = ls_sep.frisch)
```

!!! note "Aggregate labor is an outcome, not a parameter"
    With exogenous labor, `params[:L]` is a fixed input to the firm's problem. With endogenous labor it is an *outcome* of the household problem, so `compute_steady_state` iterates solve → aggregate hours → re-price to a fixed point inside each bisection step. Under Cobb-Douglas this converges on the second pass, because the firm FOC pins ``K/L`` from ``r`` alone and the wage is therefore invariant to ``L``.

### Labor in the sequence space

Hours are available as a [`HetBlock`](@ref) output — `:N` (or `:n`, `:hours`) for mean hours, `:L` for efficiency units:

```@example dsge_ha
hh_lab = HetBlock(spec_lab, ss_lab; inputs=[:r, :w], outputs=[:A, :N, :L])
J_lab = block_jacobian(hh_lab, 10)
round(J_lab[(:N, :w)][1, 1], digits=6)
```

Under GHH that number has a closed form: hours are static, so ``dN/dw = \varphi N / w``, which at this calibration is ``0.5 \times 0.8869 / 2.5099 = 0.1767``. For the same reason the Jacobian ``\partial N_t / \partial w_s`` is **diagonal** — GHH hours never depend on the continuation value, so there is no anticipation effect (the measured off-diagonal maximum is ``8 \times 10^{-13}``). Separable preferences break both properties: hours inherit the wealth effect and therefore respond to announced future prices.

---

## [Discrete-Continuous Choice: DCEGM](@id dcegm)

Many household problems pair the continuous consumption choice with a **discrete** one — retire or work, own or rent, adjust a durable or not. The discrete choice makes the value function non-concave, so the Euler equation is no longer sufficient: it admits spurious local optima, and the endogenous grid it produces is non-monotone. Iskhakov, Jørgensen, Rust & Schjerning (2017) solve this with an **upper-envelope** step.

```math
V_t(M, j, d_-) = \max_{d \in D(d_-)} v_t(M, j, d), \qquad
v_t(M, j, d) = \max_{c} u(c, d) + \beta\, E_j V_{t+1}\!\left(R(M - c) + y(d, j'),\, j',\, d\right)
```

where:
- ``M`` is cash-on-hand and ``j`` the idiosyncratic income state
- ``d_-`` is last period's discrete option and ``D(d_-)`` those still feasible (an **absorbing** option leaves only itself)
- ``u(c, d)`` is flow utility, ``R`` the gross return, and ``y(d, j')`` income received next period

One EGM sweep per option gives candidate ``(M, c, v)`` triples; where the continuation value is non-concave the ``M`` sequence loops back on itself, and the same cash-on-hand appears on two branches. `_upper_envelope` keeps the branch with the higher value and inserts the **exact** crossing, at which the value function is continuous while consumption jumps.

!!! note "Technical Note"
    Between two adjacent candidate grid points both branches are linear — the grid contains every knot — so the crossing ``M^* = M_{lo} + (M_{hi} - M_{lo}) f(M_{lo}) / (f(M_{lo}) - f(M_{hi}))`` with ``f = v_p - v_q`` is exact, not bisected. It is inserted twice, once with each branch's consumption, because consumption is genuinely discontinuous at a discrete-choice threshold.

[`dcegm_retirement_model`](@ref) builds the canonical retirement problem: work or retire (absorbing), flow utility ``\log c - \delta \mathbb{1}[\text{work}]``, and no income once retired.

```@example dsge_ha
retire = dcegm_retirement_model(; n_periods=6, beta=0.98, R=1.0, wage=20.0,
                                disutility=1.0, a_max=60.0, n_a=200)
dc = dcegm_solve(retire)
report(dc)
```

The envelope finds switching thresholds only on the **working** branch. Retirement is absorbing, so a retiree faces no further discrete choice and their problem stays concave — which is why the retired branch has zero kinks and reduces to deterministic cake-eating with the closed form ``c_t = M / \sum_{k \le T-t} \beta^k``.

```@example dsge_ha
annuity = sum(0.98^k for k in 0:2)                # three periods left at t = 4
(dcegm  = round(dcegm_policy(dc, 4, 1, 1, 45.0)[1], digits=8),
 closed = round(45.0 / annuity, digits=8),
 threshold = round(dcegm_threshold(dc, 4, 2, 1; M_lo=0.5, M_hi=60.0), digits=4))
```

The retired consumption function reproduces the closed form to machine precision, and the threshold says a worker with more than about 49 units of cash-on-hand retires at ``t = 4`` — wealthy enough that the disutility of work outweighs another wage.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `500` | Infinite-horizon iteration cap |
| `tol` | `Real` | ``10^{-8}`` | Sup-norm policy tolerance (infinite horizon) |
| `verbose` | `Bool` | `false` | Print iteration diagnostics |

| Field | Type | Description |
|-------|------|-------------|
| `M`, `c`, `v` | `Array{Vector{T},3}` | Endogenous grid, consumption, and conditional value per `(t, option, income state)` |
| `n_kinks` | `Array{Int,3}` | Switching thresholds the envelope found |
| `converged` | `Bool` | Infinite horizon: whether the policy fixed point met `tol` |

### Taste Shocks

Adding extreme-value taste shocks of scale ``\lambda`` replaces the hard maximum with a log-sum-exp and the switching rule with multinomial-logit probabilities, which makes the value function differentiable — the numerically robust variant. Setting ``\lambda = 0`` recovers the deterministic upper envelope.

```@example dsge_ha
smooth = dcegm_solve(dcegm_retirement_model(; n_periods=6, a_max=60.0, n_a=200,
                                            taste_shock_scale=0.5))
p_low  = dcegm_choice_probabilities(smooth, 4, 2, 1, 20.0)
p_high = dcegm_choice_probabilities(smooth, 4, 2, 1, 55.0)
(retire_at_20 = round(p_low[1], digits=4), retire_at_55 = round(p_high[1], digits=4))
```

Retirement probability rises with wealth, smoothly rather than as a step. As ``\lambda \to 0`` the probabilities collapse onto the deterministic rule and the consumption policy onto the upper-envelope solution.

### Simulating the Distribution

[`dcegm_simulate`](@ref) propagates a Young (2010) histogram whose transition respects the discrete choice: mass is split across options by the choice probabilities and each part is pushed forward by that option's savings policy.

```@example dsge_ha
dist = dcegm_simulate(dc, collect(range(0.01, 60.0; length=100)))
(mass = round(sum(dist.dist[3, :, :, :]), digits=12),
 retired_share = round.(dist.shares[:, 1], digits=4))
```

Mass is conserved exactly — off-grid landing points are split between the two bracketing nodes in proportion to distance. The retired share is monotone in age because retirement is absorbing, and jumps to one in the terminal period, when working carries its disutility with no future wage to compensate.

---

## Income Discretization

Idiosyncratic productivity follows an AR(1) process ``\log e' = \rho \log e + \sigma \varepsilon`` discretized onto a finite Markov chain.

### Rouwenhorst

The **Rouwenhorst (1995)** method constructs the transition matrix recursively. It is more accurate than Tauchen for highly persistent processes (``\rho > 0.9``).

```@example dsge_ha
inc7 = rouwenhorst(0.966, 0.5, 7)
(states = round.(inc7.states, digits=3),
 stationary = round.(inc7.stationary_dist, digits=4))
```

The 7-state discretization spans the ergodic support of the log-productivity process. The stationary distribution concentrates mass near the mean, with thin tails reflecting the high persistence.

!!! warning "`sigma` is the innovation standard deviation"
    `rouwenhorst(rho, sigma, n)` and `tauchen(rho, sigma, n)` interpret `sigma` as the standard deviation of the AR(1) **innovation** ``\varepsilon_t``, so the process itself has ``\mathrm{sd}(y_t) = \sigma / \sqrt{1 - \rho^2}``. The call above therefore produces a chain with an unconditional standard deviation of ``0.5 / \sqrt{1 - 0.966^2} = 1.934``, not ``0.5``.

    Calibration targets are usually quoted the other way round --- as the cross-sectional standard deviation of log earnings --- and `markov_rouwenhorst` in the Python `sequence-jacobian` toolkit uses that convention too. Pass `sigma_is=:unconditional` when your ``\sigma`` is ``\mathrm{sd}(y_t)``:

    ```@example dsge_ha
    a = rouwenhorst(0.966, 0.5, 7)                             # sd(y) = 1.934
    b = rouwenhorst(0.966, 0.5, 7; sigma_is=:unconditional)    # sd(y) = 0.500
    (innovation = round(a.states[end], digits=4),
     unconditional = round(b.states[end], digits=4))
    ```

    At ``\rho = 0.966`` the two readings differ by ``3.87\times`` in logs and ``15\times`` in variance --- more than enough to move the stationary wealth distribution off any reasonable asset grid. The built-in examples target ``\mathrm{sd}(\log e) = 0.5`` and pass `sigma_is=:unconditional` accordingly.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `rho` | `Real` | — | Persistence parameter |
| `sigma` | `Real` | — | Standard deviation of innovations |
| `n` | `Int` | — | Number of grid points |

| Field | Type | Description |
|-------|------|-------------|
| `transition` | `Matrix{T}` | ``n \times n`` Markov transition matrix |
| `states` | `Vector{T}` | Grid of log-productivity levels |
| `stationary_dist` | `Vector{T}` | Ergodic distribution over states |

### Tauchen

The **Tauchen (1986)** method uses equally spaced grid points covering ``\pm m`` standard deviations and normal CDF transition probabilities.

```@example dsge_ha
inc_t = tauchen(0.9, 0.2, 5; m=3)
(states = round.(inc_t.states, digits=3),
 stationary = round.(inc_t.stationary_dist, digits=4))
```

With lower persistence (``\rho = 0.9``), Tauchen produces a wider grid and a flatter stationary distribution than Rouwenhorst. The ``m = 3`` setting covers three unconditional standard deviations on each side.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `m` | `Int` | `3` | Number of standard deviations to cover |

---

## Distribution Tracking

Beyond the stationary distribution, [`distribution_irf`](@ref) traces the impulse response of the entire wealth distribution to an aggregate shock, and [`inequality_irf`](@ref) reports the induced Gini/percentile dynamics.

The cross-sectional wealth distribution ``\Gamma(a, e)`` evolves according to the **Young (2010) non-stochastic simulation** method. Given the savings policy ``a' = g(a, e)``, the distribution updates via a sparse transition matrix ``\Lambda``:

```math
D_{t+1} = \Lambda \, D_t
```

where ``\Lambda`` uses **lottery weights** to map off-grid savings back to grid points. If ``g(a_i, e_j)`` falls between ``a_k`` and ``a_{k+1}``, mass is split proportionally:

```math
\omega = \frac{a_{k+1} - g(a_i, e_j)}{a_{k+1} - a_k}
```

The transition matrix is sparse --- each column has at most ``2 N_e`` nonzero entries. The **stationary distribution** ``D^*`` satisfies ``D^* = \Lambda D^*`` and is found via power iteration. The `compute_steady_state` function handles distribution tracking internally. The resulting `HASteadyState` stores the distribution:

```@example dsge_ha
(shape = size(ss.distribution),
 total_mass = round(sum(ss.distribution), digits=10),
 aggregate_K = round(ss.aggregates[:K], digits=2))
```

The distribution is an ``N_a \times N_e`` matrix whose entries sum to unity. Aggregate capital integrates asset holdings against the distribution, providing the supply side of the capital market clearing condition.

---

## Steady State

The **stationary equilibrium** requires the individual problem, distribution, and prices to be mutually consistent. `compute_steady_state` bisects on the interest rate ``r``:

1. Guess ``r_{\text{mid}} = (r_{\text{lo}} + r_{\text{hi}}) / 2``
2. Compute prices ``(r, w)`` from the firm's first-order conditions given ``r_{\text{mid}}``
3. Solve the individual problem via EGM at those prices
4. Build the transition matrix and compute the stationary distribution
5. Aggregate capital supply: ``K_s = \int a \, d\Gamma(a, e)``
6. Compute capital demand ``K_d`` from the firm's FOC
7. If ``K_s > K_d``, interest rate is too low → ``r_{\text{hi}} = r_{\text{mid}}``; otherwise ``r_{\text{lo}} = r_{\text{mid}}``
8. Converge when ``|K_s - K_d| < \text{tol}``

```@example dsge_ha
spec_hank = load_ha_example(:one_asset_hank)
ss_hank = compute_steady_state(spec_hank; K_init=10.0, r_bounds=(-0.02, 0.04),
                                max_iter=80, tol=1e-4)
report(ss_hank)
```

The one-asset HANK steady state clears at a **higher** interest rate and a **lower** capital stock than the standard Aiyagari economy (``r = 0.0115`` vs ``0.0077`` per quarter; ``K = 35.70`` vs ``42.44``). The reason is impatience, not the New Keynesian block: this calibration sets ``\beta = 0.986`` against ``0.99`` for Krusell--Smith, and that difference in time preference dominates the extra dividend income. The Gini coefficient and wealth percentiles characterize the cross-sectional distribution, with the 90th percentile several times the median --- consistent with empirical wealth data.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `K_init` | `T` | `10.0` | Initial guess for aggregate capital |
| `r_bounds` | `Tuple{T,T}` | `(-0.01, 0.04)` | Bisection bounds for interest rate |
| `max_iter` | `Int` | `200` | Maximum bisection iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance on excess demand |
| `verbose` | `Bool` | `false` | Print iteration progress |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether bisection converged |
| `iterations` | `Int` | Number of bisection iterations |
| `prices` | `Dict{Symbol,T}` | Equilibrium prices (``r``, ``w``) |
| `aggregates` | `Dict{Symbol,T}` | Aggregate quantities (``K``, ``Y``, ``C``) |
| `distribution` | `Matrix{T}` | ``N_a \times N_e`` stationary distribution |
| `policies` | `Dict{Symbol,Matrix{T}}` | Policy functions (`:consumption`, `:savings`) |
| `excess_demand` | `T` | Final excess demand for capital |
| `euler_error` | `T` | ``\log_{10}`` Euler equation error |
| `grid` | `HAGrid{T}` | Asset grid used |

---

## Sequence-Space Jacobian

The **Sequence-Space Jacobian** method (Auclert, Bardóczy, Rognlie & Straub 2021) computes the ``T \times T`` Jacobian of aggregate outputs with respect to aggregate input sequences. The key idea: instead of tracking the full distribution as a state variable (as in Reiter), work directly with the impulse response sequences.

The algorithm computes a **fake news matrix** ``\mathcal{F}`` via:
1. **Backward iteration**: perturb a price at time ``s``, iterate the EGM backward to capture the expectation channel
2. **Forward iteration**: propagate the perturbed policies through the distribution forward in time
3. **Accumulation**: the true Jacobian ``\mathcal{J}`` is the cumulative sum of ``\mathcal{F}``

The resulting ``\mathcal{J}`` is converted to a minimal state-space realization via the **Ho-Kalman algorithm** (SVD of the Hankel matrix of IRF coefficients), producing a reduced `DSGESolution` compatible with all existing analysis functions.

```@example dsge_ha
sol_ssj = solve(spec; method=:ssj, ss=ss, T_horizon=50, n_reduced=15)
report(sol_ssj)
```

Composing several blocks into a model, and going to second order in the sequence space, are covered in [Block Composition and Second-Order SSJ](@ref ssj_blocks).

The SSJ method reduces the full sequence-space representation (dimension ``T``) to a compact state-space form with `n_reduced` states. The explained variance measures the fraction of aggregate dynamics captured by the truncated Ho-Kalman basis --- values above 99.9% confirm the reduction is adequate. The underlying steady state is reported alongside the reduction diagnostics.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_horizon` | `Int` | `300` | Truncation horizon for sequences |
| `n_reduced` | `Int` | `30` | Reduced state-space dimension (Ho-Kalman) |
| `dx` | `Real` | ``10^{-4}`` | Finite-difference step size |

| Field | Type | Description |
|-------|------|-------------|
| `method` | `Symbol` | Solution method (`:ssj` or `:reiter`) |
| `n_full_states` | `Int` | Full state-space dimension before reduction |
| `n_reduced` | `Int` | Reduced state-space dimension |
| `explained_variance` | `T` | Fraction of variance captured by truncation |
| `linear_solution` | `DSGESolution{T}` | Reduced-form state-space representation |
| `steady_state` | `HASteadyState{T}` | Underlying stationary equilibrium |

---

## Reiter Method

The **Reiter (2009)** method linearizes the entire system --- Euler equations, distribution evolution, and aggregate equilibrium --- around the stationary equilibrium. The distribution histogram becomes part of the state vector, yielding a large linear system that is reduced via SVD.

The implementation uses **observability-based SVD**: the reduction basis is built from the observability matrix ``[c', c' \Lambda', c' (\Lambda')^2, \ldots]'`` where ``c`` is the capital aggregation vector. This identifies the distribution directions most relevant for aggregate dynamics, achieving ``>99.9\%`` explained variance with 15--30 reduced states.

```@example dsge_ha
sol_reiter = solve(spec; method=:reiter, ss=ss, n_reduced=15)
report(sol_reiter)
```

The Reiter method produces an equivalent reduced state-space form to SSJ. The explained variance confirms that 15 reduced states capture nearly all aggregate dynamics. The two methods yield identical IRFs up to numerical precision for the same `n_reduced`, but differ in computational cost: SSJ is faster for models with few aggregate inputs, while Reiter scales better when many prices affect household decisions.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_reduced` | `Int` | `50` | Maximum reduced dimension |
| `dx` | `Real` | ``10^{-6}`` | Finite-difference step size |

---

## Krusell-Smith Method

The **Krusell-Smith (1998)** method approximates agents' forecasting rule with a **perceived law of motion** (PLM) for aggregate capital, including the aggregate shock ``z``:

```math
\log K_{t+1} = b_0 + b_1 \log K_t + b_2 z_t
```

The fitted PLM coefficients and simulated paths are returned in a [`KrusellSmithSolution`](@ref).

The algorithm iterates between simulation (using the PLM to forecast prices) and regression (updating PLM coefficients via OLS). Convergence requires ``R^2 > 0.9999``, reflecting the near-sufficiency of the first moment plus the aggregate shock for forecasting. Including ``z`` is essential for the Den Haan (2010) accuracy test below: a ``z``-free PLM produces a degenerate, fluctuation-free simulated path.

!!! note "Technical Note"
    The PLM coefficients are updated with damping: ``b^{\text{new}} = 0.5 \, b^{\text{OLS}} + 0.5 \, b^{\text{old}}``. This prevents oscillation and ensures monotone convergence.

```@example dsge_ha
sol_ks = solve(spec; method=:krusell_smith, ss=ss,
               T_sim=500, T_burn=100, max_outer=3,
               rho_z=0.95, sigma_z=0.007)
report(sol_ks)
```

The PLM ``R^2`` near unity confirms that aggregate capital plus the aggregate shock is approximately sufficient for forecasting prices --- the core insight of Krusell & Smith (1998). The three coefficients (intercept, capital slope, and shock loading) characterize the aggregate law of motion. Unlike SSJ and Reiter, the Krusell-Smith method does not produce a reduced linear state-space form, so it cannot be used directly with `irf` or `fevd`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_sim` | `Int` | `11000` | Simulation length |
| `T_burn` | `Int` | `1000` | Burn-in periods to discard |
| `max_outer` | `Int` | `20` | Maximum PLM iterations |
| `rho_z` | `Real` | `0.95` | Aggregate shock persistence |
| `sigma_z` | `Real` | `0.007` | Aggregate shock standard deviation |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether PLM coefficients converged |
| `iterations` | `Int` | Number of outer PLM iterations |
| `plm_coefficients` | `Dict{Symbol,Vector{T}}` | PLM regression coefficients per variable |
| `r_squared` | `Dict{Symbol,T}` | ``R^2`` of PLM regression per variable |
| `steady_state` | `HASteadyState{T}` | Underlying stationary equilibrium |

---

## Accuracy: the Den Haan (2010) Test

Den Haan (2010) shows that the regression ``R^2`` and standard error are **inadequate** accuracy measures for a Krusell-Smith solution: an ``R^2`` of 0.9999 can coexist with a standard deviation of aggregate capital that is off by double digits. The powerful test compares two simulations under the same shock path --- a **reference** path from the explicit cross-sectional (Young) simulation, and a **PLM-only** path that iterates the aggregate law of motion on its *own* forecasts without re-anchoring to the simulated cross-section:

```math
\varepsilon_t = 100 \cdot \left| \log K_t^{\text{ref}} - \log K_t^{\text{PLM}} \right|
```

Call [`den_haan_test`](@ref) on a [`KrusellSmithSolution`](@ref) to obtain the maximum and mean errors packaged in a [`DenHaanAccuracy`](@ref) result.

where:
- ``K_t^{\text{ref}}`` is the reference aggregate capital from the explicit distribution simulation
- ``K_t^{\text{PLM}}`` is the PLM-only path, ``\log K_{t+1}^{\text{PLM}} = b_0 + b_1 \log K_t^{\text{PLM}} + b_2 z_t``

The maximum and mean of ``\varepsilon_t`` (post burn-in) are the headline statistics; `den_haan_test` also reports the standard-deviation comparison.

```@example dsge_ha
acc = den_haan_test(sol_ks; T_sim=300, T_burn=50)
report(acc)
```

A maximum error well below 1% confirms that the perceived law of motion reproduces the aggregate dynamics implied by the full cross-section --- the benchmark for an accurate Krusell-Smith solution. The ``\sigma`` ratio near unity shows the PLM reproduces the volatility of aggregate capital, the diagnostic Den Haan (2010) emphasizes.

!!! note "Aiyagari only"
    `den_haan_test` targets the Aiyagari capital model of Den Haan (2010). For a Huggett solution the cleared aggregate is the risk-free rate, which is driven by the wealth distribution rather than the shock alone, so the test raises an informative error.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_sim` | `Int` | `10000` | Simulation length |
| `T_burn` | `Int` | `1000` | Burn-in periods to discard |
| `rho_z` | `Real` | `0.95` | Aggregate shock persistence |
| `sigma_z` | `Real` | `0.007` | Aggregate shock standard deviation |

| Field | Type | Description |
|-------|------|-------------|
| `dh_max` | `T` | Maximum percentage error between reference and PLM-only paths |
| `dh_mean` | `T` | Mean absolute percentage error |
| `sigma_ref` | `T` | Standard deviation of the reference aggregate (``\log K``) |
| `sigma_plm` | `T` | Standard deviation of the PLM-only aggregate |

---

## Two-Asset HANK

The **two-asset HANK** model (Kaplan, Moll & Violante 2018) features households holding both liquid bonds ``b`` and illiquid equity ``a``, with a portfolio **adjustment cost** ``\chi(d, a)`` for accessing the illiquid asset:

```math
V(b, a, e) = \max_{c, b', d} \; u(c) + \beta \, \mathbb{E}\bigl[V(b', a', e') \mid e\bigr]
```

subject to:
```math
c + b' + d + \chi(d, a) = (1 + r_b) \, b + w \, e + T
```
```math
a' = (1 + r_a) \, a + d, \qquad b' \geq \underline{b}
```

where:
- ``b`` is liquid bonds, ``a`` is illiquid equity
- ``d`` is the deposit/withdrawal from the illiquid account
- ``\chi(d, a) = \chi_0 |d / a|^{\chi_1} \cdot a`` is the convex adjustment cost
- ``r_b, r_a`` are liquid and illiquid returns

The individual problem is solved via **nested EGM**: an outer loop over deposit choices with an inner EGM on the liquid dimension.

```@example dsge_ha
spec_2a = load_ha_example(:two_asset_hank)
(n_dims = spec_2a.grid.n_dims,
 grid_points = spec_2a.grid.n_points,
 labels = spec_2a.grid.labels,
 has_adjustment_cost = spec_2a.individual.adjustment_cost !== nothing)
```

The two-asset model uses a two-dimensional grid (liquid ``\times`` illiquid) with 2500 total points. The presence of the adjustment cost ``\chi(d, a)`` induces an inaction region where households neither deposit nor withdraw from the illiquid account, generating realistic portfolio rebalancing behavior.

---

## Huggett (1993): The Risk-Free Rate

The Huggett (1993) economy is a pure-exchange, incomplete-markets model: agents receive a stochastic endowment ``e_t`` and trade a single risk-free bond in **zero net supply**, subject to a credit limit ``\underline{a} < 0``. With no production, the equilibrium interest rate is the price that clears the bond market, and precautionary saving against uninsurable endowment risk drives it **below** the representative-agent time-preference rate.

```math
c_t + a_{t+1} = (1 + r)\, a_t + w\, e_t, \qquad a_{t+1} \geq \underline{a}, \qquad \int a_{t+1}\, d\mu = 0
```

where:
- ``e_t \in \{e_h, e_l\} = \{1.0, 0.1\}`` is the two-state Markov endowment
- ``w`` is the aggregate endowment level (``w = 1`` in steady state)
- ``r`` is the per-period risk-free rate that clears the bond market in zero net supply
- ``\underline{a}`` is the credit limit; the `:huggett` example defaults to ``\underline{a} = -2``

The model is selected by `spec.model == :huggett`, which routes `compute_steady_state` to the zero-net-supply clearing rule (no firm FOC). Calibration follows Huggett (1993): CRRA utility (``\sigma = 1.5``), ``\beta = 0.99322``, and six model periods per year.

```@example dsge_ha
hug = load_ha_example(:huggett)               # default credit limit ā = -2
ss_hug = compute_steady_state(hug; max_iter=150, tol=5e-4)
r_annual = (1 + ss_hug.prices[:r])^6 - 1      # six model periods per year
(r_period = round(ss_hug.prices[:r], digits=5),
 r_annual = round(r_annual, digits=4),
 clearing = round(ss_hug.excess_demand, digits=6))
```

The annual risk-free rate is about ``-7\%`` --- far below the ``4.2\%`` time-preference rate of the representative-agent benchmark, and matching Huggett's Table 1. Incomplete insurance against idiosyncratic endowment risk depresses the rate: agents collectively demand the bond for self-insurance, but it is in zero net supply, so the price (the interest rate) must fall until aggregate bond holdings clear at zero. Loosening the credit limit raises the equilibrium rate toward the time-preference rate.

With an aggregate endowment shock (``w_t``, AR(1)), the model is dynamically solvable by all three methods; the clearing rate ``r_t`` falls on impact of a positive endowment shock:

```julia
sol_ssj = solve(hug; method=:ssj, ss=ss_hug, T_horizon=100, n_reduced=20)
sol_rei = solve(hug; method=:reiter, ss=ss_hug, n_reduced=30)
sol_ks  = solve(hug; method=:krusell_smith, ss=ss_hug, T_sim=2000)
```

The Sequence-Space Jacobian forms the market-clearing system ``H_U \, dr + H_Z \, dw = 0`` and solves for the rate path; the Reiter method pins ``r`` statically each period from the linearized clearing condition; and the Krusell-Smith variant fits a perceived law of motion for the clearing rate (rather than capital).

---

## [Block Composition and Second-Order SSJ](@id ssj_blocks)

A single household Jacobian prices one block. A *model* is a directed acyclic graph (DAG) of blocks, and its general-equilibrium Jacobian follows from the implicit function theorem applied along a topological ordering of that graph. `combine_blocks` assembles [`HetBlock`](@ref) household problems and [`SimpleBlock`](@ref) equation blocks into an [`SSJModel`](@ref); `ssj_jacobian` accumulates their Jacobians forward into the general-equilibrium system

```math
H_U \, dU + H_Z \, dZ = 0, \qquad dU = -H_U^{-1} H_Z \, dZ
```

where:
- ``dU`` stacks the ``T``-length deviation paths of the **unknowns** --- the aggregates the equilibrium determines
- ``dZ`` stacks the paths of the exogenous **shocks**
- ``H_U``, ``H_Z`` are the total derivatives of the **targets** (equilibrium residuals, zero in steady state) with respect to the unknowns and shocks

A `SimpleBlock` maps its inputs at declared leads and lags to its outputs, so its sequence-space Jacobian is the constant partial derivative ``\partial f_o / \partial I_{i,l}`` placed on the ``l``-th diagonal --- a banded Toeplitz matrix obtained by automatic differentiation, with no simulation. A `HetBlock` returns the dense fake-news Jacobian of the previous section, anticipation entries included.

!!! note "Technical Note"
    Unknowns and shocks must be **exogenous to the DAG** --- produced by no block. Feedback loops are closed by the GE solve, not by the graph, so a cycle among blocks is an error naming the blocks involved. Targets are equilibrium residuals: a target whose steady-state level does not vanish means the model is being linearized around a point that does not clear, which `ssj_jacobian` reports as a warning.

The Huggett economy is a two-block DAG: households map ``(r, w)`` to aggregate bond holdings ``A``, and the bond market requires ``A = 0`` in zero net supply.

```@example dsge_ha
household = HetBlock(hug, ss_hug; inputs=[:r, :w], outputs=[:A], name=:household)
bond_market = SimpleBlock(x -> [x[1]];                 # bond_mkt = A, clears at zero
                          inputs=[:A], outputs=[:bond_mkt],
                          ss_inputs=Dict(:A => household.ss_outputs[:A]),
                          name=:bond_market)
model = combine_blocks(household, bond_market; name=:huggett)
report(model)
```

The DAG report lists the blocks in topological order --- households first, then the market they feed --- together with the variables the model treats as exogenous. Here ``r`` is the unknown the bond market pins down and ``w`` is the aggregate endowment shock; both are exogenous to the graph precisely because the equilibrium, not a block, determines them.

```@example dsge_ha
gej = ssj_jacobian(model; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                   T_horizon=40, target_tol=1e-3)   # SS was solved to tol=5e-4
report(gej)
```

``H_U`` is the ``40 \times 40`` derivative of aggregate bond demand with respect to the whole rate path and ``H_Z`` its derivative with respect to the endowment path. Bond demand is about six times more sensitive to the rate path than to the endowment path (``\|H_U\|_1 \approx 50`` against ``\|H_Z\|_1 \approx 7.9``), so clearing a 2% endowment shock takes a rate move of only a few tenths of a percentage point. `target_tol` is relaxed to ``10^{-3}`` because the steady state was itself computed to `tol=5e-4`: demanding tighter clearing than the linearization point delivers would fire a warning that reflects the steady-state tolerance, not the DAG.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `unknowns` | `Vector{Symbol}` | --- | Aggregates the equilibrium determines; exogenous to the DAG |
| `targets` | `Vector{Symbol}` | --- | Equilibrium residuals, one per unknown; produced by some block |
| `shocks` | `Vector{Symbol}` | --- | Exogenous drivers; exogenous to the DAG |
| `T_horizon` | `Int` | `300` | Truncation horizon ``T`` |
| `target_tol` | `Real` | ``10^{-6}`` | Warn when a target does not vanish in steady state |

| Field | Type | Description |
|-------|------|-------------|
| `H_U` | `Matrix{T}` | ``(n_{targets} T) \times (n_{unknowns} T)`` clearing Jacobian |
| `H_Z` | `Matrix{T}` | ``(n_{targets} T) \times (n_{shocks} T)`` shock Jacobian |
| `curlyJ` | `Dict{Symbol,Dict{Symbol,Matrix{T}}}` | Total derivative of every variable w.r.t. every unknown and shock |
| `H_U_fact` | `Factorization{T}` | Cached LU factorization reused by `ssj_irf` |

### Second-Order Impulse Responses

Expanding ``H(U, Z) = 0`` to second order in the shock size gives the first-order system above plus

```math
H_U \, dU^{(2)} + \tfrac{1}{2} D^2 H[v, v] = 0, \qquad v = (dU^{(1)}, dZ)
```

where ``D^2 H[v, v]`` is the second-order sequence-space Jacobian tensor **contracted with the first-order solution path**. `ssj_irf(...; order=2)` evaluates that contraction from a central difference of two nonlinear DAG passes, so the ``T^3`` tensors --- ``2.7 \times 10^7`` entries per input pair at ``T = 300`` --- are never materialized. Because ``dU^{(2)}`` scales with the square of the shock size, the second-order solution collapses onto the first-order one as the shock shrinks.

```@example dsge_ha
dw = Dict(:w => [0.02 * 0.9^(t - 1) for t in 1:40])   # 2% AR(1) endowment shock
irf1 = ssj_irf(gej, dw)
irf2 = ssj_irf(gej, dw; order=2)
(impact_1st = round(irf1.paths[:r][1], digits=6),
 impact_2nd = round(irf2.paths[:r][1], digits=6),
 residual_1st = round(irf1.target_residual[:bond_mkt], sigdigits=3),
 residual_2nd = round(irf2.target_residual[:bond_mkt], sigdigits=3))
```

A positive endowment shock lowers the clearing rate: households want to save the windfall, but the bond is in zero net supply, so the rate must fall. The second-order correction moves the impact rate from ``-0.00314`` to ``-0.00309``, about 1.7% of the first-order response --- the precautionary-saving nonlinearity the linear solution misses. `target_residual` is the honest accuracy measure: it evaluates the DAG **nonlinearly** along the returned path and reports the largest market-clearing violation, which falls from ``3.2 \times 10^{-4}`` to ``5.0 \times 10^{-6}``, a factor of 63.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `order` | `Int` | `1` | Approximation order (`1` or `2`) |
| `fd_step` | `Real` | `1.0` | Scaling of the central difference used for the second-order contraction |
| `residual` | `Bool` | `true` | Report the nonlinear market-clearing residual along the returned path |

| Field | Type | Description |
|-------|------|-------------|
| `paths` | `Dict{Symbol,Vector{T}}` | Total deviation path of every variable in the DAG |
| `first_order` | `Dict{Symbol,Vector{T}}` | First-order component alone |
| `correction` | `Dict{Symbol,Vector{T}}` | Second-order component (empty when `order = 1`) |
| `target_residual` | `Dict{Symbol,T}` | Largest nonlinear target deviation along the returned path |

---

## [Winberry (2018) Parametric Distributions](@id ha_winberry)

The Young histogram is exact but expensive: it carries one state per grid node, so a linearized model with ``N_a = 200`` and ``N_e = 7`` has 1400 distribution states. **Winberry (2018)** instead approximates the asset density *within each income state* by a low-order exponential family, so the distribution state is a handful of moments per income state. Pass `distribution=:winberry` to a spec to select it.

Within income state ``j`` the density is

```math
g_j(a) = \exp\!\Big( \sum_{i=1}^{n} \lambda_{j,i} \, (z^i - \mu_{j,i}) - \log Z_j \Big),
\qquad z = \frac{a - m_{j,1}}{\sqrt{m_{j,2}}}
```

where:
- ``m_{j,1}, m_{j,2}, \ldots, m_{j,n}`` are the mean, the variance, and the higher **central** moments of assets conditional on income state ``j``
- ``\mu_{j,i} = m_{j,i} / m_{j,2}^{i/2}`` are those moments standardized, so ``\mu_{j,1} = 0`` and ``\mu_{j,2} = 1``
- ``\lambda_{j}`` are the exponential-family coefficients that make the density reproduce ``m_j``
- ``\log Z_j`` is the log normalizer over the reference interval ``[a_{\min}, a_{\max}]``

The moment vector ``m`` *is* the distribution state, so the linearized system carries ``N_e \times n`` distribution states instead of ``N_a \times N_e``.

!!! note "Technical Note"
    Given target moments, ``\lambda`` minimizes the log normalizer ``F(\lambda) = \log \int \exp(\sum_i \lambda_i (z^i - \mu_i)) \, da``, which is strictly convex. Its gradient ``\nabla_i F = E_g[z^i] - \mu_i`` is the moment residual itself and its Hessian is ``\mathrm{Cov}_g(z^i, z^j)``, both in closed form — the Newton step uses these exact derivatives rather than automatic differentiation, and the test suite cross-checks them against `ForwardDiff`. Two numerical details matter. The normalizer is evaluated in log space with the exponent maximum subtracted, so a density peaked at the borrowing constraint never overflows. And the monomial basis is whitened under the current density at each Newton round: on the calibrated Krusell-Smith grid the top asset node sits 33 standard deviations above the mean, so ``z^4`` spans ``10^6`` and the raw Hessian has condition number ``10^8``. Whitening through a QR factorization of ``\sqrt{p} \odot B`` restores an identity Hessian without ever squaring that condition number.

### Fitting a density to moments

[`fit_parametric_density`](@ref) solves the moment problem on its own, independently of any model. The exponential distribution with rate 1 has centered moments ``(1, 1, 2, 9)``, and the maximum-entropy density matching them is the exponential itself:

```@example dsge_ha
pd = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                            n_segments=400, n_quad=6)
(converged = pd.converged,
 lambda = round.(pd.lambda, digits=8),
 fitted_at_a1 = round(parametric_density(pd, 1.0), digits=8),
 exact_at_a1 = round(exp(-1.0), digits=8))
```

The fit recovers ``\lambda = (-1, 0, 0, 0)`` — in standardized coordinates ``z = a - 1``, so ``\exp(-z) \propto \exp(-a)`` — and reproduces the density pointwise to eight digits, not merely its first four moments. [`parametric_moments`](@ref) inverts the map, returning the moments implied by a fitted ``\lambda``.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `bounds` | `Tuple` | — | Reference interval; required unless `nodes`/`weights` are given |
| `nodes`, `weights` | `Vector` | `nothing` | Explicit quadrature in asset units, e.g. from [`winberry_quadrature`](@ref) |
| `n_segments` | `Int` | `64` | Subintervals of the default rule built from `bounds` |
| `n_quad` | `Int` | `5` | Gauss-Legendre nodes per subinterval |
| `tol` | `Real` | ``10^{-10}`` | Tolerance on the largest standardized moment residual |
| `lambda_init` | `Vector` | `nothing` | Warm start; defaults to the Gaussian ``(0, -1/2, 0, \ldots)`` |

| Field | Type | Description |
|-------|------|-------------|
| `lambda` | `Vector{T}` | Exponential-family coefficients |
| `moments` | `Vector{T}` | Target centered moments |
| `center`, `scale` | `T` | Standardization, ``m_1`` and ``\sqrt{m_2}`` |
| `log_norm` | `T` | Log normalizer in asset units |
| `converged` | `Bool` | Whether the Newton solve met `tol` |
| `residual` | `T` | Largest standardized moment residual attained |

### Steady state and solution

With `distribution=:winberry` the equilibrium is still cleared on the histogram — the accurate reference — and the parametric family is fitted afterwards at the equilibrium policy as the fixed point of the *moment* law of motion. That fixed point is a different object from the moments of the histogram, and the gap between the two aggregates is the reduction's approximation error:

```@example dsge_ha
spec_w = load_ha_example(:krusell_smith; distribution=:winberry)
ss_w = compute_steady_state(spec_w; K_init=10.0, r_bounds=(-0.02, 0.04),
                             max_iter=80, tol=1e-4, n_moments=3)
(K_histogram = round(ss_w.aggregates[:K], digits=4),
 K_parametric = round(ss_w.aggregates[:K_winberry], digits=4),
 distribution_states = length(ss_w.parametric.densities) * ss_w.parametric.n_moments,
 histogram_states = spec_w.grid.total_individual_states)
```

Three moments per income state reproduce aggregate capital to about 1.8% while carrying 21 distribution states instead of 1400. Feeding that steady state to `solve(spec; method=:reiter)` builds the linearized system on the moment state, with the same general-equilibrium closure the histogram-based Reiter method uses:

```@example dsge_ha
sol_w = solve(spec_w; method=:reiter, ss=ss_w, n_moments=3)
report(sol_w)
```

Aggregate capital is *exactly* linear in the moment state, ``K = \sum_j \text{mass}_j \, m_{j,1}``, so unlike the SVD reduction the aggregator carries no approximation error at all — the whole error sits in the shape of the density. Against the Young-based Reiter system, the aggregate capital impulse response agrees to 1.8% at two moments, 0.34% at three, and 0.30% at four. [`distribution_irf`](@ref) works unchanged: the solution stores an ``N \times (N_e \cdot n)`` basis mapping moment deviations back to a histogram deviation, so full distributional dynamics remain available from the reduced system.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `distribution` | `Symbol` | `:young` | `:young` or `:winberry`; settable on the spec or per call |
| `n_moments` | `Int` | `3` | Moments carried per income state (minimum 2) |
| `n_quad` | `Int` | `4` | Gauss-Legendre nodes per asset-grid interval |
| `winberry_tol` | `Real` | ``10^{-9}`` | Tolerance for the moment fixed point, in standardized units |

[`fit_winberry`](@ref) fits the family to any histogram directly, [`winberry_moments`](@ref) extracts the conditional moments of one, and [`winberry_histogram`](@ref) renders a fitted family back onto the asset grid for plotting or node-by-node comparison.

---

## Built-in Examples

Four canonical models are available via `load_ha_example`:

| Model | Assets | Grid | Income | Key Feature |
|-------|--------|------|--------|-------------|
| `:krusell_smith` | 1 (``a \in [0, 1000]``, geometric) | 200 pts | 7 states | Standard Aiyagari economy |
| `:one_asset_hank` | 1 (``b \in [-2, 1000]``, geometric) | 200 pts | 7 states | NK with dividends, borrowing |
| `:two_asset_hank` | 2 (liquid + illiquid) | 50 × 50 | 7 states | Portfolio choice with adjustment cost |
| `:huggett` | 1 (``a \in [-2, 4]``) | 300 pts | 2 states | Pure exchange, bond in zero net supply |
| `:endogenous_labor` | 1 (``a \in [0, 2000]``, geometric) | 200 pts | 7 states | Aiyagari with GHH endogenous hours |

```@example dsge_ha
[let s = load_ha_example(name)
    (model = name, assets = s.grid.n_dims, beta = s.individual.beta,
     grid = join(s.grid.n_points, "×"))
 end for name in [:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett]]
```

The Krusell-Smith economy is the simplest benchmark with a single asset and Cobb-Douglas production. The one-asset HANK adds New Keynesian features (sticky prices, monetary policy, dividends) and allows borrowing. The two-asset HANK introduces portfolio choice between liquid and illiquid assets, capturing the empirical finding that most household wealth is illiquid.

---

## Bayesian Estimation

Bayesian estimation of HA-DSGE models uses the **linearized reduced system** (from SSJ or Reiter) with a Kalman filter for likelihood evaluation. For each parameter draw ``\theta`` in the RWMH sampler:

1. Update model parameters
2. Re-solve the HA steady state (the expensive step)
3. Linearize via SSJ → produce a reduced `DSGESolution`
4. Build the state space → evaluate Kalman log-likelihood
5. Accept/reject via the Metropolis-Hastings ratio

```julia
using Distributions
result = estimate_dsge_bayes(spec, data, [0.36];
    priors=Dict(:alpha => Beta(5, 2)),
    observables=[:K], n_draws=5000, burnin=1000,
    ha_method=:ssj, ha_kwargs=(T_horizon=300, n_reduced=15))
```

!!! note "Technical Note"
    The inner loop re-solves the HA steady state at each draw, making estimation
    computationally intensive. For a one-asset model with 200 grid points, each
    likelihood evaluation takes approximately 0.5 seconds, yielding a 5000-draw
    chain in about 40 minutes.

The estimation output is a `BayesianDSGE` object with posterior draws, acceptance rate, and log-likelihood trace. Use `report(result)` for a formatted summary including posterior means, credible intervals, and convergence diagnostics.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `priors` | `Dict{Symbol,Distribution}` | — | Prior distributions keyed by parameter name |
| `observables` | `Vector{Symbol}` | — | Observable variable names matching data columns |
| `n_draws` | `Int` | `5000` | Total RWMH draws |
| `burnin` | `Int` | `1000` | Burn-in draws to discard |
| `ha_method` | `Symbol` | `:ssj` | Aggregate solution method (`:ssj` or `:reiter`) |
| `ha_kwargs` | `NamedTuple` | `(T_horizon=300, n_reduced=15)` | `solve` options. `T_horizon` sets the SSJ truncation length; too-small values truncate persistent HA Jacobians and bias the likelihood (default follows Auclert et al. 2021) |

---

## Complete Example

A full workflow for a Krusell-Smith (1998) economy: load the model, compute the steady state, examine the wealth distribution, solve for aggregate dynamics, and simulate panel data.

```@example dsge_ha
ks = load_ha_example(:krusell_smith)
ss_ks = compute_steady_state(ks; K_init=10.0, r_bounds=(-0.02, 0.04),
                              max_iter=80, tol=1e-4)
report(ss_ks)
```

```@example dsge_ha
ineq_ks = inequality_irf(ss_ks; T_periods=5)
(gini = round(ineq_ks[:gini][1], digits=4),
 p50  = round(ineq_ks[:p50][1], digits=2),
 p90  = round(ineq_ks[:p90][1], digits=2))
```

The Gini coefficient reflects the degree of wealth concentration in the stationary equilibrium. The gap between the median and 90th percentile illustrates the right-skewed nature of the wealth distribution --- a robust feature of heterogeneous agent models with borrowing constraints and precautionary savings.

```@example dsge_ha
panel_ks = simulate_panel(ss_ks; N_agents=1000, T_periods=200)
size(panel_ks)
```

The simulated panel tracks 1000 agents over 200 periods, providing micro-level data for computing cross-sectional moments, transition matrices, and mobility statistics.

```julia
plot_result(ss_ks)                    # wealth distribution
plot_result(ss_ks; view=:lorenz)      # Lorenz curve with Gini
plot_result(ss_ks; view=:policy)      # policy functions by income
```

---

## Common Pitfalls

1. **Bisection bounds too narrow.** If `compute_steady_state` does not converge, widen `r_bounds`. The equilibrium interest rate can be negative in Aiyagari economies with patient agents.

2. **Grid too coarse near the borrowing constraint.** The EGM interpolation is least accurate near kinks in the policy function. Use `grid_type=:double_exp` (default) for denser spacing near the lower bound.

3. **Rouwenhorst vs Tauchen for persistent income.** For ``\rho > 0.95``, Rouwenhorst is significantly more accurate. Tauchen requires very fine grids to match the stationary distribution of highly persistent processes.

4. **SSJ truncation horizon too short.** If `T_horizon` is smaller than the half-life of the aggregate shock, the Jacobian is truncated prematurely. Use ``T_{\text{horizon}} \geq 3 / (1 - \rho_z)`` as a rule of thumb.

5. **Ho-Kalman `n_reduced` too small.** Check `explained_variance` in the `HADSGESolution` --- it should exceed 0.999. If not, increase `n_reduced`.

6. **Two-asset deposit grid resolution.** The nested EGM searches over a discrete deposit grid. With too few points (`n_deposit < 20`), the optimal deposit choice may be inaccurate near the adjustment cost kink.

7. **Targets that do not vanish in steady state.** `ssj_jacobian` warns when a target's steady-state level exceeds `target_tol`, which means the DAG is being linearized around a point that does not clear. The usual cause is an asset grid whose upper bound truncates the savings policy, so mass piles up at `a_max` and ``\int a' d\mu`` exceeds ``\int a \, d\mu``. Widen the grid rather than raising the tolerance.

8. **`excess_demand` cannot see grid truncation.** This is the failure mode that motivated `ha_grid_diagnostics`, and it is silent by construction. The Young (2010) transition clamps the savings policy into ``[a_{\min}, a_{\max}]``. Clamping conserves *mass* exactly, so the distribution is still a valid probability measure and still exactly stationary --- but it destroys *assets*. Because `excess_demand` is measured on the already-clamped aggregate ``\int a \, d\mu``, it can read ``10^{-7}`` while the model fails to clear by percent. For a stationary histogram the wedge is an exact identity:

    ```math
    \int a' d\mu - \int a \, d\mu = \int \max(a' - a_{\max}, 0) \, d\mu - \int \max(a_{\min} - a', 0) \, d\mu
    ```

    so a non-zero residual *is* truncation. Read `ha_grid_diagnostics(ss)` (or the **Grid Adequacy** panel that `report(ss)` prints) after every solve, and compare `aggregates[:K]` = ``\int a \, d\mu`` against `aggregates[:A_policy]` = ``\int a' d\mu``, which is what the sequence-space household block integrates. `compute_steady_state` warns by default; use `grid_check=:error` to make it fatal or `:none` to silence it.

9. **A rate with ``\beta(1+r) \geq 1`` is not an equilibrium candidate.** Aiyagari's (1994) existence condition requires ``\beta(1+r) < 1``; above it, household wealth diverges and *no* stationary distribution exists, so no finite `a_max` will help. If `compute_steady_state` warns about this, the problem is the calibration or the clearing rule, not the grid. The bisection skips such rates outright rather than solving a household problem whose answer would be an artifact of the grid ceiling.

10. **Second-order step size.** `ssj_irf(...; order=2)` differences at the actual shock size (`fd_step=1.0`). Reduce `fd_step` when the shock is large enough to push the household problem far from its steady state; raise it when the shock is so small that the ``O(\sigma^2)`` term is lost in roundoff.

11. **DCEGM asset grids must be dense near the credit limit.** A household that saves exactly the credit limit and has no income next period consumes zero forever, so the value there is ``-\infty`` and the endogenous grid cannot reach below ``\bar{a} + c(\bar{a})``. On a uniform grid that unresolved wedge is a full asset step wide; `dcegm_retirement_model` therefore defaults to `curvature=2.0`, which shrinks it by a factor of `n_a`. Give the option a positive income floor (`pension`) if you need the constrained branch itself.

12. **Comparing DCEGM against a grid solver near a kink.** DCEGM locates the switching threshold exactly, so consumption jumps at an arbitrary real number. A value-function-iteration benchmark on a finite grid cannot represent that jump and will disagree sharply at the one or two nodes straddling it, while agreeing to grid accuracy everywhere else. Compare medians and choice indicators, not maxima.

13. **Winberry accuracy is limited by the mass at the borrowing constraint, not by the moment order.** An exponential family is a density and cannot represent an atom. On the calibrated Krusell-Smith example 6.3% of households sit exactly at ``a = 0``, and aggregate capital from the parametric fixed point is off by 7.2% at two moments, 1.8% at three, and 1.1% at five --- the returns to extra moments flatten out quickly. Use `:winberry` when the size of the linearized system is the binding constraint, and check `aggregates[:K_winberry]` against `aggregates[:K]` before trusting it.

14. **High moment orders on coarse grids do not reach the fixed-point tolerance.** The residual floor of the moment map rises with the moment order and falls with grid resolution. On a 200-node grid the fixed point converges to ``10^{-9}`` through at least five moments; on an 80-node grid it stops converging above four. `compute_steady_state` warns and leaves `converged=false` on the family rather than reporting success --- lower `n_moments` or refine the grid.

---

## References

- Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89 (5): 2375--2408. [DOI](https://doi.org/10.3982/ECTA17434)

- Bhandari, Anmol, Thomas Bourany, David Evans, and Mikhail Golosov. 2023. "A Perturbational Approach for Approximating Heterogeneous-Agent Models." NBER Working Paper 31744. [DOI](https://doi.org/10.3386/w31744)

- Carroll, Christopher D. 2006. "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems." *Economics Letters* 91 (3): 312--320. [DOI](https://doi.org/10.1016/j.econlet.2005.09.013)

- Greenwood, Jeremy, Zvi Hercowitz, and Gregory W. Huffman. 1988. "Investment, Capacity Utilization, and the Real Business Cycle." *American Economic Review* 78 (3): 402--417.
- den Haan, Wouter J. 2010. "Assessing the Accuracy of the Aggregate Law of Motion in Models with Heterogeneous Agents." *Journal of Economic Dynamics and Control* 34 (1): 79--99. [DOI](https://doi.org/10.1016/j.jedc.2009.07.006)

- Iskhakov, Fedor, Thomas H. Jørgensen, John Rust, and Bertel Schjerning. 2017. "The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks." *Quantitative Economics* 8 (2): 317--365. [DOI](https://doi.org/10.3982/QE643)

- Huggett, Mark. 1993. "The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance Economies." *Journal of Economic Dynamics and Control* 17 (5--6): 953--969. [DOI](https://doi.org/10.1016/0165-1889(93)90024-M)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)

- Krusell, Per, and Anthony A. Smith Jr. 1998. "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy* 106 (5): 867--896. [DOI](https://doi.org/10.1086/250034)

- Reiter, Michael. 2009. "Solving Heterogeneous-Agent Models by Projection and Perturbation." *Journal of Economic Dynamics and Control* 33 (3): 649--665. [DOI](https://doi.org/10.1016/j.jedc.2008.08.010)

- Rouwenhorst, K. Geert. 1995. "Asset Pricing Implications of Equilibrium Business Cycle Models." In *Frontiers of Business Cycle Research*, edited by Thomas F. Cooley, 294--330. Princeton: Princeton University Press.

- Winberry, Thomas. 2018. "A Method for Solving and Estimating Heterogeneous Agent Macro Models." *Quantitative Economics* 9 (3): 1123--1151. [DOI](https://doi.org/10.3982/QE740)

- Tauchen, George. 1986. "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions." *Economics Letters* 20 (2): 177--181. [DOI](https://doi.org/10.1016/0165-1765(86)90168-0)

- Young, Eric R. 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell--Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36--41. [DOI](https://doi.org/10.1016/j.jedc.2008.11.010)
