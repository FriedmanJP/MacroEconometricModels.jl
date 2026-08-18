# [Heterogeneous Agent DSGE](@id dsge_ha)

Standard DSGE models assume a **representative agent** whose decisions aggregate to macroeconomic outcomes. In reality, households differ in wealth, income, and consumption --- heterogeneity that shapes aggregate responses to shocks, especially monetary and fiscal policy. MacroEconometricModels.jl provides a complete toolkit for **heterogeneous agent DSGE (HA-DSGE)** models: the **Endogenous Grid Method** (Carroll 2006) and **VFI** for solving individual problems, **Young (2010) histogram** tracking for the wealth distribution, and three aggregate solution methods --- **Sequence-Space Jacobian** (Auclert, Bardóczy, Rognlie & Straub 2021), **Reiter (2009) linearization**, and **Krusell-Smith (1998) simulation**. The module supports one-asset and two-asset HANK models with Bayesian estimation. This page is part of the [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) sub-hub of the [DSGE Models](@ref dsge_page) suite; for the continuous-time formulation of the same class of models, see [Continuous Time](@ref dsge_continuous).

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

**Recipe 5: Steady-state inequality statistics**

```@example dsge_ha
ineq = inequality_irf(ss; T_periods=10)
(gini = round(ineq[:gini][1], digits=4),
 p50  = round(ineq[:p50][1], digits=2),
 p90  = round(ineq[:p90][1], digits=2))
```

Called on a steady state, `inequality_irf` returns the cross-sectional Gini and wealth percentiles repeated over `T_periods` --- nothing moves, because the stationary distribution is fixed. The 90th percentile sits several times the median, the right skew that borrowing constraints and precautionary saving produce. Pass an `HADSGESolution` instead to get genuine dynamics after an aggregate shock.

**Recipe 6: Krusell-Smith simulation method**

```@example dsge_ha
ks_result = solve(spec; method=:krusell_smith, ss=ss,
                  T_sim=500, T_burn=100, max_outer=3,
                  rho_z=0.95, sigma_z=0.007)
report(ks_result)
```

---

## Individual Problem

The full model is assembled into a [`ModelSpec`](@ref) whose `agents` NamedTuple holds one or more [`HouseholdSystem`](@ref) populations (built by [`load_ha_example`](@ref) or `@dsge` with a `heterogeneous:` block). The household payload bundles the discretized [`IncomeProcess`](@ref) and the [`IndividualProblem`](@ref) — the utility, marginal utility, budget, and borrowing-constraint fields the EGM/VFI inner loops consume. The population name is a free key (`household`, `unconstrained`, `htm`, …); `solve` dispatches on the kind, never on the key. A single `HouseholdSystem` is the usual case; two named households clear through `solve(spec; method=:ssj)` after [Multiple Household Populations](@ref ha_multipop).

`@dsge` `heterogeneous:` accepts `n_grid`, `utility` (`log`, `crra`, or `crra(σ)`), `discount`, `borrowing`, `budget`, `model` (`aiyagari` or `huggett`), and `crra`/`sigma_c`. `clock:` and `horizon:` set `ModelIR` flags; `discrete:` and `absorbing:` are stored as declarations. They do not compile [`ContinuousHouseholdSystem`](@ref), [`LifeCycleSystem`](@ref), or [`DCEGMSystem`](@ref) — use `to_spec` on the family constructor.

| Key | Status | Use instead |
|-----|--------|-------------|
| `liquid=` / `illiquid=` | deferred | [`HAGrid`](@ref) or [`load_ha_example`](@ref)`(:two_asset_hank)` |
| `labor = ghh` / `separable` | deferred | [`LaborSupply`](@ref) or [`load_ha_example`](@ref)`(:endogenous_labor)` |
| option-specific ``u(c, d)`` | deferred | [`DCEGMProblem`](@ref) / [`dcegm_retirement_model`](@ref) |

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

The steady state report displays convergence diagnostics, equilibrium prices, aggregate quantities, and wealth distribution statistics. The Euler error (in ``\log_{10}`` units) measures the accuracy of the consumption policy: the largest residual of the Euler equation over the evaluated states.

!!! warning "The Euler error is measured off-node, and the number is not comparable to node-based figures"
    `ss.euler_error` evaluates the residual at the cell **midpoints**, where the policy is genuinely an interpolant. Evaluating at the grid **nodes** — the convention this package used before v0.7.2, and the one behind most published figures — measures interpolation round-trip error rather than approximation error, because EGM *solves* the Euler equation at exactly those points. The gap is large: the shipped one-asset examples report ``-2.25`` (Krusell--Smith), ``-2.28`` (one-asset HANK) and ``-1.94`` (Huggett) off-node against ``-6.04``, ``-6.06`` and ``-4.47`` at the nodes — 2.5 to 3.8 ``\log_{10}`` units. Both statistics are kept on `ss.euler`, and `compute_steady_state(spec; euler_points=:nodes)` restores the old headline number.

    Cells whose savings policy leaves the grid (``a' > a_{\max}``) are **excluded and counted** rather than scored. `_linear_interp` flat-extrapolates above ``a_{\max}``, so the continuation consumption there is looked up at the same clamped point the solver used and the residual collapses to machine precision — which made a *truncating* model report its best accuracy exactly where it was broken.

```@example dsge_ha
(euler_error   = round(ss.euler_error; digits=4),          # off-node maximum
 mean_residual = round(ss.euler.midpoints.mean; digits=4), # ... and the mean
 at_nodes      = round(ss.euler.nodes.max; digits=4),      # the old convention
 excluded      = ss.euler.midpoints.n_offgrid)
```

### VFI with Howard Improvement

When EGM is not applicable (non-separable utility, complex constraints), **Value Function Iteration** with **Howard improvement steps** provides a robust alternative. Each VFI iteration consists of one policy maximization step followed by ``K`` policy-evaluation steps (default ``K = 20``), which are cheap linear operations that dramatically accelerate convergence. Pass `hh_solver=:vfi` to `compute_steady_state` (and to `solve`, which forwards the flag into the stationary problem) to use `_vfi_solve` instead of EGM. The default remains `:egm`. `hh_solver=:vfi` writes the Bellman value into `ss.value_fn`. The default EGM path now recovers `value_fn` afterwards by Howard policy evaluation of the equilibrium policy. One-asset VFI includes GHH and separable labor. Two-asset household VFI (`_two_asset_vfi_solve`) is the inner kernel when `hh_solver=:vfi` on a two-asset spec. `solve(...; method=:reiter, hh_solver=:vfi)` finite-differences the VFI policy. SSJ fake-news stays on EGM. `method=:krusell_smith` with `:vfi` throws (one-asset 4-D and two-asset per-period PE both use EGM).

```@example dsge_ha
spec_vfi = MacroEconometricModels._huggett_example(; n_a=40)
ss_vfi = compute_steady_state(spec_vfi; hh_solver=:vfi, max_iter=80,
                              tol=5e-3, grid_check=:none)
(isfinite_r = isfinite(ss_vfi.prices[:r]),
 V_increasing = ss_vfi.value_fn[end, 1] > ss_vfi.value_fn[1, 1])
```

!!! note "Two-asset steady states"
    One-asset models still bisect a single rate. Two-asset models use a damped `(K, r_b)` closer (the discrete-time analogue of `ct_two_asset_ge`): illiquid wealth clears against firm capital and liquid wealth against `B_supply`, with `τ = r_b B_supply`. The default `load_ha_example(:two_asset_hank)` grid is 50 × 50 × 7; shrink it for interactive work. `distribution=:winberry` on a two-asset spec still errors.

---

## [The Asset Grid](@id ha_grid)

The asset grid is the discretization choice that most affects an HA-DSGE solution. Two decisions matter, and they trade off against each other on one grid shape but not the other.

**The ceiling must not bind.** The Young (2010) transition clamps the savings policy into ``[a_{\min}, a_{\max}]``, so any mass that wants to save past ``a_{\max}`` is silently pushed back onto the top node --- see Common Pitfalls. Choose ``a_{\max}`` from the *right tail* of the wealth distribution, not from the mean: the built-in examples run to ``a_{\max} = 1000`` against an equilibrium ``K \approx 42``, a ratio of about ``24``.

**Raising the ceiling must not cost resolution at the constraint.** The `:double_exp` grid is a fixed curve rescaled by ``(a_{\max} - a_{\min})``, so its bottom spacing is *linear* in ``a_{\max}`` --- widening the grid five-fold coarsens the borrowing constraint five-fold. The `:geometric` grid is equidistant in ``\log(a - a_{\min} + \text{pivot})``, so its bottom spacing grows only logarithmically. Measured on the Krusell--Smith calibration at ``n_a = 200``:

| Grid | ``a_{\max}`` | First step ``\Delta a_1`` | Mass at floor | Mass at ceiling | Euler error (off-node) | (at nodes) |
|------|--------------|---------------------------|---------------|-----------------|------------------------|------------|
| `:double_exp` | 200 | 0.2208 | 6.41% | ``7.1 \times 10^{-3}`` | ``-1.76`` | ``-4.54`` |
| `:double_exp` | 1000 | 1.1039 | 8.15% | ``0`` | ``-1.79`` | ``-4.21`` |
| `:geometric` | 200 | 0.0085 | 6.12% | ``8.7 \times 10^{-3}`` | ``-2.32`` | ``-6.09`` |
| `:geometric` | 1000 | 0.0106 | 6.31% | ``4.4 \times 10^{-13}`` | ``-2.25`` | ``-6.04`` |

`:geometric` clears the ceiling and is the more accurate curve on both conventions --- roughly half a ``\log_{10}`` unit off-node and a full 1.8 units at the nodes. This is why the built-in one-asset examples use `grid_type=:geometric`. The default remains `:double_exp` for backward compatibility.

The two conventions disagree about what widening `:double_exp` costs. At the nodes it looks like a real degradation, ``-4.54 \to -4.21``; off-node the two are indistinguishable, ``-1.76 \to -1.79``. The node reading was an artifact: widening coarsens the grid at the constraint, which inflates the round-trip error at the nodes without changing how well the policy is approximated between them. Choose ``a_{\max}`` from the ceiling mass, not from either Euler error.

Grid *resolution* is a separate matter, and refining ``n_a`` does **not** fix a binding ceiling. On the truncating configuration, sweeping ``n_a`` from 100 to 1600 at fixed ``a_{\max} = 200`` moves the relative clearing residual only from ``1.6844\%`` to ``1.6848\%`` --- unchanged to four significant figures across a 16-fold refinement, while the solve time rises from 2.4 s to 47.6 s. Only ``a_{\max}`` fixes truncation.

### Adapting the Grid to the Density

`:geometric` and `:double_exp` are fixed curves chosen before the density is known. `adapt_ha_grid` instead solves once, reads the stationary distribution, and re-places the nodes where that density actually bends. Both the Young (2010) histogram and the EGM policy are piecewise linear on the asset grid, so their error on a cell of width ``h`` scales like ``h^2 |p''|``; equidistributing ``q = |p''|^{1/2}`` equalizes the *error* per cell rather than the *width* per cell (de Boor 1973). The monitor is

```math
M(a) = (1 - \lambda) \frac{1}{a_{\max} - a_{\min}} + \lambda \frac{q(a)}{\int q}, \qquad q = |p''|^{1/2},
```

where:
- ``p`` is the marginal asset density (histogram mass divided by the cell width)
- ``\lambda`` is `curvature`, the share of nodes allocated by curvature

and the new nodes sit at equal increments of ``\int M``. Both components integrate to 1 over the domain, so ``\lambda`` has an exact reading and ``\lambda = 0`` returns a uniform grid.

```@example dsge_ha
spec_adapted = adapt_ha_grid(spec, ss)

a_old = only(values(spec.agents)).grid.grids[1]
a_new = only(values(spec_adapted.agents)).grid.grids[1]
cutoff = a_old[findfirst(>=(0.99), cumsum(vec(sum(ss.distribution; dims=2))))]

(mass_99_below = round(cutoff; digits=1),
 nodes_before  = count(<=(cutoff), a_old),
 nodes_after   = count(<=(cutoff), a_new))
```

99% of the mass sits below ``a = 213.7`` --- about a fifth of the ``[0, 1000]`` domain --- and adaptation moves 175 of the 200 nodes there, against 163 on the shipped `:geometric` grid. Feed `spec_adapted` back to `compute_steady_state` to re-solve on the new grid.

Measured against an ``n_a = 1600`` `:geometric` reference (``r = 0.00772022``, ``K = 42.39574``) at ``n_a = 200`` with default settings:

| Grid | ``|\Delta r|`` | ``|\Delta K| / K`` |
|------|----------------|--------------------|
| `:linear` | ``2.21 \times 10^{-4}`` | ``1.07 \times 10^{-2}`` |
| `:double_exp` | ``5.11 \times 10^{-5}`` | ``2.44 \times 10^{-3}`` |
| `:geometric` (shipped) | ``2.01 \times 10^{-5}`` | ``9.62 \times 10^{-4}`` |
| `adapt_ha_grid` (defaults) | ``1.68 \times 10^{-5}`` | ``8.05 \times 10^{-4}`` |
| `adapt_ha_grid`, `monitor_cap=Inf` | ``2.92 \times 10^{-4}`` | ``1.41 \times 10^{-2}`` |

The gain over `:geometric` is modest because that curve is already tuned for this density shape; adaptation earns its keep when the shape is not known in advance. Adapt **once** --- a second round re-derives the monitor from an already-adapted grid and degrades ``|\Delta r|`` to ``4.31 \times 10^{-5}``.

!!! warning "Never disable `monitor_cap` on a model with a borrowing constraint"
    The stationary distribution has an **atom** at the constraint, and a histogram atom in a cell of width ``w`` reports ``|p''| \sim 1/w^2`` --- a discretization artifact, not curvature. Uncapped, that single node attracts 153 of 200 grid points into the bottom 0.5% of the domain and the grid becomes the *worst* of the five above. The default `monitor_cap=3.0` caps the monitor at three times its positive median.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_points` | `Vector{Int}` | `grid.n_points` | Node count per asset dimension; defaults to the current grid |
| `curvature` | `Real` | `0.9` | Share ``\lambda`` of nodes allocated by curvature rather than uniformly |
| `monitor_cap` | `Real` | `3.0` | Cap on the monitor, as a multiple of its positive median |
| `smoothing` | `Int` | `2` | Smoothing passes applied to the monitor before integrating it |

!!! note "The Euler error does not rank grids"
    The adapted grid reports a *worse* Euler error than `:geometric` on both conventions --- ``-1.76`` against ``-2.25`` off-node, ``-4.67`` against ``-6.04`` at the nodes --- while being *closer* to the high-resolution reference on both ``r`` and ``K``. Moving the honest metric off-node (issue #508) shrank the gap but did not reverse it. Rank grids against a refined reference solution, not against either Euler statistic.

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

[`dcegm_retirement_model`](@ref) builds the canonical retirement problem: work or retire (absorbing), flow utility ``\log c - \delta \mathbb{1}[\text{work}]``, and no income once retired. `solve(to_spec(retire))` dispatches to [`dcegm_solve`](@ref) on the wrapped [`DCEGMSystem`](@ref). Stationary GE is [`dcegm_steady_state`](@ref); aggregate dynamics are [`dcegm_mit`](@ref) (`method=:mit`), not sequence-space Jacobians, because the upper envelope is non-differentiable at discrete-choice thresholds. `irf(eq, H)` wraps that MIT path as an [`ImpulseResponse`](@ref).

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

`rouwenhorst(rho, sigma, n; sigma_is=:innovation)` takes three positional arguments --- persistence ``\rho``, a standard deviation, and the number of states --- and one keyword:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `sigma_is` | `Symbol` | `:innovation` | Whether `sigma` is the innovation ``\sigma_\varepsilon`` or the unconditional ``\mathrm{sd}(y_t)`` |

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

`tauchen(rho, sigma, n; m=3, sigma_is=:innovation)` adds one keyword to the same three positional arguments:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `m` | `Real` | `3` | Number of standard deviations to cover |
| `sigma_is` | `Symbol` | `:innovation` | Same convention switch as `rouwenhorst` |

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
| `r_bounds` | `Tuple{T,T}` | `nothing` | Bisection bounds; defaults to `(-0.01, 0.04)`, or `(-0.05, 1/β - 1 - 10^{-4})` when the household `model === :huggett` |
| `max_iter` | `Int` | `200` | Maximum bisection iterations |
| `tol` | `Real` | ``10^{-8}`` | Convergence tolerance on excess demand |
| `grid_check` | `Symbol` | `:warn` | Grid-adequacy check: `:warn`, `:error`, or `:none` |
| `euler_points` | `Symbol` | `:midpoints` | Where the headline Euler residual is measured (`:midpoints` or `:nodes`) |
| `verbose` | `Bool` | `false` | Print iteration progress |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether bisection converged |
| `iterations` | `Int` | Number of bisection iterations |
| `prices` | `Dict{Symbol,T}` | Equilibrium prices (``r``, ``w``) |
| `aggregates` | `Dict{Symbol,T}` | Aggregate quantities (``K``, ``Y``, ``C``) |
| `distribution` | `Array{T}` | Stationary distribution; ``N_a \times N_e`` for one asset, three-dimensional for two |
| `policies` | `Dict{Symbol,Array{T}}` | Policy functions (`:consumption`, `:savings`, and `:labor` under endogenous hours) |
| `value_fn` | `Array{T}` | Value function on the same grid as `distribution` |
| `income` | `IncomeProcess{T}` | Discretized productivity process |
| `excess_demand` | `T` | Final excess demand for capital |
| `euler_error` | `T` | ``\log_{10}`` maximum Euler residual, measured off-node |
| `euler` | `NamedTuple` | `(midpoints=…, nodes=…)`, each with `max`, `mean`, `n_evaluated`, `n_constrained`, `n_offgrid` |
| `parametric` | `WinberryFamily{T}` or `nothing` | Fitted parametric family when `distribution=:winberry` |
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

The SSJ method reduces the full sequence-space representation (dimension ``T``) to a compact state-space form with `n_reduced` states. The explained variance measures the fraction of aggregate dynamics captured by the truncated Ho-Kalman basis --- values above 99.9% confirm the reduction is adequate. The underlying steady state is reported alongside the reduction diagnostics. `historical_decomposition(sol_ssj, data, [:K])` (or `[:r]` on a Huggett solution) maps the Kalman smoother through `C_obs`, so the reported series is the aggregate, not the reduced state.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_horizon` | `Int` | `300` | Truncation horizon for sequences |
| `n_reduced` | `Int` | `30` | Reduced state-space dimension (Ho-Kalman) |

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

Because Reiter keeps a real distribution basis, the reduced solution maps back onto the histogram, so the full distributional response to an aggregate shock is available — the wealth histogram deviating period by period, and the induced Gini and percentile paths:

```julia
plot_result(sol_reiter; horizon=16, max_bins=50)      # distribution dynamics
plot_result(sol_reiter; view=:inequality, horizon=16) # Gini and percentile paths
```

```@raw html
<iframe src="../assets/plots/ha_distribution_dynamics.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ha_inequality.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Neither view is available from an `:ssj` solution: the Ho-Kalman realization's coordinates are abstract minimal-realization states with no map back to the ``(a, e)`` histogram, so `distribution_irf` raises an error there and directs you to `:reiter`.

The Reiter method produces an equivalent reduced state-space form to SSJ. The explained variance confirms that 15 reduced states capture nearly all aggregate dynamics. The two methods yield identical IRFs up to numerical precision for the same `n_reduced`, but differ in computational cost: SSJ is faster for models with few aggregate inputs, while Reiter scales better when many prices affect household decisions.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `n_reduced` | `Int` | `30` | Maximum reduced dimension |

The finite-difference step used to probe the distribution (``10^{-6}``) is fixed by the internal linearizer and is not reachable through `solve`.

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
| `rho_e` | `Real` | `0.9` | Idiosyncratic persistence used in the simulation; falls back to the household `het_params[:rho_e]` |
| `sigma_e` | `Real` | `0.01` | Idiosyncratic innovation size; falls back to the household `het_params[:sigma_e]` |

| Field | Type | Description |
|-------|------|-------------|
| `converged` | `Bool` | Whether PLM coefficients converged |
| `iterations` | `Int` | Number of outer PLM iterations |
| `plm_coefficients` | `Dict{Symbol,Vector{T}}` | PLM regression coefficients per variable |
| `r_squared` | `Dict{Symbol,T}` | ``R^2`` of PLM regression per variable |
| `steady_state` | `HASteadyState{T}` | Underlying stationary equilibrium |
| `spec` | `ModelSpec{T}` | Model solved, carried for `den_haan_test` |

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

!!! note "Capital models only"
    `den_haan_test` targets the Aiyagari capital models (`model = :aiyagari`), which covers both `:krusell_smith` and `:one_asset_hank`. For a Huggett solution the cleared aggregate is the risk-free rate, which is driven by the wealth distribution rather than the shock alone, so the test raises an informative error.

### Linearized solutions

`den_haan_test` also accepts an [`HADSGESolution`](@ref) from `:ssj` or `:reiter`. A linearized solution has no fitted aggregate law, so one is recovered from the solution itself: the reduced system is simulated for `T_fit` periods under AR(1) TFP innovations and ``\log K_{t+1}`` is regressed on ``(1, \log K_t, z_t)``. That two-state rule then goes through the identical comparison. The `source` field records which law was used --- `:plm` or `:linear`.

One subtlety is load-bearing. The Krusell-Smith machinery prices the aggregate shock as **effective capital**, ``r = \alpha Z (K e^{z})^{\alpha-1} L^{1-\alpha} - \delta``, whereas the linearizations put ``z`` in **TFP**, ``Z \leftarrow Z e^{z}``. These are not the same shock and are not related by rescaling ``z``: under Cobb-Douglas the first gives ``\partial \log r/\partial z = \alpha - 1 = -0.64`` and ``\partial \log w/\partial z = \alpha = 0.36``, the second gives ``1`` and ``1``. A law fitted under one convention must therefore be simulated under that same convention, or the statistic measures the mismatch instead of the accuracy. `den_haan_test` selects the convention from the solution type.

!!! warning "Read this before quoting the linearized number"
    For a linearized solution the statistic is **much larger than the Krusell-Smith one and is method-dependent**. On `:krusell_smith` at ``\sigma_z = 0.007``: ``\varepsilon_{\max}`` is 0.07% for the fitted PLM but 12.2% for `:ssj` and 5.5% for `:reiter` --- two solutions of the *same* model differing by more than a factor of two.

    Three errors are superimposed and this statistic does not separate them: the two-state aggregate law, the linearization itself, and the law being *recovered by regression* rather than solved as a fixed point. The ``\sigma`` comparison is the more interpretable output --- both methods show the reference cross-section more volatile than their own law predicts (ratios 2.3 and 1.3), exactly the Den Haan (2010) point that a high ``R^2`` hides a volatility miss. Treat it as a **relative** diagnostic, not an absolute certificate. Why the two linearizations disagree this much is unresolved.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `T_sim` | `Int` | `10000` / `2000` | Simulation length; the second value is the `HADSGESolution` default |
| `T_burn` | `Int` | `1000` / `200` | Burn-in periods to discard; the second value is the `HADSGESolution` default |
| `rho_z` | `Real` | `0.95` | Aggregate shock persistence |
| `sigma_z` | `Real` | `0.007` | Aggregate shock standard deviation |
| `T_fit` | `Int` | `4000` | Periods used to recover the implied law (`HADSGESolution` only) |
| `seed` | `Int` | `98765` | Seed for the shock path, so the statistic is reproducible |

| Field | Type | Description |
|-------|------|-------------|
| `source` | `Symbol` | `:plm` (fitted Krusell-Smith law) or `:linear` (recovered from `:ssj`/`:reiter`) |
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
- ``\chi(d, a) = \tfrac{1}{2}\,(d/\bar a)^2\,\bar a`` is the convex adjustment cost the shipped example uses, with ``\bar a = \max(a, 0.01)`` flooring the deposit *rate* so it stays finite at ``a = 0``
- ``r_b, r_a`` are liquid and illiquid returns

The cost is quadratic in the deposit rate and therefore smooth at ``d = 0``. The continuous-time module carries the Kaplan-Moll-Violante linear-plus-convex alternative, whose kink at zero generates a genuine inaction region — see [Continuous Time](@ref dsge_continuous).

The individual problem is solved via **nested EGM**: an outer loop over deposit choices with an inner EGM on the liquid dimension.

```@example dsge_ha
spec_2a = MacroEconometricModels._two_asset_hank_example(;
    n_liquid=8, n_illiquid=6, n_e=2, B_supply=1.0)
ss_2a = compute_steady_state(spec_2a; max_iter=12, tol=5e-2, grid_check=:none)
(n_dims = only(values(spec_2a.agents)).grid.n_dims,
 has_r_b = haskey(ss_2a.prices, :r_b),
 has_A = haskey(ss_2a.aggregates, :A),
 has_B = haskey(ss_2a.aggregates, :B))
```

The shipped `load_ha_example(:two_asset_hank)` grid is 50 × 50 × 7. The example above uses a coarse grid so the closer finishes in the docs build. Because the adjustment cost is quadratic and therefore differentiable at ``d = 0``, its marginal cost passes smoothly through zero: every household whose marginal valuations differ rebalances, by an amount that shrinks continuously to zero as ``V_a/V_b \to 1``. There is no inaction band here — generating one requires a cost with a kink at the origin, which is the `cost=:kinked` specification documented under [Continuous Time](@ref dsge_continuous).

`solve(spec_2a; method=:ssj, ss=ss_2a)` and `method=:reiter` linearize around that stationary point. `method=:krusell_smith` re-solves the two-asset household each period and fits a PLM for ``K``.

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

The model is selected by the household `model == :huggett` field, which routes `compute_steady_state` to the zero-net-supply clearing rule (no firm FOC). Calibration follows Huggett (1993): CRRA utility (``\sigma = 1.5``), ``\beta = 0.99322``, and six model periods per year.

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
| `max_iter` | `Int` | `100` | Newton iteration cap |
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
 histogram_states = only(values(spec_w.agents)).grid.total_individual_states)
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

Five canonical models are available via `load_ha_example`:

| Model | Assets | Grid | Income | Key Feature |
|-------|--------|------|--------|-------------|
| `:krusell_smith` | 1 (``a \in [0, 1000]``, geometric) | 200 pts | 7 states | Standard Aiyagari economy |
| `:one_asset_hank` | 1 (``b \in [-2, 1000]``, geometric) | 200 pts | 7 states | NK with dividends, borrowing |
| `:two_asset_hank` | 2 (liquid + illiquid) | 50 × 50 | 7 states | Portfolio choice with adjustment cost |
| `:huggett` | 1 (``a \in [-2, 4]``) | 300 pts | 2 states | Pure exchange, bond in zero net supply |
| `:endogenous_labor` | 1 (``a \in [0, 2000]``, geometric) | 200 pts | 7 states | Aiyagari with GHH endogenous hours |

```@example dsge_ha
[let s = load_ha_example(name)
    hh = only(values(s.agents))
    (model = name, assets = hh.grid.n_dims, beta = hh.individual.beta,
     grid = join(hh.grid.n_points, "×"))
 end for name in [:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett,
                  :endogenous_labor]]
```

The Krusell-Smith economy is the simplest benchmark with a single asset and Cobb-Douglas production. The one-asset HANK adds New Keynesian features (sticky prices, monetary policy, dividends) and allows borrowing; its lower ``\beta = 0.986`` is what puts its equilibrium rate above Krusell-Smith's. The two-asset HANK introduces portfolio choice between liquid and illiquid assets, capturing the empirical finding that most household wealth is illiquid. Huggett is the only one of the five with no production — its two-state endowment and tight ``[-2, 4]`` grid make it the fastest model on the page. The endogenous-labor example is Krusell-Smith with GHH hours and a wider ceiling, since labor income raises the savings target.

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

## [Multiple Household Populations](@id ha_multipop)

`solve` accepts more than one [`HouseholdSystem`](@ref) on `spec.agents`. Names are free; kinds are types — do not key on `:household`. Each population becomes its own [`HetBlock`](@ref) (or [`MitBlock`](@ref) for life-cycle and continuous-time households). Agent first-order conditions never enter Gensys.

```@example dsge_ha
s1 = load_ha_example(:huggett)
s2 = load_ha_example(:huggett)
hh1 = only(values(s1.agents))
hh2 = only(values(s2.agents))
two = ModelSpec{Float64}(
    Symbol[], Symbol[], s1.params, copy(s1.param_values),
    NamedEquation[], Function[], 0, Int[], Float64[];
    agents=(unconstrained=hh1, htm=hh2))
(n_pop = length(two.agents),
 kinds = (has_kind(two, HouseholdSystem), has_kind(two, FirmSystem)))
```

Two Huggett bond economies share the same income process and differ only by name. `has_kind` is true for `HouseholdSystem` and false for `FirmSystem`. `solve(two; method=:ssj)` builds the DAG through `combine_blocks` without calling `_hh`.

Life-cycle and continuous-time households enter the same DAG as a [`MitBlock`](@ref) — a finite-difference Jacobian of `lifecycle_transition` or `ct_mit_shock`. Discrete-continuous EGM is MIT-only: constructing `HetBlock` from a [`DCEGMSystem`](@ref) throws because the upper envelope jumps at switching thresholds.

---

## Plant Heterogeneity (Khan–Thomas)

[`FirmSystem`](@ref) is a plant grid with idiosyncratic productivity and a nonconvex fixed cost of investment, not a household budget with another name. Khan and Thomas (2008) is the first paper: establishments face persistent plant-specific and aggregate TFP, and (S,s) inaction. Hopenhayn (1992) is a later entry/exit industry equilibrium with no aggregate shock.

```@example dsge_ha
plants = khan_thomas_example(; n_k=12, n_eps=3)
spec_kt = to_spec(plants)
(kind = has_kind(spec_kt, FirmSystem),
 n_k = length(plants.k_grid),
 n_eps = length(plants.productivity.states))
```

`khan_thomas_steady_state` and `khan_thomas_mit` compute the stationary distribution and a TFP impulse of aggregate output. `solve(spec_kt)` dispatches on `FirmSystem` and does not route through household EGM.

---

## Heterogeneous Banks (Bewley Banks)

[`IntermediarySystem`](@ref) is the Jamilov and Monacelli (2026) incomplete-markets bank: net worth ``n`` on an [`HAGrid`](@ref), transitory return ``\xi``, the Gertler–Karadi incentive constraint ``\lambda \ell \le V``, and convex operating costs that break scale invariance. Gertler and Karadi (2011) is the nested representative special case (``\zeta_1 = 0``, degenerate ``\xi``), not a separate type.

```@example dsge_ha
banks = IntermediarySystem(; n_min=0.08, n_max=6.0, n_n=21, n_xi=3,
                           rho_xi=0.55, sigma_xi=0.08, kappa=1.0,
                           beta=0.99, sigma=0.94, lambda=0.20,
                           zeta1=0.02, zeta2=2.0, R=1.01, rk=0.05)
spec_bb = to_spec(banks)
(kind = has_kind(spec_bb, IntermediarySystem),
 n_endog = spec_bb.n_endog,
 xi_states = length(banks.xi.states))
```

`n_endog = 0` is partial GE. [`intermediary_steady_state`](@ref) clears the credit market; [`intermediary_mit`](@ref) traces aggregate lending after a TFP path. A bank is not a `HouseholdSystem` and has no consumption-savings EGM.

---

## [OccBin on HA Aggregates](@id ha_occbin)

[`occbin_solve`](@ref) is a piecewise-linear algorithm on residual `ModelSpec`s. A `HouseholdSystem` solution is an [`HADSGESolution`](@ref): `G1` lives on `linear_solution`, not on the wrapper, and shipped examples (`load_ha_example(:one_asset_hank)`) have a real rate, not a Taylor ``i``. Calling `occbin_solve` on those specs throws a named `ArgumentError` rather than `getfield` on a missing `G1`. An ELB fixture needs a HANK with a nominal Taylor residual and `n_endog > 0`. Continuous-time state constraints stay in the HJB — they are not OccBin.

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

2. **Grid too coarse near the borrowing constraint.** The EGM interpolation is least accurate near the kink in the policy function. Prefer `grid_type=:geometric`, whose bottom spacing grows only logarithmically in ``a_{\max}``; the `:double_exp` default is a fixed curve rescaled by ``a_{\max} - a_{\min}``, so widening the grid coarsens the constraint proportionally. All built-in one-asset examples set `:geometric` for this reason — see [The Asset Grid](@ref ha_grid).

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

15. **Two household populations are not two keys on one `HouseholdSystem`.** `solve` needs two `HouseholdSystem` values in `spec.agents`. Reusing `_hh` (the unique-household accessor) on that spec throws and names `agents_of`.

16. **`occbin_solve` on a shipped HANK is not an ELB.** Real-rate examples and `n_endog = 0` PE specs have no Taylor residual. The call errors by name; it does not linearize the dummy Cobb--Douglas block.

---

## References

- Aiyagari, S. Rao. 1994. "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109 (3): 659--684. [DOI](https://doi.org/10.2307/2118417)

- Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models." *Econometrica* 89 (5): 2375--2408. [DOI](https://doi.org/10.3982/ECTA17434)

- Bhandari, Anmol, Thomas Bourany, David Evans, and Mikhail Golosov. 2023. "A Perturbational Approach for Approximating Heterogeneous-Agent Models." NBER Working Paper 31744. [DOI](https://doi.org/10.3386/w31744)

- de Boor, Carl. 1973. "Good Approximation by Splines with Variable Knots II." In *Conference on the Numerical Solution of Differential Equations*, edited by G. A. Watson, 12--20. Lecture Notes in Mathematics 363. Berlin: Springer. [DOI](https://doi.org/10.1007/BFb0069121)

- Carroll, Christopher D. 2006. "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems." *Economics Letters* 91 (3): 312--320. [DOI](https://doi.org/10.1016/j.econlet.2005.09.013)

- Greenwood, Jeremy, Zvi Hercowitz, and Gregory W. Huffman. 1988. "Investment, Capacity Utilization, and the Real Business Cycle." *American Economic Review* 78 (3): 402--417.

- den Haan, Wouter J. 2010. "Assessing the Accuracy of the Aggregate Law of Motion in Models with Heterogeneous Agents." *Journal of Economic Dynamics and Control* 34 (1): 79--99. [DOI](https://doi.org/10.1016/j.jedc.2008.12.009)

- Huggett, Mark. 1993. "The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance Economies." *Journal of Economic Dynamics and Control* 17 (5--6): 953--969. [DOI](https://doi.org/10.1016/0165-1889(93)90024-M)

- Iskhakov, Fedor, Thomas H. Jørgensen, John Rust, and Bertel Schjerning. 2017. "The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks." *Quantitative Economics* 8 (2): 317--365. [DOI](https://doi.org/10.3982/QE643)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)

- Krusell, Per, and Anthony A. Smith Jr. 1998. "Income and Wealth Heterogeneity in the Macroeconomy." *Journal of Political Economy* 106 (5): 867--896. [DOI](https://doi.org/10.1086/250034)

- Reiter, Michael. 2009. "Solving Heterogeneous-Agent Models by Projection and Perturbation." *Journal of Economic Dynamics and Control* 33 (3): 649--665. [DOI](https://doi.org/10.1016/j.jedc.2008.08.010)

- Rouwenhorst, K. Geert. 1995. "Asset Pricing Implications of Equilibrium Business Cycle Models." In *Frontiers of Business Cycle Research*, edited by Thomas F. Cooley, 294--330. Princeton: Princeton University Press.

- Tauchen, George. 1986. "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions." *Economics Letters* 20 (2): 177--181. [DOI](https://doi.org/10.1016/0165-1765(86)90168-0)

- Winberry, Thomas. 2018. "A Method for Solving and Estimating Heterogeneous Agent Macro Models." *Quantitative Economics* 9 (3): 1123--1151. [DOI](https://doi.org/10.3982/QE740)

- Young, Eric R. 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell--Smith Algorithm and Non-Stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36--41. [DOI](https://doi.org/10.1016/j.jedc.2008.11.010)

- Gertler, Mark, and Peter Karadi. 2011. "A Model of Unconventional Monetary Policy." *Journal of Monetary Economics* 58 (1): 17--34. [DOI](https://doi.org/10.1016/j.jmoneco.2010.10.004)

- Jamilov, Rustam, and Tommaso Monacelli. 2026. "Bewley Banks." *Review of Economic Studies* 93 (3): 1889--1925. [DOI](https://doi.org/10.1093/restud/rdaf062)

- Khan, Aubhik, and Julia K. Thomas. 2008. "Idiosyncratic Shocks and the Role of Nonconvexities in Plant and Aggregate Investment Dynamics." *Econometrica* 76 (2): 395--436. [DOI](https://doi.org/10.1111/j.1468-0262.2008.00837.x)
