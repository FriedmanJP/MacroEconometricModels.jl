# [Continuous Time](@id dsge_continuous)

Continuous-time heterogeneous-agent models solved with the finite-difference methods of **Achdou, Han, Lasry, Lions & Moll (2022)**. The household problem is a **Hamilton-Jacobi-Bellman (HJB)** partial differential equation solved by an implicit upwind scheme, and the stationary wealth distribution solves the **Kolmogorov-Forward (Fokker-Planck)** equation. The elegance of the approach is that a single sparse infinitesimal generator ``A`` drives both: the HJB implicitly, and the KFE through its transpose ``A^\top``.

This page is part of the [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) sub-hub of the [DSGE Models](@ref dsge_page) suite and covers the one-asset Aiyagari model. The same machinery is the foundation for two-asset (Kaplan-Moll-Violante) models and MIT-shock transitions.

## Quick Start

```@setup ct
using MacroEconometricModels
```

**Recipe 1: Stationary equilibrium**

```@example ct
m = CTAiyagari(; alpha=0.36, rho=0.05, sigma=2.0, delta=0.05,
                 z=[0.1, 0.2], lambda=[0.5, 0.5], a_max=30.0, I=200)
ss = ct_steady_state(m; tol=1e-5)
report(ss)
```

`solve(to_spec(m))` is the same stationary equilibrium: `to_spec` wraps the household as a [`ContinuousHouseholdSystem`](@ref) and `solve` dispatches to [`ct_steady_state`](@ref) (or [`ct_two_asset_ge`](@ref) for [`CTTwoAsset`](@ref)).

**Recipe 2: Incomplete markets depress the interest rate**

```@example ct
(equilibrium_r = round(ss.r, digits=5),
 discount_rate = m.rho,
 below_rho = ss.r < m.rho)
```

**Recipe 3: More risk raises precautionary saving**

```@example ct
low_risk  = ct_steady_state(CTAiyagari(; z=[0.13, 0.17], I=200); tol=1e-5)
high_risk = ct_steady_state(CTAiyagari(; z=[0.05, 0.25], I=200); tol=1e-5)
(r_low_risk = round(low_risk.r, digits=5), r_high_risk = round(high_risk.r, digits=5))
```

---

## The HJB Equation

A household with wealth ``a`` and labor productivity ``z`` solves

```math
\rho\, v(a,z) = \max_{c}\; u(c) + \partial_a v(a,z)\,\bigl(w z + r a - c\bigr) + \sum_{z'} \lambda_{z \to z'}\,\bigl[v(a,z') - v(a,z)\bigr]
```

where:
- ``\rho`` is the discount rate and ``u(c) = c^{1-\sigma}/(1-\sigma)`` is CRRA utility
- ``w z + r a - c`` is the drift of wealth (saving)
- ``\lambda_{z \to z'}`` are the Poisson switching intensities of the two-state income process
- the state constraint ``a \geq a_{\min}`` imposes ``\partial_a v(a_{\min}, z) \geq u'(w z + r a_{\min})`` (saving cannot be negative at the borrowing limit)

`ct_hjb` solves this by an **implicit upwind** finite-difference scheme: forward differences where the drift is positive, backward differences where it is negative, and an implicit time step ``(1/\Delta + \rho) v^{n+1} - A v^{n+1} = u(c^n) + v^n/\Delta`` with a large ``\Delta`` for fast convergence.

```@example ct
r = 0.03
kl = (0.36 / (r + 0.05))^(1 / 0.64)
w = 0.64 * kl^0.36
v, c, s, A, a, converged = ct_hjb(m, r, w)
(hjb_converged = converged,
 generator_row_sums = round(maximum(abs.(vec(sum(A; dims=2)))), sigdigits=2))
```

The generator's rows sum to zero (here to machine precision), confirming that ``A`` is a valid infinitesimal generator — the discretized drift and the income switching conserve probability mass.

---

## The Kolmogorov-Forward Equation

The stationary density ``g(a,z)`` solves ``A^\top g = 0`` subject to ``\int g\, da = 1``. Because the generator from the HJB is exactly the operator governing the distribution's flow, the KFE reuses it directly:

```@example ct
da = a[2] - a[1]
g = ct_kfe(A, m.I, da)
(density_nonnegative = minimum(g) >= -1e-10,
 integrates_to_one = round(sum(g) * da, digits=8))
```

The density is nonnegative and integrates to one. Mass piles up at the borrowing constraint ``a_{\min}``, where households with low income are stuck — the continuous-time analog of the kink in the discrete-time policy function.

---

## Stationary Equilibrium

`ct_steady_state` bisects on the interest rate ``r`` until household-supplied capital ``\int a\, g`` equals firm capital demand from the Cobb-Douglas first-order condition ``r = \alpha Z (K/L)^{\alpha-1} - \delta``, with the wage ``w = (1-\alpha) Z (K/L)^{\alpha}`` and effective labor ``L = \int z\, g``. It returns a [`CTSteadyState`](@ref) holding the value function, stationary density, prices, and aggregates. The idiosyncratic income state follows a [`CTPoissonIncome`](@ref) two-state Poisson process.

```@example ct
(interest_rate = round(ss.r, digits=5),
 capital = round(ss.K, digits=4),
 effective_labor = round(ss.L, digits=4),
 fraction_constrained = round((ss.g[1,1] + ss.g[1,2]) * (ss.a[2]-ss.a[1]), digits=4))
```

The equilibrium interest rate lies strictly below the discount rate ``\rho``: incomplete markets and precautionary saving push the supply of capital up and the return down, exactly as in the discrete-time Aiyagari (1994) economy.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `r_bounds` | `Tuple` | `(0.0001, ρ-1e-4)` | Bisection bracket for the equilibrium interest rate |
| `max_iter` | `Int` | `100` | Maximum interest-rate bisection iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance on capital market clearing |
| `hjb_max_iter` | `Int` | `100` | Maximum HJB value-function iterations per rate |
| `hjb_tol` | `Real` | ``10^{-6}`` | HJB convergence tolerance |
| `Delta` | `Real` | `1000.0` | Implicit HJB time step (speed only, not the solution) |

| Field | Type | Description |
|-------|------|-------------|
| `r`, `w` | `T` | Equilibrium interest rate and wage |
| `K`, `L` | `T` | Aggregate capital ``\int a g`` and effective labor ``\int z g`` |
| `a` | `Vector{T}` | Wealth grid |
| `g` | `Matrix{T}` | Stationary density over ``(a, z)`` (``I \times 2``) |
| `v`, `c`, `s` | `Matrix{T}` | Value, consumption, and saving drift |
| `A` | `SparseMatrixCSC{T}` | Infinitesimal generator (``2I \times 2I``) |

Two views render the equilibrium. `:distribution` draws the stationary density by income state, and `:policy` overlays consumption against the saving drift so the zero-drift crossing — the target wealth each income state saves toward — is visible:

```julia
plot_result(ss; view=:distribution)
plot_result(ss; view=:policy)
```

```@raw html
<iframe src="../assets/plots/ct_distribution.html" width="100%" height="440" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/ct_policy.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## MIT-Shock Transitions

An **MIT shock** is an unanticipated, deterministic aggregate disturbance: the economy sits at a steady state, is hit by a one-time shock, and converges back along a perfect-foresight path. `ct_mit_shock` computes this transition by **shooting on the capital path** ``K_t``:

1. Given a guess ``\{K_t\}`` and the TFP path ``\{Z_t\}``, set prices ``r_t, w_t``.
2. Solve the HJB **backward** from the terminal steady-state value ``v(\cdot,T)``.
3. Solve the KFE **forward** from the initial distribution ``g(\cdot,0)``.
4. Update ``K_t = \int a\, g_t`` by relaxation until the path converges.

The converged prices, aggregates, and time-varying densities are returned in a [`CTTransition`](@ref).

```@example ct
m2 = CTAiyagari(; sigma=2.0, rho=0.05, delta=0.05, a_max=30.0, I=120)
ss0 = ct_steady_state(m2; tol=1e-6)
# Transitory 3% TFP shock, mean-reverting; horizon T = 30 (dt = 0.5).
N = 60
Z_shock = [m2.Z * (1 + 0.03 * exp(-0.4 * (n - 1) * 0.5)) for n in 1:(N+1)]
tr = ct_mit_shock(m2, ss0, Z_shock; dt=0.5, max_iter=400, tol=1e-6, relax=0.3)
(converged = tr.converged,
 r_on_impact = round(tr.r[1], digits=5),
 steady_r = round(ss0.r, digits=5),
 K_peak = round(maximum(tr.K), digits=4),
 K_steady = round(ss0.K, digits=4))
```

On impact the higher productivity raises the marginal product of capital, so the interest rate jumps above its steady-state value; investment rises and capital accumulates to a hump before depreciating back to the steady state as the shock fades. The initial capital ``K_0`` is pinned by the predetermined wealth distribution. A zero shock (``Z_t \equiv Z``) returns a path that is flat at the steady state — a useful correctness check.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dt` | `Real` | `0.25` | Time step of the transition grid |
| `max_iter` | `Int` | `300` | Maximum shooting iterations |
| `tol` | `Real` | ``10^{-6}`` | Convergence tolerance on the capital path |
| `relax` | `Real` | `0.3` | Relaxation weight on the capital-path update |

| Field | Type | Description |
|-------|------|-------------|
| `t`, `Z` | `Vector{T}` | Time grid and the TFP path that generated it |
| `K`, `C` | `Vector{T}` | Aggregate capital and consumption along the transition |
| `r`, `w` | `Vector{T}` | Interest-rate and wage paths |
| `converged` | `Bool` | Whether the shooting iteration met `tol` |
| `iterations` | `Int` | Shooting iterations used |

---

## Two-Asset HANK (Kaplan-Moll-Violante)

The two-asset model adds a second, **illiquid** asset ``a`` (return ``r_a``) alongside the **liquid** asset ``b`` (return ``r_b < r_a``). Moving funds between them — the **deposit** ``d`` — incurs a convex adjustment cost ``\chi(d) = \tfrac{\chi}{2} d^2``. Households therefore accept the low liquid return to hold high-return illiquid wealth, producing a large illiquid stock and a thin liquid buffer: the central Kaplan-Moll-Violante (2018) mechanism.

```math
\rho V(b,a,z) = \max_{c,d}\; u(c) + V_b\,(w z + r_b b - d - \tfrac{\chi}{2}d^2 - c) + V_a\,(r_a a + d) + \sum_{z'}\lambda_{z\to z'}[V(b,a,z')-V(b,a,z)]
```

where the first-order conditions are ``c = (V_b)^{-1/\sigma}`` and ``d = (V_a/V_b - 1)/\chi``. The HJB is a two-dimensional PDE solved by upwind finite differences in both ``b`` and ``a``; the stationary joint density of ``(b,a,z)`` solves the Kolmogorov-Forward equation. `ct_two_asset_solve` returns a [`CTTwoAssetSolution`](@ref) with the value function, deposit and consumption policies, joint density, and aggregates.

!!! warning "Check stationarity before reading the aggregates"
    The illiquid ceiling ``a_{\max}`` must sit below ``a^\star = 1/(\chi r_a)`` under the level-quadratic cost, or the reported illiquid stock is where the grid ends rather than where households stop saving. The calibration below puts ``a^\star = 10`` against ``a_{\max} = 8``. See the next section and issue #509.

```@example ct
tw = CTTwoAsset(; r_a=0.05, r_b=0.02, chi=2.0, rho=0.06, a_max=8.0, b_max=5.0,
                  a_power=0.5, b_power=0.5, Ib=30, Ia=30)
sol = ct_two_asset_solve(tw; tol=1e-6)
report(sol)
```

Two thirds of household wealth (an illiquid share of 0.658, ``A = 0.636`` against ``B = 0.330``) sits in the illiquid asset: the three-point return premium more than compensates for the adjustment friction, while the thin liquid balance buffers income risk. The ceiling carries ``4.7 \times 10^{-8}`` of the mass, so the illiquid stock is a genuine equilibrium quantity and not a truncation artifact. Wealthy hand-to-mouth households — illiquid wealth, no liquid buffer — are 2.3% of the population against 0.04% poor hand-to-mouth, the Kaplan-Moll-Violante composition that a one-asset model cannot generate.

`HJB converged` reports `No` here even though the solution is stationary: the sup-norm change in ``V`` stalls just above ``10^{-6}`` while the aggregates are identical to six digits across 200, 400, and 1200 iterations and the KFE residual is ``1.9 \times 10^{-15}``. Read `kfe_residual` and the stability of the aggregates alongside the flag.

[`hand_to_mouth`](@ref) splits the liquidity-poor by illiquid holdings, and [`ceiling_mass`](@ref) reports the mass on the top node of each grid. Both default to a one-grid-step threshold; pass `b_threshold`/`a_threshold` for a calibration-based definition such as a fraction of average income.

[`ct_two_asset_stationarity`](@ref) takes three keywords beyond the model. `margin` (default `0.9`, quadratic cost only) demands ``a_{\max} \le \text{margin} \cdot a^\star`` rather than the bare inequality, because an ``a^\star`` sitting just above the ceiling still dumps mass on it; the calibration above passes at the default and fails at `margin=0.75`. `solution=` supplies a solved [`CTTwoAssetSolution`](@ref) so the check can add a ceiling-mass diagnostic, and `max_ceiling_mass=` sets the threshold that diagnostic applies (default `0.05`) — passing it without `solution` throws an `ArgumentError`, since the ceiling mass is a property of the solved distribution and not of the calibration. The return is always a five-field named tuple `(ok, bound, a_star, message, ceiling_mass)`, with `ceiling_mass` set to `nothing` when no solution is supplied, so the shape is stable across call styles. `ct_two_asset_solve` runs the same diagnostic itself: it warns after solving whenever more than 5% of the stationary mass ends up on the illiquid ceiling, which catches the calibrations that satisfy the analytical bound and still report an aggregate shaped by the grid.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `max_iter` | `Int` | `200` | Maximum HJB value-function iterations |
| `tol` | `Real` | ``10^{-6}`` | Sup-norm tolerance on the value function |
| `Delta` | `Real` | `1000.0` | Implicit HJB time step (speed only, not the solution) |
| `check_stationarity` | `Bool` | `true` | Warn when the calibration cannot bound illiquid wealth |
| `V_init` | `AbstractArray` | `nothing` | Warm-start value function, used by the GE loop |

| Field | Type | Description |
|-------|------|-------------|
| `b`, `a` | `Vector{T}` | Liquid and illiquid grids |
| `V`, `c`, `d` | `Array{T,3}` | Value function, consumption, and deposit policies over ``(b, a, z)`` |
| `sb`, `sa` | `Array{T,3}` | Liquid and illiquid drifts |
| `g` | `Array{T,3}` | Stationary joint density |
| `B`, `A` | `T` | Aggregate liquid and illiquid wealth |
| `gen` | `SparseMatrixCSC{T}` | Infinitesimal generator |
| `bdelta`, `adelta` | `Vector{T}` | Trapezoidal integration weights matching the grids |
| `hjb_converged` | `Bool` | Sup-norm convergence of ``V`` **and** a KFE residual below ``10^{-6}`` |
| `kfe_residual` | `T` | Stationarity residual of the distribution |
| `hjb_iterations` | `Int` | Value-function iterations used |

---

## Adjustment Costs and Grids

The shape of ``\chi`` decides two things the aggregates depend on: whether an **inaction region** exists, and whether illiquid wealth is bounded at all. The level-quadratic default settles the first (no) and constrains the second severely; the Kaplan-Moll-Violante rate-based cost is the alternative on both counts.

### Kinked adjustment costs and the inaction region

The smooth quadratic cost is differentiable at ``d = 0``, so **every** household with ``V_a \neq V_b`` adjusts and there is no inaction region. Kaplan, Moll & Violante specify a linear-plus-convex cost on the deposit **rate** ``x = d/\bar a`` with ``\bar a = \max(\chi_3, a)``:

```math
\chi(d,a) = \left[\chi_0 |x| + \frac{|x|^{1+\chi_2}}{\chi_1^{\chi_2}(1+\chi_2)}\right]\bar a
```

where:
- ``\chi_0`` is the linear term that creates the **kink** at ``d = 0``
- ``\chi_1, \chi_2`` scale and curve the convex term
- ``\chi_3`` (`a_kink`) floors ``\bar a`` so the rate stays finite at ``a = 0``

Because the marginal cost jumps from ``-\chi_0`` to ``+\chi_0`` across ``d = 0``, the first-order condition has **no solution** while ``|V_a/V_b - 1| \le \chi_0``, and the deposit is exactly zero — a genuine **inaction region**, resolved rather than smoothed:

```math
d = \begin{cases}
\bar a\,\chi_1 (V_a/V_b - 1 - \chi_0)^{1/\chi_2} & V_a/V_b - 1 > \chi_0\\
0 & |V_a/V_b - 1| \le \chi_0\\
-\bar a\,\chi_1 (1 - V_a/V_b - \chi_0)^{1/\chi_2} & V_a/V_b - 1 < -\chi_0
\end{cases}
```

```@example ct
kink = CTTwoAsset(; cost=:kinked, chi0=0.05, chi1=0.5, chi2=2.0, a_kink=1.0, Ib=6, Ia=6)
smooth = CTTwoAsset(; cost=:quadratic, chi=2.0, Ib=6, Ia=6)

(inside_band = MacroEconometricModels._ct2_deposit(kink, 1.03, 3.0),
 outside_band = round(MacroEconometricModels._ct2_deposit(kink, 1.20, 3.0); digits=4),
 quadratic_never_inactive = MacroEconometricModels._ct2_deposit(smooth, 1.03, 3.0) > 0)
```

The kinked deposit is *exactly* zero inside the band while the quadratic one is not.

!!! warning "The level-quadratic cost cannot bound illiquid wealth"
    With `cost=:quadratic`, ``d = (V_a/V_b - 1)/\chi`` and ``V_a/V_b \ge 0``, so the largest possible **withdrawal** is the constant ``1/\chi``. Illiquid wealth accrues ``r_a a``, which grows without bound in ``a``, so a constant withdrawal cap cannot offset it and the distribution diverges above ``a^\star = 1/(\chi r_a)``. `ct_two_asset_solve` now **warns** on such a calibration rather than returning a grid artifact with `hjb_converged = true`; pass `check_stationarity=false` to silence it. See issue #509.

    The dichotomy is sharper than "pick ``a_{\max} < a^\star``". Measured at ``\chi = 8``, ``r_a = 0.05`` (so ``a^\star = 2.5``), ``r_b = 0.02``, ``\rho = 0.08``:

    | ``a_{\max}`` | ``A`` | ``A/a_{\max}`` | mass on ceiling |
    |---|---|---|---|
    | 1.0 | ``\approx 0`` | ``\approx 0`` | ``6.7 \times 10^{-26}`` |
    | 2.0 | ``\approx 0`` | ``\approx 0`` | ``1.2 \times 10^{-22}`` |
    | 3.0 | 0.701 | 0.234 | 0.234 |
    | 6.0 | 3.853 | 0.642 | 0.642 |

    Above ``a^\star`` the ratio ``A/a_{\max}`` climbs toward 1 and equals the ceiling mass exactly — raising the ceiling does not reveal a tail, it only moves where the mass piles up. Below ``a^\star`` the ceiling is clean and ``A`` is insensitive to ``a_{\max}``, but only because ``A`` is essentially **zero**: the level-quadratic cost is stationary precisely where it supports no illiquid holdings at all.

    So `:quadratic` remains the default — it is smooth, cheap, and adequate for exercising the solver — but it cannot produce a calibration with a realistic illiquid-wealth tail. Use `cost=:kinked` for that. The KMV rate-based cost has no such problem because its withdrawal **scales with** ``a``.

The kinked stationarity condition runs the other way. In the KMV parameterization ``\chi_1`` **multiplies** the withdrawal, ``|d| = \chi_1(|V_a/V_b - 1| - \chi_0)^{1/\chi_2}\,\bar a``, so the maximum withdrawal *rate* is ``\chi_1(1-\chi_0)^{1/\chi_2}`` and illiquid wealth is bounded iff that exceeds ``r_a``. A **larger** ``\chi_1`` is therefore more stationary, not less. [`ct_two_asset_stationarity`](@ref) reports this rate as `bound` and the failing margin in `message`. Once the absolute cap `dmax` binds, the withdrawal is a constant again and ``a^\star = \texttt{dmax}/r_a`` re-applies.

A passing check does not guarantee a small ceiling mass: a near-frictionless calibration (very large ``\chi_1``) makes the illiquid asset strictly dominate and drives a corner portfolio, which piles mass on the ceiling for an economic reason rather than a numerical one. Read [`ceiling_mass`](@ref) alongside the check.

Calibrating the kinked cost splits its parameters in two. ``\chi_0`` and ``\chi_2`` are dimensionless shape parameters and transfer across calibrations; ``\chi_1`` sets the scale of the deposit *rate* and does not, so the shipped defaults — KMV's own quarterly estimates ``(\chi_0, \chi_1, \chi_2, \chi_3) = (0.04383, 0.48236, 0.40176, 0.0219)`` — are a starting point rather than a calibration. Their exponent implies ``1/\chi_2 \approx 2.5``, so a ratio ``V_a/V_b \approx 2.6`` already produces a very large deposit, which `dmax` caps as in KMV.

!!! warning "`cost=:kinked` is not production-ready"
    The kinked branch reproduces the inaction region and the stationarity arithmetic correctly, but no shipped calibration of it converges reliably — it needs per-model tuning of ``\chi_1`` and `dmax` before its aggregates mean anything. It is documented here because it is the only specification that can bound illiquid wealth while supporting a realistic tail, not because it is ready to use. Tracking issue: #509.

### Grids

KMV place both grids by `PowerSpacedGrid`: ``y = \text{lo} + (\text{hi}-\text{lo})\,x^{1/k}`` on ``x \in [0,1]``, where ``k = 1`` is uniform and ``k \to 0`` is L-shaped. This matters because the deposit FOC divides by ``V_b``: a uniform grid over a wide ``[0, b_{\max}]`` leaves ``V`` nearly flat in ``b`` at the top, so ``V_b \to 0`` and the FOC deposit explodes. Set `a_power` and `b_power` below 1 to concentrate nodes near the constraint.

Integration uses the matching **trapezoidal** weights `bdelta`/`adelta` (half-width at the grid edges), which is what the density is normalized against — a flat ``\Delta b\,\Delta a`` over-counts the boundary rows.

---

## Two-Asset General Equilibrium

`ct_two_asset_ge` closes the model. A Cobb-Douglas firm rents the illiquid asset as capital and liquid government bonds are in fixed net supply ``\bar B``, financed by a lump-sum tax ``\tau = r_b \bar B``:

```math
r_a = \alpha Z (K/L)^{\alpha-1} - \delta, \qquad w = (1-\alpha) Z (K/L)^{\alpha}
```

Equilibrium requires household illiquid wealth ``A = K`` and liquid wealth ``B = \bar B``; labor ``L`` is the stationary mean of the income process. The solver iterates both conditions with damped updates, warm-starting the household block from the previous value function.

```@example ct
ge = ct_two_asset_ge(CTTwoAsset(; Ib=25, Ia=25, a_max=6.0, b_max=5.0, rho=0.06,
                                z=[0.6, 1.4], lambda=[0.4, 0.4], B_supply=0.69,
                                chi=2.0, a_power=0.5, b_power=0.5);
                     max_iter=120, tol=1e-3)
report(ge)
```

Both markets clear to better than ``10^{-3}``: capital settles at ``K = 5.58`` against an illiquid residual of ``-5.0 \times 10^{-5}``, and the bond market absorbs ``\bar B = 0.69`` at a liquid return of ``-5.4\%``. That deeply negative ``r_b`` is the mechanism, not a failure — households pay for liquidity, which is why the equilibrium illiquid return ``r_a = 6.98\%`` stands twelve points above it. `markets_cleared` and `converged` are reported **separately**: market clearing succeeds here while the inner HJB does not, and collapsing them would hide which failed.

The 18.75% mass on the illiquid ceiling is the number to watch. With ``\chi = 2`` and an equilibrium ``r_a = 0.0698`` the level-quadratic bound is ``a^\star = 1/(\chi r_a) = 7.17``, so ``a_{\max} = 6`` clears the stationarity condition — but only just, and the accumulated mass at the top node says the ceiling still shapes the illiquid distribution. Treat ``K`` here as a solver exercise rather than a calibrated statistic.

!!! warning "Bond supply is not a free parameter"
    With a real adjustment cost, liquid demand has a **floor** — households hold a buffer however negative ``r_b`` goes. A ``\bar B`` below that floor leaves ``r_b`` pinned at its lower bound with the bond market uncleared.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `K_init`, `rb_init` | `Real` | `nothing` | Starting guesses; default to the representative-agent capital and half the implied ``r_a`` |
| `max_iter` | `Int` | `60` | Maximum outer market-clearing iterations |
| `tol` | `Real` | ``10^{-4}`` | Tolerance on both market-clearing residuals |
| `relax_K` | `Real` | `0.3` | Damping on the capital update |
| `relax_rb` | `Real` | `0.02` | Damping on the liquid-return update |
| `hjb_max_iter` | `Int` | `200` | Inner HJB iterations per price vector |
| `Delta` | `Real` | `1000.0` | Implicit HJB time step |

| Field | Type | Description |
|-------|------|-------------|
| `r_a`, `r_b`, `w`, `tau` | `T` | Equilibrium illiquid return, liquid return, wage, and lump-sum tax |
| `K`, `B`, `L`, `Y` | `T` | Capital, liquid bonds, effective labor, and output |
| `solution` | `CTTwoAssetSolution{T}` | Household block at the equilibrium prices |
| `resid_illiquid`, `resid_liquid` | `T` | ``A - K`` and ``B - \bar B`` |
| `markets_cleared` | `Bool` | Both residuals within `tol` |
| `converged` | `Bool` | Market clearing **and** the inner household block converged |
| `iterations` | `Int` | Outer iterations used |

---

## Two-Asset MIT Transitions

`ct_two_asset_mit` computes the deterministic path after an unanticipated aggregate TFP shock, shooting on both the capital and liquid-return paths: backward HJB from the terminal value, forward KFE from the initial distribution.

```@example ct
N = 40
Zpath = [1.0 + 0.02 * 0.6^(n - 1) for n in 1:(N + 1)]; Zpath[1] = 1.0
tr = ct_two_asset_mit(CTTwoAsset(; Ib=25, Ia=25, a_max=6.0, b_max=5.0, rho=0.06,
                                 z=[0.6, 1.4], lambda=[0.4, 0.4], B_supply=0.69,
                                 chi=2.0, a_power=0.5, b_power=0.5),
                      ge, Zpath; dt=0.5, max_iter=80, tol=1e-5)

(K0_pinned = isapprox(tr.K[1], ge.K; rtol=1e-12),
 peak_above_ss = maximum(tr.K) > ge.K,
 returns_to_ss = round(abs(tr.K[end] - ge.K); sigdigits=2),
 r_a_up_on_impact = tr.r_a[2] > ge.r_a)
```

``K_0`` is pinned by the predetermined wealth distribution and cannot jump on impact; higher productivity raises the marginal product of capital and the wage; capital accumulates to a hump and returns to the terminal steady state.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dt` | `Real` | `0.25` | Time step of the transition grid |
| `max_iter` | `Int` | `200` | Maximum shooting iterations |
| `tol` | `Real` | ``10^{-5}`` | Path convergence tolerance |
| `relax_K` | `Real` | `0.3` | Damping on the capital-path update |
| `relax_rb` | `Real` | `0.02` | Damping on the liquid-return-path update |

| Field | Type | Description |
|-------|------|-------------|
| `t`, `Z` | `Vector{T}` | Time grid and the TFP path |
| `K`, `B`, `C` | `Vector{T}` | Capital, liquid bonds, and consumption paths |
| `r_a`, `r_b`, `w` | `Vector{T}` | Illiquid return, liquid return, and wage paths |
| `converged` | `Bool` | Whether the shooting iteration met `tol` |
| `iterations` | `Int` | Shooting iterations used |

---

## Complete Example

This example solves a one-asset continuous-time Aiyagari economy end to end: it builds the model, computes the stationary equilibrium, and reads off the equilibrium interest rate, aggregate capital, and the mass of households at the borrowing constraint.

```@example ct
# Calibrate and solve the stationary equilibrium
aiyagari = CTAiyagari(; alpha=0.36, rho=0.05, sigma=2.0, delta=0.05,
                        z=[0.1, 0.2], lambda=[0.5, 0.5], a_max=30.0, I=200)
eq = ct_steady_state(aiyagari; tol=1e-5)
report(eq)
```

```@example ct
# Key equilibrium objects
da = eq.a[2] - eq.a[1]
(interest_rate = round(eq.r, digits=5),
 below_discount_rate = eq.r < aiyagari.rho,
 aggregate_capital = round(eq.K, digits=4),
 effective_labor = round(eq.L, digits=4),
 constrained_mass = round((eq.g[1, 1] + eq.g[1, 2]) * da, digits=4))
```

The equilibrium interest rate settles below the discount rate ``\rho``: incomplete markets and precautionary saving push the supply of capital up and its return down. A nontrivial fraction of households sits at the borrowing constraint ``a_{\min}``, where the stationary density piles up --- the continuous-time counterpart of the discrete-time Aiyagari (1994) economy.

---

## Common Pitfalls

1. **Grid resolution.** The implicit upwind scheme is first-order accurate. Increase `I` (and `a_max`) for sharper policy functions and a more accurate constrained mass; `I = 500`–`1000` is typical for publication.

2. **Interest-rate bounds.** Equilibrium `r` lies in ``(0, \rho)``. As ``r \to \rho`` aggregate saving diverges, so the default upper bound is ``\rho - 10^{-4}``.

3. **Implicit step size `Delta`.** A large `Delta` (default `1000`) makes the implicit HJB iteration converge in tens of steps. It controls only the speed of the value-function iteration, not the solution.

4. **Two-asset `a_max` above ``a^\star`` reports a grid artifact.** Under `cost=:quadratic` the largest withdrawal is the constant ``1/\chi``, so illiquid wealth diverges above ``a^\star = 1/(\chi r_a)``. Call [`ct_two_asset_stationarity`](@ref) before reading any aggregate, and check [`ceiling_mass`](@ref) afterwards — a passing condition with a fat ceiling mass still means the top node is shaping the distribution.

5. **`hjb_converged = false` does not always mean an unusable solution.** The flag requires both a sup-norm change in ``V`` below `tol` and a KFE residual below ``10^{-6}``. On several two-asset calibrations the value function drifts just above the tolerance while the aggregates are stable to six digits and `kfe_residual` sits at ``10^{-15}``. Compare aggregates across two iteration caps before discarding the result.

6. **`cost=:kinked` needs calibration.** It is the only specification that bounds illiquid wealth while supporting a realistic tail, but no shipped parameterization converges reliably; ``\chi_1`` and `dmax` must be tuned per model. See issue #509.

---

## References

- Achdou, Yves, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. 2022. "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89 (1): 45--86. [DOI](https://doi.org/10.1093/restud/rdab002)

- Aiyagari, S. Rao. 1994. "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109 (3): 659--684. [DOI](https://doi.org/10.2307/2118417)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)
