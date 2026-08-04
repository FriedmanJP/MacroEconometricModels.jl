# [Overlapping Generations](@id dsge_olg)

The **Blanchard (1985) perpetual-youth** model embeds overlapping generations into an otherwise standard neoclassical growth model. Agents survive each period with probability ``\gamma`` and newborns enter with zero financial wealth, so the economy is populated by households of different ages and wealth. This generational turnover breaks the representative-agent benchmark: the equilibrium interest rate exceeds the rate of time preference, and government debt is net wealth that crowds out capital — the failure of Ricardian equivalence.

This page is part of the [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) sub-hub of the [DSGE Models](@ref dsge_page) suite. The implementation is the analytically tractable discrete-time Blanchard-Yaari case with log utility and fair annuities. For a genuine age structure — age-specific mortality, an age-earnings profile, retirement, and backward induction over age — see [True Life Cycle: Age-Dependent EGM](@ref lifecycle_olg).

## Quick Start

```@setup olg
using MacroEconometricModels
```

**Recipe 1: Steady state**

```@example olg
m = BlanchardOLG(; alpha=0.36, beta=0.96, delta=0.08, gamma=0.98)
ss = blanchard_steady_state(m)
report(ss)
```

Bisection settles on capital of 5.12 supporting consumption of 1.39 at an interest rate of 4.65%, and the log-utility marginal propensity to consume out of total wealth is ``1 - \beta\gamma = 0.0592``.

**Recipe 2: Finite horizons raise the interest rate**

```@example olg
ramsey = blanchard_steady_state(BlanchardOLG(; gamma=1.0)).r     # representative agent
olg    = blanchard_steady_state(BlanchardOLG(; gamma=0.98)).r    # perpetual youth
(ramsey_rate = round(ramsey, digits=5), olg_rate = round(olg, digits=5))
```

A 2% per-period death rate lifts the equilibrium rate from the Ramsey value ``1/\beta - 1 = 4.17\%`` to 4.65%. Newborns enter with zero wealth and dilute the aggregate stock, so capital sits below the modified golden rule and its return is correspondingly higher.

**Recipe 3: Non-Ricardian government debt**

```@example olg
no_debt = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=0.0))
debt    = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=0.1))
(Δr = round(debt.r - no_debt.r, digits=6), Δk = round(debt.k - no_debt.k, digits=4))
```

Issuing debt worth 0.1 per capita raises the interest rate by 9.1 basis points and crowds out 0.0057 of capital. Under Ricardian equivalence both differences would be exactly zero.

**Recipe 4: Saddle-path dynamics**

```@example olg
sol = blanchard_solve(m, ss)
(stable_eigenvalue = round(sol.stable_eig, digits=4),
 determinate = sol.determinate,
 consumption_slope = round(sol.policy_slope, digits=4))
```

Exactly one eigenvalue lies inside the unit circle, so the saddle path is locally unique. Capital converges at 0.8838 per period — a half-life of about six periods — and consumption rises 0.163 for every unit of capital along the path.

---

## Demographics and Annuities

Fujiwara & Teranishi (2008) embed this perpetual-youth demographic structure in a New Keynesian model to study how societal aging shapes monetary policy. Each period an agent survives with probability ``\gamma \in (0,1]`` and dies with probability ``1-\gamma``. Population is constant: the mass ``1-\gamma`` of newborns exactly replaces the deceased. Survival enters the objective as an extra discount, so the agent maximizes

```math
\mathbb{E}_t \sum_{j \geq 0} (\beta \gamma)^j \, \ln c_{t+j}
```

where:
- ``\beta`` is the pure discount factor (time preference)
- ``\gamma`` is the one-period survival probability
- ``\beta\gamma`` is the effective discount factor

Households trade **fair annuities**: an agent surrenders all wealth at death in exchange for the gross return ``(1+r)/\gamma`` while alive. The annuity premium ``1/\gamma`` exactly offsets mortality, so the *individual* Euler equation is the standard ``c_{t+1}/c_t = \beta(1+r_{t+1})``. Age matters only because newborns start with zero assets while older cohorts have accumulated wealth.

`BlanchardOLG` collects the calibration:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `alpha` | `Real` | `0.36` | Capital share in Cobb-Douglas production |
| `beta` | `Real` | `0.96` | Pure discount factor (time preference) |
| `delta` | `Real` | `0.08` | Capital depreciation rate |
| `gamma` | `Real` | `0.98` | One-period survival probability (``\gamma = 1`` is the representative agent) |
| `Z` | `Real` | `1.0` | Total factor productivity |
| `b` | `Real` | `0.0` | Per-capita government debt (net wealth) |

---

## Aggregate Dynamics

With log utility the marginal propensity to consume out of total (financial plus human) wealth is ``1-\beta\gamma``. Aggregating the individual consumption rule across cohorts and eliminating human wealth yields a two-equation system in aggregate capital ``k`` and consumption ``C``:

```math
\begin{aligned}
C_{t+1} &= (1+r_{t+1}) \left[ \beta\, C_t - \lambda\,(k_{t+1} + b) \right], \qquad \lambda = \frac{(1-\beta\gamma)(1-\gamma)}{\gamma} \\
k_{t+1} &= (1+r_t)\, k_t + w_t - C_t
\end{aligned}
```

where:
- ``r_t = \alpha Z k_t^{\alpha-1} - \delta`` and ``w_t = (1-\alpha) Z k_t^{\alpha}`` are competitive factor prices
- ``b`` is per-capita government debt (held as net wealth; taxes ``r_t b`` service it)
- ``\lambda`` is the **Blanchard correction**: it scales with the death rate ``1-\gamma`` and aggregate assets ``k+b``

The correction term is the discrete-time analog of Blanchard's continuous-time wedge ``-\nu(\rho+\nu)A``. When ``\gamma = 1`` it vanishes and the Euler collapses to the representative-agent form ``C_{t+1}/C_t = \beta(1+r_{t+1})``.

---

## Steady State

`blanchard_steady_state` solves for capital by bracketed bisection, equating the budget-implied consumption ``C = r k + w`` (``= f(k) - \delta k``; debt is net wealth and taxes ``r b`` service it, so the debt-service terms cancel in aggregate) with the Euler-implied consumption ``C = (1+r)\lambda(k+b)/[\beta(1+r)-1]``. The solver selects the high-capital root continuously connected to the Ramsey economy and returns a [`BlanchardOLGSteadyState`](@ref).

```@example olg
ss = blanchard_steady_state(BlanchardOLG(; gamma=0.96))
(capital = round(ss.k, digits=4),
 interest = round(ss.r, digits=5),
 time_preference = round(1/0.96 - 1, digits=5))
```

The equilibrium interest rate (``\approx`` 5.8% here) lies **above** the pure rate of time preference ``1/\beta-1 \approx`` 4.2%. Finite horizons require ``\beta(1+r) > 1``: because newborns dilute aggregate wealth, capital is below the modified golden rule and the return on capital is correspondingly higher. Lowering ``\gamma`` (shorter expected lives) widens this gap.

| Field | Type | Description |
|-------|------|-------------|
| `k` | `T` | Aggregate capital per capita |
| `C` | `T` | Aggregate consumption |
| `r` | `T` | Equilibrium interest rate |
| `w` | `T` | Wage |
| `H` | `T` | Aggregate human wealth |
| `mpc` | `T` | Marginal propensity to consume ``1-\beta\gamma`` |
| `b` | `T` | Per-capita government debt the steady state was solved at |
| `converged` | `Bool` | Whether the bisection converged |

The solver itself takes two keywords: `tol` (default ``10^{-10}``, the residual at which the bisection stops) and `max_iter` (default `200`). Both bind only after the 400-point downward scan has bracketed the high-capital root.

---

## Non-Ricardian Debt

Because newborns do not internalize the taxes that will service debt issued before their birth, government debt is **net wealth** in the aggregate. Higher debt raises aggregate demand for assets, bidding up the interest rate and crowding out capital.

```@example olg
[let s = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=b))
    (debt = b, r = round(s.r, digits=5), k = round(s.k, digits=4))
 end for b in (0.0, 0.05, 0.10, 0.15)]
```

The interest rate rises and capital falls monotonically with debt — Ricardian equivalence fails. In the representative-agent limit (``\gamma=1``) the correction term is zero and debt has no real effect, restoring Ricardian equivalence.

---

## Transitional Dynamics

`blanchard_solve` linearizes the ``(k, C)`` system around the steady state and solves the saddle path, returning a [`BlanchardOLGSolution`](@ref) that carries the policy matrix and eigenvalues. The 2×2 transition has one eigenvalue inside the unit circle (the stable convergence rate) and one outside, confirming determinacy. `blanchard_transition` then simulates convergence from an arbitrary initial capital stock.

```@example olg
m = BlanchardOLG(; gamma=0.98)
ss = blanchard_steady_state(m)
sol = blanchard_solve(m, ss)
path = blanchard_transition(m, sol, 0.7 * ss.k; H=40)
(k_initial = round(path.k[1], digits=3),
 k_halfway = round(path.k[20], digits=3),
 k_final = round(path.k[end], digits=3),
 steady_state_k = round(ss.k, digits=3))
```

Starting 30% below the steady state, capital rises monotonically toward ``k^*`` at the stable rate, with the interest rate falling and consumption rising along the saddle path — the standard Ramsey-style transition, modified by the perpetual-youth wedge.

!!! note "Determinacy of the saddle path"
    The linearized ``(k, C)`` system is a predetermined-plus-jump pair: capital is predetermined and consumption jumps. A determinate saddle path requires exactly one eigenvalue inside the unit circle. `sol.determinate` reports this check; when it is `false`, the calibration admits no locally unique convergent path.

`blanchard_solve` returns a `BlanchardOLGSolution`:

| Field | Type | Description |
|-------|------|-------------|
| `ss` | `BlanchardOLGSteadyState{T}` | Steady state the linearization expands around |
| `M` | `Matrix{T}` | ``2 \times 2`` linearized transition of ``(k - k^*, C - C^*)`` |
| `eigenvalues` | `Vector{ComplexF64}` | Both eigenvalues of ``M`` |
| `stable_eig` | `T` | Saddle-path (modulus ``< 1``) eigenvalue governing convergence |
| `policy_slope` | `T` | Consumption policy slope ``dC/dk`` along the saddle path |
| `determinate` | `Bool` | `true` when exactly one eigenvalue lies inside the unit circle |

`blanchard_transition(m, sol, k0; H=50)` returns a `NamedTuple` of length-``H+1`` paths:

| Field | Type | Description |
|-------|------|-------------|
| `k` | `Vector{T}` | Capital path converging to ``k^*`` |
| `C` | `Vector{T}` | Consumption path |
| `r` | `Vector{T}` | Interest-rate path (evaluated at each period's capital) |
| `w` | `Vector{T}` | Wage path |

Plotting the solution draws the stable manifold ``C = C^* + \texttt{policy\_slope}\cdot(k - k^*)`` in ``(k, C)`` space over ``k^* \pm`` `k_span`` \cdot k^*``, with the steady state marked and the stable eigenvalue and determinacy flag in the panel title. The bare steady state is a handful of scalars and has no chart — read it through `report`.

```julia
plot_result(sol)                # saddle path through the steady state
plot_result(sol; k_span=0.6)    # widen the plotted capital range
```

---

## Complete Example

```@example olg
# Compare representative-agent and perpetual-youth economies
ra  = blanchard_steady_state(BlanchardOLG(; gamma=1.0))
py  = blanchard_steady_state(BlanchardOLG(; gamma=0.95))
report(py)
```

```@example olg
(ra_interest = round(ra.r, digits=5),
 py_interest = round(py.r, digits=5),
 ra_capital = round(ra.k, digits=3),
 py_capital = round(py.k, digits=3))
```

The perpetual-youth economy has a higher interest rate and lower capital than the representative-agent benchmark with the same preferences and technology — the quantitative signature of finite horizons.

---

## [True Life Cycle: Age-Dependent EGM](@id lifecycle_olg)

Perpetual youth gives every agent the same constant survival probability, so the economy has no **age structure**: a 25-year-old and a 75-year-old face identical problems. Pension reform, demographic transition, and lifecycle inequality all turn on precisely the differences perpetual youth assumes away. [`LifeCycleOLG`](@ref) is the Auerbach–Kotlikoff / İmrohoroğlu–İmrohoroğlu–Joines alternative: agents live at most ``J`` ages, with age-specific mortality ``s_j``, a deterministic age-earnings profile ``\kappa_j``, persistent idiosyncratic productivity, and retirement onto a pay-as-you-go pension.

```math
V_j(a, e) = \max_{c,\, a' \ge \underline{a}} \; u(c) + \beta\, s_j\, E\!\left[V_{j+1}(a', e') \mid e\right],
\qquad c + a' = R_j\, a + y_j(e)
```

where:
- ``j = 1, \dots, J`` is age and ``s_j`` the probability of surviving to ``j+1`` (``s_J = 0``)
- ``y_j(e) = (1-\tau) w \kappa_j e`` while working and the pension thereafter
- ``R_j`` is the gross return on savings: ``(1+r)/s_j`` under actuarially fair annuities, ``1+r`` otherwise
- ``\underline{a}`` is the credit limit; households may borrow during life but cannot die in debt

The household problem is solved by **backward induction over age** — one endogenous-grid sweep per age, from ``J`` down to 1. This is a *finite sweep*, not a fixed point: age ``J`` is known because assets are exhausted, and every earlier age follows from the next. The only fixed point in the model is the market-clearing interest rate.

!!! warning "Income processes must be in levels"
    [`rouwenhorst`](@ref) and [`tauchen`](@ref) return the productivity grid in **logs**, symmetric about zero, so its mean is zero rather than one. Passing one straight in would zero out aggregate efficiency labor and every factor price. Use [`lifecycle_income`](@ref), which exponentiates and normalizes to unit mean; the constructor rejects any process with non-positive states.

```@example olg
lc = LifeCycleOLG(; J=40, J_retire=31, survival=0.995,
                  income=lifecycle_income(0.95, 0.2, 3),
                  a_max=50.0, n_a=120, beta=0.97, sigma=2.0, replacement=0.4)
ss_lc = lifecycle_steady_state(lc; r_bounds=(-0.01, 0.10), tol=1e-6)
report(ss_lc)
```

The equilibrium interest rate clears the capital market at a capital-output ratio near 3.1. Assets trace the classic hump: zero at birth, peaking just before retirement, then run down. The reported aggregate capital **is** the integral of the age-asset distribution, so the accounting identity holds by construction and `excess_demand` is the honest measure of how well the market clears.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `J` / `J_retire` | `Int` | `60` / `45` | Maximum age and first retired age |
| `survival` | `Real` or `Vector` | `0.99` | Survival probabilities; a scalar is broadcast, `s_J` forced to zero |
| `earnings` | `Vector` | `nothing` | Deterministic age-earnings profile ``\kappa_j``; defaults to a hump |
| `income` | `IncomeProcess` | `lifecycle_income(0.95, 0.2, 5)` | Idiosyncratic productivity in **levels** |
| `replacement` | `Real` | `0.4` | Pension as a fraction of average labor income (`0` ⇒ no social security) |
| `annuities` | `Bool` | `true` | Actuarially fair annuities, else accidental bequests rebated lump-sum |
| `n_pop` | `Real` | `0.0` | Population growth rate |
| `beta` / `sigma` | `Real` | `0.97` / `2.0` | Discount factor and CRRA curvature |
| `alpha` / `delta` / `Z` | `Real` | `0.36` / `0.06` / `1.0` | Capital share, depreciation, TFP |
| `a_max` / `n_a` | `Int` | `60.0` / `200` | Asset-grid ceiling and node count |
| `credit_limit` | `Real` | `0.0` | Borrowing floor ``\underline{a}``; households may borrow but cannot die in debt |
| `grid_type` | `Symbol` | `:double_exp` | Asset-grid curve |

`lifecycle_steady_state` bisects on the capital-labor ratio:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `r_bounds` | `Tuple` | `(-0.02, 0.10)` | Bracket for the equilibrium interest rate |
| `tol` | `Real` | ``10^{-6}`` | Tolerance on excess capital supply |
| `max_iter` | `Int` | `60` | Maximum bisection iterations |
| `bequest_iter` | `Int` | `50` | Inner iterations on the accidental-bequest fixed point |
| `verbose` | `Bool` | `false` | Print bisection progress |

| Field | Type | Description |
|-------|------|-------------|
| `r`, `w` | `T` | Equilibrium interest rate and wage |
| `K`, `L`, `Y` | `T` | Aggregate capital (the integral of `dist`), efficiency labor, output |
| `tau`, `pension` | `T` | Payroll tax and benefit balancing the pay-as-you-go budget |
| `transfer` | `T` | Lump-sum rebate of accidental bequests (`0` under annuities) |
| `c_policy`, `a_policy` | `Array{T,3}` | ``n_a \times n_e \times J`` consumption and savings policies |
| `dist` | `Array{T,3}` | ``n_a \times n_e \times J`` population distribution, sums to one |
| `cohort_mass` | `Vector{T}` | Stationary population share by age |
| `asset_profile`, `consumption_profile`, `income_profile` | `Vector{T}` | Means by age |
| `excess_demand` | `T` | Market-clearing residual |
| `converged`, `iterations` | `Bool`, `Int` | Bisection outcome |
| `spec` | `LifeCycleOLG{T}` | Model solved |

### Solving the Household Problem Directly

The two halves of the steady state are callable on their own, which is what you want when prices come from somewhere other than this model's capital market — a partial-equilibrium exercise, or a calibration where the government budget is imposed rather than solved.

[`lifecycle_policies`](@ref) runs the backward sweep at a given ``(r, w)`` and returns `(c_pol, a_pol)`, each ``n_a \times n_e \times J``. [`lifecycle_distribution`](@ref) pushes a cohort forward through a savings policy and returns the ``n_a \times n_e \times J`` population array, weighted by stationary cohort masses so it sums to one.

```@example olg
c_pol, a_pol = lifecycle_policies(lc, 0.04, 1.0; tau=0.1, pension=0.3)
dist = lifecycle_distribution(lc, a_pol)
mean_assets = vec(sum(dist .* lc.grid.grids[1]; dims=(1, 2))) ./ vec(sum(dist; dims=(1, 2)))
(policy_shape = size(c_pol),
 total_mass = round(sum(dist), digits=12),
 peak_asset_age = argmax(mean_assets),
 retirement_age = lc.J_retire)
```

The array carries unit mass by construction. Mean assets peak at age 28, three ages before retirement at 31: households accumulate through working life, stop just as earnings give way to the pension, and run the stock down thereafter. Because survival is independent of assets and productivity, mortality rescales cohorts without distorting the within-cohort distribution, so age enters the array only through the cohort weights — which is why this profile matches `ss_lc.asset_profile` in shape even though no market has been cleared here.

### The Consumption Hump Requires Imperfect Annuitization

With actuarially fair annuities survivors earn ``R_j = (1+r)/s_j``, so ``\beta s_j R_j = \beta(1+r)`` **exactly**: survival cancels out of the Euler equation and mortality cannot bend the consumption path. This is the same Blanchard–Yaari device the perpetual-youth model uses, and it is why the two families nest — but it also means an annuitized life-cycle model cannot produce a consumption hump. Switch annuities off and the Euler growth factor becomes ``(\beta s_j (1+r))^{1/\sigma}``, which falls below one once late-life mortality bites.

```@example olg
surv = lifecycle_survival(65)                       # Gompertz-Makeham mortality
common = (J=65, J_retire=45, survival=surv, income=lifecycle_income(0.95, 0.2, 3),
          a_max=60.0, n_a=110, beta=0.97, sigma=2.0, replacement=0.4)
hump = lifecycle_steady_state(LifeCycleOLG(; common..., annuities=false);
                              r_bounds=(-0.01, 0.20), tol=1e-6)
flat = lifecycle_steady_state(LifeCycleOLG(; common..., annuities=true);
                              r_bounds=(-0.01, 0.20), tol=1e-6)
(hump_peak_age = argmax(hump.consumption_profile),
 flat_peak_age = argmax(flat.consumption_profile),
 bequest_transfer = round(hump.transfer, digits=4))
```

Without annuities consumption peaks at the retirement age and declines thereafter — the life-cycle hump. With annuities it peaks at the very last age, rising monotonically throughout, because ``\beta(1+r) > 1`` and mortality has been insured away. The accidental bequests that annuities would have absorbed are instead rebated lump-sum, which is itself a fixed point: policies depend on the transfer and the transfer on the policies.

```julia
plot_result(ss_lc)                        # age profiles + cohort mass
plot_result(ss_lc; view=:distribution)    # mean and interquartile wealth by age
plot_result(ss_lc; view=:policy)          # consumption policy at three ages
```

---

## Common Pitfalls

1. **`γ = 1` is the representative-agent limit.** With certain survival the Blanchard correction vanishes, the interest rate equals ``1/\beta-1``, and Ricardian equivalence holds. Use ``\gamma < 1`` for genuine OLG effects.

2. **Large debt and multiple roots.** The OLG consumption function can admit a second, degenerate low-capital root with an implausibly high interest rate. The solver scans from high capital downward to select the economically relevant root; very large `b` may have no high-capital equilibrium (`converged` will be `false`).

3. **Log utility only.** The closed-form marginal propensity to consume ``1-\beta\gamma`` requires log utility (``\sigma = 1``). General CRRA implies a wealth- and rate-dependent propensity that this implementation does not cover. This restriction applies to `BlanchardOLG` alone — [`LifeCycleOLG`](@ref) takes any CRRA curvature.

4. **Life-cycle income processes must be in levels.** `rouwenhorst`/`tauchen` return log states with mean zero; use [`lifecycle_income`](@ref). The constructor rejects non-positive states rather than silently returning an economy with no labor.

5. **No hump under annuities.** With actuarially fair annuities survival cancels out of the Euler equation exactly, so consumption cannot turn over no matter how steep mortality is. Set `annuities=false` (and use a realistic mortality profile such as [`lifecycle_survival`](@ref)) to generate a life-cycle consumption hump.

6. **Bracket the equilibrium rate.** `lifecycle_steady_state` bisects on the capital-labor ratio and reports `converged=false` with a warning when excess capital supply does not change sign on `r_bounds`. Accidental bequests raise the equilibrium rate, so the annuity calibration's bracket is often too narrow for `annuities=false` — widen `r_bounds` rather than raising `tol`.

---

## References

- Blanchard, Olivier J. 1985. "Debt, Deficits, and Finite Horizons." *Journal of Political Economy* 93 (2): 223--247. [DOI](https://doi.org/10.1086/261297)

- Auerbach, Alan J., and Laurence J. Kotlikoff. 1987. *Dynamic Fiscal Policy*. Cambridge: Cambridge University Press. ISBN 978-0521300414.

- İmrohoroğlu, Ayşe, Selahattin İmrohoroğlu, and Douglas H. Joines. 1995. "A Life Cycle Analysis of Social Security." *Economic Theory* 6 (1): 83--114. [DOI](https://doi.org/10.1007/BF01213942)

- Yaari, Menahem E. 1965. "Uncertain Lifetime, Life Insurance, and the Theory of the Consumer." *Review of Economic Studies* 32 (2): 137--150. [DOI](https://doi.org/10.2307/2296058)

- Fujiwara, Ippei, and Yuki Teranishi. 2008. "A Dynamic New Keynesian Life-Cycle Model: Societal Aging, Demographics, and Monetary Policy." *Journal of Economic Dynamics and Control* 32 (8): 2398--2427. [DOI](https://doi.org/10.1016/j.jedc.2007.09.002)
