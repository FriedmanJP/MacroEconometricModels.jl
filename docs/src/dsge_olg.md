# [Overlapping Generations](@id dsge_olg)

The **Blanchard (1985) perpetual-youth** model embeds overlapping generations into an otherwise standard neoclassical growth model. Agents survive each period with probability ``\gamma`` and newborns enter with zero financial wealth, so the economy is populated by households of different ages and wealth. This generational turnover breaks the representative-agent benchmark: the equilibrium interest rate exceeds the rate of time preference, and government debt is net wealth that crowds out capital — the failure of Ricardian equivalence.

The implementation is the analytically tractable discrete-time Blanchard-Yaari case with log utility and fair annuities.

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

**Recipe 2: Finite horizons raise the interest rate**

```@example olg
ramsey = blanchard_steady_state(BlanchardOLG(; gamma=1.0)).r     # representative agent
olg    = blanchard_steady_state(BlanchardOLG(; gamma=0.98)).r    # perpetual youth
(ramsey_rate = round(ramsey, digits=5), olg_rate = round(olg, digits=5))
```

**Recipe 3: Non-Ricardian government debt**

```@example olg
no_debt = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=0.0))
debt    = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=0.1))
(Δr = round(debt.r - no_debt.r, digits=6), Δk = round(debt.k - no_debt.k, digits=4))
```

**Recipe 4: Saddle-path dynamics**

```@example olg
sol = blanchard_solve(m, ss)
(stable_eigenvalue = round(sol.stable_eig, digits=4),
 determinate = sol.determinate,
 consumption_slope = round(sol.policy_slope, digits=4))
```

---

## Demographics and Annuities

Each period an agent survives with probability ``\gamma \in (0,1]`` and dies with probability ``1-\gamma``. Population is constant: the mass ``1-\gamma`` of newborns exactly replaces the deceased. Survival enters the objective as an extra discount, so the agent maximizes

```math
\mathbb{E}_t \sum_{j \geq 0} (\beta \gamma)^j \, \ln c_{t+j}
```

where:
- ``\beta`` is the pure discount factor (time preference)
- ``\gamma`` is the one-period survival probability
- ``\beta\gamma`` is the effective discount factor

Households trade **fair annuities**: an agent surrenders all wealth at death in exchange for the gross return ``(1+r)/\gamma`` while alive. The annuity premium ``1/\gamma`` exactly offsets mortality, so the *individual* Euler equation is the standard ``c_{t+1}/c_t = \beta(1+r_{t+1})``. Age matters only because newborns start with zero assets while older cohorts have accumulated wealth.

---

## Aggregate Dynamics

With log utility the marginal propensity to consume out of total (financial plus human) wealth is ``1-\beta\gamma``. Aggregating the individual consumption rule across cohorts and eliminating human wealth yields a two-equation system in aggregate capital ``k`` and consumption ``C``:

```math
\begin{aligned}
C_{t+1} &= (1+r_{t+1}) \left[ \beta\, C_t - \lambda\,(k_{t+1} + b) \right], \qquad \lambda = \frac{(1-\beta\gamma)(1-\gamma)}{\gamma} \\
k_{t+1} &= (1+r_t)\, k_t + w_t - r_t\, b - C_t
\end{aligned}
```

where:
- ``r_t = \alpha Z k_t^{\alpha-1} - \delta`` and ``w_t = (1-\alpha) Z k_t^{\alpha}`` are competitive factor prices
- ``b`` is per-capita government debt (held as net wealth; taxes ``r_t b`` service it)
- ``\lambda`` is the **Blanchard correction**: it scales with the death rate ``1-\gamma`` and aggregate assets ``k+b``

The correction term is the discrete-time analog of Blanchard's continuous-time wedge ``-\nu(\rho+\nu)A``. When ``\gamma = 1`` it vanishes and the Euler collapses to the representative-agent form ``C_{t+1}/C_t = \beta(1+r_{t+1})``.

---

## Steady State

`blanchard_steady_state` solves for capital by bracketed bisection, equating the budget-implied consumption ``C = r k + w - r b`` with the Euler-implied consumption ``C = (1+r)\lambda(k+b)/[\beta(1+r)-1]``. The solver selects the high-capital root continuously connected to the Ramsey economy.

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
| `converged` | `Bool` | Whether the bisection converged |

---

## Non-Ricardian Debt

Because newborns do not internalize the taxes that will service debt issued before their birth, government debt is **net wealth** in the aggregate. Higher debt raises aggregate demand for assets, bidding up the interest rate and crowding out capital.

```@example olg
for b in (0.0, 0.05, 0.10, 0.15)
    s = blanchard_steady_state(BlanchardOLG(; gamma=0.98, b=b))
    println("b = ", b, "  →  r = ", round(s.r, digits=5), ",  k = ", round(s.k, digits=4))
end
```

The interest rate rises and capital falls monotonically with debt — Ricardian equivalence fails. In the representative-agent limit (``\gamma=1``) the correction term is zero and debt has no real effect, restoring Ricardian equivalence.

---

## Transitional Dynamics

`blanchard_solve` linearizes the ``(k, C)`` system around the steady state and solves the saddle path. The 2×2 transition has one eigenvalue inside the unit circle (the stable convergence rate) and one outside, confirming determinacy. `blanchard_transition` then simulates convergence from an arbitrary initial capital stock.

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

## Common Pitfalls

1. **`γ = 1` is the representative-agent limit.** With certain survival the Blanchard correction vanishes, the interest rate equals ``1/\beta-1``, and Ricardian equivalence holds. Use ``\gamma < 1`` for genuine OLG effects.

2. **Large debt and multiple roots.** The OLG consumption function can admit a second, degenerate low-capital root with an implausibly high interest rate. The solver scans from high capital downward to select the economically relevant root; very large `b` may have no high-capital equilibrium (`converged` will be `false`).

3. **Log utility only.** The closed-form marginal propensity to consume ``1-\beta\gamma`` requires log utility (``\sigma = 1``). General CRRA implies a wealth- and rate-dependent propensity that this implementation does not cover.

---

## References

- Blanchard, Olivier J. 1985. "Debt, Deficits, and Finite Horizons." *Journal of Political Economy* 93 (2): 223--247. [DOI](https://doi.org/10.1086/261297)

- Yaari, Menahem E. 1965. "Uncertain Lifetime, Life Insurance, and the Theory of the Consumer." *Review of Economic Studies* 32 (2): 137--150. [DOI](https://doi.org/10.2307/2296058)

- Fujiwara, Ippei, and Yuki Teranishi. 2008. "A Dynamic New Keynesian Life-Cycle Model: Societal Aging, Demographics, and Monetary Policy." *Journal of Economic Dynamics and Control* 32 (7): 2398--2427. [DOI](https://doi.org/10.1016/j.jedc.2007.09.013)
