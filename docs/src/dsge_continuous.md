# [Continuous Time](@id dsge_continuous)

Continuous-time heterogeneous-agent models solved with the finite-difference methods of **Achdou, Han, Lasry, Lions & Moll (2022)**. The household problem is a **Hamilton-Jacobi-Bellman (HJB)** partial differential equation solved by an implicit upwind scheme, and the stationary wealth distribution solves the **Kolmogorov-Forward (Fokker-Planck)** equation. The elegance of the approach is that a single sparse infinitesimal generator ``A`` drives both: the HJB implicitly, and the KFE through its transpose ``A^\top``.

This page covers the one-asset Aiyagari model. The same machinery is the foundation for two-asset (Kaplan-Moll-Violante) models and MIT-shock transitions.

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
v, c, s, A, a, converged = MacroEconometricModels.ct_hjb(m, r, w)
(hjb_converged = converged,
 generator_row_sums = round(maximum(abs.(vec(sum(A; dims=2)))), sigdigits=2))
```

The generator's rows sum to zero (here to machine precision), confirming that ``A`` is a valid infinitesimal generator — the discretized drift and the income switching conserve probability mass.

---

## The Kolmogorov-Forward Equation

The stationary density ``g(a,z)`` solves ``A^\top g = 0`` subject to ``\int g\, da = 1``. Because the generator from the HJB is exactly the operator governing the distribution's flow, the KFE reuses it directly:

```@example ct
da = a[2] - a[1]
g = MacroEconometricModels.ct_kfe(A, m.I, da)
(density_nonnegative = minimum(g) >= -1e-10,
 integrates_to_one = round(sum(g) * da, digits=8))
```

The density is nonnegative and integrates to one. Mass piles up at the borrowing constraint ``a_{\min}``, where households with low income are stuck — the continuous-time analog of the kink in the discrete-time policy function.

---

## Stationary Equilibrium

`ct_steady_state` bisects on the interest rate ``r`` until household-supplied capital ``\int a\, g`` equals firm capital demand from the Cobb-Douglas first-order condition ``r = \alpha Z (K/L)^{\alpha-1} - \delta``, with the wage ``w = (1-\alpha) Z (K/L)^{\alpha}`` and effective labor ``L = \int z\, g``.

```@example ct
(interest_rate = round(ss.r, digits=5),
 capital = round(ss.K, digits=4),
 effective_labor = round(ss.L, digits=4),
 fraction_constrained = round((ss.g[1,1] + ss.g[1,2]) * (ss.a[2]-ss.a[1]), digits=4))
```

The equilibrium interest rate lies strictly below the discount rate ``\rho``: incomplete markets and precautionary saving push the supply of capital up and the return down, exactly as in the discrete-time Aiyagari (1994) economy.

| Field | Type | Description |
|-------|------|-------------|
| `r`, `w` | `T` | Equilibrium interest rate and wage |
| `K`, `L` | `T` | Aggregate capital ``\int a g`` and effective labor ``\int z g`` |
| `a` | `Vector{T}` | Wealth grid |
| `g` | `Matrix{T}` | Stationary density over ``(a, z)`` (``I \times 2``) |
| `v`, `c`, `s` | `Matrix{T}` | Value, consumption, and saving drift |
| `A` | `SparseMatrixCSC{T}` | Infinitesimal generator (``2I \times 2I``) |

---

## Common Pitfalls

1. **Grid resolution.** The implicit upwind scheme is first-order accurate. Increase `I` (and `a_max`) for sharper policy functions and a more accurate constrained mass; `I = 500`–`1000` is typical for publication.

2. **Interest-rate bounds.** Equilibrium `r` lies in ``(0, \rho)``. As ``r \to \rho`` aggregate saving diverges, so the default upper bound is ``\rho - 10^{-4}``.

3. **Implicit step size `Delta`.** A large `Delta` (default `1000`) makes the implicit HJB iteration converge in tens of steps. It controls only the speed of the value-function iteration, not the solution.

---

## References

- Achdou, Yves, Jiequn Han, Jean-Michel Lasry, Pierre-Louis Lions, and Benjamin Moll. 2022. "Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach." *Review of Economic Studies* 89 (1): 45--86. [DOI](https://doi.org/10.1093/restud/rdab002)

- Aiyagari, S. Rao. 1994. "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109 (3): 659--684. [DOI](https://doi.org/10.2307/2118417)

- Kaplan, Greg, Benjamin Moll, and Giovanni L. Violante. 2018. "Monetary Policy According to HANK." *American Economic Review* 108 (3): 697--743. [DOI](https://doi.org/10.1257/aer.20160042)
