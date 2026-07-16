# Blanchard (1985) Perpetual-Youth OLG — Design (autonomous)

**Date:** 2026-06-23 · **Release:** v0.5.1 (item 3 of 4) · **Status:** autonomous (user asleep;
decisions documented for review). On `dev`, no remote.

## Autonomous design decisions (please review)

- **Discrete-time, closed-economy neoclassical growth with perpetual youth** (the
  analytically tractable Blanchard-Yaari case), log utility (σ=1) for the closed-form
  marginal propensity to consume `1−βγ`. This matches "discrete-time Blanchard OLG
  (Fujiwara-Teranishi 2008)" at the real-side core (no NK frictions — those are a much
  larger model; the perpetual-youth demographics + non-Ricardian result are the essence).
- **Self-contained module** `src/olg/blanchard.jl` (not the HA grid or `@dsge` paradigm):
  a `BlanchardOLG` model type, a steady-state solver, and a log-linear saddle-path solver
  with IRFs to a TFP shock. Reuses `robust_inv`, display helpers, conventions.
- **Government debt** `b` included as a parameter to demonstrate the hallmark
  **non-Ricardian** result (debt is net wealth ⟹ crowds out capital, raises `r`).

## Model (verified)

Per capita, population constant (`n=0`). Survival probability `γ ∈ (0,1]` each period;
fair annuities pay survivors gross return `(1+r)/γ`. CRRA log utility, effective discount
`βγ`. Production `f(k)=Z k^α`, `r=αZk^{α-1}−δ`, `w=(1−α)Z k^α`.

Individual Euler is standard (annuity cancels survival): `c_{t+1}/c_t = β(1+r_{t+1})`.
Aggregating with newborns entering at zero financial wealth and eliminating human wealth
`H_t = w_t + (γ/(1+r_{t+1}))H_{t+1}` from the consumption function
`C_t = (1−βγ)[(1+r_t)(k_t+b) + H_t]` yields the **aggregate system**:

```
Euler:  C_{t+1} = (1+r_{t+1}) · [ β·C_t − ((1−βγ)(1−γ)/γ)·(k_{t+1}+b) ]
Budget: k_{t+1} = (1+r_t)·k_t + w_t − τ_t − C_t,     τ_t = r_t·b   (constant debt)
Prices: r_t = αZ k_t^{α-1} − δ,   w_t = (1−α)Z k_t^α
```

- The `(1−γ)` wedge is the Blanchard correction (newborn turnover). `γ=1` ⟹ wedge 0 ⟹
  Ramsey Euler `C_{t+1}/C_t = β(1+r)` ⟹ `r* = 1/β−1`.
- For `γ<1`, SS requires `β(1+r)>1` ⟹ `r* > 1/β−1` (finite-horizon result).

## Steady state

Solve `C = r·k + w − r·b` (SS budget) `= (1+r)((1−βγ)(1−γ)/γ)(k+b) / (β(1+r)−1)`
(SS Euler) for `k*` by bracketed bisection on `k` (`r,w` functions of `k`). Return
`(k, C, r, w, H, b)`. Guard: with `b=0, γ=1`, `k*` matches the Ramsey `r=1/β−1` capital.

## Dynamics

Log-linearize `(k_t, C_t)` around SS with a TFP shock `Z_t = ρ_Z Z_{t-1} + ε_t`
(3 variables: `k` predetermined, `C` jump, `Z` exogenous). Build `(Γ0,Γ1,Ψ,Π)` and solve
via the existing linear DSGE machinery (`gensys`/companion-QZ) → state transition + impact
→ `irf`. Saddle-path determinacy (one stable, one unstable root for the `(k,C)` block).

## API

- `BlanchardOLG(; alpha, beta, delta, gamma, Z, b=0.0, rho_Z=0.9, sigma_Z=0.01)` → model.
- `blanchard_steady_state(m)` → `BlanchardOLGSteadyState{T}` (k, C, r, w, H, mpc).
- `blanchard_solve(m)` → `BlanchardOLGSolution{T}` (wraps a `DSGESolution` for `irf`/`fevd`).
- `report` / `show` for both. Export the type names + functions.

## Tests (acceptance)

1. **Ramsey limit:** `γ=1, b=0` ⟹ `r* ≈ 1/β−1` (to tol); `k*` matches `(αZ/(r+δ))^{1/(1-α)}`.
2. **Finite-horizon:** `γ<1` ⟹ `r* > 1/β−1`; the gap increases as `γ` falls (more death).
3. **Non-Ricardian:** `∂k*/∂b < 0` and `∂r*/∂b > 0` (debt crowds out capital).
4. **Dynamics:** saddle-path stable (spectral radius of the solved transition ≤ 1);
   a positive TFP shock raises `k` and `C` on impact with sensible signs.
5. **Display:** `report` runs.

## Files

- `src/olg/blanchard.jl` (new): types, steady state, dynamics, display.
- `src/MacroEconometricModels.jl`: `include` + exports.
- `test/dsge/test_blanchard_olg.jl` (new) or a section in an existing test file.
- `docs/src/` — a short OLG section (likely a new `olg.md` or appended to a DSGE page) +
  Blanchard (1985) reference; verify examples.

## References

- Blanchard, O. J. (1985). Debt, Deficits, and Finite Horizons. *JPE*, 93(2), 223–247.
- Yaari, M. E. (1965). Uncertain Lifetime, Life Insurance, and the Theory of the Consumer.
  *Review of Economic Studies*, 32(2), 137–150.
- Fujiwara, I., & Teranishi, Y. (2008). A dynamic new Keynesian life-cycle model.
  *JEDC*, 32(7), 2398–2427.
