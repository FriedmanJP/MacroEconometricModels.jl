# [Stochastic Volatility SVAR](@id id_sv_svar_page)

Stochastic-volatility (SV) identification recovers the structural impact matrix ``B_0`` by modelling each shock's log-variance as a persistent AR(1) process and estimating the full system by EM maximum likelihood (Bertsche & Braun 2022). Unlike model-free volatility methods, the estimator commits to a variance law — AR(1) log-volatility with shock-specific persistence — and in return delivers smoothed volatility paths, SV parameters, and a likelihood alongside ``B_0``.

- **Parametric SV law**: ``h_{it} = \mu_i + \phi_i (h_{i,t-1} - \mu_i) + \eta_{it}`` with shock-specific persistence ``\phi_i``
- **EM-2 estimation**: mixture Kalman smoother (E-step) plus rotation, slope, and SV-parameter updates (M-step)
- **Partial identification**: `hetero` flags the SV shocks; the rest stay homoskedastic
- **Full object**: impact matrix, VAR slopes, SV parameters, and smoothed log-volatilities in one result

For model-free volatility identification without a variance law, see [Time-Varying Volatility](@ref id_tvv_page). For regime and GARCH volatility models, see [Heteroskedasticity](@ref id_heteroskedastic_page). For an overview, see [Statistical Identification](@ref nongaussian_page). Restriction-based schemes live on [Structural Identification](@ref structural_identification_page).

```@setup svsvar
using MacroEconometricModels, Random, LinearAlgebra
rng = Xoshiro(827)
n, Tobs = 2, 800
rhos, sigs = [0.97, 0.85], [0.25, 0.20]
h = zeros(Tobs, n)
for t in 2:Tobs, i in 1:n
    h[t, i] = rhos[i] * h[t - 1, i] + sigs[i] * randn(rng)
end
E = randn(rng, Tobs, n) .* exp.(h ./ 2)
B0 = [1.0 0.4; -0.2 1.0]
A = [0.5 * Matrix{Float64}(I, n, n)]
Y = similar(E)
Y[1, :] = B0 * E[1, :]
for t in 2:Tobs
    Y[t, :] = A[1] * Y[t - 1, :] + B0 * E[t, :]
end
model = estimate_var(Y, 1; varnames=["y1", "y2"])
```

## Quick Start

**Recipe 1: Estimate the SV-SVAR by EM**

```@example svsvar
# Bertsche-Braun (2022) EM-2: rotation, slopes, and SV params jointly
sv = identify_sv_svar(Y, 1; rng=Xoshiro(828))
report(sv)
```

The SV block recovers the simulation persistence (``0.97`` and ``0.85``) and volatilities (``0.25`` and ``0.20``) to two decimals. The impact matrix ``B`` matches ``B_0`` up to signed permutation; the recovery table in the test suite pins Procrustes distance below ``0.2`` at ``T = 2000``.

**Recipe 2: Read the SV parameters**

```@example svsvar
(rhos = round.(sv.rhos, digits=2),
 sigmas = round.(sv.sigmas, digits=2),
 converged = sv.converged, iters = sv.iters)
```

**Recipe 3: Partial identification with one SV shock**

```@example svsvar
# Shock 2 stays homoskedastic: NaN SV entries, precision weight 1
sp = identify_sv_svar(Y, 1; hetero=[true, false], rng=Xoshiro(829))
(sigmas = round.(sp.sigmas, digits=2), rhos = round.(sp.rhos, digits=2))
```

**Recipe 4: Impulse responses through the standard pipeline**

```@example svsvar
ir_sv = irf(model, 12; method=:sv_em, rng=Xoshiro(828))
report(ir_sv)
```

---

## Model

Each structural shock scales with its own log-volatility following a stationary AR(1):

```math
y_t = A_1 y_{t-1} + B \varepsilon_t, \quad \varepsilon_{it} = \exp(h_{it}/2) \, u_{it}, \quad h_{it} = \mu_i + \phi_i (h_{i,t-1} - \mu_i) + \eta_{it}
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables
- ``B`` is the ``n \times n`` structural impact matrix
- ``h_{it}`` is shock ``i``'s log-variance with persistence ``|\phi_i| < 1``
- ``u_{it}`` are unit-variance shocks and ``\eta_{it}`` are volatility innovations
- ``\mu_i = -\sigma_i^2 / (2(1 - \phi_i^2))`` fixes unit unconditional shock variance

Distinct persistence parameters separate the shocks: two shocks with identical ``(\phi, \sigma)`` paths are interchangeable and their columns are not separately identified — the SV analogue of indistinct eigenvalues in regime models.

!!! note "Technical Note"
    The E-step integrates the log-``\chi^2`` state space against a normal mixture (Kim, Shephard & Chib 1998) with forward-filtering backward-sampling. The M-step updates AR(1) SV parameters by OLS-type regressions on smoothed states, VAR slopes by per-shock weighted least squares, and the impact-matrix rotation by numerical maximization over Givens angles. Impact scales stay frozen at the OLS Cholesky factor throughout: joint scale–level estimation has a finite-sample ridge that corrupts the smoothed volatilities, while the rotation is the identified object the theory concerns.

---

## Estimation

Convergence is declared on parameter stabilization (max change in ``\phi``, relative change in ``\sigma``, change in ``Q`` all below `tol`, default ``10^{-3}``), not on the log-likelihood path, which mixes E-step distributions with Monte Carlo noise. `loglik` is kept as a diagnostic. The rotation M-step warm-starts from the previous angles, so the small default `b_iter` suffices; large values chase Monte Carlo weight noise and break convergence.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `hetero` | `AbstractVector{Bool}` | All true | Flags shocks with AR(1) log-vol |
| `smoother` | `Symbol` | `:ksc` | Only supported smoother (anything else throws) |
| `maxiter` | `Int` | `500` | Maximum EM iterations |
| `tol` | `Real` | `1e-3` | Parameter-stabilization tolerance |
| `b_iter` | `Int` | `5` | Rotation optimizer steps per M-step (keep small) |
| `gibbs_burn` | `Int` | `5` | FFBS burn-in draws per E-step |
| `gibbs_draws` | `Int` | `100` | FFBS draws per E-step |
| `init` | `Symbol` | `:ols_chol` | Cholesky or Haar-random rotation start |
| `phi_init` | `Real` | `0.9` | Starting persistence |
| `s_init` | `Real` | `0.2` | Starting volatility-of-volatility |
| `theta_grid` | `Int` | `12` | Givens-grid density for start globalization (0 disables) |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Haar starts and mixture draws (pass `Xoshiro` for reproducibility) |

| Field | Type | Description |
|-------|------|-------------|
| `B` | `Matrix{T}` | ``n \times n`` structural impact matrix |
| `A` | `Vector{Matrix{T}}` | VAR slope matrices |
| `c` | `Vector{T}` | VAR intercepts |
| `mus` | `Vector{T}` | Unconditional log-volatility levels |
| `rhos` | `Vector{T}` | SV persistence (NaN for homoskedastic shocks) |
| `sigmas` | `Vector{T}` | Volatility-of-volatility (NaN for homoskedastic shocks) |
| `hetero` | `BitVector` | Flags of SV shocks |
| `H_smooth` | `Matrix{T}` | Smoothed log-volatilities (effective sample × n) |
| `loglik` | `Vector{T}` | Diagnostic log-likelihood path |
| `converged` | `Bool` | Parameter stabilization within `maxiter` |
| `iters` | `Int` | EM iterations run |

---

## Complete Example

The `compute_Q` layer exposes the rotation against `model.Sigma` for downstream analysis; the full `(A, B, SV)` object is available only from the direct call. Smoothed volatilities track the simulated variance paths shock by shock:

```@example svsvar
L = Matrix{Float64}(MacroEconometricModels.safe_cholesky(model.Sigma))
Q = Matrix{Float64}(L \ sv.B)
(norm(Q' * Q - I), size(sv.H_smooth), all(isfinite, sv.H_smooth))
```

---

## Common Pitfalls

1. **Two shocks, one volatility path.** Identical ``(\phi, \sigma)`` pairs leave the corresponding columns interchangeable. Check that `rhos` and `sigmas` differ across shocks before naming them.
2. **Raising `b_iter` to "help" convergence.** Large rotation steps overfit Monte Carlo weight noise and stall the EM loop. Leave `b_iter` at 5; raise `gibbs_draws` instead.
3. **Reading `loglik` as a convergence gauge.** The recorded path need not increase monotonically; `converged` and parameter stability are the criteria.
4. **Cost at ``n = 3``.** The Givens grid and FFBS scale steeply: ``n = 3, T = 3000`` takes about 90 seconds. Pilot timing with `maxiter` capped before long runs.
5. **Requesting another smoother.** Only `:smoother=:ksc` is implemented; anything else throws `ArgumentError`.

---

## References

- Bertsche, Dominik, and Robin Braun. 2022. "Identification of Structural Vector Autoregressions by Stochastic Volatility." *Journal of Business & Economic Statistics* 40 (1): 328--341. [DOI](https://doi.org/10.1080/07350015.2020.1813588)

- Kim, Sangjoon, Neil Shephard, and Siddhartha Chib. 1998. "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies* 65 (3): 361--393. [DOI](https://doi.org/10.1111/1467-937X.00050)
