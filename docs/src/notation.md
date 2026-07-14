# [Notation](@id notation)

This page collects the mathematical notation used throughout the documentation. Symbols are
consistent across every page: a symbol defined here carries the same meaning everywhere it appears.
Individual pages may introduce additional model-specific symbols, which they define locally on first
use.

---

## Core Symbols

| Symbol | Description |
|--------|-------------|
| ``y_t`` | ``n \times 1`` vector of endogenous variables at time ``t`` |
| ``Y`` | ``T \times n`` data matrix |
| ``x_t`` | ``n_x \times 1`` vector of state variables at time ``t`` |
| ``p`` | Number of lags in VAR |
| ``A_i`` | ``n \times n`` coefficient matrix for lag ``i`` |
| ``\Sigma`` | ``n \times n`` reduced-form error covariance |
| ``B_0`` | ``n \times n`` contemporaneous impact matrix |
| ``\varepsilon_t`` | ``n \times 1`` structural shocks |
| ``u_t`` | ``n \times 1`` reduced-form residuals |
| ``h`` | Forecast/impulse response horizon |
| ``H`` | Maximum horizon |

---

## Dimensions

| Symbol | Description |
|--------|-------------|
| ``T`` | Sample size / number of time periods |
| ``n`` | Number of endogenous variables |
| ``n_x`` | Number of state variables |
| ``n_y`` | Number of control variables |

---

## Structural Parameters and Solution Symbols

| Symbol | Description |
|--------|-------------|
| ``\theta`` | Deep structural parameters |
| ``\phi`` | Tempering parameter (SMC) |
| ``\sigma`` | Perturbation scaling parameter |
| ``z_t`` | Higher-order perturbation stacked endogenous vector (order ``\geq 2`` solutions) |

The symbol ``z_t`` denotes the stacked endogenous vector of a higher-order perturbation solution
(the pruned state-space state used at order ``\geq 2``; see [Nonlinear Solution Methods](@ref dsge_nonlinear)).
It is scoped to that context and never overlaps ``y_t``.

!!! note "Continuous-time override"
    In [Continuous Time](@ref dsge_continuous) the symbol ``z`` locally denotes the
    labor-productivity / idiosyncratic income state of the heterogeneous-agent problem, not the
    perturbation stacked vector. Continuous-time pages define ``z`` locally; the two meanings do not
    interact.

---

## Operators

- ``\cdot`` — matrix-vector product
- ``\otimes`` — Kronecker product
- ``\text{tr}(\cdot)`` — trace

---

See [Bibliography](@ref bibliography) for the full reference list backing the methods documented here.
