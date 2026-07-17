#!/usr/bin/env python3
"""Generate HA sequence-space-Jacobian reference fixtures for the Julia oracle.

Primary oracle for test/oracle/checks_ha_ssj.jl: the Python `sequence-jacobian`
toolkit of Auclert, Bardóczy, Rognlie & Straub (2021, Econometrica 89(5)). This
package cannot be installed in the read-only CI/agent environment, so run this
script OFFLINE and commit the CSVs it writes to `test/oracle/ha_ssj_ref/`; the
Julia harness then diffs our `method=:ssj` Jacobians against them.

Pinned environment (reproducibility):
    python >= 3.9
    sequence-jacobian == 1.0.0      # pip install sequence-jacobian==1.0.0
    numpy, scipy

Run from the repository root:
    python test/oracle/python/gen_ha_ssj_reference.py

Calibration MUST match the shipped Julia examples exactly (src/dsge/heterogeneous/
examples.jl), including the #231 income normalization e = exp(z)/E[exp(z)] so that
E[e] = 1 and every state has strictly positive labour income:

  Krusell–Smith  (_ks_example):
      alpha=0.36, beta=0.99, delta=0.025, sigma (CRRA)=1 (log),
      income = Rouwenhorst(rho=0.966, sigma=0.5, n=7), normalized to unit mean,
      asset grid a in [0, 200] with 200 points, borrowing constraint a >= 0,
      budget  c + a' = (1+r) a + w e.

  One-asset HANK  (_one_asset_hank_example):
      alpha=0.36, beta=0.986, delta=0.025, sigma=1,
      income = Rouwenhorst(0.966, 0.5, 7) normalized,
      asset grid a in [-2, 50] with 200 points, borrowing a >= -2,
      budget  c + a' = (1+r) a + w e + div   (div = 0 at the reference point).

Fixtures written (each a T x T CSV, comma-separated, no header):
    ks_J_r_A.csv     dA/dr   (household asset Jacobian wrt the interest rate)
    ks_J_w_A.csv     dA/dw
    hank_J_r_A.csv   dA/dr   (one-asset HANK)
    hank_J_w_A.csv   dA/dw
Optionally the GE IRFs (K, r, Y, C to a TFP shock) can be added the same way.

Keep T (sequence length) equal to the Julia harness `T_horizon` (200).
"""

import os
import numpy as np

# The sequence-jacobian public API changed across releases; this script targets
# the 1.0.x layout. If your installed version differs, adapt the imports below.
from sequence_jacobian import het, create_model            # noqa: E402
from sequence_jacobian.grids import agrid, markov_rouwenhorst  # noqa: E402

T_HORIZON = 200
OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "ha_ssj_ref")


def _rouwenhorst_unit_mean(rho, sigma, n):
    """Rouwenhorst discretization normalized to a unit-mean multiplier e.

    Mirrors examples.jl: e = exp(z) / E[exp(z)] under the stationary distribution,
    so E[e] = 1 and every state gives strictly positive w*e.
    """
    e, pi, Pi = markov_rouwenhorst(rho=rho, sigma=sigma, N=n)
    e = e / np.vdot(pi, e)      # unit-mean normalization (#231)
    return e, pi, Pi


@het(exogenous="Pi", policy="a", backward="Va", backward_init=None)
def household(Va_p, a_grid, e_grid, r, w, beta, eis, div):
    """One-asset consumption-savings household (EGM), CRRA(eis) utility.

    eis is the elasticity of intertemporal substitution = 1/sigma; sigma=1 (log)
    corresponds to eis=1. `div` is the lump-sum transfer in the HANK budget.
    """
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis] + div
    a = np.empty_like(coh)
    for e in range(len(e_grid)):
        a[e] = np.interp(coh[e], c_nextgrid[e] + a_grid, a_grid)
    a = np.maximum(a, a_grid[0])                 # borrowing constraint
    c = coh - a
    Va = (1 + r) * c ** (-1.0 / eis)
    return Va, a, c


def build_and_dump(name, *, alpha, beta, delta, a_min, a_max, n_a, rho, sigma_e, n_e):
    e_grid, pi, Pi = _rouwenhorst_unit_mean(rho, sigma_e, n_e)
    a_grid = agrid(amax=a_max, n=n_a, amin=a_min)

    hh = household.add_hetinputs([])  # prices supplied as calibration below
    ss = hh.steady_state(dict(
        Pi=Pi, a_grid=a_grid, e_grid=e_grid,
        beta=beta, eis=1.0, div=0.0,
        r=1.0 / beta - 1.0 - 1e-3, w=1.0,
    ))
    # Household Jacobians dA/dr, dA/dw over the T-horizon (the `curlyJ` blocks).
    J = hh.jacobian(ss, inputs=["r", "w"], T=T_HORIZON)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUT_DIR, f"{name}_J_r_A.csv"), J["A"]["r"], delimiter=",")
    np.savetxt(os.path.join(OUT_DIR, f"{name}_J_w_A.csv"), J["A"]["w"], delimiter=",")
    print(f"wrote {name}_J_r_A.csv, {name}_J_w_A.csv  (T={T_HORIZON})")


def main():
    # Krusell–Smith calibration
    build_and_dump("ks", alpha=0.36, beta=0.99, delta=0.025,
                   a_min=0.0, a_max=200.0, n_a=200,
                   rho=0.966, sigma_e=0.5, n_e=7)
    # One-asset HANK calibration
    build_and_dump("hank", alpha=0.36, beta=0.986, delta=0.025,
                   a_min=-2.0, a_max=50.0, n_a=200,
                   rho=0.966, sigma_e=0.5, n_e=7)
    print(f"\nFixtures written to {os.path.abspath(OUT_DIR)}. Commit them, then run:")
    print("  MACRO_ORACLE_TESTS=1 julia --project=. test/oracle/checks_ha_ssj.jl")


if __name__ == "__main__":
    main()
