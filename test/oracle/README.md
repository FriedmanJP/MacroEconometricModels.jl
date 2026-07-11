# Oracle harness — numerical cross-checks vs the `BVAR_` reference

This directory cross-validates MEMs.jl estimators against the Ferroni & Canova `BVAR_`
MATLAB/Octave toolbox (the audit reference).

**Reference path:** `/Users/chung/Downloads/BVAR_-master-2` (edit `octave/_setup.m` if it moves).
**Engine:** Octave 11.1.0 (`pkg load statistics`). MATLAB R2023b is a fallback for routines
Octave can't run.

## Layout

- `octave/_setup.m` — adds `bvartools/` + `cmintools/` to the path, loads `statistics`, and
  defines `OUTDIR` (`_out/`) and `DATADIR` (`_data/`). Source it at the top of every script.
- `octave/make_fixtures.m` — writes deterministic shared data to `_data/`.
- `octave/ref_*.m` — per-routine reference runners; each dumps CSVs to `_out/`.
- `fixtures.jl` — `load_fixture(name)` reads the same `_data/*.csv` into Julia.
- `compare.jl` — `read_ref(name)` + `compare(label, ours, theirs; rtol, atol)`.
- `checks_*.jl` — per-module Julia comparison drivers (run our code, compare to `_out/`).
- `_data/`, `_out/` — generated, gitignored.

## How to run

Always launch Octave **from the repository root** so relative paths resolve. Note `julia` is
NOT on the non-interactive PATH in this environment — call the version binary directly:

```bash
JULIA="$HOME/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia"
octave --no-gui test/oracle/octave/make_fixtures.m
octave --no-gui test/oracle/octave/ref_<name>.m
"$JULIA" --project=. test/oracle/checks_<module>.jl
```

## Reference conventions (verified)

- `rfvar3(ydata, lags, xdata, breaks, lambda, mu[, ww])` estimates `y = Xb + e` by SVD.
  **The constant is NOT added internally** — pass `xdata = ones(T,1)` for an intercept.
  Regressor block order is **most-recent-first lags then exogenous**:
  `X = [y(t-1) y(t-2) … y(t-p) | xdata]`, so with a constant `B` is `(n·p + nx)×n` with the
  **constant in the last row(s)**. Returns `B, u, xxi=(X'X)⁻¹, y, X`. Σ is computed by the caller.
- `lambda` (co-persistence) and `mu` (own-persistence) dummy observations are applied **inside
  `rfvar3`**, not in `varprior`.

## HA / sequence-jacobian cross-check (`checks_ha_ssj.jl`)

A second, **Python-based** oracle validates the heterogeneous-agent SSJ block
(Krusell–Smith + one-asset HANK + Huggett) against the `sequence-jacobian`
toolkit (Auclert, Bardóczy, Rognlie & Straub 2021). It is a **manual / weekly**
harness — guarded by `MACRO_ORACLE_TESTS` and NOT wired into `runtests.jl`.

- `python/gen_ha_ssj_reference.py` — pinned generator (`sequence-jacobian==1.0.0`);
  run it OFFLINE (the package is not installable in CI/agent) and commit the CSVs
  it writes to `ha_ssj_ref/`. The calibration mirrors `examples.jl` exactly,
  including the `#231` income normalization `e = exp(z)/E[exp(z)]`.
- `ha_ssj_ref/*.csv` — committed reference Jacobians (`ks_J_r_A`, `hank_J_r_A`, …).
- `checks_ha_ssj.jl` — solves the same models via `method=:ssj` and diffs against
  the fixtures when present. When they are absent it still runs the in-env
  consistency checks (anticipation `J[t,s]≠0` for `t<s` — the discriminator for
  the `#226` fake-news fix; Ho-Kalman realization consistency `#227`; steady-state
  market clearing; the Huggett `H_U \ (H_Z·dw)` GE-clearing identity) and reports
  per-quantity max abs/rel deviations.

```bash
MACRO_ORACLE_TESTS=1 "$JULIA" --project=. test/oracle/checks_ha_ssj.jl
```
