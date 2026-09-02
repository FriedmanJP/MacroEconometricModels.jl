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
- `sdfm_ref/` — committed FGLR Cholesky fixture (`X`, `K`, `B0`, 12-step IRF). `checks_sdfm.jl` compares `estimate_structural_dfm` within `1e-6`. Octave `ref_sdfm.m` documents the algebra; tests do not require Octave.
- `identification_ref/` — committed BVAR_ identification fixtures (`Y`, `z`, proxy `b1`, long-run impact, sign-set 16/84 bounds). `checks_identification.jl` compares without requiring Octave.
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

## Identification (`checks_identification.jl`)

Cross-checks public SVAR identification against BVAR_ committed CSVs. Uhlig MATLAB
and CRAN `vars::SVAR` are documented recipes only — no automated dump in this
harness. Numerical compare of the BVAR_ fixtures is skipped when those CSVs are
absent (never required in CI).

**BVAR_ (Ferroni & Canova, Octave 11.1.0 / MATLAB R2023b fallback).** Path:
`/Users/chung/Downloads/BVAR_-master-2` (`bvartools/iresponse_proxy.m`,
`iresponse_longrun.m`, `iresponse_sign.m`). Committed fixtures live in
`identification_ref/` (`Y`, `z`, `proxy_b1`, `long_run_Q`, sign-set 16/84
impact bounds). Tests compare Julia to those CSVs; Octave is not required at
test time.

- Proxy point estimates (`identify_proxy` `normalize=:unit_variance` vs
  `iresponse_proxy` `b1`): `rtol=1e-6`.
- Blanchard-Quah long-run impact (`chol(Σ) * identify_long_run` vs
  `iresponse_longrun` `Q`): `rtol=1e-6` after column sign alignment.
- Sign-set 16/84 impact bounds for shock 1: MC tolerance (`rtol=0.25`,
  `atol=0.20`) because Haar draws differ across RNGs.

Regenerate fixtures from the repository root:

```bash
octave --no-gui test/oracle/octave/ref_identification.m
"$JULIA" --project=. test/oracle/checks_identification.jl
```

**Uhlig MATLAB penalty (Mountford & Uhlig 2009).** Recipe only, no automated dump
in this harness. Point identification when the admissible set is a singleton. No
toolbox is bundled. The replication `.m` is the `penaltyfunction` driver from the
JAE 2009 supplement (run by hand if you have it). MATLAB is never required in CI.

**R `vars::SVAR` (Pfaff) AB-model.** Recipe only, no automated dump in this
harness. Just-identified recursive pattern should match
`estimate_svar(model, recursive_pattern(n))` / Cholesky `A \ B`. Needs `Rscript`
and CRAN `vars` if you run it by hand.

```bash
"$JULIA" --project=. test/oracle/checks_identification.jl
```
