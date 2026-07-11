# HA / sequence-jacobian reference fixtures

This directory holds the **committed** reference CSVs that
`test/oracle/checks_ha_ssj.jl` diffs our `method=:ssj` household Jacobians / GE
IRFs against. They are produced OFFLINE by
`test/oracle/python/gen_ha_ssj_reference.py` using the Python `sequence-jacobian`
toolkit (Auclert, Bardóczy, Rognlie & Straub 2021), which is not installable in
the read-only CI/agent environment.

Expected files (each a `T × T` comma-separated matrix, no header; `T = 200`):

| file | quantity |
|------|----------|
| `ks_J_r_A.csv`   | Krusell–Smith household Jacobian `dA/dr` |
| `ks_J_w_A.csv`   | Krusell–Smith `dA/dw` |
| `hank_J_r_A.csv` | one-asset HANK `dA/dr` |
| `hank_J_w_A.csv` | one-asset HANK `dA/dw` |

When these are absent, `checks_ha_ssj.jl` still runs its in-env consistency checks
(anticipation `J[t,s] ≠ 0` for `t < s`, Ho-Kalman realization, market clearing,
the Huggett `H_U \ (H_Z·dw)` GE-clearing identity) and prints a note that the
external cross-check was skipped.

To (re)generate:

```bash
pip install sequence-jacobian==1.0.0
python test/oracle/python/gen_ha_ssj_reference.py
git add test/oracle/ha_ssj_ref/*.csv
```
