# Serialization v1 fixtures

Committed `SERIALIZATION_FORMAT_VERSION == 1` files. Tests load them to
prove format 1 stays readable across refactors. **Regenerate only when
the format version bumps** (same policy as the CSV oracle fixtures).

| File | Object |
|---|---|
| `dsge_rbc.jld2` | small 3-equation RBC `DSGESolution` (`β=0.99`, `α=0.36`, `δ=0.025`) |
| `ha_ks.jld2` | coarse Huggett `HouseholdSystem` (`n_a=20`) |

```bash
julia --project=. -e 'include("test/dsge/gen_serialization_v1_fixtures.jl")'
```

Do not hand-edit the `.jld2` files.
