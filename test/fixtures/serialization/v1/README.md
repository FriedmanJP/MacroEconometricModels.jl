# Serialization v1 fixtures

Committed `SERIALIZATION_FORMAT_VERSION == 1` files. Tests load them to
prove format 1 stays readable across refactors. **Regenerate only when
the format version bumps** (same policy as the CSV oracle fixtures).

| File | Object |
|---|---|
| `dsge_rbc.jld2` | small 3-equation RBC `DSGESolution` (`β=0.99`, `α=0.36`, `δ=0.025`) |
| `ha_ks.jld2` | coarse Huggett `HouseholdSystem` (`n_a=20`) |
| `var_irf.jld2` | bundle: `VARModel` + `ImpulseResponse` (`note="v1 VAR+IRF"`) |
| `bvar.jld2` | `BVARPosterior` (`n_draws=30`, `seed=787`) |
| `arima.jld2` | `ARIMAModel` ARIMA(1,0,1) CSS |
| `garch.jld2` | `GARCHModel` GARCH(1,1) |
| `factor.jld2` | `FactorModel` (`r=2`) |
| `nowcast_dfm.jld2` | `NowcastDFM` (`r=1`, 4 monthly + 1 quarterly) |
| `did_event_study.jld2` | `EventStudyLP` (horizon 3) |
| `opp.jld2` | `OPPResult` (H=6 plug-in) |
| `teststat.jld2` | bundle: `ADFResult` + `KPSSResult` (`note="v1 teststat"`) |

```bash
julia --project=. -e 'include("test/dsge/gen_serialization_v1_fixtures.jl")'   # DSER RBC/HA
julia --project=. test/gen_serialization_v1_fixtures.jl                        # RSER family
```

Do not hand-edit the `.jld2` files.
