# DSGE: Arbitrary Lag/Lead Lengths and News Shocks — Design

**Issue**: #54
**Date**: 2026-02-25

## Goal

Extend the DSGE module to support arbitrary lag/lead depths on endogenous variables (`y[t-k]`, `y[t+k]` for any integer k) and news/anticipated shocks on exogenous variables (`ε[t-k]`). Currently only `[t]`, `[t-1]`, `[t+1]` are supported.

## Architecture: Augment at Macro Time

The `@dsge` macro performs automatic state-space augmentation before compiling residual functions. After augmentation, all offsets are reduced to ±1 or 0, so the entire downstream pipeline (linearize, gensys, BK, simulate, IRF, FEVD) works unchanged on a larger system. Auxiliary variables are hidden from user output.

## Augmentation Rules

### Endogenous lags: `var[t-k]` where k > 1

Create k-1 auxiliary endogenous variables and k-1 identity equations:

```
_var_lag1[t] = var[t-1]
_var_lag2[t] = _var_lag1[t-1]
...
_var_lag{k-1}[t] = _var_lag{k-2}[t-1]
```

Substitute: `var[t-k]` → `_var_lag{k-1}[t-1]`

### Endogenous leads: `var[t+k]` where k > 1

Create k-1 auxiliary endogenous variables and k-1 identity equations:

```
_var_fwd1[t] = var[t+1]
_var_fwd2[t] = _var_fwd1[t+1]
...
_var_fwd{k-1}[t] = _var_fwd{k-2}[t+1]
```

Substitute: `var[t+k]` → `_var_fwd{k-1}[t+1]`

### Exogenous news shocks: `ε[t-k]` where k > 0

Create k auxiliary endogenous variables (shocks become states) and k identity equations:

```
_news_ε_1[t] = ε[t]        (captures current shock realization)
_news_ε_2[t] = _news_ε_1[t-1]  (1-period-old shock)
...
_news_ε_k[t] = _news_ε_{k-1}[t-1]  (k-1-period-old shock)
```

Substitute: `ε[t-k]` → `_news_ε_k[t-1]`

Verification: `_news_ε_j[t] = ε[t-j+1]`, so `_news_ε_k[t-1] = ε[t-k]`. ✓

## Type Changes

### DSGESpec{T} — new fields

| Field | Type | Description |
|---|---|---|
| `original_endog` | `Vector{Symbol}` | User's original endogenous variable names |
| `n_original_endog` | `Int` | Length of original_endog |
| `n_original_eq` | `Int` | Number of user-written equations |
| `augmented` | `Bool` | Whether augmentation was performed |
| `max_lag` | `Int` | Maximum lag depth (1 if no augmentation) |
| `max_lead` | `Int` | Maximum lead depth (1 if no augmentation) |

### DSGESolution{T} — new fields

| Field | Type | Description |
|---|---|---|
| `original_var_indices` | `Vector{Int}` | Indices of original variables in augmented state vector |
| `original_endog` | `Vector{Symbol}` | Copy from spec |

### LinearDSGE{T} — unchanged

Matrices are just larger after augmentation.

## Pipeline Impact

| Component | Change |
|---|---|
| `@dsge` parser | Scan for deep offsets → generate auxiliaries → substitute → compile |
| `DSGESpec` | New metadata fields |
| `linearize` | None (augmented spec is standard lag-1/lead-1) |
| `gensys` / `blanchard_kahn` | None (generic matrix solvers) |
| `steady_state` | None (identity equations self-consistent at SS) |
| `simulate` | Filter output to `original_var_indices` |
| `irf` / `fevd` | Filter output to original variables |
| `perfect_foresight` | Filter output to original variables |
| `occbin` | Filter output to original variables |
| `estimate_dsge` | None (uses filtered IRF output) |
| `display.jl` | Show only first `n_original_eq` equations, use `original_endog` |
| `plot_result` | None (receives already-filtered data) |

## Steady State Correctness

At steady state (`y_t = y_{t-1} = y_{t+1} = y`, `ε = 0`):

- **Endogenous lag auxiliaries**: `_var_lag1 = var`, `_var_lag2 = _var_lag1 = var`, etc. All equal the original variable. ✓
- **News shock auxiliaries**: `_news_ε_1 = ε = 0`, `_news_ε_2 = _news_ε_1 = 0`, etc. All zero. ✓

The existing steady state solver works unchanged.

## Backward Compatibility

For models with max lag/lead = 1 and no exogenous lags: `augmented = false`, `original_endog == endog`, `original_var_indices = 1:n_endog`. Identical behavior to current code.

## Testing

1. **AR(2)**: `y[t] = a1*y[t-1] + a2*y[t-2] + σ*ε[t]` — verify augmentation, solve, IRF
2. **News shock**: `A[t] = ρ*A[t-1] + σ_0*ε_A[t] + σ_8*ε_A[t-8]` — verify shock appears at t+8
3. **Higher-order lead**: `y[t] = a*y[t+2] + ε[t]` — verify augmentation and solve
4. **Backward compat**: existing lag-1/lead-1 model unchanged
5. **Output filtering**: IRF/FEVD/simulate show only original variables
6. **Display**: show original equations only, report augmentation status
7. **Metadata**: max_lag, max_lead, n_original_endog correct
8. **Mixed**: deep lags + news shocks in same model

## References

- Beaudry, P. & Portier, F. (2006). "Stock Prices, News, and Economic Fluctuations." *AER* 96(4): 1293–1307.
- Schmitt-Grohé, S. & Uribe, M. (2012). "What's News in Business Cycles." *Econometrica* 80(6): 2733–2764.
- Barsky, R. B. & Sims, E. R. (2011). "News Shocks and Business Cycles." *JME* 58(3): 273–289.
- Fujiwara, I., Hirose, Y. & Shintani, M. (2011). "Can News Be a Major Source of Aggregate Fluctuations?" *JMCB* 43(1): 1–29.
