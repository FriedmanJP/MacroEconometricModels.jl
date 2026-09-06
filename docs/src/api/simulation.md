# [Simulation API](@id api_simulation)

Public data-generating processes for Monte Carlo experiments and estimator recovery checks. See [Simulation (DGPs)](@ref simulation_page) for the contract and examples.

---

## VAR Simulation and Population Moments

```@docs
dgp_var
lyapunov_gamma0
var_irf
var_fevd
var_hd
```

---

## Non-Gaussian and Heteroskedastic SVARs

```@docs
dgp_nongaussian_var
dgp_heteroskedastic_var
```

---

## Univariate Time Series

```@docs
dgp_arima
dgp_trend_cycle
dgp_ar2_peak
dgp_lagged_pair
dgp_state_space
dgp_unit_root_pair
```

---

## Cointegration, ARDL, and Panel VAR

```@docs
dgp_vecm
dgp_cointreg
dgp_panel_var
dgp_ardl
dgp_nardl
dgp_pmg
```

---

## Volatility

```@docs
dgp_garch_family
dgp_sv
dgp_mgarch
dgp_midas
```

---

## Factor Models and Mixed Frequencies

```@docs
dgp_dynamic_factors
dgp_mixed_frequency_panel
```

---

## Local Projections and Treatment Designs

```@docs
dgp_lp_iv
dgp_state_dependent_var
dgp_propensity
dgp_hac
```

---

## Cross-Section, Panel, and Staggered DiD

```@docs
dgp_cross_section
dgp_panel
dgp_staggered_did
```

---

## Regime Switching

```@docs
dgp_regime_switching
```

---

## GMM, Policy Bands, and DSGE Measurement

```@docs
dgp_gmm
dgp_pce_draws
dgp_dsge_observed
```

---

## Analytic Truth Helpers

```@docs
arma_spectrum
mm_aggregate
logit_ame
probit_ame
```

---
