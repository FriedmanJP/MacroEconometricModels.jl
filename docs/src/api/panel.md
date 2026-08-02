# [Panel Models API](@id api_panel)

Panel VAR, panel regression (FE/RE/FD/Between/CRE/AB/BB + IV + discrete choice), difference-in-differences, panel unit root and cointegration tests, and panel data utilities. See [Panel VAR](@ref pvar_page), [Panel Regression](@ref panel_reg_page), [Difference-in-Differences](@ref did_page), [Event Study LP](@ref event_study_page), and [Panel Tests](@ref tests_panel_page) for theory and examples.

---

## Panel VAR Types

```@docs
PVARModel
PVARStability
PVARTestResult
```

---

## Panel Regression Types

```@docs
PanelRegModel
PanelIVModel
PanelLogitModel
PanelProbitModel
```

---

## Panel Test Result Types

```@docs
PanelTestResult
PanelUnitRootSummary
```

### Panel Display

```@docs
report(::PanelRegModel)
report(::PanelIVModel)
report(::PanelLogitModel)
report(::PanelProbitModel)
report(::PanelTestResult)
```

---

## Difference-in-Differences Types

```@docs
DIDResult
EventStudyLP
BaconDecomposition
PretrendTestResult
NegativeWeightResult
HonestDiDResult
```

---

## LP-DiD Types

```@docs
LPDiDResult
```

---

## Panel VAR

### Estimation

```@docs
estimate_pvar
estimate_pvar_feols
```

### Structural Analysis

```@docs
pvar_oirf
pvar_girf
pvar_fevd
pvar_stability
pvar_irf(::PVARModel{T}, ::Int) where {T}
pvar_fevd_result(::PVARModel{T}, ::Int) where {T}
```

### Bootstrap

```@docs
pvar_bootstrap_irf
```

### Specification Tests

```@docs
pvar_hansen_j
pvar_mmsc
pvar_lag_selection
arellano_bond_ar_test
```

---

## Panel Regression

### Panel Linear Models

```@docs
estimate_xtreg
absorb_fe
```

### Panel Instrumental Variables

```@docs
estimate_xtiv
```

### Panel Discrete Choice

```@docs
estimate_xtlogit
estimate_xtprobit
```

### Panel Specification Tests

`hausman_test` and `breusch_pagan_test` are also defined for other model families — the pooled-ARDL Hausman test in [Multivariate Models API](@ref api_multivariate) and the cross-sectional heteroskedasticity test in [Cross-Sectional Models API](@ref api_cross_section). The panel methods are the ones documented here.

```@docs
hausman_test(::PanelRegModel{T}, ::PanelRegModel{T}) where {T}
breusch_pagan_test(::PanelRegModel{T}) where {T}
pesaran_cd_test
wooldridge_ar_test
modified_wald_test
f_test_fe
```

---

## Difference-in-Differences

### Estimation

```@docs
estimate_did
estimate_event_study_lp
estimate_lp_did
```

### Diagnostics

```@docs
bacon_decomposition
pretrend_test
negative_weight_check
```

### Sensitivity Analysis

```@docs
honest_did
```

---

## Panel Unit Root Tests

```@docs
llc_test
ips_test
breitung_panel_test
fisher_panel_test
hadri_test
panic_test
pesaran_cips_test
moon_perron_test
panel_unit_root_summary
```

---

## Panel Cointegration Tests

```@docs
pedroni_test
kao_test
westerlund_test
fisher_johansen_test
PedroniResult
KaoResult
WesterlundResult
FisherJohansenResult
```

---

## Panel Data Utilities

```@docs
panel_lag
panel_lead
panel_diff
add_panel_lag
add_panel_lead
add_panel_diff
```
