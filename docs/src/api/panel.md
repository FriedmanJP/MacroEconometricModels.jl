# [Panel Models API](@id api_panel)

Panel VAR, panel regression (FE/RE/FD/Between/CRE/AB/BB + IV + discrete choice), difference-in-differences, panel unit root tests, and panel data utilities. See [Panel VAR](../pvar.md), [Panel Regression](../panel_reg.md), [DiD](../did.md), and [Event Study](../event_study.md).

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

```@docs
hausman_test
breusch_pagan_test
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
panic_test
pesaran_cips_test
moon_perron_test
panel_unit_root_summary
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
