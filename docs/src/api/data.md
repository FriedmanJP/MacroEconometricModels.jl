# [Data Management API](@id api_data)

Typed data containers, built-in datasets, and data cleaning utilities. See [Data Management](../data.md) for theory and examples.

---

## Data Containers

```@docs
AbstractMacroData
TimeSeriesData
PanelData
CrossSectionData
Frequency
DataDiagnostic
DataSummary
```

---

## Validation and Cleaning

```@docs
diagnose
fix
validate_for_model
dropna
keeprows
```

---

## FRED Transformations

```@docs
apply_tcode
inverse_tcode
```

---

## Filtering

```@docs
apply_filter
```

---

## Summary Statistics

```@docs
describe_data
```

---

## Panel Data

```@docs
xtset
isbalanced
groups
ngroups
group_data
panel_summary
```

---

## Data Accessors and Conversion

```@docs
MacroEconometricModels.StatsAPI.nobs(::TimeSeriesData)
nvars
nlags
ncoefs
varnames
frequency
time_index
obs_id
desc
vardesc
rename_vars!
set_time_index!
set_obs_id!
set_desc!
set_vardesc!
dates
set_dates!
to_matrix
to_vector
load_example
```
