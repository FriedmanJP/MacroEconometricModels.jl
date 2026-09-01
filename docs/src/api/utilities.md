# [Utilities & Display API](@id api_utilities)

HAC and robust covariance estimators, the long-run-variance toolkit, output and display infrastructure, bibliographic references, logging, reproducibility manifests, model serialization, and low-level numerical utilities.

---

## Covariance Estimator Types

```@docs
AbstractCovarianceEstimator
NeweyWestEstimator
WhiteEstimator
DriscollKraayEstimator
```

---

## Covariance Estimators

```@docs
newey_west
white_vcov
driscoll_kraay
robust_vcov
long_run_variance
long_run_covariance
optimal_bandwidth_nw
register_cov_estimator!
```

---

## Long-Run Variance Toolkit

```@docs
lrvar
lrcov
lrcov_oneside
varhac
optimal_bandwidth_nw94
```

---

## Display and References

`set_display_backend` and `with_display_backend` select how `report` renders tables; `table` builds the tabular views used throughout the package. See [Visualization](@ref plotting_page) for the plotting counterpart.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/display.jl"]
Order   = [:function]
```

```@docs
refs
```

### Output Tables

```@docs
table
print_table
long_table
write_csv
```

### Logging

```@docs
set_log_level
with_min_level
```

### Reproducibility

`reproduce` is documented here in its generic form; the per-result methods for [`BVARPosterior`](@ref) and bootstrap [`ImpulseResponse`](@ref) results appear in [Multivariate Models API](@ref api_multivariate).

```@docs
ReproManifest
capture_manifest
reproduce(::Any)
ReproReport
```

### Serialization

JLD2 is a package dependency. The living catalog of saveable types is the Persistence table on the [API Reference](@ref api_page); narrative coverage (bundles, `note=`, `model_info`, compression, the executed-code caveat) lives on [Data Management](@ref data_page).

```@docs
save_model
load_model
model_info
SERIALIZATION_FORMAT_VERSION
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/utils.jl"]
Order   = [:function]
```

### Numerical Tolerances

```@docs
default_abstol
default_reltol
```

---

## Exceptions

```@docs
MacroModelError
ConvergenceError
IdentificationError
SingularSystemError
SerializationError
```

---

## License

```@docs
warranty
conditions
```
