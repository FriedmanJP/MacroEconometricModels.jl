# [Utilities & Display API](@id api_utilities)

HAC/robust covariance estimators, output/display infrastructure, bibliographic references, and low-level numerical utilities.

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
```

---

## License

```@docs
warranty
conditions
```
