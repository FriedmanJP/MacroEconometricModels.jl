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
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/utils.jl"]
Order   = [:function]
```
