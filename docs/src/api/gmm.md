# [GMM & SMM API](@id api_gmm)

Generalized Method of Moments and Simulated Method of Moments estimation. See [Generalized & Simulated Method of Moments](@ref gmm_page) for theory and examples.

---

## GMM Types

```@docs
AbstractGMMModel
GMMModel
SMMModel
GMMWeighting
```

---

## GMM Estimation

Objective, weighting matrices, the estimator itself, overidentification testing, and the linear-GMM and sandwich-covariance utilities.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm/gmm.jl"]
Order   = [:function]
```

### Simulated Method of Moments

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm/smm.jl"]
Order   = [:function]
```

### Parameter Transforms

```@docs
to_unconstrained
to_constrained
transform_jacobian
log_jacobian
```
