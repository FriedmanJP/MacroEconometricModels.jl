# [GMM & SMM API](@id api_gmm)

Generalized Method of Moments and Simulated Method of Moments estimation. See [GMM & SMM](../gmm.md) for theory and examples.

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
```

### GMM Utilities

```@docs
linear_gmm_solve
gmm_sandwich_vcov
andrews_lu_mmsc
```
