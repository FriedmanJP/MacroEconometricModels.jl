# [DSGE Models API](@id api_dsge)

Specify, solve, simulate, and estimate Dynamic Stochastic General Equilibrium models. See [DSGE Models](../dsge.md) for the full guide.

---

## DSGE Types

```@docs
AbstractDSGEModel
DSGESpec
LinearDSGE
DSGESolution
PerturbationSolution
ProjectionSolution
PerfectForesightPath
DSGEEstimation
BayesianDSGE
BayesianDSGESimulation
DSGEConstraint
VariableBound
NonlinearConstraint
ParameterTransform
OccBinConstraint
OccBinRegime
OccBinSolution
OccBinIRF
```

---

## Specification

```@docs
MacroEconometricModels.@dsge
```

---

## Steady State

```@docs
compute_steady_state
linearize
```

---

## Solution Methods

```@docs
solve
gensys
blanchard_kahn
klein
perturbation_solver
collocation_solver
pfi_solver
perfect_foresight
evaluate_policy
max_euler_error
```

### Global Solution Methods

```@docs
vfi_solver
```

---

## DSGE IRF and FEVD

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/simulation.jl", "dsge/pruning.jl"]
Order   = [:function]
```

### Simulation and Analysis

```@docs
simulate
solve_lyapunov
analytical_moments
```

---

## DSGE GMM Estimation

```@docs
estimate_dsge
```

---

## DSGE Bayesian Estimation

```@docs
estimate_dsge_bayes
posterior_summary
marginal_likelihood
bayes_factor
prior_posterior_table
posterior_predictive
```

---

## Occasionally Binding Constraints

```@docs
parse_constraint
occbin_solve
occbin_irf
```

### Constraint Constructors

```@docs
variable_bound
nonlinear_constraint
```
