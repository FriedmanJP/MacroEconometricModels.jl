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

```@docs
irf(::OccBinSolution{T}, ::Int) where {T<:AbstractFloat}
irf(::BayesianDSGE{T}, ::Int) where {T<:AbstractFloat}
fevd(::BayesianDSGE{T}, ::Int) where {T<:AbstractFloat}
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
posterior_mode
PosteriorMode
posterior_summary
marginal_likelihood
bridge_sampling_ml
bayes_factor
prior_posterior_table
posterior_predictive
mcmc_diagnostics
MCMCDiagnostics
trace
identification_diagnostics
IdentificationDiagnostics
learning_rate_check
LearningRateCheck
prior_posterior_overlap
PriorPosteriorOverlap
prior_predictive
PriorPredictiveResult
posterior_predictive_check
PosteriorPredictiveCheck
dynare_prior
InverseGamma1
```

### Observation Handling for Trending Data

```@docs
apply_prefilter
invert_prefilter
PrefilterSpec
observation_trends
ObservationTrends
detect_trend
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

---

## Heterogeneous-Agent DSGE

```@docs
HADSGESpec
HAGrid
IncomeProcess
IndividualProblem
HASteadyState
HADSGESolution
KrusellSmithSolution
DenHaanAccuracy
HAGridDiagnostics
```

### Heterogeneous-Agent Solvers and Analysis

```@docs
rouwenhorst
tauchen
load_ha_example
distribution_irf
inequality_irf
simulate_panel
den_haan_test
ha_grid_diagnostics
irf(::HADSGESolution{T}, ::Int) where {T<:AbstractFloat}
fevd(::HADSGESolution{T}, ::Int) where {T<:AbstractFloat}
report(::DenHaanAccuracy{T}) where {T}
```

### Sequence-Space Block Composition

```@docs
AbstractSSJBlock
SimpleBlock
HetBlock
SSJModel
SSJGEJacobian
SSJImpulseResponse
combine_blocks
block_jacobian
ssj_jacobian
ssj_irf
ssj_arg_order
report(::SSJModel{T}) where {T}
report(::SSJGEJacobian{T}) where {T}
report(::SSJImpulseResponse{T}) where {T}
```

### Discrete-Continuous Choice (DCEGM)

```@docs
DCEGMProblem
DCEGMSolution
DCEGMDistribution
dcegm_solve
dcegm_policy
dcegm_choice_probabilities
dcegm_threshold
dcegm_simulate
dcegm_retirement_model
report(::DCEGMSolution{T}) where {T}
report(::DCEGMDistribution{T}) where {T}
```

---

### Life-Cycle Overlapping Generations

```@docs
LifeCycleOLG
LifeCycleSteadyState
lifecycle_steady_state
lifecycle_policies
lifecycle_distribution
lifecycle_income
lifecycle_survival
report(::LifeCycleSteadyState{T}) where {T}
```

---

## Continuous-Time DSGE

```@docs
CTAiyagari
CTPoissonIncome
CTSteadyState
CTTransition
CTTwoAsset
CTTwoAssetSolution
```

### Continuous-Time Solvers

```@docs
ct_hjb
ct_kfe
ct_steady_state
ct_mit_shock
ct_two_asset_solve
report(::IO, ::CTSteadyState{T}) where {T}
report(::IO, ::CTTwoAssetSolution{T}) where {T}
```

---

## Overlapping Generations (OLG)

```@docs
BlanchardOLG
BlanchardOLGSteadyState
BlanchardOLGSolution
```

### OLG Solvers

```@docs
blanchard_steady_state
blanchard_solve
blanchard_transition
report(::IO, ::BlanchardOLGSteadyState{T}) where {T}
```
