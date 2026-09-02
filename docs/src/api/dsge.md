# [DSGE Models API](@id api_dsge)

Specify, solve, simulate, and estimate Dynamic Stochastic General Equilibrium models. See [DSGE Models](@ref dsge_page) for the representative-agent guide and [Heterogeneity & Continuous Time](@ref dsge_heterogeneity) for the heterogeneous-agent, life-cycle, and continuous-time families documented in the second half of this page.

---

## DSGE Types

```@docs
AbstractDSGEModel
ModelSpec
ModelIR
NamedEquation
NoAgents
AbstractAgentSystem
LinearDSGE
DSGESolution
nshocks(::DSGESolution)
is_determined(::DSGESolution)
is_stable(::DSGESolution)
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
DeterminacyMap
PrunedStateSpace
report(::IO, ::DeterminacyMap{T}) where {T}
report(::IO, ::PrunedStateSpace{T}) where {T}
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
evaluate_value
```

---

## DSGE Simulation, IRF, and FEVD

`simulate`, `irf`, `fevd`, and `analytical_moments` for first-order, perturbation, and projection solutions, including the pruned higher-order state space.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/simulation.jl", "dsge/pruning.jl"]
Order   = [:function]
```

```@docs
irf(::OccBinSolution{T}, ::Int) where {T<:AbstractFloat}
irf(::BayesianDSGE{T}, ::Int) where {T<:AbstractFloat}
irf(::ModelSpec, ::Int)
fevd(::BayesianDSGE{T}, ::Int) where {T<:AbstractFloat}
```

### State-Space Moments

```@docs
solve_lyapunov
pruned_state_space
```

Unconditional moments of a first-order solution. The higher-order counterpart for [`PerturbationSolution`](@ref) appears in the block above.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/analytical.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.analytical_moments
```

### Determinacy Regions

```@docs
determinacy_region
determinacy_boundary
determinacy_label
DETERMINACY_CODES
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
reproduce(::BayesianDSGE)
posterior_mode
PosteriorMode
posterior_summary
marginal_likelihood
bridge_sampling_ml
bayes_factor
prior_posterior_table
posterior_predictive
simulate(::BayesianDSGE{T}, ::Int) where {T<:AbstractFloat}
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
HouseholdSystem
DCEGMSystem
LifeCycleSystem
ContinuousHouseholdSystem
FirmSystem
IntermediarySystem
to_spec
has_kind
agents_of
HAGrid
HAGrid()
IncomeProcess
IndividualProblem
CRRAUtility
CRRAMarginalUtility
CRRAInverseMarginalUtility
HASteadyState
HADSGESolution
KrusellSmithSolution
DenHaanAccuracy
HAGridDiagnostics
LaborSupply
LaborSupply()
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
reproduce(::KrusellSmithSolution)
ha_grid_diagnostics
adaptive_asset_grid
adapt_ha_grid
labor_supply
labor_policy
irf(::HADSGESolution{T}, ::Int) where {T<:AbstractFloat}
fevd(::HADSGESolution{T}, ::Int) where {T<:AbstractFloat}
simulate(::HADSGESolution{T}, ::Int) where {T<:AbstractFloat}
```

Formatted summaries for heterogeneous-agent steady states, solutions, and accuracy diagnostics.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/heterogeneous/display.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.report
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
MitBlock
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/heterogeneous/blocks.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.report
```

### Discrete-Continuous Choice (DCEGM)

```@docs
DCEGMProblem
DCEGMUtility
DCEGMIncome
DCEGMSolution
DCEGMDistribution
DCEGMFirm
DCEGMEquilibrium
DCEGMTransition
dcegm_solve
dcegm_policy
dcegm_choice_probabilities
dcegm_threshold
dcegm_simulate
dcegm_retirement_model
dcegm_steady_state
dcegm_capital_demand
dcegm_firm_wage
dcegm_mit
irf(::DCEGMEquilibrium{T}, ::Int) where {T<:AbstractFloat}
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/heterogeneous/dcegm.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.report
```

---

### Winberry (2018) Parametric Distributions

```@docs
ParametricDensity
WinberryFamily
fit_parametric_density
parametric_density
parametric_moments
fit_winberry
winberry_moments
winberry_histogram
winberry_quadrature
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
LifeCycleTransition
lifecycle_transition
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["olg/lifecycle.jl"]
Order   = [:function]
Filter  = f -> f === MacroEconometricModels.report
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
CTTwoAssetGE
CTTwoAssetTransition
```

### Continuous-Time Solvers

```@docs
ct_hjb
ct_kfe
ct_steady_state
ct_mit_shock
ct_two_asset_solve
ct_two_asset_ge
ct_two_asset_mit
hand_to_mouth
ceiling_mass
ct_two_asset_stationarity
irf(::CTAiyagari{T}, ::Int) where {T<:AbstractFloat}
irf(::CTTwoAsset{T}, ::Int) where {T<:AbstractFloat}
report(::IO, ::CTSteadyState{T}) where {T}
report(::IO, ::CTTwoAssetSolution{T}) where {T}
report(::IO, ::CTTwoAssetGE{T}) where {T}
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
blanchard_nk_spec
report(::IO, ::BlanchardOLGSteadyState{T}) where {T}
```

---

## Plant Heterogeneity

```@docs
KhanThomasSteadyState
KhanThomasTransition
khan_thomas_example
khan_thomas_steady_state
khan_thomas_mit
irf(::KhanThomasSteadyState{T}, ::Int) where {T<:AbstractFloat}
```

---

## Heterogeneous Banks

```@docs
IntermediarySteadyState
IntermediaryPE
IntermediaryTransition
intermediary_pe
intermediary_steady_state
intermediary_mit
irf(::IntermediarySteadyState{T}, ::Int) where {T<:AbstractFloat}
```
