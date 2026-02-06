# [API Types](@id api_types)

This page documents all core types in **MacroEconometricModels.jl**.

## Module

```@docs
MacroEconometricModels.MacroEconometricModels
```

---

## ARIMA Models

```@docs
AbstractARIMAModel
ARModel
MAModel
ARMAModel
ARIMAModel
ARIMAForecast
ARIMAOrderSelection
```

---

## VAR Models

```@docs
VARModel
AbstractVARModel
```

---

## Impulse Response and FEVD

```@docs
ImpulseResponse
BayesianImpulseResponse
AbstractImpulseResponse
FEVD
BayesianFEVD
AbstractFEVD
```

---

## Historical Decomposition

```@docs
HistoricalDecomposition
BayesianHistoricalDecomposition
AbstractHistoricalDecomposition
```

---

## Factor Models

```@docs
FactorModel
DynamicFactorModel
GeneralizedDynamicFactorModel
FactorForecast
AbstractFactorModel
```

---

## Local Projections

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/types.jl"]
Order   = [:type]
```

---

## GMM Types

```@docs
AbstractGMMModel
GMMModel
GMMWeighting
```

---

## Prior Types

```@docs
MinnesotaHyperparameters
AbstractPrior
```

---

## Unit Root Test Types

```@docs
AbstractUnitRootTest
ADFResult
KPSSResult
PPResult
ZAResult
NgPerronResult
JohansenResult
VARStationarityResult
```

---

## SVAR Identification Types

```@docs
ZeroRestriction
SignRestriction
SVARRestrictions
AriasSVARResult
```

---

## Non-Gaussian SVAR Types

```@docs
AbstractNormalityTest
AbstractNonGaussianSVAR
NormalityTestResult
NormalityTestSuite
ICASVARResult
NonGaussianMLResult
MarkovSwitchingSVARResult
GARCHSVARResult
SmoothTransitionSVARResult
ExternalVolatilitySVARResult
IdentifiabilityTestResult
```

---

## Type Hierarchy

```
AbstractARIMAModel <: StatsAPI.RegressionModel
├── ARModel{T}
├── MAModel{T}
├── ARMAModel{T}
└── ARIMAModel{T}

AbstractVARModel
└── VARModel{T}

AbstractImpulseResponse
├── ImpulseResponse{T}
├── BayesianImpulseResponse{T}
└── AbstractLPImpulseResponse
    └── LPImpulseResponse{T}

AbstractFEVD
├── FEVD{T}
├── BayesianFEVD{T}
└── LPFEVD{T}

AbstractHistoricalDecomposition
├── HistoricalDecomposition{T}
└── BayesianHistoricalDecomposition{T}

AbstractFactorModel
├── FactorModel{T}
├── DynamicFactorModel{T}
└── GeneralizedDynamicFactorModel{T}

FactorForecast{T}

AbstractLPModel
├── LPModel{T}
├── LPIVModel{T}
├── SmoothLPModel{T}
├── StateLPModel{T}
└── PropensityLPModel{T}

StructuralLP{T}
LPForecast{T}

AbstractCovarianceEstimator
├── NeweyWestEstimator{T}
├── WhiteEstimator
└── DriscollKraayEstimator{T}

AbstractGMMModel
└── GMMModel{T}

AbstractPrior
└── MinnesotaHyperparameters{T}

AbstractUnitRootTest <: StatsAPI.HypothesisTest
├── ADFResult{T}
├── KPSSResult{T}
├── PPResult{T}
├── ZAResult{T}
├── NgPerronResult{T}
└── JohansenResult{T}

VARStationarityResult{T}

AbstractNormalityTest <: StatsAPI.HypothesisTest
└── NormalityTestResult{T}

NormalityTestSuite{T}

AbstractNonGaussianSVAR
├── ICASVARResult{T}
├── NonGaussianMLResult{T}
├── MarkovSwitchingSVARResult{T}
├── GARCHSVARResult{T}
├── SmoothTransitionSVARResult{T}
└── ExternalVolatilitySVARResult{T}

IdentifiabilityTestResult{T}
```
