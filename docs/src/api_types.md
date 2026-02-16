# [API Types](@id api_types)

This page documents all core types in **MacroEconometricModels.jl**.

## Module

```@docs
MacroEconometricModels.MacroEconometricModels
```

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

## Time Series Filters

```@docs
AbstractFilterResult
HPFilterResult
HamiltonFilterResult
BeveridgeNelsonResult
BaxterKingResult
BoostedHPResult
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

## VECM Models

```@docs
VECMModel
VECMForecast
VECMGrangerResult
```

---

## Analysis Result Types

```@docs
AbstractAnalysisResult
AbstractFrequentistResult
AbstractBayesianResult
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

## Panel VAR Types

```@docs
PVARModel
PVARStability
PVARTestResult
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

## Bayesian Posterior Types

```@docs
BVARPosterior
```

---

## Covariance Estimators

```@docs
AbstractCovarianceEstimator
NeweyWestEstimator
WhiteEstimator
DriscollKraayEstimator
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

## Model Comparison Types

```@docs
LRTestResult
LMTestResult
```

---

## Granger Causality Types

```@docs
GrangerCausalityResult
```

---

## Nowcasting Types

```@docs
AbstractNowcastModel
NowcastDFM
NowcastBVAR
NowcastBridge
NowcastResult
NowcastNews
```

---

## SVAR Identification Types

```@docs
ZeroRestriction
SignRestriction
SVARRestrictions
AriasSVARResult
UhligSVARResult
```

---

## Volatility Models

```@docs
AbstractVolatilityModel
ARCHModel
GARCHModel
EGARCHModel
GJRGARCHModel
SVModel
VolatilityForecast
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
AbstractMacroData
├── TimeSeriesData{T}
├── PanelData{T}
└── CrossSectionData{T}

DataDiagnostic
DataSummary
Frequency (enum: Daily, Monthly, Quarterly, Yearly, Mixed, Other)

AbstractARIMAModel <: StatsAPI.RegressionModel
├── ARModel{T}
├── MAModel{T}
├── ARMAModel{T}
└── ARIMAModel{T}

AbstractVARModel
├── VARModel{T}
└── VECMModel{T}

VECMForecast{T}
VECMGrangerResult{T}

PVARModel{T} <: StatsAPI.RegressionModel
PVARStability{T}
PVARTestResult{T} <: StatsAPI.HypothesisTest

AbstractAnalysisResult
├── AbstractFrequentistResult
│     ├── ImpulseResponse{T}, FEVD{T}, HistoricalDecomposition{T}
└── AbstractBayesianResult
      ├── BayesianImpulseResponse{T}, BayesianFEVD{T}, BayesianHistoricalDecomposition{T}

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

BVARPosterior{T}

AbstractUnitRootTest <: StatsAPI.HypothesisTest
├── ADFResult{T}
├── KPSSResult{T}
├── PPResult{T}
├── ZAResult{T}
├── NgPerronResult{T}
└── JohansenResult{T}

VARStationarityResult{T}

LRTestResult{T} <: StatsAPI.HypothesisTest
LMTestResult{T} <: StatsAPI.HypothesisTest
GrangerCausalityResult{T} <: StatsAPI.HypothesisTest

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

AbstractVolatilityModel <: StatsAPI.RegressionModel
├── ARCHModel{T}
├── GARCHModel{T}
├── EGARCHModel{T}
├── GJRGARCHModel{T}
└── SVModel{T}

VolatilityForecast{T}

AbstractNowcastModel
├── NowcastDFM{T}
├── NowcastBVAR{T}
└── NowcastBridge{T}

NowcastResult{T}
NowcastNews{T}
```
