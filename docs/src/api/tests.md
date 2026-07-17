# [Hypothesis Tests API](@id api_tests)

Unit root and cointegration tests, structural break tests, model comparison, Granger causality, and portmanteau serial-correlation tests. See [Hypothesis Tests](../tests.md) for interpretation and examples.

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

## Advanced Test Types

```@docs
FourierADFResult
FourierKPSSResult
DFGLSResult
LMUnitRootResult
ADF2BreakResult
GregoryHansenResult
AndrewsResult
BaiPerronResult
FactorBreakResult
PANICResult
PesaranCIPSResult
MoonPerronResult
LLCResult
IPSResult
BreitungPanelResult
FisherPanelResult
HadriResult
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

## Portmanteau Test Types

```@docs
LjungBoxResult
BoxPierceResult
DurbinWatsonResult
BartlettWhiteNoiseResult
FisherTestResult
```

---

## Unit Root and Cointegration Tests

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["teststat/adf.jl", "teststat/kpss.jl", "teststat/pp.jl", "teststat/za.jl", "teststat/ngperron.jl", "teststat/johansen.jl", "teststat/fourier.jl", "teststat/dfgls.jl", "teststat/lm_unitroot.jl", "teststat/adf_2break.jl", "teststat/gregory_hansen.jl", "teststat/stationarity.jl", "teststat/convenience.jl"]
Order   = [:function]
```

---

## Model Comparison Tests

```@docs
lr_test
lm_test
```

---

## Granger Causality Tests

```@docs
granger_test
granger_test_all
```

---

## Portmanteau and Serial Correlation Tests

```@docs
ljung_box_test
box_pierce_test
durbin_watson_test
bartlett_white_noise_test
fisher_test
```

---

## Structural Break Tests

```@docs
andrews_test
bai_perron_test
factor_break_test
```

---

## Residual-Based Cointegration Test Types

```@docs
EngleGrangerResult
PhillipsOuliarisResult
HansenInstabilityResult
ParkAddedResult
```

---

## Residual-Based Cointegration Tests

```@docs
engle_granger_test
phillips_ouliaris_test
hansen_instability_test
park_added_test
```

---

## Higher-Moment, Bubble & Distribution Test Types

```@docs
HEGYResult
ERSResult
BubbleResult
BDSResult
VarianceRatioResult
EDFTestResult
EqualityTestResult
CorTestResult
DumitrescuHurlinResult
```

---

## Higher-Moment, Bubble & Distribution Tests

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["teststat/hegy.jl", "teststat/bubble.jl", "teststat/bds.jl", "teststat/variance_ratio.jl", "teststat/edf.jl", "teststat/equality.jl", "teststat/dumitrescu_hurlin.jl"]
```

```@docs
cor_test
```
