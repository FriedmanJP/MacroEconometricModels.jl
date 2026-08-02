# [Hypothesis Tests API](@id api_tests)

Every hypothesis test in the package: unit root and cointegration, structural breaks, panel unit roots, model comparison, Granger causality, portmanteau serial correlation, and the higher-moment, bubble, and distribution tests. The [Hypothesis Tests](@ref tests_page) hub routes each family to the child page that interprets it.

!!! note "Where the worked examples live"
    The portmanteau tests (`ljung_box_test`, `box_pierce_test`, `durbin_watson_test`) are demonstrated on [Spectral Analysis](@ref spectral_page), next to their frequency-domain counterparts. The regression specification tests (`chow_test`, `cusum_test`, `breusch_godfrey_test`, `reset_test`, `white_test`, `breusch_pagan_test`) are on [Linear Regression](@ref regression_page), since each takes a fitted `RegModel`.

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

ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron, and Johansen, together with the
Fourier, DF-GLS, LM, and two-break variants and the Gregory-Hansen cointegration test with
a break. See [Unit Root Tests](@ref tests_unitroot_page) and
[Advanced Unit Root Tests](@ref tests_unitroot_advanced_page).

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

Unknown-break-date tests and multiple-break detection. See
[Structural Breaks](@ref tests_breaks_page).

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

Single-equation alternatives to Johansen, testing the residual of a cointegrating
regression. See [Cointegration Tests](@ref tests_cointegration_page).

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

Seasonal unit roots, explosive-bubble detection, nonlinearity, random-walk, and
distributional-comparison tests.

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["teststat/hegy.jl", "teststat/bubble.jl", "teststat/bds.jl", "teststat/variance_ratio.jl", "teststat/edf.jl", "teststat/equality.jl", "teststat/dumitrescu_hurlin.jl"]
```

```@docs
cor_test
```
