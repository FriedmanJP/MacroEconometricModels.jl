# [Hypothesis Tests](@id tests_page)

MacroEconometricModels.jl provides a comprehensive suite of statistical hypothesis tests for macroeconomic time series analysis. The battery covers pre-estimation diagnostics --- integration order, cointegration, structural breaks, panel stationarity --- and post-estimation specification checks --- Granger causality, normality, ARCH effects, independence, distributional fit, and nested model comparison.

The six child pages divide the battery by question. Two pages cover univariate integration order (standard tests and the high-power variants designed for breaks, seasonality, and bubbles), one covers residual-based cointegration, one covers structural breaks, one covers panel data, and one covers everything applied to an estimated model. This page owns the shared material: the decision tables that route a question to a test, and the StatsAPI interface every result type implements.

```@setup tests_overview
using MacroEconometricModels, Random
import MacroEconometricModels: StatsAPI
const pvalue = StatsAPI.pvalue
Random.seed!(42)
```

## Quick Start

Test a price level for a unit root with automatic lag selection:

```@example tests_overview
fred = load_example(:fred_md)
cpi = filter(isfinite, fred[:, "CPIAUCSL"])
result = adf_test(cpi; lags=:aic, regression=:constant)
report(result)
```

The test statistic of 2.15 lies far above the 5% critical value of ``-2.85``, so the test fails to reject the unit root null --- the expected verdict for a price *level*. Confirm the finding with `kpss_test`, whose null is stationarity, before differencing the series.

---

## Choosing a Method

### Integration Order

The first step before specifying a VAR, VECM, or Local Projection. See [Unit Root & Cointegration](@ref tests_unitroot_page) and [Advanced Unit Root](@ref tests_unitroot_advanced_page).

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Baseline unit root test | `adf_test` | Automatic lag selection via AIC |
| Confirmation under the opposite null | `kpss_test` | Resolves non-rejection ambiguity |
| Serially correlated errors | `pp_test` | Non-parametric variance correction |
| One unknown structural break | `za_test` | Break date estimated endogenously |
| Small sample (``T < 100``) | `ngperron_test` | GLS detrending improves size |
| Smooth or gradual breaks | `fourier_adf_test`, `fourier_kpss_test` | No break dates to specify |
| Maximum power near unity | `dfgls_test`, `ers_test` | Near-optimal against local alternatives |
| Breaks present under the null | `lm_unitroot_test` | Rejection is unambiguous |
| Two break dates | `adf_2break_test` | Grid search over both dates |
| Seasonal unit roots | `hegy_test` | Tests each seasonal frequency |
| Explosive bubble behaviour | `sadf_test`, `gsadf_test` | Right-tailed sup-ADF with date-stamping |
| Quick multi-test verdict | `unit_root_summary` | Combines ADF, KPSS, and PP |
| Every column of a data matrix | `test_all_variables` | One call screens the panel |

### Cointegration

See [Unit Root & Cointegration](@ref tests_unitroot_page) for the system approach and [Residual-Based Cointegration](@ref tests_cointegration_page) for single-equation tests.

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Cointegrating rank of a system | `johansen_test` | Trace and maximum-eigenvalue statistics |
| Single equation, rank not needed | `engle_granger_test` | ADF on the cointegrating residual |
| Serial correlation in the residual | `phillips_ouliaris_test` | Semiparametric long-run variance |
| Null of cointegration | `hansen_instability_test` | ``L_c`` parameter-instability statistic |
| Superfluous-regressor check | `park_added_test` | ``H(p,q)`` added-variables test |
| Regime shift in the relationship | `gregory_hansen_test` | Break allowed under the alternative |
| Panel cointegration | `pedroni_test`, `kao_test`, `westerlund_test` | Pool evidence across units |

### Structural Stability

Parameter instability at unknown dates. See [Structural Breaks](@ref tests_breaks_page).

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Single unknown break | `andrews_test` | sup-, exp-, and mean-Wald functionals |
| Multiple unknown breaks | `bai_perron_test` | Dynamic programming with BIC/LWZ |
| Factor loading instability | `factor_break_test` | Pooled per-series LM, sup-Wald |

### Panel Data

Panel tests account for cross-sectional dependence that invalidates pooled time-series tests. See [Panel Tests](@ref tests_panel_page).

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| Factor-driven dependence | `panic_test` | Separates common and idiosyncratic parts |
| General cross-section dependence | `pesaran_cips_test` | Augments ADF with cross-section means |
| Large ``N``, moderate ``T`` | `moon_perron_test` | Factor-adjusted pooled statistics |
| Cross-sectionally independent panel | `llc_test`, `ips_test`, `fisher_panel_test` | First-generation, no factor structure |
| Null of panel stationarity | `hadri_test` | KPSS-type null for panels |
| Heterogeneous panel causality | `dh_causality_test` | Coefficients vary across units |
| Comprehensive panel check | `panel_unit_root_summary` | Three tests, one table |

### Post-Estimation Diagnostics

Specification tests that validate the assumptions underlying inference. See [Model Diagnostics](@ref tests_diagnostics_page).

| Feature needed | Recommended | Why |
|----------------|-------------|-----|
| VAR stability | `is_stationary` | Companion eigenvalue check |
| Predictive causality | `granger_test`, `granger_test_all` | Pairwise and block Wald tests |
| Residual normality | `normality_test_suite` | Seven tests in one call |
| Conditional heteroskedasticity | `arch_lm_test`, `ljung_box_squared` | Detects remaining ARCH effects |
| Nonlinear dependence | `bds_test` | Finds structure linear tests miss |
| Distributional fit | `edf_test` | Anderson-Darling and Kolmogorov-Smirnov |
| Random-walk behaviour | `variance_ratio_test` | Heteroskedasticity-robust Lo-MacKinlay |
| Nested model comparison | `lr_test`, `lm_test` | Likelihood ratio and score tests |
| Group and rank comparisons | `equality_test`, `cor_test` | Distribution equality and rank correlation |
| PVAR instrument validity | `pvar_hansen_j` | Overidentifying restrictions |
| PVAR lag order | `pvar_mmsc`, `pvar_lag_selection` | Andrews-Lu MMSC criteria |

---

## Child Pages

- [Unit Root & Cointegration](@ref tests_unitroot_page) --- ADF, KPSS, Phillips-Perron, Zivot-Andrews, and Ng-Perron tests, the `unit_root_summary` and `test_all_variables` batch utilities, and the Johansen system cointegration test
- [Advanced Unit Root](@ref tests_unitroot_advanced_page) --- Fourier ADF/KPSS, DF-GLS and ERS point-optimal, HEGY seasonal roots, LM unit root with 0/1/2 breaks, two-break ADF, and SADF/GSADF bubble detection
- [Residual-Based Cointegration](@ref tests_cointegration_page) --- Engle-Granger two-step, Phillips-Ouliaris, Hansen ``L_c`` instability, and Park ``H(p,q)`` added-variables tests
- [Structural Breaks](@ref tests_breaks_page) --- Andrews single-break, Bai-Perron multiple-break, factor model break tests, and Gregory-Hansen cointegration with a regime shift
- [Panel Tests](@ref tests_panel_page) --- first- and second-generation panel unit root tests, PANIC, Pesaran CIPS, Moon-Perron, panel cointegration, Dumitrescu-Hurlin causality, and Panel VAR specification tests
- [Model Diagnostics](@ref tests_diagnostics_page) --- VAR stationarity, Granger causality, normality suite, ARCH diagnostics, BDS independence, EDF goodness-of-fit, variance-ratio tests, and nested model comparison

---

## StatsAPI Interface

All test result types implement the StatsAPI.jl interface, providing a uniform way to extract test statistics, p-values, and degrees of freedom regardless of the specific test.

```@example tests_overview
fred = load_example(:fred_md)
cpi = filter(isfinite, fred[:, "CPIAUCSL"])
result = adf_test(cpi; lags=:aic, regression=:constant)

# Uniform interface across all test types
nobs(result)      # number of observations
dof(result)       # degrees of freedom
pvalue(result)    # p-value
```

### Type Hierarchy

The results form a three-level hierarchy rooted at `StatsAPI.HypothesisTest`:

```
StatsAPI.HypothesisTest
  AbstractUnitRootTest      # unit root, cointegration, break, and panel results
    ADFResult{T}, KPSSResult{T}, PPResult{T}, ZAResult{T}, NgPerronResult{T}, ...
  AbstractNormalityTest     # multivariate normality results
    NormalityTestResult{T}
  GrangerCausalityResult{T}, LRTestResult{T}, LMTestResult{T}, PVARTestResult{T}, ...
```

`AbstractUnitRootTest` is the widest branch: every unit root, cointegration, structural break, and panel result type descends from it, including the newer `BubbleResult`, `HEGYResult`, `ERSResult`, `EDFTestResult`, `VarianceRatioResult`, `BDSResult`, and `DumitrescuHurlinResult` types. The complete catalog with field documentation lives on the [Hypothesis Tests API](@ref api_tests) page.

For tests with multiple statistics (Ng-Perron returns four, Moon-Perron returns two), `pvalue()` returns the primary statistic's p-value (MZt for Ng-Perron, ``t_a^*`` for Moon-Perron). Access individual statistics through the result fields documented on each child page.

!!! note "Two results are not hypothesis tests"
    `is_stationary` returns a `VARStationarityResult`, an eigenvalue diagnostic with no p-value. The `NormalityTestSuite` returned by `normality_test_suite` is a container holding multiple `NormalityTestResult` objects. Iterate over `suite.results` for individual p-values, or use `report(suite)` for a consolidated display.

---

## Common Pitfalls

1. **Applying time-series unit root tests to panels.** Standard ADF, KPSS, and PP tests assume independent observations and produce invalid inference under cross-sectional dependence. For panel data, use `panic_test`, `pesaran_cips_test`, or `moon_perron_test`, which explicitly account for common factor structures.

2. **Running Granger causality on non-stationary data.** The asymptotic ``\chi^2`` distribution for the Granger Wald test requires the VAR to be stationary. Apply unit root tests first, difference or cointegration-adjust the data, and verify `is_stationary(model)` before interpreting Granger causality results.

---

## References

- Enders, Walter. 2014. *Applied Econometric Time Series*. 4th ed.
  Hoboken, NJ: Wiley. ISBN 978-1-118-80856-6.

- Hamilton, James D. 1994. *Time Series Analysis*.
  Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
