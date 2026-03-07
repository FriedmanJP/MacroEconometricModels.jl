# [Hypothesis Tests](@id tests_page)

MacroEconometricModels.jl provides a comprehensive suite of statistical hypothesis tests for macroeconomic time series analysis. The test battery covers pre-estimation diagnostics (unit root, cointegration, structural breaks, panel stationarity), post-estimation specification checks (Granger causality, normality, ARCH effects, model comparison), and panel-level instrument validation. All result types implement the StatsAPI interface for uniform access to test statistics, p-values, and degrees of freedom.

- **[Unit Root & Cointegration](tests_unitroot.md)**: ADF, KPSS, Phillips-Perron, Zivot-Andrews, and Ng-Perron tests for univariate stationarity; Johansen cointegration test
- **[Advanced Unit Root](tests_unitroot_advanced.md)**: Fourier ADF/KPSS, DF-GLS/ERS, LM unit root (0/1/2 breaks), and two-break ADF for improved power under structural breaks
- **[Structural Breaks](tests_breaks.md)**: Andrews single-break, Bai-Perron multiple-break, and factor loading stability tests for detecting parameter instability
- **[Panel Tests](tests_panel.md)**: PANIC, Pesaran CIPS, and Moon-Perron tests for panel unit roots with cross-sectional dependence
- **[Model Diagnostics](tests_diagnostics.md)**: Granger causality, multivariate normality, ARCH-LM, Ljung-Box, Hansen J, Andrews-Lu MMSC, and likelihood ratio / Lagrange multiplier tests

```@setup tests_overview
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

**Recipe 1: ADF unit root test**

```@example tests_overview
# Test CPI price level for a unit root
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))
result = adf_test(cpi; lags=:aic, regression=:constant)
report(result)
```

**Recipe 2: Andrews structural break test**

```@example tests_overview
# Detect a structural break in a linear regression
T = 200
X = hcat(ones(T), randn(T))
y = X * [1.0, 2.0] + randn(T) * 0.5
y[101:end] .+= X[101:end, 2] .* 3.0   # break at midpoint

result = andrews_test(y, X; test=:supwald, trimming=0.15)
report(result)
```

**Recipe 3: PANIC panel unit root test**

```@example tests_overview
# Panel data: 100 time periods, 20 cross-section units
X_panel = cumsum(randn(100, 20), dims=1)   # I(1) panel with common factors
result = panic_test(X_panel; r=:auto, method=:pooled)
report(result)
```

**Recipe 4: Normality test suite on VAR residuals**

```@example tests_overview
# Estimate a VAR and test residual normality
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
m = estimate_var(Y, 2)

suite = normality_test_suite(m)
report(suite)
```

---

## Test Taxonomy

The package exports over 30 test functions organized into ten categories. Each function returns a typed result struct with a `show` method for publication-quality display and full StatsAPI compatibility.

### Unit Root Tests

Tests for univariate stationarity and integration order. See [Unit Root & Cointegration](tests_unitroot.md) for full documentation.

1. `adf_test` -- Augmented Dickey-Fuller test (Dickey & Fuller 1979). Null: unit root
2. `kpss_test` -- KPSS stationarity test (Kwiatkowski et al. 1992). Null: stationarity
3. `pp_test` -- Phillips-Perron non-parametric unit root test (Phillips & Perron 1988)
4. `za_test` -- Zivot-Andrews unit root test with endogenous structural break (Zivot & Andrews 1992)
5. `ngperron_test` -- Ng-Perron GLS-detrended tests with improved size properties (Ng & Perron 2001)
6. `fourier_adf_test` -- Fourier ADF unit root test with flexible Fourier form (Enders & Lee 2012). Null: unit root
7. `fourier_kpss_test` -- Fourier KPSS stationarity test with Fourier terms (Becker, Enders & Lee 2006). Null: stationarity
8. `dfgls_test` -- DF-GLS / ERS point-optimal unit root test with GLS detrending (Elliott, Rothenberg & Stock 1996). Null: unit root
9. `lm_unitroot_test` -- LM unit root test with 0, 1, or 2 structural breaks (Schmidt-Phillips 1992; Lee-Strazicich 2003, 2013). Null: unit root with breaks
10. `adf_2break_test` -- Two-break ADF unit root test (Narayan & Popp 2010). Null: unit root with two breaks

### Cointegration

Tests for multivariate long-run equilibrium relationships. See [Unit Root & Cointegration](tests_unitroot.md).

11. `johansen_test` -- Johansen trace and maximum eigenvalue tests (Johansen 1991)
12. `gregory_hansen_test` -- Gregory-Hansen cointegration test with structural break (Gregory & Hansen 1996). Null: no cointegration

### Structural Breaks

Tests for parameter instability at unknown break dates. See [Structural Breaks](tests_breaks.md).

13. `andrews_test` -- Andrews (1993) sup-Wald / Andrews-Ploberger (1994) exp- and mean-Wald tests for a single structural break
14. `bai_perron_test` -- Bai-Perron (1998, 2003) multiple structural break test with dynamic programming and BIC/LWZ selection
15. `factor_break_test` -- Structural break tests for factor models: Breitung-Eickmeier (2011) loading stability, Chen-Dolado-Gonzalo (2014) factor number change, Han-Inoue (2015) sup-Wald

### Panel Unit Root

Tests for panel stationarity under cross-sectional dependence. See [Panel Tests](tests_panel.md).

16. `panic_test` -- Bai-Ng (2004, 2010) PANIC decomposition into common factors and idiosyncratic components
17. `pesaran_cips_test` -- Pesaran (2007) cross-sectionally augmented IPS test
18. `moon_perron_test` -- Moon-Perron (2004) factor-adjusted pooled t-statistics
19. `panel_unit_root_summary` -- Runs all three panel tests and prints a consolidated summary table

### VAR Diagnostics

Specification tests for Vector Autoregressive models. See [Model Diagnostics](tests_diagnostics.md).

20. `is_stationary` -- Companion matrix eigenvalue check for VAR stability
21. `granger_test` -- Pairwise or block Granger causality Wald test (Granger 1969)
22. `granger_test_all` -- All-pairs Granger causality matrix

### Panel VAR

GMM instrument validation and lag selection for Panel VARs. See [Model Diagnostics](tests_diagnostics.md).

23. `pvar_hansen_j` -- Hansen (1982) J-test for overidentifying restrictions
24. `pvar_mmsc` -- Andrews-Lu (2001) model and moment selection criteria (BIC/AIC/HQIC)
25. `pvar_lag_selection` -- Optimal lag order via MMSC comparison across specifications

### Normality

Multivariate normality tests for model residuals. See [Model Diagnostics](tests_diagnostics.md).

26. `jarque_bera_test` -- Multivariate and component-wise Jarque-Bera (Jarque & Bera 1980)
27. `mardia_test` -- Mardia skewness, kurtosis, and combined tests (Mardia 1970)
28. `doornik_hansen_test` -- Doornik-Hansen omnibus test (Doornik & Hansen 2008)
29. `henze_zirkler_test` -- Henze-Zirkler invariant test (Henze & Zirkler 1990)
30. `normality_test_suite` -- Runs all seven normality tests and returns a consolidated `NormalityTestSuite`

### ARCH Diagnostics

Tests for conditional heteroskedasticity in residuals. See [Model Diagnostics](tests_diagnostics.md).

31. `arch_lm_test` -- Engle (1982) ARCH-LM test (``T \cdot R^2`` from squared residual regression)
32. `ljung_box_squared` -- Ljung-Box test on squared residuals for remaining ARCH effects

### Model Comparison

Generic nested model comparison tests. See [Model Diagnostics](tests_diagnostics.md).

33. `lr_test` -- Likelihood ratio test for any pair of nested models with `loglikelihood` (Wilks 1938)
34. `lm_test` -- Lagrange multiplier (score) test for ARIMA, VAR, ARCH, and GARCH families (Rao 1948)

### Convenience

Batch testing utilities for multi-variable workflows.

35. `unit_root_summary` -- Runs ADF, KPSS, PP, and optionally Fourier ADF, DF-GLS on a single series and reports a combined verdict
36. `test_all_variables` -- Applies a unit root test (including Fourier ADF, DF-GLS, LM unit root) to every column of a data matrix

---

## Decision Flowchart

### Pre-Estimation: Integration Order

Determining the integration order of each variable is the first step before specifying a VAR, VECM, or Local Projection.

| Situation | Recommended test | Rationale |
|-----------|-----------------|-----------|
| Standard unit root test | `adf_test` | Baseline; automatic lag selection via AIC |
| Confirm ADF finding | `kpss_test` | Opposite null hypothesis eliminates ambiguity |
| Autocorrelation concerns | `pp_test` | Non-parametric; no lag specification needed |
| Suspected structural break | `za_test` | ADF has low power against break-stationary alternatives |
| Small sample (``T < 100``) | `ngperron_test` | GLS detrending improves size and power |
| Multiple variables, I(1) | `johansen_test` | Tests for cointegrating rank before VECM estimation |
| Smooth structural breaks | `fourier_adf_test` | Fourier terms capture gradual shifts without specifying break dates |
| Best power against near-unit-root | `dfgls_test` | GLS detrending; reports ERS Pt and MGLS statistics |
| Unit root with breaks under H₀ | `lm_unitroot_test` | Breaks included under the null; 0/1/2 breaks |
| Two known-type structural breaks | `adf_2break_test` | Grid search over two break dates |
| Quick multi-test diagnosis | `unit_root_summary` | Combines ADF + KPSS + PP with automatic verdict |

### Pre-Estimation: Panel Data

Panel unit root tests account for cross-sectional dependence that invalidates standard IPS-type tests.

| Situation | Recommended test | Rationale |
|-----------|-----------------|-----------|
| Factor-driven dependence | `panic_test` | Separates common and idiosyncratic components |
| General cross-section dependence | `pesaran_cips_test` | Augments ADF with cross-section averages |
| Large N, moderate T | `moon_perron_test` | Factor-adjusted pooled statistics |
| Comprehensive panel check | `panel_unit_root_summary` | All three tests with consolidated output |

### Structural Stability

Structural break tests detect changes in regression parameters at unknown dates.

| Situation | Recommended test | Rationale |
|-----------|-----------------|-----------|
| Single unknown break | `andrews_test` | sup-Wald for point detection; exp/mean for average evidence |
| Multiple unknown breaks | `bai_perron_test` | Dynamic programming with BIC/LWZ break selection |
| Factor loading instability | `factor_break_test` | Loading CUSUM, factor number change, or sup-Wald |
| Cointegration with regime shift | `gregory_hansen_test` | Three models (C/CT/CS), three statistics (ADF*/Zt*/Za*) |

### Post-Estimation Diagnostics

After estimating a model, specification tests validate the assumptions underlying inference.

| Situation | Recommended test | Rationale |
|-----------|-----------------|-----------|
| VAR stability | `is_stationary` | Eigenvalue check ensures non-explosive dynamics |
| Predictive causality | `granger_test` / `granger_test_all` | Pairwise and block Wald tests |
| Residual normality | `normality_test_suite` | Seven tests covering skewness, kurtosis, and omnibus |
| Conditional heteroskedasticity | `arch_lm_test` / `ljung_box_squared` | Detects remaining ARCH effects |
| Nested model comparison | `lr_test` / `lm_test` | Likelihood ratio and score-based nesting tests |
| PVAR instrument validity | `pvar_hansen_j` | Overidentifying restrictions |
| PVAR model selection | `pvar_mmsc` / `pvar_lag_selection` | Andrews-Lu MMSC criteria |

---

## StatsAPI Interface

All test result types implement the StatsAPI.jl interface. This provides a uniform way to extract test statistics, p-values, and degrees of freedom regardless of the specific test.

```@example tests_overview
fred = load_example(:fred_md)
cpi = filter(isfinite, to_vector(fred[:, "CPIAUCSL"]))
result = adf_test(cpi; lags=:aic, regression=:constant)

# Uniform interface across all test types
nobs(result)      # number of observations
dof(result)       # degrees of freedom
pvalue(result)    # p-value
```

### Type Hierarchy

```
StatsAPI.HypothesisTest
  AbstractUnitRootTest
    ADFResult{T}
    KPSSResult{T}
    PPResult{T}
    ZAResult{T}
    NgPerronResult{T}
    FourierADFResult{T}
    FourierKPSSResult{T}
    DFGLSResult{T}
    LMUnitRootResult{T}
    ADF2BreakResult{T}
    GregoryHansenResult{T}
    JohansenResult{T}
    AndrewsResult{T}
    BaiPerronResult{T}
    PANICResult{T}
    PesaranCIPSResult{T}
    MoonPerronResult{T}
    FactorBreakResult{T}
  AbstractNormalityTest
    NormalityTestResult{T}
  GrangerCausalityResult{T}
  PVARTestResult{T}
  LRTestResult{T}
  LMTestResult{T}
  VARStationarityResult{T}
```

Every result type supports `nobs()`, `dof()`, and `pvalue()`. For tests with multiple statistics (Ng-Perron returns four, Moon-Perron returns two), `pvalue()` returns the primary statistic's p-value (MZt for Ng-Perron, ``t_a^*`` for Moon-Perron). Access individual statistics through the result fields documented on each sub-page.

!!! note "Technical Note"
    The `NormalityTestSuite` returned by `normality_test_suite` is not itself a `HypothesisTest` subtype --- it is a container holding multiple `NormalityTestResult` objects. Iterate over `suite.results` to access individual test p-values, or use `report(suite)` for a consolidated display.

---

## Common Pitfalls

1. **Conflating ADF and KPSS null hypotheses.** The ADF null is "unit root" while the KPSS null is "stationarity." A non-significant ADF result does not confirm a unit root --- it means insufficient evidence against one. Always run both tests and consult the decision matrix in the [Unit Root & Cointegration](tests_unitroot.md) page.

2. **Ignoring structural breaks in unit root tests.** Standard ADF and PP tests have low power against trend-stationary series with a structural break. If economic events suggest a regime change (e.g., the Great Moderation, the Global Financial Crisis), use `za_test` or pre-test with `andrews_test` before concluding non-stationarity.

3. **Applying time-series unit root tests to panels.** Standard ADF, KPSS, and PP tests assume independent observations and produce invalid inference under cross-sectional dependence. For panel data, use `panic_test`, `pesaran_cips_test`, or `moon_perron_test`, which explicitly account for common factor structures.

4. **Running Granger causality on non-stationary data.** The asymptotic ``\chi^2`` distribution for the Granger Wald test requires the VAR to be stationary. Apply unit root tests first, difference or cointegration-adjust the data, and verify `is_stationary(model)` before interpreting Granger causality results.

5. **Over-relying on the Hansen J-test in Panel VARs.** The J-test has low power when the instrument count is large relative to the cross-section dimension ``N``. A non-rejection does not validate instruments --- it may reflect low power. Combine J-test results with Andrews-Lu MMSC criteria and economic reasoning about instrument relevance.

---

## References

- Enders, Walter. 2014. *Applied Econometric Time Series*. 4th ed.
  Hoboken, NJ: Wiley. ISBN 978-1-118-80856-6.

- Hamilton, James D. 1994. *Time Series Analysis*.
  Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.

- Lutkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
