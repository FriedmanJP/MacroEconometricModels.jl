# Examples

This chapter provides comprehensive worked examples demonstrating the main functionality of **MacroEconometricModels.jl**. Each example includes complete code, economic interpretation, and best practices. The examples follow the natural empirical workflow: load and prepare data, test properties, estimate univariate and multivariate models, and report results.

### Quick Reference

| # | Example | Key Functions | Description |
|---|---------|---------------|-------------|
| 1 | FRED-MD Data Pipeline | `load_example`, `apply_tcode`, `diagnose`, `estimate_var` | Real data workflow from FRED-MD to VAR estimation |
| 2 | Hypothesis Tests | `adf_test`, `kpss_test`, `johansen_test`, `granger_test` | ADF, KPSS, Zivot-Andrews, Ng-Perron, Johansen, Granger causality |
| 3 | Time Series Filters | `hp_filter`, `hamilton_filter`, `beveridge_nelson`, `baxter_king`, `boosted_hp` | All 5 filters compared on simulated GDP |
| 4 | ARIMA Models | `estimate_ar`, `estimate_arma`, `auto_arima`, `forecast` | AR, MA, ARMA estimation, order selection, forecasting |
| 5 | Volatility Models | `estimate_garch`, `estimate_egarch`, `estimate_sv`, `news_impact_curve` | ARCH/GARCH/SV estimation, diagnostics, forecasting |
| 6 | Three-Variable VAR | `estimate_var`, `irf`, `fevd`, `identify_arias` | Frequentist VAR with Cholesky, sign, long-run, and Arias (2018) identification |
| 7 | Bayesian VAR with Minnesota Prior | `estimate_bvar`, `optimize_hyperparameters` | Minnesota prior, conjugate posterior estimation, credible intervals |
| 8 | VECM Analysis | `estimate_vecm`, `johansen_test`, `to_var`, `forecast` | Cointegration, VECM estimation, IRF, forecast |
| 9 | Local Projections | `estimate_lp`, `estimate_lp_iv`, `structural_lp`, `lp_fevd` | Standard, IV, smooth, state-dependent, structural LP, and LP-FEVD |
| 10 | Factor Model for Large Panels | `estimate_factors`, `ic_criteria`, `forecast` | Large panel factor extraction, Bai-Ng criteria, forecasting with CIs |
| 11 | Panel VAR Analysis | `estimate_pvar`, `pvar_oirf`, `pvar_fevd`, `pvar_bootstrap_irf` | Full PVAR workflow: GMM, specification tests, structural analysis |
| 12 | GMM Estimation | `estimate_gmm`, `j_test` | IV regression via GMM, overidentification test |
| 13 | Non-Gaussian Identification | `identify_fastica`, `normality_test_suite`, `test_shock_gaussianity` | ICA, ML, heteroskedastic identification |
| 14 | Complete Workflow | Multiple | Unit roots → lag selection → VAR → BVAR → LP comparison |
| 15 | Table Output (LaTeX & HTML) | `set_display_backend`, `print_table`, `table` | Export tables for papers, slides, and web |
| 16 | Bibliographic References | `refs` | Multi-format references for models and methods |

---

## Example 1: FRED-MD Data Pipeline

This example demonstrates a complete empirical workflow using the built-in FRED-MD dataset: loading data, applying transformations, data cleaning, and VAR estimation. See [Data Management](data.md) for data container details and [Examples](examples.md) for additional workflows.

```julia
using MacroEconometricModels, Random

Random.seed!(42)

# --- Step 1: Load the FRED-MD dataset ---
md = load_example(:fred_md)
desc(md)                                # Dataset description and vintage info
```

### Explore the Dataset

```julia
# Variable descriptions
vardesc(md, "INDPRO")                   # "IP Index"
vardesc(md, "UNRATE")                   # "Civilian Unemployment Rate"
vardesc(md, "CPIAUCSL")                 # "CPI: All Items"

# Bibliographic reference
refs(md)                                # McCracken & Ng (2016)
```

### Transform and Clean

```julia
# Apply recommended FRED transformation codes (log-diff, diff, etc.)
md_transformed = apply_tcode(md, md.tcode)

# Diagnose data issues (NaN from differencing, constant columns, etc.)
diag = diagnose(md_transformed)

# Clean: remove rows with missing values
clean = fix(md_transformed; method=:listwise)
```

**Interpretation.** The FRED transformation codes ensure stationarity: code 1 = no transform, 2 = first difference, 4 = log, 5 = log first difference, etc. Differencing introduces NaN in the first row(s), which `fix` removes. Always diagnose before estimation to catch remaining data issues.

### Subset and Estimate

```julia
# Select 4 key macroeconomic variables
subset = clean[:, ["INDPRO", "UNRATE", "CPIAUCSL", "FEDFUNDS"]]

# Summary statistics
describe_data(subset)

# Estimate VAR(4) — quarterly lag structure for monthly data
model = estimate_var(subset, 4)

# Structural analysis
irfs = irf(model, 24; method=:cholesky)     # 24-month horizon
fvd = fevd(model, 24)
```

**Interpretation.** The four-variable system captures industrial production, unemployment, inflation, and monetary policy --- the core variables for monetary VAR analysis. With monthly data, 4 lags covers one quarter of dynamics. The Cholesky ordering places slow-moving real variables first and the policy rate last, consistent with the standard recursive identification in monetary economics. The impulse responses trace the transmission of a monetary policy shock through output, unemployment, and prices.

## Example 2: Hypothesis Tests

This example demonstrates comprehensive unit root testing before fitting VAR models. Pre-estimation analysis is the first step in any empirical macro workflow. See [Hypothesis Tests](hypothesis_tests.md) for theoretical background.

### Individual Unit Root Tests

```julia
using MacroEconometricModels
using Random
using Statistics

Random.seed!(42)

# Generate data: mix of I(0) and I(1) series
T = 200
y_stationary = randn(T)                      # I(0): stationary
y_random_walk = cumsum(randn(T))             # I(1): unit root
y_trend_stat = 0.1 .* (1:T) .+ randn(T)      # Trend stationary
y_with_break = vcat(randn(100), randn(100) .+ 2)  # Structural break

# === ADF Test ===
println("="^60)
println("ADF Test (H₀: unit root)")
println("="^60)

adf_stat = adf_test(y_stationary; lags=:aic, regression=:constant)
println("\nStationary series:")
println("  Statistic: ", round(adf_stat.statistic, digits=3))
println("  P-value: ", round(adf_stat.pvalue, digits=4))
println("  Lags: ", adf_stat.lags)

adf_rw = adf_test(y_random_walk; lags=:aic, regression=:constant)
println("\nRandom walk:")
println("  Statistic: ", round(adf_rw.statistic, digits=3))
println("  P-value: ", round(adf_rw.pvalue, digits=4))
```

The ADF test statistic is compared to non-standard critical values (Dickey-Fuller distribution, not Student-t). For the stationary series, the large negative test statistic yields a small p-value, rejecting the unit root null. For the random walk, the test statistic is close to zero, failing to reject. The number of augmenting lags selected by AIC controls for residual serial correlation.

### KPSS Complementary Test

```julia
# === KPSS Test ===
println("\n" * "="^60)
println("KPSS Test (H₀: stationarity)")
println("="^60)

kpss_stat = kpss_test(y_stationary; regression=:constant)
println("\nStationary series:")
println("  Statistic: ", round(kpss_stat.statistic, digits=4))
println("  P-value: ", kpss_stat.pvalue > 0.10 ? ">0.10" : round(kpss_stat.pvalue, digits=4))
println("  Bandwidth: ", kpss_stat.bandwidth)

kpss_rw = kpss_test(y_random_walk; regression=:constant)
println("\nRandom walk:")
println("  Statistic: ", round(kpss_rw.statistic, digits=4))
println("  P-value: ", kpss_rw.pvalue < 0.01 ? "<0.01" : round(kpss_rw.pvalue, digits=4))
```

### Combining ADF and KPSS for Robust Inference

```julia
# === Combined Analysis ===
println("\n" * "="^60)
println("Combined ADF + KPSS Analysis")
println("="^60)

function unit_root_decision(y; name="Series")
    adf = adf_test(y; lags=:aic)
    kpss = kpss_test(y)

    adf_reject = adf.pvalue < 0.05  # Reject unit root
    kpss_reject = kpss.pvalue < 0.05  # Reject stationarity

    decision = if adf_reject && !kpss_reject
        "I(0) - Stationary"
    elseif !adf_reject && kpss_reject
        "I(1) - Unit root"
    elseif adf_reject && kpss_reject
        "Conflicting (possible structural break)"
    else
        "Inconclusive"
    end

    println("\n$name:")
    println("  ADF p-value: ", round(adf.pvalue, digits=4))
    println("  KPSS p-value: ", round(kpss.pvalue, digits=4))
    println("  Decision: $decision")

    return decision
end

unit_root_decision(y_stationary; name="Stationary series")
unit_root_decision(y_random_walk; name="Random walk")
unit_root_decision(y_trend_stat; name="Trend stationary")
```

### Testing for Structural Breaks

```julia
# === Zivot-Andrews Test ===
println("\n" * "="^60)
println("Zivot-Andrews Test (H₀: unit root without break)")
println("="^60)

za_result = za_test(y_with_break; regression=:constant, trim=0.15)
println("\nSeries with structural break:")
println("  Minimum t-stat: ", round(za_result.statistic, digits=3))
println("  P-value: ", round(za_result.pvalue, digits=4))
println("  Break index: ", za_result.break_index)
println("  Break at: ", round(za_result.break_fraction * 100, digits=1), "% of sample")

# Compare with standard ADF
adf_break = adf_test(y_with_break)
println("\n  ADF (ignoring break): p=", round(adf_break.pvalue, digits=4))
println("  ZA (allowing break): p=", round(za_result.pvalue, digits=4))
```

### Ng-Perron Tests for Small Samples

```julia
# === Ng-Perron Tests ===
println("\n" * "="^60)
println("Ng-Perron Tests (improved size properties)")
println("="^60)

# Generate smaller sample
y_small = cumsum(randn(80))
np_result = ngperron_test(y_small; regression=:constant)

println("\nSmall sample (n=80):")
println("  MZα: ", round(np_result.MZa, digits=3),
        " (5% CV: ", np_result.critical_values[:MZa][5], ")")
println("  MZt: ", round(np_result.MZt, digits=3),
        " (5% CV: ", np_result.critical_values[:MZt][5], ")")
println("  MSB: ", round(np_result.MSB, digits=4),
        " (5% CV: ", np_result.critical_values[:MSB][5], ")")
println("  MPT: ", round(np_result.MPT, digits=3),
        " (5% CV: ", np_result.critical_values[:MPT][5], ")")
```

### Johansen Cointegration Test

```julia
# === Johansen Cointegration Test ===
println("\n" * "="^60)
println("Johansen Cointegration Test")
println("="^60)

# Generate cointegrated system
T_coint = 200
u1, u2, u3 = cumsum(randn(T_coint)), cumsum(randn(T_coint)), randn(T_coint)
Y_coint = hcat(
    u1 + 0.1*randn(T_coint),           # I(1)
    u1 + 0.5*u2 + 0.1*randn(T_coint),  # Cointegrated with first
    u2 + 0.1*randn(T_coint)            # I(1)
)

johansen = johansen_test(Y_coint, 2; deterministic=:constant)

println("\nCointegrated system (3 variables):")
println("  Estimated rank: ", johansen.rank)
println("\n  Trace test:")
for r in 0:2
    stat = round(johansen.trace_stats[r+1], digits=2)
    cv = round(johansen.critical_values_trace[r+1, 2], digits=2)
    reject = stat > cv ? "Reject" : "Fail to reject"
    println("    H₀: r ≤ $r: stat=$stat, 5% CV=$cv → $reject")
end

println("\n  Eigenvalues: ", round.(johansen.eigenvalues, digits=4))

if johansen.rank > 0
    println("\n  Cointegrating vector(s):")
    for i in 1:johansen.rank
        println("    β$i: ", round.(johansen.eigenvectors[:, i], digits=3))
    end
end
```

The Johansen trace test sequentially tests hypotheses about the cointegration rank. When the trace statistic exceeds the critical value, we reject the null and move to the next rank. The estimated cointegrating vectors ``\beta`` represent long-run equilibrium relationships: deviations from ``\beta' y_t`` are stationary even though the individual series are I(1). The adjustment coefficients ``\alpha`` govern how quickly variables correct back toward equilibrium.

### Granger Causality

```julia
# === Granger Causality Tests ===
println("\n" * "="^60)
println("Granger Causality Tests")
println("="^60)

# Generate data with known causal structure:
# Variable 2 depends on lagged Variable 1
T_obs = 300
Y_gc = zeros(T_obs, 3)
Y_gc[1, :] = randn(3)
for t in 2:T_obs
    Y_gc[t, 1] = 0.5 * Y_gc[t-1, 1] + randn()
    Y_gc[t, 2] = 0.3 * Y_gc[t-1, 1] + 0.2 * Y_gc[t-1, 2] + randn()  # 1 causes 2
    Y_gc[t, 3] = 0.4 * Y_gc[t-1, 3] + randn()                         # independent
end

m = estimate_var(Y_gc, 2)

# Pairwise test: does variable 1 Granger-cause variable 2?
g = granger_test(m, 1, 2)
println("\n1 → 2: Wald = ", round(g.statistic, digits=2),
        ", p = ", round(g.pvalue, digits=4))

# Variable 3 should be independent of others
println("3 → 1: p = ", round(granger_test(m, 3, 1).pvalue, digits=4))
println("3 → 2: p = ", round(granger_test(m, 3, 2).pvalue, digits=4))

# Block test: do variables 1 and 3 jointly Granger-cause variable 2?
g_block = granger_test(m, [1, 3], 2)
println("\nBlock [1,3] → 2: Wald = ", round(g_block.statistic, digits=2),
        ", p = ", round(g_block.pvalue, digits=4))

# Full causality table (n × n matrix, nothing on diagonal)
results = granger_test_all(m)
```

The Granger causality test examines whether lagged values of one variable help predict another. A significant result (low p-value) indicates that past values of the cause variable contain predictive information for the effect variable beyond what is captured by the effect variable's own lags and those of other variables. The block test extends this to groups of variables, testing their joint predictive power.

### Testing All Variables Before VAR

```julia
# === Multi-Variable Pre-VAR Analysis ===
println("\n" * "="^60)
println("Pre-VAR Unit Root Analysis")
println("="^60)

# Typical macro dataset
Y_macro = hcat(
    cumsum(randn(T)),           # GDP (I(1))
    0.8*cumsum(randn(T)[1:T]),  # Inflation (I(1))
    cumsum(randn(T)),           # Interest rate (I(1))
    randn(T)                    # Output gap (I(0))
)
var_names = ["GDP", "Inflation", "Rate", "Output Gap"]

# Test all variables
results = test_all_variables(Y_macro; test=:adf)

println("\nUnit root test results:")
println("-"^50)
n_i1 = 0
for (i, r) in enumerate(results)
    status = r.pvalue > 0.05 ? "I(1)" : "I(0)"
    n_i1 += r.pvalue > 0.05
    println("  $(var_names[i]): p=$(round(r.pvalue, digits=3)) → $status")
end

println("\nSummary: $n_i1 of $(size(Y_macro, 2)) variables appear I(1)")

# Recommendation
if n_i1 == size(Y_macro, 2)
    println("\nRecommendation: All variables I(1)")
    println("  → Test for cointegration")
    println("  → If cointegrated: use VECM")
    println("  → If not: use VAR in first differences")
elseif n_i1 == 0
    println("\nRecommendation: All variables I(0)")
    println("  → Use VAR in levels")
else
    println("\nRecommendation: Mixed I(0)/I(1)")
    println("  → Consider ARDL bounds test")
    println("  → Or difference I(1) variables")
end
```

### Complete Pre-Estimation Workflow

```julia
# === Complete Workflow ===
println("\n" * "="^60)
println("Complete Pre-Estimation Workflow")
println("="^60)

function pre_estimation_analysis(Y; var_names=nothing, α=0.05)
    T, n = size(Y)
    var_names = isnothing(var_names) ? ["Var$i" for i in 1:n] : var_names

    println("\n1. Individual Unit Root Tests")
    println("-"^40)

    integration_orders = zeros(Int, n)
    for i in 1:n
        adf = adf_test(Y[:, i]; lags=:aic)
        kpss = kpss_test(Y[:, i])

        if adf.pvalue < α && kpss.pvalue > α
            integration_orders[i] = 0
            status = "I(0)"
        elseif adf.pvalue > α && kpss.pvalue < α
            integration_orders[i] = 1
            status = "I(1)"
        else
            integration_orders[i] = -1  # Inconclusive
            status = "Inconclusive"
        end
        println("  $(var_names[i]): $status (ADF p=$(round(adf.pvalue, digits=3)), KPSS p=$(round(kpss.pvalue, digits=3)))")
    end

    n_i1 = sum(integration_orders .== 1)
    n_i0 = sum(integration_orders .== 0)

    println("\n2. Summary")
    println("-"^40)
    println("  I(0) variables: $n_i0")
    println("  I(1) variables: $n_i1")
    println("  Inconclusive: $(n - n_i0 - n_i1)")

    # Cointegration test if all I(1)
    if n_i1 >= 2
        println("\n3. Cointegration Test")
        println("-"^40)
        joh = johansen_test(Y, 2)
        println("  Estimated cointegration rank: ", joh.rank)

        if joh.rank > 0
            println("  → Cointegration detected")
            println("  → Recommendation: VECM with rank=$(joh.rank)")
        else
            println("  → No cointegration")
            println("  → Recommendation: VAR in first differences")
        end
    elseif n_i0 == n
        println("\n3. Recommendation")
        println("-"^40)
        println("  All series stationary → VAR in levels")
    end

    return (integration_orders=integration_orders, n_i0=n_i0, n_i1=n_i1)
end

# Run complete analysis
result = pre_estimation_analysis(Y_macro; var_names=var_names)
```

---

## Example 3: Time Series Filters

This example compares all five trend-cycle decomposition filters on a simulated quarterly GDP-like series. See [Time Series Filters](filters.md) for theory, return values, and individual filter options.

```julia
using MacroEconometricModels, Random, Statistics

Random.seed!(42)

# Simulated quarterly GDP-like series (200 quarters)
# cumsum with drift produces an I(1) series with a stochastic trend
y = cumsum(0.5 .+ randn(200))

println("Simulated GDP series: T=$(length(y)) quarters")
println("  Mean growth: ", round(mean(diff(y)), digits=3))

# === Apply all five filters ===

# Hodrick-Prescott (lambda=1600 for quarterly data)
hp  = hp_filter(y; lambda=1600.0)

# Hamilton regression filter (h=8, p=4)
ham = hamilton_filter(y; h=8, p=4)

# Beveridge-Nelson decomposition via ARIMA
bn  = beveridge_nelson(y; p=2, q=0)

# Baxter-King symmetric band-pass (6-32 quarters, K=12 lead/lag)
bk  = baxter_king(y; pl=6, pu=32, K=12)

# Boosted HP (iterated HP with BIC stopping)
bhp = boosted_hp(y; stopping=:BIC)

# === Compare cycle standard deviations ===
println("\nCycle standard deviations:")
println("  HP:       ", round(std(cycle(hp)), digits=4))
println("  Hamilton: ", round(std(cycle(ham)), digits=4))
println("  BN:       ", round(std(cycle(bn)), digits=4))
println("  BK:       ", round(std(cycle(bk)), digits=4))
println("  bHP:      ", round(std(cycle(bhp)), digits=4))

# === Trend comparison at selected dates ===
println("\nTrend values at t=100:")
println("  HP:       ", round(trend(hp)[100], digits=2))
println("  Hamilton: ", round(trend(ham)[100], digits=2))
println("  BN:       ", round(trend(bn)[100], digits=2))
println("  bHP:      ", round(trend(bhp)[100], digits=2))
```

The HP filter is the most common choice in macroeconomics, but Hamilton (2018) argues it can induce spurious dynamics. The Hamilton filter uses a regression-based approach that avoids end-of-sample bias. Beveridge-Nelson decomposes the series using its ARIMA representation, while Baxter-King isolates business-cycle frequencies (6--32 quarters) via a symmetric moving average. The boosted HP iteratively re-applies the HP filter, improving trend estimation for series with structural changes. Comparing cycle standard deviations reveals how aggressively each filter extracts the cyclical component.

---

## Example 4: ARIMA Models

This example demonstrates univariate time series modeling with ARIMA models: estimation, order selection, diagnostics, and forecasting.

```julia
using MacroEconometricModels
using Random
using Statistics

Random.seed!(42)

# Generate ARMA(1,1) data
T = 300
y = zeros(T)
e = randn(T)
for t in 2:T
    y[t] = 0.7 * y[t-1] + e[t] + 0.3 * e[t-1]
end

# === AR(2) via OLS ===
ar = estimate_ar(y, 2)
println("AR(2) Estimation")
println("  Coefficients: ", round.(coef(ar), digits=4))
println("  AIC: ", round(aic(ar), digits=2))
println("  BIC: ", round(bic(ar), digits=2))

# === ARMA(1,1) via CSS-MLE ===
arma = estimate_arma(y, 1, 1)
println("\nARMA(1,1) Estimation")
println("  AR coef: ", round(arma.ar_coefs[1], digits=4))
println("  MA coef: ", round(arma.ma_coefs[1], digits=4))
println("  AIC: ", round(aic(arma), digits=2))

# === Automatic order selection ===
best = auto_arima(y)
println("\nauto_arima selection:")
println("  Best model: ARIMA($(best.p),$(best.d),$(best.q))")
println("  AIC: ", round(aic(best), digits=2))

# === Information criteria table ===
ict = ic_table(y, 4, 4)
println("\nIC table (top 5 by AIC):")
for i in 1:min(5, size(ict, 1))
    println("  p=$(Int(ict[i,1])), q=$(Int(ict[i,2])): AIC=$(round(ict[i,3], digits=1)), BIC=$(round(ict[i,4], digits=1))")
end

# === Forecast ===
fc = forecast(arma, 12; conf_level=0.95)
println("\nARMA(1,1) Forecasts:")
for h in [1, 4, 8, 12]
    println("  h=$h: $(round(fc.forecast[h], digits=3)) [$(round(fc.ci_lower[h], digits=3)), $(round(fc.ci_upper[h], digits=3))]")
end
```

The `auto_arima` function performs a grid search over (p,d,q) combinations, selecting the model that minimizes AIC. The CSS-MLE estimation method initializes parameters via conditional sum of squares (CSS), then refines via exact maximum likelihood using the Kalman filter. Forecast confidence intervals widen with the horizon, reflecting accumulating prediction uncertainty.

---

## Example 5: Volatility Models

This example estimates ARCH, GARCH, EGARCH, and GJR-GARCH models on the same data, compares their news impact curves, runs diagnostics, and forecasts volatility. See also [Volatility Models](volatility.md) for theory and return value tables.

```julia
using MacroEconometricModels
using Random
using Statistics

Random.seed!(42)

# === Generate GARCH(1,1) data with leverage effect ===
T = 1000
y = zeros(T)
h = zeros(T)
h[1] = 1.0

for t in 2:T
    z = randn()
    h[t] = 0.01 + 0.08 * y[t-1]^2 + 0.12 * (y[t-1] < 0 ? 1 : 0) * y[t-1]^2 + 0.85 * h[t-1]
    y[t] = sqrt(h[t]) * z
end

println("Simulated T=$T observations from GJR-GARCH(1,1)")
println("Sample kurtosis: ", round(kurtosis(y), digits=2))

# === Step 1: Test for ARCH effects ===
stat, pval, q = arch_lm_test(y, 5)
println("\nARCH-LM test (q=5): stat=$(round(stat, digits=2)), p=$(round(pval, digits=6))")

stat2, pval2, K = ljung_box_squared(y, 10)
println("Ljung-Box squared (K=10): stat=$(round(stat2, digits=2)), p=$(round(pval2, digits=6))")

# === Step 2: Estimate competing models ===
garch   = estimate_garch(y, 1, 1)
egarch  = estimate_egarch(y, 1, 1)
gjr     = estimate_gjr_garch(y, 1, 1)

println("\n" * "="^60)
println("Model Comparison")
println("="^60)
println("              AIC         BIC     Persistence")
println("  GARCH:   ", round(aic(garch), digits=1),
        "    ", round(bic(garch), digits=1),
        "    ", round(persistence(garch), digits=4))
println("  EGARCH:  ", round(aic(egarch), digits=1),
        "    ", round(bic(egarch), digits=1),
        "    ", round(persistence(egarch), digits=4))
println("  GJR:     ", round(aic(gjr), digits=1),
        "    ", round(bic(gjr), digits=1),
        "    ", round(persistence(gjr), digits=4))

# === Step 3: News impact curves ===
nic_g  = news_impact_curve(garch)
nic_e  = news_impact_curve(egarch)
nic_j  = news_impact_curve(gjr)

println("\nNews Impact at epsilon = -2 vs epsilon = +2:")
idx_neg = findfirst(x -> x >= -2.0, nic_g.shocks)
idx_pos = findfirst(x -> x >= 2.0, nic_g.shocks)

println("  GARCH:  var(-2) = ", round(nic_g.variance[idx_neg], digits=4),
        "   var(+2) = ", round(nic_g.variance[idx_pos], digits=4))
println("  EGARCH: var(-2) = ", round(nic_e.variance[idx_neg], digits=4),
        "   var(+2) = ", round(nic_e.variance[idx_pos], digits=4))
println("  GJR:    var(-2) = ", round(nic_j.variance[idx_neg], digits=4),
        "   var(+2) = ", round(nic_j.variance[idx_pos], digits=4))

# === Step 4: Residual diagnostics ===
println("\nResidual ARCH-LM test (q=5):")
for (name, m) in [("GARCH", garch), ("EGARCH", egarch), ("GJR", gjr)]
    _, p, _ = arch_lm_test(m, 5)
    status = p > 0.05 ? "Pass" : "FAIL"
    println("  $name: p=$(round(p, digits=4))  $status")
end

# === Step 5: Volatility forecasts ===
H = 20
fc_g = forecast(garch, H)
fc_e = forecast(egarch, H)
fc_j = forecast(gjr, H)

println("\nVolatility forecasts (conditional variance):")
println("  h    GARCH    EGARCH   GJR      Uncond")
for h_idx in [1, 5, 10, 20]
    println("  $h_idx    ",
            round(fc_g.forecast[h_idx], digits=4), "  ",
            round(fc_e.forecast[h_idx], digits=4), "  ",
            round(fc_j.forecast[h_idx], digits=4), "  ",
            round(unconditional_variance(garch), digits=4))
end

# === Step 6: Stochastic Volatility ===
println("\nEstimating SV model via KSC Gibbs sampler...")
sv = estimate_sv(y; n_samples=2000, burnin=1000)

println("SV posterior summary:")
println("  mu:      ", round(mean(sv.mu_post), digits=3))
println("  phi:     ", round(mean(sv.phi_post), digits=3))
println("  sigma_eta: ", round(mean(sv.sigma_eta_post), digits=3))

fc_sv = forecast(sv, H)
println("\nSV forecast at h=1:  ", round(fc_sv.forecast[1], digits=4))
println("SV forecast at h=20: ", round(fc_sv.forecast[end], digits=4))
```

The GJR-GARCH model should provide the best fit (lowest AIC/BIC) since the data was generated from a GJR-GARCH DGP with a leverage effect. The news impact curves reveal the asymmetry: for EGARCH and GJR-GARCH, the variance response to ``\varepsilon = -2`` exceeds that for ``\varepsilon = +2``; for symmetric GARCH, they are equal. All models' standardized residuals should pass the ARCH-LM test after fitting, confirming that the variance dynamics are adequately captured.

---

## Example 6: Three-Variable VAR Analysis

This example walks through a complete analysis of a macroeconomic VAR with GDP growth, inflation, and the federal funds rate.

### Setup and Data Generation

```julia
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

Random.seed!(42)

# Generate realistic macro data from a VAR(1) DGP
T = 200
n = 3
p = 2

# True VAR(1) coefficients (persistent, cross-correlated)
A_true = [0.85 0.10 -0.15;   # GDP responds to own lag, inflation, rate
          0.05 0.70  0.00;   # Inflation mainly AR
          0.10 0.20  0.80]   # Rate responds to GDP and inflation

# Shock covariance (correlated shocks)
Σ_true = [1.00 0.50 0.20;
          0.50 0.80 0.10;
          0.20 0.10 0.60]

# Generate data
Y = zeros(T, n)
Y[1, :] = randn(n)
chol_Σ = cholesky(Σ_true).L

for t in 2:T
    Y[t, :] = A_true * Y[t-1, :] + chol_Σ * randn(n)
end

var_names = ["GDP Growth", "Inflation", "Fed Funds Rate"]
println("Data: T=$T observations, n=$n variables")
```

### Frequentist VAR Estimation

```julia
# Estimate VAR(2) model via OLS
model = estimate_var(Y, p)

# Model diagnostics
println("Log-likelihood: ", loglikelihood(model))
println("AIC: ", aic(model))
println("BIC: ", bic(model))

# Check stability (eigenvalues inside unit circle)
F = companion_matrix(model.B, n, p)
eigenvalues = eigvals(F)
println("Max eigenvalue modulus: ", maximum(abs.(eigenvalues)))
println("Stable: ", maximum(abs.(eigenvalues)) < 1)
```

The AIC and BIC values measure the trade-off between fit and parsimony. Lower values indicate a better model. The maximum eigenvalue modulus should be strictly less than 1 for the VAR to be stationary; values close to 1 indicate high persistence, while values near 0 suggest rapid mean-reversion.

### Cholesky-Identified IRF

```julia
# Compute 20-period IRF with Cholesky identification
# Ordering: GDP → Inflation → Rate (contemporaneous causality)
H = 20
irfs = irf(model, H; method=:cholesky)

# Display impact responses (horizon 0)
println("\nImpact responses (B₀):")
println("  GDP shock → GDP: ", round(irfs.irf[1, 1, 1], digits=3))
println("  GDP shock → Inflation: ", round(irfs.irf[1, 2, 1], digits=3))
println("  GDP shock → Rate: ", round(irfs.irf[1, 3, 1], digits=3))

# Long-run responses (horizon H)
println("\nLong-run responses (h=$H):")
println("  GDP shock → GDP: ", round(irfs.irf[H+1, 1, 1], digits=3))
```

### Sign Restriction Identification

```julia
# Sign restrictions: Demand shock raises GDP and inflation on impact
function check_demand_shock(irf_array)
    # irf_array is (H+1) × n × n
    # Check: Shock 1 → Variable 1 (GDP) positive
    #        Shock 1 → Variable 2 (Inflation) positive
    return irf_array[1, 1, 1] > 0 && irf_array[1, 2, 1] > 0
end

# Estimate with sign restrictions
irfs_sign = irf(model, H; method=:sign, check_func=check_demand_shock, n_draws=1000)

println("\nSign-identified demand shock:")
println("  GDP response: ", round(irfs_sign.irf[1, 1, 1], digits=3))
println("  Inflation response: ", round(irfs_sign.irf[1, 2, 1], digits=3))
```

The Cholesky identification assumes a recursive causal ordering (GDP → Inflation → Rate), meaning GDP responds only to its own shocks contemporaneously. Sign restrictions provide a theory-based alternative: requiring both GDP and inflation to rise on impact identifies a "demand shock" without imposing a specific causal ordering. If sign restrictions accept many draws, the set-identified IRFs will show wider bands than point-identified Cholesky responses.

### Forecast Error Variance Decomposition

```julia
# Compute FEVD
fevd_result = fevd(model, H; method=:cholesky)

# Variance decomposition at horizon 1, 4, and 20
for h in [1, 4, 20]
    println("\nFEVD at horizon $h:")
    for i in 1:n
        println("  $(var_names[i]):")
        for j in 1:n
            pct = round(fevd_result.fevd[h, i, j] * 100, digits=1)
            println("    Shock $j: $pct%")
        end
    end
end
```

The FEVD shows the proportion of each variable's forecast error variance attributable to each structural shock. At short horizons, own shocks typically dominate. As the horizon increases, cross-variable transmission becomes more important, and the FEVD converges to the unconditional variance decomposition. If shock 1 explains a large share of GDP variance at long horizons, it is the primary driver of GDP fluctuations in the model.

### Long-Run (Blanchard-Quah) Identification

Long-run restrictions identify shocks by constraining their cumulative effects. The classic application distinguishes supply and demand shocks where demand shocks have no long-run effect on output.

```julia
using MacroEconometricModels, Random

Random.seed!(42)
Y = randn(200, 3)
model = estimate_var(Y, 2)

# Blanchard-Quah long-run identification
irfs_lr = irf(model, 20; method=:long_run)
```

**Interpretation.** The long-run restriction forces the cumulative IRF of shock 1 on variable 2 to zero at the infinite horizon. This is implemented via the Blanchard-Quah decomposition of the long-run multiplier matrix. The method is particularly useful for bivariate supply-demand identification.

### Arias et al. (2018) Zero and Sign Restrictions

The Arias et al. (2018) algorithm provides a unified framework for imposing both zero and sign restrictions on impulse responses. It draws orthogonal rotation matrices uniformly conditional on the restrictions.

```julia
using MacroEconometricModels, Random

Random.seed!(42)
Y = randn(200, 3)
model = estimate_var(Y, 2)

# Define restrictions: shock 1 has positive effect on var 1, zero on var 3
restrictions = SVARRestrictions(3)
add_sign_restriction!(restrictions, 1, 1, :positive, 0)   # shock 1 → var 1 positive at h=0
add_zero_restriction!(restrictions, 1, 3, 0)               # shock 1 → var 3 zero at h=0

# Arias identification (draws uniform rotations conditional on restrictions)
result = identify_arias(model, restrictions, 20; n_draws=500)
irfs_arias = irf(model, 20; method=:arias, restrictions=restrictions, n_draws=500)
```

**Interpretation.** The Arias algorithm guarantees draws from the correct posterior over the identified set, unlike accept-reject approaches. The zero restriction imposes exact equality while sign restrictions constrain the sign at specified horizons. Report the median and pointwise credible bands from the accepted draws.

---

## Example 7: Bayesian VAR with Minnesota Prior

This example demonstrates Bayesian estimation with automatic hyperparameter optimization.

### Hyperparameter Optimization

```julia
using MacroEconometricModels

# Find optimal shrinkage using marginal likelihood (Giannone et al. 2015)
println("Optimizing hyperparameters...")
best_hyper = optimize_hyperparameters(Y, p; grid_size=20)

println("Optimal hyperparameters:")
println("  τ (overall tightness): ", round(best_hyper.tau, digits=4))
println("  d (lag decay): ", best_hyper.d)
```

The optimal `tau` value reflects the degree of shrinkage that maximizes the marginal likelihood. A small `tau` (e.g., 0.05) means strong shrinkage toward the random walk prior, appropriate for large systems or short samples. A larger `tau` (e.g., 0.5-1.0) allows the data more influence, appropriate when the sample is informative relative to the model complexity.

### BVAR Estimation

```julia
# Estimate BVAR with optimized Minnesota prior
println("\nEstimating BVAR with conjugate NIW sampler...")
post = estimate_bvar(Y, p;
    n_draws = 1000,
    prior = :minnesota,
    hyper = best_hyper
)

# Posterior summary (coefficients from first equation)
println("\nPosterior summary for GDP equation:")
# Access posterior draws: post.B_draws, post.Sigma_draws
```

### Bayesian IRF with Credible Intervals

```julia
# Bayesian IRF with Cholesky identification
birf_chol = irf(post, H; method=:cholesky)

# Extract median and 68% credible intervals
# birf_chol.quantiles is (H+1) × n × n × 3 array
# [:, :, :, 1] = 16th percentile
# [:, :, :, 2] = median
# [:, :, :, 3] = 84th percentile

println("\nBayesian IRF of GDP to own shock:")
for h in [0, 4, 8, 12, 20]
    med = round(birf_chol.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf_chol.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf_chol.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

### Bayesian Sign Restrictions

```julia
# Bayesian IRF with sign restrictions
birf_sign = irf(post, H;
    method = :sign,
    check_func = check_demand_shock
)

println("\nBayesian sign-restricted demand shock → GDP:")
for h in [0, 4, 8, 12]
    med = round(birf_sign.quantiles[h+1, 1, 1, 2], digits=3)
    lo = round(birf_sign.quantiles[h+1, 1, 1, 1], digits=3)
    hi = round(birf_sign.quantiles[h+1, 1, 1, 3], digits=3)
    println("  h=$h: $med [$lo, $hi]")
end
```

---

## Example 8: VECM Analysis

This example demonstrates Vector Error Correction Model (VECM) estimation for cointegrated systems: testing for cointegration rank, estimating the VECM, computing impulse responses via VAR conversion, forecasting, and Granger causality decomposition. See [VECM](vecm.md) for theory and return value tables.

```julia
using MacroEconometricModels, Random

Random.seed!(42)

# === Generate cointegrated data ===
# Two linked variables (sharing a common stochastic trend) + one independent
T_obs = 200
Y = cumsum(randn(T_obs, 3), dims=1)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T_obs)  # Y2 cointegrated with Y1

println("Data: T=$T_obs, n=3 (Y1 and Y2 cointegrated, Y3 independent)")

# === Step 1: Test for cointegration ===
joh = johansen_test(Y, 2)
println("\nJohansen test:")
println("  Estimated cointegration rank: ", joh.rank)
for r in 0:2
    stat = round(joh.trace_stats[r+1], digits=2)
    cv = round(joh.critical_values_trace[r+1, 2], digits=2)
    reject = stat > cv ? "Reject" : "Fail to reject"
    println("  H₀: r ≤ $r: stat=$stat, 5% CV=$cv → $reject")
end

# === Step 2: Estimate VECM with automatic rank ===
vecm = estimate_vecm(Y, 2)
report(vecm)

# === Step 3: Examine cointegrating vectors and adjustment speeds ===
println("\nβ (cointegrating vectors):")
println(vecm.beta)
println("α (adjustment speeds):")
println(vecm.alpha)

# === Step 4: Impulse responses via VAR conversion ===
irfs = irf(vecm, 20; method=:cholesky)

println("\nIRF of shock 1 → variable 1:")
for h in [0, 4, 8, 12, 20]
    println("  h=$h: ", round(irfs.irf[h+1, 1, 1], digits=3))
end

# === Step 5: Forecast with bootstrap CIs ===
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=200)

println("\nVECM forecast (variable 1):")
for h in [1, 5, 10]
    println("  h=$h: ", round(fc.forecast[h, 1], digits=3))
end

# === Step 6: Granger causality (short-run, long-run, strong) ===
println("\nGranger causality (VECM decomposition):")
for i in 1:3, j in 1:3
    i == j && continue
    g = granger_causality_vecm(vecm, i, j)
    println("  Var $i → Var $j: p=$(round(g.strong_pvalue, digits=4))")
end

# === Step 7: Convert to VAR for FEVD ===
var_model = to_var(vecm)
decomp = fevd(var_model, 20)
```

**Interpretation.** The cointegrating vector ``\beta`` identifies the long-run equilibrium. If ``\beta \approx [1, -1, 0]'``, this implies ``y_{1,t} - y_{2,t}`` is stationary --- variables 1 and 2 share a common stochastic trend. The adjustment coefficients ``\alpha`` show how each variable responds when the system deviates from equilibrium. A significant ``\alpha_i`` indicates that variable ``i`` adjusts to restore the long-run relationship. The VECM-specific Granger causality decomposes predictive power into short-run (lagged differences) and long-run (error correction) channels.

---

## Example 9: Local Projections

This example demonstrates various LP methods for estimating impulse responses.

### Standard Local Projection

```julia
using MacroEconometricModels

# Estimate LP-IRF with Newey-West standard errors
H = 20
shock_var = 1  # GDP as the shock variable

lp_model = estimate_lp(Y, shock_var, H;
    lags = 4,
    cov_type = :newey_west,
    bandwidth = 0  # Automatic bandwidth selection
)

# Extract IRF with confidence intervals
lp_result = lp_irf(lp_model; conf_level = 0.95)

println("LP-IRF of shock to variable 1 → variable 1:")
for h in 0:4:H
    val = round(lp_result.values[h+1, 1], digits=3)
    se = round(lp_result.se[h+1, 1], digits=3)
    println("  h=$h: $val (SE: $se)")
end
```

### LP with Instrumental Variables

```julia
# Generate external instrument (e.g., monetary policy shock proxy)
Random.seed!(123)
Z = 0.5 * Y[:, 3] + randn(T, 1)  # Correlated with rate but exogenous

# Estimate LP-IV
shock_var = 3  # Instrument for rate shock
lpiv_model = estimate_lp_iv(Y, shock_var, Z, H;
    lags = 4,
    cov_type = :newey_west
)

# Check instrument strength
weak_test = weak_instrument_test(lpiv_model; threshold = 10.0)
println("\nFirst-stage F-statistics by horizon:")
for h in 0:4:H
    F = round(weak_test.F_stats[h+1], digits=2)
    status = F > 10 ? "✓" : "⚠ weak"
    println("  h=$h: F=$F $status")
end
println("All horizons pass F>10: ", weak_test.passes_threshold)

# Extract IRF
lpiv_result = lp_iv_irf(lpiv_model)
```

### Smooth Local Projection

```julia
# Estimate smooth LP with B-splines
smooth_model = estimate_smooth_lp(Y, 1, H;
    degree = 3,      # Cubic splines
    n_knots = 4,     # Interior knots
    lambda = 1.0,    # Smoothing parameter
    lags = 4
)

# Cross-validate lambda
optimal_lambda = cross_validate_lambda(Y, 1, H;
    lambda_grid = 10.0 .^ (-4:0.5:2),
    k_folds = 5
)
println("\nOptimal smoothing parameter: ", round(optimal_lambda, digits=4))

# Compare standard vs smooth LP
comparison = compare_smooth_lp(Y, 1, H; lambda = optimal_lambda)
println("Variance reduction ratio: ", round(comparison.variance_reduction, digits=3))
```

### State-Dependent Local Projection

```julia
# Construct state variable (moving average of GDP growth)
gdp_level = cumsum(Y[:, 1])  # Integrate growth to get level
gdp_growth = [NaN; diff(gdp_level)]

# 4-period moving average, standardized
state_var = zeros(T)
for t in 4:T
    state_var[t] = mean(Y[t-3:t, 1])
end
state_var = (state_var .- mean(state_var[4:end])) ./ std(state_var[4:end])

# Estimate state-dependent LP
state_model = estimate_state_lp(Y, 1, state_var, H;
    gamma = 1.5,           # Transition speed
    threshold = :median,    # Threshold at median
    lags = 4
)

# Extract regime-specific IRFs
irf_both = state_irf(state_model; regime = :both)

println("\nState-dependent IRFs (shock 1 → variable 1):")
println("Expansion vs Recession comparison:")
for h in [0, 4, 8, 12]
    exp_val = round(irf_both.expansion.values[h+1, 1], digits=3)
    rec_val = round(irf_both.recession.values[h+1, 1], digits=3)
    diff = round(exp_val - rec_val, digits=3)
    println("  h=$h: Expansion=$exp_val, Recession=$rec_val, Diff=$diff")
end

# Test for regime differences
diff_test = test_regime_difference(state_model)
println("\nJoint test for regime differences:")
println("  Average |t|: ", round(diff_test.joint_test.avg_t_stat, digits=2))
println("  p-value: ", round(diff_test.joint_test.p_value, digits=4))
```

### Structural LP (Plagborg-Moller & Wolf 2021)

Structural LP extends standard LP to jointly identify multiple shocks, enabling direct comparison with VAR-based IRFs. The `structural_lp` function estimates impulse responses for all shocks simultaneously.

```julia
using MacroEconometricModels, Random

Random.seed!(42)
Y = randn(200, 3)

# Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# Multi-shock IRFs: response of variable 2 to shock 1
slp.irfs[1, 2, :]  # H+1 vector of responses

# Historical decomposition from structural LP
hd = historical_decomposition(slp)
```

**Interpretation.** Structural LP produces IRFs that are asymptotically equivalent to VAR-based IRFs under correct specification (Plagborg-Moller and Wolf 2021). The key advantage is robustness to lag length misspecification. Use `structural_lp` when you want multi-shock identification without committing to a specific VAR lag order.

### LP-FEVD (Gorodnichenko & Lee 2019)

LP-FEVD decomposes forecast error variance using local projection methods, providing a model-free alternative to VAR-based FEVD.

```julia
using MacroEconometricModels, Random

Random.seed!(42)
Y = randn(200, 3)

# First estimate structural LP
slp = structural_lp(Y, 20; method=:cholesky, lags=4)

# LP-FEVD with R² estimator
lfevd = lp_fevd(slp; estimator=:r2)

# Decomposition at horizon h: fraction of variable j's forecast error due to shock i
lfevd.decomposition  # (H+1) × n_shocks × n_vars array
```

**Interpretation.** The R² estimator measures the fraction of h-step-ahead forecast error variance attributable to each structural shock. Unlike VAR-FEVD, LP-FEVD does not require invertibility of the MA representation and is robust to lag truncation. The decomposition should sum to approximately 1 across shocks at each horizon. See [Innovation Accounting](innovation_accounting.md) for a detailed comparison of VAR-FEVD and LP-FEVD.

---

## Example 10: Factor Model for Large Panels

This example demonstrates factor extraction and selection from a large macroeconomic panel.

### Simulate Large Panel Data

```julia
using MacroEconometricModels
using Random
using Statistics

Random.seed!(42)

# Panel dimensions
T = 150   # Time periods
N = 50    # Variables
r_true = 3  # True number of factors

# Generate true factors (with persistence)
F_true = zeros(T, r_true)
for j in 1:r_true
    F_true[1, j] = randn()
    for t in 2:T
        F_true[t, j] = 0.8 * F_true[t-1, j] + 0.3 * randn()
    end
end

# Factor loadings (sparse structure)
Λ_true = randn(N, r_true)
# Make first 15 vars load strongly on factor 1, etc.
Λ_true[1:15, 1] .*= 2
Λ_true[16:30, 2] .*= 2
Λ_true[31:45, 3] .*= 2

# Generate panel
X = F_true * Λ_true' + 0.5 * randn(T, N)

println("Panel: T=$T, N=$N, true r=$r_true")
```

### Determine Number of Factors

```julia
# Bai-Ng information criteria
r_max = 10
ic = ic_criteria(X, r_max)

println("\nBai-Ng information criteria:")
println("  IC1 selects: ", ic.r_IC1, " factors")
println("  IC2 selects: ", ic.r_IC2, " factors")
println("  IC3 selects: ", ic.r_IC3, " factors")
println("  (True: $r_true factors)")

# IC values for each r
println("\nIC values by number of factors:")
for r in 1:r_max
    println("  r=$r: IC1=$(round(ic.IC1[r], digits=4)), IC2=$(round(ic.IC2[r], digits=4))")
end
```

### Estimate Factor Model

```julia
# Use IC2's recommendation
r_opt = ic.r_IC2

# Estimate factor model
fm = estimate_factors(X, r_opt; standardize = true)

println("\nEstimated factor model:")
println("  Number of factors: ", fm.r)
println("  Factors dimension: ", size(fm.factors))
println("  Loadings dimension: ", size(fm.loadings))

# Variance explained
println("\nVariance explained:")
for j in 1:r_opt
    pct = round(fm.explained_variance[j] * 100, digits=1)
    cum = round(fm.cumulative_variance[j] * 100, digits=1)
    println("  Factor $j: $pct% (cumulative: $cum%)")
end
```

### Model Diagnostics

```julia
# R² for each variable
r2_vals = r2(fm)

println("\nR² statistics:")
println("  Mean: ", round(mean(r2_vals), digits=3))
println("  Median: ", round(median(r2_vals), digits=3))
println("  Min: ", round(minimum(r2_vals), digits=3))
println("  Max: ", round(maximum(r2_vals), digits=3))

# Variables well-explained (R² > 0.5)
well_explained = sum(r2_vals .> 0.5)
println("  Variables with R² > 0.5: $well_explained / $N")

# Factor-true factor correlation (up to rotation)
println("\nFactor recovery (correlation with true factors):")
for j in 1:r_opt
    cors = [abs(cor(fm.factors[:, j], F_true[:, k])) for k in 1:r_true]
    best_match = argmax(cors)
    println("  Estimated factor $j matches true factor $best_match: r=$(round(cors[best_match], digits=3))")
end
```

The Bai-Ng information criteria select the number of factors by balancing fit against complexity. IC2 tends to perform best in simulations. High correlations between estimated and true factors (above 0.9) confirm reliable factor recovery. The R² values show how well the common factors explain each variable; variables with low R² are primarily driven by idiosyncratic shocks and contribute less to the common component.

### Factor Model Forecasting

```julia
# Forecast 12 steps ahead with theoretical (analytical) CIs
fc = forecast(fm, 12; ci_method=:theoretical, conf_level=0.95)

println("\nFactor forecast with 95% CIs:")
println("  Factors: ", size(fc.factors))        # 12×r
println("  Observables: ", size(fc.observables)) # 12×N
println("  CI method: ", fc.ci_method)

# SEs should increase with horizon (growing uncertainty)
println("\nFactor 1 SE by horizon:")
for h in [1, 4, 8, 12]
    println("  h=$h: SE=$(round(fc.factors_se[h, 1], digits=4))")
end

# Bootstrap CIs (non-parametric, no Gaussian assumption)
fc_boot = forecast(fm, 12; ci_method=:bootstrap, n_boot=500, conf_level=0.90)

println("\nBootstrap vs theoretical CI widths (Factor 1, h=12):")
width_theory = fc.factors_upper[12, 1] - fc.factors_lower[12, 1]
width_boot = fc_boot.factors_upper[12, 1] - fc_boot.factors_lower[12, 1]
println("  Theoretical: ", round(width_theory, digits=3))
println("  Bootstrap: ", round(width_boot, digits=3))
```

The theoretical SEs grow monotonically with the forecast horizon for stationary factor dynamics, reflecting accumulating forecast uncertainty. Bootstrap CIs are useful when factor innovations may be non-Gaussian or exhibit conditional heteroskedasticity.

### Dynamic Factor Model Forecasting

```julia
# Estimate DFM with VAR(2) factor dynamics
dfm = estimate_dynamic_factors(X, r_opt, 2)

# Forecast with all CI methods
fc_none = forecast(dfm, 12)                                    # Point only
fc_theo = forecast(dfm, 12; ci_method=:theoretical)            # Analytical CIs
fc_boot = forecast(dfm, 12; ci_method=:bootstrap, n_boot=500)  # Bootstrap CIs
fc_sim  = forecast(dfm, 12; ci_method=:simulation, n_boot=500) # Simulation CIs

println("\nDFM forecast comparison (Observable 1, h=12):")
println("  Point forecast: ", round(fc_none.observables[12, 1], digits=3))
println("  Theoretical CI: [", round(fc_theo.observables_lower[12, 1], digits=3),
        ", ", round(fc_theo.observables_upper[12, 1], digits=3), "]")
println("  Bootstrap CI:   [", round(fc_boot.observables_lower[12, 1], digits=3),
        ", ", round(fc_boot.observables_upper[12, 1], digits=3), "]")
```

The DFM supports four CI methods: `:theoretical` (fastest, assumes Gaussian innovations), `:bootstrap` (residual resampling), `:simulation` (full Monte Carlo draws), and the legacy `ci=true` interface which maps to `:simulation`.

---

## Example 11: Panel VAR Analysis

This example demonstrates the full Panel VAR workflow: data construction, lag selection, GMM estimation, specification tests, structural analysis, and bootstrap confidence intervals. See [Panel VAR](pvar.md) for theory and method details.

```julia
using MacroEconometricModels, DataFrames, Random

Random.seed!(42)

# --- Step 1: Construct panel data ---
N, T_total, m = 50, 20, 3
data = zeros(N * T_total, m)
for i in 1:N
    mu_i = randn(m)
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu_i + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.1 * randn(m)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)
```

### Lag Selection

```julia
# Andrews-Lu MMSC-based lag selection (max 4 lags)
lag_result = pvar_lag_selection(pd, 4)
```

**Interpretation.** The lag selection procedure computes MMSC-AIC, MMSC-BIC, and MMSC-HQIC for each candidate lag length. Lower values indicate better fit-complexity trade-off. Choose the lag minimizing BIC for a conservative choice.

### Two-Step GMM Estimation

```julia
# Estimate PVAR with two-step GMM and Windmeijer correction
model = estimate_pvar(pd, 2; steps=:twostep)
```

### Specification Tests

```julia
# Hansen J-test for overidentifying restrictions
j = pvar_hansen_j(model)

# Andrews-Lu MMSC for moment selection
mmsc = pvar_mmsc(model)

# Stability check (all eigenvalues inside unit circle)
stab = pvar_stability(model)
```

**Interpretation.** A non-rejected Hansen J-test (large p-value) supports the validity of the instruments. The MMSC criteria help select among alternative moment conditions. All eigenvalues of the companion matrix should be inside the unit circle for the PVAR to be stable.

### Structural Analysis

```julia
# Orthogonalized IRF (Cholesky identification)
irfs = pvar_oirf(model, 10)

# Generalized IRF (order-invariant, Pesaran & Shin 1998)
girfs = pvar_girf(model, 10)

# Forecast error variance decomposition
fv = pvar_fevd(model, 10)
```

### Bootstrap Confidence Intervals

```julia
# Group-level block bootstrap for IRF confidence intervals
boot_irfs = pvar_bootstrap_irf(model, 10; n_draws=200)
```

**Interpretation.** The bootstrap resamples entire cross-sectional units (groups) to preserve within-unit dependence. With N=50 groups and 200 draws, the resulting confidence intervals account for both estimation uncertainty and cross-sectional heterogeneity. Report median IRFs with 90% bootstrap bands.

### Alternative Estimators

```julia
# Fixed-Effects OLS (within estimator)
fe_model = estimate_pvar_feols(pd, 2)

# System GMM (Blundell & Bond 1998)
sys_model = estimate_pvar(pd, 2; steps=:twostep, system_instruments=true)
```

**Interpretation.** FE-OLS is consistent when T is large relative to N but suffers from Nickell bias in short panels. System GMM adds level equations with lagged differences as instruments, improving efficiency when the first-difference instruments are weak (near unit root). Compare coefficient estimates across estimators as a robustness check.

---

## Example 12: GMM Estimation

This example demonstrates GMM estimation of a simple model with moment conditions.

### Define Moment Conditions

```julia
using MacroEconometricModels

# Example: IV regression via GMM
# Model: y = x'β + ε
# Moment conditions: E[z(y - x'β)] = 0

# Generate data with endogeneity
Random.seed!(42)
n_obs = 500
n_params = 2

# Instruments
Z = randn(n_obs, 3)

# Endogenous regressor (correlated with error)
u = randn(n_obs)
X = hcat(ones(n_obs), Z[:, 1] + 0.5 * u + 0.2 * randn(n_obs))

# Outcome
β_true = [1.0, 2.0]
Y = X * β_true + u

# Data bundle
data = (Y = Y, X = X, Z = hcat(ones(n_obs), Z))

# Moment function: E[Z'(Y - Xβ)] = 0
function moment_conditions(theta, data)
    residuals = data.Y - data.X * theta
    data.Z .* residuals  # n_obs × n_moments matrix
end
```

### GMM Estimation

```julia
# Initial values
theta0 = zeros(n_params)

# Two-step efficient GMM
gmm_result = estimate_gmm(moment_conditions, theta0, data;
    weighting = :two_step,
    hac = true
)

println("GMM Estimation Results:")
println("  True β: ", β_true)
println("  Estimated β: ", round.(gmm_result.theta, digits=4))
println("  Converged: ", gmm_result.converged)
println("  Iterations: ", gmm_result.iterations)

# Standard errors
se = sqrt.(diag(gmm_result.vcov))
println("\n  Standard errors: ", round.(se, digits=4))

# Confidence intervals
z = 1.96
for i in 1:n_params
    lo = round(gmm_result.theta[i] - z * se[i], digits=4)
    hi = round(gmm_result.theta[i] + z * se[i], digits=4)
    println("  β[$i]: 95% CI = [$lo, $hi]")
end
```

### J-Test for Overidentification

```julia
# Test overidentifying restrictions
j_result = j_test(gmm_result)

println("\nHansen J-test:")
println("  J-statistic: ", round(j_result.J_stat, digits=4))
println("  Degrees of freedom: ", j_result.df)
println("  p-value: ", round(j_result.p_value, digits=4))
println("  Reject at 5%: ", j_result.reject_05)
```

The GMM estimates should be close to the true values ``\beta = [1.0, 2.0]`` when instruments are valid and strong. The standard errors from two-step efficient GMM are asymptotically optimal. The Hansen J-test evaluates whether the moment conditions are jointly satisfied: a large p-value (failing to reject) indicates that the instruments are valid and the model is correctly specified. Rejection suggests either invalid instruments or model misspecification.

---

## Example 13: Non-Gaussian Identification

When structural shocks are non-Gaussian, statistical independence provides identification without imposing economic restrictions like recursive ordering or sign constraints. This example demonstrates the full non-Gaussian identification workflow: testing for non-Gaussianity, ICA-based and ML-based identification, and post-estimation specification tests.

### Setup: Generate Non-Gaussian Data

```julia
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

Random.seed!(42)

# True structural parameters
T = 500
n = 3

# True B₀ (structural impact matrix)
B0_true = [1.0  0.0  0.0;
           0.5  1.0  0.0;
           0.3 -0.2  1.0]

# Non-Gaussian structural shocks (Student-t with 5 df)
# Heavy tails provide the non-Gaussianity needed for identification
eps = zeros(T, n)
for j in 1:n
    # Standardized t(5): mean 0, variance 1
    raw = randn(T) ./ sqrt.(rand(Chisq(5), T) ./ 5)
    eps[:, j] = raw ./ std(raw)
end

# True VAR(1) dynamics
A_true = [0.7 0.1 0.0;
          0.0 0.6 0.1;
          0.0 0.0 0.5]

# Generate reduced-form data: u_t = B₀ ε_t
Y = zeros(T, n)
Y[1, :] = B0_true * eps[1, :]
for t in 2:T
    Y[t, :] = A_true * Y[t-1, :] + B0_true * eps[t, :]
end

println("Data: T=$T, n=$n (non-Gaussian DGP with t(5) shocks)")
```

### Step 1: Test for Non-Gaussianity

Before using non-Gaussian identification, verify that residuals are indeed non-Gaussian:

```julia
# Estimate VAR
model = estimate_var(Y, 1)

# Run the full normality test suite
suite = normality_test_suite(model)

println("Multivariate Normality Tests (H₀: residuals are Gaussian)")
println("="^55)
for r in suite.results
    stars = r.pvalue < 0.01 ? "***" : r.pvalue < 0.05 ? "**" : r.pvalue < 0.10 ? "*" : ""
    println("  $(r.test_name): stat=$(round(r.statistic, digits=2)), p=$(round(r.pvalue, digits=4)) $stars")
end
```

All four tests (Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler) should reject normality when the true shocks are t-distributed. If normality is not rejected, non-Gaussian identification may lack power and Cholesky or sign restrictions should be preferred.

You can also run individual tests:

```julia
# Individual tests
jb = jarque_bera_test(model)
mardia = mardia_test(model; type=:both)
dh = doornik_hansen_test(model)
hz = henze_zirkler_test(model)

println("\nDetailed Mardia test:")
println("  Skewness stat: ", round(mardia.statistic, digits=2))
println("  P-value: ", round(mardia.pvalue, digits=4))
```

### Step 2: ICA-Based Identification

ICA (Independent Component Analysis) recovers structurally independent shocks by maximizing statistical independence:

```julia
# FastICA identification (default: logcosh contrast)
ica_result = identify_fastica(model; contrast=:logcosh, approach=:deflation)

println("\nFastICA Identification")
println("="^40)
println("  Converged: ", ica_result.converged)
println("  Iterations: ", ica_result.iterations)
println("  Objective: ", round(ica_result.objective, digits=6))

# Structural impact matrix B₀
println("\nEstimated B₀ (structural impact matrix):")
for i in 1:n
    println("  ", [round(ica_result.B0[i, j], digits=3) for j in 1:n])
end
```

Compare different ICA algorithms:

```julia
# JADE (Joint Approximate Diagonalization of Eigenmatrices)
jade_result = identify_jade(model)

# SOBI (Second-Order Blind Identification — exploits temporal structure)
sobi_result = identify_sobi(model; lags=1:12)

# Distance-covariance ICA
dcov_result = identify_dcov(model)

println("\nComparison of ICA methods:")
println("  FastICA converged: ", ica_result.converged, " (iter: ", ica_result.iterations, ")")
println("  JADE converged:    ", jade_result.converged, " (iter: ", jade_result.iterations, ")")
println("  SOBI converged:    ", sobi_result.converged, " (iter: ", sobi_result.iterations, ")")
println("  dCov converged:    ", dcov_result.converged, " (iter: ", dcov_result.iterations, ")")
```

FastICA is the fastest and most commonly used, but JADE is more robust when multiple shocks have similar kurtosis. SOBI exploits temporal dependence and works even with mildly non-Gaussian shocks.

### Step 3: Compute IRFs with ICA Identification

The rotation matrix `Q` from ICA integrates directly with the standard `irf()` and `fevd()` functions:

```julia
# IRF using FastICA-identified structure
irfs_ica = irf(model, 20; method=:fastica)

println("\nFastICA-identified IRF (shock 1 → all variables):")
for h in [0, 4, 8, 12, 20]
    vals = [round(irfs_ica.irf[h+1, v, 1], digits=3) for v in 1:n]
    println("  h=$h: ", vals)
end

# FEVD using ICA identification
fevd_ica = fevd(model, 20; method=:fastica)

println("\nFEVD at h=20 (ICA-identified):")
for v in 1:n
    shares = [round(fevd_ica.fevd[21, v, s] * 100, digits=1) for s in 1:n]
    println("  Variable $v: ", shares, "%")
end
```

Unlike Cholesky identification, the ICA-based IRFs do not depend on variable ordering. The same data produces the same structural shocks regardless of how the columns of ``Y`` are arranged.

### Step 4: ML-Based Identification

Maximum likelihood methods parameterize the shock distribution and jointly estimate ``B_0`` and the distributional parameters:

```julia
# Student-t ML identification
ml_t = identify_student_t(model)

println("\nStudent-t ML Identification")
println("="^40)
println("  Converged: ", ml_t.converged)
println("  Log-likelihood (non-Gaussian): ", round(ml_t.loglik, digits=2))
println("  Log-likelihood (Gaussian):     ", round(ml_t.loglik_gaussian, digits=2))
println("  AIC: ", round(ml_t.aic, digits=2))
println("  BIC: ", round(ml_t.bic, digits=2))

# Estimated degrees of freedom for each shock
if haskey(ml_t.dist_params, :nu)
    println("  Estimated ν (df): ", round.(ml_t.dist_params[:nu], digits=2))
end

# Standard errors for B₀ elements
println("\nB₀ standard errors:")
for i in 1:n
    println("  ", [round(ml_t.se[i, j], digits=4) for j in 1:n])
end
```

The Student-t ML approach provides standard errors for ``B_0`` elements, unlike ICA which only gives point estimates. Compare with other distributional assumptions:

```julia
# Mixture of normals
ml_mix = identify_mixture_normal(model; n_components=2)

# Pseudo-maximum likelihood (robust, no distributional assumption)
ml_pml = identify_pml(model)

# Unified interface — select distribution via keyword
ml_auto = identify_nongaussian_ml(model; distribution=:student_t)

println("\nML method comparison (AIC):")
println("  Student-t:      AIC = ", round(ml_t.aic, digits=2))
println("  Mixture normal: AIC = ", round(ml_mix.aic, digits=2))
println("  PML:            AIC = ", round(ml_pml.aic, digits=2))
```

Lower AIC indicates a better distributional fit. The PML estimator is semiparametrically efficient and does not require specifying the shock distribution.

### Step 5: Heteroskedasticity-Based Identification

When shocks exhibit time-varying volatility, changes in the covariance structure can identify the structural model:

```julia
# External volatility regimes (e.g., pre/post Great Moderation)
regime = vcat(ones(Int, 250), 2 * ones(Int, 250))  # Two regimes
vol_result = identify_external_volatility(model, regime; regimes=2)

println("\nExternal Volatility Identification")
println("="^40)
println("  Regime 1 shock variances: ",
        [round(vol_result.Lambda_vecs[1][j], digits=3) for j in 1:n])
println("  Regime 2 shock variances: ",
        [round(vol_result.Lambda_vecs[2][j], digits=3) for j in 1:n])
```

### Step 6: Post-Estimation Specification Tests

Verify that the identification assumptions hold:

```julia
# Test 1: Are recovered shocks non-Gaussian?
gauss_test = test_shock_gaussianity(ica_result)
println("\nShock Gaussianity Test (H₀: shocks are Gaussian)")
println("  Statistic: ", round(gauss_test.statistic, digits=2))
println("  P-value: ", round(gauss_test.pvalue, digits=4))
println("  Non-Gaussian: ", gauss_test.identified)

# Test 2: Are recovered shocks independent?
indep_test = test_shock_independence(ica_result; max_lag=10)
println("\nShock Independence Test (H₀: shocks are independent)")
println("  Statistic: ", round(indep_test.statistic, digits=2))
println("  P-value: ", round(indep_test.pvalue, digits=4))
println("  Independent: ", indep_test.identified)

# Test 3: Identification strength (bootstrap)
strength_test = test_identification_strength(model; method=:fastica, n_bootstrap=499)
println("\nIdentification Strength Test")
println("  Statistic: ", round(strength_test.statistic, digits=4))
println("  P-value: ", round(strength_test.pvalue, digits=4))
println("  Strongly identified: ", strength_test.identified)

# Test 4: Gaussian vs non-Gaussian likelihood ratio
lr_test = test_gaussian_vs_nongaussian(model; method=:fastica, n_bootstrap=499)
println("\nLR Test: Gaussian vs Non-Gaussian")
println("  LR statistic: ", round(lr_test.statistic, digits=2))
println("  P-value: ", round(lr_test.pvalue, digits=4))
```

A valid non-Gaussian SVAR requires: (1) rejection of shock Gaussianity (non-Gaussian shocks are needed for identification), (2) failure to reject shock independence (the identified shocks should be independent), and (3) strong identification (the structural parameters are precisely estimated). If the Gaussianity test fails to reject, the data may not contain enough non-Gaussianity to identify the model, and traditional Cholesky or sign restrictions should be used instead.

### Comparing Cholesky vs ICA Identification

```julia
# Cholesky IRF (ordering-dependent)
irfs_chol = irf(model, 20; method=:cholesky)

# ICA IRF (ordering-independent)
irfs_ica = irf(model, 20; method=:fastica)

println("\nCholesky vs FastICA IRF comparison (shock 1 → variable 1):")
println("  h   Cholesky   FastICA")
for h in [0, 4, 8, 12, 20]
    chol_val = round(irfs_chol.irf[h+1, 1, 1], digits=3)
    ica_val = round(irfs_ica.irf[h+1, 1, 1], digits=3)
    println("  $h    $chol_val      $ica_val")
end
```

When the true DGP is recursive (lower-triangular ``B_0``), Cholesky and ICA should yield similar IRFs. Large discrepancies suggest that the recursive assumption may be misspecified, and the data-driven ICA identification should be preferred.

---

## Example 14: Complete Workflow

This example shows a complete empirical workflow combining multiple techniques.

```julia
using MacroEconometricModels
using Random
using Statistics

Random.seed!(2024)

# === Step 1: Data Preparation ===
T, n = 200, 4
Y = randn(T, n)
for t in 2:T
    Y[t, :] = 0.6 * Y[t-1, :] + 0.3 * randn(n)
end
var_names = ["Output", "Inflation", "Rate", "Exchange Rate"]

# === Step 2: Lag Selection ===
println("="^50)
println("Step 1: Lag Selection")
println("="^50)

aics = Float64[]
bics = Float64[]
for p in 1:8
    m = estimate_var(Y, p)
    push!(aics, aic(m))
    push!(bics, bic(m))
end
p_aic = argmin(aics)
p_bic = argmin(bics)
println("AIC selects p=$p_aic, BIC selects p=$p_bic")
p = p_bic  # Use BIC's conservative choice

# === Step 3: VAR Estimation ===
println("\n" * "="^50)
println("Step 2: VAR Estimation")
println("="^50)

model = estimate_var(Y, p)
println("Estimated VAR($p)")
println("Log-likelihood: ", round(loglikelihood(model), digits=2))

# === Step 4: Frequentist IRF ===
println("\n" * "="^50)
println("Step 3: Impulse Response Analysis")
println("="^50)

H = 20
irfs = irf(model, H; method=:cholesky)
fevd_res = fevd(model, H; method=:cholesky)

# === Step 5: Bayesian Estimation ===
println("\n" * "="^50)
println("Step 4: Bayesian Analysis")
println("="^50)

# Optimize priors
best_hyper = optimize_hyperparameters(Y, p; grid_size=15)
println("Optimal τ: ", round(best_hyper.tau, digits=4))

# BVAR with conjugate NIW sampler
post = estimate_bvar(Y, p; n_draws=1000,
                     prior=:minnesota, hyper=best_hyper)

# Bayesian IRF
birf = irf(post, H; method=:cholesky)

# === Step 6: Local Projections Comparison ===
println("\n" * "="^50)
println("Step 5: LP vs VAR Comparison")
println("="^50)

lp_model = estimate_lp(Y, 1, H; lags=p, cov_type=:newey_west)
lp_result = lp_irf(lp_model)

println("IRF(1→1) at h=0:")
println("  VAR: ", round(irfs.irf[1, 1, 1], digits=3))
println("  LP: ", round(lp_result.values[1, 1], digits=3))

println("\nIRF(1→1) at h=8:")
println("  VAR: ", round(irfs.irf[9, 1, 1], digits=3))
println("  LP: ", round(lp_result.values[9, 1], digits=3))

# === Step 7: Robustness Check with Smooth LP ===
smooth_lp = estimate_smooth_lp(Y, 1, H; lambda=1.0, lags=p)
smooth_result = smooth_lp_irf(smooth_lp)

println("\nSmooth LP variance reduction: ",
        round(mean(smooth_result.se.^2) / mean(lp_result.se.^2), digits=3))

println("\n" * "="^50)
println("Analysis Complete!")
println("="^50)
```

Comparing VAR and LP impulse responses at the same horizon provides a robustness check. Under correct specification, both estimators are consistent for the same causal parameter (Plagborg-Moller & Wolf, 2021), but LP is less efficient. Large discrepancies suggest potential dynamic misspecification in the VAR. The smooth LP variance reduction ratio measures efficiency gains from B-spline regularization; values well below 1.0 indicate substantial noise reduction from imposing smoothness.

---

## Example 15: Table Output — Text, LaTeX, and HTML

All `show`, `print_table`, and `Base.show` methods in MacroEconometricModels route through a unified PrettyTables backend. Switching from terminal text to LaTeX or HTML output requires a single call to `set_display_backend`. This is useful for embedding results directly into papers (LaTeX), slides (HTML), or reports.

### Setup

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

# Estimate a VAR and compute IRFs + FEVD
Y = randn(200, 3)
for t in 2:200
    Y[t, :] = [0.8 0.1 0.0; 0.05 0.7 0.0; 0.1 0.2 0.75] * Y[t-1, :] + 0.3 * randn(3)
end

model = estimate_var(Y, 2)
irfs = irf(model, 12; method=:cholesky, ci_type=:bootstrap, n_boot=500)
fevd_result = fevd(model, 12; method=:cholesky)
```

### Text Output (Default)

The default backend is `:text`, producing terminal-friendly borderless tables:

```julia
# Confirm default backend
get_display_backend()   # :text

# Print IRF table for variable 1, shock 1
print_table(irfs, 1, 1)
```

Output:

```
           IRF: Var 1 ← Shock 1
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      h      IRF    CI_lo    CI_hi
  ────────────────────────────────
      1   1.0000   1.0000   1.0000
      4   0.5765   0.3821   0.7542
      8   0.2134   0.0512   0.3891
     12   0.0712  -0.0203   0.1744
```

```julia
# Print FEVD table for variable 2
print_table(fevd_result, 2)
```

### LaTeX Output for Papers

Switch to LaTeX to get tables ready for `\input{}` in your `.tex` file:

```julia
# Switch to LaTeX backend
set_display_backend(:latex)

# Print IRF table — output is now LaTeX
print_table(irfs, 1, 1)
```

Output:

```latex
\begin{table}
  \caption{IRF: Var 1 ← Shock 1}
  \begin{tabular}{rrrr}
    \hline
    h & IRF & CI\_lo & CI\_hi \\
    \hline
    1 & 1.0 & 1.0 & 1.0 \\
    4 & 0.5765 & 0.3821 & 0.7542 \\
    8 & 0.2134 & 0.0512 & 0.3891 \\
    12 & 0.0712 & -0.0203 & 0.1744 \\
    \hline
  \end{tabular}
\end{table}
```

To save LaTeX output directly to a file:

```julia
set_display_backend(:latex)

# Write IRF table to file
open("tables/irf_table.tex", "w") do io
    print_table(io, irfs, 1, 1)
end

# Write FEVD table to file
open("tables/fevd_table.tex", "w") do io
    print_table(io, fevd_result, 2)
end
```

Then in your LaTeX document:

```latex
\begin{document}
Table~\ref{tab:irf} reports the impulse responses...
\input{tables/irf_table.tex}
\end{document}
```

### HTML Output for Slides and Web

Switch to HTML for Jupyter notebooks, web dashboards, or HTML-based presentations:

```julia
# Switch to HTML backend
set_display_backend(:html)

# Print IRF table — output is now an HTML <table>
print_table(irfs, 1, 1)
```

Output:

```html
<table>
  <caption>IRF: Var 1 ← Shock 1</caption>
  <tr><th>h</th><th>IRF</th><th>CI_lo</th><th>CI_hi</th></tr>
  <tr><td>1</td><td>1.0</td><td>1.0</td><td>1.0</td></tr>
  <tr><td>4</td><td>0.5765</td><td>0.3821</td><td>0.7542</td></tr>
  ...
</table>
```

To save HTML output to a file:

```julia
set_display_backend(:html)

open("tables/irf_table.html", "w") do io
    print_table(io, irfs, 1, 1)
end
```

### Switching Backends in a Workflow

You can switch backends freely within a session. A common pattern for a research workflow:

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

Y = randn(200, 3)
for t in 2:200
    Y[t, :] = [0.8 0.1 0.0; 0.05 0.7 0.0; 0.1 0.2 0.75] * Y[t-1, :] + 0.3 * randn(3)
end

model = estimate_var(Y, 2)
H = 20
irfs = irf(model, H; method=:cholesky, ci_type=:bootstrap, n_boot=500)
fevd_result = fevd(model, H; method=:cholesky)
hd_result = historical_decomposition(model)

# === Step 1: Inspect in terminal ===
set_display_backend(:text)
print_table(irfs, 1, 1)       # Quick look at IRF
print_table(fevd_result, 1)    # Quick look at FEVD

# === Step 2: Export LaTeX for the paper ===
set_display_backend(:latex)

open("tables/irf_gdp.tex", "w") do io
    print_table(io, irfs, 1, 1; horizons=[1, 4, 8, 12, 20])
end

open("tables/fevd_gdp.tex", "w") do io
    print_table(io, fevd_result, 1; horizons=[1, 4, 8, 12, 20])
end

# === Step 3: Export HTML for slides ===
set_display_backend(:html)

open("slides/irf_gdp.html", "w") do io
    print_table(io, irfs, 1, 1)
end

# === Step 4: Reset to text for continued interactive work ===
set_display_backend(:text)
```

### Using `table()` to Extract Raw Data

The `table()` function returns a plain `Matrix` that you can manipulate, pass to DataFrames, or export with CSV:

```julia
using DataFrames, CSV

# Extract IRF as a matrix: columns are [h, IRF, CI_lo, CI_hi]
irf_data = table(irfs, 1, 1; horizons=[1, 4, 8, 12, 20])

# Convert to DataFrame
df = DataFrame(irf_data, [:h, :IRF, :CI_lo, :CI_hi])

# Save as CSV
CSV.write("tables/irf_data.csv", df)

# Extract FEVD as a matrix: columns are [h, Shock1, Shock2, ..., ShockN]
fevd_data = table(fevd_result, 1; horizons=[1, 4, 8, 12, 20])
```

### Backend Affects `show()` Too

The display backend also controls how objects render when printed in the REPL or displayed in Jupyter:

```julia
set_display_backend(:latex)

# REPL display is now LaTeX
model    # VARModel show → LaTeX tables
irfs     # ImpulseResponse show → LaTeX tables

set_display_backend(:text)  # Reset
```

This means in a Jupyter notebook, you can set the backend to `:html` once at the top:

```julia
# Top of Jupyter notebook
using MacroEconometricModels
set_display_backend(:html)

# All subsequent cells render as formatted HTML tables
model = estimate_var(Y, 2)
irfs = irf(model, 12; method=:cholesky)
irfs   # Displays as an HTML table
```

### Summary of Output Functions

| Function | Returns | Use Case |
|---|---|---|
| `table(result, ...)` | `Matrix` | Raw numeric data for custom processing, CSV export |
| `print_table([io], result, ...)` | Nothing (prints) | Formatted output via current backend (text/LaTeX/HTML) |
| `show(io, result)` | Nothing (prints) | REPL display, also respects backend |
| `set_display_backend(:text)` | Nothing | Terminal output (default) |
| `set_display_backend(:latex)` | Nothing | LaTeX `\begin{tabular}` output |
| `set_display_backend(:html)` | Nothing | HTML `<table>` output |
| `get_display_backend()` | `Symbol` | Check current backend |

---

## Example 16: Bibliographic References

The `refs()` function returns bibliographic references for any model, result type, or identification method. References are available in four formats: AEA text (default), BibTeX, LaTeX `\bibitem`, and HTML with clickable DOI links.

### Basic Usage

```julia
using MacroEconometricModels
using Random

Random.seed!(42)

# Estimate a model
Y = randn(200, 3)
for t in 2:200
    Y[t, :] = 0.5 * Y[t-1, :] + 0.3 * randn(3)
end
model = estimate_var(Y, 2)

# Get references for this model type (AEA text format)
refs(model)
```

Output:
```
Sims, Christopher A. 1980. "Macroeconomics and Reality." Econometrica 48 (1): 1-48.
Lutkepohl, Helmut. 2005. New Introduction to Multiple Time Series Analysis. Berlin: Springer.
```

### Multiple Output Formats

```julia
# BibTeX format — paste into your .bib file
refs(model; format=:bibtex)

# LaTeX \bibitem format
refs(model; format=:latex)

# HTML with clickable DOI links
refs(model; format=:html)
```

### References by Method Name

```julia
# References for identification methods
refs(:cholesky)       # Cholesky decomposition
refs(:fastica)        # FastICA for SVAR
refs(:sign)           # Sign restrictions
refs(:johansen)       # Johansen cointegration
refs(:garch)          # GARCH models

# References for specific result types
garch = estimate_garch(randn(500), 1, 1)
refs(garch)           # Bollerslev (1986)

sv = estimate_sv(randn(500); n_samples=500, burnin=200)
refs(sv)              # Taylor (1986), Kim et al. (1998), Omori et al. (2007)
```

### Export to .bib File

```julia
# Write BibTeX entries for all models used in your analysis
open("references.bib", "w") do io
    refs(io, model; format=:bibtex)
    println(io)
    refs(io, :fastica; format=:bibtex)
    println(io)
    refs(io, :johansen; format=:bibtex)
end
```

The `refs()` function covers all 45+ references in the package's database, including every estimation method, identification scheme, and test. This ensures correct citation of the methods used in your empirical analysis.

---

## Best Practices

### Data Preparation

1. **Stationarity**: Test for unit roots using ADF and KPSS together
   - Both fail to reject → inconclusive, consider structural breaks
   - ADF rejects, KPSS doesn't → stationary (I(0))
   - ADF doesn't reject, KPSS rejects → unit root (I(1))
2. **Structural Breaks**: Use Zivot-Andrews test if visual inspection suggests breaks
3. **Cointegration**: For I(1) variables, test for cointegration before differencing
4. **Outliers**: Check for and handle outliers
5. **Missing data**: Factor models can handle some missing data; VARs require complete data
6. **Scaling**: For factor models, standardize variables

### Model Selection

1. **Lag length**: Use information criteria (BIC is more conservative)
2. **Number of factors**: Use Bai-Ng criteria; prefer IC2 or IC3
3. **Prior tightness**: Optimize via marginal likelihood for large models

### Identification

1. **Economic theory**: Base restrictions on economic reasoning
2. **Robustness**: Try multiple identification schemes
3. **Narrative**: Use historical knowledge when available
4. **Non-Gaussian**: Test residuals with `normality_test_suite` first; if non-Gaussian, ICA/ML methods provide ordering-free identification
5. **Specification tests**: Validate non-Gaussian identification with `test_shock_gaussianity` and `test_shock_independence`

### Inference

1. **HAC standard errors**: Always use for LP at horizons > 0
2. **Credible intervals**: Report 68% and 90% bands for Bayesian
3. **Bootstrap**: Use for frequentist VAR confidence intervals

### Reporting

1. **Present both**: VAR and LP estimates as robustness check
2. **Horizon selection**: Focus on economically meaningful horizons
3. **FEVD**: Report at multiple horizons (short, medium, long-run)
4. **LaTeX export**: Use `set_display_backend(:latex)` then `print_table(io, ...)` for paper-ready tables
5. **HTML export**: Use `set_display_backend(:html)` for Jupyter notebooks and web reports
6. **Raw data**: Use `table(result, ...)` to extract matrices for custom formatting or CSV export

---
