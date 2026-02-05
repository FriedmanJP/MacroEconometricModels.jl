# Examples

This chapter provides comprehensive worked examples demonstrating the main functionality of **MacroEconometricModels.jl**. Each example includes complete code, economic interpretation, and best practices.

### Quick Reference

| # | Example | Key Functions | Description |
|---|---------|---------------|-------------|
| 1 | Three-Variable VAR | `estimate_var`, `irf`, `fevd` | Frequentist VAR with Cholesky and sign restriction identification |
| 2 | Bayesian VAR with Minnesota Prior | `estimate_bvar`, `optimize_hyperparameters` | Minnesota prior, MCMC estimation, credible intervals |
| 3 | Local Projections | `estimate_lp`, `estimate_lp_iv`, `estimate_smooth_lp` | Standard, IV, smooth, and state-dependent LP |
| 4 | Factor Model for Large Panels | `estimate_factors`, `ic_criteria` | Large panel factor extraction, Bai-Ng criteria, FAVAR |
| 5 | GMM Estimation | `estimate_gmm`, `j_test` | IV regression via GMM, overidentification test |
| 6 | Complete Workflow | Multiple | Lag selection → VAR → BVAR → LP comparison |
| 7 | Unit Root Testing | `adf_test`, `kpss_test`, `johansen_test` | ADF, KPSS, Zivot-Andrews, Ng-Perron, Johansen |

---

## Example 1: Three-Variable VAR Analysis

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
model = fit(VARModel, Y, p)

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

---

## Example 2: Bayesian VAR with Minnesota Prior

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

### BVAR Estimation with MCMC

```julia
# Estimate BVAR with optimized Minnesota prior
println("\nEstimating BVAR with MCMC...")
chain = estimate_bvar(Y, p;
    n_samples = 2000,
    n_adapts = 500,
    prior = :minnesota,
    hyper = best_hyper
)

# Posterior summary (coefficients from first equation)
println("\nPosterior summary for GDP equation:")
# Access posterior draws and compute statistics
```

### Bayesian IRF with Credible Intervals

```julia
# Bayesian IRF with Cholesky identification
birf_chol = irf(chain, p, n, H; method=:cholesky)

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
birf_sign = irf(chain, p, n, H;
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

## Example 3: Local Projections

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

---

## Example 4: Factor Model for Large Panels

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

---

## Example 5: GMM Estimation

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

## Example 6: Complete Workflow

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
    m = fit(VARModel, Y, p)
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

model = fit(VARModel, Y, p)
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

# BVAR with MCMC
chain = estimate_bvar(Y, p; n_samples=1000, n_adapts=300,
                      prior=:minnesota, hyper=best_hyper)

# Bayesian IRF
birf = irf(chain, p, n, H; method=:cholesky)

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

Comparing VAR and LP impulse responses at the same horizon provides a robustness check. Under correct specification, both estimators are consistent for the same causal parameter (Plagborg-Møller & Wolf, 2021), but LP is less efficient. Large discrepancies suggest potential dynamic misspecification in the VAR. The smooth LP variance reduction ratio measures efficiency gains from B-spline regularization; values well below 1.0 indicate substantial noise reduction from imposing smoothness.

---

## Example 7: Unit Root Testing and Pre-Estimation Analysis

This example demonstrates comprehensive unit root testing before fitting VAR models.

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

### Inference

1. **HAC standard errors**: Always use for LP at horizons > 0
2. **Credible intervals**: Report 68% and 90% bands for Bayesian
3. **Bootstrap**: Use for frequentist VAR confidence intervals

### Reporting

1. **Present both**: VAR and LP estimates as robustness check
2. **Horizon selection**: Focus on economically meaningful horizons
3. **FEVD**: Report at multiple horizons (short, medium, long-run)
