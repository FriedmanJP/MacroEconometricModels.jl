# Examples

This chapter provides comprehensive worked examples demonstrating the main functionality of **MacroEconometricModels.jl**. Each example includes complete code, economic interpretation, and best practices.

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

---

## Best Practices

### Data Preparation

1. **Stationarity**: Ensure data is stationary (difference if needed)
2. **Outliers**: Check for and handle outliers
3. **Missing data**: Factor models can handle some missing data; VARs require complete data
4. **Scaling**: For factor models, standardize variables

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
