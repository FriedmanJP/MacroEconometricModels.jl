# Factor Model Example
# This script demonstrates how to use the Static Factor Model functionality

using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

# Set random seed for reproducibility
Random.seed!(42)

println("="^70)
println("Factor Model Example - MacroEconometricModels Package")
println("="^70)

# ============================================================================
# Example 1: Basic Factor Model Estimation
# ============================================================================
println("\n--- Example 1: Basic Factor Model Estimation ---\n")

# Generate synthetic data with known factor structure
T = 200  # Number of time periods
N = 30   # Number of variables
r_true = 5  # True number of factors

# Generate true factors
F_true = randn(T, r_true)

# Generate true loadings
Λ_true = randn(N, r_true)

# Generate observed data: X = F * Λ' + noise
noise_std = 0.5
X = F_true * Λ_true' + noise_std * randn(T, N)

println("Data dimensions: T=$T observations, N=$N variables")
println("True number of factors: $r_true")

# Estimate factor model
model = estimate_factors(X, r_true)

println("\nEstimated factor model:")
println("  Number of factors: $(model.r)")
println("  Factors shape: $(size(model.factors))")
println("  Loadings shape: $(size(model.loadings))")
println("  Data was standardized: $(model.standardized)")

# Print variance explained
println("\nVariance explained by each factor:")
for i in 1:r_true
    println("  Factor $i: $(round(model.explained_variance[i] * 100, digits=2))%")
end
println("  Cumulative (first $r_true factors): $(round(model.cumulative_variance[r_true] * 100, digits=2))%")

# ============================================================================
# Example 2: Model Evaluation and Diagnostics
# ============================================================================
println("\n--- Example 2: Model Evaluation and Diagnostics ---\n")

# Compute R² for each variable
r2_values = r2(model)
println("R² statistics:")
println("  Mean R²: $(round(mean(r2_values), digits=3))")
println("  Median R²: $(round(median(r2_values), digits=3))")
println("  Min R²: $(round(minimum(r2_values), digits=3))")
println("  Max R²: $(round(maximum(r2_values), digits=3))")

# Get fitted values and residuals
X_fitted = predict(model)
resid = residuals(model)

println("\nModel fit:")
println("  Number of observations: $(nobs(model))")
println("  Degrees of freedom: $(dof(model))")

# Compute overall fit statistic
total_var = sum(var(X, dims=1))
residual_var = sum(var(resid, dims=1))
overall_r2 = 1 - residual_var / total_var
println("  Overall R²: $(round(overall_r2, digits=3))")

# ============================================================================
# Example 3: Determining the Optimal Number of Factors
# ============================================================================
println("\n--- Example 3: Determining Optimal Number of Factors ---\n")

# Compute information criteria for different numbers of factors
max_factors = 10
ic = ic_criteria(X, max_factors)

println("Information criteria suggest:")
println("  IC1: $(ic.r_IC1) factors")
println("  IC2: $(ic.r_IC2) factors")
println("  IC3: $(ic.r_IC3) factors")
println("  (True number: $r_true)")

# Print IC values for each number of factors
println("\nDetailed IC values:")
println("r\tIC1\t\tIC2\t\tIC3")
for i in 1:max_factors
    println("$i\t$(round(ic.IC1[i], digits=4))\t$(round(ic.IC2[i], digits=4))\t$(round(ic.IC3[i], digits=4))")
end

# ============================================================================
# Example 4: Scree Plot Data
# ============================================================================
println("\n--- Example 4: Scree Plot Data ---\n")

# Extract scree plot data
scree_data = scree_plot_data(model)

println("Scree plot - Variance explained:")
println("Factor\tIndividual\tCumulative")
for i in 1:min(15, length(scree_data.factors))
    println("$(scree_data.factors[i])\t$(round(scree_data.explained_variance[i] * 100, digits=2))%\t\t$(round(scree_data.cumulative_variance[i] * 100, digits=2))%")
end

# ============================================================================
# Example 5: Comparison of Standardized vs Non-Standardized Estimation
# ============================================================================
println("\n--- Example 5: Standardized vs Non-Standardized Estimation ---\n")

# Estimate with standardization (default)
model_std = estimate_factors(X, r_true; standardize=true)

# Estimate without standardization
model_nostd = estimate_factors(X, r_true; standardize=false)

r2_std = mean(r2(model_std))
r2_nostd = mean(r2(model_nostd))

println("Mean R² with standardization: $(round(r2_std, digits=3))")
println("Mean R² without standardization: $(round(r2_nostd, digits=3))")

# ============================================================================
# Example 6: Forecasting with Factor Models (FAVAR-style)
# ============================================================================
println("\n--- Example 6: Using Factors for Forecasting ---\n")

# Extract estimated factors
F_estimated = model.factors

println("Factors can now be used in downstream analysis:")
println("  - As predictors in VAR models (FAVAR)")
println("  - For forecasting target variables")
println("  - For structural analysis")
println("  - For dimensionality reduction")

# Example: Correlation between true and estimated factors
# Note: Factors are identified only up to rotation, so we check correlation magnitude
println("\nCorrelation between true and estimated factors (absolute values):")
println("(Note: Factors are identified up to rotation)")
for i in 1:min(3, r_true)
    # Find the estimated factor most correlated with true factor i
    correlations = [abs(cor(F_true[:, i], F_estimated[:, j])) for j in 1:r_true]
    max_cor = maximum(correlations)
    println("  True factor $i matches with correlation: $(round(max_cor, digits=3))")
end

# ============================================================================
# Example 7: Working with Real Data Patterns
# ============================================================================
println("\n--- Example 7: Realistic Macroeconomic Data Scenario ---\n")

# Simulate data with realistic features
T_macro = 150
N_macro = 50

# Create factors representing economic concepts
# Factor 1: Real activity
# Factor 2: Inflation
# Factor 3: Financial conditions
r_macro = 3

F_macro = zeros(T_macro, r_macro)
# Add persistence (AR process)
for i in 1:r_macro
    F_macro[1, i] = randn()
    for t in 2:T_macro
        F_macro[t, i] = 0.8 * F_macro[t-1, i] + 0.3 * randn()
    end
end

# Loadings with variable strength across series
Λ_macro = randn(N_macro, r_macro)
# Make some variables load strongly on specific factors
Λ_macro[1:10, 1] .*= 2.0  # First 10 variables load strongly on factor 1
Λ_macro[11:20, 2] .*= 2.0  # Next 10 on factor 2
Λ_macro[21:30, 3] .*= 2.0  # Next 10 on factor 3

X_macro = F_macro * Λ_macro' + 0.5 * randn(T_macro, N_macro)

# Determine optimal number of factors
ic_macro = ic_criteria(X_macro, 8)
r_optimal = ic_macro.r_IC2  # Using IC2

println("Macroeconomic data simulation:")
println("  Time periods: $T_macro")
println("  Number of variables: $N_macro")
println("  True factors: $r_macro")
println("  Optimal factors (IC2): $r_optimal")

# Estimate with optimal number
model_macro = estimate_factors(X_macro, r_optimal)

println("\nModel performance:")
println("  Mean R²: $(round(mean(r2(model_macro)), digits=3))")
println("  Variance explained (first $r_optimal factors): $(round(model_macro.cumulative_variance[r_optimal] * 100, digits=2))%")

# Check which variables are well-explained
r2_macro = r2(model_macro)
well_explained = sum(r2_macro .> 0.7)
println("  Variables with R² > 0.7: $well_explained / $N_macro")

println("\n" * "="^70)
println("Examples completed successfully!")
println("="^70)
