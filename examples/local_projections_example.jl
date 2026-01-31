# =============================================================================
# Local Projections Example
# =============================================================================
# This example demonstrates Local Projection methods available in
# MacroEconometricModels.jl, following the methodology of Jordà (2005).

using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

# Set random seed for reproducibility
Random.seed!(42)

# =============================================================================
# 1. Generate Synthetic Data
# =============================================================================
println("=" ^ 60)
println("Local Projections Example")
println("=" ^ 60)

# Generate VAR(2) process for demonstration
T = 200  # Sample size
n = 3    # Number of variables (e.g., output, inflation, interest rate)

# VAR coefficients
A1 = [0.5 0.1 -0.1;
      0.05 0.6 0.0;
      0.1 0.15 0.7]

A2 = [0.2 0.05 0.0;
      0.0 0.2 0.0;
      0.05 0.05 0.1]

# Generate data
Y = zeros(T, n)
Y[1:2, :] = randn(2, n)

for t in 3:T
    Y[t, :] = A1 * Y[t-1, :] + A2 * Y[t-2, :] + 0.5 * randn(n)
end

println("\nData generated: T = $T observations, n = $n variables")

# =============================================================================
# 2. Basic Local Projections (Jordà 2005)
# =============================================================================
println("\n" * "-" ^ 60)
println("2. Basic Local Projections (Jordà 2005)")
println("-" ^ 60)

# Estimate LP with Newey-West HAC standard errors
horizon = 20
shock_var = 1  # Shock to first variable

lp_model = estimate_lp(Y, shock_var, horizon;
                       lags=2,
                       cov_type=:newey_west)

# Extract IRF
lp_irf_result = lp_irf(lp_model)

println("\nLP-IRF for variable 1 response to shock 1:")
println("  Horizon 0: $(round(lp_irf_result.values[1, 1, 1], digits=4))")
println("  Horizon 5: $(round(lp_irf_result.values[6, 1, 1], digits=4))")
println("  Horizon 10: $(round(lp_irf_result.values[11, 1, 1], digits=4))")
println("  Horizon 20: $(round(lp_irf_result.values[21, 1, 1], digits=4))")

println("\nConfidence intervals (95%) at horizon 10:")
println("  Lower: $(round(lp_irf_result.ci_lower[11, 1, 1], digits=4))")
println("  Upper: $(round(lp_irf_result.ci_upper[11, 1, 1], digits=4))")

# =============================================================================
# 3. Compare VAR and LP IRFs
# =============================================================================
println("\n" * "-" ^ 60)
println("3. Compare VAR and LP IRFs")
println("-" ^ 60)

# Estimate VAR for comparison
var_model = estimate_var(Y, 2)
var_irf_result = irf(var_model, horizon; method=:cholesky)

# Compare results
println("\nComparison of VAR vs LP IRFs (Variable 1 response to Shock 1):")
println("  Horizon | VAR IRF  | LP IRF")
println("  " * "-" ^ 30)
for h in [0, 5, 10, 15, 19]
    var_val = round(var_irf_result.values[h+1, 1, 1], digits=4)
    lp_val = round(lp_irf_result.values[h+1, 1, 1], digits=4)
    println("  $h       | $var_val   | $lp_val")
end

# =============================================================================
# 4. Cumulative IRFs
# =============================================================================
println("\n" * "-" ^ 60)
println("4. Cumulative IRFs")
println("-" ^ 60)

cum_irf = cumulative_irf(lp_irf_result)

println("\nCumulative IRF for variable 1:")
println("  Horizon 5: $(round(cum_irf.values[6, 1, 1], digits=4))")
println("  Horizon 10: $(round(cum_irf.values[11, 1, 1], digits=4))")
println("  Horizon 20: $(round(cum_irf.values[21, 1, 1], digits=4))")

# =============================================================================
# 5. Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("Summary: Local Projection Methods")
println("=" ^ 60)
println("""
This example demonstrated:

1. Basic LP (Jordà 2005)
   - Direct projection estimation
   - HAC standard errors (Newey-West)
   - Confidence intervals

2. VAR vs LP comparison
   - LP is more robust to misspecification
   - VAR is more efficient under correct specification

3. Cumulative IRFs
   - Useful for level responses to permanent shocks

For more details, see the documentation at:
https://chung9207.github.io/MacroEconometricModels.jl/dev/
""")

println("Example completed successfully!")
