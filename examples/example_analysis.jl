using MacroEconometricModels
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Turing
using Random
using Dates
using StatsAPI

# ==========================================
# 1. Setup & Data Loading
# ==========================================
println("Starting MacroEconometricModels Example Analysis...")

# Create Synthetic 'Real' Data
# Variables: GDP Growth, Inflation, Interest Rate
Random.seed!(123)
T = 100
dates = Date(2000, 1, 1) + Month.(0:T-1)
Y = zeros(T, 3)
# VAR(1) Process
const A = [0.8 0.1 -0.1; 0.05 0.7 0.0; 0.1 0.2 0.8]
const C = [0.5; 0.02; 1.0]
const Sigma = [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0] * 0.1
chol_L = cholesky(Sigma).L

Y[1, :] = C
for t in 2:T
    u = chol_L * randn(3)
    Y[t, :] = C + A * Y[t-1, :] + u
end

df = DataFrame(Date=dates, GDP=Y[:, 1], CPI=Y[:, 2], Rate=Y[:, 3])
CSV.write("examples/macro_data.csv", df)
println("Created synthetic macro data at examples/macro_data.csv")

# Load Data
data = Matrix(df[:, [:GDP, :CPI, :Rate]])
var_names = ["GDP", "CPI", "Rate"]
n_vars = 3
lags = 2
horizon = 12

# ==========================================
# 2. Frequentist Analysis
# ==========================================
println("\n--- Frequentist Estimation (OLS) ---")
model = fit(VARModel, data, lags)
println("Model Estimated. AIC: $(StatsAPI.aic(model))")

# Identification Schemes
println("Running Structural Analysis...")

# A. Cholesky
println(">> Cholesky Identification")
irf_chol = irf(model, horizon; method=:cholesky)
fevd_chol = fevd(model, horizon; method=:cholesky)
println("   IRF & FEVD computed.")

# B. Sign Restrictions
# Restriction: Shock 1 (Demand?) -> GDP (+), CPI (+)
println(">> Sign Restrictions")
check_sign(irf_val) = irf_val[1, 1, 1] > 0 && irf_val[1, 2, 1] > 0
irf_sign = irf(model, horizon; method=:sign, check_func=check_sign)
fevd_sign = fevd(model, horizon; method=:sign, check_func=check_sign)
println("   IRF & FEVD computed.")

# C. Narrative Restrictions
# Period 10: Shock 1 was positive.
println(">> Narrative Restrictions")
narrative_chk(shocks) = shocks[10, 1] > 0
# Re-using sign check
irf_narr = irf(model, horizon; method=:narrative, check_func=check_sign, narrative_check=narrative_chk)
# Note: FEVD for Narrative uses specific identified Q.
fevd_narr = fevd(model, horizon; method=:narrative, check_func=check_sign, narrative_check=narrative_chk)
println("   IRF & FEVD computed.")

# D. Long Run Restrictions (Blanchard-Quah)
println(">> Long-Run Identification")
irf_lr = irf(model, horizon; method=:long_run)
fevd_lr = fevd(model, horizon; method=:long_run)
println("   IRF & FEVD computed.")


# ==========================================
# 3. Bayesian Analysis
# ==========================================
println("\n--- Bayesian Estimation (Turing.jl) ---")
# Using Minnesota Prior
hyper = MinnesotaHyperparameters(0.5, 2.0, 1.0, 1.0, 1.0) # Loose prior
println("Estimating BVAR with Minnesota Prior...")
# n_samples small for example speed
chain = estimate_bvar(data, lags; n_samples=500, n_adapts=200, prior=:minnesota, hyper=hyper)
println("Bayesian Estimation Complete.")

# NEW: BGR (2010) Prior Optimization
println("\n>> Optimizing Priors (BGR 2010)...")
# Automatically find optimal overall tightness (tau)
best_hyper = optimize_hyperparameters(data, lags; grid_size=10)
println("   Optimal Tau found: $(best_hyper.tau)")

println("   Estimating BVAR with Optimized Priors...")
chain_opt = estimate_bvar(data, lags; n_samples=500, n_adapts=200, prior=:minnesota, hyper=best_hyper)
println("   Authorized BVAR Estimation Complete.")

# Bayesian Structural Analysis
# Note: Narrative not supported for Bayesian in this script.
println("Running Bayesian Structural Analysis...")

# A. Cholesky
println(">> Bayesian Cholesky")
birf_chol = irf(chain, lags, n_vars, horizon; method=:cholesky)
bfevd_chol = fevd(chain, lags, n_vars, horizon; method=:cholesky)
println("   Bayesian IRF/FEVD computed. Median Impact Var 1 from Shock 1: $(birf_chol.quantiles[1, 1, 1, 2])")

# B. Sign
println(">> Bayesian Sign Restrictions")
birf_sign = irf(chain, lags, n_vars, horizon; method=:sign, check_func=check_sign)
bfevd_sign = fevd(chain, lags, n_vars, horizon; method=:sign, check_func=check_sign)
println("   Bayesian Sign IRF/FEVD computed.")

# C. Long Run
println(">> Bayesian Long-Run Identification")
birf_lr = irf(chain, lags, n_vars, horizon; method=:long_run)
bfevd_lr = fevd(chain, lags, n_vars, horizon; method=:long_run)
println("   Bayesian Long-Run IRF/FEVD computed.")

# D. Narrative (New)
println(">> Bayesian Narrative Identification")

# Reusing narrative check (Period 10, Shock 1 > 0)
birf_narr = irf(chain, lags, n_vars, horizon; method=:narrative, check_func=check_sign, narrative_check=narrative_chk, data=data)
bfevd_narr = fevd(chain, lags, n_vars, horizon; method=:narrative, check_func=check_sign, narrative_check=narrative_chk, data=data)
println("   Bayesian Narrative IRF/FEVD computed. Median Impact Var 1: $(birf_narr.quantiles[1,1,1,2])")

println("\nAll Examples Completed Successfully.")
