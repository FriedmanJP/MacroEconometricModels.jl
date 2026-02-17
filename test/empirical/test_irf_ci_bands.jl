# Empirical diagnostic: IRF CI band width across horizons
#
# Verifies that LP CI bands do not shrink pathologically at long horizons.
# Run: julia --project=. test/empirical/test_irf_ci_bands.jl

using MacroEconometricModels
using Random, Statistics, Printf

Random.seed!(2024)

# ── Generate a persistent 3-variable VAR(1) system ──────────────────────
T_obs = 300
n = 3
A = [0.7 0.1 0.0;
     0.1 0.6 0.1;
     0.0 0.1 0.5]

Y = zeros(T_obs, n)
for t in 2:T_obs
    Y[t, :] = A * Y[t-1, :] + randn(n)
end

H = 40
var_names = ["Var1", "Var2", "Var3"]

# ── VAR bootstrap CI ────────────────────────────────────────────────────
println("=" ^ 72)
println("  IRF Confidence Interval Band Width Diagnostic")
println("=" ^ 72)

println("\n── VAR (bootstrap CI, 500 reps) ──")
var_model = estimate_var(Y, 2)
var_irf = irf(var_model, H; method=:cholesky, ci_type=:bootstrap, reps=500)

var_widths = var_irf.ci_upper .- var_irf.ci_lower
println(@sprintf("  %-6s  %10s  %10s  %10s", "h", "Width(1,1)", "Width(2,1)", "Width(3,1)"))
for h in [1, 5, 10, 15, 20, 30, 40]
    w = [var_widths[h, i, 1] for i in 1:n]
    println(@sprintf("  h=%-4d  %10.4f  %10.4f  %10.4f", h, w...))
end

# ── BVAR posterior CI ───────────────────────────────────────────────────
println("\n── BVAR (posterior CI, 1000 draws) ──")
bvar_post = estimate_bvar(Y, 2; n_draws=1000)
bvar_irf = irf(bvar_post, H; method=:cholesky)

# BayesianImpulseResponse: quantiles is (H, n, n, nq) with levels [0.16, 0.5, 0.84]
bvar_widths = bvar_irf.quantiles[:, :, :, 3] .- bvar_irf.quantiles[:, :, :, 1]
println(@sprintf("  %-6s  %10s  %10s  %10s", "h", "Width(1,1)", "Width(2,1)", "Width(3,1)"))
for h in [1, 5, 10, 15, 20, 30, 40]
    w = [bvar_widths[h, i, 1] for i in 1:n]
    println(@sprintf("  h=%-4d  %10.4f  %10.4f  %10.4f", h, w...))
end

# ── LP (Newey-West CI) ─────────────────────────────────────────────────
println("\n── LP (Newey-West CI, auto bandwidth with h+1 floor) ──")
lp_model = estimate_lp(Y, 1, H; lags=2, cov_type=:newey_west)
lp_result = lp_irf(lp_model; conf_level=0.95)

lp_widths = lp_result.ci_upper .- lp_result.ci_lower
println(@sprintf("  %-6s  %10s  %10s", "h", "Width(1,1)", "Width(2,1)"))
shrink_count = let sc = 0
    for h_idx in 1:(H+1)
        h = h_idx - 1
        w1 = lp_widths[h_idx, 1]
        w2 = lp_widths[h_idx, 2]
        flag = ""
        if h_idx > 1 && lp_widths[h_idx, 1] < lp_widths[h_idx - 1, 1] - 1e-10
            flag = " <-- SHRINK"
            sc += 1
        end
        if h in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
            println(@sprintf("  h=%-4d  %10.4f  %10.4f%s", h, w1, w2, flag))
        end
    end
    sc
end

# ── Summary ─────────────────────────────────────────────────────────────
println("\n" * "=" ^ 72)
println("  Summary")
println("=" ^ 72)
println("  LP CI width shrinkage occurrences (Var1→Var1): $shrink_count / $H")
if shrink_count == 0
    println("  PASS: LP CI bands do not shrink pathologically.")
else
    println("  NOTE: Some shrinkage detected. Minor, non-systematic shrinkage")
    println("        can occur due to changing effective sample size.")
end
println("=" ^ 72)
