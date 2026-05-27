# Tier 1: Fernandez-Villaverde, Rubio-Ramirez, Sargent & Watson (2007)
# "ABCs (and Ds) of Understanding VARs", AER, 97(3), 1021-1026
# Permanent Income Model — ABCD Test
#
# Dynare source: DSGE_mod/FV_et_al_2007/FV_et_al_2007_ABCD.mod
# Mat ref: dynare_results/fv_2007.mat
#
# 3 endogenous: y, c, y_m_c
# 1 exogenous:  w
# All variables in LEVELS (linear model, SS = 0).
using MacroEconometricModels, MAT, Printf

# Parameters
const _R = 1.2
const _sigma_w = 1.0

spec = @dsge begin
    parameters: R_gross = 1.2, sigma_w = 1.0
    endogenous: y, c, y_m_c
    exogenous: w

    # (1) Consumption follows a random walk plus permanent income innovation
    c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]

    # (2) Income minus consumption (savings)
    y_m_c[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]

    # (3) Identity: y - c = y_m_c
    y_m_c[t] = y[t] - c[t]

    steady_state = begin
        [0.0, 0.0, 0.0]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 20)
ss = sol.spec.steady_state

println("=" ^ 60)
println("  FV et al. (2007) — Permanent Income ABCD")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "fv_2007.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
irfs = data["irfs"]

let
    # ── Steady State ──
    # Dynare endo order: y(1), c(2), y_m_c(3)
    println("\n=== Steady State ===")
    ss_pass = true
    for (name, j_idx, d_idx) in [("y", 1, 1), ("c", 2, 2), ("y_m_c", 3, 3)]
        j_val = ss[j_idx]
        d_val = d_ss[d_idx]
        diff = abs(j_val - d_val)
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-10s  Julia=%12.8f  Dynare=%12.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── IRFs ──
    # Dynare keys: y_w, c_w, y_m_c_w (for shock w)
    println("\n=== IRFs ===")
    irf_pass = true
    irf_map = [("y_w", 1), ("c_w", 2), ("y_m_c_w", 3)]
    for (d_key, j_idx) in irf_map
        haskey(irfs, d_key) || continue
        d_vals = vec(irfs[d_key])
        H = min(length(d_vals), 20)
        j_vals = [ir.values[h, j_idx, 1] for h in 1:H]
        max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
        ok = max_diff < 1e-4
        irf_pass = irf_pass && ok
        @printf("  %-12s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
    end

    println("\n  Overall: SS=", ss_pass ? "PASS" : "FAIL",
            ", IRF=", irf_pass ? "PASS" : "FAIL")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
