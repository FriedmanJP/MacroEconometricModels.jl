# Tier 1: Collard (2001), Example 1 — Dynare Replication
# Dynare source: DSGE_mod/Collard_2001/Collard_2001_example1.mod
#
# Simple RBC with two correlated technology shocks.
# 6 endogenous: y, c, k, a, h, b
# 2 exogenous : e, u  (stderr 0.009 each, cross-corr phi=0.1)
#
# Shock covariance Σ = [0.009² φ·0.009²; φ·0.009² 0.009²]
# Cholesky: L = [0.009 0; 0.0009 0.008954886933959579]
# We define orthogonal shocks e1, e2 and mix via Cholesky in the equations.
using MacroEconometricModels, MAT, Printf

# ── Parameters ──
const _alpha  = 0.36
const _rho    = 0.95
const _tau    = 0.025
const _beta   = 0.99
const _delta  = 0.025
const _psi    = 0       # NOTE: psi=0 in Collard
const _theta  = 2.95
const _phi    = 0.1

# Cholesky of shock covariance
const _L11 = 0.009
const _L21 = 0.0009                 # phi * 0.009
const _L22 = 0.008954886933959579   # 0.009 * sqrt(1 - phi^2)

# ── Model ──
# Dynare equations (level variables, a and b are log-TFP processes):
#   (1) c*theta*h^(1+psi) = (1-alpha)*y
#   (2) k = beta*((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k)
#   (3) y = exp(a)*k(-1)^alpha*h^(1-alpha)
#   (4) k = exp(b)*(y-c)+(1-delta)*k(-1)
#   (5) a = rho*a(-1)+tau*b(-1) + e
#   (6) b = tau*a(-1)+rho*b(-1) + u
# With Cholesky orthogonalization:
#   (5') a = rho*a(-1)+tau*b(-1) + L11*e1
#   (6') b = tau*a(-1)+rho*b(-1) + L21*e1 + L22*e2

spec = @dsge begin
    parameters: alpha = 0.36, rho_a = 0.95, tau = 0.025, beta_disc = 0.99,
                delta = 0.025, psi = 0.0, theta = 2.95,
                L11 = 0.009, L21 = 0.0009, L22 = 0.008954886933959579
    endogenous: y, c, k, a, h, b
    exogenous: e1, e2

    # (1) Intratemporal FOC
    c[t] * theta * h[t]^(1 + psi) = (1 - alpha) * y[t]

    # (2) Euler equation (capital)
    k[t] = beta_disc * ((exp(b[t]) * c[t]) / (exp(b[t+1]) * c[t+1])) * (exp(b[t+1]) * alpha * y[t+1] + (1 - delta) * k[t])

    # (3) Production function
    y[t] = exp(a[t]) * k[t-1]^alpha * h[t]^(1 - alpha)

    # (4) Resource constraint
    k[t] = exp(b[t]) * (y[t] - c[t]) + (1 - delta) * k[t-1]

    # (5) Technology shock a (orthogonalized)
    a[t] = rho_a * a[t-1] + tau * b[t-1] + L11 * e1[t]

    # (6) Technology shock b (orthogonalized)
    b[t] = tau * a[t-1] + rho_a * b[t-1] + L21 * e1[t] + L22 * e2[t]

    steady_state = begin
        a_ss = 0.0
        b_ss = 0.0
        # From initval (verified analytically)
        y_ss = 1.08068253095672
        c_ss = 0.80359242014163
        h_ss = 0.29175631001732
        k_ss = 11.08360443260358
        [y_ss, c_ss, k_ss, a_ss, h_ss, b_ss]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 40)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  Collard (2001), Example 1")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "collard_2001.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

let
    # ── Steady State comparison ──
    # Dynare endo order: y, c, k, a, h, b
    # Our endo order  : y, c, k, a, h, b  (same)
    println("\n=== Steady State ===")
    ss_pass = true
    for (name, j_idx, d_idx) in [("y",1,1), ("c",2,2), ("k",3,3), ("a",4,4), ("h",5,5), ("b",6,6)]
        d_val = d_ss[d_idx]
        diff  = abs(ss[j_idx] - d_val)
        ok    = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-6s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, ss[j_idx], d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── IRF comparison ──
    # Dynare IRFs are level deviations from SS, response to 1-unit orthogonalized shock.
    # Our shocks e1, e2 are unit-variance; Cholesky mixing is in the equations.
    # Dynare shock "e" = our e1 (first Cholesky column), "u" = our e2 (second column).
    # So: y_e → ir.values[:, 1, 1], y_u → ir.values[:, 1, 2], etc.
    println("\n=== IRFs (level deviations) ===")
    irf_pass = true

    var_defs = [("y",1), ("c",2), ("k",3), ("a",4), ("h",5), ("b",6)]
    shock_defs = [("e",1), ("u",2)]  # Dynare shock name → our shock index

    for (sn, si) in shock_defs
        for (vn, vi) in var_defs
            d_key = vn * "_" * sn
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 40)
            j_vals = [ir.values[h, vi, si] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-10s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
