# Tier 1: RBC with Capital Stock Shock — Dynare Replication
# Dynare source: DSGE_mod/RBC_capitalstock_shock/RBC_capitalstock_shock.mod
#
# Standard RBC model with an additional shock to the capital stock.
# Uses exp() substitution for log-linearization: x = log(X), so exp(x) = X.
# IRFs are therefore in percentage deviations from SS (log-deviations).
#
# 6 endogenous (in log space): y, c, k, l, z, invest
# 2 exogenous: eps_z (TFP), eps_cap (capital destruction)
#
# Key feature: capital stock at t is affected by a contemporaneous shock:
#   exp(k) = exp(-eps_cap) * (exp(invest(-1)) + (1-delta)*exp(k(-1)))
#
# Dynare uses `var eps_z = 1; var eps_cap = 1;` → unit shock.
using MacroEconometricModels, MAT, Printf

# ── Parameters ──
# Calibration from .mod file
const _alpha = 0.33
const _i_y   = 0.25
const _k_y   = 10.4
const _rho   = 0.97

# Derived parameters (from Dynare's steady_state_model)
const _delta  = _i_y / _k_y                            # 0.024038461538...
const _beta   = 1.0 / (_alpha / _k_y + (1 - _delta))   # 0.990238095238...
const _l_ss   = 0.33
const _k_ss   = ((1/_beta - (1 - _delta)) / _alpha)^(1/(_alpha - 1)) * _l_ss
const _i_ss   = _delta * _k_ss
const _y_ss   = _k_ss^_alpha * _l_ss^(1 - _alpha)
const _c_ss   = _y_ss - _i_ss
const _psi    = (1 - _alpha) * (_k_ss / _l_ss)^_alpha * (1 - _l_ss) / _c_ss

spec = @dsge begin
    parameters: alpha = 0.33, rho_z = 0.97,
                beta_disc = 0.9923664122137404, delta = 0.024038461538461536,
                psi_d = 1.8137373737373732
    endogenous: y, c, k, l, z, invest
    exogenous: eps_z, eps_cap

    # (1) Labor FOC: psi*exp(c)/(1-exp(l)) = (1-alpha)*exp(z)*(exp(k)/exp(l))^alpha
    psi_d * exp(c[t]) / (1 - exp(l[t])) = (1 - alpha) * exp(z[t]) * (exp(k[t]) / exp(l[t]))^alpha

    # (2) Euler equation
    1 / exp(c[t]) = beta_disc / exp(c[t+1]) * (alpha * exp(z[t+1]) * (exp(k[t+1]) / exp(l[t+1]))^(alpha - 1) + (1 - delta))

    # (3) Capital accumulation with capital stock shock
    exp(k[t]) = exp(-eps_cap[t]) * (exp(invest[t-1]) + (1 - delta) * exp(k[t-1]))

    # (4) Production function
    exp(y[t]) = exp(z[t]) * exp(k[t])^alpha * exp(l[t])^(1 - alpha)

    # (5) TFP process
    z[t] = rho_z * z[t-1] + eps_z[t]

    # (6) Market clearing
    exp(invest[t]) = exp(y[t]) - exp(c[t])

    steady_state = begin
        alpha_v = alpha
        delta_v = delta
        beta_v = beta_disc
        l_ss_v = 0.33
        k_ss_v = ((1/beta_v - (1 - delta_v)) / alpha_v)^(1/(alpha_v - 1)) * l_ss_v
        i_ss_v = delta_v * k_ss_v
        y_ss_v = k_ss_v^alpha_v * l_ss_v^(1 - alpha_v)
        c_ss_v = y_ss_v - i_ss_v
        # Return log values matching endogenous order: y, c, k, l, z, invest
        [log(y_ss_v), log(c_ss_v), log(k_ss_v), log(l_ss_v), 0.0, log(i_ss_v)]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 20)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  RBC with Capital Stock Shock")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "rbc_capitalstock.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

let
    # ── Steady State comparison ──
    # Dynare endo order: y(1), c(2), k(3), l(4), z(5), invest(6)
    # Our endo order   : y(1), c(2), k(3), l(4), z(5), invest(6)  (same)
    # Dynare stores log-level SS because of exp() substitution
    println("\n=== Steady State ===")
    ss_pass = true
    for (name, j_idx, d_idx) in [("y",1,1), ("c",2,2), ("k",3,3), ("l",4,4), ("z",5,5), ("invest",6,6)]
        diff = abs(ss[j_idx] - d_ss[d_idx])
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-10s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, ss[j_idx], d_ss[d_idx], diff, ok ? "PASS" : "FAIL")
    end

    # ── IRF comparison ──
    # Model is log-linearized (exp() substitution) → IRFs are already % deviations
    println("\n=== IRFs ===")
    irf_pass = true

    # Variables: y=1, c=2, k=3, l=4, z=5, invest=6
    # Shocks: eps_z=1, eps_cap=2
    var_map = [("y",1), ("c",2), ("k",3), ("l",4), ("z",5), ("invest",6)]
    shock_map = [("eps_z",1), ("eps_cap",2)]

    for (sn, si) in shock_map
        for (vn, vi) in var_map
            d_key = vn * "_" * sn
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 20)
            j_vals = [ir.values[h, vi, si] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-20s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
