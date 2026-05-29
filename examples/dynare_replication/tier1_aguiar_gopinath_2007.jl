# Tier 1: Aguiar & Gopinath (2007) — Emerging Market Business Cycles — Dynare Replication
# Dynare source: DSGE_mod/Aguiar_Gopinath_2007/Aguiar_Gopinath_2007.mod
#
# Emerging market business cycle model with trend and transitory TFP shocks.
# Mexico calibration (@#define mexico=1). Detrended model with exp(g) growth.
#
# 21 endogenous: c, k, y, b, q, g, l, u, z, uc, ul, c_y, i_y, invest, nx,
#                i_y_percentage, c_y_percentage, log_y, log_c, log_i, delta_y
# 2 exogenous:  eps_z (transitory TFP), eps_g (trend shock)
#
# Dynare uses `predetermined_variables k b`:
#   k, b in equations = beginning-of-period stock (our k[t-1], b[t-1])
#   k(+1) = our k[t], k(+2) = our k[t+1]
#   b(+1) = our b[t], b(+2) = our b[t+1]
#
# Dynare IRFs use `stderr 1` → unit shock. Level deviations from SS.
using MacroEconometricModels, MAT, Printf

# ── Parameters ──
# beta = 1/1.02, gamma = 0.36, b_share = 0.1, psi = 0.001
# alpha = 0.68, sigma = 2, delta = 0.05, phi = 4
# Mexico: mu_g = log(1.0066), rho_z = 0.95, rho_g = 0.01
# Derived: r_star = 0.02916638148458217, b_star = 0.06451768392329356

spec = @dsge begin
    parameters: mu_g = 0.006578315360122507, sigma_risk = 2.0, rho_g = 0.01, delta = 0.05,
                phi = 4.0, psi_debt = 0.001, alpha = 0.68, rho_z = 0.95,
                gamma_util = 0.36, beta_disc = 0.9803921568627451,
                b_star = 0.06451768392329356, r_star = 0.02916638148458217
    endogenous: c, k, y, b, q, g, l, u, z, uc, ul, c_y, i_y, invest, nx,
                i_y_pct, c_y_pct, log_y, log_c, log_i, delta_y
    exogenous: eps_z, eps_g

    # (1) Production function: y = exp(z)*k^(1-alpha)*(exp(g)*l)^alpha
    # With predetermined k: Dynare k → our k[t-1]
    y[t] = exp(z[t]) * k[t-1]^(1-alpha) * (exp(g[t]) * l[t])^alpha

    # (2) Transitory shock: z = rho_z*z(-1) + eps_z
    z[t] = rho_z * z[t-1] + eps_z[t]

    # (3) Trend shock: g = (1-rho_g)*mu_g + rho_g*g(-1) + eps_g
    g[t] = (1-rho_g)*mu_g + rho_g*g[t-1] + eps_g[t]

    # (4) Utility: u = (c^gamma*(1-l)^(1-gamma))^(1-sigma)/(1-sigma)
    u[t] = (c[t]^gamma_util * (1-l[t])^(1-gamma_util))^(1-sigma_risk) / (1-sigma_risk)

    # (5) Marginal utility of consumption: uc = gamma*u/c*(1-sigma)
    uc[t] = gamma_util * u[t] / c[t] * (1-sigma_risk)

    # (6) Disutility of labor: ul = -(1-gamma)*u/(1-l)*(1-sigma)
    ul[t] = -(1-gamma_util) * u[t] / (1-l[t]) * (1-sigma_risk)

    # (7) Resource constraint:
    # c + exp(g)*k(+1) = y + (1-delta)*k - phi/2*(exp(g)*k(+1)/k - exp(mu_g))^2*k - b + q*exp(g)*b(+1)
    # With predetermined: k(+1)→k[t], k→k[t-1], b→b[t-1], b(+1)→b[t]
    c[t] + exp(g[t])*k[t] = y[t] + (1-delta)*k[t-1] - phi/2*(exp(g[t])*k[t]/k[t-1] - exp(mu_g))^2*k[t-1] - b[t-1] + q[t]*exp(g[t])*b[t]

    # (8) Price of debt: 1/q = 1 + r_star + psi*(exp(b(+1) - b_star) - 1)
    # With predetermined: b(+1)→b[t]
    1/q[t] = 1 + r_star + psi_debt*(exp(b[t] - b_star) - 1)

    # (9) FOC capital (Euler):
    # uc*(1+phi*(exp(g)*k(+1)/k-exp(mu_g)))*exp(g)
    #   = beta*exp(g*(gamma*(1-sigma)))*uc(+1)*(1-delta+(1-alpha)*y(+1)/k(+1)
    #     - phi/2*(2*(exp(g(+1))*k(+2)/k(+1)-exp(mu_g))*(-1)*exp(g(+1))*k(+2)/k(+1)
    #            + (exp(g(+1))*k(+2)/k(+1)-exp(mu_g))^2))
    # With predetermined: k(+1)→k[t], k(+2)→k[t+1], k→k[t-1], g(+1)→g[t+1]
    # y(+1)/k(+1) → y[t+1]/k[t]
    uc[t] * (1 + phi*(exp(g[t])*k[t]/k[t-1] - exp(mu_g))) * exp(g[t]) = beta_disc * exp(g[t]*(gamma_util*(1-sigma_risk))) * uc[t+1] * (1-delta + (1-alpha)*y[t+1]/k[t] - phi/2*(2*(exp(g[t+1])*k[t+1]/k[t] - exp(mu_g))*(-1)*exp(g[t+1])*k[t+1]/k[t] + (exp(g[t+1])*k[t+1]/k[t] - exp(mu_g))^2))

    # (10) FOC labor: ul + uc*alpha*y/l = 0
    ul[t] + uc[t]*alpha*y[t]/l[t] = 0

    # (11) Euler for bonds: uc*exp(g)*q = beta*exp(g*(gamma*(1-sigma)))*uc(+1)
    uc[t]*exp(g[t])*q[t] = beta_disc*exp(g[t]*(gamma_util*(1-sigma_risk)))*uc[t+1]

    # (12) Investment: invest = exp(g)*k(+1)-(1-delta)*k+phi/2*(exp(g)*k(+1)/k-exp(mu_g))^2*k
    # With predetermined: k(+1)→k[t], k→k[t-1]
    invest[t] = exp(g[t])*k[t] - (1-delta)*k[t-1] + phi/2*(exp(g[t])*k[t]/k[t-1] - exp(mu_g))^2*k[t-1]

    # (13-17) Definitional equations
    c_y[t] = c[t] / y[t]
    i_y[t] = invest[t] / y[t]
    # nx = (b - exp(g)*q*b(+1))/y  →  with predetermined: b→b[t-1], b(+1)→b[t]
    nx[t] = (b[t-1] - exp(g[t])*q[t]*b[t]) / y[t]

    i_y_pct[t] = log(i_y[t])
    c_y_pct[t] = log(c_y[t])
    log_y[t] = log(y[t])
    log_c[t] = log(c[t])
    log_i[t] = log(invest[t])

    # (21) Growth rate of output: delta_y = log(y) - log(y(-1)) + g(-1)
    delta_y[t] = log(y[t]) - log(y[t-1]) + g[t-1]

    steady_state = begin
        q_ss = beta_disc * exp(mu_g)^(gamma_util*(1-sigma_risk)-1)
        YKbar = ((1/q_ss) - (1-delta)) / (1-alpha)
        c_y_ss = 1 + (1-exp(mu_g)-delta)*(1/YKbar) - (1-exp(mu_g)*q_ss)*0.1
        l_ss = (alpha*gamma_util) / (c_y_ss - gamma_util*c_y_ss + alpha*gamma_util)
        k_ss = (((exp(mu_g)^alpha)*(l_ss^alpha))/YKbar)^(1/alpha)
        y_ss = k_ss^(1-alpha) * (l_ss*exp(mu_g))^alpha
        c_ss = c_y_ss * y_ss
        invest_ss = (exp(mu_g) - 1 + delta) * k_ss
        nx_ss = (y_ss - c_ss - invest_ss) / y_ss

        b_ss = 0.1 * y_ss  # b_star
        z_ss = 0.0
        g_ss = mu_g
        u_ss = (c_ss^gamma_util * (1-l_ss)^(1-gamma_util))^(1-sigma_risk) / (1-sigma_risk)
        uc_ss = gamma_util * u_ss / c_ss * (1-sigma_risk)
        ul_ss = -(1-gamma_util) * u_ss / (1-l_ss) * (1-sigma_risk)
        i_y_ss = invest_ss / y_ss
        i_y_pct_ss = log(i_y_ss)
        c_y_pct_ss = log(c_y_ss)
        log_y_ss = log(y_ss)
        log_c_ss = log(c_ss)
        log_i_ss = log(invest_ss)
        delta_y_ss = mu_g

        # Order: c, k, y, b, q, g, l, u, z, uc, ul, c_y, i_y, invest, nx,
        #        i_y_pct, c_y_pct, log_y, log_c, log_i, delta_y
        [c_ss, k_ss, y_ss, b_ss, q_ss, g_ss, l_ss, u_ss, z_ss, uc_ss, ul_ss,
         c_y_ss, i_y_ss, invest_ss, nx_ss, i_y_pct_ss, c_y_pct_ss,
         log_y_ss, log_c_ss, log_i_ss, delta_y_ss]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 40)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  Aguiar & Gopinath (2007) — Mexico Calibration")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "aguiar_gopinath_2007.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

let
    # ── Steady State comparison ──
    # Dynare endo order (21 vars):
    # c(1), k(2), y(3), b(4), q(5), g(6), l(7), u(8), z(9), uc(10), ul(11),
    # c_y(12), i_y(13), invest(14), nx(15), i_y_percentage(16), c_y_percentage(17),
    # log_y(18), log_c(19), log_i(20), delta_y(21)
    # Our endo order: same
    println("\n=== Steady State ===")
    ss_pass = true
    names_map = [("c",1,1), ("k",2,2), ("y",3,3), ("b",4,4), ("q",5,5), ("g",6,6),
                 ("l",7,7), ("u",8,8), ("z",9,9), ("uc",10,10), ("ul",11,11),
                 ("c_y",12,12), ("i_y",13,13), ("invest",14,14), ("nx",15,15),
                 ("i_y_pct",16,16), ("c_y_pct",17,17), ("log_y",18,18),
                 ("log_c",19,19), ("log_i",20,20), ("delta_y",21,21)]
    for (name, j_idx, d_idx) in names_map
        diff = abs(ss[j_idx] - d_ss[d_idx])
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-15s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, ss[j_idx], d_ss[d_idx], diff, ok ? "PASS" : "FAIL")
    end

    # ── IRF comparison ──
    # Dynare IRFs are level deviations from SS.
    # Variable mapping from stoch_simul outputs.
    # First stoch_simul: nx, c_y_percentage, i_y_percentage
    # Second stoch_simul: log_y, log_c, log_i, g
    # All share eps_z and eps_g shocks.
    println("\n=== IRFs (level deviations from SS) ===")
    irf_pass = true

    # Map: (dynare_irf_name, our_var_idx)
    # Only compare variables from AG2007 stoch_simul lists:
    #   First call:  nx, c_y_percentage, i_y_percentage
    #   Second call: log_y, log_c, log_i, g
    # Other IRFs (z, r, etc.) in the .mat are contamination from prior model runs
    var_map = [
        ("nx", 15), ("c_y_percentage", 17), ("i_y_percentage", 16),
        ("log_y", 18), ("log_c", 19), ("log_i", 20), ("g", 6),
    ]
    shock_map = [("eps_z",1), ("eps_g",2)]

    for (sn, si) in shock_map
        for (vn, vi) in var_map
            vi == -1 && continue
            d_key = vn * "_" * sn
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 40)
            j_vals = [ir.values[h, vi, si] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-30s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
