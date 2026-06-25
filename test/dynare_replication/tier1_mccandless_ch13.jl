# Tier 1: McCandless (2008), Chapter 13 — Open Economy RBC — Dynare Replication
# Dynare source: DSGE_mod/McCandless_2008/McCandless_2008_Chapter_13.mod
#
# Open economy model with cash-in-advance, capital adjustment costs, and
# foreign bonds. Model has 2-period-ahead leads p(+2) and c(+2) which
# require auxiliary variables to reduce to 1-period leads.
#
# 16 endogenous (14 core + 2 auxiliary):
#   w, r, c, k, h, m, p, pstar, g_var, lam, b, rf, e_var, x, fp, fc
# where fp[t]=p[t+1] and fc[t]=c[t+1] are lead-reduction auxiliaries.
#
# 3 exogenous: eps_lambda (TFP), eps_g (money growth), eps_pstar (foreign price)
# All shocks have stderr 1 in Dynare; sigma coefficients embedded in equations.
# Shock processes are LINEAR AR(1) in levels (not log-AR).
#
# No predetermined_variables in this model — standard timing convention.
using MacroEconometricModels, MAT, Printf

spec = @dsge begin
    parameters: kappa = 0.5, beta_disc = 0.99, delta = 0.025, theta = 0.36,
                rstar = 0.03, a_risk = 0.01, B_param = -2.58,
                gamma_lambda = 0.95, gamma_g = 0.95, gamma_pstar = 0.95,
                sigma_lambda = 0.01, sigma_g = 0.01, sigma_pstar = 0.01
    endogenous: w, r, c, k, h, m, p, pstar, g_var, lam, b, rf, e_var, x, fp, fc
    exogenous: eps_lambda, eps_g, eps_pstar

    # Auxiliary variable definitions: fp[t] = p[t+1], fc[t] = c[t+1]
    fp[t] = p[t+1]
    fc[t] = c[t+1]

    # (1) Euler equation (bonds):
    # Original: e/(p(+1)*c(+1)) = beta*e(+1)*(1+rf)/(p(+2)*c(+2))
    # p(+1)*c(+1) = fp[t]*fc[t], p(+2)*c(+2) = fp[t+1]*fc[t+1]
    e_var[t] / (fp[t] * fc[t]) = beta_disc * e_var[t+1] * (1 + rf[t]) / (fp[t+1] * fc[t+1])

    # (2) FOC capital:
    # Original: p/(p(+1)*c(+1))*(1+kappa*(k-k(-1))) = beta*p(+1)/(p(+2)*c(+2))*(r(+1)+1-delta+kappa*(k(+1)-k))
    p[t] / (fp[t] * fc[t]) * (1 + kappa * (k[t] - k[t-1])) = beta_disc * fp[t] / (fp[t+1] * fc[t+1]) * (r[t+1] + (1-delta) + kappa * (k[t+1] - k[t]))

    # (3) FOC hours: B/w + beta*p/(p(+1)*c(+1)) = 0
    B_param / w[t] + beta_disc * p[t] / (fp[t] * fc[t]) = 0

    # (4) Cash-in-advance: p*c = m
    p[t] * c[t] = m[t]

    # (5) Budget constraint:
    # m/p + e*b/p + k + kappa/2*(k-k(-1))^2 = w*h + r*k(-1) + (1-delta)*k(-1) + e*(1+rf(-1))*b(-1)/p
    m[t]/p[t] + e_var[t]*b[t]/p[t] + k[t] + kappa/2*(k[t]-k[t-1])^2 = w[t]*h[t] + r[t]*k[t-1] + (1-delta)*k[t-1] + e_var[t]*(1+rf[t-1])*b[t-1]/p[t]

    # (6) Firm FOC labor: w = (1-theta)*lambda*k(-1)^theta*h^(-theta)
    w[t] = (1-theta) * lam[t] * k[t-1]^theta * h[t]^(-theta)

    # (7) Firm FOC capital: r = theta*lambda*k(-1)^(theta-1)*h^(1-theta)
    r[t] = theta * lam[t] * k[t-1]^(theta-1) * h[t]^(1-theta)

    # (8) Foreign bonds evolution: b = (1+rf(-1))*b(-1) + pstar*x
    b[t] = (1 + rf[t-1]) * b[t-1] + pstar[t] * x[t]

    # (9) Foreign interest rate: rf = rstar - a*b/pstar
    rf[t] = rstar - a_risk * b[t] / pstar[t]

    # (10) Exchange rate definition: e = p/pstar
    e_var[t] = p[t] / pstar[t]

    # (11) Money growth: m = g*m(-1)
    m[t] = g_var[t] * m[t-1]

    # (12) LOM TFP (linear AR(1) in levels):
    # lambda = (1-gamma_lambda) + gamma_lambda*lambda(-1) + sigma_lambda*eps_lambda
    lam[t] = (1-gamma_lambda) + gamma_lambda*lam[t-1] + sigma_lambda*eps_lambda[t]

    # (13) LOM money growth (linear AR(1)):
    # g = (1-gamma_g)*1 + gamma_g*g(-1) + sigma_g*eps_g
    g_var[t] = (1-gamma_g) + gamma_g*g_var[t-1] + sigma_g*eps_g[t]

    # (14) LOM foreign price (linear AR(1)):
    # pstar = (1-gamma_pstar)*1 + gamma_pstar*pstar(-1) + sigma_pstar*eps_pstar
    pstar[t] = (1-gamma_pstar) + gamma_pstar*pstar[t-1] + sigma_pstar*eps_pstar[t]

    steady_state = begin
        r_ss = 1/beta_disc - (1-delta)
        rf_ss = 1/beta_disc - 1
        b_ss = (rstar + 1 - 1/beta_disc) / a_risk
        x_ss = ((1-beta_disc)^2 - (1-beta_disc)*beta_disc*rstar) / (a_risk*beta_disc^2)
        w_ss = (1-theta)*(theta/r_ss)^(theta/(1-theta))
        pstar_ss = 1.0
        pistar_ss = 1.0
        c_ss = beta_disc*w_ss/(-B_param*pistar_ss)
        m_pss = c_ss
        k_ss = theta*(m_pss - rf_ss*b_ss)/(r_ss - theta*delta)
        h_ss = r_ss*(1-theta)/(w_ss*theta)*k_ss
        lam_ss = 1.0
        g_ss = 1.0
        p_ss = 1.0
        m_ss = m_pss * p_ss
        e_ss = 1.0
        fp_ss = p_ss   # p(+1) at SS = p_ss
        fc_ss = c_ss   # c(+1) at SS = c_ss
        # Order: w, r, c, k, h, m, p, pstar, g_var, lam, b, rf, e_var, x, fp, fc
        [w_ss, r_ss, c_ss, k_ss, h_ss, m_ss, p_ss, pstar_ss, g_ss, lam_ss, b_ss, rf_ss, e_ss, x_ss, fp_ss, fc_ss]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 100)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  McCandless (2008), Chapter 13 — Open Economy")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "mccandless_ch13.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

let
    # ── Steady State comparison ──
    # Dynare endo order: w(1), r(2), c(3), k(4), h(5), m(6), p(7), pstar(8),
    #   g(9), lambda(10), b(11), rf(12), e(13), x(14), AUX_ENDO_LEAD_23(15), AUX_ENDO_LEAD_45(16)
    # Our endo order: w(1), r(2), c(3), k(4), h(5), m(6), p(7), pstar(8),
    #   g_var(9), lam(10), b(11), rf(12), e_var(13), x(14), fp(15), fc(16)
    println("\n=== Steady State ===")
    ss_pass = true
    core_pairs = [("w",1,1), ("r",2,2), ("c",3,3), ("k",4,4), ("h",5,5),
                  ("m",6,6), ("p",7,7), ("pstar",8,8), ("g",9,9), ("lambda",10,10),
                  ("b",11,11), ("rf",12,12), ("e",13,13), ("x",14,14)]
    for (name, j_idx, d_idx) in core_pairs
        diff = abs(ss[j_idx] - d_ss[d_idx])
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-10s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, ss[j_idx], d_ss[d_idx], diff, ok ? "PASS" : "FAIL")
    end
    # Aux variables: our fp_ss = p_ss, fc_ss = c_ss
    # Dynare AUX_ENDO_LEAD values at SS
    for (name, j_idx, d_idx) in [("AUX15",15,15), ("AUX16",16,16)]
        @printf("  %-10s  Julia=%14.8f  Dynare=%14.8f  (info)\n",
                name, ss[j_idx], d_ss[d_idx])
    end

    # ── IRF comparison ──
    # Dynare IRFs are level deviations from SS. Both ours and Dynare are the same.
    println("\n=== IRFs (level deviations from SS) ===")
    irf_pass = true

    # Variable mapping: (dynare_name, our_idx)
    # Only compare variables from Dynare's stoch_simul list: k c w b m p e rf r
    # Other IRFs (h, y, etc.) in the .mat are contamination from prior model runs
    var_map = [("w",1), ("r",2), ("c",3), ("k",4), ("m",6), ("p",7),
               ("b",11), ("rf",12), ("e",13)]
    shock_map = [("eps_lambda",1), ("eps_g",2), ("eps_pstar",3)]

    for (sn, si) in shock_map
        for (vn, vi) in var_map
            d_key = vn * "_" * sn
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 100)
            j_vals = [ir.values[h, vi, si] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-25s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    # Also check y (which is not a separate variable in our model, but check if present)
    # y_eps_lambda etc. - we don't have y as endogenous but could check via w*h/(1-theta) or similar

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
