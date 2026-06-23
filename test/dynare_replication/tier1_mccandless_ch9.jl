# Tier 1: McCandless (2008), Chapter 9 — Money-in-Utility RBC — Dynare Replication
# Dynare source: DSGE_mod/McCandless_2008/McCandless_2008_Chapter_9.mod
#
# Money-in-utility function model. Money growth and TFP follow log-AR(1) processes.
# Superneutrality: money growth shock has zero effect on real variables.
#
# 10 endogenous: w, r, c, k, h, m, p, log_g, log_lam, y
# 2 exogenous:  eps_g (money growth), eps_lambda (TFP)
#
# Dynare uses `predetermined_variables k` → k in equations is beginning-of-period stock.
# Translation: Dynare k → our k[t-1], Dynare k(+1) → our k[t].
#
# Dynare shocks: stderr 0.01 for each (separate stoch_simul calls in .mod).
# We embed 0.01 coefficient in the shock equations.
using MacroEconometricModels, MAT, Printf

# ── Pre-computed Parameters ──
# From .mod: beta=0.99, delta=0.025, theta=0.36, A=1.72, h_0=0.583,
#   gamma=0.95, pi=0.48, g_bar=1, D=0.01
# B = A*log(1-h_0)/h_0 = -2.5804987621875424
# r_ss = 1/beta - 1 + delta = 0.03510101010101017
# w_ss = (1-theta)*(r/theta)^(theta/(theta-1)) = 2.3705976394178108
# c_ss = -w/B = 0.9186587004630864
# k_ss = c / ((r*(1-theta)/(w*theta))^(1-theta) - delta) = 12.670664119390215
# h_ss = 0.3335328530913403
# y_ss = 1.2354253034478417
# m_ss = p*D*g*c/(g-beta) = 0.9186587004630856

spec = @dsge begin
    parameters: beta_disc = 0.99, delta = 0.025, theta = 0.36,
                B_param = -2.5804987621875424,
                gamma_ar = 0.95, pi_ar = 0.48, D_coef = 0.01
    endogenous: w, r, c, k, h, m, p, log_g, log_lam, y
    exogenous: eps_g, eps_lambda

    # (9.1) Budget constraint:
    # Dynare: c + k(+1) + m/p = w*h + r*k + (1-delta)*k + m(-1)/p + (g-1)*m(-1)/p
    # Our timing: k(+1)→k[t], k→k[t-1], m→m[t], m(-1)→m[t-1]
    c[t] + k[t] + m[t]/p[t] = w[t]*h[t] + r[t]*k[t-1] + (1-delta)*k[t-1] + m[t-1]/p[t] + (exp(log_g[t])-1)*m[t-1]/p[t]

    # (9.2) Euler equation money:
    # 1/c = beta*p/(c(+1)*p(+1)) + D*p/m
    1/c[t] = beta_disc * p[t] / (c[t+1]*p[t+1]) + D_coef * p[t] / m[t]

    # (9.3) FOC hours:
    # 1/c = -B/w  →  w/c = -B  →  1/c = -B_param/w
    1/c[t] = -B_param / w[t]

    # (9.4) FOC capital:
    # 1/c = beta/c(+1)*(r(+1)+1-delta)
    # With predetermined k: r(+1) uses k(+1) which is our k[t]
    1/c[t] = beta_disc / c[t+1] * (r[t+1] + 1 - delta)

    # Money growth definition: m = g*m(-1)
    # Our timing: m[t] = g[t] * m[t-1]
    m[t] = exp(log_g[t]) * m[t-1]

    # Production function: y = lambda*k^theta*h^(1-theta)
    # With predetermined k: uses k which is our k[t-1]
    y[t] = exp(log_lam[t]) * k[t-1]^theta * h[t]^(1-theta)

    # Firm FOC labor: w = (1-theta)*lambda*k^theta*h^(-theta)
    w[t] = (1-theta) * exp(log_lam[t]) * k[t-1]^theta * h[t]^(-theta)

    # Firm FOC capital: r = theta*lambda*(k/h)^(theta-1)
    r[t] = theta * exp(log_lam[t]) * (k[t-1] / h[t])^(theta-1)

    # Money growth process: log(g) = (1-pi)*log(g_bar) + pi*log(g(-1)) + eps_g
    # g_bar=1 → log(g_bar)=0, so: log_g[t] = pi*log_g[t-1] + 0.01*eps_g[t]
    log_g[t] = pi_ar * log_g[t-1] + 0.01 * eps_g[t]

    # TFP process: log(lambda) = gamma*log(lambda(-1)) + eps_lambda
    # log_lam[t] = gamma*log_lam[t-1] + 0.01*eps_lambda[t]
    log_lam[t] = gamma_ar * log_lam[t-1] + 0.01 * eps_lambda[t]

    steady_state = begin
        r_ss = 1/beta_disc - 1 + delta
        w_ss = (1-theta)*(r_ss/theta)^(theta/(theta-1))
        c_ss = -w_ss/B_param
        k_ss = c_ss / ((r_ss*(1-theta)/(w_ss*theta))^(1-theta) - delta)
        h_ss = r_ss*(1-theta)/(w_ss*theta)*k_ss
        y_ss = k_ss*(r_ss*(1-theta)/(w_ss*theta))^(1-theta)
        g_ss = 1.0
        lambda_ss = 1.0
        p_ss = 1.0
        m_ss = p_ss * D_coef * g_ss * c_ss / (g_ss - beta_disc)
        log_g_ss = 0.0     # log(1) = 0
        log_lam_ss = 0.0   # log(1) = 0
        # Order: w, r, c, k, h, m, p, log_g, log_lam, y
        [w_ss, r_ss, c_ss, k_ss, h_ss, m_ss, p_ss, log_g_ss, log_lam_ss, y_ss]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 100)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  McCandless (2008), Chapter 9 — Money-in-Utility")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "mccandless_ch9.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

let
    # ── Steady State comparison ──
    # Dynare endo order: w(1), r(2), c(3), k(4), h(5), m(6), p(7), g(8), lambda(9), y(10)
    # Our endo order:    w(1), r(2), c(3), k(4), h(5), m(6), p(7), log_g(8), log_lam(9), y(10)
    # Dynare stores level SS; our log_g_ss=0 maps to g_ss=1, log_lam_ss=0 maps to lambda_ss=1
    println("\n=== Steady State ===")
    ss_pass = true
    level_pairs = [("w",1,1), ("r",2,2), ("c",3,3), ("k",4,4), ("h",5,5),
                   ("m",6,6), ("p",7,7), ("y",10,10)]
    for (name, j_idx, d_idx) in level_pairs
        diff = abs(ss[j_idx] - d_ss[d_idx])
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-10s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, ss[j_idx], d_ss[d_idx], diff, ok ? "PASS" : "FAIL")
    end
    # g and lambda: compare exp(our_log) vs dynare_level
    for (name, j_idx, d_idx) in [("g",8,8), ("lambda",9,9)]
        j_val = exp(ss[j_idx])   # our log_g → exp → level
        d_val = d_ss[d_idx]
        diff = abs(j_val - d_val)
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-10s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── IRF comparison ──
    # Dynare IRFs are level deviations from SS. Our solver also produces level deviations.
    # For log_g and log_lam: our IRF is d(log_g), Dynare IRF is d(g).
    # Linearized: d(g) ≈ g_ss * d(log_g) = 1 * d(log_g).
    # Similarly d(lambda) ≈ lambda_ss * d(log_lam) = 1 * d(log_lam).
    println("\n=== IRFs (level deviations from SS) ===")
    irf_pass = true

    # Direct level variables
    level_vars = [("w",1), ("r",2), ("c",3), ("k",4), ("h",5), ("m",6), ("p",7), ("y",10)]
    # Log-to-level variables: our IRF → multiply by SS level (=1 for both g and lambda)
    log_vars = [("g",8,1.0), ("lambda",9,1.0)]  # (dynare_name, our_idx, ss_level)

    shock_info = [("eps_g",1), ("eps_lambda",2)]

    for (sn, si) in shock_info
        for (vn, vi) in level_vars
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
        for (vn, vi, ss_lev) in log_vars
            d_key = vn * "_" * sn
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 100)
            # Our log-dev * ss_level ≈ Dynare level-dev
            j_vals = [ir.values[h, vi, si] * ss_lev for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-25s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
