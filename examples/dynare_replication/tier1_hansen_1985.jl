# Tier 1: Hansen (1985), Indivisible Labor — Dynare Replication
# Dynare source: DSGE_mod/Hansen_1985/Hansen_1985.mod
#
# Indivisible labor RBC model.
# 9 endogenous: c, w, r, y, h, k, invest, lambda, productivity
# 1 exogenous : eps_a (stderr sigma_eps = 0.00712)
#
# Dynare uses `loglinear` → IRFs are in log-deviations (% from SS).
# Our solver produces level deviations; convert via: log_dev = level_dev / ss_level.
#
# Key equation: log(lambda) = gamma*log(lambda(-1)) + eps_a
# Rewritten as: log_lam[t] = gamma*log_lam[t-1] + sigma_eps*eps_a[t]
# with lambda[t] = exp(log_lam[t]).
using MacroEconometricModels, MAT, Printf

# ── Parameters ──
const _beta      = 0.99
const _delta     = 0.025
const _theta     = 0.36
const _gamma     = 0.95
const _A         = 2.0
const _sigma_eps = 0.00712
const _h0        = 0.53

# Derived (indivisible labor)
const _B = -_A * log(1 - _h0) / _h0   # 2.849141827464275

# Steady state values (level)
const _lambda_ss = 1.0
const _h_ss  = (1 - _theta) * (1/_beta - (1 - _delta)) / (_B * (1/_beta - (1 - _delta) - _theta * _delta))
const _k_ss  = _h_ss * ((1/_beta - (1 - _delta)) / (_theta * _lambda_ss))^(1/(_theta - 1))
const _i_ss  = _delta * _k_ss
const _y_ss  = _lambda_ss * _k_ss^_theta * _h_ss^(1 - _theta)
const _c_ss  = _y_ss - _delta * _k_ss
const _r_ss  = 1/_beta - (1 - _delta)
const _w_ss  = (1 - _theta) * _y_ss / _h_ss
const _prod_ss = _y_ss / _h_ss

# ── Model ──
# We use a variable `log_lam` = log(lambda) to avoid log() in equations.
# lambda = exp(log_lam), so wherever the Dynare model uses `lambda`, we use exp(log_lam).
#
# Dynare equations:
#   (1) 1/c = beta*(1/c(+1))*(r(+1)+(1-delta))
#   (2) (1-theta)*(y/h) = B*c              [indivisible labor]
#   (3) c = y + (1-delta)*k(-1) - k
#   (4) k = (1-delta)*k(-1) + invest
#   (5) y = lambda*k(-1)^theta*h^(1-theta)
#   (6) r = theta*(y/k(-1))
#   (7) w = (1-theta)*(y/h)
#   (8) log(lambda) = gamma*log(lambda(-1)) + eps_a
#   (9) productivity = y/h

spec = @dsge begin
    parameters: beta_disc = 0.99, delta = 0.025, theta = 0.36, gamma_ar = 0.95,
                B_param = 2.849141827464275, sigma_eps = 0.00712
    endogenous: c, w, r, y, h, k, invest, log_lam, productivity
    exogenous: eps_a

    # (1) Euler equation
    1/c[t] = beta_disc * (1/c[t+1]) * (r[t+1] + (1 - delta))

    # (2) Indivisible labor FOC
    (1 - theta) * (y[t] / h[t]) = B_param * c[t]

    # (3) Resource constraint
    c[t] = y[t] + (1 - delta) * k[t-1] - k[t]

    # (4) Capital accumulation
    k[t] = (1 - delta) * k[t-1] + invest[t]

    # (5) Production function (lambda = exp(log_lam))
    y[t] = exp(log_lam[t]) * k[t-1]^theta * h[t]^(1 - theta)

    # (6) Rental rate of capital
    r[t] = theta * (y[t] / k[t-1])

    # (7) Real wage
    w[t] = (1 - theta) * (y[t] / h[t])

    # (8) TFP process (log_lam = log(lambda))
    log_lam[t] = gamma_ar * log_lam[t-1] + sigma_eps * eps_a[t]

    # (9) Productivity
    productivity[t] = y[t] / h[t]

    steady_state = begin
        B_val = B_param
        lam_ss = 1.0
        h_ss = (1 - theta) * (1/beta_disc - (1 - delta)) / (B_val * (1/beta_disc - (1 - delta) - theta * delta))
        k_ss = h_ss * ((1/beta_disc - (1 - delta)) / (theta * lam_ss))^(1/(theta - 1))
        i_ss = delta * k_ss
        y_ss = lam_ss * k_ss^theta * h_ss^(1 - theta)
        c_ss = y_ss - delta * k_ss
        r_ss = 1/beta_disc - (1 - delta)
        w_ss = (1 - theta) * y_ss / h_ss
        prod_ss = y_ss / h_ss
        log_lam_ss = 0.0  # log(1) = 0
        # Order: c, w, r, y, h, k, invest, log_lam, productivity
        [c_ss, w_ss, r_ss, y_ss, h_ss, k_ss, i_ss, log_lam_ss, prod_ss]
    end
end

sol = solve(spec; method=:gensys)
ir  = irf(sol, 20)
ss  = sol.spec.steady_state

println("=" ^ 60)
println("  Hansen (1985), Indivisible Labor")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "hansen_1985.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data  = matread(dynare_mat)
d_ss  = vec(data["steady_state"])
irfs  = data["irfs"]

# ── Steady State comparison ──
# Dynare endo order: c, w, r, y, h, k, invest, lambda, productivity
# Dynare SS values are in LOG (because loglinear option stores log SS)
# Our SS is in levels.  Compare: log(our_ss) vs dynare_ss
println("\n=== Steady State (comparing log values) ===")
let
    ss_pass = true
    # Our order: c=1, w=2, r=3, y=4, h=5, k=6, invest=7, log_lam=8, productivity=9
    # Dynare  : c=1, w=2, r=3, y=4, h=5, k=6, invest=7, lambda=8, productivity=9
    pairs = [("c",1,1), ("w",2,2), ("r",3,3), ("y",4,4), ("h",5,5),
             ("k",6,6), ("invest",7,7), ("productivity",9,9)]

    for (name, j_idx, d_idx) in pairs
        j_val = log(ss[j_idx])   # Our level → log
        d_val = d_ss[d_idx]      # Dynare stores log(level)
        diff  = abs(j_val - d_val)
        ok    = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-15s  log(Julia)=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end
    # lambda: our log_lam (=0) vs Dynare log(lambda) (=0)
    j_val = ss[8]      # log_lam_ss = 0
    d_val = d_ss[8]     # Dynare lambda SS (in log = log(1) = 0)
    diff  = abs(j_val - d_val)
    ok    = diff < 1e-6
    ss_pass = ss_pass && ok
    @printf("  %-15s  log(Julia)=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
            "lambda", j_val, d_val, diff, ok ? "PASS" : "FAIL")

    # ── IRF comparison ──
    # Dynare loglinear IRFs = log-deviations from SS (percentage deviations).
    # Our solver gives level deviations.
    # Convert: log_dev[h] = level_dev[h] / ss_level
    # For log_lam: our IRF is already the log-deviation of lambda (since log_lam_ss=0,
    #   and lambda = exp(log_lam), linearized: dlambda/lambda ≈ d(log_lam) = IRF of log_lam).
    println("\n=== IRFs (log-deviations from SS) ===")
    irf_pass = true

    # Variable mapping: (dynare_name, our_idx, our_ss_level, is_log_lam)
    var_map = [
        ("c",        1, ss[1], false),
        ("y",        4, ss[4], false),
        ("h",        5, ss[5], false),
        ("k",        6, ss[6], false),
        ("invest",   7, ss[7], false),
        ("productivity", 9, ss[9], false),
    ]

    for (vn, vi, ss_lev, _) in var_map
        d_key = vn * "_eps_a"
        haskey(irfs, d_key) || continue
        d_vals = vec(irfs[d_key])
        H = min(length(d_vals), 20)
        # Our level deviations → log deviations
        j_vals = [ir.values[h, vi, 1] / ss_lev for h in 1:H]
        max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
        ok = max_diff < 1e-4
        irf_pass = irf_pass && ok
        @printf("  %-20s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
    end

    # ── Variance Decomposition ──
    d_names_raw = data["endo_names"]
    if d_names_raw isa Matrix
        d_names_parsed = vec([strip(string(d_names_raw[i,1])) for i in 1:size(d_names_raw, 1)])
    else
        d_names_parsed = vec([strip(string(x)) for x in d_names_raw])
    end
    d_endo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_names_parsed))

    vd_pass = true
    if haskey(data, "variance_decomposition")
        fv = fevd(sol, 1000)
        d_vd = data["variance_decomposition"]

        d_exo_names_raw = data["exo_names"]
        if d_exo_names_raw isa Matrix
            d_exo_names = vec([strip(string(d_exo_names_raw[i,1])) for i in 1:size(d_exo_names_raw, 1)])
        else
            d_exo_names = vec([strip(string(x)) for x in d_exo_names_raw])
        end
        d_exo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_exo_names))

        j_vd = fv.proportions[:, :, end] .* 100.0
        our_var_names = string.(spec.endog)
        our_shock_names = string.(spec.exog)

        println("\n=== Variance Decomposition (asymptotic, %) ===")
        @printf("  %-4s %-15s %-10s %10s %10s %10s\n", "", "Variable", "Shock", "Julia", "Dynare", "Diff")
        println("  ", "-"^60)

        for (j_vi, vn_sym) in enumerate(our_var_names)
            vn = vn_sym == "log_lam" ? "lambda" : vn_sym
            d_vi = get(d_endo_idx, vn, 0)
            (d_vi == 0 || d_vi > size(d_vd, 1)) && continue
            for (j_si, sn) in enumerate(our_shock_names)
                d_si = get(d_exo_idx, sn, 0)
                (d_si == 0 || d_si > size(d_vd, 2)) && continue
                j_val = j_vd[j_vi, j_si]
                d_val = d_vd[d_vi, d_si]
                diff = abs(j_val - d_val)
                ok = diff < 1.0
                vd_pass = vd_pass && ok
                @printf("  %s  %-15s %-10s %10.4f %10.4f %10.4f\n",
                        ok ? "✓" : "✗", vn, sn, j_val, d_val, diff)
            end
        end
        println("  VD: ", vd_pass ? "ALL PASS" : "SOME FAIL")
    else
        println("\n  (No variance_decomposition in .mat — skipping)")
    end

    # ── Autocorrelation comparison ──
    acorr_pass = true
    if haskey(data, "autocorr")
        d_acorr = data["autocorr"]
        Sigma_y = MacroEconometricModels.solve_lyapunov(sol.G1, sol.impact)
        Gamma_1 = sol.G1 * Sigma_y

        println("\n=== Autocorrelation (lag 1) ===")
        @printf("  %-4s %-15s %14s %14s %10s\n", "", "Variable", "Julia", "Dynare", "Diff")
        println("  ", "-"^55)

        for (j_i, vn_sym) in enumerate(string.(spec.endog))
            vn = vn_sym == "log_lam" ? "lambda" : vn_sym
            d_i = get(d_endo_idx, vn, 0)
            (d_i == 0 || d_i > size(d_acorr, 1)) && continue
            j_val = Sigma_y[j_i, j_i] > 0 ? Gamma_1[j_i, j_i] / Sigma_y[j_i, j_i] : 0.0
            d_val = d_acorr[d_i, d_i]
            diff = abs(j_val - d_val)
            ok = diff < 0.01
            acorr_pass = acorr_pass && ok
            @printf("  %s  %-15s %14.8f %14.8f %10.6f\n",
                    ok ? "✓" : "✗", vn, j_val, d_val, diff)
        end
        println("  Autocorrelation: ", acorr_pass ? "ALL PASS" : "SOME FAIL")
    end

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL"), " *
            "VD=$(vd_pass ? "PASS" : "FAIL"), Acorr=$(acorr_pass ? "PASS" : "FAIL")")
end
