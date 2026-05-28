# Tier 1: RBC Baseline (King & Rebelo, 1999) — Dynare Replication
# Dynare source: DSGE_mod/RBC_baseline/RBC_baseline.mod
using MacroEconometricModels, MAT, Printf

# Derived calibration (from Dynare's steady_state_model block)
const _alpha = 0.33; const _i_y = 0.25; const _k_y = 10.4
const _x = 0.0055; const _n = 0.0027; const _gshare = 0.2038
const _gammax = (1 + _n) * (1 + _x)
const _delta = _i_y / _k_y - _x - _n - _n * _x
const _beta = (1 + _x) * (1 + _n) / (_alpha / _k_y + (1 - _delta))
const _l_ss = 0.33
const _k_ss = ((1/_beta * _gammax - (1 - _delta)) / _alpha)^(1/(_alpha - 1)) * _l_ss
const _y_ss = _k_ss^_alpha * _l_ss^(1 - _alpha)
const _g_ss_val = _gshare * _y_ss

spec = @dsge begin
    parameters: sigma_c = 1.0, alpha = 0.33, rhoz = 0.97, rhog = 0.989,
                beta_disc = 0.9924281390931616, gammax = 1.0082148500000001,
                delta = 0.015823611538461537, gshare = 0.2038,
                psi_d = 2.4904852257470322,
                g_ss = 0.21313019787746235, sigma_z = 0.66, sigma_g = 1.04
    endogenous: y, c, k, l, z, ghat, r, w, invest
    exogenous: eps_z, eps_g

    c[t]^(-sigma_c) = beta_disc / gammax * c[t+1]^(-sigma_c) * (alpha * exp(z[t+1]) * (k[t] / l[t+1])^(alpha - 1) + (1 - delta))
    psi_d * c[t]^sigma_c * (1 / (1 - l[t])) = w[t]
    gammax * k[t] = (1 - delta) * k[t-1] + invest[t]
    y[t] = invest[t] + c[t] + g_ss * exp(ghat[t])
    y[t] = exp(z[t]) * k[t-1]^alpha * l[t]^(1 - alpha)
    w[t] = (1 - alpha) * y[t] / l[t]
    r[t] = 4 * alpha * y[t] / k[t-1]
    z[t] = rhoz * z[t-1] + sigma_z * eps_z[t]
    ghat[t] = rhog * ghat[t-1] + sigma_g * eps_g[t]

    steady_state = begin
        l_ss = 0.33
        k_ss = (((1 / beta_disc) * gammax - (1 - delta)) / alpha)^(1 / (alpha - 1)) * l_ss
        invest_ss = (gammax - 1 + delta) * k_ss
        y_ss = k_ss^alpha * l_ss^(1 - alpha)
        c_ss = y_ss - invest_ss - g_ss
        w_ss = (1 - alpha) * y_ss / l_ss
        r_ss = 4 * alpha * y_ss / k_ss
        [y_ss, c_ss, k_ss, l_ss, 0.0, 0.0, r_ss, w_ss, invest_ss]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 40)
ss = sol.spec.steady_state

println("="^60)
println("  RBC Baseline (King & Rebelo, 1999)")
println("="^60)
println("  determined = ", is_determined(sol))

# Compare with Dynare .mat
dynare_mat = joinpath(@__DIR__, "dynare_results", "rbc_baseline.mat")
if isfile(dynare_mat)
    data = matread(dynare_mat)
    d_ss = vec(data["steady_state"])
    irfs = data["irfs"]

    # Steady state (Dynare has 15 vars including log defs, we have 9)
    println("\n=== Steady State ===")
    dynare_ss_ref = Dict("y"=>d_ss[1], "c"=>d_ss[2], "k"=>d_ss[3], "l"=>d_ss[4],
                         "z"=>d_ss[5], "ghat"=>d_ss[6], "r"=>d_ss[7], "w"=>d_ss[8], "invest"=>d_ss[9])
    for (i, v) in enumerate(string.(spec.endog))
        d_val = get(dynare_ss_ref, v, NaN)
        isnan(d_val) && continue
        diff = abs(ss[i] - d_val)
        @printf("  %-10s  Julia=%12.8f  Dynare=%12.8f  diff=%8.2e  %s\n",
                v, ss[i], d_val, diff, diff < 1e-6 ? "PASS" : "FAIL")
    end

    # IRFs (convert level deviations to log deviations for log_ variables)
    println("\n=== IRFs ===")
    vars = [("log_y",1,true), ("log_c",2,true), ("log_k",3,true), ("log_l",4,true),
            ("z",5,false), ("ghat",6,false), ("r",7,false), ("log_w",8,true), ("log_invest",9,true)]
    shocks = ["eps_z", "eps_g"]
    local all_irf_pass = true
    for sn in shocks, (dname, idx, is_log) in vars
        d_key = dname * "_" * sn
        haskey(irfs, d_key) || continue
        d_vals = vec(irfs[d_key])
        si = sn == "eps_z" ? 1 : 2
        H = min(length(d_vals), 40)
        j_vals = [ir.values[h, idx, si] for h in 1:H]
        j_conv = is_log ? j_vals ./ ss[idx] : j_vals
        max_diff = maximum(abs.(j_conv .- d_vals[1:H]))
        ok = max_diff < 1e-4
        all_irf_pass = all_irf_pass && ok
        @printf("  %-25s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
    end
    println("  IRFs: ", all_irf_pass ? "ALL PASS" : "SOME FAIL")

    # ── Variance Decomposition ──
    d_names_raw = data["endo_names"]
    if d_names_raw isa Matrix
        d_names_parsed = vec([strip(string(d_names_raw[i,1])) for i in 1:size(d_names_raw, 1)])
    else
        d_names_parsed = vec([strip(string(x)) for x in d_names_raw])
    end
    d_endo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_names_parsed))

    local all_vd_pass = true
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
        @printf("  %-4s %-12s %-10s %10s %10s %10s\n", "", "Variable", "Shock", "Julia", "Dynare", "Diff")
        println("  ", "-"^60)

        for (j_vi, vn) in enumerate(our_var_names)
            d_vi = get(d_endo_idx, vn, 0)
            (d_vi == 0 || d_vi > size(d_vd, 1)) && continue
            for (j_si, sn) in enumerate(our_shock_names)
                d_si = get(d_exo_idx, sn, 0)
                (d_si == 0 || d_si > size(d_vd, 2)) && continue
                j_val = j_vd[j_vi, j_si]
                d_val = d_vd[d_vi, d_si]
                diff = abs(j_val - d_val)
                ok = diff < 1.0
                all_vd_pass = all_vd_pass && ok
                @printf("  %s  %-12s %-10s %10.4f %10.4f %10.4f\n",
                        ok ? "✓" : "✗", vn, sn, j_val, d_val, diff)
            end
        end
        println("  VD: ", all_vd_pass ? "ALL PASS" : "SOME FAIL")
    else
        println("\n  (No variance_decomposition in .mat — skipping)")
    end

    # ── Moments comparison ──
    local all_acorr_pass = true
    if haskey(data, "autocorr")
        d_acorr = data["autocorr"]
        Sigma_y = MacroEconometricModels.solve_lyapunov(sol.G1, sol.impact)
        Gamma_1 = sol.G1 * Sigma_y

        println("\n=== Autocorrelation (lag 1) ===")
        @printf("  %-4s %-12s %14s %14s %10s\n", "", "Variable", "Julia", "Dynare", "Diff")
        println("  ", "-"^55)

        for (j_i, vn) in enumerate(string.(spec.endog))
            d_i = get(d_endo_idx, vn, 0)
            (d_i == 0 || d_i > size(d_acorr, 1)) && continue
            j_val = Sigma_y[j_i, j_i] > 0 ? Gamma_1[j_i, j_i] / Sigma_y[j_i, j_i] : 0.0
            d_val = d_acorr[d_i, d_i]
            diff = abs(j_val - d_val)
            ok = diff < 0.01
            all_acorr_pass = all_acorr_pass && ok
            @printf("  %s  %-12s %14.8f %14.8f %10.6f\n",
                    ok ? "✓" : "✗", vn, j_val, d_val, diff)
        end
        println("  Autocorrelation: ", all_acorr_pass ? "ALL PASS" : "SOME FAIL")
    end

    println("\n  Overall: IRF=$(all_irf_pass ? "PASS" : "FAIL"), " *
            "VD=$(all_vd_pass ? "PASS" : "FAIL"), " *
            "Acorr=$(all_acorr_pass ? "PASS" : "FAIL")")
else
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
end
