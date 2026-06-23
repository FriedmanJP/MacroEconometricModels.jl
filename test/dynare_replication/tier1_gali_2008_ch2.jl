# Tier 1: Gali (2008) Chapter 2 — Classical Monetary Economy
# Dynare source: DSGE_mod/Gali_2008/Gali_2008_chapter_2.mod
# Mat ref: dynare_results/gali_2008_ch2.mat
#
# 9 endogenous: C, W_real, Pi, A, N, R, realinterest, Y, m_growth_ann
# 2 exogenous:  eps_A, eps_m
#
# Non-linear model in LEVELS. Dynare IRFs are linear deviations from SS.
# For level variables: our level_dev matches Dynare's directly.
using MacroEconometricModels, MAT, Printf

# Parameters from the .mod file
const _alppha = 0.33
const _betta = 0.99
const _rho = 0.9
const _siggma = 1.0
const _phi = 1.0
const _phi_pi = 1.5
const _eta = 4.0

# Pre-compute SS values
const _N_ss = (1 - _alppha)^(1 / ((1 - _siggma) * _alppha + _phi + _siggma))
const _C_ss = _N_ss^(1 - _alppha)
const _W_real_ss = (1 - _alppha) * _N_ss^(-_alppha)
const _R_ss = 1 / _betta
const _Y_ss = _C_ss

spec = @dsge begin
    parameters: alppha = 0.33, betta = 0.99, rho_a = 0.9, siggma = 1.0,
                phi_frisch = 1.0, phi_pi = 1.5, eta_money = 4.0
    endogenous: C, W_real, Pi, A, N, R, realinterest, Y, m_growth_ann
    exogenous: eps_A, eps_m

    # (1) FOC Wages, eq. (6)
    W_real[t] = C[t]^siggma * N[t]^phi_frisch

    # (2) Euler equation eq. (7)
    1/R[t] = betta * (C[t+1]/C[t])^(-siggma) / Pi[t+1]

    # (3) Production function eq. (8)
    A[t] * N[t]^(1 - alppha) = C[t]

    # (4) FOC wages firm, eq. (13)
    W_real[t] = (1 - alppha) * A[t] * N[t]^(-alppha)

    # (5) Definition Real interest rate
    realinterest[t] = R[t] / Pi[t+1]

    # (6) Monetary Policy Rule, eq. (22)
    R[t] = (1/betta) * Pi[t]^phi_pi + eps_m[t]

    # (7) Market Clearing, eq. (15)
    C[t] = Y[t]

    # (8) Technology Shock
    log(A[t]) = rho_a * log(A[t-1]) + eps_A[t]

    # (9) Money growth
    m_growth_ann[t] = 4*(log(Y[t]) - log(Y[t-1]) - eta_money*(log(R[t]) - log(R[t-1])) + log(Pi[t]))

    steady_state = begin
        A_ss = 1.0
        R_ss = 1.0 / betta
        Pi_ss = 1.0
        ri_ss = R_ss
        N_ss = (1 - alppha)^(1 / ((1 - siggma) * alppha + phi_frisch + siggma))
        C_ss = A_ss * N_ss^(1 - alppha)
        W_ss = (1 - alppha) * A_ss * N_ss^(-alppha)
        Y_ss = C_ss
        mg_ss = 0.0
        [C_ss, W_ss, Pi_ss, A_ss, N_ss, R_ss, ri_ss, Y_ss, mg_ss]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 20)
ss = sol.spec.steady_state

println("=" ^ 60)
println("  Gali (2008) Chapter 2 — Classical Monetary Economy")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "gali_2008_ch2.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
irfs = data["irfs"]

# Dynare endo order: C(1), W_real(2), Pi(3), A(4), N(5), R(6), realinterest(7), Y(8), m_growth_ann(9)
# Our endo order  : C(1), W_real(2), Pi(3), A(4), N(5), R(6), realinterest(7), Y(8), m_growth_ann(9)

let
    println("\n=== Steady State ===")
    ss_pass = true
    names = ["C", "W_real", "Pi", "A", "N", "R", "realinterest", "Y", "m_growth_ann"]
    for (i, name) in enumerate(names)
        j_val = ss[i]
        d_val = d_ss[i]
        diff = abs(j_val - d_val)
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-15s  Julia=%12.8f  Dynare=%12.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── IRFs ──
    # Dynare level-deviation IRFs: C_eps_A, Y_eps_A, Pi_eps_A, R_eps_A, etc.
    # Our solver gives level deviations directly for non-log models.
    println("\n=== IRFs ===")
    irf_pass = true

    # Technology shock (eps_A) — real vars respond, nominal don't
    irf_map_A = [
        ("C_eps_A", 1, 1), ("Y_eps_A", 8, 1), ("Pi_eps_A", 3, 1),
        ("R_eps_A", 6, 1), ("realinterest_eps_A", 7, 1),
        ("m_growth_ann_eps_A", 9, 1)
    ]
    # Monetary shock (eps_m) — real vars should not respond (neutrality)
    irf_map_m = [
        ("C_eps_m", 1, 2), ("Y_eps_m", 8, 2), ("Pi_eps_m", 3, 2),
        ("R_eps_m", 6, 2), ("realinterest_eps_m", 7, 2),
        ("m_growth_ann_eps_m", 9, 2)
    ]

    for irf_map in [irf_map_A, irf_map_m]
        for (d_key, j_idx, s_idx) in irf_map
            haskey(irfs, d_key) || continue
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 20)
            j_vals = [ir.values[h, j_idx, s_idx] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            ok = max_diff < 1e-4
            irf_pass = irf_pass && ok
            @printf("  %-25s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=", ss_pass ? "PASS" : "FAIL",
            ", IRF=", irf_pass ? "PASS" : "FAIL")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
