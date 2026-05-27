# Tier 1: Gali (2015) Chapter 2 — Classical Monetary Economy (2nd Edition)
# Dynare source: DSGE_mod/Gali_2015/Gali_2015_chapter_2.mod
# Mat ref: dynare_results/gali_2015_ch2.mat
#
# 12 endogenous: C, W_real, Pi, A, N, R, realinterest, Y, nu, m_growth_ann, Q, Z
# 3 exogenous:  eps_a, eps_z, eps_nu
#
# Non-linear model in LEVELS. Dynare IRFs are linear deviations from SS.
using MacroEconometricModels, MAT, Printf

# Parameters from the .mod file
const _alppha = 0.25
const _betta = 0.99
const _rho_a = 0.9
const _rho_z = 0.5
const _rho_nu = 0.5
const _siggma = 1.0
const _varphi = 5.0
const _phi_pi = 1.5
const _eta = 3.77

# Pre-compute SS
const _N_ss = (1 - _alppha)^(1 / ((1 - _siggma) * _alppha + _varphi + _siggma))
const _C_ss = _N_ss^(1 - _alppha)
const _W_real_ss = (1 - _alppha) * _N_ss^(-_alppha)
const _R_ss = 1 / _betta
const _Q_ss = _betta

spec = @dsge begin
    parameters: alppha = 0.25, betta = 0.99, rho_a = 0.9, rho_z = 0.5, rho_nu = 0.5,
                siggma = 1.0, varphi = 5.0, phi_pi = 1.5, eta_money = 3.77
    endogenous: C, W_real, Pi, A, N, R, realinterest, Y, nu, m_growth_ann, Q, Z
    exogenous: eps_a, eps_z, eps_nu

    # (1) FOC Wages, eq. (7)
    W_real[t] = C[t]^siggma * N[t]^varphi

    # (2) Euler equation eq. (8)
    Q[t] = betta * (C[t+1]/C[t])^(-siggma) * (Z[t+1]/Z[t]) / Pi[t+1]

    # (3) Definition nominal interest rate
    R[t] = 1/Q[t]

    # (4) Production function eq. (12)
    Y[t] = A[t] * N[t]^(1 - alppha)

    # (5) FOC wages firm, eq. (14)
    W_real[t] = (1 - alppha) * A[t] * N[t]^(-alppha)

    # (6) Definition Real interest rate
    R[t] = realinterest[t] * Pi[t+1]

    # (7) Monetary Policy Rule
    R[t] = (1/betta) * Pi[t]^phi_pi * exp(nu[t])

    # (8) Market Clearing
    C[t] = Y[t]

    # (9) Technology Shock
    log(A[t]) = rho_a * log(A[t-1]) + eps_a[t]

    # (10) Preference Shock
    log(Z[t]) = rho_z * log(Z[t-1]) + eps_z[t]

    # (11) Monetary policy shock
    nu[t] = rho_nu * nu[t-1] + eps_nu[t]

    # (12) Money growth
    m_growth_ann[t] = 4*(log(C[t]) - log(C[t-1]) - eta_money*(log(R[t]) - log(R[t-1])) + log(Pi[t]))

    steady_state = begin
        A_ss = 1.0
        Z_ss = 1.0
        R_ss = 1.0 / betta
        Pi_ss = 1.0
        Q_ss = betta
        ri_ss = R_ss
        N_ss = (1 - alppha)^(1 / ((1 - siggma) * alppha + varphi + siggma))
        C_ss = A_ss * N_ss^(1 - alppha)
        W_ss = (1 - alppha) * A_ss * N_ss^(-alppha)
        Y_ss = C_ss
        nu_ss = 0.0
        mg_ss = 0.0
        # Order: C, W_real, Pi, A, N, R, realinterest, Y, nu, m_growth_ann, Q, Z
        [C_ss, W_ss, Pi_ss, A_ss, N_ss, R_ss, ri_ss, Y_ss, nu_ss, mg_ss, Q_ss, Z_ss]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 20)
ss = sol.spec.steady_state

println("=" ^ 60)
println("  Gali (2015) Chapter 2 — Classical Monetary Economy")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "gali_2015_ch2.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
irfs = data["irfs"]

# Dynare endo order: C(1), W_real(2), Pi(3), A(4), N(5), R(6), realinterest(7), Y(8), nu(9), m_growth_ann(10), Q(11), Z(12)

let
    println("\n=== Steady State ===")
    ss_pass = true
    names = ["C", "W_real", "Pi", "A", "N", "R", "realinterest", "Y", "nu", "m_growth_ann", "Q", "Z"]
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
    println("\n=== IRFs ===")
    irf_pass = true

    # Technology shock (eps_a, shock index 1)
    # Preference shock (eps_z, shock index 2)
    # Monetary policy shock (eps_nu, shock index 3)
    irf_checks = [
        # Technology shock
        ("C_eps_a", 1, 1), ("Y_eps_a", 8, 1), ("Pi_eps_a", 3, 1),
        ("R_eps_a", 6, 1), ("realinterest_eps_a", 7, 1),
        ("m_growth_ann_eps_a", 10, 1),
        # Preference shock — real vars should not respond (neutrality)
        ("C_eps_z", 1, 2), ("Y_eps_z", 8, 2), ("Pi_eps_z", 3, 2),
        ("R_eps_z", 6, 2), ("realinterest_eps_z", 7, 2),
        ("m_growth_ann_eps_z", 10, 2),
        # Monetary policy shock
        ("C_eps_nu", 1, 3), ("Y_eps_nu", 8, 3), ("Pi_eps_nu", 3, 3),
        ("R_eps_nu", 6, 3), ("realinterest_eps_nu", 7, 3),
        ("m_growth_ann_eps_nu", 10, 3),
    ]

    for (d_key, j_idx, s_idx) in irf_checks
        haskey(irfs, d_key) || continue
        d_vals = vec(irfs[d_key])
        H = min(length(d_vals), 20)
        j_vals = [ir.values[h, j_idx, s_idx] for h in 1:H]
        max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
        ok = max_diff < 1e-4
        irf_pass = irf_pass && ok
        @printf("  %-25s  max|diff|=%8.2e  %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
    end

    println("\n  Overall: SS=", ss_pass ? "PASS" : "FAIL",
            ", IRF=", irf_pass ? "PASS" : "FAIL")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
