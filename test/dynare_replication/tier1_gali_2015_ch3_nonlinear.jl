# Tier 1: Gali (2015) Chapter 3 — Baseline NK Model (Nonlinear)
# Dynare source: DSGE_mod/Gali_2015/Gali_2015_chapter_3_nonlinear.mod
# Mat ref: dynare_results/gali_2015_ch3_nl.mat
#
# money_growth_rule=1 variant (money supply rule, not Taylor rule)
#
# 29 endogenous + we skip some pure-log definitions to keep the system clean.
# Core NK model: C, W_real, Pi, A, N, R, realinterest, Y, Q, Z, S, Pi_star,
#                x_aux_1, x_aux_2, MC, M_real, P, money_growth, money_growth_ann,
#                i_ann, pi_ann, r_real_ann, log_y, log_W_real, log_N, log_m_nominal,
#                log_P, log_A, log_Z
#
# Dynare IRFs use different shock sizes for the 3 stoch_simul runs:
#   eps_m: stderr 0.0025, eps_z: stderr 0.005, eps_a: stderr 0.01
# We embed these in the equations as sigma_* * eps_*[t].
using MacroEconometricModels, MAT, Printf

# Parameters
const _siggma = 1.0
const _varphi = 5.0
const _phi_pi = 1.5
const _phi_y = 0.125
const _theta = 0.75
const _rho_m = 0.5
const _rho_z = 0.5
const _rho_a = 0.9
const _betta = 0.99
const _eta = 3.77
const _alppha = 0.25
const _epsilon = 9.0
const _tau = 0.0

# Steady state (from steady_state_model block in .mod)
const _MC_ss = (_epsilon - 1) / _epsilon / (1 - _tau)
const _R_ss = 1 / _betta
const _Pi_ss = 1.0
const _Q_ss = _betta
const _N_ss = ((1 - _alppha) * _MC_ss)^(1 / ((1 - _siggma) * _alppha + _varphi + _siggma))
const _C_ss = _N_ss^(1 - _alppha)
const _W_real_ss = _C_ss^_siggma * _N_ss^_varphi
const _Y_ss = _C_ss
const _S_ss = 1.0
const _Pi_star_ss = 1.0
const _x_aux_1_ss = _C_ss^(-_siggma) * _Y_ss * _MC_ss / (1 - _betta * _theta)
const _x_aux_2_ss = _C_ss^(-_siggma) * _Y_ss / (1 - _betta * _theta)
const _M_real_ss = _Y_ss / _R_ss^_eta
const _P_ss = 1.0

spec = @dsge begin
    parameters: siggma = 1.0, varphi = 5.0, theta = 0.75, betta = 0.99,
                eta_money = 3.77, alppha = 0.25, epsilon_d = 9.0, tau = 0.0,
                rho_m = 0.5, rho_z = 0.5, rho_a = 0.9,
                sigma_m = 0.0025, sigma_z = 0.005, sigma_a = 0.01
    endogenous: C, W_real, Pi, A, N, R, realinterest, Y, Q, Z,
                S, Pi_star, x_aux_1, x_aux_2, MC, M_real,
                i_ann, pi_ann, r_real_ann, P, log_m_nominal,
                log_y, log_W_real, log_N, log_P, log_A, log_Z,
                money_growth, money_growth_ann
    exogenous: eps_a, eps_z, eps_m

    # (1) FOC Wages
    W_real[t] = C[t]^siggma * N[t]^varphi

    # (2) Euler equation
    Q[t] = betta * (C[t+1]/C[t])^(-siggma) * (Z[t+1]/Z[t]) / Pi[t+1]

    # (3) Nominal interest rate
    R[t] = 1/Q[t]

    # (4) Aggregate output (with price dispersion)
    Y[t] = A[t] * (N[t] / S[t])^(1 - alppha)

    # (5) Real interest rate
    R[t] = realinterest[t] * Pi[t+1]

    # (6) Market clearing
    C[t] = Y[t]

    # (7) Technology shock
    log(A[t]) = rho_a * log(A[t-1]) + sigma_a * eps_a[t]

    # (8) Preference shock (MINUS sign as in Dynare .mod)
    log(Z[t]) = rho_z * log(Z[t-1]) - sigma_z * eps_z[t]

    # (9) Money growth definition
    money_growth[t] = log(M_real[t] / M_real[t-1] * Pi[t])

    # (10) Money growth exogenous process
    money_growth[t] = rho_m * money_growth[t-1] + sigma_m * eps_m[t]

    # (11) Annualized money growth
    money_growth_ann[t] = 4 * money_growth[t]

    # (12) Marginal cost
    MC[t] = W_real[t] / ((1 - alppha) * Y[t] / N[t] * S[t])

    # (13) LOM prices
    1 = theta * Pi[t]^(epsilon_d - 1) + (1 - theta) * Pi_star[t]^(1 - epsilon_d)

    # (14) LOM price dispersion
    S[t] = (1 - theta) * Pi_star[t]^(-epsilon_d/(1 - alppha)) + theta * Pi[t]^(epsilon_d/(1 - alppha)) * S[t-1]

    # (15) FOC price setting
    Pi_star[t]^(1 + epsilon_d * alppha / (1 - alppha)) = x_aux_1[t] / x_aux_2[t] * (1 - tau) * epsilon_d / (epsilon_d - 1)

    # (16) Auxiliary recursion 1
    x_aux_1[t] = Z[t] * C[t]^(-siggma) * Y[t] * MC[t] + betta * theta * Pi[t+1]^(epsilon_d + alppha * epsilon_d / (1 - alppha)) * x_aux_1[t+1]

    # (17) Auxiliary recursion 2
    x_aux_2[t] = Z[t] * C[t]^(-siggma) * Y[t] + betta * theta * Pi[t+1]^(epsilon_d - 1) * x_aux_2[t+1]

    # (18) log output
    log_y[t] = log(Y[t])

    # (19) log real wage
    log_W_real[t] = log(W_real[t])

    # (20) log hours
    log_N[t] = log(N[t])

    # (21) Annualized inflation
    pi_ann[t] = 4 * log(Pi[t])

    # (22) Annualized nominal interest rate
    i_ann[t] = 4 * log(R[t])

    # (23) Annualized real interest rate
    r_real_ann[t] = 4 * log(realinterest[t])

    # (24) Real money demand
    M_real[t] = Y[t] / R[t]^eta_money

    # (25) Log nominal money stock
    log_m_nominal[t] = log(M_real[t] * P[t])

    # (26) Price level definition
    Pi[t] = P[t] / P[t-1]

    # (27) Log price level
    log_P[t] = log(P[t])

    # (28) Log TFP
    log_A[t] = log(A[t])

    # (29) Log preference
    log_Z[t] = log(Z[t])

    steady_state = begin
        A_ss = 1.0; Z_ss = 1.0; S_ss = 1.0; Pi_star_ss = 1.0; P_ss = 1.0
        MC_val = (epsilon_d - 1) / epsilon_d / (1 - tau)
        R_ss = 1.0 / betta; Pi_ss = 1.0; Q_ss = betta; ri_ss = R_ss
        N_ss = ((1 - alppha) * MC_val)^(1 / ((1 - siggma) * alppha + varphi + siggma))
        C_ss = A_ss * N_ss^(1 - alppha)
        W_ss = C_ss^siggma * N_ss^varphi
        Y_ss = C_ss
        mg_ss = 0.0; mga_ss = 0.0
        x1_ss = C_ss^(-siggma) * Y_ss * MC_val / (1 - betta * theta * Pi_ss^(epsilon_d / (1 - alppha)))
        x2_ss = C_ss^(-siggma) * Y_ss / (1 - betta * theta * Pi_ss^(epsilon_d - 1))
        M_real_ss = Y_ss / R_ss^eta_money
        log_y_ss = log(Y_ss)
        log_W_real_ss = log(W_ss)
        log_N_ss = log(N_ss)
        pi_ann_ss = 4 * log(Pi_ss)
        i_ann_ss = 4 * log(R_ss)
        r_real_ann_ss = 4 * log(ri_ss)
        log_m_nominal_ss = log(M_real_ss * P_ss)
        log_P_ss = 0.0; log_A_ss = 0.0; log_Z_ss = 0.0
        # C, W_real, Pi, A, N, R, realinterest, Y, Q, Z,
        # S, Pi_star, x_aux_1, x_aux_2, MC, M_real,
        # i_ann, pi_ann, r_real_ann, P, log_m_nominal,
        # log_y, log_W_real, log_N, log_P, log_A, log_Z,
        # money_growth, money_growth_ann
        [C_ss, W_ss, Pi_ss, A_ss, N_ss, R_ss, ri_ss, Y_ss, Q_ss, Z_ss,
         S_ss, Pi_star_ss, x1_ss, x2_ss, MC_val, M_real_ss,
         i_ann_ss, pi_ann_ss, r_real_ann_ss, P_ss, log_m_nominal_ss,
         log_y_ss, log_W_real_ss, log_N_ss, log_P_ss, log_A_ss, log_Z_ss,
         mg_ss, mga_ss]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 15)
ss = sol.spec.steady_state

println("=" ^ 60)
println("  Gali (2015) Chapter 3 — Baseline NK (Nonlinear)")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "gali_2015_ch3_nl.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
d_names_raw = data["endo_names"]
d_names = d_names_raw isa Matrix ? vec(d_names_raw) : d_names_raw
irfs = data["irfs"]

# Build Dynare name -> index mapping
dynare_idx = Dict{String,Int}()
for (i, n) in enumerate(d_names)
    dynare_idx[strip(string(n))] = i
end

let
    # Steady state comparison — match our variables to Dynare
    our_names = ["C", "W_real", "Pi", "A", "N", "R", "realinterest", "Y", "Q", "Z",
                 "S", "Pi_star", "x_aux_1", "x_aux_2", "MC", "M_real",
                 "i_ann", "pi_ann", "r_real_ann", "P", "log_m_nominal",
                 "log_y", "log_W_real", "log_N", "log_P", "log_A", "log_Z",
                 "money_growth", "money_growth_ann"]

    println("\n=== Steady State ===")
    ss_pass = true
    for (j_idx, name) in enumerate(our_names)
        haskey(dynare_idx, name) || continue
        d_idx = dynare_idx[name]
        j_val = ss[j_idx]
        d_val = d_ss[d_idx]
        diff = abs(j_val - d_val)
        ok = diff < 1e-6
        ss_pass = ss_pass && ok
        @printf("  %-15s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── IRFs ──
    # Dynare IRFs are linear deviations from steady state for a 1-std-dev shock.
    # Since we embedded sigma_m, sigma_z, sigma_a in the equations, our unit-shock IRFs
    # already incorporate the scaling. So our IRFs should match Dynare directly.
    #
    # Dynare .mat has IRFs from 3 separate stoch_simul runs:
    # Run 1: eps_m (sigma=0.0025), Run 2: eps_z (sigma=0.005), Run 3: eps_a (sigma=0.01)
    # Our shock order: eps_a(1), eps_z(2), eps_m(3)

    # Map: Julia var name -> Julia var index for log-definition vars
    j_idx_map = Dict(n => i for (i, n) in enumerate(our_names))

    println("\n=== IRFs ===")
    irf_pass = true

    # Technology shock (eps_a is shock 1 in our model)
    irf_checks_a = [
        ("log_y_eps_a", "log_y", 1),
        ("log_N_eps_a", "log_N", 1),
        ("log_W_real_eps_a", "log_W_real", 1),
        ("pi_ann_eps_a", "pi_ann", 1),
        ("i_ann_eps_a", "i_ann", 1),
        ("r_real_ann_eps_a", "r_real_ann", 1),
        ("log_A_eps_a", "log_A", 1),
        ("log_m_nominal_eps_a", "log_m_nominal", 1),
        ("log_P_eps_a", "log_P", 1),
    ]
    # Preference shock (eps_z is shock 2, but enters with minus sign)
    irf_checks_z = [
        ("log_y_eps_z", "log_y", 2),
        ("log_N_eps_z", "log_N", 2),
        ("log_W_real_eps_z", "log_W_real", 2),
        ("pi_ann_eps_z", "pi_ann", 2),
        ("i_ann_eps_z", "i_ann", 2),
        ("r_real_ann_eps_z", "r_real_ann", 2),
        ("log_Z_eps_z", "log_Z", 2),
        ("log_m_nominal_eps_z", "log_m_nominal", 2),
        ("log_P_eps_z", "log_P", 2),
    ]
    # Monetary shock (eps_m is shock 3)
    irf_checks_m = [
        ("log_y_eps_m", "log_y", 3),
        ("log_N_eps_m", "log_N", 3),
        ("log_W_real_eps_m", "log_W_real", 3),
        ("pi_ann_eps_m", "pi_ann", 3),
        ("i_ann_eps_m", "i_ann", 3),
        ("r_real_ann_eps_m", "r_real_ann", 3),
        ("money_growth_ann_eps_m", "money_growth_ann", 3),
        ("log_m_nominal_eps_m", "log_m_nominal", 3),
        ("log_P_eps_m", "log_P", 3),
    ]

    for irf_checks in [irf_checks_a, irf_checks_z, irf_checks_m]
        for (d_key, j_name, s_idx) in irf_checks
            haskey(irfs, d_key) || continue
            haskey(j_idx_map, j_name) || continue
            j_idx = j_idx_map[j_name]
            d_vals = vec(irfs[d_key])
            H = min(length(d_vals), 15)
            j_vals = [ir.values[h, j_idx, s_idx] for h in 1:H]
            max_diff = maximum(abs.(j_vals .- d_vals[1:H]))
            scale = max(maximum(abs.(d_vals[1:H])), 1e-10)
            rel_diff = max_diff / scale
            ok = max_diff < 1e-4 || (max_diff < 1e-3 && rel_diff < 0.01)
            irf_pass = irf_pass && ok
            @printf("  %-30s  max|diff|=%8.2e  rel=%8.2e  %s\n",
                    d_key, max_diff, rel_diff, ok ? "PASS" : "FAIL")
        end
    end

    println("\n  Overall: SS=", ss_pass ? "PASS" : "FAIL",
            ", IRF=", irf_pass ? "PASS" : "FAIL")
    println("  ", (ss_pass && irf_pass) ? "ALL PASS" : "SOME FAIL")
end
