# Tier 1: Ascari & Sbordone (2014) — NK with Trend Inflation
# "The Macroeconomics of Trend Inflation", JEL, 52(3), 679-739
# Dynare source: DSGE_mod/Ascari_Sbordone_2014/Ascari_Sbordone_2014.mod
# Mat ref: dynare_results/ascari_sbordone_2014.mat
#
# Configuration: MP_shock=1, log_utility=1, trend_inflation=0
# Parameters: phi_pi=2, phi_y=0.125, rho_i=0.8, phi_par=1 (Frisch=1)
# Shocks: e_v(stderr=1), e_a(stderr=0), e_zeta(stderr=0)
#
# NOTE: The Dynare .mat file has INCONSISTENT SS and IRFs because the .mod file
# runs many stoch_simul calls with different trend_inflation values. The .mat SS
# is from trend_inflation=8 (last steady call in a loop), while the .mat IRFs are
# from trend_inflation=6 (last stoch_simul with irf>0). We verify:
#   1. SS computation against trend_inflation=8 reference
#   2. Model solves and IRFs have correct qualitative properties
#   3. Self-consistency: our trend_inflation=0 solution is internally valid
using MacroEconometricModels, MAT, Printf

# ── Parameters ──
const _beta = 0.99
const _alpha = 0.0
const _theta = 0.75
const _epsilon = 10.0
const _sigma = 1.0
const _phi_par = 1.0
const _var_rho = 0.0
const _rho_v = 0.0
const _rho_a = 0.0
const _rho_zeta = 0.0
const _phi_pi = 2.0
const _phi_y = 0.125
const _rho_i = 0.8

# ── d_n from zero-inflation benchmark ──
# phi = y^(1-sigma)/(1-theta*beta) = (1/3)^0 / 0.2575 = 3.8834951..
# psi = p_star * phi / (epsilon/((epsilon-1)*(1-alpha))) = 1 * phi / (10/9) = phi * 0.9 = 3.4951456..
# w = psi * (1-theta*beta) / (y^(1-sigma)) = psi * 0.2575 / 1 = 0.9
# d_n = w / (N^phi * y^sigma) = 0.9 / ((1/3)^1 * (1/3)^1) = 0.9 / (1/9) = 8.1
const _d_n = 8.1

# ── Steady state computation function ──
function compute_ascari_ss(trend_inflation; beta=_beta, alpha=_alpha, theta=_theta,
                           epsilon=_epsilon, sigma=_sigma, phi_par=_phi_par,
                           var_rho=_var_rho, d_n=_d_n)
    Pi_bar = (1 + trend_inflation/100)^(1/4)
    i_lev = 1/beta * Pi_bar - 1
    i_bar = i_lev
    p_star_lev = ((1 - theta*Pi_bar^((epsilon-1)*(1-var_rho)))/(1-theta))^(1/(1-epsilon))
    s_lev = (1-theta)/(1-theta*Pi_bar^((epsilon*(1-var_rho))/(1-alpha))) * p_star_lev^(-epsilon/(1-alpha))
    y_lev = (p_star_lev^(1+(epsilon*alpha)/(1-alpha)) * (epsilon/((epsilon-1)*(1-alpha)) * ((1-beta*theta*Pi_bar^((epsilon-1)*(1-var_rho)))/(1-beta*theta*Pi_bar^(epsilon*(1-var_rho)/(1-alpha)))) * d_n * s_lev^phi_par)^(-1))^(((phi_par+1)/(1-alpha)-(1-sigma))^(-1))
    N_lev = s_lev * y_lev^(1/(1-alpha))
    Y_bar = y_lev
    phi_lev = y_lev^(1-sigma) / (1-theta*beta*Pi_bar^((epsilon-1)*(1-var_rho)))
    psi_lev = p_star_lev^(1+epsilon*alpha/(1-alpha)) * phi_lev / (epsilon/((epsilon-1)*(1-alpha)))
    w_lev = (psi_lev - theta*beta*Pi_bar^((-var_rho*epsilon)/(1-alpha))*Pi_bar^(epsilon/(1-alpha))*psi_lev) / (1.0^(-1/(1-alpha)) * y_lev^((1/(1-alpha))-sigma))
    MC_lev = 1/(1-alpha) * w_lev * 1.0^(1/(alpha-1)) * y_lev^(alpha/(1-alpha))
    ri_lev = (1+i_lev)/Pi_bar
    A_tilde_lev = 1.0 / s_lev
    Avg_lev = 1/MC_lev
    Marg_lev = p_star_lev/MC_lev
    pag_lev = 1/p_star_lev
    Utility_val = (1-beta)^(-1) * (log(y_lev) - d_n * N_lev^(1+phi_par)/(1+phi_par))

    ss_log = [log(y_lev), log(i_lev), log(Pi_bar), log(N_lev), log(w_lev), log(p_star_lev),
              log(psi_lev), log(phi_lev), 0.0, log(MC_lev), log(ri_lev),
              0.0, log(s_lev), 0.0, log(A_tilde_lev), Utility_val,
              log(Avg_lev), log(Marg_lev), log(pag_lev)]
    return (ss_log=ss_log, Pi_bar=Pi_bar, i_bar=i_bar, Y_bar=Y_bar)
end

# trend_inflation=0 for our model
const _ss0 = compute_ascari_ss(0.0)
# trend_inflation=8 for .mat SS verification
const _ss8 = compute_ascari_ss(8.0)

spec = @dsge begin
    parameters: beta_disc = 0.99, alpha_val = 0.0, theta_val = 0.75, epsilon_d = 10.0,
                sigma_c = 1.0, phi_par_val = 1.0, var_rho_val = 0.0,
                rho_v = 0.0, rho_a = 0.0, rho_zeta = 0.0,
                phi_pi = 2.0, phi_y = 0.125, rho_i = 0.8,
                Pi_bar_p = 1.0, i_bar_p = 0.010101010101010102,
                Y_bar_p = 0.3333333333333333, d_n_p = 8.1
    endogenous: y, i, pi, N, w, p_star, psi, phi_v,
                A, MC_real, real_interest, zeta, s, v,
                A_tilde, Utility, Average_markup, Marginal_markup,
                price_adjustment_gap
    exogenous: e_v, e_a, e_zeta

    # (1) Euler equation
    1/(exp(y[t])^sigma_c) = beta_disc * (1 + exp(i[t])) / (exp(pi[t+1]) * exp(y[t+1])^sigma_c)

    # (2) Labor FOC
    exp(w[t]) = d_n_p * exp(zeta[t]) * exp(N[t])^phi_par_val * exp(y[t])^sigma_c

    # (3) Optimal price
    exp(p_star[t]) = ((1 - theta_val * exp(pi[t-1])^((1-epsilon_d)*var_rho_val) * exp(pi[t])^(epsilon_d-1)) / (1 - theta_val))^(1/(1-epsilon_d))

    # (4) FOC price setting
    exp(p_star[t])^(1 + epsilon_d*alpha_val/(1-alpha_val)) = (epsilon_d/((epsilon_d-1)*(1-alpha_val))) * exp(psi[t]) / exp(phi_v[t])

    # (5) Recursive LOM psi
    exp(psi[t]) = exp(w[t]) * exp(A[t])^(-1/(1-alpha_val)) * exp(y[t])^(1/(1-alpha_val) - sigma_c) + theta_val * beta_disc * exp(pi[t])^((-var_rho_val*epsilon_d)/(1-alpha_val)) * exp(pi[t+1])^(epsilon_d/(1-alpha_val)) * exp(psi[t+1])

    # (6) Recursive LOM phi
    exp(phi_v[t]) = exp(y[t])^(1-sigma_c) + theta_val * beta_disc * exp(pi[t])^(var_rho_val*(1-epsilon_d)) * exp(pi[t+1])^(epsilon_d-1) * exp(phi_v[t+1])

    # (7) Aggregate production
    exp(N[t]) = exp(s[t]) * (exp(y[t]) / exp(A[t]))^(1/(1-alpha_val))

    # (8) LOM price dispersion
    exp(s[t]) = (1-theta_val) * exp(p_star[t])^(-epsilon_d/(1-alpha_val)) + theta_val * exp(pi[t-1])^((-epsilon_d*var_rho_val)/(1-alpha_val)) * exp(pi[t])^(epsilon_d/(1-alpha_val)) * exp(s[t-1])

    # (9) Monetary policy rule (Taylor rule with interest rate smoothing)
    (1+exp(i[t]))/(1+i_bar_p) = ((1+exp(i[t-1]))/(1+i_bar_p))^rho_i * ((exp(pi[t])/Pi_bar_p)^phi_pi * (exp(y[t])/Y_bar_p)^phi_y)^(1-rho_i) * exp(v[t])

    # (10) Real marginal costs
    exp(MC_real[t]) = 1/(1-alpha_val) * exp(w[t]) * exp(A[t])^(1/(alpha_val-1)) * exp(y[t])^(alpha_val/(1-alpha_val))

    # (11) Real interest rate
    exp(real_interest[t]) = (1 + exp(i[t])) / exp(pi[t+1])

    # (12) Utility (log case)
    Utility[t] = y[t] - d_n_p * exp(zeta[t]) * exp(N[t])^(1+phi_par_val) / (1+phi_par_val) + beta_disc * Utility[t+1]

    # (13-15) Shock processes
    v[t] = rho_v * v[t-1] + e_v[t]
    A[t] = rho_a * A[t-1] + e_a[t]
    zeta[t] = rho_zeta * zeta[t-1] + e_zeta[t]

    # (16-19) Definitions
    exp(A_tilde[t]) = exp(A[t]) / exp(s[t])
    exp(Average_markup[t]) = 1 / exp(MC_real[t])
    exp(Marginal_markup[t]) = exp(p_star[t]) / exp(MC_real[t])
    exp(price_adjustment_gap[t]) = 1 / exp(p_star[t])

    steady_state = begin
        y_lev = 1.0/3; i_lev = 1.0/beta_disc - 1; pi_lev = 1.0
        s_lev = 1.0; p_star_lev = 1.0; N_lev = 1.0/3
        phi_lev = y_lev^(1-sigma_c) / (1-theta_val*beta_disc)
        psi_lev = p_star_lev * phi_lev / (epsilon_d/((epsilon_d-1)*(1-alpha_val)))
        w_lev = psi_lev * (1-theta_val*beta_disc) / (y_lev^(1/(1-alpha_val)-sigma_c))
        MC_lev = w_lev; ri_lev = (1+i_lev)/pi_lev
        A_tilde_lev = 1.0/s_lev
        Util = (1-beta_disc)^(-1) * (log(y_lev) - d_n_p * N_lev^(1+phi_par_val)/(1+phi_par_val))
        [log(y_lev), log(i_lev), log(pi_lev), log(N_lev), log(w_lev), log(p_star_lev),
         log(psi_lev), log(phi_lev), 0.0, log(MC_lev), log(ri_lev),
         0.0, log(s_lev), 0.0, log(A_tilde_lev), Util,
         log(1/MC_lev), log(p_star_lev/MC_lev), log(1/p_star_lev)]
    end
end

sol = solve(spec; method=:gensys)
ir = irf(sol, 7)
ss = sol.spec.steady_state

println("=" ^ 60)
println("  Ascari & Sbordone (2014) — NK with Trend Inflation")
println("=" ^ 60)
println("  determined = ", is_determined(sol))

# ── Load Dynare reference ──
dynare_mat = joinpath(@__DIR__, "dynare_results", "ascari_sbordone_2014.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found — run run_dynare_reference.m first)")
    exit(0)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
d_names_raw = data["endo_names"]
d_names = d_names_raw isa Matrix ? vec(d_names_raw) : d_names_raw
irfs = data["irfs"]

dynare_idx = Dict{String,Int}()
for (i, n) in enumerate(d_names)
    dynare_idx[strip(string(n))] = i
end

let
    our_names = ["y","i","pi","N","w","p_star","psi","phi","A","MC_real","real_interest",
                 "zeta","s","v","A_tilde","Utility","Average_markup","Marginal_markup",
                 "price_adjustment_gap"]

    # ── (1) Verify SS computation: trend_inflation=8 vs .mat ──
    println("\n=== SS Verification: Our trend_inflation=8 SS vs .mat ===")
    ss8_log = _ss8.ss_log
    ss8_pass = true
    for (j_idx, name) in enumerate(our_names)
        haskey(dynare_idx, name) || continue
        d_idx = dynare_idx[name]
        j_val = ss8_log[j_idx]
        d_val = d_ss[d_idx]
        diff = abs(j_val - d_val)
        ok = diff < 1e-6
        ss8_pass = ss8_pass && ok
        @printf("  %-20s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
                name, j_val, d_val, diff, ok ? "PASS" : "FAIL")
    end

    # ── (2) Verify model solution with trend_inflation=0 ──
    println("\n=== Model Solution: trend_inflation=0 ===")
    println("  Model solves with determinacy = ", is_determined(sol))
    println("  SS vector (first 5): ", round.(ss[1:5]; digits=8))

    # Verify our SS matches the computed SS for trend_inflation=0
    ss0_log = _ss0.ss_log
    ss0_pass = true
    for (j_idx, name) in enumerate(our_names)
        j_val = ss[j_idx]
        c_val = ss0_log[j_idx]
        diff = abs(j_val - c_val)
        ok = diff < 1e-10
        ss0_pass = ss0_pass && ok
    end
    println("  SS matches computed SS (trend_inflation=0): ", ss0_pass ? "PASS" : "FAIL")

    # ── (3) IRF qualitative checks ──
    println("\n=== IRFs (trend_inflation=0, MP shock) ===")
    println("  NOTE: .mat IRFs come from a DIFFERENT trend_inflation (=6 in the")
    println("  Dynare .mod file's parameter sweep). Direct comparison not possible.")
    println("  Verifying qualitative properties instead:")

    # v_e_v should match exactly (same process regardless of trend_inflation)
    v_match = haskey(irfs, "v_e_v") && maximum(abs.([ir.values[h,14,1] for h in 1:7] .- vec(irfs["v_e_v"])[1:7])) < 1e-10
    println("  v_e_v matches .mat: ", v_match ? "PASS" : "FAIL")

    # Qualitative: contractionary monetary shock (positive v) should decrease output
    y_negative = ir.values[1,1,1] < 0
    println("  y responds negatively to MP shock: ", y_negative ? "PASS" : "FAIL")

    # Qualitative: inflation falls on impact
    pi_negative = ir.values[1,3,1] < 0
    println("  pi responds negatively to MP shock: ", pi_negative ? "PASS" : "FAIL")

    # Qualitative: real interest rate rises
    ri_positive = ir.values[1,11,1] > 0
    println("  real_interest rises on MP shock: ", ri_positive ? "PASS" : "FAIL")

    # With trend_inflation=0, s should not move (price dispersion = 0 at zero inflation)
    s_zero = maximum(abs.([ir.values[h,13,1] for h in 1:7])) < 1e-8
    println("  s unchanged at zero trend inflation: ", s_zero ? "PASS" : "FAIL")

    # Print IRFs for reference
    println("\n  IRF values (trend_inflation=0):")
    for (name, idx) in [("y",1), ("i",2), ("pi",3), ("real_interest",11), ("s",13), ("v",14)]
        vals = [ir.values[h, idx, 1] for h in 1:7]
        @printf("  %-20s  %s\n", name, join([@sprintf("%.4f", v) for v in vals], "  "))
    end

    all_pass = ss8_pass && v_match && y_negative && pi_negative && ri_positive && s_zero
    println("\n  Overall: SS(trend_infl=8)=", ss8_pass ? "PASS" : "FAIL",
            ", Qualitative=", (v_match && y_negative && pi_negative && ri_positive && s_zero) ? "PASS" : "FAIL")
    println("  ", all_pass ? "ALL PASS" : "SOME FAIL")
end
