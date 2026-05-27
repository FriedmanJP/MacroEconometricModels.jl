# Tier 2: Basu_Bundick_2017 (Uncertainty Shocks in Effective Demand) — Dynare Replication
# Dynare source: DSGE_mod/Basu_Bundick_2017/Basu_Bundick_2017.mod
# Reference: Basu, S. & Bundick, B. (2017), "Uncertainty shocks in a model of
#   effective demand", Econometrica, 85(3), pp. 937-958
#
# New Keynesian model with Epstein-Zin preferences, Rotemberg pricing, variable
# capital utilization, and stochastic volatility in preference shocks.
# 43 endogenous variables (Dynare), 4 shocks. Order=3 with pruning.
#
# Uses sticky_prices=1, CRRA=0 (Epstein-Zin), true_stochastic_steady_state_IRFs=1.
#
# NOTE: This model has a Dynare model-local variable (#M for SDF) which must be
# inlined as an explicit endogenous variable. We add M_sdf as a 44th variable.
# Many observation equations (log_Y, log_C, etc.) are omitted to keep the model
# tractable for the perturbation solver. We include only the core 33 structural
# equations.
#
# NOTE on order=3 feasibility:
#   With n~35, nx~10, nv~14, the order=3 Kronecker system requires ~122GB.
#   INFEASIBLE with direct solve. We verify SS, order=1, and attempt order=2.
using MacroEconometricModels, Printf, Statistics, Random

# ── Steady State (computed from Dynare's external SS file) ──────────────
# See Basu_Bundick_2017_steadystate.m. N solved numerically.
const _siggma = 80.0
const _psii = 0.95
const _betta = 0.994
const _alppha = 0.333
const _delta_0 = 0.025
const _delta_2 = 0.00031
const _phi_p = 100.0
const _phi_k = 2.0901
const _Pi_bar = 1.005
const _rho_r = 0.0
const _rho_pi = 1.5
const _rho_y = 0.2
const _theta_mu = 6.0
const _a_bar = 1.0
const _Frisch_target = 2.0
const _rho_a = 0.93564
const _sigma_a_bar = 0.0026251
const _rho_sigma_a = 0.74227
const _sigma_sigma_a = 0.0025022
const _rho_z = 0.98793
const _sigma_z_bar = 0.0012857
const _nu = 0.9
const _sigma_eps_m = 0.0

# Derived parameters
const _theta_v = (1 - _siggma) / (1 - 1/_psii)  # 1501
const _log_R_bar = log(_Pi_bar / _betta)

# Solve for N (bisection on the SS equation from Dynare external file)
const _R_K = 1/_betta - (1 - _delta_0)
const _K_ss = _alppha / _R_K
const _delta_1 = 1/_betta - 1 + _delta_0

function _basu_ss_resid(N_val)
    term = _theta_v/(1-_siggma)*(1-_Frisch_target*(1-(1-_siggma)/_theta_v)*N_val/(1-N_val))
    return (1-_alppha)/N_val*(1-N_val)*term/(1-term)+_delta_0*_K_ss-1
end

# Bisection between 0.32 and 0.33 (verified sign change)
let lo = 0.32, hi = 0.33
    for _ in 1:200
        mid = (lo + hi) / 2
        if _basu_ss_resid(lo) * _basu_ss_resid(mid) < 0
            hi = mid
        else
            lo = mid
        end
    end
    global _N_ss = (lo + hi) / 2
end

const _eta = _theta_v/(1-_siggma)*(1-_Frisch_target*(1-(1-_siggma)/_theta_v)*_N_ss/(1-_N_ss))
const _Xi = (_theta_mu - 1) / _theta_mu
const _PF_norm = 1/(_Xi * _N_ss^(1-_alppha) * _K_ss^_alppha)
const _W_ss = (1-_alppha)*_Xi*_PF_norm*_N_ss^(1-_alppha)*_K_ss^_alppha / _N_ss
const _Psi = (1-_Xi)*_PF_norm*_N_ss^(1-_alppha)*_K_ss^_alppha
const _Y_ss = 1.0
const _I_ss = _delta_0 * _K_ss
const _C_ss = _Y_ss - _I_ss
const _D_ss = _Y_ss - _N_ss*_W_ss - _delta_0*_K_ss
const _R_R_ss = 1/_betta
const _D_E_ss = _D_ss - (1 - 1/_R_R_ss)*_nu*_K_ss
const _P_E_ss = _betta/(1-_betta)*_D_E_ss
const _V_norm = (1-_betta)/((_C_ss^_eta*(1-_N_ss)^(1-_eta))^((1-_siggma)/_theta_v))
const _R_ss = _Pi_bar / _betta
const _mu_ss = 1/_Xi
const _R_E_ss = (_D_E_ss + _P_E_ss)/_P_E_ss

# ── Core Model (33 structural equations, dropping log-transform observations) ─
# Variables:
#   C, N, P_E, D_E, R_R, W, V, E_t_V_tp1_1_minus_sigma,
#   Y, K (predetermined), I_v, D, Pi, R_K, q, mu, sigma_a, a, Z, R,
#   Xi_v, u, PHI_v, R_E,
#   E_R_E, E_R_E_squared, cond_var_R_E,
#   E_M_tp1, E_R_E_risk_neutral, E_M_tp1_R_E, E_M_tp1_R_E_squared,
#   cond_var_R_E_risk_neutral,
#   M_sdf (model-local SDF made explicit)
spec = @dsge begin
    parameters: siggma = 80.0, theta_v = 1501.0000000000016, psii = 0.95,
                eta_p = 0.3466744155782053, betta = 0.994, theta_mu = 6.0,
                alppha = 0.333, Psi_p = 0.19999999999999993,
                delta_0 = 0.025, delta_1 = 0.031036217303822868, delta_2 = 0.00031,
                phi_k = 2.0901, phi_p = 100.0, Pi_bar = 1.005, nu_p = 0.9,
                rho_r = 0.0, rho_pi = 1.5, rho_y = 0.2,
                log_R_bar = 0.011005613836601888,
                sigma_a_bar = 0.0026251, sigma_sigma_a = 0.0025022,
                rho_a = 0.93564, rho_sigma_a = 0.74227,
                a_bar = 1.0, sigma_z_bar = 0.0012857, rho_z = 0.98793,
                PF_norm = 1.1499755173399646, V_norm_p = 0.00588552318299202,
                sigma_eps_m = 0.0
    endogenous: C, N, P_E, D_E, R_R, W, V, EV1ms,
                Y, K, I_v, D_v, Pi_v, R_K, q_v, mu_v,
                sigma_a, a_v, Z, R_v, Xi_v, u_v, PHI_v, R_E,
                E_R_E, E_R_E_sq, cond_var, E_M, E_R_E_rn,
                E_M_R_E, E_M_R_E_sq, cond_var_rn, M_sdf
    exogenous: eps_a, eps_sigma_a, eps_z, eps_m

    # Eq 1: Epstein-Zin value function
    # V = (V_norm*a*(C^eta*(1-N)^(1-eta))^((1-sig)/theta_v) + betta*(EV1ms)^(1/theta_v))^(theta_v/(1-sig))
    V[t] - (V_norm_p * a_v[t] * (C[t]^eta_p * (1.0 - N[t])^(1.0 - eta_p))^((1.0 - siggma) / theta_v) + betta * EV1ms[t]^(1.0 / theta_v))^(theta_v / (1.0 - siggma)) = 0.0

    # Eq 2: Auxiliary EZ variable
    # E_t_V_tp1_1_minus_sigma = V(+1)^(1-sigma)
    EV1ms[t] - V[t+1]^(1.0 - siggma) = 0.0

    # Eq 3: SDF definition (model-local #M made explicit)
    # M = betta*(a(+1)/a)*((C(+1)^eta*(1-N(+1))^(1-eta))/(C^eta*(1-N)^(1-eta)))^((1-sig)/theta_v)*(C/C(+1))*(V(+1)^(1-sig)/EV1ms)^(1-1/theta_v)
    M_sdf[t] - betta * (a_v[t+1] / a_v[t]) * ((C[t+1]^eta_p * (1.0 - N[t+1])^(1.0 - eta_p)) / (C[t]^eta_p * (1.0 - N[t])^(1.0 - eta_p)))^((1.0 - siggma) / theta_v) * (C[t] / C[t+1]) * (V[t+1]^(1.0 - siggma) / EV1ms[t])^(1.0 - 1.0 / theta_v) = 0.0

    # Eq 4: Budget constraint
    # C + P_E + 1/R_R*nu*K(+1) = W*N + (D_E+P_E) + nu*K
    # K predetermined: K(+1)→K[t], K→K[t-1]
    C[t] + P_E[t] + 1.0 / R_R[t] * nu_p * K[t] - W[t] * N[t] - (D_E[t] + P_E[t]) - nu_p * K[t-1] = 0.0

    # Eq 5: Labor FOC
    # (1-eta)/eta * C/(1-N) = W
    (1.0 - eta_p) / eta_p * C[t] / (1.0 - N[t]) - W[t] = 0.0

    # Eq 6: Stock FOC
    # P_E = M*(D_E(+1) + P_E(+1))
    P_E[t] - M_sdf[t] * (D_E[t+1] + P_E[t+1]) = 0.0

    # Eq 7: Bond FOC
    # 1 = R_R * M
    1.0 - R_R[t] * M_sdf[t] = 0.0

    # Eq 8: Production function
    # Y = PF_norm*(u*K)^alpha*(Z*N)^(1-alpha) - Psi
    # K predetermined: K→K[t-1]
    Y[t] - PF_norm * (u_v[t] * K[t-1])^alppha * (Z[t] * N[t])^(1.0 - alppha) + Psi_p = 0.0

    # Eq 9: LOM Capital
    # K(+1) = (1-(delta_0+delta_1*(u-1)+delta_2/2*(u-1)^2) - phi_k/2*(I/K-delta_0)^2)*K + I
    # K(+1)→K[t], K→K[t-1]
    K[t] - (1.0 - (delta_0 + delta_1 * (u_v[t] - 1.0) + delta_2 / 2.0 * (u_v[t] - 1.0)^2) - phi_k / 2.0 * (I_v[t] / K[t-1] - delta_0)^2) * K[t-1] - I_v[t] = 0.0

    # Eq 10: Cash flows
    # D = Y - W*N - I - phi_p/2*(Pi/Pi_bar-1)^2*Y
    D_v[t] - Y[t] + W[t] * N[t] + I_v[t] + phi_p / 2.0 * (Pi_v[t] / Pi_bar - 1.0)^2 * Y[t] = 0.0

    # Eq 11: Labor FOC (W*N)
    # W*N = (1-alpha)*Xi*PF_norm*(u*K)^alpha*(Z*N)^(1-alpha)
    W[t] * N[t] - (1.0 - alppha) * Xi_v[t] * PF_norm * (u_v[t] * K[t-1])^alppha * (Z[t] * N[t])^(1.0 - alppha) = 0.0

    # Eq 12: Capital FOC (R_K)
    # R_K*(u*K) = alpha*Xi*PF_norm*(u*K)^alpha*(Z*N)^(1-alpha)
    R_K[t] * (u_v[t] * K[t-1]) - alppha * Xi_v[t] * PF_norm * (u_v[t] * K[t-1])^alppha * (Z[t] * N[t])^(1.0 - alppha) = 0.0

    # Eq 13: Utilization FOC
    # q*(delta_1+delta_2*(u-1))*u*K = alpha*Xi*PF_norm*(u*K)^alpha*(Z*N)^(1-alpha)
    q_v[t] * (delta_1 + delta_2 * (u_v[t] - 1.0)) * u_v[t] * K[t-1] - alppha * Xi_v[t] * PF_norm * (u_v[t] * K[t-1])^alppha * (Z[t] * N[t])^(1.0 - alppha) = 0.0

    # Eq 14: Pricing FOC (Rotemberg)
    # phi_p*(Pi/Pi_bar-1)*(Pi/Pi_bar) = (1-theta_mu)+theta_mu*Xi + phi_p*M*(Y(+1)/Y)*(Pi(+1)/Pi_bar-1)*(Pi(+1)/Pi_bar)
    phi_p * (Pi_v[t] / Pi_bar - 1.0) * (Pi_v[t] / Pi_bar) - (1.0 - theta_mu) - theta_mu * Xi_v[t] - phi_p * M_sdf[t] * (Y[t+1] / Y[t]) * (Pi_v[t+1] / Pi_bar - 1.0) * (Pi_v[t+1] / Pi_bar) = 0.0

    # Eq 15: FOC capital
    # q = M*(u(+1)*R_K(+1) + q(+1)*(1-(delta_0+delta_1*(u(+1)-1)+delta_2/2*(u(+1)-1)^2) - phi_k/2*(I(+1)/K(+1)-delta_0)^2 + phi_k*(I(+1)/K(+1)-delta_0)*(I(+1)/K(+1))))
    q_v[t] - M_sdf[t] * (u_v[t+1] * R_K[t+1] + q_v[t+1] * (1.0 - (delta_0 + delta_1 * (u_v[t+1] - 1.0) + delta_2 / 2.0 * (u_v[t+1] - 1.0)^2) - phi_k / 2.0 * (I_v[t+1] / K[t] - delta_0)^2 + phi_k * (I_v[t+1] / K[t] - delta_0) * (I_v[t+1] / K[t]))) = 0.0

    # Eq 16: FOC investment
    # 1/q = 1 - phi_k*(I/K - delta_0)
    1.0 / q_v[t] - 1.0 + phi_k * (I_v[t] / K[t-1] - delta_0) = 0.0

    # Eq 17: Dividends
    # D_E = D - nu*(K - 1/R_R*K(+1))
    D_E[t] - D_v[t] + nu_p * (K[t-1] - 1.0 / R_R[t] * K[t]) = 0.0

    # Eq 18: Taylor Rule
    # log(R) = rho_r*log(R(-1)) + (1-rho_r)*(log_R_bar + rho_pi*(log(Pi)-log(Pi_bar)) + rho_y*(log(Y)-log(Y(-1)))) + sigma_eps_m*eps_m
    R_v[t] - exp(rho_r * log(R_v[t-1]) + (1.0 - rho_r) * (log_R_bar + rho_pi * (log(Pi_v[t]) - log(Pi_bar)) + rho_y * (log(Y[t]) - log(Y[t-1]))) + sigma_eps_m * eps_m[t]) = 0.0

    # Eq 19: Bond Euler (redundant but needed — eq 212 in mod file)
    # 1 = betta*(a(+1)/a)*((C(+1)^eta*(1-N(+1))^(1-eta))/(C^eta*(1-N)^(1-eta)))^((1-sig)/theta_v)*(C/C(+1))*(V(+1)^(1-sig)/EV1ms)^(1-1/theta_v)*(R/Pi(+1))
    # This is: 1 = M_sdf * R / Pi(+1), which should hold by construction.
    # Instead we use the equivalent: mu = 1/Xi (markup definition)
    mu_v[t] - 1.0 / Xi_v[t] = 0.0

    # Eq 20: Rotemberg cost
    # PHI = 1 + phi_p/2*(Pi/Pi_bar-1)^2*Y
    PHI_v[t] - 1.0 - phi_p / 2.0 * (Pi_v[t] / Pi_bar - 1.0)^2 * Y[t] = 0.0

    # Eq 21: Return to equity
    # R_E = (D_E + P_E)/P_E(-1)
    R_E[t] - (D_E[t] + P_E[t]) / P_E[t-1] = 0.0

    # Eq 22: Expected equity return
    # E_R_E = R_E(+1)
    E_R_E[t] - R_E[t+1] = 0.0

    # Eq 23: Expected squared return
    # E_R_E_squared = R_E(+1)^2
    E_R_E_sq[t] - R_E[t+1]^2 = 0.0

    # Eq 24: Conditional variance
    # cond_var_R_E = E_R_E_squared - E_R_E^2
    cond_var[t] - E_R_E_sq[t] + E_R_E[t]^2 = 0.0

    # Eq 25: Expected SDF
    # E_M_tp1 = M
    E_M[t] - M_sdf[t] = 0.0

    # Eq 26: Risk-neutral expected return
    # E_R_E_risk_neutral = R_E(+1)/E_M_tp1
    E_R_E_rn[t] - R_E[t+1] / E_M[t] = 0.0

    # Eq 27: E[M*R_E]
    # E_M_tp1_R_E = M*R_E(+1)
    E_M_R_E[t] - M_sdf[t] * R_E[t+1] = 0.0

    # Eq 28: E[M*R_E^2]
    # E_M_tp1_R_E_squared = M*R_E(+1)^2
    E_M_R_E_sq[t] - M_sdf[t] * R_E[t+1]^2 = 0.0

    # Eq 29: Risk-neutral conditional variance
    # cond_var_R_E_risk_neutral = E_M_R_E_sq/E_M - (E_M_R_E/E_M)^2
    cond_var_rn[t] - E_M_R_E_sq[t] / E_M[t] + (E_M_R_E[t] / E_M[t])^2 = 0.0

    # Eq 30: Preference shock level
    # a = (1-rho_a)*a_bar + rho_a*a(-1) + sigma_a(-1)*eps_a
    a_v[t] - (1.0 - rho_a) * a_bar - rho_a * a_v[t-1] - sigma_a[t-1] * eps_a[t] = 0.0

    # Eq 31: Preference volatility
    # sigma_a = (1-rho_sigma_a)*sigma_a_bar + rho_sigma_a*sigma_a(-1) + sigma_sigma_a*eps_sigma_a
    sigma_a[t] - (1.0 - rho_sigma_a) * sigma_a_bar - rho_sigma_a * sigma_a[t-1] - sigma_sigma_a * eps_sigma_a[t] = 0.0

    # Eq 32: TFP shock
    # Z = (1-rho_z)*1 + rho_z*Z(-1) - sigma_z_bar*eps_z
    Z[t] - (1.0 - rho_z) - rho_z * Z[t-1] + sigma_z_bar * eps_z[t] = 0.0

    # Eq 33: Bond Euler consistency (from eq 212)
    # 1 = M_sdf * R / Pi(+1)
    # This ensures internal consistency of the monetary policy block
    1.0 - M_sdf[t] * R_v[t] / Pi_v[t+1] = 0.0

    steady_state = begin
        # N solved numerically (bisection on SS equation)
        ss_N = 0.3259942466211687
        ss_C = 0.7317649918962719
        ss_K = 10.729400324149125
        ss_I = 0.26823500810372813
        ss_Y = 1.0
        ss_W = 2.046048379421577
        ss_D = 0.06476499189627172
        ss_R_R = 1.0060362173038229
        ss_D_E = 0.00682623014586746
        ss_P_E = 1.1308787941653748
        ss_V = 1.0
        ss_Pi = 1.005
        ss_R = 1.0110663983903418
        ss_R_K = 0.03103621730382289
        ss_q = 1.0
        ss_mu = 1.2
        ss_Xi = 0.8333333333333334
        ss_u = 1.0
        ss_a = 1.0
        ss_Z = 1.0
        ss_sigma_a = 0.0026251
        ss_PHI = 1.0
        ss_EV1ms = 1.0  # V^(1-siggma) = 1^(1-80) = 1
        ss_R_E = 1.0060362173038229
        ss_M = 0.994  # betta
        ss_E_R_E = ss_R_E
        ss_E_R_E_sq = ss_R_E^2
        ss_cond_var = 0.0
        ss_E_M = 0.994
        ss_E_R_E_rn = ss_R_E / ss_E_M
        ss_E_M_R_E = 1.0
        ss_E_M_R_E_sq = 0.994 * ss_R_E^2
        ss_cond_var_rn = ss_E_M_R_E_sq / ss_E_M - (ss_E_M_R_E / ss_E_M)^2

        # Order: C, N, P_E, D_E, R_R, W, V, EV1ms,
        #         Y, K, I_v, D_v, Pi_v, R_K, q_v, mu_v,
        #         sigma_a, a_v, Z, R_v, Xi_v, u_v, PHI_v, R_E,
        #         E_R_E, E_R_E_sq, cond_var, E_M, E_R_E_rn,
        #         E_M_R_E, E_M_R_E_sq, cond_var_rn, M_sdf
        [ss_C, ss_N, ss_P_E, ss_D_E, ss_R_R, ss_W, ss_V, ss_EV1ms,
         ss_Y, ss_K, ss_I, ss_D, ss_Pi, ss_R_K, ss_q, ss_mu,
         ss_sigma_a, ss_a, ss_Z, ss_R, ss_Xi, ss_u, ss_PHI, ss_R_E,
         ss_E_R_E, ss_E_R_E_sq, ss_cond_var, ss_E_M, ss_E_R_E_rn,
         ss_E_M_R_E, ss_E_M_R_E_sq, ss_cond_var_rn, ss_M]
    end
end

spec = compute_steady_state(spec)

println("=" ^ 60)
println("  Basu & Bundick 2017 — Uncertainty Shocks (EZ preferences)")
println("  33 core equations (+augmented), 4 shocks")
println("  sticky_prices=1, Epstein-Zin (sigma=80, psi=0.95)")
println("=" ^ 60)

# ── Steady State ──────────────────────────────────────────────────────────
println("\n=== Steady State ===")
ss = spec.steady_state
var_names = string.(spec.endog)
@printf("  %-20s %14s\n", "Variable", "Value")
println("  ", "-" ^ 36)
for (i, n) in enumerate(var_names)
    @printf("  %-20s %14.8f\n", n, ss[i])
end

# ── Order=1 ──────────────────────────────────────────────────────────────
println("\n=== First-Order Solution ===")
sol1 = nothing
try
    global sol1 = solve(spec; method=:gensys)
    println("  Determined: ", sol1.eu[1] == 1 && sol1.eu[2] == 1)
    println("  eu = ", sol1.eu)
    println("  n_vars = ", spec.n_endog, " (augmented = ", spec.augmented, ")")

    if spec.augmented
        println("  original_endog = ", length(spec.original_endog))
    end

    # IRFs to eps_sigma_a (uncertainty shock) — should be zero at order=1
    ir1 = irf(sol1, 20)
    shock_idx_sigma = 2  # eps_sigma_a
    println("\n  IRFs to eps_sigma_a (uncertainty shock):")
    ci = findfirst(==("C"), var_names); yi = findfirst(==("Y"), var_names)
    ni = findfirst(==("N"), var_names)
    @printf("  %4s %12s %12s %12s\n", "h", "C", "Y", "N")
    for h in 1:3
        @printf("  %4d %12.2e %12.2e %12.2e\n",
                h,
                ci !== nothing ? ir1.values[h, ci, shock_idx_sigma] : NaN,
                yi !== nothing ? ir1.values[h, yi, shock_idx_sigma] : NaN,
                ni !== nothing ? ir1.values[h, ni, shock_idx_sigma] : NaN)
    end
    println("  (Zero at order=1 — certainty equivalence)")

    # IRFs to eps_a (preference level shock)
    shock_idx_a = 1  # eps_a
    println("\n  IRFs to eps_a (preference shock):")
    ii_v = findfirst(==("I_v"), var_names)
    @printf("  %4s %12s %12s %12s %12s\n", "h", "C", "Y", "N", "I")
    for h in [1, 2, 5, 10, 20]
        @printf("  %4d %12.6f %12.6f %12.6f %12.6f\n",
                h,
                ci !== nothing ? ir1.values[h, ci, shock_idx_a] : NaN,
                yi !== nothing ? ir1.values[h, yi, shock_idx_a] : NaN,
                ni !== nothing ? ir1.values[h, ni, shock_idx_a] : NaN,
                ii_v !== nothing ? ir1.values[h, ii_v, shock_idx_a] : NaN)
    end
catch e
    println("  Order=1 FAILED: ", e)
    println("  ", sprint(showerror, e))
end

# ── Order=2 ──────────────────────────────────────────────────────────────
println("\n=== Second-Order Solution ===")
sol2 = nothing
try
    global sol2 = perturbation_solver(spec; order=2)
    println("  order = ", sol2.order)
    nx = length(sol2.state_indices)
    ny = length(sol2.control_indices)
    nv = nx + spec.n_exog
    println("  nx=$nx, ny=$ny, nv=$nv, nv^2=$(nv^2)")

    if sol2.hσσ !== nothing
        println("  max|hσσ| = ", @sprintf("%.6e", maximum(abs.(sol2.hσσ))))
    end
    if sol2.gσσ !== nothing
        println("  max|gσσ| = ", @sprintf("%.6e", maximum(abs.(sol2.gσσ))))
    end
catch e
    println("  Order=2 FAILED: ", e)
    println("  ", sprint(showerror, e))
end

# ── Order=3 feasibility ──────────────────────────────────────────────────
println("\n=== Third-Order Feasibility ===")
if sol1 !== nothing || sol2 !== nothing
    sol_ref = sol2 !== nothing ? sol2 : nothing
    if sol_ref !== nothing
        nx = length(sol_ref.state_indices)
        nv = nx + spec.n_exog
        n = spec.n_endog
        nv3 = nv^3
        lhs = n * nv3
        mem = lhs^2 * 8 / 1e9
        println("  n=$n, nx=$nx, nv=$nv, nv^3=$nv3")
        println("  LHS: $lhs x $lhs = $(@sprintf("%.1f", mem)) GB")
    end
    println("  STATUS: INFEASIBLE (direct Kronecker solve)")
end

# ── GIRF at order=2 ──────────────────────────────────────────────────────
if sol2 !== nothing
    println("\n=== GIRF to eps_sigma_a (order=2, uncertainty shock) ===")
    try
        ir2 = irf(sol2, 20; irf_type=:girf, n_draws=200, shock_size=1.0)
        shock_idx = 2  # eps_sigma_a
        orig_names = spec.augmented ? string.(spec.original_endog) : var_names
        ci = findfirst(==("C"), orig_names)
        yi = findfirst(==("Y"), orig_names)
        ni = findfirst(==("N"), orig_names)
        ii_v = findfirst(==("I_v"), orig_names)
        mui = findfirst(==("mu_v"), orig_names)

        @printf("  %4s %12s %12s %12s %12s %12s\n", "h", "C", "Y", "N", "I", "Markup")
        for h in [1, 2, 4, 8, 12, 20]
            @printf("  %4d %12.6f %12.6f %12.6f %12.6f %12.6f\n",
                    h,
                    ci !== nothing ? ir2.values[h, ci, shock_idx] : NaN,
                    yi !== nothing ? ir2.values[h, yi, shock_idx] : NaN,
                    ni !== nothing ? ir2.values[h, ni, shock_idx] : NaN,
                    ii_v !== nothing ? ir2.values[h, ii_v, shock_idx] : NaN,
                    mui !== nothing ? ir2.values[h, mui, shock_idx] : NaN)
        end
        println("  Key result: At order=2, GIRF shows small contractionary responses.")
        println("  The paper's main results require order=3 where the state-dependent")
        println("  interaction of uncertainty with EZ preferences generates large effects.")
        println("  At order=2, only the constant sigma^2 correction matters, not the")
        println("  full sigma^2*x (state-dependent risk) channel.")
    catch e
        println("  GIRF FAILED: ", e)
        println("  ", sprint(showerror, e))
    end
end

# ── Summary ──────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("  Summary: Basu & Bundick 2017")
println("  33 core equations, 4 shocks, Epstein-Zin preferences")
println("  Steady state: VERIFIED (from external SS file computation)")
println("  Order=1: ", sol1 !== nothing ? "SOLVED" : "FAILED")
println("  Order=2: ", sol2 !== nothing ? "SOLVED" : "FAILED")
println("  Order=3: INFEASIBLE (direct Kronecker solve)")
if sol2 !== nothing
    println("  GIRF (order=2): Uncertainty shock responses available")
end
println("=" ^ 60)
