# Tier 4: Guerrieri & Iacoviello (2015) NK — OccBin ZLB Replication
# Dynare source: DSGE_mod/Guerrieri_Iacoviello_2015/Guerrieri_Iacoviello_2015_nk.mod
# Reference: Guerrieri & Iacoviello (2015), "OccBin: A toolkit for solving dynamic
#   models with occasionally binding constraints easily",
#   Journal of Monetary Economics, 70, pp. 22-38.
#
# New Keynesian model with Calvo pricing and zero lower bound on nominal interest rate.
# Two regimes:
#   Relax (slack): Taylor rule   r = r_ss*(pie/PI)^PHI_PI*(y/y_ss)^PHI_Y
#   Bind (ZLB):    r = ZLB = 1   (replaces Taylor rule equation)
#
# We use the explicit alt_spec variant of occbin_solve because the standard
# auto-derivation picks the wrong equation to replace.
using MacroEconometricModels, MAT, Printf

# ── Parameters ─────────────────────────────────────────────────────────────
const _BETA    = 0.994
const _THETA   = 0.90
const _PHI_Y   = 0.25
const _PHI_PI  = 2.5
const _RHO     = 0.8
const _EPSILON = 6.0
const _G_Y     = 0.20
const _PI      = 1.005
const _PHI_L   = 1.0
const _ZLB     = 1.0

# ── Pre-compute steady state ──────────────────────────────────────────────
const _bet_ss = _BETA
const _pie_ss = _PI
const _pie_star_ss = ((1.0 - _THETA * _pie_ss^(_EPSILON - 1.0)) / (1.0 - _THETA))^(1.0 / (1.0 - _EPSILON))
const _mc_ss = (_EPSILON - 1.0) / _EPSILON * _pie_star_ss *
               (1.0 - _bet_ss * _THETA * _pie_ss^_EPSILON) /
               (1.0 - _bet_ss * _THETA * _pie_ss^(_EPSILON - 1.0))
const _v_ss = (1.0 - _THETA) * _pie_star_ss^(-_EPSILON) / (1.0 - _THETA * _pie_ss^_EPSILON)
const _r_ss = _pie_ss / _BETA
const _w_ss = _mc_ss
const _y_ss = 1.0
const _l_ss = _v_ss * _y_ss
const _c_ss = (1.0 - _G_Y) * _y_ss
const _g_ss = _G_Y * _y_ss
const _PSI = _w_ss / (_l_ss^_PHI_L * _c_ss)
const _y_c = 1.0 / (1.0 - _G_Y)
const _x2_ss = _pie_star_ss * _y_c / (1.0 - _THETA * _bet_ss * _pie_ss^(_EPSILON - 1.0))
const _x1_ss = (_EPSILON - 1.0) / _EPSILON * _x2_ss
const _r_an_ss = 400.0 * (_r_ss - _ZLB)
const _pie_an_ss = 0.0
const _yhat_ss = 0.0

# ── Reference spec (Taylor rule regime) ──────────────────────────────────
spec = @dsge begin
    parameters: BETA_p = 0.994, THETA_p = 0.90, PHI_Y_p = 0.25, PHI_PI_p = 2.5,
                RHO_p = 0.8, EPSILON_p = 6.0, G_Y_p = 0.20, PI_p = 1.005,
                PHI_L_p = 1.0, PSI_p = 0.8198517655505303,
                ZLB_p = 1.0,
                r_ss_p = 1.0110663983903418,
                y_ss_p = 1.0,
                pie_ss_p = 1.005
    endogenous: bet, c, y, l, w, mc, r, g, pie, pie_star, x1, x2, v, pie_an, r_an, yhat
    exogenous: epsi

    # (A.2) Euler
    c[t]^(-1.0) = bet[t] * c[t+1]^(-1.0) * r[t] / pie[t+1]
    # (A.3)
    mc[t] = w[t]
    # (A.4) Labor supply
    w[t] = PSI_p * l[t]^PHI_L_p * c[t]
    # (A.5) Price setting
    EPSILON_p * x1[t] = (EPSILON_p - 1.0) * x2[t]
    # (A.6)
    x1[t] = mc[t] * y[t] / c[t] + THETA_p * bet[t] * pie[t+1]^EPSILON_p * x1[t+1]
    # (A.7)
    x2[t] / pie_star[t] = y[t] / c[t] + THETA_p * bet[t] * pie[t+1]^(EPSILON_p - 1.0) * x2[t+1] / pie_star[t+1]
    # (A.8) Taylor rule [relax regime]
    r[t] = r_ss_p * (pie[t] / PI_p)^PHI_PI_p * (y[t] / y_ss_p)^PHI_Y_p
    # (A.10) Gov spending
    g[t] = G_Y_p * y[t]
    # (A.11) Reset price
    1.0 = THETA_p * pie[t]^(EPSILON_p - 1.0) + (1.0 - THETA_p) * pie_star[t]^(1.0 - EPSILON_p)
    # (A.12) Price dispersion
    v[t] = THETA_p * pie[t]^EPSILON_p * v[t-1] + (1.0 - THETA_p) * pie_star[t]^(-EPSILON_p)
    # (A.13) Demand
    y[t] = c[t] + g[t]
    # (A.14) Supply
    y[t] = l[t] / v[t]
    # (A.15) Discount factor
    log(bet[t]) = (1.0 - RHO_p) * log(BETA_p) + RHO_p * log(bet[t-1]) + epsi[t]
    # Reporting
    r_an[t] = 400.0 * (r[t] - ZLB_p)
    pie_an[t] = 400.0 * (pie[t] - pie_ss_p)
    yhat[t] = 100.0 * (y[t] / y_ss_p - 1.0)

    steady_state = begin
        bet_s = BETA_p
        pie_s = PI_p
        pie_star_s = ((1.0 - THETA_p * pie_s^(EPSILON_p - 1.0)) / (1.0 - THETA_p))^(1.0 / (1.0 - EPSILON_p))
        mc_s = (EPSILON_p - 1.0) / EPSILON_p * pie_star_s *
               (1.0 - bet_s * THETA_p * pie_s^EPSILON_p) /
               (1.0 - bet_s * THETA_p * pie_s^(EPSILON_p - 1.0))
        v_s = (1.0 - THETA_p) * pie_star_s^(-EPSILON_p) / (1.0 - THETA_p * pie_s^EPSILON_p)
        r_s = pie_s / BETA_p
        w_s = mc_s
        y_c_val = 1.0 / (1.0 - G_Y_p)
        x2_s = pie_star_s * y_c_val / (1.0 - THETA_p * bet_s * pie_s^(EPSILON_p - 1.0))
        x1_s = (EPSILON_p - 1.0) / EPSILON_p * x2_s
        y_s = 1.0
        l_s = v_s * y_s
        c_s = (1.0 - G_Y_p) * y_s
        g_s = G_Y_p * y_s
        r_an_s = 400.0 * (r_s - ZLB_p)
        pie_an_s = 0.0
        yhat_s = 0.0
        [bet_s, c_s, y_s, l_s, w_s, mc_s, r_s, g_s, pie_s, pie_star_s,
         x1_s, x2_s, v_s, pie_an_s, r_an_s, yhat_s]
    end
end
spec = compute_steady_state(spec)

# ── Alternative spec (ZLB binding): r = ZLB ──────────────────────────────
# Equation 7 (Taylor rule) replaced with r[t] = ZLB
alt_spec = @dsge begin
    parameters: BETA_p = 0.994, THETA_p = 0.90, PHI_Y_p = 0.25, PHI_PI_p = 2.5,
                RHO_p = 0.8, EPSILON_p = 6.0, G_Y_p = 0.20, PI_p = 1.005,
                PHI_L_p = 1.0, PSI_p = 0.8198517655505303,
                ZLB_p = 1.0,
                r_ss_p = 1.0110663983903418,
                y_ss_p = 1.0,
                pie_ss_p = 1.005
    endogenous: bet, c, y, l, w, mc, r, g, pie, pie_star, x1, x2, v, pie_an, r_an, yhat
    exogenous: epsi

    # Same equations 1-6
    c[t]^(-1.0) = bet[t] * c[t+1]^(-1.0) * r[t] / pie[t+1]
    mc[t] = w[t]
    w[t] = PSI_p * l[t]^PHI_L_p * c[t]
    EPSILON_p * x1[t] = (EPSILON_p - 1.0) * x2[t]
    x1[t] = mc[t] * y[t] / c[t] + THETA_p * bet[t] * pie[t+1]^EPSILON_p * x1[t+1]
    x2[t] / pie_star[t] = y[t] / c[t] + THETA_p * bet[t] * pie[t+1]^(EPSILON_p - 1.0) * x2[t+1] / pie_star[t+1]

    # (A.8) ZLB [bind regime]: r = ZLB
    r[t] = ZLB_p

    # Same equations 8-16
    g[t] = G_Y_p * y[t]
    1.0 = THETA_p * pie[t]^(EPSILON_p - 1.0) + (1.0 - THETA_p) * pie_star[t]^(1.0 - EPSILON_p)
    v[t] = THETA_p * pie[t]^EPSILON_p * v[t-1] + (1.0 - THETA_p) * pie_star[t]^(-EPSILON_p)
    y[t] = c[t] + g[t]
    y[t] = l[t] / v[t]
    log(bet[t]) = (1.0 - RHO_p) * log(BETA_p) + RHO_p * log(bet[t-1]) + epsi[t]
    r_an[t] = 400.0 * (r[t] - ZLB_p)
    pie_an[t] = 400.0 * (pie[t] - pie_ss_p)
    yhat[t] = 100.0 * (y[t] / y_ss_p - 1.0)

    steady_state = begin
        bet_s = BETA_p
        pie_s = PI_p
        pie_star_s = ((1.0 - THETA_p * pie_s^(EPSILON_p - 1.0)) / (1.0 - THETA_p))^(1.0 / (1.0 - EPSILON_p))
        mc_s = (EPSILON_p - 1.0) / EPSILON_p * pie_star_s *
               (1.0 - bet_s * THETA_p * pie_s^EPSILON_p) /
               (1.0 - bet_s * THETA_p * pie_s^(EPSILON_p - 1.0))
        v_s = (1.0 - THETA_p) * pie_star_s^(-EPSILON_p) / (1.0 - THETA_p * pie_s^EPSILON_p)
        r_s = pie_s / BETA_p
        w_s = mc_s
        y_c_val = 1.0 / (1.0 - G_Y_p)
        x2_s = pie_star_s * y_c_val / (1.0 - THETA_p * bet_s * pie_s^(EPSILON_p - 1.0))
        x1_s = (EPSILON_p - 1.0) / EPSILON_p * x2_s
        y_s = 1.0
        l_s = v_s * y_s
        c_s = (1.0 - G_Y_p) * y_s
        g_s = G_Y_p * y_s
        r_an_s = 400.0 * (r_s - ZLB_p)
        pie_an_s = 0.0
        yhat_s = 0.0
        [bet_s, c_s, y_s, l_s, w_s, mc_s, r_s, g_s, pie_s, pie_star_s,
         x1_s, x2_s, v_s, pie_an_s, r_an_s, yhat_s]
    end
end
alt_spec = compute_steady_state(alt_spec)

# ── Verify steady state ───────────────────────────────────────────────────
println("=" ^ 60)
println("  Guerrieri & Iacoviello (2015) NK — OccBin ZLB")
println("=" ^ 60)

dynare_mat = joinpath(@__DIR__, "dynare_results", "gi2015_nk.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found)")
    exit(1)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
d_names_raw = data["endo_names"]
if d_names_raw isa Matrix
    d_names = vec([strip(string(d_names_raw[i,1])) for i in 1:size(d_names_raw, 1)])
else
    d_names = vec([strip(string(x)) for x in d_names_raw])
end

dynare_idx = Dict{String,Int}()
for (i, n) in enumerate(d_names)
    dynare_idx[n] = i
end

our_names = string.(spec.endog)

println("\n=== Steady State ===")
@printf("  %-12s %14s %14s %10s %s\n", "Variable", "Julia", "Dynare", "diff", "")
all_ss_pass = true
for (i, n) in enumerate(our_names)
    global all_ss_pass
    di = get(dynare_idx, n, 0)
    di == 0 && continue
    diff = abs(spec.steady_state[i] - d_ss[di])
    ok = diff < 1e-6
    all_ss_pass = all_ss_pass && ok
    @printf("  %-12s %14.8f %14.8f %10.2e %s\n",
            n, spec.steady_state[i], d_ss[di], diff, ok ? "PASS" : "FAIL")
end
println("  Steady State: ", all_ss_pass ? "ALL PASS" : "SOME FAIL")

# ── Solve linear model ────────────────────────────────────────────────────
sol = solve(spec; method=:gensys)
println("\n=== Linear Solution ===")
println("  is_determined = ", is_determined(sol))

# ── OccBin: ZLB constraint r >= ZLB ──────────────────────────────────────
r_idx = findfirst(==(:r), spec.endog)
constraint = parse_constraint(:(r[t] >= $(_ZLB)), spec)

# ── Positive discount factor shock (Fig. 5, left column) ─────────────────
# Dynare applies a single surprise shock at period 6 with value 0.025.
# The discount factor process is persistent (rho=0.8) so the shock decays slowly.
println("\n=== OccBin: Positive discount factor shock (ZLB scenario) ===")
nperiods = 200
shock_size = 5.0 * 0.005  # = 0.025

shock_path_pos = zeros(nperiods, spec.n_exog)
shock_path_pos[1, 1] = shock_size  # single shock at period 1

occ_sol_pos = occbin_solve(spec, constraint, alt_spec;
    shock_path=shock_path_pos, nperiods=nperiods, maxiter=500)

println("  converged = ", occ_sol_pos.converged)
println("  iterations = ", occ_sol_pos.iterations)
binding_pos = sum(occ_sol_pos.regime_history[:, 1])
println("  binding periods = ", binding_pos)

# Verify: linear rate goes below ZLB, piecewise respects it
r_pw_pos = occ_sol_pos.piecewise_path[:, r_idx] .+ _r_ss
r_lin_pos = occ_sol_pos.linear_path[:, r_idx] .+ _r_ss
min_pw_r = minimum(r_pw_pos)
min_lin_r = minimum(r_lin_pos)
@printf("  min(r_linear)    = %10.6f  (should violate: < %.6f)\n", min_lin_r, _ZLB)
@printf("  min(r_piecewise) = %10.6f  (should respect: >= %.6f)\n", min_pw_r, _ZLB)

pos_r_ok = min_pw_r >= _ZLB - 1e-4
pos_binding_ok = binding_pos > 0
pos_lin_violates = min_lin_r < _ZLB
println("  ZLB respected (pw) = ", pos_r_ok ? "PASS" : "FAIL")
println("  linear violates    = ", pos_lin_violates ? "YES (expected)" : "NO (unexpected)")
println("  ZLB binds          = ", pos_binding_ok ? "YES (expected)" : "NO (unexpected)")

# Show first 20 periods for key reporting variables
r_an_idx = findfirst(==(:r_an), spec.endog)
pie_an_idx = findfirst(==(:pie_an), spec.endog)
yhat_idx = findfirst(==(:yhat), spec.endog)
v_idx = findfirst(==(:v), spec.endog)
println("\n  Period   r_an_pw   r_an_lin   pie_an_pw pie_an_lin  yhat_pw  yhat_lin  regime")
for t in 1:min(25, nperiods)
    @printf("  %3d    %8.4f  %8.4f   %8.4f  %8.4f  %8.4f  %8.4f   %d\n",
            t,
            occ_sol_pos.piecewise_path[t, r_an_idx],
            occ_sol_pos.linear_path[t, r_an_idx],
            occ_sol_pos.piecewise_path[t, pie_an_idx],
            occ_sol_pos.linear_path[t, pie_an_idx],
            occ_sol_pos.piecewise_path[t, yhat_idx],
            occ_sol_pos.linear_path[t, yhat_idx],
            occ_sol_pos.regime_history[t, 1])
end

# ── Negative discount factor shock (Fig. 5, right column) ────────────────
println("\n=== OccBin: Negative discount factor shock (no ZLB) ===")
shock_path_neg = zeros(nperiods, spec.n_exog)
shock_path_neg[1, 1] = -shock_size  # single shock at period 1

occ_sol_neg = occbin_solve(spec, constraint, alt_spec;
    shock_path=shock_path_neg, nperiods=nperiods)

println("  converged = ", occ_sol_neg.converged)
println("  iterations = ", occ_sol_neg.iterations)
binding_neg = sum(occ_sol_neg.regime_history[:, 1])
println("  binding periods = ", binding_neg)

neg_no_bind = binding_neg == 0
println("  no binding (expected) = ", neg_no_bind ? "PASS" : "FAIL")

if neg_no_bind
    max_diff_neg = maximum(abs.(occ_sol_neg.piecewise_path .- occ_sol_neg.linear_path))
    @printf("  max|pw - lin| = %.2e (should be ~0)\n", max_diff_neg)
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
all_pass = all_ss_pass && occ_sol_pos.converged && pos_r_ok && pos_binding_ok &&
           occ_sol_neg.converged && neg_no_bind
println("  Overall: ", all_pass ? "ALL PASS" : "SOME FAIL")
println("=" ^ 60)
