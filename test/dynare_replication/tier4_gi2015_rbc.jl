# Tier 4: Guerrieri & Iacoviello (2015) RBC — OccBin Replication
# Dynare source: DSGE_mod/Guerrieri_Iacoviello_2015/Guerrieri_Iacoviello_2015_rbc.mod
# Reference: Guerrieri & Iacoviello (2015), "OccBin: A toolkit for solving dynamic
#   models with occasionally binding constraints easily",
#   Journal of Monetary Economics, 70, pp. 22-38.
#
# RBC model with irreversible investment constraint (i >= PHI*i_ss).
# Two regimes:
#   Relax (slack): lam = 0    (Lagrange multiplier on investment constraint)
#   Bind:          iv = PHI * iv_ss  (investment pinned at lower bound)
#
# We use the explicit alt_spec variant of occbin_solve because the standard
# auto-derivation doesn't know which equation to substitute (it picks the
# equation with the largest Jacobian entry for iv, but Dynare replaces the
# lam=0 equation).
using MacroEconometricModels, MAT, Printf

# ── Parameters ─────────────────────────────────────────────────────────────
const _ALPHA = 0.33
const _DELTA = 0.1
const _BETA  = 0.96
const _GAMMA = 2.0
const _RHO   = 0.9
const _PHI   = 0.975

# ── Pre-compute steady state values ───────────────────────────────────────
const _a_ss  = 1.0
const _k_ss  = (((1.0/_BETA) - 1.0 + _DELTA) / _ALPHA)^(1.0 / (_ALPHA - 1.0))
const _c_ss  = -_DELTA * _k_ss + _k_ss^_ALPHA
const _iv_ss = _DELTA * _k_ss
const _lam_ss = 0.0
const _iv_bind = _PHI * _iv_ss

# ── Reference (unconstrained) spec: lam = 0 ──────────────────────────────
spec = @dsge begin
    parameters: ALPHA = 0.33, DELTA = 0.1, BETA = 0.96, GAMMA = 2.0,
                RHO = 0.9, PHI = 0.975,
                iv_ss_val = 0.3532878917156419,
                c_ss_val = 1.1633520474676697,
                k_ss_val = 3.5328789171564186
    endogenous: a, c, iv, k, lam, chat, ivhat, khat
    exogenous: epsi

    # (1) Consumption Euler equation (eq. A.1)
    c[t]^(-GAMMA) - lam[t] = BETA * (c[t+1]^(-GAMMA) * (1.0 - DELTA + ALPHA * a[t+1] * k[t]^(ALPHA - 1.0)) - (1.0 - DELTA) * lam[t+1])

    # (2) Resource constraint (eq. 7)
    c[t] + iv[t] = a[t] * k[t-1]^ALPHA

    # (3) Capital accumulation (eq. 8)
    k[t] = (1.0 - DELTA) * k[t-1] + iv[t]

    # (4) Technology process (eq. 10)
    log(a[t]) = RHO * log(a[t-1]) + epsi[t]

    # (5) Reporting: investment percent deviation
    ivhat[t] = 100.0 * (iv[t] / iv_ss_val - 1.0)

    # (6) Reporting: consumption percent deviation
    chat[t] = 100.0 * (c[t] / c_ss_val - 1.0)

    # (7) Reporting: capital percent deviation
    khat[t] = 100.0 * (k[t] / k_ss_val - 1.0)

    # (8) Relax regime: lam = 0
    lam[t] = 0.0

    steady_state = begin
        a_s   = 1.0
        k_s   = ((1.0/BETA - 1.0 + DELTA) / ALPHA)^(1.0 / (ALPHA - 1.0))
        c_s   = -DELTA * k_s + k_s^ALPHA
        iv_s  = DELTA * k_s
        lam_s = 0.0
        [a_s, c_s, iv_s, k_s, lam_s, 0.0, 0.0, 0.0]
    end
end
spec = compute_steady_state(spec)

# ── Alternative (binding) spec: iv = PHI * iv_ss ─────────────────────────
# Same model but equation 8 is replaced: iv[t] = PHI * iv_ss
alt_spec = @dsge begin
    parameters: ALPHA = 0.33, DELTA = 0.1, BETA = 0.96, GAMMA = 2.0,
                RHO = 0.9, PHI = 0.975,
                iv_ss_val = 0.3532878917156419,
                c_ss_val = 1.1633520474676697,
                k_ss_val = 3.5328789171564186
    endogenous: a, c, iv, k, lam, chat, ivhat, khat
    exogenous: epsi

    # Equations 1-7 identical to reference regime
    c[t]^(-GAMMA) - lam[t] = BETA * (c[t+1]^(-GAMMA) * (1.0 - DELTA + ALPHA * a[t+1] * k[t]^(ALPHA - 1.0)) - (1.0 - DELTA) * lam[t+1])
    c[t] + iv[t] = a[t] * k[t-1]^ALPHA
    k[t] = (1.0 - DELTA) * k[t-1] + iv[t]
    log(a[t]) = RHO * log(a[t-1]) + epsi[t]
    ivhat[t] = 100.0 * (iv[t] / iv_ss_val - 1.0)
    chat[t] = 100.0 * (c[t] / c_ss_val - 1.0)
    khat[t] = 100.0 * (k[t] / k_ss_val - 1.0)

    # (8) Bind regime: iv = PHI * iv_ss (replaces lam = 0)
    iv[t] = PHI * iv_ss_val

    steady_state = begin
        a_s   = 1.0
        k_s   = ((1.0/BETA - 1.0 + DELTA) / ALPHA)^(1.0 / (ALPHA - 1.0))
        c_s   = -DELTA * k_s + k_s^ALPHA
        iv_s  = DELTA * k_s
        lam_s = 0.0
        [a_s, c_s, iv_s, k_s, lam_s, 0.0, 0.0, 0.0]
    end
end
alt_spec = compute_steady_state(alt_spec)

# ── Verify steady state ───────────────────────────────────────────────────
println("=" ^ 60)
println("  Guerrieri & Iacoviello (2015) RBC — OccBin")
println("=" ^ 60)

dynare_mat = joinpath(@__DIR__, "dynare_results", "gi2015_rbc.mat")
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

# ── Solve unconstrained model ─────────────────────────────────────────────
sol = solve(spec; method=:gensys)
println("\n=== Linear Solution ===")
println("  is_determined = ", is_determined(sol))

# ── OccBin with explicit alternative regime ──────────────────────────────
# Constraint: iv >= PHI * iv_ss (in level form)
iv_idx = findfirst(==(:iv), spec.endog)
constraint_bound = _PHI * _iv_ss
constraint = parse_constraint(:(iv[t] >= $constraint_bound), spec)

# ── Negative TFP shock (Fig. 3, left column): epsi = -0.04 ───────────────
println("\n=== OccBin: Negative TFP shock (epsi = -0.04) ===")
nperiods = 100
shock_path_neg = zeros(nperiods, spec.n_exog)
shock_path_neg[1, 1] = -0.04

occ_sol_neg = occbin_solve(spec, constraint, alt_spec;
    shock_path=shock_path_neg, nperiods=nperiods)

println("  converged = ", occ_sol_neg.converged)
println("  iterations = ", occ_sol_neg.iterations)
binding_neg = sum(occ_sol_neg.regime_history[:, 1])
println("  binding periods = ", binding_neg)

# Verify constraint is respected in piecewise path
iv_pw_neg = occ_sol_neg.piecewise_path[:, iv_idx] .+ _iv_ss
iv_lin_neg = occ_sol_neg.linear_path[:, iv_idx] .+ _iv_ss
min_pw_iv = minimum(iv_pw_neg)
min_lin_iv = minimum(iv_lin_neg)
@printf("  min(iv_linear)    = %10.6f  (should violate: < %.6f)\n", min_lin_iv, _iv_bind)
@printf("  min(iv_piecewise) = %10.6f  (should respect: >= %.6f)\n", min_pw_iv, _iv_bind)

neg_iv_ok = min_pw_iv >= _iv_bind - 1e-4
neg_binding_ok = binding_neg > 0
neg_lin_violates = min_lin_iv < _iv_bind
println("  constraint respected = ", neg_iv_ok ? "PASS" : "FAIL")
println("  linear violates     = ", neg_lin_violates ? "YES (expected)" : "NO (unexpected)")
println("  constraint binds    = ", neg_binding_ok ? "YES (expected)" : "NO (unexpected)")

# Show first 15 periods (percent deviations)
ivhat_idx = findfirst(==(:ivhat), spec.endog)
chat_idx = findfirst(==(:chat), spec.endog)
khat_idx = findfirst(==(:khat), spec.endog)
println("\n  Period   ivhat_pw   ivhat_lin   chat_pw   chat_lin   khat_pw   khat_lin  regime")
for t in 1:min(20, nperiods)
    @printf("  %3d    %9.4f  %9.4f   %8.4f  %8.4f   %8.4f  %8.4f   %d\n",
            t,
            occ_sol_neg.piecewise_path[t, ivhat_idx],
            occ_sol_neg.linear_path[t, ivhat_idx],
            occ_sol_neg.piecewise_path[t, chat_idx],
            occ_sol_neg.linear_path[t, chat_idx],
            occ_sol_neg.piecewise_path[t, khat_idx],
            occ_sol_neg.linear_path[t, khat_idx],
            occ_sol_neg.regime_history[t, 1])
end

# ── Positive TFP shock (Fig. 3, right column): epsi = +0.04 ──────────────
println("\n=== OccBin: Positive TFP shock (epsi = +0.04) ===")
shock_path_pos = zeros(nperiods, spec.n_exog)
shock_path_pos[1, 1] = 0.04

occ_sol_pos = occbin_solve(spec, constraint, alt_spec;
    shock_path=shock_path_pos, nperiods=nperiods)

println("  converged = ", occ_sol_pos.converged)
println("  iterations = ", occ_sol_pos.iterations)
binding_pos = sum(occ_sol_pos.regime_history[:, 1])
println("  binding periods = ", binding_pos)

# For positive shock, constraint should NOT bind
pos_no_bind = binding_pos == 0
println("  no binding (expected) = ", pos_no_bind ? "PASS" : "FAIL")

if pos_no_bind
    max_diff_pos = maximum(abs.(occ_sol_pos.piecewise_path .- occ_sol_pos.linear_path))
    @printf("  max|pw - lin| = %.2e (should be ~0)\n", max_diff_pos)
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
all_pass = all_ss_pass && occ_sol_neg.converged && neg_iv_ok && neg_binding_ok &&
           occ_sol_pos.converged && pos_no_bind
println("  Overall: ", all_pass ? "ALL PASS" : "SOME FAIL")
println("=" ^ 60)
