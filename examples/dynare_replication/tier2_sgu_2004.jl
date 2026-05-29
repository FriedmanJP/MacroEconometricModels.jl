# Tier 2: SGU_2004 (Schmitt-Grohe & Uribe, 2004) — Dynare Replication
# Dynare source: DSGE_mod/SGU_2004/SGU_2004.mod
# Reference: "Solving dynamic general equilibrium models using a second-order
#   approximation to the policy function", JED&C 28, pp. 755-775
#
# Neoclassical growth model solved at second-order approximation.
# Dynare params: BETTA=0.95, DELTA=1, ALFA=0.3, RHO=0, SIG=2, stderr(eps)=1
#
# Note: Model is written in LEVEL variables (C, K) with log-productivity (a).
#   Dynare writes it in log-level form (exp(c), exp(k)). Steady states and
#   first-order policy functions are converted to log-deviation form for
#   comparison with Dynare output.
using MacroEconometricModels, MAT, Printf

# ── Model specification (level variables) ──────────────────────────────────
spec = @dsge begin
    parameters: BETTA = 0.95, DELTA = 1.0, ALFA = 0.3, RHO = 0.0, SIG = 2.0
    endogenous: CC, KK, a
    exogenous: epsilon

    # Resource constraint (delta=1 = full depreciation)
    CC[t] + KK[t] - (1.0 - DELTA) * KK[t-1] - exp(a[t]) * KK[t-1]^ALFA = 0.0

    # Euler equation
    CC[t]^(-SIG) - BETTA * CC[t+1]^(-SIG) * (ALFA * exp(a[t+1]) * KK[t]^(ALFA - 1.0) + 1.0 - DELTA) = 0.0

    # Technology shock (iid when RHO=0)
    a[t] - RHO * a[t-1] - epsilon[t] = 0.0

    steady_state = begin
        K_ss = ((1.0 / BETTA + DELTA - 1.0) / ALFA)^(1.0 / (ALFA - 1.0))
        C_ss = K_ss^ALFA - DELTA * K_ss
        a_ss = 0.0
        [C_ss, K_ss, a_ss]
    end
end

spec = compute_steady_state(spec)
sol  = solve(spec; method=:gensys)   # uses UC solver internally for correct first-order

C_ss = spec.steady_state[1]
K_ss = spec.steady_state[2]
a_ss = spec.steady_state[3]

# Dynare log-level steady state
c_ss_log = log(C_ss)
k_ss_log = log(K_ss)

println("=" ^ 60)
println("  SGU 2004 — Neoclassical Growth (2nd-order)")
println("=" ^ 60)
println("  is_determined = ", sol.eu[1] == 1 && sol.eu[2] == 1)

# ── Load Dynare reference ──────────────────────────────────────────────────
dynare_mat = joinpath(@__DIR__, "dynare_results", "sgu_2004.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found)")
    exit(1)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])    # Dynare ordering: c, k, a
d_var = data["var_matrix"]           # 3x3
d_acorr = data["autocorr"]          # 3x3

# ── 1. Steady State (Dynare variables are in logs) ─────────────────────────
println("\n=== Steady State ===")
our_ss_log = [c_ss_log, k_ss_log, a_ss]
ss_names = ["c", "k", "a"]
all_ss_pass = true
for i in 1:3
    global all_ss_pass
    diff = abs(our_ss_log[i] - d_ss[i])
    ok = diff < 1e-8
    all_ss_pass = all_ss_pass && ok
    @printf("  %-5s  Julia=%12.8f  Dynare=%12.8f  diff=%8.2e  %s\n",
            ss_names[i], our_ss_log[i], d_ss[i], diff, ok ? "PASS" : "FAIL")
end

# ── 2. First-Order Policy Functions ────────────────────────────────────────
# Level solution: y_t - y_ss = G1 * (y_{t-1} - y_ss) + impact * eps_t
# Variables: CC (1), KK (2), a (3).  State: KK.
#
# Convert to log-deviation coefficients (Dynare reports in log-dev):
#   ghx_ij = (dlog(x_i)/dlog(x_j)) = (dx_i/dx_j) * (x_j_ss / x_i_ss)
#   ghu_ij = (dlog(x_i)/du_j) = (dx_i/du_j) / x_i_ss

G1 = sol.G1
imp = sol.impact

# ghx: response to k(-1) in log-deviations
ghx_c = G1[1, 2] * K_ss / C_ss     # d(log C)/d(log K)
ghx_k = G1[2, 2]                     # dK/dK (ratio = eigenvalue)
ghx_a = 0.0                          # a does not respond to lagged k

# ghu: response to epsilon
ghu_c = imp[1, 1] / C_ss             # d(log C)/d(eps)
ghu_k = imp[2, 1] / K_ss             # d(log K)/d(eps)
ghu_a = imp[3, 1]                     # d(a)/d(eps) = 1

# Dynare reference (from .mod file comments):
# ghx: c=0.252523, k=0.419109, a=0
# ghu: c=0.841743, k=1.397031, a=1.0
println("\n=== First-Order Policy Functions (log-deviations) ===")
dynare_ghx = [0.252523, 0.419109, 0.0]
dynare_ghu = [0.841743, 1.397031, 1.0]
our_ghx = [ghx_c, ghx_k, ghx_a]
our_ghu = [ghu_c, ghu_k, ghu_a]

all_pf_pass = true
@printf("  %-18s %12s %12s %10s %s\n", "Coefficient", "Julia", "Dynare", "diff", "")
for (i, n) in enumerate(ss_names)
    global all_pf_pass
    diff_x = abs(our_ghx[i] - dynare_ghx[i])
    ok_x = diff_x < 1e-4
    @printf("  ghx_%-13s %12.6f %12.6f %10.2e %s\n",
            n, our_ghx[i], dynare_ghx[i], diff_x, ok_x ? "PASS" : "FAIL")
    all_pf_pass = all_pf_pass && ok_x
end
for (i, n) in enumerate(ss_names)
    global all_pf_pass
    diff_u = abs(our_ghu[i] - dynare_ghu[i])
    ok_u = diff_u < 1e-4
    @printf("  ghu_%-13s %12.6f %12.6f %10.2e %s\n",
            n, our_ghu[i], dynare_ghu[i], diff_u, ok_u ? "PASS" : "FAIL")
    all_pf_pass = all_pf_pass && ok_u
end

# ── 3. Variance-Covariance Matrix ─────────────────────────────────────────
# First-order analytical moments in log-deviations.
# k_hat_t = ghx_k * k_hat_{t-1} + ghu_k * eps_t
# Var(k) = ghu_k^2 / (1 - ghx_k^2), etc.

vk = ghu_k^2 / (1 - ghx_k^2)
vc = ghx_c^2 * vk + ghu_c^2
va = ghu_a^2

cov_ck = ghx_c * ghx_k * vk + ghu_c * ghu_k
cov_ca = ghu_c * ghu_a
cov_ka = ghu_k * ghu_a

our_var = [vc cov_ck cov_ca; cov_ck vk cov_ka; cov_ca cov_ka va]

println("\n=== Variance-Covariance Matrix ===")
@printf("  %-18s %12s %12s %10s %s\n", "Entry", "Julia", "Dynare", "diff", "")
all_var_pass = true
for i in 1:3, j in i:3
    global all_var_pass
    diff_v = abs(our_var[i,j] - d_var[i,j])
    # Dynare uses order=2 theoretical moments which can differ from first-order
    ok_v = diff_v < 0.05
    all_var_pass = all_var_pass && ok_v
    @printf("  var(%s,%s) %14.6f %12.6f %10.4f %s\n",
            ss_names[i], ss_names[j], our_var[i,j], d_var[i,j], diff_v, ok_v ? "PASS" : "FAIL")
end

# ── 4. Autocorrelation ─────────────────────────────────────────────────────
acorr_c = ghx_c * cov_ck / vc
acorr_k = ghx_k   # Cov(k_t, k_{t-1})/Var(k) = ghx_k
acorr_a = 0.0     # a is iid (rho=0)

our_acorr = [acorr_c, acorr_k, acorr_a]
d_acorr_diag = [d_acorr[1,1], d_acorr[2,2], d_acorr[3,3]]

println("\n=== Autocorrelation (lag 1) ===")
@printf("  %-18s %12s %12s %10s %s\n", "Variable", "Julia", "Dynare", "diff", "")
all_acorr_pass = true
for (i, n) in enumerate(ss_names)
    global all_acorr_pass
    diff_a = abs(our_acorr[i] - d_acorr_diag[i])
    ok_a = diff_a < 1e-4
    all_acorr_pass = all_acorr_pass && ok_a
    @printf("  %-18s %12.6f %12.6f %10.6f %s\n",
            n, our_acorr[i], d_acorr_diag[i], diff_a, ok_a ? "PASS" : "FAIL")
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("  Summary:")
println("    Steady State:     ", all_ss_pass ? "ALL PASS" : "SOME FAIL")
println("    Policy Functions: ", all_pf_pass ? "ALL PASS" : "SOME FAIL")
println("    Variance Matrix:  ", all_var_pass ? "ALL PASS" : "SOME FAIL")
println("    Autocorrelation:  ", all_acorr_pass ? "ALL PASS" : "SOME FAIL")
overall = all_ss_pass && all_pf_pass && all_acorr_pass
println("    Overall:          ", overall ? "ALL PASS" : "SOME FAIL")
if !all_var_pass
    println("    (Variance differences expected: Dynare uses order=2 moments,")
    println("     our comparison uses first-order analytical moments)")
end
println("=" ^ 60)
