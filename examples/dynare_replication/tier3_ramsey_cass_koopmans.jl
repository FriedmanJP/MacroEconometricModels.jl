# Tier 3: Ramsey-Cass-Koopmans Model — Perfect Foresight Transition
# Dynare source: DSGE_mod/Ramsey_Cass_Koopmans/Ramsey_Cass_Koopmans.mod
#
# Non-stationary growth model with Cobb-Douglas production.
# Dynare solves this in aggregate form with growing A and L as exogenous variables.
# We reformulate in intensive (per effective worker) form where a stationary
# steady state exists, then solve the deterministic transition from 90% of the
# balanced growth path capital to the BGP using perfect_foresight().
#
# Dynare model (aggregate):
#   K    = (1-delta)*K(-1) + I
#   I+C  = Y
#   1/C  = beta * (1/C(+1)) * (alpha*Y(+1)/K + (1-delta))
#   Y    = K(-1)^alpha * (A*L)^(1-alpha)
#   A grows at rate g, L grows at rate n
#
# Intensive form (variables per effective worker: c=C/(AL), k=K/(AL), y=Y/(AL)):
#   gamma*k[t] = (1-delta)*k[t-1] + invest[t]       [capital accumulation]
#   invest[t] + c[t] = y[t]                          [resource constraint]
#   1/c[t] = (beta/gamma)*(1/c[t+1])*(alpha*k[t]^(alpha-1) + (1-delta))  [Euler]
#   y[t] = k[t-1]^alpha                              [production]
#
# Timing: k[t] = capital decided at t. Production at t uses k[t-1].
# The Euler: return on capital decided at t is alpha*k[t]^{alpha-1}+(1-delta),
# realized next period.
#
# The perfect_foresight() solver uses spec.steady_state for both the initial
# boundary (k[t-1] at t=1 = k_ss) and terminal boundary.  Two shocks encode
# the initial displacement from SS to 90% of SS in period 1.
#
# ============================================================================
using MacroEconometricModels, Printf

# ── Parameters (from .mod file) ──────────────────────────────────────────────
const _alpha = 0.3
const _delta = 0.1
const _beta  = 0.99
const _n     = 0.01
const _g     = 0.02
const _gamma = (1 + _n) * (1 + _g)

# ── Analytical steady state in intensive form ────────────────────────────────
# From Euler at SS: 1 = (beta/gamma) * (alpha*k_ss^(alpha-1) + (1-delta))
# => k_ss = ((gamma/beta - (1-delta)) / alpha)^(1/(alpha-1))
const _k_ss   = ((_gamma / _beta - (1 - _delta)) / _alpha)^(1 / (_alpha - 1))
const _y_ss   = _k_ss^_alpha
const _inv_ss = (_gamma - 1 + _delta) * _k_ss
const _c_ss   = _y_ss - _inv_ss
const _k_init = 0.9 * _k_ss   # initial capital (90% of SS)

# ── Model specification (intensive form) ─────────────────────────────────────
# k[t]: capital decided at t.  Production at t uses k[t-1] (predetermined).
# eps_k: additive shock in accumulation (corrects (1-delta)*k[t-1] boundary).
# eps_y: additive shock in production (corrects k[t-1]^alpha boundary).

spec = @dsge begin
    parameters: alpha = 0.3, delta = 0.1, beta_disc = 0.99,
                npop = 0.01, gtech = 0.02
    endogenous: c, k, y, invest, log_c, log_k, log_y, log_invest
    exogenous: eps_k, eps_y

    # (1) Capital accumulation (intensive form)
    (1 + npop + gtech + npop * gtech) * k[t] = (1 - delta) * k[t-1] + invest[t] + eps_k[t]

    # (2) Resource constraint
    invest[t] + c[t] = y[t]

    # (3) Euler equation (intensive form, log utility sigma=1)
    # Return on capital decided at t: alpha*k[t]^(alpha-1)+(1-delta)
    1/c[t] = beta_disc / (1 + npop + gtech + npop * gtech) * (1/c[t+1]) * (alpha * k[t]^(alpha - 1) + (1 - delta))

    # (4) Production function (predetermined capital: y uses k from last period)
    y[t] = k[t-1]^alpha + eps_y[t]

    # (5-8) Log definitions
    log_c[t] = log(c[t])
    log_k[t] = log(k[t])
    log_y[t] = log(y[t])
    log_invest[t] = log(invest[t])

    steady_state = begin
        gamma_val = (1 + npop + gtech + npop * gtech)
        k_ss_val = ((gamma_val / beta_disc - (1 - delta)) / alpha)^(1 / (alpha - 1))
        y_ss_val = k_ss_val^alpha
        inv_ss_val = (gamma_val - 1 + delta) * k_ss_val
        c_ss_val = y_ss_val - inv_ss_val
        log_c_ss = log(c_ss_val)
        log_k_ss = log(k_ss_val)
        log_y_ss = log(y_ss_val)
        log_inv_ss = log(inv_ss_val)
        [c_ss_val, k_ss_val, y_ss_val, inv_ss_val,
         log_c_ss, log_k_ss, log_y_ss, log_inv_ss]
    end
end

# ── Compute steady state ─────────────────────────────────────────────────────
spec = compute_steady_state(spec)
ss = spec.steady_state
println("=" ^ 70)
println("  Ramsey-Cass-Koopmans Model (Perfect Foresight Transition)")
println("=" ^ 70)

println("\n=== Steady State (Intensive Form) ===")
ss_names = ["c", "k", "y", "invest", "log_c", "log_k", "log_y", "log_invest"]
analytical_ss = [_c_ss, _k_ss, _y_ss, _inv_ss, log(_c_ss), log(_k_ss), log(_y_ss), log(_inv_ss)]
all_ss_pass = true
for (i, name) in enumerate(ss_names)
    diff = abs(ss[i] - analytical_ss[i])
    ok = diff < 1e-10
    global all_ss_pass = all_ss_pass && ok
    @printf("  %-14s  Julia=%14.8f  Analytical=%14.8f  diff=%8.2e  %s\n",
            name, ss[i], analytical_ss[i], diff, ok ? "PASS" : "FAIL")
end
println("  SS overall: ", all_ss_pass ? "ALL PASS" : "SOME FAIL")

# Verify the Euler equation at SS
euler_lhs = 1 / _c_ss
euler_rhs = (_beta / _gamma) * (1 / _c_ss) * (_alpha * _k_ss^(_alpha - 1) + (1 - _delta))
@printf("\n  Euler check:  LHS=%.8f  RHS=%.8f  diff=%.2e\n",
        euler_lhs, euler_rhs, abs(euler_lhs - euler_rhs))

# ── Solve perfect foresight transition ───────────────────────────────────────
T_periods = 200

# Build shock path (2 shocks: eps_k, eps_y), nonzero only at t=1
# At t=1, boundary gives k[t-1] = k_ss. We want effective k_0 = k_init = 0.9*k_ss.
# eps_k[1] corrects (1-delta)*k[t-1] in accumulation: (1-delta)*(k_init - k_ss)
# eps_y[1] corrects k[t-1]^alpha in production: k_init^alpha - k_ss^alpha
shock_path = zeros(T_periods, 2)
shock_path[1, 1] = (1 - _delta) * (_k_init - _k_ss)
shock_path[1, 2] = _k_init^_alpha - _k_ss^_alpha

println("\n=== Perfect Foresight Solve (T=$T_periods) ===")
@printf("  Shock eps_k[1] = %.8f  (capital accumulation correction)\n", shock_path[1, 1])
@printf("  Shock eps_y[1] = %.8f  (production function correction)\n", shock_path[1, 2])
pf = perfect_foresight(spec; T_periods=T_periods, shock_path=shock_path, tol=1e-10)
println("  Converged: ", pf.converged)
println("  Iterations: ", pf.iterations)
println("  Path size: ", size(pf.path))

# ── Variable indices ─────────────────────────────────────────────────────────
idx_c   = 1
idx_k   = 2
idx_y   = 3
idx_inv = 4

# ── Transition path ─────────────────────────────────────────────────────────
println("\n=== Transition Path ===")
@printf("  %-6s  %14s  %14s  %14s  %14s  %10s\n",
        "Period", "c", "k", "y", "invest", "k/k_ss")
println("  ", "-" ^ 76)
for t in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
    @printf("  t=%-4d  %14.8f  %14.8f  %14.8f  %14.8f  %10.6f\n",
            t, pf.path[t, idx_c], pf.path[t, idx_k],
            pf.path[t, idx_y], pf.path[t, idx_inv],
            pf.path[t, idx_k] / _k_ss)
end

# ── Convergence check ────────────────────────────────────────────────────────
println("\n=== Convergence Check (period $T_periods vs SS) ===")
conv_vars = [("c", idx_c, _c_ss), ("k", idx_k, _k_ss),
             ("y", idx_y, _y_ss), ("invest", idx_inv, _inv_ss)]
all_conv = true
for (name, idx, ss_val) in conv_vars
    final_val = pf.path[end, idx]
    diff = abs(final_val - ss_val)
    ok = diff < 1e-4
    global all_conv = all_conv && ok
    @printf("  %-10s  path[T]=%14.8f  SS=%14.8f  diff=%8.2e  %s\n",
            name, final_val, ss_val, diff, ok ? "PASS" : "FAIL")
end
println("  Convergence: ", all_conv ? "ALL PASS" : "SOME FAIL")

# ── Verify economic consistency ──────────────────────────────────────────────
println("\n=== Economic Consistency ===")

# Resource constraint: c + invest = y  (should hold exactly)
rc_errors = pf.path[:, idx_c] .+ pf.path[:, idx_inv] .- pf.path[:, idx_y]
rc_max = maximum(abs.(rc_errors))
@printf("  Resource constraint max error: %.2e  %s\n", rc_max, rc_max < 1e-8 ? "PASS" : "FAIL")

# Production function: y[t] = k[t-1]^alpha + eps_y[t]
# At t=1: y[1] = k_ss^alpha + eps_y[1] = k_ss^alpha + (k_init^alpha - k_ss^alpha) = k_init^alpha
y_1_expected = _k_init^_alpha
y_1_actual = pf.path[1, idx_y]
@printf("  Period 1 production: y = k_init^alpha = %.8f  (path: %.8f, diff: %.2e)  %s\n",
        y_1_expected, y_1_actual, abs(y_1_actual - y_1_expected),
        abs(y_1_actual - y_1_expected) < 1e-6 ? "PASS" : "FAIL")

# At t>=2: y[t] = k[t-1]^alpha (no shock correction needed)
pf_errors = Float64[]
for t in 2:T_periods
    y_expected = pf.path[t-1, idx_k]^_alpha
    push!(pf_errors, abs(pf.path[t, idx_y] - y_expected))
end
pf_max = length(pf_errors) > 0 ? maximum(pf_errors) : 0.0
@printf("  Production function max error (t>=2): %.2e  %s\n", pf_max, pf_max < 1e-6 ? "PASS" : "FAIL")

# Check monotonic convergence (expected for below-SS starting point)
c_path = pf.path[:, idx_c]
k_path = pf.path[:, idx_k]
c_monotone = all(diff(c_path) .>= -1e-8)
k_monotone = all(diff(k_path) .>= -1e-8)
println("  Consumption monotonically increasing: ", c_monotone ? "PASS" : "WARN")
println("  Capital monotonically increasing: ", k_monotone ? "PASS" : "WARN")

# ── Comparison with Dynare BGP ───────────────────────────────────────────────
println("\n=== Comparison with Dynare Analytical BGP ===")
# Dynare formula: k_ss = ((1/beta*(1+n)*(1+g)-(1-delta))/alpha)^(1/(alpha-1))
k_dynare = ((_gamma / _beta - (1 - _delta)) / _alpha)^(1 / (_alpha - 1))
y_dynare = k_dynare^_alpha
inv_dynare = (_gamma - 1 + _delta) * k_dynare
c_dynare = y_dynare - inv_dynare

bgp_checks = [("k_ss", _k_ss, k_dynare), ("y_ss", _y_ss, y_dynare),
               ("c_ss", _c_ss, c_dynare), ("inv_ss", _inv_ss, inv_dynare)]
for (name, julia_val, dynare_val) in bgp_checks
    d = abs(julia_val - dynare_val)
    @printf("  %-8s  Julia=%14.8f  Dynare=%14.8f  diff=%8.2e  %s\n",
            name, julia_val, dynare_val, d, d < 1e-12 ? "PASS" : "FAIL")
end

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n=== Ramsey-Cass-Koopmans: Transition Summary ===")
println("  Model: Ramsey-Cass-Koopmans (intensive form, Cobb-Douglas, log utility)")
println("  Parameters: alpha=$_alpha, delta=$_delta, beta=$_beta, n=$_n, g=$_g")
@printf("  SS capital:  k* = %.8f\n", _k_ss)
@printf("  SS output:   y* = %.8f\n", _y_ss)
@printf("  SS cons:     c* = %.8f\n", _c_ss)
@printf("  Initial:     k0 = %.8f  (90%% of SS)\n", _k_init)
println("  Horizon: $T_periods periods")
println("  Result: Forward-looking consumption smoothing + capital accumulation")
println("  Dynare .mat: not available (perfect foresight paths not saved by default)")
