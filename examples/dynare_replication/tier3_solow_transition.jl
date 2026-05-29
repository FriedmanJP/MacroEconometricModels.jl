# Tier 3: Solow Model Steady-State Transition — Perfect Foresight
# Dynare source: DSGE_mod/Solow_model/Solow_SS_transition.mod
#
# Solow-Swan economy in intensive (per effective worker) form with Cobb-Douglas
# production.  Starts capital at 90% of steady state and traces the deterministic
# transition to the balanced growth path.
#
# The Dynare file uses initval/endval + perfect_foresight_solver with 200 periods.
# Because the model is purely backward-looking the transition is unique.
#
# Approach:  The Dynare model writes production as y = k^alpha where k is
# predetermined (decided last period).  In our timing convention:
#   y[t]   = k[t-1]^alpha              (production uses lagged capital)
#   (1+ng)*k[t] = (1-delta)*k[t-1] + invest[t]   (capital decided today)
#
# The perfect_foresight() solver uses spec.steady_state for both the initial
# boundary (y_0 = y_ss) and the terminal boundary (y_{T+1} = y_ss).  We model
# the initial capital displacement as a one-period additive shock eps_k in the
# accumulation equation that shifts the *lagged* capital term at t=1 (since the
# boundary provides k[t-1]=k_ss at t=1, the shock corrects it to k_init).
#
# ============================================================================
using MacroEconometricModels, Printf

# ── Parameters (from .mod file) ──────────────────────────────────────────────
const _s     = 0.2
const _alpha = 0.3
const _delta = 0.1
const _n     = 0.01
const _g     = 0.02
const _gamma = 1 + _n + _g + _n * _g

# ── Analytical steady state in intensive form ────────────────────────────────
# (1+ng) k_ss = (1-delta) k_ss + s k_ss^alpha
# => k_ss = ((delta+n+g+ng)/s)^(1/(alpha-1))
const _k_ss     = ((_delta + _n + _g + _n * _g) / _s)^(1 / (_alpha - 1))
const _y_ss     = _k_ss^_alpha
const _c_ss     = (1 - _s) * _y_ss
const _inv_ss   = _s * _y_ss
const _k_init   = 0.9 * _k_ss  # initial capital (90% of SS)

# ── Model specification ──────────────────────────────────────────────────────
# Timing: k[t] is capital decided at period t.  Production at t uses k[t-1].
# This matches Dynare's predetermined variable treatment.
#
# eps_k enters the accumulation equation.  At t=1 the boundary provides
# k[t-1] = k_ss.  We set eps_k[1] so that the effective lagged capital becomes
# k_init:
#   eps_k[1] = (1-delta)*(k_init - k_ss)  + (k_init^alpha - k_ss^alpha)
# The first term corrects the (1-delta)*k[t-1] piece in the accumulation eqn;
# the second corrects for the fact that investment s*y also changes with y=k[t-1]^alpha.
# Actually, since invest = s*y = s*k[t-1]^alpha, and we also have the production
# and saving equations to satisfy, the shock only needs to correct the direct
# k[t-1] term in the accumulation equation.  The production equation y[t]=k[t-1]^alpha
# at t=1 uses k[t-1]=k_ss from the boundary — which is wrong if we want k_0=0.9*k_ss.
#
# To handle this cleanly, we add a SECOND shock eps_y that enters the production
# function at t=1 to correct for the mismatch:
#   y[t] = k[t-1]^alpha + eps_y[t]
#   eps_y[1] = k_init^alpha - k_ss^alpha   (correction to production)
#   eps_k[1] = (1-delta)*(k_init - k_ss)   (correction to capital accumulation)

spec = @dsge begin
    parameters: s_rate = 0.2, alpha = 0.3, delta = 0.1, npop = 0.01, gtech = 0.02
    endogenous: k, y, c, invest, log_k, log_y, log_c, log_invest,
                g_k_intensive, g_k_per_capita, g_k_aggregate
    exogenous: eps_k, eps_y

    # Law of motion capital (intensive form)
    # k[t] is capital decided today = (1-delta)*k[t-1] + invest[t]
    (1 + npop + gtech + npop * gtech) * k[t] = (1 - delta) * k[t-1] + invest[t] + eps_k[t]

    # Resource constraint
    invest[t] + c[t] = y[t]

    # Behavioral rule (Solow saving)
    c[t] = (1 - s_rate) * y[t]

    # Production function (intensive form: y = k(-1)^alpha, predetermined k)
    y[t] = k[t-1]^alpha + eps_y[t]

    # Log definitions
    log_y[t] = log(y[t])
    log_c[t] = log(c[t])
    log_invest[t] = log(invest[t])
    log_k[t] = log(k[t])

    # Growth rates
    g_k_intensive[t] = log(k[t]) - log(k[t-1])
    g_k_per_capita[t] = g_k_intensive[t] + gtech
    g_k_aggregate[t] = g_k_intensive[t] + gtech + npop

    steady_state = begin
        k_ss_val = ((delta + npop + gtech + npop * gtech) / s_rate)^(1 / (alpha - 1))
        y_ss_val = k_ss_val^alpha
        c_ss_val = (1 - s_rate) * y_ss_val
        inv_ss_val = s_rate * y_ss_val
        log_k_ss = log(k_ss_val)
        log_y_ss = log(y_ss_val)
        log_c_ss = log(c_ss_val)
        log_inv_ss = log(inv_ss_val)
        g_k_int_ss = 0.0
        g_k_pc_ss = gtech
        g_k_agg_ss = gtech + npop
        [k_ss_val, y_ss_val, c_ss_val, inv_ss_val,
         log_k_ss, log_y_ss, log_c_ss, log_inv_ss,
         g_k_int_ss, g_k_pc_ss, g_k_agg_ss]
    end
end

# ── Compute steady state ─────────────────────────────────────────────────────
spec = compute_steady_state(spec)
ss = spec.steady_state
println("=" ^ 70)
println("  Solow Model Steady-State Transition (Perfect Foresight)")
println("=" ^ 70)

println("\n=== Steady State (Intensive Form) ===")
ss_names = ["k", "y", "c", "invest", "log_k", "log_y", "log_c", "log_invest",
            "g_k_intensive", "g_k_per_capita", "g_k_aggregate"]
analytical_ss = [_k_ss, _y_ss, _c_ss, _inv_ss, log(_k_ss), log(_y_ss), log(_c_ss),
                 log(_inv_ss), 0.0, _g, _g + _n]
all_ss_pass = true
for (i, name) in enumerate(ss_names)
    diff = abs(ss[i] - analytical_ss[i])
    ok = diff < 1e-10
    global all_ss_pass = all_ss_pass && ok
    @printf("  %-18s  Julia=%14.8f  Analytical=%14.8f  diff=%8.2e  %s\n",
            name, ss[i], analytical_ss[i], diff, ok ? "PASS" : "FAIL")
end
println("  SS overall: ", all_ss_pass ? "ALL PASS" : "SOME FAIL")

# ── Solve perfect foresight transition ───────────────────────────────────────
T_periods = 200

# Build shock path: two shocks (eps_k, eps_y), nonzero only at t=1
# At t=1, boundary gives k[t-1] = k_ss. We want effective k_0 = k_init = 0.9*k_ss.
# eps_k[1] corrects (1-delta)*k[t-1]: shift by (1-delta)*(k_init - k_ss)
# eps_y[1] corrects k[t-1]^alpha: shift by k_init^alpha - k_ss^alpha
shock_path = zeros(T_periods, 2)
shock_path[1, 1] = (1 - _delta) * (_k_init - _k_ss)          # eps_k correction
shock_path[1, 2] = _k_init^_alpha - _k_ss^_alpha              # eps_y correction

println("\n=== Perfect Foresight Solve (T=$T_periods) ===")
@printf("  Shock eps_k[1] = %.8f  (capital accumulation correction)\n", shock_path[1, 1])
@printf("  Shock eps_y[1] = %.8f  (production function correction)\n", shock_path[1, 2])
pf = perfect_foresight(spec; T_periods=T_periods, shock_path=shock_path, tol=1e-10)
println("  Converged: ", pf.converged)
println("  Iterations: ", pf.iterations)
println("  Path size: ", size(pf.path))

# ── Variable indices ─────────────────────────────────────────────────────────
idx_k   = 1
idx_y   = 2
idx_c   = 3
idx_inv = 4

# ── Transition path ─────────────────────────────────────────────────────────
println("\n=== Transition Path ===")
@printf("  %-6s  %14s  %14s  %14s  %14s  %10s\n",
        "Period", "k", "y", "c", "invest", "k/k_ss")
println("  ", "-" ^ 76)
for t in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
    @printf("  t=%-4d  %14.8f  %14.8f  %14.8f  %14.8f  %10.6f\n",
            t, pf.path[t, idx_k], pf.path[t, idx_y],
            pf.path[t, idx_c], pf.path[t, idx_inv],
            pf.path[t, idx_k] / _k_ss)
end

# ── Direct forward simulation for comparison ─────────────────────────────────
# Since Solow is purely backward-looking, we can verify with direct iteration:
#   y_t = k_{t-1}^alpha
#   invest_t = s * y_t
#   (1+ng) * k_t = (1-delta)*k_{t-1} + invest_t
println("\n=== Direct Forward Simulation (verification) ===")
k_direct = zeros(T_periods + 1)
k_direct[1] = _k_init  # k_0 = 0.9 * k_ss (initial capital)
y_direct = zeros(T_periods)
c_direct = zeros(T_periods)
inv_direct = zeros(T_periods)

for t in 1:T_periods
    y_direct[t] = k_direct[t]^_alpha
    inv_direct[t] = _s * y_direct[t]
    c_direct[t] = (1 - _s) * y_direct[t]
    k_direct[t+1] = ((1 - _delta) * k_direct[t] + inv_direct[t]) / _gamma
end

@printf("  %-6s  %14s  %14s  %14s  %14s\n",
        "Period", "k_direct", "y_direct", "c_direct", "inv_direct")
println("  ", "-" ^ 66)
for t in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]
    @printf("  t=%-4d  %14.8f  %14.8f  %14.8f  %14.8f\n",
            t, k_direct[t+1], y_direct[t], c_direct[t], inv_direct[t])
end

# Compare PF path with direct simulation
println("\n=== PF vs Direct Simulation Comparison ===")
@printf("  %-6s  %14s  %14s  %14s  %14s\n",
        "Period", "k(PF)", "k(Direct)", "diff(k)", "diff(y)")
println("  ", "-" ^ 66)
all_match = true
for t in [1, 2, 3, 5, 10, 20, 50, 100, 200]
    k_diff = abs(pf.path[t, idx_k] - k_direct[t+1])
    y_diff = abs(pf.path[t, idx_y] - y_direct[t])
    ok = k_diff < 1e-4 && y_diff < 1e-4
    global all_match = all_match && ok
    @printf("  t=%-4d  %14.8f  %14.8f  %10.2e  %10.2e  %s\n",
            t, pf.path[t, idx_k], k_direct[t+1], k_diff, y_diff,
            ok ? "PASS" : "FAIL")
end
println("  PF vs Direct: ", all_match ? "ALL PASS" : "SOME FAIL")

# ── Convergence check ────────────────────────────────────────────────────────
println("\n=== Convergence to SS ===")
conv_checks = [("k", pf.path[end, idx_k], _k_ss), ("y", pf.path[end, idx_y], _y_ss),
               ("c", pf.path[end, idx_c], _c_ss), ("invest", pf.path[end, idx_inv], _inv_ss)]
all_conv = true
for (name, val, ss_val) in conv_checks
    diff = abs(val - ss_val)
    ok = diff < 1e-4
    global all_conv = all_conv && ok
    @printf("  %-10s  path[T]=%14.8f  SS=%14.8f  diff=%8.2e  %s\n",
            name, val, ss_val, diff, ok ? "PASS" : "FAIL")
end
println("  Convergence: ", all_conv ? "ALL PASS" : "SOME FAIL")

# Resource constraint: c + invest = y
rc_max_err = maximum(abs.(pf.path[:, idx_c] .+ pf.path[:, idx_inv] .- pf.path[:, idx_y]))
@printf("\n  Resource constraint max error: %.2e  %s\n",
        rc_max_err, rc_max_err < 1e-8 ? "PASS" : "FAIL")

# Saving rule: c = (1-s)*y
sr_max_err = maximum(abs.(pf.path[:, idx_c] .- (1 - _s) .* pf.path[:, idx_y]))
@printf("  Saving rule max error: %.2e  %s\n",
        sr_max_err, sr_max_err < 1e-8 ? "PASS" : "FAIL")

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n=== Solow Model: Transition Summary ===")
println("  Model: Solow-Swan (intensive form, Cobb-Douglas)")
println("  Parameters: s=$_s, alpha=$_alpha, delta=$_delta, n=$_n, g=$_g")
@printf("  SS capital:  k* = %.8f\n", _k_ss)
@printf("  Initial:     k0 = %.8f  (90%% of SS)\n", _k_init)
println("  Horizon: $T_periods periods")
println("  Result: Capital monotonically converges from 0.9*k_ss to k_ss")
println("  Dynare .mat: not available (perfect foresight paths not saved by default)")
