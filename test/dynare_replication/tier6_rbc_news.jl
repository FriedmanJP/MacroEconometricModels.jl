# Tier 6: RBC News Shock Model (Beaudry & Portier 2004 style) — Dynare Replication
# Dynare source: DSGE_mod/RBC_news_shock_model/RBC_news_shock_model.mod
# Reference: Beaudry, P. & Portier, F. (2004), "An exploration into Pigou's theory
#   of cycles", Journal of Monetary Economics, 51, pp. 1183-1216.
#
# Standard RBC model with additively separable utility and TFP news shocks.
# The TFP process includes an 8-period anticipated (news) shock:
#   z[t] = rho * z[t-1] + eps_z_surprise[t] + eps_z_news[t-8]
#
# The @dsge parser automatically augments deep exogenous lags via auxiliary variables:
#   eps_z_news[t-8] → chain of 8 auxiliary endogenous vars (__news_eps_z_news_1..8)
#
# Variables are in exp() form (log-linearization via variable substitution).
using MacroEconometricModels, MAT, Printf

# ── Parameters (calibrated to US data, from Dynare .mod) ──────────────────
const _sigma  = 1.0
const _alpha  = 0.33
const _i_y    = 0.25
const _k_y    = 10.4
const _x      = 0.0055     # per-capita output growth
const _n      = 0.0027     # population growth
const _rhoz   = 0.97
const _gammax = (1.0 + _n) * (1.0 + _x)

# Derived parameters (from Dynare steady_state_model block)
const _delta  = _i_y / _k_y - _x - _n - _n * _x
const _beta   = _gammax / (_alpha / _k_y + (1.0 - _delta))

# Steady state (calibrated to l_ss = 0.33)
const _l_ss = 0.33
const _k_ss = ((1.0 / _beta * _gammax - (1.0 - _delta)) / _alpha)^(1.0 / (_alpha - 1.0)) * _l_ss
const _i_ss = (_x + _n + _delta + _n * _x) * _k_ss
const _y_ss = _k_ss^_alpha * _l_ss^(1.0 - _alpha)
const _c_ss = _y_ss - _i_ss
const _psi  = (1.0 - _alpha) * (_k_ss / _l_ss)^_alpha * (1.0 - _l_ss) / _c_ss^_sigma
const _w_ss = (1.0 - _alpha) * _y_ss / _l_ss
const _r_ss = 4.0 * _alpha * _y_ss / _k_ss

# ── Model specification ───────────────────────────────────────────────────
# The model uses exp() variable substitution (y = log(Y), etc.)
# except r which is already in percent.
# The news shock eps_z_news[t-8] triggers automatic augmentation.
spec = @dsge begin
    parameters: sigma_p = 1.0, alpha_p = 0.33, rhoz_p = 0.97,
                beta_p = 0.9924281390931616, gammax_p = 1.0082148499999999,
                delta_p = 0.015823611538461537, psi_p = 1.813737373737375
    endogenous: y, c, k, l, z, r, w, invest
    exogenous: eps_z_news, eps_z_surprise

    # (1) Euler equation
    exp(c[t])^(-sigma_p) = beta_p / gammax_p * exp(c[t+1])^(-sigma_p) * (alpha_p * exp(z[t+1]) * (exp(k[t]) / exp(l[t+1]))^(alpha_p - 1.0) + (1.0 - delta_p))

    # (2) Labor FOC
    psi_p * exp(c[t])^sigma_p * 1.0 / (1.0 - exp(l[t])) = exp(w[t])

    # (3) LOM capital
    gammax_p * exp(k[t]) = (1.0 - delta_p) * exp(k[t-1]) + exp(invest[t])

    # (4) Resource constraint
    exp(y[t]) = exp(invest[t]) + exp(c[t])

    # (5) Production function
    exp(y[t]) = exp(z[t]) * exp(k[t-1])^alpha_p * exp(l[t])^(1.0 - alpha_p)

    # (6) Real wage / firm FOC labor
    exp(w[t]) = (1.0 - alpha_p) * exp(y[t]) / exp(l[t])

    # (7) Annualized real interest rate / firm FOC capital
    r[t] = 4.0 * alpha_p * exp(y[t]) / exp(k[t-1])

    # (8) TFP process with 8-period anticipated news shock
    z[t] = rhoz_p * z[t-1] + eps_z_surprise[t] + eps_z_news[t-8]

    steady_state = begin
        gammax_v = gammax_p
        delta_v  = delta_p
        beta_v   = beta_p
        l_s  = 0.33
        k_s  = ((1.0/beta_v * gammax_v - (1.0 - delta_v)) / alpha_p)^(1.0/(alpha_p - 1.0)) * l_s
        i_s  = (gammax_v - 1.0 + delta_v) * k_s
        y_s  = k_s^alpha_p * l_s^(1.0 - alpha_p)
        c_s  = y_s - i_s
        w_s  = (1.0 - alpha_p) * y_s / l_s
        r_s  = 4.0 * alpha_p * y_s / k_s
        z_s  = 0.0
        # Variables in log form (exp(var) = level)
        invest_s = log(i_s)
        w_log    = log(w_s)
        y_log    = log(y_s)
        k_log    = log(k_s)
        c_log    = log(c_s)
        l_log    = log(l_s)
        # Return: y, c, k, l, z, r, w, invest, then 8 news aux vars (all zero)
        vcat([y_log, c_log, k_log, l_log, z_s, r_s, w_log, invest_s],
             zeros(8))
    end
end

spec = compute_steady_state(spec)

# ── Verify model setup ────────────────────────────────────────────────────
println("=" ^ 60)
println("  RBC News Shock Model (Beaudry & Portier 2004 style)")
println("=" ^ 60)
println("  n_endog = ", spec.n_endog, " (8 original + 8 news auxiliaries)")
println("  n_exog  = ", spec.n_exog)
println("  augmented = ", spec.augmented)

# ── Solve ──────────────────────────────────────────────────────────────────
sol = solve(spec; method=:gensys)
println("  is_determined = ", is_determined(sol))

# ── Load Dynare reference ─────────────────────────────────────────────────
dynare_mat = joinpath(@__DIR__, "dynare_results", "rbc_news.mat")
if !isfile(dynare_mat)
    println("  (No Dynare .mat file found)")
    exit(1)
end

data = matread(dynare_mat)
d_ss = vec(data["steady_state"])
d_irfs = data["irfs"]
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

# ── 1. Steady State Comparison ────────────────────────────────────────────
println("\n=== Steady State ===")
# Our first 8 variables are the original ones; Dynare has 16 (8 original + 8 AUX_EXO_LAG)
# Map: y=1, c=2, k=3, l=4, z=5, r=6, w=7, invest=8
our_to_dynare = Dict(
    "y" => 1, "c" => 2, "k" => 3, "l" => 4,
    "z" => 5, "r" => 6, "w" => 7, "invest" => 8
)

our_names = string.(spec.original_endog)

@printf("  %-10s %14s %14s %10s %s\n", "Variable", "Julia", "Dynare", "diff", "")
all_ss_pass = true
for (i, n) in enumerate(our_names)
    global all_ss_pass
    di = get(our_to_dynare, n, 0)
    di == 0 && continue
    diff = abs(spec.steady_state[i] - d_ss[di])
    ok = diff < 1e-6
    all_ss_pass = all_ss_pass && ok
    @printf("  %-10s %14.8f %14.8f %10.2e %s\n",
            n, spec.steady_state[i], d_ss[di], diff, ok ? "PASS" : "FAIL")
end
println("  Steady State: ", all_ss_pass ? "ALL PASS" : "SOME FAIL")

# ── 2. IRF Comparison ────────────────────────────────────────────────────
# Standard IRFs from stoch_simul(order=1, irf=40)
# Dynare reports IRFs in log-deviation form (since model uses exp() substitution)
ir = irf(sol, 40)

println("\n=== IRF Comparison (H=40) ===")

# Build maps for our augmented variable space
# Original variables are at indices 1:8 in the augmented spec
orig_var_idx = Dict{String,Int}()
for (i, n) in enumerate(our_names)
    orig_var_idx[n] = i
end

# Shock map: in our augmented model, eps_z_news=1, eps_z_surprise=2
shock_names = ["eps_z_news", "eps_z_surprise"]

@printf("  %-30s %12s %s\n", "Response", "MaxAbsDiff", "Status")
println("  ", "-" ^ 55)
all_irf_pass = true
for (si, sn) in enumerate(shock_names)
    for (vn, vi) in orig_var_idx
        global all_irf_pass
        d_key = vn * "_" * sn
        haskey(d_irfs, d_key) || continue
        d_vec = vec(d_irfs[d_key])
        H = min(length(d_vec), 40)

        # Our IRF values: ir.values[h, var_idx, shock_idx]
        j_vec = [ir.values[h, vi, si] for h in 1:H]

        max_diff = maximum(abs.(j_vec .- d_vec[1:H]))

        # Use relative tolerance for large responses
        max_abs_d = maximum(abs.(d_vec[1:H]))
        if max_abs_d > 1e-6
            rel_diff = max_diff / max_abs_d
            ok = rel_diff < 0.02 || max_diff < 1e-4  # 2% relative or 1e-4 absolute
        else
            ok = max_diff < 1e-4
        end
        all_irf_pass = all_irf_pass && ok
        @printf("  %-30s %12.2e %s\n", d_key, max_diff, ok ? "PASS" : "FAIL")
    end
end

# ── 3. Detailed IRF comparison for key variables ─────────────────────────
println("\n=== Detailed IRF: z to eps_z_news (first 15 periods) ===")
z_idx = orig_var_idx["z"]
news_shock_idx = 1
z_dynare = vec(d_irfs["z_eps_z_news"])
@printf("  %4s %14s %14s %10s\n", "h", "Julia", "Dynare", "diff")
for h in 1:min(15, length(z_dynare))
    j_val = ir.values[h, z_idx, news_shock_idx]
    d_val = z_dynare[h]
    @printf("  %4d %14.8f %14.8f %10.2e\n", h, j_val, d_val, abs(j_val - d_val))
end

println("\n=== Detailed IRF: y to eps_z_news (first 15 periods) ===")
y_idx = orig_var_idx["y"]
y_dynare = vec(d_irfs["y_eps_z_news"])
@printf("  %4s %14s %14s %10s\n", "h", "Julia", "Dynare", "diff")
for h in 1:min(15, length(y_dynare))
    j_val = ir.values[h, y_idx, news_shock_idx]
    d_val = y_dynare[h]
    @printf("  %4d %14.8f %14.8f %10.2e\n", h, j_val, d_val, abs(j_val - d_val))
end

println("\n=== Detailed IRF: y to eps_z_surprise (first 15 periods) ===")
surp_shock_idx = 2
y_surp_dynare = vec(d_irfs["y_eps_z_surprise"])
@printf("  %4s %14s %14s %10s\n", "h", "Julia", "Dynare", "diff")
for h in 1:min(15, length(y_surp_dynare))
    j_val = ir.values[h, y_idx, surp_shock_idx]
    d_val = y_surp_dynare[h]
    @printf("  %4d %14.8f %14.8f %10.2e\n", h, j_val, d_val, abs(j_val - d_val))
end

# ── 4. News shock behavior check ─────────────────────────────────────────
println("\n=== News Shock Behavior Check ===")
# Key property: eps_z_news at t=1 should NOT affect z until t=9 (8-period anticipation)
# z IRF to news shock should be ~0 for periods 1-8, then nonzero from period 9
z_news_irfs = [ir.values[h, z_idx, news_shock_idx] for h in 1:40]
max_z_pre_arrive = maximum(abs.(z_news_irfs[1:8]))
z_at_arrival = z_news_irfs[9]
println("  max|z(1:8)| to news = ", @sprintf("%.2e", max_z_pre_arrive),
        " (should be ~0)")
println("  z(9) to news         = ", @sprintf("%.6f", z_at_arrival),
        " (should be nonzero)")
news_timing_ok = max_z_pre_arrive < 1e-8 && abs(z_at_arrival) > 0.1
println("  News timing correct  = ", news_timing_ok ? "PASS" : "FAIL")

# Anticipation effects: y and c should respond BEFORE z arrives (forward-looking)
y_news_irfs = [ir.values[h, y_idx, news_shock_idx] for h in 1:40]
c_news_irfs = [ir.values[h, orig_var_idx["c"], news_shock_idx] for h in 1:40]
y_pre = maximum(abs.(y_news_irfs[1:8]))
c_pre = maximum(abs.(c_news_irfs[1:8]))
println("  max|y(1:8)| to news = ", @sprintf("%.6f", y_pre), " (should be >0, anticipation)")
println("  max|c(1:8)| to news = ", @sprintf("%.6f", c_pre), " (should be >0, anticipation)")
anticipation_ok = y_pre > 0.01 && c_pre > 0.01
println("  Anticipation effect = ", anticipation_ok ? "PASS" : "FAIL")

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
overall = all_ss_pass && all_irf_pass && news_timing_ok && anticipation_ok
println("  Summary:")
println("    Steady State:     ", all_ss_pass ? "ALL PASS" : "SOME FAIL")
println("    IRFs:             ", all_irf_pass ? "ALL PASS" : "SOME FAIL")
println("    News timing:      ", news_timing_ok ? "PASS" : "FAIL")
println("    Anticipation:     ", anticipation_ok ? "PASS" : "FAIL")
println("    Overall:          ", overall ? "ALL PASS" : "SOME FAIL")
println("=" ^ 60)
