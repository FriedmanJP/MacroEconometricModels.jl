# DSGE Bayesian Estimation Suite
#
# Part A: SW07 full-scale estimation (3 params, SMC)
# Part B: Toy model estimation (all methods: SMC, MH, SMC²)
#
# Reference: Smets & Wouters (2007), AER 97(3), 586-606.

using MacroEconometricModels, Printf, LinearAlgebra, Random, Statistics, Distributions

const _sw = MacroEconometricModels._suppress_warnings

println("=" ^ 70)
println("  DSGE Bayesian Estimation Suite")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════════════
# Part A: SW07 — Full-Scale Estimation
# ═══════════════════════════════════════════════════════════════════════════
println("\n  Part A: Smets-Wouters 2007 (40 vars, 7 shocks)")
println("  " * "-" ^ 50)

include(joinpath(@__DIR__, "sw07_model.jl"))

Sigma_e = diagm([stderr_ea, stderr_eb, stderr_eg, stderr_eqs,
                  stderr_em, stderr_epinf, stderr_ew].^2)
effective_ss = (I - sol.G1) \ sol.C_sol

# Generate synthetic observables
T_obs = 150
sim_raw = simulate(sol, T_obs; rng=MersenneTwister(2007))

obs_vars = [:dy, :dc, :dinve, :dw, :pinfobs, :robs, :labobs]
obs_idx = [vi[v] for v in obs_vars]

Y_obs = zeros(7, T_obs)
for t in 1:T_obs, (oi, si) in enumerate(obs_idx)
    Y_obs[oi, t] = sim_raw[t, si] + effective_ss[si]
end

println("  Data: T=$T_obs, 7 observables")

# Kalman log-likelihood at mode
Z_k = zeros(7, spec.n_endog)
for (oi, si) in enumerate(obs_idx)
    Z_k[oi, si] = 1.0
end
d_k = Float64[effective_ss[si] for si in obs_idx]
H_k = 1e-8 * Matrix{Float64}(I, 7, 7)

ss_k = MacroEconometricModels.DSGEStateSpace{Float64}(
    sol.G1, sol.impact, Z_k, d_k, H_k, Sigma_e
)
ll_mode = MacroEconometricModels._kalman_loglikelihood(ss_k, Y_obs)
println("  logL at mode:  $(round(ll_mode, digits=2))")
@assert isfinite(ll_mode)

# Likelihood function test
est_params_sw = [:crhoa_p, :crhob_p, :crr_p]
true_vals_sw = Float64[param_values[p] for p in est_params_sw]

ll_fn = MacroEconometricModels._build_likelihood_fn(
    spec, est_params_sw, Y_obs, obs_vars, nothing, :gensys, NamedTuple()
)
ll_true = ll_fn(true_vals_sw)
ll_wrong = ll_fn([0.5, 0.1, 0.5])
@assert isfinite(ll_true) && ll_true > ll_wrong
println("  logL(true) > logL(wrong): PASS")

# SMC estimation
priors_sw = Dict(
    :crhoa_p => Beta(5.0, 1.0),
    :crhob_p => Beta(1.5, 5.0),
    :crr_p   => Beta(5.0, 2.0),
)

println("  Running SMC (N=500, 3 params)...")
result_sw = estimate_dsge_bayes(
    spec, Y_obs', true_vals_sw;
    priors=priors_sw, method=:smc, observables=obs_vars,
    n_smc=500, n_mh_steps=2, ess_target=0.5,
    rng=MersenneTwister(42)
)

ps = posterior_summary(result_sw)
sw_pass = true
@printf("\n  %-12s %8s %8s %8s %8s\n", "Parameter", "True", "Post.μ", "Post.σ", "In CI?")
println("  ", "-" ^ 44)
for (i, pn) in enumerate(est_params_sw)
    info = ps[pn]
    tv = true_vals_sw[i]
    in_ci = info[:ci_lower] <= tv <= info[:ci_upper]
    global sw_pass &= in_ci
    @printf("  %-12s %8.4f %8.4f %8.4f %8s\n",
            pn, tv, info[:mean], info[:std], in_ci ? "✓" : "✗")
end
println("  SW07 SMC: ", sw_pass ? "PASS" : "FAIL")
println("  Marginal lik: $(round(result_sw.log_marginal_likelihood, digits=2))")

# ═══════════════════════════════════════════════════════════════════════════
# Part B: Toy Model — All Estimation Methods
# ═══════════════════════════════════════════════════════════════════════════
println("\n\n" * "=" ^ 70)
println("  Part B: Toy Model (all methods)")
println("=" ^ 70)

toy_spec = _sw() do
    @dsge begin
        parameters: rho_y = 0.5, rho_pi = 0.5, phi = 0.5, sigma_y = 1.0, sigma_pi = 1.0
        endogenous: y, pi_v
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + sigma_y * eps_y[t]
        pi_v[t] = rho_pi * pi_v[t-1] + phi * y[t-1] + sigma_pi * eps_pi[t]
        steady_state = [0.0, 0.0]
    end
end
toy_spec = compute_steady_state(toy_spec)

true_toy_spec = _sw() do
    @dsge begin
        parameters: rho_y = 0.8, rho_pi = 0.6, phi = 0.5, sigma_y = 1.0, sigma_pi = 1.0
        endogenous: y, pi_v
        exogenous: eps_y, eps_pi
        y[t] = rho_y * y[t-1] + sigma_y * eps_y[t]
        pi_v[t] = rho_pi * pi_v[t-1] + phi * y[t-1] + sigma_pi * eps_pi[t]
        steady_state = [0.0, 0.0]
    end
end
true_toy_spec = compute_steady_state(true_toy_spec)
toy_sol = _sw() do; solve(true_toy_spec; method=:gensys); end

toy_data = simulate(toy_sol, 400; rng=MersenneTwister(123))
toy_priors = Dict(:rho_y => Beta(2, 2), :rho_pi => Beta(2, 2))
toy_true = [0.8, 0.6]

function _check_recovery(result, toy_true, label)
    ps = posterior_summary(result)
    pass = true
    for (i, pn) in enumerate([:rho_y, :rho_pi])
        info = ps[pn]
        in_ci = info[:ci_lower] <= toy_true[i] <= info[:ci_upper]
        pass &= in_ci
        @printf("      %-8s true=%.2f  post=%.3f±%.3f  %s\n",
                pn, toy_true[i], info[:mean], info[:std], in_ci ? "✓" : "✗")
    end
    println("      $label: ", pass ? "PASS" : "FAIL")
    return pass
end

# B1: SMC
println("\n  B1: SMC + Kalman")
result_smc = _sw() do
    estimate_dsge_bayes(toy_spec, toy_data, [0.5, 0.5];
        priors=toy_priors, method=:smc, observables=[:y, :pi_v],
        n_smc=500, n_mh_steps=2, ess_target=0.5, rng=MersenneTwister(42))
end
smc_pass = _check_recovery(result_smc, toy_true, "SMC")

# B2: MH
println("\n  B2: RWMH")
result_mh = _sw() do
    estimate_dsge_bayes(toy_spec, toy_data, [0.5, 0.5];
        priors=toy_priors, method=:mh, observables=[:y, :pi_v],
        n_draws=5000, burnin=1000, rng=MersenneTwister(42))
end
mh_pass = _check_recovery(result_mh, toy_true, "MH")
println("      Acceptance: $(round(result_mh.acceptance_rate * 100, digits=1))%")

# B3: SMC²
println("\n  B3: SMC²")
result_smc2 = _sw() do
    estimate_dsge_bayes(toy_spec, toy_data, [0.5, 0.5];
        priors=toy_priors, method=:smc2, observables=[:y, :pi_v],
        n_smc=200, n_particles=100, n_mh_steps=1, ess_target=0.5,
        measurement_error=[0.1, 0.1], rng=MersenneTwister(42))
end
smc2_pass = _check_recovery(result_smc2, toy_true, "SMC²")

# B4: Bayes factor
println("\n  B4: Model Comparison")
bf = bayes_factor(result_smc, result_smc)
bf_pass = abs(bf) < 1.0
println("      log BF(M,M) = $(round(bf, digits=4)) ≈ 0: ", bf_pass ? "PASS" : "FAIL")

# B5: Posterior IRF/FEVD/Simulate
println("\n  B5: Posterior Analysis")
birf  = _sw() do; irf(result_smc, 20; n_draws=10, rng=MersenneTwister(1)); end
bfevd = _sw() do; fevd(result_smc, 20; n_draws=10, rng=MersenneTwister(1)); end
bsim  = _sw() do; simulate(result_smc, 50; n_draws=10, rng=MersenneTwister(1)); end
@assert birf isa BayesianImpulseResponse && all(isfinite.(birf.point_estimate))
@assert bfevd isa BayesianFEVD && all(x -> 0 <= x <= 1+1e-6, bfevd.point_estimate)
@assert bsim isa BayesianDSGESimulation
println("      Posterior IRF/FEVD/Simulate: PASS")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("  Summary")
println("=" ^ 70)
println("  A. SW07 (40 vars, 7 shocks, 3 est. params):")
println("     Kalman logL:       PASS")
println("     SMC recovery:      $(sw_pass ? "PASS" : "FAIL")")
println("  B. Toy model (2 vars, 2 shocks, 2 est. params):")
println("     SMC + Kalman:      $(smc_pass ? "PASS" : "FAIL")")
println("     RWMH:              $(mh_pass ? "PASS" : "FAIL")")
println("     SMC²:              $(smc2_pass ? "PASS" : "FAIL")")
println("     Bayes factor:      $(bf_pass ? "PASS" : "FAIL")")
println("     Posterior analysis: PASS")
println("=" ^ 70)
