# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
const _hh = MacroEconometricModels._hh
using LinearAlgebra
using SparseArrays
using Random
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end
if !@isdefined(NUMERICAL)
    const NUMERICAL = get(ENV, "MACRO_NUMERICAL_CI", "") == "1"
end

# Shared Huggett (1993) credit-limit −2 steady state (T209/#308): three testsets
# (the Table-1 SS loop, SSJ, and Reiter) recompute the identical cl=−2 equilibrium.
# Solve it ONCE here at the stricter (tol=5e-4) bar and reuse everywhere.
# Keep n_a=200 even under FAST — Table 1 atol=0.015 and the #234 @test_broken
# KS-SSJ items are not monotone in the grid (see HA Bayesian note below).
const _HUG_SPEC_M2 = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=200)
const _HUG_SS_M2 = compute_steady_state(_HUG_SPEC_M2; max_iter=FAST ? 80 : 200, tol=5e-4)

@testset "HA-DSGE Types" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 22: HA Bayesian estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "HA Bayesian estimation" begin
    spec = load_ha_example(:krusell_smith)
    # [T206] NOTE: the plan's asset-grid shrink (n_a 200→60/80) was dropped — perturbing the
    # KS-SSJ grid non-monotonically stabilizes the reduced realization and flips the #234
    # @test_broken truncation assertions to unexpected passes (n_a=60 flips T049's L1475;
    # n_a=80 also flips _build_ha_likelihood_fn's ll_val). Per the plan's flip-guard fallback
    # we keep the full-size spec and cut only draws + T_data (+ the shared-solve hoist).
    # The Ho-Kalman spectral radius is CHAOTIC, not monotone, in the calibration: the
    # a_max/grid_type change that fixed the asset-grid truncation was checked against all
    # three @test_broken items (all still -Inf at :geometric a_max=1000, whereas
    # :double_exp a_max=1000 flips all three). Re-measure them after ANY change to
    # a_max, n_a or grid_type — an unexpected pass is reported as a suite FAILURE.

    @testset "_update_ha_params" begin
        param_names = [:alpha]
        theta = [0.30]
        new_spec = MacroEconometricModels._update_ha_params(spec, param_names, theta)
        @test new_spec isa ModelSpec
        @test new_spec.param_values[:alpha] ≈ 0.30
        @test _hh(new_spec).het_params[:alpha] ≈ 0.36  # het_params has its own copy
        @test _hh(new_spec).individual.beta ≈ 0.99  # unchanged

        # Update beta
        param_names2 = [:beta]
        theta2 = [0.98]
        new_spec2 = MacroEconometricModels._update_ha_params(spec, param_names2, theta2)
        @test _hh(new_spec2).individual.beta ≈ 0.98
    end

    # Full KS SS + SSJ + MH is the HA-DSGE ceiling. Windows smoke (FAST) and the
    # Ubuntu 1.10 numerical cell keep the cheap helper above; macos/ubuntu LTS
    # still run the rest.
    # Do not `return` — inside the wrapping HA-DSGE Types testset that aborts siblings.
    if !(FAST || NUMERICAL)
    # Compute steady state for generating fake data
    ss = compute_steady_state(spec; K_init=10.0, r_bounds=(-0.02, 0.04), max_iter=50, tol=1e-3)
    K_ss = ss.aggregates[:K]
    T_data = 16
    rng = Random.MersenneTwister(42)
    data_K = K_ss .+ 0.1 .* randn(rng, T_data)  # K with noise

    # [T206] hoist one shared :ssj solve to avoid re-solving in the two helper testsets below.
    sol_shared = solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=10)

    @testset "_build_ha_likelihood_fn" begin
        # Solve model first to have a valid solution for observation equation
        @test sol_shared isa HADSGESolution{Float64}

        param_names = [:alpha]
        ll_fn = MacroEconometricModels._build_ha_likelihood_fn(
            spec, param_names, reshape(data_K, 1, :),
            [:K], nothing, :ssj, (T_horizon=30, n_reduced=10)
        )

        ll_val = ll_fn([0.36])
        # #234 honesty consequence: with the silent G1 eigenvalue rescale removed, the KS-SSJ
        # Ho-Kalman realization is truthfully explosive (reduced ρ≈1.003 ≥ 1) at the small FAST
        # size used here (n_reduced=10), so the Kalman likelihood is honestly -Inf rather than a
        # finite value — this assertion encoded the pre-#234 silently-stabilized behavior.
        # Follow-up: stabilize the reduced realization at small n_reduced (the runtime warning
        # flags a probable incomplete GE block / mis-scaled Jacobian) — NOT a silent rescale.
        @test_broken isfinite(ll_val)
        @test ll_val < 0  # -Inf < 0 still holds; it is the finiteness that broke (see above)

        # Likelihood should handle bad parameter values gracefully
        ll_bad = ll_fn([0.001])  # extreme parameter
        @test ll_bad == -Inf || ll_bad < ll_val + 100  # either fails or worse
    end

    @testset "_build_ha_observation_equation" begin
        sol = sol_shared

        Z, d, H = MacroEconometricModels._build_ha_observation_equation(
            sol, [:K], nothing
        )
        n_states = size(sol.linear_solution.G1, 1)
        @test size(Z) == (1, n_states)
        @test length(d) == 1
        @test size(H) == (1, 1)
        @test d[1] ≈ K_ss atol=1.0  # steady state K
        @test H[1, 1] == 0  # zero default measurement error (T042)
        @test all(iszero, H)

        # #228/T129: Z is the C_obs row for the matched aggregate (:K), NOT a silent
        # unit-loading at an arbitrary reduced-state index.
        @test Z ≈ reshape(sol.C_obs[1, :], 1, :)

        # Custom measurement error
        Z2, d2, H2 = MacroEconometricModels._build_ha_observation_equation(
            sol, [:K], [0.5]
        )
        @test H2[1, 1] ≈ 0.25  # 0.5^2

        # #228/T129: an observable absent from the reduced system's aggregate outputs
        # raises an informative error naming it (the SSJ realization exposes only :K),
        # instead of the old silent arbitrary-index fallback.
        err = try
            MacroEconometricModels._build_ha_observation_equation(sol, [:K, :Y], nothing)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("Y", err.msg)
        @test_throws ErrorException MacroEconometricModels._build_ha_observation_equation(
            sol, [:nonexistent], nothing)
    end

    @testset "estimate_dsge_bayes dispatch" begin
        # Very small run to verify the method dispatches correctly
        priors = Dict(:alpha => Distributions.Normal(0.36, 0.05))
        rng_est = Random.MersenneTwister(123)

        result = estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), [0.36];
            priors=priors,
            observables=[:K],
            n_draws=6,
            burnin=2,
            ha_method=:ssj,
            ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001,
            adapt_interval=50,  # no adaptation in 6 draws
            rng=rng_est
        )

        @test result isa BayesianDSGE{Float64}
        @test result.solved_at === :posterior_mean  # normal path (#149/T050)
        @test size(result.theta_draws, 2) == 1  # one parameter
        @test size(result.theta_draws, 1) == 4  # n_draws - burnin = 6 - 2
        @test length(result.log_posterior) == 4
        @test result.method === :rwmh
        @test result.acceptance_rate >= 0.0
        @test result.acceptance_rate <= 1.0
        @test length(result.param_names) == 1
        @test result.param_names[1] === :alpha

        # Posterior summary should work
        ps = posterior_summary(result)
        @test haskey(ps, :alpha)
        @test isfinite(ps[:alpha][:mean])

        # #136: theta0 as a Dict (order-independent) is accepted through the HA method;
        # a wrong-length positional vector errors informatively before any solve.
        result_dict = estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), Dict(:alpha => 0.36);
            priors=priors, observables=[:K], n_draws=6, burnin=2,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001, adapt_interval=50, rng=Random.MersenneTwister(7))
        @test result_dict isa BayesianDSGE{Float64}
        @test_throws ArgumentError estimate_dsge_bayes(
            spec, reshape(data_K, T_data, 1), [0.36, 0.9];   # length 2, but 1 prior
            priors=priors, observables=[:K], n_draws=10,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10))

        # #142: n×T data (1×T_data) resolves identically to T×n (same internal matrix →
        # identical draws under the same rng); a shape matching neither dim to n_obs errors.
        result_nt = estimate_dsge_bayes(
            spec, reshape(data_K, 1, T_data), Dict(:alpha => 0.36);
            priors=priors, observables=[:K], n_draws=6, burnin=2,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10),
            proposal_scale=0.001, adapt_interval=50, rng=Random.MersenneTwister(7))
        @test result_nt.theta_draws ≈ result_dict.theta_draws
        @test_throws ArgumentError estimate_dsge_bayes(
            spec, randn(3, T_data), [0.36];                  # neither dim == n_obs (1)
            priors=priors, observables=[:K], n_draws=10,
            ha_method=:ssj, ha_kwargs=(T_horizon=30, n_reduced=10))
    end

    @testset "T049: default T_horizon >= 300 (truncation)" begin
        # (A) Pin the signature default cheaply (no horizon-300 solve — those cost minutes):
        #     the signature's ha_kwargs default uses this const.
        @test MacroEconometricModels._HA_DEFAULT_T_HORIZON >= 300

        # (B) Truncation is non-negligible: the likelihood depends on the horizon (compared
        #     at cheap horizons; KS ρ_z=0.95 ⇒ 0.95^30≈0.21 vs 0.95^60≈0.046 tail alive).
        ll30 = MacroEconometricModels._build_ha_likelihood_fn(
            spec, [:alpha], reshape(data_K, 1, :), [:K], nothing, :ssj,
            (T_horizon=30, n_reduced=15))([0.36])
        ll60 = MacroEconometricModels._build_ha_likelihood_fn(
            spec, [:alpha], reshape(data_K, 1, :), [:K], nothing, :ssj,
            (T_horizon=60, n_reduced=15))([0.36])
        # #234 honesty consequence (see the _build_ha_likelihood_fn testset): at these small
        # FAST sizes (n_reduced=15) the truthful KS-SSJ realization is explosive, so both
        # likelihoods are -Inf. Follow-up: stabilize the reduced realization; broken pending that.
        @test_broken isfinite(ll30) && isfinite(ll60)
        @test_broken abs(ll30 - ll60) > 1e-6
    end

    @testset "posterior-mean solution built at the mean, marked (#149/T050)" begin
        # KS always yields a determinate, finite reduced solution for ANY θ (even NaN/Inf),
        # so the mean-solve-fails → highest-posterior-draw branch — which mirrors the
        # unit-tested aggregate [T044]/#143 path — is not reachable with this fast example.
        # We verify the reachable guarantees of the fix: (a) the container is built at the
        # POSTERIOR MEAN θ and marked, NOT silently at the original pre-estimation spec (the
        # removed E-25 bug); (b) when no candidate yields a supported HADSGESolution the
        # helper errors LOUDLY rather than silently substituting.
        post_draws = reshape([0.4, 0.5, 0.6], 3, 1)   # mean = 0.5 (≠ spec's alpha=0.36)
        post_lp    = [-3.0, -1.0, -2.0]
        linear_sol, ss_result, solved_at, theta_used =
            MacroEconometricModels._build_ha_result_solution(
                spec, [:alpha], post_draws, post_lp, [:K], nothing,
                :ssj, (T_horizon=30, n_reduced=10))
        @test solved_at === :posterior_mean
        @test theta_used ≈ [0.5]                    # built at the mean, not spec's 0.36
        @test all(isfinite, linear_sol.G1)

        # No candidate solves (unsupported method ⇒ no HADSGESolution) ⇒ loud error, never a
        # silent original-spec substitution.
        @test_throws ErrorException MacroEconometricModels._build_ha_result_solution(
            spec, [:alpha], reshape([0.36], 1, 1), [0.0], [:K], nothing,
            :badmethod, NamedTuple())
    end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 23: Clearing closure (Aiyagari regression — refactor must not change behavior)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Clearing closure (Aiyagari regression)" begin
    spec = load_ha_example(:krusell_smith)
    @test _hh(spec).model == :aiyagari                       # new field defaults correctly

    if !(FAST || NUMERICAL)
    ss = compute_steady_state(spec; r_bounds=(-0.02, 0.04), max_iter=100, tol=1e-3)
    @test ss.aggregates[:K] > 0
    @test isfinite(ss.prices[:r])
    @test haskey(ss.prices, :w)                         # Cobb-Douglas wage still produced
    @test abs(ss.excess_demand) < 5e-3                  # market essentially clears
    @test -0.01 < ss.prices[:r] < 1 / _hh(spec).individual.beta - 1  # r* below time-pref rate
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 24: Huggett (1993) — pure-exchange risk-free bond, zero net supply
# ─────────────────────────────────────────────────────────────────────────────

@testset "Huggett (1993) steady state" begin
    # Six model periods per year (Huggett 1993): annualize the per-period rate.
    annualize(rp) = (1 + rp)^6 - 1
    # Table 1 (σ = 1.5): credit limit => equilibrium annual risk-free rate.
    targets = (FAST || NUMERICAL) ? [(-2.0, -0.071)] :
              [(-2.0, -0.071), (-4.0, 0.023), (-6.0, 0.034), (-8.0, 0.040)]

    r_annuals = Float64[]
    for (cl, r_target) in targets
        a_max = cl <= -6 ? 18.0 : 8.0
        if cl == -2.0                       # reuse the shared cl=−2 SS (a_max=8.0, n_a=200)
            spec = _HUG_SPEC_M2
            ss = _HUG_SS_M2
        else
            spec = MacroEconometricModels._huggett_example(; credit_limit=cl, a_max=a_max, n_a=200)
            ss = compute_steady_state(spec; max_iter=200, tol=5e-4)
        end
        @test _hh(spec).model == :huggett
        @test ss.converged
        @test abs(ss.excess_demand) < 3e-3                 # bond market clears (∫a' ≈ 0)
        r_ann = annualize(ss.prices[:r])
        push!(r_annuals, r_ann)
        # Reproduces Huggett (1993) Table 1 within method/grid tolerance (~1.5pp)
        @test isapprox(r_ann, r_target; atol=0.015)
        # Precautionary saving keeps r* below the time-preference rate (1/β − 1)
        @test r_ann < annualize((1 - _hh(spec).individual.beta) / _hh(spec).individual.beta)
    end

    # Huggett's comparative static: r* rises as the credit limit loosens.
    @test issorted(r_annuals)

    # load_ha_example(:huggett) is the default (credit limit −2) economy.
    spec0 = load_ha_example(:huggett)
    @test _hh(spec0).model == :huggett
    @test _hh(spec0).individual.borrowing_constraint[1] == -2.0
    @test _hh(spec0).income.states == [1.0, 0.1]
end

@testset "Huggett SSJ" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2      # reuse shared cl=−2 SS (T209/#308)
    Th = 20
    sol = solve(spec; method=:ssj, ss=ss, T_horizon=Th, n_reduced=10)
    @test sol isa HADSGESolution
    @test sol.method === :ssj
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6  # stable
    @test haskey(sol.jacobians, :H_U)                                  # clearing Jacobian
    @test haskey(sol.jacobians, :H_Z)                                  # shock Jacobian
    # A positive aggregate endowment shock lowers the clearing risk-free rate on impact.
    H_U = sol.jacobians[:H_U]; H_Z = sol.jacobians[:H_Z]
    dr = -(H_U \ (H_Z * [0.9^(t - 1) for t in 1:Th]))
    @test dr[1] < 0
end

@testset "Huggett Reiter" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2      # reuse shared cl=−2 SS (T209/#308)
    sol = solve(spec; method=:reiter, ss=ss, n_reduced=15)
    @test sol isa HADSGESolution
    @test sol.method === :reiter
    @test maximum(abs.(eigvals(sol.linear_solution.G1))) <= 1 + 1e-6   # stable
    # #234: eu is now derived from the true spectral radius, so a genuinely stable
    # reduced system reports determinate (not a hardcoded [1,1] on a rescaled G1).
    @test MacroEconometricModels.is_determined(sol.linear_solution)
    @test MacroEconometricModels.is_stable(sol.linear_solution)
    @test sol.explained_variance > 0.5
    @test size(sol.linear_solution.G1, 1) == sol.n_reduced + 1         # state [d̃; w]
end

@testset "Huggett Krusell-Smith" begin
    spec = MacroEconometricModels._huggett_example(; credit_limit=-2.0, a_max=8.0,
                                                    n_a=60)
    ss = compute_steady_state(spec; max_iter=50, tol=1e-3)
    sol = solve(spec; method=:krusell_smith, ss=ss,
                T_sim=120, T_burn=30, max_outer=2)
    @test sol isa KrusellSmithSolution
    @test haskey(sol.plm_coefficients, :r)        # PLM forecasts the clearing rate, not K
    @test sol.r_squared[:r] > 0.7                 # rate is near-linear in the endowment shock
    b = sol.plm_coefficients[:r]
    @test abs(b[1] - ss.prices[:r]) < 0.01        # PLM intercept ≈ steady-state rate
    @test b[2] < 0                                # positive endowment shock lowers r
end

@testset "Den Haan (2010) accuracy" begin
    # --- Aiyagari capital model (z-augmented PLM makes the test meaningful) ---
    if !(FAST || NUMERICAL)
    ks_spec = load_ha_example(:krusell_smith)
    ss_a = compute_steady_state(ks_spec; r_bounds=(-0.02, 0.04), max_iter=80, tol=1e-3)
    ks = solve(ks_spec; method=:krusell_smith, ss=ss_a, T_sim=200, T_burn=100, max_outer=3)
    @test length(ks.plm_coefficients[:K]) == 3          # z-augmented PLM

    dh = den_haan_test(ks; T_sim=150, T_burn=100)
    @test dh isa DenHaanAccuracy
    @test dh.aggregate === :K
    @test isfinite(dh.dh_max) && dh.dh_max >= dh.dh_mean >= 0
    @test dh.sigma_ref > 0 && dh.sigma_plm > 0
    @test length(dh.ref_path) == 150 && length(dh.plm_path) == 150
    @test dh.sigma_plm > 0.2 * dh.sigma_ref             # PLM reproduces the fluctuations
    @test dh.dh_max < 1.0                               # accurate: well under 1% (Den Haan)
    report(dh)                                          # display smoke test
    end

    # --- Huggett: rate accuracy test is intentionally unsupported (errors clearly) ---
    # Reuse the shared cl=−2 SS — no extra solve (the guard fires on _hh(spec).model).
    ks_h = KrusellSmithSolution{Float64}(
        _HUG_SS_M2, Dict(:r => [_HUG_SS_M2.prices[:r], 0.0]), Dict(:r => 1.0),
        _HUG_SPEC_M2, false, 0)
    @test_throws ErrorException den_haan_test(ks_h)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 26 (#352/T253): sequence-space block composition (DAG) + 2nd-order SSJ
# ─────────────────────────────────────────────────────────────────────────────

# A pure-SimpleBlock DAG is exactly solvable by hand, so it pins down every piece
# of the composition machinery (shift matrices, topological sort, forward
# accumulation, the GE solve, and the second-order contraction) against closed
# forms rather than against snapshots.
@testset "SSJ blocks — SimpleBlock algebra" begin
    Th = 12

    # y_t = 2·u_t + 3·z_t ;  q_t = y_t − 0.5·u_{t-1} + 0.25·y_{t+1}
    blk1 = SimpleBlock(x -> [2 * x[1] + 3 * x[2]];
                       inputs=[:u, :z], outputs=[:y],
                       ss_inputs=Dict(:u => 0.0, :z => 0.0), name=:one)
    blk2 = SimpleBlock(x -> [x[3] - 0.5 * x[1] + 0.25 * x[2]];
                       inputs=[:u, :y], outputs=[:q],
                       lags=Dict(:u => [1], :y => [-1, 0]),
                       ss_inputs=Dict(:u => 0.0, :y => 0.0), name=:two)

    # Argument order: inputs in declaration order, lags ascending within an input.
    @test ssj_arg_order(blk1) == [(:u, 0), (:z, 0)]
    @test ssj_arg_order(blk2) == [(:u, 1), (:y, -1), (:y, 0)]
    @test blk1.ss_outputs[:y] == 0.0

    J1 = block_jacobian(blk1, Th)
    @test J1[(:y, :u)] ≈ 2 * Matrix(I, Th, Th)
    @test J1[(:y, :z)] ≈ 3 * Matrix(I, Th, Th)

    # Shift matrices: lag l ⇒ ones on M[t, t-l]; out-of-window entries dropped.
    S_lag = zeros(Th, Th); for t in 2:Th; S_lag[t, t-1] = 1.0; end
    S_lead = zeros(Th, Th); for t in 1:(Th-1); S_lead[t, t+1] = 1.0; end
    J2 = block_jacobian(blk2, Th)
    @test J2[(:q, :u)] ≈ -0.5 .* S_lag
    @test J2[(:q, :y)] ≈ Matrix(I, Th, Th) .+ 0.25 .* S_lead

    model = combine_blocks(blk1, blk2; name=:toy)
    @test [b.name for b in model.blocks] == [:one, :two]      # topological order
    @test model.exogenous == [:u, :z]
    @test model.endogenous == [:y, :q]
    @test model.ss_values[:q] == 0.0

    # Supplying the blocks out of order must not change the sorted DAG.
    @test [b.name for b in combine_blocks(blk2, blk1).blocks] == [:one, :two]

    gej = ssj_jacobian(model; unknowns=[:u], targets=[:q], shocks=[:z], T_horizon=Th)
    # Chain rule by hand: dq/du = ∂q/∂u + (∂q/∂y)(∂y/∂u); dq/dz = (∂q/∂y)(∂y/∂z).
    H_U_hand = J2[(:q, :u)] .+ J2[(:q, :y)] * J1[(:y, :u)]
    H_Z_hand = J2[(:q, :y)] * J1[(:y, :z)]
    @test gej.H_U ≈ H_U_hand
    @test gej.H_Z ≈ H_Z_hand
    @test size(gej.H_U) == (Th, Th) && size(gej.H_Z) == (Th, Th)

    dz = [0.5^(t - 1) for t in 1:Th]
    r1 = ssj_irf(gej, Dict(:z => dz))
    du_hand = -(H_U_hand \ (H_Z_hand * dz))
    @test r1.paths[:u] ≈ du_hand
    @test r1.paths[:y] ≈ J1[(:y, :u)] * du_hand .+ J1[(:y, :z)] * dz
    @test r1.paths[:z] ≈ dz
    @test r1.order == 1 && isempty(r1.correction)
    # A linear DAG clears exactly at first order.
    @test r1.target_residual[:q] < 1e-12
    @test maximum(abs, r1.paths[:q]) < 1e-12

    # Second order on a LINEAR DAG must vanish identically.
    r2 = ssj_irf(gej, Dict(:z => dz); order=2)
    @test r2.order == 2
    @test maximum(abs, r2.correction[:u]) < 1e-9
    @test r2.paths[:u] ≈ du_hand atol=1e-9

    # Convenience single-shock method.
    @test ssj_irf(gej, :z, dz).paths[:u] ≈ du_hand
end

@testset "SSJ blocks — second-order closed form" begin
    Th = 8
    # y_t = u_t + u_t² ;  q_t = y_t − z_t.  Equilibrium: u + u² = z.
    b1 = SimpleBlock(x -> [x[1] + x[1]^2];
                     inputs=[:u], outputs=[:y],
                     ss_inputs=Dict(:u => 0.0), name=:quad)
    b2 = SimpleBlock(x -> [x[1] - x[2]];
                     inputs=[:y, :z], outputs=[:q],
                     ss_inputs=Dict(:y => 0.0, :z => 0.0), name=:clear)
    gej = ssj_jacobian(combine_blocks(b1, b2; name=:quadtoy);
                       unknowns=[:u], targets=[:q], shocks=[:z], T_horizon=Th)
    @test gej.H_U ≈ Matrix(I, Th, Th)           # ∂(u+u²)/∂u = 1 at u=0

    dz = fill(0.05, Th)
    r1 = ssj_irf(gej, Dict(:z => dz))
    @test r1.paths[:u] ≈ dz                     # first order: du = dz

    r2 = ssj_irf(gej, Dict(:z => dz); order=2)
    # Second order: u + u² = z with u = z + u₂ ⇒ u₂ = −z².
    @test r2.correction[:u] ≈ -dz .^ 2 rtol=1e-8
    @test r2.paths[:u] ≈ dz .- dz .^ 2 rtol=1e-8
    # D²y[v,v] = 2·(du¹)² ⇒ the second-order y path is J·u₂ + ½·2z² = −z² + z² = 0.
    @test maximum(abs, r2.correction[:y]) < 1e-8
    # Exact root of u + u² = z:  u* = (√(1+4z) − 1)/2.  Second order beats first.
    u_exact = (sqrt.(1 .+ 4 .* dz) .- 1) ./ 2
    @test maximum(abs, r2.paths[:u] .- u_exact) < maximum(abs, r1.paths[:u] .- u_exact)
    @test r2.target_residual[:q] < r1.target_residual[:q]
end

@testset "SSJ blocks — HetBlock and DAG composition" begin
    spec = load_ha_example(:krusell_smith)
    # Converged (default) tolerance, not the loose tol=1e-4 used elsewhere: the
    # firm SimpleBlock below is built at K_ss = ∫a dμ while `ss.prices` are
    # evaluated at the firm's K_demand, so the two price sets differ by exactly
    # |dp/dK|·|excess_demand|. Testing the block's Cobb-Douglas algebra against
    # the price function therefore needs a steady state where those coincide.
    ss = compute_steady_state(spec; r_bounds=(-0.01, 0.04), max_iter=80)
    Th = 20

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A, :C], name=:household)
    @test hh isa HetBlock{Float64}
    @test hh.ss_inputs[:r] == ss.prices[:r]
    @test hh.ss_outputs[:A] ≈ dot(vec(ss.policies[:savings]),
                                  MacroEconometricModels._normalized_distribution(ss))

    # The block Jacobian IS the fake-news Jacobian — no reimplementation drift.
    Jb = block_jacobian(hh, Th)
    @test Set(keys(Jb)) == Set([(:A, :r), (:A, :w), (:C, :r), (:C, :w)])
    @test Jb[(:A, :r)] == MacroEconometricModels._ssj_jacobian(
        ss, _hh(spec).individual, _hh(spec).grid, _hh(spec).income, :r, :A; T_horizon=Th, dx=hh.dx)

    # Nonlinear path evaluation reproduces the steady state on a flat input path.
    flat = Dict(:r => fill(ss.prices[:r], Th), :w => fill(ss.prices[:w], Th))
    base = MacroEconometricModels._block_evaluate(hh, flat, Th)
    @test maximum(abs, base[:A] .- hh.ss_outputs[:A]) < 1e-6
    @test maximum(abs, base[:C] .- hh.ss_outputs[:C]) < 1e-6

    # INDEPENDENT ORACLE: the fake-news Jacobian must equal a central finite
    # difference of the *nonlinear* transition path (backward EGM + forward Young
    # histogram) — two implementations sharing no code beyond the EGM step. The
    # anticipation columns (t < s) are the ones the pre-#226 brute force got wrong.
    hh_fine = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], dx=1e-5)
    J_fine = block_jacobian(hh_fine, Th)[(:A, :r)]
    fd_step = 1e-6
    for s in (1, 4, 9)
        pp = deepcopy(flat); pp[:r][s] += fd_step
        pm = deepcopy(flat); pm[:r][s] -= fd_step
        col = (MacroEconometricModels._block_evaluate(hh_fine, pp, Th)[:A] .-
               MacroEconometricModels._block_evaluate(hh_fine, pm, Th)[:A]) ./ (2fd_step)
        @test maximum(abs, col .- J_fine[:, s]) < 1e-5 * maximum(abs, J_fine[:, s])
    end
    @test any(abs(J_fine[t, s]) > 1e-8 for t in 1:Th for s in (t+1):Th)   # anticipation

    # ── Three-block DAG: firm (lagged capital) → household → asset market ────
    alpha = spec.param_values[:alpha]
    delta = spec.param_values[:delta]
    K_ss = ss.aggregates[:K]
    firm = SimpleBlock(
        x -> [alpha * x[2] * x[1]^(alpha - 1) - delta,
              (1 - alpha) * x[2] * x[1]^alpha,
              x[2] * x[1]^alpha];
        inputs=[:K, :Z], outputs=[:r, :w, :Y],
        lags=Dict(:K => [1]),
        ss_inputs=Dict(:K => K_ss, :Z => 1.0), name=:firm)
    @test firm.ss_outputs[:r] ≈ ss.prices[:r] atol=1e-6
    @test firm.ss_outputs[:w] ≈ ss.prices[:w] atol=1e-6

    hh1 = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household)
    mkt = SimpleBlock(x -> [x[1] - x[2]];
                      inputs=[:A, :K], outputs=[:asset_mkt],
                      ss_inputs=Dict(:A => hh1.ss_outputs[:A], :K => K_ss),
                      name=:asset_market)
    dag = combine_blocks(firm, hh1, mkt; name=:ks_dag)
    @test [b.name for b in dag.blocks] == [:firm, :household, :asset_market]
    @test dag.exogenous == [:K, :Z]
    @test dag.endogenous == [:r, :w, :Y, :A, :asset_mkt]

    # HISTORICAL NOTE: the Krusell-Smith example used to truncate its asset grid
    # (~5.6% of mass pinned at a_max = 200, so ∫a'dμ exceeded ∫a dμ by ~1.7%) and
    # the asset market did NOT clear at the linearization point — this assertion
    # read `dag.ss_values[:asset_mkt] > 1e-3` and the GE assembler's target_tol
    # guard fired. The example now clears; the household block's ∫a'dμ and the
    # steady state's ∫a dμ agree to floating-point. The guard itself is still
    # covered independently on the toy DAG below.
    @test abs(dag.ss_values[:asset_mkt]) < 1e-9

    gej = ssj_jacobian(dag; unknowns=[:K], targets=[:asset_mkt], shocks=[:Z],
                       T_horizon=Th, target_tol=Inf)
    # Forward accumulation vs the chain rule computed by hand from block Jacobians.
    Jf = block_jacobian(firm, Th)
    Jh = block_jacobian(hh1, Th)
    dA_dK = Jh[(:A, :r)] * Jf[(:r, :K)] .+ Jh[(:A, :w)] * Jf[(:w, :K)]
    @test gej.curlyJ[:A][:K] ≈ dA_dK
    @test gej.H_U ≈ dA_dK .- Matrix(I, Th, Th)
    @test gej.curlyJ[:Y][:Z] ≈ Jf[(:Y, :Z)]

    dZ = Dict(:Z => [0.01 * 0.9^(t - 1) for t in 1:Th])
    r1 = ssj_irf(gej, dZ; residual=false)
    # The linearized clearing condition holds exactly whatever the steady-state wedge.
    @test maximum(abs, gej.H_U * r1.paths[:K] .+ gej.H_Z * dZ[:Z]) < 1e-8
    @test r1.paths[:K][1] > 0                    # positive TFP shock raises capital
    @test r1.paths[:r][1] ≈ alpha * 0.01 * K_ss^(alpha - 1) atol=1e-10  # K lagged ⇒ r_1 ← Z_1
    report(dag)                                   # display smoke tests
    report(gej)
    @test occursin("SSJModel", sprint(show, dag))
    @test occursin("SSJGEJacobian", sprint(show, gej))
    @test occursin("HetBlock", sprint(show, hh1))
    @test occursin("SimpleBlock", sprint(show, firm))
end

@testset "SSJ blocks — Huggett GE and second order" begin
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2       # reuse the shared cl=−2 SS
    Th = 40

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household)
    bond = SimpleBlock(x -> [x[1]];
                       inputs=[:A], outputs=[:bond_mkt],
                       ss_inputs=Dict(:A => hh.ss_outputs[:A]), name=:bond_market)
    dag = combine_blocks(hh, bond; name=:huggett_dag)
    # Zero net supply: the Huggett steady state genuinely clears, so no warning.
    @test abs(dag.ss_values[:bond_mkt]) < 1e-3
    gej = ssj_jacobian(dag; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                       T_horizon=Th, target_tol=1e-2)

    # The two-block DAG reproduces the hard-wired GE close of `_ssj_solve` exactly.
    J_ref_U = MacroEconometricModels._ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid,
                                                   _hh(spec).income, :r, :A; T_horizon=Th)
    J_ref_Z = MacroEconometricModels._ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid,
                                                   _hh(spec).income, :w, :A; T_horizon=Th)
    @test gej.H_U == J_ref_U
    @test gej.H_Z == J_ref_Z
    dw = [0.9^(t - 1) for t in 1:Th]
    @test ssj_irf(gej, Dict(:w => dw); residual=false).paths[:r] ≈ -(J_ref_U \ (J_ref_Z * dw))

    # Routing `solve(:ssj)` through the DAG must not start emitting the target guard:
    # for the zero-net-supply close the target level IS ss.excess_demand, already
    # reported by report(ss), so warning again on every solve is pure noise.
    logs, _ = Test.collect_test_logs() do
        solve(spec; method=:ssj, ss=ss, T_horizon=30, n_reduced=12)
    end
    @test !any(occursin("does not vanish in steady state", string(r.message)) for r in logs)

    # ── Second order ────────────────────────────────────────────────────────
    sigma = 0.02
    dZ = Dict(:w => [sigma * 0.9^(t - 1) for t in 1:Th])
    o1 = ssj_irf(gej, dZ)
    o2 = ssj_irf(gej, dZ; order=2)
    @test o2.order == 2
    @test haskey(o2.correction, :r) && haskey(o2.correction, :A)
    # Precautionary saving makes the block genuinely nonlinear: nonzero correction.
    @test maximum(abs, o2.correction[:r]) > 1e-10
    # By construction the target is zero to second order: 𝒥·dU² + ½D²H = 0. Scale
    # against one of the two cancelling terms — NOT against the first-order :A path,
    # which the GE solve itself drives to ~1e-17 (bond_mkt IS A here).
    cancel_scale = maximum(abs, gej.H_U * o2.correction[:r])
    @test cancel_scale > 1e-12
    @test maximum(abs, o2.correction[:bond_mkt]) < 1e-8 * cancel_scale
    # The honest accuracy measure: the nonlinear clearing residual must improve.
    @test o2.target_residual[:bond_mkt] < o1.target_residual[:bond_mkt]

    # dU² is O(σ²) while dU¹ is O(σ), so halving the shock halves the relative
    # correction — this is what "collapses onto the first order" means.
    ratios = Float64[]
    for s in (0.02, 0.01, 0.005)
        rr = ssj_irf(gej, Dict(:w => [s * 0.9^(t - 1) for t in 1:Th]);
                     order=2, residual=false)
        push!(ratios, maximum(abs, rr.correction[:r]) / maximum(abs, rr.first_order[:r]))
    end
    @test issorted(ratios; rev=true)
    @test 1.6 < ratios[1] / ratios[2] < 2.4
    @test 1.6 < ratios[2] / ratios[3] < 2.4

    report(o2)                                    # display smoke test
    @test occursin("SSJImpulseResponse", sprint(show, o2))
end

@testset "SSJ blocks — validation and errors" begin
    ok = SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                     ss_inputs=Dict(:a => 1.0), name=:ok)

    # Construction-time validation
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=Symbol[], outputs=[:b],
                                           ss_inputs=Dict{Symbol,Float64}())
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=Symbol[],
                                           ss_inputs=Dict(:a => 1.0))
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a, :a], outputs=[:b],
                                           ss_inputs=Dict(:a => 1.0))
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                                           ss_inputs=Dict{Symbol,Float64}())    # missing SS
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                                           ss_inputs=Dict(:a => 1.0),
                                           lags=Dict(:q => [1]))                # unknown lag key
    @test_throws ArgumentError SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b, :c],
                                           ss_inputs=Dict(:a => 1.0))           # arity mismatch

    # DAG assembly
    @test_throws ArgumentError combine_blocks()
    dup = SimpleBlock(x -> [2 * x[1]]; inputs=[:a], outputs=[:b],
                      ss_inputs=Dict(:a => 1.0), name=:dup)
    @test_throws ArgumentError combine_blocks(ok, dup)                # duplicate output
    self = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:b],
                       ss_inputs=Dict(:b => 1.0), name=:self)
    @test_throws ArgumentError combine_blocks(self)                   # self loop
    back = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:a],
                       ss_inputs=Dict(:b => 1.0), name=:back)
    @test_throws ArgumentError combine_blocks(ok, back)               # cycle

    # Inconsistent steady state between producer and consumer is warned about.
    consumer = SimpleBlock(x -> [x[1]]; inputs=[:b], outputs=[:c],
                           ss_inputs=Dict(:b => 5.0), name=:consumer)
    @test_logs (:warn, r"inconsistent steady state") match_mode=:any begin
        combine_blocks(ok, consumer)
    end

    model = combine_blocks(ok; name=:tiny)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:b],
                                            shocks=[:a], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:b], targets=[:b],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:a],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=Symbol[], targets=Symbol[],
                                            shocks=Symbol[], T_horizon=4, target_tol=Inf)
    @test_throws ArgumentError ssj_jacobian(model; unknowns=[:a], targets=[:b],
                                            shocks=Symbol[], T_horizon=1, target_tol=Inf)
    # A non-vanishing target level is warned about (ok's steady-state :b is 1.0).
    @test_logs (:warn, r"does not vanish in steady state") match_mode=:any begin
        ssj_jacobian(model; unknowns=[:a], targets=[:b], shocks=Symbol[], T_horizon=4)
    end

    # A model with no shocks must still assemble and solve (typed empty H_Z path).
    gej = ssj_jacobian(model; unknowns=[:a], targets=[:b], shocks=Symbol[],
                       T_horizon=4, target_tol=Inf)
    @test size(gej.H_Z) == (4, 0)
    @test all(iszero, ssj_irf(gej, Dict{Symbol,Vector{Float64}}()).paths[:a])
    @test_throws ArgumentError ssj_irf(gej, Dict(:zz => zeros(4)))          # undeclared shock
    @test_throws ArgumentError ssj_irf(gej, Dict{Symbol,Vector{Float64}}(); order=3)
    @test_throws ArgumentError ssj_irf(gej, Dict{Symbol,Vector{Float64}}();
                                       order=2, fd_step=0.0)

    # A singular clearing Jacobian is reported, not silently inverted.
    dead = SimpleBlock(x -> [0.0 * x[1]]; inputs=[:a], outputs=[:b],
                       ss_inputs=Dict(:a => 1.0), name=:dead)
    @test_throws ErrorException ssj_jacobian(combine_blocks(dead);
                                             unknowns=[:a], targets=[:b],
                                             shocks=Symbol[], T_horizon=4,
                                             target_tol=Inf)

    # HetBlock validation
    spec = _HUG_SPEC_M2; ss = _HUG_SS_M2
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:not_a_price], outputs=[:A])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:r], outputs=[:nonsense])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=Symbol[], outputs=[:A])
    @test_throws ArgumentError HetBlock(spec, ss; inputs=[:r], outputs=Symbol[])
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 27 (#353/T254): DCEGM — discrete-continuous choice
# ─────────────────────────────────────────────────────────────────────────────

@testset "DCEGM upper envelope" begin
    UE = MacroEconometricModels._upper_envelope
    SEG = MacroEconometricModels._monotone_segments

    @test SEG([1.0, 2.0, 3.0]) == [1:3]
    @test SEG([1.0, 2.0, 3.0, 2.5, 3.5]) == [1:3, 4:5]
    @test SEG([1.0, 2.0, 1.5]) == [1:2]                  # trailing single point dropped
    @test isempty(SEG(Float64[]))

    # A monotone correspondence passes through untouched.
    M1 = [1.0, 2.0, 3.0]; c1 = [0.5, 1.0, 1.5]; v1 = [0.0, 1.0, 2.0]
    Me, ce, ve, nk = UE(M1, c1, v1)
    @test Me == M1 && ce == c1 && ve == v1 && nk == 0

    # Two branches that genuinely CROSS inside their overlap.
    #   A: v = M           on [1, 5]
    #   B: v = 2M − 3.5    on [2, 5]   ⇒  v_A = v_B ⟺ M = 3.5, strictly between knots
    Ma = [1.0, 2.0, 3.0, 4.0, 5.0]; ca = [0.5, 1.0, 1.5, 2.0, 2.5]; va = [1.0, 2.0, 3.0, 4.0, 5.0]
    Mb = [2.0, 3.0, 4.0, 5.0];      cb = [9.0, 9.5, 10.0, 10.5];    vb = [0.5, 2.5, 4.5, 6.5]
    Me, ce, ve, nk = UE(vcat(Ma, Mb), vcat(ca, cb), vcat(va, vb))
    @test nk == 1
    @test all(diff(Me) .> 0)                              # strictly increasing output
    k = findfirst(i -> Me[i+1] == nextfloat(Me[i]), 1:(length(Me)-1))
    @test k !== nothing
    @test Me[k] ≈ 3.5                                     # exact crossing, not a grid point
    @test ve[k] ≈ ve[k+1] ≈ 3.5                           # value is continuous at a kink
    @test ce[k] ≈ 1.75 && ce[k+1] ≈ 9.75                  # consumption jumps
    # Defining property: the envelope dominates every branch everywhere it is defined.
    for (m, v) in zip(Me, ve)
        for (Ms, vs) in ((Ma, va), (Mb, vb))
            Ms[1] <= m <= Ms[end] || continue
            @test v >= MacroEconometricModels._seg_interp(Ms, vs, m) - 1e-12
        end
    end

    # A crossing that lands exactly ON a knot is still a kink: the branches tie at
    # M = 3 and consumption jumps immediately above it. Rounding it away would lose
    # the switching threshold entirely.
    vb_knot = [1.0, 3.0, 5.0, 7.0]                        # B: v = 2M − 3 ⇒ tie at M = 3
    Me, ce, ve, nk = UE(vcat(Ma, Mb), vcat(ca, cb), vcat(va, vb_knot))
    @test nk == 1
    @test all(diff(Me) .> 0)
    k = findfirst(i -> Me[i+1] == nextfloat(Me[i]), 1:(length(Me)-1))
    @test k !== nothing && Me[k] == 3.0
    @test ve[k] ≈ ve[k+1] ≈ 3.0
    @test ce[k] ≈ 1.5 && ce[k+1] ≈ 9.5

    # A switch at a SUPPORT BOUNDARY is not a crossing: branch B starts already
    # dominating, so there is no interior kink to insert.
    Mc_ = [1.0, 2.0, 3.0, 2.5, 3.5, 4.5]
    cc_ = [0.5, 1.0, 1.5, 0.2, 0.3, 0.4]
    vc_ = [0.0, 1.0, 2.0, 3.0, 3.4, 3.8]
    Me, ce, ve, nk = UE(Mc_, cc_, vc_)
    @test nk == 0
    @test all(diff(Me) .> 0)
    @test ve[findfirst(≈(2.5), Me)] ≈ 3.0                 # the dominating branch is kept

    @test_throws ArgumentError UE([1.0, 2.0], [1.0], [1.0, 2.0])
end

@testset "DCEGM retirement model" begin
    prob = dcegm_retirement_model(; n_periods=6, beta=0.98, R=1.0, wage=20.0,
                                  disutility=1.0, a_max=60.0, n_a=80)
    @test prob isa DCEGMProblem{Float64}
    @test prob.options == [:retire, :work] && prob.absorbing == [true, false]
    sol = dcegm_solve(prob)
    @test sol isa DCEGMSolution{Float64}
    @test sol.converged && sol.n_periods == 6

    # ANALYTIC ORACLE: once retired (absorbing, no pension, R = 1, log utility) the
    # problem is deterministic cake-eating with the closed form c_t = M / Σ_{k≤T−t} β^k.
    for t in (6, 5, 3, 1), Mt in (5.0, 20.0, 45.0)
        annuity = sum(0.98^k for k in 0:(6 - t))
        @test dcegm_policy(sol, t, 1, 1, Mt)[1] ≈ Mt / annuity rtol=1e-12
    end

    # The discrete choice makes the WORKING branch non-concave: the envelope deletes
    # secondary segments and inserts switching thresholds. Retirement is absorbing,
    # so its own branch is concave and needs none.
    @test sum(sol.n_kinks[:, 2, :]) > 0
    @test sum(sol.n_kinks[:, 1, :]) == 0

    # At every inserted kink the two value branches coincide while consumption jumps —
    # the defining property of an upper-envelope crossing.
    for t in 1:6, d in 1:2
        Mv = sol.M[t, d, 1]; cv = sol.c[t, d, 1]; vv = sol.v[t, d, 1]
        @test all(diff(Mv) .> 0)
        for i in 1:(length(Mv) - 1)
            Mv[i+1] == nextfloat(Mv[i]) || continue
            @test vv[i] ≈ vv[i+1] rtol=1e-8
            @test abs(cv[i] - cv[i+1]) > 1e-6
        end
    end

    # ── INDEPENDENT ORACLE: dense-grid backward-induction VFI on the same model ──
    # No EGM, no envelope, no Euler equation — just brute-force maximization.
    # Julia 1 numerical CI / FAST skip this grid; LTS still runs it.
    if !(FAST || NUMERICAL)
    function _vfi_retirement(; T_end, beta, R, wage, delta, Mmax, nM, nC)
        Mg = collect(range(1e-4, Mmax; length=nM))
        V = fill(-Inf, T_end, nM, 2); C = zeros(T_end, nM, 2); D = zeros(Int, T_end, nM, 2)
        u(c, d) = c > 0 ? log(c) - (d == 2 ? delta : 0.0) : -Inf
        for i in 1:nM, dp in 1:2
            V[T_end, i, dp] = u(Mg[i], 1); C[T_end, i, dp] = Mg[i]; D[T_end, i, dp] = 1
        end
        for t in (T_end-1):-1:1, i in 1:nM, dp in 1:2
            best = -Inf; bc = 0.0; bd = 0
            for d in (dp == 1 ? (1:1) : (1:2))
                inc = d == 2 ? wage : 0.0
                for k in 0:(nC-1)
                    c = Mg[i] * (k + 1) / nC
                    Mn = R * (Mg[i] - c) + inc
                    Vn = if Mn <= Mg[1]; V[t+1, 1, d]
                         elseif Mn >= Mg[end]; V[t+1, end, d]
                         else
                             q = searchsortedfirst(Mg, Mn) - 1
                             w = (Mn - Mg[q]) / (Mg[q+1] - Mg[q])
                             (1 - w) * V[t+1, q, d] + w * V[t+1, q+1, d]
                         end
                    val = u(c, d) + beta * Vn
                    val > best && (best = val; bc = c; bd = d)
                end
            end
            V[t, i, dp] = best; C[t, i, dp] = bc; D[t, i, dp] = bd
        end
        return Mg, C, D
    end
    Mg, C_vfi, D_vfi = _vfi_retirement(; T_end=6, beta=0.98, R=1.0, wage=20.0,
                                      delta=1.0, Mmax=60.0, nM=80, nC=200)
    step = Mg[2] - Mg[1]

    for t in 2:4
        errs = Float64[]; mism = 0
        for (i, m) in enumerate(Mg)
            m < 0.5 && continue
            d = argmax(dcegm_choice_probabilities(sol, t, 2, 1, m))
            d != D_vfi[t, i, 2] && (mism += 1)
            push!(errs, abs(dcegm_policy(sol, t, d, 1, m)[1] - C_vfi[t, i, 2]) /
                        max(C_vfi[t, i, 2], 1e-8))
        end
        @test mism == 0                                        # discrete choice agrees
        @test sort(errs)[cld(length(errs), 2)] < 2e-3          # median within VFI resolution
        # Large disagreements occur only where the policy is genuinely discontinuous:
        # a grid-based VFI cannot resolve a jump, DCEGM locates it exactly.
        @test count(>(1e-2), errs) <= sum(sol.n_kinks[t, :, :]) + 1
    end

    # Retirement threshold vs the VFI switch point, within one oracle grid step.
    for t in (4, 5)
        thr = dcegm_threshold(sol, t, 2, 1; M_lo=0.5, M_hi=60.0)
        idx = findlast(i -> D_vfi[t, i, 2] == 2, 1:length(Mg))
        @test idx !== nothing
        @test isfinite(thr)
        @test abs(thr - Mg[idx]) <= 2 * step
    end
    @test all(D_vfi[2, i, 2] == 2 for i in 1:length(Mg))   # oracle agrees never-retire
    end
    # Early in life the worker never retires on this bracket — honestly reported as NaN.
    @test isnan(dcegm_threshold(sol, 2, 2, 1; M_lo=0.5, M_hi=60.0))
    # Retirement is absorbing, so there is no two-option choice left to threshold.
    @test_throws ArgumentError dcegm_threshold(sol, 3, 1, 1; M_lo=1.0, M_hi=10.0)
    @test_throws ArgumentError dcegm_threshold(sol, 3, 2, 1; M_lo=10.0, M_hi=1.0)

    report(sol)                                                # display smoke tests
    @test occursin("DCEGMSolution", sprint(show, sol))
    @test occursin("DCEGMProblem", sprint(show, prob))
end

@testset "DCEGM taste shocks" begin
    base = dcegm_solve(dcegm_retirement_model(; n_periods=5, beta=0.98, R=1.0,
                                              wage=20.0, disutility=1.0,
                                              a_max=60.0, n_a=80))
    Ms = collect(2.0:2.0:55.0)
    devs = Float64[]; spreads = Float64[]
    for lam in (1.0, 0.05, 0.01, 0.002)
        s = dcegm_solve(dcegm_retirement_model(; n_periods=5, beta=0.98, R=1.0,
                                               wage=20.0, disutility=1.0,
                                               a_max=60.0, n_a=80,
                                               taste_shock_scale=lam))
        push!(devs, maximum(abs(dcegm_policy(s, 2, 2, 1, m)[1] -
                                dcegm_policy(base, 2, 2, 1, m)[1]) for m in Ms))
        # Mean distance of the choice probabilities from the deterministic 0/1 rule.
        # The MAXIMUM is the wrong statistic: at the indifference point the
        # probabilities are 1/2 for every λ, so only the *measure* of the interior
        # region shrinks, not its peak.
        push!(spreads, sum(minimum(dcegm_choice_probabilities(s, 3, 2, 1, m))
                           for m in Ms) / length(Ms))
    end
    # The smoothed solution collapses onto the deterministic upper envelope as λ → 0.
    @test issorted(devs; rev=true)
    @test devs[1] > 1.0                                   # λ = 1 genuinely differs
    @test devs[end] < 0.01
    @test issorted(spreads; rev=true)
    @test spreads[end] < 1e-3

    s = dcegm_solve(dcegm_retirement_model(; n_periods=5, a_max=60.0, n_a=60,
                                           taste_shock_scale=0.5))
    p = dcegm_choice_probabilities(s, 3, 2, 1, 30.0)
    @test length(p) == 2 && sum(p) ≈ 1.0 && all(p .>= 0)
    # After retiring, work is infeasible: probability exactly zero, not merely small.
    pr = dcegm_choice_probabilities(s, 3, 1, 1, 30.0)
    @test pr == [1.0, 0.0]
end

@testset "DCEGM distribution and simulation" begin
    prob = dcegm_retirement_model(; n_periods=7, beta=0.98, R=1.02, wage=20.0,
                                  disutility=0.8, sigma=0.15, n_shocks=3,
                                  a_max=80.0, n_a=60)
    @test length(prob.income_process.states) == 3
    @test sum(prob.income_process.stationary_dist) ≈ 1.0
    @test dot(prob.income_process.stationary_dist, prob.income_process.states) ≈ 1.0 rtol=1e-6
    sol = dcegm_solve(prob)

    grid = collect(range(0.01, 80.0; length=120))
    dist = dcegm_simulate(sol, grid)
    @test dist isa DCEGMDistribution{Float64}
    @test dist.n_periods == 7
    # The Young lottery splits off-grid landings between neighbours, so mass is exact.
    for t in 1:7
        @test sum(@view dist.dist[t, :, :, :]) ≈ 1.0 atol=1e-12
        @test sum(@view dist.shares[t, :]) ≈ 1.0 atol=1e-12
    end
    @test all(dist.dist .>= 0)
    # Retirement is absorbing, so its share can only rise with age.
    @test issorted(dist.shares[:, 1])
    @test dist.shares[1, 2] ≈ 1.0                       # everyone starts working
    @test all(isfinite, dist.consumption) && all(dist.consumption .> 0)
    @test all(dist.assets .>= -1e-12)
    report(dist)                                        # display smoke test
    @test occursin("DCEGMDistribution", sprint(show, dist))

    # Custom initial condition: all mass at one node, everyone already retired.
    init = zeros(length(grid), 3)
    init[60, :] .= prob.income_process.stationary_dist
    d2 = dcegm_simulate(sol, grid; init=init, init_option=:retire, n_periods=4)
    @test d2.n_periods == 4
    @test all(d2.shares[:, 1] .≈ 1.0)                   # absorbing: nobody returns to work
    @test sum(@view d2.dist[1, :, :, :]) ≈ 1.0

    @test_throws ArgumentError dcegm_simulate(sol, [3.0, 1.0, 2.0])
    @test_throws ArgumentError dcegm_simulate(sol, [1.0])
    @test_throws ArgumentError dcegm_simulate(sol, grid; n_periods=0)
    @test_throws ArgumentError dcegm_simulate(sol, grid; n_periods=99)
    @test_throws ArgumentError dcegm_simulate(sol, grid; init_option=:nope)
    @test_throws ArgumentError dcegm_simulate(sol, grid; init=zeros(3, 3))
end

@testset "DCEGM infinite horizon and validation" begin
    # Stationary policy: a pension keeps the retired branch finite at the constraint.
    p = dcegm_retirement_model(; n_periods=0, beta=0.95, R=1.01, wage=5.0,
                               disutility=0.5, a_max=40.0, n_a=60, pension=1.0)
    s = dcegm_solve(p; max_iter=300, tol=1e-7)
    @test s.converged
    @test s.iterations > 1 && s.sup_diff < 1e-7
    @test s.n_periods == 1
    # A stationary solution can be simulated for any number of periods.
    d = dcegm_simulate(s, collect(range(0.01, 40.0; length=80)); n_periods=12)
    @test d.n_periods == 12
    @test all(sum(@view d.dist[t, :, :, :]) ≈ 1.0 for t in 1:12)

    # Non-convergence is reported, not silently accepted.
    s1 = dcegm_solve(p; max_iter=1, tol=1e-12)
    @test !s1.converged && s1.iterations == 1

    # ── Constructor validation ──────────────────────────────────────────────
    inc = rouwenhorst(0.5, 0.1, 2)
    base = (utility=(c, d) -> log(c), utility_prime=(c, d) -> 1 / c,
            utility_prime_inv=(m, d) -> 1 / m, income=(d, j) -> 1.0,
            income_process=inc)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=Symbol[], absorbing=Bool[], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a, :b], absorbing=[true], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a, :a], absorbing=[true, false], asset_grid=[0.0, 1.0])
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0])            # too few points
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[1.0, 0.0])       # unsorted
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.5, 1.0])       # ≠ credit limit
    @test_throws ArgumentError DCEGMProblem(; beta=1.5, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0])       # β outside (0,1)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0], n_periods=-1)
    @test_throws ArgumentError DCEGMProblem(; beta=0.95, R=1.0, base...,
        options=[:a], absorbing=[false], asset_grid=[0.0, 1.0], taste_shock_scale=-1)
    @test_throws ArgumentError dcegm_retirement_model(; n_shocks=0)
    @test_throws ArgumentError dcegm_retirement_model(; curvature=0.5)

    # A degenerate problem leaves too few usable grid points and says so.
    bad = DCEGMProblem(; beta=0.95, R=1.0,
        utility=(c, d) -> c > 0 ? log(c) : -Inf, utility_prime=(c, d) -> c > 0 ? 1 / c : Inf,
        utility_prime_inv=(m, d) -> m > 0 ? 1 / m : Inf, income=(d, j) -> 0.0,
        options=[:only], absorbing=[true], asset_grid=[0.0, 1.0],
        income_process=inc, n_periods=3)
    @test_throws ErrorException dcegm_solve(bad)
end


# ─────────────────────────────────────────────────────────────────────────────
# Winberry (2018) parametric distribution dynamics (#356/T257)
# ─────────────────────────────────────────────────────────────────────────────

# Small Aiyagari spec: the Winberry end-to-end tests solve TWO steady states and
# TWO linearizations, so the shipped 200x7 examples would dominate this file.
function _win_small_spec(; distribution::Symbol=:young, n_a::Int=80, n_e::Int=3)
    u, up, upi = MacroEconometricModels._crra_utility(1.0)
    income = MacroEconometricModels._unit_mean_lognormal_income(0.90, 0.30, n_e)
    grid = HAGrid(; assets=(0.0, 300.0, n_a), income_states=n_e, grid_type=:geometric)
    ip = IndividualProblem{Float64}(u, up, upi, 0.99,
                                    MacroEconometricModels._ks_budget,
                                    [0.0], nothing, 1)
    aggregation = Pair{Symbol,Function}[:K => MacroEconometricModels._agg_var1]
    het = Dict{Symbol,Float64}(:alpha => 0.36, :delta => 0.025, :Z => 1.0, :L => 1.0,
                               :rho_z => 0.95, :sigma_z => 0.007)
    hh = HouseholdSystem{Float64}(ip, income, grid, aggregation, het;
                                  distribution=distribution)
    return MacroEconometricModels._wrap_ha_spec(hh;
        params=[:alpha, :delta, :rho_z, :sigma_z],
        param_values=Dict{Symbol,Float64}(:alpha => 0.36, :delta => 0.025,
                                          :rho_z => 0.95, :sigma_z => 0.007))
end

@testset "Winberry parametric density (#356/T257)" begin

    @testset "Gauss-Legendre and composite quadrature are exact" begin
        # A k-point Gauss-Legendre rule integrates polynomials of degree 2k-1 exactly.
        for k in 2:6
            x, w = MacroEconometricModels._gauss_legendre(Float64, k)
            @test length(x) == k && length(w) == k
            @test sum(w) ≈ 2.0 atol=1e-14
            for d in 0:(2k - 1)
                exact = iseven(d) ? 2 / (d + 1) : 0.0
                @test sum(w .* x .^ d) ≈ exact atol=1e-12
            end
        end
        # Composite rule on arbitrary (unequal) segments: same exactness, and the
        # weights integrate the domain width.
        edges = [0.0, 0.3, 1.7, 5.0]
        nodes, wts = MacroEconometricModels._composite_quadrature(edges, 4)
        @test length(nodes) == 3 * 4
        @test sum(wts) ≈ 5.0 atol=1e-12
        for d in 0:7
            @test sum(wts .* nodes .^ d) ≈ 5.0^(d + 1) / (d + 1) atol=1e-9
        end
        # Grid-derived rule inherits the asset grid as its segment edges.
        g = HAGrid(; assets=(0.0, 50.0, 40), income_states=2)
        nq, wq = winberry_quadrature(g; n_quad=3)
        @test length(nq) == 39 * 3
        @test sum(wq) ≈ 50.0 atol=1e-10
        @test all(g.grids[1][1] .<= nq .<= g.grids[1][end])
        g2 = HAGrid(; liquid=(0.0, 5.0, 10), illiquid=(0.0, 5.0, 10), income_states=2)
        @test_throws ArgumentError winberry_quadrature(g2)
    end

    @testset "analytic oracles: the max-entropy fit IS the known density" begin
        # (a) Matching mean 0 and variance 1 on a wide symmetric interval must return
        #     the Gaussian exactly: g ∝ exp(−z²/2), i.e. λ = (0, −1/2).
        pd = fit_parametric_density([0.0, 1.0]; bounds=(-8.0, 8.0),
                                    n_segments=200, n_quad=6)
        @test pd.converged
        @test pd.lambda[1] ≈ 0.0 atol=1e-10
        @test pd.lambda[2] ≈ -0.5 atol=1e-7
        @test pd.residual < 1e-10
        for a in (-2.0, -0.5, 0.0, 1.0, 2.5)
            @test parametric_density(pd, a) ≈ exp(-a^2 / 2) / sqrt(2π) rtol=1e-6
        end

        # (b) The exponential distribution with rate 1 has centered moments
        #     (1, 1, 2, 9); the four-moment max-entropy fit must recover exp(−a)
        #     POINTWISE, not merely match the moments. In standardized coordinates
        #     z = a − 1, so the answer is λ = (−1, 0, 0, 0).
        pd4 = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                     n_segments=400, n_quad=6, tol=1e-12)
        # `tol` is RELATIVE to the target scale since #514, so the flag is portable
        # on the ill-conditioned four-moment basis again. Measured residual 4.1e-15
        # against an effective tolerance of 9e-12 — 2177x of headroom.
        @test pd4.converged
        @test pd4.residual < 1e-6
        @test pd4.lambda[1] ≈ -1.0 atol=1e-8
        @test all(abs.(pd4.lambda[2:end]) .< 1e-8)
        for a in (0.0, 0.25, 1.0, 2.0, 5.0)
            @test parametric_density(pd4, a) ≈ exp(-a) rtol=1e-6
        end
    end

    @testset "moment round trip (fit ∘ moments = identity)" begin
        nodes, wts = MacroEconometricModels._composite_quadrature(
            collect(range(-6.0, 12.0; length=301)), 5)
        # `converged` used to compare an ABSOLUTE residual against `tol`, which the
        # four-moment basis (Hessian cond ~1e8) could not meet portably — its
        # residual is 2.8e-10 here against a 1e-10 request, so the flag was asserted
        # only for the well-conditioned bases. Since #514 the test is relative to the
        # target scale max(1, max|mu|), which is 3.5 for this basis, so 2.8e-10 sits
        # inside an effective 3.5e-10 and every basis can assert the flag again.
        for targets in ([2.0, 4.0], [2.0, 4.0, 3.0], [1.0, 2.0, 1.5, 14.0])
            pd = MacroEconometricModels._fit_parametric_density(
                copy(targets), nodes, wts; tol=1e-10)
            @test pd.converged
            @test pd.residual < 1e-6            # a hard bound for every basis
            @test parametric_moments(pd, nodes, wts) ≈ targets rtol=1e-7
            # The density integrates to one over the reference interval.
            @test sum(wts .* [parametric_density(pd, a) for a in nodes]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "#514: convergence is relative to the target scale, not absolute" begin
        MEM = MacroEconometricModels
        # The residual is a moment mismatch in STANDARDIZED units, so an absolute
        # threshold demands more relative precision the larger the targets are --
        # and they are largest exactly where the basis is worst conditioned.
        for (mom, want) in (([0.0, 1.0], 1.0), ([2.0, 4.0, 3.0], 1.0),
                            ([1.0, 2.0, 1.5, 14.0], 3.5), ([1.0, 1.0, 2.0, 9.0], 9.0))
            _, _, mu = MEM._standardized_targets(collect(Float64, mom))
            @test max(1.0, maximum(abs, mu)) ≈ want
        end

        # Every basis converges, including the two four-moment ones that could not
        # meet an absolute tolerance.
        pd4 = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                     n_segments=400, n_quad=6, tol=1e-12)
        @test pd4.converged
        @test pd4.residual < 1e-12 * 9.0        # inside the RELATIVE tolerance

        # A tolerance no arithmetic can meet still converges, because below
        # sqrt(eps) relative the residual is gradient noise rather than a mismatch
        # the solve could act on. The fit is fully accurate there.
        pd_floor = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                          n_segments=400, n_quad=6, tol=1e-30)
        @test pd_floor.converged
        @test pd_floor.residual < sqrt(eps(Float64)) * 9.0
        @test pd_floor.lambda[1] ≈ -1.0 atol=1e-8
        for a in (0.0, 1.0, 5.0)
            @test parametric_density(pd_floor, a) ≈ exp(-a) rtol=1e-6
        end

        # ... and that leniency must NOT rescue a fit that genuinely failed. Both
        # of these stall at an O(1) residual, nowhere near the floor.
        infeasible = fit_parametric_density([0.0, 1.0, 3.0, 1.0]; bounds=(-8.0, 8.0),
                                            tol=1e-30)
        @test !infeasible.converged
        @test infeasible.residual > 1.0
        starved = fit_parametric_density([1.0, 1.0, 2.0, 9.0]; bounds=(0.0, 40.0),
                                         n_segments=400, n_quad=6, max_iter=1)
        @test !starved.converged
        @test starved.residual > 1.0
    end

    @testset "analytic gradient/Hessian match ForwardDiff" begin
        # The fit uses closed-form derivatives of the log-normalizer rather than AD.
        # Cross-check both against ForwardDiff on the same objective (the docstring
        # promises this).
        FD = MacroEconometricModels.ForwardDiff
        nodes, wts = MacroEconometricModels._composite_quadrature(
            collect(range(-5.0, 9.0; length=201)), 5)
        moments = [1.5, 2.25, 1.2, 16.0]
        center, scale, mu = MacroEconometricModels._standardized_targets(moments)
        B = MacroEconometricModels._winberry_basis(nodes, center, scale, mu)
        F(lam) = begin
            u = B * lam
            umax = maximum(u)
            umax + log(sum(wts .* exp.(u .- umax)))
        end
        for lam in ([0.0, -0.5, 0.0, 0.0], [-0.4, -0.3, 0.05, -0.02])
            _, p = MacroEconometricModels._log_normalizer(B * lam, wts)
            grad_analytic = B' * p
            hess_analytic = B' * (B .* p) - grad_analytic * grad_analytic'
            @test grad_analytic ≈ FD.gradient(F, lam) rtol=1e-8
            @test hess_analytic ≈ FD.hessian(F, lam) rtol=1e-6
            # The Hessian is a covariance matrix, hence symmetric PSD.
            @test hess_analytic ≈ hess_analytic' atol=1e-12
            @test minimum(eigvals(Symmetric(hess_analytic))) > -1e-12
        end
        # ∇ = 0 is exactly the moment-matching condition: at the converged λ the
        # analytic gradient vanishes and the fitted density's central moments are
        # the targets.
        pd = MacroEconometricModels._fit_parametric_density(copy(moments), nodes, wts;
                                                            tol=1e-12)
        @test pd.converged
        _, p_star = MacroEconometricModels._log_normalizer(B * pd.lambda, wts)
        @test maximum(abs, B' * p_star) < 1e-11
        @test parametric_moments(pd, nodes, wts) ≈ moments rtol=1e-7
    end

    @testset "input validation" begin
        @test_throws ArgumentError fit_parametric_density([1.0]; bounds=(0.0, 1.0))
        @test_throws ArgumentError fit_parametric_density([1.0, -1.0]; bounds=(0.0, 1.0))
        @test_throws ArgumentError fit_parametric_density([1.0, 1.0])   # no quadrature
        @test_throws ArgumentError fit_parametric_density([1.0, 1.0]; nodes=[0.0, 1.0],
                                                          weights=[0.5])
        nodes, wts = MacroEconometricModels._composite_quadrature([0.0, 4.0], 5)
        @test_throws ArgumentError MacroEconometricModels._fit_parametric_density(
            [1.0, 1.0], nodes, wts; lambda_init=[0.0, 0.0, 0.0])
        ws = _win_small_spec()
        @test_throws ArgumentError HouseholdSystem{Float64}(
            _hh(ws).individual, _hh(ws).income, _hh(ws).grid,
            _hh(ws).aggregation, _hh(ws).het_params;
            distribution=:histogram)
    end

    @testset "histogram ↔ moments" begin
        g = HAGrid(; assets=(0.0, 20.0, 60), income_states=2)
        a = g.grids[1]
        d = zeros(60, 2)
        d[:, 1] .= exp.(-a ./ 3); d[:, 2] .= exp.(-a ./ 8)
        d ./= sum(d)
        M, mass = winberry_moments(d, g; n_moments=4)
        @test size(M) == (2, 4)
        @test sum(mass) ≈ 1.0 atol=1e-12
        # Rows must equal the discrete conditional moments, computed independently.
        for j in 1:2
            p = d[:, j] ./ mass[j]
            m1 = sum(p .* a)
            @test M[j, 1] ≈ m1 rtol=1e-12
            for i in 2:4
                @test M[j, i] ≈ sum(p .* (a .- m1) .^ i) rtol=1e-10
            end
        end
        # Flattened input is accepted and gives the same answer.
        @test first(winberry_moments(vec(d), g; n_moments=4)) ≈ M
        @test_throws ArgumentError winberry_moments(d, g; n_moments=1)

        # Explicit tol for the same reason as above: under the 1e-10 default this fit
        # lands at 7.5e-11 (75% of the threshold), so `converged` is decided by
        # rounding rather than by the fit. The lambda vector and the reconstructed
        # histogram are identical to 4e-10 / 1e-11 across tol in [1e-10, 1e-6].
        fam = fit_winberry(d, g; n_moments=3, tol=1e-8)
        @test fam isa WinberryFamily{Float64}
        @test fam.converged
        @test length(fam.densities) == 2
        @test fam.n_moments == 3
        h = winberry_histogram(fam, g)
        @test length(h) == 120
        @test all(h .>= 0)
        @test sum(h) ≈ 1.0 atol=1e-12
        # Per-income-state mass is preserved by the rendering.
        for j in 1:2
            @test sum(h[((j - 1) * 60 + 1):(j * 60)]) ≈ mass[j] rtol=1e-10
        end
        # The rendered histogram carries roughly the family's own mean.
        @test sum(h .* repeat(a, 2)) ≈ sum(mass .* M[:, 1]) rtol=1e-3
    end

    # Remaining testsets solve shipped HA examples (SS + Reiter). Smoke / numerical
    # CI keep the quadrature / max-entropy oracles above; LTS still runs these.
    if !(FAST || NUMERICAL)
    @testset "moment fixed point is genuinely stationary and tracks Young" begin
        spec = _win_small_spec()
        ss = compute_steady_state(spec; grid_check=:none)
        a_pol = ss.policies[:savings]
        nodes, wts = winberry_quadrature(ss.grid; n_quad=4)
        M_young, mass_y = winberry_moments(ss.distribution, ss.grid; n_moments=3)
        K_young = sum(mass_y .* M_young[:, 1])
        @test K_young ≈ ss.aggregates[:K] rtol=1e-10

        errs = Float64[]
        for nm in (2, 3, 4)
            st = MacroEconometricModels._winberry_stationary(
                a_pol, ss.grid, ss.income; n_moments=nm)
            @test st.converged
            @test size(st.moments) == (3, nm)
            # Income-state masses are the ergodic distribution of the income chain.
            @test st.mass ≈ vec(sum(ss.distribution; dims=1)) rtol=1e-8
            # It really is a FIXED POINT: one more application of the law of motion
            # leaves it where it is (this is the property the linearization needs).
            M_next, _, _ = MacroEconometricModels._winberry_forward(
                st.moments, st.mass, a_pol, ss.grid, ss.income, nodes, wts;
                lambda_warm=st.lambdas)
            dev = MacroEconometricModels._winberry_to_state(
                M_next .- st.moments, MacroEconometricModels._winberry_scales(st.moments))
            @test maximum(abs, dev) < 1e-8
            K_w = sum(st.mass .* st.moments[:, 1])
            push!(errs, abs(K_w - K_young) / K_young)
            # A parametric solve started with no guess must find the same point.
            st_cold = MacroEconometricModels._winberry_stationary(
                a_pol, ss.grid, ss.income; n_moments=nm, M_init=nothing)
            @test st_cold.moments ≈ st.moments rtol=1e-6
        end
        # The reduction is accurate, and more moments do not make it worse.
        @test all(errs .< 0.10)
        @test errs[3] <= errs[1] + 1e-12
    end

    @testset "steady state with distribution=:winberry" begin
        spec_y = _win_small_spec()
        spec_w = _win_small_spec(; distribution=:winberry)
        @test _hh(spec_y).distribution === :young
        @test _hh(spec_w).distribution === :winberry
        ss_y = compute_steady_state(spec_y; grid_check=:none)
        ss_w = compute_steady_state(spec_w; grid_check=:none)

        # The equilibrium is cleared on the histogram either way, so prices and
        # aggregates are identical — only the extra parametric object differs.
        @test ss_y.parametric === nothing
        @test ss_w.parametric isa WinberryFamily{Float64}
        @test ss_w.prices[:r] == ss_y.prices[:r]
        @test ss_w.aggregates[:K] == ss_y.aggregates[:K]
        @test ss_w.parametric.converged
        @test ss_w.parametric.n_moments == 3
        @test length(ss_w.parametric.densities) == 3
        @test sum(ss_w.parametric.mass) ≈ 1.0 atol=1e-12
        @test all(pd -> pd.residual < 1e-9, ss_w.parametric.densities)

        # aggregates[:K_winberry] is the family's OWN stationary aggregate, so the
        # gap against :K is the reduction error — small but not zero.
        @test haskey(ss_w.aggregates, :K_winberry)
        @test !haskey(ss_y.aggregates, :K_winberry)
        rel = abs(ss_w.aggregates[:K_winberry] - ss_w.aggregates[:K]) / ss_w.aggregates[:K]
        @test 0 < rel < 0.10

        # n_moments is honoured, and more moments do not degrade the aggregate.
        ss_w5 = compute_steady_state(spec_w; grid_check=:none, n_moments=5)
        @test ss_w5.parametric.n_moments == 5
        rel5 = abs(ss_w5.aggregates[:K_winberry] - ss_w5.aggregates[:K]) / ss_w5.aggregates[:K]
        @test rel5 <= rel + 1e-10

        # `distribution=` on the call overrides the spec in both directions.
        @test compute_steady_state(spec_y; grid_check=:none,
                                   distribution=:winberry).parametric !== nothing
        @test compute_steady_state(spec_w; grid_check=:none,
                                   distribution=:young).parametric === nothing
        @test_throws ArgumentError compute_steady_state(spec_y; grid_check=:none,
                                                        distribution=:bogus)
    end

    @testset "Reiter linearization on the moment state" begin
        spec_y = _win_small_spec()
        spec_w = _win_small_spec(; distribution=:winberry)
        ss_y = compute_steady_state(spec_y; grid_check=:none)
        ss_w = compute_steady_state(spec_w; grid_check=:none)
        sol_y = solve(spec_y; method=:reiter, ss=ss_y)
        sol_w = solve(spec_w; method=:reiter, ss=ss_w)

        n_e = _hh(spec_w).grid.n_income
        # The distribution state is n_income × n_moments — far fewer than the
        # histogram's n_a × n_income, and fewer than the SVD reduction as well.
        @test sol_w.n_reduced == n_e * 3
        @test sol_w.n_reduced < sol_y.n_reduced
        @test sol_w.n_reduced < _hh(spec_w).grid.total_individual_states
        @test sol_w.method === :reiter
        @test is_determined(sol_w)
        @test maximum(abs, eigvals(sol_w.linear_solution.G1)) < 1.0
        @test 0.5 < sol_w.explained_variance <= 1.0

        # The reduction basis maps moment deviations back to the full histogram, so
        # distribution IRFs work unchanged — and every column is mass-preserving.
        @test size(sol_w.reduction_basis) == (_hh(spec_w).grid.total_individual_states,
                                              sol_w.n_reduced)
        @test maximum(abs, vec(sum(sol_w.reduction_basis; dims=1))) < 1e-8
        di = distribution_irf(sol_w, 6)
        @test size(di) == (_hh(spec_w).grid.n_points[1], n_e, 6)
        @test maximum(abs, di) > 0
        @test abs(sum(di[:, :, 1])) < 1e-8

        # Aggregate capital IRFs agree with the Young-based Reiter system. K is the
        # state just after the distribution block in both.
        function _agg_path(sol, H)
            G1 = sol.linear_solution.G1
            x = sol.linear_solution.impact[:, 1]
            out = zeros(H)
            for h in 1:H
                out[h] = x[sol.n_reduced + 1]
                x = G1 * x
            end
            return out
        end
        H = 20
        iy = _agg_path(sol_y, H)
        iw = _agg_path(sol_w, H)
        scale = maximum(abs, iy)
        @test scale > 0
        @test maximum(abs, iw .- iy) / scale < 0.05
        @test cor(iy, iw) > 0.999

        # More moments must not move the aggregate IRF much further away.
        sol_w5 = solve(spec_w; method=:reiter,
                       ss=compute_steady_state(spec_w; grid_check=:none, n_moments=5),
                       n_moments=5)
        @test sol_w5.n_reduced == n_e * 5
        @test maximum(abs, _agg_path(sol_w5, H) .- iy) / scale < 0.05
    end

    @testset "Huggett closure and the built-in examples" begin
        spec = load_ha_example(:huggett; distribution=:winberry)
        @test _hh(spec).distribution === :winberry
        @test _hh(load_ha_example(:huggett)).distribution === :young
        @test _hh(load_ha_example(:krusell_smith; distribution=:winberry)).distribution === :winberry
        ss = compute_steady_state(spec; grid_check=:none)
        @test ss.parametric isa WinberryFamily{Float64}
        # Huggett is zero net supply: the parametric family's own aggregate must
        # also be (nearly) zero, without ever having been told so.
        @test abs(ss.aggregates[:K_winberry]) < 1e-2
        sol = solve(spec; method=:reiter, ss=ss)
        @test sol.n_reduced == _hh(spec).grid.n_income * 3
        @test is_determined(sol)
        @test maximum(abs, eigvals(sol.linear_solution.G1)) < 1.0
    end

    @testset "display" begin
        spec = _win_small_spec(; distribution=:winberry)
        ss = compute_steady_state(spec; grid_check=:none)
        fam = ss.parametric
        str_f = sprint(show, fam)
        @test occursin("WinberryFamily", str_f)
        @test occursin("3 moments", str_f)
        @test occursin("converged=true", str_f)
        str_d = sprint(show, fam.densities[1])
        @test occursin("ParametricDensity", str_d)
        @test occursin("converged=true", str_d)
        # `report` writes to stdout; on Julia 1.12 redirect_stdout no longer accepts
        # an IOBuffer, so capture through a temporary file.
        out = mktemp() do path, f
            redirect_stdout(() -> report(ss), f)
            flush(f)
            read(path, String)
        end
        @test occursin("Winberry Parametric Family", out)
        @test occursin("K_winberry", out)
    end
    end

end

end # @testset "HA-DSGE Types"

@testset "#508: Euler-error metric measures approximation, not round-trip" begin
    MEM = MacroEconometricModels

    @testset "analytic fixture (hand-computed residuals)" begin
        # Everything below is exactly computable with pen and paper.
        #   a_grid = [0,1,2,3,4], one income state, u = log c so u'(c) = 1/c,
        #   beta = 0.96, r = 0.02, c(a) = 1 + a (linear, so the interpolant is
        #   EXACT at midpoints), a'(a) = 1.5 for every a.
        # Then c(a') = 2.5 always, E[u'(c')] = 0.4, and
        #   resid(a) = |1 - beta(1+r)*0.4*(1+a)| = |1 - 0.39168*(1+a)|.
        a_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
        c_pol = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1)
        a_pol = reshape(fill(1.5, 5), 5, 1)
        ip = MEM.IndividualProblem{Float64}(
            log, c -> 1 / c, u -> 1 / u, 0.96,
            (a, e, p) -> (1 + p[:r]) * a + e, [0.0], nothing, 1)
        grid = MEM.HAGrid{Float64}([a_grid], [5], 1, 1, [(0.0, 4.0)], [:assets])
        income = MEM.IncomeProcess{Float64}(reshape([1.0], 1, 1), [1.0], [1.0], :income)
        prices = Dict(:r => 0.02, :w => 1.0)

        sn = MEM._euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=:nodes)
        sm = MEM._euler_error_stats(c_pol, a_pol, ip, grid, income, prices; points=:midpoints)

        @test sn.n_evaluated == 5 && sn.n_constrained == 0 && sn.n_offgrid == 0
        @test sm.n_evaluated == 4 && sm.n_constrained == 0 && sm.n_offgrid == 0
        @test sn.max ≈ log10(0.9584) atol = 1e-12
        @test sn.mean ≈ log10(0.505024) atol = 1e-12
        @test sm.max ≈ log10(0.76256) atol = 1e-12
        @test sm.mean ≈ log10(0.39168) atol = 1e-12
        @test sm.points === :midpoints && sn.points === :nodes
        # the scalar wrapper is exactly the `max` field
        @test MEM._compute_euler_error(c_pol, a_pol, ip, grid, income, prices) == sm.max
        @test_throws ArgumentError MEM._euler_error_stats(c_pol, a_pol, ip, grid,
                                                          income, prices; points=:bogus)

        # Off-grid cells are excluded and counted, not scored: with a' = 10 > a_max
        # the last node leaves the grid, and so does the last midpoint, whose
        # interpolated a' is (1.5 + 10)/2 = 5.75.
        a_off = reshape([1.5, 1.5, 1.5, 1.5, 10.0], 5, 1)
        on = MEM._euler_error_stats(c_pol, a_off, ip, grid, income, prices; points=:nodes)
        om = MEM._euler_error_stats(c_pol, a_off, ip, grid, income, prices; points=:midpoints)
        @test on.n_offgrid == 1 && on.n_evaluated == 4
        @test om.n_offgrid == 1 && om.n_evaluated == 3
        # The remaining cells are untouched, so the max is the surviving maximum.
        @test on.max ≈ log10(0.60832) atol = 1e-12
        @test om.max ≈ log10(0.41248) atol = 1e-12

        # Constrained cells are excluded too (the Euler equation is an inequality there).
        a_con = reshape([0.0, 1.5, 1.5, 1.5, 1.5], 5, 1)
        cn = MEM._euler_error_stats(c_pol, a_con, ip, grid, income, prices; points=:nodes)
        @test cn.n_constrained == 1 && cn.n_evaluated == 4
    end

    @testset "shipped examples: the node metric flatters by 2.5-3.8 log10 units" begin
        for (ex, mid, nodes) in ((:krusell_smith, -2.2531, -6.0397),
                                 (:one_asset_hank, -2.2781, -6.0555),
                                 (:huggett, -1.9363, -4.4699))
            ss = compute_steady_state(load_ha_example(ex))
            @test ss.euler !== nothing
            # The headline number is now the off-node one.
            @test ss.euler_error ≈ mid atol = 1e-3
            @test ss.euler.midpoints.max ≈ mid atol = 1e-3
            @test ss.euler.nodes.max ≈ nodes atol = 1e-3
            # The node metric is optimistic by construction, never pessimistic.
            @test ss.euler.nodes.max < ss.euler.midpoints.max
            # mean < max, and both are finite and reported.
            @test ss.euler.midpoints.mean < ss.euler.midpoints.max
            @test isfinite(ss.euler.midpoints.mean)
            @test ss.euler.midpoints.n_evaluated > 0

            # The old convention is still reachable for continuity.
            ss_n = compute_steady_state(load_ha_example(ex); euler_points=:nodes)
            @test ss_n.euler_error ≈ nodes atol = 1e-3
        end
        @test_throws ArgumentError compute_steady_state(load_ha_example(:huggett);
                                                        euler_points=:bogus)
    end

    @testset "a truncating model no longer reports the better accuracy" begin
        # Same pre-fix Krusell-Smith clone the grid diagnostics use. Under the node
        # metric its 22 truncated cells were excused to ~1e-11 while the interior sat
        # at 2.5e-3, so truncation bought accuracy. Off-node it is scored worse than
        # the shipped calibration, which is the point of the change.
        base = load_ha_example(:krusell_smith)
        raw = rouwenhorst(0.966, 0.5, 7)
        e = exp.(raw.states); e ./= dot(raw.stationary_dist, e)
        old_inc = IncomeProcess{Float64}(raw.transition, e, raw.stationary_dist, :income)
        old = MacroEconometricModels._replace_household(base; income=old_inc,
            grid=HAGrid(; assets=(0.0, 200.0, 200), income_states=7))
        ss_bad = compute_steady_state(old; grid_check=:none)
        ss_good = compute_steady_state(base)

        @test ss_bad.euler_error > ss_good.euler_error      # measured -1.66 vs -2.25
        @test ss_bad.euler.midpoints.n_offgrid > 0          # and its cells do leave the grid
        @test ss_good.euler.midpoints.n_offgrid == 0
        # Under the OLD metric the gap was 3.4 log10 units the other way, which is
        # what made a truncating fit look respectable.
        @test ss_bad.euler.nodes.max ≈ -2.5952 atol = 1e-3
        @test ss_bad.euler.nodes.max < ss_bad.euler.midpoints.max
    end

    @testset "report(ss) names the convention" begin
        ss = compute_steady_state(load_ha_example(:huggett))
        io = IOBuffer(); report(io, ss); s = String(take!(io))
        @test occursin("Euler error", s)
        @test occursin("midpoints", s)          # the convention is stated, not implied
        @test occursin("mean (log10)", s)
        @test occursin("at grid nodes", s)
    end
end
