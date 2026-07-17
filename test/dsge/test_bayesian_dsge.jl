# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using StatsAPI
using LinearAlgebra
using Random
using Statistics
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _suppress_warnings = MacroEconometricModels._suppress_warnings

@testset "Bayesian DSGE" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Type Definitions and Construction
# ─────────────────────────────────────────────────────────────────────────────

@testset "BayesianDSGE <: AbstractDSGEModel" begin
    @test BayesianDSGE <: AbstractDSGEModel
    @test AbstractDSGEModel <: StatsAPI.StatisticalModel
    @test BayesianDSGE{Float64} <: AbstractDSGEModel
end

@testset "DSGEPrior construction" begin
    # Direct constructor
    names = [:alpha, :beta, :sigma]
    dists = Distribution[Normal(0.33, 0.1), Beta(5.0, 2.0), InverseGamma(2.0, 0.5)]
    lb = [-Inf, 0.0, 0.0]
    ub = [Inf, 1.0, Inf]
    prior = MacroEconometricModels.DSGEPrior{Float64}(names, dists, lb, ub)

    @test prior.param_names == [:alpha, :beta, :sigma]
    @test length(prior.distributions) == 3
    @test prior.lower == [-Inf, 0.0, 0.0]
    @test prior.upper == [Inf, 1.0, Inf]

    # Convenience constructor from Dict
    d = Dict(:alpha => Normal(0.33, 0.1),
             :beta => Beta(5.0, 2.0),
             :sigma => InverseGamma(2.0, 0.5))
    lower_d = Dict(:beta => 0.0, :sigma => 0.0)
    upper_d = Dict(:beta => 1.0)
    prior2 = MacroEconometricModels.DSGEPrior(d; lower=lower_d, upper=upper_d)

    @test length(prior2.param_names) == 3
    @test :alpha in prior2.param_names
    @test :beta in prior2.param_names
    @test :sigma in prior2.param_names
    # Sorted order
    @test issorted(prior2.param_names)
    # Check bounds are correctly assigned
    idx_beta = findfirst(==(:beta), prior2.param_names)
    @test prior2.lower[idx_beta] == 0.0
    @test prior2.upper[idx_beta] == 1.0
    idx_sigma = findfirst(==(:sigma), prior2.param_names)
    @test prior2.lower[idx_sigma] == 0.0
    @test prior2.upper[idx_sigma] == Inf
    idx_alpha = findfirst(==(:alpha), prior2.param_names)
    @test prior2.lower[idx_alpha] == -Inf
    @test prior2.upper[idx_alpha] == Inf
end

@testset "DSGEPrior validation" begin
    # Mismatched lengths
    @test_throws AssertionError MacroEconometricModels.DSGEPrior{Float64}(
        [:a, :b], [Normal()], [0.0, 0.0], [1.0, 1.0]
    )
    @test_throws AssertionError MacroEconometricModels.DSGEPrior{Float64}(
        [:a], [Normal()], [0.0, 0.0], [1.0]
    )
    @test_throws AssertionError MacroEconometricModels.DSGEPrior{Float64}(
        [:a], [Normal()], [0.0], [1.0, 2.0]
    )
end

@testset "DSGEStateSpace construction" begin
    n_states = 4
    n_obs = 2
    n_shocks = 2

    G1 = 0.5 * Matrix{Float64}(I, n_states, n_states)
    impact = randn(n_states, n_shocks)
    Z = randn(n_obs, n_states)
    d = zeros(n_obs)
    H = Matrix{Float64}(0.01 * I(n_obs))
    Q = Matrix{Float64}(I(n_shocks))

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)

    @test size(ss.G1) == (n_states, n_states)
    @test size(ss.impact) == (n_states, n_shocks)
    @test size(ss.Z) == (n_obs, n_states)
    @test length(ss.d) == n_obs
    @test size(ss.H) == (n_obs, n_obs)
    @test size(ss.Q) == (n_shocks, n_shocks)

    # Verify H_inv is correct
    @test ss.H_inv * ss.H ≈ I(n_obs) atol=1e-10

    # Verify log_det_H
    @test ss.log_det_H ≈ logdet(H) atol=1e-10
end

@testset "DSGEStateSpace validation" begin
    # Non-square G1
    @test_throws AssertionError MacroEconometricModels.DSGEStateSpace{Float64}(
        randn(3, 4), randn(3, 2), randn(2, 3), zeros(2),
        0.01 * Matrix{Float64}(I(2)), Matrix{Float64}(I(2))
    )
    # Z columns don't match G1
    @test_throws AssertionError MacroEconometricModels.DSGEStateSpace{Float64}(
        0.5 * Matrix{Float64}(I(3)), randn(3, 2), randn(2, 4), zeros(2),
        0.01 * Matrix{Float64}(I(2)), Matrix{Float64}(I(2))
    )
    # d length mismatch
    @test_throws AssertionError MacroEconometricModels.DSGEStateSpace{Float64}(
        0.5 * Matrix{Float64}(I(3)), randn(3, 2), randn(2, 3), zeros(3),
        0.01 * Matrix{Float64}(I(2)), Matrix{Float64}(I(2))
    )
end

@testset "NonlinearStateSpace construction" begin
    nx = 2  # states
    ny = 1  # controls
    nv = nx + 1  # states + shocks
    n_obs = 2

    hx = randn(nx, nv)
    gx = randn(ny, nv)
    eta = zeros(nv, 1)
    eta[nx+1, 1] = 1.0
    ss = ones(nx + ny)
    Z = randn(n_obs, ny)
    d = zeros(n_obs)
    H = Matrix{Float64}(0.01 * I(n_obs))

    # Order 1
    nlss = MacroEconometricModels.NonlinearStateSpace{Float64}(
        hx, gx, eta, ss, [1, 2], [3], 1,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
        Z, d, H
    )
    @test nlss.order == 1
    @test nlss.hxx === nothing
    @test nlss.hxxx === nothing
    @test nlss.H_inv * nlss.H ≈ I(n_obs) atol=1e-10
    @test nlss.log_det_H ≈ logdet(H) atol=1e-10

    # Order 2
    hxx = randn(nx, nv * nv)
    gxx = randn(ny, nv * nv)
    hsig = randn(nx)
    gsig = randn(ny)
    nlss2 = MacroEconometricModels.NonlinearStateSpace{Float64}(
        hx, gx, eta, ss, [1, 2], [3], 2,
        hxx, gxx, hsig, gsig,
        nothing, nothing, nothing, nothing, nothing, nothing,
        Z, d, H
    )
    @test nlss2.order == 2
    @test nlss2.hxx !== nothing
    @test size(nlss2.hxx) == (nx, nv * nv)
    @test nlss2.hxxx === nothing

    # Order 3
    hxxx = randn(nx, nv * nv * nv)
    gxxx = randn(ny, nv * nv * nv)
    hsx = randn(nx, nv)
    gsx = randn(ny, nv)
    hsss = randn(nx)
    gsss = randn(ny)
    nlss3 = MacroEconometricModels.NonlinearStateSpace{Float64}(
        hx, gx, eta, ss, [1, 2], [3], 3,
        hxx, gxx, hsig, gsig,
        hxxx, gxxx, hsx, gsx, hsss, gsss,
        Z, d, H
    )
    @test nlss3.order == 3
    @test nlss3.hxxx !== nothing
    @test size(nlss3.hxxx) == (nx, nv * nv * nv)
end

@testset "NonlinearStateSpace validation" begin
    # Invalid order
    @test_throws AssertionError MacroEconometricModels.NonlinearStateSpace{Float64}(
        randn(2, 3), randn(1, 3), zeros(3, 1), ones(3), [1, 2], [3], 4,
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
        randn(2, 1), zeros(2), 0.01 * Matrix{Float64}(I(2))
    )
end

@testset "PFWorkspace allocation — order 1" begin
    n_states = 4
    n_obs = 2
    n_shocks = 2
    N = 100

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N)

    @test size(ws.particles) == (n_states, N)
    @test size(ws.particles_new) == (n_states, N)
    @test length(ws.log_weights) == N
    @test length(ws.weights) == N
    @test ws.weights ≈ fill(1.0 / N, N)
    @test length(ws.ancestors) == N
    @test length(ws.cumweights) == N
    @test size(ws.shocks) == (n_shocks, N)
    @test size(ws.innovations) == (n_obs, N)
    @test size(ws.tmp_obs) == (n_obs, N)

    # Order 1: no higher-order buffers
    @test ws.kron_buffer === nothing
    @test ws.kron3_buffer === nothing
    @test ws.particles_fo === nothing
    @test ws.particles_so === nothing
    @test ws.particles_to === nothing

    # No CSMC by default
    @test ws.reference_trajectory === nothing
    @test ws.reference_ancestors === nothing
end

@testset "Particle-filter likelihood correct under adaptive resampling (#128)" begin
    # On a small linear-Gaussian state space the Kalman likelihood is exact. The bootstrap PF
    # log-likelihood must match it even when adaptive resampling SKIPS steps (low threshold):
    # the ratio-form increment logsumexp(L_t)-logsumexp(L_{t-1}) keeps the carried non-uniform
    # weights, whereas the pre-fix logsumexp(log g_t)-log N increment biased every skipped step.
    ns, no, nsh = 2, 1, 1
    G1 = [0.6 0.0; 0.0 0.4]
    impact = reshape([1.0, 0.5], ns, nsh)
    Z = [1.0 1.0]
    d = [0.0]
    H = reshape([0.05], no, no)
    Q = reshape([1.0], nsh, nsh)
    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)
    T_obs = 50
    data = let rng = MersenneTwister(1), x = zeros(ns), dat = zeros(no, T_obs), Hc = sqrt(H[1, 1])
        for t in 1:T_obs
            x = G1 * x + impact * randn(rng, nsh)
            dat[:, t] = Z * x .+ d .+ Hc * randn(rng, no)
        end
        dat
    end
    ll_kalman = MacroEconometricModels._kalman_loglikelihood(ss, data)
    N = 2500
    for thr in (0.1, 0.5, 1.0)   # 0.1 skips most resamples yet must still match Kalman
        lls = Float64[]
        for s in 1:50
            ws = MacroEconometricModels._allocate_pf_workspace(Float64, ns, no, nsh, N)
            push!(lls, MacroEconometricModels._bootstrap_particle_filter!(
                ws, ss, data, T_obs; threshold=thr, rng=MersenneTwister(1000 + s)))
        end
        @test isapprox(mean(lls), ll_kalman; rtol=0.05)
    end
end

@testset "PFWorkspace allocation — order 2" begin
    n_states = 4
    n_obs = 2
    n_shocks = 2
    N = 50
    nv = n_states + n_shocks  # augmented dimension

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                        nv=nv, order=2)

    @test ws.kron_buffer !== nothing
    @test size(ws.kron_buffer) == (nv * nv, N)
    @test ws.kron3_buffer === nothing
    @test ws.particles_fo !== nothing
    @test size(ws.particles_fo) == (n_states, N)
    @test ws.particles_so !== nothing
    @test size(ws.particles_so) == (n_states, N)
    @test ws.particles_to === nothing
end

@testset "PFWorkspace allocation — order 3" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 30
    nv = n_states + n_shocks

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                        nv=nv, order=3)

    @test ws.kron_buffer !== nothing
    @test size(ws.kron_buffer) == (nv * nv, N)
    @test ws.kron3_buffer !== nothing
    @test size(ws.kron3_buffer) == (nv * nv * nv, N)
    @test ws.particles_fo !== nothing
    @test ws.particles_so !== nothing
    @test ws.particles_to !== nothing
    @test size(ws.particles_to) == (n_states, N)
end

@testset "PFWorkspace allocation — CSMC" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 20
    T_obs = 100

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                        T_obs=T_obs)

    @test ws.reference_trajectory !== nothing
    @test size(ws.reference_trajectory) == (n_states, T_obs)
    @test ws.reference_ancestors !== nothing
    @test length(ws.reference_ancestors) == T_obs
end

@testset "PFWorkspace resize" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 50
    nv = n_states + n_shocks

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                        nv=nv, order=2)
    @test size(ws.particles, 2) == N

    N_new = 100
    MacroEconometricModels._resize_pf_workspace!(ws, N_new)

    @test size(ws.particles) == (n_states, N_new)
    @test size(ws.particles_new) == (n_states, N_new)
    @test length(ws.log_weights) == N_new
    @test length(ws.weights) == N_new
    @test length(ws.ancestors) == N_new
    @test size(ws.shocks) == (n_shocks, N_new)
    @test size(ws.innovations) == (n_obs, N_new)
    @test size(ws.tmp_obs) == (n_obs, N_new)
    @test size(ws.kron_buffer) == (nv * nv, N_new)
    @test ws.kron3_buffer === nothing
    @test size(ws.particles_fo) == (n_states, N_new)
    @test size(ws.particles_so) == (n_states, N_new)
    @test ws.particles_to === nothing
end

@testset "SMCState construction" begin
    n_params = 3
    N_smc = 200

    state = MacroEconometricModels.SMCState{Float64}(
        randn(n_params, N_smc),
        zeros(N_smc),
        zeros(N_smc),
        zeros(N_smc),
        Float64[0.0, 0.5, 1.0],
        Float64[200.0, 180.0, 190.0],
        Float64[0.3, 0.25],
        0.0,
        MacroEconometricModels.PFWorkspace{Float64}[],
        Matrix{Float64}(I(n_params))
    )

    @test size(state.theta_particles) == (n_params, N_smc)
    @test length(state.log_weights) == N_smc
    @test length(state.phi_schedule) == 3
    @test state.log_marginal_likelihood == 0.0
    @test size(state.proposal_cov) == (n_params, n_params)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Observation Equation Builder & DSGE Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

@testset "Build observation equation" begin
    # Create a simple 3-variable DSGE spec
    spec = _suppress_warnings() do
        @dsge begin
            parameters: rho_y = 0.8, rho_pi = 0.5, rho_r = 0.6
            endogenous: y, pi_var, r
            exogenous: eps_y, eps_pi, eps_r
            y[t] = rho_y * y[t-1] + eps_y[t]
            pi_var[t] = rho_pi * pi_var[t-1] + eps_pi[t]
            r[t] = rho_r * r[t-1] + eps_r[t]
        end
    end

    # Build observation equation for 2 observables (y and r)
    observables = [:y, :r]
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, observables, nothing)

    @test size(Z) == (2, 3)  # 2 observables, 3 states
    # Z should select y (index 1) and r (index 3)
    @test Z[1, 1] == 1.0
    @test Z[1, 2] == 0.0
    @test Z[1, 3] == 0.0
    @test Z[2, 1] == 0.0
    @test Z[2, 2] == 0.0
    @test Z[2, 3] == 1.0

    # d should be steady-state values at observable indices
    # (zeros when steady_state not yet computed)
    @test length(d) == 2
    @test d[1] == 0.0  # y steady state (default zero)
    @test d[2] == 0.0  # r steady state (default zero)

    # Verify with computed steady state
    spec_ss = compute_steady_state(spec)
    Z2, d2, _ = MacroEconometricModels._build_observation_equation(spec_ss, observables, nothing)
    @test d2[1] == spec_ss.steady_state[1]
    @test d2[2] == spec_ss.steady_state[3]

    # H defaults to ZERO measurement error (T042); n_obs=2 ≤ n_shocks=3, no singularity.
    @test size(H) == (2, 2)
    @test all(iszero, H)
    @test H == zeros(2, 2)
    @test H[1, 2] == 0.0

    # Error: unknown observable
    @test_throws ArgumentError MacroEconometricModels._build_observation_equation(
        spec, [:y, :nonexistent], nothing)

    # Error: empty observables
    @test_throws ArgumentError MacroEconometricModels._build_observation_equation(
        spec, Symbol[], nothing)
end

@testset "Build observation equation with measurement error" begin
    spec = _suppress_warnings() do
        @dsge begin
            parameters: rho = 0.8
            endogenous: y, c
            exogenous: eps_y, eps_c
            y[t] = rho * y[t-1] + eps_y[t]
            c[t] = rho * c[t-1] + eps_c[t]
        end
    end

    me = [0.1, 0.2]
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y, :c], me)

    @test H ≈ diagm([0.01, 0.04])  # diag(me.^2)
    @test H[1, 1] ≈ 0.01
    @test H[2, 2] ≈ 0.04
    @test H[1, 2] ≈ 0.0

    # Wrong length measurement error
    @test_throws ArgumentError MacroEconometricModels._build_observation_equation(
        spec, [:y, :c], [0.1])
end

@testset "Stochastic singularity & auto measurement error (T042)" begin
    spec1 = _suppress_warnings() do
        @dsge begin
            parameters: rho = 0.5
            endogenous: y, k
            exogenous: eps
            y[t] = rho * y[t-1] + eps[t]
            k[t] = 0.5 * k[t-1] + y[t]
            steady_state = [0.0, 0.0]
        end
    end

    _suppress_warnings() do
        # (a) Default zero ME when n_obs ≤ n_shocks (2 obs, 2 shocks).
        spec2 = @dsge begin
            parameters: rho = 0.8
            endogenous: y, c
            exogenous: eps_y, eps_c
            y[t] = rho * y[t-1] + eps_y[t]
            c[t] = rho * c[t-1] + eps_c[t]
        end
        _, _, H0 = MacroEconometricModels._build_observation_equation(spec2, [:y, :c], nothing)
        @test all(iszero, H0)

        # (b) Builder throws on stochastic singularity (2 obs > 1 shock).
        @test_throws MacroEconometricModels.StochasticSingularityError MacroEconometricModels._build_observation_equation(
            spec1, [:y, :k], nothing)

        # (c) Explicit vector ME with n_obs>n_shocks does NOT throw.
        _, _, Hv = MacroEconometricModels._build_observation_equation(spec1, [:y, :k], [0.01, 0.01])
        @test Hv ≈ diagm([1e-4, 1e-4])

        # (d) Entry-point singularity check fires before sampling.
        spec1c = compute_steady_state(spec1)
        data1 = simulate(solve(spec1c; method=:gensys), 50; rng=Random.MersenneTwister(1))
        @test_throws MacroEconometricModels.StochasticSingularityError estimate_dsge_bayes(
            spec1c, data1, [0.5]; priors=Dict(:rho => Beta(2, 2)),
            method=:smc, observables=[:y, :k], measurement_error=nothing,
            n_smc=20, rng=Random.MersenneTwister(0))
    end

    # (e) :auto scales per-observable to √(0.1·var) and warns (outside suppression).
    dm = reshape(Float64[1.0, 2.0, 3.0, 4.0, 5.0], 1, 5)
    me = @test_logs (:warn,) match_mode=:any MacroEconometricModels._resolve_measurement_error(
        :auto, dm, [:y])
    @test me[1] ≈ sqrt(0.1 * var(dm[1, :]))
    # per-observable scaling: a higher-variance series gets a larger SD
    dm2 = [ones(1, 5); 10.0 .* Float64[1.0 2.0 3.0 4.0 5.0]]
    me2 = MacroEconometricModels._resolve_measurement_error(:auto, dm2, [:a, :b])
    @test me2[2] > me2[1]
end

@testset "Build state space from DSGESolution" begin
    spec = _suppress_warnings() do
        @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: eps_y
            y[t] = rho * y[t-1] + eps_y[t]
        end
    end

    sol = _suppress_warnings() do
        solve(spec; method=:gensys)
    end

    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], nothing)
    ss = MacroEconometricModels._build_state_space(sol, Z, d, H)

    @test ss isa MacroEconometricModels.DSGEStateSpace{Float64}
    @test ss.G1 == sol.G1
    @test ss.impact == sol.impact
    @test ss.Z == Z
    @test ss.d == d
    @test ss.H == H
    @test ss.Q ≈ I(size(sol.impact, 2))
end

@testset "Kalman loglikelihood: AR(1)" begin
    Random.seed!(42)

    # AR(1): y_t = rho * y_{t-1} + eps_t
    rho_true = 0.8
    sigma_eps = 0.5
    T_sim = 200

    # Simulate data
    y = zeros(T_sim)
    for t in 2:T_sim
        y[t] = rho_true * y[t-1] + sigma_eps * randn()
    end

    # Build state space for correct model
    n_states = 1
    n_obs = 1
    n_shocks = 1
    G1 = fill(rho_true, 1, 1)
    impact = fill(sigma_eps, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H_me = fill(1e-6, 1, 1)  # tiny measurement error
    Q = ones(1, 1)

    ss_correct = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H_me, Q)

    # Data as n_obs x T_obs
    data = reshape(y, 1, T_sim)

    ll_correct = MacroEconometricModels._kalman_loglikelihood(ss_correct, data)

    @test isfinite(ll_correct)
    @test ll_correct < 0.0  # log-likelihood should be negative

    # Misspecified model: wrong rho
    G1_wrong = fill(0.2, 1, 1)
    ss_wrong = MacroEconometricModels.DSGEStateSpace{Float64}(G1_wrong, impact, Z, d, H_me, Q)
    ll_wrong = MacroEconometricModels._kalman_loglikelihood(ss_wrong, data)

    @test isfinite(ll_wrong)
    @test ll_correct > ll_wrong  # correct model should have higher log-likelihood
end

@testset "Kalman diffuse init: scale-invariant under a unit root (T040)" begin
    # Transition with one unit root (random walk) + one stationary direction.
    G1 = [1.0 0.0; 0.0 0.6]
    impact = [0.4 0.0; 0.0 0.3]
    Z = Matrix{Float64}(I, 2, 2)
    d = zeros(2)
    H = 1e-6 * Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)

    x = zeros(2, 200); rng = Random.MersenneTwister(20240709)
    for t in 2:200; x[:, t] = G1 * x[:, t-1] + impact * randn(rng, 2); end
    data = x

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)
    ll = _suppress_warnings() do
        MacroEconometricModels._kalman_loglikelihood(ss, data)
    end
    @test isfinite(ll)

    # Rescale ONLY the nonstationary STATE direction (Z compensates), so the OBSERVED
    # process is unchanged ⇒ the likelihood must be (near-)invariant. G1 diagonal ⇒
    # D*G1*Dinv == G1, so the transition is unchanged.
    c = 100.0; D = [c 0.0; 0.0 1.0]; Dinv = [1/c 0.0; 0.0 1.0]
    ss_scaled = MacroEconometricModels.DSGEStateSpace{Float64}(G1, D*impact, Z*Dinv, d, H, Q)
    ll_scaled = _suppress_warnings() do
        MacroEconometricModels._kalman_loglikelihood(ss_scaled, data)
    end
    # κ=1e6 diffuse init leaves an O(log c)≈4.6 residual; P0=10*I blows up (fails atol=15).
    @test isapprox(ll, ll_scaled; atol=15.0)
end

@testset "Kalman init: non-stability error propagates, not swallowed (T040)" begin
    G1 = fill(0.5, 1, 1)             # stationary ⇒ takes the solve_lyapunov branch
    impact_bad = zeros(2, 1)         # 2 rows vs n_states=1 ⇒ solve_lyapunov throws
    Z = ones(1, 1); d = zeros(1); H = fill(1e-6, 1, 1); Q = ones(1, 1)
    ss_bad = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact_bad, Z, d, H, Q)
    data = reshape(randn(Random.MersenneTwister(1), 20), 1, 20)
    # Old code swallowed this into P0=10I and returned a finite ll; now it propagates.
    @test_throws ArgumentError _suppress_warnings() do
        MacroEconometricModels._kalman_loglikelihood(ss_bad, data)
    end
end

@testset "Kalman loglikelihood: 2D with missing data" begin
    Random.seed!(123)

    # 2D VAR(1): x_t = G1 * x_{t-1} + eps_t
    G1 = [0.7 0.1; 0.0 0.5]
    impact = [0.3 0.0; 0.0 0.4]
    T_sim = 150

    # Simulate
    x = zeros(2, T_sim)
    for t in 2:T_sim
        x[:, t] = G1 * x[:, t-1] + impact * randn(2)
    end

    Z = Matrix{Float64}(I, 2, 2)
    d = zeros(2)
    H_me = 1e-4 * Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H_me, Q)

    # Complete data log-likelihood
    ll_complete = MacroEconometricModels._kalman_loglikelihood(ss, x)
    @test isfinite(ll_complete)
    @test ll_complete < 0.0

    # Add NaN at specific positions
    x_missing = copy(x)
    x_missing[1, 10] = NaN   # first variable missing at t=10
    x_missing[2, 25] = NaN   # second variable missing at t=25
    x_missing[1, 50] = NaN   # first variable missing at t=50
    x_missing[2, 50] = NaN   # both missing at t=50

    ll_missing = MacroEconometricModels._kalman_loglikelihood(ss, x_missing)
    @test isfinite(ll_missing)
    @test ll_missing != ll_complete  # should differ from complete data

    # All NaN for one period should still produce finite result
    x_all_nan = copy(x)
    x_all_nan[:, 30] .= NaN
    ll_all_nan = MacroEconometricModels._kalman_loglikelihood(ss, x_all_nan)
    @test isfinite(ll_all_nan)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Particle Filter Compute Kernels
# ─────────────────────────────────────────────────────────────────────────────

@testset "Linear transition kernel" begin
    Random.seed!(101)
    n_states = 3
    N = 100
    n_shocks = 2

    G1 = 0.5 * Matrix{Float64}(I, n_states, n_states)
    G1[1, 2] = 0.1
    impact = randn(n_states, n_shocks)

    S_old = randn(n_states, N)
    E = randn(n_shocks, N)
    S_new = zeros(n_states, N)

    MacroEconometricModels._pf_transition_linear!(S_new, S_old, E, G1, impact)

    # Verify against naive loop
    for k in 1:N
        expected = G1 * S_old[:, k] + impact * E[:, k]
        @test S_new[:, k] ≈ expected atol=1e-12
    end
end

@testset "Log-weight computation" begin
    Random.seed!(102)
    n_obs = 2
    N = 50
    n_states = 3

    Z = randn(n_obs, n_states)
    d = randn(n_obs)
    H = [0.1 0.01; 0.01 0.2]
    H_inv = Matrix{Float64}(inv(H))
    log_det_H = logdet(H)

    S = randn(n_states, N)
    y_t = randn(n_obs)

    log_w = zeros(N)
    innovations = zeros(n_obs, N)
    tmp_obs = zeros(n_obs, N)

    MacroEconometricModels._pf_log_weights!(log_w, innovations, tmp_obs,
                                             S, y_t, Z, d, H_inv, log_det_H)

    # Verify against naive implementation (full Gaussian log-density)
    log2pi_const = n_obs * log(2 * pi)
    for k in 1:N
        inn = y_t - Z * S[:, k] - d
        expected = -0.5 * (log2pi_const + log_det_H + dot(inn, H_inv * inn))
        @test log_w[k] ≈ expected atol=1e-10
    end
end

@testset "Systematic resampling" begin
    Random.seed!(103)
    N = 1000

    # Concentrated weights: [0.5, 0.3, 0.2, 0, 0, ...]
    weights = zeros(N)
    weights[1] = 0.5
    weights[2] = 0.3
    weights[3] = 0.2

    ancestors = zeros(Int, N)
    cumweights = zeros(N)

    MacroEconometricModels._systematic_resample!(ancestors, weights, cumweights, N,
                                                  Random.MersenneTwister(42))

    # Count how many times each original particle is selected
    counts = zeros(Int, N)
    for a in ancestors
        counts[a] += 1
    end

    # Particle 1 should get ~50%, particle 2 ~30%, particle 3 ~20%
    @test counts[1] == 500  # systematic resampling is deterministic given u
    @test counts[2] == 300
    @test counts[3] == 200
    @test sum(counts[4:end]) == 0

    # All ancestors should be valid
    @test all(1 .<= ancestors .<= N)
end

@testset "Logsumexp" begin
    # Basic correctness
    x = [1.0, 2.0, 3.0]
    @test MacroEconometricModels._logsumexp(x) ≈ log(exp(1.0) + exp(2.0) + exp(3.0)) atol=1e-12

    # Numerical stability with large values
    x_large = [1000.0, 1001.0, 1002.0]
    result = MacroEconometricModels._logsumexp(x_large)
    @test isfinite(result)
    @test result ≈ 1000.0 + log(1.0 + exp(1.0) + exp(2.0)) atol=1e-10

    # Numerical stability with very negative values
    x_neg = [-1000.0, -1001.0, -1002.0]
    result_neg = MacroEconometricModels._logsumexp(x_neg)
    @test isfinite(result_neg)
    @test result_neg ≈ -1000.0 + log(1.0 + exp(-1.0) + exp(-2.0)) atol=1e-10

    # Single element
    @test MacroEconometricModels._logsumexp([5.0]) ≈ 5.0

    # All -Inf
    @test MacroEconometricModels._logsumexp([-Inf, -Inf]) == -Inf
end

@testset "Kronecker buffer fill (2nd-order)" begin
    Random.seed!(104)
    nv = 3
    N = 20
    V = randn(nv, N)
    buffer = zeros(nv * nv, N)

    MacroEconometricModels._fill_kron_buffer!(buffer, V, nv)

    # Verify: buffer[(i-1)*nv+j, k] == V[i,k] * V[j,k]
    for k in 1:N, i in 1:nv, j in 1:nv
        idx = (i - 1) * nv + j
        @test buffer[idx, k] ≈ V[i, k] * V[j, k] atol=1e-14
    end
end

@testset "Kronecker buffer fill (3rd-order)" begin
    Random.seed!(105)
    nv = 2
    N = 10
    V = randn(nv, N)
    buffer = zeros(nv * nv * nv, N)

    MacroEconometricModels._fill_kron3_buffer!(buffer, V, nv)

    # Verify: buffer[(i-1)*nv^2 + (j-1)*nv + l, k] == V[i,k]*V[j,k]*V[l,k]
    for k in 1:N, i in 1:nv, j in 1:nv, l in 1:nv
        idx = (i - 1) * nv * nv + (j - 1) * nv + l
        @test buffer[idx, k] ≈ V[i, k] * V[j, k] * V[l, k] atol=1e-14
    end
end

@testset "PFWorkspace allocation with 3rd-order" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 30
    nv = n_states + n_shocks  # 4

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                        nv=nv, order=3)

    # Verify buffer sizes
    @test size(ws.kron_buffer) == (nv * nv, N)     # 16 x 30
    @test size(ws.kron3_buffer) == (nv^3, N)       # 64 x 30
    @test size(ws.particles_fo) == (n_states, N)   # 3 x 30
    @test size(ws.particles_so) == (n_states, N)   # 3 x 30
    @test size(ws.particles_to) == (n_states, N)   # 3 x 30
end

@testset "Bootstrap particle filter: AR(1)" begin
    Random.seed!(200)

    # AR(1): y_t = rho * y_{t-1} + sigma * eps_t,  observed with measurement error
    rho = 0.8
    sigma_eps = 0.5
    sigma_me = 0.1
    T_sim = 200

    # Simulate data
    x = zeros(T_sim)
    y = zeros(T_sim)
    for t in 2:T_sim
        x[t] = rho * x[t-1] + sigma_eps * randn()
    end
    for t in 1:T_sim
        y[t] = x[t] + sigma_me * randn()
    end

    # Build state space
    G1 = fill(rho, 1, 1)
    impact = fill(sigma_eps, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H = fill(sigma_me^2, 1, 1)
    Q = ones(1, 1)

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)
    data = reshape(y, 1, T_sim)

    # Kalman log-likelihood (exact)
    ll_kalman = MacroEconometricModels._kalman_loglikelihood(ss, data)

    # Bootstrap PF log-likelihood (N=500, average over multiple runs for stability)
    N_particles = 500
    n_runs = 10
    ll_pf_runs = zeros(n_runs)

    for r in 1:n_runs
        ws = MacroEconometricModels._allocate_pf_workspace(Float64, 1, 1, 1, N_particles)
        ll_pf_runs[r] = MacroEconometricModels._bootstrap_particle_filter!(
            ws, ss, data, T_sim; rng=Random.MersenneTwister(r * 1000))
    end

    ll_pf_mean = mean(ll_pf_runs)

    @test isfinite(ll_pf_mean)
    @test isfinite(ll_kalman)

    # PF should approximate Kalman within 15% relative error
    # (PF is noisy, so we use a generous tolerance)
    rel_error = abs(ll_pf_mean - ll_kalman) / abs(ll_kalman)
    @test rel_error < 0.15
end


@testset "Conditional SMC" begin
    Random.seed!(400)

    rho = 0.8
    sigma_eps = 0.5
    sigma_me = 0.1
    T_sim = 100

    x = zeros(T_sim)
    y = zeros(T_sim)
    for t in 2:T_sim
        x[t] = rho * x[t-1] + sigma_eps * randn()
    end
    for t in 1:T_sim
        y[t] = x[t] + sigma_me * randn()
    end

    G1 = fill(rho, 1, 1)
    impact = fill(sigma_eps, 1, 1)
    Z = ones(1, 1)
    d = zeros(1)
    H = fill(sigma_me^2, 1, 1)
    Q = ones(1, 1)

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)
    data = reshape(y, 1, T_sim)

    N_particles = 200

    # First run: bootstrap PF to get initial reference trajectory
    ws = MacroEconometricModels._allocate_pf_workspace(Float64, 1, 1, 1, N_particles;
                                                        T_obs=T_sim)
    ll_init = MacroEconometricModels._bootstrap_particle_filter!(
        ws, ss, data, T_sim; rng=Random.MersenneTwister(500), store_trajectory=true)

    @test isfinite(ll_init)
    @test ws.reference_trajectory !== nothing

    # CSMC run
    ll_csmc = MacroEconometricModels._conditional_smc!(
        ws, ss, data, T_sim; rng=Random.MersenneTwister(600))

    @test isfinite(ll_csmc)
    @test ll_csmc < 0.0  # log-likelihood should be negative

    # Error: no reference trajectory
    ws_no_ref = MacroEconometricModels._allocate_pf_workspace(Float64, 1, 1, 1, N_particles)
    @test_throws ArgumentError MacroEconometricModels._conditional_smc!(
        ws_no_ref, ss, data, T_sim)
end

@testset "Normalize log weights" begin
    log_w = [log(0.2), log(0.3), log(0.5)]
    weights = zeros(3)

    MacroEconometricModels._normalize_log_weights!(weights, log_w)

    @test sum(weights) ≈ 1.0 atol=1e-12
    @test weights[1] ≈ 0.2 atol=1e-12
    @test weights[2] ≈ 0.3 atol=1e-12
    @test weights[3] ≈ 0.5 atol=1e-12

    # With shifted log weights (should give same result)
    log_w_shifted = log_w .+ 1000.0
    MacroEconometricModels._normalize_log_weights!(weights, log_w_shifted)

    @test sum(weights) ≈ 1.0 atol=1e-12
    @test weights[1] ≈ 0.2 atol=1e-12
    @test weights[2] ≈ 0.3 atol=1e-12
    @test weights[3] ≈ 0.5 atol=1e-12
end

@testset "Resample particles" begin
    Random.seed!(106)
    n_states = 3
    N = 10

    S_old = randn(n_states, N)
    S_new = zeros(n_states, N)
    ancestors = [1, 1, 3, 3, 3, 5, 5, 7, 7, 7]

    MacroEconometricModels._resample_particles!(S_new, S_old, ancestors)

    for i in 1:N
        @test S_new[:, i] ≈ S_old[:, ancestors[i]]
    end
end

@testset "Stationary initialization" begin
    Random.seed!(107)

    G1 = [0.5 0.1; 0.0 0.3]
    impact = [0.2 0.0; 0.0 0.3]
    Z = Matrix{Float64}(I, 2, 2)
    d = zeros(2)
    H = 0.01 * Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)

    ss = MacroEconometricModels.DSGEStateSpace{Float64}(G1, impact, Z, d, H, Q)

    N = 1000
    ws = MacroEconometricModels._allocate_pf_workspace(Float64, 2, 2, 2, N)

    MacroEconometricModels._pf_initialize_stationary!(ws, ss; rng=Random.MersenneTwister(42))

    # Particles should be drawn from N(0, P0) where P0 = solve_lyapunov(G1, impact)
    # Check that sample covariance is roughly correct
    P0 = solve_lyapunov(G1, impact)
    sample_cov = ws.particles * ws.particles' / N

    # With N=1000, sample covariance should be within ~10% of theoretical
    @test norm(sample_cov - P0) / norm(P0) < 0.3  # generous tolerance

    # Weights should be uniform
    @test all(ws.weights .≈ 1.0 / N)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: SMC & MH Samplers
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adaptive tempering bisection" begin
    N = 100
    log_liks = randn(Random.MersenneTwister(123), N)
    log_weights = fill(-log(N), N)          # uniform incoming weights (5-arg signature, #133)
    phi_old = 0.0
    ess_target = 0.5

    phi_new = MacroEconometricModels._adaptive_tempering(log_liks, log_weights, phi_old, ess_target, N)
    @test phi_new > phi_old
    @test phi_new <= 1.0

    # ESS of the cumulative weights at this phi (reduces to incremental under uniform incoming)
    lw = log_weights .+ (phi_new - phi_old) .* log_liks
    w = exp.(lw .- MacroEconometricModels._logsumexp(lw))
    w ./= sum(w)
    ess = 1.0 / sum(abs2, w)
    @test abs(ess - ess_target * N) < 5.0
end

@testset "Log prior evaluation" begin
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(2.0, 0.5))
    prior = MacroEconometricModels.DSGEPrior(priors;
        lower=Dict(:ρ => 0.0, :σ => 0.0),
        upper=Dict(:ρ => 1.0))

    # Valid parameter vector (sorted order: ρ, σ)
    θ_valid = [0.5, 0.3]
    lp = MacroEconometricModels._log_prior(θ_valid, prior)
    @test isfinite(lp)
    @test lp ≈ logpdf(Beta(2, 2), 0.5) + logpdf(InverseGamma(2.0, 0.5), 0.3)

    # Out of bounds
    θ_oob = [-0.1, 0.3]
    lp_oob = MacroEconometricModels._log_prior(θ_oob, prior)
    @test lp_oob == -Inf

    θ_oob2 = [0.5, -0.1]
    lp_oob2 = MacroEconometricModels._log_prior(θ_oob2, prior)
    @test lp_oob2 == -Inf
end

@testset "Build likelihood function" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:gensys)
    data_mat = simulate(sol, 100; rng=Random.MersenneTwister(42))'  # n_obs × T

    ll_fn = MacroEconometricModels._build_likelihood_fn(spec, [:ρ], data_mat,
        [:y], nothing, :gensys, NamedTuple())

    # Should return finite log-likelihood at valid parameter
    ll = ll_fn([0.8])
    @test isfinite(ll)
    @test ll < 0.0

    # Should return -Inf for explosive parameter
    ll_bad = ll_fn([1.5])
    @test ll_bad == -Inf
    end
end

@testset "Likelihood closure: narrow catch + failure counting (T041)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    good = simulate(sol, 100; rng=Random.MersenneTwister(42))'   # 1×100

    # (a) A genuine bug (dimension mismatch: 2-row data vs 1 observable) PROPAGATES
    #     rather than being swallowed to -Inf.
    bad = vcat(good, good)   # 2×100
    ll_fn_bad = MacroEconometricModels._build_likelihood_fn(spec, [:ρ], bad,
        [:y], nothing, :gensys, NamedTuple())
    @test_throws DimensionMismatch ll_fn_bad([0.5])

    # (b) A legitimate indeterminacy is caught, returns -Inf, and is COUNTED.
    fails = Threads.Atomic{Int}(0); evals = Threads.Atomic{Int}(0)
    ll_fn = MacroEconometricModels._build_likelihood_fn(spec, [:ρ], good,
        [:y], nothing, :gensys, NamedTuple(); failures=fails, evals=evals)
    @test ll_fn([1.5]) == -Inf          # explosive ⇒ !is_determined
    @test fails[] == 1 && evals[] == 1
    @test isfinite(ll_fn([0.6]))
    @test fails[] == 1 && evals[] == 2
    end
end

@testset "SMC records and surfaces the likelihood failure count (T041)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    data = simulate(sol, 200; rng=Random.MersenneTwister(7))'
    # Uniform(0,1.4) prior ⇒ ~29% of particles are explosive (ρ>1) ⇒ many failed evals.
    priors = Dict(:ρ => Uniform(0.0, 1.4))
    result = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=200, n_mh_steps=1, rng=Random.MersenneTwister(123))
    @test result.n_lik_evals > 0
    @test result.n_failed_draws > 0
    @test occursin("Failed lik. evals", sprint(show, result))
    end
end

@testset "posterior_predictive drops failed draws, no zero-fill (T041)" begin
    # NOTE: not wrapped in _suppress_warnings — @test_warn must see the "dropped" @warn.
    spec = _suppress_warnings() do
        s = @dsge begin
            parameters: ρ = 0.5, σ = 0.5
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
            steady_state = [0.0]
        end
        compute_steady_state(s)
    end
    sol, ss = MacroEconometricModels._build_solution_at_theta(spec, [:ρ], [0.5],
        [:y], nothing, :gensys, NamedTuple())
    # Half the draws are explosive (ρ=1.5, indeterminate ⇒ dropped), half valid (ρ=0.5).
    n = 40
    td = reshape(vcat(fill(0.5, 20), fill(1.5, 20)), 40, 1)
    prior = MacroEconometricModels.DSGEPrior{Float64}([:ρ], [Uniform(0.0, 2.0)], [0.0], [2.0])
    post = BayesianDSGE{Float64}(td, zeros(n), [:ρ], prior, 0.0, :smc, 0.5,
        Float64[], Float64[], spec, sol, ss)   # 12-arg compat ctor
    Y = @test_warn r"dropped" posterior_predictive(post, 30; T_periods=25,
        rng=Random.MersenneTwister(9))
    @test size(Y, 1) < 30                                    # dropped, not zero-filled to 30
    @test all(any(!iszero, Y[s, :, :]) for s in 1:size(Y, 1))   # no zero path survived
end

@testset "SMC with Kalman: AR(1) recovery" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 200; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))
    result = MacroEconometricModels._smc_sample(spec, data, [:ρ],
        MacroEconometricModels.DSGEPrior(priors; lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99)),
        [0.5];
        n_smc=200, n_mh_steps=1, ess_target=0.5,
        observables=[:y], measurement_error=nothing,
        solver=:gensys, solver_kwargs=NamedTuple(),
        rng=Random.MersenneTwister(123))

    @test length(result.phi_schedule) > 1
    @test result.phi_schedule[end] ≈ 1.0
    @test isfinite(result.log_marginal_likelihood)

    # Posterior mean should be close to true value
    w = exp.(result.log_weights .- MacroEconometricModels._logsumexp(result.log_weights))
    post_mean = sum(result.theta_particles[1, :] .* w)
    @test abs(post_mean - 0.8) < 0.3
    end
end

@testset "SMC prior init errors on exhausted retries (T051 #150)" begin
    _suppress_warnings() do
        spec = @dsge begin
            parameters: ρ = 0.5, σ = 0.5
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
            steady_state = [0.0]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        data = simulate(sol, 50; rng=Random.MersenneTwister(7))'  # n_obs × T

        # Prior support (Normal(5.0, 0.01)) lies entirely OUTSIDE the [0.01, 0.99] bounds →
        # every draw is rejected → the initializer must fail loudly, not substitute a midpoint.
        bad_prior = MacroEconometricModels.DSGEPrior(
            Dict(:ρ => Normal(5.0, 0.01));
            lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99))

        err = try
            MacroEconometricModels._smc_sample(spec, data, [:ρ], bad_prior, [0.5];
                n_smc=4, observables=[:y], solver=:gensys,
                rng=Random.MersenneTwister(123))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("ρ", msg)     # names the offending parameter
        @test occursin("100", msg)   # names the rejection count
    end
end

@testset "SMC tempering: max_stages guard (#145 T046)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 200; rng=Random.MersenneTwister(42))'
    prior = MacroEconometricModels.DSGEPrior(Dict(:ρ => Beta(2, 2));
        lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99))
    # An informative 200-obs AR(1) SMC needs many tempering stages; capping at 2 must raise.
    @test_throws ErrorException MacroEconometricModels._smc_sample(
        spec, data, [:ρ], prior, [0.5];
        n_smc=100, n_mh_steps=1, ess_target=0.5, observables=[:y],
        solver=:gensys, max_stages=2, rng=Random.MersenneTwister(123))
    end
end

@testset "_check_tempering_progress: min_dphi + max_stages guards (#145 T046)" begin
    # A degenerate step Δφ < min_dphi while φ < 1 raises (stall).
    @test_throws ErrorException MacroEconometricModels._check_tempering_progress(
        3, 500, 0.5, 0.5 + 1e-9, 1e-6)
    # Exceeding the stage cap raises.
    @test_throws ErrorException MacroEconometricModels._check_tempering_progress(
        501, 500, 0.9, 0.95, 1e-6)
    # A legitimate final jump to exactly φ=1 is never flagged, even if the step is small.
    @test MacroEconometricModels._check_tempering_progress(10, 500, 0.9999, 1.0, 1e-6) === nothing
    # Normal within-schedule progress does not raise.
    @test MacroEconometricModels._check_tempering_progress(5, 500, 0.3, 0.5, 1e-6) === nothing
end

@testset "SMC² tempering: max_stages guard (#145 T046)" begin
    FAST && return
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 100; rng=Random.MersenneTwister(42))'
    prior = MacroEconometricModels.DSGEPrior(Dict(:ρ => Beta(2, 2));
        lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99))
    @test_throws ErrorException MacroEconometricModels._smc2_sample(
        spec, data, [:ρ], prior, [0.5];
        n_smc=40, n_particles=20, n_mh_steps=1, ess_target=0.5,
        observables=[:y], measurement_error=[0.5], solver=:gensys,
        max_stages=2, rng=Random.MersenneTwister(5))
    end
end

@testset "Adaptive RWMH: AR(1) recovery" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 200; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))
    draws, log_post, acc_rate = MacroEconometricModels._mh_sample(spec, data, [:ρ],
        MacroEconometricModels.DSGEPrior(priors; lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99)),
        [0.5];
        n_draws=2000, burnin=500, adapt_interval=100,
        observables=[:y], measurement_error=nothing,
        solver=:gensys, solver_kwargs=NamedTuple(),
        rng=Random.MersenneTwister(123))

    @test size(draws, 1) == 2000
    @test size(draws, 2) == 1
    @test length(log_post) == 2000
    @test 0.0 < acc_rate < 1.0

    # Posterior mean (after burnin) should be near truth
    post_draws = draws[501:end, 1]
    @test abs(mean(post_draws) - 0.8) < 0.3
    end
end

@testset "RWMH freezes proposal after burn-in and windows the acceptance signal (T038)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 200; rng=Random.MersenneTwister(42))'

    prior = MacroEconometricModels.DSGEPrior(Dict(:ρ => Beta(2, 2));
        lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99))

    # burnin=300 ≥ 2*adapt_interval ⇒ adaptation fires at 100,150,…,300 then freezes;
    # n_draws=1500 ≫ burnin so an un-frozen sampler would keep moving the proposal.
    draws, log_post, acc_rate, diag = MacroEconometricModels._mh_sample(
        spec, data, [:ρ], prior, [0.5];
        n_draws=1500, burnin=300, adapt_interval=50,
        observables=[:y], measurement_error=nothing,
        solver=:gensys, solver_kwargs=NamedTuple(),
        rng=Random.MersenneTwister(123))

    # 1. FREEZE: proposal at end of burn-in == proposal at end of run.
    @test diag.proposal_L_at_burnin ≈ diag.proposal_L
    @test diag.scale_at_burnin == diag.scale_factor

    # 2. Adaptation actually occurred (so the freeze is meaningful): the burn-in proposal
    #    has moved away from the initial c2·I.
    init_L = cholesky(Hermitian((2.38^2) * Matrix{Float64}(I, 1, 1))).L
    @test !isapprox(diag.proposal_L_at_burnin, init_L)

    # 3. WINDOWED SIGNAL: recorded, in [0,1], and NOT equal to the cumulative signal.
    @test !isempty(diag.window_acc_history)
    @test all(0.0 .<= diag.window_acc_history .<= 1.0)
    @test diag.window_acc_history != diag.cum_acc_history

    # 4. Recovery sanity.
    @test 0.0 < acc_rate < 1.0
    @test abs(mean(draws[301:end, 1]) - 0.8) < 0.3
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: SMC² (Nested PF inside SMC)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adaptive N_x in SMC²" begin
    N_x = 50
    var_ll = 100.0  # high variance → should double
    threshold = 10.0
    new_N_x = MacroEconometricModels._adapt_n_particles(N_x, var_ll, threshold)
    @test new_N_x == 100  # doubled

    var_ll_low = 1.0
    new_N_x2 = MacroEconometricModels._adapt_n_particles(N_x, var_ll_low, threshold)
    @test new_N_x2 == N_x
end

@testset "PF likelihood function" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    data_mat = simulate(sol, 50; rng=Random.MersenneTwister(42))'

    # Bootstrap PF needs nonzero measurement error to weight particles (T042 default is
    # zero ME); [0.01] reproduces the former 1e-4·I default (0.01² = 1e-4).
    pf_ll_fn = MacroEconometricModels._build_pf_likelihood_fn(spec, [:ρ], data_mat,
        [:y], [0.01], :gensys, NamedTuple(), 100)

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, 1, 1, 1, 100; T_obs=50)
    rng = Random.MersenneTwister(123)

    ll = pf_ll_fn([0.8], ws, rng)
    @test isfinite(ll)
    @test ll < 0.0

    # Explosive parameter should give -Inf
    ll_bad = pf_ll_fn([1.5], ws, rng)
    @test ll_bad == -Inf
    end
end

@testset "SMC² with particle filter: AR(1)" begin
    FAST && return
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 100; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))
    result = MacroEconometricModels._smc2_sample(spec, data, [:ρ],
        MacroEconometricModels.DSGEPrior(priors; lower=Dict(:ρ => 0.01), upper=Dict(:ρ => 0.99)),
        [0.5];
        n_smc=50, n_particles=50, n_mh_steps=1, ess_target=0.5,
        observables=[:y], measurement_error=nothing,
        solver=:gensys, solver_kwargs=NamedTuple(),
        rng=Random.MersenneTwister(123))

    @test result.phi_schedule[end] ≈ 1.0
    @test isfinite(result.log_marginal_likelihood)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Public API — estimate_dsge_bayes
# ─────────────────────────────────────────────────────────────────────────────

@testset "estimate_dsge_bayes: SMC + Kalman" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    sim_data = simulate(sol_true, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=200, n_mh_steps=1,
        rng=Random.MersenneTwister(123))

    @test result isa BayesianDSGE{Float64}
    @test result.method == :smc
    @test size(result.theta_draws, 2) == 1
    @test length(result.param_names) == 1
    @test result.param_names[1] == :ρ
    @test isfinite(result.log_marginal_likelihood)
    @test 0.0 < result.acceptance_rate <= 1.0
    @test result.phi_schedule[end] ≈ 1.0
    end
end

@testset "estimate_dsge_bayes: MH" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:mh, observables=[:y],
        n_draws=1000, burnin=200,
        rng=Random.MersenneTwister(123))

    @test result isa BayesianDSGE{Float64}
    @test result.method == :rwmh
    # Burn-in is discarded (E-03 / #122): stored posterior draws = n_draws - burnin, with
    # log_posterior sliced consistently. Pre-fix this returned all 1000 draws (burnin no-op).
    @test size(result.theta_draws, 1) == 1000 - 200
    @test length(result.log_posterior) == 1000 - 200

    # keep_burnin=true retains the full chain (smaller run to stay fast)
    result_full = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:mh, observables=[:y],
        n_draws=300, burnin=100, keep_burnin=true,
        rng=Random.MersenneTwister(7))
    @test size(result_full.theta_draws, 1) == 300
    end
end

@testset "estimate_dsge_bayes: auto data transpose" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 100; rng=rng)  # Returns T × 1

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))
    @test result isa BayesianDSGE{Float64}
    end
end

@testset "estimate_dsge_bayes: invalid method" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sim_data = randn(1, 100)
    priors = Dict(:ρ => Beta(2, 2))
    @test_throws ArgumentError estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:invalid, observables=[:y])
    end
end

# ── E-12 / H-12 / #136: theta0 as Dict/NamedTuple + length validation ──

@testset "_resolve_theta0: order-independent Dict/NamedTuple + length validation" begin
    _rt = MacroEconometricModels._resolve_theta0
    pnames = [:alpha, :rho, :sigma]                        # sorted prior keys

    # Dict / NamedTuple in scrambled order → values land on the RIGHT parameters.
    @test _rt(Dict(:sigma => 0.3, :alpha => 0.1, :rho => 0.9), pnames, Float64) == [0.1, 0.9, 0.3]
    @test _rt((sigma = 0.3, alpha = 0.1, rho = 0.9), pnames, Float64) == [0.1, 0.9, 0.3]
    # Positional vector (must already be in sorted order) passes through.
    @test _rt([0.1, 0.9, 0.3], pnames, Float64) == [0.1, 0.9, 0.3]
    # Type conversion.
    @test _rt(Dict(:alpha => 0.1, :rho => 0.9, :sigma => 0.3), pnames, Float32) isa Vector{Float32}
    # Wrong-length positional vector → informative ArgumentError (was a late opaque failure).
    @test_throws ArgumentError _rt([0.1, 0.9], pnames, Float64)
    # Dict missing a parameter → ArgumentError.
    @test_throws ArgumentError _rt(Dict(:alpha => 0.1, :rho => 0.9), pnames, Float64)
    # Dict with an unknown parameter → ArgumentError.
    @test_throws ArgumentError _rt(
        Dict(:alpha => 0.1, :rho => 0.9, :sigma => 0.3, :bogus => 0.0), pnames, Float64)
end

@testset "estimate_dsge_bayes: accepts Dict theta0, errors on wrong length (#136)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 100; rng=Random.MersenneTwister(42))
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(3.0, 1.0))

    # Dict theta0 in scrambled order runs end-to-end (order-independent).
    r = estimate_dsge_bayes(spec, sim_data, Dict(:σ => 0.5, :ρ => 0.5);
        priors=priors, method=:smc, observables=[:y], n_smc=100,
        rng=Random.MersenneTwister(1))
    @test r isa BayesianDSGE{Float64}
    # Wrong-length positional vector errors before sampling.
    @test_throws ArgumentError estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y], n_smc=100,
        rng=Random.MersenneTwister(1))
    end
end

# ── E-18 / #142: data orientation resolved by matching n_obs (T×n convention) ──

@testset "_orient_data: orientation by n_obs, not by size comparison" begin
    _od = MacroEconometricModels._orient_data
    @test size(_od(reshape(collect(1.0:20), 10, 2), 2, Float64)) == (2, 10)   # T×n → n_obs×T_obs
    @test size(_od(reshape(collect(1.0:20), 2, 10), 2, Float64)) == (2, 10)    # n×T → as-is
    A = randn(Random.MersenneTwister(1), 40, 3)
    @test _od(A, 3, Float64) == _od(permutedims(A), 3, Float64)                 # same internal matrix
    # Neither dimension equals n_obs → informative ArgumentError (was a silent best-guess).
    @test_throws ArgumentError _od(randn(3, 100), 1, Float64)
end

@testset "estimate_dsge_bayes: T×n and n×T give the same likelihood (#142)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sim = simulate(solve(spec; method=:gensys), 100; rng=Random.MersenneTwister(42))  # 100×1 (T×n)
    priors = Dict(:ρ => Beta(2, 2))

    r_tn = estimate_dsge_bayes(spec, sim, [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=100, rng=Random.MersenneTwister(5))
    r_nt = estimate_dsge_bayes(spec, permutedims(sim), [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=100, rng=Random.MersenneTwister(5))
    @test r_tn.log_marginal_likelihood ≈ r_nt.log_marginal_likelihood
    # A shape where neither dimension equals n_obs errors instead of guessing.
    @test_throws ArgumentError estimate_dsge_bayes(spec, randn(3, 100), [0.5];
        priors=priors, method=:smc, observables=[:y], n_smc=50)
    end
end

# ── #136 / #142 parity: posterior_mode & posterior_predictive_check route through
#    the shared _resolve_theta0 / _orient_data helpers (previously positional-only
#    theta0 + the best-guess _orient_bayes_data heuristic) ──

@testset "posterior_mode: Dict/NamedTuple theta0 + n_obs orientation (#136/#142)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data = simulate(solve(spec; method=:gensys), 120; rng=Random.MersenneTwister(7))  # 120×1 (T×n)
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(3.0, 1.0))

    pm_vec  = posterior_mode(spec, data, [0.5, 0.5]; priors=priors, observables=[:y])
    # Scrambled Dict / NamedTuple land on the right parameters → identical mode.
    pm_dict = posterior_mode(spec, data, Dict(:σ => 0.5, :ρ => 0.5); priors=priors, observables=[:y])
    pm_nt   = posterior_mode(spec, data, (σ = 0.5, ρ = 0.5); priors=priors, observables=[:y])
    @test pm_dict.mode ≈ pm_vec.mode
    @test pm_nt.mode ≈ pm_vec.mode
    # n×T data (row count == n_obs) accepted and gives the same mode as T×n.
    pm_nxt = posterior_mode(spec, permutedims(data), [0.5, 0.5]; priors=priors, observables=[:y])
    @test pm_nxt.mode ≈ pm_vec.mode

    # Wrong-length vector, missing/unknown Dict key, and neither-dim-matches shape all error.
    @test_throws ArgumentError posterior_mode(spec, data, [0.5]; priors=priors, observables=[:y])
    @test_throws ArgumentError posterior_mode(spec, data, Dict(:ρ => 0.5); priors=priors, observables=[:y])
    @test_throws ArgumentError posterior_mode(spec, randn(3, 120), [0.5, 0.5];
                                              priors=priors, observables=[:y])
    end
end

@testset "posterior_predictive_check: T×n and n×T data agree, bad shape errors (#142)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data = simulate(solve(spec; method=:gensys), 120; rng=Random.MersenneTwister(7))
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(3.0, 1.0))
    fit = estimate_dsge_bayes(spec, data, [0.5, 0.5]; priors=priors, method=:smc,
                              observables=[:y], n_smc=80, rng=Random.MersenneTwister(3))
    ppc_tn = posterior_predictive_check(fit; data=data, n_draws=40, rng=Random.MersenneTwister(1))
    ppc_nt = posterior_predictive_check(fit; data=permutedims(data), n_draws=40,
                                        rng=Random.MersenneTwister(1))
    @test ppc_tn.p_values ≈ ppc_nt.p_values
    @test_throws ArgumentError posterior_predictive_check(fit; data=randn(3, 120), n_draws=10)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Posterior Analysis
# ─────────────────────────────────────────────────────────────────────────────

@testset "posterior_summary" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    ps = posterior_summary(result)
    @test haskey(ps, :ρ)
    @test haskey(ps[:ρ], :mean)
    @test haskey(ps[:ρ], :median)
    @test haskey(ps[:ρ], :std)
    @test haskey(ps[:ρ], :ci_lower)
    @test haskey(ps[:ρ], :ci_upper)
    @test ps[:ρ][:ci_lower] < ps[:ρ][:mean] < ps[:ρ][:ci_upper]
    end
end

@testset "posterior_summary: quantiles equal Statistics.quantile (#144)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sim_data = simulate(solve(spec; method=:gensys), 200; rng=rng)
    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=300, rng=Random.MersenneTwister(1))

    ps = posterior_summary(result)
    d = result.theta_draws[:, 1]
    # Reported median/CI bounds are interpolated Statistics.quantile of the (unweighted,
    # post-terminal-resample) draws — the old order-statistic indexing did not match.
    @test ps[:ρ][:median]   ≈ quantile(d, 0.5)
    @test ps[:ρ][:ci_lower] ≈ quantile(d, 0.025)
    @test ps[:ρ][:ci_upper] ≈ quantile(d, 0.975)
    end
end

@testset "marginal_likelihood" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 100; rng=Random.MersenneTwister(42))

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    ml = marginal_likelihood(result)
    @test isfinite(ml)
    @test ml < 0
    end
end

@testset "bayes_factor" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 100; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    r1 = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=100, rng=Random.MersenneTwister(1))
    r2 = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=100, rng=Random.MersenneTwister(2))

    bf = bayes_factor(r1, r2)
    @test isfinite(bf)
    end
end

# ── E-04 / #130: RWMH marginal likelihood = Geweke (1999) modified harmonic mean ──

@testset "_geweke_mhm: recovers analytic Gaussian marginal likelihood" begin
    # If the posterior is exactly N(μ, Σ) and the kernel is c·N(θ;μ,Σ), then the
    # marginal likelihood (integral of the kernel) equals c, so log ML = log c.
    # The MHM estimator applied to draws from N(μ,Σ) must recover log c — this is
    # the precise, deterministic check that the estimator (not just "some finite
    # number") is correct.
    rng = Random.MersenneTwister(20260708)
    μ = [0.5, -0.3]
    Σ = [0.04 0.01; 0.01 0.09]
    mvn = MvNormal(μ, Σ)
    S = 20_000
    draws = Matrix(rand(rng, mvn, S)')          # S×d
    log_c = -50.0
    kernel = [log_c + logpdf(mvn, draws[s, :]) for s in 1:S]

    est = MacroEconometricModels._geweke_mhm(draws, kernel)
    @test isfinite(est)
    @test isapprox(est, log_c; atol=0.3)         # observed ≈ -49.99

    # Truncation fraction p should not materially move the estimate (Geweke's point).
    est_p9 = MacroEconometricModels._geweke_mhm(draws, kernel; p=0.9)
    est_p1 = MacroEconometricModels._geweke_mhm(draws, kernel; p=0.1)
    @test isapprox(est_p9, log_c; atol=0.4)
    @test isapprox(est_p1, log_c; atol=0.4)
end

@testset "_geweke_mhm: short-chain / degenerate guards return NaN" begin
    _suppress_warnings() do
        # S < 10·d → NaN, not a silently wrong number.
        short = randn(Random.MersenneTwister(1), 5, 2)
        @test isnan(MacroEconometricModels._geweke_mhm(short, fill(-1.0, 5)))
        # No finite kernel value → NaN.
        big = randn(Random.MersenneTwister(2), 100, 1)
        @test isnan(MacroEconometricModels._geweke_mhm(big, fill(-Inf, 100)))
    end
end

@testset "marginal_likelihood: :mh MHM agrees with :smc (not max log posterior)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    sim_data = simulate(sol_true, 200; rng=Random.MersenneTwister(2024))

    priors = Dict(:ρ => Beta(2, 2))
    r_smc = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=400, rng=Random.MersenneTwister(11))
    r_mh = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:mh,
        observables=[:y], n_draws=6000, burnin=2000, rng=Random.MersenneTwister(12))

    ml_smc = marginal_likelihood(r_smc)
    ml_mh  = marginal_likelihood(r_mh)
    @test isfinite(ml_smc)
    @test isfinite(ml_mh)
    # Regression against the old behaviour: the estimator is a genuine evidence
    # estimate strictly below the max log posterior kernel (the peaked-posterior
    # Occam factor), NOT `maximum(log_posterior)` (≈ -128.4 here) as before.
    @test ml_mh < maximum(r_mh.log_posterior)
    # SMC (tempering path) and MHM estimate the same log marginal likelihood and
    # agree within a documented Monte Carlo tolerance: the observed gap is ≈ 0.5
    # (within SMC's own ≈ 0.5 cross-seed spread); the old max-log-posterior would
    # miss by ≈ 2.5 and fail this bound.
    @test isapprox(ml_smc, ml_mh; atol=2.0)
    end
end

# ── E-08 / #131: SMC marginal-likelihood increment under non-uniform weights ──

@testset "SMC log marginal likelihood: ratio increment (resample-invariant)" begin
    # The per-stage ML increment must be logsumexp(lw+inc)−logsumexp(lw), the
    # weighted-average tempering factor, NOT logsumexp(inc)−log N (valid only when
    # incoming weights are uniform). A low ess_target takes few, large tempering
    # steps that leave many stages *un*-resampled (non-uniform incoming weights),
    # while a high ess_target resamples almost every stage. Both configurations
    # estimate the *same* marginal likelihood, so their log-ML must agree within
    # Monte Carlo error. Pre-fix (uniform form) the two configs disagreed by ≈0.36;
    # the ratio form brings them to ≈0.05 (well inside MC noise).
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.7, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    sim_data = simulate(sol_true, 150; rng=Random.MersenneTwister(99))
    priors = Dict(:ρ => Beta(2, 2), :σ => Gamma(2, 0.5))

    run_cfg(ess) = [marginal_likelihood(estimate_dsge_bayes(spec, sim_data, [0.5, 1.0];
                        priors=priors, method=:smc, observables=[:y], n_smc=400,
                        ess_target=ess, rng=Random.MersenneTwister(sd)))
                    for sd in 1:6]

    ml_lo = run_cfg(0.50)   # few, large steps → non-resampled stages
    ml_hi = run_cfg(0.90)   # many, small steps → resamples ≈ every stage
    @test all(isfinite, ml_lo)
    @test all(isfinite, ml_hi)
    @test all(<(0), ml_lo)
    # Resample-invariance of the marginal-likelihood estimate. Documented MC
    # tolerance 0.3; the pre-fix uniform form biased ml_lo by ≈0.36 and fails this.
    @test isapprox(sum(ml_lo)/6, sum(ml_hi)/6; atol=0.3)
    end
end

# ── E-10 / #133: adaptive tempering targets ESS of the CUMULATIVE weights ──

@testset "_adaptive_tempering: ESS on cumulative weights (no overshoot)" begin
    lse = MacroEconometricModels._logsumexp
    N = 300
    rng = Random.MersenneTwister(7)
    log_liks = 1.5 .* randn(rng, N)
    log_weights = 0.7 .* randn(rng, N)      # non-uniform incoming cumulative weights
    phi_old = 0.3
    ess_target = 0.5

    ess_cum(phi) = (lw = log_weights .+ (phi - phi_old) .* log_liks;
                    w = exp.(lw .- lse(lw)); 1.0 / sum(abs2, w))
    ess_inc(phi) = (inc = (phi - phi_old) .* log_liks;
                    w = exp.(inc .- lse(inc)); 1.0 / sum(abs2, w))

    # Valid bracket: cumulative ESS starts above target and drops below it by φ=1.
    @test ess_cum(phi_old) > ess_target * N
    @test ess_cum(1.0) < ess_target * N

    phi_new = MacroEconometricModels._adaptive_tempering(
        log_liks, log_weights, phi_old, ess_target, N)
    @test phi_old < phi_new <= 1.0
    # Realized ESS on the CUMULATIVE weights hits the target (bisection tol ≈1 in ESS units).
    @test isapprox(ess_cum(phi_new), ess_target * N; atol=2.0)
    # The uniform-base (incremental-only) ESS at this φ is materially different — the old
    # computation would have chosen a different φ and overshot the cumulative ESS target.
    @test abs(ess_inc(phi_new) - ess_cum(phi_new)) > 5.0
end

# ── E-09 / #132: terminal resample → stored SMC draws are equal-weighted ──

@testset "_terminal_resample!: unweighted quantiles match weighted pre-resample set" begin
    lse = MacroEconometricModels._logsumexp
    np, Np = 2, 500
    rng = Random.MersenneTwister(11)
    theta = randn(rng, np, Np); theta[1, :] .+= 3.0
    lw0 = 1.2 .* randn(rng, Np)              # deliberately non-uniform terminal weights
    state = MacroEconometricModels.SMCState{Float64}(
        copy(theta), copy(lw0), randn(rng, Np), randn(rng, Np),
        Float64[0.0, 1.0], Float64[], Float64[], 0.0,
        MacroEconometricModels.PFWorkspace{Float64}[], Matrix{Float64}(I, np, np))

    # weighted median of param 1 on the pre-resample particle set
    w = exp.(lw0 .- lse(lw0))
    perm = sortperm(theta[1, :]); cw = cumsum(w[perm]); cw ./= cw[end]
    wq50_before = theta[1, perm][findfirst(>=(0.5), cw)]

    did = MacroEconometricModels._terminal_resample!(state, Np, Random.MersenneTwister(123))
    @test did                                                    # non-uniform → resampled
    weights_after = exp.(state.log_weights .- lse(state.log_weights))
    @test all(≈(1 / Np), weights_after)                          # stored weights uniform
    # plain (unweighted) median of resampled draws ≈ weighted median of the original set
    @test isapprox(quantile(state.theta_particles[1, :], 0.5), wq50_before; atol=0.3)

    # No-op when weights are already uniform (e.g. the final stage resampled) — no double resample.
    state_u = MacroEconometricModels.SMCState{Float64}(
        copy(theta), fill(-log(Float64(Np)), Np), zeros(Np), zeros(Np),
        Float64[0.0, 1.0], Float64[], Float64[], 0.0,
        MacroEconometricModels.PFWorkspace{Float64}[], Matrix{Float64}(I, np, np))
    @test MacroEconometricModels._terminal_resample!(state_u, Np, Random.MersenneTwister(1)) == false
end

# ── E-06 / #134: SMC² PMMH mutation (chunked workspaces, unconditional PF) ──

@testset "_chunk_ranges: contiguous non-overlapping partition of 1:N" begin
    _cr = MacroEconometricModels._chunk_ranges
    # Even/uneven splits cover exactly 1:N with no overlap, no threadid() aliasing.
    for (N, k) in ((100, 4), (101, 4), (7, 3), (10, 1), (5, 5))
        rgs = _cr(N, k)
        @test length(rgs) == k
        @test reduce(vcat, collect.(rgs)) == collect(1:N)   # covers 1:N in order, once each
        @test maximum(length.(rgs)) - minimum(length.(rgs)) <= 1   # near-equal
    end
    # N < k → trailing chunks are empty (no phantom particles)
    rgs = _cr(3, 8)
    @test length(rgs) == 8
    @test count(!isempty, rgs) == 3
    @test reduce(vcat, collect.(rgs)) == collect(1:3)
end

@testset "SMC² PMMH recovers the Kalman posterior (linear model)" begin
    # On a linear model the bootstrap PF is a noisy estimator of the exact Kalman
    # likelihood, so the redesigned PMMH SMC² kernel must recover the same posterior
    # as :smc (exact Kalman). Verifies the mutation targets the correct distribution.
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.7, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    sim_data = simulate(sol_true, 120; rng=Random.MersenneTwister(2026))
    priors = Dict(:ρ => Beta(2, 2))
    merr = [0.1]

    r_smc = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:smc,
        observables=[:y], n_smc=300, measurement_error=merr,
        rng=Random.MersenneTwister(1))
    r_smc2 = estimate_dsge_bayes(spec, sim_data, [0.5]; priors=priors, method=:smc2,
        observables=[:y], n_smc=200, n_particles=200, n_mh_steps=2,
        measurement_error=merr, solver=:gensys,
        rng=Random.MersenneTwister(101))

    ρ_smc = mean(r_smc.theta_draws[:, 1])
    ρ_smc2 = mean(r_smc2.theta_draws[:, 1])
    # Both recover the true ρ region (data posterior ≈ 0.69, well away from prior mean 0.5).
    @test ρ_smc2 > 0.6
    # SMC² PMMH agrees with the exact-Kalman SMC reference within MC error (observed ≈ 0.002).
    @test isapprox(ρ_smc2, ρ_smc; atol=0.05)
    end
end

# ── E-11 / #135: SMC² N_x adaptation on estimator variance + exchange step ──

@testset "_pf_estimator_variance: estimator noise, not posterior spread (#135)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.7, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    sim = simulate(sol_true, 80; rng=Random.MersenneTwister(7))
    data = Matrix(reshape(sim[:, 1], 1, :))          # n_obs × T_obs
    T_obs = size(data, 2)
    merr = [0.4]
    n_states = spec.n_endog; n_shocks = spec.n_exog
    mkpool(N_x, k=1) = [MacroEconometricModels._allocate_pf_workspace(
        Float64, n_states, 1, n_shocks, N_x; T_obs=T_obs) for _ in 1:k]

    # A DIFFUSE θ-population: ρ ranges widely, so the likelihood level varies a lot
    # across particles ⇒ var(ll across θ) is large (this is what the OLD trigger fired on).
    diffuse = reshape(collect(range(0.2, 0.9; length=40)), 1, 40)
    rng = Random.MersenneTwister(123)

    est400 = MacroEconometricModels._pf_estimator_variance(
        spec, [:ρ], diffuse, [:y], merr, :gensys, NamedTuple(), mkpool(400),
        data, T_obs, rng; n_probe=10, n_rep=3)
    est50 = MacroEconometricModels._pf_estimator_variance(
        spec, [:ρ], diffuse, [:y], merr, :gensys, NamedTuple(), mkpool(50),
        data, T_obs, rng; n_probe=10, n_rep=3)

    # var(ll across θ) — the OLD (wrong) trigger quantity — on the same diffuse set.
    ws = mkpool(400)[1]; lls = Float64[]
    for i in 1:size(diffuse, 2)
        rr = Random.MersenneTwister(hash((:av, i)))
        ll = MacroEconometricModels._solve_and_run_pf(
            spec, [:ρ], diffuse[:, i], [:y], merr, :gensys, NamedTuple(),
            ws, data, T_obs, rr)
        isfinite(ll) && push!(lls, ll)
    end
    var_across = var(lls)

    @test var_across > 10                 # OLD trigger (threshold 10) would double N_x here
    @test est400 < 3                      # NEW estimator-variance trigger does NOT (Chopin ≈1–3)
    @test est50 > est400                  # estimator variance falls with more inner particles
    # Decision: on this diffuse-but-well-estimated posterior, N_x stays put (no spurious doubling).
    @test MacroEconometricModels._adapt_n_particles(400, est400, 3.0) == 400
    end
end

@testset "_exchange_step!: recomputes all θ-likelihoods at the new N_x (#135)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.7, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sim = simulate(solve(true_spec; method=:gensys), 60; rng=Random.MersenneTwister(3))
    data = Matrix(reshape(sim[:, 1], 1, :)); T_obs = size(data, 2)
    merr = [0.4]; N = 20; N_x = 200
    n_states = spec.n_endog; n_shocks = spec.n_exog
    npool = max(Threads.nthreads(), 1)
    pool = [MacroEconometricModels._allocate_pf_workspace(
        Float64, n_states, 1, n_shocks, N_x; T_obs=T_obs) for _ in 1:npool]

    thetas = reshape(collect(range(0.55, 0.85; length=N)), 1, N)
    SENTINEL = -1.0e5
    state = MacroEconometricModels.SMCState{Float64}(
        copy(thetas), fill(-log(Float64(N)), N),
        fill(SENTINEL, N), zeros(N),          # stale (old-N_x) sentinel likelihoods
        Float64[0.0, 1.0], Float64[], Float64[], 0.0,
        MacroEconometricModels.PFWorkspace{Float64}[], Matrix{Float64}(I, 1, 1))

    MacroEconometricModels._exchange_step!(
        state, spec, [:ρ], [:y], merr, :gensys, NamedTuple(),
        pool, data, T_obs, Random.MersenneTwister(99))

    @test all(isfinite, state.log_likelihoods)          # all recomputed to finite values
    @test all(state.log_likelihoods .!= SENTINEL)       # no stale (old-N_x) estimate remains
    end
end

@testset "SMC² init likelihoods: threaded chunks == serial (#146 #147)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.7, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sim = simulate(solve(true_spec; method=:gensys), 60; rng=Random.MersenneTwister(3))
    data = Matrix(reshape(sim[:, 1], 1, :)); T_obs = size(data, 2)
    merr = [0.4]; N = 12; N_x = 200
    n_states = spec.n_endog; n_shocks = spec.n_exog
    npool = max(Threads.nthreads(), 1)

    pf_ll_fn = MacroEconometricModels._build_pf_likelihood_fn(spec, [:ρ], data, [:y],
        merr, :gensys, NamedTuple(), N_x)
    thetas = reshape(collect(range(0.55, 0.85; length=N)), 1, N)
    # Fixed per-particle seeds ⇒ threaded and serial must agree exactly.
    seeds = UInt64[hash((j, UInt64(0x5EED))) for j in 1:N]

    # THREADED: chunked helper with one PF workspace per chunk.
    pool = [MacroEconometricModels._allocate_pf_workspace(
        Float64, n_states, 1, n_shocks, N_x; T_obs=T_obs) for _ in 1:npool]
    ll_thr = fill(-Inf, N); sols_thr = Vector{Any}(undef, N); fill!(sols_thr, nothing)
    MacroEconometricModels._smc2_init_likelihoods!(ll_thr, sols_thr, thetas, spec, [:ρ],
        [:y], merr, :gensys, NamedTuple(), pf_ll_fn, pool, data, T_obs, seeds)

    # SERIAL reference: single workspace, same per-particle seeds and closure.
    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, 1, n_shocks, N_x; T_obs=T_obs)
    ll_ser = fill(-Inf, N)
    for j in 1:N
        rr = Random.MersenneTwister(seeds[j])
        ll_ser[j] = pf_ll_fn(Vector{Float64}(thetas[:, j]), ws, rr)
    end

    @test ll_thr == ll_ser              # bit-identical, thread-count-independent (#146/#147)
    @test all(isfinite, ll_thr)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Display, Report, Refs, Plot
# ─────────────────────────────────────────────────────────────────────────────

@testset "BayesianDSGE show" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test occursin("Bayesian DSGE Estimation", output)
    @test occursin("Method", output)
    @test occursin("Posterior Summary", output)
    @test occursin("Parameter", output)
    @test occursin("Mean", output)
    @test occursin("Std", output)
    @test occursin("Prior vs Posterior", output)
    end
end

@testset "BayesianDSGE report" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    # report() calls show(stdout, result); capture stdout
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("Bayesian DSGE Estimation", output)
    @test occursin("Log marginal lik.", output)
    end
end

@testset "BayesianDSGE refs" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    io = IOBuffer()
    refs(io, result)
    output = String(take!(io))
    @test occursin("Herbst", output)
    @test occursin("Schorfheide", output)
    end
end

@testset "plot_result(BayesianDSGE)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    p = plot_result(result)
    @test p isa PlotOutput
    @test occursin("d3", p.html)
    @test occursin("Prior", p.html)
    @test occursin("Posterior", p.html)
    @test occursin("Bayesian DSGE", p.html)
    end
end

@testset "StatsAPI methods for BayesianDSGE" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, sim_data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))

    c = StatsAPI.coef(result)
    @test length(c) == 1
    @test isfinite(c[1])
    @test c[1] ≈ mean(result.theta_draws[:, 1])

    @test StatsAPI.islinear(result) == false
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: TimeSeriesData Dispatch
# ─────────────────────────────────────────────────────────────────────────────

@testset "estimate_dsge_bayes with TimeSeriesData" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim = simulate(sol, 100; rng=rng)

    # Wrap in TimeSeriesData
    ts = TimeSeriesData(sim; varnames=["y"], frequency=Quarterly)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, ts, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=100, rng=Random.MersenneTwister(1))
    @test result isa BayesianDSGE{Float64}
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10: End-to-End Integration
# ─────────────────────────────────────────────────────────────────────────────

@testset "E2E: 2-variable model, SMC + Kalman" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    # Simple RBC-like model
    spec = @dsge begin
        parameters: ρ = 0.5, α = 0.3
        endogenous: y, k
        exogenous: ε
        y[t] = α * k[t-1] + ε[t]
        k[t] = ρ * k[t-1] + y[t]
        steady_state = [0.0, 0.0]
    end
    spec = compute_steady_state(spec)

    # True: ρ=0.7, α=0.3
    true_spec = @dsge begin
        parameters: ρ = 0.7, α = 0.3
        endogenous: y, k
        exogenous: ε
        y[t] = α * k[t-1] + ε[t]
        k[t] = ρ * k[t-1] + y[t]
        steady_state = [0.0, 0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 300; rng=rng)

    priors = Dict(
        :ρ => Beta(2, 2),
        :α => Normal(0.3, 0.1)
    )
    # 1 shock but 2 observables ⇒ n_obs>n_shocks: must supply explicit ME (T042).
    result = estimate_dsge_bayes(spec, data, [0.5, 0.3];
        priors=priors, method=:smc, observables=[:y, :k],
        measurement_error=[0.01, 0.01],
        n_smc=300, n_mh_steps=1,
        rng=Random.MersenneTwister(123))

    @test result isa BayesianDSGE{Float64}
    @test length(result.param_names) == 2

    ps = posterior_summary(result)
    # ρ should be recovered within tolerance
    @test abs(ps[:ρ][:mean] - 0.7) < 0.3

    # report should not error
    io = IOBuffer()
    show(io, result)
    s = String(take!(io))
    @test length(s) > 0

    # plot should not error
    p = plot_result(result)
    @test p isa PlotOutput
    end
end

@testset "E2E: MH baseline" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    data = simulate(sol, 200; rng=rng)

    priors = Dict(:ρ => Beta(2, 2))
    result = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, observables=[:y],
        n_draws=500, burnin=100,
        rng=Random.MersenneTwister(1))

    @test result.method == :mh || result.method == :rwmh
    @test size(result.theta_draws, 1) == 500 - 100   # burn-in discarded (E-03 / #122)
    @test 0.0 < result.acceptance_rate < 1.0
    end
end

# =============================================================================
# Nonlinear Particle Filter Tests
# =============================================================================

@testset "PFWorkspace allocation — augmented_buffer and kron_cross_buffer" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 30
    nv = n_states + n_shocks

    # Order 2: augmented_buffer allocated, kron_cross_buffer not
    ws2 = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                          nv=nv, order=2)
    @test ws2.augmented_buffer !== nothing
    @test size(ws2.augmented_buffer) == (nv, N)
    @test ws2.kron_cross_buffer === nothing

    # Order 3: both allocated
    ws3 = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                          nv=nv, order=3)
    @test ws3.augmented_buffer !== nothing
    @test size(ws3.augmented_buffer) == (nv, N)
    @test ws3.kron_cross_buffer !== nothing
    @test size(ws3.kron_cross_buffer) == (nv * nv, N)
end

@testset "PFWorkspace allocation — nx parameter" begin
    n_states = 5  # n_endog
    n_obs = 2
    n_shocks = 1
    N = 20
    nx = 2  # only 2 state variables out of 5 endogenous
    nv = nx + n_shocks

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                         nv=nv, nx=nx, order=2)
    @test size(ws.particles) == (n_states, N)       # full endogenous
    @test size(ws.particles_fo) == (nx, N)           # state variables only
    @test size(ws.particles_so) == (nx, N)
    @test ws.particles_to === nothing
    @test size(ws.augmented_buffer) == (nv, N)       # nx + n_shocks
    @test size(ws.kron_buffer) == (nv * nv, N)
end

@testset "PFWorkspace resize — augmented_buffer and kron_cross_buffer" begin
    n_states = 3
    n_obs = 2
    n_shocks = 1
    N = 30
    nv = n_states + n_shocks

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_states, n_obs, n_shocks, N;
                                                         nv=nv, order=3)
    N_new = 60
    MacroEconometricModels._resize_pf_workspace!(ws, N_new)

    @test size(ws.augmented_buffer) == (nv, N_new)
    @test size(ws.kron_cross_buffer) == (nv * nv, N_new)
    @test size(ws.particles_fo) == (n_states, N_new)
    @test size(ws.particles_so) == (n_states, N_new)
    @test size(ws.particles_to) == (n_states, N_new)
end

@testset "Nonlinear PF: _fill_kron_cross_buffer!" begin
    nv = 3
    N = 4
    V1 = Float64[1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
    V2 = Float64[0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8; 0.9 1.0 1.1 1.2]
    buffer = zeros(Float64, nv * nv, N)

    MacroEconometricModels._fill_kron_cross_buffer!(buffer, V1, V2, nv)

    # Check: buffer[(i-1)*nv+j, k] = V1[i,k] * V2[j,k]
    for k in 1:N, i in 1:nv, j in 1:nv
        idx = (i - 1) * nv + j
        @test buffer[idx, k] ≈ V1[i, k] * V2[j, k]
    end
end

@testset "Nonlinear PF: _pf_initialize_nonlinear!" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:perturbation, order=2)
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], nothing)
    nlss = MacroEconometricModels._build_nonlinear_state_space(sol, Z, d, H)

    nx = length(sol.state_indices)
    ny = length(sol.control_indices)
    n_endog = spec.n_endog
    n_eps = spec.n_exog
    nv = nx + n_eps
    N = 100

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, 1, n_eps, N;
                                                         nv=nv, nx=nx, order=2)
    MacroEconometricModels._pf_initialize_nonlinear!(ws, nlss; rng=rng)

    # particles_fo should have non-zero dispersion from Lyapunov
    @test any(ws.particles_fo .!= 0.0)
    # particles_so should be all zeros (steady state correction)
    @test all(ws.particles_so .== 0.0)
    # particles should have values at state indices
    for (si_idx, si) in enumerate(sol.state_indices)
        @test all(ws.particles[si, :] .== ws.particles_fo[si_idx, :])
    end
    # weights should be uniform
    @test all(ws.weights .≈ 1.0 / N)
    end
end

@testset "Nonlinear PF: _pf_transition_pruned! matches pruning.jl" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:perturbation, order=2)

    # Simulate with pruning.jl for reference
    rng_ref = Random.MersenneTwister(99)
    ref_data = simulate(sol, 5; rng=rng_ref)

    # Now replicate using the PF transition kernel on a single particle
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], nothing)
    nlss = MacroEconometricModels._build_nonlinear_state_space(sol, Z, d, H)

    nx = length(sol.state_indices)
    n_endog = spec.n_endog
    n_eps = spec.n_exog
    nv = nx + n_eps
    N = 1  # single particle for exact comparison

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, 1, n_eps, N;
                                                         nv=nv, nx=nx, order=2)
    # Initialize to zero state (matching pruning.jl initial conditions)
    fill!(ws.particles_fo, 0.0)
    fill!(ws.particles_so, 0.0)
    fill!(ws.particles, 0.0)

    # Replay same shocks as simulate()
    rng_replay = Random.MersenneTwister(99)
    T_periods = 5
    e = randn(rng_replay, T_periods, n_eps)

    for t in 1:T_periods
        # Set shocks in workspace
        ws.shocks[:, 1] .= e[t, :]
        MacroEconometricModels._pf_transition_pruned!(ws, nlss)

        # Check that particles match simulate() output (in deviations from SS)
        for (si_idx, si) in enumerate(sol.state_indices)
            ref_val = ref_data[t, si] - sol.steady_state[si]
            @test ws.particles[si, 1] ≈ ref_val atol=1e-10
        end
        for (ci_idx, ci) in enumerate(sol.control_indices)
            ref_val = ref_data[t, ci] - sol.steady_state[ci]
            @test ws.particles[ci, 1] ≈ ref_val atol=1e-10
        end
    end
    end
end

@testset "Nonlinear PF: _pf_resample_pruned!" begin
    nx = 2
    N = 5
    n_endog = 3
    n_obs = 1
    n_shocks = 1
    nv = nx + n_shocks

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, n_obs, n_shocks, N;
                                                         nv=nv, nx=nx, order=2)
    # Set up known particles_fo and particles_so
    ws.particles_fo .= reshape(1.0:Float64(nx * N), nx, N)
    ws.particles_so .= reshape(100.0:Float64(99 + nx * N), nx, N)

    # Ancestors: all pick particle 3
    ws.ancestors .= 3

    MacroEconometricModels._pf_resample_pruned!(ws, N, nx, 2)

    # After resampling, all columns of particles_fo and particles_so should equal column 3
    for k in 1:N
        @test ws.particles_fo[:, k] == ws.particles_fo[:, 1]  # all same
    end
    # Verify they came from the original column 3
    expected_fo = Float64[(3 - 1) * nx + i for i in 1:nx] .+ 1.0 .- 1.0
    # Original column 3: indices (2*nx+1):(3*nx) in the 1:nx*N range
    for i in 1:nx
        @test ws.particles_fo[i, 1] ≈ Float64((3 - 1) * nx + i)
    end
end

@testset "Nonlinear bootstrap PF: finite log-likelihood (order 2)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:perturbation, order=2)

    # Generate data
    data_sim = simulate(sol, 50; rng=Random.MersenneTwister(1))
    data = Matrix{Float64}(data_sim')  # n_obs x T_obs

    # Bootstrap PF needs nonzero measurement error (T042 default is zero ME);
    # [0.01] reproduces the former 1e-4·I default.
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], [0.01])
    nlss = MacroEconometricModels._build_nonlinear_state_space(sol, Z, d, H)

    nx = length(sol.state_indices)
    n_endog = spec.n_endog
    n_eps = spec.n_exog
    nv = nx + n_eps
    N = 200

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, 1, n_eps, N;
                                                         nv=nv, nx=nx, order=2)
    ll = MacroEconometricModels._bootstrap_particle_filter!(ws, nlss, data, 50;
                                                              rng=Random.MersenneTwister(42))

    @test isfinite(ll)
    @test ll > -1e6  # not absurdly large negative
    end
end

@testset "Nonlinear CSMC: finite log-likelihood (order 2)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:perturbation, order=2)

    T_obs = 30
    data_sim = simulate(sol, T_obs; rng=Random.MersenneTwister(1))
    data = Matrix{Float64}(data_sim')

    # Bootstrap PF/CSMC need nonzero measurement error (T042 default is zero ME).
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], [0.01])
    nlss = MacroEconometricModels._build_nonlinear_state_space(sol, Z, d, H)

    nx = length(sol.state_indices)
    n_endog = spec.n_endog
    n_eps = spec.n_exog
    nv = nx + n_eps
    N = 100

    ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, 1, n_eps, N;
                                                         nv=nv, nx=nx, order=2, T_obs=T_obs)
    # Initialize reference trajectory via a bootstrap PF run
    ll1 = MacroEconometricModels._bootstrap_particle_filter!(ws, nlss, data, T_obs;
                                                               store_trajectory=true,
                                                               rng=Random.MersenneTwister(10))
    @test isfinite(ll1)

    # Now run CSMC using the stored reference trajectory
    ll2 = MacroEconometricModels._conditional_smc!(ws, nlss, data, T_obs;
                                                      rng=Random.MersenneTwister(20))
    @test isfinite(ll2)
    @test ll2 > -1e6
    end
end

@testset "Nonlinear PF vs Kalman: order 1 comparison" begin
    _suppress_warnings() do
    # For a linear (order=1) model, the nonlinear PF should give similar results
    # to the exact Kalman filter (within Monte Carlo noise)
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.1
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Solve both ways
    sol_lin = solve(spec; method=:gensys)
    sol_pert = solve(spec; method=:perturbation, order=1)

    T_obs = 50
    data_sim = simulate(sol_lin, T_obs; rng=Random.MersenneTwister(1))
    data = Matrix{Float64}(data_sim')

    # Kalman log-likelihood. Bootstrap PF needs nonzero measurement error (T042 default
    # is zero ME); use the same H for both so the PF-vs-Kalman comparison is fair.
    Z, d, H = MacroEconometricModels._build_observation_equation(spec, [:y], [0.01])
    ss = MacroEconometricModels._build_state_space(sol_lin, Z, d, H)
    ll_kalman = MacroEconometricModels._kalman_loglikelihood(ss, data)

    # Nonlinear PF log-likelihood (should approximate Kalman)
    nlss = MacroEconometricModels._build_nonlinear_state_space(sol_pert, Z, d, H)
    nx = length(sol_pert.state_indices)
    n_endog = spec.n_endog
    n_eps = spec.n_exog
    nv = nx + n_eps
    N = 2000  # many particles for accuracy

    # Average over multiple runs for stability
    ll_pf_runs = Float64[]
    for seed in 1:5
        ws = MacroEconometricModels._allocate_pf_workspace(Float64, n_endog, 1, n_eps, N;
                                                             nv=nv, nx=nx, order=1)
        ll = MacroEconometricModels._bootstrap_particle_filter!(ws, nlss, data, T_obs;
                                                                   rng=Random.MersenneTwister(seed))
        push!(ll_pf_runs, ll)
    end
    ll_pf_mean = mean(ll_pf_runs)

    # PF should approximate Kalman — within 10% relative error
    @test isfinite(ll_kalman)
    @test isfinite(ll_pf_mean)
    rel_error = abs(ll_pf_mean - ll_kalman) / abs(ll_kalman)
    @test rel_error < 0.1
    end
end

@testset "Kalman guard: PerturbationSolution order >= 2 returns -Inf" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Generate data
    sol = solve(spec; method=:gensys)
    data_sim = simulate(sol, 20; rng=Random.MersenneTwister(1))
    data = Matrix{Float64}(data_sim')

    # Build Kalman likelihood function with perturbation solver (order=2)
    ll_fn = MacroEconometricModels._build_likelihood_fn(
        spec, [:ρ], data, [:y], nothing, :perturbation, (order=2,))
    ll = ll_fn([0.9])

    # Should return -Inf since Kalman can't handle order >= 2
    @test ll == -Inf
    end
end

@testset "_build_solution_at_theta dispatches on PerturbationSolution" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Linear solver → DSGEStateSpace
    sol_lin, ss_lin = MacroEconometricModels._build_solution_at_theta(
        spec, [:ρ], [0.9], [:y], nothing, :gensys, NamedTuple())
    @test sol_lin isa MacroEconometricModels.DSGESolution
    @test ss_lin isa MacroEconometricModels.DSGEStateSpace

    # Perturbation solver → NonlinearStateSpace
    sol_pert, ss_pert = MacroEconometricModels._build_solution_at_theta(
        spec, [:ρ], [0.9], [:y], nothing, :perturbation, (order=2,))
    @test sol_pert isa MacroEconometricModels.PerturbationSolution
    @test ss_pert isa MacroEconometricModels.NonlinearStateSpace
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Projection Particle Filter
# ─────────────────────────────────────────────────────────────────────────────

@testset "ProjectionStateSpace construction" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:projection, degree=3, scale=5.0)
    @test sol isa MacroEconometricModels.ProjectionSolution

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.01 * I, 1, 1)

    pss = MacroEconometricModels._build_projection_state_space(sol, Z, d, H)
    @test pss isa MacroEconometricModels.ProjectionStateSpace{Float64}
    @test size(pss.coefficients, 1) == 1  # n_vars
    @test size(pss.coefficients, 2) == size(pss.multi_indices, 1)  # n_basis
    @test pss.max_degree == maximum(pss.multi_indices)
    @test length(pss.state_indices) == 1
    @test length(pss.steady_state) == 1
    @test size(pss.impact) == (1, 1)
    @test length(pss.scale) == 1
    @test length(pss.shift) == 1
    @test size(pss.Z) == (1, 1)
    @test size(pss.H_inv) == (1, 1)
    @test pss.log_det_H isa Float64
    end
end

@testset "Projection PF: transition kernel matches evaluate_policy" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:projection, degree=3, scale=5.0)

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.01 * I, 1, 1)
    pss = MacroEconometricModels._build_projection_state_space(sol, Z, d, H)

    nx = length(pss.state_indices)
    n_vars = size(pss.coefficients, 1)
    n_basis = size(pss.coefficients, 2)
    N = 50
    n_shocks = size(pss.impact, 2)
    n_endog = length(pss.steady_state)

    # Allocate workspace with projection buffers
    ws = MacroEconometricModels._allocate_pf_workspace(
        Float64, n_endog, 1, n_shocks, N;
        proj_nx=nx, proj_n_basis=n_basis,
        proj_max_degree=pss.max_degree, proj_n_vars=n_vars)

    # Set particles to known states and zero shocks
    rng = Random.MersenneTwister(42)
    states_before = zeros(N)
    for k in 1:N
        x_k = 0.001 * randn(rng)
        ws.particles[1, k] = x_k
        states_before[k] = x_k
    end
    fill!(ws.shocks, zero(Float64))

    # Run transition kernel
    MacroEconometricModels._pf_transition_projection!(ws, pss)

    # Compare with point-by-point evaluate_policy for each particle
    for k in 1:N
        y_ref = evaluate_policy(sol, [states_before[k]])
        @test isapprox(ws.particles[1, k], y_ref[1]; atol=1e-10)
    end

    # All particles should be finite
    @test all(isfinite, ws.particles)

    # Zero-allocation check (warmup + measure)
    # Restore particles for another run
    for k in 1:N
        ws.particles[1, k] = states_before[k]
    end
    MacroEconometricModels._pf_transition_projection!(ws, pss)  # warmup
    for k in 1:N
        ws.particles[1, k] = states_before[k]
    end
    allocs = @allocated MacroEconometricModels._pf_transition_projection!(ws, pss)
    @test allocs == 0
    end
end

@testset "Projection bootstrap PF: finite log-likelihood (collocation)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:projection, degree=3, scale=5.0)

    # Generate synthetic data
    sim_data = simulate(sol, 50; rng=Random.MersenneTwister(42))
    data = reshape(sim_data[:, 1], 1, :)  # 1 × T_obs

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.01 * I, 1, 1)
    pss = MacroEconometricModels._build_projection_state_space(sol, Z, d, H)

    nx = length(pss.state_indices)
    n_vars = size(pss.coefficients, 1)
    n_basis = size(pss.coefficients, 2)
    N = 200
    n_shocks = size(pss.impact, 2)
    n_endog = length(pss.steady_state)
    T_obs = size(data, 2)

    ws = MacroEconometricModels._allocate_pf_workspace(
        Float64, n_endog, 1, n_shocks, N;
        proj_nx=nx, proj_n_basis=n_basis,
        proj_max_degree=pss.max_degree, proj_n_vars=n_vars,
        T_obs=T_obs)

    rng = Random.MersenneTwister(123)
    ll = MacroEconometricModels._bootstrap_particle_filter!(
        ws, pss, data, T_obs; rng=rng)

    @test isfinite(ll)
    @test ll > -1e6  # not absurdly negative
    end
end

@testset "Projection bootstrap PF: finite log-likelihood (PFI)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:pfi, degree=3, scale=5.0)
    @test sol isa MacroEconometricModels.ProjectionSolution
    @test sol.method == :pfi

    sim_data = simulate(sol, 50; rng=Random.MersenneTwister(42))
    data = reshape(sim_data[:, 1], 1, :)

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.01 * I, 1, 1)
    pss = MacroEconometricModels._build_projection_state_space(sol, Z, d, H)

    nx = length(pss.state_indices)
    n_vars = size(pss.coefficients, 1)
    n_basis = size(pss.coefficients, 2)
    N = 200
    n_shocks = size(pss.impact, 2)
    n_endog = length(pss.steady_state)
    T_obs = size(data, 2)

    ws = MacroEconometricModels._allocate_pf_workspace(
        Float64, n_endog, 1, n_shocks, N;
        proj_nx=nx, proj_n_basis=n_basis,
        proj_max_degree=pss.max_degree, proj_n_vars=n_vars,
        T_obs=T_obs)

    rng = Random.MersenneTwister(123)
    ll = MacroEconometricModels._bootstrap_particle_filter!(
        ws, pss, data, T_obs; rng=rng)

    @test isfinite(ll)
    @test ll > -1e6
    end
end

@testset "Projection CSMC: conditional SMC with reference trajectory" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:projection, degree=3, scale=5.0)

    sim_data = simulate(sol, 30; rng=Random.MersenneTwister(42))
    data = reshape(sim_data[:, 1], 1, :)

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.01 * I, 1, 1)
    pss = MacroEconometricModels._build_projection_state_space(sol, Z, d, H)

    nx = length(pss.state_indices)
    n_vars = size(pss.coefficients, 1)
    n_basis = size(pss.coefficients, 2)
    N = 100
    n_shocks = size(pss.impact, 2)
    n_endog = length(pss.steady_state)
    T_obs = size(data, 2)

    ws = MacroEconometricModels._allocate_pf_workspace(
        Float64, n_endog, 1, n_shocks, N;
        proj_nx=nx, proj_n_basis=n_basis,
        proj_max_degree=pss.max_degree, proj_n_vars=n_vars,
        T_obs=T_obs)

    rng = Random.MersenneTwister(456)
    ll = MacroEconometricModels._conditional_smc!(
        ws, pss, data, T_obs; rng=rng)

    @test isfinite(ll)
    @test ll > -1e6
    end
end

@testset "Projection PF vs Kalman: linear model comparison" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Generate data from linear model
    sol_lin = solve(spec; method=:gensys)
    sim = simulate(sol_lin, 50; rng=Random.MersenneTwister(42))
    data = reshape(sim[:, 1], 1, :)

    Z = ones(Float64, 1, 1)
    d = zeros(Float64, 1)
    H = Matrix{Float64}(0.1 * I, 1, 1)

    # Kalman log-likelihood
    ss = MacroEconometricModels._build_state_space(sol_lin, Z, d, H)
    ll_kalman = MacroEconometricModels._kalman_loglikelihood(ss, data)

    # Projection PF log-likelihood (average over multiple runs for stability)
    sol_proj = solve(spec; method=:projection, degree=5, scale=5.0)
    pss = MacroEconometricModels._build_projection_state_space(sol_proj, Z, d, H)

    nx = length(pss.state_indices)
    n_vars = size(pss.coefficients, 1)
    n_basis = size(pss.coefficients, 2)
    N = 500
    n_shocks = size(pss.impact, 2)
    n_endog = length(pss.steady_state)
    T_obs = size(data, 2)

    lls = Float64[]
    for seed in 1:10
        ws = MacroEconometricModels._allocate_pf_workspace(
            Float64, n_endog, 1, n_shocks, N;
            proj_nx=nx, proj_n_basis=n_basis,
            proj_max_degree=pss.max_degree, proj_n_vars=n_vars)
        rng = Random.MersenneTwister(seed)
        ll = MacroEconometricModels._bootstrap_particle_filter!(
            ws, pss, data, T_obs; rng=rng)
        push!(lls, ll)
    end
    ll_pf_mean = mean(lls)

    # PF should be in the right ballpark (within 30% of Kalman)
    @test isfinite(ll_pf_mean)
    @test abs(ll_pf_mean - ll_kalman) / abs(ll_kalman) < 0.3
    end
end

@testset "_build_solution_at_theta: ProjectionSolution dispatch" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Projection solver → ProjectionStateSpace
    sol_proj, ss_proj = MacroEconometricModels._build_solution_at_theta(
        spec, [:ρ], [0.9], [:y], nothing, :projection, (degree=3, scale=5.0))
    @test sol_proj isa MacroEconometricModels.ProjectionSolution
    @test ss_proj isa MacroEconometricModels.ProjectionStateSpace

    # PFI solver → ProjectionStateSpace
    sol_pfi, ss_pfi = MacroEconometricModels._build_solution_at_theta(
        spec, [:ρ], [0.9], [:y], nothing, :pfi, (degree=3, scale=5.0))
    @test sol_pfi isa MacroEconometricModels.ProjectionSolution
    @test ss_pfi isa MacroEconometricModels.ProjectionStateSpace
    @test sol_pfi.method == :pfi
    end
end

@testset "MH mutation dispatch for projection solver" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data_obs = randn(MersenneTwister(42), 1, 30) .* 0.02
    priors = Dict(:ρ => Normal(0.5, 0.2))
    θ0 = [0.5]

    result = estimate_dsge_bayes(
        spec, data_obs, θ0;
        priors=priors, method=:smc2, observables=[:y],
        n_smc=10, n_particles=25, n_mh_steps=3,
        ess_target=0.5, measurement_error=[0.005],
        solver=:projection, solver_kwargs=(degree=3, scale=5.0),
        rng=MersenneTwister(123))
    @test result isa MacroEconometricModels.BayesianDSGE
    @test isfinite(result.log_marginal_likelihood)
    @test any(r -> r > 0, result.ess_history)
    end
end

@testset "Collocation solver warm-starting with initial_coeffs" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    # Solve once to get reference coefficients
    sol1 = solve(spec; method=:projection, degree=3, scale=5.0)
    coeffs1 = sol1.coefficients

    # Solve again with warm-starting — should produce same result
    sol2 = solve(spec; method=:projection, degree=3, scale=5.0,
                 initial_coeffs=copy(coeffs1))
    @test sol2 isa MacroEconometricModels.ProjectionSolution
    @test isapprox(sol2.coefficients, coeffs1, atol=1e-6)

    # Solve at slightly different parameter — warm-start should still converge
    spec2 = @dsge begin
        parameters: ρ = 0.85, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec2 = compute_steady_state(spec2)
    sol3 = solve(spec2; method=:projection, degree=3, scale=5.0,
                 initial_coeffs=copy(coeffs1))
    @test sol3 isa MacroEconometricModels.ProjectionSolution
    @test isfinite(MacroEconometricModels.max_euler_error(sol3))
    end
end

@testset "PFI solver warm-starting with initial_coeffs" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    sol1 = solve(spec; method=:pfi, degree=3, scale=5.0)
    coeffs1 = sol1.coefficients

    # Warm-start with own coefficients → same result
    sol2 = solve(spec; method=:pfi, degree=3, scale=5.0,
                 initial_coeffs=copy(coeffs1))
    @test sol2 isa MacroEconometricModels.ProjectionSolution
    @test isapprox(sol2.coefficients, coeffs1, atol=1e-6)

    # Warm-start at nearby parameter
    spec2 = @dsge begin
        parameters: ρ = 0.85, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec2 = compute_steady_state(spec2)
    sol3 = solve(spec2; method=:pfi, degree=3, scale=5.0,
                 initial_coeffs=copy(coeffs1))
    @test sol3 isa MacroEconometricModels.ProjectionSolution
    @test isfinite(MacroEconometricModels.max_euler_error(sol3))
    end
end

@testset "SMC² per-particle solution storage" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data_obs = randn(MersenneTwister(42), 1, 30) .* 0.02
    priors = Dict(:ρ => Normal(0.5, 0.2))
    θ0 = [0.5]

    # Standard SMC² still works after adding solution storage
    result = estimate_dsge_bayes(
        spec, data_obs, θ0;
        priors=priors, method=:smc2, observables=[:y],
        n_smc=10, n_particles=25, n_mh_steps=2,
        ess_target=0.5, measurement_error=[0.005],
        solver=:projection, solver_kwargs=(degree=3, scale=5.0),
        rng=MersenneTwister(999))
    @test result isa MacroEconometricModels.BayesianDSGE
    @test isfinite(result.log_marginal_likelihood)
    end
end

@testset "Delayed acceptance kwarg passthrough" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data_obs = randn(MersenneTwister(42), 1, 30) .* 0.02
    priors = Dict(:ρ => Normal(0.5, 0.2))
    θ0 = [0.5]

    # delayed_acceptance kwarg should be accepted and produce valid result
    result = estimate_dsge_bayes(
        spec, data_obs, θ0;
        priors=priors, method=:smc2, observables=[:y],
        n_smc=10, n_particles=25, n_mh_steps=2,
        ess_target=0.5, measurement_error=[0.005],
        solver=:projection, solver_kwargs=(degree=3, scale=5.0),
        delayed_acceptance=true, n_screen=15,
        rng=MersenneTwister(555))
    @test result isa MacroEconometricModels.BayesianDSGE
    @test isfinite(result.log_marginal_likelihood)
    end
end

@testset "Two-stage delayed acceptance MH correctness" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data_obs = randn(MersenneTwister(42), 1, 50) .* 0.02
    priors = Dict(:ρ => Normal(0.5, 0.2))
    θ0 = [0.5]

    # Run standard SMC²
    result_std = estimate_dsge_bayes(
        spec, data_obs, θ0;
        priors=priors, method=:smc2, observables=[:y],
        n_smc=20, n_particles=50, n_mh_steps=2,
        ess_target=0.5, measurement_error=[0.005],
        solver=:projection, solver_kwargs=(degree=3, scale=5.0),
        rng=MersenneTwister(777))

    # Run delayed acceptance SMC²
    result_da = estimate_dsge_bayes(
        spec, data_obs, θ0;
        priors=priors, method=:smc2, observables=[:y],
        n_smc=20, n_particles=50, n_mh_steps=2,
        ess_target=0.5, measurement_error=[0.005],
        solver=:projection, solver_kwargs=(degree=3, scale=5.0),
        delayed_acceptance=true, n_screen=30,
        rng=MersenneTwister(777))

    @test result_da isa MacroEconometricModels.BayesianDSGE
    @test isfinite(result_da.log_marginal_likelihood)

    # Posterior means should be in same ballpark (both target same posterior)
    rho_std = mean(result_std.theta_draws[:, 1])
    rho_da = mean(result_da.theta_draws[:, 1])
    @test abs(rho_std - rho_da) < 0.5  # same ballpark, Monte Carlo variance
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section: Bayesian DSGE IRF / FEVD / Simulate
# ─────────────────────────────────────────────────────────────────────────────

# Build a simple BayesianDSGE result for testing IRF/FEVD/simulate
_bayes_dsge_irf_test_result = _suppress_warnings() do
    spec = @dsge begin
        parameters: rho = 0.5, sig = 0.01
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sig * e[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    data_obs = randn(MersenneTwister(42), 1, 30) .* 0.02
    priors = Dict(:rho => Normal(0.5, 0.2))
    theta0 = [0.5]
    estimate_dsge_bayes(
        spec, data_obs, theta0;
        priors=priors, method=:smc, observables=[:y],
        n_smc=30, n_mh_steps=1, ess_target=0.5,
        measurement_error=[0.01],
        rng=MersenneTwister(123))
end

@testset "irf(::BayesianDSGE)" begin
    result = _bayes_dsge_irf_test_result
    _suppress_warnings() do
        birf = irf(result, 10; n_draws=10, rng=MersenneTwister(42))
        @test birf isa BayesianImpulseResponse{Float64}
        @test birf.horizon == 10
        @test length(birf.variables) >= 1
        @test length(birf.shocks) >= 1
        @test size(birf.point_estimate) == (10, length(birf.variables), length(birf.shocks))
        @test birf.quantile_levels == Float64[0.05, 0.16, 0.84, 0.95]
        @test size(birf.quantiles, ndims(birf.quantiles)) == 4  # 4 quantile levels

        # Custom quantiles
        birf2 = irf(result, 5; n_draws=5, quantiles=[0.1, 0.9], rng=MersenneTwister(42))
        @test birf2.quantile_levels == Float64[0.1, 0.9]
        @test size(birf2.quantiles, ndims(birf2.quantiles)) == 2

        # show should not error
        io = IOBuffer()
        show(io, birf)
        @test length(take!(io)) > 0
    end
end

@testset "fevd(::BayesianDSGE)" begin
    result = _bayes_dsge_irf_test_result
    _suppress_warnings() do
        bfevd = fevd(result, 10; n_draws=10, rng=MersenneTwister(42))
        @test bfevd isa BayesianFEVD{Float64}
        @test bfevd.horizon == 10
        @test length(bfevd.variables) >= 1
        @test length(bfevd.shocks) >= 1
        @test bfevd.quantile_levels == Float64[0.05, 0.16, 0.84, 0.95]

        # FEVD proportions should be in [0, 1]
        @test all(x -> 0.0 <= x <= 1.0 + 1e-10, bfevd.point_estimate)
    end
end

@testset "simulate(::BayesianDSGE)" begin
    result = _bayes_dsge_irf_test_result
    _suppress_warnings() do
        bsim = simulate(result, 20; n_draws=10, rng=MersenneTwister(42))
        @test bsim isa BayesianDSGESimulation{Float64}
        @test bsim.T_periods == 20
        @test length(bsim.variables) >= 1
        @test bsim.quantile_levels == Float64[0.05, 0.16, 0.84, 0.95]
        @test size(bsim.point_estimate) == (20, length(bsim.variables))
        @test size(bsim.all_paths, 2) == 20
        @test size(bsim.all_paths, 3) == length(bsim.variables)

        # show and report should not error
        io = IOBuffer()
        show(io, bsim)
        @test length(take!(io)) > 0
        report(bsim)
    end
end

@testset "_respec + PF effective-SS offset preserve linear=true (E-07 / #115)" begin
    a, b, c = 0.5, 0.3, 1.0
    spec = DSGESpec{Float64}(
        [:y], [:eps], [:a, :b, :c], Dict{Symbol,Float64}(:a => a, :b => b, :c => c),
        Expr[:(0 + 0)],
        Function[(yt, yl, yle, eps, th) -> yt[1] - th[:a] * yle[1] - th[:b] * yl[1] - th[:c] - eps[1]],
        1, [1], Float64[], nothing; linear=true,
    )
    # _respec must carry every parse/SS-affecting flag to the rebuilt spec — esp. linear=true,
    # which the hand-written rebuild sites dropped (re-parsing the model as nonlinear).
    new_pv = copy(spec.param_values); new_pv[:a] = 0.4
    re = MacroEconometricModels._respec(spec, new_pv)
    @test re.linear == true
    @test re.n_expect == spec.n_expect
    @test re.forward_indices == spec.forward_indices
    @test re.max_lag == spec.max_lag
    @test re.max_lead == spec.max_lead
    @test re.augmented == spec.augmented
    @test re.param_values[:a] == 0.4

    # Kalman and particle-filter paths share _effective_obs_offset: linear=true constant models
    # get the effective SS (I-G1)⁻¹·C_sol as the observation offset, not the zero raw offset.
    sspec = compute_steady_state(spec)
    sol = solve(sspec; method=:gensys)
    d_eff = MacroEconometricModels._effective_obs_offset(zeros(1), sspec, sol, [:y])
    @test d_eff[1] ≈ c / (1 - a - b) atol=1e-6
    @test !(d_eff[1] ≈ 0.0)
end

@testset "Posterior-mean solve failure falls back to highest-posterior draw (T044)" begin
    _suppress_warnings() do
    # Forward-looking model: a_ ≤ 0.9 determinate, a_ ≥ 1.2 indeterminate (verified).
    spec = @dsge begin
        parameters: a_ = 0.5, s = 0.5
        endogenous: y
        exogenous: eps
        y[t] = a_ * y[t+1] + s * eps[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    prior = MacroEconometricModels.DSGEPrior{Float64}([:a_], [Uniform(0.0, 3.0)], [0.0], [3.0])

    # Mean of [0.5, 1.6, 1.7] = 1.27 is INDETERMINATE ⇒ fall back to the highest-posterior
    # draw (argmax log-posterior = index 1 ⇒ a_=0.5, determinate). The whole call must still
    # return a usable BayesianDSGE rather than erroring after sampling.
    result = MacroEconometricModels._mh_to_bayesian_dsge(
        reshape([0.5, 1.6, 1.7], 3, 1), [5.0, 1.0, 0.5], 0.4, prior, [:a_],
        spec, [:y], nothing, :gensys, NamedTuple())
    @test result isa BayesianDSGE{Float64}
    @test result.solved_at == :highest_posterior_draw
    @test is_determined(result.solution)
    @test occursin("Solution built at", sprint(show, result))

    # Control: a determinate mean stays :posterior_mean.
    result2 = MacroEconometricModels._mh_to_bayesian_dsge(
        reshape([0.5, 0.6, 0.7], 3, 1), [1.0, 2.0, 3.0], 0.4, prior, [:a_],
        spec, [:y], nothing, :gensys, NamedTuple())
    @test result2.solved_at == :posterior_mean
    @test is_determined(result2.solution)

    # Shared helper directly (SMC path uses the same seam).
    sol, ss, tag = MacroEconometricModels._build_solution_mean_or_hpd(
        spec, [:a_], [1.5], reshape([0.5, 1.5], 2, 1), [3.0, 0.1],
        [:y], nothing, :gensys, NamedTuple())
    @test tag == :highest_posterior_draw
    @test is_determined(sol)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Posterior mode + Laplace ML + mode-seeded RWMH proposal (T233 / #332)
# ─────────────────────────────────────────────────────────────────────────────

@testset "posterior_mode: mode finding + Laplace ML (#332)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 300; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))

    pm = posterior_mode(spec, data, [0.5]; priors=priors, observables=[:y])

    @test pm isa PosteriorMode{Float64}
    @test length(pm.mode) == 1
    @test pm.param_names == [:ρ]
    @test pm.converged
    # Mode recovers the true ρ = 0.8 from T=300 simulated observations
    @test abs(pm.mode[1] - 0.8) < 0.15
    # Hessian/inverse-Hessian are PD scalars here
    @test pm.hessian[1, 1] > 0
    @test pm.inv_hessian[1, 1] > 0
    @test pm.inv_hessian[1, 1] ≈ 1 / pm.hessian[1, 1] rtol=1e-6
    @test isfinite(pm.log_posterior)
    @test isfinite(pm.log_likelihood)
    @test isfinite(pm.laplace_log_ml)
    # Laplace ML must be below the log posterior maximum plus the Gaussian volume
    # term only when det H > 1; sanity: it is finite and differs from log posterior
    @test pm.laplace_log_ml != pm.log_posterior

    # Mode is a maximizer: log posterior at the mode ≥ at nearby points
    ll_fn = MacroEconometricModels._build_likelihood_fn(
        spec, [:ρ], Matrix{Float64}(data), [:y], nothing, :gensys, NamedTuple())
    prior = MacroEconometricModels._build_bayes_prior(priors)
    lp(θ) = ll_fn([θ]) + MacroEconometricModels._log_prior([θ], prior)
    @test pm.log_posterior ≥ lp(pm.mode[1] + 0.05) - 1e-8
    @test pm.log_posterior ≥ lp(pm.mode[1] - 0.05) - 1e-8

    # transform=false path reaches the same interior mode
    pm2 = posterior_mode(spec, data, [0.5]; priors=priors, observables=[:y],
                         transform=false)
    @test abs(pm2.mode[1] - pm.mode[1]) < 0.02

    # Laplace log-ML agrees with the SMC tempering-path log-ML (documented
    # tolerance: 1.0 nat on this 1-parameter linear-Gaussian model)
    smc_fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=300, rng=Random.MersenneTwister(11))
    @test abs(pm.laplace_log_ml - marginal_likelihood(smc_fit)) < 1.0

    # show does not error
    io = IOBuffer()
    show(io, pm)
    out = String(take!(io))
    @test occursin("Posterior Mode", out)
    end
end

@testset "posterior_mode: RWMH proposal seeding (proposal=:mode)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 300; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))

    fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, proposal=:mode,
        n_draws=1500, burnin=500, observables=[:y],
        rng=Random.MersenneTwister(7))

    @test fit isa BayesianDSGE{Float64}
    # Mode-seeded inverse-Hessian proposal achieves a reasonable acceptance rate
    # (documented target 0.2–0.4; assert a safe envelope)
    @test 0.10 < fit.acceptance_rate < 0.60
    # Posterior mean near truth
    @test abs(mean(fit.theta_draws[:, 1]) - 0.8) < 0.2

    # Invalid proposal symbol throws
    @test_throws ArgumentError estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, proposal=:bogus,
        n_draws=10, burnin=2, observables=[:y])
    end
end

@testset "posterior_mode: non-PD Hessian fallback (flat posterior)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    # κ multiplies a zero regressor: the likelihood is flat in κ, and the
    # Uniform(0,1) prior contributes zero curvature → H ≈ 0 → not PD
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5, κ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + κ * 0.0 * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    data = simulate(sol, 100; rng=rng)'

    priors = Dict(:κ => Uniform(0.0, 1.0))
    pm = @test_logs (:warn, r"not finite positive definite") match_mode=:any begin
        posterior_mode(spec, data, [0.5]; priors=priors, observables=[:y])
    end
    @test isnan(pm.laplace_log_ml)
    # Diagonal fallback proposal is still usable
    @test isfinite(pm.inv_hessian[1, 1])
    @test pm.inv_hessian[1, 1] > 0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# MCMC convergence diagnostics (T234 / #333)
# ─────────────────────────────────────────────────────────────────────────────

@testset "MCMC diagnostics internals: rank-normalized R̂/ESS/Geweke (#333)" begin
    M = MacroEconometricModels
    rng = Random.MersenneTwister(2026)

    # Tied ranks average
    @test M._tied_ranks([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]

    # Rank normalization: monotone, mean ≈ 0
    z = M._rank_normalize(reshape(collect(1.0:100.0), 50, 2))
    @test size(z) == (50, 2)
    @test abs(mean(z)) < 1e-10
    @test issorted(vec(z))

    # iid chain: R̂ ≈ 1, ESS ≈ S, Geweke accepts
    x = randn(rng, 4000)
    @test M._rhat_rank(x) < 1.02
    @test 0.5 * 4000 < M._ess_bulk(x) < 2.0 * 4000
    @test M._ess_tail(x) > 400
    @test abs(M._geweke_z(x)) < 3.5

    # Sticky AR(1) chain (φ = 0.99): ESS collapses
    y = zeros(4000)
    for t in 2:4000
        y[t] = 0.99 * y[t-1] + 0.1 * randn(rng)
    end
    @test M._ess_bulk(y) < 0.2 * 4000

    # Drifting chain: Geweke rejects
    drift = collect(range(0.0, 3.0; length=2000)) .+ 0.1 .* randn(rng, 2000)
    @test abs(M._geweke_z(drift)) > 3.0

    # Degenerate constant chain → NaN diagnostics (no crash)
    @test isnan(M._ess_bulk(fill(1.0, 100)))
    @test isnan(M._rhat_rank(fill(1.0, 100)))
    @test isnan(M._ess_tail(fill(1.0, 100)))
end

@testset "mcmc_diagnostics + trace/acf accessors + low-ESS warning (#333)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 200; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))
    fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, n_draws=2000, burnin=500,
        observables=[:y], rng=Random.MersenneTwister(123))

    # mcmc_diagnostics returns per-parameter stats on the retained chain
    d = mcmc_diagnostics(fit)
    @test d isa MCMCDiagnostics{Float64}
    @test d.param_names == [:ρ]
    @test d.n_draws == 1500                    # burnin discarded
    @test isfinite(d.rhat[1])
    @test d.rhat[1] < 1.2                      # short but reasonably mixed chain
    @test 0 < d.ess_bulk[1] <= 1500 * log10(1500.0)
    @test 0 < d.ess_tail[1]
    @test isfinite(d.geweke_z[1])
    @test 0 <= d.geweke_p[1] <= 1
    @test abs(d.mean[1] - 0.8) < 0.3

    # show does not error and prints the table
    io = IOBuffer()
    show(io, d)
    out = String(take!(io))
    @test occursin("R-hat", out)
    @test occursin("ESS", out)

    # trace accessor
    tr = trace(fit, :ρ)
    @test tr == fit.theta_draws[:, 1]
    @test length(tr) == 1500
    @test_throws ArgumentError trace(fit, :bogus)

    # acf accessor dispatches to the spectral acf
    a = acf(fit, :ρ; lags=10)
    @test length(a.acf) == 10
    @test all(abs.(a.acf) .<= 1.0 .+ 1e-12)

    # low-ESS warning path: an impossible threshold forces the flag
    ps = @test_logs (:warn, r"bulk ESS below") match_mode=:any begin
        posterior_summary(fit; min_ess=10^9)
    end
    @test ps[:ρ][:low_ess] == 1.0
    @test haskey(ps[:ρ], :ess_bulk)
    pt = prior_posterior_table(fit; min_ess=10^9)
    @test pt[1].low_ess === true

    # relaxed threshold: no flag
    ps0 = posterior_summary(fit; min_ess=0)
    @test ps0[:ρ][:low_ess] == 0.0

    # SMC results carry no ESS annotation (weighted particles, not a chain)
    smc_fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:smc, n_smc=100,
        observables=[:y], rng=Random.MersenneTwister(5))
    pss = posterior_summary(smc_fit)
    @test !haskey(pss[:ρ], :ess_bulk)
    # mcmc_diagnostics on SMC draws warns but still computes
    d2 = @test_logs (:warn, r"not a Markov chain") match_mode=:any begin
        mcmc_diagnostics(smc_fit)
    end
    @test d2 isa MCMCDiagnostics{Float64}
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Bridge sampling marginal likelihood (T235 / #334)
# ─────────────────────────────────────────────────────────────────────────────

@testset "bridge_sampling_ml: agrees with SMC and Laplace (#334)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)

    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    sol_true = solve(true_spec; method=:gensys)
    data = simulate(sol_true, 300; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2))

    fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, proposal=:mode,
        n_draws=3000, burnin=1000, observables=[:y],
        rng=Random.MersenneTwister(7))

    # Estimation context is now stored on the result
    @test !isempty(fit.data)
    @test size(fit.data, 1) == 1                  # n_obs × T_obs orientation
    @test fit.observables == [:y]
    @test fit.solver == :gensys

    bml = bridge_sampling_ml(fit; rng=Random.MersenneTwister(3))
    @test isfinite(bml)

    # Documented tolerance: 1 nat against the SMC tempering path and Laplace
    smc_fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:smc, observables=[:y],
        n_smc=300, rng=Random.MersenneTwister(11))
    @test abs(bml - marginal_likelihood(smc_fit)) < 1.0

    pm = posterior_mode(spec, data, [0.5]; priors=priors, observables=[:y])
    @test abs(bml - pm.laplace_log_ml) < 1.0

    # Student-t proposal agrees closely with the normal proposal
    bml_t = bridge_sampling_ml(fit; proposal=:t, df=5, rng=Random.MersenneTwister(3))
    @test isfinite(bml_t)
    @test abs(bml_t - bml) < 0.5

    # Works on SMC draws too (context stored for all methods)
    bml_smc = bridge_sampling_ml(smc_fit; rng=Random.MersenneTwister(3))
    @test isfinite(bml_smc)
    @test abs(bml_smc - bml) < 0.5

    # Invalid proposal family throws
    @test_throws ArgumentError bridge_sampling_ml(fit; proposal=:bogus)
    end
end

@testset "bridge_sampling_ml: failure paths return NaN + warn (#334)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    data = simulate(sol, 100; rng=rng)'
    priors = Dict(:ρ => Beta(2, 2))

    # Chain too short → NaN + warning
    fit_short = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:mh, n_draws=15, burnin=5, observables=[:y],
        rng=Random.MersenneTwister(9))
    bml = @test_logs (:warn, r"chain too short") match_mode=:any begin
        bridge_sampling_ml(fit_short)
    end
    @test isnan(bml)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Identification diagnostics (T236 / #335)
# ─────────────────────────────────────────────────────────────────────────────

@testset "identification_diagnostics: Iskrev rank test (#335)" begin
    # Well-identified AR(1): ρ and σ identified from mean/variance/autocovariance
    spec_ok = @dsge begin
        parameters: ρ = 0.6, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec_ok = compute_steady_state(spec_ok)
    idd = identification_diagnostics(spec_ok, [:ρ, :σ]; observables=[:y])
    @test idd isa IdentificationDiagnostics{Float64}
    @test idd.identified
    @test idd.rank == 2
    @test idd.n_params == 2
    @test isempty(idd.null_space) || size(idd.null_space, 2) == 0
    @test minimum(idd.singular_values) > idd.tol

    # Deliberately unidentified: a and b enter only as the product a·b
    spec_bad = @dsge begin
        parameters: a = 0.5, b = 0.9, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = a * b * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec_bad = compute_steady_state(spec_bad)
    idd2 = @test_logs (:warn, r"unidentified") match_mode=:any begin
        identification_diagnostics(spec_bad, [:a, :b]; observables=[:y])
    end
    @test !idd2.identified
    @test idd2.rank == 1
    @test size(idd2.null_space, 2) == 1
    # Null direction ∝ (a, -b) = (0.5, -0.9) up to sign and normalization
    v = idd2.null_space[:, 1]
    v = v / v[1]
    @test v[2] ≈ -0.9 / 0.5 atol=1e-3

    # show names the unidentified combination
    io = IOBuffer()
    show(io, idd2)
    out = String(take!(io))
    @test occursin("Unidentified direction", out)
    @test occursin("NO", out)

    # Input validation
    @test_throws ArgumentError identification_diagnostics(spec_ok, Symbol[])
    @test_throws ArgumentError identification_diagnostics(spec_ok, [:ρ]; observables=[:zzz])
    @test_throws ArgumentError identification_diagnostics(spec_ok, [:ρ]; theta=[0.5, 0.5])
end

@testset "learning_rate_check + prior_posterior_overlap (#335)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    # κ multiplies a zero regressor → data are uninformative about κ
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5, κ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + κ * 0.0 * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5, κ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + κ * 0.0 * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 300; rng=rng)'

    priors = Dict(:ρ => Beta(2, 2), :κ => Beta(2, 2))
    fit = estimate_dsge_bayes(spec, data, [0.5, 0.5];
        priors=priors, method=:smc, n_smc=300, observables=[:y],
        rng=Random.MersenneTwister(11))

    # Overlap: κ's posterior is essentially the prior; ρ's is not
    ppo = prior_posterior_overlap(fit)
    @test ppo isa PriorPosteriorOverlap{Float64}
    ik = findfirst(==(:κ), ppo.param_names)
    ir = findfirst(==(:ρ), ppo.param_names)
    @test ppo.flagged[ik]
    @test !ppo.flagged[ir]
    @test ppo.overlap[ik] > 0.8
    @test ppo.overlap[ir] < 0.5
    io = IOBuffer()
    show(io, ppo)
    @test occursin("Overlap", String(take!(io)))

    # KPS learning rate: ρ's posterior variance shrinks with T, κ's does not
    lrc = learning_rate_check(fit; fractions=[0.4, 1.0], n_smc=200,
                              rng=Random.MersenneTwister(3))
    @test lrc isa LearningRateCheck{Float64}
    @test lrc.flagged[findfirst(==(:κ), lrc.param_names)]
    @test lrc.learning_rate[findfirst(==(:ρ), lrc.param_names)] > 0.2
    @test size(lrc.post_vars) == (2, 2)
    io2 = IOBuffer()
    show(io2, lrc)
    @test occursin("Learning-Rate", String(take!(io2)))

    # Validation
    @test_throws ArgumentError learning_rate_check(fit; fractions=[0.5])
    @test_throws ArgumentError learning_rate_check(fit; fractions=[0.0, 1.0])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Prior/posterior predictive checks (T237 / #336)
# ─────────────────────────────────────────────────────────────────────────────

@testset "prior_predictive + posterior_predictive_check (#336)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    priors = Dict(:ρ => Beta(2, 2))

    # Prior predictive: draw-level statistic distribution, sensible variance
    ppr = prior_predictive(spec, priors; n_draws=100, T_periods=150,
                           observables=[:y], rng=Random.MersenneTwister(1))
    @test ppr isa PriorPredictiveResult{Float64}
    @test ppr.n_draws == 100
    @test 0 < ppr.n_effective <= 100
    @test size(ppr.stats) == (ppr.n_effective, length(ppr.stat_names))
    @test "mean_y" in ppr.stat_names
    @test "var_y" in ppr.stat_names
    @test "ar1_y" in ppr.stat_names
    j = findfirst(==("var_y"), ppr.stat_names)
    @test isfinite(mean(ppr.stats[:, j]))
    @test mean(ppr.stats[:, j]) > 0
    io = IOBuffer()
    show(io, ppr)
    @test occursin("Prior Predictive", String(take!(io)))

    # Posterior predictive check on a correctly specified model: interior p-values
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 300; rng=rng)'
    fit = estimate_dsge_bayes(spec, data, [0.5];
        priors=priors, method=:smc, n_smc=200, observables=[:y],
        rng=Random.MersenneTwister(11))

    ppc = posterior_predictive_check(fit; n_draws=150, rng=Random.MersenneTwister(2))
    @test ppc isa PosteriorPredictiveCheck{Float64}
    @test ppc.n_effective > 0
    @test length(ppc.p_values) == length(ppc.stat_names) == length(ppc.observed)
    @test size(ppc.replicated) == (ppc.n_effective, length(ppc.stat_names))
    jv = findfirst(==("var_y"), ppc.stat_names)
    ja = findfirst(==("ar1_y"), ppc.stat_names)
    @test 0.01 < ppc.p_values[jv] < 0.99      # model reproduces the variance
    @test 0.01 < ppc.p_values[ja] < 0.99      # ... and the persistence
    io2 = IOBuffer()
    show(io2, ppc)
    @test occursin("p-value", String(take!(io2)))

    # Deliberately misspecified: prior pins ρ ≈ 0.2 while the data have ρ = 0.8
    # → replicated persistence is far below observed → extreme p-value
    tight = Dict(:ρ => Beta(60, 240))
    fit_bad = estimate_dsge_bayes(spec, data, [0.2];
        priors=tight, method=:smc, n_smc=200, observables=[:y],
        rng=Random.MersenneTwister(12))
    ppc_bad = posterior_predictive_check(fit_bad; n_draws=150,
                                         rng=Random.MersenneTwister(3))
    @test ppc_bad.p_values[findfirst(==("ar1_y"), ppc_bad.stat_names)] < 0.05

    # Custom stats via NamedTuple contract
    ppc_c = posterior_predictive_check(fit; n_draws=50, rng=Random.MersenneTwister(4),
        stats=Y -> (skew_y = mean(((Y[:, 1] .- mean(Y[:, 1])) ./ std(Y[:, 1])).^3),))
    @test ppc_c.stat_names == ["skew_y"]
    @test length(ppc_c.p_values) == 1

    # Explicit data argument (T_obs × n_obs orientation) matches stored data
    ppc_d = posterior_predictive_check(fit; data=Matrix(data'), n_draws=50,
                                       rng=Random.MersenneTwister(5))
    @test ppc_d.observed ≈ ppc.observed

    # Regression: model with MORE endogenous series than observables — the
    # default stats must index by the observables, not by the data columns
    spec2 = @dsge begin
        parameters: ρ2 = 0.5, φ2 = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = ρ2 * y[t-1] + e[t]
        i[t] = φ2 * y[t]
    end
    spec2 = compute_steady_state(spec2)
    data2 = simulate(solve(spec2; method=:gensys), 150; rng=Random.MersenneTwister(6))
    # Pass only the observed column (dev convention: data columns == observables, #142)
    fit2 = estimate_dsge_bayes(spec2, data2[:, [1]], [0.5];
        priors=Dict(:ρ2 => Beta(2, 2)), method=:smc, n_smc=100,
        observables=[:y], rng=Random.MersenneTwister(13))
    ppc2 = posterior_predictive_check(fit2; n_draws=25,
                                      rng=Random.MersenneTwister(14))
    @test ppc2.stat_names == ["mean_y", "var_y", "ar1_y"]   # no phantom cross-corrs
    @test all(isfinite, ppc2.observed)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter transformations for samplers (T238 / #337)
# ─────────────────────────────────────────────────────────────────────────────

@testset "sampler transforms: round trip + log-Jacobian (#337)" begin
    # Positive (0,∞), interval (0,1), unbounded, shifted interval (2,5)
    pt = ParameterTransform([0.0, 0.0, -Inf, 2.0], [1.0, Inf, Inf, 5.0])
    theta = [0.3, 1.7, -0.4, 4.2]
    y = to_unconstrained(pt, theta)
    @test to_constrained(pt, y) ≈ theta rtol=1e-12

    # log_jacobian equals the log-abs product of the diagonal Jacobian
    J = transform_jacobian(pt, y)
    @test log_jacobian(pt, y) ≈ sum(log(abs(J[i, i])) for i in 1:4) rtol=1e-10

    # Finite-difference check of log|dθᵢ/dyᵢ| coordinate by coordinate
    h = 1e-6
    for i in 1:4
        yp = copy(y); yp[i] += h
        ym = copy(y); ym[i] -= h
        d_num = (to_constrained(pt, yp)[i] - to_constrained(pt, ym)[i]) / (2h)
        @test log(abs(J[i, i])) ≈ log(abs(d_num)) atol=1e-6
    end

    # Numerically stable at extreme y where naive σ(y)(1-σ(y)) under/overflows
    yext = [-800.0, -800.0, -800.0, 900.0]
    @test isfinite(log_jacobian(pt, yext))
end

@testset "RWMH transform=true: boundary-safe walk (#337)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)

    # σ = 0.05 sits near the zero boundary of its positive support
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.05
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.05
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 300; rng=rng)'
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(3.0, 0.2))

    # burnin=2000: dev's burn-in-frozen proposal adaptation (#139) needs enough
    # tuning events before the freeze for the natural-space (transform=false) walk
    # to mix on this near-boundary problem; with burnin=1000 the frozen baseline
    # proposal stays under-scaled and the two parameterizations disagree.
    fit_t = estimate_dsge_bayes(spec, data, [0.5, 0.1];
        priors=priors, method=:mh, transform=true,
        n_draws=5000, burnin=2000, observables=[:y],
        rng=Random.MersenneTwister(7))
    fit_u = estimate_dsge_bayes(spec, data, [0.5, 0.1];
        priors=priors, method=:mh, transform=false,
        n_draws=5000, burnin=2000, observables=[:y],
        rng=Random.MersenneTwister(7))

    # Posterior means agree across parameterizations within Monte Carlo error
    mt = vec(mean(fit_t.theta_draws; dims=1))
    mu = vec(mean(fit_u.theta_draws; dims=1))
    @test abs(mt[1] - mu[1]) < 0.1
    @test abs(mt[2] - mu[2]) < 0.05
    # ... and recover the truth (ρ = 0.8, σ = 0.05)
    @test abs(mt[1] - 0.8) < 0.15
    @test abs(mt[2] - 0.05) < 0.05
    @test fit_t.acceptance_rate > 0.1

    # The transformed walk never leaves the support by construction —
    # no proposal is wasted on out-of-bounds values
    @test all(fit_t.theta_draws[:, 2] .> 0)
    @test all(0 .< fit_t.theta_draws[:, 1] .< 1)

    # transform composes with the mode-seeded proposal (Σ mapped through D⁻¹)
    fit_mt = estimate_dsge_bayes(spec, data, [0.5, 0.1];
        priors=priors, method=:mh, transform=true, proposal=:mode,
        n_draws=1500, burnin=500, observables=[:y],
        rng=Random.MersenneTwister(9))
    @test 0.1 < fit_mt.acceptance_rate < 0.6
    @test abs(mean(fit_mt.theta_draws[:, 1]) - 0.8) < 0.15
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Dynare prior-convention shims (T239 / #338)
# ─────────────────────────────────────────────────────────────────────────────

@testset "dynare_prior: moment-matched constructors (#338)" begin
    # normal — passthrough
    d = dynare_prior(:normal, 0.3, 0.05)
    @test d isa Normal
    @test mean(d) ≈ 0.3 && std(d) ≈ 0.05

    # gamma — k = m²/s², θ = s²/m
    g = dynare_prior(:gamma, 1.5, 0.25)
    @test g isa Gamma
    @test mean(g) ≈ 1.5 atol=1e-12
    @test std(g) ≈ 0.25 atol=1e-12
    @test g.α ≈ 1.5^2 / 0.25^2 atol=1e-10

    # beta — analytic shape parameters
    b = dynare_prior(:beta, 0.7, 0.1)
    @test b isa Beta
    @test mean(b) ≈ 0.7 atol=1e-12
    @test std(b) ≈ 0.1 atol=1e-12
    k0 = 0.7 * 0.3 / 0.01 - 1
    @test b.α ≈ 0.7 * k0 atol=1e-10
    @test b.β ≈ 0.3 * k0 atol=1e-10

    # generalized beta on (p3, p4) = (0, 0.9)
    bs = dynare_prior(:beta, 0.5, 0.1; lower=0.0, upper=0.9)
    @test mean(bs) ≈ 0.5 atol=1e-10
    @test std(bs) ≈ 0.1 atol=1e-10
    @test minimum(bs) ≈ 0.0 atol=1e-12
    @test maximum(bs) ≈ 0.9 atol=1e-12

    # uniform — from (mean, std) or explicit (lower, upper)
    u = dynare_prior(:uniform, 0.5, 0.1)
    @test u isa Uniform
    @test mean(u) ≈ 0.5 && std(u) ≈ 0.1
    @test dynare_prior(:uniform, 0.0, 0.0; lower=1.0, upper=3.0) == Uniform(1.0, 3.0)

    # validation
    @test_throws ArgumentError dynare_prior(:beta, 0.5, 0.6)     # std² ≥ m(1−m)
    @test_throws ArgumentError dynare_prior(:beta, 1.2, 0.1)     # mean outside (0,1)
    @test_throws ArgumentError dynare_prior(:gamma, -1.0, 1.0)
    @test_throws ArgumentError dynare_prior(:uniform, 0.0, 0.0; lower=2.0, upper=1.0)
    @test_throws ArgumentError dynare_prior(:bogus, 1.0, 1.0)
end

@testset "dynare_prior: inverse gamma on σ (InverseGamma1) (#338)" begin
    # (mean, std) → (s, ν) inversion reproduces the requested σ-moments
    ig = dynare_prior(:inv_gamma, 0.02, 0.05)
    @test ig isa InverseGamma1
    @test mean(ig) ≈ 0.02 rtol=1e-6
    @test std(ig) ≈ 0.05 rtol=1e-6

    # The density is Dynare's TYPE-1 kernel on σ (NOT σ²):
    # log p(x₁) − log p(x₂) = −(ν+1)Δlog x − (s/2)Δ(1/x²)
    x1, x2 = 0.03, 0.08
    lhs = logpdf(ig, x1) - logpdf(ig, x2)
    rhs = -(ig.nu + 1) * (log(x1) - log(x2)) - ig.s / 2 * (1 / x1^2 - 1 / x2^2)
    @test lhs ≈ rhs rtol=1e-10

    # σ² ~ InverseGamma(ν/2, s/2) — the exact Dynare correspondence
    @test cdf(ig, 0.05) ≈ cdf(InverseGamma(ig.nu / 2, ig.s / 2), 0.05^2) rtol=1e-10

    # ... and this differs from naively putting the numbers into InverseGamma
    naive = InverseGamma(0.02, 0.05)
    @test !(logpdf(ig, 0.05) ≈ logpdf(naive, 0.05))

    # proper density: integrates to 1, quantile/cdf round trip, support
    xs = range(1e-6, 2.0; length=200_000)
    @test sum(pdf.(Ref(ig), xs)) * step(xs) ≈ 1.0 atol=5e-3
    @test cdf(ig, quantile(ig, 0.3)) ≈ 0.3 rtol=1e-8
    @test logpdf(ig, -0.1) == -Inf
    @test minimum(ig) == 0.0 && maximum(ig) == Inf

    # draw-based moments on a milder prior (ν ≈ 14.7 — 4th moment exists)
    ig_mild = dynare_prior(:inv_gamma, 0.5, 0.1)
    rng = Random.MersenneTwister(1)
    draws = [rand(rng, ig_mild) for _ in 1:100_000]
    @test mean(draws) ≈ 0.5 rtol=0.02
    @test std(draws) ≈ 0.1 rtol=0.05

    # :inv_gamma2 is on the variance with matched moments
    igv = dynare_prior(:inv_gamma2, 0.004, 0.002)
    @test igv isa InverseGamma
    @test mean(igv) ≈ 0.004 rtol=1e-12
    @test std(igv) ≈ 0.002 rtol=1e-12

    # direct (s, ν) constructor validation
    @test_throws ArgumentError InverseGamma1(-1.0, 3.0)
    @test_throws ArgumentError InverseGamma1(1.0, -3.0)
end

@testset "dynare_prior: end-to-end in estimate_dsge_bayes (#338)" begin
    _suppress_warnings() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    ts = @dsge begin
        parameters: ρ = 0.8, σ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    ts = compute_steady_state(ts)
    data = simulate(solve(ts; method=:gensys), 200; rng=rng)'

    priors = Dict{Symbol,Distribution}(
        :ρ => dynare_prior(:beta, 0.5, 0.2),
        :σ => dynare_prior(:inv_gamma, 0.5, 0.3))
    fit = estimate_dsge_bayes(spec, data, [0.5, 0.5];
        priors=priors, method=:smc, n_smc=200, observables=[:y],
        rng=Random.MersenneTwister(3))
    ps = posterior_summary(fit)
    @test abs(ps[:ρ][:mean] - 0.8) < 0.25
    @test abs(ps[:σ][:mean] - 0.5) < 0.2
    end
end

end  # @testset "Bayesian DSGE"

# =============================================================================
# Regression (#139/#141 interaction): posterior-draw irf/fevd/simulate on a
# model with more endogenous variables than shocks must not trip the
# stochastic-singularity guard (the rebuild skips the observation equation)
# =============================================================================

@testset "posterior irf/fevd/simulate with n_endog > n_shocks" begin
    Random.seed!(13901)
    spec = @dsge begin
        parameters: ρ = 0.9, α = 0.33
        endogenous: Y, K, A
        exogenous: ε
        A[t] = ρ * A[t-1] + ε[t]
        Y[t] = A[t] + α * K[t-1]
        K[t] = 0.9 * K[t-1] + 0.1 * Y[t]
    end
    spec2 = compute_steady_state(spec)
    sol = solve(spec2; method=:gensys)
    Y = simulate(sol, 150)
    b = estimate_dsge_bayes(spec, Y[:, [1]], [0.8];
                            priors=Dict(:ρ => Beta(5, 2)),
                            method=:mh, n_draws=200, burnin=80,
                            observables=[:Y])
    r = irf(b, 8; n_draws=4)
    @test size(r.point_estimate) == (8, 3, 1)
    f = fevd(b, 8; n_draws=4)
    @test f !== nothing
    s = simulate(b, 20; n_draws=4)
    @test s !== nothing
end
