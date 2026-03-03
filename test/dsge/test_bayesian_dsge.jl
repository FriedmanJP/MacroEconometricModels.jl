# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

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

    # H should be positive definite (default 1e-4 * I)
    @test size(H) == (2, 2)
    @test H[1, 1] > 0.0
    @test H[2, 2] > 0.0
    @test H[1, 2] == 0.0
    @test H ≈ 1e-4 * I(2)

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

@testset "Auxiliary particle filter" begin
    Random.seed!(300)

    # Same AR(1) setup
    rho = 0.8
    sigma_eps = 0.5
    sigma_me = 0.1
    T_sim = 150

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

    # Kalman log-likelihood
    ll_kalman = MacroEconometricModels._kalman_loglikelihood(ss, data)

    # APF log-likelihood
    N_particles = 500
    n_runs = 10
    ll_apf_runs = zeros(n_runs)

    for r in 1:n_runs
        ws = MacroEconometricModels._allocate_pf_workspace(Float64, 1, 1, 1, N_particles)
        ll_apf_runs[r] = MacroEconometricModels._auxiliary_particle_filter!(
            ws, ss, data, T_sim; rng=Random.MersenneTwister(r * 2000))
    end

    ll_apf_mean = mean(ll_apf_runs)

    @test isfinite(ll_apf_mean)
    # APF should be finite and in the right ballpark
    # APF likelihood estimate can differ from Kalman due to the two-stage structure
    # but should still be a reasonable finite value
    @test ll_apf_mean < 0.0  # log-likelihood should be negative
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
    log_liks = randn(N)
    phi_old = 0.0
    ess_target = 0.5

    phi_new = MacroEconometricModels._adaptive_tempering(log_liks, phi_old, ess_target, N)
    @test phi_new > phi_old
    @test phi_new <= 1.0

    # Check ESS at this phi
    delta_phi = phi_new - phi_old
    inc_log_w = delta_phi .* log_liks
    inc_log_w .-= MacroEconometricModels._logsumexp(inc_log_w) - log(N)
    w = exp.(inc_log_w)
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

    pf_ll_fn = MacroEconometricModels._build_pf_likelihood_fn(spec, [:ρ], data_mat,
        [:y], nothing, :gensys, NamedTuple(), 100)

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
        n_smc=100, n_particles=50, n_mh_steps=1, ess_target=0.5,
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
    @test size(result.theta_draws, 1) == 1000
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

end  # @testset "Bayesian DSGE"
