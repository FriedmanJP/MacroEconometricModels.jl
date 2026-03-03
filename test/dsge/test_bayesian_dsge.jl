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

end  # @testset "Bayesian DSGE"
