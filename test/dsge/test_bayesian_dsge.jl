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

end  # @testset "Bayesian DSGE"
