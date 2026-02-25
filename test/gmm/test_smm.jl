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
using Random
using LinearAlgebra
using Statistics

const _suppress_warnings = MacroEconometricModels._suppress_warnings

@testset "SMM Estimation" begin

@testset "Parameter Transforms" begin
    @testset "ParameterTransform construction" begin
        pt = ParameterTransform([0.0, -Inf, 0.0], [1.0, Inf, Inf])
        @test pt.lower == [0.0, -Inf, 0.0]
        @test pt.upper == [1.0, Inf, Inf]
    end

    @testset "Identity transform (unbounded)" begin
        pt = ParameterTransform([-Inf], [Inf])
        @test to_unconstrained(pt, [2.5]) ≈ [2.5]
        @test to_constrained(pt, [2.5]) ≈ [2.5]
    end

    @testset "Exp/log transform (lower bounded)" begin
        pt = ParameterTransform([0.0], [Inf])
        theta = [2.0]
        phi = to_unconstrained(pt, theta)
        @test phi ≈ [log(2.0)]
        @test to_constrained(pt, phi) ≈ theta
    end

    @testset "Negative exp transform (upper bounded)" begin
        pt = ParameterTransform([-Inf], [0.0])
        theta = [-3.0]
        phi = to_unconstrained(pt, theta)
        @test to_constrained(pt, phi) ≈ theta atol=1e-10
    end

    @testset "Logistic transform (bounded interval)" begin
        pt = ParameterTransform([0.0], [1.0])
        theta = [0.5]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
        # Boundary behavior
        @test to_constrained(pt, [-100.0])[1] >= 0.0
        @test to_constrained(pt, [100.0])[1] <= 1.0
        # Moderate values strictly inside bounds
        @test to_constrained(pt, [-10.0])[1] > 0.0
        @test to_constrained(pt, [10.0])[1] < 1.0
    end

    @testset "Round-trip multiple parameters" begin
        pt = ParameterTransform([0.0, -1.0, 0.0, -Inf], [1.0, 1.0, Inf, Inf])
        theta = [0.3, 0.0, 2.5, -1.0]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
    end

    @testset "Jacobian diagonal" begin
        pt = ParameterTransform([0.0, -Inf], [1.0, Inf])
        phi = [0.0, 3.0]
        J = transform_jacobian(pt, phi)
        @test size(J) == (2, 2)
        @test J[1, 2] == 0.0  # diagonal
        @test J[2, 1] == 0.0
        @test J[1, 1] > 0.0   # positive for logistic
        @test J[2, 2] == 1.0  # identity for unbounded
    end
end

@testset "GMM with Parameter Transforms" begin
    rng = Random.MersenneTwister(42)
    true_mu = 0.7
    data = true_mu .+ 0.1 .* randn(rng, 200, 1)

    function mean_moments(theta, data)
        data .- theta[1]
    end

    bounds = ParameterTransform([0.0], [1.0])
    result = estimate_gmm(mean_moments, [0.5], data;
                          weighting=:identity, bounds=bounds)
    @test result.converged
    @test abs(result.theta[1] - true_mu) < 0.05
    @test result.theta[1] > 0.0
    @test result.theta[1] < 1.0
end

@testset "autocovariance_moments" begin
    rng = Random.MersenneTwister(123)
    data = randn(rng, 500, 2)
    m = autocovariance_moments(data; lags=1)
    # k=2, lags=1: k*(k+1)/2 + k*lags = 3 + 2 = 5 moments
    @test length(m) == 5
    # First element: var(y1) — using 1/n divisor
    @test m[1] ≈ sum((data[:,1] .- mean(data[:,1])).^2) / 500 atol=1e-10
end

@testset "SMMModel construction and interface" begin
    theta = [0.5, 0.3]
    vcov_mat = [0.01 0.0; 0.0 0.02]
    n_moments = 5
    W = Matrix{Float64}(I, n_moments, n_moments)
    g_bar = zeros(n_moments)
    weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)

    smm = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov_mat, n_moments, 2, 200, weighting, W, g_bar,
        0.5, 0.48, true, 10, 5
    )

    @test coef(smm) == theta
    @test nobs(smm) == 200
    @test stderror(smm) ≈ sqrt.(diag(vcov_mat))
    @test smm.sim_ratio == 5

    io = IOBuffer()
    show(io, smm)
    str = String(take!(io))
    @test occursin("SMM", str)
end

@testset "SMMModel j_test" begin
    theta = [0.5, 0.3]
    vcov_mat = [0.01 0.0; 0.0 0.02]
    n_moments = 5
    W = Matrix{Float64}(I, n_moments, n_moments)
    g_bar = zeros(n_moments)
    weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)

    # Overidentified case
    smm = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov_mat, n_moments, 2, 200, weighting, W, g_bar,
        2.5, 0.47, true, 10, 5
    )
    jt = j_test(smm)
    @test jt.df == 3  # 5 moments - 2 params
    @test jt.J_stat == 2.5
    @test jt.p_value == 0.47

    # Just-identified case
    smm_just = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov_mat, 2, 2, 200, weighting,
        Matrix{Float64}(I, 2, 2), zeros(2),
        0.0, 1.0, true, 10, 5
    )
    jt_just = j_test(smm_just)
    @test jt_just.df == 0
    @test jt_just.J_stat == 0.0
end

@testset "SMMModel is_overidentified and overid_df" begin
    theta = [0.5]
    vcov_mat = reshape([0.01], 1, 1)
    W = Matrix{Float64}(I, 3, 3)
    g_bar = zeros(3)
    weighting = MacroEconometricModels.GMMWeighting{Float64}(:identity, 100, 1e-8)

    smm = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov_mat, 3, 1, 100, weighting, W, g_bar,
        1.0, 0.6, true, 5, 3
    )
    @test MacroEconometricModels.is_overidentified(smm) == true
    @test MacroEconometricModels.overid_df(smm) == 2
end

@testset "SMMModel confint" begin
    theta = [0.5, 0.3]
    vcov_mat = [0.01 0.0; 0.0 0.04]
    n_moments = 5
    W = Matrix{Float64}(I, n_moments, n_moments)
    g_bar = zeros(n_moments)
    weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)

    smm = MacroEconometricModels.SMMModel{Float64}(
        theta, vcov_mat, n_moments, 2, 200, weighting, W, g_bar,
        0.5, 0.48, true, 10, 5
    )
    ci = confint(smm)
    @test size(ci) == (2, 2)
    @test ci[1, 1] < 0.5 < ci[1, 2]  # CI contains point estimate
    @test ci[2, 1] < 0.3 < ci[2, 2]
end

@testset "smm_weighting_matrix" begin
    rng = Random.MersenneTwister(42)
    data = randn(rng, 200, 2)
    # Use hac=false to avoid bandwidth estimation issues with per-observation
    # moment contributions from autocovariance_moments (which are degenerate for
    # single observations)
    W = MacroEconometricModels.smm_weighting_matrix(data, d -> autocovariance_moments(d; lags=1); hac=false)
    @test size(W) == (5, 5)
    @test issymmetric(round.(W, digits=10))  # approximately symmetric

    # With explicit bandwidth (avoids automatic bandwidth NaN issue)
    W2 = MacroEconometricModels.smm_weighting_matrix(data, d -> autocovariance_moments(d; lags=1); hac=true, bandwidth=5)
    @test size(W2) == (5, 5)
end

@testset "estimate_smm — AR(1) recovery" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(42)
        true_rho = 0.8
        true_sigma = 0.5
        T_obs = 500
        y = zeros(T_obs)
        for t in 2:T_obs
            y[t] = true_rho * y[t-1] + true_sigma * randn(rng)
        end
        data = reshape(y, :, 1)

        function sim_ar1(theta, T_periods, burn; rng=Random.default_rng())
            rho, sigma = theta
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn)
                sim[t] = rho * sim[t-1] + abs(sigma) * randn(rng)
            end
            reshape(sim[(burn+1):end], :, 1)
        end

        function my_moments(d)
            autocovariance_moments(d; lags=1)
        end

        # Use bounds to properly constrain the parameter space:
        # rho in (-1, 1), sigma in (0, Inf)
        bounds = ParameterTransform([-1.0, 0.0], [1.0, Inf])
        result = estimate_smm(sim_ar1, my_moments, [0.5, 0.3], data;
                              sim_ratio=5, burn=100, weighting=:two_step,
                              bounds=bounds,
                              rng=Random.MersenneTwister(123))

        @test result isa SMMModel{Float64}
        @test result.converged
        @test abs(result.theta[1] - true_rho) < 0.15
        @test abs(result.theta[2] - true_sigma) < 0.15
        @test result.sim_ratio == 5
        @test length(stderror(result)) == 2
        @test all(stderror(result) .> 0)
    end
end

@testset "estimate_smm with bounds" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(42)
        true_rho = 0.8
        true_sigma = 0.5
        T_obs = 500
        y = zeros(T_obs)
        for t in 2:T_obs
            y[t] = true_rho * y[t-1] + true_sigma * randn(rng)
        end
        data = reshape(y, :, 1)

        function sim_ar1_bounded(theta, T_periods, burn; rng=Random.default_rng())
            rho, sigma = theta
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn)
                sim[t] = rho * sim[t-1] + sigma * randn(rng)
            end
            reshape(sim[(burn+1):end], :, 1)
        end

        bounds = ParameterTransform([-1.0, 0.0], [1.0, Inf])
        result = estimate_smm(sim_ar1_bounded, autocovariance_moments, [0.5, 0.3], data;
                              sim_ratio=5, burn=100, bounds=bounds,
                              rng=Random.MersenneTwister(123))

        @test result.converged
        @test -1.0 < result.theta[1] < 1.0
        @test result.theta[2] > 0.0
    end
end

@testset "estimate_smm — identity weighting" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(99)
        y = zeros(300)
        for t in 2:300
            y[t] = 0.6 * y[t-1] + 0.4 * randn(rng)
        end
        data = reshape(y, :, 1)

        function sim_fn_identity(theta, T_periods, burn; rng=Random.default_rng())
            rho = theta[1]
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn)
                sim[t] = rho * sim[t-1] + randn(rng)
            end
            reshape(sim[(burn+1):end], :, 1)
        end

        result = estimate_smm(sim_fn_identity, d -> autocovariance_moments(d; lags=1),
                              [0.3], data;
                              sim_ratio=5, burn=50, weighting=:identity,
                              rng=Random.MersenneTwister(42))
        @test result isa SMMModel{Float64}
    end
end

@testset "j_test on SMMModel from estimate_smm" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(42)
        y = zeros(300)
        for t in 2:300
            y[t] = 0.7 * y[t-1] + 0.5 * randn(rng)
        end
        data = reshape(y, :, 1)

        function sim_fn_jtest(theta, T_periods, burn; rng=Random.default_rng())
            rho, sigma = theta
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn)
                sim[t] = rho * sim[t-1] + sigma * randn(rng)
            end
            reshape(sim[(burn+1):end], :, 1)
        end

        result = estimate_smm(sim_fn_jtest, d -> autocovariance_moments(d; lags=2),
                              [0.5, 0.3], data;
                              sim_ratio=5, burn=100, weighting=:two_step,
                              rng=Random.MersenneTwister(55))
        # k=1 variable, lags=2: k*(k+1)/2 + k*lags = 1 + 2 = 3 moments, 2 params → overid
        jt = j_test(result)
        @test jt.df == result.n_moments - result.n_params
        @test jt.J_stat >= 0
        @test 0 <= jt.p_value <= 1
    end
end

@testset "SMMModel report and show" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(42)
        y = zeros(200)
        for t in 2:200; y[t] = 0.7 * y[t-1] + randn(rng); end
        data = reshape(y, :, 1)

        function sim_fn_report(theta, T_periods, burn; rng=Random.default_rng())
            rho = theta[1]
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn); sim[t] = rho * sim[t-1] + randn(rng); end
            reshape(sim[(burn+1):end], :, 1)
        end

        result = estimate_smm(sim_fn_report, d -> autocovariance_moments(d; lags=1),
                              [0.5], data; sim_ratio=3, burn=50,
                              rng=Random.MersenneTwister(42))
        io = IOBuffer()
        show(io, result)
        str = String(take!(io))
        @test occursin("SMM", str)
        @test occursin("Sim ratio", str)
        @test redirect_stdout(devnull) do; report(result) end === nothing  # should not error
    end
end

@testset "SMM refs()" begin
    _suppress_warnings() do
        rng = Random.MersenneTwister(42)
        y = zeros(200)
        for t in 2:200; y[t] = 0.7 * y[t-1] + randn(rng); end
        data = reshape(y, :, 1)

        function sim_fn_refs(theta, T_periods, burn; rng=Random.default_rng())
            rho = theta[1]
            sim = zeros(T_periods + burn)
            for t in 2:(T_periods + burn); sim[t] = rho * sim[t-1] + randn(rng); end
            reshape(sim[(burn+1):end], :, 1)
        end

        result = estimate_smm(sim_fn_refs, d -> autocovariance_moments(d; lags=1),
                              [0.5], data; sim_ratio=3, burn=50,
                              rng=Random.MersenneTwister(42))

        io = IOBuffer()
        refs(io, result)
        str = String(take!(io))
        @test occursin("Ruge-Murcia", str) || occursin("Lee", str) || occursin("Hansen", str)
    end
end

end  # outer testset
