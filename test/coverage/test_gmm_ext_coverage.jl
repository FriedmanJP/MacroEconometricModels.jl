# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Coverage tests for GMM transforms, SMM edge cases, and JuMP/PATH extensions
#
# Targets:
#   src/gmm/transforms.jl  — lower-only, upper-only, non-zero offset bounds; Jacobian edges
#   src/gmm/smm.jl         — weighting=:identity path, moment count validation
#   ext/MacroEconometricModelsJuMPExt.jl  — constrained SS & PF edge cases
#   ext/MacroEconometricModelsPATHExt.jl  — MCP SS & PF edge cases

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using StatsAPI

Random.seed!(9006)

const _suppress_warnings = MacroEconometricModels._suppress_warnings

# ============================================================================
# 1. ParameterTransform — missed branches in transforms.jl
# ============================================================================

@testset "ParameterTransform coverage" begin

    @testset "Lower-only bound (a, Inf) with a != 0" begin
        # This hits lines 73-74 in to_unconstrained and line 103 in to_constrained
        # and line 131 in transform_jacobian: the (a, Inf) with finite lo != 0 branch
        pt = ParameterTransform([2.0], [Inf])
        theta = [5.0]
        phi = to_unconstrained(pt, theta)
        @test phi[1] ≈ log(5.0 - 2.0)  # log(theta - a)
        theta_back = to_constrained(pt, phi)
        @test theta_back[1] ≈ 5.0 atol=1e-12  # round-trip

        # Jacobian for (a, Inf) branch: d(a + exp(phi))/dphi = exp(phi)
        J = transform_jacobian(pt, phi)
        @test J[1, 1] ≈ exp(phi[1]) atol=1e-10
        @test J[1, 1] > 0.0

        # Verify constrained value respects lower bound
        @test to_constrained(pt, [-10.0])[1] > 2.0
        @test to_constrained(pt, [0.0])[1] ≈ 2.0 + 1.0  # a + exp(0) = a + 1
    end

    @testset "Upper-only bound (-Inf, b) with b != 0" begin
        # This hits lines 79-80 in to_unconstrained and lines 107 in to_constrained
        # and line 135 in transform_jacobian: the (-Inf, b) with finite hi != 0 branch
        pt = ParameterTransform([-Inf], [3.0])
        theta = [1.0]
        phi = to_unconstrained(pt, theta)
        @test phi[1] ≈ log(3.0 - 1.0)  # log(b - theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back[1] ≈ 1.0 atol=1e-12  # round-trip

        # Jacobian for (-Inf, b) branch: d(b - exp(phi))/dphi = -exp(phi)
        J = transform_jacobian(pt, phi)
        @test J[1, 1] ≈ -exp(phi[1]) atol=1e-10
        @test J[1, 1] < 0.0

        # Verify constrained value respects upper bound
        @test to_constrained(pt, [10.0])[1] < 3.0
        @test to_constrained(pt, [0.0])[1] ≈ 3.0 - 1.0  # b - exp(0) = b - 1
    end

    @testset "Mixed bound types in a single transform" begin
        # Combines ALL branch types in one ParameterTransform:
        # 1: (-Inf, Inf) identity
        # 2: (0, Inf) exp/log
        # 3: (-Inf, 0) -exp
        # 4: (2, Inf) shifted exp
        # 5: (-Inf, 5) shifted negative exp
        # 6: (0, 1) logistic
        pt = ParameterTransform(
            [-Inf, 0.0, -Inf, 2.0, -Inf, 0.0],
            [ Inf, Inf,  0.0, Inf,  5.0, 1.0]
        )
        theta = [3.5, 2.0, -1.5, 4.0, 2.0, 0.7]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10

        # Jacobian should be diagonal
        J = transform_jacobian(pt, phi)
        for i in 1:6, j in 1:6
            if i != j
                @test J[i, j] == 0.0
            end
        end
        # Identity element
        @test J[1, 1] == 1.0
        # (0, Inf) element
        @test J[2, 2] ≈ exp(phi[2]) atol=1e-10
        # (-Inf, 0) element
        @test J[3, 3] ≈ -exp(phi[3]) atol=1e-10
        # (2, Inf) element
        @test J[4, 4] ≈ exp(phi[4]) atol=1e-10
        # (-Inf, 5) element
        @test J[5, 5] ≈ -exp(phi[5]) atol=1e-10
        # (0, 1) logistic element
        e = exp(-phi[6])
        @test J[6, 6] ≈ (1.0 - 0.0) * e / (1.0 + e)^2 atol=1e-10
    end

    @testset "ParameterTransform construction from integers" begin
        # The outer constructor that converts Real vectors to Float64
        pt = ParameterTransform([0, -1], [1, 1])
        @test pt isa ParameterTransform{Float64}
        @test pt.lower == [0.0, -1.0]
        @test pt.upper == [1.0, 1.0]
    end

    @testset "ParameterTransform assertion errors" begin
        # Length mismatch
        @test_throws AssertionError ParameterTransform([0.0, 1.0], [1.0])
        # lower >= upper (non-Inf)
        @test_throws AssertionError ParameterTransform([2.0], [1.0])
    end

    @testset "Negative lower-only bound" begin
        # (a, Inf) where a < 0
        pt = ParameterTransform([-3.0], [Inf])
        theta = [0.5]
        phi = to_unconstrained(pt, theta)
        @test phi[1] ≈ log(0.5 - (-3.0))  # log(theta - a) = log(3.5)
        @test to_constrained(pt, phi)[1] ≈ 0.5 atol=1e-12
    end

    @testset "Negative upper-only bound" begin
        # (-Inf, b) where b < 0
        pt = ParameterTransform([-Inf], [-2.0])
        theta = [-5.0]
        phi = to_unconstrained(pt, theta)
        @test phi[1] ≈ log(-2.0 - (-5.0))  # log(b - theta) = log(3.0)
        @test to_constrained(pt, phi)[1] ≈ -5.0 atol=1e-12
    end

    @testset "Float32 parameter transform" begin
        pt = ParameterTransform(Float32[0.0], Float32[1.0])
        @test pt isa ParameterTransform{Float32}
        theta = Float32[0.5]
        phi = to_unconstrained(pt, theta)
        @test eltype(phi) == Float32
        J = transform_jacobian(pt, phi)
        @test eltype(J) == Float32
    end
end

# ============================================================================
# 2. SMM — identity weighting path and edge cases
# ============================================================================

@testset "SMM coverage" begin

    @testset "estimate_smm — identity weighting with overidentification" begin
        # Exercises the weighting=:identity path with sandwich vcov computation
        # (lines 463-467 in smm.jl) and ensures J-stat is computed correctly
        _suppress_warnings() do
            rng = Random.MersenneTwister(9006)
            true_rho = 0.7
            true_sigma = 0.4
            T_obs = 400
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

            # Use 2 lags for overidentification: 1*(1+1)/2 + 1*2 = 3 moments for 2 params
            result = estimate_smm(sim_ar1, d -> autocovariance_moments(d; lags=2),
                                  [0.5, 0.3], data;
                                  sim_ratio=3, burn=50, weighting=:identity,
                                  rng=Random.MersenneTwister(42))

            @test result isa SMMModel{Float64}
            @test result.weighting.method == :identity
            @test result.n_moments == 3  # overidentified
            @test result.n_params == 2
            @test size(result.vcov) == (2, 2)
            @test all(isfinite, result.vcov)

            # J-test should be available for overidentified model
            jt = j_test(result)
            @test jt.df == 1
            @test jt.J_stat >= 0
            @test 0 <= jt.p_value <= 1
        end
    end

    @testset "estimate_smm — identity weighting just-identified" begin
        # Just-identified identity weighting: sandwich formula with Omega computation
        _suppress_warnings() do
            rng = Random.MersenneTwister(123)
            y = zeros(300)
            for t in 2:300
                y[t] = 0.5 * y[t-1] + randn(rng)
            end
            data = reshape(y, :, 1)

            function sim_fn(theta, T_periods, burn; rng=Random.default_rng())
                rho = theta[1]
                sim = zeros(T_periods + burn)
                for t in 2:(T_periods + burn)
                    sim[t] = rho * sim[t-1] + randn(rng)
                end
                reshape(sim[(burn+1):end], :, 1)
            end

            # 1 variable, 1 lag: k*(k+1)/2 + k*lags = 1 + 1 = 2 moments for 1 param
            # This is overidentified (2 > 1), but identity weighting takes the sandwich path
            result = estimate_smm(sim_fn, d -> autocovariance_moments(d; lags=1),
                                  [0.3], data;
                                  sim_ratio=3, burn=25, weighting=:identity,
                                  rng=Random.MersenneTwister(55))
            @test result isa SMMModel{Float64}
            @test result.weighting.method == :identity
            @test all(isfinite, stderror(result))
        end
    end

    @testset "SMMModel assertion failures" begin
        W = Matrix{Float64}(I, 3, 3)
        weighting = MacroEconometricModels.GMMWeighting{Float64}(:identity, 100, 1e-8)

        # n_moments < n_params
        @test_throws AssertionError MacroEconometricModels.SMMModel{Float64}(
            [0.5, 0.3, 0.1], [0.01 0.0 0.0; 0.0 0.02 0.0; 0.0 0.0 0.01],
            2, 3, 100, weighting,
            Matrix{Float64}(I, 2, 2), zeros(2),
            0.0, 1.0, true, 5, 3
        )

        # sim_ratio < 1
        @test_throws AssertionError MacroEconometricModels.SMMModel{Float64}(
            [0.5], reshape([0.01], 1, 1),
            3, 1, 100, weighting, W, zeros(3),
            0.0, 1.0, true, 5, 0
        )

        # theta length mismatch
        @test_throws AssertionError MacroEconometricModels.SMMModel{Float64}(
            [0.5, 0.3], reshape([0.01], 1, 1),
            3, 1, 100, weighting, W, zeros(3),
            0.0, 1.0, true, 5, 3
        )

        # W size mismatch
        @test_throws AssertionError MacroEconometricModels.SMMModel{Float64}(
            [0.5], reshape([0.01], 1, 1),
            3, 1, 100, weighting,
            Matrix{Float64}(I, 2, 2), zeros(3),
            0.0, 1.0, true, 5, 3
        )

        # g_bar length mismatch
        @test_throws AssertionError MacroEconometricModels.SMMModel{Float64}(
            [0.5], reshape([0.01], 1, 1),
            3, 1, 100, weighting, W, zeros(2),
            0.0, 1.0, true, 5, 3
        )
    end

    @testset "SMM StatsAPI completeness" begin
        theta = [0.5]
        vcov_mat = reshape([0.01], 1, 1)
        W = Matrix{Float64}(I, 3, 3)
        g_bar = zeros(3)
        weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)
        smm = MacroEconometricModels.SMMModel{Float64}(
            theta, vcov_mat, 3, 1, 200, weighting, W, g_bar,
            1.5, 0.3, true, 10, 5
        )

        @test StatsAPI.dof(smm) == 1
        @test StatsAPI.islinear(smm) == false
        @test MacroEconometricModels.is_overidentified(smm) == true
        @test MacroEconometricModels.overid_df(smm) == 2

        # confint
        ci = confint(smm)
        @test size(ci) == (1, 2)
        @test ci[1, 1] < 0.5 < ci[1, 2]

        ci90 = confint(smm; level=0.90)
        @test ci90[1, 2] - ci90[1, 1] < ci[1, 2] - ci[1, 1] + 1e-10
    end

    @testset "SMMModel show with just-identified (no J-test block)" begin
        theta = [0.5]
        vcov_mat = reshape([0.01], 1, 1)
        W = Matrix{Float64}(I, 1, 1)
        g_bar = zeros(1)
        weighting = MacroEconometricModels.GMMWeighting{Float64}(:identity, 100, 1e-8)
        smm = MacroEconometricModels.SMMModel{Float64}(
            theta, vcov_mat, 1, 1, 100, weighting, W, g_bar,
            0.0, 1.0, false, 50, 3
        )

        io = IOBuffer()
        show(io, smm)
        str = String(take!(io))
        @test occursin("SMM", str)
        @test occursin("Converged", str)
        @test occursin("No", str)  # converged = false
        # Just-identified: no Hansen J-test block
        @test !occursin("Hansen J-test", str)
    end

    @testset "SMMModel show with overidentified (J-test block)" begin
        theta = [0.5]
        vcov_mat = reshape([0.01], 1, 1)
        W = Matrix{Float64}(I, 3, 3)
        g_bar = zeros(3)
        weighting = MacroEconometricModels.GMMWeighting{Float64}(:two_step, 100, 1e-8)
        smm = MacroEconometricModels.SMMModel{Float64}(
            theta, vcov_mat, 3, 1, 200, weighting, W, g_bar,
            2.5, 0.28, true, 10, 5
        )

        io = IOBuffer()
        show(io, smm)
        str = String(take!(io))
        @test occursin("Hansen J-test", str)
        @test occursin("J-statistic", str)
        @test occursin("P-value", str)
    end

    @testset "autocovariance_moments with integer input" begin
        # Exercises the Real-to-Float64 conversion fallback on line 212
        data_int = ones(Int, 50, 2)
        m = autocovariance_moments(data_int; lags=1)
        @test length(m) == 5
        @test all(m .== 0.0)  # constant data has zero (co)variance
    end

    @testset "estimate_smm moment count assertion" begin
        # n_moments < n_params should fail
        _suppress_warnings() do
            rng = Random.MersenneTwister(42)
            data = randn(rng, 100, 1)

            function sim_fn_bad(theta, T_periods, burn; rng=Random.default_rng())
                reshape(randn(rng, T_periods), :, 1)
            end

            # moments_fn returns 1 moment but theta has 3 params
            @test_throws AssertionError estimate_smm(
                sim_fn_bad,
                d -> [mean(d)],
                [0.1, 0.2, 0.3], data;
                sim_ratio=2, burn=10,
                rng=Random.MersenneTwister(1))
        end
    end

    @testset "smm_data_covariance" begin
        rng = Random.MersenneTwister(42)
        data = randn(rng, 200, 2)
        Omega = MacroEconometricModels.smm_data_covariance(
            data, d -> autocovariance_moments(d; lags=1); hac=false)
        @test size(Omega) == (5, 5)
        @test all(isfinite, Omega)

        # With HAC + explicit bandwidth
        Omega_hac = MacroEconometricModels.smm_data_covariance(
            data, d -> autocovariance_moments(d; lags=1); hac=true, bandwidth=3)
        @test size(Omega_hac) == (5, 5)
    end

    @testset "estimate_smm with bounds and identity weighting" begin
        # Combines bounds + identity weighting to exercise both paths simultaneously
        _suppress_warnings() do
            rng = Random.MersenneTwister(77)
            y = zeros(300)
            for t in 2:300
                y[t] = 0.6 * y[t-1] + 0.4 * randn(rng)
            end
            data = reshape(y, :, 1)

            function sim_bounded(theta, T_periods, burn; rng=Random.default_rng())
                rho = theta[1]
                sim = zeros(T_periods + burn)
                for t in 2:(T_periods + burn)
                    sim[t] = clamp(rho, -0.99, 0.99) * sim[t-1] + randn(rng)
                end
                reshape(sim[(burn+1):end], :, 1)
            end

            bounds = ParameterTransform([-1.0], [1.0])
            result = estimate_smm(sim_bounded, d -> autocovariance_moments(d; lags=1),
                                  [0.3], data;
                                  sim_ratio=3, burn=25, weighting=:identity,
                                  bounds=bounds,
                                  rng=Random.MersenneTwister(42))
            @test result isa SMMModel{Float64}
            @test -1.0 < result.theta[1] < 1.0
        end
    end
end

# ============================================================================
# 3. JuMP + Ipopt extension coverage
# ============================================================================

_jump_available = try
    @eval import JuMP
    @eval import Ipopt
    true
catch
    false
end

if _jump_available

@testset "JuMP Extension coverage" begin

    @testset "Constrained SS with initial_guess kwarg" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Provide explicit initial_guess (exercises that branch in the extension)
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, lower=-1.0)],
                initial_guess=[0.5])
            @test abs(spec_c.steady_state[1]) < 0.1
        end
    end

    @testset "Constrained SS with only upper bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Upper bound only — exercises the c.upper !== nothing branch
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, upper=1.0)])
            @test spec_c.steady_state[1] <= 1.0 + 1e-4
        end
    end

    @testset "Constrained PF with both bounds and NL constraint" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(10, 1)
            shocks[1, 1] = 2.0

            # Variable bound + nonlinear constraint together
            nlc = nonlinear_constraint(
                (y, y_lag, y_lead, e, theta) -> y[1] - 5.0;
                label="cap_y")
            pf = solve(spec; method=:perfect_foresight, T_periods=10,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, lower=-1.0), nlc])
            @test pf isa PerfectForesightPath
            @test size(pf.path) == (10, 1)
        end
    end

    @testset "Constrained PF with upper-only variable bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(15, 1)
            shocks[1, 1] = 3.0

            # Upper-only bound
            pf = solve(spec; method=:perfect_foresight, T_periods=15,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, upper=5.0)])
            @test pf isa PerfectForesightPath
            @test pf.converged
            @test all(pf.path[:, 1] .<= 5.0 + 1e-4)
        end
    end
end

end  # _jump_available

# ============================================================================
# 4. PATH extension coverage
# ============================================================================

_path_available = try
    @eval import PATHSolver
    true
catch
    false
end

if _path_available

@testset "PATH Extension coverage" begin

    @testset "MCP SS with initial_guess kwarg" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Provide explicit initial_guess (exercises that branch in PATH ext)
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, lower=-1.0)],
                solver=:path,
                initial_guess=[0.2])
            @test abs(spec_c.steady_state[1]) < 0.1
        end
    end

    @testset "MCP SS with upper-only bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Upper bound only on PATH
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, upper=1.0)],
                solver=:path)
            @test spec_c.steady_state[1] <= 1.0 + 1e-4
        end
    end

    @testset "MCP PF single-period (t==1 && t==T_periods)" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # T_periods=1 triggers the t==1 && t==T_periods branch (line 107)
            shocks = zeros(1, 1)
            shocks[1, 1] = 0.5
            pf = solve(spec; method=:perfect_foresight, T_periods=1,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, lower=-5.0)],
                       solver=:path)
            @test pf isa PerfectForesightPath
            @test size(pf.path) == (1, 1)
        end
    end

    @testset "MCP PF with upper bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(10, 1)
            shocks[1, 1] = 3.0

            pf = solve(spec; method=:perfect_foresight, T_periods=10,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, upper=0.5)],
                       solver=:path)
            @test pf isa PerfectForesightPath
            @test pf.converged
            @test all(pf.path[:, 1] .<= 0.5 + 1e-4)
        end
    end
end

end  # _path_available

# ============================================================================
# 5. GMM with ParameterTransform bounds (additional branch coverage)
# ============================================================================

@testset "GMM bounds coverage" begin

    @testset "estimate_gmm with shifted lower bound" begin
        Random.seed!(9006)
        n = 200
        # True mean is 5.0; use bounds [2, Inf) to exercise (a, Inf) branch via GMM
        data = 5.0 .+ 0.5 .* randn(n, 1)
        moment_fn(theta, d) = d .- theta[1]

        bounds = ParameterTransform([2.0], [Inf])
        result = estimate_gmm(moment_fn, [3.0], data;
                              weighting=:identity, bounds=bounds)
        @test result.theta[1] > 2.0
        @test isapprox(result.theta[1], 5.0, atol=0.3)
    end

    @testset "estimate_gmm with upper-only bound" begin
        Random.seed!(9007)
        n = 200
        # True mean is -3.0; use bounds (-Inf, 0] to exercise (-Inf, b) branch via GMM
        data = -3.0 .+ 0.5 .* randn(n, 1)
        moment_fn(theta, d) = d .- theta[1]

        bounds = ParameterTransform([-Inf], [0.0])
        result = estimate_gmm(moment_fn, [-2.0], data;
                              weighting=:identity, bounds=bounds)
        @test result.theta[1] < 0.0
        @test isapprox(result.theta[1], -3.0, atol=0.3)
    end

    @testset "estimate_gmm with shifted upper bound" begin
        Random.seed!(9008)
        n = 200
        # True mean is 1.0; use bounds (-Inf, 3) to exercise (-Inf, b) with b != 0
        data = 1.0 .+ 0.3 .* randn(n, 1)
        moment_fn(theta, d) = d .- theta[1]

        bounds = ParameterTransform([-Inf], [3.0])
        result = estimate_gmm(moment_fn, [0.5], data;
                              weighting=:identity, bounds=bounds)
        @test result.theta[1] < 3.0
        @test isapprox(result.theta[1], 1.0, atol=0.3)
    end
end
