# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Statistics
using LinearAlgebra
using Random
import StatsAPI

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "GMM Estimation" begin

    # =========================================================================
    # Test Setup: IV regression with known solution
    # y = X*beta + eps, E[Z'eps] = 0
    # Overidentified: 3 instruments for 2 parameters
    # =========================================================================

    @testset "GMMWeighting type" begin
        gw = MacroEconometricModels.GMMWeighting()
        @test gw.method == :two_step
        @test gw.max_iter == 100
        @test gw.tol == 1e-8

        gw2 = MacroEconometricModels.GMMWeighting(method=:identity)
        @test gw2.method == :identity

        gw3 = MacroEconometricModels.GMMWeighting(method=:iterated, max_iter=50, tol=1e-6)
        @test gw3.method == :iterated
        @test gw3.max_iter == 50
        @test gw3.tol == 1e-6

        # Invalid method
        @test_throws ArgumentError MacroEconometricModels.GMMWeighting(method=:invalid)
    end

    @testset "identity_weighting" begin
        W = MacroEconometricModels.identity_weighting(4)
        @test W == Matrix{Float64}(I, 4, 4)
        @test eltype(W) == Float64

        W32 = MacroEconometricModels.identity_weighting(3, Float32)
        @test W32 == Matrix{Float32}(I, 3, 3)
        @test eltype(W32) == Float32
    end

    @testset "numerical_gradient" begin
        # Test with known analytical gradient
        # f(x) = [x1^2 + x2, x1*x2], gradient = [2x1 x2; 1 x1]
        f(x) = [x[1]^2 + x[2], x[1] * x[2]]
        x0 = [2.0, 3.0]

        J = MacroEconometricModels.numerical_gradient(f, x0)
        # Analytical: [2*2 1; 3 2] = [4 1; 3 2]
        @test size(J) == (2, 2)
        @test isapprox(J[1, 1], 4.0, atol=1e-5)
        @test isapprox(J[1, 2], 1.0, atol=1e-5)
        @test isapprox(J[2, 1], 3.0, atol=1e-5)
        @test isapprox(J[2, 2], 2.0, atol=1e-5)

        # Test with single-parameter function
        g(x) = [x[1]^3]
        J_g = MacroEconometricModels.numerical_gradient(g, [1.0])
        @test size(J_g) == (1, 1)
        @test isapprox(J_g[1, 1], 3.0, atol=1e-5)  # d/dx(x^3) = 3x^2

        # Multivariate output, multivariate input
        h(x) = [sum(x), prod(x), x[1] - x[2]]
        J_h = MacroEconometricModels.numerical_gradient(h, [1.0, 2.0, 3.0])
        @test size(J_h) == (3, 3)
        @test isapprox(J_h[1, :], [1.0, 1.0, 1.0], atol=1e-5)  # d(sum)/dx_i = 1
    end

    @testset "gmm_objective" begin
        rng = Random.MersenneTwister(42)
        # Simple moment function: E[data - theta] = 0
        moment_fn(theta, data) = data .- theta[1]'
        data = randn(rng, 100, 2) .+ 3.0
        W = Matrix{Float64}(I, 2, 2)

        obj = MacroEconometricModels.gmm_objective([3.0], moment_fn, data, W)
        @test obj >= 0.0  # Objective is non-negative
        @test obj < 1.0   # Should be close to zero at true value

        # At wrong parameter, objective should be larger
        obj_wrong = MacroEconometricModels.gmm_objective([0.0], moment_fn, data, W)
        @test obj_wrong > obj
    end

    @testset "optimal_weighting_matrix" begin
        rng = Random.MersenneTwister(42)

        n = 200
        k = 3

        # Simple OLS moment conditions: E[X'(y - X*beta)] = 0
        X = randn(rng, n, 2)
        beta_true = [1.0, 2.0]
        y = X * beta_true + 0.5 * randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        W = MacroEconometricModels.optimal_weighting_matrix(moment_fn, beta_true, data; hac=false)
        @test size(W) == (2, 2)
        @test issymmetric(round.(W, digits=10))  # Should be approximately symmetric
        @test all(eigvals(Symmetric(W)) .> -1e-8)  # Should be PSD

        # With HAC
        W_hac = MacroEconometricModels.optimal_weighting_matrix(moment_fn, beta_true, data; hac=true)
        @test size(W_hac) == (2, 2)
    end

    @testset "estimate_gmm - identity weighting" begin
        rng = Random.MersenneTwister(100)
        n = 300

        # OLS as GMM: E[X'(y - X*beta)] = 0  (just-identified)
        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:identity)

        @test result isa MacroEconometricModels.GMMModel
        @test length(result.theta) == 2
        @test result.n_params == 2
        @test result.n_moments == 2
        @test result.n_obs == n
        @test isapprox(result.theta[1], beta_true[1], atol=0.3)
        @test isapprox(result.theta[2], beta_true[2], atol=0.3)
        @test result.J_stat >= 0
        @test result.converged || result.iterations > 0
    end

    @testset "estimate_gmm - two_step weighting" begin
        rng = Random.MersenneTwister(200)
        n = 300

        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:two_step)

        @test result isa MacroEconometricModels.GMMModel
        @test isapprox(result.theta[1], beta_true[1], atol=0.3)
        @test isapprox(result.theta[2], beta_true[2], atol=0.3)
        @test result.weighting.method == :two_step
    end

    @testset "estimate_gmm - iterated weighting" begin
        rng = Random.MersenneTwister(300)
        n = 300

        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:iterated)

        @test result isa MacroEconometricModels.GMMModel
        @test isapprox(result.theta[1], beta_true[1], atol=0.3)
        @test isapprox(result.theta[2], beta_true[2], atol=0.3)
        @test result.weighting.method == :iterated
    end

    @testset "estimate_gmm - optimal weighting" begin
        rng = Random.MersenneTwister(350)
        n = 300

        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:optimal)

        @test result isa MacroEconometricModels.GMMModel
        @test isapprox(result.theta[1], beta_true[1], atol=0.3)
        @test isapprox(result.theta[2], beta_true[2], atol=0.3)
        @test result.weighting.method == :optimal
    end

    @testset "estimate_gmm - overidentified IV" begin
        rng = Random.MersenneTwister(400)
        n = 500

        # IV regression: y = X*beta + eps, X correlated with eps
        # Use Z as instruments (3 instruments for 1 parameter => overidentified)
        Z = randn(rng, n, 3)
        eps = randn(rng, n)
        X = Z * [0.5, 0.3, 0.2] + 0.5 * eps  # X correlated with eps
        beta_true = [2.0]
        y = X .* beta_true[1] + eps

        data = hcat(y, X, Z)

        # Moment conditions: E[Z' * (y - X*beta)] = 0
        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:2]
            Z_d = d[:, 3:5]
            resid = y_d - X_d * theta
            Z_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0], data; weighting=:two_step)

        @test result.n_moments == 3
        @test result.n_params == 1
        @test MacroEconometricModels.is_overidentified(result)
        @test MacroEconometricModels.overid_df(result) == 2
        @test isapprox(result.theta[1], beta_true[1], atol=0.5)
    end

    @testset "j_test" begin
        rng = Random.MersenneTwister(500)
        n = 500

        # Overidentified IV
        Z = randn(rng, n, 3)
        eps = randn(rng, n)
        X = Z * [0.5, 0.3, 0.2] + 0.5 * eps
        beta_true = [2.0]
        y = X .* beta_true[1] + eps

        data = hcat(y, X, Z)
        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:2]
            Z_d = d[:, 3:5]
            resid = y_d - X_d * theta
            Z_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0], data; weighting=:two_step)
        jt = MacroEconometricModels.j_test(result)

        @test jt.df == 2
        @test jt.J_stat >= 0
        @test 0 <= jt.p_value <= 1
        @test jt.reject_05 isa Bool

        # Just-identified case: J-test not applicable
        moment_fn_ji(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:2]
            resid = y_d - X_d * theta
            X_d .* resid
        end
        data_ji = hcat(y, X)
        result_ji = estimate_gmm(moment_fn_ji, [0.0], data_ji; weighting=:identity)
        jt_ji = MacroEconometricModels.j_test(result_ji)

        @test jt_ji.df == 0
        @test jt_ji.J_stat == 0.0
        @test jt_ji.p_value == 1.0
        @test jt_ji.reject_05 == false
        @test haskey(jt_ji, :message)
    end

    @testset "gmm_summary" begin
        rng = Random.MersenneTwister(600)
        n = 300

        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:two_step)
        s = MacroEconometricModels.gmm_summary(result)

        @test length(s.theta) == 2
        @test length(s.se) == 2
        @test all(s.se .> 0)
        @test length(s.t_stats) == 2
        @test length(s.p_values) == 2
        @test all(0 .<= s.p_values .<= 1)
        @test s.n_moments == 2
        @test s.n_params == 2
        @test s.n_obs == n
        @test s.weighting == :two_step
        @test s.converged isa Bool
        @test s.j_test isa NamedTuple
    end

    @testset "GMMModel StatsAPI interface" begin
        rng = Random.MersenneTwister(700)
        n = 300

        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:two_step)

        @test coef(result) == result.theta
        @test vcov(result) == result.vcov
        @test nobs(result) == n
        @test dof(result) == 2
        @test islinear(result) == false

        se = stderror(result)
        @test length(se) == 2
        @test all(se .> 0)

        ci = confint(result)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .< ci[:, 2])  # Lower < upper

        ci_90 = confint(result; level=0.90)
        # 90% CI should be narrower than 95% CI
        @test all(ci_90[:, 2] - ci_90[:, 1] .<= ci[:, 2] - ci[:, 1] .+ 1e-10)
    end

    @testset "is_overidentified and overid_df" begin
        rng = Random.MersenneTwister(800)
        n = 200

        X = randn(rng, n, 2)
        y = X * [1.0, 2.0] + randn(rng, n)
        data = hcat(y, X)

        # Just-identified
        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:identity)
        @test !MacroEconometricModels.is_overidentified(result)
        @test MacroEconometricModels.overid_df(result) == 0

        # Overidentified (add extra instrument)
        Z = hcat(X, randn(rng, n))
        data_ov = hcat(y, X, Z)
        moment_fn_ov(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            Z_d = d[:, 4:6]
            resid = y_d - X_d * theta
            Z_d .* resid
        end

        result_ov = estimate_gmm(moment_fn_ov, [0.0, 0.0], data_ov; weighting=:identity)
        @test MacroEconometricModels.is_overidentified(result_ov)
        @test MacroEconometricModels.overid_df(result_ov) == 1
    end

    @testset "lp_gmm_moments" begin
        rng = Random.MersenneTwister(900)
        n_obs = 100
        n_vars = 3
        Y = randn(rng, n_obs, n_vars)
        lags = 2
        shock_var = 1
        h = 1

        k = 2 + n_vars * lags  # intercept + shock + lagged controls
        theta = zeros(k)

        moments = MacroEconometricModels.lp_gmm_moments(Y, shock_var, h, theta, lags)

        t_start = lags + 1
        t_end = n_obs - h
        T_eff = t_end - t_start + 1
        @test size(moments) == (T_eff, k)
        @test !any(isnan, moments)
    end

    @testset "estimate_lp_gmm" begin
        rng = Random.MersenneTwister(1000)
        n_obs = 150
        n_vars = 2
        Y = randn(rng, n_obs, n_vars)
        horizon = 4

        # LP-GMM may fail if optimal_weighting_matrix returns Hermitian (type mismatch)
        # Use identity weighting to avoid this issue
        models = MacroEconometricModels.estimate_lp_gmm(Y, 1, horizon; lags=2, weighting=:identity)

        @test length(models) == horizon + 1
        for (h, m) in enumerate(models)
            @test m isa MacroEconometricModels.GMMModel
            @test length(m.theta) == 2 + n_vars * 2  # intercept + shock + 2 vars * 2 lags
        end
    end

    @testset "Single parameter estimation" begin
        rng = Random.MersenneTwister(1100)
        n = 300

        # Simple mean estimation: E[y - mu] = 0
        y_data = randn(rng, n) .+ 5.0
        data = reshape(y_data, n, 1)

        moment_fn(theta, d) = d .- theta[1]

        result = estimate_gmm(moment_fn, [0.0], data; weighting=:identity)
        @test length(result.theta) == 1
        @test isapprox(result.theta[1], 5.0, atol=0.3)
    end

    @testset "vcov matrix properties" begin
        rng = Random.MersenneTwister(1200)
        n = 300

        X = randn(rng, n, 3)
        beta_true = [1.0, -0.5, 0.3]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:4]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, zeros(3), data; weighting=:two_step)

        V = result.vcov
        @test size(V) == (3, 3)
        @test isapprox(V, V', atol=1e-10)  # Symmetric
        @test all(diag(V) .>= 0)  # Non-negative diagonal

        # DGP-08 (#797, :462): vcov against the hand-computed sandwich (exact —
        # realized max rel dev 1e-9) and the homoskedastic σ²(X'X)⁻¹ limit.
        # The analytic leg is loose by design: sandwich sampling noise is
        # ~30% at n=300 — it pins the scale/shape, the sandwich pins the math.
        ro = estimate_gmm(moment_fn, zeros(3), data; weighting=:two_step, hac=false)
        e = y - X * ro.theta
        G = X .* e
        Om = (G' * G) / n
        V_hand = n * inv(Symmetric(X' * X)) * Om * inv(Symmetric(X' * X))
        @test ro.vcov ≈ V_hand rtol=1e-6
        s2 = var(e)
        @test ro.vcov ≈ s2 * inv(Symmetric(X' * X)) rtol=0.5
    end

    @testset "Iterated weighting" begin
        rng = Random.MersenneTwister(3201)
        n = 100
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [1.0, -0.5, 0.3]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:4]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, zeros(3), data; weighting=:iterated, max_iter=20)
        @test result isa GMMModel
        @test isfinite(result.J_stat)
        @test length(result.theta) == 3
    end

    @testset "Identity weighting (one-step)" begin
        rng = Random.MersenneTwister(3202)
        n = 100
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [1.0, -0.5, 0.3]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:4]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, zeros(3), data; weighting=:identity)
        @test result isa GMMModel
        @test length(result.theta) == 3
    end

    @testset "J-test direct" begin
        rng = Random.MersenneTwister(3203)
        n = 200
        X = hcat(ones(n), randn(rng, n, 3))  # 4 instruments for 3 parameters = overid
        beta_true = [1.0, -0.5, 0.3]
        y = X[:, 1:3] * beta_true + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:5]
            resid = y_d - d[:, 2:4] * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, zeros(3), data; weighting=:two_step)
        j = j_test(result)
        @test j isa NamedTuple
        @test haskey(j, :J_stat)
        @test haskey(j, :p_value)
        @test j.J_stat >= 0
        @test 0 <= j.p_value <= 1
    end

    @testset "StatsAPI methods on GMMModel" begin
        rng = Random.MersenneTwister(3204)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        y = X * [1.0, 0.5] + randn(rng, n)
        data = hcat(y, X)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            resid = y_d - X_d * theta
            X_d .* resid
        end

        result = estimate_gmm(moment_fn, zeros(2), data; weighting=:two_step)
        @test StatsAPI.nobs(result) == n
        @test length(StatsAPI.coef(result)) == 2
    end

end

# =============================================================================
# T089 (#188): M-29 identity-weighting J p-value, M-31 numerical_gradient step
# =============================================================================

@testset "T089: GMM identity J p-value + numerical_gradient step kwarg" begin

    @testset "M-29: J p-value invalid under identity weighting" begin
        rng = Random.MersenneTwister(18901)
        n = 300
        X = randn(rng, n, 2)
        y = X * [1.0, -0.5] + randn(rng, n)
        Z = hcat(X, randn(rng, n))  # extra instrument -> overidentified
        data = hcat(y, X, Z)

        moment_fn(theta, d) = begin
            y_d = d[:, 1]
            X_d = d[:, 2:3]
            Z_d = d[:, 4:6]
            resid = y_d - X_d * theta
            Z_d .* resid
        end

        r_id = estimate_gmm(moment_fn, zeros(2), data; weighting=:identity)
        @test MacroEconometricModels.is_overidentified(r_id)
        @test r_id.J_stat >= 0 && isfinite(r_id.J_stat)
        @test isnan(r_id.J_pvalue)

        jt = MacroEconometricModels.j_test(r_id)
        @test isnan(jt.p_value)
        @test jt.reject_05 == false
        @test haskey(jt, :message) && occursin("identity", jt.message)

        io = IOBuffer()
        show(io, r_id)
        out = String(take!(io))
        @test occursin("n/a", out)
        @test occursin("identity weighting", out)

        # Efficient weighting keeps a valid chi-squared p-value
        r_ts = estimate_gmm(moment_fn, zeros(2), data; weighting=:two_step)
        @test 0.0 <= r_ts.J_pvalue <= 1.0
        io2 = IOBuffer()
        show(io2, r_ts)
        @test !occursin("n/a", String(take!(io2)))
    end

    @testset "M-31: numerical_gradient step kwarg" begin
        f(x) = [x[1]^2 + 2x[2], 3x[1] * x[2]]
        x0 = [1.0, 2.0]
        analytic = [2.0 2.0; 6.0 3.0]

        J_default = numerical_gradient(f, x0)
        @test J_default ≈ analytic atol = 1e-6

        J_step = numerical_gradient(f, x0; step=1e-5)
        @test J_step ≈ analytic atol = 1e-6
    end

end

# =============================================================================
# DGP-08 (#797): truth arms on the shared dgp_gmm / dgp_var simulators
# =============================================================================

@testset "DGP-08 weighting values on hetero overidentified IV" begin
    # dgp_gmm with heteroskedastic errors: one-step and efficient weights must
    # genuinely differ; iterated must agree with two-step; the optimal matrix
    # must equal the hand-computed Ω⁻¹. Realized: W diff 0.49, iterated≈2step
    # to 1e-6, optimal==hand exactly.
    # Deliberately NO "two-step SE smaller" assertion: the reported SEs estimate
    # different probability limits (sandwich vs efficient-form), and an R=25 MC
    # gives dispersion ratio 1.006 — the efficiency gap is below MC resolution
    # on this design. Comparing their magnitudes would test noise, not theory.
    rng = Random.MersenneTwister(42)
    d = dgp_gmm(rng; kind=:iv, beta=[1.0, 0.5], n=1000, hetero=true, overid_k=2)
    data = hcat(d.y, d.X, d.Z)
    moment_fn(theta, dd) = dd[:, 4:6] .* (dd[:, 1] - dd[:, 2:3] * theta)

    r_id = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:identity)
    r_ts = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:two_step, hac=false)
    r_it = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:iterated, hac=false)

    @test r_id.theta ≈ d.beta atol=0.15
    @test r_ts.theta ≈ d.beta atol=0.15
    @test maximum(abs, r_ts.W - Matrix{Float64}(I, 3, 3)) > 0.3
    @test r_it.theta ≈ r_ts.theta atol=1e-3

    G = moment_fn(d.beta, data)
    Gc = G .- mean(G, dims=1)
    W_hand = inv(Symmetric((Gc' * Gc) / size(G, 1)))
    W_opt = MacroEconometricModels.optimal_weighting_matrix(moment_fn, d.beta, data;
                                                            hac=false)
    @test W_opt ≈ W_hand rtol=1e-8
end

@testset "DGP-08 J-test size and power" begin
    # Size: valid design over 40 seeds (12 in FAST), rejection rate ≤ 0.15
    # (expected 0.05, se ≈ 0.034 — ~3σ headroom). Power: invalid_k=1 puts the
    # violation on the overidentifying instrument (see dgp_gmm docstring) and
    # must reject at 1% (realized J ≈ 217).
    moment_fn(theta, dd) = dd[:, 4:6] .* (dd[:, 1] - dd[:, 2:3] * theta)
    nreps = FAST ? 12 : 40
    rej = 0
    for s in 1:nreps
        dd = dgp_gmm(Random.MersenneTwister(2000 + s); kind=:iv, beta=[1.0, 0.5],
                     n=1000, hetero=true, overid_k=2)
        res = estimate_gmm(moment_fn, [0.0, 0.0], hcat(dd.y, dd.X, dd.Z);
                           weighting=:two_step, hac=false)
        jt = MacroEconometricModels.j_test(res)
        @test jt.df == 1
        rej += jt.p_value < 0.05
    end
    @test rej / nreps <= (FAST ? 0.25 : 0.15)

    dd_bad = dgp_gmm(Random.MersenneTwister(7); kind=:iv, beta=[1.0, 0.5],
                     n=1000, hetero=true, overid_k=2, invalid_k=1)
    jt_bad = MacroEconometricModels.j_test(estimate_gmm(moment_fn, [0.0, 0.0],
        hcat(dd_bad.y, dd_bad.X, dd_bad.Z); weighting=:two_step, hac=false))
    @test jt_bad.p_value < 0.01

    # df tracks the instrument count: overid_k=4 → 5 moments − 2 params = 3.
    dd4 = dgp_gmm(Random.MersenneTwister(7); kind=:iv, beta=[1.0, 0.5],
                  n=1000, hetero=true, overid_k=4)
    moment_fn5(theta, dd) = dd[:, 4:8] .* (dd[:, 1] - dd[:, 2:3] * theta)
    jt4 = MacroEconometricModels.j_test(estimate_gmm(moment_fn5, [0.0, 0.0],
        hcat(dd4.y, dd4.X, dd4.Z); weighting=:two_step, hac=false))
    @test jt4.df == 3
end

@testset "DGP-08 LP-GMM IRF recovery on dgp_var" begin
    # The stale Hermitian comment is retired: estimate_lp_gmm works under
    # :two_step (verified — optimal_weighting_matrix returns Matrix). LP-GMM is
    # just-identified (Z = X), so both weightings must agree AND recover the
    # closed-form IRF (realized max dev 0.04 at T=2000; bound 0.1).
    rng = Random.MersenneTwister(3)
    A = [0.5 0.1; 0.2 0.4]
    B0 = [1.0 0.0; 0.3 1.0]
    Y = dgp_var(rng; A=A, B0=B0, T=2000).Y
    models = MacroEconometricModels.estimate_lp_gmm(Y, 1, 4; lags=1, weighting=:two_step)
    models_id = MacroEconometricModels.estimate_lp_gmm(Y, 1, 4; lags=1, weighting=:identity)
    IRF = var_irf(A, B0, 4)
    @test length(models) == 5
    for h in 0:4
        @test models[h + 1].theta[2] ≈ IRF[h + 1, 1, 1] atol=0.1
        @test models[h + 1].theta[2] ≈ models_id[h + 1].theta[2] atol=1e-3
    end

    # lp_gmm_moments at non-zero θ against a hand-stacked row (k = 2 + 2·1).
    th = [0.1, 0.5, -0.2, 0.3]
    M = MacroEconometricModels.lp_gmm_moments(Y, 1, 2, th, 1)
    @test size(M) == (2000 - 2 - 1, 4)
    x = [1.0, Y[2, 1], Y[1, 1], Y[1, 2]]
    @test vec(M[1, :]) ≈ x .* (Y[4, 1] - dot(x, th))
end

@testset "DGP-08 weak identification design" begin
    # pi1 = 0.07 delivers first-stage F ≈ 3.8 (asserted 2–6). The estimators run
    # without complaint — no first-stage diagnostic is reported anywhere, which
    # is filed as a separate feature (#815).
    rw = Random.MersenneTwister(9)
    dw = dgp_gmm(rw; kind=:iv, beta=[1.0, 0.5], n=1000, hetero=false,
                 overid_k=2, pi1=0.07)
    x = dw.X[:, 2]
    Z = dw.Z
    k = size(Z, 2)
    Pz = Z * inv(Symmetric(Z' * Z)) * Z'
    Fstat = ((dot(x, Pz * x) - dot(x, ones(length(x)))^2 / length(x)) / (k - 1)) /
            ((dot(x, x) - dot(x, Pz * x)) / (length(x) - k))
    @test 2.0 < Fstat < 6.0

    moment_fn(theta, dd) = dd[:, 4:6] .* (dd[:, 1] - dd[:, 2:3] * theta)
    res = estimate_gmm(moment_fn, [0.0, 0.0], hcat(dw.y, dw.X, dw.Z);
                       weighting=:two_step, hac=false)
    @test res isa MacroEconometricModels.GMMModel
    @test all(isfinite, res.theta)
end
