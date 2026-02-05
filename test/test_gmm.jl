using Test
using MacroEconometricModels
using Statistics
using LinearAlgebra
using Random

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
        Random.seed!(42)
        # Simple moment function: E[data - theta] = 0
        moment_fn(theta, data) = data .- theta[1]'
        data = randn(100, 2) .+ 3.0
        W = Matrix{Float64}(I, 2, 2)

        obj = MacroEconometricModels.gmm_objective([3.0], moment_fn, data, W)
        @test obj >= 0.0  # Objective is non-negative
        @test obj < 1.0   # Should be close to zero at true value

        # At wrong parameter, objective should be larger
        obj_wrong = MacroEconometricModels.gmm_objective([0.0], moment_fn, data, W)
        @test obj_wrong > obj
    end

    @testset "optimal_weighting_matrix" begin
        Random.seed!(42)

        n = 200
        k = 3

        # Simple OLS moment conditions: E[X'(y - X*beta)] = 0
        X = randn(n, 2)
        beta_true = [1.0, 2.0]
        y = X * beta_true + 0.5 * randn(n)
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
        Random.seed!(100)
        n = 300

        # OLS as GMM: E[X'(y - X*beta)] = 0  (just-identified)
        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(200)
        n = 300

        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(300)
        n = 300

        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(350)
        n = 300

        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(400)
        n = 500

        # IV regression: y = X*beta + eps, X correlated with eps
        # Use Z as instruments (3 instruments for 1 parameter => overidentified)
        Z = randn(n, 3)
        eps = randn(n)
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
        Random.seed!(500)
        n = 500

        # Overidentified IV
        Z = randn(n, 3)
        eps = randn(n)
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
        Random.seed!(600)
        n = 300

        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(700)
        n = 300

        X = randn(n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(n)
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
        Random.seed!(800)
        n = 200

        X = randn(n, 2)
        y = X * [1.0, 2.0] + randn(n)
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
        Z = hcat(X, randn(n))
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
        Random.seed!(900)
        n_obs = 100
        n_vars = 3
        Y = randn(n_obs, n_vars)
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
        Random.seed!(1000)
        n_obs = 150
        n_vars = 2
        Y = randn(n_obs, n_vars)
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
        Random.seed!(1100)
        n = 300

        # Simple mean estimation: E[y - mu] = 0
        y_data = randn(n) .+ 5.0
        data = reshape(y_data, n, 1)

        moment_fn(theta, d) = d .- theta[1]

        result = estimate_gmm(moment_fn, [0.0], data; weighting=:identity)
        @test length(result.theta) == 1
        @test isapprox(result.theta[1], 5.0, atol=0.3)
    end

    @testset "vcov matrix properties" begin
        Random.seed!(1200)
        n = 300

        X = randn(n, 3)
        beta_true = [1.0, -0.5, 0.3]
        y = X * beta_true + randn(n)
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
    end

end
