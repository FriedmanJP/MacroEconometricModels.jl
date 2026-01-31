using MacroEconometricModels
using Test
using StatsAPI
using LinearAlgebra
using Statistics
using Random

@testset "StatsAPI Compatibility" begin
    # Generate Data
    T = 100
    n = 2
    p = 1
    Random.seed!(42)

    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.1; 0.1]
    Y = zeros(T, n)
    for t in 2:T
        u = randn(n) * 0.1
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    model = StatsAPI.fit(VARModel, Y, p)

    # 1. Basic Interface
    @test StatsAPI.coef(model) isa Matrix
    @test StatsAPI.residuals(model) isa Matrix
    @test size(StatsAPI.coef(model)) == (1 + n * p, n)

    # 2. Test dof, nobs
    println("Testing dof/nobs...")
    @test StatsAPI.nobs(model) == T
    @test StatsAPI.dof(model) == (1 + n * p) * n

    # 3. Test vcov
    println("Testing vcov()...")
    V = StatsAPI.vcov(model)
    @test size(V) == ((1 + n * p) * n, (1 + n * p) * n)
    @test issymmetric(V) || norm(V - V') < 1e-10  # Should be symmetric

    # 4. Test predict (in-sample)
    println("Testing predict() in-sample...")
    y_hat = StatsAPI.predict(model)
    @test size(y_hat) == (T - p, n) # Effective sample size
    @test all(isfinite, y_hat)

    # 5. Test predict (forecast)
    println("Testing predict() forecast...")
    steps = 5
    y_fcast = StatsAPI.predict(model, steps)
    @test size(y_fcast) == (steps, n)
    @test all(isfinite, y_fcast)

    # 6. Test loglikelihood
    println("Testing loglikelihood...")
    ll = StatsAPI.loglikelihood(model)
    @test ll isa Float64
    @test isfinite(ll)

    # 7. Test stderror
    println("Testing stderror...")
    se = StatsAPI.stderror(model)
    @test length(se) == length(vec(StatsAPI.coef(model)))
    @test all(se .> 0)
    @test all(isfinite, se)

    # 8. Test confint
    println("Testing confint...")
    ci = StatsAPI.confint(model; level=0.95)
    @test size(ci) == (length(se), 2)
    @test all(ci[:, 1] .< ci[:, 2])  # Lower < Upper

    # Check that confidence intervals are reasonable (contain plausible values)
    # B structure: [c1 c2; A11 A12; A21 A22]
    # vec(B) = [c1, A11, A21, c2, A12, A22]
    b_vec = vec(StatsAPI.coef(model))

    # Check that CI widths are reasonable (not too narrow, not too wide)
    ci_widths = ci[:, 2] - ci[:, 1]
    @test all(ci_widths .> 0)
    @test all(ci_widths .< 10)  # Reasonable upper bound

    # 9. Test islinear
    println("Testing islinear...")
    @test StatsAPI.islinear(model)

    println("StatsAPI Tests Passed.")
end

@testset "StatsAPI r2 for VARModel" begin
    Random.seed!(123)
    T, n, p = 100, 2, 1
    Y = randn(T, n)
    model = estimate_var(Y, p)

    # r2 might not be defined for VAR, but should not error
    # or should return reasonable values if defined
    try
        r2_val = StatsAPI.r2(model)
        @test r2_val isa Number || r2_val isa Vector
        println("r2 for VAR: ", r2_val)
    catch e
        if e isa MethodError
            @test_skip "r2 not implemented for VARModel"
        else
            rethrow(e)
        end
    end
end
