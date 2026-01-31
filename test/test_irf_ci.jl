using MacroEconometricModels
using Test
using Random
using LinearAlgebra
using Statistics

@testset "IRF Confidence Intevals" begin
    # Generate Data
    T = 100
    n = 2
    p = 1
    Random.seed!(123)

    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]
    Y = zeros(T, n)
    for t in 2:T
        u = randn(n)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    model = estimate_var(Y, p)

    # 1. Test Frequentist Bootstrap
    println("Testing Bootstrap IRF...")
    irf_boot = irf(model, 5; method=:cholesky, ci_type=:bootstrap, reps=50, conf_level=0.90)

    @test irf_boot isa ImpulseResponse
    @test irf_boot.ci_type == :bootstrap
    @test !isempty(irf_boot.ci_lower)
    @test !isempty(irf_boot.ci_upper)
    @test size(irf_boot.ci_lower) == (5, n, n)
    @test all(irf_boot.ci_lower .<= irf_boot.ci_upper)

    # 2. Test Frequentist Theoretical (Asymptotic)
    println("Testing Theoretical IRF...")
    irf_theo = irf(model, 5; method=:cholesky, ci_type=:theoretical, reps=50, conf_level=0.90)

    @test irf_theo isa ImpulseResponse
    @test irf_theo.ci_type == :theoretical
    @test all(irf_theo.ci_lower .<= irf_theo.ci_upper)

    # Compare widths?
    # Theoretical and Bootstrap should be somewhat similar in width for large T.
    # With T=100 they might differ, but should overlap.
    @test isapprox(mean(irf_boot.values), mean(irf_theo.values), atol=1e-5) # Point estimates same

    # 3. Test Bayesian 68% Credible Interval (Default)
    println("Testing Bayesian 68% CI...")
    # Mock chain or run small one
    # Running small chain
    # Running small chain with IS (Importance Sampling) for robustness/speed in verification
    chain = estimate_bvar(Y, p; n_samples=100, sampler=:is)
    irf_bayes = irf(chain, p, n, 5)

    @test size(irf_bayes.quantiles, 4) == 3 # [0.16, 0.5, 0.84]
    # Check 16th <= 50th <= 84th
    @test all(irf_bayes.quantiles[:, :, :, 1] .<= irf_bayes.quantiles[:, :, :, 2])
    @test all(irf_bayes.quantiles[:, :, :, 2] .<= irf_bayes.quantiles[:, :, :, 3])

    println("IRF CI Tests Passed.")
end
