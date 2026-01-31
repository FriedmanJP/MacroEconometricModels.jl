using MacroEconometricModels
using Test
using MCMCChains
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

@testset "Minnesota Prior Tests" begin
    println("Generating Data for Minnesota Test...")
    T = 50
    n = 2
    p = 1
    true_A = [0.8 0.0; 0.0 0.8] # Persistent
    true_c = [0.0; 0.0]
    Y = zeros(T, n)
    for t in 2:T
        u = randn(2) * 0.5
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 1. Test Dummy Generation
    hyper = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0)
    Y_d, X_d = gen_dummy_obs(Y, p, hyper)

    # Check dimensions
    # n=2, p=1
    # AR priors: n*p = 2 rows
    # Sum coeffs: n = 2 rows
    # Dummy initial: 1 row
    # Covariance: n = 2 rows
    # Total = 2 + 2 + 1 + 2 = 7 rows
    @test size(Y_d, 1) == 7
    @test size(Y_d, 2) == 2
    @test size(X_d, 1) == 7
    @test size(X_d, 2) == 3 # 1 + 2*1

    println("Dummy Observations Generated.")

    # 2. Test Estimation with Minnesota Prior
    println("Estimating BVAR with Minnesota...")
    chain = estimate_bvar(Y, p; n_samples=50, n_adapts=20, prior=:minnesota, hyper=hyper)

    @test chain isa Chains
    @test size(chain, 1) == 50

    # Basic check: posterior mean should be somewhat reasonable (within bounds)
    # Just checking it runs and returns.
    println("Estimation Complete.")
end
