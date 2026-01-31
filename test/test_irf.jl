using MacroEconometricModels
using Test
using MCMCChains
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

@testset "IRF Tests with Theoretical Verification" begin
    println("Generating Data for IRF Verification...")
    # 1. Setup Data with Known DGP
    # VAR(1): Y_t = A Y_{t-1} + u_t, u_t ~ N(0, I)
    # A = 0.5 * I
    T = 500
    n = 2
    p = 1
    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]
    Sigma_true = [1.0 0.0; 0.0 1.0] # Identity
    L_true = [1.0 0.0; 0.0 1.0]      # Cholesky of Identity is Identity

    Y = zeros(T, n)
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    model = estimate_var(Y, p)
    println("Frequentist Estimation Done.")

    # 2. Frequentist IRF (Cholesky) vs Theoretical
    println("Testing Frequentist IRF (Cholesky)...")
    irf_freq = irf(model, 6; method=:cholesky) # Horizon 6 (lags 0 to 5)

    # Theoretical IRF: Phi_h * P
    # P = L_true = I
    # Phi_h = A^h
    # Since A is diagonal 0.5:
    # IRF at h (lag h-1) = 0.5^(h-1) * I

    for h in 1:6
        lag = h - 1
        theoretical_impact = (0.5^lag) * I(2)
        estimated_impact = irf_freq.values[h, :, :]

        # Check diagonal elements
        @test isapprox(estimated_impact[1, 1], theoretical_impact[1, 1], atol=0.1)
        @test isapprox(estimated_impact[2, 2], theoretical_impact[2, 2], atol=0.1)

        # Check off-diagonal (should be close to 0)
        @test abs(estimated_impact[1, 2]) < 0.1
        @test abs(estimated_impact[2, 1]) < 0.1
    end

    # 3. Frequentist IRF (Sign) - Basic check logic remains
    println("Testing Frequentist IRF (Sign)...")
    check_func(irf) = irf[1, 1, 1] > 0
    irf_sign_res = irf(model, 6; method=:sign, check_func=check_func)
    @test irf_sign_res.values[1, 1, 1] > 0

    # 4. Bayesian IRF
    println("Testing Bayesian Estimation...")
    try
        # Estimate chain
        # Need adequate samples for mean convergence
        chain = estimate_bvar(Y, p; n_samples=200, n_adapts=100)
        println("Bayesian Estimation Done.")

        println("Testing Bayesian IRF...")
        irf_bayes = irf(chain, p, n, 6; method=:cholesky)
        println("Bayesian IRF Done.")

        @test irf_bayes isa BayesianImpulseResponse

        # Check Mean IRF against Theoretical
        for h in 1:6
            lag = h - 1
            theoretical_impact = (0.5^lag) * I(2)
            bayes_mean = irf_bayes.mean[h, :, :]

            # Allow slightly larger tolerance for Bayesian mean with finite chain
            @test isapprox(bayes_mean[1, 1], theoretical_impact[1, 1], atol=0.2)
            @test isapprox(bayes_mean[2, 2], theoretical_impact[2, 2], atol=0.2)
        end

    catch e
        println("ERROR CAUGHT:")
        showerror(stdout, e)
        println()
        rethrow(e)
    end
end
