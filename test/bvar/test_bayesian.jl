using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "BVAR Bayesian Parameter Recovery" begin
    println("Generating Data for Bayesian Verification...")

    # 1. Generate Synthetic Data
    T = 100
    n = 2
    p = 1
    Random.seed!(42)

    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]

    Y = zeros(T, n)
    for t in 2:T
        u = randn(2)  # Unit variance
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 2. Direct Sampler Parameter Recovery (Primary Test)
    @testset "Direct Sampler Parameter Recovery" begin
        println("Estimating BVAR (direct)...")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 30 : 100), sampler=:direct)
        @test post isa BVARPosterior

        # Extract and check parameter recovery
        b_vecs, _ = MacroEconometricModels.extract_chain_parameters(post)
        means_arr = vec(mean(b_vecs, dims=1))

        println("Recovered Means: ", means_arr)

        # Check intercepts (should be near 0)
        @test abs(means_arr[1]) < 0.5
        @test abs(means_arr[4]) < 0.5

        # Check diagonal A elements (should be near 0.5)
        @test isapprox(means_arr[2], 0.5, atol=0.35)
        @test isapprox(means_arr[6], 0.5, atol=0.35)

        # Check off-diagonal A elements (should be near 0)
        @test abs(means_arr[3]) < 0.35
        @test abs(means_arr[5]) < 0.35

        println("Direct Sampler Parameter Recovery Verified.")
    end

    # 3. Gibbs Sampler Smoke Test
    @testset "Gibbs Sampler Smoke Test" begin
        println("Estimating BVAR (Gibbs)...")
        post_gibbs = estimate_bvar(Y, p;
            n_draws=(FAST ? 20 : 50), sampler=:gibbs, burnin=(FAST ? 20 : 50), thin=1
        )
        @test post_gibbs isa BVARPosterior
        @test post_gibbs.n_draws == (FAST ? 20 : 50)
        @test post_gibbs.sampler == :gibbs
        println("Gibbs Sampler Smoke Test Passed.")
    end

    # ==========================================================================
    # Robustness Tests
    # ==========================================================================

    @testset "Reproducibility" begin
        println("Testing BVAR reproducibility...")
        Random.seed!(77777)
        Y_rep = zeros(80, 2)
        for t in 2:80
            Y_rep[t, :] = 0.5 * Y_rep[t-1, :] + randn(2)
        end

        Random.seed!(88888)
        post1 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct)

        Random.seed!(88888)
        post2 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct)

        # Same random seed should give same results
        @test post1.B_draws ≈ post2.B_draws
        @test post1.Sigma_draws ≈ post2.Sigma_draws
        println("Reproducibility test passed.")
    end

    @testset "Numerical Stability - Near-Collinear Data" begin
        println("Testing numerical stability with near-collinear data...")
        Random.seed!(11111)
        T_nc = 80
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(T_nc)

        post_nc = estimate_bvar(Y_nc, 1; n_draws=50, sampler=:direct)
        @test post_nc isa BVARPosterior

        # Check all parameters are finite
        @test all(isfinite.(post_nc.B_draws))
        println("Numerical stability test passed.")
    end

    @testset "Edge Cases" begin
        println("Testing edge cases...")
        Random.seed!(22222)

        # Single variable BVAR
        Y_single = randn(80, 1)
        post_single = estimate_bvar(Y_single, 1; n_draws=50)
        @test post_single isa BVARPosterior

        # Verify parameter dimensions for single variable
        # k = 1 + n*p = 1 + 1*1 = 2
        @test size(post_single.B_draws, 2) == 2  # intercept + 1 AR coefficient
        @test size(post_single.B_draws, 3) == 1  # 1 variable
        println("Edge case tests passed.")
    end

    @testset "Posterior Draws Structure" begin
        println("Testing posterior draws structure...")
        Random.seed!(33333)
        Y_diag = zeros(80, 2)
        for t in 2:80
            Y_diag[t, :] = 0.5 * Y_diag[t-1, :] + randn(2)
        end

        post_diag = estimate_bvar(Y_diag, 1; n_draws=50, sampler=:direct)

        # Check structure
        @test post_diag.n_draws == 50
        @test post_diag.p == 1
        @test post_diag.n == 2

        # All samples should be finite
        @test all(isfinite.(post_diag.B_draws))
        @test all(isfinite.(post_diag.Sigma_draws))

        # Sigma draws should be symmetric positive definite
        for s in 1:post_diag.n_draws
            S = post_diag.Sigma_draws[s, :, :]
            @test isapprox(S, S', atol=1e-10)
            @test all(eigvals(Symmetric(S)) .> -1e-10)
        end

        # Posterior mean should be reasonable (not extreme)
        b_vecs, _ = MacroEconometricModels.extract_chain_parameters(post_diag)
        mean_b = vec(mean(b_vecs, dims=1))
        @test all(abs.(mean_b) .< 10.0)  # Not exploding
        println("Posterior draws structure test passed.")
    end

    @testset "Posterior Model Extraction" begin
        println("Testing posterior model extraction...")
        Random.seed!(44444)
        Y_post = zeros(80, 2)
        for t in 2:80
            Y_post[t, :] = 0.5 * Y_post[t-1, :] + randn(2)
        end

        post = estimate_bvar(Y_post, 1; n_draws=50)

        # Extract posterior mean model
        mean_model = posterior_mean_model(post; data=Y_post)
        @test mean_model isa VARModel
        @test all(isfinite.(mean_model.B))
        @test all(isfinite.(mean_model.Sigma))

        # Extract posterior median model
        med_model = posterior_median_model(post; data=Y_post)
        @test med_model isa VARModel
        @test all(isfinite.(med_model.B))

        # Test deprecated wrapper signatures
        mean_model2 = posterior_mean_model(post, 1, 2; data=Y_post)
        @test mean_model2 isa VARModel

        println("Posterior model extraction test passed.")
    end

    @testset "Minnesota prior with BVAR" begin
        Random.seed!(99887)
        Y_mn = randn(80, 2)
        hyper = MinnesotaHyperparameters(tau=0.2, decay=2.0, omega=0.5)
        post_mn = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper, n_draws=100)
        @test post_mn isa BVARPosterior
        @test post_mn.prior == :minnesota
        println("Minnesota prior BVAR test passed.")
    end

    @testset "BVAR sampler variants" begin
        Random.seed!(99886)
        Y_sv = randn(60, 2)

        # Direct sampler
        @testset "Direct sampler" begin
            post_direct = estimate_bvar(Y_sv, 1; sampler=:direct, n_draws=50)
            @test post_direct isa BVARPosterior
            @test post_direct.sampler == :direct
            println("Direct sampler test passed.")
        end

        # Gibbs sampler
        @testset "Gibbs sampler" begin
            post_gibbs = estimate_bvar(Y_sv, 1; sampler=:gibbs, n_draws=50, burnin=100)
            @test post_gibbs isa BVARPosterior
            @test post_gibbs.sampler == :gibbs
            println("Gibbs sampler test passed.")
        end

        # Unknown sampler
        @testset "Unknown sampler error" begin
            @test_throws ArgumentError estimate_bvar(Y_sv, 1; sampler=:nonexistent, n_draws=50)
        end
    end

    # 8. BVARPosterior show() method
    @testset "BVARPosterior show method" begin
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct)
        io = IOBuffer()
        show(io, post)
        out = String(take!(io))
        @test length(out) > 0
        @test occursin("Bayesian VAR", out)
        @test occursin("Mean", out)
        @test occursin("2.5%", out)
        @test occursin("97.5%", out)
        @test occursin("Posterior Mean", out)
        println("BVARPosterior show test passed.")
    end
end
