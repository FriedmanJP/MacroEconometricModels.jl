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

using MacroEconometricModels
using Test
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
        post = estimate_bvar(Y, p; n_draws=50)
        println("Bayesian Estimation Done.")

        println("Testing Bayesian IRF...")
        irf_bayes = irf(post, 6; method=:cholesky)
        println("Bayesian IRF Done.")

        @test irf_bayes isa BayesianImpulseResponse

        # Check Mean IRF against Theoretical
        for h in 1:6
            lag = h - 1
            theoretical_impact = (0.5^lag) * I(2)
            bayes_mean = irf_bayes.mean[h, :, :]

            # Allow larger tolerance for smaller chain
            @test isapprox(bayes_mean[1, 1], theoretical_impact[1, 1], atol=0.3)
            @test isapprox(bayes_mean[2, 2], theoretical_impact[2, 2], atol=0.3)
        end

    catch e
        println("ERROR CAUGHT:")
        showerror(stdout, e)
        println()
        rethrow(e)
    end
end

# =============================================================================
# Cumulative IRF (Issue #15)
# =============================================================================
@testset "Cumulative IRF" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    H = 20

    @testset "VAR cumulative IRF" begin
        irf_result = irf(model, H)
        cirf = cumulative_irf(irf_result)

        @test cirf isa ImpulseResponse
        @test size(cirf.values) == size(irf_result.values)
        @test cirf.horizon == irf_result.horizon

        # Verify cumulative = cumsum along horizon dimension
        expected = cumsum(irf_result.values, dims=1)
        @test cirf.values ≈ expected

        # CI bands should also be cumulated
        @test cirf.ci_lower ≈ cumsum(irf_result.ci_lower, dims=1)
        @test cirf.ci_upper ≈ cumsum(irf_result.ci_upper, dims=1)
    end

    @testset "Bayesian cumulative IRF" begin
        post = estimate_bvar(Y, 2; n_draws=200)
        birf = irf(post, H)
        bcirf = cumulative_irf(birf)

        @test bcirf isa BayesianImpulseResponse
        @test size(bcirf.mean) == size(birf.mean)
        @test bcirf.mean ≈ cumsum(birf.mean, dims=1)
        @test bcirf.quantiles ≈ cumsum(birf.quantiles, dims=1)
    end
end

# =============================================================================
# compute_irf exported (Issue #20)
# =============================================================================
@testset "compute_irf exported" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    n = 3

    Q = Matrix{Float64}(I, n, n)
    result = compute_irf(model, Q, 10)
    @test size(result) == (10, n, n)
    @test !any(isnan, result)
end

# =============================================================================
# Sign Identified Set (Issue #21)
# =============================================================================
@testset "Sign Identified Set" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    n = 3
    H = 10

    # Accept-all check function for testing
    check_all(irf_result) = true

    result = identify_sign(model, H, check_all; max_draws=50, store_all=true)

    @test result isa SignIdentifiedSet
    @test result.n_accepted == 50
    @test result.n_total == 50
    @test result.acceptance_rate ≈ 1.0
    @test length(result.Q_draws) == 50
    @test size(result.irf_draws) == (50, H, n, n)

    # irf_bounds
    lower, upper = irf_bounds(result)
    @test size(lower) == (H, n, n)
    @test size(upper) == (H, n, n)
    @test all(lower .<= upper)

    # irf_median
    med = irf_median(result)
    @test size(med) == (H, n, n)
    @test !any(isnan, med)

    # show method
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("Sign-Identified Set", output)
    @test occursin("50", output)
end
