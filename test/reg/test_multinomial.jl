# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using MacroEconometricModels: dof_residual
using LinearAlgebra, Statistics, Random, Distributions

@testset "Multinomial Logit Regression" begin

    # =========================================================================
    # Helper: generate multinomial data from known DGP
    # =========================================================================

    function generate_mlogit_data(rng, n, beta_true)
        # beta_true is K x (J-1), base category = 1
        K = size(beta_true, 1)
        Jm1 = size(beta_true, 2)
        J = Jm1 + 1

        X = [ones(n) randn(rng, n, K - 1)]

        # Compute probabilities via softmax
        V = X * beta_true  # n x (J-1)
        probs = zeros(n, J)
        for i in 1:n
            v_max = max(zero(Float64), maximum(V[i, :]))
            denom = exp(-v_max)  # base category
            for j in 1:Jm1
                denom += exp(V[i, j] - v_max)
            end
            probs[i, 1] = exp(-v_max) / denom
            for j in 1:Jm1
                probs[i, j+1] = exp(V[i, j] - v_max) / denom
            end
        end

        # Draw from multinomial
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(rng)
            cumprob = 0.0
            y[i] = J
            for j in 1:J
                cumprob += probs[i, j]
                if u < cumprob
                    y[i] = j
                    break
                end
            end
        end

        (y, X)
    end

    # =========================================================================
    # 3-category coefficient recovery
    # =========================================================================

    @testset "3-category coefficient recovery" begin
        rng = MersenneTwister(2024)
        n = 5000
        # K=3 (intercept + 2 vars), J-1=2 alternatives
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])

        @test m isa MultinomialLogitModel{Float64}
        @test m.converged
        @test size(m.beta) == (3, 2)

        # Coefficient recovery (within ~0.15 for n=5000)
        for j in 1:2
            for k in 1:3
                @test abs(m.beta[k, j] - beta_true[k, j]) < 0.2
            end
        end

        # Categories
        @test length(m.categories) == 3
    end

    # =========================================================================
    # 4-category coefficient recovery
    # =========================================================================

    @testset "4-category coefficient recovery" begin
        rng = MersenneTwister(9999)
        n = 8000
        # K=3 (intercept + 2 vars), J-1=3 alternatives
        beta_true = [0.3 -0.5  0.2;
                     0.8 -0.3  0.6;
                    -0.4  0.7 -0.2]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])

        @test m isa MultinomialLogitModel{Float64}
        @test m.converged
        @test size(m.beta) == (3, 3)
        @test length(m.categories) == 4

        # Coefficient recovery
        for j in 1:3
            for k in 1:3
                @test abs(m.beta[k, j] - beta_true[k, j]) < 0.2
            end
        end
    end

    # =========================================================================
    # StatsAPI interface compliance
    # =========================================================================

    @testset "StatsAPI interface" begin
        rng = MersenneTwister(1111)
        n = 1000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])

        K = 3
        J = 3
        Jm1 = 2
        P = K * Jm1  # 6

        # coef returns vec(beta) of length K*(J-1)
        @test length(coef(m)) == P
        @test coef(m) == vec(m.beta)

        # vcov is K(J-1) x K(J-1) = 6x6
        @test size(vcov(m)) == (P, P)
        @test issymmetric(vcov(m)) || norm(vcov(m) - vcov(m)') < 1e-10

        # nobs
        @test nobs(m) == n

        # dof = K*(J-1)
        @test dof(m) == P

        # dof_residual
        @test dof_residual(m) == n - P

        # loglikelihood
        @test loglikelihood(m) < 0
        @test loglikelihood(m) == m.loglik

        # aic, bic
        @test aic(m) == m.aic
        @test bic(m) == m.bic
        @test aic(m) > 0
        @test bic(m) > 0

        # islinear
        @test islinear(m) == false

        # stderror: length K*(J-1)
        se = stderror(m)
        @test length(se) == P
        @test all(se .> 0)

        # confint: K*(J-1) x 2
        ci = confint(m)
        @test size(ci) == (P, 2)
        @test all(ci[:, 1] .< ci[:, 2])

        # predict: n x J matrix
        probs = predict(m)
        @test size(probs) == (n, J)
        @test all(probs .>= 0)
        @test all(abs.(sum(probs, dims=2) .- 1.0) .< 1e-10)
    end

    # =========================================================================
    # Predict: n x J matrix, rows sum to 1, all probs >= 0
    # =========================================================================

    @testset "Out-of-sample prediction" begin
        rng = MersenneTwister(3333)
        n = 1000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X)

        # New data prediction
        rng2 = MersenneTwister(4444)
        X_new = [ones(50) randn(rng2, 50, 2)]
        probs_new = predict(m, X_new)
        @test size(probs_new) == (50, 3)
        @test all(probs_new .>= 0)
        @test all(abs.(sum(probs_new, dims=2) .- 1.0) .< 1e-10)

        # Error on wrong dimensions
        @test_throws ArgumentError predict(m, randn(10, 5))
    end

    # =========================================================================
    # Robust SEs (:hc1 differs from :ols)
    # =========================================================================

    @testset "Robust standard errors" begin
        rng = MersenneTwister(7777)
        n = 2000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m_ols = estimate_mlogit(y, X; cov_type=:ols)
        m_hc1 = estimate_mlogit(y, X; cov_type=:hc1)

        # Coefficients should be the same
        @test coef(m_ols) == coef(m_hc1)

        # Standard errors should differ
        se_ols = stderror(m_ols)
        se_hc1 = stderror(m_hc1)
        @test se_ols != se_hc1

        # Both should be positive
        @test all(se_ols .> 0)
        @test all(se_hc1 .> 0)

        # HC0 should also work
        m_hc0 = estimate_mlogit(y, X; cov_type=:hc0)
        @test all(stderror(m_hc0) .> 0)
    end

    # =========================================================================
    # Category remapping
    # =========================================================================

    @testset "Category remapping" begin
        rng = MersenneTwister(5555)
        n = 1000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        # Remap y to non-standard categories: 10, 20, 30
        y_remapped = [yi == 1 ? 10 : yi == 2 ? 20 : 30 for yi in y]

        m1 = estimate_mlogit(y, X)
        m2 = estimate_mlogit(y_remapped, X)

        # Same coefficients
        @test coef(m1) ≈ coef(m2)

        # Categories stored correctly
        @test m2.categories == [10, 20, 30]
    end

    # =========================================================================
    # Display output
    # =========================================================================

    @testset "Display output" begin
        rng = MersenneTwister(8888)
        n = 500
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Multinomial Logit", output)
        @test occursin("Categories", output)
        @test occursin("Alternative", output)
        @test occursin("const", output)
        @test occursin("x1", output)
        @test occursin("x2", output)

        # report() dispatches to show(stdout, m) -- test it doesn't error
        io2 = IOBuffer()
        show(io2, m)
        @test length(String(take!(io2))) > 0
    end

    # =========================================================================
    # Float64 fallback
    # =========================================================================

    @testset "Float64 fallback" begin
        rng = MersenneTwister(6666)
        n = 500
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        # Pass integer X
        X_int = round.(Int, X)
        m = estimate_mlogit(y, X_int)
        @test m isa MultinomialLogitModel{Float64}
    end

    # =========================================================================
    # Model fit statistics
    # =========================================================================

    @testset "Model fit statistics" begin
        rng = MersenneTwister(4242)
        n = 2000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]

        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X)

        # Pseudo R-squared should be between 0 and 1
        @test 0 < m.pseudo_r2 < 1

        # Log-likelihood should be negative
        @test m.loglik < 0
        @test m.loglik_null < 0

        # Model should fit better than null
        @test m.loglik > m.loglik_null

        # Null log-likelihood = n * log(1/J)
        J = length(m.categories)
        @test m.loglik_null ≈ n * log(1.0 / J)

        # AIC = -2*loglik + 2*K
        @test m.aic ≈ -2 * m.loglik + 2 * dof(m)

        # BIC = -2*loglik + log(n)*K
        @test m.bic ≈ -2 * m.loglik + log(n) * dof(m)
    end

    # =========================================================================
    # Multinomial Logit — Marginal Effects
    # =========================================================================

    @testset "Multinomial Logit — marginal effects" begin
        rng = MersenneTwister(3030)
        n = 5000
        beta_true = [0.5 -0.3;
                     1.0 -0.5;
                    -0.5  0.8]
        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
        me = marginal_effects(m)

        # Returns a NamedTuple with effects, varnames, categories
        @test haskey(me, :effects)
        @test haskey(me, :varnames)
        @test haskey(me, :categories)

        # K x J matrix
        K = size(m.X, 2)
        J = length(m.categories)
        @test size(me.effects) == (K, J)

        # Key property: AMEs sum to ~0 across categories for each variable
        row_sums = sum(me.effects, dims=2)
        for k in 1:K
            @test abs(row_sums[k]) < 1e-10
        end

        # Variable names preserved
        @test me.varnames == ["const", "x1", "x2"]
    end

    @testset "Multinomial Logit — marginal effects 4-category" begin
        rng = MersenneTwister(4040)
        n = 8000
        beta_true = [0.3 -0.5  0.2;
                     0.8 -0.3  0.6;
                    -0.4  0.7 -0.2]
        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
        me = marginal_effects(m)

        K = 3
        J = 4
        @test size(me.effects) == (K, J)

        # Row sums should be ~0
        row_sums = sum(me.effects, dims=2)
        for k in 1:K
            @test abs(row_sums[k]) < 1e-10
        end
    end

    # =========================================================================
    # Hausman IIA Test
    # =========================================================================

    @testset "Hausman IIA test — structure" begin
        rng = MersenneTwister(5050)
        n = 5000
        # Use 4-category model so omitting one leaves 3 categories
        beta_true = [0.3 -0.5  0.2;
                     0.8 -0.3  0.6;
                    -0.4  0.7 -0.2]
        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
        ht = hausman_iia(m; omit_category=4)

        # Check structure
        @test haskey(ht, :statistic)
        @test haskey(ht, :pvalue)
        @test haskey(ht, :df)
        @test haskey(ht, :omitted_category)

        # Types
        @test ht.statistic isa Float64
        @test ht.pvalue isa Float64
        @test ht.df isa Int
        @test ht.statistic >= 0

        # p-value should be in [0, 1]
        @test 0 <= ht.pvalue <= 1

        # Omitted category label
        @test ht.omitted_category == m.categories[4]

        # df = K * (J_new - 1) where J_new = 3
        K = size(m.X, 2)
        @test ht.df == K * 2
    end

    @testset "Hausman IIA test — different omit categories" begin
        rng = MersenneTwister(6060)
        n = 5000
        # 4-category model to ensure enough categories after omission
        beta_true = [0.3 -0.5  0.2;
                     0.8 -0.3  0.6;
                    -0.4  0.7 -0.2]
        y, X = generate_mlogit_data(rng, n, beta_true)

        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])

        # Omit different categories
        ht2 = hausman_iia(m; omit_category=2)
        @test ht2.statistic >= 0
        @test 0 <= ht2.pvalue <= 1

        ht4 = hausman_iia(m; omit_category=4)
        @test ht4.statistic >= 0
        @test 0 <= ht4.pvalue <= 1

        # Invalid omit_category
        @test_throws ArgumentError hausman_iia(m; omit_category=0)
        @test_throws ArgumentError hausman_iia(m; omit_category=5)
    end

    # =========================================================================
    # Input validation
    # =========================================================================

    @testset "Input validation" begin
        rng = MersenneTwister(1234)
        X = randn(rng, 100, 2)

        # Only 2 categories should fail
        y_binary = rand(rng, [1, 2], 100)
        @test_throws ArgumentError estimate_mlogit(y_binary, X)

        # Mismatched dimensions
        @test_throws ArgumentError estimate_mlogit([1, 2, 3, 1], randn(rng, 5, 2))

        # Invalid cov_type
        y3 = rand(rng, [1, 2, 3], 100)
        @test_throws ArgumentError estimate_mlogit(y3, X; cov_type=:hc2)
    end

end
