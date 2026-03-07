# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using MacroEconometricModels: dof_residual
using LinearAlgebra, Statistics, Random, Distributions

@testset "Ordered Logit/Probit Regression" begin

    # =========================================================================
    # Helper: generate ordered data
    # =========================================================================

    function _link_cdf(link::Symbol, x::Float64)
        if link == :logit
            return 1.0 / (1.0 + exp(-x))
        else
            return cdf(Normal(), x)
        end
    end

    function generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)
        K = length(beta_true)
        X = randn(rng, n, K)
        xb = X * beta_true
        J = length(cutpoints_true) + 1

        y = Vector{Int}(undef, n)
        u = rand(rng, n)
        for i in 1:n
            assigned = false
            for j in 1:(J-1)
                p_j = _link_cdf(link, cutpoints_true[j] - xb[i])
                if u[i] < p_j
                    y[i] = j
                    assigned = true
                    break
                end
            end
            if !assigned
                y[i] = J
            end
        end
        (y, X)
    end

    # =========================================================================
    # Ordered Logit: 3-category coefficient recovery
    # =========================================================================

    @testset "Ordered Logit 3-category recovery" begin
        rng = MersenneTwister(2024)
        n = 5000
        beta_true = [1.0, -0.5]
        cutpoints_true = [0.0, 1.5]

        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2"])

        @test m isa OrderedLogitModel{Float64}
        @test m.converged
        @test length(coef(m)) == 2
        @test length(m.cutpoints) == 2

        # Coefficient recovery (within ~0.15 of true values for n=5000)
        @test abs(coef(m)[1] - beta_true[1]) < 0.15
        @test abs(coef(m)[2] - beta_true[2]) < 0.15
        @test abs(m.cutpoints[1] - cutpoints_true[1]) < 0.15
        @test abs(m.cutpoints[2] - cutpoints_true[2]) < 0.15

        # Cutpoints are ordered
        @test m.cutpoints[1] < m.cutpoints[2]

        # Categories
        @test length(m.categories) == 3
    end

    # =========================================================================
    # Ordered Logit: 5-category coefficient recovery
    # =========================================================================

    @testset "Ordered Logit 5-category recovery" begin
        rng = MersenneTwister(9999)
        n = 8000
        beta_true = [0.8, -0.6, 0.4]
        cutpoints_true = [-1.5, -0.3, 0.8, 2.0]

        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2", "x3"])

        @test m isa OrderedLogitModel{Float64}
        @test m.converged
        @test length(coef(m)) == 3
        @test length(m.cutpoints) == 4

        # Coefficient recovery
        for k in 1:3
            @test abs(coef(m)[k] - beta_true[k]) < 0.15
        end

        # Cutpoints ordered
        for j in 2:4
            @test m.cutpoints[j] > m.cutpoints[j-1]
        end

        # 5 categories
        @test length(m.categories) == 5
    end

    # =========================================================================
    # Ordered Probit: 3-category coefficient recovery
    # =========================================================================

    @testset "Ordered Probit 3-category recovery" begin
        rng = MersenneTwister(5678)
        n = 5000
        beta_true = [0.8, -0.5]
        cutpoints_true = [0.0, 1.0]

        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:probit)

        m = estimate_oprobit(y, X; varnames=["x1", "x2"])

        @test m isa OrderedProbitModel{Float64}
        @test m.converged
        @test length(coef(m)) == 2
        @test length(m.cutpoints) == 2

        # Coefficient recovery
        @test abs(coef(m)[1] - beta_true[1]) < 0.15
        @test abs(coef(m)[2] - beta_true[2]) < 0.15
        @test abs(m.cutpoints[1] - cutpoints_true[1]) < 0.15
        @test abs(m.cutpoints[2] - cutpoints_true[2]) < 0.15

        # Cutpoints ordered
        @test m.cutpoints[1] < m.cutpoints[2]
    end

    # =========================================================================
    # StatsAPI interface compliance
    # =========================================================================

    @testset "StatsAPI interface" begin
        rng = MersenneTwister(1111)
        n = 1000
        beta_true = [1.0, -0.5]
        cutpoints_true = [0.0, 1.5]
        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2"])

        # coef returns beta only
        @test length(coef(m)) == 2

        # vcov is (K+J-1) x (K+J-1) = 4x4
        @test size(vcov(m)) == (4, 4)
        @test issymmetric(vcov(m)) || norm(vcov(m) - vcov(m)') < 1e-10

        # nobs
        @test nobs(m) == n

        # dof = K + J - 1
        @test dof(m) == 4

        # dof_residual
        @test dof_residual(m) == n - 4

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

        # stderror: length K+J-1
        se = stderror(m)
        @test length(se) == 4
        @test all(se .> 0)

        # confint: (K+J-1) x 2
        ci = confint(m)
        @test size(ci) == (4, 2)
        @test all(ci[:, 1] .< ci[:, 2])

        # predict: n x J matrix
        probs = predict(m)
        @test size(probs) == (n, 3)
        @test all(probs .>= 0)
        @test all(abs.(sum(probs, dims=2) .- 1.0) .< 1e-10)

        # Same tests for ordered probit
        mp = estimate_oprobit(y, X; varnames=["x1", "x2"])
        @test mp isa OrderedProbitModel{Float64}
        @test length(coef(mp)) == 2
        @test size(vcov(mp)) == (4, 4)
        @test islinear(mp) == false
    end

    # =========================================================================
    # Out-of-sample prediction
    # =========================================================================

    @testset "Out-of-sample prediction" begin
        rng = MersenneTwister(3333)
        n = 1000
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        m = estimate_ologit(y, X)

        # New data prediction
        rng2 = MersenneTwister(4444)
        X_new = randn(rng2, 50, 2)
        probs_new = predict(m, X_new)
        @test size(probs_new) == (50, 3)
        @test all(probs_new .>= 0)
        @test all(abs.(sum(probs_new, dims=2) .- 1.0) .< 1e-10)

        # Error on wrong dimensions
        @test_throws ArgumentError predict(m, randn(10, 5))
    end

    # =========================================================================
    # Robust SEs (cov_type=:hc1 differs from :ols)
    # =========================================================================

    @testset "Robust standard errors" begin
        rng = MersenneTwister(7777)
        n = 2000
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        m_ols = estimate_ologit(y, X; cov_type=:ols)
        m_hc1 = estimate_ologit(y, X; cov_type=:hc1)

        # Coefficients should be the same
        @test coef(m_ols) == coef(m_hc1)
        @test m_ols.cutpoints == m_hc1.cutpoints

        # Standard errors should differ
        se_ols = stderror(m_ols)
        se_hc1 = stderror(m_hc1)
        @test se_ols != se_hc1

        # Both should be positive
        @test all(se_ols .> 0)
        @test all(se_hc1 .> 0)

        # Same for probit
        mp_ols = estimate_oprobit(y, X; cov_type=:ols)
        mp_hc1 = estimate_oprobit(y, X; cov_type=:hc1)
        @test stderror(mp_ols) != stderror(mp_hc1)
    end

    # =========================================================================
    # Category remapping
    # =========================================================================

    @testset "Category remapping" begin
        rng = MersenneTwister(5555)
        n = 1000
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        # Remap y to non-standard categories: 10, 20, 30
        y_remapped = [yi == 1 ? 10 : yi == 2 ? 20 : 30 for yi in y]

        m1 = estimate_ologit(y, X)
        m2 = estimate_ologit(y_remapped, X)

        # Same coefficients
        @test coef(m1) ≈ coef(m2)
        @test m1.cutpoints ≈ m2.cutpoints

        # Categories stored correctly
        @test m2.categories == [10, 20, 30]
    end

    # =========================================================================
    # Display output
    # =========================================================================

    @testset "Display output" begin
        rng = MersenneTwister(8888)
        n = 500
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2"])

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Ordered Logit", output)
        @test occursin("Categories", output)
        @test occursin("Coefficients", output)
        @test occursin("Cutpoints", output)
        @test occursin("cut1", output)
        @test occursin("cut2", output)

        # Probit display
        mp = estimate_oprobit(y, X; varnames=["x1", "x2"])
        io2 = IOBuffer()
        show(io2, mp)
        output2 = String(take!(io2))
        @test occursin("Ordered Probit", output2)

        # report() dispatches to show(stdout, m) — test it doesn't error
        io3 = IOBuffer()
        show(io3, m)
        @test length(String(take!(io3))) > 0
    end

    # =========================================================================
    # Float64 fallback
    # =========================================================================

    @testset "Float64 fallback" begin
        rng = MersenneTwister(6666)
        n = 500
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        # Pass integer X
        X_int = round.(Int, X)
        m = estimate_ologit(y, X_int)
        @test m isa OrderedLogitModel{Float64}

        mp = estimate_oprobit(y, X_int)
        @test mp isa OrderedProbitModel{Float64}
    end

    # =========================================================================
    # McFadden pseudo R-squared
    # =========================================================================

    @testset "Model fit statistics" begin
        rng = MersenneTwister(4242)
        n = 2000
        y, X = generate_ordered_data(rng, n, [1.0, -0.5], [0.0, 1.5]; link=:logit)

        m = estimate_ologit(y, X)

        # Pseudo R-squared should be between 0 and 1
        @test 0 < m.pseudo_r2 < 1

        # Log-likelihood should be negative
        @test m.loglik < 0
        @test m.loglik_null < 0

        # Model should fit better than null
        @test m.loglik > m.loglik_null

        # AIC = -2*loglik + 2*K
        @test m.aic ≈ -2 * m.loglik + 2 * dof(m)

        # BIC = -2*loglik + log(n)*K
        @test m.bic ≈ -2 * m.loglik + log(n) * dof(m)
    end

    # =========================================================================
    # Ordered Logit — Marginal Effects
    # =========================================================================

    @testset "Ordered Logit — marginal effects" begin
        rng = MersenneTwister(3030)
        n = 3000
        beta_true = [1.0, -0.5]
        cutpoints_true = [0.0, 1.5]
        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2"])
        me = marginal_effects(m)

        # Returns a NamedTuple with effects, varnames, categories
        @test haskey(me, :effects)
        @test haskey(me, :varnames)
        @test haskey(me, :categories)

        # K x J matrix
        K = length(coef(m))
        J = length(m.categories)
        @test size(me.effects) == (K, J)

        # Key property: AMEs sum to ~0 across categories for each variable
        row_sums = sum(me.effects, dims=2)
        for k in 1:K
            @test abs(row_sums[k]) < 1e-10
        end

        # Variable names preserved
        @test me.varnames == ["x1", "x2"]

        # Same test for ordered probit
        mp = estimate_oprobit(y, X; varnames=["x1", "x2"])
        me_p = marginal_effects(mp)
        @test size(me_p.effects) == (K, J)
        row_sums_p = sum(me_p.effects, dims=2)
        for k in 1:K
            @test abs(row_sums_p[k]) < 1e-10
        end
    end

    @testset "Ordered Logit — marginal effects 5-category" begin
        rng = MersenneTwister(4040)
        n = 5000
        beta_true = [0.8, -0.6, 0.4]
        cutpoints_true = [-1.5, -0.3, 0.8, 2.0]
        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2", "x3"])
        me = marginal_effects(m)

        K = 3
        J = 5
        @test size(me.effects) == (K, J)

        # Row sums should be ~0
        row_sums = sum(me.effects, dims=2)
        for k in 1:K
            @test abs(row_sums[k]) < 1e-10
        end
    end

    # =========================================================================
    # Brant Test
    # =========================================================================

    @testset "Brant test — structure" begin
        rng = MersenneTwister(5050)
        n = 2000
        beta_true = [1.0, -0.5]
        cutpoints_true = [0.0, 1.5]
        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X; varnames=["x1", "x2"])
        bt = brant_test(m)

        # Check structure
        @test haskey(bt, :statistic)
        @test haskey(bt, :pvalue)
        @test haskey(bt, :df)
        @test haskey(bt, :per_variable)
        @test haskey(bt, :binary_coefs)

        # Types
        @test bt.statistic isa Float64
        @test bt.pvalue isa Float64
        @test bt.df isa Int
        @test length(bt.per_variable) == 2

        # df should be K*(J-2) = 2*1 = 2
        K = length(coef(m))
        J = length(m.categories)
        @test bt.df == K * (J - 2)

        # binary_coefs: K x (J-1)
        @test size(bt.binary_coefs) == (K, J - 1)

        # p-value should be in [0, 1]
        @test 0 <= bt.pvalue <= 1
        for pv in bt.per_variable
            @test 0 <= pv <= 1
        end
    end

    @testset "Brant test — proportional odds data should not reject" begin
        # Data generated from ordered logit (proportional odds holds by construction)
        rng = MersenneTwister(6060)
        n = 5000
        beta_true = [1.0, -0.5]
        cutpoints_true = [0.0, 1.5]
        y, X = generate_ordered_data(rng, n, beta_true, cutpoints_true; link=:logit)

        m = estimate_ologit(y, X)
        bt = brant_test(m)

        # Should not reject at 0.01 level (proportional odds holds)
        @test bt.pvalue > 0.01
    end

    @testset "Brant test — violation of proportional odds" begin
        # Generate data that violates proportional odds
        rng = MersenneTwister(7070)
        n = 5000
        K = 2
        X = randn(rng, n, K)

        # Use DIFFERENT slopes for each binary split
        # Split 1 (y <= 1 vs y > 1): beta = [2.0, -1.0]
        # Split 2 (y <= 2 vs y > 2): beta = [0.2, 0.5]
        # This strongly violates proportional odds
        y = Vector{Int}(undef, n)
        u = rand(rng, n)
        for i in 1:n
            p1 = 1.0 / (1.0 + exp(-(0.0 - X[i,:]' * [2.0, -1.0])))
            p2 = 1.0 / (1.0 + exp(-(1.5 - X[i,:]' * [0.2, 0.5])))
            if u[i] < p1
                y[i] = 1
            elseif u[i] < p2
                y[i] = 2
            else
                y[i] = 3
            end
        end

        m = estimate_ologit(y, X)
        bt = brant_test(m)

        # Should reject at 0.05 level
        @test bt.pvalue < 0.05
    end

    # =========================================================================
    # Edge case: exactly 3 categories required
    # =========================================================================

    @testset "Input validation" begin
        rng = MersenneTwister(1234)
        X = randn(rng, 100, 2)

        # Only 2 categories should fail
        y_binary = rand(rng, [1, 2], 100)
        @test_throws ArgumentError estimate_ologit(y_binary, X)

        # Mismatched dimensions
        @test_throws ArgumentError estimate_ologit([1, 2, 3, 1], randn(rng, 5, 2))
    end

end
