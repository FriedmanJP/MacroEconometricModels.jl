using Test
using MacroEconometricModels
using MacroEconometricModels: dof_residual
using LinearAlgebra, Statistics, Random

@testset "Cross-Sectional Regression (Tasks 2-4)" begin

    # =========================================================================
    # Task 2: Type construction and StatsAPI interface
    # =========================================================================

    @testset "RegModel type construction and StatsAPI" begin
        # Manually construct a RegModel to test StatsAPI accessors
        n, k = 100, 3
        rng = MersenneTwister(42)
        X = hcat(ones(n), randn(rng, n, k - 1))
        beta_true = [1.0, 2.0, -0.5]
        y = X * beta_true + 0.3 * randn(rng, n)
        fitted_vals = X * beta_true
        resid = y .- fitted_vals
        ssr = dot(resid, resid)
        tss = sum((yi - mean(y))^2 for yi in y)

        m = RegModel{Float64}(
            y, X, beta_true, Matrix{Float64}(I, k, k) * 0.01,
            resid, fitted_vals,
            ssr, tss, 1.0 - ssr / tss, 0.98, 100.0, 0.0,
            -50.0, 106.0, 120.0,
            ["const", "x1", "x2"], :ols, :hc1,
            nothing, nothing, nothing, nothing, nothing, nothing
        )

        # StatsAPI accessors
        @test coef(m) == beta_true
        @test length(coef(m)) == k
        @test size(vcov(m)) == (k, k)
        @test length(residuals(m)) == n
        @test length(predict(m)) == n
        @test nobs(m) == n
        @test dof(m) == k
        @test dof_residual(m) == n - k
        @test loglikelihood(m) == -50.0
        @test aic(m) == 106.0
        @test bic(m) == 120.0
        @test islinear(m) == true
        @test r2(m) == m.r2
        @test length(stderror(m)) == k
        @test all(stderror(m) .> 0)

        # Confidence intervals (t-distribution)
        ci = confint(m)
        @test size(ci) == (k, 2)
        @test all(ci[:, 1] .< coef(m))
        @test all(ci[:, 2] .> coef(m))

        # show method
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("OLS Regression", output)
        @test occursin("R-squared", output)
    end

    @testset "LogitModel type construction and StatsAPI" begin
        n, k = 50, 2
        m = LogitModel{Float64}(
            ones(n), randn(n, k), [0.5, -0.3],
            Matrix{Float64}(I, k, k) * 0.04,
            randn(n), fill(0.5, n),
            -30.0, -34.0, 0.12, 64.0, 68.0,
            ["const", "x1"], true, 5, :ols
        )

        @test coef(m) == [0.5, -0.3]
        @test nobs(m) == n
        @test dof(m) == k
        @test dof_residual(m) == n - k
        @test loglikelihood(m) == -30.0
        @test aic(m) == 64.0
        @test bic(m) == 68.0
        @test islinear(m) == false

        # Confidence intervals (normal distribution)
        ci = confint(m)
        @test size(ci) == (k, 2)
        @test all(ci[:, 1] .< coef(m))
        @test all(ci[:, 2] .> coef(m))

        # show method
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Logit Regression", output)
    end

    @testset "ProbitModel type construction and StatsAPI" begin
        n, k = 50, 2
        m = ProbitModel{Float64}(
            ones(n), randn(n, k), [0.3, -0.2],
            Matrix{Float64}(I, k, k) * 0.03,
            randn(n), fill(0.5, n),
            -28.0, -34.0, 0.18, 60.0, 64.0,
            ["const", "x1"], true, 4, :ols
        )

        @test coef(m) == [0.3, -0.2]
        @test nobs(m) == n
        @test islinear(m) == false

        # Confidence intervals (normal distribution)
        ci = confint(m)
        @test size(ci) == (k, 2)

        # show method
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Probit Regression", output)
    end

    @testset "MarginalEffects type construction" begin
        k = 3
        me = MarginalEffects{Float64}(
            [0.1, 0.2, -0.05],
            [0.01, 0.02, 0.01],
            [10.0, 10.0, -5.0],
            [0.0, 0.0, 0.0],
            [0.08, 0.16, -0.07],
            [0.12, 0.24, -0.03],
            ["x1", "x2", "x3"],
            :ame, 0.95
        )

        @test me.effects == [0.1, 0.2, -0.05]
        @test me.type == :ame
        @test me.conf_level == 0.95
        @test length(me.varnames) == k

        # show method
        io = IOBuffer()
        show(io, me)
        output = String(take!(io))
        @test occursin("Average Marginal Effects", output)
    end

    # =========================================================================
    # Task 3: Covariance Estimators
    # =========================================================================

    @testset "Covariance estimators" begin

        @testset "HC0-HC3 ordering on heteroskedastic data" begin
            rng = MersenneTwister(123)
            n = 200
            k = 3
            X = hcat(ones(n), randn(rng, n, k - 1))
            beta_true = [1.0, 2.0, -0.5]
            # Heteroskedastic errors: sigma proportional to |x1|
            sigma_i = 0.5 .+ abs.(X[:, 2])
            y = X * beta_true + sigma_i .* randn(rng, n)
            resid = y .- X * (X \ y)

            XtXinv = inv(X' * X)

            V_ols = MacroEconometricModels._reg_vcov(X, resid, :ols, XtXinv)
            V_hc0 = MacroEconometricModels._reg_vcov(X, resid, :hc0, XtXinv)
            V_hc1 = MacroEconometricModels._reg_vcov(X, resid, :hc1, XtXinv)
            V_hc2 = MacroEconometricModels._reg_vcov(X, resid, :hc2, XtXinv)
            V_hc3 = MacroEconometricModels._reg_vcov(X, resid, :hc3, XtXinv)

            # All should be positive definite
            for V in [V_ols, V_hc0, V_hc1, V_hc2, V_hc3]
                @test all(diag(V) .> 0)
            end

            # HC3 >= HC2 >= HC0 on diagonal (by construction from leverage corrections)
            for j in 1:k
                @test diag(V_hc3)[j] >= diag(V_hc2)[j]
                @test diag(V_hc2)[j] >= diag(V_hc0)[j]
            end

            # HC1 >= HC0 (by the n/(n-k) correction)
            for j in 1:k
                @test diag(V_hc1)[j] >= diag(V_hc0)[j]
            end
        end

        @testset "Cluster-robust covariance" begin
            rng = MersenneTwister(456)
            n = 200
            G = 20  # 20 clusters of 10 each
            X = hcat(ones(n), randn(rng, n, 2))
            clusters = repeat(1:G, inner=10)
            beta_true = [1.0, 0.5, -0.3]

            # Within-cluster correlated errors
            y = zeros(n)
            for g in 1:G
                idx = findall(==(g), clusters)
                group_shock = randn(rng)
                y[idx] = X[idx, :] * beta_true .+ group_shock .+ 0.3 .* randn(rng, length(idx))
            end
            resid = y .- X * (X \ y)
            XtXinv = inv(X' * X)

            V_cl = MacroEconometricModels._reg_vcov(X, resid, :cluster, XtXinv; clusters=clusters)

            # Should be PSD
            @test all(diag(V_cl) .> 0)

            # Cluster-robust SE should generally be larger than HC1 with clustered data
            V_hc1 = MacroEconometricModels._reg_vcov(X, resid, :hc1, XtXinv)
            # At least one cluster SE should exceed HC1 SE
            @test any(diag(V_cl) .> diag(V_hc1))
        end

        @testset "Covariance error handling" begin
            X = ones(10, 2)
            resid = randn(10)
            XtXinv = Matrix{Float64}(I, 2, 2)

            # Invalid cov_type
            @test_throws ArgumentError MacroEconometricModels._reg_vcov(X, resid, :invalid, XtXinv)

            # :cluster without clusters argument
            @test_throws ArgumentError MacroEconometricModels._reg_vcov(X, resid, :cluster, XtXinv)

            # Cluster with wrong length
            @test_throws ArgumentError MacroEconometricModels._reg_vcov(X, resid, :cluster, XtXinv; clusters=[1,2,3])
        end
    end

    # =========================================================================
    # Task 4: OLS / WLS Estimation
    # =========================================================================

    @testset "OLS estimation — coefficient recovery" begin
        rng = MersenneTwister(789)
        n = 500
        beta_true = [1.0, 2.0, -0.5]
        k = length(beta_true)
        X = hcat(ones(n), randn(rng, n, k - 1))
        y = X * beta_true + 0.3 * randn(rng, n)

        m = estimate_reg(y, X)

        # Coefficient recovery
        @test all(abs.(coef(m) .- beta_true) .< 0.1)

        # R-squared should be high with low noise
        @test r2(m) > 0.9

        # Residuals should be roughly mean-zero
        @test abs(mean(residuals(m))) < 0.1

        # Fitted values
        @test predict(m) ≈ X * coef(m)

        # Method and cov_type
        @test m.method == :ols
        @test m.cov_type == :hc1
    end

    @testset "OLS — robust SEs same coefs, different SEs" begin
        rng = MersenneTwister(101)
        n = 300
        X = hcat(ones(n), randn(rng, n, 2))
        sigma_i = 0.5 .+ 2.0 .* abs.(X[:, 2])
        y = X * [1.0, 0.5, -0.3] + sigma_i .* randn(rng, n)

        m_ols = estimate_reg(y, X; cov_type=:ols)
        m_hc3 = estimate_reg(y, X; cov_type=:hc3)

        # Same coefficients
        @test coef(m_ols) ≈ coef(m_hc3) atol=1e-10

        # Different standard errors (heteroskedasticity present)
        @test stderror(m_ols) != stderror(m_hc3)

        # Same R-squared
        @test r2(m_ols) ≈ r2(m_hc3) atol=1e-10
    end

    @testset "OLS — F-test significant" begin
        rng = MersenneTwister(202)
        n = 200
        X = hcat(ones(n), randn(rng, n, 3))
        y = X * [1.0, 3.0, -2.0, 1.5] + 0.5 * randn(rng, n)

        m = estimate_reg(y, X)
        @test m.f_stat > 10.0
        @test m.f_pval < 0.01
    end

    @testset "OLS — AIC/BIC finite" begin
        rng = MersenneTwister(303)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        y = X * [1.0, 0.5] + randn(rng, n)

        m = estimate_reg(y, X)
        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test isfinite(loglikelihood(m))
    end

    @testset "OLS — varnames auto-generated and custom" begin
        rng = MersenneTwister(404)
        n = 50
        X = hcat(ones(n), randn(rng, n, 2))
        y = randn(rng, n)

        # Auto-generated
        m1 = estimate_reg(y, X)
        @test m1.varnames == ["x1", "x2", "x3"]

        # Custom
        m2 = estimate_reg(y, X; varnames=["const", "income", "age"])
        @test m2.varnames == ["const", "income", "age"]
    end

    @testset "WLS estimation" begin
        rng = MersenneTwister(505)
        n = 200
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [1.0, 2.0, -1.0]

        # Heteroskedastic errors proportional to x1^2
        sigma_i = 0.3 .+ 2.0 .* X[:, 2].^2
        y = X * beta_true + sqrt.(sigma_i) .* randn(rng, n)

        # WLS with inverse variance weights
        w = 1.0 ./ sigma_i
        m = estimate_reg(y, X; weights=w)

        @test m.method == :wls
        @test m.weights !== nothing
        # WLS should still recover coefficients reasonably
        @test all(abs.(coef(m) .- beta_true) .< 0.5)
    end

    @testset "Intercept-only model" begin
        rng = MersenneTwister(606)
        n = 100
        X = ones(n, 1)
        y = 5.0 .+ randn(rng, n)

        m = estimate_reg(y, X; varnames=["const"])

        # Coefficient should be close to sample mean
        @test abs(coef(m)[1] - mean(y)) < 1e-10

        # R-squared should be 0 for intercept-only
        @test abs(r2(m)) < 1e-10

        # F-stat: no slopes to test
        @test m.f_stat == 0.0
        @test m.f_pval == 1.0
    end

    @testset "NaN validation" begin
        n = 50
        X = hcat(ones(n), randn(n))
        y = randn(n)
        y[10] = NaN

        @test_throws ArgumentError estimate_reg(y, X)

        y2 = randn(n)
        X2 = hcat(ones(n), randn(n))
        X2[5, 2] = NaN
        @test_throws ArgumentError estimate_reg(y2, X2)
    end

    @testset "Dimension mismatch" begin
        @test_throws ArgumentError estimate_reg(randn(50), randn(60, 2))
    end

    @testset "n <= k error" begin
        @test_throws ArgumentError estimate_reg(randn(3), randn(3, 4))
    end

    @testset "Float fallback (Integer input)" begin
        rng = MersenneTwister(707)
        n = 50
        X = hcat(ones(Int, n), rand(rng, 0:10, n))
        y = rand(rng, 0:20, n)

        # Should auto-convert to Float64 and work
        m = estimate_reg(y, X)
        @test length(coef(m)) == 2
        @test nobs(m) == n
    end

    @testset "Cluster estimation via estimate_reg" begin
        rng = MersenneTwister(808)
        n = 200
        G = 20
        clusters = repeat(1:G, inner=10)
        X = hcat(ones(n), randn(rng, n, 2))
        y = X * [1.0, 0.5, -0.3] + randn(rng, n)

        m = estimate_reg(y, X; cov_type=:cluster, clusters=clusters)
        @test m.cov_type == :cluster
        @test all(stderror(m) .> 0)
    end

    @testset "show method with IV fields" begin
        # Construct a model pretending to be IV for show coverage
        n, k = 50, 2
        rng = MersenneTwister(909)
        X = hcat(ones(n), randn(rng, n))
        y = randn(rng, n)
        beta = X \ y
        fitted_vals = X * beta
        resid = y .- fitted_vals
        ssr = dot(resid, resid)
        tss = sum((yi - mean(y))^2 for yi in y)

        m = RegModel{Float64}(
            y, X, beta, Matrix{Float64}(I, k, k) * 0.01,
            resid, fitted_vals,
            ssr, tss, 1.0 - ssr / tss, 0.5, 5.0, 0.05,
            -100.0, 204.0, 210.0,
            ["const", "x1"], :iv, :hc1,
            nothing,
            randn(rng, n, 3), [2], 15.0, 2.5, 0.3   # IV fields
        )

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("IV/2SLS", output)
        @test occursin("1st-stage F", output)
    end

    @testset "confint at different levels" begin
        rng = MersenneTwister(1010)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        y = X * [1.0, 2.0] + 0.5 * randn(rng, n)

        m = estimate_reg(y, X)

        ci90 = confint(m; level=0.90)
        ci95 = confint(m; level=0.95)
        ci99 = confint(m; level=0.99)

        # Wider levels produce wider intervals
        for j in 1:2
            width90 = ci90[j, 2] - ci90[j, 1]
            width95 = ci95[j, 2] - ci95[j, 1]
            width99 = ci99[j, 2] - ci99[j, 1]
            @test width99 > width95 > width90
        end
    end

end
