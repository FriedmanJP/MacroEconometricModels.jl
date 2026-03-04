using Test
using MacroEconometricModels
using MacroEconometricModels: dof_residual
using LinearAlgebra, Statistics, Random, Distributions

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

# =============================================================================
# Task 5: IV/2SLS Estimation
# =============================================================================

@testset "IV/2SLS Estimation (Task 5)" begin

    @testset "2SLS recovers true beta better than OLS" begin
        rng = MersenneTwister(5001)
        n = 1000
        beta_true = [1.0, 2.0]

        # Instruments
        z1 = randn(rng, n)
        z2 = randn(rng, n)

        # Endogenous regressor: correlated with error
        v = randn(rng, n)
        u = 0.8 * v + 0.6 * randn(rng, n)  # corr(u, v) != 0
        x_endog = 0.5 * z1 + 0.4 * z2 + v

        y = beta_true[1] .+ beta_true[2] .* x_endog .+ u

        X = hcat(ones(n), x_endog)
        Z = hcat(ones(n), z1, z2)

        # OLS is biased
        m_ols = estimate_reg(y, X)

        # IV should be closer to true beta
        m_iv = estimate_iv(y, X, Z; endogenous=[2])

        @test m_iv.method == :iv
        @test abs(coef(m_iv)[2] - beta_true[2]) < abs(coef(m_ols)[2] - beta_true[2])
        @test abs(coef(m_iv)[2] - beta_true[2]) < 0.3
    end

    @testset "First-stage F > 10 (strong instruments)" begin
        rng = MersenneTwister(5002)
        n = 500
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        v = randn(rng, n)
        u = 0.5 * v + randn(rng, n)
        x_endog = 0.7 * z1 + 0.5 * z2 + v
        y = 1.0 .+ 2.0 .* x_endog .+ u

        X = hcat(ones(n), x_endog)
        Z = hcat(ones(n), z1, z2)

        m = estimate_iv(y, X, Z; endogenous=[2])
        @test m.first_stage_f > 10.0
    end

    @testset "Sargan test: overidentified has stat, exactly identified is nothing" begin
        rng = MersenneTwister(5003)
        n = 500
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        z3 = randn(rng, n)
        v = randn(rng, n)
        u = 0.5 * v + randn(rng, n)
        x_endog = 0.5 * z1 + 0.3 * z2 + 0.2 * z3 + v
        y = 1.0 .+ 2.0 .* x_endog .+ u

        X = hcat(ones(n), x_endog)

        # Overidentified: 3 excluded instruments for 1 endogenous
        Z_over = hcat(ones(n), z1, z2, z3)
        m_over = estimate_iv(y, X, Z_over; endogenous=[2])
        @test m_over.sargan_stat !== nothing
        @test m_over.sargan_pval !== nothing
        # Valid instruments → p > 0.05 typically
        @test m_over.sargan_pval > 0.01

        # Exactly identified: 1 excluded instrument for 1 endogenous
        Z_exact = hcat(ones(n), z1)
        m_exact = estimate_iv(y, X, Z_exact; endogenous=[2])
        @test m_exact.sargan_stat === nothing
        @test m_exact.sargan_pval === nothing
    end

    @testset "IV — robust covariance types" begin
        rng = MersenneTwister(5004)
        n = 300
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        v = randn(rng, n)
        u = 0.5 * v + randn(rng, n)
        x_endog = 0.5 * z1 + 0.3 * z2 + v
        y = 1.0 .+ 1.5 .* x_endog .+ u

        X = hcat(ones(n), x_endog)
        Z = hcat(ones(n), z1, z2)

        for ct in [:ols, :hc0, :hc1, :hc2, :hc3]
            m = estimate_iv(y, X, Z; endogenous=[2], cov_type=ct)
            @test m.cov_type == ct
            @test all(stderror(m) .> 0)
            @test all(isfinite.(coef(m)))
        end
    end

    @testset "IV — error handling" begin
        n = 50
        X = hcat(ones(n), randn(n))
        Z = hcat(ones(n))  # only 1 instrument < 2 regressors
        y = randn(n)

        # Order condition violation
        @test_throws ArgumentError estimate_iv(y, X, Z; endogenous=[2])

        # Empty endogenous
        Z2 = hcat(ones(n), randn(n))
        @test_throws ArgumentError estimate_iv(y, X, Z2; endogenous=Int[])

        # Invalid endogenous index
        @test_throws ArgumentError estimate_iv(y, X, Z2; endogenous=[5])
    end

    @testset "IV — Float fallback" begin
        rng = MersenneTwister(5005)
        n = 100
        z1 = rand(rng, 0:10, n)
        x = z1 .+ rand(rng, 0:3, n)
        y = 2 .* x .+ rand(rng, 0:5, n)

        X = hcat(ones(Int, n), x)
        Z = hcat(ones(Int, n), z1)

        m = estimate_iv(y, X, Z; endogenous=[2])
        @test length(coef(m)) == 2
    end

    @testset "IV — show method" begin
        rng = MersenneTwister(5006)
        n = 200
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        v = randn(rng, n)
        u = 0.5 * v + randn(rng, n)
        x_endog = 0.5 * z1 + 0.3 * z2 + v
        y = 1.0 .+ 2.0 .* x_endog .+ u

        X = hcat(ones(n), x_endog)
        Z = hcat(ones(n), z1, z2)

        m = estimate_iv(y, X, Z; endogenous=[2])

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("IV/2SLS", output)
        @test occursin("1st-stage F", output)
    end

    @testset "IV — AIC/BIC/loglik finite" begin
        rng = MersenneTwister(5007)
        n = 200
        z1 = randn(rng, n)
        v = randn(rng, n)
        u = 0.3 * v + randn(rng, n)
        x_endog = 0.6 * z1 + v
        y = 1.0 .+ 1.0 .* x_endog .+ u

        X = hcat(ones(n), x_endog)
        Z = hcat(ones(n), z1)

        m = estimate_iv(y, X, Z; endogenous=[2])
        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test isfinite(loglikelihood(m))
    end

end

# =============================================================================
# Task 6: Logit Estimation
# =============================================================================

@testset "Logit Estimation (Task 6)" begin

    @testset "Logit coefficient recovery" begin
        rng = MersenneTwister(6001)
        n = 1000
        beta_true = [0.0, 1.5, -1.0]
        X = hcat(ones(n), randn(rng, n, 2))
        p = 1.0 ./ (1.0 .+ exp.(-X * beta_true))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        # Coefficient recovery (atol=0.3)
        for j in 1:3
            @test abs(coef(m)[j] - beta_true[j]) < 0.3
        end
    end

    @testset "Logit convergence and predictions" begin
        rng = MersenneTwister(6002)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [0.5, 1.0, -0.5]
        p = 1.0 ./ (1.0 .+ exp.(-X * beta_true))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        # Convergence
        @test m.converged == true
        @test m.iterations > 0

        # Predicted probabilities in (0, 1)
        @test all(0 .< predict(m) .< 1)

        # Pseudo R-squared > 0
        @test m.pseudo_r2 > 0

        # Log-likelihood > null log-likelihood
        @test m.loglik > m.loglik_null
    end

    @testset "Logit StatsAPI interface" begin
        rng = MersenneTwister(6003)
        n = 300
        X = hcat(ones(n), randn(rng, n))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        @test nobs(m) == n
        @test dof(m) == 2
        @test !islinear(m)

        ci = confint(m)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .< coef(m))
        @test all(ci[:, 2] .> coef(m))
    end

    @testset "Logit deviance residuals finite" begin
        rng = MersenneTwister(6004)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0, -0.5]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        @test all(isfinite.(residuals(m)))
        @test length(residuals(m)) == n
    end

    @testset "Logit AIC/BIC" begin
        rng = MersenneTwister(6005)
        n = 400
        X = hcat(ones(n), randn(rng, n))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 0.8]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test isfinite(loglikelihood(m))
    end

    @testset "Logit robust covariance" begin
        rng = MersenneTwister(6006)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0, -0.5]))
        y = Float64.(rand(rng, n) .< p)

        for ct in [:ols, :hc0, :hc1]
            m = estimate_logit(y, X; cov_type=ct)
            @test m.cov_type == ct
            @test all(stderror(m) .> 0)
        end
    end

    @testset "Logit error handling" begin
        n = 50
        X = hcat(ones(n), randn(n))

        # Non-binary y
        @test_throws ArgumentError estimate_logit(randn(n), X)

        # Dimension mismatch
        @test_throws ArgumentError estimate_logit(Float64.([0,1,0,1,1]), randn(10, 2))
    end

    @testset "Logit Float fallback" begin
        rng = MersenneTwister(6007)
        n = 100
        X = hcat(ones(Int, n), rand(rng, 0:1, n))
        y = rand(rng, 0:1, n)

        m = estimate_logit(y, X)
        @test length(coef(m)) == 2
    end

    @testset "Logit show method" begin
        rng = MersenneTwister(6008)
        n = 200
        X = hcat(ones(n), randn(rng, n))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Logit Regression", output)
        @test occursin("Pseudo R-sq", output)
    end

end

# =============================================================================
# Task 7: Probit Estimation
# =============================================================================

@testset "Probit Estimation (Task 7)" begin

    @testset "Probit coefficient recovery" begin
        rng = MersenneTwister(7001)
        n = 1000
        beta_true = [0.0, 1.0, -0.8]
        X = hcat(ones(n), randn(rng, n, 2))
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * beta_true)
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)

        # Coefficient recovery
        for j in 1:3
            @test abs(coef(m)[j] - beta_true[j]) < 0.3
        end
    end

    @testset "Probit convergence and predictions" begin
        rng = MersenneTwister(7002)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [0.3, 0.8, -0.5]
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * beta_true)
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)

        @test m.converged == true
        @test m.iterations > 0
        @test all(0 .< predict(m) .< 1)
        @test m.pseudo_r2 > 0
    end

    @testset "Probit beta ≈ logit beta / 1.6" begin
        rng = MersenneTwister(7003)
        n = 2000
        beta_true_logit = [0.0, 2.0, -1.5]
        X = hcat(ones(n), randn(rng, n, 2))
        p = 1.0 ./ (1.0 .+ exp.(-X * beta_true_logit))
        y = Float64.(rand(rng, n) .< p)

        m_logit = estimate_logit(y, X)
        m_probit = estimate_probit(y, X)

        # Probit coefficients should be approximately logit / 1.6
        for j in 1:3
            ratio = abs(coef(m_logit)[j]) > 0.1 ?
                coef(m_probit)[j] / coef(m_logit)[j] : NaN
            if !isnan(ratio)
                @test abs(ratio - 1.0 / 1.6) < 0.15
            end
        end
    end

    @testset "Probit StatsAPI interface" begin
        rng = MersenneTwister(7004)
        n = 300
        X = hcat(ones(n), randn(rng, n))
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * [0.0, 0.8])
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)

        @test nobs(m) == n
        @test dof(m) == 2
        @test !islinear(m)

        ci = confint(m)
        @test size(ci) == (2, 2)
    end

    @testset "Probit deviance residuals finite" begin
        rng = MersenneTwister(7005)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * [0.0, 0.8, -0.5])
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)
        @test all(isfinite.(residuals(m)))
    end

    @testset "Probit robust covariance" begin
        rng = MersenneTwister(7006)
        n = 500
        X = hcat(ones(n), randn(rng, n))
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * [0.0, 1.0])
        y = Float64.(rand(rng, n) .< p)

        for ct in [:ols, :hc0, :hc1]
            m = estimate_probit(y, X; cov_type=ct)
            @test m.cov_type == ct
            @test all(stderror(m) .> 0)
        end
    end

    @testset "Probit error handling" begin
        n = 50
        X = hcat(ones(n), randn(n))

        # Non-binary y
        @test_throws ArgumentError estimate_probit(randn(n), X)
    end

    @testset "Probit Float fallback" begin
        rng = MersenneTwister(7007)
        n = 100
        X = hcat(ones(Int, n), rand(rng, 0:1, n))
        y = rand(rng, 0:1, n)

        m = estimate_probit(y, X)
        @test length(coef(m)) == 2
    end

    @testset "Probit show method" begin
        rng = MersenneTwister(7008)
        n = 200
        X = hcat(ones(n), randn(rng, n))
        d = Distributions.Normal()
        p = Distributions.cdf.(d, X * [0.0, 0.8])
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)

        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Probit Regression", output)
        @test occursin("Pseudo R-sq", output)
    end

end

# =============================================================================
# Task 8: Marginal Effects and Odds Ratios
# =============================================================================

@testset "Marginal Effects and Odds Ratios (Task 8)" begin

    # ---- Shared data for logit ----
    rng_logit = MersenneTwister(8001)
    n_logit = 1000
    beta_true_logit = [0.0, 1.5, -1.0]
    X_logit = hcat(ones(n_logit), randn(rng_logit, n_logit, 2))
    p_logit = 1.0 ./ (1.0 .+ exp.(-X_logit * beta_true_logit))
    y_logit = Float64.(rand(rng_logit, n_logit) .< p_logit)
    m_logit = estimate_logit(y_logit, X_logit; varnames=["const", "x1", "x2"])

    # ---- Shared data for probit ----
    rng_probit = MersenneTwister(8002)
    n_probit = 1000
    beta_true_probit = [0.0, 1.0, -0.8]
    X_probit = hcat(ones(n_probit), randn(rng_probit, n_probit, 2))
    d_norm = Distributions.Normal()
    p_probit = Distributions.cdf.(d_norm, X_probit * beta_true_probit)
    y_probit = Float64.(rand(rng_probit, n_probit) .< p_probit)
    m_probit = estimate_probit(y_probit, X_probit; varnames=["const", "x1", "x2"])

    @testset "Logit AME — basic properties" begin
        me = marginal_effects(m_logit)

        @test me.type == :ame
        @test length(me.effects) == 3
        @test length(me.se) == 3
        @test length(me.z_stat) == 3
        @test length(me.p_values) == 3
        @test me.conf_level == 0.95
        @test me.varnames == ["const", "x1", "x2"]

        # SEs should be positive
        @test all(me.se .> 0)

        # CIs should bracket effects
        @test all(me.ci_lower .< me.effects)
        @test all(me.ci_upper .> me.effects)

        # AME for slope coefficients should be non-trivial
        # For logit: AME_j ≈ mean(p*(1-p)) * beta_j
        # With beta2 = 1.5, AME should be positive
        @test me.effects[2] > 0
        # With beta3 = -1.0, AME should be negative
        @test me.effects[3] < 0
    end

    @testset "Logit MEM — basic properties" begin
        me = marginal_effects(m_logit; type=:mem)

        @test me.type == :mem
        @test length(me.effects) == 3
        @test all(me.se .> 0)
        @test all(me.ci_lower .< me.effects)
        @test all(me.ci_upper .> me.effects)

        # MEM should have same sign as AME for monotone link
        me_ame = marginal_effects(m_logit; type=:ame)
        for j in 2:3
            @test sign(me.effects[j]) == sign(me_ame.effects[j])
        end
    end

    @testset "Logit MER — basic properties" begin
        me = marginal_effects(m_logit; type=:mer, at=Dict(2 => 0.0, 3 => 0.0))

        @test me.type == :mer
        @test length(me.effects) == 3
        @test all(me.se .> 0)
    end

    @testset "MER error — missing at argument" begin
        @test_throws ArgumentError marginal_effects(m_logit; type=:mer)
    end

    @testset "MER error — invalid column index" begin
        @test_throws ArgumentError marginal_effects(m_logit; type=:mer, at=Dict(99 => 0.0))
    end

    @testset "Invalid type error" begin
        @test_throws ArgumentError marginal_effects(m_logit; type=:invalid)
    end

    @testset "Probit AME — basic properties" begin
        me = marginal_effects(m_probit)

        @test me.type == :ame
        @test length(me.effects) == 3
        @test all(me.se .> 0)
        @test all(me.ci_lower .< me.effects)
        @test all(me.ci_upper .> me.effects)

        # Probit AME for x1 (beta=1.0) should be positive
        @test me.effects[2] > 0
        # Probit AME for x2 (beta=-0.8) should be negative
        @test me.effects[3] < 0
    end

    @testset "Probit MEM and MER" begin
        me_mem = marginal_effects(m_probit; type=:mem)
        @test me_mem.type == :mem
        @test length(me_mem.effects) == 3

        me_mer = marginal_effects(m_probit; type=:mer, at=Dict(2 => 0.5))
        @test me_mer.type == :mer
        @test length(me_mer.effects) == 3
    end

    @testset "AME magnitude: logit vs probit" begin
        # AME for logit and probit on similar data should be comparable
        # For balanced data, AME_logit ≈ AME_probit (approximately)
        me_l = marginal_effects(m_logit)
        me_p = marginal_effects(m_probit)

        # Both should identify same direction for slopes
        @test sign(me_l.effects[2]) == sign(me_p.effects[2])
        @test sign(me_l.effects[3]) == sign(me_p.effects[3])
    end

    @testset "Confidence level affects CI width" begin
        me_90 = marginal_effects(m_logit; conf_level=0.90)
        me_95 = marginal_effects(m_logit; conf_level=0.95)
        me_99 = marginal_effects(m_logit; conf_level=0.99)

        for j in 1:3
            w90 = me_90.ci_upper[j] - me_90.ci_lower[j]
            w95 = me_95.ci_upper[j] - me_95.ci_lower[j]
            w99 = me_99.ci_upper[j] - me_99.ci_lower[j]
            @test w99 > w95 > w90
        end
    end

    @testset "MarginalEffects show method" begin
        me = marginal_effects(m_logit)
        io = IOBuffer()
        show(io, me)
        output = String(take!(io))
        @test occursin("Average Marginal Effects", output)

        me_mem = marginal_effects(m_logit; type=:mem)
        io2 = IOBuffer()
        show(io2, me_mem)
        output2 = String(take!(io2))
        @test occursin("Marginal Effects at Mean", output2)
    end

    @testset "Odds ratios — basic properties" begin
        result = odds_ratio(m_logit)

        @test length(result.or) == 3
        @test length(result.se) == 3
        @test length(result.ci_lower) == 3
        @test length(result.ci_upper) == 3
        @test result.varnames == ["const", "x1", "x2"]

        # OR = exp(beta), should be positive
        @test all(result.or .> 0)

        # OR should equal exp(beta) exactly
        @test result.or ≈ exp.(coef(m_logit))

        # SEs positive
        @test all(result.se .> 0)

        # CIs should bracket ORs
        @test all(result.ci_lower .< result.or)
        @test all(result.ci_upper .> result.or)

        # For positive beta (x1 ~ 1.5), OR > 1
        @test result.or[2] > 1.0

        # For negative beta (x2 ~ -1.0), OR < 1
        @test result.or[3] < 1.0
    end

    @testset "Odds ratios — confidence level" begin
        or_90 = odds_ratio(m_logit; conf_level=0.90)
        or_95 = odds_ratio(m_logit; conf_level=0.95)
        or_99 = odds_ratio(m_logit; conf_level=0.99)

        # Same ORs regardless of conf level
        @test or_90.or ≈ or_95.or
        @test or_95.or ≈ or_99.or

        # Wider CIs for higher confidence
        for j in 1:3
            w90 = or_90.ci_upper[j] - or_90.ci_lower[j]
            w95 = or_95.ci_upper[j] - or_95.ci_lower[j]
            w99 = or_99.ci_upper[j] - or_99.ci_lower[j]
            @test w99 > w95 > w90
        end
    end

    @testset "Odds ratios — SE via delta method" begin
        result = odds_ratio(m_logit)
        se_beta = sqrt.(diag(m_logit.vcov_mat))

        # Delta method: SE(OR_j) = OR_j * SE(beta_j)
        expected_se = result.or .* se_beta
        @test result.se ≈ expected_se
    end

end

# =============================================================================
# Task 9: Diagnostics — VIF and Classification Table
# =============================================================================

@testset "Diagnostics — VIF and Classification Table (Task 9)" begin

    @testset "VIF — low collinearity" begin
        rng = MersenneTwister(9001)
        n = 500
        X = hcat(ones(n), randn(rng, n, 3))
        y = X * [1.0, 2.0, -0.5, 0.3] + 0.5 * randn(rng, n)

        m = estimate_reg(y, X; varnames=["const", "x1", "x2", "x3"])
        v = vif(m)

        # With independent regressors, VIF should be near 1
        @test length(v) == 3  # excludes intercept
        @test all(v .> 0.9)
        @test all(v .< 5.0)
    end

    @testset "VIF — high collinearity" begin
        rng = MersenneTwister(9002)
        n = 500
        x1 = randn(rng, n)
        x2 = x1 .+ 0.01 .* randn(rng, n)  # nearly collinear with x1
        X = hcat(ones(n), x1, x2)
        y = X * [1.0, 1.0, 1.0] + randn(rng, n)

        m = estimate_reg(y, X; varnames=["const", "x1", "x2"])
        v = vif(m)

        # VIF should be very high for both collinear regressors
        @test length(v) == 2
        @test all(v .> 10.0)
    end

    @testset "VIF — single regressor" begin
        rng = MersenneTwister(9003)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        y = X * [1.0, 2.0] + randn(rng, n)

        m = estimate_reg(y, X)
        v = vif(m)

        # With single non-intercept regressor, VIF should be 1
        @test length(v) == 1
        @test v[1] ≈ 1.0 atol=0.1
    end

    @testset "Classification table — logit" begin
        rng = MersenneTwister(9004)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [0.0, 2.0, -1.5]
        p = 1.0 ./ (1.0 .+ exp.(-X * beta_true))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)
        ct = classification_table(m)

        # Check keys
        @test haskey(ct, "confusion")
        @test haskey(ct, "accuracy")
        @test haskey(ct, "sensitivity")
        @test haskey(ct, "specificity")
        @test haskey(ct, "precision")
        @test haskey(ct, "f1_score")
        @test haskey(ct, "n")
        @test haskey(ct, "threshold")

        # Confusion matrix 2x2
        @test size(ct["confusion"]) == (2, 2)

        # All metrics in [0, 1]
        @test 0 <= ct["accuracy"] <= 1
        @test 0 <= ct["sensitivity"] <= 1
        @test 0 <= ct["specificity"] <= 1
        @test 0 <= ct["precision"] <= 1
        @test 0 <= ct["f1_score"] <= 1

        # n should match
        @test ct["n"] == n

        # Confusion matrix should sum to n
        @test sum(ct["confusion"]) ≈ n

        # Accuracy = (TN + TP) / n
        conf = ct["confusion"]
        tn, fp, fn, tp = conf[1,1], conf[1,2], conf[2,1], conf[2,2]
        @test ct["accuracy"] ≈ (tp + tn) / n

        # With strong signal, accuracy should be decent
        @test ct["accuracy"] > 0.6
    end

    @testset "Classification table — probit" begin
        rng = MersenneTwister(9005)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        d_norm = Distributions.Normal()
        p = Distributions.cdf.(d_norm, X * [0.0, 1.5, -1.0])
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)
        ct = classification_table(m)

        @test size(ct["confusion"]) == (2, 2)
        @test sum(ct["confusion"]) ≈ n
        @test 0 <= ct["accuracy"] <= 1
        @test ct["accuracy"] > 0.6
    end

    @testset "Classification table — custom threshold" begin
        rng = MersenneTwister(9006)
        n = 300
        X = hcat(ones(n), randn(rng, n))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        ct_50 = classification_table(m; threshold=0.5)
        ct_30 = classification_table(m; threshold=0.3)
        ct_70 = classification_table(m; threshold=0.7)

        # Lower threshold → more positives → higher sensitivity, lower specificity
        @test ct_30["sensitivity"] >= ct_50["sensitivity"]
        @test ct_70["specificity"] >= ct_50["specificity"]

        @test ct_50["threshold"] == 0.5
        @test ct_30["threshold"] ≈ 0.3
    end

    @testset "Classification table — perfect separation" begin
        # If model is very good, accuracy should be near 1
        rng = MersenneTwister(9007)
        n = 200
        x = randn(rng, n)
        X = hcat(ones(n), x)
        # Very strong signal
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 10.0]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)
        ct = classification_table(m)

        @test ct["accuracy"] > 0.85
    end

    @testset "F1 score consistency" begin
        rng = MersenneTwister(9008)
        n = 400
        X = hcat(ones(n), randn(rng, n, 2))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.5, -1.0]))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)
        ct = classification_table(m)

        # F1 = 2 * precision * recall / (precision + recall)
        prec = ct["precision"]
        recall = ct["sensitivity"]
        if prec + recall > 0
            expected_f1 = 2 * prec * recall / (prec + recall)
            @test ct["f1_score"] ≈ expected_f1
        end
    end

end

# =============================================================================
# Task 10: Predict Dispatches
# =============================================================================

@testset "Predict Dispatches (Task 10)" begin

    @testset "RegModel predict — new data" begin
        rng = MersenneTwister(10001)
        n = 200
        beta_true = [1.0, 2.0, -0.5]
        k = length(beta_true)
        X = hcat(ones(n), randn(rng, n, k - 1))
        y = X * beta_true + 0.3 * randn(rng, n)

        m = estimate_reg(y, X)

        # Predict on training data should match fitted values
        y_hat_train = predict(m, X)
        @test y_hat_train ≈ predict(m) atol=1e-10

        # Predict on new data
        n_new = 50
        X_new = hcat(ones(n_new), randn(rng, n_new, k - 1))
        y_hat = predict(m, X_new)

        @test length(y_hat) == n_new
        @test all(isfinite.(y_hat))

        # Should equal X_new * beta
        @test y_hat ≈ X_new * coef(m) atol=1e-10
    end

    @testset "RegModel predict — dimension mismatch" begin
        rng = MersenneTwister(10002)
        n = 100
        X = hcat(ones(n), randn(rng, n, 2))
        y = randn(rng, n)
        m = estimate_reg(y, X)

        # Wrong number of columns
        @test_throws ArgumentError predict(m, randn(10, 5))
    end

    @testset "LogitModel predict — new data" begin
        rng = MersenneTwister(10003)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [0.0, 1.5, -1.0]
        p = 1.0 ./ (1.0 .+ exp.(-X * beta_true))
        y = Float64.(rand(rng, n) .< p)

        m = estimate_logit(y, X)

        # Predict on training data should match fitted values
        p_hat_train = predict(m, X)
        @test p_hat_train ≈ predict(m) atol=1e-8

        # Predict on new data
        n_new = 50
        X_new = hcat(ones(n_new), randn(rng, n_new, 2))
        p_hat = predict(m, X_new)

        @test length(p_hat) == n_new
        @test all(isfinite.(p_hat))

        # Predictions should be probabilities in (0, 1)
        @test all(0 .< p_hat .< 1)

        # Should equal logistic function applied to linear predictor
        eta = X_new * coef(m)
        expected = 1.0 ./ (1.0 .+ exp.(-eta))
        @test p_hat ≈ expected atol=1e-10
    end

    @testset "LogitModel predict — dimension mismatch" begin
        rng = MersenneTwister(10004)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        p = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0]))
        y = Float64.(rand(rng, n) .< p)
        m = estimate_logit(y, X)

        @test_throws ArgumentError predict(m, randn(10, 5))
    end

    @testset "ProbitModel predict — new data" begin
        rng = MersenneTwister(10005)
        n = 500
        X = hcat(ones(n), randn(rng, n, 2))
        beta_true = [0.0, 1.0, -0.8]
        d_norm = Distributions.Normal()
        p = Distributions.cdf.(d_norm, X * beta_true)
        y = Float64.(rand(rng, n) .< p)

        m = estimate_probit(y, X)

        # Predict on training data should match fitted values
        p_hat_train = predict(m, X)
        @test p_hat_train ≈ predict(m) atol=1e-8

        # Predict on new data
        n_new = 50
        X_new = hcat(ones(n_new), randn(rng, n_new, 2))
        p_hat = predict(m, X_new)

        @test length(p_hat) == n_new
        @test all(isfinite.(p_hat))

        # Predictions should be probabilities in (0, 1)
        @test all(0 .< p_hat .< 1)

        # Should equal Phi(X_new * beta)
        eta = X_new * coef(m)
        expected = Distributions.cdf.(d_norm, eta)
        @test p_hat ≈ expected atol=1e-10
    end

    @testset "ProbitModel predict — dimension mismatch" begin
        rng = MersenneTwister(10006)
        n = 100
        X = hcat(ones(n), randn(rng, n))
        d_norm = Distributions.Normal()
        p = Distributions.cdf.(d_norm, X * [0.0, 0.8])
        y = Float64.(rand(rng, n) .< p)
        m = estimate_probit(y, X)

        @test_throws ArgumentError predict(m, randn(10, 5))
    end

    @testset "Predict consistency across models" begin
        # Generate data, fit all three model types, check predict dimensions
        rng = MersenneTwister(10007)
        n = 200
        X = hcat(ones(n), randn(rng, n, 2))

        # RegModel
        y_cont = X * [1.0, 0.5, -0.3] + randn(rng, n)
        m_reg = estimate_reg(y_cont, X)
        y_hat_reg = predict(m_reg, X)
        @test length(y_hat_reg) == n

        # LogitModel
        p_logit = 1.0 ./ (1.0 .+ exp.(-X * [0.0, 1.0, -0.5]))
        y_bin = Float64.(rand(rng, n) .< p_logit)
        m_logit = estimate_logit(y_bin, X)
        y_hat_logit = predict(m_logit, X)
        @test length(y_hat_logit) == n
        @test all(0 .< y_hat_logit .< 1)

        # ProbitModel
        m_probit = estimate_probit(y_bin, X)
        y_hat_probit = predict(m_probit, X)
        @test length(y_hat_probit) == n
        @test all(0 .< y_hat_probit .< 1)
    end

end
