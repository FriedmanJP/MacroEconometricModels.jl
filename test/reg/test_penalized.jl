# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using MacroEconometricModels: _soft, _standardize, _ridge_fit
using LinearAlgebra, Statistics, Random
import StatsAPI

# =============================================================================
# Deterministic DGP shared with the offline sklearn oracle (no RNG mismatch).
# X[i,j] = sin(0.3i + 0.7j) + 0.5 cos(0.2 i j); y = X*beta_true + 0.4 sin(1.1 i).
# Standardization uses population std (1/n), matching glmnet/sklearn below.
# =============================================================================
function _penalized_dgp(n::Int=50, p::Int=8)
    X = [sin(0.3i + 0.7j) + 0.5 * cos(0.2 * i * j) for i in 1:n, j in 1:p]
    beta_true = [1.5, 0.0, -2.0, 0.0, 0.0, 0.8, 0.0, 0.0]
    y = X * beta_true .+ [0.4 * sin(1.1i) for i in 1:n]
    (X, y, beta_true)
end

@testset "Penalized Regression (EV-03, #411)" begin

    X, y, beta_true = _penalized_dgp()
    n, p = size(X)

    # =========================================================================
    # Soft-threshold operator
    # =========================================================================
    @testset "soft-threshold operator" begin
        @test _soft(3.0, 1.0) == 2.0
        @test _soft(-3.0, 1.0) == -2.0
        @test _soft(0.5, 1.0) == 0.0
        @test _soft(0.0, 1.0) == 0.0
    end

    # =========================================================================
    # Ridge exactness: standardized coefficients equal the closed form to 1e-8.
    # Convention: glmnet (1/2n)-SSE scaling => (Xs'Xs + n*lambda*I)^{-1} Xs'yc,
    # equivalently (Xs'Xs/n + lambda I)^{-1} (Xs'yc/n).
    # =========================================================================
    @testset "ridge closed-form exactness" begin
        Xs, xbar, xsd, yc, ybar = _standardize(X, y)
        for λ in (0.05, 0.1, 0.5, 1.0)
            m = estimate_ridge(y, X; lambda=λ)
            β_ref = (Xs'Xs + n * λ * I) \ (Xs'yc)
            @test maximum(abs.(m.beta_std .- β_ref)) < 1e-8
            # internal helper agrees
            β_h, df_h = _ridge_fit(Xs, yc, λ)
            @test maximum(abs.(β_h .- β_ref)) < 1e-8
            # effective df in (0, p]
            @test 0 < df_h <= p + 1e-8
        end
    end

    # =========================================================================
    # Zero-lambda LASSO == OLS (estimate_reg) to 1e-8. Catches unstandardize bug.
    # =========================================================================
    @testset "lambda=0 LASSO equals OLS" begin
        Xf = hcat(ones(n), X)
        ols = estimate_reg(y, Xf; cov_type=:ols)
        m = estimate_lasso(y, X; lambda=[0.0])
        β_pen = vcat(m.beta0, m.beta)      # [intercept; slopes]
        @test maximum(abs.(β_pen .- ols.beta)) < 1e-8
    end

    # =========================================================================
    # predict == X*beta_natural + beta0 exactly (missing-unstandardize guard).
    # =========================================================================
    @testset "predict equals X*beta_natural + beta0" begin
        for (mk, kw) in (
            (estimate_lasso, (; lambda=0.05)),
            (estimate_ridge, (; lambda=0.1)),
            (estimate_elastic_net, (; alpha=0.5, lambda=0.1)))
            m = mk(y, X; kw...)
            @test maximum(abs.(predict(m) .- (X * m.beta .+ m.beta0))) < 1e-12
            @test predict(m) == m.fitted
            # predict on new data matches the same linear form
            Xnew = X[1:5, :]
            @test maximum(abs.(predict(m, Xnew) .- (Xnew * m.beta .+ m.beta0))) < 1e-12
        end
    end

    # =========================================================================
    # Cross-implementation oracle: Python sklearn.linear_model (v1.7.1).
    # Objective match: sklearn ElasticNet(alpha=lambda, l1_ratio=alpha_mix,
    #   fit_intercept=False) on standardized Xs & centered yc reproduces glmnet's
    #   (1/2n)||yc-Xs w||^2 + lambda[alpha*|w|_1 + (1-alpha)/2 |w|^2].
    # Reference coefficients (standardized scale) generated offline with:
    #   from sklearn.linear_model import ElasticNet
    #   m = ElasticNet(alpha=lam, l1_ratio=a, fit_intercept=False,
    #                  tol=1e-14, max_iter=2_000_000).fit(Xs, yc)
    # on the DGP above (Xs population-standardized, yc centered).
    # =========================================================================
    @testset "sklearn ElasticNet oracle (standardized coefficients)" begin
        # (alpha_mix, lambda) => expected standardized coefficient vector
        cases = [
            (1.0, 0.05, [0.9859756971, 0.0, -1.6111597386, 0.0, 0.0,
                         0.4060995892, 0.0, 0.0126389901]),
            (1.0, 0.15, [0.6071304302, 0.0, -1.6246995185, 0.0, 0.0,
                         0.0271442121, 0.0, 0.0460664429]),
            (0.5, 0.05, [0.9954060040, -0.0044250909, -1.5090040393, 0.0, 0.0,
                         0.4508842292, 0.0, 0.1040736155]),
            (0.5, 0.10, [0.8359197046, -0.0046001543, -1.4133808241, -0.0448933970, 0.0,
                         0.3405105708, 0.0044425167, 0.1654284929]),
            (0.0, 0.10, [0.8449335646, -0.0683706333, -1.2250447188, -0.1086930408,
                         -0.0670572723, 0.4075301988, 0.0706785910, 0.1983390141]),
        ]
        for (a, lam, ref) in cases
            m = estimate_elastic_net(y, X; alpha=a, lambda=lam, tol=1e-12)
            @test maximum(abs.(m.beta_std .- ref)) < 1e-6
        end
    end

    # =========================================================================
    # Elastic-net alpha=0 limit reduces to ridge (internal consistency).
    # =========================================================================
    @testset "alpha=0 elastic net equals ridge" begin
        m0 = estimate_elastic_net(y, X; alpha=0.0, lambda=0.2)
        mr = estimate_ridge(y, X; lambda=0.2)
        @test maximum(abs.(m0.beta_std .- mr.beta_std)) < 1e-8
    end

    # =========================================================================
    # LASSO active-set size monotone non-increasing as lambda rises.
    # =========================================================================
    @testset "LASSO active-set monotonicity in lambda" begin
        m = estimate_lasso(y, X; select=:bic)         # builds full path
        L = length(m.lambda_path)
        # path is descending in lambda: index 1 = largest lambda
        active_counts = [count(!iszero, @view m.coef_path[:, k]) for k in 1:L]
        # As lambda decreases (k increases), the active set grows in aggregate. Exact
        # step-by-step monotonicity is not guaranteed for correlated predictors (a variable
        # can leave as another enters), so we assert the true global/robust properties:
        #   (i) largest lambda zeros (nearly) everything,
        #   (ii) smallest lambda has the (weakly) largest support,
        #   (iii) the trend is overwhelmingly non-decreasing.
        @test active_counts[1] <= 1
        @test active_counts[end] == maximum(active_counts)
        @test active_counts[end] >= active_counts[1]
        nondecr = count(k -> active_counts[k+1] >= active_counts[k], 1:L-1)
        @test nondecr >= L - 2                         # at most one transient drop
    end

    # =========================================================================
    # Adaptive LASSO recovers the known sparse support at the CV lambda.
    # True support: {1, 3, 6}. Seeded CV for reproducibility.
    # =========================================================================
    @testset "adaptive LASSO sparse support recovery" begin
        Random.seed!(0)
        # stronger-signal DGP so recovery is unambiguous
        nn, pp = 120, 10
        Xr = [sin(0.21i + 0.5j) + 0.4cos(0.13 * i * j) for i in 1:nn, j in 1:pp]
        bt = zeros(pp); bt[1] = 2.5; bt[3] = -3.0; bt[6] = 1.8
        yr = Xr * bt .+ [0.15 * sin(0.9i) for i in 1:nn]
        m = estimate_lasso(yr, Xr; adaptive=true, cv=:kfold, nfolds=5, seed=7)
        true_support = Set([1, 3, 6])
        # all true predictors selected
        @test issubset(true_support, Set(m.active_set))
        # coefficient signs correct on the true support
        @test m.beta[1] > 0 && m.beta[3] < 0 && m.beta[6] > 0
        @test m.adaptive == true
    end

    # =========================================================================
    # CV selection: kfold and timeseries both run and produce sane output.
    # timeseries must NOT shuffle (contiguous folds).
    # =========================================================================
    @testset "CV selection (kfold & timeseries)" begin
        mk = estimate_lasso(y, X; cv=:kfold, nfolds=5, seed=42)
        @test mk.cv_mse !== nothing
        @test length(mk.cv_mse) == length(mk.lambda_path)
        @test all(isfinite, mk.cv_mse)
        @test mk.lambda_min > 0
        # 1-SE lambda is >= lambda_min (more parsimonious)
        @test mk.lambda_1se >= mk.lambda_min - 1e-12
        @test mk.select == :cv

        mt = estimate_lasso(y, X; cv=:timeseries, nfolds=5)
        @test mt.cv === :timeseries
        @test mt.cv_mse !== nothing
        @test all(isfinite, mt.cv_mse)
    end

    # =========================================================================
    # IC selection: aic/bic/ebic all run; ebic penalty >= bic penalty.
    # =========================================================================
    @testset "IC selection (aic/bic/ebic)" begin
        for sel in (:aic, :bic, :ebic)
            m = estimate_lasso(y, X; select=sel)
            @test m.select == sel
            @test isfinite(m.aic) && isfinite(m.bic) && isfinite(m.ebic)
            @test m.lambda > 0
        end
        m = estimate_lasso(y, X; select=:bic)
        @test m.ebic >= m.bic - 1e-8   # EBIC adds a nonnegative penalty
    end

    # =========================================================================
    # Post-LASSO OLS refit: coefficients on the selected support equal OLS on
    # that support (unbiased, de-shrunk).
    # =========================================================================
    @testset "post-LASSO OLS refit" begin
        m = estimate_lasso(y, X; lambda=0.05, post=true)
        @test m.post == true
        supp = m.active_set
        @test !isempty(supp)
        Xsup = hcat(ones(n), X[:, supp])
        b_ols = Xsup \ y
        @test abs(m.beta0 - b_ols[1]) < 1e-8
        @test maximum(abs.(m.beta[supp] .- b_ols[2:end])) < 1e-8
        # coefficients off the support are exactly zero
        @test all(iszero, m.beta[setdiff(1:p, supp)])
    end

    # =========================================================================
    # Wrappers and API surface.
    # =========================================================================
    @testset "wrappers, coef, StatsAPI, dof" begin
        ml = estimate_lasso(y, X; lambda=0.05)
        @test ml.alpha == 1.0
        mr = estimate_ridge(y, X; lambda=0.1)
        @test mr.alpha == 0.0
        me = estimate_elastic_net(y, X; alpha=0.3, lambda=0.05)
        @test me.alpha == 0.3
        @test coef(ml) === ml.beta
        @test nobs(ml) == n
        @test length(residuals(ml)) == n
        @test isfinite(loglikelihood(ml))
        @test StatsAPI.dof(ml) ≈ ml.df_star + 1
        @test 0 <= r2(ml) <= 1
    end

    # =========================================================================
    # Input validation.
    # =========================================================================
    @testset "input validation" begin
        @test_throws ArgumentError estimate_elastic_net(y, X; alpha=1.5)
        @test_throws ArgumentError estimate_elastic_net(y, X; select=:foo)
        @test_throws ArgumentError estimate_elastic_net(y, X; cv=:foo)
        @test_throws ArgumentError estimate_elastic_net(y[1:10], X)  # length mismatch
        @test_throws ArgumentError estimate_lasso(y, X; lambda=0.05, varnames=["a"])
    end

    # =========================================================================
    # Integer / mixed input coercion.
    # =========================================================================
    @testset "integer input coercion" begin
        Xi = round.(Int, X .* 3)
        yi = round.(Int, y .* 3)
        m = estimate_lasso(yi, Xi; lambda=0.1)
        @test m isa PenalizedRegModel{Float64}
    end

    # =========================================================================
    # Display: report / show / refs render without error.
    # =========================================================================
    @testset "report, show, refs" begin
        m = estimate_lasso(y, X; lambda=0.05)
        buf = IOBuffer()
        show(buf, m)
        s = String(take!(buf))
        @test occursin("LASSO", s)
        @test occursin("Coefficients", s)
        @test occursin("no valid standard errors", s)

        # ridge & elastic-net variant labels
        buf2 = IOBuffer(); show(buf2, estimate_ridge(y, X; lambda=0.1))
        @test occursin("Ridge", String(take!(buf2)))
        buf3 = IOBuffer(); show(buf3, estimate_elastic_net(y, X; alpha=0.5, lambda=0.05))
        @test occursin("Elastic Net", String(take!(buf3)))
        buf4 = IOBuffer(); show(buf4, estimate_lasso(y, X; lambda=0.05, adaptive=true, post=true))
        s4 = String(take!(buf4))
        @test occursin("Adaptive", s4) && occursin("post-OLS", s4)

        # refs
        rbuf = IOBuffer()
        refs(rbuf, m; format=:text)
        rs = String(take!(rbuf))
        @test occursin("Tibshirani", rs)
    end

    # =========================================================================
    # Plotting: path and CV views produce PlotOutput; bad view errors.
    # =========================================================================
    @testset "plot_result path & cv" begin
        m = estimate_lasso(y, X; cv=:kfold, nfolds=5, seed=1)
        pp = plot_result(m; view=:path)
        @test pp isa MacroEconometricModels.PlotOutput
        @test occursin("svg", lowercase(pp.html)) || occursin("d3", lowercase(pp.html))
        pc = plot_result(m; view=:cv)
        @test pc isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(m; view=:bogus)
        # CV view on a non-CV (fixed-lambda) fit errors
        mf = estimate_lasso(y, X; lambda=0.05)
        @test_throws ArgumentError plot_result(mf; view=:cv)
    end
end
