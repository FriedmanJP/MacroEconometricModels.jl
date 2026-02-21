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

using Test
using MacroEconometricModels
using Statistics
using Random
using StatsAPI
using LinearAlgebra

# Set seed for reproducibility
Random.seed!(42)

@testset "AR Model Estimation" begin
    @testset "AR(1) OLS estimation" begin
        # Generate AR(1) data: yₜ = 0.5 + 0.7yₜ₋₁ + εₜ
        n = 500
        phi_true = 0.7
        c_true = 0.5
        sigma_true = 1.0

        y = zeros(n)
        y[1] = c_true / (1 - phi_true) + randn()
        for t in 2:n
            y[t] = c_true + phi_true * y[t-1] + sigma_true * randn()
        end

        model = estimate_ar(y, 1; method=:ols)

        @test model.p == 1
        @test length(model.phi) == 1
        @test abs(model.phi[1] - phi_true) < 0.1  # Within 0.1 of true
        @test abs(model.c - c_true) < 0.3
        @test model.converged == true
        @test model.method == :ols
        @test length(model.residuals) == n - 1
        @test !isnan(model.loglik)
        @test !isnan(model.aic)
        @test !isnan(model.bic)
    end

    @testset "AR(2) OLS estimation" begin
        n = 500
        phi_true = [0.5, 0.3]

        y = zeros(n)
        y[1:2] = randn(2)
        for t in 3:n
            y[t] = phi_true[1] * y[t-1] + phi_true[2] * y[t-2] + randn()
        end

        model = estimate_ar(y, 2; method=:ols, include_intercept=false)

        @test model.p == 2
        @test length(model.phi) == 2
        @test abs(model.phi[1] - phi_true[1]) < 0.15
        @test abs(model.phi[2] - phi_true[2]) < 0.15
    end

    @testset "AR(1) MLE estimation" begin
        n = 300
        phi_true = 0.8

        y = zeros(n)
        y[1] = randn()
        for t in 2:n
            y[t] = phi_true * y[t-1] + randn()
        end

        model = estimate_ar(y, 1; method=:mle, include_intercept=false)

        @test model.method == :mle
        @test abs(model.phi[1] - phi_true) < 0.1
    end

    @testset "AR StatsAPI interface" begin
        y = randn(200)
        model = estimate_ar(y, 2)

        @test nobs(model) == 200
        @test length(coef(model)) == 3  # c, phi1, phi2
        @test length(residuals(model)) == 198
        @test length(predict(model)) == 198
        @test !isnan(loglikelihood(model))
        @test !isnan(aic(model))
        @test !isnan(bic(model))
        @test dof(model) == 4  # c, phi1, phi2, sigma2
        @test 0 <= r2(model) <= 1
        @test islinear(model)
    end
end

@testset "MA Model Estimation" begin
    @testset "MA(1) CSS+MLE estimation" begin
        # Generate MA(1) data: yₜ = εₜ + θεₜ₋₁
        n = 500
        theta_true = 0.6
        sigma_true = 1.0

        eps = sigma_true * randn(n + 1)
        y = [eps[t] + theta_true * eps[t-1] for t in 2:n+1]

        model = estimate_ma(y, 1; method=:css_mle)

        @test model.q == 1
        @test length(model.theta) == 1
        @test abs(model.theta[1] - theta_true) < 0.15
        @test model.method == :css_mle
        @test !isnan(model.loglik)
    end

    @testset "MA(2) estimation" begin
        n = 500
        theta_true = [0.4, 0.3]

        eps = randn(n + 2)
        y = [eps[t] + theta_true[1] * eps[t-1] + theta_true[2] * eps[t-2] for t in 3:n+2]

        model = estimate_ma(y, 2)

        @test model.q == 2
        @test length(model.theta) == 2
        # MA estimation is harder, use relaxed tolerance
        @test abs(model.theta[1] - theta_true[1]) < 0.2
        @test abs(model.theta[2] - theta_true[2]) < 0.2
    end

    @testset "MA(1) invertibility check" begin
        y = randn(200)
        model = estimate_ma(y, 1)

        # Estimated theta should be invertible (|θ| < 1)
        @test abs(model.theta[1]) < 1.0
    end
end

@testset "ARMA Model Estimation" begin
    @testset "ARMA(1,1) CSS+MLE estimation" begin
        # Generate ARMA(1,1) data
        n = 500
        phi_true = 0.6
        theta_true = 0.4

        eps = randn(n + 1)
        y = zeros(n)
        y[1] = eps[1]
        for t in 2:n
            y[t] = phi_true * y[t-1] + eps[t] + theta_true * eps[t-1]
        end

        model = estimate_arma(y, 1, 1; method=:css_mle)

        @test model.p == 1
        @test model.q == 1
        @test length(model.phi) == 1
        @test length(model.theta) == 1
        @test abs(model.phi[1] - phi_true) < 0.2
        @test abs(model.theta[1] - theta_true) < 0.2
    end

    @testset "ARMA(0,0) is white noise" begin
        y = randn(200)
        model = estimate_arma(y, 0, 0)

        @test model.p == 0
        @test model.q == 0
        @test isempty(model.phi)
        @test isempty(model.theta)
        @test abs(model.c - mean(y)) < 0.1
    end

    @testset "ARMA(p,0) matches AR(p)" begin
        y = randn(200)

        arma = estimate_arma(y, 2, 0; method=:css_mle)
        ar = estimate_ar(y, 2; method=:ols)

        # Coefficients should be similar
        @test abs(arma.phi[1] - ar.phi[1]) < 0.1
        @test abs(arma.phi[2] - ar.phi[2]) < 0.1
    end

    @testset "ARMA stationarity and invertibility" begin
        y = randn(200)
        model = estimate_arma(y, 2, 2)

        # Check stationarity via companion matrix
        if model.p > 0
            p = model.p
            if p == 1
                @test abs(model.phi[1]) < 1.0
            else
                F = zeros(p, p)
                F[1, :] = model.phi
                F[2:p, 1:p-1] = I(p-1)
                @test maximum(abs.(eigvals(F))) < 1.0
            end
        end

        # Check invertibility
        if model.q > 0
            q = model.q
            if q == 1
                @test abs(model.theta[1]) < 1.0
            else
                F = zeros(q, q)
                F[1, :] = model.theta
                F[2:q, 1:q-1] = I(q-1)
                @test maximum(abs.(eigvals(F))) < 1.0
            end
        end
    end
end

@testset "ARIMA Model Estimation" begin
    @testset "ARIMA(1,1,0) - random walk with drift" begin
        n = 300
        phi_true = 0.5
        drift = 0.1

        # Generate I(1) process
        y_diff = zeros(n)
        y_diff[1] = randn()
        for t in 2:n
            y_diff[t] = drift + phi_true * y_diff[t-1] + randn()
        end
        y = cumsum(y_diff)

        model = estimate_arima(y, 1, 1, 0)

        @test model.p == 1
        @test model.d == 1
        @test model.q == 0
        @test length(model.y_diff) == n - 1
        @test abs(model.phi[1] - phi_true) < 0.2
    end

    @testset "ARIMA(0,1,1) - IMA model" begin
        n = 300
        theta_true = 0.5

        # Generate I(1) MA(1) process
        eps = randn(n + 1)
        y_diff = [eps[t] + theta_true * eps[t-1] for t in 2:n+1]
        y = cumsum(y_diff)

        model = estimate_arima(y, 0, 1, 1)

        @test model.p == 0
        @test model.d == 1
        @test model.q == 1
        @test isempty(model.phi)
        @test length(model.theta) == 1
    end

    @testset "ARIMA(p,0,q) matches ARMA(p,q)" begin
        y = randn(200)

        arima = estimate_arima(y, 1, 0, 1)
        arma = estimate_arma(y, 1, 1)

        @test arima.p == arma.p
        @test arima.q == arma.q
        @test abs(arima.phi[1] - arma.phi[1]) < 0.01
        @test abs(arima.theta[1] - arma.theta[1]) < 0.01
    end

    @testset "ARIMA with d=2" begin
        # I(2) process
        y = cumsum(cumsum(randn(200)))

        model = estimate_arima(y, 1, 2, 0)

        @test model.d == 2
        @test length(model.y_diff) == 198  # Two differences
    end
end

@testset "Forecasting" begin
    @testset "AR forecast" begin
        # Stationary AR(1)
        phi = 0.7
        n = 200
        y = zeros(n)
        y[1] = randn()
        for t in 2:n
            y[t] = phi * y[t-1] + randn()
        end

        model = estimate_ar(y, 1)
        fc = forecast(model, 10)

        @test fc.horizon == 10
        @test length(fc.forecast) == 10
        @test length(fc.se) == 10
        @test all(fc.ci_lower .< fc.forecast .< fc.ci_upper)

        # Forecast SE should increase with horizon
        @test issorted(fc.se)

        # h-step forecast should converge to unconditional mean
        fc_long = forecast(model, 100)
        unconditional_mean = model.c / (1 - model.phi[1])
        @test abs(fc_long.forecast[end] - unconditional_mean) < 0.5
    end

    @testset "MA forecast" begin
        y = randn(200)
        model = estimate_ma(y, 2)
        fc = forecast(model, 10)

        @test fc.horizon == 10
        @test length(fc.forecast) == 10

        # MA forecast beyond q should be constant (at mean)
        @test abs(fc.forecast[end] - fc.forecast[5]) < 0.01
    end

    @testset "ARMA forecast" begin
        y = randn(200)
        model = estimate_arma(y, 1, 1)
        fc = forecast(model, 12)

        @test fc.horizon == 12
        @test length(fc.forecast) == 12
        @test all(fc.se .> 0)
    end

    @testset "ARIMA forecast integration" begin
        # Random walk
        y = cumsum(randn(200))
        model = estimate_arima(y, 0, 1, 0; include_intercept=false)
        fc = forecast(model, 10)

        @test fc.horizon == 10
        # Forecast should start near last observation
        @test abs(fc.forecast[1] - y[end]) < 2 * fc.se[1]
    end

    @testset "Confidence interval coverage" begin
        # Generate known process
        phi = 0.5
        n = 200
        y = zeros(n)
        y[1] = randn()
        for t in 2:n
            y[t] = phi * y[t-1] + randn()
        end

        model = estimate_ar(y, 1)
        fc = forecast(model, 5; conf_level=0.95)

        @test fc.conf_level == 0.95
        @test all(fc.ci_lower .< fc.ci_upper)
    end

    @testset "StatsAPI predict with horizon" begin
        y = randn(200)
        model = estimate_ar(y, 2)

        # predict(model) returns fitted values
        @test length(predict(model)) == length(model.fitted)

        # predict(model, h) returns forecasts
        fc = predict(model, 5)
        @test length(fc) == 5
    end
end

@testset "Order Selection" begin
    @testset "Select correct AR order" begin
        # Generate AR(2) data
        n = 200
        phi_true = [0.5, 0.3]

        y = zeros(n)
        y[1:2] = randn(2)
        for t in 3:n
            y[t] = phi_true[1] * y[t-1] + phi_true[2] * y[t-2] + randn()
        end

        result = select_arima_order(y, 2, 0; criterion=:bic)

        # BIC should select AR(2) or nearby
        @test result.best_p_bic in [1, 2]
        @test result.best_q_bic == 0
    end

    @testset "Select correct MA order" begin
        # Generate MA(1) data
        n = 200
        theta_true = 0.6

        eps = randn(n + 1)
        y = [eps[t] + theta_true * eps[t-1] for t in 2:n+1]

        result = select_arima_order(y, 0, 1; criterion=:bic)

        # BIC should select MA(1)
        @test result.best_p_bic == 0
        @test result.best_q_bic in [0, 1]
    end

    @testset "IC matrix dimensions" begin
        y = randn(200)
        result = select_arima_order(y, 1, 1)

        @test size(result.aic_matrix) == (2, 2)  # (max_p+1, max_q+1)
        @test size(result.bic_matrix) == (2, 2)
    end

    @testset "Best model is fitted" begin
        y = randn(200)
        result = select_arima_order(y, 1, 1)

        @test isa(result.best_model_aic, AbstractARIMAModel)
        @test isa(result.best_model_bic, AbstractARIMAModel)
        @test ar_order(result.best_model_bic) == result.best_p_bic
        @test ma_order(result.best_model_bic) == result.best_q_bic
    end

    @testset "auto_arima" begin
        # Random walk
        y = cumsum(randn(200))
        model = auto_arima(y; max_p=1, max_q=1, max_d=1)

        @test isa(model, AbstractARIMAModel)
    end
end

@testset "Edge Cases" begin
    @testset "AR(1) near unit root" begin
        phi = 0.99
        n = 200
        y = zeros(n)
        y[1] = randn()
        for t in 2:n
            y[t] = phi * y[t-1] + randn()
        end

        model = estimate_ar(y, 1)
        @test model.converged || model.method == :ols
        @test abs(model.phi[1]) < 1.0  # Should still be stationary
    end

    @testset "Short time series" begin
        y = randn(30)

        # Should still work with small n
        ar = estimate_ar(y, 1)
        @test length(ar.residuals) == 29

        ma = estimate_ma(y, 1)
        @test length(ma.residuals) == 30
    end

    @testset "p=0, q=0" begin
        y = randn(100)
        model = estimate_arma(y, 0, 0)

        @test isempty(model.phi)
        @test isempty(model.theta)
        @test abs(model.c - mean(y)) < 0.2
    end

    @testset "High order ARMA" begin
        y = randn(500)
        model = estimate_arma(y, 4, 3)

        @test model.p == 4
        @test model.q == 3
        @test length(model.phi) == 4
        @test length(model.theta) == 3
    end

    @testset "Input validation" begin
        y = randn(100)

        @test_throws ArgumentError estimate_ar(y, -1)
        @test_throws ArgumentError estimate_ma(y, -1)
        @test_throws ArgumentError estimate_arima(y, 1, -1, 0)

        # Too short
        y_short = randn(5)
        @test_throws ArgumentError estimate_ar(y_short, 1)
    end
end

@testset "Type Accessors" begin
    y = randn(200)

    ar = estimate_ar(y, 2)
    @test ar_order(ar) == 2
    @test ma_order(ar) == 0
    @test diff_order(ar) == 0

    ma = estimate_ma(y, 3)
    @test ar_order(ma) == 0
    @test ma_order(ma) == 3
    @test diff_order(ma) == 0

    arma = estimate_arma(y, 2, 1)
    @test ar_order(arma) == 2
    @test ma_order(arma) == 1
    @test diff_order(arma) == 0

    y_rw = cumsum(randn(200))
    arima = estimate_arima(y_rw, 1, 1, 1)
    @test ar_order(arima) == 1
    @test ma_order(arima) == 1
    @test diff_order(arima) == 1
end

@testset "Display Methods" begin
    y = randn(200)

    ar = estimate_ar(y, 2)
    @test contains(repr(ar), "AR(2)")

    ma = estimate_ma(y, 1)
    @test contains(repr(ma), "MA(1)")

    arma = estimate_arma(y, 1, 1)
    @test contains(repr(arma), "ARMA(1,1)")

    y_rw = cumsum(randn(200))
    arima = estimate_arima(y_rw, 1, 1, 0)
    @test contains(repr(arima), "ARIMA(1,1,0)")

    fc = forecast(ar, 5)
    @test contains(repr(fc), "Forecast")

    result = select_arima_order(y, 2, 2)
    @test contains(repr(result), "Order Selection")
end

@testset "Numerical Stability" begin
    @testset "Large variance data" begin
        y = 1000 .* randn(200)
        model = estimate_ar(y, 1)
        @test !isnan(model.loglik)
        @test !isinf(model.loglik)
    end

    @testset "Small variance data" begin
        y = 0.001 .* randn(200)
        model = estimate_ar(y, 1)
        @test !isnan(model.loglik)
    end

    @testset "Data with trend" begin
        y = collect(1.0:200.0) .+ randn(200)
        # First difference should remove trend
        model = estimate_arima(y, 1, 1, 0)
        @test !isnan(model.loglik)
    end
end

@testset "fit Interface" begin
    y = randn(200)

    ar = fit(ARModel, y, 2)
    @test isa(ar, ARModel)

    ma = fit(MAModel, y, 1)
    @test isa(ma, MAModel)

    arma = fit(ARMAModel, y, 1, 1)
    @test isa(arma, ARMAModel)

    y_rw = cumsum(randn(200))
    arima = fit(ARIMAModel, y_rw, 1, 1, 0)
    @test isa(arima, ARIMAModel)
end

# =============================================================================
# StatsAPI Interface Tests for MA, ARMA, ARIMA
# =============================================================================

@testset "StatsAPI Interface - MA Model" begin
    Random.seed!(7000)
    n = 300
    # Generate MA(1) data
    eps = randn(n + 1)
    theta_true = 0.6
    y = [eps[t] + theta_true * eps[t-1] for t in 2:(n+1)]

    model = estimate_ma(y, 1)

    # nobs
    @test nobs(model) == n
    @test nobs(model) == length(model.y)

    # dof: p + q + 2 (intercept + sigma2) = 0 + 1 + 2 = 3
    @test dof(model) == 3

    # StatsAPI.dof_residual
    @test StatsAPI.dof_residual(model) == length(model.residuals) - dof(model) + 1

    # coef: [intercept, theta]
    c = coef(model)
    @test length(c) == 2
    @test c[1] ≈ model.c
    @test c[2] ≈ model.theta[1]

    # residuals
    r = residuals(model)
    @test length(r) == length(model.residuals)
    @test r == model.residuals

    # predict (fitted values)
    f = predict(model)
    @test length(f) == length(model.fitted)
    @test f == model.fitted

    # fitted + residuals ≈ y (for the effective sample)
    y_eff = model.y[end-length(r)+1:end]
    @test isapprox(f + r, y_eff, atol=1e-8)

    # loglikelihood
    ll = loglikelihood(model)
    @test !isnan(ll)
    @test isfinite(ll)
    @test ll == model.loglik

    # aic, bic
    @test !isnan(aic(model))
    @test !isnan(bic(model))
    @test aic(model) == model.aic
    @test bic(model) == model.bic

    # r2
    r2_val = r2(model)
    @test 0 <= r2_val <= 1 || r2_val ≈ 0  # Can be 0 for weak fit

    # islinear
    @test islinear(model) == true

    # ar_order, ma_order, diff_order
    @test MacroEconometricModels.ar_order(model) == 0
    @test MacroEconometricModels.ma_order(model) == 1
    @test MacroEconometricModels.diff_order(model) == 0
end

@testset "StatsAPI Interface - MA(2) Model" begin
    Random.seed!(7100)
    n = 300
    eps = randn(n + 2)
    y = [eps[t] + 0.5 * eps[t-1] - 0.3 * eps[t-2] for t in 3:(n+2)]

    model = estimate_ma(y, 2)

    @test nobs(model) == n
    @test dof(model) == 4  # 0 + 2 + 2
    c = coef(model)
    @test length(c) == 3  # intercept + 2 theta
    @test islinear(model)
end

@testset "StatsAPI Interface - ARMA Model" begin
    Random.seed!(7200)
    n = 400
    # Generate ARMA(1,1) data
    phi_true = 0.5
    theta_true = 0.3
    eps = randn(n)
    y = zeros(n)
    y[1] = eps[1]
    for t in 2:n
        y[t] = phi_true * y[t-1] + eps[t] + theta_true * eps[t-1]
    end

    model = estimate_arma(y, 1, 1)

    # nobs
    @test nobs(model) == n

    # dof: p + q + 2 = 1 + 1 + 2 = 4
    @test dof(model) == 4

    # StatsAPI.dof_residual
    @test StatsAPI.dof_residual(model) == length(model.residuals) - dof(model) + 1

    # coef: [intercept, phi, theta]
    c = coef(model)
    @test length(c) == 3

    # residuals and fitted
    r = residuals(model)
    f = predict(model)
    @test length(r) > 0
    @test length(f) > 0
    @test length(r) == length(f)
    y_eff = model.y[end-length(r)+1:end]
    @test isapprox(f + r, y_eff, atol=1e-8)

    # loglikelihood, aic, bic
    @test isfinite(loglikelihood(model))
    @test isfinite(aic(model))
    @test isfinite(bic(model))

    # r2
    r2_val = r2(model)
    @test r2_val >= 0

    @test islinear(model)

    # ar_order, ma_order, diff_order
    @test MacroEconometricModels.ar_order(model) == 1
    @test MacroEconometricModels.ma_order(model) == 1
    @test MacroEconometricModels.diff_order(model) == 0
end

@testset "StatsAPI Interface - ARMA(2,1) Model" begin
    Random.seed!(7300)
    n = 400
    eps = randn(n)
    y = zeros(n)
    y[1:2] = randn(2)
    for t in 3:n
        y[t] = 0.4 * y[t-1] + 0.2 * y[t-2] + eps[t] + 0.3 * eps[t-1]
    end

    model = estimate_arma(y, 2, 1)

    @test nobs(model) == n
    @test dof(model) == 5  # 2 + 1 + 2
    c = coef(model)
    @test length(c) == 4  # intercept + 2 phi + 1 theta
end

@testset "StatsAPI Interface - ARIMA Model" begin
    Random.seed!(7400)
    n = 300
    # Generate ARIMA(1,1,1) data
    eps = randn(n)
    z = zeros(n)
    z[1] = eps[1]
    for t in 2:n
        z[t] = 0.5 * z[t-1] + eps[t] + 0.3 * eps[t-1]
    end
    y = cumsum(z)  # Integration order 1

    model = estimate_arima(y, 1, 1, 1)

    # nobs (on original series)
    @test nobs(model) == n

    # dof: p + q + 2 = 1 + 1 + 2 = 4
    @test dof(model) == 4

    # StatsAPI.dof_residual
    @test StatsAPI.dof_residual(model) == length(model.residuals) - dof(model) + 1

    # coef: [intercept, phi, theta]
    c = coef(model)
    @test length(c) == 3

    # residuals and fitted
    r = residuals(model)
    f = predict(model)
    @test length(r) > 0
    @test length(f) > 0

    # loglikelihood, aic, bic
    @test isfinite(loglikelihood(model))
    @test isfinite(aic(model))
    @test isfinite(bic(model))

    # r2
    r2_val = r2(model)
    @test r2_val >= 0

    @test islinear(model)

    # ar_order, ma_order, diff_order
    @test MacroEconometricModels.ar_order(model) == 1
    @test MacroEconometricModels.ma_order(model) == 1
    @test MacroEconometricModels.diff_order(model) == 1

    # ARIMA-specific fields
    @test length(model.y_diff) == n - 1
    @test model.d == 1
    @test model.p == 1
    @test model.q == 1
end

@testset "StatsAPI Interface - ARIMA(2,1,0) Model" begin
    Random.seed!(7500)
    n = 300
    z = zeros(n)
    z[1:2] = randn(2)
    for t in 3:n
        z[t] = 0.4 * z[t-1] + 0.2 * z[t-2] + randn()
    end
    y = cumsum(z)

    model = estimate_arima(y, 2, 1, 0)

    @test nobs(model) == n
    @test dof(model) == 4  # 2 + 0 + 2
    c = coef(model)
    @test length(c) == 3  # intercept + 2 phi + 0 theta
    @test MacroEconometricModels.diff_order(model) == 1
end

@testset "StatsAPI Interface - ARIMA(0,1,2) Model" begin
    Random.seed!(7600)
    n = 300
    eps = randn(n + 2)
    z = [eps[t] + 0.4 * eps[t-1] - 0.2 * eps[t-2] for t in 3:(n+2)]
    y = cumsum(z)

    model = estimate_arima(y, 0, 1, 2)

    @test nobs(model) == n
    @test dof(model) == 4  # 0 + 2 + 2
    c = coef(model)
    @test length(c) == 3  # intercept + 0 phi + 2 theta
end

@testset "StatsAPI consistency checks" begin
    Random.seed!(7700)

    # For each model type, check StatsAPI.dof_residual == nobs_residuals - dof + 1
    y = randn(300)

    for (ModelType, args) in [
        (MAModel, (y, 2)),
        (ARMAModel, (y, 1, 1)),
    ]
        model = fit(ModelType, args...)
        @test StatsAPI.dof_residual(model) == length(residuals(model)) - dof(model) + 1
    end

    y_rw = cumsum(randn(300))
    model_arima = fit(ARIMAModel, y_rw, 1, 1, 1)
    @test StatsAPI.dof_residual(model_arima) == length(residuals(model_arima)) - dof(model_arima) + 1
end

@testset "Display methods - MA/ARMA/ARIMA" begin
    Random.seed!(7800)
    y = randn(200)
    y_rw = cumsum(randn(200))

    # MA show
    ma = estimate_ma(y, 1)
    io = IOBuffer()
    show(io, ma)
    output = String(take!(io))
    @test occursin("MA(1)", output)
    @test occursin("θ[1]", output)

    # ARMA show
    arma = estimate_arma(y, 1, 1)
    io = IOBuffer()
    show(io, arma)
    output = String(take!(io))
    @test occursin("ARMA(1,1)", output)
    @test occursin("φ[1]", output)
    @test occursin("θ[1]", output)

    # ARIMA show
    arima = estimate_arima(y_rw, 1, 1, 1)
    io = IOBuffer()
    show(io, arima)
    output = String(take!(io))
    @test occursin("ARIMA(1,1,1)", output)
end

@testset "ARIMA d=2 (double differencing)" begin
    Random.seed!(6001)
    # Generate I(2) series
    y_i2 = cumsum(cumsum(randn(200)))
    m = estimate_arima(y_i2, 1, 2, 1; method=:css_mle)
    @test m isa MacroEconometricModels.ARIMAModel{Float64}
    @test m.d == 2
    fc = forecast(m, 5)
    @test length(fc.forecast) == 5
end

@testset "auto_arima include_intercept=false" begin
    Random.seed!(6002)
    y = randn(100)
    m = auto_arima(y; include_intercept=false, max_p=1, max_q=1)
    @test m isa MacroEconometricModels.AbstractARIMAModel
    fc = forecast(m, 3)
    @test length(fc.forecast) == 3
end

@testset "select_arima_order larger grid" begin
    Random.seed!(6003)
    y = randn(100)
    result = select_arima_order(y, 2, 1)
    @test result isa MacroEconometricModels.ARIMAOrderSelection
    @test result.best_p_aic >= 0
    @test result.best_q_aic >= 0
    @test result.best_p_bic >= 0
    @test result.best_q_bic >= 0
end

@testset "Forecast with conf_level=0.99" begin
    Random.seed!(6004)
    y = randn(100)
    m = estimate_ar(y, 2)
    fc = forecast(m, 5; conf_level=0.99)
    @test length(fc.forecast) == 5
    @test all(fc.ci_upper .> fc.ci_lower)
    # 99% CI should be wider than default 95%
    fc95 = forecast(m, 5; conf_level=0.95)
    @test all(fc.ci_upper .- fc.ci_lower .>= fc95.ci_upper .- fc95.ci_lower .- 1e-10)
end

@testset "conf_level accepts Real" begin
    Random.seed!(42)
    y = cumsum(randn(100))
    m = estimate_ar(y, 2)
    # Should accept Float64 and Float32 without error
    fc1 = forecast(m, 5; conf_level=0.9)
    @test fc1.conf_level ≈ 0.9
    fc2 = forecast(m, 5; conf_level=Float32(0.9))
    @test fc2.conf_level ≈ Float32(0.9) atol=1e-6
end
