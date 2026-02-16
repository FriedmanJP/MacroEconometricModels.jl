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

"""
Coverage improvement tests for volatility models (ARCH/GARCH/SV).

Targets uncovered paths:
- SV leverage variant (_ksc_ffbs_leverage, _ksc_draw_rho)
- SV Student-t deeper coverage (_ksc_draw_lambda, _ksc_draw_nu)
- GARCH/ARCH edge cases: persistence >= 1, halflife/unconditional_variance = Inf
- StatsAPI gaps for EGARCH/GJR-GARCH (coef, residuals, predict, loglikelihood, aic, bic)
- VolatilityForecast display with horizon > 10
"""

using Test
using MacroEconometricModels
using Random
using Statistics
using StatsAPI

# =============================================================================
# SV Leverage Variant
# =============================================================================

@testset "SV leverage variant" begin
    Random.seed!(5001)
    # Simulate data with leverage-like properties
    n = 100
    y = randn(n) .* exp.(cumsum(0.15 .* randn(n)) ./ 2)

    m = estimate_sv(y; n_samples=40, burnin=20, leverage=true)

    @test m isa SVModel{Float64}
    @test m.leverage == true
    @test m.dist == :normal
    @test nobs(m) == n
    @test length(m.mu_post) == 40
    @test length(m.phi_post) == 40
    @test length(m.sigma_eta_post) == 40
    @test size(m.h_draws) == (40, n)
    @test all(m.volatility_mean .> 0)
    @test all(isfinite.(m.mu_post))
    @test all(isfinite.(m.phi_post))

    # Display should include "leverage"
    io = IOBuffer()
    show(io, m)
    output = String(take!(io))
    @test occursin("leverage", lowercase(output))
end

# =============================================================================
# SV Student-t Deeper Coverage
# =============================================================================

@testset "SV Student-t deeper coverage" begin
    Random.seed!(5002)
    n = 100
    y = randn(n) .* exp.(cumsum(0.1 .* randn(n)) ./ 2)

    m = estimate_sv(y; n_samples=40, burnin=20, dist=:studentt)

    @test m isa SVModel{Float64}
    @test m.dist == :studentt
    @test m.leverage == false
    @test all(m.volatility_mean .> 0)
    @test all(m.sigma_eta_post .> 0)

    # Display should include "Student-t"
    io = IOBuffer()
    show(io, m)
    output = String(take!(io))
    @test occursin("Student-t", output)

    # Forecast should work
    Random.seed!(5003)
    fc = forecast(m, 5)
    @test fc isa VolatilityForecast{Float64}
    @test fc.model_type == :sv
    @test all(fc.forecast .> 0)
end

# =============================================================================
# SV Leverage + Student-t Combined
# =============================================================================

@testset "SV leverage + Student-t combined" begin
    Random.seed!(5004)
    n = 100
    y = randn(n) .* exp.(cumsum(0.12 .* randn(n)) ./ 2)

    m = estimate_sv(y; n_samples=30, burnin=15, dist=:studentt, leverage=true)

    @test m.leverage == true
    @test m.dist == :studentt
    @test all(m.volatility_mean .> 0)
end

# =============================================================================
# GARCH/ARCH Edge Cases: persistence >= 1
# =============================================================================

@testset "Volatility edge cases: high persistence" begin
    @testset "GARCH halflife and unconditional_variance at boundary" begin
        Random.seed!(5010)
        y = randn(300)

        # Estimate a model, then manually create one with extreme parameters
        m = estimate_garch(y, 1, 1)

        # Construct a model with persistence >= 1 using the GARCHModel constructor
        m_extreme = GARCHModel(
            m.y, m.p, m.q, m.mu,
            m.omega,
            [0.6],    # alpha high
            [0.5],    # beta high => persistence = 1.1
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_extreme) >= 1.0
        @test halflife(m_extreme) == Inf
        @test unconditional_variance(m_extreme) == Inf
    end

    @testset "EGARCH persistence edge case" begin
        Random.seed!(5011)
        y = randn(300)
        m = estimate_egarch(y, 1, 1)

        # Construct with beta sum >= 1
        m_extreme = EGARCHModel(
            m.y, m.p, m.q, m.mu, m.omega,
            m.alpha,
            m.gamma,
            [1.1],    # beta >= 1
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_extreme) >= 1.0
        @test halflife(m_extreme) == Inf
        @test unconditional_variance(m_extreme) == Inf
    end

    @testset "GJR-GARCH persistence edge case" begin
        Random.seed!(5012)
        y = randn(300)
        m = estimate_gjr_garch(y, 1, 1)

        # Construct with persistence >= 1
        m_extreme = GJRGARCHModel(
            m.y, m.p, m.q, m.mu, m.omega,
            [0.4],    # alpha
            [0.3],    # gamma => alpha + gamma/2 + beta = 0.4 + 0.15 + 0.5 = 1.05
            [0.5],    # beta
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_extreme) >= 1.0
        @test halflife(m_extreme) == Inf
        @test unconditional_variance(m_extreme) == Inf
    end

    @testset "ARCH halflife edge case: persistence <= 0" begin
        Random.seed!(5013)
        y = randn(300)
        m = estimate_arch(y, 1)

        # Construct with zero alpha
        m_zero = ARCHModel(
            m.y, m.q, m.mu, m.omega,
            [0.0],    # alpha = 0 => persistence = 0
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_zero) <= 0.0
        @test halflife(m_zero) == Inf

        # Construct with persistence >= 1
        m_high = ARCHModel(
            m.y, m.q, m.mu, m.omega,
            [1.1],    # alpha >= 1
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_high) >= 1.0
        @test halflife(m_high) == Inf
        @test unconditional_variance(m_high) == Inf
    end

    @testset "GARCH halflife edge case: persistence <= 0" begin
        Random.seed!(5014)
        y = randn(300)
        m = estimate_garch(y, 1, 1)

        m_zero = GARCHModel(
            m.y, m.p, m.q, m.mu, m.omega,
            [0.0],    # alpha = 0
            [0.0],    # beta = 0  => persistence = 0
            m.conditional_variance, m.standardized_residuals,
            m.residuals, m.fitted, m.loglik, m.aic, m.bic,
            m.method, m.converged, m.iterations
        )

        @test persistence(m_zero) <= 0.0
        @test halflife(m_zero) == Inf
    end
end

# =============================================================================
# EGARCH / GJR-GARCH StatsAPI Completeness
# =============================================================================

@testset "EGARCH/GJR StatsAPI completeness" begin
    Random.seed!(5020)
    y = randn(300)

    @testset "EGARCH StatsAPI" begin
        m = estimate_egarch(y, 1, 1)
        @test StatsAPI.nobs(m) == 300
        @test length(StatsAPI.coef(m)) == 5  # mu + omega + alpha + gamma + beta
        @test length(StatsAPI.residuals(m)) == 300
        @test length(StatsAPI.predict(m)) == 300
        @test isfinite(StatsAPI.loglikelihood(m))
        @test isfinite(StatsAPI.aic(m))
        @test isfinite(StatsAPI.bic(m))
        @test StatsAPI.dof(m) == 2 + 2 * 1 + 1
        @test StatsAPI.islinear(m) == false
        @test arch_order(m) == 1
        @test garch_order(m) == 1
    end

    @testset "GJR-GARCH StatsAPI" begin
        m = estimate_gjr_garch(y, 1, 1)
        @test StatsAPI.nobs(m) == 300
        @test length(StatsAPI.coef(m)) == 5
        @test length(StatsAPI.residuals(m)) == 300
        @test length(StatsAPI.predict(m)) == 300
        @test isfinite(StatsAPI.loglikelihood(m))
        @test isfinite(StatsAPI.aic(m))
        @test isfinite(StatsAPI.bic(m))
        @test StatsAPI.dof(m) == 2 + 2 * 1 + 1
        @test StatsAPI.islinear(m) == false
        @test arch_order(m) == 1
        @test garch_order(m) == 1
    end
end

# =============================================================================
# VolatilityForecast Display: horizon > 10 (triggers "..." branch)
# =============================================================================

@testset "VolatilityForecast display h > 10" begin
    Random.seed!(5030)
    y = randn(300)
    m = estimate_garch(y, 1, 1)
    fc = forecast(m, 15)

    io = IOBuffer()
    show(io, fc)
    output = String(take!(io))
    @test occursin("...", output)
    @test occursin("5 more", output)
end
