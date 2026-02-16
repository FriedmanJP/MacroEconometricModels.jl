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
Coverage improvement tests for ARIMA models.

Targets uncovered paths:
- ic_table() function (never called in tests)
- auto_arima with criterion=:aic
- select_arima_order with d>0 and argument validation
- _select_d_heuristic edge cases (max_d=0)
- Display methods: ARModel show, ARIMAForecast h>10, ARIMAOrderSelection show
- StatsAPI: r2 edge, dof_residual explicit
"""

using Test
using MacroEconometricModels
using Random
using Statistics
using StatsAPI

Random.seed!(6001)

# =============================================================================
# ic_table() Tests
# =============================================================================

@testset "ic_table" begin
    y = randn(200)
    result = select_arima_order(y, 2, 2; criterion=:bic)

    @testset "BIC table" begin
        tbl = MacroEconometricModels.ic_table(result; criterion=:bic)
        @test tbl isa String
        @test occursin("BIC", tbl)
        @test occursin("p\\q", tbl)
    end

    @testset "AIC table" begin
        tbl = MacroEconometricModels.ic_table(result; criterion=:aic)
        @test tbl isa String
        @test occursin("AIC", tbl)
    end
end

# =============================================================================
# auto_arima with criterion=:aic
# =============================================================================

@testset "auto_arima criterion=:aic" begin
    Random.seed!(6002)
    y = randn(200)

    model = auto_arima(y; criterion=:aic, max_p=2, max_q=2, max_d=1)
    @test model isa MacroEconometricModels.AbstractARIMAModel
    @test isfinite(model.aic)
end

# =============================================================================
# select_arima_order with d > 0
# =============================================================================

@testset "select_arima_order with differencing" begin
    Random.seed!(6003)
    y = cumsum(randn(200))  # I(1) series

    result = select_arima_order(y, 2, 2; d=1, criterion=:bic)
    @test result isa MacroEconometricModels.ARIMAOrderSelection
    @test result.best_p_bic >= 0
    @test result.best_q_bic >= 0
end

# =============================================================================
# select_arima_order argument validation
# =============================================================================

@testset "select_arima_order validation" begin
    y = randn(100)
    @test_throws ArgumentError select_arima_order(y, -1, 1)
    @test_throws ArgumentError select_arima_order(y, 1, -1)
    @test_throws ArgumentError select_arima_order(y, 1, 1; d=-1)
    @test_throws ArgumentError select_arima_order(y, 1, 1; criterion=:hqic)
end

# =============================================================================
# _select_d_heuristic edge case: max_d = 0
# =============================================================================

@testset "_select_d_heuristic edge cases" begin
    y = randn(100)

    # max_d = 0 should immediately return 0
    d = MacroEconometricModels._select_d_heuristic(y, 0)
    @test d == 0

    # Stationary data should select d = 0
    d = MacroEconometricModels._select_d_heuristic(y, 2)
    @test d == 0  # variance shouldn't decrease enough for differencing
end

# =============================================================================
# auto_arima with integer input (Float64 conversion)
# =============================================================================

@testset "auto_arima integer input" begin
    y_int = round.(Int, randn(200) .* 100)
    model = auto_arima(y_int; max_p=1, max_q=1, max_d=0)
    @test model isa MacroEconometricModels.AbstractARIMAModel
end

# =============================================================================
# select_arima_order with integer input
# =============================================================================

@testset "select_arima_order integer input" begin
    y_int = round.(Int, randn(150) .* 100)
    result = select_arima_order(y_int, 1, 1)
    @test result isa MacroEconometricModels.ARIMAOrderSelection
end

# =============================================================================
# Display Methods
# =============================================================================

@testset "ARIMA display methods" begin
    Random.seed!(6010)

    @testset "ARModel display" begin
        y = randn(200)
        m = estimate_ar(y, 2)
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("AR(2)", output)
        @test occursin("φ[1]", output)
        @test occursin("φ[2]", output)
    end

    @testset "ARIMAForecast display h > 10" begin
        y = randn(200)
        m = estimate_ar(y, 1)
        fc = forecast(m, 15)
        io = IOBuffer()
        show(io, fc)
        output = String(take!(io))
        @test occursin("...", output)
        @test occursin("5 more", output)
    end

    @testset "ARIMAForecast display h <= 10" begin
        y = randn(200)
        m = estimate_ar(y, 1)
        fc = forecast(m, 5)
        io = IOBuffer()
        show(io, fc)
        output = String(take!(io))
        @test occursin("Forecast", output)
        @test !occursin("...", output)
    end

    @testset "ARIMAOrderSelection display" begin
        y = randn(200)
        result = select_arima_order(y, 2, 2)
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test occursin("Order Selection", output)
        @test occursin("AIC", output)
        @test occursin("BIC", output)
    end
end
