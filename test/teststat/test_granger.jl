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

using Random

Random.seed!(54321)

# Shared test data
Y_gc = randn(200, 3)
m2 = estimate_var(Y_gc, 2)

# =============================================================================
# Pairwise Granger Test
# =============================================================================

@testset "Pairwise Granger Test" begin
    g = granger_test(m2, 1, 2)
    @test g isa GrangerCausalityResult
    @test g.statistic >= 0
    @test 0 <= g.pvalue <= 1
    @test g.df == 2  # p = 2 lags
    @test g.cause == [1]
    @test g.effect == 2
    @test g.n == 3
    @test g.p == 2
    @test g.nobs == 198  # 200 - 2
    @test g.test_type == :pairwise

    # StatsAPI interface
    @test nobs(g) == 198
    @test dof(g) == 2

    # Different variable pair
    g2 = granger_test(m2, 3, 1)
    @test g2 isa GrangerCausalityResult
    @test g2.cause == [3]
    @test g2.effect == 1
    @test g2.statistic >= 0

    # Test all pairs produce valid results
    for i in 1:3, j in 1:3
        i == j && continue
        r = granger_test(m2, i, j)
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.df == 2
    end
end

# =============================================================================
# Block (Multivariate) Granger Test
# =============================================================================

@testset "Block Granger Test" begin
    g = granger_test(m2, [1, 2], 3)
    @test g isa GrangerCausalityResult
    @test g.statistic >= 0
    @test 0 <= g.pvalue <= 1
    @test g.df == 2 * 2  # p * |cause| = 2 * 2 = 4
    @test g.cause == [1, 2]
    @test g.effect == 3
    @test g.test_type == :block

    # Single-element vector is pairwise
    g_single = granger_test(m2, [1], 2)
    @test g_single.test_type == :pairwise
    @test g_single.df == 2
    @test g_single.cause == [1]

    # Cause ordering is sorted
    g_unordered = granger_test(m2, [3, 1], 2)
    @test g_unordered.cause == [1, 3]
    @test g_unordered.df == 4
end

# =============================================================================
# granger_test_all
# =============================================================================

@testset "granger_test_all" begin
    results = granger_test_all(m2)
    @test size(results) == (3, 3)

    # Diagonal is nothing
    for i in 1:3
        @test results[i, i] === nothing
    end

    # Off-diagonal entries are valid
    for i in 1:3, j in 1:3
        i == j && continue
        r = results[i, j]
        @test r isa GrangerCausalityResult
        @test r.cause == [j]
        @test r.effect == i
        @test r.statistic >= 0
        @test 0 <= r.pvalue <= 1
        @test r.df == 2
        @test r.test_type == :pairwise
    end

    # Results should match individual pairwise tests
    for i in 1:3, j in 1:3
        i == j && continue
        individual = granger_test(m2, j, i)
        @test results[i, j].statistic ≈ individual.statistic atol=1e-10
        @test results[i, j].pvalue ≈ individual.pvalue atol=1e-10
    end
end

# =============================================================================
# Error Handling
# =============================================================================

@testset "Granger Test Error Handling" begin
    # cause == effect
    @test_throws ArgumentError granger_test(m2, 1, 1)
    @test_throws ArgumentError granger_test(m2, [1, 2], 1)

    # Out of range
    @test_throws ArgumentError granger_test(m2, 0, 1)
    @test_throws ArgumentError granger_test(m2, 4, 1)
    @test_throws ArgumentError granger_test(m2, 1, 0)
    @test_throws ArgumentError granger_test(m2, 1, 4)
    @test_throws ArgumentError granger_test(m2, [0, 1], 2)
    @test_throws ArgumentError granger_test(m2, [4], 2)

    # Empty cause
    @test_throws ArgumentError granger_test(m2, Int[], 1)

    # Duplicate cause indices
    @test_throws ArgumentError granger_test(m2, [1, 1], 2)

    # Effect in cause set
    @test_throws ArgumentError granger_test(m2, [1, 2], 2)
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "Granger Test Edge Cases" begin
    # VAR(1) — single lag
    rng1 = Random.MersenneTwister(999)
    Y1 = randn(rng1, 100, 2)
    m1 = estimate_var(Y1, 1)
    g1 = granger_test(m1, 1, 2)
    @test g1.df == 1  # p = 1
    @test g1.statistic >= 0
    @test 0 <= g1.pvalue <= 1

    # 2-variable system
    results_2var = granger_test_all(m1)
    @test size(results_2var) == (2, 2)
    @test results_2var[1, 1] === nothing
    @test results_2var[2, 2] === nothing
    @test results_2var[1, 2] isa GrangerCausalityResult
    @test results_2var[2, 1] isa GrangerCausalityResult

    # VAR with more lags
    rng4 = Random.MersenneTwister(888)
    Y4 = randn(rng4, 200, 3)
    m4 = estimate_var(Y4, 4)
    g4 = granger_test(m4, 1, 2)
    @test g4.df == 4
    @test g4.p == 4

    # Block test with all but effect
    g_block = granger_test(m4, [1, 2], 3)
    @test g_block.df == 8  # 4 lags * 2 cause variables
end

# =============================================================================
# Show Methods
# =============================================================================

@testset "Granger Test Display" begin
    g = granger_test(m2, 1, 2)
    io = IOBuffer()
    show(io, g)
    output = String(take!(io))
    @test contains(output, "Granger Causality Test")
    @test contains(output, "Wald statistic")
    @test contains(output, "P-value")
    @test contains(output, "Variable 1")
    @test contains(output, "Variable 2")

    # Block test display
    g_block = granger_test(m2, [1, 2], 3)
    io2 = IOBuffer()
    show(io2, g_block)
    output2 = String(take!(io2))
    @test contains(output2, "Block")
    @test contains(output2, "Variables [1, 2]")

    # All-pairs display
    results = granger_test_all(m2)
    io3 = IOBuffer()
    show(io3, MIME"text/plain"(), results)
    output3 = String(take!(io3))
    @test contains(output3, "Granger Causality P-values")
    @test contains(output3, "Var 1")
end

# =============================================================================
# refs() Integration
# =============================================================================

@testset "Granger Test References" begin
    g = granger_test(m2, 1, 2)
    io = IOBuffer()
    refs(io, g)
    output = String(take!(io))
    @test contains(output, "Granger")
    @test contains(output, "1969")

    # Symbol dispatch
    io2 = IOBuffer()
    refs(io2, :granger)
    output2 = String(take!(io2))
    @test contains(output2, "Granger")
    @test contains(output2, "1969")
end
