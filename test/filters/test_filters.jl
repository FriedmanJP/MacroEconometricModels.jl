# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
using Random
using Statistics
using LinearAlgebra
using SparseArrays

@testset "Time Series Filters" begin

    # =============================================================================
    # HP Filter
    # =============================================================================
    @testset "HP Filter" begin
        Random.seed!(42)
        y = cumsum(randn(200))

        @testset "basic functionality" begin
            r = hp_filter(y)
            @test r isa HPFilterResult{Float64}
            @test length(r.trend) == 200
            @test length(r.cycle) == 200
            @test r.lambda == 1600.0
            @test r.T_obs == 200
            # Trend + cycle = original
            @test r.trend .+ r.cycle ≈ y
        end

        @testset "lambda = 0 => trend = y" begin
            r = hp_filter(y; lambda=0.0)
            @test r.trend ≈ y
            @test all(abs.(r.cycle) .< 1e-12)
        end

        @testset "large lambda => near-linear trend" begin
            r = hp_filter(y; lambda=1e10)
            # Trend should be approximately linear
            trend_diff2 = diff(diff(r.trend))
            @test maximum(abs.(trend_diff2)) < 0.1
        end

        @testset "different lambda values" begin
            r6 = hp_filter(y; lambda=6.25)
            r1600 = hp_filter(y; lambda=1600.0)
            r129600 = hp_filter(y; lambda=129600.0)
            # Higher lambda => smoother trend => larger cycle variance
            @test std(r6.cycle) < std(r1600.cycle)
            @test std(r1600.cycle) < std(r129600.cycle)
        end

        @testset "accessors" begin
            r = hp_filter(y)
            @test trend(r) === r.trend
            @test cycle(r) === r.cycle
        end

        @testset "Float32 input" begin
            y32 = Float32.(y)
            r = hp_filter(y32)
            @test r isa HPFilterResult{Float32}
            @test r.trend .+ r.cycle ≈ y32
        end

        @testset "Integer input (fallback)" begin
            yi = round.(Int, y .* 10)
            r = hp_filter(yi)
            @test r isa HPFilterResult{Float64}
        end

        @testset "edge cases" begin
            @test_throws ArgumentError hp_filter([1.0, 2.0])  # too short
            @test_throws ArgumentError hp_filter(y; lambda=-1.0)  # negative lambda
        end

        @testset "display" begin
            r = hp_filter(y)
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Hodrick-Prescott", s)
            @test occursin("Lambda", s)
        end

        @testset "report" begin
            r = hp_filter(y)
            # report() delegates to show(), so just test show works
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Hodrick-Prescott", s)
        end

        @testset "refs" begin
            r = hp_filter(y)
            io = IOBuffer()
            refs(io, r)
            s = String(take!(io))
            @test occursin("Hodrick", s)
        end
    end

    # =============================================================================
    # Hamilton Filter
    # =============================================================================
    @testset "Hamilton Filter" begin
        Random.seed!(42)
        y = cumsum(randn(200))

        @testset "basic functionality" begin
            r = hamilton_filter(y)
            @test r isa HamiltonFilterResult{Float64}
            @test r.h == 8
            @test r.p == 4
            @test r.T_obs == 200
            # Effective length: T - h - p + 1 = 200 - 8 - 4 + 1 = 189
            @test length(r.cycle) == 189
            @test length(r.trend) == 189
            @test r.valid_range == 12:200
            @test length(r.beta) == 5  # intercept + 4 lags
        end

        @testset "trend + cycle = y at valid range" begin
            r = hamilton_filter(y)
            @test r.trend .+ r.cycle ≈ y[r.valid_range]
        end

        @testset "OLS correctness" begin
            r = hamilton_filter(y; h=2, p=2)
            # Check that residuals are orthogonal to regressors
            n_eff = length(r.cycle)
            X = hcat(ones(n_eff), [y[t] for t in (r.valid_range.start - 2):(r.valid_range.stop - 2)],
                     [y[t] for t in (r.valid_range.start - 3):(r.valid_range.stop - 3)])
            @test norm(X' * r.cycle) < 1e-8 * n_eff
        end

        @testset "custom h and p" begin
            r = hamilton_filter(y; h=24, p=12)
            @test r.h == 24
            @test r.p == 12
            @test length(r.cycle) == 200 - 24 - 12 + 1
        end

        @testset "accessors" begin
            r = hamilton_filter(y)
            @test trend(r) === r.trend
            @test cycle(r) === r.cycle
        end

        @testset "Float32 input" begin
            y32 = Float32.(y)
            r = hamilton_filter(y32)
            @test r isa HamiltonFilterResult{Float32}
        end

        @testset "Integer input (fallback)" begin
            yi = round.(Int, y .* 10)
            r = hamilton_filter(yi)
            @test r isa HamiltonFilterResult{Float64}
        end

        @testset "edge cases" begin
            @test_throws ArgumentError hamilton_filter(y; h=0)
            @test_throws ArgumentError hamilton_filter(y; p=0)
            @test_throws ArgumentError hamilton_filter(ones(10); h=8, p=4)  # too short
        end

        @testset "display" begin
            r = hamilton_filter(y)
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Hamilton", s)
            @test occursin("2018", s)
        end

        @testset "refs" begin
            r = hamilton_filter(y)
            io = IOBuffer()
            refs(io, r)
            s = String(take!(io))
            @test occursin("Hamilton", s)
        end
    end

    # =============================================================================
    # Beveridge-Nelson Decomposition
    # =============================================================================
    @testset "Beveridge-Nelson" begin
        Random.seed!(42)

        @testset "basic functionality" begin
            # Random walk + stationary AR(1)
            y = cumsum(randn(200)) + 0.3 * sin.(2π * (1:200) / 20)
            r = beveridge_nelson(y)
            @test r isa BeveridgeNelsonResult{Float64}
            @test length(r.permanent) == 200
            @test length(r.transitory) == 200
            @test r.T_obs == 200
            @test r.arima_order[2] == 1  # d = 1
            # permanent + transitory = y
            @test r.permanent .+ r.transitory ≈ y
        end

        @testset "manual order" begin
            y = cumsum(randn(200))
            r = beveridge_nelson(y; p=2, q=1)
            @test r.arima_order == (2, 1, 1)
            @test r.permanent .+ r.transitory ≈ y
        end

        @testset "pure random walk (p=0, q=0 white noise differences)" begin
            Random.seed!(123)
            y = cumsum(randn(200))
            r = beveridge_nelson(y; p=0, q=0)
            # When Δy is white noise, transitory = 0, permanent = y
            @test r.permanent ≈ y
            @test all(abs.(r.transitory) .< 1e-10)
            @test r.long_run_multiplier ≈ 1.0
        end

        @testset "accessors" begin
            y = cumsum(randn(200))
            r = beveridge_nelson(y; p=1, q=0)
            @test trend(r) === r.permanent
            @test cycle(r) === r.transitory
        end

        @testset "Float32 input" begin
            y = Float32.(cumsum(randn(200)))
            # auto_arima may fail with Float32 in some cases, use manual order
            r = beveridge_nelson(y; p=1, q=0)
            @test r isa BeveridgeNelsonResult{Float32}
        end

        @testset "Integer input (fallback)" begin
            yi = round.(Int, cumsum(randn(200)) .* 10)
            r = beveridge_nelson(yi; p=1, q=0)
            @test r isa BeveridgeNelsonResult{Float64}
        end

        @testset "edge cases" begin
            @test_throws ArgumentError beveridge_nelson(ones(5))  # too short
            @test_throws ArgumentError beveridge_nelson(ones(200); max_terms=0)
        end

        @testset "display" begin
            y = cumsum(randn(200))
            r = beveridge_nelson(y; p=1, q=0)
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Beveridge-Nelson", s)
        end

        @testset "refs" begin
            y = cumsum(randn(200))
            r = beveridge_nelson(y; p=1, q=0)
            io = IOBuffer()
            refs(io, r)
            s = String(take!(io))
            @test occursin("Beveridge", s)
        end
    end

    # =============================================================================
    # Baxter-King Band-Pass Filter
    # =============================================================================
    @testset "Baxter-King" begin
        Random.seed!(42)
        y = cumsum(randn(200))

        @testset "basic functionality" begin
            r = baxter_king(y)
            @test r isa BaxterKingResult{Float64}
            @test r.pl == 6
            @test r.pu == 32
            @test r.K == 12
            @test r.T_obs == 200
            @test length(r.cycle) == 200 - 2 * 12  # 176
            @test length(r.trend) == 176
            @test r.valid_range == 13:188
        end

        @testset "trend + cycle = y at valid range" begin
            r = baxter_king(y)
            @test r.trend .+ r.cycle ≈ y[r.valid_range]
        end

        @testset "weights sum to approximately zero" begin
            r = baxter_king(y)
            # a_0 + 2 * sum(a_1:a_K) should be approximately 0
            total = r.weights[1] + 2 * sum(r.weights[2:end])
            @test abs(total) < 1e-10
        end

        @testset "sinusoid passthrough" begin
            # Create a pure sinusoid in the passband
            t = 1:300
            # Period = 16 quarters (in [6, 32] band) => should pass through
            y_sin = sin.(2π .* t ./ 16)
            r = baxter_king(y_sin; K=12)
            # Filtered cycle should retain most of the signal
            @test cor(r.cycle, y_sin[r.valid_range]) > 0.9
        end

        @testset "sinusoid rejection" begin
            # Create a pure sinusoid outside the passband
            t = 1:300
            # Period = 3 quarters (outside [6, 32] band) => should be attenuated
            y_sin = sin.(2π .* t ./ 3)
            r = baxter_king(y_sin; K=12)
            # Cycle amplitude should be very small relative to input
            @test std(r.cycle) / std(y_sin) < 0.2
        end

        @testset "custom parameters" begin
            r = baxter_king(y; pl=2, pu=8, K=6)
            @test r.pl == 2
            @test r.pu == 8
            @test r.K == 6
            @test length(r.cycle) == 200 - 12
        end

        @testset "accessors" begin
            r = baxter_king(y)
            @test trend(r) === r.trend
            @test cycle(r) === r.cycle
        end

        @testset "Float32 input" begin
            y32 = Float32.(y)
            r = baxter_king(y32)
            @test r isa BaxterKingResult{Float32}
        end

        @testset "Integer input (fallback)" begin
            yi = round.(Int, y .* 10)
            r = baxter_king(yi)
            @test r isa BaxterKingResult{Float64}
        end

        @testset "edge cases" begin
            @test_throws ArgumentError baxter_king(y; pl=1)  # pl < 2
            @test_throws ArgumentError baxter_king(y; pl=32, pu=6)  # pu <= pl
            @test_throws ArgumentError baxter_king(y; K=0)  # K < 1
            @test_throws ArgumentError baxter_king(ones(20); K=12)  # too short
        end

        @testset "display" begin
            r = baxter_king(y)
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Baxter-King", s)
        end

        @testset "refs" begin
            r = baxter_king(y)
            io = IOBuffer()
            refs(io, r)
            s = String(take!(io))
            @test occursin("Baxter", s)
        end
    end

    # =============================================================================
    # Boosted HP Filter
    # =============================================================================
    @testset "Boosted HP" begin
        Random.seed!(42)
        y = cumsum(randn(200))

        @testset "BIC stopping" begin
            r = boosted_hp(y; stopping=:BIC)
            @test r isa BoostedHPResult{Float64}
            @test length(r.trend) == 200
            @test length(r.cycle) == 200
            @test r.lambda == 1600.0
            @test r.stopping == :BIC
            @test r.iterations >= 1
            @test length(r.bic_path) >= 1
            @test r.trend .+ r.cycle ≈ y
        end

        @testset "ADF stopping" begin
            r = boosted_hp(y; stopping=:ADF, sig_p=0.10)
            @test r.stopping == :ADF
            @test length(r.adf_pvalues) >= 1
            @test r.trend .+ r.cycle ≈ y
        end

        @testset "fixed stopping" begin
            r = boosted_hp(y; stopping=:fixed, max_iter=5)
            @test r.stopping == :fixed
            @test r.iterations == 5
            @test r.trend .+ r.cycle ≈ y
        end

        @testset "matches HP at iter 1 approximately" begin
            hp_r = hp_filter(y)
            # boosted_hp with BIC at iteration 1 should be close to HP
            # (not identical because BIC may stop at 1)
            r = boosted_hp(y; stopping=:fixed, max_iter=1)
            @test r.iterations == 1
            @test r.trend ≈ hp_r.trend atol=1e-10
            @test r.cycle ≈ hp_r.cycle atol=1e-10
        end

        @testset "accessors" begin
            r = boosted_hp(y)
            @test trend(r) === r.trend
            @test cycle(r) === r.cycle
        end

        @testset "Float32 input" begin
            y32 = Float32.(y)
            r = boosted_hp(y32; stopping=:fixed, max_iter=3)
            @test r isa BoostedHPResult{Float32}
        end

        @testset "Integer input (fallback)" begin
            yi = round.(Int, y .* 10)
            r = boosted_hp(yi; stopping=:fixed, max_iter=3)
            @test r isa BoostedHPResult{Float64}
        end

        @testset "IC monotonicity at stopping" begin
            # When BIC stops, the last IC value should be >= the previous one
            r = boosted_hp(y; stopping=:BIC)
            if length(r.bic_path) >= 2
                @test r.bic_path[end] >= r.bic_path[end-1]
            end
        end

        @testset "IC formula at iteration 1" begin
            # IC(1) = var(c₁)/var(c₁) + log(n)/tr(I-S) * tr(B₁)
            # = 1 + log(n)/tr(I-S) * tr(S)   since B₁ = I - (I-S) = S
            r = boosted_hp(y; stopping=:fixed, max_iter=1)
            # Access internal helpers to verify
            n = length(y)
            lam = 1600.0
            A = MacroEconometricModels._hp_penalty(n, lam)
            eigs_A = eigvals(Matrix(A))
            eig_ImS = [1.0 - 1.0/e for e in eigs_A]
            tr_ImS = sum(eig_ImS)
            tr_S = sum(1.0 ./ eigs_A)
            expected_ic1 = 1.0 + log(Float64(n)) / tr_ImS * tr_S
            # Run with BIC to get IC path
            r2 = boosted_hp(y; stopping=:BIC)
            @test r2.bic_path[1] ≈ expected_ic1 rtol=1e-10
        end

        @testset "IC decreasing early iterations" begin
            # For a random walk, boosting should reduce IC in early iterations
            r = boosted_hp(y; stopping=:fixed, max_iter=10)
            # Compute IC path manually
            n = length(y)
            lam = 1600.0
            A = MacroEconometricModels._hp_penalty(n, lam)
            eigs_A = eigvals(Matrix(A))
            eig_ImS = [1.0 - 1.0/e for e in eigs_A]
            F = cholesky(A)
            yv = Float64.(y)
            tau = Vector{Float64}(F \ yv)
            cyc = yv .- tau
            var_hp = var(cyc)
            ic_vals = Float64[]
            for iter in 1:5
                push!(ic_vals, MacroEconometricModels._phillips_shi_ic(
                    var(cyc), var_hp, eig_ImS, n, iter))
                cyc_trend = Vector{Float64}(F \ cyc)
                cyc = cyc .- cyc_trend
            end
            # IC should decrease for at least the first few iterations
            @test ic_vals[2] < ic_vals[1]
        end

        @testset "edge cases" begin
            @test_throws ArgumentError boosted_hp([1.0, 2.0])  # too short
            @test_throws ArgumentError boosted_hp(y; lambda=-1.0)
            @test_throws ArgumentError boosted_hp(y; max_iter=0)
            @test_throws ArgumentError boosted_hp(y; stopping=:invalid)
        end

        @testset "display" begin
            r = boosted_hp(y; stopping=:fixed, max_iter=3)
            io = IOBuffer()
            show(io, r)
            s = String(take!(io))
            @test occursin("Boosted HP", s)
            @test occursin("Phillips", s)
        end

        @testset "refs" begin
            r = boosted_hp(y; stopping=:fixed, max_iter=3)
            io = IOBuffer()
            refs(io, r)
            s = String(take!(io))
            @test occursin("Phillips", s)
        end
    end

    # =============================================================================
    # Symbol dispatch for refs()
    # =============================================================================
    @testset "Symbol dispatch refs" begin
        for sym in [:hp_filter, :hamilton_filter, :beveridge_nelson, :baxter_king, :boosted_hp]
            io = IOBuffer()
            refs(io, sym)
            s = String(take!(io))
            @test length(s) > 10
        end
    end

    # =============================================================================
    # AbstractFilterResult type hierarchy
    # =============================================================================
    @testset "Type hierarchy" begin
        Random.seed!(42)
        y = cumsum(randn(200))
        @test hp_filter(y) isa AbstractFilterResult
        @test hamilton_filter(y) isa AbstractFilterResult
        @test beveridge_nelson(y; p=1, q=0) isa AbstractFilterResult
        @test baxter_king(y) isa AbstractFilterResult
        @test boosted_hp(y; stopping=:fixed, max_iter=2) isa AbstractFilterResult
    end

end
