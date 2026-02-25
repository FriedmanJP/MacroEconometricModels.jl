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
using Random
using DataFrames
using Statistics

const _suppress_warnings = MacroEconometricModels._suppress_warnings

@testset "Data Module" begin

    # =========================================================================
    # 1. TimeSeriesData Construction
    # =========================================================================
    @testset "TimeSeriesData Construction" begin
        # From Matrix
        @testset "Matrix constructor" begin
            Random.seed!(42)
            mat = randn(100, 3)
            d = TimeSeriesData(mat)
            @test nobs(d) == 100
            @test nvars(d) == 3
            @test varnames(d) == ["x1", "x2", "x3"]
            @test frequency(d) == Other
            @test d.tcode == [1, 1, 1]
            @test d.time_index == collect(1:100)
            @test size(d) == (100, 3)
            @test length(d) == 300
        end

        @testset "Matrix with kwargs" begin
            mat = randn(50, 2)
            d = TimeSeriesData(mat; varnames=["GDP", "CPI"],
                              frequency=Quarterly, tcode=[5, 5],
                              time_index=collect(1980:2029))
            @test varnames(d) == ["GDP", "CPI"]
            @test frequency(d) == Quarterly
            @test d.tcode == [5, 5]
            @test time_index(d) == collect(1980:2029)
        end

        # From Vector (univariate)
        @testset "Vector constructor" begin
            v = randn(200)
            d = TimeSeriesData(v; varname="GDP", frequency=Monthly)
            @test nobs(d) == 200
            @test nvars(d) == 1
            @test varnames(d) == ["GDP"]
            @test frequency(d) == Monthly
            @test size(d) == (200, 1)
        end

        # Non-float input
        @testset "Non-float conversion" begin
            mat_int = [1 2; 3 4; 5 6]
            d = TimeSeriesData(mat_int)
            @test eltype(d.data) == Float64
            @test d.data[1, 1] == 1.0

            v_int = [1, 2, 3, 4, 5]
            d2 = TimeSeriesData(v_int)
            @test eltype(d2.data) == Float64
        end

        # DataFrame constructor
        @testset "DataFrame constructor" begin
            df = DataFrame(a=randn(50), b=randn(50), c=["x" for _ in 1:50])
            d = TimeSeriesData(df; frequency=Yearly)
            @test nvars(d) == 2  # only numeric columns
            @test varnames(d) == ["a", "b"]
            @test frequency(d) == Yearly
        end

        @testset "DataFrame with missing" begin
            df = DataFrame(x=[1.0, missing, 3.0], y=[4.0, 5.0, 6.0])
            d = TimeSeriesData(df)
            @test isnan(d.data[2, 1])  # missing → NaN
            @test d.data[2, 2] == 5.0
        end

        # Validation
        @testset "Construction validation" begin
            @test_throws ArgumentError TimeSeriesData(randn(5, 2); varnames=["a"])
            @test_throws ArgumentError TimeSeriesData(randn(5, 2); tcode=[1])
            @test_throws ArgumentError TimeSeriesData(randn(5, 2); tcode=[1, 8])
            @test_throws ArgumentError TimeSeriesData(randn(5, 2); time_index=[1, 2])
            @test_throws ArgumentError TimeSeriesData(Matrix{Float64}(undef, 0, 3))
        end

        # Float32
        @testset "Float32 support" begin
            mat32 = randn(Float32, 50, 2)
            d = TimeSeriesData(mat32)
            @test eltype(d.data) == Float32
        end

        # All frequencies
        @testset "All frequency values" begin
            for freq in (Daily, Monthly, Quarterly, Yearly, Mixed, Other)
                d = TimeSeriesData(randn(10, 1); frequency=freq)
                @test frequency(d) == freq
            end
        end
    end

    # =========================================================================
    # 2. PanelData Construction
    # =========================================================================
    @testset "PanelData Construction" begin
        @testset "xtset balanced" begin
            df = DataFrame(id=repeat(1:3, inner=50), t=repeat(1:50, 3),
                          x=randn(150), y=randn(150))
            pd = xtset(df, :id, :t)
            @test nobs(pd) == 150
            @test nvars(pd) == 2
            @test ngroups(pd) == 3
            @test isbalanced(pd)
            @test groups(pd) == ["1", "2", "3"]
            @test size(pd) == (150, 2)
            @test length(pd) == 300
        end

        @testset "xtset unbalanced" begin
            df = DataFrame(id=[1,1,1,2,2], t=[1,2,3,1,2],
                          x=randn(5))
            pd = xtset(df, :id, :t)
            @test nobs(pd) == 5
            @test ngroups(pd) == 2
            @test !isbalanced(pd)
        end

        @testset "xtset with frequency" begin
            df = DataFrame(id=repeat(1:2, inner=4), t=repeat(1:4, 2), x=randn(8))
            pd = xtset(df, :id, :t; frequency=Quarterly)
            @test frequency(pd) == Quarterly
        end

        @testset "xtset validation" begin
            df = DataFrame(id=[1,1], t=[1,1], x=[1.0, 2.0])
            @test_throws ArgumentError xtset(df, :id, :t)  # duplicate pair
            @test_throws ArgumentError xtset(df, :nonexistent, :t)
            @test_throws ArgumentError xtset(df, :id, :nonexistent)
        end

        @testset "xtset no numeric columns" begin
            df = DataFrame(id=[1,2], t=[1,2], name=["a","b"])
            @test_throws ArgumentError xtset(df, :id, :t)
        end

        @testset "group_data extraction" begin
            df = DataFrame(id=repeat(1:3, inner=20), t=repeat(1:20, 3),
                          x=randn(60), y=randn(60))
            pd = xtset(df, :id, :t)
            g1 = group_data(pd, 1)
            @test g1 isa TimeSeriesData
            @test nobs(g1) == 20
            @test nvars(g1) == 2

            g2 = group_data(pd, "2")
            @test nobs(g2) == 20

            @test_throws ArgumentError group_data(pd, 0)
            @test_throws ArgumentError group_data(pd, 4)
            @test_throws ArgumentError group_data(pd, "nonexistent")
        end

        @testset "panel_summary" begin
            df = DataFrame(id=repeat(1:2, inner=30), t=repeat(1:30, 2),
                          x=randn(60), y=randn(60))
            pd = xtset(df, :id, :t)
            io = IOBuffer()
            panel_summary(io, pd)
            s = String(take!(io))
            @test occursin("2 groups", s)
            @test occursin("60 total", s)
            @test occursin("balanced", s)
        end
    end

    # =========================================================================
    # 3. CrossSectionData Construction
    # =========================================================================
    @testset "CrossSectionData Construction" begin
        @testset "Matrix constructor" begin
            mat = randn(50, 3)
            d = CrossSectionData(mat; varnames=["income", "age", "edu"])
            @test nobs(d) == 50
            @test nvars(d) == 3
            @test varnames(d) == ["income", "age", "edu"]
            @test obs_id(d) == collect(1:50)
            @test size(d) == (50, 3)
        end

        @testset "Non-float conversion" begin
            d = CrossSectionData([1 2; 3 4])
            @test eltype(d.data) == Float64
        end

        @testset "DataFrame constructor" begin
            df = DataFrame(x=randn(10), y=randn(10))
            d = CrossSectionData(df)
            @test nobs(d) == 10
            @test nvars(d) == 2
        end

        @testset "Validation" begin
            @test_throws ArgumentError CrossSectionData(randn(5, 2); varnames=["a"])
            @test_throws ArgumentError CrossSectionData(randn(5, 2); obs_id=[1])
        end
    end

    # =========================================================================
    # 4. Accessors and Indexing
    # =========================================================================
    @testset "Accessors and Indexing" begin
        Random.seed!(123)
        mat = randn(100, 3)
        d = TimeSeriesData(mat; varnames=["GDP", "CPI", "FFR"])

        @testset "Column indexing by name" begin
            gdp = d[:, "GDP"]
            @test gdp isa Vector
            @test length(gdp) == 100
            @test gdp ≈ mat[:, 1]
        end

        @testset "Column indexing by index" begin
            col2 = d[:, 2]
            @test col2 ≈ mat[:, 2]
        end

        @testset "Multi-column indexing" begin
            sub = d[:, ["GDP", "FFR"]]
            @test sub isa TimeSeriesData
            @test nvars(sub) == 2
            @test varnames(sub) == ["GDP", "FFR"]
            @test sub.data[:, 1] ≈ mat[:, 1]
            @test sub.data[:, 2] ≈ mat[:, 3]
        end

        @testset "Indexing errors" begin
            @test_throws ArgumentError d[:, "NONEXISTENT"]
            @test_throws ArgumentError d[:, ["GDP", "NONEXISTENT"]]
            @test_throws BoundsError d[:, 0]
            @test_throws BoundsError d[:, 4]
        end

        @testset "CrossSectionData indexing" begin
            cs = CrossSectionData(randn(10, 2); varnames=["a", "b"])
            @test length(cs[:, "a"]) == 10
            @test_throws ArgumentError cs[:, "c"]
        end

        @testset "Matrix/Vector conversion" begin
            @test Matrix(d) === d.data
            d1 = TimeSeriesData(randn(50))
            @test Vector(d1) isa Vector
            @test length(Vector(d1)) == 50
            @test_throws ArgumentError Vector(d)  # multivariate
        end

        @testset "rename_vars!" begin
            d2 = TimeSeriesData(randn(20, 2); varnames=["a", "b"])
            rename_vars!(d2, "a" => "alpha")
            @test varnames(d2) == ["alpha", "b"]

            rename_vars!(d2, ["x", "y"])
            @test varnames(d2) == ["x", "y"]

            @test_throws ArgumentError rename_vars!(d2, "nonexistent" => "z")
            @test_throws ArgumentError rename_vars!(d2, ["a"])  # wrong length
        end

        @testset "rename_vars! PanelData" begin
            df = DataFrame(id=[1,1,2,2], t=[1,2,1,2], x=randn(4), y=randn(4))
            pd = xtset(df, :id, :t)
            rename_vars!(pd, "x" => "alpha")
            @test varnames(pd) == ["alpha", "y"]
        end

        @testset "rename_vars! CrossSectionData" begin
            cs = CrossSectionData(randn(5, 2); varnames=["a", "b"])
            rename_vars!(cs, "a" => "alpha")
            @test varnames(cs) == ["alpha", "b"]
            rename_vars!(cs, ["x", "y"])
            @test varnames(cs) == ["x", "y"]
        end

        @testset "set_time_index!" begin
            d3 = TimeSeriesData(randn(10, 1))
            set_time_index!(d3, collect(2001:2010))
            @test time_index(d3) == collect(2001:2010)
            @test_throws ArgumentError set_time_index!(d3, [1, 2])
        end

        @testset "set_obs_id!" begin
            cs = CrossSectionData(randn(5, 1))
            set_obs_id!(cs, [10, 20, 30, 40, 50])
            @test obs_id(cs) == [10, 20, 30, 40, 50]
            @test_throws ArgumentError set_obs_id!(cs, [1])
        end
    end

    # =========================================================================
    # 5. Validation
    # =========================================================================
    @testset "Validation" begin
        @testset "diagnose clean data" begin
            d = TimeSeriesData(randn(100, 3))
            diag = diagnose(d)
            @test diag.is_clean
            @test all(==(0), diag.n_nan)
            @test all(==(0), diag.n_inf)
            @test !any(diag.is_constant)
            @test !diag.is_short
        end

        @testset "diagnose NaN" begin
            mat = randn(50, 2)
            mat[5, 1] = NaN
            mat[10, 1] = NaN
            mat[3, 2] = NaN
            d = TimeSeriesData(mat)
            diag = diagnose(d)
            @test !diag.is_clean
            @test diag.n_nan == [2, 1]
        end

        @testset "diagnose Inf" begin
            mat = randn(50, 2)
            mat[1, 1] = Inf
            mat[2, 2] = -Inf
            d = TimeSeriesData(mat)
            diag = diagnose(d)
            @test !diag.is_clean
            @test diag.n_inf == [1, 1]
        end

        @testset "diagnose constant" begin
            mat = hcat(ones(50), randn(50))
            d = TimeSeriesData(mat)
            diag = diagnose(d)
            @test !diag.is_clean
            @test diag.is_constant[1]
            @test !diag.is_constant[2]
        end

        @testset "diagnose short" begin
            d = TimeSeriesData(randn(5, 2))
            diag = diagnose(d)
            @test !diag.is_clean
            @test diag.is_short
        end

        @testset "diagnose display" begin
            mat = randn(50, 2)
            mat[1, 1] = NaN
            d = TimeSeriesData(mat; varnames=["GDP", "CPI"])
            diag = diagnose(d)
            io = IOBuffer()
            show(io, diag)
            s = String(take!(io))
            @test occursin("issues detected", s)
        end

        @testset "diagnose clean display" begin
            d = TimeSeriesData(randn(20, 2))
            diag = diagnose(d)
            io = IOBuffer()
            show(io, diag)
            s = String(take!(io))
            @test occursin("clean", s)
        end

        @testset "fix listwise" begin
            mat = [1.0 2.0; NaN 3.0; 4.0 5.0; 6.0 NaN; 7.0 8.0;
                   9.0 10.0; 11.0 12.0; 13.0 14.0; 15.0 16.0; 17.0 18.0;
                   19.0 20.0; 21.0 22.0]
            d = TimeSeriesData(mat)
            d_fixed = fix(d; method=:listwise)
            @test nobs(d_fixed) == 10  # 2 rows dropped
            @test diagnose(d_fixed).is_clean
        end

        @testset "fix interpolate" begin
            mat = [1.0; NaN; 3.0; 4.0; NaN; NaN; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0]
            d = TimeSeriesData(mat)
            d_fixed = fix(d; method=:interpolate)
            @test nobs(d_fixed) == 12
            @test !any(isnan, Matrix(d_fixed))
            @test d_fixed.data[2, 1] ≈ 2.0  # interpolated
        end

        @testset "fix mean" begin
            mat = [1.0; NaN; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0]
            d = TimeSeriesData(mat)
            d_fixed = fix(d; method=:mean)
            @test nobs(d_fixed) == 12
            @test !any(isnan, Matrix(d_fixed))
            # Mean of finite values: (1+3+4+5+6+7+8+9+10+11+12)/11 ≈ 6.909...
            finite_mean = mean([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            @test d_fixed.data[2, 1] ≈ finite_mean
        end

        @testset "fix Inf replacement" begin
            mat = [1.0; Inf; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0; 12.0]
            d = TimeSeriesData(mat)
            d_fixed = fix(d; method=:listwise)
            @test !any(isinf, Matrix(d_fixed))
        end

        @testset "fix drops constant columns" begin
            _suppress_warnings() do
                mat = hcat(ones(20), randn(20))
                d = TimeSeriesData(mat; varnames=["const_col", "good_col"])
                d_fixed = fix(d)
                @test nvars(d_fixed) == 1
                @test varnames(d_fixed) == ["good_col"]
            end
        end

        @testset "fix invalid method" begin
            d = TimeSeriesData(randn(20, 1))
            @test_throws ArgumentError fix(d; method=:invalid)
        end

        @testset "validate_for_model" begin
            d_multi = TimeSeriesData(randn(100, 3))
            d_uni = TimeSeriesData(randn(100, 1))

            # Multivariate models need n_vars ≥ 2
            @test validate_for_model(d_multi, :var) === nothing
            @test validate_for_model(d_multi, :vecm) === nothing
            @test validate_for_model(d_multi, :bvar) === nothing
            @test validate_for_model(d_multi, :factors) === nothing
            @test_throws ArgumentError validate_for_model(d_uni, :var)
            @test_throws ArgumentError validate_for_model(d_uni, :vecm)

            # Univariate models need n_vars == 1
            @test validate_for_model(d_uni, :arima) === nothing
            @test validate_for_model(d_uni, :arch) === nothing
            @test validate_for_model(d_uni, :hp_filter) === nothing
            @test validate_for_model(d_uni, :adf) === nothing
            @test_throws ArgumentError validate_for_model(d_multi, :arima)
            @test_throws ArgumentError validate_for_model(d_multi, :sv)

            # Flexible models accept any
            @test validate_for_model(d_multi, :lp) === nothing
            @test validate_for_model(d_uni, :lp) === nothing
            @test validate_for_model(d_multi, :gmm) === nothing

            # Unknown model type
            @test_throws ArgumentError validate_for_model(d_uni, :unknown_model)
        end
    end

    # =========================================================================
    # 6. Transformations
    # =========================================================================
    @testset "Transformations" begin
        @testset "tcode 1 — levels" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test apply_tcode(y, 1) == y
        end

        @testset "tcode 2 — first difference" begin
            y = [1.0, 3.0, 6.0, 10.0]
            @test apply_tcode(y, 2) ≈ [2.0, 3.0, 4.0]
        end

        @testset "tcode 3 — second difference" begin
            y = [1.0, 3.0, 6.0, 10.0, 15.0]
            @test apply_tcode(y, 3) ≈ [1.0, 1.0, 1.0]
        end

        @testset "tcode 4 — log" begin
            y = [1.0, exp(1.0), exp(2.0)]
            @test apply_tcode(y, 4) ≈ [0.0, 1.0, 2.0]
        end

        @testset "tcode 5 — diff of log" begin
            y = [100.0, 110.0, 121.0]
            result = apply_tcode(y, 5)
            @test length(result) == 2
            @test result ≈ [log(110/100), log(121/110)]
        end

        @testset "tcode 6 — second diff of log" begin
            y = exp.([1.0, 2.0, 4.0, 7.0, 11.0])
            result = apply_tcode(y, 6)
            @test length(result) == 3
        end

        @testset "tcode 7 — delta pct change" begin
            y = [100.0, 105.0, 112.0, 115.0]
            result = apply_tcode(y, 7)
            @test length(result) == 2
        end

        @testset "tcode validation" begin
            @test_throws ArgumentError apply_tcode([1.0, 2.0], 0)
            @test_throws ArgumentError apply_tcode([1.0, 2.0], 8)
            @test_throws ArgumentError apply_tcode([-1.0, 2.0], 4)  # non-positive
            @test_throws ArgumentError apply_tcode([0.0, 2.0], 5)   # zero
        end

        @testset "TimeSeriesData apply_tcode per-variable" begin
            d = TimeSeriesData(rand(50, 3) .+ 1.0; varnames=["a", "b", "c"])
            d2 = apply_tcode(d, [5, 5, 1])
            @test nobs(d2) == 49  # one row lost due to tcode 5
            @test nvars(d2) == 3
            @test d2.tcode == [5, 5, 1]
        end

        @testset "TimeSeriesData apply_tcode uniform" begin
            d = TimeSeriesData(rand(50, 2) .+ 1.0)
            d2 = apply_tcode(d, 5)
            @test nobs(d2) == 49
            @test d2.tcode == [5, 5]
        end

        @testset "Transform length validation" begin
            d = TimeSeriesData(randn(50, 2))
            @test_throws ArgumentError apply_tcode(d, [1])  # wrong length
        end

        @testset "PanelData apply_tcode per-variable" begin
            df = DataFrame(id=repeat(1:3, inner=50),
                           t=repeat(1:50, 3),
                           x=rand(150) .+ 1.0,
                           y=rand(150) .+ 1.0)
            pd = xtset(df, :id, :t)
            pd2 = apply_tcode(pd, [5, 5])
            @test pd2 isa PanelData
            @test ngroups(pd2) == 3
            @test nvars(pd2) == 2
            @test pd2.tcode == [5, 5]
            # Each group loses 1 row: 3 * 49 = 147
            @test nobs(pd2) == 147
            @test isbalanced(pd2)
        end

        @testset "PanelData apply_tcode uniform" begin
            df = DataFrame(id=repeat(1:2, inner=30),
                           t=repeat(1:30, 2),
                           x=rand(60) .+ 1.0)
            pd = xtset(df, :id, :t)
            pd2 = apply_tcode(pd, 5)
            @test nobs(pd2) == 58  # 2 * 29
            @test pd2.tcode == [5]
        end

        @testset "PanelData apply_tcode mixed codes" begin
            df = DataFrame(id=repeat(1:2, inner=50),
                           t=repeat(1:50, 2),
                           x=rand(100) .+ 1.0,
                           y=randn(100))
            pd = xtset(df, :id, :t)
            pd2 = apply_tcode(pd, [5, 1])  # log-diff x, leave y in levels
            # tcode 5 loses 1 row per group, tcode 1 loses 0
            # max_lost = 1 per group → 49 obs per group
            @test nobs(pd2) == 98
            @test pd2.tcode == [5, 1]
        end

        @testset "PanelData apply_tcode second diff" begin
            df = DataFrame(id=repeat(1:2, inner=50),
                           t=repeat(1:50, 2),
                           x=rand(100) .+ 1.0,
                           y=rand(100) .+ 1.0)
            pd = xtset(df, :id, :t)
            pd2 = apply_tcode(pd, [6, 5])  # tcode 6 loses 2, tcode 5 loses 1
            # max_lost = 2 per group → 48 obs per group
            @test nobs(pd2) == 96
        end

        @testset "PanelData apply_tcode metadata propagation" begin
            df = DataFrame(id=repeat(1:2, inner=30),
                           t=repeat(1:30, 2),
                           x=rand(60) .+ 1.0,
                           y=rand(60) .+ 1.0)
            vd = Dict("x" => "X variable", "y" => "Y variable")
            pd = xtset(df, :id, :t; desc="Test panel", vardesc=vd)
            pd2 = apply_tcode(pd, [5, 1])
            @test desc(pd2) == "Test panel"
            @test vardesc(pd2, "x") == "X variable"
            @test vardesc(pd2, "y") == "Y variable"
            @test groups(pd2) == ["1", "2"]
        end

        @testset "PanelData apply_tcode validation" begin
            df = DataFrame(id=repeat(1:2, inner=10),
                           t=repeat(1:10, 2),
                           x=rand(20) .+ 1.0,
                           y=rand(20) .+ 1.0)
            pd = xtset(df, :id, :t)
            @test_throws ArgumentError apply_tcode(pd, [1])  # wrong length
        end

        @testset "inverse_tcode tcode 1" begin
            y = [1.0, 2.0, 3.0]
            @test inverse_tcode(y, 1) == y
        end

        @testset "inverse_tcode tcode 2 round-trip" begin
            y = [100.0, 102.0, 105.0, 103.0, 108.0]
            yd = apply_tcode(y, 2)
            recovered = inverse_tcode(yd, 2; x_prev=[y[1]])
            @test recovered ≈ y[2:end]
        end

        @testset "inverse_tcode tcode 3 round-trip" begin
            y = [100.0, 102.0, 105.0, 103.0, 108.0]
            yd = apply_tcode(y, 3)
            recovered = inverse_tcode(yd, 3; x_prev=y[1:2])
            @test recovered ≈ y[3:end]
        end

        @testset "inverse_tcode tcode 4 round-trip" begin
            y = [1.0, 2.0, 3.0]
            yd = apply_tcode(y, 4)
            @test inverse_tcode(yd, 4) ≈ y
        end

        @testset "inverse_tcode tcode 5 round-trip" begin
            y = [100.0, 105.0, 110.0, 108.0]
            yd = apply_tcode(y, 5)
            recovered = inverse_tcode(yd, 5; x_prev=[y[1]])
            @test recovered ≈ y[2:end] atol=1e-10
        end

        @testset "inverse_tcode tcode 6 round-trip" begin
            y = [100.0, 105.0, 110.0, 115.0, 112.0]
            yd = apply_tcode(y, 6)
            recovered = inverse_tcode(yd, 6; x_prev=y[1:2])
            @test recovered ≈ y[3:end] atol=1e-10
        end

        @testset "inverse_tcode tcode 7 round-trip" begin
            y = [100.0, 105.0, 112.0, 115.0]
            yd = apply_tcode(y, 7)
            recovered = inverse_tcode(yd, 7; x_prev=y[1:2])
            @test recovered ≈ y[3:end] atol=1e-10
        end

        @testset "inverse_tcode missing x_prev" begin
            @test_throws ArgumentError inverse_tcode([1.0, 2.0], 2)
            @test_throws ArgumentError inverse_tcode([1.0, 2.0], 3)
            @test_throws ArgumentError inverse_tcode([1.0, 2.0], 5)
            @test_throws ArgumentError inverse_tcode([1.0, 2.0], 6)
            @test_throws ArgumentError inverse_tcode([1.0, 2.0], 7)
        end

        @testset "inverse_tcode x_prev too short" begin
            @test_throws ArgumentError inverse_tcode([1.0], 3; x_prev=[1.0])
            @test_throws ArgumentError inverse_tcode([1.0], 6; x_prev=[1.0])
            @test_throws ArgumentError inverse_tcode([1.0], 7; x_prev=[1.0])
        end

        @testset "inverse_tcode tcode validation" begin
            @test_throws ArgumentError inverse_tcode([1.0], 0)
            @test_throws ArgumentError inverse_tcode([1.0], 8)
        end
    end

    # =========================================================================
    # 7. Summary Statistics
    # =========================================================================
    @testset "Summary Statistics" begin
        @testset "describe_data basic" begin
            Random.seed!(42)
            d = TimeSeriesData(randn(200, 3); varnames=["GDP", "CPI", "FFR"],
                              frequency=Quarterly)
            s = describe_data(d)
            @test s isa DataSummary
            @test s.n_vars == 3
            @test s.T_obs == 200
            @test s.frequency == Quarterly
            @test all(s.n .== 200)
            @test length(s.mean) == 3
            @test length(s.std) == 3
            @test all(s.min .<= s.mean)
            @test all(s.mean .<= s.max)
            @test all(s.p25 .<= s.median)
            @test all(s.median .<= s.p75)
        end

        @testset "describe_data with NaN" begin
            mat = randn(100, 2)
            mat[1, 1] = NaN
            mat[2, 1] = NaN
            d = TimeSeriesData(mat)
            s = describe_data(d)
            @test s.n[1] == 98
            @test s.n[2] == 100
        end

        @testset "describe_data display" begin
            d = TimeSeriesData(randn(50, 2); varnames=["a", "b"])
            s = describe_data(d)
            io = IOBuffer()
            show(io, s)
            str = String(take!(io))
            @test occursin("Summary Statistics", str)
            @test occursin("a", str)
            @test occursin("b", str)
        end

        @testset "describe_data CrossSectionData" begin
            cs = CrossSectionData(randn(50, 2); varnames=["inc", "age"])
            s = describe_data(cs)
            @test s.n_vars == 2
            @test all(s.n .== 50)
        end

        @testset "describe_data PanelData" begin
            redirect_stdout(devnull) do
                df = DataFrame(id=repeat(1:2, inner=30), t=repeat(1:30, 2),
                              x=randn(60), y=randn(60))
                pd = xtset(df, :id, :t)
                s = describe_data(pd)
                @test s isa DataSummary
                @test s.n_vars == 2
            end
        end

        @testset "quantile edge cases" begin
            # Single value
            d = TimeSeriesData(fill(5.0, 1, 1))
            s = describe_data(d)
            @test s.mean[1] == 5.0
            @test s.p25[1] == 5.0
            @test s.median[1] == 5.0
        end
    end

    # =========================================================================
    # 8. Conversion and Estimation Dispatch
    # =========================================================================
    @testset "Conversion" begin
        @testset "to_matrix" begin
            d = TimeSeriesData(randn(50, 3))
            @test to_matrix(d) === d.data
        end

        @testset "to_vector" begin
            d = TimeSeriesData(randn(50, 1))
            @test to_vector(d) isa Vector
            @test length(to_vector(d)) == 50
        end

        @testset "to_vector by index" begin
            d = TimeSeriesData(randn(50, 3))
            @test length(to_vector(d, 2)) == 50
            @test_throws BoundsError to_vector(d, 0)
            @test_throws BoundsError to_vector(d, 4)
        end

        @testset "to_vector by name" begin
            d = TimeSeriesData(randn(50, 3); varnames=["a", "b", "c"])
            v = to_vector(d, "b")
            @test v ≈ d.data[:, 2]
            @test_throws ArgumentError to_vector(d, "z")
        end

        @testset "to_vector multivariate error" begin
            d = TimeSeriesData(randn(50, 3))
            @test_throws ArgumentError to_vector(d)
        end

        @testset "estimate_var dispatch" begin
            Random.seed!(42)
            d = TimeSeriesData(randn(100, 3); varnames=["y1", "y2", "y3"])
            model = estimate_var(d, 2)
            @test model isa VARModel
            @test nobs(model) > 0
        end

        @testset "estimate_ar dispatch" begin
            Random.seed!(42)
            d = TimeSeriesData(randn(100); varname="gdp")
            model = estimate_ar(d, 2)
            @test model isa ARModel
        end

        @testset "hp_filter dispatch" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(100)); varname="gdp")
            result = hp_filter(d)
            @test result isa HPFilterResult
        end

        @testset "adf_test dispatch" begin
            Random.seed!(42)
            d = TimeSeriesData(randn(100); varname="y")
            result = adf_test(d)
            @test result isa ADFResult
        end
    end

    # =========================================================================
    # 9. Edge Cases
    # =========================================================================
    @testset "Edge Cases" begin
        @testset "Single observation" begin
            d = TimeSeriesData(reshape([1.0], 1, 1))
            @test nobs(d) == 1
            @test nvars(d) == 1
            diag = diagnose(d)
            @test diag.is_short  # < 10 obs
        end

        @testset "Single variable" begin
            d = TimeSeriesData(randn(50, 1))
            @test nvars(d) == 1
            @test Vector(d) isa Vector
        end

        @testset "Large varnames" begin
            d = TimeSeriesData(randn(10, 15))
            @test nvars(d) == 15
            @test length(varnames(d)) == 15
        end

        @testset "TimeSeriesData from DataFrame with only string cols" begin
            df = DataFrame(a=["x","y","z"], b=["a","b","c"])
            @test_throws ArgumentError TimeSeriesData(df)
        end

        @testset "CrossSectionData from DataFrame with only string cols" begin
            df = DataFrame(a=["x","y","z"])
            @test_throws ArgumentError CrossSectionData(df)
        end

        @testset "fix all constant" begin
            _suppress_warnings() do
                mat = hcat(ones(20), 2.0 .* ones(20))
                d = TimeSeriesData(mat)
                @test_throws ArgumentError fix(d)
            end
        end

        @testset "Transform too short" begin
            d = TimeSeriesData(rand(2, 2) .+ 1.0)
            @test_throws ArgumentError apply_tcode(d, [3, 3])  # needs 2 rows lost, leaves 0
            d3 = TimeSeriesData(rand(1, 1) .+ 1.0)
            @test_throws ArgumentError apply_tcode(d3, [3])
        end

        @testset "Interpolate all NaN column" begin
            _suppress_warnings() do
                mat = fill(NaN, 20, 1)
                d = TimeSeriesData(mat)
                # All NaN → constant after interpolation → error
                @test_throws ArgumentError fix(d; method=:interpolate)
                @test_throws ArgumentError fix(d; method=:mean)
            end
        end
    end

    # =========================================================================
    # 10. Display
    # =========================================================================
    @testset "Display" begin
        @testset "TimeSeriesData show" begin
            d = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "FFR"],
                              frequency=Quarterly)
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test occursin("TimeSeriesData", s)
            @test occursin("100 obs", s)
            @test occursin("3 vars", s)
            @test occursin("Quarterly", s)
            @test occursin("GDP", s)
        end

        @testset "TimeSeriesData MIME show" begin
            d = TimeSeriesData(randn(50, 2))
            io = IOBuffer()
            show(io, MIME"text/plain"(), d)
            s = String(take!(io))
            @test occursin("TimeSeriesData", s)
        end

        @testset "PanelData show" begin
            df = DataFrame(id=repeat(1:2, inner=10), t=repeat(1:10, 2), x=randn(20))
            pd = xtset(df, :id, :t)
            io = IOBuffer()
            show(io, pd)
            s = String(take!(io))
            @test occursin("PanelData", s)
            @test occursin("2 groups", s)
            @test occursin("balanced", s)
        end

        @testset "CrossSectionData show" begin
            d = CrossSectionData(randn(50, 2); varnames=["a", "b"])
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test occursin("CrossSectionData", s)
            @test occursin("50 obs", s)
        end

        @testset "TimeSeriesData many vars (no varnames in show)" begin
            d = TimeSeriesData(randn(10, 15))
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test occursin("15 vars", s)
            @test !occursin("x1", s)  # too many vars, not listed
        end
    end

    # =========================================================================
    # 11. Descriptions (desc / vardesc)
    # =========================================================================
    @testset "Descriptions" begin
        @testset "desc at construction" begin
            d = TimeSeriesData(randn(50, 2);
                varnames=["GDP", "CPI"],
                desc="US macro data")
            @test desc(d) == "US macro data"
        end

        @testset "desc default empty" begin
            d = TimeSeriesData(randn(50, 2))
            @test desc(d) == ""
        end

        @testset "set_desc!" begin
            d = TimeSeriesData(randn(50, 2))
            set_desc!(d, "Updated description")
            @test desc(d) == "Updated description"
            set_desc!(d, "")
            @test desc(d) == ""
        end

        @testset "vardesc at construction" begin
            vd = Dict("GDP" => "Real GDP growth", "CPI" => "Consumer prices")
            d = TimeSeriesData(randn(50, 2);
                varnames=["GDP", "CPI"], vardesc=vd)
            @test vardesc(d) == vd
            @test vardesc(d, "GDP") == "Real GDP growth"
            @test vardesc(d, "CPI") == "Consumer prices"
        end

        @testset "vardesc default empty" begin
            d = TimeSeriesData(randn(50, 2); varnames=["GDP", "CPI"])
            @test isempty(vardesc(d))
            @test_throws ArgumentError vardesc(d, "GDP")
        end

        @testset "set_vardesc! single" begin
            d = TimeSeriesData(randn(50, 2); varnames=["GDP", "CPI"])
            set_vardesc!(d, "GDP", "Real GDP")
            @test vardesc(d, "GDP") == "Real GDP"
            @test_throws ArgumentError set_vardesc!(d, "NONEXISTENT", "x")
        end

        @testset "set_vardesc! dict" begin
            d = TimeSeriesData(randn(50, 2); varnames=["GDP", "CPI"])
            set_vardesc!(d, Dict("GDP" => "Real GDP", "CPI" => "CPI index"))
            @test vardesc(d, "GDP") == "Real GDP"
            @test vardesc(d, "CPI") == "CPI index"
            @test_throws ArgumentError set_vardesc!(d, Dict("BAD" => "x"))
        end

        @testset "desc propagates through subsetting" begin
            vd = Dict("GDP" => "Real GDP", "CPI" => "Consumer prices", "FFR" => "Fed rate")
            d = TimeSeriesData(randn(50, 3);
                varnames=["GDP", "CPI", "FFR"],
                desc="US macro", vardesc=vd)
            sub = d[:, ["GDP", "FFR"]]
            @test desc(sub) == "US macro"
            @test vardesc(sub, "GDP") == "Real GDP"
            @test vardesc(sub, "FFR") == "Fed rate"
            @test !haskey(vardesc(sub), "CPI")  # filtered out
        end

        @testset "desc propagates through apply_tcode" begin
            vd = Dict("a" => "Variable a", "b" => "Variable b")
            d = TimeSeriesData(rand(50, 2) .+ 1.0;
                varnames=["a", "b"], desc="test data", vardesc=vd)
            d2 = apply_tcode(d, [5, 1])
            @test desc(d2) == "test data"
            @test vardesc(d2, "a") == "Variable a"
            @test vardesc(d2, "b") == "Variable b"
        end

        @testset "desc propagates through fix" begin
            mat = [1.0 2.0; NaN 3.0; 4.0 5.0; 6.0 7.0; 8.0 9.0;
                   10.0 11.0; 12.0 13.0; 14.0 15.0; 16.0 17.0; 18.0 19.0;
                   20.0 21.0; 22.0 23.0]
            vd = Dict("x1" => "First var", "x2" => "Second var")
            d = TimeSeriesData(mat; desc="with desc", vardesc=vd)
            d_fixed = fix(d; method=:listwise)
            @test desc(d_fixed) == "with desc"
            @test vardesc(d_fixed, "x1") == "First var"
        end

        @testset "desc propagates through group_data" begin
            df = DataFrame(id=repeat(1:2, inner=10), t=repeat(1:10, 2),
                          x=randn(20), y=randn(20))
            vd = Dict("x" => "X variable", "y" => "Y variable")
            pd = xtset(df, :id, :t; desc="Panel dataset", vardesc=vd)
            @test desc(pd) == "Panel dataset"
            @test vardesc(pd, "x") == "X variable"

            g1 = group_data(pd, 1)
            @test desc(g1) == "Panel dataset"
            @test vardesc(g1, "x") == "X variable"
            @test vardesc(g1, "y") == "Y variable"
        end

        @testset "rename_vars! updates vardesc keys" begin
            vd = Dict("a" => "Alpha variable", "b" => "Beta variable")
            d = TimeSeriesData(randn(20, 2); varnames=["a", "b"], vardesc=vd)
            rename_vars!(d, "a" => "x")
            @test vardesc(d, "x") == "Alpha variable"
            @test !haskey(vardesc(d), "a")

            rename_vars!(d, ["p", "q"])
            @test vardesc(d, "p") == "Alpha variable"
            @test vardesc(d, "q") == "Beta variable"
        end

        @testset "CrossSectionData desc/vardesc" begin
            vd = Dict("inc" => "Income", "age" => "Age in years")
            d = CrossSectionData(randn(30, 2);
                varnames=["inc", "age"], desc="Survey data", vardesc=vd)
            @test desc(d) == "Survey data"
            @test vardesc(d, "inc") == "Income"
            set_desc!(d, "Updated survey")
            @test desc(d) == "Updated survey"
        end

        @testset "desc in show output" begin
            d = TimeSeriesData(randn(50, 2);
                varnames=["GDP", "CPI"], desc="Quarterly US data")
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test occursin("Quarterly US data", s)
        end

        @testset "no desc line when empty" begin
            d = TimeSeriesData(randn(50, 2); varnames=["GDP", "CPI"])
            io = IOBuffer()
            show(io, d)
            s = String(take!(io))
            @test !occursin("\n", s)  # single line, no desc
        end

        @testset "Vector constructor with desc" begin
            d = TimeSeriesData(randn(50); varname="GDP", desc="GDP series",
                vardesc=Dict("GDP" => "Real GDP"))
            @test desc(d) == "GDP series"
            @test vardesc(d, "GDP") == "Real GDP"
        end

        @testset "DataFrame constructor with desc" begin
            df = DataFrame(x=randn(20), y=randn(20))
            d = TimeSeriesData(df; desc="From DataFrame",
                vardesc=Dict("x" => "X col", "y" => "Y col"))
            @test desc(d) == "From DataFrame"
            @test vardesc(d, "x") == "X col"
        end
    end

    # =========================================================================
    # 12. Example datasets (FRED-MD, FRED-QD)
    # =========================================================================
    @testset "Example datasets" begin
        @test_throws ArgumentError load_example(:nonexistent)

        @testset "FRED-MD" begin
            md = load_example(:fred_md)
            @test md isa TimeSeriesData{Float64}
            @test nobs(md) == 804
            @test nvars(md) == 126
            @test frequency(md) == Monthly
            @test desc(md) == "FRED-MD Monthly Database, January 2026 Vintage (McCracken & Ng 2016)"
            @test length(md.tcode) == 126
            @test all(t -> 1 <= t <= 7, md.tcode)

            # Variable descriptions
            @test vardesc(md, "INDPRO") == "IP Index"
            @test vardesc(md, "RPI") == "Real Personal Income"
            @test length(vardesc(md)) == 126

            # Source refs
            @test md.source_refs == [:mccracken_ng2016]

            # refs() works
            io = IOBuffer()
            refs(io, md)
            s = String(take!(io))
            @test occursin("McCracken", s)
            @test occursin("FRED-MD", s)

            # refs(:fred_md) symbol dispatch
            io2 = IOBuffer()
            refs(io2, :fred_md)
            s2 = String(take!(io2))
            @test s == s2

            # Data is numeric and not all NaN
            @test !all(isnan, md.data[:, 1])
            @test size(md.data) == (804, 126)

            # Subsetting preserves source_refs
            sub = md[:, ["INDPRO", "UNRATE"]]
            @test sub.source_refs == [:mccracken_ng2016]
            @test nvars(sub) == 2
        end

        @testset "FRED-QD" begin
            qd = load_example(:fred_qd)
            @test qd isa TimeSeriesData{Float64}
            @test nobs(qd) == 268
            @test nvars(qd) == 245
            @test frequency(qd) == Quarterly
            @test occursin("FRED-QD", desc(qd))
            @test all(t -> 1 <= t <= 7, qd.tcode)

            # Variable descriptions
            @test occursin("Gross Domestic Product", vardesc(qd, "GDPC1"))
            @test length(vardesc(qd)) == 245

            # Source refs
            @test qd.source_refs == [:mccracken_ng2020]

            # refs() works
            io = IOBuffer()
            refs(io, qd)
            s = String(take!(io))
            @test occursin("McCracken", s)
            @test occursin("FRED-QD", s)

            # Data dimensions
            @test size(qd.data) == (268, 245)
        end

        @testset "Penn World Table" begin
            pwt = load_example(:pwt)
            @test pwt isa PanelData{Float64}
            @test nobs(pwt) == 2812  # 38 countries × 74 years
            @test nvars(pwt) == 42
            @test ngroups(pwt) == 38
            @test isbalanced(pwt)
            @test frequency(pwt) == Yearly
            @test occursin("Penn World Table", desc(pwt))

            # Country groups
            g = groups(pwt)
            @test length(g) == 38
            @test "AUS" ∈ g
            @test "USA" ∈ g
            @test "JPN" ∈ g
            @test "DEU" ∈ g

            # Variable names
            vn = varnames(pwt)
            @test "rgdpna" ∈ vn
            @test "pop" ∈ vn
            @test "emp" ∈ vn
            @test "hc" ∈ vn
            @test "avh" ∈ vn
            @test "xr" ∈ vn

            # Variable descriptions
            @test occursin("hours worked", vardesc(pwt, "avh"))
            @test occursin("GDP", vardesc(pwt, "rgdpna"))
            @test length(vardesc(pwt)) == 42

            # Source refs
            @test pwt.source_refs == [:feenstra_etal2015]

            # refs() works
            io = IOBuffer()
            refs(io, pwt)
            s = String(take!(io))
            @test occursin("Feenstra", s)
            @test occursin("Penn World Table", s)

            # refs(:pwt) symbol dispatch
            io2 = IOBuffer()
            refs(io2, :pwt)
            s2 = String(take!(io2))
            @test s == s2

            # Data dimensions
            @test size(pwt.data) == (2812, 42)

            # Extract single country as TimeSeriesData
            usa = group_data(pwt, "USA")
            @test usa isa TimeSeriesData{Float64}
            @test nobs(usa) == 74
            @test nvars(usa) == 42
            @test usa.time_index[1] == 1950
            @test usa.time_index[end] == 2023
            @test usa.source_refs == [:feenstra_etal2015]

            # Data is numeric and not all NaN
            rgdpna_idx = findfirst(==("rgdpna"), varnames(pwt))
            @test !all(isnan, pwt.data[:, rgdpna_idx])

            # USA GDP should be positive
            usa_rgdpna = usa[:, "rgdpna"]
            @test all(x -> !isnan(x) ? x > 0 : true, usa_rgdpna)

            # Extract by index
            aus = group_data(pwt, 1)
            @test aus isa TimeSeriesData{Float64}
            @test nobs(aus) == 74
        end
    end

    # =========================================================================
    # 13. apply_filter
    # =========================================================================
    @testset "apply_filter" begin

        # --- TimeSeriesData: single symbol for all variables ---
        @testset "Single symbol all vars — HP cycle" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])
            d_hp = apply_filter(d, :hp; component=:cycle)
            @test d_hp isa TimeSeriesData
            @test nobs(d_hp) == 200  # HP preserves length
            @test nvars(d_hp) == 3
            @test varnames(d_hp) == ["GDP", "CPI", "FFR"]
            # Cycle should have near-zero mean relative to original
            @test abs(mean(d_hp.data[:, 1])) < 30.0
        end

        @testset "Single symbol all vars — HP trend" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1))
            d_trend = apply_filter(d, :hp; component=:trend)
            @test nobs(d_trend) == 200
            @test nvars(d_trend) == 3
        end

        @testset "Single symbol — Hamilton" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1))
            d_ham = apply_filter(d, :hamilton; component=:cycle)
            # Hamilton drops h+p-1 obs from start
            @test nobs(d_ham) < 200
            @test nobs(d_ham) > 150
            @test nvars(d_ham) == 3
        end

        @testset "Single symbol — Baxter-King" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1))
            d_bk = apply_filter(d, :bk; component=:cycle)
            @test nobs(d_bk) < 200  # BK loses 2K obs
            @test nvars(d_bk) == 3
        end

        @testset "Single symbol — Boosted HP" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1))
            d_bhp = apply_filter(d, :boosted_hp; component=:cycle)
            @test nobs(d_bhp) == 200
            @test nvars(d_bhp) == 3
        end

        # --- Per-variable symbol specs ---
        @testset "Per-variable symbols" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])
            d2 = apply_filter(d, [:hp, :hp, :hp]; component=:cycle)
            @test nobs(d2) == 200
            @test nvars(d2) == 3
        end

        @testset "Per-variable with nothing pass-through" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])
            d2 = apply_filter(d, [:hp, nothing, :hp]; component=:cycle)
            @test nobs(d2) == 200
            @test nvars(d2) == 3
            # Pass-through column should be unchanged
            @test d2.data[:, 2] ≈ d.data[:, 2]
        end

        @testset "All nothing — identity" begin
            Random.seed!(42)
            d = TimeSeriesData(randn(100, 2))
            d2 = apply_filter(d, [nothing, nothing])
            @test d2.data ≈ d.data
            @test nobs(d2) == 100
        end

        # --- Tuple component overrides ---
        @testset "Tuple per-variable overrides" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])
            d2 = apply_filter(d, [(:hp, :trend), (:hp, :cycle), nothing])
            @test nobs(d2) == 200
            @test nvars(d2) == 3
        end

        # --- vars keyword ---
        @testset "vars keyword with String" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1); varnames=["GDP","CPI","FFR"])
            d2 = apply_filter(d, :hp; vars=["GDP", "CPI"], component=:cycle)
            @test nobs(d2) == 200
            # FFR should be pass-through
            @test d2.data[:, 3] ≈ d.data[:, 3]
        end

        @testset "vars keyword with Int" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 3), dims=1))
            d2 = apply_filter(d, :hp; vars=[1, 3], component=:cycle)
            @test nobs(d2) == 200
            # Column 2 should be pass-through
            @test d2.data[:, 2] ≈ d.data[:, 2]
        end

        @testset "vars keyword invalid var" begin
            d = TimeSeriesData(randn(50, 2); varnames=["a", "b"])
            @test_throws ArgumentError apply_filter(d, :hp; vars=["nonexistent"])
        end

        @testset "vars keyword invalid index" begin
            d = TimeSeriesData(randn(50, 2))
            @test_throws BoundsError apply_filter(d, :hp; vars=[0])
            @test_throws BoundsError apply_filter(d, :hp; vars=[3])
        end

        # --- Pre-computed filter results ---
        @testset "Pre-computed filter result" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1))
            r1 = hp_filter(d.data[:, 1])
            d2 = apply_filter(d, [r1, :hp]; component=:cycle)
            @test nobs(d2) == 200
            @test nvars(d2) == 2
        end

        @testset "Pre-computed Hamilton result" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1))
            r1 = hamilton_filter(d.data[:, 1])
            d2 = apply_filter(d, [r1, nothing])
            # Hamilton trims, nothing doesn't — common range from Hamilton
            @test nobs(d2) < 200
        end

        # --- Mixed specs with trimming ---
        @testset "Mixed HP + Hamilton — common range trimming" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1))
            d2 = apply_filter(d, [:hp, :hamilton]; component=:cycle)
            # Should trim to Hamilton's valid range (shorter)
            r_ham = hamilton_filter(d.data[:, 2])
            @test nobs(d2) == length(r_ham.valid_range)
        end

        @testset "Mixed HP + BK — common range trimming" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1))
            d2 = apply_filter(d, [:hp, :bk]; component=:cycle)
            r_bk = baxter_king(d.data[:, 2])
            @test nobs(d2) == length(r_bk.valid_range)
        end

        # --- Metadata propagation ---
        @testset "Metadata propagation" begin
            Random.seed!(42)
            vd = Dict("GDP" => "Real GDP", "CPI" => "Consumer prices")
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1);
                varnames=["GDP", "CPI"], frequency=Quarterly,
                desc="Test data", vardesc=vd,
                source_refs=[:mccracken_ng2016])
            d2 = apply_filter(d, :hp; component=:cycle)
            @test desc(d2) == "Test data"
            @test vardesc(d2, "GDP") == "Real GDP"
            @test vardesc(d2, "CPI") == "Consumer prices"
            @test d2.source_refs == [:mccracken_ng2016]
            @test frequency(d2) == Quarterly
            @test varnames(d2) == ["GDP", "CPI"]
        end

        @testset "Time index trimmed correctly" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1);
                time_index=collect(1:200))
            d2 = apply_filter(d, :hamilton; component=:cycle)
            # Time index should start after Hamilton's initial obs loss
            @test d2.time_index[1] > 1
            @test d2.time_index[end] == 200
            @test length(d2.time_index) == nobs(d2)
        end

        # --- Single variable ---
        @testset "Single variable" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200)); varname="GDP")
            d2 = apply_filter(d, :hp; component=:cycle)
            @test nobs(d2) == 200
            @test nvars(d2) == 1
        end

        # --- Error cases ---
        @testset "Invalid filter symbol" begin
            d = TimeSeriesData(randn(50, 2))
            @test_throws ArgumentError apply_filter(d, :invalid_filter)
        end

        @testset "Mismatched specs length" begin
            d = TimeSeriesData(randn(50, 3))
            @test_throws ArgumentError apply_filter(d, [:hp, :hp])  # 2 specs for 3 vars
        end

        @testset "Invalid component" begin
            d = TimeSeriesData(cumsum(randn(50, 1), dims=1))
            @test_throws ArgumentError apply_filter(d, [:hp]; component=:invalid)
        end

        # --- kwargs forwarding ---
        @testset "kwargs forwarded to filter — HP lambda" begin
            Random.seed!(42)
            d = TimeSeriesData(cumsum(randn(200, 2), dims=1))
            d_default = apply_filter(d, :hp; component=:cycle)
            d_smooth = apply_filter(d, :hp; component=:cycle, lambda=100.0)
            # Different lambda should produce different cycles
            @test !(d_default.data[:, 1] ≈ d_smooth.data[:, 1])
        end

        # --- PanelData ---
        @testset "PanelData — single symbol" begin
            Random.seed!(42)
            df = DataFrame(
                id=repeat(1:3, inner=100),
                t=repeat(1:100, 3),
                x=cumsum(randn(300)),
                y=cumsum(randn(300)))
            pd = xtset(df, :id, :t)
            pd_hp = apply_filter(pd, :hp; component=:cycle)
            @test pd_hp isa PanelData
            @test ngroups(pd_hp) == 3
            @test nvars(pd_hp) == 2
            # Each group has 100 obs (HP preserves length), total = 300
            @test nobs(pd_hp) == 300
            @test isbalanced(pd_hp)
        end

        @testset "PanelData — Hamilton (shorter)" begin
            Random.seed!(42)
            df = DataFrame(
                id=repeat(1:2, inner=100),
                t=repeat(1:100, 2),
                x=cumsum(randn(200)))
            pd = xtset(df, :id, :t)
            pd_ham = apply_filter(pd, :hamilton; component=:cycle)
            @test pd_ham isa PanelData
            @test nobs(pd_ham) < 200  # each group lost obs
            @test ngroups(pd_ham) == 2
        end

        @testset "PanelData — vars keyword" begin
            Random.seed!(42)
            df = DataFrame(
                id=repeat(1:2, inner=100),
                t=repeat(1:100, 2),
                x=cumsum(randn(200)),
                y=randn(200))
            pd = xtset(df, :id, :t)
            pd2 = apply_filter(pd, :hp; vars=["x"], component=:cycle)
            @test nvars(pd2) == 2
            @test nobs(pd2) == 200  # HP preserves length
        end

        @testset "PanelData — per-variable specs vector" begin
            Random.seed!(42)
            df = DataFrame(
                id=repeat(1:2, inner=100),
                t=repeat(1:100, 2),
                x=cumsum(randn(200)),
                y=cumsum(randn(200)))
            pd = xtset(df, :id, :t)
            pd2 = apply_filter(pd, [:hp, nothing]; component=:cycle)
            @test nvars(pd2) == 2
            @test nobs(pd2) == 200
        end

        @testset "PanelData metadata propagation" begin
            Random.seed!(42)
            df = DataFrame(
                id=repeat(1:2, inner=50),
                t=repeat(1:50, 2),
                x=cumsum(randn(100)))
            pd = xtset(df, :id, :t; desc="Test panel",
                       vardesc=Dict("x" => "X variable"))
            pd2 = apply_filter(pd, :hp; component=:cycle)
            @test desc(pd2) == "Test panel"
            @test vardesc(pd2, "x") == "X variable"
            @test groups(pd2) == ["1", "2"]
        end
    end

    # =========================================================================
    # 14. source_refs field
    # =========================================================================
    @testset "source_refs" begin
        # Default is empty
        d = TimeSeriesData(randn(50, 3))
        @test d.source_refs == Symbol[]

        # Set at construction
        d2 = TimeSeriesData(randn(50, 2);
            source_refs=[:mccracken_ng2016])
        @test d2.source_refs == [:mccracken_ng2016]

        # Propagation through apply_tcode
        d3 = TimeSeriesData(rand(50, 2) .+ 1.0;
            source_refs=[:mccracken_ng2016])
        d3t = apply_tcode(d3, [5, 5])
        @test d3t.source_refs == [:mccracken_ng2016]

        # Propagation through fix
        mat = randn(50, 2)
        mat[3, 1] = NaN
        d4 = TimeSeriesData(mat; source_refs=[:mccracken_ng2016])
        d4f = fix(d4)
        @test d4f.source_refs == [:mccracken_ng2016]

        # Propagation through subsetting
        d5 = TimeSeriesData(randn(50, 3);
            varnames=["a", "b", "c"],
            source_refs=[:mccracken_ng2016])
        sub = d5[:, ["a", "b"]]
        @test sub.source_refs == [:mccracken_ng2016]

        # refs() errors on empty source_refs
        @test_throws ArgumentError refs(d)

        # CrossSectionData with source_refs
        cs = CrossSectionData(randn(30, 2);
            source_refs=[:mccracken_ng2016])
        @test cs.source_refs == [:mccracken_ng2016]
    end

    # =================================================================
    # apply_tcode Convenience (Issue #14)
    # =================================================================
    @testset "apply_tcode Convenience" begin
        data = rand(200, 3) .+ 1.0  # strictly positive
        tcodes = [5, 5, 1]
        d = TimeSeriesData(data; varnames=["GDP", "CPI", "FFR"], tcode=tcodes)

        # Convenience method: apply_tcode(d) uses d.tcode
        d2 = apply_tcode(d)
        @test d2 isa TimeSeriesData
        @test d2.n_vars == 3
        @test d2.T_obs < d.T_obs  # rows lost from differencing

        # Should match explicit call
        d3 = apply_tcode(d, tcodes)
        @test d2.data ≈ d3.data
        @test d2.varnames == d3.varnames
    end

    # =================================================================
    # TimeSeriesData Dates (Issue #19)
    # =================================================================
    @testset "TimeSeriesData Dates" begin
        data = randn(5, 2)
        date_strings = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"]

        d = TimeSeriesData(data; varnames=["x", "y"], dates=date_strings)

        # dates accessor
        @test dates(d) == date_strings
        @test length(dates(d)) == 5

        # Date-based indexing (single date → row vector)
        row = d["2020-03", :]
        @test row isa AbstractVector
        @test row ≈ data[3, :]

        # Date-based indexing (multiple dates)
        sub = d[["2020-02", "2020-03", "2020-04"], :]
        @test sub isa TimeSeriesData
        @test nobs(sub) == 3
        @test sub.data ≈ data[2:4, :]

        # set_dates!
        d2 = TimeSeriesData(randn(3, 2))
        new_dates = ["2021-Q1", "2021-Q2", "2021-Q3"]
        set_dates!(d2, new_dates)
        @test dates(d2) == new_dates

        # set_dates! with wrong length should error
        @test_throws ArgumentError set_dates!(d2, ["a", "b"])
    end

end
