using Test
using MacroEconometricModels
using Random
using DataFrames
using Statistics

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
            mat = hcat(ones(20), randn(20))
            d = TimeSeriesData(mat; varnames=["const_col", "good_col"])
            d_fixed = fix(d)
            @test nvars(d_fixed) == 1
            @test varnames(d_fixed) == ["good_col"]
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
            df = DataFrame(id=repeat(1:2, inner=30), t=repeat(1:30, 2),
                          x=randn(60), y=randn(60))
            pd = xtset(df, :id, :t)
            io = IOBuffer()
            # Redirect stdout to capture panel_summary output
            s = describe_data(pd)
            @test s isa DataSummary
            @test s.n_vars == 2
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
            mat = hcat(ones(20), 2.0 .* ones(20))
            d = TimeSeriesData(mat)
            @test_throws ArgumentError fix(d)
        end

        @testset "Transform too short" begin
            d = TimeSeriesData(rand(2, 2) .+ 1.0)
            @test_throws ArgumentError apply_tcode(d, [3, 3])  # needs 2 rows lost, leaves 0
            d3 = TimeSeriesData(rand(1, 1) .+ 1.0)
            @test_throws ArgumentError apply_tcode(d3, [3])
        end

        @testset "Interpolate all NaN column" begin
            mat = fill(NaN, 20, 1)
            d = TimeSeriesData(mat)
            # All NaN → constant after interpolation → error
            @test_throws ArgumentError fix(d; method=:interpolate)
            @test_throws ArgumentError fix(d; method=:mean)
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
    # 11. Examples stub
    # =========================================================================
    @testset "Examples" begin
        @test_throws ArgumentError load_example(:sw2001)
        @test_throws ArgumentError load_example(:nonexistent)
    end

end
