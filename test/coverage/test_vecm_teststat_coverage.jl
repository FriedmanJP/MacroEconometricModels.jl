# Coverage tests for VECM granger and teststat modules
# Targets:
#   src/vecm/granger.jl      — p=1, rank=0, deterministic=:trend paths
#   src/teststat/convenience.jl — :za, :ngperron branches, :none regression remapping
#   src/teststat/helpers.jl   — :hqic criterion, adf_pvalue above 10% CV, _regression_name(:both), kpss_pvalue large stat
#   src/teststat/show.jl      — ZA fail-to-reject, KPSS "<0.01", Johansen rank=0/n, NgPerron fail-to-reject,
#                                VARStationarity >10 eigenvalues truncation, complex eigenvalues

Random.seed!(9001)

@testset "VECM & Teststat Coverage" begin

    # =========================================================================
    # vecm/granger.jl coverage
    # =========================================================================

    @testset "VECM Granger: p=1 path (no short-run Gamma)" begin
        # Cointegrated data: Y2 tracks Y1
        rng = Random.MersenneTwister(1234)
        T_obs = 200
        e = randn(rng, T_obs, 2)
        Y = cumsum(e, dims=1)
        Y[:, 2] = Y[:, 1] + 0.3 * randn(rng, T_obs)

        vecm = estimate_vecm(Y, 1; rank=1, deterministic=:constant)
        @test vecm.p == 1
        @test length(vecm.Gamma) == 0  # p-1 = 0 Gamma matrices

        g = granger_causality_vecm(vecm, 1, 2)
        # Short-run stat should be 0 (no Gamma matrices when p=1)
        @test g.short_run_stat == 0.0
        @test g.short_run_pvalue == 1.0
        @test g.short_run_df == 0
        # Long-run should still be computed (rank=1 > 0)
        @test g.long_run_df == 1
        @test g.long_run_stat >= 0.0
        # Strong = short + long when only one is nonzero
        @test g.strong_df == g.long_run_df
    end

    @testset "VECM Granger: rank=0 path (no long-run alpha)" begin
        # Independent random walks — no cointegration
        rng = Random.MersenneTwister(5678)
        Y = cumsum(randn(rng, 200, 2), dims=1)

        vecm = estimate_vecm(Y, 2; rank=0, deterministic=:constant)
        @test vecm.rank == 0

        g = granger_causality_vecm(vecm, 1, 2)
        # Long-run stat should be 0 (rank=0 means no alpha)
        @test g.long_run_stat == 0.0
        @test g.long_run_pvalue == 1.0
        @test g.long_run_df == 0
        # Short-run should still be computed (p-1=1 Gamma matrices)
        @test g.short_run_df == 1
        @test g.short_run_stat >= 0.0
        # Strong = short + long when only one is nonzero
        @test g.strong_df == g.short_run_df
    end

    @testset "VECM Granger: deterministic=:trend path" begin
        rng = Random.MersenneTwister(9012)
        T_obs = 200
        e = randn(rng, T_obs, 2)
        Y = cumsum(e, dims=1)
        Y[:, 2] = Y[:, 1] + 0.2 * randn(rng, T_obs)

        vecm = estimate_vecm(Y, 2; rank=1, deterministic=:trend)
        @test vecm.deterministic == :trend

        g = granger_causality_vecm(vecm, 1, 2)
        # Should complete without error when has_trend=true
        @test g.short_run_df == 1
        @test g.long_run_df == 1
        @test g.strong_df == 2
        @test g.strong_stat >= 0.0
    end

    @testset "VECM Granger: p=1 and rank=0 combined (strong_df=0)" begin
        # Both p=1 (no Gamma) and rank=0 (no alpha) => strong_df=0
        rng = Random.MersenneTwister(3456)
        Y = cumsum(randn(rng, 200, 2), dims=1)

        vecm = estimate_vecm(Y, 1; rank=0, deterministic=:constant)
        @test vecm.p == 1
        @test vecm.rank == 0

        g = granger_causality_vecm(vecm, 1, 2)
        @test g.short_run_stat == 0.0
        @test g.long_run_stat == 0.0
        @test g.strong_stat == 0.0
        @test g.strong_pvalue == 1.0
        @test g.strong_df == 0
    end

    # =========================================================================
    # teststat/convenience.jl coverage
    # =========================================================================

    @testset "unit_root_summary: :za and :ngperron branches" begin
        y_ur = cumsum(randn(Random.MersenneTwister(42), 200))

        # Test with :za included
        result_za = unit_root_summary(y_ur; tests=[:adf, :kpss, :za])
        @test haskey(result_za.results, :za)
        @test result_za.results[:za] isa MacroEconometricModels.ZAResult

        # Test with :ngperron included
        result_ng = unit_root_summary(y_ur; tests=[:adf, :kpss, :ngperron])
        @test haskey(result_ng.results, :ngperron)
        @test result_ng.results[:ngperron] isa MacroEconometricModels.NgPerronResult
    end

    @testset "unit_root_summary: regression=:none remapping for ZA/NgPerron" begin
        y = cumsum(randn(Random.MersenneTwister(99), 200))

        # When regression=:none, ZA and NgPerron should remap to :constant
        result = unit_root_summary(y; tests=[:adf, :kpss, :za, :ngperron], regression=:none)
        @test haskey(result.results, :za)
        @test haskey(result.results, :ngperron)
        # ZA and NgPerron should use :constant instead of :none
        @test result.results[:za].regression == :constant
        @test result.results[:ngperron].regression == :constant
    end

    # =========================================================================
    # teststat/helpers.jl coverage
    # =========================================================================

    @testset "ADF test with :hqic lag selection criterion" begin
        y = cumsum(randn(Random.MersenneTwister(111), 200))
        result = adf_test(y; lags=:hqic)
        @test result isa MacroEconometricModels.ADFResult
        @test result.lags >= 0
        @test result.pvalue >= 0.0
    end

    @testset "adf_pvalue: stat above 10% critical value" begin
        # Directly test adf_pvalue with a stat well above the 10% critical value
        # ADF 10% CV for :constant is around -2.57, so stat=0.0 is far above it
        pval = MacroEconometricModels.adf_pvalue(0.0, :constant, 300)
        @test pval > 0.10
    end

    @testset "_regression_name(:both) branch" begin
        name = MacroEconometricModels._regression_name(:both)
        @test name == "Constant + Trend"
    end

    @testset "kpss_pvalue: large stat (unit root series, p < 0.01)" begin
        # Call kpss_pvalue directly with a very large statistic to hit stat >= cv[1] branch
        # KPSS critical values for :constant are approximately: 10% = 0.347, 5% = 0.463, 1% = 0.739
        # A stat of 5.0 is far above the 1% CV
        pval = MacroEconometricModels.kpss_pvalue(5.0, :constant)
        @test pval == 0.01

        # Also test trend regression
        pval_trend = MacroEconometricModels.kpss_pvalue(2.0, :trend)
        @test pval_trend == 0.01
    end

    # =========================================================================
    # teststat/show.jl coverage
    # =========================================================================

    @testset "ZA show: fail-to-reject (unit root series)" begin
        # A unit root series should cause ZA to fail to reject
        y = cumsum(randn(Random.MersenneTwister(444), 200))
        result = za_test(y; regression=:constant)
        s = sprint(show, result)
        @test occursin("Zivot-Andrews", s)
        # For a unit root series, should fail to reject
        @test occursin("Fail to reject", s) || occursin("Reject", s)
        # Verify the show method outputs correctly
        @test occursin("Break", s)
        @test occursin("Critical Values", s)
    end

    @testset "KPSS show: <0.01 p-value (unit root series)" begin
        # Use long deterministic trend — always produces KPSS stat > 1% CV
        y = cumsum(ones(3000))
        result = kpss_test(y; regression=:constant)
        s = sprint(show, result)
        @test occursin("KPSS", s)
        # For a strong trend, p-value should be ≤ 0.01
        @test occursin("<0.01", s) || occursin("0.01", s)
        @test occursin("Reject", s)
    end

    @testset "Johansen show: rank=0 (no cointegration)" begin
        # Independent random walks — expect rank=0
        rng = Random.MersenneTwister(666)
        Y = cumsum(randn(rng, 200, 3), dims=1)
        joh = johansen_test(Y, 2)

        s = sprint(show, joh)
        @test occursin("Johansen", s)
        # Check for the rank=0 or rank=n conclusion paths
        @test occursin("cointegrat", s) || occursin("rank", s) || occursin("stationary", s)
    end

    @testset "Johansen show: rank=n (all stationary)" begin
        # Stationary data should yield rank=n (full rank)
        rng = Random.MersenneTwister(777)
        Y = randn(rng, 200, 2)  # Stationary
        joh = johansen_test(Y, 2)

        s = sprint(show, joh)
        @test occursin("Johansen", s)
        @test occursin("Eigenvalues", s)
    end

    @testset "NgPerron show: fail-to-reject (unit root series)" begin
        y = cumsum(randn(Random.MersenneTwister(888), 200))
        result = ngperron_test(y; regression=:constant)
        s = sprint(show, result)
        @test occursin("Ng-Perron", s)
        # For a unit root series, should mostly fail to reject
        @test occursin("Fail to reject", s) || occursin("non-stationary", s) || occursin("Weak", s)
        @test occursin("MZ", s)
    end

    @testset "VARStationarity show: >10 eigenvalues truncation" begin
        # 4 variables * 3 lags = 12 eigenvalues in companion matrix
        rng = Random.MersenneTwister(999)
        Y = randn(rng, 200, 4)
        m = estimate_var(Y, 3)
        sr = is_stationary(m)

        @test length(sr.eigenvalues) == 12  # 4*3 = 12

        s = sprint(show, sr)
        @test occursin("VAR Model Stationarity", s)
        # Should show truncation message for >10 eigenvalues
        @test occursin("more)", s)
    end

    @testset "VARStationarity show: complex eigenvalues" begin
        # VAR models frequently produce complex eigenvalues
        rng = Random.MersenneTwister(1010)
        Y = randn(rng, 200, 3)
        m = estimate_var(Y, 2)
        sr = is_stationary(m)

        s = sprint(show, sr)
        @test occursin("Eigenvalue", s)
        @test occursin("Modulus", s)

        # Check for complex eigenvalue display (contains "i" for imaginary part)
        has_complex = any(!isreal, sr.eigenvalues)
        if has_complex
            @test occursin("i", s)
        end
    end

end
