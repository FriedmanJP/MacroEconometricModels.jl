# Miscellaneous coverage tests for MacroEconometricModels.jl
# Targets small coverage gaps in:
#   - src/data/summary_stats.jl (CrossSectionData dispatch, all-NaN, single-obs)
#   - src/filters/boosted_hp.jl (ADF never-reject path, zero-variance IC)
#   - src/arima/estimation.jl (css_mle method, unknown method, Yule-Walker catch, MLE SE catch)
#   - src/bvar/types.jl (size dim=3 error, BVARForecast show with :median)

Random.seed!(9004)

@testset "Misc Coverage Tests" begin

    # =========================================================================
    # 1. src/data/summary_stats.jl — CrossSectionData dispatch
    # =========================================================================
    @testset "describe_data(CrossSectionData)" begin
        X = randn(50, 3)
        cs = CrossSectionData(X; varnames=["x1", "x2", "x3"])
        s = describe_data(cs)
        @test s isa MacroEconometricModels.DataSummary
        @test s.n_vars == 3
        @test s.T_obs == 50
        @test s.frequency == MacroEconometricModels.Other
        @test all(s.n .== 50)
        # Trigger show
        str = sprint(show, s)
        @test occursin("Summary Statistics", str)
    end

    # =========================================================================
    # 2. src/data/summary_stats.jl — all-NaN column
    # =========================================================================
    @testset "describe_data with all-NaN column" begin
        Y = randn(30, 3)
        Y[:, 2] .= NaN  # entire column is NaN
        ts = TimeSeriesData(Y; varnames=["a", "b_nan", "c"])
        s = describe_data(ts)
        @test s.n[2] == 0           # no finite observations
        @test isnan(s.mean[2])
        @test isnan(s.std[2])
        @test isnan(s.min[2])
        @test isnan(s.max[2])
        @test isnan(s.skewness[2])
        @test isnan(s.kurtosis[2])
        # Other columns should be fine
        @test s.n[1] == 30
        @test s.n[3] == 30
    end

    # =========================================================================
    # 3. src/data/summary_stats.jl — single-observation column (nf == 1)
    # =========================================================================
    @testset "describe_data with single-observation column" begin
        Y = fill(NaN, 20, 2)
        Y[5, 1] = 3.14     # only one finite value in column 1
        Y[:, 2] .= randn(20)  # normal column
        ts = TimeSeriesData(Y; varnames=["single_obs", "normal"])
        s = describe_data(ts)
        @test s.n[1] == 1
        @test s.std[1] == 0.0       # nf == 1 -> std = 0.0
        @test s.mean[1] == 3.14
        @test s.min[1] == 3.14
        @test s.max[1] == 3.14
        @test s.skewness[1] == 0.0  # nf <= 2 -> skewness = 0.0
        @test s.kurtosis[1] == 0.0  # nf <= 2 -> kurtosis = 0.0
    end

    # =========================================================================
    # 4. src/data/summary_stats.jl — two observations (nf == 2, s > 0)
    # =========================================================================
    @testset "describe_data with two-observation column" begin
        Y = fill(NaN, 20, 2)
        Y[3, 1] = 1.0
        Y[7, 1] = 5.0  # two finite values, std > 0 but nf <= 2
        Y[:, 2] .= randn(20)
        ts = TimeSeriesData(Y; varnames=["two_obs", "normal"])
        s = describe_data(ts)
        @test s.n[1] == 2
        @test s.skewness[1] == 0.0  # nf <= 2 -> 0.0
        @test s.kurtosis[1] == 0.0
    end

    # =========================================================================
    # 5. src/filters/boosted_hp.jl — ADF stopping where ADF never rejects
    # =========================================================================
    @testset "boosted_hp ADF never rejects" begin
        # A strong random walk: ADF should not reject within very few iterations
        rng = Random.MersenneTwister(42)
        y = cumsum(randn(rng, 200))  # strong unit root
        result = boosted_hp(y; stopping=:ADF, max_iter=3, sig_p=0.001)
        @test result isa MacroEconometricModels.BoostedHPResult
        # When ADF never rejects, all p-values >= sig_p, so it uses last iteration
        @test result.iterations == 3
        @test length(result.adf_pvalues) >= 1
        @test all(p -> p >= 0.001, result.adf_pvalues)
    end

    # =========================================================================
    # 6. src/filters/boosted_hp.jl — zero-variance edge case in _phillips_shi_ic
    # =========================================================================
    @testset "boosted_hp zero variance in _phillips_shi_ic" begin
        # A constant series: var(cycle) = 0 after HP filter
        y = fill(5.0, 100)
        # BIC stopping: var_cyc_hp will be 0, triggering var_hp <= 0 -> Inf path
        result = boosted_hp(y; stopping=:BIC)
        @test result isa MacroEconometricModels.BoostedHPResult
        # With constant input, HP filter gives trend = input, cycle = 0
        @test all(abs.(result.cycle) .< 1e-10)
    end

    # =========================================================================
    # 7. src/arima/estimation.jl — estimate_arma with :css_mle method
    # =========================================================================
    @testset "estimate_arma css_mle method" begin
        rng = Random.MersenneTwister(1234)
        y = randn(rng, 200)
        m = estimate_arma(y, 1, 1; method=:css_mle)
        @test m isa ARMAModel
        @test m.method == :css_mle
        @test length(m.phi) == 1
        @test length(m.theta) == 1
        @test isfinite(m.loglik)
    end

    # =========================================================================
    # 8. src/arima/estimation.jl — estimate_arma with unknown method
    # =========================================================================
    @testset "estimate_arma unknown method" begin
        y = randn(100)
        @test_throws ArgumentError estimate_arma(y, 1, 1; method=:unknown)
    end

    # =========================================================================
    # 9. src/arima/estimation.jl — _yule_walker catch path (singular system)
    # =========================================================================
    @testset "_yule_walker singular fallback" begin
        # A constant series has zero autocorrelation -> singular Toeplitz matrix
        y_const = fill(3.0, 100)
        phi = MacroEconometricModels._yule_walker(y_const, 2)
        @test length(phi) == 2
        # Should return fallback small coefficients
        @test all(isfinite, phi)
    end

    # =========================================================================
    # 10. src/arima/estimation.jl — _arima_mle_stderror catch path
    # =========================================================================
    @testset "_arima_mle_stderror Hessian catch" begin
        # Estimate an MA(1) with near-degenerate data to stress the Hessian
        rng = Random.MersenneTwister(777)
        y = randn(rng, 50)
        m = estimate_ma(y, 1; method=:mle)
        # Even if Hessian is well-conditioned, stderror should return finite values
        se = stderror(m)
        @test length(se) == 2  # c + theta
        @test all(isfinite, se)

        # Force the catch path by testing with a very short series
        # where MLE may produce a near-singular Hessian
        y_short = randn(rng, 15)
        m_short = estimate_ma(y_short, 1; method=:css)
        # stderror uses _arima_mle_stderror regardless — test it completes
        se_short = stderror(m_short)
        @test length(se_short) == 2
    end

    # =========================================================================
    # 11. src/bvar/types.jl — size(post, 3) error
    # =========================================================================
    @testset "BVARPosterior size dim=3 error" begin
        Y = randn(100, 3)
        post = estimate_bvar(Y, 2; n_draws=50)
        @test size(post, 1) == 50
        @test length(post) == 50
        @test_throws ErrorException size(post, 3)
    end

    # =========================================================================
    # 12. src/bvar/types.jl — BVARForecast show with :median point estimate
    # =========================================================================
    @testset "BVARForecast show with :median" begin
        Y = randn(100, 2)
        post = estimate_bvar(Y, 2; n_draws=100, varnames=["GDP", "INF"])
        fc = forecast(post, 5; point_estimate=:median)
        @test fc isa MacroEconometricModels.BVARForecast
        @test fc.point_estimate == :median
        str = sprint(show, fc)
        @test occursin("Post. Median", str)
        @test occursin("GDP", str)
        @test occursin("INF", str)
    end

    # =========================================================================
    # 13. src/bvar/types.jl — BVARForecast show with :mean point estimate
    # =========================================================================
    @testset "BVARForecast show with :mean" begin
        Y = randn(100, 2)
        post = estimate_bvar(Y, 2; n_draws=100, varnames=["X1", "X2"])
        fc = forecast(post, 3; point_estimate=:mean)
        @test fc.point_estimate == :mean
        str = sprint(show, fc)
        @test occursin("Post. Mean", str)
    end

end
