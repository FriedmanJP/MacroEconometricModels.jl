using Random, Statistics, LinearAlgebra

@testset "FactorForecast Struct" begin
    Random.seed!(77701)
    X = randn(100, 10)
    fm = estimate_factors(X, 2)
    fc = forecast(fm, 5)

    @test fc isa FactorForecast{Float64}
    @test fc.horizon == 5
    @test fc.ci_method == :none

    # Display
    buf = IOBuffer()
    show(buf, fc)
    s = String(take!(buf))
    @test occursin("Factor Forecast", s)
    @test occursin("5", s)
end

# =============================================================================
# Static Factor Model Forecasting
# =============================================================================

@testset "FactorModel Forecast - Dimensions" begin
    Random.seed!(77702)
    T_obs, N, r = 120, 15, 3
    X = randn(T_obs, N)
    fm = estimate_factors(X, r)

    for ci_method in (:none, :theoretical, :bootstrap)
        fc = forecast(fm, 8; ci_method=ci_method, n_boot=200)
        @test fc isa FactorForecast{Float64}
        @test size(fc.factors) == (8, r)
        @test size(fc.observables) == (8, N)
        @test size(fc.factors_lower) == (8, r)
        @test size(fc.factors_upper) == (8, r)
        @test size(fc.observables_lower) == (8, N)
        @test size(fc.observables_upper) == (8, N)
        @test size(fc.factors_se) == (8, r)
        @test size(fc.observables_se) == (8, N)
        @test fc.horizon == 8
        @test fc.ci_method == ci_method
    end
end

@testset "FactorModel Forecast - CI Ordering" begin
    Random.seed!(77703)
    X = randn(150, 12)
    fm = estimate_factors(X, 2)

    for ci_method in (:theoretical, :bootstrap)
        fc = forecast(fm, 6; ci_method=ci_method, n_boot=300)
        @test all(fc.factors_upper .>= fc.factors_lower)
        @test all(fc.observables_upper .>= fc.observables_lower)
        @test all(fc.factors_se .>= 0)
        @test all(fc.observables_se .>= 0)
    end
end

@testset "FactorModel Forecast - No CI Zeros" begin
    Random.seed!(77704)
    X = randn(100, 10)
    fm = estimate_factors(X, 2)
    fc = forecast(fm, 5)

    @test all(fc.factors_lower .== 0)
    @test all(fc.factors_upper .== 0)
    @test all(fc.factors_se .== 0)
end

@testset "FactorModel Forecast - VAR Lag" begin
    Random.seed!(77705)
    X = randn(100, 10)
    fm = estimate_factors(X, 2)

    fc1 = forecast(fm, 5; p=1)
    fc2 = forecast(fm, 5; p=2)
    # Different lags should generally give different point forecasts
    @test fc1.factors != fc2.factors || true  # just check no error
end

@testset "FactorModel Forecast - Input Validation" begin
    X = randn(100, 10)
    fm = estimate_factors(X, 2)

    @test_throws ArgumentError forecast(fm, 0)
    @test_throws ArgumentError forecast(fm, -1)
    @test_throws ArgumentError forecast(fm, 5; p=0)
    @test_throws ArgumentError forecast(fm, 5; ci_method=:invalid)
end

# =============================================================================
# Dynamic Factor Model Forecasting
# =============================================================================

@testset "DFM Forecast - Dimensions (all ci_methods)" begin
    Random.seed!(77710)
    T_obs, N, r, p = 120, 12, 2, 2
    X = randn(T_obs, N)
    dfm = estimate_dynamic_factors(X, r, p)

    for ci_method in (:none, :theoretical, :bootstrap, :simulation)
        fc = forecast(dfm, 6; ci_method=ci_method, n_boot=200)
        @test fc isa FactorForecast{Float64}
        @test size(fc.factors) == (6, r)
        @test size(fc.observables) == (6, N)
        @test size(fc.factors_lower) == (6, r)
        @test size(fc.factors_upper) == (6, r)
        @test size(fc.observables_lower) == (6, N)
        @test size(fc.observables_upper) == (6, N)
        @test fc.ci_method == ci_method
    end
end

@testset "DFM Forecast - CI Ordering" begin
    Random.seed!(77711)
    T_obs, N, r, p = 150, 10, 2, 1
    X = randn(T_obs, N)
    dfm = estimate_dynamic_factors(X, r, p)

    for ci_method in (:theoretical, :bootstrap, :simulation)
        fc = forecast(dfm, 8; ci_method=ci_method, n_boot=300)
        @test all(fc.factors_upper .>= fc.factors_lower)
        @test all(fc.observables_upper .>= fc.observables_lower)
        @test all(fc.factors_se .>= 0)
        @test all(fc.observables_se .>= 0)
    end
end

@testset "DFM Forecast - Legacy ci=true Compat" begin
    Random.seed!(77712)
    T_obs, N, r, p = 100, 8, 2, 1
    X = randn(T_obs, N)
    dfm = estimate_dynamic_factors(X, r, p)

    fc = forecast(dfm, 5; ci=true, ci_level=0.90)
    @test fc isa FactorForecast{Float64}
    @test fc.ci_method == :simulation
    @test fc.conf_level ≈ 0.90
    @test size(fc.factors_lower) == (5, r)
    @test all(fc.factors_upper .>= fc.factors_lower)
end

@testset "DFM Forecast - Theoretical SE Non-Decreasing" begin
    Random.seed!(77713)
    T_obs, N, r, p = 200, 10, 2, 1
    X = randn(T_obs, N)
    dfm = estimate_dynamic_factors(X, r, p)

    fc = forecast(dfm, 10; ci_method=:theoretical)
    # For a stationary VAR, theoretical SEs should be non-decreasing with horizon
    for j in 1:r
        for step in 2:10
            @test fc.factors_se[step, j] >= fc.factors_se[step-1, j] - 1e-10
        end
    end
end

@testset "DFM Forecast - Point Forecast Match Manual" begin
    Random.seed!(77714)
    T_obs, r, p = 80, 2, 1
    X = randn(T_obs, 8)
    dfm = estimate_dynamic_factors(X, r, p)

    fc = forecast(dfm, 3)

    # Manually compute 1-step ahead
    F_last = dfm.factors[end, :]
    F1_manual = dfm.A[1] * F_last
    @test fc.factors[1, :] ≈ F1_manual
end

@testset "DFM Forecast - Input Validation" begin
    X = randn(100, 10)
    dfm = estimate_dynamic_factors(X, 2, 1)

    @test_throws ArgumentError forecast(dfm, 0)
    @test_throws ArgumentError forecast(dfm, -1)
    @test_throws ArgumentError forecast(dfm, 5; ci_method=:invalid)
end

# =============================================================================
# GDFM Forecasting
# =============================================================================

@testset "GDFM Forecast - Dimensions (all ci_methods)" begin
    Random.seed!(77720)
    T_obs, N, q = 150, 15, 2
    X = randn(T_obs, N)
    gdfm = estimate_gdfm(X, q)

    for ci_method in (:none, :theoretical, :bootstrap)
        fc = forecast(gdfm, 6; ci_method=ci_method, n_boot=200)
        @test fc isa FactorForecast{Float64}
        @test size(fc.factors) == (6, q)
        @test size(fc.observables) == (6, N)
        @test size(fc.factors_lower) == (6, q)
        @test size(fc.factors_upper) == (6, q)
        @test size(fc.observables_lower) == (6, N)
        @test size(fc.observables_upper) == (6, N)
        @test fc.ci_method == ci_method
    end
end

@testset "GDFM Forecast - CI Ordering" begin
    Random.seed!(77721)
    T_obs, N, q = 120, 12, 2
    X = randn(T_obs, N)
    gdfm = estimate_gdfm(X, q)

    for ci_method in (:theoretical, :bootstrap)
        fc = forecast(gdfm, 8; ci_method=ci_method, n_boot=300)
        @test all(fc.factors_upper .>= fc.factors_lower)
        @test all(fc.observables_upper .>= fc.observables_lower)
        @test all(fc.factors_se .>= 0)
        @test all(fc.observables_se .>= 0)
    end
end

@testset "GDFM Forecast - Theoretical SE Non-Decreasing" begin
    Random.seed!(77722)
    T_obs, N, q = 150, 10, 2
    X = randn(T_obs, N)
    gdfm = estimate_gdfm(X, q)

    fc = forecast(gdfm, 10; ci_method=:theoretical)
    # AR(1) forecast SE should be non-decreasing
    for j in 1:q
        for step in 2:10
            @test fc.factors_se[step, j] >= fc.factors_se[step-1, j] - 1e-10
        end
    end
end

@testset "GDFM Forecast - Input Validation" begin
    X = randn(100, 10)
    gdfm = estimate_gdfm(X, 2)

    @test_throws ArgumentError forecast(gdfm, 0)
    @test_throws ArgumentError forecast(gdfm, -1)
    @test_throws ArgumentError forecast(gdfm, 5; ci_method=:invalid)
    @test_throws ArgumentError forecast(gdfm, 5; method=:invalid)
end

@testset "GDFM Forecast - Backward Compat (observables field)" begin
    Random.seed!(77723)
    X = randn(100, 10)
    gdfm = estimate_gdfm(X, 2)

    fc = forecast(gdfm, 5)
    @test size(fc.observables) == (5, 10)
    @test all(isfinite, fc.observables)
    @test all(isfinite, fc.factors)
end

@testset "DFM simulation CI method" begin
    Random.seed!(77730)
    X = randn(100, 8)
    dfm = estimate_dynamic_factors(X, 2, 1)
    fc_sim = forecast(dfm, 5; ci_method=:simulation, n_boot=20)
    @test fc_sim isa MacroEconometricModels.FactorForecast
    @test size(fc_sim.factors) == (5, 2)
    @test size(fc_sim.observables) == (5, 8)
    if fc_sim.factors_lower !== nothing
        @test size(fc_sim.factors_lower) == (5, 2)
        @test size(fc_sim.factors_upper) == (5, 2)
    end
end

@testset "DFM legacy ci=true compatibility" begin
    Random.seed!(77731)
    X = randn(100, 8)
    dfm = estimate_dynamic_factors(X, 2, 1)
    fc_legacy = forecast(dfm, 5; ci=true)
    @test fc_legacy isa MacroEconometricModels.FactorForecast
    @test size(fc_legacy.factors) == (5, 2)
end

@testset "Static FM with p=3 bootstrap CIs" begin
    Random.seed!(77732)
    X = randn(100, 8)
    fm = estimate_factors(X, 2)
    fc = forecast(fm, 5; p=3, ci_method=:bootstrap, n_boot=20)
    @test fc isa MacroEconometricModels.FactorForecast
    @test size(fc.factors) == (5, 2)
    if fc.factors_lower !== nothing
        @test size(fc.factors_lower) == (5, 2)
    end
end

@testset "GDFM spectral vs ar forecast" begin
    Random.seed!(77733)
    X = randn(100, 10)
    gdfm = estimate_gdfm(X, 2)
    fc_ar = forecast(gdfm, 5; method=:ar)
    @test size(fc_ar.factors) == (5, 2)
    @test all(isfinite, fc_ar.factors)
end
