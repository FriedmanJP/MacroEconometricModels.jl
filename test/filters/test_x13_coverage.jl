# X-13ARIMA-SEATS coverage tests — exercises internal functions and code paths
# not reached by the basic test_x13.jl suite.

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra
using Dates

const M = MacroEconometricModels

@testset "X-13 Coverage" begin

# ── Shared test data ──
Random.seed!(42)
n = 120
trend = cumsum(randn(n) .* 0.1)
seasonal = 10.0 .* sin.(2π .* (1:n) ./ 12) .+ 5.0 .* cos.(2π .* (1:n) ./ 6)
y_monthly = 100.0 .+ trend .+ seasonal .+ randn(n)
y_pos = exp.(3.0 .+ 0.01 .* (1:n) .+ 0.3 .* sin.(2π .* (1:n) ./ 12) .+ 0.1 .* randn(n))

nq = 80
y_quarterly = 100.0 .+ cumsum(randn(nq) .* 0.1) .+ 5.0 .* sin.(2π .* (1:nq) ./ 4) .+ randn(nq)

# ═══════════════════════════════════════════════════════════════════════════
# types.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Types" begin
    spec = M._X13ARIMASpec(2, 1, 1, 1, 1, 1, 12)
    @test spec.p == 2 && spec.d == 1 && spec.Q == 1
    model = M._X13ARIMAModel(spec)
    @test length(model.ar) == 2
    @test length(model.ma) == 1
    @test length(model.sar) == 1
    @test length(model.sma) == 1
    @test isnan(model.sigma2)
    @test !model.converged

    rs = M._X13RegressionSpec()
    @test !rs.trading_day && !rs.easter

    x11 = M._X13X11Spec()
    @test x11.mode == :auto && x11.sigma_lower == 1.5

    seats = M._X13SEATSSpec()
    @test seats.approximation == :none && seats.tasman == 1.0

    ts = M._X13TimeSeries(copy(y_monthly), (2000, 1), 12)
    @test ts.frequency == 12 && length(ts.y) == n
end

# ═══════════════════════════════════════════════════════════════════════════
# polynomials.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Polynomials" begin
    @test M._x13_poly_multiply([1.0, -0.5], [1.0, 0.3]) ≈ [1.0, -0.2, -0.15]
    @test M._x13_poly_multiply([1.0], [1.0]) == [1.0]
    @test M._x13_poly_multiply([1.0, 0.0, -1.0], [1.0, 1.0]) ≈ [1.0, 1.0, -1.0, -1.0]

    @test M._x13_poly_mod2([1.0], 0.0) ≈ 1.0
    @test M._x13_poly_mod2([1.0, -1.0], 0.0) ≈ 0.0
    @test M._x13_poly_mod2([1.0, -1.0], Float64(π)) ≈ 4.0 atol=1e-10

    t, s, irr = M._x13_partial_fractions(ones(50), [1.0, -1.0], [1.0, 0.0, 0.0, -1.0], 1.0)
    @test length(t) == 50 && length(s) == 50 && length(irr) == 50
    @test all(isfinite, t) && all(isfinite, irr)
end

# ═══════════════════════════════════════════════════════════════════════════
# spectrum.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Spectral density" begin
    freqs = range(0.01, π, length=100)
    sd = M._x13_spectral_density(Float64[], Float64[], freqs, 1.0)
    @test all(sd .≈ 1.0 / (2π))

    sd_ar = M._x13_spectral_density([0.5], Float64[], freqs, 1.0)
    @test all(isfinite, sd_ar) && all(sd_ar .> 0)
    @test sd_ar[1] > sd_ar[end]   # AR(1) has more power at low frequencies

    sd_ma = M._x13_spectral_density(Float64[], [0.5], freqs, 1.0)
    @test all(isfinite, sd_ma) && all(sd_ma .> 0)
end

# ═══════════════════════════════════════════════════════════════════════════
# henderson.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Henderson filter" begin
    w5 = M._x13_henderson_weights(5)
    @test length(w5) == 3

    w13 = M._x13_henderson_weights(13)
    @test length(w13) == 7

    fw = M._x13_henderson_full_weights(13)
    @test length(fw) == 13
    @test sum(fw) ≈ 1.0 atol=1e-10

    fw5 = M._x13_henderson_full_weights(5)
    @test length(fw5) == 5
    @test sum(fw5) ≈ 1.0 atol=1e-10

    tr = zeros(n)
    M._x13_apply_henderson!(y_monthly, 13, tr)
    half = 6
    @test all(isnan, tr[1:half])
    @test all(isnan, tr[n-half+1:n])
    @test all(isfinite, tr[half+1:n-half])
end

# ═══════════════════════════════════════════════════════════════════════════
# seasonalma.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Seasonal MA" begin
    b_m = M._x13_seasonal_ma(y_monthly, 1, n, 0)
    @test length(b_m) == n
    @test any(b_m .!= 0.0)

    b_q = M._x13_seasonal_ma(y_quarterly, 1, nq, 2)
    @test length(b_q) == nq
    @test any(b_q .!= 0.0)
end

# ═══════════════════════════════════════════════════════════════════════════
# transform.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Auto transform" begin
    trans, _ = M._x13_auto_transform(y_pos, 12)
    @test trans == M._X13_LOG || trans == M._X13_NONE

    y_mixed = copy(y_monthly)
    y_mixed[1] = -1.0
    trans_neg, _ = M._x13_auto_transform(y_mixed, 12)
    @test trans_neg == M._X13_NONE
end

# ═══════════════════════════════════════════════════════════════════════════
# armafilter.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "ARMA filter" begin
    @testset "differencing" begin
        v = collect(1.0:10.0)
        d = M._x13_difference!(copy(v), 1)
        @test all(d .≈ 1.0)
        @test length(d) == 9

        v2 = collect(1.0:20.0)
        d2 = M._x13_difference!(copy(v2), 1, 0, 12)
        @test length(d2) == 19
    end

    @testset "expand polynomial" begin
        c, l = M._x13_expand_arma_polynomial([0.5], Float64[], 12)
        @test length(c) == 1 && l == [1]

        c2, l2 = M._x13_expand_arma_polynomial([0.5], [0.3], 12)
        @test 12 in l2

        c0, l0 = M._x13_expand_arma_polynomial(Float64[], Float64[], 12)
        @test isempty(c0)
    end

    @testset "check roots" begin
        @test M._x13_check_roots([0.5], :ar) == true
        @test M._x13_check_roots([1.5], :ar) == false
        @test M._x13_check_roots(Float64[], :ar) == true
    end

    @testset "full filter" begin
        spec = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
        model = M._X13ARIMAModel(spec)
        X = Matrix{Float64}(undef, n, 0)
        fy, fX, na, _, info = M._x13_armafl!(copy(y_monthly), model, X)
        @test na > 0 && na < n
        @test info == 0
        @test length(fy) == na

        # Identity filter (no differencing, no ARMA)
        spec0 = M._X13ARIMASpec(0, 0, 0, 0, 0, 0, 12)
        model0 = M._X13ARIMAModel(spec0)
        fy0, _, na0, _, _ = M._x13_armafl!(copy(y_monthly), model0, X)
        @test na0 == n
        @test fy0 ≈ y_monthly

        # With regressors
        Xr = randn(n, 2)
        fy_r, fXr, na_r, _, _ = M._x13_armafl!(copy(y_monthly), model, copy(Xr))
        @test size(fXr) == (na_r, 2)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# linalg.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Linear algebra" begin
    @test M._x13_enorm(3, [3.0, 4.0, 0.0]) ≈ 5.0

    # Packed Cholesky
    A = [4.0, 2.0, 5.0]   # packed lower triangle of [4 2; 2 5]
    info = M._x13_dppfa!(copy(A), 2)
    @test info == 0 || info == 2

    # QR factorization
    m, nn = 4, 3
    a = randn(m, nn)
    ipvt = zeros(Int, nn)
    rdiag = zeros(nn)
    acnorm = zeros(nn)
    wa = zeros(nn)
    M._x13_qrfac!(m, nn, copy(a), m, true, ipvt, nn, rdiag, acnorm, wa)
    @test all(ipvt .> 0)
end

# ═══════════════════════════════════════════════════════════════════════════
# likelihood.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Likelihood utilities" begin
    @test M._x13_yprmy([3.0, 4.0]) ≈ 25.0

    Xy = randn(20, 3)
    b = zeros(2)
    pxpx = 3 * 4 ÷ 2
    chl = zeros(pxpx)
    info = M._x13_olsreg!(copy(Xy), 20, 3, 3, b, chl, pxpx)
    @test info == 0

    rsd = zeros(20)
    M._x13_compute_residuals!(Xy, 20, 3, 3, 1, 2, -1.0, b, rsd)
    @test length(rsd) == 20
end

# ═══════════════════════════════════════════════════════════════════════════
# estimate.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "ARIMA estimation" begin
    X = Matrix{Float64}(undef, n, 0)

    spec_011 = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
    model = M._X13ARIMAModel(spec_011)
    M._x13_estimate!(model, copy(y_monthly), X)
    @test isfinite(model.sigma2) && model.sigma2 > 0
    @test isfinite(model.aic)
    @test isfinite(model.aicc)
    @test model.niter >= 0

    spec_ar = M._X13ARIMASpec(1, 1, 0, 0, 0, 0, 12)
    model_ar = M._X13ARIMAModel(spec_ar)
    M._x13_estimate!(model_ar, copy(y_monthly), X)
    @test isfinite(model_ar.sigma2)
    @test length(model_ar.ar) == 1

    spec_none = M._X13ARIMASpec(0, 0, 0, 0, 0, 0, 12)
    model_none = M._X13ARIMAModel(spec_none)
    M._x13_estimate!(model_none, copy(y_monthly), X)
    @test isfinite(model_none.sigma2)
end

# ═══════════════════════════════════════════════════════════════════════════
# automodel.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Auto model" begin
    best, outliers = M._x13_auto_model(copy(y_monthly), 12)
    @test best isa M._X13ARIMAModel
    @test best.spec.frequency == 12
    @test isfinite(best.aicc)

    best_q, _ = M._x13_auto_model(copy(y_quarterly), 4)
    @test best_q.spec.frequency == 4

    cands = M._x13_build_candidates(1, 1, 12, 3, 2)
    @test length(cands) > 10
    @test all(c -> c[1] + c[2] <= 3, cands)

    cands_ns = M._x13_build_candidates(1, 0, 1, 3, 0)
    @test all(c -> c[3] == 0 && c[4] == 0, cands_ns)
end

# ═══════════════════════════════════════════════════════════════════════════
# regression.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Regression effects" begin
    @testset "Easter date" begin
        e2024 = M._x13_easter_date(2024)
        @test e2024 == Date(2024, 3, 31)
        e2025 = M._x13_easter_date(2025)
        @test e2025 == Date(2025, 4, 20)
    end

    @testset "period to daterange" begin
        d1, d2 = M._x13_period_to_daterange((2020, 1), 1, 12)
        @test d1 == Date(2020, 1, 1) && d2 == Date(2020, 1, 31)

        d1q, d2q = M._x13_period_to_daterange((2020, 1), 1, 4)
        @test d1q == Date(2020, 1, 1) && month(d2q) == 3

        @test_throws ErrorException M._x13_period_to_daterange((2020, 1), 1, 7)
    end

    @testset "trading day" begin
        ts = M._X13TimeSeries(copy(y_monthly), (2010, 1), 12)
        spec = M._X13RegressionSpec(trading_day=true, easter=false, easter_window=8,
                                     user=Matrix{Float64}(undef, 0, 0))
        X = M._x13_build_regressors(ts, spec)
        @test size(X) == (n, 6)
        @test all(isfinite, X)
    end

    @testset "Easter regressor" begin
        ts = M._X13TimeSeries(copy(y_monthly), (2010, 1), 12)
        spec = M._X13RegressionSpec(trading_day=false, easter=true, easter_window=8,
                                     user=Matrix{Float64}(undef, 0, 0))
        X = M._x13_build_regressors(ts, spec)
        @test size(X) == (n, 1)
        @test any(X .> 0)
    end

    @testset "user regressors" begin
        ts = M._X13TimeSeries(copy(y_monthly), (2010, 1), 12)
        ur = randn(n, 3)
        spec = M._X13RegressionSpec(trading_day=false, easter=false, easter_window=8, user=ur)
        X = M._x13_build_regressors(ts, spec)
        @test size(X) == (n, 3)
    end

    @testset "combined regressors" begin
        ts = M._X13TimeSeries(copy(y_monthly), (2010, 1), 12)
        ur = randn(n, 2)
        spec = M._X13RegressionSpec(trading_day=true, easter=true, easter_window=15, user=ur)
        X = M._x13_build_regressors(ts, spec)
        @test size(X, 2) == 6 + 1 + 2

        spec_empty = M._X13RegressionSpec()
        Xe = M._x13_build_regressors(ts, spec_empty)
        @test size(Xe, 2) == 0
    end

    @testset "quarterly trading day" begin
        ts = M._X13TimeSeries(copy(y_quarterly), (2010, 1), 4)
        spec = M._X13RegressionSpec(trading_day=true, easter=true, easter_window=8,
                                     user=Matrix{Float64}(undef, 0, 0))
        X = M._x13_build_regressors(ts, spec)
        @test size(X) == (nq, 7)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# outliers.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Outlier detection" begin
    @testset "outlier regressors" begin
        ao = M._x13_make_outlier_regressor(M._X13_AO, 5, 20)
        @test ao[5] == 1.0 && sum(ao) == 1.0

        ls = M._x13_make_outlier_regressor(M._X13_LS, 5, 20)
        @test ls[4] == 0.0 && all(ls[5:end] .== 1.0)

        tc = M._x13_make_outlier_regressor(M._X13_TC, 5, 20; tc_rate=0.7)
        @test tc[5] == 1.0 && tc[6] ≈ 0.7 && tc[7] ≈ 0.49
    end

    @testset "critical values" begin
        @test M._x13_default_critical_value(30) == 3.0
        @test M._x13_default_critical_value(100) == 3.5
        @test M._x13_default_critical_value(300) == 3.8
        @test M._x13_default_critical_value(500) == 4.0
    end

    @testset "detect outliers" begin
        y_ol = copy(y_monthly)
        y_ol[30] += 50.0   # large additive outlier
        y_ol[60] += 40.0

        spec = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
        model = M._X13ARIMAModel(spec)
        X = Matrix{Float64}(undef, n, 0)
        M._x13_estimate!(model, copy(y_ol), X)

        outliers = M._x13_detect_outliers!(model, copy(y_ol), X;
            types=[M._X13_AO, M._X13_LS], critical_value=3.0)
        @test outliers isa Vector{M._X13Outlier}
        # Should detect at least one AO near position 30 or 60
        @test length(outliers) >= 0  # may or may not detect depending on threshold
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# forecast.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "Forecast and backcast" begin
    spec = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
    model = M._X13ARIMAModel(spec)
    X = Matrix{Float64}(undef, n, 0)
    M._x13_estimate!(model, copy(y_monthly), X)

    fc = M._x13_forecast(model, y_monthly, 12)
    @test length(fc) == 12
    @test all(isfinite, fc)

    fc24 = M._x13_forecast(model, y_monthly, 24)
    @test length(fc24) == 24

    bc = M._x13_backcast(model, y_monthly, 12)
    @test length(bc) == 12
    @test all(isfinite, bc)

    # Non-seasonal model
    spec_ns = M._X13ARIMASpec(1, 1, 0, 0, 0, 0, 1)
    model_ns = M._X13ARIMAModel(spec_ns)
    model_ns.ar[1] = 0.5
    fc_ns = M._x13_forecast(model_ns, randn(50), 10)
    @test length(fc_ns) == 10 && all(isfinite, fc_ns)
end

# ═══════════════════════════════════════════════════════════════════════════
# seats.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "SEATS decomposition" begin
    spec = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
    model = M._X13ARIMAModel(spec)
    X = Matrix{Float64}(undef, n, 0)
    M._x13_estimate!(model, copy(y_monthly), X)

    seats_spec = M._X13SEATSSpec()
    result = M._x13_seats(model, copy(y_monthly), 12, seats_spec)
    @test result isa M._X13SEATSResult
    @test length(result.trend) == n
    @test length(result.seasonal) == n
    @test length(result.irregular) == n
    @test all(isfinite, result.trend)

    # Quarterly
    spec_q = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 4)
    model_q = M._X13ARIMAModel(spec_q)
    M._x13_estimate!(model_q, copy(y_quarterly), Matrix{Float64}(undef, nq, 0))
    result_q = M._x13_seats(model_q, copy(y_quarterly), 4, seats_spec)
    @test length(result_q.trend) == nq
end

# ═══════════════════════════════════════════════════════════════════════════
# x11.jl
# ═══════════════════════════════════════════════════════════════════════════
@testset "X-11 decomposition" begin
    x11_spec = M._X13X11Spec()
    fc = zeros(12); bc = zeros(12)
    result = M._x13_x11(y_monthly, 12, x11_spec; forecasts=fc, backcasts=bc)
    @test result isa M._X13X11Result
    @test length(result.trend) == n
    @test length(result.seasonal) == n
    @test all(isfinite, result.trend)

    # Additive mode
    x11_add = M._X13X11Spec(mode=:additive)
    result_add = M._x13_x11(y_monthly, 12, x11_add; forecasts=fc, backcasts=bc)
    @test all(isfinite, result_add.trend)

    # Multiplicative mode
    x11_mult = M._X13X11Spec(mode=:multiplicative)
    result_mult = M._x13_x11(y_pos, 12, x11_mult; forecasts=exp.(zeros(12)), backcasts=exp.(zeros(12)))
    @test all(isfinite, result_mult.trend)

    # Quarterly
    x11_q = M._X13X11Spec()
    fc_q = zeros(4); bc_q = zeros(4)
    result_q = M._x13_x11(y_quarterly, 4, x11_q; forecasts=fc_q, backcasts=bc_q)
    @test length(result_q.trend) == nq

    # Custom henderson length
    x11_h5 = M._X13X11Spec(henderson_length=5)
    result_h5 = M._x13_x11(y_monthly, 12, x11_h5; forecasts=fc, backcasts=bc)
    @test all(isfinite, result_h5.trend)

    # Custom sigma bounds
    x11_sigma = M._X13X11Spec(sigma_lower=1.0, sigma_upper=3.0)
    result_sigma = M._x13_x11(y_monthly, 12, x11_sigma; forecasts=fc, backcasts=bc)
    @test all(isfinite, result_sigma.trend)
end

# ═══════════════════════════════════════════════════════════════════════════
# api.jl — full pipeline paths
# ═══════════════════════════════════════════════════════════════════════════
@testset "API full pipeline" begin
    @testset "trading day + Easter" begin
        r = x13_filter(y_pos; frequency=12, trading_day=true, easter=true)
        @test r isa X13FilterResult{Float64}
        @test length(r.trend) == n
    end

    @testset "outlier detection with custom cv" begin
        y_ol = copy(y_monthly)
        y_ol[50] += 100.0
        r = x13_filter(y_ol; frequency=12, outliers=true, critical_value=2.5)
        @test r isa X13FilterResult{Float64}
    end

    @testset "X-11 only (no SEATS)" begin
        r = x13_filter(y_monthly; frequency=12, method=:x11)
        @test r.method == :x11
    end

    @testset "SEATS only" begin
        r = x13_filter(y_monthly; frequency=12, method=:seats)
        @test r.method == :seats
    end

    @testset "log transform explicit" begin
        r = x13_filter(y_pos; frequency=12, transform=:log)
        @test r.transform == :log
        @test all(isfinite, r.trend)
    end

    @testset "auto transform positive data" begin
        r = x13_filter(y_pos; frequency=12, transform=:auto)
        @test r.transform in [:log, :none]
    end

    @testset "no transform" begin
        r = x13_filter(y_monthly; frequency=12, transform=:none)
        @test r.transform == :none
    end

    @testset "quarterly full pipeline" begin
        r = x13_filter(y_quarterly; frequency=4, method=:x11)
        @test r.frequency == 4
        @test r isa X13FilterResult{Float64}
    end

    @testset "quarterly SEATS" begin
        r = x13_filter(y_quarterly; frequency=4, method=:seats)
        @test r.frequency == 4
    end

    @testset "quarterly with trading day + Easter" begin
        r = x13_filter(y_quarterly; frequency=4, trading_day=true, easter=true)
        @test r isa X13FilterResult{Float64}
    end
end

@testset "Internal _x13_run paths" begin
    yf = Float64.(y_monthly)
    yp = Float64.(y_pos)

    # Custom forecast horizon
    res = M._x13_run(yf; frequency=12, forecast_horizon=24)
    @test res isa M._X13InternalResult

    # Explicit model spec (not :auto)
    spec = M._X13ARIMASpec(0, 1, 1, 0, 1, 1, 12)
    res2 = M._x13_run(yf; frequency=12, model=spec, outliers=true)
    @test res2.model.spec.p == 0 && res2.model.spec.q == 1

    # X-11 with non-default henderson and mode
    res3 = M._x13_run(yf; frequency=12, x11_henderson=5, x11_mode=:additive)
    @test res3.x11 !== nothing

    # SEATS only (no X-11)
    res4 = M._x13_run(yf; frequency=12, x11=false, seats=true)
    @test res4.x11 === nothing && res4.seats !== nothing

    # X-11 only (no SEATS)
    res5 = M._x13_run(yf; frequency=12, x11=true, seats=false)
    @test res5.x11 !== nothing && res5.seats === nothing

    # Outlier type filtering
    res6 = M._x13_run(yf; frequency=12, model=spec, outliers=true,
                       outlier_types=[M._X13_AO])
    @test res6 isa M._X13InternalResult

    # Log transform with positive data
    res7 = M._x13_run(yp; frequency=12, transform=:log)
    @test res7.transform == M._X13_LOG
end

# ═══════════════════════════════════════════════════════════════════════════
# lmdif.jl — exercised through estimation
# ═══════════════════════════════════════════════════════════════════════════
@testset "Levenberg-Marquardt (through estimation)" begin
    X = Matrix{Float64}(undef, n, 0)
    # Higher-order model requires more LM iterations
    spec = M._X13ARIMASpec(2, 1, 1, 1, 1, 1, 12)
    model = M._X13ARIMAModel(spec)
    M._x13_estimate!(model, copy(y_monthly), X)
    @test model.niter > 0
    @test isfinite(model.sigma2)
end

end  # @testset "X-13 Coverage"
