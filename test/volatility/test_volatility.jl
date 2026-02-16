using Test
using MacroEconometricModels
using Random
using Statistics
if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# Fix seed for reproducibility
Random.seed!(42)

# =============================================================================
# Helper: Simulate ARCH(1) data
# =============================================================================
function simulate_arch1(n::Int; omega=0.1, alpha1=0.3, mu=0.0)
    y = zeros(n)
    h = zeros(n)
    h[1] = omega / (1.0 - alpha1)
    y[1] = sqrt(h[1]) * randn()
    for t in 2:n
        h[t] = omega + alpha1 * y[t-1]^2
        y[t] = mu + sqrt(h[t]) * randn()
    end
    y
end

# =============================================================================
# Helper: Simulate GARCH(1,1) data
# =============================================================================
function simulate_garch11(n::Int; omega=0.01, alpha1=0.05, beta1=0.90, mu=0.0)
    y = zeros(n)
    h = zeros(n)
    h[1] = omega / (1.0 - alpha1 - beta1)
    y[1] = mu + sqrt(h[1]) * randn()
    for t in 2:n
        h[t] = omega + alpha1 * (y[t-1] - mu)^2 + beta1 * h[t-1]
        y[t] = mu + sqrt(h[t]) * randn()
    end
    y
end

# =============================================================================
# Helper: Simulate GJR-GARCH(1,1) data
# =============================================================================
function simulate_gjr11(n::Int; omega=0.01, alpha1=0.03, gamma1=0.07, beta1=0.85, mu=0.0)
    y = zeros(n)
    h = zeros(n)
    h[1] = omega / (1.0 - alpha1 - gamma1/2 - beta1)
    y[1] = mu + sqrt(h[1]) * randn()
    for t in 2:n
        eps = y[t-1] - mu
        indicator = eps < 0.0 ? 1.0 : 0.0
        h[t] = omega + (alpha1 + gamma1 * indicator) * eps^2 + beta1 * h[t-1]
        y[t] = mu + sqrt(h[t]) * randn()
    end
    y
end

# =============================================================================
# ARCH Estimation Tests
# =============================================================================

@testset "ARCH estimation" begin
    Random.seed!(123)
    y_arch = simulate_arch1(1000; omega=0.2, alpha1=0.4)

    @testset "ARCH(1) basic" begin
        m = estimate_arch(y_arch, 1)
        @test m isa ARCHModel{Float64}
        @test m.q == 1
        @test m.omega > 0
        @test all(m.alpha .> 0)
        @test length(m.conditional_variance) == length(y_arch)
        @test length(m.standardized_residuals) == length(y_arch)
        @test all(m.conditional_variance .> 0)
        @test isfinite(m.loglik)
        @test m.aic > m.loglik * (-2) - 1  # AIC ≈ -2ℓ + 2k
    end

    @testset "ARCH(1) stationarity" begin
        m = estimate_arch(y_arch, 1)
        @test persistence(m) < 1.0
        @test persistence(m) > 0.0
    end

    @testset "ARCH(1) unconditional variance" begin
        m = estimate_arch(y_arch, 1)
        uv = unconditional_variance(m)
        @test isfinite(uv)
        @test uv > 0
        # Should be roughly close to sample variance
        @test uv < var(y_arch) * 5
    end

    @testset "ARCH(2)" begin
        m2 = estimate_arch(y_arch, 2)
        @test m2.q == 2
        @test length(m2.alpha) == 2
        @test all(m2.alpha .> 0)
    end

    @testset "ARCH input validation" begin
        @test_throws ArgumentError estimate_arch(randn(10), 1)  # Too few obs
        @test_throws ArgumentError estimate_arch(randn(100), 0)  # q < 1
    end

    @testset "ARCH halflife" begin
        m = estimate_arch(y_arch, 1)
        hl = halflife(m)
        @test isfinite(hl)
        @test hl > 0
    end

    @testset "ARCH Integer input" begin
        y_int = round.(Int, y_arch .* 100)
        m = estimate_arch(y_int, 1)
        @test m isa ARCHModel{Float64}
    end
end

# =============================================================================
# GARCH Estimation Tests
# =============================================================================

@testset "GARCH estimation" begin
    Random.seed!(456)
    y_garch = simulate_garch11(1000; omega=0.01, alpha1=0.05, beta1=0.90)

    @testset "GARCH(1,1) basic" begin
        m = estimate_garch(y_garch, 1, 1)
        @test m isa GARCHModel{Float64}
        @test m.p == 1
        @test m.q == 1
        @test m.omega > 0
        @test all(m.alpha .> 0)
        @test all(m.beta .> 0)
        @test length(m.conditional_variance) == length(y_garch)
        @test all(m.conditional_variance .> 0)
    end

    @testset "GARCH(1,1) persistence" begin
        m = estimate_garch(y_garch, 1, 1)
        p = persistence(m)
        @test p < 1.0
        @test p > 0.5  # GARCH(1,1) typically has high persistence
    end

    @testset "GARCH(1,1) unconditional variance" begin
        m = estimate_garch(y_garch, 1, 1)
        uv = unconditional_variance(m)
        @test isfinite(uv)
        @test uv > 0
    end

    @testset "GARCH(2,1)" begin
        m = estimate_garch(y_garch, 2, 1)
        @test m.p == 2
        @test length(m.beta) == 2
    end

    @testset "GARCH(1,2)" begin
        m = estimate_garch(y_garch, 1, 2)
        @test m.q == 2
        @test length(m.alpha) == 2
    end

    @testset "GARCH arch/garch_order" begin
        m = estimate_garch(y_garch, 1, 1)
        @test arch_order(m) == 1
        @test garch_order(m) == 1
    end

    @testset "GARCH Integer input" begin
        y_int = round.(Int, y_garch .* 1000)
        m = estimate_garch(y_int)
        @test m isa GARCHModel{Float64}
    end
end

# =============================================================================
# EGARCH Estimation Tests
# =============================================================================

@testset "EGARCH estimation" begin
    Random.seed!(789)
    # Simulate with leverage effect (larger negative shocks → higher vol)
    n = 1000
    y_lev = zeros(n)
    h = zeros(n)
    log_h = zeros(n)
    E_abs_z = sqrt(2/pi)
    log_h[1] = -1.0
    h[1] = exp(log_h[1])
    y_lev[1] = sqrt(h[1]) * randn()
    for t in 2:n
        z = y_lev[t-1] / sqrt(h[t-1])
        log_h[t] = -0.3 + 0.15 * (abs(z) - E_abs_z) + (-0.08) * z + 0.95 * log_h[t-1]
        h[t] = exp(log_h[t])
        y_lev[t] = sqrt(h[t]) * randn()
    end

    @testset "EGARCH(1,1) basic" begin
        m = estimate_egarch(y_lev, 1, 1)
        @test m isa EGARCHModel{Float64}
        @test m.p == 1
        @test m.q == 1
        @test length(m.gamma) == 1
        @test all(m.conditional_variance .> 0)
    end

    @testset "EGARCH leverage detection" begin
        m = estimate_egarch(y_lev, 1, 1)
        # Leverage coefficient should be negative for equity-like data
        @test m.gamma[1] < 0.1  # Allow some estimation noise
    end

    @testset "EGARCH persistence" begin
        m = estimate_egarch(y_lev, 1, 1)
        @test persistence(m) > 0.0
        @test persistence(m) < 1.0
    end

    @testset "EGARCH unconditional variance" begin
        m = estimate_egarch(y_lev, 1, 1)
        uv = unconditional_variance(m)
        @test isfinite(uv)
        @test uv > 0
    end

    @testset "EGARCH StatsAPI dof" begin
        m = estimate_egarch(y_lev, 1, 1)
        @test dof(m) == 2 + 2*1 + 1  # mu + omega + q alphas + q gammas + p betas
    end
end

# =============================================================================
# GJR-GARCH Estimation Tests
# =============================================================================

@testset "GJR-GARCH estimation" begin
    Random.seed!(101)
    y_gjr = simulate_gjr11(1000; omega=0.01, alpha1=0.03, gamma1=0.07, beta1=0.85)

    @testset "GJR-GARCH(1,1) basic" begin
        m = estimate_gjr_garch(y_gjr, 1, 1)
        @test m isa GJRGARCHModel{Float64}
        @test m.p == 1
        @test m.q == 1
        @test length(m.gamma) == 1
        @test m.omega > 0
        @test all(m.conditional_variance .> 0)
    end

    @testset "GJR-GARCH asymmetry" begin
        m = estimate_gjr_garch(y_gjr, 1, 1)
        @test m.gamma[1] > 0  # Asymmetry should be detected
    end

    @testset "GJR-GARCH persistence" begin
        m = estimate_gjr_garch(y_gjr, 1, 1)
        p = persistence(m)
        @test p < 1.0
        @test p > 0.0
    end

    @testset "GJR-GARCH unconditional variance" begin
        m = estimate_gjr_garch(y_gjr, 1, 1)
        uv = unconditional_variance(m)
        @test isfinite(uv)
        @test uv > 0
    end
end

# =============================================================================
# Volatility Forecasting Tests
# =============================================================================

@testset "Volatility forecasting" begin
    Random.seed!(202)
    y = simulate_garch11(500; omega=0.01, alpha1=0.05, beta1=0.90)

    @testset "ARCH forecast" begin
        Random.seed!(303)
        m = estimate_arch(y, 1)
        fc = forecast(m, 10)
        @test fc isa VolatilityForecast{Float64}
        @test fc.horizon == 10
        @test fc.model_type == :arch
        @test length(fc.forecast) == 10
        @test all(fc.forecast .> 0)
        @test all(fc.ci_lower .> 0)
        @test all(fc.ci_upper .>= fc.ci_lower)
        @test all(fc.se .>= 0)
    end

    @testset "GARCH forecast" begin
        Random.seed!(303)
        m = estimate_garch(y, 1, 1)
        fc = forecast(m, 10)
        @test fc isa VolatilityForecast{Float64}
        @test fc.model_type == :garch
        @test length(fc.forecast) == 10
        @test all(fc.forecast .> 0)
        @test all(fc.ci_upper .>= fc.ci_lower)
    end

    @testset "EGARCH forecast" begin
        Random.seed!(303)
        m = estimate_egarch(y, 1, 1)
        fc = forecast(m, 5)
        @test fc isa VolatilityForecast{Float64}
        @test fc.model_type == :egarch
        @test length(fc.forecast) == 5
        @test all(fc.forecast .> 0)
    end

    @testset "GJR-GARCH forecast" begin
        Random.seed!(303)
        m = estimate_gjr_garch(y, 1, 1)
        fc = forecast(m, 5)
        @test fc isa VolatilityForecast{Float64}
        @test fc.model_type == :gjr_garch
        @test all(fc.forecast .> 0)
    end

    @testset "Forecast horizon validation" begin
        m = estimate_arch(y, 1)
        @test_throws ArgumentError forecast(m, 0)
        @test_throws ArgumentError forecast(m, -1)
    end

    @testset "Forecast mean reversion" begin
        Random.seed!(404)
        m = estimate_garch(y, 1, 1)
        fc = forecast(m, 100; n_sim=5000)
        uv = unconditional_variance(m)
        if isfinite(uv) && uv > 0
            # Long-horizon forecast should move toward unconditional variance
            @test abs(fc.forecast[end] - uv) < abs(fc.forecast[1] - uv) + uv * 0.5
        end
    end
end

# =============================================================================
# StatsAPI Compliance Tests
# =============================================================================

@testset "StatsAPI compliance" begin
    Random.seed!(505)
    y = simulate_garch11(500)

    @testset "ARCH StatsAPI" begin
        m = estimate_arch(y, 1)
        @test nobs(m) == 500
        @test length(coef(m)) == 3  # mu + omega + alpha
        @test length(residuals(m)) == 500
        @test length(predict(m)) == 500
        @test isfinite(loglikelihood(m))
        @test isfinite(aic(m))
        @test isfinite(bic(m))
        @test dof(m) == 3
        @test islinear(m) == false
    end

    @testset "GARCH StatsAPI" begin
        m = estimate_garch(y, 1, 1)
        @test nobs(m) == 500
        @test length(coef(m)) == 4  # mu + omega + alpha + beta
        @test dof(m) == 4
        @test islinear(m) == false
        @test aic(m) < bic(m)  # BIC penalizes more for n > e²
    end

    @testset "EGARCH StatsAPI" begin
        m = estimate_egarch(y, 1, 1)
        @test length(coef(m)) == 5  # mu + omega + alpha + gamma + beta
        @test dof(m) == 5
    end

    @testset "GJR-GARCH StatsAPI" begin
        m = estimate_gjr_garch(y, 1, 1)
        @test length(coef(m)) == 5  # mu + omega + alpha + gamma + beta
        @test dof(m) == 5
    end
end

# =============================================================================
# Diagnostics Tests
# =============================================================================

@testset "Diagnostics" begin
    @testset "ARCH-LM on white noise" begin
        Random.seed!(606)
        y_wn = randn(500)
        result = arch_lm_test(y_wn, 5)
        @test result.pvalue > 0.01  # White noise should not reject
        @test result.q == 5
        @test result.statistic >= 0
    end

    @testset "ARCH-LM on ARCH data" begin
        Random.seed!(607)
        y_arch = simulate_arch1(1000; omega=0.1, alpha1=0.5)
        result = arch_lm_test(y_arch, 5)
        @test result.pvalue < 0.05  # Should detect ARCH effects
    end

    @testset "ARCH-LM on fitted model" begin
        Random.seed!(608)
        y = simulate_arch1(500; omega=0.1, alpha1=0.3)
        m = estimate_arch(y, 1)
        result = arch_lm_test(m, 5)
        # After fitting ARCH, standardized residuals should have less ARCH effect
        @test result.statistic >= 0
        @test 0 <= result.pvalue <= 1
    end

    @testset "ARCH-LM validation" begin
        @test_throws ArgumentError arch_lm_test(randn(5), 10)  # Too few obs
        @test_throws ArgumentError arch_lm_test(randn(100), 0)  # q < 1
    end

    @testset "Ljung-Box squared" begin
        Random.seed!(609)
        y_wn = randn(500)
        result = ljung_box_squared(y_wn, 10)
        @test result.pvalue > 0.01  # No correlation in squared white noise
        @test result.K == 10
        @test result.statistic >= 0
    end

    @testset "Ljung-Box squared on ARCH data" begin
        Random.seed!(610)
        y = simulate_arch1(1000; omega=0.1, alpha1=0.5)
        z = y ./ std(y)
        result = ljung_box_squared(z, 10)
        @test result.pvalue < 0.05  # Should detect serial correlation in z²
    end

    @testset "Ljung-Box on fitted model" begin
        Random.seed!(611)
        y = simulate_garch11(500)
        m = estimate_garch(y, 1, 1)
        result = ljung_box_squared(m, 10)
        @test result.statistic >= 0
        @test 0 <= result.pvalue <= 1
    end
end

# =============================================================================
# News Impact Curve Tests
# =============================================================================

@testset "News impact curve" begin
    Random.seed!(707)
    y = simulate_garch11(500; omega=0.01, alpha1=0.05, beta1=0.90)

    @testset "GARCH NIC symmetry" begin
        m = estimate_garch(y, 1, 1)
        nic = news_impact_curve(m)
        @test length(nic.shocks) == 200
        @test length(nic.variance) == 200
        @test all(nic.variance .> 0)
        # Symmetric: variance at shock = +s should equal variance at shock = -s
        mid = length(nic.shocks) ÷ 2
        @test abs(nic.variance[1] - nic.variance[end]) / nic.variance[1] < 0.1
    end

    @testset "GJR-GARCH NIC asymmetry" begin
        y_gjr = simulate_gjr11(1000; gamma1=0.1)
        m = estimate_gjr_garch(y_gjr, 1, 1)
        nic = news_impact_curve(m)
        # Negative shocks should produce higher variance than positive shocks
        neg_idx = findfirst(x -> x < 0, nic.shocks)
        pos_idx = findlast(x -> x > 0, nic.shocks)
        # At symmetric points, negative side should be steeper
        @test nic.variance[1] > nic.variance[end]  # Far negative > far positive
    end

    @testset "EGARCH NIC" begin
        m = estimate_egarch(y, 1, 1)
        nic = news_impact_curve(m)
        @test length(nic.shocks) == 200
        @test all(nic.variance .> 0)
    end

    @testset "NIC custom range" begin
        m = estimate_garch(y, 1, 1)
        nic = news_impact_curve(m; range=(-5.0, 5.0), n_points=100)
        @test length(nic.shocks) == 100
    end
end

# =============================================================================
# Stochastic Volatility Tests
# =============================================================================

@testset "SV estimation" begin
    Random.seed!(808)
    # Simulate SV data (reduced from n=200)
    n_sv = 100
    mu_true = -1.0
    phi_true = 0.95
    sigma_eta_true = 0.2
    h = zeros(n_sv)
    h[1] = mu_true
    for t in 2:n_sv
        h[t] = mu_true + phi_true * (h[t-1] - mu_true) + sigma_eta_true * randn()
    end
    y_sv = exp.(h ./ 2) .* randn(n_sv)

    # Estimate :normal once and reuse across subtests
    Random.seed!(909)
    m_normal = estimate_sv(y_sv; n_samples=(FAST ? 20 : 50), burnin=(FAST ? 10 : 25), dist=:normal)

    @testset "Basic SV" begin
        @test m_normal isa SVModel{Float64}
        @test nobs(m_normal) == n_sv
        @test length(m_normal.mu_post) == (FAST ? 20 : 50)
        @test length(m_normal.phi_post) == (FAST ? 20 : 50)
        @test length(m_normal.sigma_eta_post) == (FAST ? 20 : 50)
        @test length(m_normal.volatility_mean) == n_sv
        @test size(m_normal.volatility_quantiles) == (n_sv, 3)
        @test all(m_normal.volatility_mean .> 0)
        @test m_normal.dist == :normal
        @test m_normal.leverage == false
    end

    @testset "SV persistence" begin
        phi_mean = mean(m_normal.phi_post)
        # With high persistence data, posterior should be near unit
        @test abs(phi_mean) < 1.0
        @test phi_mean > 0.0  # Should be positive
    end

    @testset "SV StatsAPI" begin
        @test nobs(m_normal) == n_sv
        @test length(coef(m_normal)) == 3  # mu, phi, sigma_eta
        @test length(residuals(m_normal)) == n_sv
        @test length(predict(m_normal)) == n_sv
        @test islinear(m_normal) == false
    end

    @testset "SV posterior shapes" begin
        @test all(m_normal.sigma_eta_post .> 0)  # sigma_eta is positive
        @test all(abs.(m_normal.phi_post) .< 1.0)  # phi in (-1, 1)
    end

    @testset "SV Student-t" begin
        Random.seed!(1010)
        m_t = estimate_sv(y_sv; n_samples=(FAST ? 20 : 50), burnin=(FAST ? 10 : 25), dist=:studentt)
        @test m_t isa SVModel{Float64}
        @test m_t.dist == :studentt
        @test all(m_t.volatility_mean .> 0)
    end

    @testset "SV input validation" begin
        @test_throws ArgumentError estimate_sv(randn(10))  # Too few obs
    end

    @testset "SV forecast" begin
        Random.seed!(1111)
        fc = forecast(m_normal, 5)
        @test fc isa VolatilityForecast{Float64}
        @test fc.model_type == :sv
        @test fc.horizon == 5
        @test all(fc.forecast .> 0)
        @test all(fc.ci_upper .> fc.ci_lower)
    end
end

# =============================================================================
# Display Tests
# =============================================================================

@testset "Display methods" begin
    Random.seed!(1212)
    y = simulate_garch11(300)

    @testset "ARCHModel display" begin
        m = estimate_arch(y, 1)
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("ARCH(1)", output)
        @test occursin("ω", output)
        @test occursin("α[1]", output)
    end

    @testset "GARCHModel display" begin
        m = estimate_garch(y, 1, 1)
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("GARCH(1,1)", output)
        @test occursin("β[1]", output)
    end

    @testset "EGARCHModel display" begin
        m = estimate_egarch(y, 1, 1)
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("EGARCH(1,1)", output)
        @test occursin("γ[1]", output)
    end

    @testset "GJRGARCHModel display" begin
        m = estimate_gjr_garch(y, 1, 1)
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("GJR-GARCH(1,1)", output)
    end

    @testset "VolatilityForecast display" begin
        m = estimate_garch(y, 1, 1)
        fc = forecast(m, 5)
        io = IOBuffer()
        show(io, fc)
        output = String(take!(io))
        @test occursin("Volatility Forecast", output)
        @test occursin("garch", output)
    end

    @testset "SVModel display" begin
        Random.seed!(1313)
        y_sv = randn(100) .* exp.(cumsum(0.1 .* randn(100)) ./ 2)
        m = estimate_sv(y_sv; n_samples=(FAST ? 20 : 30), burnin=(FAST ? 10 : 15))
        io = IOBuffer()
        show(io, m)
        output = String(take!(io))
        @test occursin("Stochastic Volatility", output)
        @test occursin("φ", output)
    end
end
