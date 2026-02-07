using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics

# =============================================================================
# Helper: generate cointegrated data
# =============================================================================

function gen_cointegrated_data(T_obs::Int, n::Int; rank::Int=1, seed::Int=42)
    rng = MersenneTwister(seed)
    # Generate I(1) common trends
    trends = cumsum(randn(rng, T_obs, n), dims=1)
    Y = copy(trends)
    # Create cointegrating relationships by making some variables
    # linear combinations of others plus stationary noise
    for r in 1:min(rank, n-1)
        Y[:, r+1] = Y[:, 1] + 0.1 * randn(rng, T_obs)
    end
    Y
end

# =============================================================================
# Johansen Estimation
# =============================================================================

@testset "VECM Johansen Estimation" begin
    Random.seed!(42)

    @testset "Basic estimation" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2)

        @test m isa VECMModel{Float64}
        @test nvars(m) == 3
        @test nlags(m) == 2
        @test m.p == 2
        @test m.method == :johansen
        @test m.deterministic == :constant
        @test effective_nobs(m) > 0
        @test effective_nobs(m) == size(m.U, 1)
        @test size(m.U, 2) == 3
        @test size(m.Sigma) == (3, 3)
        @test issymmetric(round.(m.Sigma, digits=10))
        @test isfinite(m.aic)
        @test isfinite(m.bic)
        @test isfinite(m.hqic)
        @test isfinite(m.loglik)
    end

    @testset "Rank detection" begin
        # Rank 1 system
        Y = gen_cointegrated_data(300, 3; rank=1, seed=123)
        m = estimate_vecm(Y, 2)
        @test m.rank >= 0
        @test m.rank <= 3
        @test size(m.alpha) == (3, m.rank)
        @test size(m.beta) == (3, m.rank)
        @test size(m.Pi) == (3, 3)

        # Explicit rank
        m2 = estimate_vecm(Y, 2; rank=1)
        @test m2.rank == 1
        @test size(m2.alpha) == (3, 1)
        @test size(m2.beta) == (3, 1)
    end

    @testset "Rank 2 system" begin
        Y = gen_cointegrated_data(300, 4; rank=2, seed=456)
        m = estimate_vecm(Y, 2; rank=2)
        @test m.rank == 2
        @test size(m.alpha) == (4, 2)
        @test size(m.beta) == (4, 2)
        @test size(m.Pi) == (4, 4)
    end

    @testset "Deterministic specifications" begin
        Y = gen_cointegrated_data(200, 3; rank=1)

        for det in (:none, :constant, :trend)
            m = estimate_vecm(Y, 2; rank=1, deterministic=det)
            @test m.deterministic == det
            @test m isa VECMModel{Float64}
        end
    end

    @testset "Different lag orders" begin
        Y = gen_cointegrated_data(200, 3; rank=1)

        m1 = estimate_vecm(Y, 1; rank=1)
        @test m1.p == 1
        @test isempty(m1.Gamma)

        m2 = estimate_vecm(Y, 2; rank=1)
        @test m2.p == 2
        @test length(m2.Gamma) == 1
        @test size(m2.Gamma[1]) == (3, 3)

        m3 = estimate_vecm(Y, 3; rank=1)
        @test m3.p == 3
        @test length(m3.Gamma) == 2
    end

    @testset "Pi = alpha * beta'" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        @test m.Pi ≈ m.alpha * m.beta' atol=1e-10
    end

    @testset "Phillips normalization" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        # First r rows of beta should form identity
        @test m.beta[1, 1] ≈ 1.0 atol=1e-10

        Y4 = gen_cointegrated_data(300, 4; rank=2, seed=456)
        m2 = estimate_vecm(Y4, 2; rank=2)
        @test m2.beta[1:2, :] ≈ Matrix{Float64}(I, 2, 2) atol=1e-8
    end

    @testset "Johansen result stored" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2)
        @test m.johansen_result isa JohansenResult
        @test m.johansen_result.rank >= 0
    end
end

# =============================================================================
# Engle-Granger Estimation
# =============================================================================

@testset "VECM Engle-Granger Estimation" begin
    Random.seed!(42)

    @testset "Basic bivariate" begin
        Y = gen_cointegrated_data(200, 2; rank=1)
        m = estimate_vecm(Y, 2; method=:engle_granger)
        @test m isa VECMModel{Float64}
        @test m.rank == 1
        @test m.method == :engle_granger
        @test m.johansen_result === nothing
        @test size(m.alpha) == (2, 1)
        @test size(m.beta) == (2, 1)
        @test m.beta[1, 1] ≈ 1.0  # normalized on first variable
    end

    @testset "Multivariate" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; method=:engle_granger)
        @test m.rank == 1
        @test size(m.alpha) == (3, 1)
        @test size(m.beta) == (3, 1)
    end

    @testset "Rank must be 1" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        @test_throws ArgumentError estimate_vecm(Y, 2; method=:engle_granger, rank=2)
    end
end

# =============================================================================
# Rank Zero (No Cointegration)
# =============================================================================

@testset "VECM Rank Zero" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=0)

    @test m.rank == 0
    @test size(m.alpha) == (3, 0)
    @test size(m.beta) == (3, 0)
    @test m.Pi ≈ zeros(3, 3) atol=1e-15
    @test m isa VECMModel{Float64}
    @test isfinite(m.aic)
end

# =============================================================================
# to_var() Conversion
# =============================================================================

@testset "VECM to VAR Conversion" begin
    Random.seed!(42)

    @testset "Dimensions" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)

        @test v isa VARModel{Float64}
        @test nvars(v) == 3
        @test v.p == 2
        @test size(v.B) == (1 + 3*2, 3)
        @test size(v.Y) == size(Y)
        @test isfinite(v.aic)
        @test isfinite(v.bic)
    end

    @testset "VAR(1) conversion" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 1; rank=1)
        v = to_var(m)
        @test v.p == 1
        @test size(v.B) == (1 + 3, 3)
    end

    @testset "Coefficient reconstruction" begin
        # For VAR(2): A1 = Pi + I + Gamma1, A2 = -Gamma1
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)

        n = 3
        In = Matrix{Float64}(I, n, n)
        A1_expected = m.Pi + In + m.Gamma[1]
        A2_expected = -m.Gamma[1]

        A = extract_ar_coefficients(v.B, n, 2)
        @test A[1] ≈ A1_expected atol=1e-10
        @test A[2] ≈ A2_expected atol=1e-10

        # Intercept
        @test v.B[1, :] ≈ m.mu atol=1e-10
    end

    @testset "VAR(3) conversion" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 3; rank=1)
        v = to_var(m)

        n = 3
        In = Matrix{Float64}(I, n, n)
        A = extract_ar_coefficients(v.B, n, 3)

        # A1 = Pi + I + Gamma1
        @test A[1] ≈ m.Pi + In + m.Gamma[1] atol=1e-10
        # A2 = Gamma2 - Gamma1
        @test A[2] ≈ m.Gamma[2] - m.Gamma[1] atol=1e-10
        # A3 = -Gamma2
        @test A[3] ≈ -m.Gamma[2] atol=1e-10
    end

    @testset "Companion eigenvalues" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)
        F = companion_matrix(v.B, nvars(v), v.p)
        eigvals_F = abs.(eigvals(F))
        # Cointegrated system: should have unit roots approximately
        @test maximum(eigvals_F) >= 0.5  # at least some persistence
    end
end

# =============================================================================
# IRF / FEVD / HD via VECM
# =============================================================================

@testset "VECM Innovation Accounting" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "IRF dispatch" begin
        r = irf(m, 10)
        @test r isa ImpulseResponse{Float64}
        @test size(r.values) == (10, 3, 3)

        # With CIs
        r2 = irf(m, 10; ci_type=:bootstrap, reps=50)
        @test r2.ci_type == :bootstrap
        @test size(r2.ci_lower) == (10, 3, 3)
    end

    @testset "FEVD dispatch" begin
        f = fevd(m, 10)
        @test f isa FEVD{Float64}
        @test size(f.proportions) == (3, 3, 10)
        # Proportions sum to 1 at each horizon
        for h in 1:10
            for v in 1:3
                @test sum(f.proportions[v, :, h]) ≈ 1.0 atol=1e-8
            end
        end
    end

    @testset "Historical decomposition dispatch" begin
        T_eff = effective_nobs(to_var(m))
        hd = historical_decomposition(m, T_eff)
        @test hd isa HistoricalDecomposition{Float64}
        @test hd.method == :cholesky
    end
end

# =============================================================================
# Forecasting
# =============================================================================

@testset "VECM Forecasting" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "Point forecast" begin
        fc = forecast(m, 10)
        @test fc isa VECMForecast{Float64}
        @test size(fc.levels) == (10, 3)
        @test size(fc.differences) == (10, 3)
        @test fc.horizon == 10
        @test fc.ci_method == :none
        @test all(isfinite, fc.levels)

        # Differences should be consistent with levels
        expected_diff = diff(vcat(Y[end:end, :], fc.levels), dims=1)
        @test fc.differences ≈ expected_diff atol=1e-10
    end

    @testset "Bootstrap CIs" begin
        fc = forecast(m, 5; ci_method=:bootstrap, reps=100)
        @test fc.ci_method == :bootstrap
        @test size(fc.ci_lower) == (5, 3)
        @test size(fc.ci_upper) == (5, 3)
        # CI lower < point < CI upper (approximately)
        for j in 1:3
            @test fc.ci_lower[1, j] <= fc.levels[1, j] + 1.0  # some tolerance
        end
    end

    @testset "Simulation CIs" begin
        fc = forecast(m, 5; ci_method=:simulation, reps=100)
        @test fc.ci_method == :simulation
        @test all(isfinite, fc.ci_lower)
        @test all(isfinite, fc.ci_upper)
    end

    @testset "Forecast from rank 0" begin
        m0 = estimate_vecm(Y, 2; rank=0)
        fc = forecast(m0, 5)
        @test all(isfinite, fc.levels)
    end

    @testset "Forecast from VAR(1)" begin
        m1 = estimate_vecm(Y, 1; rank=1)
        fc = forecast(m1, 5)
        @test all(isfinite, fc.levels)
    end
end

# =============================================================================
# Granger Causality
# =============================================================================

@testset "VECM Granger Causality" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "Basic test" begin
        g = granger_causality_vecm(m, 1, 2)
        @test g isa VECMGrangerResult{Float64}
        @test g.cause_var == 1
        @test g.effect_var == 2

        # Statistics are non-negative
        @test g.short_run_stat >= 0
        @test g.long_run_stat >= 0
        @test g.strong_stat >= 0

        # P-values in [0, 1]
        @test 0 <= g.short_run_pvalue <= 1
        @test 0 <= g.long_run_pvalue <= 1
        @test 0 <= g.strong_pvalue <= 1

        # Degrees of freedom are positive
        @test g.short_run_df >= 0
        @test g.long_run_df >= 0
        @test g.strong_df >= 0
        @test g.strong_df == g.short_run_df + g.long_run_df
    end

    @testset "All variable pairs" begin
        for i in 1:3, j in 1:3
            if i != j
                g = granger_causality_vecm(m, i, j)
                @test g.cause_var == i
                @test g.effect_var == j
                @test isfinite(g.strong_pvalue)
            end
        end
    end

    @testset "Error for same variable" begin
        @test_throws ArgumentError granger_causality_vecm(m, 1, 1)
    end

    @testset "Error for out of range" begin
        @test_throws ArgumentError granger_causality_vecm(m, 0, 1)
        @test_throws ArgumentError granger_causality_vecm(m, 1, 4)
    end

    @testset "Rank 0 model" begin
        m0 = estimate_vecm(Y, 2; rank=0)
        g = granger_causality_vecm(m0, 1, 2)
        @test g.long_run_stat == 0.0
        @test g.long_run_pvalue == 1.0
        @test g.long_run_df == 0
    end
end

# =============================================================================
# Select Rank
# =============================================================================

@testset "VECM Rank Selection" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(300, 3; rank=1, seed=123)

    r_trace = select_vecm_rank(Y, 2; criterion=:trace)
    @test r_trace >= 0
    @test r_trace <= 3

    r_max = select_vecm_rank(Y, 2; criterion=:max_eigen)
    @test r_max >= 0
    @test r_max <= 3
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

@testset "VECM StatsAPI" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @test coef(m) isa Vector{Float64}
    @test length(coef(m)) > 0
    @test residuals(m) === m.U
    @test nobs(m) == 200
    @test aic(m) ≈ m.aic
    @test bic(m) ≈ m.bic
    @test loglikelihood(m) ≈ m.loglik
    @test islinear(m) == true
    @test dof(m) > 0

    # predict returns in-sample fitted differences
    fitted = predict(m)
    @test size(fitted) == (effective_nobs(m), 3)
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "VECM Edge Cases" begin
    Random.seed!(42)

    @testset "Input validation" begin
        Y = randn(20, 3)
        @test_throws ArgumentError estimate_vecm(Y, 2; deterministic=:invalid)
        @test_throws ArgumentError estimate_vecm(Y, 2; method=:invalid)
        @test_throws ArgumentError estimate_vecm(Y, 0)
        @test_throws ArgumentError estimate_vecm(Y, 2; rank=-1)
        @test_throws ArgumentError estimate_vecm(Y, 2; rank=4)

        Y_small = randn(5, 3)
        @test_throws ArgumentError estimate_vecm(Y_small, 2)
    end

    @testset "Full rank" begin
        Y = gen_cointegrated_data(200, 3; rank=1)
        m = estimate_vecm(Y, 2; rank=3)
        @test m.rank == 3
        @test size(m.alpha) == (3, 3)
        @test size(m.beta) == (3, 3)
    end

    @testset "Float32 input" begin
        Y = Float32.(gen_cointegrated_data(200, 3; rank=1))
        m = estimate_vecm(Y, 2; rank=1)
        @test m isa VECMModel{Float32}
    end

    @testset "Integer input" begin
        Y = round.(Int, gen_cointegrated_data(200, 3; rank=1) .* 10)
        m = estimate_vecm(Y, 2; rank=1)
        @test m isa VECMModel{Float64}  # promoted via @float_fallback
    end

    @testset "Bivariate system" begin
        Y = gen_cointegrated_data(200, 2; rank=1)
        m = estimate_vecm(Y, 2; rank=1)
        @test nvars(m) == 2
        @test m.rank == 1
        v = to_var(m)
        @test nvars(v) == 2
    end
end

# =============================================================================
# Display
# =============================================================================

@testset "VECM Display" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "show" begin
        buf = IOBuffer()
        show(buf, m)
        s = String(take!(buf))
        @test occursin("VECM", s)
        @test occursin("Rank", s)
    end

    @testset "report" begin
        buf = IOBuffer()
        old_stdout = stdout
        rd, wr = redirect_stdout()
        try
            report(m)
        finally
            redirect_stdout(old_stdout)
            close(wr)
        end
        output = String(read(rd))
        close(rd)
        @test occursin("VECM", output)
        @test occursin("Cointegrating", output)
    end

    @testset "VECMForecast show" begin
        fc = forecast(m, 5)
        buf = IOBuffer()
        show(buf, fc)
        s = String(take!(buf))
        @test occursin("VECM Forecast", s)
    end

    @testset "VECMGrangerResult show" begin
        g = granger_causality_vecm(m, 1, 2)
        buf = IOBuffer()
        show(buf, g)
        s = String(take!(buf))
        @test occursin("Granger", s)
    end

    @testset "refs" begin
        buf = IOBuffer()
        refs(buf, m)
        s = String(take!(buf))
        @test occursin("Johansen", s)
    end

    @testset "refs symbol dispatch" begin
        buf = IOBuffer()
        refs(buf, :vecm)
        s = String(take!(buf))
        @test occursin("Johansen", s)

        buf2 = IOBuffer()
        refs(buf2, :engle_granger)
        s2 = String(take!(buf2))
        @test occursin("Engle", s2)
    end
end

# =============================================================================
# Accessor Functions
# =============================================================================

@testset "VECM Accessors" begin
    Random.seed!(42)
    Y = gen_cointegrated_data(200, 3; rank=1)
    m = estimate_vecm(Y, 2; rank=1)

    @test nvars(m) == 3
    @test nlags(m) == 2
    @test cointegrating_rank(m) == 1
    @test effective_nobs(m) == size(m.U, 1)
    @test ncoefs(m) > 0
end
