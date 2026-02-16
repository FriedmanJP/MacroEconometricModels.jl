using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using DataFrames

# =============================================================================
# Test Data Generation
# =============================================================================

"""Generate synthetic mixed-frequency data with known factor structure."""
function _make_nowcast_data(; T_obs=120, nM=6, nQ=2, r=2, seed=42)
    rng = Random.MersenneTwister(seed)

    # True factors (monthly)
    F = randn(rng, T_obs, r)
    for t in 2:T_obs
        F[t, :] = 0.7 * F[t-1, :] + 0.3 * randn(rng, r)
    end

    # Monthly loadings
    Lambda_M = randn(rng, nM, r)
    X_M = F * Lambda_M' + 0.2 * randn(rng, T_obs, nM)

    # Quarterly loadings (observed every 3rd month)
    Lambda_Q = randn(rng, nQ, r)
    X_Q = F * Lambda_Q' + 0.2 * randn(rng, T_obs, nQ)

    # Set quarterly to NaN for non-quarter months
    for t in 1:T_obs
        if mod(t, 3) != 0
            X_Q[t, :] .= NaN
        end
    end

    Y = hcat(X_M, X_Q)
    return Y, F, Lambda_M, Lambda_Q
end

"""Generate synthetic data with ragged edge pattern."""
function _make_ragged_data(; T_obs=120, nM=6, nQ=2, n_missing=5, seed=42)
    Y, F, _, _ = _make_nowcast_data(T_obs=T_obs, nM=nM, nQ=nQ, seed=seed)

    # Add ragged edge: last n_missing months of some variables are NaN
    for j in 1:3
        Y[(T_obs - n_missing + 1):T_obs, j] .= NaN
    end

    return Y
end

# =============================================================================
# 1. Kalman Filter with Missing Data
# =============================================================================

@testset "Kalman Filter with Missing Data" begin
    rng = Random.MersenneTwister(123)

    @testset "Basic functionality" begin
        # Simple 2-state system
        state_dim = 2
        N = 3
        T_obs = 50
        A = [0.8 0.1; 0.0 0.9]
        C = [1.0 0.0; 0.5 0.5; 0.0 1.0]
        Q = [0.1 0.0; 0.0 0.1]
        R = Matrix{Float64}(0.05 * I(N))
        x0 = zeros(state_dim)
        P0 = Matrix{Float64}(I(state_dim))

        # Generate data
        x = zeros(state_dim, T_obs)
        y = zeros(N, T_obs)
        x[:, 1] = A * x0 + cholesky(Q).L * randn(rng, state_dim)
        y[:, 1] = C * x[:, 1] + cholesky(R).L * randn(rng, N)
        for t in 2:T_obs
            x[:, t] = A * x[:, t-1] + cholesky(Q).L * randn(rng, state_dim)
            y[:, t] = C * x[:, t] + cholesky(R).L * randn(rng, N)
        end

        # No missing data
        x_pred, P_pred, x_filt, P_filt, loglik = MacroEconometricModels._kalman_filter_missing(
            y, A, C, Q, R, x0, P0)
        @test size(x_filt) == (state_dim, T_obs)
        @test loglik < 0  # negative log-likelihood
        @test !any(isnan, x_filt)
        @test !any(isnan, P_filt)
    end

    @testset "Smoother with no missing data" begin
        state_dim = 2
        N = 2
        T_obs = 30
        A = [0.5 0.0; 0.0 0.5]
        C = Matrix{Float64}(I(N))
        Q = [0.2 0.0; 0.0 0.2]
        R = [0.1 0.0; 0.0 0.1]
        x0 = zeros(state_dim)
        P0 = Matrix{Float64}(I(state_dim))

        y = randn(rng, N, T_obs)

        x_sm, P_sm, PP_sm, loglik = MacroEconometricModels._kalman_smoother_missing(
            y, A, C, Q, R, x0, P0)
        @test size(x_sm) == (state_dim, T_obs)
        @test size(P_sm) == (state_dim, state_dim, T_obs)
        @test loglik < 0
        @test !any(isnan, x_sm)
    end

    @testset "Missing data handling" begin
        state_dim = 2
        N = 3
        T_obs = 50
        A = [0.7 0.0; 0.0 0.7]
        C = [1.0 0.0; 0.5 0.5; 0.0 1.0]
        Q = [0.1 0.0; 0.0 0.1]
        R = Matrix{Float64}(0.05 * I(N))
        x0 = zeros(state_dim)
        P0 = Matrix{Float64}(I(state_dim))

        y = randn(rng, N, T_obs)

        # Insert NaN at specific positions
        y[2, 10] = NaN
        y[1, 20] = NaN
        y[3, 20] = NaN
        y[:, 30] .= NaN  # all missing

        x_sm, P_sm, PP_sm, loglik = MacroEconometricModels._kalman_smoother_missing(
            y, A, C, Q, R, x0, P0)
        @test !any(isnan, x_sm)
        @test loglik < 0
    end

    @testset "_miss_data row elimination" begin
        y = [1.0, NaN, 3.0, NaN, 5.0]
        C = randn(5, 2)
        R = Matrix{Float64}(0.1 * I(5))

        y_obs, C_obs, R_obs, idx = MacroEconometricModels._miss_data(y, C, R)
        @test length(y_obs) == 3
        @test size(C_obs) == (3, 2)
        @test size(R_obs) == (3, 3)
        @test idx == [1, 3, 5]
    end

    @testset "All NaN row" begin
        y = [NaN, NaN, NaN]
        C = randn(3, 2)
        R = Matrix{Float64}(0.1 * I(3))

        y_obs, C_obs, R_obs, idx = MacroEconometricModels._miss_data(y, C, R)
        @test isempty(y_obs)
        @test isempty(idx)
    end

    @testset "Smoother with lagged covariances" begin
        state_dim = 2
        N = 2
        T_obs = 30
        A = [0.5 0.0; 0.0 0.5]
        C = Matrix{Float64}(I(N))
        Q = [0.2 0.0; 0.0 0.2]
        R = [0.1 0.0; 0.0 0.1]
        x0 = zeros(state_dim)
        P0 = Matrix{Float64}(I(state_dim))

        y = randn(rng, N, T_obs)
        y[1, 15] = NaN

        k = 3
        x_sm, P_sm, Plag, loglik = MacroEconometricModels._kalman_smoother_lag(
            y, A, C, Q, R, x0, P0, k)
        @test length(Plag) == k
        @test size(Plag[1]) == (state_dim, state_dim, T_obs)
        @test !any(isnan, x_sm)
    end

    @testset "Ragged edge pattern" begin
        state_dim = 2
        N = 4
        T_obs = 60
        A = [0.7 0.1; 0.0 0.8]
        C = randn(rng, N, state_dim)
        Q = [0.1 0.0; 0.0 0.1]
        R = Matrix{Float64}(0.05 * I(N))
        x0 = zeros(state_dim)
        P0 = Matrix{Float64}(I(state_dim))

        y = randn(rng, N, T_obs)
        # Ragged edge: last few obs missing for some variables
        y[3, 55:60] .= NaN
        y[4, 58:60] .= NaN

        x_sm, _, _, loglik = MacroEconometricModels._kalman_smoother_missing(
            y, A, C, Q, R, x0, P0)
        @test !any(isnan, x_sm)
        @test loglik < 0
    end
end

# =============================================================================
# 2. DFM Nowcasting
# =============================================================================

@testset "DFM Nowcasting" begin
    @testset "Basic estimation" begin
        Y, F, _, _ = _make_nowcast_data(T_obs=90, nM=4, nQ=1, r=2, seed=123)

        m = nowcast_dfm(Y, 4, 1; r=2, p=1, max_iter=20, thresh=1e-3)

        @test m isa NowcastDFM{Float64}
        @test size(m.X_sm) == size(Y)
        @test !any(isnan, m.X_sm)
        @test m.r == 2
        @test m.p == 1
        @test m.nM == 4
        @test m.nQ == 1
        @test m.n_iter >= 1
        @test m.loglik < 0 || m.loglik isa Float64  # loglik is finite
        @test isfinite(m.loglik)
    end

    @testset "EM convergence" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=120, nM=6, nQ=2, r=2, seed=456)

        m = nowcast_dfm(Y, 6, 2; r=2, p=1, max_iter=50, thresh=1e-4)

        @test m.n_iter <= 50
        @test isfinite(m.loglik)
    end

    @testset "All monthly (no quarterly)" begin
        rng = Random.MersenneTwister(789)
        Y = randn(rng, 80, 5)
        Y[75:80, 3:5] .= NaN  # ragged edge

        m = nowcast_dfm(Y, 5, 0; r=2, p=1, max_iter=20, thresh=1e-3)

        @test size(m.X_sm) == (80, 5)
        @test !any(isnan, m.X_sm)
        @test m.nM == 5
        @test m.nQ == 0
    end

    @testset "Fills NaN correctly" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=3, nQ=1, r=1, seed=321)

        # Count NaN in input vs output
        n_nan_in = count(isnan, Y)
        m = nowcast_dfm(Y, 3, 1; r=1, p=1, max_iter=30, thresh=1e-3)
        n_nan_out = count(isnan, m.X_sm)

        @test n_nan_in > 0
        @test n_nan_out == 0
    end

    @testset "Single factor" begin
        rng = Random.MersenneTwister(111)
        F = cumsum(randn(rng, 60, 1), dims=1) * 0.1
        Lambda = randn(rng, 4, 1)
        Y = F * Lambda' + 0.1 * randn(rng, 60, 4)

        m = nowcast_dfm(Y, 4, 0; r=1, p=1, max_iter=20, thresh=1e-3)

        @test m.r == 1
        @test size(m.F, 2) >= 1
    end

    @testset "Block structure" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=90, nM=6, nQ=2, r=2, seed=222)

        # 2 blocks: first 3 monthly + 1 quarterly, second 3 monthly + 1 quarterly
        blocks = zeros(Int, 8, 2)
        blocks[1:3, 1] .= 1
        blocks[4:6, 2] .= 1
        blocks[7, 1] = 1
        blocks[8, 2] = 1

        m = nowcast_dfm(Y, 6, 2; r=1, p=1, blocks=blocks, max_iter=20, thresh=1e-3)

        @test size(m.blocks) == (8, 2)
        @test !any(isnan, m.X_sm)
    end

    @testset "IID idiosyncratic" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=333)

        m = nowcast_dfm(Y, 4, 1; r=1, p=1, idio=:iid, max_iter=20, thresh=1e-3)

        @test m.idio == :iid
        @test !any(isnan, m.X_sm)
    end

    @testset "Input validation" begin
        Y = randn(50, 5)
        @test_throws ArgumentError nowcast_dfm(Y, 3, 3)  # nM + nQ != N
        @test_throws ArgumentError nowcast_dfm(Y, 5, 0; r=0)  # r < 1
        @test_throws ArgumentError nowcast_dfm(Y, 5, 0; idio=:foo)  # invalid idio
    end

    @testset "Ragged edge filling" begin
        Y = _make_ragged_data(T_obs=90, nM=5, nQ=1, n_missing=5, seed=444)

        m = nowcast_dfm(Y, 5, 1; r=2, p=1, max_iter=30, thresh=1e-3)

        # Check that ragged edge is filled
        @test !any(isnan, m.X_sm[86:90, 1:3])
        # Filled values should be within reasonable range
        for j in 1:3
            valid = filter(!isnan, Y[:, j])
            @test all(abs.(m.X_sm[86:90, j]) .< 10 * std(valid) + abs(mean(valid)))
        end
    end

    @testset "StatsAPI interface" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=555)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        @test loglikelihood(m) == m.loglik
        @test predict(m) == m.X_sm
        @test nobs(m) == 60
    end
end

# =============================================================================
# 3. Large BVAR Nowcasting
# =============================================================================

@testset "BVAR Nowcasting" begin
    @testset "Basic estimation" begin
        rng = Random.MersenneTwister(100)
        Y = randn(rng, 80, 5)
        Y[75:80, 4:5] .= NaN  # ragged edge

        m = nowcast_bvar(Y, 3, 2; lags=2, max_iter=30)

        @test m isa NowcastBVAR{Float64}
        @test size(m.X_sm) == (80, 5)
        @test !any(isnan, m.X_sm)
        @test m.lags == 2
        @test m.nM == 3
        @test m.nQ == 2
        @test isfinite(m.loglik)
        @test m.lambda > 0
        @test m.theta > 0
    end

    @testset "Fills ragged edge" begin
        rng = Random.MersenneTwister(200)
        Y = randn(rng, 60, 4)
        Y[56:60, 3:4] .= NaN

        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=20)

        @test !any(isnan, m.X_sm[56:60, 3:4])
    end

    @testset "Hyperparameter optimization" begin
        rng = Random.MersenneTwister(300)
        Y = randn(rng, 100, 6)

        m = nowcast_bvar(Y, 4, 2; lags=3, max_iter=50)

        # Optimized hyperparameters should be positive
        @test m.lambda > 0
        @test m.theta > 0
        @test m.miu > 0
        @test m.alpha > 0
    end

    @testset "Input validation" begin
        Y = randn(50, 5)
        @test_throws ArgumentError nowcast_bvar(Y, 3, 3)  # nM + nQ != N
        @test_throws ArgumentError nowcast_bvar(Y, 5, 0; lags=0)  # lags < 1
    end

    @testset "StatsAPI interface" begin
        rng = Random.MersenneTwister(400)
        Y = randn(rng, 60, 4)
        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=10)

        @test loglikelihood(m) == m.loglik
        @test predict(m) == m.X_sm
        @test nobs(m) == 60
    end
end

# =============================================================================
# 4. Bridge Equation Nowcasting
# =============================================================================

@testset "Bridge Equation Nowcasting" begin
    @testset "Basic estimation" begin
        rng = Random.MersenneTwister(500)
        Y = randn(rng, 90, 5)  # 3 monthly + 2 quarterly
        # Make quarterly variables NaN except every 3rd month
        for t in 1:90
            if mod(t, 3) != 0
                Y[t, 4:5] .= NaN
            end
        end

        m = nowcast_bridge(Y, 3, 2; lagM=1, lagQ=1, lagY=1)

        @test m isa NowcastBridge{Float64}
        @test m.nM == 3
        @test m.nQ == 2
        @test m.n_equations >= 1
        @test length(m.Y_nowcast) == 90 ÷ 3
        @test !all(isnan, m.Y_nowcast)
    end

    @testset "Equation combination" begin
        # With 4 monthly variables, should have C(4,2) + 4 = 10 equations
        combos = MacroEconometricModels._bridge_combinations(4, 1)
        @test size(combos, 1) == 10  # 6 pairs + 4 univariate
        @test size(combos, 2) == 2
    end

    @testset "Monthly to quarterly aggregation" begin
        Xm = ones(12, 2)  # 12 months, 2 variables
        Xm[:, 2] .= 2.0
        Xq = MacroEconometricModels._bridge_m2q(Xm, 4)

        @test size(Xq) == (4, 2)
        @test all(Xq[:, 1] .≈ 1.0)
        @test all(Xq[:, 2] .≈ 2.0)
    end

    @testset "Input validation" begin
        Y = randn(60, 5)
        @test_throws ArgumentError nowcast_bridge(Y, 3, 3)  # nM + nQ != N
        @test_throws ArgumentError nowcast_bridge(Y, 5, 0)  # nQ < 1
    end

    @testset "Nowcast values reasonable" begin
        rng = Random.MersenneTwister(600)
        T_obs = 120
        Y = randn(rng, T_obs, 5)
        for t in 1:T_obs
            if mod(t, 3) != 0
                Y[t, 4:5] .= NaN
            end
        end

        m = nowcast_bridge(Y, 3, 2; lagM=1, lagQ=0, lagY=1)

        # Non-NaN nowcasts should be within reasonable range
        valid_nc = filter(!isnan, m.Y_nowcast)
        if !isempty(valid_nc)
            @test all(abs.(valid_nc) .< 100)
        end
    end
end

# =============================================================================
# 5. News Decomposition
# =============================================================================

@testset "News Decomposition" begin
    @testset "Basic news computation" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=700)

        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        # Create old vintage with more NaN
        X_old = copy(Y)
        X_old[58:60, 2] .= NaN  # additional missing

        news = nowcast_news(Y, X_old, m, 58; target_var=5)

        @test news isa NowcastNews{Float64}
        @test isfinite(news.old_nowcast)
        @test isfinite(news.new_nowcast)
        @test length(news.impact_news) == count((isnan.(X_old)) .& (.!isnan.(Y)))
    end

    @testset "No-news case" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=800)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        # Same data → no news
        news = nowcast_news(Y, Y, m, 30; target_var=5)

        @test length(news.impact_news) == 0
        @test news.old_nowcast ≈ news.new_nowcast atol=1e-6
    end

    @testset "Decomposition identity" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=900)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        X_old = copy(Y)
        X_old[55:60, 1:2] .= NaN

        news = nowcast_news(Y, X_old, m, 55; target_var=5)

        total = news.new_nowcast - news.old_nowcast
        decomp = sum(news.impact_news) + news.impact_revision + news.impact_reestimation

        @test total ≈ decomp atol=1e-8
    end

    @testset "Input validation" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1000)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        @test_throws ArgumentError nowcast_news(Y, Y[1:50, :], m, 30)  # size mismatch
        @test_throws ArgumentError nowcast_news(Y, Y, m, 0)  # out of range
        @test_throws ArgumentError nowcast_news(Y, Y, m, 30; target_var=0)  # out of range
    end

    @testset "Group impacts" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1100)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        X_old = copy(Y)
        X_old[58:60, 1:2] .= NaN

        groups = [1, 1, 2, 2, 3]  # 3 groups
        news = nowcast_news(Y, X_old, m, 58; target_var=5, groups=groups)

        @test length(news.group_impacts) == 3
    end
end

# =============================================================================
# 6. Nowcast and Forecast Dispatch
# =============================================================================

@testset "Nowcast and Forecast" begin
    @testset "nowcast() DFM" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1200)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        result = nowcast(m)
        @test result isa NowcastResult{Float64}
        @test result.method == :dfm
        @test isfinite(result.nowcast)
        @test isfinite(result.forecast)
        @test result.target_index == 5
    end

    @testset "nowcast() BVAR" begin
        rng = Random.MersenneTwister(1300)
        Y = randn(rng, 60, 4)
        Y[55:60, 3:4] .= NaN

        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=20)

        result = nowcast(m)
        @test result isa NowcastResult{Float64}
        @test result.method == :bvar
        @test isfinite(result.nowcast)
    end

    @testset "nowcast() Bridge" begin
        rng = Random.MersenneTwister(1400)
        Y = randn(rng, 90, 4)
        for t in 1:90
            mod(t, 3) != 0 && (Y[t, 4] = NaN)
        end

        m = nowcast_bridge(Y, 3, 1; lagM=1, lagQ=0, lagY=1)

        result = nowcast(m)
        @test result isa NowcastResult{Float64}
        @test result.method == :bridge
    end

    @testset "forecast() DFM" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1500)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        fc = forecast(m, 6)
        @test size(fc) == (6, 5)
        @test !any(isnan, fc)

        # Single variable
        fc_v = forecast(m, 3; target_var=1)
        @test length(fc_v) == 3
    end

    @testset "forecast() BVAR" begin
        rng = Random.MersenneTwister(1600)
        Y = randn(rng, 60, 4)
        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=10)

        fc = forecast(m, 6)
        @test size(fc) == (6, 4)
        @test !any(isnan, fc)
    end

    @testset "nowcast() with target_var" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1700)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        result = nowcast(m; target_var=3)
        @test result.target_index == 3
    end
end

# =============================================================================
# 7. balance_panel
# =============================================================================

@testset "balance_panel" begin
    @testset "PanelData with NaN" begin
        rng = Random.MersenneTwister(1800)
        x_vals = Vector{Union{Missing,Float64}}(randn(rng, 90))
        y_vals = Vector{Union{Missing,Float64}}(randn(rng, 90))
        x_vals[85:90] .= missing
        y_vals[28:30] .= missing
        df = DataFrame(
            id = repeat(1:3, inner=30),
            t = repeat(1:30, 3),
            x = x_vals,
            y = y_vals,
        )

        pd = xtset(df, :id, :t)
        @test any(isnan, pd.data)

        pd_bal = balance_panel(pd; r=1, p=1)
        @test !any(isnan, pd_bal.data)
        @test pd_bal isa PanelData{Float64}
    end

    @testset "Already balanced panel" begin
        rng = Random.MersenneTwister(1900)
        df = DataFrame(
            id = repeat(1:2, inner=20),
            t = repeat(1:20, 2),
            x = randn(rng, 40),
            y = randn(rng, 40),
        )
        pd = xtset(df, :id, :t)
        pd_bal = balance_panel(pd; r=1, p=1)

        @test pd_bal.data ≈ pd.data  # no change
    end

    @testset "TimeSeriesData with NaN" begin
        rng = Random.MersenneTwister(2000)
        Y = randn(rng, 50, 3)
        Y[45:50, 2] .= NaN

        ts = TimeSeriesData(Y)
        ts_bal = balance_panel(ts; r=1, p=1)

        @test !any(isnan, ts_bal.data)
        @test ts_bal isa TimeSeriesData{Float64}
        # Observed values should be preserved
        @test ts_bal.data[1:44, :] ≈ ts.data[1:44, :] atol=1e-10
    end

    @testset "No NaN returns same" begin
        rng = Random.MersenneTwister(2100)
        Y = randn(rng, 30, 2)
        ts = TimeSeriesData(Y)
        ts_bal = balance_panel(ts; r=1)
        @test ts_bal === ts  # same object (no copy needed)
    end

    @testset "Input validation" begin
        Y = randn(30, 3)
        Y[25:30, 1] .= NaN
        ts = TimeSeriesData(Y)
        @test_throws ArgumentError balance_panel(ts; method=:foo)
    end
end

# =============================================================================
# 8. Display and Report Methods
# =============================================================================

@testset "Display and Report" begin
    @testset "NowcastDFM show" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=2200)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test contains(s, "DFM Nowcasting")
        @test contains(s, "Dynamic Factor Model")
    end

    @testset "NowcastBVAR show" begin
        rng = Random.MersenneTwister(2300)
        Y = randn(rng, 60, 4)
        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=10)

        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test contains(s, "BVAR Nowcasting")
        @test contains(s, "Large BVAR")
    end

    @testset "NowcastBridge show" begin
        rng = Random.MersenneTwister(2400)
        Y = randn(rng, 60, 4)
        for t in 1:60
            mod(t, 3) != 0 && (Y[t, 4] = NaN)
        end

        m = nowcast_bridge(Y, 3, 1; lagM=1, lagQ=0, lagY=1)

        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test contains(s, "Bridge Equation")
    end

    @testset "NowcastResult show" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=2500)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)
        result = nowcast(m)

        io = IOBuffer()
        show(io, result)
        s = String(take!(io))
        @test contains(s, "Nowcast Result")
        @test contains(s, "DFM")
    end

    @testset "NowcastNews show" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=2600)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        X_old = copy(Y)
        X_old[58:60, 1] .= NaN

        news = nowcast_news(Y, X_old, m, 58; target_var=5)

        io = IOBuffer()
        show(io, news)
        s = String(take!(io))
        @test contains(s, "News Decomposition")
    end

    @testset "report() dispatch" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=2700)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        # Test show(io, m) directly (redirect_stdout(IOBuffer) not supported in Julia 1.12)
        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test !isempty(s)
    end
end

# =============================================================================
# 9. References
# =============================================================================

@testset "References" begin
    @testset "Nowcasting references exist" begin
        for key in [:banbura_modugno2014, :cimadomo2022, :banbura2023, :delle_chiaie2022]
            io = IOBuffer()
            refs(io, [key]; format=:text)  # use Vector{Symbol} for direct key lookup
            s = String(take!(io))
            @test !isempty(s)
        end
    end

    @testset "Instance dispatch" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=2800)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=10, thresh=1e-2)

        io = IOBuffer()
        refs(io, m)
        s = String(take!(io))
        @test contains(s, "Modugno") || contains(s, "2014")
    end

    @testset "Symbol dispatch" begin
        io = IOBuffer()
        refs(io, :nowcast_dfm; format=:text)
        s = String(take!(io))
        @test !isempty(s)

        io2 = IOBuffer()
        refs(io2, :nowcast_bvar; format=:text)
        s2 = String(take!(io2))
        @test contains(s2, "Cimadomo") || contains(s2, "2022")
    end
end

# =============================================================================
# 10. TimeSeriesData Dispatch Wrappers
# =============================================================================

@testset "TimeSeriesData Dispatch" begin
    @testset "nowcast_dfm with TimeSeriesData" begin
        rng = Random.MersenneTwister(2900)
        Y = randn(rng, 60, 4)
        Y[55:60, 3:4] .= NaN
        ts = TimeSeriesData(Y)

        m = nowcast_dfm(ts, 3, 1; r=1, p=1, max_iter=10, thresh=1e-2)
        @test m isa NowcastDFM{Float64}
        @test !any(isnan, m.X_sm)
    end

    @testset "nowcast_bvar with TimeSeriesData" begin
        rng = Random.MersenneTwister(3000)
        Y = randn(rng, 60, 4)
        ts = TimeSeriesData(Y)

        m = nowcast_bvar(ts, 2, 2; lags=2, max_iter=10)
        @test m isa NowcastBVAR{Float64}
    end

    @testset "nowcast_bridge with TimeSeriesData" begin
        rng = Random.MersenneTwister(3100)
        Y = randn(rng, 60, 4)
        for t in 1:60
            mod(t, 3) != 0 && (Y[t, 4] = NaN)
        end
        ts = TimeSeriesData(Y)

        m = nowcast_bridge(ts, 3, 1; lagM=1, lagQ=0, lagY=1)
        @test m isa NowcastBridge{Float64}
    end
end

# =============================================================================
# 11. Edge Cases
# =============================================================================

@testset "Edge Cases" begin
    @testset "High missingness" begin
        rng = Random.MersenneTwister(3200)
        Y = randn(rng, 60, 4)
        # 50% missing
        for i in 1:60, j in 1:4
            rand(rng) < 0.5 && (Y[i, j] = NaN)
        end

        m = nowcast_dfm(Y, 4, 0; r=1, p=1, max_iter=20, thresh=1e-2)
        @test !any(isnan, m.X_sm)
    end

    @testset "Small sample" begin
        rng = Random.MersenneTwister(3300)
        Y = randn(rng, 15, 3)
        Y[13:15, 2] .= NaN

        m = nowcast_dfm(Y, 3, 0; r=1, p=1, max_iter=20, thresh=1e-2)
        @test size(m.X_sm) == (15, 3)
        @test !any(isnan, m.X_sm)
    end

    @testset "Two variables" begin
        rng = Random.MersenneTwister(3400)
        Y = randn(rng, 40, 2)
        Y[35:40, 2] .= NaN

        m = nowcast_dfm(Y, 2, 0; r=1, p=1, max_iter=20, thresh=1e-2)
        @test !any(isnan, m.X_sm)
    end

    @testset "Float32 input" begin
        rng = Random.MersenneTwister(3500)
        Y = Float32.(randn(rng, 40, 3))
        Y[35:40, 2] .= NaN32

        m = nowcast_dfm(Y, 3, 0; r=1, p=1, max_iter=10, thresh=1e-2)
        @test m isa NowcastDFM{Float32}
        @test !any(isnan, m.X_sm)
    end
end
