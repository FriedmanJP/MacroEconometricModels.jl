# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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

    @testset "Fewer series than factors: N < r·n_blocks (#209 R-30)" begin
        # N = 4 series but r·n_blocks = 2·3 = 6 requested factors. Init extracts only
        # n_eig = min(6,4) = 4; the EM M-step must use the SAME factor count or its
        # factor-block indexing diverges from init. Before the fix it used
        # min(r·n_blocks, state_dim) = 6 and disagreed.
        Y, _, _, _ = _make_nowcast_data(T_obs=80, nM=3, nQ=1, r=2, seed=777)
        blocks = zeros(Int, 4, 3)
        blocks[1, 1] = 1
        blocks[2, 2] = 1
        blocks[3, 3] = 1
        blocks[4, 1] = 1
        m = nowcast_dfm(Y, 3, 1; r=2, p=1, blocks=blocks, max_iter=15, thresh=1e-3)
        @test size(m.blocks) == (4, 3)
        @test !any(isnan, m.X_sm)
    end

    @testset "IID idiosyncratic" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=333)

        m = nowcast_dfm(Y, 4, 1; r=1, p=1, idio=:iid, max_iter=20, thresh=1e-3)

        @test m.idio == :iid
        @test !any(isnan, m.X_sm)
    end

    @testset "Mariano-Murasawa temporal aggregation (#38)" begin
        rng = Random.MersenneTwister(3838)
        T_obs = 120; nM = 3; nQ = 2; r = 2

        Y = randn(rng, T_obs, nM + nQ)
        for j in (nM+1):(nM+nQ)
            for t in 1:T_obs
                mod(t, 3) != 0 && (Y[t, j] = NaN)
            end
        end

        m = nowcast_dfm(Y, nM, nQ; r=r, p=1, max_iter=30)

        # State dimension: r*max(p,5) + nM(ar1) + 5*nQ = 2*5+3+10 = 23
        n_f = r
        p_eff = 5
        @test size(m.A, 1) == n_f * p_eff + nM + 5 * nQ

        # Quarterly factor loadings must have [1,2,3,2,1] structure
        weights = [1.0, 2.0, 3.0, 2.0, 1.0]
        for q in 1:nQ
            i = nM + q
            for c in 1:n_f
                base_load = m.C[i, c]  # w=1 at lag 0
                if abs(base_load) > 1e-10
                    for k in 1:4
                        @test m.C[i, k * n_f + c] ≈ weights[k + 1] * base_load
                    end
                end
            end
        end

        # Monthly variables: zero loadings on lagged factor states
        for i in 1:nM, k in 1:4, c in 1:n_f
            @test m.C[i, k * n_f + c] == 0.0
        end

        # With nQ=0, state dim should use p not max(p,5)
        Y_monthly = randn(rng, 80, 4)
        m0 = nowcast_dfm(Y_monthly, 4, 0; r=2, p=1, max_iter=10)
        @test size(m0.A, 1) == 2 * 1 + 4  # r*p + nM (ar1), no quarterly
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
    @testset "GLP hyperparameter sanity flag (B4/T173 / #571)" begin
        # T173: a hyperparameter parked on the |log-param| ≤ 5 box edge must be reported,
        # never presented bare. Which panels pin is data-dependent once the dummy design
        # and the NIW marginal likelihood are correct (#571/#572), so assert the detector's
        # invariant rather than one panel's outcome: pinned ⇔ !converged, and !converged
        # ⇒ show() carries the warning.
        m = nowcast_bvar(randn(Random.MersenneTwister(9), 50, 10), 6, 4; lags=5)
        log_pars = log.([m.lambda, m.theta, m.miu, m.alpha])
        @test m.converged == !any(x -> abs(x) >= 5 - 1e-3, log_pars)
        @test m.converged || occursin("WARNING", sprint(show, m))
        @test isfinite(m.loglik) && m.loglik > -1e9   # not the degenerate -1e10 sentinel
        # A well-conditioned interior fit converges and does NOT warn.
        m2 = nowcast_bvar(randn(Random.MersenneTwister(300), 100, 6), 4, 2; lags=3, max_iter=50)
        @test m2.converged
        @test !occursin("WARNING", sprint(show, m2))
        # Display half of the detector: flag down ⇒ warning, whatever the fit produced.
        m_pinned = MacroEconometricModels.NowcastBVAR{Float64}(
            m2.X_sm, m2.beta, m2.sigma, exp(5.0), m2.theta, m2.miu, m2.alpha,
            m2.lags, m2.loglik, m2.nM, m2.nQ, m2.data, false)
        @test occursin("WARNING", sprint(show, m_pinned))
    end

    @testset "Litterman non-conjugate prior (#602)" begin
        MEM = MacroEconometricModels
        rng = Random.MersenneTwister(602)
        N, lags, T_obs = 3, 2, 40
        Y = randn(rng, T_obs, N)
        for t in 2:T_obs
            Y[t, :] .+= 0.6 .* Y[t-1, :]
        end
        sar = [std(diff(Y[:, j])) for j in 1:N]
        lam, th, tc, mu, al = 0.4, 1.3, 0.7, 1.1, 0.9

        beta, sigma, logml = MEM._litterman_estimate(Y, lags, sar, lam, th, tc, mu, al)

        # Rebuild the regression independently.
        Y_dep = Y[(lags+1):end, :]
        T_eff = size(Y_dep, 1)
        X = ones(T_eff, 1)
        for l in 1:lags
            X = hcat(X, Y[(lags+1-l):(end-l), :])
        end
        y_bar = [mean(Y[1:lags, j]) for j in 1:N]

        ref_logml = 0.0
        for m in 1:N
            A, c = MEM._litterman_prior_rows(m, N, lags, sar, y_bar, lam, th, tc, mu, al)
            P = A'A
            b = A \ c
            s2 = sar[m]^2
            # (1) posterior mean is the plain generalized-ridge / GLS solution
            @test (P + X'X / s2) \ (P * b + X' * Y_dep[:, m] / s2) ≈ beta[:, m] atol = 1e-10
            # (2) marginal likelihood equals a direct T x T Gaussian density, written out by
            #     hand — the completion-of-square algebra must be exact, not close. Under
            #     beta ~ N(b, P^-1) and known s2, y ~ N(X b, s2 I + X P^-1 X').
            Cov = Symmetric(s2 * Matrix(I, T_eff, T_eff) + X * inv(P) * X')
            e = Y_dep[:, m] - X * b
            ref_logml += -0.5 * (T_eff * log(2π) + logdet(Cov) + dot(e, Cov \ e))
        end
        @test logml ≈ ref_logml atol = 1e-8

        # (3) Sigma is the FIXED diag(sigma_ar^2), not a posterior mode
        @test sigma ≈ Matrix(Diagonal(sar .^ 2))

        # (4) theta_cross = 1 reproduces the conjugate Minnesota asymmetry exactly: for a
        #     given regressor the cross/own prior SD ratio is sigma_m/sigma_j, which is the
        #     ONLY own-vs-cross asymmetry a Sigma-kron-V prior can express.
        A1, _ = MEM._litterman_prior_rows(1, N, lags, sar, y_bar, lam, th, 1.0, 0.0, 0.0)
        A2, _ = MEM._litterman_prior_rows(2, N, lags, sar, y_bar, lam, th, 1.0, 0.0, 0.0)
        col = 1 + 2                                    # lag 1, variable 2
        @test (1 / A1[col, col]) / (1 / A2[col, col]) ≈ sar[1] / sar[2] atol = 1e-12
        # ...and theta_cross scales exactly that ratio, which is the point of the feature.
        A1h, _ = MEM._litterman_prior_rows(1, N, lags, sar, y_bar, lam, th, 0.5, 0.0, 0.0)
        @test (1 / A1h[col, col]) ≈ 0.5 * (1 / A1[col, col]) atol = 1e-12
        # own-lag rows are untouched by theta_cross
        own_col = 1 + 1
        A1o, _ = MEM._litterman_prior_rows(1, N, lags, sar, y_bar, lam, th, 0.5, 0.0, 0.0)
        @test A1o[own_col, own_col] ≈ A1[own_col, own_col] atol = 1e-12

        # (5) theta_cross is identified: the criterion is not flat in it
        mls = [MEM._litterman_estimate(Y, lags, sar, lam, th, t, mu, al)[3]
               for t in (0.1, 0.5, 1.0, 3.0)]
        @test length(unique(round.(mls, digits=6))) == 4
        @test all(isfinite, mls)

        # (6) end-to-end through the public API
        rng2 = Random.MersenneTwister(6021)
        Yn = randn(rng2, 70, 4)
        for t in 2:70
            Yn[t, :] .+= 0.5 .* Yn[t-1, :]
        end
        Yn[68:70, 4] .= NaN
        ml_fit = nowcast_bvar(Yn, 3, 1; lags=2, max_iter=120, prior=:litterman)
        @test ml_fit.prior == :litterman
        @test isfinite(ml_fit.theta_cross) && ml_fit.theta_cross > 0
        @test isfinite(ml_fit.loglik) && ml_fit.loglik > -1e9
        @test !any(isnan, ml_fit.X_sm)
        @test occursin("Litterman", sprint(show, ml_fit))

        # (7) the conjugate prior must REFUSE theta_cross rather than silently ignore it
        cj_fit = nowcast_bvar(Yn, 3, 1; lags=2, max_iter=120)
        @test cj_fit.prior == :conjugate
        @test isnan(cj_fit.theta_cross)
        @test occursin("fixed by", sprint(show, cj_fit))
        @test_throws ArgumentError nowcast_bvar(Yn, 3, 1; lags=2, theta_cross0=0.5)
        @test_throws ArgumentError nowcast_bvar(Yn, 3, 1; lags=2, prior=:bogus)

        # (8) back-compat: the pre-#602 13-argument positional constructor still works
        legacy = MEM.NowcastBVAR{Float64}(
            cj_fit.X_sm, cj_fit.beta, cj_fit.sigma, cj_fit.lambda, cj_fit.theta,
            cj_fit.miu, cj_fit.alpha, cj_fit.lags, cj_fit.loglik, cj_fit.nM,
            cj_fit.nQ, cj_fit.data, cj_fit.converged)
        @test legacy.prior == :conjugate
        @test isnan(legacy.theta_cross)

        # (9) box-edge warning names the edge that was actually hit
        pinned_lo = MEM.NowcastBVAR{Float64}(
            ml_fit.X_sm, ml_fit.beta, ml_fit.sigma, ml_fit.lambda, ml_fit.theta,
            ml_fit.miu, exp(-5.0), ml_fit.lags, ml_fit.loglik, ml_fit.nM, ml_fit.nQ,
            ml_fit.data, false, exp(-5.0), :litterman)
        s_lo = sprint(show, pinned_lo)
        @test occursin("floor", s_lo)
        @test occursin("theta_cross", s_lo)
        @test !occursin("ceiling", s_lo)
        pinned_hi = MEM.NowcastBVAR{Float64}(
            ml_fit.X_sm, ml_fit.beta, ml_fit.sigma, exp(5.0), ml_fit.theta,
            ml_fit.miu, ml_fit.alpha, ml_fit.lags, ml_fit.loglik, ml_fit.nM, ml_fit.nQ,
            ml_fit.data, false, ml_fit.theta_cross, :litterman)
        s_hi = sprint(show, pinned_hi)
        @test occursin("ceiling", s_hi)
        @test occursin("lambda", s_hi)
    end

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

    @testset "Ragged-edge fill conditions on observed variables (T105 #204)" begin
        # 2-var VAR(1), B_1 = diag(0.5), strong contemporaneous innovation correlation
        # sigma_12 = 0.9. When var1 is observed and var2 missing, the Kalman-smoothed var2 equals
        # the conditional expectation sigma_21/sigma_11 * u1 (= 0.9 * observed); the old
        # interpolation + deterministic-projection fill ignored the observed var1 and gave 0.
        beta = zeros(3, 2); beta[2, 1] = 0.5; beta[3, 2] = 0.5
        sigma = [1.0 0.9; 0.9 1.0]
        Yp = zeros(6, 2); Yp[4, 1] = 3.0; Yp[4, 2] = NaN
        Xp = MacroEconometricModels._bvar_smooth_missing(Yp, beta, sigma, 1, 6)
        @test Xp[4, 1] == 3.0                    # observed entries preserved exactly
        @test Xp[4, 2] ≈ 2.7 atol=1e-3           # conditioned on the observed correlated variable
        Yn = zeros(6, 2); Yn[4, 1] = -3.0; Yn[4, 2] = NaN
        Xn = MacroEconometricModels._bvar_smooth_missing(Yn, beta, sigma, 1, 6)
        @test Xn[4, 2] ≈ -2.7 atol=1e-3          # sign follows the observed variable
        @test all(isfinite, Xp) && all(isfinite, Xn)
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

    @testset "Minnesota dummy design is one-nonzero-per-row (#571/#572)" begin
        # Replaces the old cross-variable-entry test: packing the own-lag and every cross
        # entry of an equation into ONE row restricts their SUM, so each lag block has rank
        # 1, X_d'X_d is singular and the NIW marginal likelihood collapses to the -1e10
        # sentinel. Cross-vs-own relative tightness is not a free hyperparameter of a
        # conjugate NIW prior (it is √(Σ_mm/Σ_jj)); theta is the lag-decay exponent (#572).
        rng = Random.MersenneTwister(500)
        N, lags = 3, 2
        Y0 = randn(rng, lags, N)
        sigma_ar = [1.0, 2.0, 3.0]
        lambda, theta = 0.2, 2.0
        k = 1 + N * lags

        Y_d, X_d = MacroEconometricModels._bvar_dummy_obs(Y0, lags, sigma_ar,
                                                          lambda, theta, 1.0, 2.0)

        # One restriction per Minnesota row, and the stacked design restricts every column
        for r in 1:(N * lags)
            @test count(!iszero, X_d[r, :]) == 1
        end
        @test rank(X_d) == k
        @test isfinite(logdet(X_d' * X_d))

        # Row (lag, i) scale = sigma_i * lag^theta / lambda; only lag 1 carries the RW mean
        @test X_d[1, 2] ≈ sigma_ar[1] * 1.0^theta / lambda
        @test Y_d[1, 1] ≈ X_d[1, 2]
        @test X_d[N + 1, 2 + N] ≈ sigma_ar[1] * 2.0^theta / lambda
        @test all(iszero, Y_d[(N + 1):(2 * N), :])

        # Larger theta shrinks the higher lags harder and leaves lag 1 untouched
        _, X_hi = MacroEconometricModels._bvar_dummy_obs(Y0, lags, sigma_ar,
                                                         lambda, 3.0, 1.0, 2.0)
        @test X_hi[N + 1, 2 + N] > X_d[N + 1, 2 + N]
        @test X_hi[1, 2] ≈ X_d[1, 2]

        # Closing inverse-Wishart scale block (Y = diag(sigma), X = 0): without it the
        # random-walk B solves every dummy row exactly and the prior SSR is singular.
        @test all(iszero, X_d[(end - N + 1):end, :])
        @test Y_d[(end - N + 1):end, :] ≈ diagm(sigma_ar)
    end

    @testset "Marginal likelihood is a smooth surface (#571/#572)" begin
        # 4-variable AR(0.7) panel: the log marginal likelihood must be finite at the
        # default start and vary smoothly in the hyperparameters. With the rank-1 lag
        # blocks, K_prior was singular, logdet_safe returned -Inf and EVERY evaluation
        # clamped to -1e10 — a flat plateau that pinned the optimizer at the box wall.
        rng = Random.MersenneTwister(4242)
        N, T_burn, T_obs = 4, 50, 120
        Yb = zeros(T_burn + T_obs, N)
        for t in 2:(T_burn + T_obs), j in 1:N
            Yb[t, j] = 0.7 * Yb[t - 1, j] + randn(rng)
        end
        Y = Yb[(T_burn + 1):end, :]
        sigma_ar = [std(diff(Y[:, j])) for j in 1:N]
        ml(lam, th) = MacroEconometricModels._bvar_estimate(Y, 2, sigma_ar, lam, th,
                                                            1.0, 2.0)[3]

        @test isfinite(ml(0.2, 1.0)) && ml(0.2, 1.0) > -1e9
        for lam in (0.02, 0.05, 0.2), th in (0.5, 1.0, 1.1)
            @test isfinite(ml(lam, th)) && ml(lam, th) > -1e9
        end
        # Smooth through theta = 1: the midpoint of the neighbours matches the value there
        @test ml(0.2, 1.0) ≈ (ml(0.2, 0.99) + ml(0.2, 1.01)) / 2 rtol=1e-4
        @test ml(0.2, 0.5) < ml(0.2, 1.1)   # monotone over this stretch, no spike at 1

        # The optimizer now reaches an interior optimum on this panel
        m = nowcast_bvar(Y, 3, 1; lags=2)
        @test m.converged
        @test all(x -> abs(log(x)) < 5 - 1e-3, (m.lambda, m.theta, m.miu, m.alpha))
    end

    @testset "StatsAPI interface" begin
        rng = Random.MersenneTwister(400)
        Y = randn(rng, 60, 4)
        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=10)

        @test loglikelihood(m) == m.loglik
        @test predict(m) == m.X_sm
        @test nobs(m) == 60
    end

    @testset "Handles near-singular data without NaN" begin
        # Construct data with near-collinear columns to stress the optimizer
        rng = Random.MersenneTwister(999)
        base = randn(rng, 80, 1)
        Y = hcat(base, base .+ 1e-8 * randn(rng, 80, 1),
                 base .+ 1e-8 * randn(rng, 80, 1))
        Y[75:80, 3] .= NaN

        m = nowcast_bvar(Y, 2, 1; lags=2, max_iter=30)
        @test m isa NowcastBVAR{Float64}
        @test isfinite(m.loglik)
        @test !any(isnan, m.X_sm)
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

    @testset "Bridge nowcasts the incomplete current quarter (T104 #203)" begin
        rng = Random.MersenneTwister(203)
        T_obs = 92                          # NOT a multiple of 3 → a partial current quarter
        Y = randn(rng, T_obs, 5)            # 3 monthly + 2 quarterly
        for t in 1:T_obs
            mod(t, 3) != 0 && (Y[t, 4:5] .= NaN)
        end
        m = nowcast_bridge(Y, 3, 2; lagM=1, lagQ=1, lagY=1)
        # ceil-division emits a row for the incomplete quarter (floor ÷ dropped it: 30 → 31).
        @test length(m.Y_nowcast) == cld(T_obs, 3)
        @test length(m.Y_nowcast) == 31
        @test isfinite(m.Y_nowcast[end])    # the current partial quarter is actually nowcast
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

# Shared T=60/nM=4/nQ=1/r=1 DFM fit (seed=700, heavier max_iter=20/thresh=1e-3 config
# valid for all reused News/Dispatch/Display testsets). Deduplicates 13 redundant EM refits.
_NC_Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=700)
_NC_M = nowcast_dfm(_NC_Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

@testset "News Decomposition" begin
    @testset "Basic news computation" begin
        Y = _NC_Y
        m = _NC_M

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
        Y = _NC_Y
        m = _NC_M

        # Same data → no news
        news = nowcast_news(Y, Y, m, 30; target_var=5)

        @test length(news.impact_news) == 0
        @test news.old_nowcast ≈ news.new_nowcast atol=1e-6
    end

    @testset "Decomposition identity" begin
        Y = _NC_Y
        m = _NC_M

        X_old = copy(Y)
        X_old[55:60, 1:2] .= NaN

        news = nowcast_news(Y, X_old, m, 55; target_var=5)

        total = news.new_nowcast - news.old_nowcast
        decomp = sum(news.impact_news) + news.impact_revision + news.impact_reestimation

        @test total ≈ decomp atol=1e-8
    end

    @testset "Joint news fully explains the revision (T094 #194)" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=80, nM=4, nQ=1, r=1, seed=717)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=30, thresh=1e-4)
        X_old = copy(Y)
        X_old[74:80, 1:3] .= NaN                 # withhold several recent releases across periods
        news = nowcast_news(Y, X_old, m, 78; target_var=5)
        @test length(news.impact_news) > 1
        # The joint weights B = Cov(F,I)·Var(I)^{-1} make the news explain the ENTIRE smoothed
        # revision, so the re-estimation residual is ~0. The old per-release scalar gains split
        # overlapping information wrongly and dumped a large residual into impact_reestimation.
        rev = news.new_nowcast - news.old_nowcast
        @test abs(news.impact_reestimation) <= 1e-6 * (abs(rev) + 1)
    end

    @testset "Revisions are not news (#573)" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=80, nM=4, nQ=1, r=1, seed=717)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=30, thresh=1e-4)

        # A pure revision: one cell observed in BOTH vintages changes value. The news
        # weights are derived for cells MISSING in the old vintage, so applying them here
        # misattributed part of the impact to re-estimation (#573). With parameters held
        # fixed the whole nowcast delta must land in impact_revision.
        X_old = copy(Y)
        X_new = copy(Y)
        X_new[79, 2] += 0.5
        news = nowcast_news(X_new, X_old, m, 79; target_var=5)
        delta = news.new_nowcast - news.old_nowcast

        @test length(news.impact_news) == 0
        @test news.impact_revision ≈ delta atol=1e-10
        @test abs(news.impact_reestimation) <= 1e-10
        # Non-vacuity guard: the revision actually moved the nowcast. The magnitude is
        # seed-stream dependent (Julia 1.10's RNG gives variable 2 a near-zero loading,
        # delta ≈ 1e-5), so the bar sits well above the 1e-10 identity atol, not at 1e-4.
        @test abs(delta) > 1e-8

        # Re-standardization noise (~1e-13) must not register as a revision — the exact
        # `!=` test flagged every observed cell and built a T·N-square revision system.
        X_noise = copy(Y) .+ 1e-13
        news_noise = nowcast_news(X_noise, Y, m, 79; target_var=5)
        @test news_noise.impact_revision == 0.0

        # News and revisions together: the decomposition identity still holds
        X_old2 = copy(Y)
        X_old2[76:80, 1:2] .= NaN
        X_new2 = copy(Y)
        X_new2[60, 3] += 0.3
        news2 = nowcast_news(X_new2, X_old2, m, 78; target_var=5)
        @test length(news2.impact_news) > 1
        @test news2.new_nowcast - news2.old_nowcast ≈
              sum(news2.impact_news) + news2.impact_revision + news2.impact_reestimation atol=1e-8
        @test abs(news2.impact_reestimation) <= 1e-6 * (abs(news2.new_nowcast - news2.old_nowcast) + 1)
    end

    @testset "Kalman lagged smoother cross-covariance recursion (T094 #194)" begin
        # Plag[j][:,:,t] = Cov(x_t, x_{t-j} | Y_T) must match the analytic joint-Gaussian
        # posterior covariance. The old j>=2 recursion J_{t-1}·Plag[j-1][t-1] was wrong; the
        # correct one is Plag[j-1][t]·J_{t-j}'.
        Random.seed!(7)
        A = [0.7 0.1; 0.0 0.5]; C = reshape([1.0, 0.5], 1, 2)
        Q = [0.3 0.0; 0.0 0.2]; R = reshape([0.4], 1, 1)
        x0 = [0.2, -0.1]; P0 = [1.0 0.2; 0.2 0.8]
        Tn, sd, N = 6, 2, 1
        y = randn(N, Tn)
        _, _, Plag, _ = MacroEconometricModels._kalman_smoother_lag(y, A, C, Q, R, x0, P0, Tn - 1)
        Vt = Vector{Matrix{Float64}}(undef, Tn); Vt[1] = A * P0 * A' + Q
        for t in 2:Tn; Vt[t] = A * Vt[t-1] * A' + Q; end
        SX = zeros(sd * Tn, sd * Tn)
        for t in 1:Tn, s in 1:Tn
            if t >= s
                Mm = copy(Vt[s]); for _ in 1:(t - s); Mm = A * Mm; end
                SX[(t-1)*sd+1:t*sd, (s-1)*sd+1:s*sd] = Mm
                SX[(s-1)*sd+1:s*sd, (t-1)*sd+1:t*sd] = Mm'
            end
        end
        H = zeros(N * Tn, sd * Tn); for t in 1:Tn; H[(t-1)*N+1:t*N, (t-1)*sd+1:t*sd] = C; end
        Rb = zeros(N * Tn, N * Tn); for t in 1:Tn; Rb[(t-1)*N+1:t*N, (t-1)*N+1:t*N] = R; end
        Spost = SX - SX * H' * inv(H * SX * H' + Rb) * H * SX
        for j in 1:3, t in (j + 1):Tn
            blk = Spost[(t-1)*sd+1:t*sd, (t-j-1)*sd+1:(t-j)*sd]
            @test Plag[j][:, :, t] ≈ blk atol=1e-9
        end
    end

    @testset "Input validation" begin
        Y = _NC_Y
        m = _NC_M

        @test_throws ArgumentError nowcast_news(Y, Y[1:50, :], m, 30)  # size mismatch
        @test_throws ArgumentError nowcast_news(Y, Y, m, 0)  # out of range
        @test_throws ArgumentError nowcast_news(Y, Y, m, 30; target_var=0)  # out of range
        @test_throws ArgumentError nowcast_news(Y, Y, m, 30; groups=[1,1,2,2,3], group_names=["A", "B"])
    end

    @testset "Group impacts" begin
        Y = _NC_Y
        m = _NC_M

        X_old = copy(Y)
        X_old[58:60, 1:2] .= NaN

        groups = [1, 1, 2, 2, 3]  # 3 groups
        news = nowcast_news(Y, X_old, m, 58; target_var=5, groups=groups)

        @test length(news.group_impacts) == 3

        # group_names auto-generated
        @test length(news.group_names) == 3
        @test news.group_names[1] == "Group 1"
        @test news.group_names[3] == "Group 3"

        # group_names explicit
        news2 = nowcast_news(Y, X_old, m, 58; target_var=5, groups=groups,
                             group_names=["Ind. Prod.", "Retail", "GDP"])
        @test news2.group_names == ["Ind. Prod.", "Retail", "GDP"]
    end

    @testset "Default group_names without groups" begin
        Y = _NC_Y
        m = _NC_M

        X_old = copy(Y)
        X_old[58:60, 1:2] .= NaN

        news = nowcast_news(Y, X_old, m, 58; target_var=5)
        @test length(news.group_names) == 5  # one per variable
        @test news.group_names[1] == "Var1"
        @test news.group_names[5] == "Var5"
    end
end

# =============================================================================
# 6. Nowcast and Forecast Dispatch
# =============================================================================

@testset "Nowcast and Forecast" begin
    @testset "nowcast() DFM" begin
        Y = _NC_Y
        m = _NC_M

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
        Y = _NC_Y
        m = _NC_M

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
        Y = _NC_Y
        m = _NC_M

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
        Y = _NC_Y
        m = _NC_M

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
        Y = _NC_Y
        m = _NC_M
        result = nowcast(m)

        io = IOBuffer()
        show(io, result)
        s = String(take!(io))
        @test contains(s, "Nowcast Result")
        @test contains(s, "DFM")
    end

    @testset "NowcastNews show" begin
        Y = _NC_Y
        m = _NC_M

        X_old = copy(Y)
        X_old[58:60, 1] .= NaN

        news = nowcast_news(Y, X_old, m, 58; target_var=5)

        io = IOBuffer()
        show(io, news)
        s = String(take!(io))
        @test contains(s, "News Decomposition")
    end

    @testset "report() dispatch" begin
        Y = _NC_Y
        m = _NC_M

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
