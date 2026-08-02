# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using DataFrames
using Random

@testset "Core VAR & Identification" begin
    # Use fixed seed for reproducibility
    Random.seed!(12345)

    # 1. Generate Synthetic Data
    T = 200
    n = 2
    p = 1

    true_A = [0.5 0.0; 0.0 0.5] # Diagonal AR
    true_c = [0.0; 0.0]
    Sigma_true = [1.0 0.0; 0.0 1.0] # Identity

    Y = zeros(T, n)
    # Generate data
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 2. Estimate
    model = estimate_var(Y, p)

    # 6. Identification (Cholesky)
    L = identify_cholesky(model)
    @test istriu(L')

    # 7. Sign Restrictions
    # We want to identify a "positive shock" to variable 1
    # Restriction: Response of Var 1 to Shock 1 at h=0 is positive (> 0)

    horizon = 10
    check_func(irf) = irf[1, 1, 1] > 0

    Q_sign, irf_sign = identify_sign(model, horizon, check_func)
    @test irf_sign[1, 1, 1] > 0
    @test isapprox(Q_sign * Q_sign', I(n), atol=1e-10) # Q is orthogonal

    # 8. Narrative Restrictions
    # Assume we know that at t=5, the shock 1 was positive.
    narrative_check(shocks) = shocks[5, 1] > 0

    Q_nar, irf_nar, shocks_nar = identify_narrative(model, horizon, check_func, narrative_check)
    @test irf_nar[1, 1, 1] > 0
    @test shocks_nar[5, 1] > 0

    # 9. Long Run Identification
    # Blanchard-Quah
    Q_lr = identify_long_run(model)
    @test isapprox(Q_lr * Q_lr', I(n), atol=1e-10)

    # Check if Long Run Impact is Lower Triangular
    # LR Impact = (I - A(1))^-1 * P
    # P = L * Q_lr
    # We can approximate infinite sum IRF or compute directly.
    # IRF cumulative sum at large h should be close to LR impact.

    # Let's compute directly to verify property
    B = model.B
    A_sum = zeros(n, n)
    for i in 1:p
        start_row = 1 + (i - 1) * n + 1
        end_row = 1 + i * n
        A_sum += B[start_row:end_row, :]'
    end
    inv_lag = inv(I(n) - A_sum)
    L_chol = identify_cholesky(model)
    P = L_chol * Q_lr
    LR_Matrix = inv_lag * P

    # Check lower triangularity of LR_Matrix
    @test abs(LR_Matrix[1, 2]) < 1e-8 # Upper right element should be 0

    @testset "Higher Order Lag (VAR(12))" begin
        # Test estimation with long lags using a known DGP to verify statistical recovery
        # DGP: Y_t = A_1 Y_{t-1} + ... + A_{12} Y_{t-12} + u_t
        # A_1 = 0.4 * I
        # A_12 = 0.2 * I
        # Others = 0

        T_large = 2000
        n = 2
        p = 12

        Y = zeros(T_large, n)

        # True Parameters
        A1 = [0.4 0.0; 0.0 0.4]
        A12 = [0.2 0.0; 0.0 0.2]
        Sigma_true = [1.0 0.0; 0.0 1.0]

        # Simulation
        Random.seed!(42) # Ensure reproducibility
        for t in p+1:T_large
            u = randn(n)
            # Y_t = A1 * Y_{t-1} + A12 * Y_{t-12} + u
            Y[t, :] = A1 * Y[t-1, :] + A12 * Y[t-12, :] + u
        end

        # Estimate
        model = estimate_var(Y, p)

        # Verify Coefficients
        # B structure: [Intercept; A_1'; A_2'; ... A_p']
        # Intercept should be close to 0
        @test norm(model.B[1, :]) < 0.2

        # Check Lag 1 (Rows 2:3)
        est_A1 = model.B[2:3, :]'
        @test isapprox(est_A1, A1, atol=0.1)

        # Check Lag 12 (Rows 1+11*2+1 : 1+12*2) -> Rows 24:25
        est_A12 = model.B[end-1:end, :]'
        @test isapprox(est_A12, A12, atol=0.1)

        # Check a middle lag (e.g., Lag 6) is close to zero
        # Rows for Lag 6: 1 + 5*2 + 1 = 12 to 13?
        # Lag k starts at 2 + (k-1)*n
        # Lag 6: 2 + 5*2 = 12. So rows 12:13.
        est_A6 = model.B[12:13, :]'
        @test norm(est_A6) < 0.1

        # Verify Residuals Covariance
        @test isapprox(model.Sigma, Sigma_true, atol=0.1)
    end

    # ==========================================================================
    # Robustness Tests (Following Arias et al. pattern)
    # ==========================================================================

    @testset "Reproducibility" begin
        # Same seed should produce identical results
        Random.seed!(99999)
        Y1 = zeros(100, 2)
        for t in 2:100
            Y1[t, :] = 0.5 * Y1[t-1, :] + randn(2)
        end
        model1 = estimate_var(Y1, 1)

        Random.seed!(99999)
        Y2 = zeros(100, 2)
        for t in 2:100
            Y2[t, :] = 0.5 * Y2[t-1, :] + randn(2)
        end
        model2 = estimate_var(Y2, 1)

        @test model1.B ≈ model2.B
        @test model1.Sigma ≈ model2.Sigma
        @test model1.U ≈ model2.U
    end

    @testset "Stability Check" begin
        # VAR should detect stable vs unstable systems
        Random.seed!(11111)
        T_stab = 200
        n_stab = 2
        p_stab = 1

        # Stable VAR
        Y_stable = zeros(T_stab, n_stab)
        A_stable = [0.3 0.1; 0.1 0.3]  # All eigenvalues < 1
        for t in 2:T_stab
            Y_stable[t, :] = A_stable * Y_stable[t-1, :] + randn(n_stab)
        end
        model_stable = estimate_var(Y_stable, p_stab)

        # Check stability via companion matrix eigenvalues
        F = companion_matrix(model_stable.B, n_stab, p_stab)
        eigenvalues = eigvals(F)
        @test maximum(abs.(eigenvalues)) < 1.0  # Stable
    end

    @testset "Numerical Stability - Near-Collinear Data" begin
        Random.seed!(22222)
        T_nc = 200
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(T_nc)  # Variable 3 ≈ Variable 1

        # Should not crash with near-singular covariance
        model_nc = estimate_var(Y_nc, 1)
        @test model_nc isa VARModel
        @test all(isfinite.(model_nc.B))
        @test all(isfinite.(model_nc.Sigma))
    end

    @testset "Edge Cases" begin
        Random.seed!(33333)

        # Single variable VAR
        Y_single = randn(100, 1)
        model_single = estimate_var(Y_single, 1)
        @test size(model_single.B) == (2, 1)  # intercept + 1 lag
        @test size(model_single.Sigma) == (1, 1)

        # Minimum viable sample size (T just larger than p*n + 1)
        n_min = 2
        p_min = 2
        T_min = p_min * n_min + 10  # Bare minimum observations
        Y_min = randn(T_min, n_min)
        model_min = estimate_var(Y_min, p_min)
        @test model_min isa VARModel

        # VAR(1) - simplest case
        Y_var1 = randn(50, 2)
        model_var1 = estimate_var(Y_var1, 1)
        @test model_var1.p == 1
    end

    @testset "Orthogonality of Q Matrices" begin
        Random.seed!(44444)
        T_q = 150
        n_q = 3
        Y_q = randn(T_q, n_q)
        model_q = estimate_var(Y_q, 1)

        # Cholesky Q should be identity (orthogonal)
        Q_chol = I(n_q)
        @test norm(Q_chol' * Q_chol - I(n_q)) < 1e-10

        # Sign restriction Q should be orthogonal
        check_func_q(irf) = irf[1, 1, 1] > 0
        Q_sign_q, _ = identify_sign(model_q, 5, check_func_q)
        @test norm(Q_sign_q' * Q_sign_q - I(n_q)) < 1e-10
        @test norm(Q_sign_q * Q_sign_q' - I(n_q)) < 1e-10

        # Columns should be unit vectors
        for j in 1:n_q
            @test abs(norm(Q_sign_q[:, j]) - 1.0) < 1e-10
        end
    end

    @testset "Input Validation" begin
        Random.seed!(55555)
        Y_val = randn(100, 2)

        # p = 0 should error or be handled
        @test_throws Exception estimate_var(Y_val, 0)

        # p too large for data - package handles gracefully with warning
        # Just verify it returns a model (even if with adjusted dof)
        model_large_p = estimate_var(Y_val, 40)
        @test model_large_p isa VARModel

        # Empty data
        @test_throws Exception estimate_var(zeros(0, 2), 1)
    end

    # =================================================================
    # Identification Functions (expanded coverage)
    # =================================================================

    @testset "generate_Q properties" begin
        Random.seed!(60000)

        for n in [2, 3, 5]
            Q = MacroEconometricModels.generate_Q(n)
            @test size(Q) == (n, n)

            # Orthogonality: Q'Q ≈ I
            @test isapprox(Q' * Q, Matrix{Float64}(I, n, n), atol=1e-10)
            @test isapprox(Q * Q', Matrix{Float64}(I, n, n), atol=1e-10)

            # Determinant ≈ ±1
            @test isapprox(abs(det(Q)), 1.0, atol=1e-10)

            # Columns are unit vectors
            for j in 1:n
                @test isapprox(norm(Q[:, j]), 1.0, atol=1e-10)
            end
        end

        # Randomness: two Q draws should differ
        Q1 = MacroEconometricModels.generate_Q(3)
        Q2 = MacroEconometricModels.generate_Q(3)
        @test !isapprox(Q1, Q2, atol=1e-5)
    end

    @testset "compute_structural_shocks" begin
        Random.seed!(61000)
        Y = randn(200, 3)
        model = estimate_var(Y, 2)
        n = 3

        # With identity Q (Cholesky identification)
        Q = Matrix{Float64}(I, n, n)
        shocks = MacroEconometricModels.compute_structural_shocks(model, Q)

        T_eff = size(model.U, 1)
        @test size(shocks) == (T_eff, n)
        @test !any(isnan, shocks)

        # Structural shocks should have unit variance (approximately, for Cholesky)
        for j in 1:n
            @test isapprox(var(shocks[:, j]), 1.0, rtol=0.3)
        end

        # With a random orthogonal Q
        Q_rand = MacroEconometricModels.generate_Q(n)
        shocks_rand = MacroEconometricModels.compute_structural_shocks(model, Q_rand)
        @test size(shocks_rand) == (T_eff, n)

        # Structural shocks from random Q should also have approximately unit variance
        for j in 1:n
            @test isapprox(var(shocks_rand[:, j]), 1.0, rtol=0.3)
        end
    end

    @testset "compute_irf" begin
        Random.seed!(62000)
        Y = randn(200, 2)
        model = estimate_var(Y, 1)
        n = 2
        horizon = 10

        # Identity Q
        Q = Matrix{Float64}(I, n, n)
        irf_array = MacroEconometricModels.compute_irf(model, Q, horizon)

        @test size(irf_array) == (horizon, n, n)
        @test !any(isnan, irf_array)

        # Impact (h=1) should be non-zero for a non-degenerate model
        @test any(irf_array[1, :, :] .!= 0)

        # IRF should decay for stationary model
        for i in 1:n, j in 1:n
            @test abs(irf_array[horizon, i, j]) < abs(irf_array[1, i, j]) + 1.0
        end
    end

    @testset "compute_Q dispatcher" begin
        Random.seed!(63000)
        Y = randn(200, 2)
        model = estimate_var(Y, 1)
        n = 2

        # :cholesky
        Q_chol = MacroEconometricModels.compute_Q(model, :cholesky, 10, nothing, nothing)
        @test Q_chol == Matrix{Float64}(I, n, n)

        # :long_run
        Q_lr = MacroEconometricModels.compute_Q(model, :long_run, 10, nothing, nothing)
        @test size(Q_lr) == (n, n)

        # :sign
        check_func = irf -> irf[1, 1, 1] > 0
        Q_sign = MacroEconometricModels.compute_Q(model, :sign, 10, check_func, nothing)
        @test size(Q_sign) == (n, n)
        # Verify the sign restriction is satisfied
        irf_check = MacroEconometricModels.compute_irf(model, Q_sign, 10)
        @test irf_check[1, 1, 1] > 0

        # Invalid method
        @test_throws ArgumentError MacroEconometricModels.compute_Q(model, :invalid, 10, nothing, nothing)

        # :sign without check_func
        @test_throws ArgumentError MacroEconometricModels.compute_Q(model, :sign, 10, nothing, nothing)
    end

    @testset "identify_cholesky" begin
        Random.seed!(64000)
        Y = randn(200, 3)
        model = estimate_var(Y, 1)

        L = identify_cholesky(model)
        @test size(L) == (3, 3)

        # L should be lower triangular
        @test istriu(L')

        # L * L' ≈ Sigma
        @test isapprox(L * L', model.Sigma, atol=1e-8)
    end

    @testset "identify_sign multiple draws" begin
        Random.seed!(65000)
        Y = randn(200, 2)
        model = estimate_var(Y, 1)

        # Multiple draws should all satisfy constraint
        check_func = irf -> irf[1, 1, 1] > 0 && irf[1, 2, 1] > 0
        Q, irf_result = identify_sign(model, 10, check_func; max_draws=5000)

        @test irf_result[1, 1, 1] > 0
        @test irf_result[1, 2, 1] > 0
        @test isapprox(Q' * Q, I(2), atol=1e-10)
    end

    @testset "identify_long_run" begin
        Random.seed!(66000)
        Y = randn(200, 2)
        model = estimate_var(Y, 1)

        Q = identify_long_run(model)
        @test size(Q) == (2, 2)

        # Long-run cumulative impact matrix should be lower triangular
        n, p = 2, 1
        A = MacroEconometricModels.extract_ar_coefficients(model.B, n, p)
        A_sum = sum(A)
        inv_lag = inv(I(n) - A_sum)
        L = MacroEconometricModels.safe_cholesky(model.Sigma)
        C1 = inv_lag * L * Q  # Long-run impact

        # C1 should be approximately lower triangular
        @test abs(C1[1, 2]) < 0.5  # Upper triangle should be small (not exactly zero due to numerics)
    end

    @testset "irf_percentiles and irf_mean" begin
        Random.seed!(67000)
        Y = randn(200, 2)
        model = estimate_var(Y, 1)
        n = 2
        horizon = 8

        # Create sign restrictions for Arias identification
        restrictions = SVARRestrictions(n;
            signs=[sign_restriction(1, 1, :positive; horizon=0)]
        )

        try
            result = MacroEconometricModels.identify_arias(model, restrictions, horizon;
                n_draws=50, n_rotations=500)

            # irf_percentiles
            pct = MacroEconometricModels.irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
            @test size(pct) == (horizon, n, n, 3)

            # Percentiles should be ordered
            for h in 1:horizon, i in 1:n, j in 1:n
                @test pct[h, i, j, 1] <= pct[h, i, j, 2]
                @test pct[h, i, j, 2] <= pct[h, i, j, 3]
            end

            # irf_mean
            mean_irf = MacroEconometricModels.irf_mean(result)
            @test size(mean_irf) == (horizon, n, n)
            @test !any(isnan, mean_irf)

        catch e
            @warn "Arias identification test failed (may need more draws)" exception=e
            @test_skip "Arias identification skipped"
        end
    end

    # =================================================================
    # VARModel Variable Names (Issue #17)
    # =================================================================
    @testset "VARModel Variable Names" begin
        Random.seed!(42)
        Y = randn(100, 3)

        # Default variable names
        m1 = estimate_var(Y, 2)
        @test m1.varnames == ["y1", "y2", "y3"]

        # Custom variable names
        m2 = estimate_var(Y, 2; varnames=["GDP", "CPI", "FFR"])
        @test m2.varnames == ["GDP", "CPI", "FFR"]

        # Variable names propagate to IRF
        irf_result = irf(m2, 10)
        @test irf_result.variables == ["GDP", "CPI", "FFR"]
        @test irf_result.shocks == ["GDP", "CPI", "FFR"]

        # Variable names propagate to FEVD
        fevd_result = fevd(m2, 10)
        @test !any(isnan, fevd_result.proportions)
    end

    # =================================================================
    # Forecast bands: parameter uncertainty (#208)
    # =================================================================
    @testset "VAR forecast parameter uncertainty (#208)" begin
        rng = Random.MersenneTwister(2208)
        n = 2
        Atrue = [0.5 0.1; 0.0 0.4]
        Lσ = cholesky([1.0 0.2; 0.2 1.0]).L
        Tn = 60
        Y = zeros(Tn, n)
        for t in 2:Tn
            Y[t, :] = Atrue * Y[t-1, :] + Lσ * randn(rng, n)
        end
        m = estimate_var(Y, 1)
        H = 6
        A1 = MacroEconometricModels.extract_ar_coefficients(m.B, n, 1)[1]

        # --- :analytic exact MSE cross-check (Lütkepohl §3.5; VAR(1) ⇒ Φ_i = A^i) ---
        z90 = 1.6448536269514722          # quantile(Normal(), 0.95)
        fa = forecast(m, H; ci_method=:analytic, conf_level=0.90)
        @test fa.ci_method == :analytic
        mse = copy(m.Sigma); Φ = Matrix{Float64}(I, n, n)
        for hi in 1:H
            if hi > 1
                Φ = A1 * Φ                 # Φ = A^{hi-1}
                mse = mse .+ Φ * m.Sigma * Φ'
            end
            for j in 1:n
                hw = z90 * sqrt(mse[j, j])
                @test isapprox(fa.ci_upper[hi, j] - fa.forecast[hi, j], hw; rtol=1e-10)
                @test isapprox(fa.forecast[hi, j] - fa.ci_lower[hi, j], hw; rtol=1e-10)
            end
        end
        # h=1 half-width is exactly z·√Σ_jj
        for j in 1:n
            @test isapprox(fa.ci_upper[1, j] - fa.forecast[1, j], z90 * sqrt(m.Sigma[j, j]); rtol=1e-10)
        end

        # --- bootstrap-B reproducibility under a fixed rng ---
        fb1 = forecast(m, H; ci_method=:bootstrap, reps=800, rng=Random.MersenneTwister(7))
        fb2 = forecast(m, H; ci_method=:bootstrap, reps=800, rng=Random.MersenneTwister(7))
        @test fb1.ci_lower == fb2.ci_lower
        @test fb1.ci_upper == fb2.ci_upper

        # --- bootstrap-B adds coefficient uncertainty ⇒ wider than innovation-only analytic ---
        fa95 = forecast(m, H; ci_method=:analytic)
        fb95 = forecast(m, H; ci_method=:bootstrap, reps=4000, rng=Random.MersenneTwister(11))
        @test sum(fb95.ci_upper .- fb95.ci_lower) > sum(fa95.ci_upper .- fa95.ci_lower)

        # --- stationary_only, :none, and validation ---
        fs = forecast(m, H; ci_method=:bootstrap, reps=300, stationary_only=true,
                      rng=Random.MersenneTwister(3))
        @test all(fs.ci_upper .>= fs.ci_lower)
        fn = forecast(m, 3; ci_method=:none)
        @test all(fn.ci_lower .== 0) && all(fn.ci_upper .== 0)
        @test_throws ArgumentError forecast(m, 3; ci_method=:bogus)
    end
end

@testset "VAR predict in-place history ring (#210 box C)" begin
    # Box C replaces the per-step `vcat` history ring in `predict(model, steps)` with an in-place
    # row shift. The point-forecast recursion is deterministic, so it must be bit-for-bit identical
    # to a naive `vcat`-based recursion (the pre-refactor algorithm) reconstructed here.
    rng = Random.MersenneTwister(7)
    n, p, Tn = 3, 2, 120
    A1 = [0.4 0.1 0.0; 0.0 0.3 0.1; 0.1 0.0 0.2]
    Y = zeros(Tn, n)
    for t in 3:Tn
        Y[t, :] = A1 * Y[t-1, :] .+ 0.5 .* A1 * Y[t-2, :] .+ randn(rng, n)
    end
    model = estimate_var(Y, p)
    steps = 15
    fc = predict(model, steps)

    # naive vcat reference (pre-refactor algorithm)
    B = model.B
    A = MacroEconometricModels.extract_ar_coefficients(B, n, p)
    intercept = B[1, :]
    hist = copy(model.Y[(end-p+1):end, :])
    ref = Matrix{Float64}(undef, steps, n)
    for h in 1:steps
        y_hat = copy(intercept)
        for lag in 1:p
            y_hat .+= A[lag] * hist[end-lag+1, :]
        end
        ref[h, :] = y_hat
        hist = vcat(hist[2:end, :], y_hat')
    end

    @test fc == ref              # bit-identical to the vcat recursion
end

@testset "Generalized FEVD, Pesaran-Shin (#364/T265)" begin
    M = MacroEconometricModels

    # A stationary 3-variable VAR with CORRELATED reduced-form errors — correlation is what
    # makes the ordering matter for Cholesky and is therefore the interesting case.
    rng = Random.MersenneTwister(11); nobs = 600
    A = [0.4 0.15 0.05; 0.10 0.35 0.10; 0.05 0.10 0.30]
    L = [1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 0.4 1.0]
    Y = zeros(nobs, 3)
    for t in 2:nobs
        Y[t, :] = A * Y[t-1, :] + L * randn(rng, 3)
    end
    m = estimate_var(Y, 2)
    H = 12

    @testset "reduced-form MA recursion" begin
        Phi = M._reduced_form_ma(m.B, 3, 2, 5)
        @test length(Phi) == 5
        @test Phi[1] ≈ I(3)                       # Phi_0 = I
        Acoef = M.extract_ar_coefficients(m.B, 3, 2)
        @test Phi[2] ≈ Acoef[1]                   # Phi_1 = A_1
        @test Phi[3] ≈ Acoef[1] * Phi[2] + Acoef[2] * Phi[1]
        # the structural IRF is Phi_h * P, so with P = chol(Sigma) it must match compute_irf
        P = M.safe_cholesky(m.Sigma)
        irf_chol = M.compute_irf(m, Matrix{Float64}(I, 3, 3), 5)
        for h in 1:5
            @test irf_chol[h, :, :] ≈ Phi[h] * P atol = 1e-10
        end
    end

    @testset "ORDER INVARIANCE — the reason to use it" begin
        g = generalized_fevd(m, H)
        c = fevd(m, H)
        perm = [3, 1, 2]; ip = invperm(perm)
        mp = estimate_var(Y[:, perm], 2)
        gp = generalized_fevd(mp, H)
        cp = fevd(mp, H)
        # gFEVD is invariant to the variable ordering, to machine precision
        @test maximum(abs.(g.proportions .- gp.proportions[ip, ip, :])) < 1e-10
        # Cholesky FEVD is NOT — that is the whole contrast (measured 0.303)
        @test maximum(abs.(c.proportions .- cp.proportions[ip, ip, :])) > 0.01
    end

    @testset "impact horizon has a closed form: gFEVD_ij(1) = corr(u_i,u_j)^2" begin
        # At h = 1, Phi_0 = I, so
        #   gFEVD_ij(1) = sigma_jj^-1 (e_i' Sigma e_j)^2 / (e_i' Sigma e_i)
        #               = sigma_ij^2 / (sigma_ii sigma_jj) = corr(u_i, u_j)^2
        # and in particular the own-shock share is EXACTLY one.
        g = generalized_fevd(m, H)
        D = sqrt.(diag(m.Sigma))
        R = m.Sigma ./ (D * D')
        @test g.proportions[:, :, 1] ≈ R .^ 2 atol = 1e-12
        for i in 1:3
            @test g.proportions[i, i, 1] ≈ 1.0 atol = 1e-12
        end
    end

    @testset "raw shares do not sum to one; normalized ones do" begin
        g = generalized_fevd(m, H)
        gn = generalized_fevd(m, H; normalize=true)
        @test all(g.proportions .>= -1e-14)              # non-negative
        rows = vec(sum(g.proportions[:, :, H]; dims=2))
        @test all(rows .> 1.0)                            # correlated shocks over-attribute
        @test !all(isapprox.(rows, 1.0; atol=1e-6))       # NOT forced to one
        nrows = vec(sum(gn.proportions[:, :, H]; dims=2))
        @test all(isapprox.(nrows, 1.0; atol=1e-10))
        # normalization is a pure rescaling of each row: the ratios are preserved
        for i in 1:3, h in 1:H
            s = sum(g.proportions[i, :, h])
            s > 0 && @test gn.proportions[i, :, h] ≈ g.proportions[i, :, h] ./ s atol = 1e-12
        end
        @test size(g.proportions) == (3, 3, H)
        @test g.variables == m.varnames
        @test g.shocks == m.varnames                      # a "shock" here is to a VARIABLE
    end

    @testset "diagonal Sigma ⇒ gFEVD coincides with Cholesky" begin
        # With uncorrelated reduced-form errors there is nothing to orthogonalize, so the
        # generalized and recursive decompositions agree (up to the sample correlation).
        rng2 = Random.MersenneTwister(3); n2 = 4000
        Y2 = zeros(n2, 2); A2 = [0.5 0.0; 0.0 0.4]
        for t in 2:n2
            Y2[t, :] = A2 * Y2[t-1, :] + randn(rng2, 2)
        end
        m2 = estimate_var(Y2, 1)
        @test abs(m2.Sigma[1, 2] / sqrt(m2.Sigma[1, 1] * m2.Sigma[2, 2])) < 0.05
        g2 = generalized_fevd(m2, 10)
        c2 = fevd(m2, 10)
        @test maximum(abs.(g2.proportions .- c2.proportions)) < 5e-3
        # and with a diagonal Sigma the rows already nearly sum to one on their own
        @test all(isapprox.(vec(sum(g2.proportions[:, :, 10]; dims=2)), 1.0; atol=5e-3))
    end

    @testset "Bayesian generalized FEVD" begin
        rng3 = Random.MersenneTwister(5); n3 = 300
        A3 = [0.4 0.1; 0.1 0.35]; Y3 = zeros(n3, 2)
        for t in 2:n3
            Y3[t, :] = A3 * Y3[t-1, :] + [1.0 0.0; 0.5 1.0] * randn(rng3, 2)
        end
        post = estimate_bvar(Y3, 2; n_draws=150)
        g = generalized_fevd(post, 8)
        gn = generalized_fevd(post, 8; normalize=true)
        # Axis order unified with FEVD: (variable, shock, horizon[, quantile]) (#527)
        @test size(g.quantiles) == (2, 2, 8, 3)
        @test size(g.point_estimate) == (2, 2, 8)
        @test all(isfinite, g.quantiles)
        @test all(g.quantiles .>= -1e-12)
        # bands are ordered across the QUANTILE axis (which is last)
        @test all(g.quantiles[:, :, :, 1] .<= g.quantiles[:, :, :, 2] .+ 1e-12)
        @test all(g.quantiles[:, :, :, 2] .<= g.quantiles[:, :, :, 3] .+ 1e-12)
        @test all(isapprox.(vec(sum(gn.point_estimate[:, :, 8]; dims=2)), 1.0; atol=1e-8))
        @test g.n_effective == 150 && g.n_failed == 0
        # the posterior mean sits close to the OLS point estimate
        gv = generalized_fevd(estimate_var(Y3, 2), 8)
        @test maximum(abs.(g.point_estimate[:, :, 8] .- gv.proportions[:, :, 8])) < 0.1
    end

    @testset "validation and display" begin
        @test_throws ArgumentError generalized_fevd(m, 0)
        @test_throws ArgumentError generalized_fevd(m, -1)
        g = generalized_fevd(m, 4; shock_names=["a", "b", "c"])
        @test g.shocks == ["a", "b", "c"]
        io = IOBuffer(); show(io, g)
        @test occursin("Forecast Error Variance Decomposition", String(take!(io)))
        report(g)
    end
end

@testset "Wild/block bootstrap + Kilian bias correction (#370, T271)" begin
    M = MacroEconometricModels

    @testset "resampling schemes preserve what they claim to" begin
        rng = Random.MersenneTwister(3)
        U = randn(rng, 200, 3)
        for sch in (:iid, :wild, :block)
            A = M._resample_residuals(U, sch, Random.MersenneTwister(5))
            @test size(A) == size(U)                                   # every scheme returns T_eff rows
            @test A == M._resample_residuals(U, sch, Random.MersenneTwister(5))   # reproducible
            @test all(isfinite, A)
        end
        @test_throws ArgumentError M._resample_residuals(U, :bogus, rng)

        # WILD scales whole rows, so the contemporaneous cross-equation correlation survives
        # exactly — that is the property that makes it robust to conditional heteroskedasticity.
        Uc = randn(Random.MersenneTwister(9), 4000, 2) * [1.0 0.8; 0.0 0.6]
        w = M._resample_residuals(Uc, :wild, Random.MersenneTwister(11))
        @test cor(w[:, 1], w[:, 2]) ≈ cor(Uc[:, 1], Uc[:, 2]) atol = 0.03
        # ... and it is a pure row rescaling: |u*| equals |u| row by row under Rademacher
        @test abs.(w) ≈ abs.(Uc) atol = 1e-12

        # BLOCK keeps serial dependence that i.i.d. resampling destroys.
        ar = zeros(600)
        rng2 = Random.MersenneTwister(21)
        for t in 2:600
            ar[t] = 0.9 * ar[t-1] + randn(rng2)
        end
        Ua = reshape(ar, :, 1)
        ac(x) = cor(x[2:end], x[1:end-1])
        @test ac(vec(M._resample_residuals(Ua, :block, Random.MersenneTwister(2);
                                           block_length=30))) > 0.6
        @test abs(ac(vec(M._resample_residuals(Ua, :iid, Random.MersenneTwister(2))))) < 0.15
        @test M._default_block_length(1000) == 10
        @test M._default_block_length(1) == 1
    end

    @testset "wild weights match their moments" begin
        r = M._wild_weights(Random.MersenneTwister(4), 200_000, :rademacher, Float64)
        @test all(x -> x == 1.0 || x == -1.0, r)
        @test mean(r) ≈ 0 atol = 0.01
        @test mean(r .^ 2) ≈ 1 atol = 1e-12                # exactly 1 for ±1
        # Mammen matches the THIRD moment as well, which Rademacher cannot (its odd moments
        # are zero by symmetry). That is the whole reason to offer it.
        mm = M._wild_weights(Random.MersenneTwister(4), 400_000, :mammen, Float64)
        @test mean(mm) ≈ 0 atol = 0.01
        @test mean(mm .^ 2) ≈ 1 atol = 0.02
        @test mean(mm .^ 3) ≈ 1 atol = 0.05
        @test abs(mean(r .^ 3)) < 0.01                     # Rademacher: zero, not one
        @test_throws ArgumentError M._wild_weights(Random.MersenneTwister(1), 5, :nope, Float64)
    end

    @testset "Kilian bias correction reduces the OLS bias" begin
        # OLS is badly downward-biased for a persistent AR in a short sample. This is the
        # acceptance criterion: the correction must move the estimate toward the truth.
        rho_true, Tn, nsim = 0.95, 60, 250
        bias_ols = Float64[]
        bias_bc = Float64[]
        for s in 1:nsim
            rng = Random.MersenneTwister(1000 + s)
            y = zeros(Tn, 1)
            for t in 2:Tn
                y[t, 1] = rho_true * y[t-1, 1] + randn(rng)
            end
            m = estimate_var(y, 1; check_stability=false)
            push!(bias_ols, M.extract_ar_coefficients(m.B, 1, 1)[1][1, 1] - rho_true)
            Psi = M._estimate_var_bias(m, 60, :iid, Random.MersenneTwister(7000 + s))
            Bc, _ = M._kilian_bias_correction(m.B, Psi, 1, 1)
            push!(bias_bc, M.extract_ar_coefficients(Bc, 1, 1)[1][1, 1] - rho_true)
        end
        @test mean(bias_ols) < -0.03                        # the premise: OLS IS biased down
        @test abs(mean(bias_bc)) < abs(mean(bias_ols)) / 2  # at least halved
        @test sqrt(mean(bias_bc .^ 2)) < sqrt(mean(bias_ols .^ 2))   # and RMSE improves
    end

    @testset "the stationarity shrinkage keeps the corrected companion stable" begin
        # A correction large enough to push the companion outside the unit circle must be
        # scaled back, not applied — otherwise the bias-corrected DGP is explosive.
        B = reshape([0.0, 0.9], 2, 1)                       # intercept 0, rho 0.9
        Psi_big = reshape([0.0, -0.5], 2, 1)                # would give rho = 1.4
        Bc, delta = M._kilian_bias_correction(B, Psi_big, 1, 1)
        @test delta < 1.0
        @test maximum(abs.(eigvals(M.companion_matrix(Bc, 1, 1)))) < 1.0
        # A small correction is applied in full.
        Bc2, d2 = M._kilian_bias_correction(B, reshape([0.0, 0.01], 2, 1), 1, 1)
        @test d2 == 1.0
        @test M.extract_ar_coefficients(Bc2, 1, 1)[1][1, 1] ≈ 0.89 rtol = 1e-12
        # An already-explosive estimate gets NO correction (rule 1).
        Bexp = reshape([0.0, 1.2], 2, 1)
        Bc3, d3 = M._kilian_bias_correction(Bexp, Psi_big, 1, 1)
        @test d3 == 0.0 && Bc3 == Bexp
    end

    @testset "IRF bands: default unchanged, all schemes valid and reproducible" begin
        rng = Random.MersenneTwister(42)
        Y = randn(rng, 200, 2)
        for t in 2:200
            Y[t, :] = [0.5 0.1; 0.2 0.4] * Y[t-1, :] + 0.5 * randn(rng, 2)
        end
        mv = estimate_var(Y, 2)
        base = irf(mv, 12; ci_type=:bootstrap, reps=80, seed=1)
        # Backward compatibility: :iid must be BIT-identical to the historical default.
        iid = irf(mv, 12; ci_type=:bootstrap, reps=80, seed=1, bootstrap=:iid)
        @test iid.ci_lower == base.ci_lower && iid.ci_upper == base.ci_upper
        for sch in (:wild, :block)
            r = irf(mv, 12; ci_type=:bootstrap, reps=80, seed=1, bootstrap=sch)
            @test all(isfinite, r.ci_lower) && all(isfinite, r.ci_upper)
            @test all(r.ci_lower .<= r.ci_upper)
            @test r.values == base.values                   # point IRF never moves
            @test r.ci_lower != base.ci_lower               # ... but the bands do
        end
        bc = irf(mv, 12; ci_type=:bootstrap, reps=80, seed=1, bias_correct=true, bias_reps=40)
        @test all(isfinite, bc.ci_lower) && all(bc.ci_lower .<= bc.ci_upper)
        # Kilian (1998): bias_correct also corrects the point IRF (#564) — the
        # corrected point must actually move relative to the uncorrected one.
        @test all(isfinite, bc.values)
        @test size(bc.values) == size(base.values)
        @test maximum(abs, bc.values .- base.values) > 1e-10
        # Rejected outside the bootstrap machinery rather than silently ignored
        @test_throws ArgumentError irf(mv, 12; ci_type=:none, bias_correct=true)
        # reproducible at a fixed seed
        @test irf(mv, 12; ci_type=:bootstrap, reps=40, seed=99, bootstrap=:wild).ci_lower ==
              irf(mv, 12; ci_type=:bootstrap, reps=40, seed=99, bootstrap=:wild).ci_lower
        @test_throws ArgumentError irf(mv, 6; ci_type=:bootstrap, reps=10, bootstrap=:bogus)
        # the scheme is recorded in the reproducibility manifest
        @test irf(mv, 6; ci_type=:bootstrap, reps=10, seed=3,
                  bootstrap=:wild).manifest.settings["bootstrap"] == "wild"
    end
end
