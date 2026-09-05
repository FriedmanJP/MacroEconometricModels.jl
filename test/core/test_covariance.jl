# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Statistics
using LinearAlgebra
using Random

@testset "Covariance Estimators" begin

    # =========================================================================
    # Kernel Weights
    # =========================================================================

    @testset "kernel_weight" begin
        # Bartlett kernel
        @test MacroEconometricModels.kernel_weight(0, 5, :bartlett) == 1.0
        @test MacroEconometricModels.kernel_weight(5, 5, :bartlett) ≈ 1 - 5/6  # x = 5/6
        @test MacroEconometricModels.kernel_weight(0, 0, :bartlett) == 0.0  # bandwidth=0 returns 0

        # All kernels return 1 at j=0 (except bandwidth=0)
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            @test MacroEconometricModels.kernel_weight(0, 5, kernel) ≈ 1.0
        end

        # All kernels return 0 when |x| > 1 (j > bandwidth+1)
        for kernel in [:bartlett, :parzen, :tukey_hanning]
            @test MacroEconometricModels.kernel_weight(10, 3, kernel) == 0.0
        end

        # Bartlett: linearly decreasing
        w1 = MacroEconometricModels.kernel_weight(1, 5, :bartlett)
        w2 = MacroEconometricModels.kernel_weight(2, 5, :bartlett)
        w3 = MacroEconometricModels.kernel_weight(3, 5, :bartlett)
        @test w1 > w2 > w3

        # Parzen kernel: values in [0, 1]
        for j in 0:5
            w = MacroEconometricModels.kernel_weight(j, 5, :parzen)
            @test 0 <= w <= 1
        end

        # Tukey-Hanning: values in [0, 1]
        for j in 0:5
            w = MacroEconometricModels.kernel_weight(j, 5, :tukey_hanning)
            @test 0 <= w <= 1
        end

        # Quadratic spectral: test non-zero for j > 0
        w_qs = MacroEconometricModels.kernel_weight(1, 5, :quadratic_spectral)
        @test w_qs != 0.0

        # Unknown kernel
        @test_throws ArgumentError MacroEconometricModels.kernel_weight(1, 5, :unknown)

        # Float32 type
        w32 = MacroEconometricModels.kernel_weight(1, 5, :bartlett, Float32)
        @test w32 isa Float32
    end

    # =========================================================================
    # Optimal Bandwidth Selection
    # =========================================================================

    @testset "optimal_bandwidth_nw" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        # White noise: bandwidth should be small
        x_wn = randn(rng, 200)
        bw_wn = MacroEconometricModels.optimal_bandwidth_nw(x_wn)
        @test bw_wn >= 0
        @test bw_wn <= 20  # Should be small for white noise

        # Persistent series (stationary AR(1), DGP-02 #791): bandwidth larger
        x_pers = dgp_arima(rng; phi=[0.9], sigma=0.1, T=200).y
        bw_pers = MacroEconometricModels.optimal_bandwidth_nw(x_pers)
        @test bw_pers >= 0
        @test bw_pers > bw_wn  # persistence needs lags; iid does not

        # Short series
        bw_short = MacroEconometricModels.optimal_bandwidth_nw(randn(rng, 3))
        @test bw_short == 0

        # Multivariate version
        X_mv = randn(rng, 200, 3)
        bw_mv = MacroEconometricModels.optimal_bandwidth_nw(X_mv)
        @test bw_mv >= 0

        # Empty multivariate
        bw_empty = MacroEconometricModels.optimal_bandwidth_nw(zeros(10, 0))
        @test bw_empty == 0
    end

    @testset "optimal_bandwidth_nw — Andrews (1991) plug-in (T052)" begin
        # Persistent AR(1) with ρ≈0.5 so the plug-in bandwidth comfortably exceeds
        # the old degenerate floor(n^(1/3))=5 clamp but stays under the Schwert cap.
        rng = MersenneTwister(4242)  # DGP-02: explicit rng
        n = 200
        x = dgp_arima(rng; phi=[0.5], T=n).y
        # Recompute ρ̂ with the SAME estimator the function uses → the pins are exact
        # regardless of the realized draw.
        rho = dot(@view(x[1:end-1]), @view(x[2:end])) / dot(@view(x[1:end-1]), @view(x[1:end-1]))
        ra = min(abs(rho), 0.99)
        schwert = floor(Int, 12 * (n / 100)^(1 / 4))          # = 14 for n=200

        # (1) Bartlett exact pin: α(1)=4ρ²/(1−ρ²)², constant 1.1447, exponent 1/3
        a1 = 4ra^2 / (1 - ra^2)^2
        exp_bart = min(ceil(Int, 1.1447 * (a1 * n)^(1 / 3)), schwert)
        @test MacroEconometricModels.optimal_bandwidth_nw(x) == exp_bart
        @test MacroEconometricModels.optimal_bandwidth_nw(x; kernel=:bartlett) == exp_bart

        # (2) q=2 kernels: α(2)=4ρ²/(1−ρ)⁴ with kernel-specific Andrews constants
        a2 = 4ra^2 / (1 - ra)^4
        @test MacroEconometricModels.optimal_bandwidth_nw(x; kernel=:parzen) ==
              min(ceil(Int, 2.6614 * (a2 * n)^(1 / 5)), schwert)
        @test MacroEconometricModels.optimal_bandwidth_nw(x; kernel=:quadratic_spectral) ==
              min(ceil(Int, 1.3221 * (a2 * n)^(1 / 5)), schwert)
        @test MacroEconometricModels.optimal_bandwidth_nw(x; kernel=:tukey_hanning) ==
              min(ceil(Int, 1.7462 * (a2 * n)^(1 / 5)), schwert)

        # (3) Truncation-cap fix: bandwidth now exceeds the old floor(n^(1/3)) clamp
        @test MacroEconometricModels.optimal_bandwidth_nw(x) > floor(Int, n^(1 / 3))

        # (4) Unknown kernel throws
        @test_throws ArgumentError MacroEconometricModels.optimal_bandwidth_nw(x; kernel=:bogus)

        # (5) Mislabel/α-gap: the old code paired the Bartlett constant/exponent with
        #     the α(2) plug-in → a strictly larger (uncapped) bandwidth than the
        #     kernel-consistent Bartlett value.
        old_uncapped = ceil(Int, 1.1447 * (a2 * n)^(1 / 3))
        new_uncapped = ceil(Int, 1.1447 * (a1 * n)^(1 / 3))
        @test new_uncapped < old_uncapped

        # (6) Smoke: every kernel flows finitely through the public estimators
        X = hcat(ones(n), randn(rng, n, 2)); u = randn(rng, n)
        for k in (:bartlett, :parzen, :quadratic_spectral, :tukey_hanning)
            @test all(isfinite, MacroEconometricModels.newey_west(X, u; kernel=k))
            @test isfinite(MacroEconometricModels.long_run_variance(x; kernel=k))
            @test all(isfinite, MacroEconometricModels.long_run_covariance(X; kernel=k))
        end
    end

    @testset "QS kernel infinite support (T053)" begin
        # (A) analytic pin: QS weight is nonzero for lags beyond the bandwidth.
        #     x = 10/(3+1) = 2.5, z = 6π·2.5/5 = 3π ⇒ sin(3π)=0, cos(3π)=−1
        #     ⇒ w = 25/(12π²·6.25)·1 = 1/(3π²).
        w_qs = MacroEconometricModels.kernel_weight(10, 3, :quadratic_spectral)
        @test w_qs ≈ 1 / (3π^2) atol = 1e-12
        @test w_qs > 0
        # Compact-support kernels still truncate to exactly 0 at |x|>1.
        for k in (:bartlett, :parzen, :tukey_hanning)
            @test MacroEconometricModels.kernel_weight(10, 3, k) == 0.0
        end

        # AR(1), ρ=0.7 (project convention: explicit MersenneTwister seed).
        rng = Random.MersenneTwister(53)
        n = 200
        x = dgp_arima(rng; phi=[0.7], T=n).y  # DGP-02 #791: shared simulator
        xd = x .- Statistics.mean(x)
        bw = 4
        γ(j) = sum(@view(xd[j+1:n]) .* @view(xd[1:n-j])) / n
        wqs(j) = MacroEconometricModels.kernel_weight(j, bw, :quadratic_spectral)

        # (B) QS full-range summation (to n−1) differs from the old bw-truncated sum.
        lrv_qs = MacroEconometricModels.long_run_variance(x; bandwidth=bw, kernel=:quadratic_spectral)
        S0 = sum(abs2, xd) / n
        trunc_qs = S0 + sum(2 * wqs(j) * γ(j) for j in 1:bw)
        tail = sum(2 * wqs(j) * γ(j) for j in (bw+1):(n-1))
        @test !isapprox(lrv_qs, trunc_qs; rtol=1e-6)   # summation range genuinely changed
        @test lrv_qs > trunc_qs                         # positive serial corr ⇒ positive tail mass
        @test abs(tail) > 0
        @test lrv_qs ≈ trunc_qs + tail atol = 1e-10     # exact reconstruction

        # (C) compact kernels are a strict no-op off the QS path (jmax==bw).
        for k in (:bartlett, :parzen, :tukey_hanning)
            manual = S0 + sum(2 * MacroEconometricModels.kernel_weight(j, bw, k) * γ(j) for j in 1:bw)
            @test MacroEconometricModels.long_run_variance(x; bandwidth=bw, kernel=k) ≈ manual rtol = 1e-12
        end
    end

    @testset "Andrews–Monahan prewhitening (T054)" begin
        # (A) Scalar reduction (k=1, X=ones): the VAR(1) collapses to a scalar AR(1)
        #     with ρ = Σu_{t-1}u_t / Σu_{t-1}², whitened û_t = u_t − ρ u_{t-1} (t=2..n,
        #     length n−1, NOT spliced with u[1]), recolored by 1/(1−ρ)².
        rng = MersenneTwister(71)  # DGP-02: explicit rng
        n = 300
        u = dgp_arima(rng; phi=[0.6], T=n).y  # DGP-02 #791: shared simulator
        bw = 5
        ulag = @view u[1:n-1]; ulead = @view u[2:n]
        rho = dot(ulag, ulead) / dot(ulag, ulag)
        uhat = ulead .- rho .* ulag                 # length n-1
        mh = length(uhat)
        Sstar = sum(abs2, uhat)
        for j in 1:bw
            w = MacroEconometricModels.kernel_weight(j, bw, :bartlett)
            Sstar += 2 * w * sum(@view(uhat[j+1:mh]) .* @view(uhat[1:mh-j]))
        end
        V_ref = (1 / n) * (Sstar / (1 - rho)^2) * (1 / n)
        V_scalar = MacroEconometricModels.newey_west(reshape(ones(n), n, 1), u;
                                                     prewhiten=true, bandwidth=bw)
        @test V_scalar[1, 1] ≈ V_ref rtol = 1e-9

        # (B) Multivariate: the new VAR(1) prewhitening genuinely differs from both the
        #     non-prewhitened estimate and the OLD scalar-AR(1) prewhitening.
        rng = MersenneTwister(72)  # DGP-02: explicit rng (fresh, was global reseed)
        xreg = dgp_arima(rng; phi=[0.7], T=n).y  # DGP-02 #791: shared simulator
        X = hcat(ones(n), xreg)
        uu = dgp_arima(rng; phi=[0.5], T=n).y  # DGP-02 #791
        V_new = MacroEconometricModels.newey_west(X, uu; prewhiten=true, bandwidth=bw)
        V_noprew = MacroEconometricModels.newey_west(X, uu; prewhiten=false, bandwidth=bw)
        # inline reconstruction of the OLD scalar-AR(1) prewhitening (spliced first obs)
        rl = @view uu[1:n-1]; ld = @view uu[2:n]
        rho_s = dot(rl, ld) / dot(rl, rl)
        u_pw = vcat([uu[1]], uu[2:n] .- rho_s .* uu[1:n-1])
        Xu_old = X .* u_pw
        S_old = Xu_old' * Xu_old
        for j in 1:bw
            w = MacroEconometricModels.kernel_weight(j, bw, :bartlett)
            Gj = @view(Xu_old[j+1:n, :])' * @view(Xu_old[1:n-j, :])
            S_old .+= w * (Gj + Gj')
        end
        S_old ./= (1 - rho_s)^2
        XtXi = inv(X'X)
        V_old = XtXi * S_old * XtXi
        @test !isapprox(V_new, V_noprew; rtol=1e-3)
        @test !isapprox(V_new, V_old; rtol=1e-3)
        @test isapprox(V_new, V_new'; atol=1e-10)

        # (C) Stability guard: a near-unit-root moment VAR(1) → warned fallback to no
        #     prewhitening (bit-for-bit equal to prewhiten=false).
        rng = MersenneTwister(99)  # DGP-02: explicit rng
        m = 500
        u_rw = dgp_arima(rng; phi=[0.995], T=m).y  # DGP-02 #791: shared simulator
        Xc = reshape(ones(m), m, 1)
        V_fallback = @test_logs (:warn,) match_mode = :any MacroEconometricModels.newey_west(
            Xc, u_rw; prewhiten=true, bandwidth=4)
        @test V_fallback ≈ MacroEconometricModels.newey_west(Xc, u_rw; prewhiten=false, bandwidth=4)

        # (D) Helper contract: stable moments → (n-1)×k whitened + k×k A; near-unit-root → nothing.
        rng = MersenneTwister(7)  # DGP-02: explicit rng
        Gm = randn(rng, 300, 2)
        Ghat, A = MacroEconometricModels._prewhiten_moments(Gm)
        @test size(Ghat) == (299, 2)
        @test size(A) == (2, 2)
        g1 = dgp_arima(rng; phi=[0.995], T=m).y  # DGP-02 #791: shared simulator
        g2 = dgp_arima(rng; phi=[0.99], T=m).y  # DGP-02 #791
        Gh2, _ = MacroEconometricModels._prewhiten_moments(hcat(g1, g2); radius_cap=0.97)
        @test Gh2 === nothing
    end

    @testset "long_run_covariance PSD gate (T063 SUB-2)" begin
        # The isposdef gate is behavior-preserving: the result is symmetric + PSD on both
        # the Bartlett (PD) and QS (may be non-PD → projected) paths.
        rng = MersenneTwister(63)  # DGP-02: explicit rng
        X = hcat(ones(200), randn(rng, 200, 2))
        for k in (:bartlett, :quadratic_spectral)
            S = MacroEconometricModels.long_run_covariance(X; bandwidth=4, kernel=k)
            @test isapprox(S, S'; atol=1e-12)
            @test minimum(eigvals(Symmetric(S))) ≥ -1e-10
        end
    end

    @testset "System HAC cross-equation blocks (T055)" begin
        rng = MersenneTwister(313)  # DGP-02: explicit rng
        n = 300; k = 2; n_eq = 2
        X = hcat(ones(n), randn(rng, n))
        fac = randn(rng, n)                  # common factor → correlated equations
        u1 = randn(rng, n) .+ fac
        u2 = randn(rng, n) .+ fac
        U = hcat(u1, u2)
        bw = 4
        XtXinv = inv(X'X)
        blk1 = 1:k; blk2 = (k+1):(2k)

        # (a) newey_west — nonzero cross-block, symmetric, PSD, diagonal-block backward compat
        Vnw = MacroEconometricModels.newey_west(X, U; bandwidth=bw)
        @test size(Vnw) == (k * n_eq, k * n_eq)
        @test isapprox(Vnw, Vnw'; atol=1e-10)
        @test minimum(eigvals(Symmetric(Vnw))) ≥ -1e-8
        @test opnorm(Vnw[blk1, blk2]) > 1e-6
        @test Vnw[blk1, blk1] ≈ MacroEconometricModels.newey_west(X, u1; bandwidth=bw) atol = 1e-9
        @test Vnw[blk2, blk2] ≈ MacroEconometricModels.newey_west(X, u2; bandwidth=bw) atol = 1e-9

        # (b) white_vcov — all HC variants
        for variant in (:hc0, :hc1, :hc2, :hc3)
            Vw = MacroEconometricModels.white_vcov(X, U; variant=variant)
            @test isapprox(Vw, Vw'; atol=1e-10)
            @test minimum(eigvals(Symmetric(Vw))) ≥ -1e-8
            @test opnorm(Vw[blk1, blk2]) > 1e-6
            @test Vw[blk1, blk1] ≈ MacroEconometricModels.white_vcov(X, u1; variant=variant) atol = 1e-9
            @test Vw[blk2, blk2] ≈ MacroEconometricModels.white_vcov(X, u2; variant=variant) atol = 1e-9
        end
        # exact cross-block identity (HC0): V[1:k, k+1:2k] = (X'X)^{-1}(Xu1'Xu2)(X'X)^{-1}
        Xu1 = X .* u1; Xu2 = X .* u2
        V0 = MacroEconometricModels.white_vcov(X, U; variant=:hc0)
        @test V0[blk1, blk2] ≈ XtXinv * (Xu1' * Xu2) * XtXinv atol = 1e-10

        # (c) driscoll_kraay
        Vdk = MacroEconometricModels.driscoll_kraay(X, U; bandwidth=bw)
        @test isapprox(Vdk, Vdk'; atol=1e-10)
        @test opnorm(Vdk[blk1, blk2]) > 1e-6
        @test Vdk[blk1, blk1] ≈ MacroEconometricModels.driscoll_kraay(X, u1; bandwidth=bw) atol = 1e-9
        @test Vdk[blk2, blk2] ≈ MacroEconometricModels.driscoll_kraay(X, u2; bandwidth=bw) atol = 1e-9

        # Perfectly-correlated equations ⇒ all four k×k blocks equal (HC0)
        Vd = MacroEconometricModels.white_vcov(X, hcat(u1, u1); variant=:hc0)
        @test Vd[blk1, blk1] ≈ Vd[blk1, blk2] atol = 1e-10
        @test Vd[blk1, blk2] ≈ Vd[blk2, blk2] atol = 1e-10

        # Independent equations ⇒ cross-block small relative to diagonal
        rng = MersenneTwister(314)  # DGP-02: explicit rng
        ni = 2000
        Xi = hcat(ones(ni), randn(rng, ni))
        Vi = MacroEconometricModels.white_vcov(Xi, hcat(randn(rng, ni), randn(rng, ni)); variant=:hc0)
        @test opnorm(Vi[blk1, blk2]) < 0.15 * opnorm(Vi[blk1, blk1])
    end

    # =========================================================================
    # Newey-West HAC Estimator
    # =========================================================================

    @testset "newey_west - univariate residuals" begin
        rng = MersenneTwister(100)  # DGP-02: explicit rng
        n = 200
        k = 3
        X = hcat(ones(n), randn(rng, n, k - 1))
        beta = [1.0, 2.0, -0.5]
        residuals = randn(rng, n)

        # Bartlett kernel (default)
        V_nw = MacroEconometricModels.newey_west(X, residuals)
        @test size(V_nw) == (k, k)
        @test isapprox(V_nw, V_nw', atol=1e-10)  # Symmetric
        @test all(eigvals(Symmetric(V_nw)) .> -1e-8)  # PSD

        # All 4 kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V = MacroEconometricModels.newey_west(X, residuals; kernel=kernel)
            @test size(V) == (k, k)
            @test isapprox(V, V', atol=1e-10)
        end

        # Fixed bandwidth
        V_bw = MacroEconometricModels.newey_west(X, residuals; bandwidth=5)
        @test size(V_bw) == (k, k)

        # Prewhitening
        V_pw = MacroEconometricModels.newey_west(X, residuals; prewhiten=true)
        @test size(V_pw) == (k, k)
        @test isapprox(V_pw, V_pw', atol=1e-10)

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.newey_west(X, residuals; XtX_inv=XtX_inv)
        @test isapprox(V_cached, V_nw, atol=1e-10)
    end

    @testset "newey_west - multivariate residuals" begin
        rng = MersenneTwister(200)  # DGP-02: explicit rng
        n = 200
        k = 2
        n_eq = 3
        X = hcat(ones(n), randn(rng, n))
        residuals = randn(rng, n, n_eq)

        V = MacroEconometricModels.newey_west(X, residuals)
        @test size(V) == (k * n_eq, k * n_eq)

        # Single column treated as vector
        V_single = MacroEconometricModels.newey_west(X, residuals[:, 1:1])
        @test size(V_single) == (k, k)
    end

    # =========================================================================
    # White Heteroscedasticity-Robust Estimator
    # =========================================================================

    @testset "white_vcov - all HC variants" begin
        rng = MersenneTwister(300)  # DGP-02: explicit rng
        n = 100
        k = 3
        X = hcat(ones(n), randn(rng, n, k - 1))
        residuals = randn(rng, n) .* exp.(0.5 * randn(rng, n))  # Heteroscedastic

        for variant in [:hc0, :hc1, :hc2, :hc3]
            V = MacroEconometricModels.white_vcov(X, residuals; variant=variant)
            @test size(V) == (k, k)
            @test isapprox(V, V', atol=1e-10)  # Symmetric
        end

        # HC1 should give larger values than HC0 (finite sample correction)
        V_hc0 = MacroEconometricModels.white_vcov(X, residuals; variant=:hc0)
        V_hc1 = MacroEconometricModels.white_vcov(X, residuals; variant=:hc1)
        @test all(diag(V_hc1) .>= diag(V_hc0) .- 1e-10)

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.white_vcov(X, residuals; variant=:hc0, XtX_inv=XtX_inv)
        @test isapprox(V_cached, V_hc0, atol=1e-10)
    end

    @testset "white_vcov - multivariate residuals" begin
        rng = MersenneTwister(400)  # DGP-02: explicit rng
        n = 100
        k = 2
        n_eq = 2
        X = hcat(ones(n), randn(rng, n))
        residuals = randn(rng, n, n_eq)

        V = MacroEconometricModels.white_vcov(X, residuals)
        @test size(V) == (k * n_eq, k * n_eq)
    end

    # =========================================================================
    # Driscoll-Kraay Estimator
    # =========================================================================

    @testset "driscoll_kraay - univariate" begin
        rng = MersenneTwister(500)  # DGP-02: explicit rng
        n = 200
        k = 3
        X = hcat(ones(n), randn(rng, n, k - 1))
        u = randn(rng, n)

        V = MacroEconometricModels.driscoll_kraay(X, u)
        @test size(V) == (k, k)
        @test isapprox(V, V', atol=1e-10)

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V_k = MacroEconometricModels.driscoll_kraay(X, u; kernel=kernel)
            @test size(V_k) == (k, k)
        end

        # With precomputed XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        V_cached = MacroEconometricModels.driscoll_kraay(X, u; XtX_inv=XtX_inv)
        @test isapprox(V_cached, V, atol=1e-10)
    end

    @testset "driscoll_kraay - multivariate" begin
        rng = MersenneTwister(600)  # DGP-02: explicit rng
        n = 200
        k = 2
        n_eq = 3
        X = hcat(ones(n), randn(rng, n))
        U = randn(rng, n, n_eq)

        V = MacroEconometricModels.driscoll_kraay(X, U)
        @test size(V) == (k * n_eq, k * n_eq)
    end

    # =========================================================================
    # Covariance Estimator Dispatch
    # =========================================================================

    @testset "robust_vcov dispatch" begin
        rng = MersenneTwister(700)  # DGP-02: explicit rng
        n = 100
        k = 2
        X = hcat(ones(n), randn(rng, n))
        residuals = randn(rng, n)

        # NeweyWestEstimator
        nw_est = MacroEconometricModels.NeweyWestEstimator()
        V_nw = MacroEconometricModels.robust_vcov(X, residuals, nw_est)
        @test size(V_nw) == (k, k)

        # WhiteEstimator
        white_est = MacroEconometricModels.WhiteEstimator()
        V_white = MacroEconometricModels.robust_vcov(X, residuals, white_est)
        @test size(V_white) == (k, k)

        # DriscollKraayEstimator
        dk_est = MacroEconometricModels.DriscollKraayEstimator()
        V_dk = MacroEconometricModels.robust_vcov(X, residuals, dk_est)
        @test size(V_dk) == (k, k)

        # Multivariate dispatch
        residuals_mv = randn(rng, n, 2)
        V_nw_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, nw_est)
        @test size(V_nw_mv) == (k * 2, k * 2)

        V_white_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, white_est)
        @test size(V_white_mv) == (k * 2, k * 2)

        V_dk_mv = MacroEconometricModels.robust_vcov(X, residuals_mv, dk_est)
        @test size(V_dk_mv) == (k * 2, k * 2)
    end

    # =========================================================================
    # Estimator Type Constructors
    # =========================================================================

    @testset "Estimator constructors" begin
        nw = MacroEconometricModels.NeweyWestEstimator()
        @test nw.bandwidth == 0
        @test nw.kernel == :bartlett
        @test nw.prewhiten == false

        nw2 = MacroEconometricModels.NeweyWestEstimator(bandwidth=5, kernel=:parzen, prewhiten=true)
        @test nw2.bandwidth == 5
        @test nw2.kernel == :parzen
        @test nw2.prewhiten == true

        @test_throws ArgumentError MacroEconometricModels.NeweyWestEstimator(bandwidth=-1)
        @test_throws ArgumentError MacroEconometricModels.NeweyWestEstimator(kernel=:invalid)

        dk = MacroEconometricModels.DriscollKraayEstimator()
        @test dk.bandwidth == 0
        @test dk.kernel == :bartlett

        @test_throws ArgumentError MacroEconometricModels.DriscollKraayEstimator{Float64}(-1)
    end

    # =========================================================================
    # precompute_XtX_inv
    # =========================================================================

    @testset "precompute_XtX_inv" begin
        rng = MersenneTwister(800)  # DGP-02: explicit rng
        n = 100
        k = 3
        X = hcat(ones(n), randn(rng, n, k - 1))

        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        @test size(XtX_inv) == (k, k)

        # Verify it's actually (X'X)^{-1}
        XtX = X' * X
        @test isapprox(XtX_inv * XtX, Matrix{Float64}(I, k, k), atol=1e-8)
        @test isapprox(XtX * XtX_inv, Matrix{Float64}(I, k, k), atol=1e-8)

        # Caching: results from NW with/without cached should match
        residuals = randn(rng, n)
        V1 = MacroEconometricModels.newey_west(X, residuals)
        V2 = MacroEconometricModels.newey_west(X, residuals; XtX_inv=XtX_inv)
        @test isapprox(V1, V2, atol=1e-10)
    end

    # =========================================================================
    # Long-Run Variance
    # =========================================================================

    @testset "long_run_variance" begin
        rng = MersenneTwister(900)  # DGP-02: explicit rng

        # White noise: long-run variance ≈ variance
        n = 1000
        x_wn = randn(rng, n)
        lrv = MacroEconometricModels.long_run_variance(x_wn)
        @test lrv > 0
        @test isapprox(lrv, var(x_wn), rtol=0.3)  # Approximately variance for white noise

        # AR(1) with rho = 0.5: theoretical LRV = sigma^2 / (1-rho)^2
        rho = 0.5
        x_ar = dgp_arima(rng; phi=[rho], T=n).y  # DGP-02 #791: shared simulator
        lrv_ar = MacroEconometricModels.long_run_variance(x_ar)
        theoretical_lrv = 1.0 / (1 - rho)^2  # sigma^2 / (1-rho)^2
        @test lrv_ar > 0
        @test isapprox(lrv_ar, theoretical_lrv, rtol=0.5)

        # Fixed bandwidth
        lrv_bw = MacroEconometricModels.long_run_variance(x_wn; bandwidth=3)
        @test lrv_bw > 0

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            lrv_k = MacroEconometricModels.long_run_variance(x_ar; kernel=kernel)
            @test lrv_k > 0
        end

        # Very short series (single element: var returns NaN, that's expected)
        lrv_short = MacroEconometricModels.long_run_variance([1.0])
        @test isnan(lrv_short) || lrv_short >= 0
    end

    # =========================================================================
    # Long-Run Covariance
    # =========================================================================

    @testset "long_run_covariance" begin
        rng = MersenneTwister(1000)  # DGP-02: explicit rng
        n = 300
        k = 3
        X = randn(rng, n, k)

        lrc = MacroEconometricModels.long_run_covariance(X)
        @test size(lrc) == (k, k)
        @test isapprox(lrc, lrc', atol=1e-10)  # Symmetric
        @test all(eigvals(Symmetric(lrc)) .>= -1e-10)  # PSD

        # Different kernels
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            lrc_k = MacroEconometricModels.long_run_covariance(X; kernel=kernel)
            @test size(lrc_k) == (k, k)
            @test isapprox(lrc_k, lrc_k', atol=1e-10)
        end

        # Fixed bandwidth
        lrc_bw = MacroEconometricModels.long_run_covariance(X; bandwidth=5)
        @test size(lrc_bw) == (k, k)

        # Short series
        X_short = randn(rng, 1, 2)
        lrc_short = MacroEconometricModels.long_run_covariance(X_short)
        @test size(lrc_short) == (2, 2)
    end

    @testset "HAC recovers known long-run variance (DGP-02 #791)" begin
        # AR(1) ρ=0.7, σ=1: LRV = σ²/(1−ρ)² = 11.11. MA(1) θ=0.5: LRV =
        # σ²(1+θ)² = 2.25. With X=ones, n·V[1,1] estimates the LRV for the
        # sandwich (NW), rescaled (DK), and direct (long_run_variance) forms.
        # rtol 0.25 covers Bartlett-kernel downward bias plus sampling noise
        # of the long-run average at T=2000.
        rng = MersenneTwister(90210)
        T = 2000
        X1 = ones(T, 1)
        ar = dgp_arima(rng; phi=[0.7], T=T).y
        ma = dgp_arima(rng; theta=[0.5], T=T).y

        Vnw_ar = MacroEconometricModels.newey_west(X1, ar)
        @test isapprox(T * Vnw_ar[1, 1], 1 / (1 - 0.7)^2; rtol=0.25)
        Vnw_ma = MacroEconometricModels.newey_west(X1, ma)
        @test isapprox(T * Vnw_ma[1, 1], (1 + 0.5)^2; rtol=0.25)

        # A Newey-West that dropped every lag term (= White) cannot see serial
        # correlation: NW exceeds White by a wide margin on these DGPs.
        Vw_ar = MacroEconometricModels.white_vcov(X1, ar; variant=:hc0)
        @test Vnw_ar[1, 1] > 3 * Vw_ar[1, 1]   # LRV 11.1 vs γ0 ≈ 2.0
        Vw_ma = MacroEconometricModels.white_vcov(X1, ma; variant=:hc0)
        @test Vnw_ma[1, 1] > 1.3 * Vw_ma[1, 1]  # LRV 2.25 vs γ0 = 1.25

        # Driscoll-Kraay sees the same LRV through the moment LRV ...
        Vdk_ar = MacroEconometricModels.driscoll_kraay(X1, ar)
        @test isapprox(T * Vdk_ar[1, 1], 1 / (1 - 0.7)^2; rtol=0.25)
        Vdk_ma = MacroEconometricModels.driscoll_kraay(X1, ma)
        @test isapprox(T * Vdk_ma[1, 1], (1 + 0.5)^2; rtol=0.3)

        # ... and inflates under a common serially correlated shock: f+e1
        # carries LRV(f)+1 ≈ 12.1 versus ≈ 1 without the common shock.
        f = dgp_arima(rng; phi=[0.7], T=T).y
        e1 = randn(rng, T)
        e2 = randn(rng, T)
        Vdk_com = MacroEconometricModels.driscoll_kraay(X1, f .+ e1)
        Vdk_idio = MacroEconometricModels.driscoll_kraay(X1, e2)
        @test Vdk_com[1, 1] > 5 * Vdk_idio[1, 1]

        # long_run_covariance on [persistent, iid]: known diagonal, ~0 off-diag.
        S = MacroEconometricModels.long_run_covariance(hcat(ar, e2))
        @test isapprox(S[1, 1], 1 / (1 - 0.7)^2; rtol=0.25)
        @test isapprox(S[2, 2], 1.0; rtol=0.3)
        @test abs(S[1, 2]) < 1.0  # independent series: no long-run comovement
    end

    @testset "precompute_XtX_inv caching pattern" begin
        rng = MersenneTwister(8801)  # DGP-02: explicit rng
        X = randn(rng, 80, 4)
        u = randn(rng, 80)
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)

        # Newey-West with cached XtX_inv
        V_nw = MacroEconometricModels.newey_west(X, u; XtX_inv=XtX_inv)
        V_nw2 = MacroEconometricModels.newey_west(X, u)
        @test norm(V_nw - V_nw2) < 1e-10

        # White with cached XtX_inv
        V_w = MacroEconometricModels.white_vcov(X, u; XtX_inv=XtX_inv)
        V_w2 = MacroEconometricModels.white_vcov(X, u)
        @test norm(V_w - V_w2) < 1e-10

        # Driscoll-Kraay with cached XtX_inv
        V_dk = MacroEconometricModels.driscoll_kraay(X, u; XtX_inv=XtX_inv)
        V_dk2 = MacroEconometricModels.driscoll_kraay(X, u)
        @test norm(V_dk - V_dk2) < 1e-10
    end

    @testset "NW at bw=0 matches White HC0 for white noise" begin
        # Regression test: for white noise residuals, auto-bandwidth ≈ 0,
        # so NW should approximate White HC0 (both are sandwich estimators
        # with the same meat when there is no autocorrelation).
        rng = MersenneTwister(12345)  # DGP-02: explicit rng
        n = 500
        k = 3
        X = hcat(ones(n), randn(rng, n, k - 1))
        u = randn(rng, n)  # white noise — auto bandwidth should be 0 or very small

        V_nw = MacroEconometricModels.newey_west(X, u; bandwidth=0)
        V_white = MacroEconometricModels.white_vcov(X, u; variant=:hc0)

        # With bandwidth=0, NW should exactly equal White HC0
        bw_auto = MacroEconometricModels.optimal_bandwidth_nw(u)
        if bw_auto == 0
            @test isapprox(V_nw, V_white, rtol=1e-10)
        else
            # Even if auto bw > 0, they should be close for white noise
            @test isapprox(V_nw, V_white, rtol=0.3)
        end

        # Explicitly force bw=0: NW with bw=0 must exactly equal White HC0
        V_nw_bw0 = MacroEconometricModels.newey_west(X, u; bandwidth=0)
        bw_check = MacroEconometricModels.optimal_bandwidth_nw(u)
        if bw_check == 0
            @test isapprox(V_nw_bw0, V_white, rtol=1e-10)
        end
    end

    @testset "Newey-West with fixed bandwidth and all kernels" begin
        rng = MersenneTwister(8802)  # DGP-02: explicit rng
        X = randn(rng, 80, 3)
        u = randn(rng, 80)
        for kernel in [:bartlett, :parzen, :quadratic_spectral, :tukey_hanning]
            V = MacroEconometricModels.newey_west(X, u; bandwidth=5, kernel=kernel)
            @test size(V) == (3, 3)
            @test issymmetric(V) || norm(V - V') < 1e-12
        end
    end

end
